"""
Disk-backed external chunk counter for TB-scale out-of-core tokenizer training.

Provides StreamingChunkCounter (aliased as DiskChunkCounter) which accumulates
token counts in configurable in-memory buffers (default 500MB), flushes sorted
binary runs to disk, and executes an external k-way min-heap tournament merge
to support repeated, sequential EM iteration with bounded RAM (one buffer plus
one sparse-index entry per ``sparse_index_step`` records, not the corpus).
"""

from __future__ import annotations

import atexit
import bisect
import heapq
import os
import shutil
import signal
import struct
import sys
import tempfile
import weakref
from collections import Counter
from collections.abc import ItemsView, Iterator, Mapping, ValuesView
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Global registry of active counters to guarantee cleanup on exit or signals
_ACTIVE_COUNTERS: Dict[int, weakref.ReferenceType[StreamingChunkCounter]] = {}
_CLEANUP_HANDLERS_REGISTERED = False


def _cleanup_all_active_counters() -> None:
    """Closes and removes temporary storage for all live counters."""
    for ref in list(_ACTIVE_COUNTERS.values()):
        counter = ref()
        if counter is not None:
            try:
                counter.close()
            except Exception:
                pass


def _ensure_atexit_cleanup() -> None:
    """Registers the atexit temp-dir cleanup exactly once (always safe)."""
    global _CLEANUP_HANDLERS_REGISTERED
    if _CLEANUP_HANDLERS_REGISTERED:
        return
    atexit.register(_cleanup_all_active_counters)
    _CLEANUP_HANDLERS_REGISTERED = True


def _install_signal_handlers() -> None:
    """Chains SIGINT/SIGTERM handlers that clean up temp files on kill.

    Explicit opt-in only (see ``install_signal_handlers``): installing from a
    constructor would otherwise mutate process-global signal state behind the
    caller's back, changing exit codes (SIG_DFL becomes SystemExit) and
    fighting host frameworks' own handlers.
    """
    _ensure_atexit_cleanup()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            prev_handler = signal.getsignal(sig)
            if prev_handler not in (signal.SIG_IGN, None):

                def _make_handler(old_h: Any, s: int):
                    def _handler(signum: int, frame: Any) -> None:
                        _cleanup_all_active_counters()
                        if callable(old_h):
                            old_h(signum, frame)
                        elif signum == signal.SIGINT:
                            raise KeyboardInterrupt()
                        else:
                            sys.exit(128 + signum)

                    return _handler

                signal.signal(sig, _make_handler(prev_handler, sig))
    except (ValueError, AttributeError):
        # Platform does not support signal or not running in main thread
        pass


class _BinaryRunReader:
    """Sequential reader for a single sorted binary run file.

    Format per record:
        [str_len: 4 bytes unsigned int (<I)]
        [count:   8 bytes unsigned int (<Q)]
        [utf-8 bytes: str_len bytes]
    """

    __slots__ = ("file_path", "buffer_size", "_file")

    def __init__(self, file_path: str, buffer_size: int = 64 * 1024) -> None:
        self.file_path = file_path
        self.buffer_size = buffer_size
        self._file = open(file_path, "rb", buffering=buffer_size)

    def read_next(self) -> Optional[Tuple[str, int]]:
        header = self._file.read(12)
        if not header or len(header) < 12:
            return None
        str_len, count = struct.unpack("<IQ", header)
        token_bytes = self._file.read(str_len)
        if len(token_bytes) < str_len:
            raise IOError(f"Unexpected EOF while reading record from {self.file_path}")
        return token_bytes.decode("utf-8"), count

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> _BinaryRunReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class _ChunkItemsView(ItemsView[str, int]):
    """ItemsView implementation that streams records sequentially from disk."""

    def __init__(self, counter: StreamingChunkCounter) -> None:
        super().__init__(counter)
        self._counter = counter

    def __len__(self) -> int:
        return len(self._counter)

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        k, v = item
        if not isinstance(k, str) or not isinstance(v, int):
            return False
        return self._counter.get(k) == v

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        return self._counter._iter_items()


class _ChunkValuesView(ValuesView[int]):
    """ValuesView implementation that streams frequency counts from disk."""

    def __init__(self, counter: StreamingChunkCounter) -> None:
        super().__init__(counter)
        self._counter = counter

    def __len__(self) -> int:
        return len(self._counter)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int):
            return False
        for v in self:
            if v == value:
                return True
        return False

    def __iter__(self) -> Iterator[int]:
        for _, count in self._counter._iter_items():
            yield count


class StreamingChunkCounter(Mapping[str, int]):
    """Disk-backed chunk counter for out-of-core corpus aggregation.

    Accumulates chunk frequencies in a memory buffer up to ``chunk_size_bytes``.
    When the threshold is reached, items are sorted and flushed to a binary run file.
    Upon finalization, an external k-way min-heap merge consolidates all runs into
    a single sorted binary run file (with intermediate cascading passes if run count
    exceeds ``max_open_runs`` to avoid file descriptor limits).

    Implements ``Mapping[str, int]`` so it directly substitutes for ``Counter[str]``
    in vocabulary building and EM training loops with zero in-memory re-materialization.

    Read/freeze contract: accumulating (``add``/``update``/``__setitem__``) must
    finish before reading. The first read (``__getitem__``/``get``/
    ``__contains__``/``__len__``/iteration/``total``/``most_common``) finalizes
    the counter — merging spilled runs and freezing further accumulation
    (later writes raise ``RuntimeError``). Accumulate fully, then read.
    """

    def __init__(
        self,
        temp_dir: Optional[Union[str, Path]] = None,
        chunk_size_bytes: int = 500 * 1024 * 1024,
        max_open_runs: int = 64,
        io_buffer_size: int = 256 * 1024,
        sparse_index_step: int = 256,
        install_signal_handlers: bool = False,
    ) -> None:
        """Args:
        temp_dir: Parent directory for the counter's temp workspace (created if missing).
        chunk_size_bytes: In-memory buffer budget before spilling a sorted run to disk.
        max_open_runs: Max run files merged in one pass (cascade above this).
        io_buffer_size: File buffering for run I/O.
        sparse_index_step: Index every Nth merged record for lookups.
        install_signal_handlers: When True, chain process SIGINT/SIGTERM
            handlers that clean up temp files on kill. Off by default:
            constructing a counter must not mutate process-global signal
            state (atexit cleanup is always installed and covers normal
            exit paths).
        """
        if chunk_size_bytes <= 0:
            raise ValueError(f"chunk_size_bytes must be positive, got {chunk_size_bytes}")
        if max_open_runs < 2:
            raise ValueError(f"max_open_runs must be at least 2, got {max_open_runs}")

        _ensure_atexit_cleanup()
        if install_signal_handlers:
            _install_signal_handlers()

        if temp_dir is not None:
            base_dir = Path(temp_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            self._dir = tempfile.mkdtemp(prefix="uniqtoken_counter_", dir=str(base_dir))
        else:
            self._dir = tempfile.mkdtemp(prefix="uniqtoken_counter_")

        self.chunk_size_bytes = chunk_size_bytes
        self.max_open_runs = max_open_runs
        self.io_buffer_size = io_buffer_size
        self.sparse_index_step = sparse_index_step

        self._buffer: Dict[str, int] = {}
        self._current_buffer_bytes = 0
        self._run_files: List[str] = []
        self._run_counter = 0

        self._finalized = False
        self._merged_file: Optional[str] = None
        self._total_unique_chunks = 0
        self._total_frequency = 0
        self._sparse_index: List[Tuple[str, int]] = []
        self._closed = False
        _ACTIVE_COUNTERS[id(self)] = weakref.ref(self)

    @property
    def is_streaming(self) -> bool:
        """Indicator for seed builder and trainers that this counter is out-of-core."""
        return True

    @property
    def dir_path(self) -> str:
        """Path to the underlying temporary directory."""
        return self._dir

    def _ensure_open(self) -> None:
        """Raises if the counter was closed (use-after-close is a bug, fail loudly)."""
        if self._closed:
            raise RuntimeError("StreamingChunkCounter is closed.")

    def add(self, chunk: str, count: int = 1) -> None:
        """Adds occurrences of ``chunk``."""
        self._ensure_open()
        if self._finalized:
            raise RuntimeError("Cannot add items to StreamingChunkCounter after finalization.")
        if count <= 0:
            return

        old_count = self._buffer.get(chunk)
        if old_count is None:
            self._buffer[chunk] = count
            # Byte length of chunk + 12-byte header overhead
            self._current_buffer_bytes += len(chunk.encode("utf-8")) + 12
        else:
            self._buffer[chunk] = old_count + count

        if self._current_buffer_bytes >= self.chunk_size_bytes:
            self._flush_buffer()

    def update(self, iterable: Union[Iterable[str], Mapping[str, int]]) -> None:
        """Accumulates counts from an iterable of strings or a Mapping."""
        self._ensure_open()
        if self._finalized:
            raise RuntimeError("Cannot update StreamingChunkCounter after finalization.")

        if isinstance(iterable, Mapping):
            for k, v in iterable.items():
                self.add(k, v)
        else:
            for item in iterable:
                self.add(item, 1)

    def __setitem__(self, key: str, value: int) -> None:
        """Sets the buffered count for ``key`` before finalization.

        Note: only the in-memory buffer entry is replaced. If ``key`` already
        spilled to a run file, the merge sums both records — prefer ``add``
        for accumulation and treat ``__setitem__`` as buffer-local.
        """
        self._ensure_open()
        if self._finalized:
            raise RuntimeError("Cannot set items on StreamingChunkCounter after finalization.")
        old = self._buffer.get(key)
        if old is None:
            self._current_buffer_bytes += len(key.encode("utf-8")) + 12
        self._buffer[key] = value
        if self._current_buffer_bytes >= self.chunk_size_bytes:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flushes in-memory buffer to a sorted binary run file."""
        if not self._buffer:
            return

        sorted_items = sorted(self._buffer.items(), key=lambda x: x[0])
        run_path = os.path.join(self._dir, f"run_{self._run_counter:06d}.bin")
        self._run_counter += 1

        with open(run_path, "wb", buffering=self.io_buffer_size) as f:
            for token, count in sorted_items:
                token_bytes = token.encode("utf-8")
                f.write(struct.pack("<IQ", len(token_bytes), count))
                f.write(token_bytes)

        self._run_files.append(run_path)
        self._buffer.clear()
        self._current_buffer_bytes = 0

    def _merge_runs_to_file(
        self,
        input_paths: List[str],
        output_path: str,
        build_index: bool = False,
    ) -> None:
        """K-way min-heap merge across input binary runs into output_path.

        Duplicates across input runs are summed. Output records are written in
        lexicographically sorted order.
        """
        readers = [_BinaryRunReader(p, buffer_size=self.io_buffer_size) for p in input_paths]
        heap: List[Tuple[str, int, int]] = []  # (token, reader_index, count)

        try:
            for idx, reader in enumerate(readers):
                item = reader.read_next()
                if item is not None:
                    heapq.heappush(heap, (item[0], idx, item[1]))

            with open(output_path, "wb", buffering=self.io_buffer_size) as out_f:
                curr_offset = 0
                while heap:
                    curr_token, curr_idx, curr_count = heapq.heappop(heap)
                    total_count = curr_count

                    # Advance reader that produced curr_token
                    nxt = readers[curr_idx].read_next()
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], curr_idx, nxt[1]))

                    # Drain all other runs that currently have the identical token at the top
                    while heap and heap[0][0] == curr_token:
                        _, dup_idx, dup_count = heapq.heappop(heap)
                        total_count += dup_count
                        dup_nxt = readers[dup_idx].read_next()
                        if dup_nxt is not None:
                            heapq.heappush(heap, (dup_nxt[0], dup_idx, dup_nxt[1]))

                    token_bytes = curr_token.encode("utf-8")
                    rec_len = len(token_bytes)
                    record_bytes = 12 + rec_len

                    if build_index and (self._total_unique_chunks % self.sparse_index_step == 0):
                        self._sparse_index.append((curr_token, curr_offset))

                    out_f.write(struct.pack("<IQ", rec_len, total_count))
                    out_f.write(token_bytes)

                    curr_offset += record_bytes
                    if build_index:
                        self._total_unique_chunks += 1
                        self._total_frequency += total_count
        finally:
            for reader in readers:
                reader.close()

    def finalize(self) -> None:
        """Consolidates all runs into a single merged binary run file."""
        self._ensure_open()
        if self._finalized:
            return

        # 1. Flush any remaining in-memory items
        if self._buffer:
            self._flush_buffer()

        # 2. Case: No items ever added
        if not self._run_files:
            self._finalized = True
            self._total_unique_chunks = 0
            self._total_frequency = 0
            return

        run_files = list(self._run_files)

        # 3. Intermediate cascade multi-pass merge if runs exceed max_open_runs
        cascade_round = 0
        while len(run_files) > self.max_open_runs:
            next_runs: List[str] = []
            for i in range(0, len(run_files), self.max_open_runs):
                batch = run_files[i : i + self.max_open_runs]
                if len(batch) == 1:
                    next_runs.append(batch[0])
                    continue
                cascade_file = os.path.join(self._dir, f"cascade_{cascade_round:04d}_{i:04d}.bin")
                self._merge_runs_to_file(batch, cascade_file, build_index=False)
                # Remove obsolete batch files immediately to free disk space
                for bf in batch:
                    try:
                        os.remove(bf)
                    except OSError:
                        pass
                next_runs.append(cascade_file)
            run_files = next_runs
            cascade_round += 1

        # 4. Final consolidation pass
        merged_file = os.path.join(self._dir, "merged_runs.bin")
        self._sparse_index.clear()
        self._total_unique_chunks = 0
        self._total_frequency = 0

        if len(run_files) == 1:
            # Single run file already has unique, sorted entries; index while copying/scanning
            single_run = run_files[0]
            if single_run != merged_file:
                # Build index and compute stats directly from the single run file
                with _BinaryRunReader(single_run, buffer_size=self.io_buffer_size) as reader:
                    curr_offset = 0
                    while True:
                        rec = reader.read_next()
                        if rec is None:
                            break
                        tok, cnt = rec
                        rec_bytes = 12 + len(tok.encode("utf-8"))
                        if self._total_unique_chunks % self.sparse_index_step == 0:
                            self._sparse_index.append((tok, curr_offset))
                        curr_offset += rec_bytes
                        self._total_unique_chunks += 1
                        self._total_frequency += cnt
                try:
                    os.replace(single_run, merged_file)
                except OSError:
                    shutil.move(single_run, merged_file)
        else:
            self._merge_runs_to_file(run_files, merged_file, build_index=True)
            for rf in run_files:
                try:
                    os.remove(rf)
                except OSError:
                    pass

        self._merged_file = merged_file
        self._run_files = [merged_file]
        self._finalized = True

    def _lookup_count(self, key: str) -> Optional[int]:
        """Performs sparse-indexed binary search for key in merged_runs.bin."""
        self.finalize()
        if not self._merged_file or not os.path.exists(self._merged_file) or not self._sparse_index:
            return None

        # Binary search sparse index for starting byte offset
        idx = bisect.bisect_right(self._sparse_index, (key, float("inf"))) - 1
        if idx < 0:
            idx = 0
        start_offset = self._sparse_index[idx][1]

        with open(self._merged_file, "rb", buffering=self.io_buffer_size) as f:
            f.seek(start_offset)
            while True:
                header = f.read(12)
                if not header or len(header) < 12:
                    break
                str_len, count = struct.unpack("<IQ", header)
                token_bytes = f.read(str_len)
                token = token_bytes.decode("utf-8")
                if token == key:
                    return count
                if token > key:
                    break
        return None

    def __getitem__(self, key: str) -> int:
        """Returns count of ``key``, or 0 if missing (matching collections.Counter).

        Finalizes first (see class docstring): reads observe total counts,
        never buffer-only partials.
        """
        self._ensure_open()
        if not isinstance(key, str):
            return 0
        self.finalize()
        cnt = self._lookup_count(key)
        return cnt if cnt is not None else 0

    def get(self, key: str, default: Any = None) -> Any:
        """Returns count of ``key``, or ``default`` if missing (matching collections.Counter.get).

        Finalizes first (see class docstring).
        """
        self._ensure_open()
        if not isinstance(key, str):
            return default
        self.finalize()
        cnt = self._lookup_count(key)
        return cnt if cnt is not None else default

    def __contains__(self, key: object) -> bool:
        """Returns True if ``key`` has occurrence >= 1. Finalizes first (see class docstring)."""
        self._ensure_open()
        if not isinstance(key, str):
            return False
        self.finalize()
        return self._lookup_count(key) is not None

    def __len__(self) -> int:
        """Returns total number of unique chunks."""
        self.finalize()
        return self._total_unique_chunks

    def total(self) -> int:
        """Returns sum of all chunk frequencies (matching collections.Counter.total)."""
        self.finalize()
        return self._total_frequency

    def __iter__(self) -> Iterator[str]:
        """Yields chunks in sorted order directly from disk."""
        for token, _ in self._iter_items():
            yield token

    def _iter_items(self) -> Iterator[Tuple[str, int]]:
        """Yields (chunk, count) pairs in sorted order directly from disk with O(1) RAM."""
        self.finalize()
        if not self._merged_file or not os.path.exists(self._merged_file):
            return

        with open(self._merged_file, "rb", buffering=self.io_buffer_size) as f:
            while True:
                header = f.read(12)
                if not header or len(header) < 12:
                    break
                str_len, count = struct.unpack("<IQ", header)
                token_bytes = f.read(str_len)
                token = token_bytes.decode("utf-8")
                yield token, count

    def items(self) -> ItemsView[str, int]:
        """Yields (chunk, count) pairs in sorted order directly from disk with O(1) RAM."""
        return _ChunkItemsView(self)

    def values(self) -> ValuesView[int]:
        """Yields frequency counts in sorted chunk order."""
        return _ChunkValuesView(self)

    def most_common(self, n: Optional[int] = None) -> List[Tuple[str, int]]:
        """Returns the n most common chunks and their counts."""
        self.finalize()
        if n is not None:
            if n <= 0:
                return []
            return heapq.nlargest(n, self._iter_items(), key=lambda x: x[1])
        return sorted(self._iter_items(), key=lambda x: (-x[1], x[0]))

    def to_counter(self) -> Counter[str]:
        """Materializes in-memory collections.Counter (only for debugging or small runs)."""
        return Counter(dict(self.items()))

    def close(self) -> None:
        """Closes all resources and deletes the temporary run directory."""
        if self._closed:
            return
        self._closed = True
        _ACTIVE_COUNTERS.pop(id(self), None)
        try:
            if os.path.exists(self._dir):
                shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            # Best-effort: close() also runs from __del__/atexit during
            # interpreter shutdown, where even os.path may be torn down.
            pass

    def __enter__(self) -> StreamingChunkCounter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# Alias for explicit clarity
DiskChunkCounter = StreamingChunkCounter

__all__ = ["StreamingChunkCounter", "DiskChunkCounter"]
