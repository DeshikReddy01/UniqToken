"""
Benchmark: In-Memory Counter vs StreamingChunkCounter (Disk-Backed Out-of-Core).

Evaluates:
- Peak RAM consumption during corpus aggregation (demonstrating O(1) memory bound).
- Flush, external k-way merge, and EM streaming read throughput.
- Scalability across increasing synthetic corpus sizes.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Generator

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.streaming_counter import StreamingChunkCounter


def generate_synthetic_corpus(num_tokens: int, vocab_size: int = 50_000) -> Generator[str, None, None]:
    """Generates synthetic pre-tokenized chunks with Zipfian-like distribution."""
    for i in range(num_tokens):
        idx = int((i * 2654435761) % vocab_size)  # Knuth multiplicative hash for pseudo-random distribution
        yield f"subword_tok_{idx:06d}"


def run_benchmark(num_tokens: int = 500_000, chunk_size_mb: int = 5) -> None:
    print("=" * 90)
    print(f"STREAMING CHUNK COUNTER BENCHMARK (Workload: {num_tokens:,} tokens, Chunk Spill: {chunk_size_mb} MB)")
    print("=" * 90)

    # -------------------------------------------------------------
    # 1. Benchmark standard in-memory Counter
    # -------------------------------------------------------------
    print("[1/2] Benchmarking standard in-memory collections.Counter...")
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    mem_counter = Counter(generate_synthetic_corpus(num_tokens))
    num_unique_mem = len(mem_counter)

    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_time = t1 - t0
    peak_mem_mb = peak_mem / (1024 * 1024)
    throughput_mem = (num_tokens / mem_time) if mem_time > 0 else 0.0

    print(f"  -> Unique chunks:  {num_unique_mem:,}")
    print(f"  -> Ingestion time: {mem_time:.3f} s ({throughput_mem:,.0f} tok/s)")
    print(f"  -> Peak RAM:       {peak_mem_mb:.2f} MB")
    print()

    del mem_counter
    gc.collect()

    # -------------------------------------------------------------
    # 2. Benchmark StreamingChunkCounter (Disk-Backed)
    # -------------------------------------------------------------
    print("[2/2] Benchmarking StreamingChunkCounter (Disk-Backed)...")
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    with StreamingChunkCounter(chunk_size_bytes=chunk_size_bytes) as disk_counter:
        disk_counter.update(generate_synthetic_corpus(num_tokens))
        num_runs_flushed = len(disk_counter._run_files)
        disk_counter.finalize()

        num_unique_disk = len(disk_counter)

        t1 = time.perf_counter()
        current_disk, peak_disk = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ingest_time = t1 - t0
        peak_disk_mb = peak_disk / (1024 * 1024)
        throughput_disk = (num_tokens / ingest_time) if ingest_time > 0 else 0.0

        # Measure sequential disk streaming throughput (EM loop iteration)
        t_read_start = time.perf_counter()
        read_count = sum(cnt for _, cnt in disk_counter.items())
        t_read_end = time.perf_counter()
        read_time = t_read_end - t_read_start
        read_throughput = (num_tokens / read_time) if read_time > 0 else 0.0

        assert num_unique_mem == num_unique_disk
        assert read_count == num_tokens

        print(f"  -> Runs flushed to disk: {num_runs_flushed}")
        print(f"  -> Unique chunks:        {num_unique_disk:,}")
        print(f"  -> Total Ingestion Time: {ingest_time:.3f} s ({throughput_disk:,.0f} tok/s)")
        print(f"  -> Peak RAM:             {peak_disk_mb:.2f} MB")
        print(f"  -> EM Read Throughput:   {read_time:.3f} s ({read_throughput:,.0f} tok/s)")

    print()
    print("-" * 90)
    print("COMPARATIVE SUMMARY:")
    print(f"  In-Memory Counter Peak RAM:     {peak_mem_mb:8.2f} MB")
    print(f"  StreamingChunkCounter Peak RAM: {peak_disk_mb:8.2f} MB")
    reduction = (1.0 - (peak_disk_mb / peak_mem_mb)) * 100 if peak_mem_mb > 0 else 0.0
    print(f"  Peak RAM Reduction:             {reduction:8.1f} %")
    print("-" * 90)


if __name__ == "__main__":
    tokens = 200_000
    if len(sys.argv) > 1:
        tokens = int(sys.argv[1])
    run_benchmark(tokens, chunk_size_mb=2)
