from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Dict, List, Optional, Set, Tuple, Union

from .bpe_model import BPEModel
from .byte_codec import ByteFallbackEngine


class FlatBucketQueue:
    """Integer-frequency priority queue (Dial's flat buckets) for BPE pairs.

    Pairs are grouped into buckets indexed by their integer frequency. Moving
    a pair between frequency buckets is O(1) via hash set operations, and
    ``pop_max`` advances a high-water mark pointer across non-empty buckets in
    O(1) amortized time. Deterministic tie-breaking among pairs sharing the
    same frequency is maintained with a per-bucket min-heap in O(log B) time,
    where B is the number of tied pairs at that frequency.

    Stale tombstones cannot accumulate: a pair only ever lives in the single
    bucket matching its live count, and stale heap entries are discarded
    lazily when popped.
    """

    def __init__(self) -> None:
        self._buckets: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)
        self._bucket_heaps: Dict[int, List[Tuple[str, Tuple[str, str]]]] = defaultdict(list)
        self._counts: Dict[Tuple[str, str], int] = {}
        self._max_freq: int = 0

    def add(self, pair: Tuple[str, str], freq: int) -> None:
        """Insert ``pair`` with an initial positive frequency.

        If ``pair`` already exists, its previous bucket membership is
        discarded before updating to the new frequency, so a pair is never
        present in two buckets at once.
        """
        if freq <= 0:
            return
        old_freq = self._counts.get(pair)
        if old_freq is not None and old_freq in self._buckets:
            self._buckets[old_freq].discard(pair)
        self._counts[pair] = freq
        self._buckets[freq].add(pair)
        heapq.heappush(self._bucket_heaps[freq], (pair[0] + pair[1], pair))
        if freq > self._max_freq:
            self._max_freq = freq

    def update(self, pair: Tuple[str, str], delta: int) -> None:
        """Adjust ``pair``'s frequency by ``delta``, moving it between buckets.

        Pairs whose new frequency is zero or negative are dropped entirely, so
        no stale entries are ever left behind. Updating an unknown pair with a
        positive delta introduces it; a negative delta is a no-op.
        """
        old_freq = self._counts.get(pair, 0)
        new_freq = old_freq + delta
        if old_freq > 0 and old_freq in self._buckets:
            self._buckets[old_freq].discard(pair)
        if new_freq > 0:
            self._counts[pair] = new_freq
            self._buckets[new_freq].add(pair)
            heapq.heappush(self._bucket_heaps[new_freq], (pair[0] + pair[1], pair))
            if new_freq > self._max_freq:
                self._max_freq = new_freq
        else:
            self._counts.pop(pair, None)

    def pop_max(self) -> Optional[Tuple[str, str]]:
        """Remove and return the pair with the highest frequency, or None.

        The per-bucket heap is not mutated on ``update``/``remove`` (that would
        require O(B) deletion); entries are instead validated lazily when
        popped, so stale heap entries for pairs that moved or were removed are
        simply skipped.
        """
        while self._max_freq > 0:
            bucket = self._buckets[self._max_freq]
            if not bucket:
                self._buckets.pop(self._max_freq, None)
                self._bucket_heaps.pop(self._max_freq, None)
                self._max_freq -= 1
                continue
            heap = self._bucket_heaps[self._max_freq]
            while heap:
                _, pair = heapq.heappop(heap)
                if pair in bucket:
                    bucket.remove(pair)
                    self._counts.pop(pair, None)
                    if not bucket:
                        self._buckets.pop(self._max_freq, None)
                        self._bucket_heaps.pop(self._max_freq, None)
                        self._max_freq -= 1
                    return pair
            # Heap exhausted without finding a live member: the bucket holds
            # only stale entries, so drop it and keep scanning downward.
            self._buckets.pop(self._max_freq, None)
            self._bucket_heaps.pop(self._max_freq, None)
            self._max_freq -= 1
        return None

    def get_count(self, pair: Tuple[str, str]) -> int:
        """Current frequency of ``pair`` (0 when absent)."""
        return self._counts.get(pair, 0)

    def remove(self, pair: Tuple[str, str]) -> None:
        """Remove ``pair`` from the queue regardless of its frequency."""
        freq = self._counts.pop(pair, None)
        if freq and freq in self._buckets:
            self._buckets[freq].discard(pair)

    def __len__(self) -> int:
        return len(self._counts)


class BPETrainer:
    """
    Byte-Pair Encoding (BPE) Model Trainer.

    Mines adjacent symbol pair co-occurrences and constructs an optimal merge table
    up to target_vocab_size or num_merges.
    """

    def __init__(
        self,
        target_vocab_size: Optional[int] = None,
        num_merges: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
        byte_fallback: bool = True,
    ):
        if target_vocab_size is None and num_merges is None:
            target_vocab_size = 1000
        if target_vocab_size is not None and target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must not be negative")
        self.target_vocab_size = target_vocab_size
        self.num_merges = num_merges
        self.special_tokens = list(special_tokens or ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"])
        self.byte_fallback = byte_fallback

    def train(self, chunks: Union[List[str], Iterable[str], Mapping[str, int]], verbose: bool = False) -> BPEModel:
        """
        Trains BPE merge ranks and vocabulary from pre-tokenized chunks.
        """
        # 1. Base vocabulary initialization
        vocab: Set[str] = set(self.special_tokens)
        if self.byte_fallback:
            for b in range(256):
                vocab.add(ByteFallbackEngine.byte_to_token(b))

        # Count word frequencies and represent words as tuple of characters
        if isinstance(chunks, Mapping):
            word_counts = chunks
        else:
            word_counts = Counter(chunks)
        splits: Dict[str, List[str]] = {}
        for word in word_counts:
            char_list = list(word)
            splits[word] = char_list
            for c in char_list:
                vocab.add(c)

        if self.target_vocab_size is not None and len(vocab) > self.target_vocab_size:
            raise ValueError(
                f"target_vocab_size ({self.target_vocab_size}) is smaller than "
                f"the required initial vocabulary ({len(vocab)})"
            )

        merges: Dict[Tuple[str, str], int] = {}
        rank = 0

        target_size = self.target_vocab_size if self.target_vocab_size is not None else float("inf")
        max_merges = self.num_merges if self.num_merges is not None else float("inf")

        # 2. Inverted index and flat-bucket priority queue for O(1) merge extraction
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for word, freq in word_counts.items():
            syms = splits[word]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)

        # Build the initial frequency-bucketed queue. Every pair lives in the
        # bucket for its current count, so no stale tombstones accumulate.
        queue = FlatBucketQueue()
        for p, freq in pair_counts.items():
            if freq > 0:
                queue.add(p, freq)

        while len(vocab) < target_size and rank < max_merges:
            best_pair = queue.pop_max()
            if best_pair is None or pair_counts.get(best_pair, 0) < 1:
                break

            new_token = best_pair[0] + best_pair[1]
            if new_token in vocab:
                # Two different pairs can concat to the same string ("a"+"bc" vs
                # "ab"+"c"); recording the merge would burn a rank without growing
                # the vocab, so drop the pair instead of merging it.
                pair_counts.pop(best_pair, None)
                pair_to_words.pop(best_pair, None)
                continue

            # Record merge
            merges[best_pair] = rank
            rank += 1
            vocab.add(new_token)

            first, second = best_pair
            affected_words = list(pair_to_words.get(best_pair, set()))

            for word in affected_words:
                old_syms = splits[word]
                freq = word_counts[word]

                # Decrement old pairs
                for i in range(len(old_syms) - 1):
                    p = (old_syms[i], old_syms[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_to_words[p].discard(word)
                    queue.update(p, -freq)

                # Form new symbols
                new_syms: List[str] = []
                i = 0
                while i < len(old_syms):
                    if i < len(old_syms) - 1 and old_syms[i] == first and old_syms[i + 1] == second:
                        new_syms.append(new_token)
                        i += 2
                    else:
                        new_syms.append(old_syms[i])
                        i += 1
                splits[word] = new_syms

                # Increment new pairs
                for i in range(len(new_syms) - 1):
                    p = (new_syms[i], new_syms[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(word)
                    queue.update(p, +freq)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            if verbose and rank % 500 == 0:
                print(f"[BPE Trainer] Merge {rank:>5}: {best_pair} -> {new_token!r} | Vocab: {len(vocab):,}")

        # 3. Build token-to-id mapping
        token_to_id: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}

        curr_id = 0
        for st in self.special_tokens:
            if st not in token_to_id:
                token_to_id[st] = curr_id
                id_to_token[curr_id] = st
                curr_id += 1

        for tok in sorted(vocab):
            if tok not in token_to_id:
                token_to_id[tok] = curr_id
                id_to_token[curr_id] = tok
                curr_id += 1

        return BPEModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            special_tokens=self.special_tokens,
            byte_fallback=self.byte_fallback,
        )
