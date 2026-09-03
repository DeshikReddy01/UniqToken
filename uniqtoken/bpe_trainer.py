from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .bpe_model import BPEModel
from .byte_codec import ByteFallbackEngine


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

    def train(self, chunks: List[str], verbose: bool = False) -> BPEModel:
        """
        Trains BPE merge ranks and vocabulary from pre-tokenized chunks.
        """
        # 1. Base vocabulary initialization
        vocab: Set[str] = set(self.special_tokens)
        if self.byte_fallback:
            for b in range(256):
                vocab.add(ByteFallbackEngine.byte_to_token(b))

        # Count word frequencies and represent words as tuple of characters
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

        import heapq

        # 2. Inverted index and Max-Heap for O(1) merge extraction
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for word, freq in word_counts.items():
            syms = splits[word]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)

        # Build initial max heap (-freq, word_concat, pair)
        heap = [(-freq, p[0] + p[1], p) for p, freq in pair_counts.items() if freq > 0]
        heapq.heapify(heap)

        while len(vocab) < target_size and rank < max_merges:
            best_pair = None
            while heap:
                neg_f, _, p = heapq.heappop(heap)
                cur_f = pair_counts.get(p, 0)
                if cur_f <= 0:
                    continue
                if cur_f != -neg_f:
                    # Count drifted since entry was pushed; re-insert with current
                    # frequency so the pair remains an active merge candidate.
                    heapq.heappush(heap, (-cur_f, p[0] + p[1], p))
                    continue
                best_pair = p
                break

            if best_pair is None or pair_counts[best_pair] < 1:
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
                    heapq.heappush(heap, (-pair_counts[p], p[0] + p[1], p))

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
