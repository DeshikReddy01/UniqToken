from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from bpe_model import BPEModel
from byte_codec import ByteFallbackEngine


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

        # 2. Greedy merge loop
        while len(vocab) < target_size and rank < max_merges:
            pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

            for word, freq in word_counts.items():
                word_symbols = splits[word]
                for i in range(len(word_symbols) - 1):
                    pair = (word_symbols[i], word_symbols[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Select pair with highest frequency (lexical tie breaker for determinism)
            best_pair = max(
                pair_counts.keys(),
                key=lambda p: (pair_counts[p], p[0] + p[1]),
            )

            if pair_counts[best_pair] < 1:
                break

            # Record merge
            merges[best_pair] = rank
            rank += 1
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)

            # Apply merge to all active splits
            first, second = best_pair
            for word in word_counts:
                word_symbols = splits[word]
                new_symbols: List[str] = []
                i = 0
                while i < len(word_symbols):
                    if i < len(word_symbols) - 1 and word_symbols[i] == first and word_symbols[i + 1] == second:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(word_symbols[i])
                        i += 1
                splits[word] = new_symbols

            if verbose and rank % 100 == 0:
                print(f"[BPE Trainer] Merge {rank:>4}: {best_pair} -> {new_token!r} (Freq: {pair_counts[best_pair]})")

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
