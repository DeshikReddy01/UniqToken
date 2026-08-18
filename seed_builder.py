from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass(frozen=True)
class SeedToken:
    """
    Represents a token in the seed vocabulary.

    is_required: True for special tokens, byte fallbacks, and the base alphabet.
                 These tokens are IMMUNE to pruning during Unigram EM iterations.
    """

    token: str
    frequency: int
    is_required: bool
    source: str  # "special" | "byte" | "alphabet" | "ngram"
    length: int


class SeedVocabularyBuilder:
    """
    Deterministic Seed Vocabulary Builder with Irreducible Floor Validation.
    """

    DEFAULT_SPECIAL_TOKENS = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|endoftext|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]

    def __init__(
        self,
        target_vocab_size: int = 8000,
        seed_multiplier: float = 3.0,
        max_ngram_length: int = 16,
        min_frequency: int = 2,
        byte_fallback: bool = True,
        special_tokens: List[str] | None = None,
        ranking_strategy: str = "char_savings",
    ):
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if seed_multiplier <= 0:
            raise ValueError("seed_multiplier must be greater than zero")
        if max_ngram_length < 1:
            raise ValueError("max_ngram_length must be at least one")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one")
        if ranking_strategy not in {"char_savings", "frequency"}:
            raise ValueError("ranking_strategy must be 'char_savings' or 'frequency'")
        self.target_vocab_size = target_vocab_size
        self.seed_multiplier = seed_multiplier
        self.seed_vocab_size = int(target_vocab_size * seed_multiplier)
        self.max_ngram_length = max_ngram_length
        self.min_frequency = min_frequency
        self.byte_fallback = byte_fallback
        self.special_tokens = special_tokens if special_tokens is not None else list(self.DEFAULT_SPECIAL_TOKENS)
        self.ranking_strategy = ranking_strategy

    def collect_special_tokens(self) -> List[SeedToken]:
        tokens: List[SeedToken] = []
        for token in self.special_tokens:
            tokens.append(
                SeedToken(
                    token=token,
                    frequency=1,
                    is_required=True,
                    source="special",
                    length=len(token),
                )
            )
        return tokens

    def collect_byte_tokens(self) -> List[SeedToken]:
        if not self.byte_fallback:
            return []

        tokens: List[SeedToken] = []
        for b in range(256):
            byte_repr = f"<0x{b:02X}>"
            tokens.append(
                SeedToken(
                    token=byte_repr,
                    frequency=1,
                    is_required=True,
                    source="byte",
                    length=len(byte_repr),
                )
            )
        return tokens

    def collect_base_alphabet(self, chunk_counts: Counter[str]) -> List[SeedToken]:
        char_counts: Counter[str] = Counter()
        for chunk, count in chunk_counts.items():
            for char in chunk:
                char_counts[char] += count

        # Deterministic sorting: frequency descending, then unicode codepoint ascending
        sorted_chars = sorted(char_counts.items(), key=lambda x: (-x[1], x[0]))

        tokens: List[SeedToken] = []
        for char, count in sorted_chars:
            tokens.append(
                SeedToken(
                    token=char,
                    frequency=max(count, 1),
                    is_required=True,
                    source="alphabet",
                    length=1,
                )
            )
        return tokens

    def mine_ngrams(self, chunk_counts: Counter[str]) -> Counter[str]:
        ngram_counts: Counter[str] = Counter()
        max_len = self.max_ngram_length

        for chunk, chunk_freq in chunk_counts.items():
            if chunk in self.special_tokens or (chunk.startswith("<|") and chunk.endswith("|>")):
                continue

            chunk_len = len(chunk)
            for start in range(chunk_len):
                end_limit = min(chunk_len + 1, start + max_len + 1)
                for end in range(start + 1, end_limit):
                    sub = chunk[start:end]
                    ngram_counts[sub] += chunk_freq

        return ngram_counts

    def filter_candidates(self, ngram_counts: Counter[str], protected_tokens: Set[str]) -> Dict[str, int]:
        filtered: Dict[str, int] = {}
        for token, count in ngram_counts.items():
            if token in protected_tokens:
                continue
            if count >= self.min_frequency:
                filtered[token] = count
        return filtered

    def rank_candidates(self, candidate_counts: Dict[str, int]) -> List[str]:
        """
        Deterministic Candidate Ranking:
        - Primary key: Score (character savings or frequency) descending.
        - Secondary key: Frequency descending.
        - Tertiary key: Length descending.
        - Quaternary tie-breaker: Lexicographical string ascending.
        """
        if self.ranking_strategy == "char_savings":
            return sorted(
                candidate_counts.keys(),
                key=lambda t: (
                    -(len(t) - 1) * candidate_counts[t],
                    -candidate_counts[t],
                    -len(t),
                    t,
                ),
            )
        else:
            return sorted(
                candidate_counts.keys(),
                key=lambda t: (-candidate_counts[t], -len(t), t),
            )

    def build_seed_vocab(
        self, pre_tokenized_chunks: Iterable[str], enforce_target_floor: bool = True
    ) -> List[SeedToken]:
        """
        Assembles the complete Seed Vocabulary pool.
        """
        chunk_counts: Counter[str] = Counter(pre_tokenized_chunks)
        seen_tokens: Set[str] = set()
        seed_vocab: List[SeedToken] = []

        # 1. Required Tokens
        for entry in self.collect_special_tokens():
            if entry.token not in seen_tokens:
                seed_vocab.append(entry)
                seen_tokens.add(entry.token)

        for entry in self.collect_byte_tokens():
            if entry.token not in seen_tokens:
                seed_vocab.append(entry)
                seen_tokens.add(entry.token)

        for entry in self.collect_base_alphabet(chunk_counts):
            if entry.token not in seen_tokens:
                seed_vocab.append(entry)
                seen_tokens.add(entry.token)

        num_required = len(seed_vocab)

        # 2. Floor Validation
        if enforce_target_floor and self.target_vocab_size < num_required:
            raise ValueError(
                f"target_vocab_size ({self.target_vocab_size}) is smaller than the required token floor ({num_required}). "
                f"Set target_vocab_size >= {num_required} or disable byte_fallback / reduce special tokens."
            )

        # 3. Mine, Filter, and Rank Candidates
        raw_ngrams = self.mine_ngrams(chunk_counts)
        filtered_candidates = self.filter_candidates(raw_ngrams, seen_tokens)
        ranked_candidates = self.rank_candidates(filtered_candidates)

        # 4. Fill Seed Capacity
        candidate_budget = max(0, self.seed_vocab_size - len(seed_vocab))
        for token in ranked_candidates[:candidate_budget]:
            if token in seen_tokens:
                continue

            seed_vocab.append(
                SeedToken(
                    token=token,
                    frequency=filtered_candidates[token],
                    is_required=False,
                    source="ngram",
                    length=len(token),
                )
            )
            seen_tokens.add(token)

        return seed_vocab

    def stats(self, vocabulary: List[SeedToken]) -> dict:
        required = [t for t in vocabulary if t.is_required]
        candidates = [t for t in vocabulary if not t.is_required]
        return {
            "total": len(vocabulary),
            "required": len(required),
            "candidates": len(candidates),
            "target_vocab_size": self.target_vocab_size,
            "seed_vocab_size": self.seed_vocab_size,
            "max_ngram_length": self.max_ngram_length,
            "min_frequency": self.min_frequency,
            "sources": Counter(t.source for t in vocabulary),
        }
