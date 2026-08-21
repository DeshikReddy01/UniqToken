from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from seed_builder import SeedToken, SeedVocabularyBuilder
from trie import PrefixTrie
from unigram_lattice import UnigramLattice
from byte_codec import ByteFallbackEngine

try:
    import caliper_core

    _HAS_CALIPER_CORE = True
except ImportError:
    _HAS_CALIPER_CORE = False


@dataclass
class UnigramModel:
    """
    Trained Unigram Tokenizer Model.
    """

    vocab: Dict[str, float]  # token -> log_probability
    token_to_id: Dict[str, int]  # token -> integer ID
    id_to_token: Dict[int, str]  # integer ID -> token
    special_tokens: List[str]
    max_subword_len: int = 16
    byte_fallback: bool = True

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _cache_signature(self) -> Tuple[int, int]:
        # ponytail: assumes vocab dict replaced not mutated in-place; add version counter if external mutation needed
        return (id(self.vocab), len(self.vocab))

    def _sync_cache(self) -> None:
        """Invalidates caches if the vocab object or size changed since they were built."""
        sig = self._cache_signature()
        if self.__dict__.get("_cache_sig") != sig:
            self.clear_cache()
            self._cache_sig = sig

    def _get_trie(self) -> PrefixTrie:
        """Builds (and caches) a PrefixTrie over the current vocab for fast lattice search."""
        self._sync_cache()
        trie = self.__dict__.get("_trie")
        if trie is None:
            trie = PrefixTrie.from_vocab(self.vocab)
            self._trie = trie
        return trie

    def _get_rust_trie(self) -> Optional[Any]:
        """Builds (and caches) a native RustPrefixTrie if caliper_core is available."""
        if not _HAS_CALIPER_CORE:
            return None
        self._sync_cache()
        rust_trie = self.__dict__.get("_rust_trie")
        if rust_trie is None:
            rust_trie = caliper_core.RustPrefixTrie()
            for token, log_p in self.vocab.items():
                tid = self.token_to_id.get(token)
                rust_trie.insert(token, log_p, tid)
            self._rust_trie = rust_trie
        return rust_trie

    _FAST_PATH_MAX_LEN = 6
    _MAX_CACHE_SIZE = 10000

    def clear_cache(self) -> None:
        """Clears cached PrefixTrie and segmentation memoization tables."""
        if "_trie" in self.__dict__:
            del self.__dict__["_trie"]
        if "_rust_trie" in self.__dict__:
            del self.__dict__["_rust_trie"]
        if "_seg_cache" in self.__dict__:
            del self.__dict__["_seg_cache"]
        if "_cache_sig" in self.__dict__:
            del self.__dict__["_cache_sig"]

    def _get_seg_cache(self) -> Dict[str, List[Tuple[str, int, int]]]:
        self._sync_cache()
        cache = self.__dict__.get("_seg_cache")
        if cache is None:
            cache = {}
            self._seg_cache = cache
        return cache

    def _encode_fast(self, text: str) -> Optional[List[Tuple[str, int, int]]]:
        """
        Exact Viterbi 1-best for short chunks without constructing a lattice object.

        Reproduces UnigramLattice edge semantics (including byte-fallback
        only-when-no-vocab-edge and longest-edge-first tie-breaking). Returns
        (token, start, end) triples, or None to defer to the full lattice.
        """
        length = len(text)
        if length < 2 or length > self._FAST_PATH_MAX_LEN:
            return None

        max_len = min(self.max_subword_len, length)
        neg_inf = float("-inf")
        dp = [neg_inf] * (length + 1)
        back: List[Optional[Tuple[int, List[str]]]] = [None] * (length + 1)
        dp[0] = 0.0

        for i in range(1, length + 1):
            best_score = neg_inf
            best_edge: Optional[Tuple[int, List[str]]] = None
            start = max(0, i - max_len)
            for j in range(start, i):
                token = text[j:i]
                log_p = self.vocab.get(token)
                if log_p is not None and dp[j] > neg_inf:
                    score = dp[j] + log_p
                    if score > best_score:
                        best_score = score
                        best_edge = (j, [token])

            # Byte fallback edge from (i - 1) to i applies if no vocab edge starts at (i - 1)
            if self.byte_fallback and dp[i - 1] > neg_inf:
                has_vocab_edge = False
                for end in range(i, min(length + 1, i - 1 + max_len + 1)):
                    if text[i - 1 : end] in self.vocab:
                        has_vocab_edge = True
                        break
                if not has_vocab_edge:
                    char = text[i - 1]
                    byte_tokens = ByteFallbackEngine.char_to_byte_tokens(char)
                    log_p = sum(self.vocab.get(b, -UnigramLattice.DEFAULT_BYTE_PENALTY) for b in byte_tokens)
                    score = dp[i - 1] + log_p
                    if score > best_score:
                        best_score = score
                        best_edge = (i - 1, byte_tokens)

            if best_edge is None:
                return None  # disconnected; defer to the full lattice

            dp[i] = best_score
            back[i] = best_edge

        result: List[Tuple[str, int, int]] = []
        i = length
        while i > 0:
            edge = back[i]
            assert edge is not None
            j, tokens = edge
            for token in reversed(tokens):
                result.append((token, j, i))
            i = j
        result.reverse()
        return result

    def encode(self, text: str) -> List[str]:
        if len(text) == 1 and text in self.vocab:
            return [text]
        spans = self.encode_with_spans(text)
        return [token for token, _, _ in spans]

    def encode_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Encode text and retain normalized character spans for every output token."""
        if len(text) == 1 and text in self.vocab:
            return [(text, 0, 1)]

        cache = self._get_seg_cache()
        cached = cache.get(text)
        if cached is not None:
            return list(cached)

        # 1. Native Rust Viterbi engine dispatch (if caliper_core compiled)
        rust_trie = self._get_rust_trie()
        if rust_trie is not None:
            try:
                rust_spans = caliper_core.rust_viterbi_decode(
                    text,
                    rust_trie,
                    self.byte_fallback,
                )
                spans = [(s.token, s.start, s.end) for s in rust_spans]
                if len(text) <= 64 and len(cache) < self._MAX_CACHE_SIZE:
                    cache[text] = spans
                return spans
            except Exception:
                pass

        # 2. Pure Python fast path or full lattice DAG
        fast = self._encode_fast(text)
        if fast is not None:
            spans = fast
        else:
            lattice = UnigramLattice(
                text,
                self.vocab,
                max_subword_len=self.max_subword_len,
                byte_fallback=self.byte_fallback,
                trie=self._get_trie(),
            )
            edges, _ = lattice.viterbi_edges()
            spans = [(token, edge.start, edge.end) for edge in edges for token in edge.tokens]

        if len(text) <= 64 and len(cache) < self._MAX_CACHE_SIZE:
            cache[text] = spans
        return spans

    def sample(self, text: str, alpha: float = 0.5) -> List[str]:
        if len(text) == 1 and text in self.vocab:
            return [text]
        lattice = UnigramLattice(
            text,
            self.vocab,
            max_subword_len=self.max_subword_len,
            byte_fallback=self.byte_fallback,
            trie=self._get_trie(),
        )
        return lattice.sample(alpha=alpha)

    def encode_to_ids(self, text: str) -> List[int]:
        tokens = self.encode(text)
        unk_id = self.token_to_id.get("<|unk|>", 0)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def sample_to_ids(self, text: str, alpha: float = 0.5) -> List[int]:
        tokens = self.sample(text, alpha=alpha)
        unk_id = self.token_to_id.get("<|unk|>", 0)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, token_ids: List[int], space_char: str = "\u2581") -> str:
        from byte_codec import ByteFallbackEngine

        tokens = [self.id_to_token.get(i, "<|unk|>") for i in token_ids]
        return ByteFallbackEngine.decode_tokens(tokens, space_char=space_char)


class UnigramTrainer:
    """
    Expectation-Maximization (EM) & Iterative Likelihood Pruning Trainer for Unigram Tokenizer.
    """

    def __init__(
        self,
        target_vocab_size: int = 8000,
        seed_multiplier: float = 3.0,
        max_ngram_length: int = 16,
        min_frequency: int = 2,
        byte_fallback: bool = True,
        prune_rate: float = 0.20,
        em_sub_iterations: int = 2,
        special_tokens: List[str] | None = None,
        ranking_strategy: str = "char_savings",
        adaptive_multiplier: bool = False,
        max_edges_per_node: Optional[int] = None,
        min_edge_log_prob: Optional[float] = None,
        convergence_tolerance: float = 1e-4,
        script_balance_temperature: Optional[float] = None,
        min_boundary_entropy: Optional[float] = None,
    ):
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if seed_multiplier <= 0:
            raise ValueError("seed_multiplier must be greater than zero")
        if max_ngram_length < 1:
            raise ValueError("max_ngram_length must be at least one")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one")
        if not 0 < prune_rate <= 1:
            raise ValueError("prune_rate must be in the interval (0, 1]")
        if em_sub_iterations < 1:
            raise ValueError("em_sub_iterations must be at least one")
        self.target_vocab_size = target_vocab_size
        self.seed_multiplier = seed_multiplier
        self.max_ngram_length = max_ngram_length
        self.min_frequency = min_frequency
        self.byte_fallback = byte_fallback
        self.prune_rate = prune_rate
        self.em_sub_iterations = em_sub_iterations
        self.special_tokens = special_tokens
        self.ranking_strategy = ranking_strategy
        self.adaptive_multiplier = adaptive_multiplier
        self.max_edges_per_node = max_edges_per_node
        self.min_edge_log_prob = min_edge_log_prob
        self.convergence_tolerance = convergence_tolerance
        self.script_balance_temperature = script_balance_temperature
        self.min_boundary_entropy = min_boundary_entropy

    def train(self, pre_tokenized_chunks: Iterable[str], verbose: bool = True) -> UnigramModel:
        """
        Runs the full EM training and pruning loop with convergence checks and beam pruning.
        """
        # Step 1: Pre-aggregate chunk frequencies
        chunk_counts = Counter(pre_tokenized_chunks)

        # Step 2: Build Seed Vocabulary
        seed_builder = SeedVocabularyBuilder(
            target_vocab_size=self.target_vocab_size,
            seed_multiplier=self.seed_multiplier,
            max_ngram_length=self.max_ngram_length,
            min_frequency=self.min_frequency,
            byte_fallback=self.byte_fallback,
            special_tokens=self.special_tokens,
            ranking_strategy=self.ranking_strategy,
            adaptive_multiplier=self.adaptive_multiplier,
            script_balance_temperature=self.script_balance_temperature,
            min_boundary_entropy=self.min_boundary_entropy,
        )

        seed_tokens: List[SeedToken] = seed_builder.build_seed_vocab(chunk_counts)
        required_tokens: Set[str] = {t.token for t in seed_tokens if t.is_required}

        # Step 3: Initialize log probabilities from seed counts
        total_seed_freq = sum(t.frequency for t in seed_tokens)
        current_vocab_log_probs: Dict[str, float] = {
            t.token: math.log(max(t.frequency, 1) / total_seed_freq) for t in seed_tokens
        }

        if verbose:
            print(
                f"[EM Trainer] Seed Vocab Size: {len(current_vocab_log_probs)} (Target: {self.target_vocab_size}, Required: {len(required_tokens)})"
            )

        round_num = 1

        # Step 4: Iterative EM Optimization & Pruning Loop
        while len(current_vocab_log_probs) > self.target_vocab_size:
            # --- E-STEP & M-STEP SUB-ITERATIONS ---
            prev_log_lik = -float("inf")
            for sub_iter in range(self.em_sub_iterations):
                expected_counts: Dict[str, float] = {}
                total_corpus_log_lik = 0.0
                trie = PrefixTrie.from_vocab(current_vocab_log_probs)

                for chunk, count in chunk_counts.items():
                    # Skip special tokens from lattice segmentation
                    if chunk in required_tokens and (chunk.startswith("<|") and chunk.endswith("|>")):
                        expected_counts[chunk] = expected_counts.get(chunk, 0.0) + count
                        continue

                    lattice = UnigramLattice(
                        text=chunk,
                        vocab_log_probs=current_vocab_log_probs,
                        max_subword_len=self.max_ngram_length,
                        byte_fallback=self.byte_fallback,
                        trie=trie,
                        max_edges_per_node=self.max_edges_per_node,
                        min_edge_log_prob=self.min_edge_log_prob,
                    )

                    chunk_exp, chunk_log_lik = lattice.forward_backward()
                    total_corpus_log_lik += chunk_log_lik * count

                    for tok, exp_val in chunk_exp.items():
                        expected_counts[tok] = expected_counts.get(tok, 0.0) + (exp_val * count)

                # Check convergence
                delta_log_lik = abs(total_corpus_log_lik - prev_log_lik)
                prev_log_lik = total_corpus_log_lik

                # M-Step: Update token log probabilities
                total_expected = sum(expected_counts.values())
                if total_expected <= 0:
                    total_expected = 1.0
                current_vocab_log_probs = {
                    tok: math.log(max(expected_counts.get(tok, 1e-12) / total_expected, 1e-12))
                    for tok in current_vocab_log_probs
                }

                if sub_iter > 0 and delta_log_lik < self.convergence_tolerance:
                    break

            # --- PRUNING STEP ---
            current_size = len(current_vocab_log_probs)
            if current_size <= self.target_vocab_size:
                break

            excess = current_size - self.target_vocab_size
            num_to_prune = max(1, int(excess * self.prune_rate))
            # Candidates are scored by their contribution to corpus likelihood
            # Score(t) = E[count(t)] * log p(t). Scores closest to zero have
            # the smallest expected effect and are pruned first.
            candidate_scores: List[Tuple[str, float]] = []

            for tok, log_p in current_vocab_log_probs.items():
                if tok in required_tokens:
                    continue  # Required tokens are immune to pruning
                exp_c = expected_counts.get(tok, 0.0)
                # Score measures entropy reduction from having this token
                score = exp_c * log_p
                candidate_scores.append((tok, score))

            # Deterministic: remove the least negative scores before valuable
            # high-count tokens whose contribution is strongly negative.
            candidate_scores.sort(key=lambda x: (-x[1], x[0]))

            # Prune lowest candidates
            tokens_to_remove = set(tok for tok, _ in candidate_scores[:num_to_prune])

            new_vocab_log_probs: Dict[str, float] = {
                tok: log_p for tok, log_p in current_vocab_log_probs.items() if tok not in tokens_to_remove
            }

            current_vocab_log_probs = new_vocab_log_probs

            if verbose:
                print(
                    f"[EM Round {round_num:>2}] Vocab: {len(current_vocab_log_probs):>5} | Pruned: {len(tokens_to_remove):>4} | Corpus LogLik: {total_corpus_log_lik:.2f}"
                )

            round_num += 1

        # Step 5: Final Probability Re-normalization
        total_p = sum(math.exp(log_p) for log_p in current_vocab_log_probs.values())
        final_vocab = {tok: math.log(math.exp(log_p) / total_p) for tok, log_p in current_vocab_log_probs.items()}

        # Step 6: Build Integer Token IDs
        # Sort tokens deterministically: special tokens first, then bytes, then alphabet/subwords
        sorted_tokens = sorted(
            final_vocab.keys(),
            key=lambda t: (
                0 if t.startswith("<|") and t.endswith("|>") else (1 if t.startswith("<0x") else 2),
                -len(t),
                t,
            ),
        )

        token_to_id = {tok: idx for idx, tok in enumerate(sorted_tokens)}
        id_to_token = {idx: tok for idx, tok in enumerate(sorted_tokens)}

        return UnigramModel(
            vocab=final_vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=list(seed_builder.special_tokens),
            max_subword_len=self.max_ngram_length,
            byte_fallback=self.byte_fallback,
        )
