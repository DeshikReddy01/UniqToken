from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from byte_codec import ByteFallbackEngine
from unigram_trainer import UnigramModel


class CrossEntropyMerging:
    """
    Cross-Entropy Merging (CEM) post-training vocabulary optimizer.

    Extends an already-trained Unigram model by greedily adding merged tokens
    whose introduction least increases corpus cross-entropy (Gee et al., 2024).
    Existing token IDs are preserved, so downstream embedding weights remain
    valid; the additions are pure vocabulary growth chosen by loss impact.

    The score of merging adjacent tokens ``(a, b)`` into ``a + b`` is::

        f * (log P(a) + log P(b) - log(f / N))

    where ``f`` is the pair's corpus count and ``N`` the total number of
    adjacent token pairs. A negative score means the merge reduces expected
    loss; the pair with the lowest score is merged first and the corpus token
    stream is updated incrementally (so newly merged tokens can participate in
    further merges, mirroring BPE's greedy hierarchy).

    With ``cross_word=True`` the optimizer becomes a SuperBPE pass ("space
    travel"): the corpus is treated as one continuous token stream (instead of
    resetting at chunk boundaries) and only merges whose result contains the
    space character are accepted, producing tokens such as ``the\u2581``,
    ``\u2581quick`` and ``the\u2581quick`` that span word boundaries.
    """

    def __init__(
        self,
        max_merges: int = 200,
        max_score: float = 0.0,
        verbose: bool = False,
        cross_word: bool = False,
        space_char: str = "\u2581",
    ):
        if max_merges < 0:
            raise ValueError("max_merges must not be negative")
        self.max_merges = max_merges
        self.max_score = max_score
        self.verbose = verbose
        self.cross_word = cross_word
        self.space_char = space_char
        self.merges: List[Tuple[str, str, str, float, int]] = []

    def optimize(self, model: UnigramModel, chunks: Iterable[str]) -> UnigramModel:
        """Returns a new model with CEM/SuperBPE-merged tokens; IDs of existing tokens are unchanged."""
        if self.max_merges == 0:
            return model

        special_tokens = set(model.special_tokens)
        byte_pattern = ByteFallbackEngine.BYTE_TOKEN_PATTERN
        max_len = model.max_subword_len
        new_probs: Dict[str, float] = {}
        new_count: Dict[str, int] = {}

        def log_prob(token: str) -> float:
            lp = model.vocab.get(token)
            if lp is not None:
                return lp
            return new_probs[token]

        def mergeable(token: str) -> bool:
            if token in special_tokens:
                return False
            if byte_pattern.match(token):
                return False
            return token in model.vocab or token in new_probs

        if self.cross_word:
            streams: List[List[str]] = [[tok for chunk in chunks if chunk for tok in model.encode(chunk)]]
        else:
            streams = [model.encode(chunk) for chunk in chunks if chunk]

        for _ in range(self.max_merges):
            pair_counts: Counter = Counter()
            for stream in streams:
                for a, b in zip(stream, stream[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            total_pairs = sum(pair_counts.values())

            best_pair: Tuple[str, str, str] | None = None
            best_score = float("inf")
            best_log_p = 0.0
            for (a, b), f in pair_counts.items():
                if not mergeable(a) or not mergeable(b):
                    continue
                merged = a + b
                if self.cross_word and self.space_char not in merged:
                    continue
                if len(merged) > max_len or merged in model.vocab or merged in new_probs or merged in special_tokens:
                    continue
                log_p_hat = math.log(f / total_pairs)
                score = f * (log_prob(a) + log_prob(b) - log_p_hat)
                if score < best_score:
                    best_score = score
                    best_pair = (a, b, merged)
                    best_log_p = log_p_hat

            if best_pair is None or best_score >= self.max_score:
                break

            a, b, merged = best_pair
            pair_count = pair_counts[(a, b)]
            self.merges.append((a, b, merged, best_score, pair_count))
            new_probs[merged] = best_log_p
            new_count[merged] = pair_count

            for stream in streams:
                merged_stream: List[str] = []
                i = 0
                n = len(stream)
                while i < n:
                    if i < n - 1 and stream[i] == a and stream[i + 1] == b:
                        merged_stream.append(merged)
                        i += 2
                    else:
                        merged_stream.append(stream[i])
                        i += 1
                stream[:] = merged_stream

            if self.verbose:
                label = "SuperBPE" if self.cross_word else "CEM"
                print(
                    f"[{label}] Merge {len(self.merges):>4}: "
                    f"{a!r} + {b!r} -> {merged!r} "
                    f"(freq={pair_count}, score={best_score:.3f})"
                )

        if not new_probs:
            return model

        # Re-normalize the probability distribution over old + new tokens.
        probs: Dict[str, float] = {tok: math.exp(lp) for tok, lp in model.vocab.items()}
        for tok, lp in new_probs.items():
            probs[tok] = math.exp(lp)
        total_p = sum(probs.values())
        updated_vocab = {tok: math.log(p / total_p) for tok, p in probs.items()}

        token_to_id = dict(model.token_to_id)
        id_to_token = dict(model.id_to_token)
        next_id = len(token_to_id)
        for tok in new_probs:
            token_to_id[tok] = next_id
            id_to_token[next_id] = tok
            next_id += 1

        return UnigramModel(
            vocab=updated_vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=list(model.special_tokens),
            max_subword_len=max_len,
            byte_fallback=model.byte_fallback,
        )
