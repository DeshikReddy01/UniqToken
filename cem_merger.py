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
        min_pmi: Optional[float] = None,
    ):
        if max_merges < 0:
            raise ValueError("max_merges must not be negative")
        self.max_merges = max_merges
        self.max_score = max_score
        self.verbose = verbose
        self.cross_word = cross_word
        self.space_char = space_char
        self.min_pmi = min_pmi
        self.merges: List[Tuple[str, str, str, float, int]] = []

    def optimize(self, model: UnigramModel, chunks: Iterable[str]) -> UnigramModel:
        """Returns a new model with CEM/SuperBPE-merged tokens; IDs of existing tokens are unchanged."""
        self.merges.clear()
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

        from collections import defaultdict

        unique_chunks = set(chunk for chunk in chunks if chunk)
        chunk_enc_map = {chunk: model.encode(chunk) for chunk in unique_chunks}

        if self.cross_word:
            streams: List[List[str]] = []
            cur_stream: List[str] = []
            for chunk in chunks:
                if not chunk:
                    continue
                cur_stream.extend(chunk_enc_map[chunk])
                if len(cur_stream) >= 200:
                    streams.append(cur_stream)
                    cur_stream = []
            if cur_stream:
                streams.append(cur_stream)
        else:
            streams = [chunk_enc_map[chunk] for chunk in chunks if chunk]

        # 1. Build initial inverted pair index
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_to_streams: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        total_pairs = 0

        for s_idx, stream in enumerate(streams):
            for i in range(len(stream) - 1):
                p = (stream[i], stream[i + 1])
                pair_counts[p] += 1
                pair_to_streams[p].add(s_idx)
                total_pairs += 1

        import heapq

        def compute_pair_score(a: str, b: str, f: int, tot: int) -> Tuple[float, float, str]:
            log_p_hat = math.log(f / max(tot, 1))
            score = f * (log_prob(a) + log_prob(b) - log_p_hat)
            return score, log_p_hat, a + b

        heap: List[Tuple[float, int, float, Tuple[str, str]]] = []
        for (a, b), f in pair_counts.items():
            if f < 2:
                continue
            if self.cross_word and (self.space_char not in a and self.space_char not in b):
                continue
            if not mergeable(a) or not mergeable(b):
                continue
            m = a + b
            if len(m) > max_len or m in model.vocab or m in new_probs or m in special_tokens:
                continue
            sc, lp_hat, _ = compute_pair_score(a, b, f, total_pairs)
            if self.min_pmi is not None:
                pmi = (lp_hat - (log_prob(a) + log_prob(b))) / math.log(2)
                if pmi < self.min_pmi:
                    continue
            if sc < self.max_score:
                heap.append((sc, f, lp_hat, (a, b)))
        heapq.heapify(heap)

        for _ in range(self.max_merges):
            if not pair_counts or total_pairs <= 0:
                break

            best_pair: Tuple[str, str, str] | None = None
            best_score = float("inf")
            best_log_p = 0.0

            while heap:
                sc, f_in_heap, lp_hat, (a, b) = heapq.heappop(heap)
                cur_f = pair_counts.get((a, b), 0)
                if cur_f >= 2 and cur_f == f_in_heap:
                    if not mergeable(a) or not mergeable(b):
                        continue
                    m = a + b
                    if len(m) > max_len or m in model.vocab or m in new_probs or m in special_tokens:
                        continue
                    sc, lp_hat, _ = compute_pair_score(a, b, cur_f, total_pairs)
                    if self.min_pmi is not None:
                        pmi = (lp_hat - (log_prob(a) + log_prob(b))) / math.log(2)
                        if pmi < self.min_pmi:
                            continue
                    if sc < self.max_score:
                        best_score = sc
                        best_pair = (a, b, m)
                        best_log_p = lp_hat
                        break

            if best_pair is None or best_score >= self.max_score:
                break

            a, b, merged = best_pair
            pair_count = pair_counts[(a, b)]
            self.merges.append((a, b, merged, best_score, pair_count))
            new_probs[merged] = best_log_p
            new_count[merged] = pair_count

            # Incremental update on affected streams only
            affected_streams = list(pair_to_streams.get((a, b), set()))
            for s_idx in affected_streams:
                stream = streams[s_idx]
                old_len = len(stream)

                # Decrement old pairs
                for i in range(old_len - 1):
                    p = (stream[i], stream[i + 1])
                    pair_counts[p] -= 1
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_to_streams[p].discard(s_idx)
                total_pairs -= (old_len - 1)

                # Form new stream
                new_stream: List[str] = []
                i = 0
                n = len(stream)
                while i < n:
                    if i < n - 1 and stream[i] == a and stream[i + 1] == b:
                        new_stream.append(merged)
                        i += 2
                    else:
                        new_stream.append(stream[i])
                        i += 1
                streams[s_idx] = new_stream

                # Increment new pairs
                for i in range(len(new_stream) - 1):
                    p = (new_stream[i], new_stream[i + 1])
                    pair_counts[p] += 1
                    pair_to_streams[p].add(s_idx)
                    a_p, b_p = p
                    if pair_counts[p] >= 2 and (not self.cross_word or self.space_char in a_p or self.space_char in b_p):
                        if mergeable(a_p) and mergeable(b_p):
                            sc, lp_hat, _ = compute_pair_score(a_p, b_p, pair_counts[p], total_pairs)
                            if sc < self.max_score:
                                heapq.heappush(heap, (sc, pair_counts[p], lp_hat, p))
                total_pairs += (len(new_stream) - 1)

            pair_counts.pop((a, b), None)
            pair_to_streams.pop((a, b), None)

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
        next_id = max(id_to_token, default=-1) + 1
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
