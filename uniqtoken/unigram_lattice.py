from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .byte_codec import ByteFallbackEngine


def logsumexp(log_probs: List[float]) -> float:
    """
    Numerically stable log-sum-exp:
    logsumexp([x_1, x_2, ...]) = max_x + log(sum(exp(x_i - max_x)))
    """
    if not log_probs:
        return -float("inf")
    max_val = max(log_probs)
    if max_val == -float("inf"):
        return -float("inf")
    sum_exp = sum(math.exp(x - max_val) for x in log_probs)
    return max_val + math.log(sum_exp)


@dataclass
class LatticeEdge:
    start: int
    end: int
    tokens: List[str]
    log_prob: float
    cost: float


class UnigramLattice:
    """
    Directed Acyclic Graph (DAG) with:
    1. Viterbi 1-Best Optimal Decoder (Deterministic Inference).
    2. Forward-Backward Expectation Engine (EM Optimization).
    3. Forward-Filtering Backward-Sampling (FFBS) Subword Regularization.
    """

    DEFAULT_BYTE_PENALTY: float = 10.0

    def __init__(
        self,
        text: str,
        vocab_log_probs: Dict[str, float],
        max_subword_len: int = 16,
        byte_fallback: bool = True,
        trie: Optional[object] = None,
        max_edges_per_node: Optional[int] = None,
        min_edge_log_prob: Optional[float] = None,
        unk_token: str = "<|unk|>",
    ):
        if max_subword_len < 1:
            raise ValueError("max_subword_len must be at least one")
        if max_edges_per_node is not None and max_edges_per_node < 1:
            raise ValueError("max_edges_per_node must be at least one")
        self.text = text
        self.length = len(text)
        self.vocab = vocab_log_probs
        self.max_subword_len = max_subword_len
        self.byte_fallback = byte_fallback
        self.trie = trie
        self.max_edges_per_node = max_edges_per_node
        self.min_edge_log_prob = min_edge_log_prob
        self.unk_token = unk_token

        self.begin_nodes: List[List[LatticeEdge]] = [[] for _ in range(self.length + 1)]
        self.end_nodes: List[List[LatticeEdge]] = [[] for _ in range(self.length + 1)]

        self._build_graph()

    def _build_graph(self) -> None:
        for i in range(self.length):
            has_edge_at_i = False

            if self.trie is not None and hasattr(self.trie, "find_matches"):
                matches = self.trie.find_matches(self.text, i, self.max_subword_len)
                for end_j, subword, log_p in matches:
                    if self.min_edge_log_prob is not None and log_p < self.min_edge_log_prob:
                        continue
                    edge = LatticeEdge(
                        start=i,
                        end=end_j,
                        tokens=[subword],
                        log_prob=log_p,
                        cost=-log_p,
                    )
                    self.begin_nodes[i].append(edge)
                    self.end_nodes[end_j].append(edge)
                    has_edge_at_i = True
            else:
                max_j = min(self.length + 1, i + self.max_subword_len + 1)
                for j in range(i + 1, max_j):
                    subword = self.text[i:j]
                    if subword in self.vocab:
                        log_p = self.vocab[subword]
                        if self.min_edge_log_prob is not None and log_p < self.min_edge_log_prob:
                            continue
                        edge = LatticeEdge(
                            start=i,
                            end=j,
                            tokens=[subword],
                            log_prob=log_p,
                            cost=-log_p,
                        )
                        self.begin_nodes[i].append(edge)
                        self.end_nodes[j].append(edge)
                        has_edge_at_i = True

            if not has_edge_at_i:
                if self.byte_fallback:
                    char = self.text[i]
                    try:
                        byte_tokens = ByteFallbackEngine.char_to_byte_tokens(char)
                    except (ValueError, UnicodeEncodeError):
                        byte_tokens = ByteFallbackEngine.char_to_byte_tokens("\ufffd")
                    total_log_p = sum(self.vocab.get(b, -self.DEFAULT_BYTE_PENALTY) for b in byte_tokens)
                    edge = LatticeEdge(
                        start=i,
                        end=i + 1,
                        tokens=byte_tokens,
                        log_prob=total_log_p,
                        cost=-total_log_p,
                    )
                elif self.unk_token in self.vocab:
                    # No byte fallback: map the OOV character to the unk token so
                    # the lattice stays connected instead of crashing downstream.
                    log_p = self.vocab[self.unk_token]
                    edge = LatticeEdge(
                        start=i,
                        end=i + 1,
                        tokens=[self.unk_token],
                        log_prob=log_p,
                        cost=-log_p,
                    )
                else:
                    # No fallback available; node is genuinely disconnected.
                    continue
                self.begin_nodes[i].append(edge)
                self.end_nodes[i + 1].append(edge)

        # Beam Pruning: Cap incoming edges per node if max_edges_per_node is specified
        if self.max_edges_per_node is not None:
            k = self.max_edges_per_node
            for j in range(1, self.length + 1):
                if len(self.end_nodes[j]) > k:
                    # ponytail: APPROXIMATE pruning — edges are ranked by local
                    # edge cost only, not by best_cost[start] + edge.cost, so a
                    # locally cheap edge can be globally inferior. This bounds
                    # lattice size; exact decoding is preserved only when k is
                    # large enough that no true-best edge is pruned. Upgrade:
                    # rank by forward DP scores computed before pruning.
                    self.end_nodes[j].sort(key=lambda e: e.cost)
                    # ponytail: O(k) removal per node not O(n*k) scan; upgrade to heap if n>100k
                    pruned = self.end_nodes[j][k:]
                    self.end_nodes[j] = self.end_nodes[j][:k]
                    for e in pruned:
                        # remove pruned edge from its start bucket by identity
                        lst = self.begin_nodes[e.start]
                        self.begin_nodes[e.start] = [x for x in lst if x is not e]

    def viterbi_edges(self) -> Tuple[List[LatticeEdge], float]:
        """Return the edges in the single most probable segmentation."""
        best_cost: List[float] = [float("inf")] * (self.length + 1)
        best_edge: List[Optional[LatticeEdge]] = [None] * (self.length + 1)
        best_cost[0] = 0.0

        for j in range(1, self.length + 1):
            for edge in self.end_nodes[j]:
                i = edge.start
                cost_through_edge = best_cost[i] + edge.cost
                if cost_through_edge < best_cost[j]:
                    best_cost[j] = cost_through_edge
                    best_edge[j] = edge

        edges: List[LatticeEdge] = []
        curr = self.length

        while curr > 0:
            selected_edge = best_edge[curr]
            if selected_edge is None:
                raise ValueError(
                    f"Lattice disconnected at index {curr} of {self.text!r}: no "
                    f"vocabulary/fallback edge covers this character. Train with "
                    f"byte_fallback=True or ensure {self.unk_token!r} is in the vocabulary."
                )
            edges.append(selected_edge)
            curr = selected_edge.start

        edges.reverse()
        total_log_likelihood = -best_cost[self.length]
        return edges, total_log_likelihood

    def viterbi(self) -> Tuple[List[str], float]:
        """Compute the single 1-best segmentation used during standard inference."""
        edges, total_log_likelihood = self.viterbi_edges()
        flat_tokens = [token for edge in edges for token in edge.tokens]
        return flat_tokens, total_log_likelihood

    def forward_backward(self) -> Tuple[Dict[str, float], float]:
        """
        Computes expected token counts across ALL valid segmentations in log-space.
        """
        if self.length == 0:
            return {}, 0.0

        log_alpha: List[float] = [-float("inf")] * (self.length + 1)
        log_alpha[0] = 0.0

        for j in range(1, self.length + 1):
            incoming_scores = [log_alpha[edge.start] + edge.log_prob for edge in self.end_nodes[j]]
            log_alpha[j] = logsumexp(incoming_scores)

        total_marginal_log_lik = log_alpha[self.length]
        if total_marginal_log_lik == -float("inf"):
            raise ValueError(
                f"Lattice disconnected for {self.text!r}: no vocabulary/fallback "
                f"edge covers every character. Train with byte_fallback=True or "
                f"ensure {self.unk_token!r} is in the vocabulary."
            )

        log_beta: List[float] = [-float("inf")] * (self.length + 1)
        log_beta[self.length] = 0.0

        for i in range(self.length - 1, -1, -1):
            outgoing_scores = [edge.log_prob + log_beta[edge.end] for edge in self.begin_nodes[i]]
            log_beta[i] = logsumexp(outgoing_scores)

        expected_counts: Dict[str, float] = {}

        for j in range(1, self.length + 1):
            for edge in self.end_nodes[j]:
                log_posterior = log_alpha[edge.start] + edge.log_prob + log_beta[edge.end] - total_marginal_log_lik
                posterior = math.exp(log_posterior)
                for tok in edge.tokens:
                    expected_counts[tok] = expected_counts.get(tok, 0.0) + posterior

        return expected_counts, total_marginal_log_lik

    def sample(self, alpha: float = 0.5) -> List[str]:
        """
        Subword Regularization via Forward-Filtering Backward-Sampling (FFBS).

        Samples a valid segmentation from the distribution:
            P(x | W) ~ exp( alpha * sum(log p(t_i)) )

        - alpha -> inf: approaches deterministic Viterbi 1-best path.
        - alpha = 1.0 : exact unigram model posterior sampling.
        - alpha < 1.0 : flattened temperature (more diverse subwords / single characters).
        """
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite value greater than zero")
        if self.length == 0:
            return []

        # 1. Forward Filter with Temperature alpha
        log_alpha: List[float] = [-float("inf")] * (self.length + 1)
        log_alpha[0] = 0.0

        for j in range(1, self.length + 1):
            incoming = [log_alpha[edge.start] + (alpha * edge.log_prob) for edge in self.end_nodes[j]]
            log_alpha[j] = logsumexp(incoming)

        if log_alpha[self.length] == -float("inf"):
            raise ValueError(
                f"Lattice disconnected for {self.text!r}: no vocabulary/fallback "
                f"edge covers every character. Train with byte_fallback=True or "
                f"ensure {self.unk_token!r} is in the vocabulary."
            )

        # 2. Backward Sampling
        flat_tokens: List[str] = []
        curr = self.length

        while curr > 0:
            edges = self.end_nodes[curr]
            if not edges:
                raise ValueError(
                    f"Lattice disconnected at index {curr} of {self.text!r}: no "
                    f"vocabulary/fallback edge covers this character. Train with "
                    f"byte_fallback=True or ensure {self.unk_token!r} is in the vocabulary."
                )

            # Compute transition probabilities for incoming edges ending at curr
            edge_log_probs: List[float] = []
            for edge in edges:
                edge_score = log_alpha[edge.start] + (alpha * edge.log_prob) - log_alpha[curr]
                edge_log_probs.append(edge_score)

            # Softmax normalization to obtain sampling weights
            edge_probs = [math.exp(lp) for lp in edge_log_probs]
            prob_sum = sum(edge_probs)
            if prob_sum > 0:
                normalized_probs = [p / prob_sum for p in edge_probs]
            else:
                normalized_probs = [1.0 / len(edges)] * len(edges)

            # Sample one edge according to normalized_probs
            chosen_edge = random.choices(edges, weights=normalized_probs, k=1)[0]

            for t in reversed(chosen_edge.tokens):
                flat_tokens.append(t)
            curr = chosen_edge.start

        flat_tokens.reverse()
        return flat_tokens
