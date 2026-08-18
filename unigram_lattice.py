import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from byte_codec import ByteFallbackEngine


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
    ):
        if max_subword_len < 1:
            raise ValueError("max_subword_len must be at least one")
        self.text = text
        self.length = len(text)
        self.vocab = vocab_log_probs
        self.max_subword_len = max_subword_len
        self.byte_fallback = byte_fallback
        self.trie = trie

        self.begin_nodes: List[List[LatticeEdge]] = [[] for _ in range(self.length + 1)]
        self.end_nodes: List[List[LatticeEdge]] = [[] for _ in range(self.length + 1)]

        self._build_graph()

    def _build_graph(self) -> None:
        for i in range(self.length):
            has_edge_at_i = False

            if self.trie is not None and hasattr(self.trie, "find_matches"):
                matches = self.trie.find_matches(self.text, i, self.max_subword_len)
                for end_j, subword, log_p in matches:
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

            if not has_edge_at_i and self.byte_fallback:
                char = self.text[i]
                byte_tokens = ByteFallbackEngine.char_to_byte_tokens(char)
                total_log_p = sum(self.vocab.get(b, -self.DEFAULT_BYTE_PENALTY) for b in byte_tokens)
                edge = LatticeEdge(
                    start=i,
                    end=i + 1,
                    tokens=byte_tokens,
                    log_prob=total_log_p,
                    cost=-total_log_p,
                )
                self.begin_nodes[i].append(edge)
                self.end_nodes[i + 1].append(edge)

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
                raise RuntimeError(f"Lattice disconnected at index {curr} for {self.text!r}")
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
            raise RuntimeError(f"Lattice disconnected for {self.text!r}")

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
            raise RuntimeError(f"Lattice disconnected for {self.text!r}")

        # 2. Backward Sampling
        flat_tokens: List[str] = []
        curr = self.length

        while curr > 0:
            edges = self.end_nodes[curr]
            if not edges:
                raise RuntimeError(f"Lattice disconnected at index {curr} for {self.text!r}")

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
