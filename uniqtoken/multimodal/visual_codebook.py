from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple, Union


class VisualCodebook:
    """
    Discrete Vector-Quantized (VQ) Visual Codebook Tokenizer with EMA Training.

    Quantizes continuous image patch pixel vectors into discrete visual tokens
    (<|vis_0000|> to <|vis_{K-1}|>) using nearest-neighbor codebook projection.
    Supports online EMA (Exponential Moving Average) updates for codebook training.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 16 * 16 * 3,
        seed: int = 42,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if num_embeddings > 10000:
            # <|vis_{i:04d}|> is fixed-width to 4 digits; wider indices would
            # break the format contract shared with decoders.
            raise ValueError("num_embeddings must not exceed 10000 (fixed-width <|vis_NNNN|> token format)")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0 < ema_decay < 1:
            raise ValueError("ema_decay must be in (0, 1)")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite positive number")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Initialize codebook vectors e_1 .. e_K
        self._rng = random.Random(seed)
        self.codebook: List[List[float]] = [
            [self._rng.gauss(0.0, 1.0 / math.sqrt(embedding_dim)) for _ in range(embedding_dim)]
            for _ in range(num_embeddings)
        ]

        # EMA statistics for online training
        self._ema_cluster_size: List[float] = [0.0] * num_embeddings
        self._ema_embed_sum: List[List[float]] = [[0.0] * embedding_dim for _ in range(num_embeddings)]
        self._update_count = 0

        # Precompute squared norms for efficient distance computation
        self._codebook_norms: List[float] = [sum(c * c for c in code_vec) for code_vec in self.codebook]

    def get_special_tokens(self) -> List[str]:
        """
        Returns the list of all discrete visual token identifiers.
        """
        return [f"<|vis_{i:04d}|>" for i in range(self.num_embeddings)]

    def quantize_patch(self, patch_vector: List[float]) -> Tuple[int, str, float]:
        """
        Maps a continuous patch vector to the nearest codebook index:
            k* = argmin_k ||patch - e_k||^2
        Returns (codebook_index, token_string, quantization_error).
        Uses efficient distance computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        """
        if len(patch_vector) != self.embedding_dim:
            raise ValueError(f"Patch vector dimension {len(patch_vector)} != embedding_dim {self.embedding_dim}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in patch_vector):
            raise ValueError("patch_vector must contain only finite numeric values")

        patch_norm = sum(p * p for p in patch_vector)

        best_idx = 0
        min_dist = float("inf")

        for idx, code_vec in enumerate(self.codebook):
            # Compute Euclidean distance squared efficiently
            dot_product = sum(p * c for p, c in zip(patch_vector, code_vec))
            dist = patch_norm + self._codebook_norms[idx] - 2 * dot_product
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        token_str = f"<|vis_{best_idx:04d}|>"
        return best_idx, token_str, min_dist

    def quantize_patches(self, patch_vectors: List[List[float]]) -> List[str]:
        """
        Quantizes a list of image patch vectors into a list of discrete visual token strings.
        """
        return [self.quantize_patch(vec)[1] for vec in patch_vectors]

    def dequantize_token(self, token_str: str) -> List[float]:
        """
        Reconstructs the continuous patch pixel vector from a discrete visual token string.
        """
        if not isinstance(token_str, str) or not re.fullmatch(r"<\|vis_\d{4}\|>", token_str):
            raise ValueError(f"Invalid visual token format: {token_str!r}")

        idx = int(token_str[6:-2])
        if not 0 <= idx < self.num_embeddings:
            raise ValueError(f"Visual token index {idx} out of range (0-{self.num_embeddings - 1})")

        return list(self.codebook[idx])

    def update_ema(self, patch_vectors: List[List[float]], indices: List[int]) -> None:
        """
        Updates codebook using Exponential Moving Average (EMA) of assigned vectors.
        This implements the standard VQ-VAE codebook update rule.

        Args:
            patch_vectors: List of input patch vectors
            indices: List of assigned codebook indices for each patch
        """
        if len(patch_vectors) != len(indices):
            raise ValueError("patch_vectors and indices must have same length")
        if not patch_vectors:
            return

        # Validate and aggregate the complete batch before mutating EMA state.
        # Applying decay once per vector makes the result depend on batch size.
        counts: Dict[int, int] = {}
        sums: Dict[int, List[float]] = {}
        for vec, idx in zip(patch_vectors, indices):
            if len(vec) != self.embedding_dim:
                raise ValueError(f"Patch vector dimension {len(vec)} != embedding_dim {self.embedding_dim}")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vec):
                raise ValueError("patch_vectors must contain only finite numeric values")
            if not 0 <= idx < self.num_embeddings:
                raise ValueError(f"Codebook index {idx} out of range")

            counts[idx] = counts.get(idx, 0) + 1
            batch_sum = sums.setdefault(idx, [0.0] * self.embedding_dim)
            for d, value in enumerate(vec):
                batch_sum[d] += value

        self._update_count += 1
        batch_weight = 1 - self.ema_decay
        decay = self.ema_decay
        for idx, count in counts.items():
            self._ema_cluster_size[idx] = decay * self._ema_cluster_size[idx] + batch_weight * count
            b_sum = sums[idx]
            e_sum = self._ema_embed_sum[idx]
            for d in range(self.embedding_dim):
                e_sum[d] = decay * e_sum[d] + batch_weight * b_sum[d]

        for idx in range(self.num_embeddings):
            if idx not in counts:
                self._ema_cluster_size[idx] *= decay
                e_sum = self._ema_embed_sum[idx]
                if any(e_sum):
                    for d in range(self.embedding_dim):
                        e_sum[d] *= decay

        # Periodically re-normalize codebook vectors from EMA statistics
        # This is done lazily to avoid overhead on every update
        if self._update_count % 100 == 0:
            self._rebuild_from_ema()

    def finalize(self) -> None:
        """Flushes accumulated EMA statistics and rebuilds codebook vectors."""
        self._rebuild_from_ema()

    def flush(self) -> None:
        """Alias for finalize()."""
        self.finalize()

    def _rebuild_from_ema(self) -> None:
        """Rebuilds codebook vectors from EMA statistics with Laplace smoothing."""
        for idx in range(self.num_embeddings):
            n = self._ema_cluster_size[idx]
            if n < self.epsilon:
                continue

            # Apply Laplace smoothing to avoid dead codes
            smoothed_n = n + self.epsilon
            for d in range(self.embedding_dim):
                self.codebook[idx][d] = self._ema_embed_sum[idx][d] / smoothed_n

        # Update precomputed norms
        self._codebook_norms = [sum(c * c for c in code_vec) for code_vec in self.codebook]

    def kmeans_init(self, patch_vectors: List[List[float]], max_iter: int = 10) -> None:
        """
        Initializes codebook using k-means++ on the provided patch vectors.
        Useful for warm-starting the codebook before EMA training.
        """
        if len(patch_vectors) < self.num_embeddings:
            raise ValueError(f"Need at least {self.num_embeddings} patches for k-means init, got {len(patch_vectors)}")
        if max_iter < 1:
            raise ValueError("max_iter must be at least one")
        for vector in patch_vectors:
            if len(vector) != self.embedding_dim:
                raise ValueError(f"Patch vector dimension {len(vector)} != embedding_dim {self.embedding_dim}")

        # K-means++ initialization
        centroids = []
        # Pick first centroid randomly
        first_idx = self._rng.randrange(len(patch_vectors))
        centroids.append(patch_vectors[first_idx][:])

        for _ in range(1, self.num_embeddings):
            # Compute distances to nearest centroid
            min_dists = []
            for vec in patch_vectors:
                min_dist = float("inf")
                for c in centroids:
                    dist = sum((v - c) ** 2 for v, c in zip(vec, c))
                    if dist < min_dist:
                        min_dist = dist
                min_dists.append(min_dist)

            # Sample next centroid proportional to squared distance
            total = sum(min_dists)
            if total == 0:
                centroids.append(patch_vectors[self._rng.randrange(len(patch_vectors))][:])
                continue
            r = self._rng.random() * total
            cumsum = 0.0
            for i, d in enumerate(min_dists):
                cumsum += d
                if cumsum >= r:
                    centroids.append(patch_vectors[i][:])
                    break
            else:
                # float rounding: cumsum never reached r — take the farthest point
                centroids.append(patch_vectors[max(range(len(min_dists)), key=min_dists.__getitem__)][:])

        # Run k-means iterations
        for _ in range(max_iter):
            # Assign each patch to nearest centroid
            assignments: List[List[List[float]]] = [[] for _ in range(self.num_embeddings)]
            for vec in patch_vectors:
                best_idx = 0
                min_dist = float("inf")
                for idx, c in enumerate(centroids):
                    dist = sum((v - c) ** 2 for v, c in zip(vec, c))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                assignments[best_idx].append(vec)

            # Update centroids
            new_centroids = []
            for idx in range(self.num_embeddings):
                if assignments[idx]:
                    new_c = [0.0] * self.embedding_dim
                    for vec in assignments[idx]:
                        for d in range(self.embedding_dim):
                            new_c[d] += vec[d]
                    for d in range(self.embedding_dim):
                        new_c[d] /= len(assignments[idx])
                    new_centroids.append(new_c)
                else:
                    # Re-initialize empty cluster randomly
                    new_centroids.append(patch_vectors[self._rng.randrange(len(patch_vectors))][:])

            centroids = new_centroids

        # Update codebook
        self.codebook = centroids
        self._codebook_norms = [sum(c * c for c in code_vec) for code_vec in self.codebook]
        # Reset EMA stats
        self._ema_cluster_size = [0.0] * self.num_embeddings
        self._ema_embed_sum = [[0.0] * self.embedding_dim for _ in range(self.num_embeddings)]
        self._update_count = 0

    def get_codebook_state(self) -> Dict:
        """Returns serializable codebook state for saving, rebuilding from EMA first."""
        self.finalize()
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "seed": self.seed,
            "ema_decay": self.ema_decay,
            "epsilon": self.epsilon,
            "codebook": [vector[:] for vector in self.codebook],
            "ema_cluster_size": list(self._ema_cluster_size),
            "ema_embed_sum": [vector[:] for vector in self._ema_embed_sum],
            "update_count": self._update_count,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Saves codebook state to a JSON file, rebuilding from EMA first."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.get_codebook_state(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VisualCodebook":
        """Loads codebook from a saved JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Dict) -> "VisualCodebook":
        """Reconstructs VisualCodebook from saved state."""
        cb = cls(
            num_embeddings=state["num_embeddings"],
            embedding_dim=state["embedding_dim"],
            seed=state["seed"],
            ema_decay=state.get("ema_decay", 0.99),
            epsilon=state.get("epsilon", 1e-5),
        )
        codebook = state["codebook"]
        if len(codebook) != cb.num_embeddings:
            raise ValueError("visual codebook state has an invalid codebook size")
        if any(len(vector) != cb.embedding_dim for vector in codebook):
            raise ValueError("visual codebook state has an invalid vector dimension")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for vector in codebook for value in vector
        ):
            raise ValueError("visual codebook state contains non-finite values")
        # copy: avoid aliasing the caller's nested lists
        cb.codebook = [vector[:] for vector in codebook]
        cb._ema_cluster_size = state.get("ema_cluster_size", [0.0] * cb.num_embeddings)
        cb._ema_embed_sum = state.get(
            "ema_embed_sum",
            [[0.0] * cb.embedding_dim for _ in range(cb.num_embeddings)],
        )
        if len(cb._ema_cluster_size) != cb.num_embeddings or len(cb._ema_embed_sum) != cb.num_embeddings:
            raise ValueError("visual codebook state has invalid EMA dimensions")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in cb._ema_cluster_size
        ):
            raise ValueError("visual codebook EMA cluster sizes must be finite and non-negative")
        if any(
            len(vector) != cb.embedding_dim
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
            for vector in cb._ema_embed_sum
        ):
            raise ValueError("visual codebook EMA sums have invalid values")
        cb._update_count = state.get("update_count", 0)
        if not isinstance(cb._update_count, int) or isinstance(cb._update_count, bool) or cb._update_count < 0:
            raise ValueError("visual codebook update_count must be a non-negative integer")
        cb._codebook_norms = [sum(c * c for c in code_vec) for code_vec in cb.codebook]
        return cb
