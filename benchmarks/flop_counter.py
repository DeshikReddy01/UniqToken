"""
Authoritative Transformer Analytical FLOP Counter & Compute Accounting Engine.

Single Source of Truth for Phase Three:
- Implements explicit layer-by-layer analytical FLOP formulas under standard conventions.
- Eliminates hardcoded constants.
- Generates reproducible formula hashes for machine-readable audit records.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple


FLOP_FORMULA_VERSION = "v1.0"


@dataclass(frozen=True)
class TransformerFLOPReport:
    version: str
    formula_sha256: str
    vocab_size: int
    batch_size: int
    seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    backbone_fwd_flops_per_seq: int
    lm_head_fwd_flops_per_seq: int
    total_fwd_flops_per_seq: int
    fwd_flops_per_batch: int
    train_flops_per_step: int  # 3x forward multiplier


def compute_transformer_flops_per_step(
    vocab_size: int,
    batch_size: int = 16,
    seq_len: int = 64,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    d_ff: int = 128,
) -> TransformerFLOPReport:
    """
    Computes analytical FLOPs under the preregistered counting convention:
    - 1 MAC = 2 FLOPs (2MNK for matmuls)
    - Non-matmul conventions: Softmax = 3TV, Attention Softmax = 3HT^2, Activation = 4Td_ff, LayerNorm = 8Td
    - Backward pass = 2x Forward FLOPs (Total Step = 3x Forward)
    """
    T = seq_len
    d = d_model
    L = num_layers
    H = num_heads
    V = vocab_size
    B = batch_size

    # 1. Per-layer Backbone Forward FLOPs
    qkv_o_proj = 8 * T * (d**2)
    attn_scores_map = 4 * (T**2) * d
    attn_softmax = 3 * H * (T**2)
    ffn_matmuls = 4 * T * d * d_ff
    ffn_activation = 4 * T * d_ff
    layer_norms = 8 * T * d

    layer_fwd = qkv_o_proj + attn_scores_map + attn_softmax + ffn_matmuls + ffn_activation + layer_norms
    backbone_fwd_seq = L * layer_fwd

    # 2. LM Output Head & Softmax Forward FLOPs
    lm_head_proj = 2 * T * d * V
    lm_softmax = 3 * T * V
    lm_head_fwd_seq = lm_head_proj + lm_softmax

    # 3. Total Forward FLOPs
    total_fwd_seq = backbone_fwd_seq + lm_head_fwd_seq
    fwd_batch = B * total_fwd_seq

    # 4. Total Training FLOPs per Step (3x Forward multiplier: 1 Fwd + 2 Bwd)
    train_step = 3 * fwd_batch

    # Hash the implementation source code for cryptographic auditability
    src = inspect.getsource(compute_transformer_flops_per_step)
    formula_sha256 = hashlib.sha256(src.encode("utf-8")).hexdigest()

    return TransformerFLOPReport(
        version=FLOP_FORMULA_VERSION,
        formula_sha256=formula_sha256,
        vocab_size=V,
        batch_size=B,
        seq_len=T,
        d_model=d,
        num_layers=L,
        num_heads=H,
        d_ff=d_ff,
        backbone_fwd_flops_per_seq=backbone_fwd_seq,
        lm_head_fwd_flops_per_seq=lm_head_fwd_seq,
        total_fwd_flops_per_seq=total_fwd_seq,
        fwd_flops_per_batch=fwd_batch,
        train_flops_per_step=train_step,
    )


def plan_training_steps_for_target_flops(
    target_flops: int,
    vocab_size: int,
    batch_size: int = 16,
    seq_len: int = 64,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    d_ff: int = 128,
) -> Tuple[int, int, float, TransformerFLOPReport]:
    """
    Computes required optimizer steps to match target cumulative training FLOPs.
    Returns: (steps, actual_flops, flop_relative_error, flop_report)
    """
    report = compute_transformer_flops_per_step(
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    steps = max(int(round(target_flops / report.train_flops_per_step)), 1)
    actual_flops = steps * report.train_flops_per_step
    rel_error = abs(actual_flops - target_flops) / max(target_flops, 1)

    return steps, actual_flops, rel_error, report


if __name__ == "__main__":
    rep = compute_transformer_flops_per_step(8192)
    print("Authoritative FLOP Verification at V=8,192:")
    print(f"  Backbone Fwd / Seq: {rep.backbone_fwd_flops_per_seq:,} (Expected: 10,665,984)")
    print(f"  LM Head Fwd / Seq:  {rep.lm_head_fwd_flops_per_seq:,} (Expected: 68,681,728)")
    print(f"  Fwd / Batch (B=16): {rep.fwd_flops_per_batch:,} (Expected: 1,269,563,392)")
    print(f"  Train FLOPs / Step: {rep.train_flops_per_step:,} (Expected: 3,808,690,176)")
    print(f"  Formula SHA-256:    {rep.formula_sha256[:16]}... (v{rep.version})")
