"""
Phase Five: Causal Evaluation of Space-Boundary Pre-Tokenizer Alignment Patch.
Compares Patched Caliper against SentencePiece and Boundary-BPE at V = 16,384
across 3 paired seeds (101, 202, 303) under strict matched FLOP compute.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from tokenizer import CustomTokenizer
from benchmarks.run_phase_three_strict_matched import (
    TARGET_TRAINING_FLOPS,
    generate_rich_multilingual_corpus,
    train_and_eval_strict_transformer,
)


def run_phase_five_evaluation(
    target_vocab: int = 16384,
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 500,
) -> Dict[str, Dict[str, float]]:
    import sentencepiece as spm

    print("=" * 165)
    print("PHASE FIVE: CAUSAL EVALUATION OF SPACE-BOUNDARY PRE-TOKENIZATION PATCH")
    print(f"Scale: V = {target_vocab:,} | Seeds: {seeds} | Matched Analytical FLOPs: {TARGET_TRAINING_FLOPS:.3e}")
    print("Hypothesis: Space-prefixed lexical units eliminate ~0.96 token/word tax and drive BPB < 2.40")
    print("=" * 165)

    all_results: Dict[str, List[Dict[str, float]]] = {
        "SentencePiece (Anchor)": [],
        "Boundary-BPE (Anchor)": [],
        "Caliper (Patched Boundary)": [],
    }

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n---> [Paired Seed {seed_idx}/{len(seeds)}: {seed}]")
        train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=num_docs, seed=seed)
        combined_val = "\n".join(val_by_lang.values())
        total_val_bytes = len(combined_val.encode("utf-8"))
        words = [w for w in combined_val.split() if w]
        num_words = len(words)

        # 1. SentencePiece Anchor
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            sp_corpus = tmp / "train.txt"
            sp_corpus.write_text("\n".join(train_docs), encoding="utf-8")
            sp_prefix = tmp / "sp_model"
            spm.SentencePieceTrainer.train(
                input=str(sp_corpus),
                model_prefix=str(sp_prefix),
                model_type="unigram",
                vocab_size=target_vocab,
                character_coverage=1.0,
                byte_fallback=True,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            sp_proc = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
            sp_tokens = list(sp_proc.encode_as_pieces(combined_val))
            sp_enc = lambda t: sp_proc.encode(t, out_type=int)
            sp_vocab_size = sp_proc.get_piece_size()

        sp_counts = Counter(sp_tokens)
        sp_active_cov = len(sp_counts) / sp_vocab_size
        sp_tok_bytes = [len(t.encode("utf-8")) for t in sp_tokens]
        sp_pct_ge_6b = sum(1 for b in sp_tok_bytes if b >= 6) / max(len(sp_tok_bytes), 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=sp_enc,
            vocab_size=sp_vocab_size,
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(sp_tokens), 1)
        fertility = len(sp_tokens) / max(num_words, 1)

        all_results["SentencePiece (Anchor)"].append({
            "true_lm_bpb": lm_bpb,
            "val_loss": val_loss,
            "bytes_per_token": bpt,
            "fertility": fertility,
            "active_vocab_pct": sp_active_cov * 100.0,
            "p50_bytes": float(np.percentile(sp_tok_bytes, 50)),
            "p90_bytes": float(np.percentile(sp_tok_bytes, 90)),
            "pct_ge_6b": sp_pct_ge_6b,
        })
        print(f"  [SentencePiece (Anchor)     ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {sp_active_cov*100.0:.1f}% | >=6B: {sp_pct_ge_6b:.1f}% | p50/p90: {np.percentile(sp_tok_bytes, 50):.0f}/{np.percentile(sp_tok_bytes, 90):.0f}B", flush=True)

        # 2. Boundary-BPE Anchor
        b_bpe = BPETrainer(target_vocab_size=target_vocab, byte_fallback=True)
        bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
        bpe_model = b_bpe.train(bpe_chunks, verbose=False)
        bpe_tokens = bpe_model.encode(combined_val)
        bpe_counts = Counter(bpe_tokens)
        bpe_active_cov = len(bpe_counts) / len(bpe_model.vocab)
        bpe_tok_bytes = [len(t.encode("utf-8")) for t in bpe_tokens]
        bpe_pct_ge_6b = sum(1 for b in bpe_tok_bytes if b >= 6) / max(len(bpe_tok_bytes), 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: bpe_model.encode_to_ids(t),
            vocab_size=len(bpe_model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(bpe_tokens), 1)
        fertility = len(bpe_tokens) / max(num_words, 1)

        all_results["Boundary-BPE (Anchor)"].append({
            "true_lm_bpb": lm_bpb,
            "val_loss": val_loss,
            "bytes_per_token": bpt,
            "fertility": fertility,
            "active_vocab_pct": bpe_active_cov * 100.0,
            "p50_bytes": float(np.percentile(bpe_tok_bytes, 50)),
            "p90_bytes": float(np.percentile(bpe_tok_bytes, 90)),
            "pct_ge_6b": bpe_pct_ge_6b,
        })
        print(f"  [Boundary-BPE (Anchor)      ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {bpe_active_cov*100.0:.1f}% | >=6B: {bpe_pct_ge_6b:.1f}% | p50/p90: {np.percentile(bpe_tok_bytes, 50):.0f}/{np.percentile(bpe_tok_bytes, 90):.0f}B", flush=True)

        # 3. Caliper with Patched Boundary Pre-tokenizer
        sbp_merges = min(target_vocab // 10, 1500)
        base_target = max(target_vocab - sbp_merges, 1000)
        actual_merges = target_vocab - base_target

        tok_base = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=base_target,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            pruning_length_exponent=0.0,
            min_frequency=1,
            verbose=False,
        )
        pretok_chunks = [tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))]
        cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
        sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
        cal_tok = CustomTokenizer(normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model)

        cal_tokens = cal_tok.encode(combined_val)
        cal_counts = Counter(cal_tokens)
        cal_active_cov = len(cal_counts) / len(cal_tok.model.vocab)
        cal_tok_bytes = [len(t.encode("utf-8")) for t in cal_tokens]
        cal_pct_ge_6b = sum(1 for b in cal_tok_bytes if b >= 6) / max(len(cal_tok_bytes), 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: cal_tok.encode_to_ids(t),
            vocab_size=len(cal_tok.model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(cal_tokens), 1)
        fertility = len(cal_tokens) / max(num_words, 1)

        all_results["Caliper (Patched Boundary)"].append({
            "true_lm_bpb": lm_bpb,
            "val_loss": val_loss,
            "bytes_per_token": bpt,
            "fertility": fertility,
            "active_vocab_pct": cal_active_cov * 100.0,
            "p50_bytes": float(np.percentile(cal_tok_bytes, 50)),
            "p90_bytes": float(np.percentile(cal_tok_bytes, 90)),
            "pct_ge_6b": cal_pct_ge_6b,
        })
        print(f"  [Caliper (Patched Boundary) ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {cal_active_cov*100.0:.1f}% | >=6B: {cal_pct_ge_6b:.1f}% | p50/p90: {np.percentile(cal_tok_bytes, 50):.0f}/{np.percentile(cal_tok_bytes, 90):.0f}B", flush=True)

    print("\n" + "=" * 165)
    print("PHASE FIVE: CAUSAL EVALUATION SUMMARY REPORT (MEAN ACROSS 3 SEEDS AT 16K SCALE)")
    print("=" * 165)
    print(f"{'Configuration':<28} | {'True LM BPB':<12} | {'Token CE':<10} | {'Bytes/Tok':<10} | {'Fertility':<10} | {'Active Vocab %':<15} | {'>=6B Tokens %':<15} | {'p50/p90 Length':<15}")
    print("-" * 165)

    summary: Dict[str, Dict[str, float]] = {}

    for v in ["SentencePiece (Anchor)", "Boundary-BPE (Anchor)", "Caliper (Patched Boundary)"]:
        bpb_m = float(np.mean([r["true_lm_bpb"] for r in all_results[v]]))
        ce_m = float(np.mean([r["val_loss"] for r in all_results[v]]))
        bpt_m = float(np.mean([r["bytes_per_token"] for r in all_results[v]]))
        fert_m = float(np.mean([r["fertility"] for r in all_results[v]]))
        act_m = float(np.mean([r["active_vocab_pct"] for r in all_results[v]]))
        ge6_m = float(np.mean([r["pct_ge_6b"] for r in all_results[v]]))
        p50_m = float(np.mean([r["p50_bytes"] for r in all_results[v]]))
        p90_m = float(np.mean([r["p90_bytes"] for r in all_results[v]]))

        summary[v] = {
            "true_lm_bpb": bpb_m,
            "val_loss": ce_m,
            "bytes_per_token": bpt_m,
            "fertility": fert_m,
            "active_vocab_pct": act_m,
            "pct_ge_6b": ge6_m,
            "p50_bytes": p50_m,
            "p90_bytes": p90_m,
        }

        len_str = f"{p50_m:.0f}B / {p90_m:.0f}B"
        print(f"{v:<28} | {bpb_m:<12.3f} | {ce_m:<10.3f} | {bpt_m:<10.2f} | {fert_m:<10.2f} | {act_m:<15.1f}% | {ge6_m:<15.1f}% | {len_str:<15}")
    print("=" * 165 + "\n")

    # Generate Pareto Trajectory Plot
    plt.figure(figsize=(10, 7), dpi=300)
    sp_bpb = summary["SentencePiece (Anchor)"]["true_lm_bpb"]
    sp_ce = summary["SentencePiece (Anchor)"]["val_loss"]
    bpe_bpb = summary["Boundary-BPE (Anchor)"]["true_lm_bpb"]
    bpe_ce = summary["Boundary-BPE (Anchor)"]["val_loss"]
    cal_bpb = summary["Caliper (Patched Boundary)"]["true_lm_bpb"]
    cal_ce = summary["Caliper (Patched Boundary)"]["val_loss"]

    # Unpatched baseline anchor for reference: BPB 2.752, CE 9.074
    plt.scatter([2.752], [9.074], color="gray", marker="x", s=160, label="Caliper Baseline (Unpatched)", zorder=5)
    plt.annotate("Caliper (Unpatched)\nBPB: 2.752, CE: 9.07", (2.752 + 0.02, 9.074 - 0.03), fontsize=9, color="gray")

    plt.scatter([sp_bpb], [sp_ce], color="#1f77b4", marker="o", s=180, edgecolors="black", label="SentencePiece (Anchor)", zorder=6)
    plt.annotate(f"SentencePiece\nBPB: {sp_bpb:.3f}, CE: {sp_ce:.2f}", (sp_bpb + 0.02, sp_ce - 0.03), fontsize=9, color="#1f77b4")

    plt.scatter([bpe_bpb], [bpe_ce], color="#2ca02c", marker="s", s=180, edgecolors="black", label="Boundary-BPE (Anchor)", zorder=6)
    plt.annotate(f"Boundary-BPE\nBPB: {bpe_bpb:.3f}, CE: {bpe_ce:.2f}", (bpe_bpb + 0.02, bpe_ce - 0.03), fontsize=9, color="#2ca02c")

    plt.scatter([cal_bpb], [cal_ce], color="#d62728", marker="*", s=260, edgecolors="black", label="Caliper (Patched Boundary)", zorder=7)
    plt.annotate(f"Caliper (Patched Boundary)\nBPB: {cal_bpb:.3f}, CE: {cal_ce:.2f}", (cal_bpb + 0.02, cal_ce + 0.03), fontsize=10, fontweight="bold", color="#d62728")

    # Arrow showing movement
    plt.annotate("", xy=(cal_bpb, cal_ce), xytext=(2.752, 9.074),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.5, ls="--"))

    plt.title("Phase Five: Causal Effect of Space-Boundary Pre-Tokenizer Patch (V = 16,384)", fontsize=12, fontweight="bold")
    plt.xlabel("True LM BPB (lower is better) -> [Optimal: Left]", fontsize=11)
    plt.ylabel("Token Cross-Entropy Loss (lower is better) -> [Optimal: Bottom]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")

    plot_path = Path(__file__).resolve().parent / "phase_five_boundary_pareto.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Pareto causal evaluation plot saved to {plot_path}")

    return summary


if __name__ == "__main__":
    run_phase_five_evaluation(target_vocab=16384, seeds=[101, 202, 303], num_docs=500)
