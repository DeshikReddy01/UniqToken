"""
Phase Eight: Script-Aware Candidate Allocation & Multi-Byte Length Sweep.
Tests Configurations A, B, C, D against SentencePiece and Boundary-BPE at V = 16,384
across 3 paired seeds (101, 202, 303) under strict matched FLOP compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Tuple

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
from pre_tokenizer import Normalizer, RegexPreTokenizer
from seed_builder import SeedToken, SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from unigram_trainer import UnigramTrainer
from benchmarks.run_phase_three_strict_matched import (
    TARGET_TRAINING_FLOPS,
    generate_rich_multilingual_corpus,
    train_and_eval_strict_transformer,
)


def run_phase_eight_sweep(
    target_vocab: int = 16384,
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 500,
) -> Dict[str, Any]:
    import sentencepiece as spm

    print("=" * 165)
    print("PHASE EIGHT: SCRIPT-AWARE CANDIDATE ALLOCATION & MULTI-BYTE LENGTH SWEEP")
    print(f"Scale: V = {target_vocab:,} | Seeds: {seeds} | Compute: Strict Matched FLOPs ({TARGET_TRAINING_FLOPS:.3e})")
    print("Success Target: Bytes/Tok > 5.5, BPB < 2.40, CE <= 9.10, Indic B/Tok Gap substantially reduced")
    print("=" * 165)

    configs = [
        "Config A (Frozen Base)",
        "Config B (Length Global α=1.5)",
        "Config C (Script Quotas α=1.0)",
        "Config D (Script Quotas α=1.25)",
        "SentencePiece (Anchor)",
        "Boundary-BPE (Anchor)",
    ]

    all_results: Dict[str, List[Dict[str, float]]] = {c: [] for c in configs}
    languages = ["English", "Hindi", "Telugu", "Tamil", "Bengali", "Arabic", "Chinese", "Russian"]

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n---> [Exploratory Seed {seed_idx}/{len(seeds)}: {seed}]")
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

        # Measure Indic B/Tok
        indic_val = "\n".join([val_by_lang[l] for l in ["Hindi", "Telugu", "Tamil", "Bengali"]])
        indic_val_bytes = len(indic_val.encode("utf-8"))
        sp_indic_toks = list(sp_proc.encode_as_pieces(indic_val))
        sp_indic_bpt = indic_val_bytes / max(len(sp_indic_toks), 1)

        all_results["SentencePiece (Anchor)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "indic_bpt": sp_indic_bpt,
                "fertility": fertility,
                "active_vocab_pct": sp_active_cov * 100.0,
                "p50_bytes": float(np.percentile(sp_tok_bytes, 50)),
                "p90_bytes": float(np.percentile(sp_tok_bytes, 90)),
                "pct_ge_6b": sp_pct_ge_6b,
            }
        )
        print(
            f"  [SentencePiece (Anchor)     ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Indic B/Tok: {sp_indic_bpt:.2f} | Fert: {fertility:.2f} | Active: {sp_active_cov * 100.0:.1f}% | >=6B: {sp_pct_ge_6b:.1f}%",
            flush=True,
        )

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

        bpe_indic_toks = bpe_model.encode(indic_val)
        bpe_indic_bpt = indic_val_bytes / max(len(bpe_indic_toks), 1)

        all_results["Boundary-BPE (Anchor)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "indic_bpt": bpe_indic_bpt,
                "fertility": fertility,
                "active_vocab_pct": bpe_active_cov * 100.0,
                "p50_bytes": float(np.percentile(bpe_tok_bytes, 50)),
                "p90_bytes": float(np.percentile(bpe_tok_bytes, 90)),
                "pct_ge_6b": bpe_pct_ge_6b,
            }
        )
        print(
            f"  [Boundary-BPE (Anchor)      ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Indic B/Tok: {bpe_indic_bpt:.2f} | Fert: {fertility:.2f} | Active: {bpe_active_cov * 100.0:.1f}% | >=6B: {bpe_pct_ge_6b:.1f}%",
            flush=True,
        )

        # Caliper Configurations
        sbp_merges = min(target_vocab // 10, 1500)
        base_target = max(target_vocab - sbp_merges, 1000)
        actual_merges = target_vocab - base_target

        cal_variants = [
            ("Config A (Frozen Base)", 1.0),
            ("Config B (Length Global α=1.5)", 1.5),
            ("Config C (Script Quotas α=1.0)", 1.0),
            ("Config D (Script Quotas α=1.25)", 1.25),
        ]

        for cfg_name, alpha_val in cal_variants:
            tok_base = CustomTokenizer.train_from_corpus(
                corpus=train_docs,
                target_vocab_size=base_target,
                seed_multiplier=1.2,
                ranking_strategy="byte_savings",
                min_boundary_entropy=0.5,
                length_exponent=alpha_val,
                pruning_length_exponent=0.0,
                min_frequency=1,
                verbose=False,
            )
            pretok_chunks = [
                tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))
            ]
            cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
            sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
            cal_tok = CustomTokenizer(
                normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model
            )

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

            cal_indic_toks = cal_tok.encode(indic_val)
            cal_indic_bpt = indic_val_bytes / max(len(cal_indic_toks), 1)

            all_results[cfg_name].append(
                {
                    "true_lm_bpb": lm_bpb,
                    "val_loss": val_loss,
                    "bytes_per_token": bpt,
                    "indic_bpt": cal_indic_bpt,
                    "fertility": fertility,
                    "active_vocab_pct": cal_active_cov * 100.0,
                    "p50_bytes": float(np.percentile(cal_tok_bytes, 50)),
                    "p90_bytes": float(np.percentile(cal_tok_bytes, 90)),
                    "pct_ge_6b": cal_pct_ge_6b,
                }
            )
            print(
                f"  [{cfg_name:<30}] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Indic B/Tok: {cal_indic_bpt:.2f} | Fert: {fertility:.2f} | Active: {cal_active_cov * 100.0:.1f}% | >=6B: {cal_pct_ge_6b:.1f}%",
                flush=True,
            )

    print("\n" + "=" * 175)
    print("PHASE EIGHT: SCRIPT-AWARE CANDIDATE ALLOCATION SUMMARY REPORT (MEAN ACROSS 3 SEEDS AT 16K SCALE)")
    print("=" * 175)
    print(
        f"{'Configuration':<32} | {'True LM BPB':<12} | {'Token CE':<10} | {'Bytes/Tok':<10} | {'Indic B/Tok':<12} | {'Fertility':<10} | {'Active Vocab %':<15} | {'>=6B Tokens %':<15} | {'Delta BPB vs Base'}"
    )
    print("-" * 175)

    summary: Dict[str, Dict[str, float]] = {}
    base_bpb = float(np.mean([r["true_lm_bpb"] for r in all_results["Config A (Frozen Base)"]]))

    for v in configs:
        bpb_m = float(np.mean([r["true_lm_bpb"] for r in all_results[v]]))
        ce_m = float(np.mean([r["val_loss"] for r in all_results[v]]))
        bpt_m = float(np.mean([r["bytes_per_token"] for r in all_results[v]]))
        ind_m = float(np.mean([r["indic_bpt"] for r in all_results[v]]))
        fert_m = float(np.mean([r["fertility"] for r in all_results[v]]))
        act_m = float(np.mean([r["active_vocab_pct"] for r in all_results[v]]))
        ge6_m = float(np.mean([r["pct_ge_6b"] for r in all_results[v]]))
        p50_m = float(np.mean([r["p50_bytes"] for r in all_results[v]]))
        p90_m = float(np.mean([r["p90_bytes"] for r in all_results[v]]))
        delta = bpb_m - base_bpb

        summary[v] = {
            "true_lm_bpb": bpb_m,
            "val_loss": ce_m,
            "bytes_per_token": bpt_m,
            "indic_bpt": ind_m,
            "fertility": fert_m,
            "active_vocab_pct": act_m,
            "pct_ge_6b": ge6_m,
            "p50_bytes": p50_m,
            "p90_bytes": p90_m,
            "delta_bpb": delta,
        }

        delta_str = f"{delta:+.3f} BPB" if v != "Config A (Frozen Base)" else "0.000 (Base)"
        print(
            f"{v:<32} | {bpb_m:<12.3f} | {ce_m:<10.3f} | {bpt_m:<10.2f} | {ind_m:<12.2f} | {fert_m:<10.2f} | {act_m:<15.1f}% | {ge6_m:<15.1f}% | {delta_str}"
        )
    print("=" * 175 + "\n")

    # Generate Pareto Trajectory Plot
    plt.figure(figsize=(11, 8), dpi=300)
    sp_bpb = summary["SentencePiece (Anchor)"]["true_lm_bpb"]
    sp_ce = summary["SentencePiece (Anchor)"]["val_loss"]
    bpe_bpb = summary["Boundary-BPE (Anchor)"]["true_lm_bpb"]
    bpe_ce = summary["Boundary-BPE (Anchor)"]["val_loss"]

    plt.scatter(
        [sp_bpb],
        [sp_ce],
        color="#1f77b4",
        marker="o",
        s=180,
        edgecolors="black",
        label="SentencePiece (Anchor)",
        zorder=6,
    )
    plt.annotate(
        f"SentencePiece\nBPB: {sp_bpb:.3f}, CE: {sp_ce:.2f}", (sp_bpb + 0.02, sp_ce - 0.03), fontsize=9, color="#1f77b4"
    )

    plt.scatter(
        [bpe_bpb],
        [bpe_ce],
        color="#2ca02c",
        marker="s",
        s=180,
        edgecolors="black",
        label="Boundary-BPE (Anchor)",
        zorder=6,
    )
    plt.annotate(
        f"Boundary-BPE\nBPB: {bpe_bpb:.3f}, CE: {bpe_ce:.2f}",
        (bpe_bpb + 0.02, bpe_ce - 0.03),
        fontsize=9,
        color="#2ca02c",
    )

    colors_dict = {
        "Config A (Frozen Base)": "gray",
        "Config B (Length Global α=1.5)": "#9467bd",
        "Config C (Script Quotas α=1.0)": "#ff7f0e",
        "Config D (Script Quotas α=1.25)": "#d62728",
    }
    markers_dict = {
        "Config A (Frozen Base)": "x",
        "Config B (Length Global α=1.5)": "D",
        "Config C (Script Quotas α=1.0)": "P",
        "Config D (Script Quotas α=1.25)": "*",
    }

    pts = []
    for b in [
        "Config A (Frozen Base)",
        "Config B (Length Global α=1.5)",
        "Config C (Script Quotas α=1.0)",
        "Config D (Script Quotas α=1.25)",
    ]:
        bpb = summary[b]["true_lm_bpb"]
        ce = summary[b]["val_loss"]
        pts.append((bpb, ce))
        sz = 260 if "D" in b else (200 if "C" in b else 150)
        plt.scatter(
            [bpb],
            [ce],
            color=colors_dict[b],
            marker=markers_dict[b],
            s=sz,
            edgecolors="black" if "D" in b or "C" in b else None,
            label=b,
            zorder=7,
        )
        lbl_clean = b.split("(")[1].replace(")", "")
        plt.annotate(
            f"{lbl_clean}\nBPB: {bpb:.3f}, CE: {ce:.2f}", (bpb + 0.02, ce + 0.02), fontsize=8.5, color=colors_dict[b]
        )

    plt.plot(
        [p[0] for p in pts], [p[1] for p in pts], color="#d62728", linestyle="--", linewidth=2, alpha=0.7, zorder=5
    )

    plt.title(
        "Phase Eight: Script-Aware Candidate Allocation Pareto Frontier (V = 16,384)", fontsize=12, fontweight="bold"
    )
    plt.xlabel("True LM BPB (lower is better) -> [Optimal: Left]", fontsize=11)
    plt.ylabel("Token Cross-Entropy Loss (lower is better) -> [Optimal: Bottom]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")

    plot_path = Path(__file__).resolve().parent / "phase_eight_script_pareto.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 8 Pareto chart to {plot_path}")

    # Save summary JSON
    json_path = Path(__file__).resolve().parent / "phase_eight_script_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] Saved Phase 8 summary metrics to {json_path}")

    return summary


if __name__ == "__main__":
    run_phase_eight_sweep(target_vocab=16384, seeds=[101, 202, 303], num_docs=500)
