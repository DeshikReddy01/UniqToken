"""
Phase Eleven: Vocabulary Scaling Experiment across 8K, 16K, 32K, and 64K Scales.
Evaluates SentencePiece, Boundary-BPE, and Frozen Caliper Config B across 3 paired seeds (101, 202, 303)
under strict matched FLOP compute (1.333e+11) on CUDA.
Maps the empirical Pareto scaling trajectory as vocabulary capacity expands.
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


@dataclass
class ScaleRecord:
    vocab_size: int
    seed: int
    model_name: str
    true_lm_bpb: float
    token_ce_loss: float
    bytes_per_token: float
    indic_bpt: float
    fertility: float
    active_vocab_pct: float
    pct_ge_6b: float
    p50_bytes: float
    p90_bytes: float
    wall_clock_sec: float


def run_phase_eleven_scaling(
    vocab_scales: List[int] = [8192, 16384, 32768, 65536],
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 600,
) -> Dict[str, Any]:
    import sentencepiece as spm

    print("=" * 175)
    print("PHASE ELEVEN: VOCABULARY SCALING EXPERIMENT (8K -> 16K -> 32K -> 64K)")
    print(
        f"Scales: {vocab_scales} | Seeds: {seeds} (N = {len(seeds)} paired) | Matched FLOPs: {TARGET_TRAINING_FLOPS:.3e}"
    )
    print(
        "Question: Does Caliper's Pareto advantage over Boundary-BPE and SentencePiece scale and widen with vocabulary capacity?"
    )
    print("=" * 175)

    all_records: List[ScaleRecord] = []
    languages = ["English", "Hindi", "Telugu", "Tamil", "Bengali", "Arabic", "Chinese", "Russian"]

    for V in vocab_scales:
        print(f"\n======================================== SCALE: V = {V:,} ========================================")
        sbp_merges = min(V // 10, 3000)
        base_target = max(V - sbp_merges, 1000)
        actual_merges = V - base_target

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n---> [Scale V = {V:,} | Paired Seed {seed_idx}/{len(seeds)}: {seed}]")
            train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=num_docs, seed=seed)
            combined_val = "\n".join(val_by_lang.values())
            total_val_bytes = len(combined_val.encode("utf-8"))
            words = [w for w in combined_val.split() if w]
            num_words = len(words)

            indic_val = "\n".join([val_by_lang[l] for l in ["Hindi", "Telugu", "Tamil", "Bengali"]])
            indic_val_bytes = len(indic_val.encode("utf-8"))

            # 1. SentencePiece-Unigram
            t0 = time.time()
            with TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                sp_corpus = tmp / "train.txt"
                sp_corpus.write_text("\n".join(train_docs), encoding="utf-8")
                sp_prefix = tmp / "sp_model"
                spm.SentencePieceTrainer.train(
                    input=str(sp_corpus),
                    model_prefix=str(sp_prefix),
                    model_type="unigram",
                    vocab_size=V,
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
            sp_indic_toks = list(sp_proc.encode_as_pieces(indic_val))
            sp_indic_bpt = indic_val_bytes / max(len(sp_indic_toks), 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=sp_enc,
                vocab_size=sp_vocab_size,
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )
            sp_bpt = total_val_bytes / max(len(sp_tokens), 1)
            sp_fert = len(sp_tokens) / max(num_words, 1)

            rec_sp = ScaleRecord(
                vocab_size=V,
                seed=seed,
                model_name="SentencePiece-Unigram",
                true_lm_bpb=lm_bpb,
                token_ce_loss=val_loss,
                bytes_per_token=sp_bpt,
                indic_bpt=sp_indic_bpt,
                fertility=sp_fert,
                active_vocab_pct=sp_active_cov * 100.0,
                pct_ge_6b=sp_pct_ge_6b,
                p50_bytes=float(np.percentile(sp_tok_bytes, 50)),
                p90_bytes=float(np.percentile(sp_tok_bytes, 90)),
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_sp)
            print(
                f"  [SentencePiece (Anchor)     ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {sp_bpt:.2f} | Indic: {sp_indic_bpt:.2f} | Fert: {sp_fert:.2f} | Active: {sp_active_cov * 100.0:.1f}%",
                flush=True,
            )

            # 2. Boundary-BPE
            t0 = time.time()
            b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
            bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
            bpe_model = b_bpe.train(bpe_chunks, verbose=False)
            bpe_tokens = bpe_model.encode(combined_val)
            bpe_counts = Counter(bpe_tokens)
            bpe_active_cov = len(bpe_counts) / len(bpe_model.vocab)
            bpe_tok_bytes = [len(t.encode("utf-8")) for t in bpe_tokens]
            bpe_pct_ge_6b = sum(1 for b in bpe_tok_bytes if b >= 6) / max(len(bpe_tok_bytes), 1) * 100.0
            bpe_indic_toks = bpe_model.encode(indic_val)
            bpe_indic_bpt = indic_val_bytes / max(len(bpe_indic_toks), 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=lambda t: bpe_model.encode_to_ids(t),
                vocab_size=len(bpe_model.vocab),
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )
            bpe_bpt = total_val_bytes / max(len(bpe_tokens), 1)
            bpe_fert = len(bpe_tokens) / max(num_words, 1)

            rec_bpe = ScaleRecord(
                vocab_size=V,
                seed=seed,
                model_name="Boundary-BPE",
                true_lm_bpb=lm_bpb,
                token_ce_loss=val_loss,
                bytes_per_token=bpe_bpt,
                indic_bpt=bpe_indic_bpt,
                fertility=bpe_fert,
                active_vocab_pct=bpe_active_cov * 100.0,
                pct_ge_6b=bpe_pct_ge_6b,
                p50_bytes=float(np.percentile(bpe_tok_bytes, 50)),
                p90_bytes=float(np.percentile(bpe_tok_bytes, 90)),
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_bpe)
            print(
                f"  [Boundary-BPE (Anchor)      ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpe_bpt:.2f} | Indic: {bpe_indic_bpt:.2f} | Fert: {bpe_fert:.2f} | Active: {bpe_active_cov * 100.0:.1f}%",
                flush=True,
            )

            # 3. Frozen Caliper Config B
            t0 = time.time()
            tok_base = CustomTokenizer.train_from_corpus(
                corpus=train_docs,
                target_vocab_size=base_target,
                seed_multiplier=1.2,
                ranking_strategy="byte_savings",
                min_boundary_entropy=0.5,
                length_exponent=1.5,
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
            cal_indic_toks = cal_tok.encode(indic_val)
            cal_indic_bpt = indic_val_bytes / max(len(cal_indic_toks), 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=lambda t: cal_tok.encode_to_ids(t),
                vocab_size=len(cal_tok.model.vocab),
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )
            cal_bpt = total_val_bytes / max(len(cal_tokens), 1)
            cal_fert = len(cal_tokens) / max(num_words, 1)

            rec_cal = ScaleRecord(
                vocab_size=V,
                seed=seed,
                model_name="Caliper-SuperBPE (Config B)",
                true_lm_bpb=lm_bpb,
                token_ce_loss=val_loss,
                bytes_per_token=cal_bpt,
                indic_bpt=cal_indic_bpt,
                fertility=cal_fert,
                active_vocab_pct=cal_active_cov * 100.0,
                pct_ge_6b=cal_pct_ge_6b,
                p50_bytes=float(np.percentile(cal_tok_bytes, 50)),
                p90_bytes=float(np.percentile(cal_tok_bytes, 90)),
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_cal)
            print(
                f"  [Caliper-SuperBPE (Config B)] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {cal_bpt:.2f} | Indic: {cal_indic_bpt:.2f} | Fert: {cal_fert:.2f} | Active: {cal_active_cov * 100.0:.1f}%",
                flush=True,
            )

    # Aggregations across scales
    print("\n" + "=" * 175)
    print("PHASE ELEVEN: COMPREHENSIVE VOCABULARY SCALING SUMMARY (MEAN ACROSS 3 SEEDS)")
    print("=" * 175)
    print(
        f"{'Scale':<8} | {'Model Architecture':<30} | {'True LM BPB':<12} | {'Token CE':<10} | {'Bytes/Tok':<10} | {'Indic B/Tok':<12} | {'Fertility':<10} | {'Active Vocab %':<15} | {'>=6B %'}"
    )
    print("-" * 175)

    models = ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]
    summary_scaling: Dict[int, Dict[str, Dict[str, float]]] = {}

    for V in vocab_scales:
        summary_scaling[V] = {}
        for m_name in models:
            recs = [r for r in all_records if r.vocab_size == V and r.model_name == m_name]
            bpb_m = float(np.mean([r.true_lm_bpb for r in recs]))
            ce_m = float(np.mean([r.token_ce_loss for r in recs]))
            bpt_m = float(np.mean([r.bytes_per_token for r in recs]))
            ind_m = float(np.mean([r.indic_bpt for r in recs]))
            fert_m = float(np.mean([r.fertility for r in recs]))
            act_m = float(np.mean([r.active_vocab_pct for r in recs]))
            ge6_m = float(np.mean([r.pct_ge_6b for r in recs]))

            summary_scaling[V][m_name] = {
                "true_lm_bpb": bpb_m,
                "token_ce_loss": ce_m,
                "bytes_per_token": bpt_m,
                "indic_bpt": ind_m,
                "fertility": fert_m,
                "active_vocab_pct": act_m,
                "pct_ge_6b": ge6_m,
            }
            print(
                f"V={V:<6} | {m_name:<30} | {bpb_m:<12.3f} | {ce_m:<10.3f} | {bpt_m:<10.2f} | {ind_m:<12.2f} | {fert_m:<10.2f} | {act_m:<15.1f}% | {ge6_m:<8.1f}%"
            )
        print("-" * 175)
    print("=" * 175 + "\n")

    # 4-Panel Publication Scaling Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel A: True LM BPB vs Vocabulary Size
    ax_a = axes[0, 0]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        bpb_vals = [summary_scaling[V][m_name]["true_lm_bpb"] for V in vocab_scales]
        sz = 14 if "*" in mk else 8
        ax_a.plot(vocab_scales, bpb_vals, mk, color=col, label=m_name, linewidth=2.2, markersize=sz)
        for V, val in zip(vocab_scales, bpb_vals):
            ax_a.annotate(f"{val:.3f}", (V, val + 0.02), fontsize=8.5, color=col, ha="center")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(vocab_scales)
    ax_a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_a.set_title("Panel A: True LM BPB vs Vocabulary Capacity (lower is better)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_a.set_ylabel("True LM BPB", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend()

    # Panel B: Token CE Loss vs Vocabulary Size
    ax_b = axes[0, 1]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        ce_vals = [summary_scaling[V][m_name]["token_ce_loss"] for V in vocab_scales]
        sz = 14 if "*" in mk else 8
        ax_b.plot(vocab_scales, ce_vals, mk, color=col, label=m_name, linewidth=2.2, markersize=sz)
        for V, val in zip(vocab_scales, ce_vals):
            ax_b.annotate(f"{val:.2f}", (V, val + 0.03), fontsize=8.5, color=col, ha="center")
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(vocab_scales)
    ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b.set_title("Panel B: Token Cross-Entropy Loss vs Vocabulary Capacity", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_b.set_ylabel("Token Cross-Entropy Loss (nats)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend()

    # Panel C: Bytes per Token vs Vocabulary Size
    ax_c = axes[1, 0]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        bpt_vals = [summary_scaling[V][m_name]["bytes_per_token"] for V in vocab_scales]
        sz = 14 if "*" in mk else 8
        ax_c.plot(vocab_scales, bpt_vals, mk, color=col, label=m_name, linewidth=2.2, markersize=sz)
        for V, val in zip(vocab_scales, bpt_vals):
            ax_c.annotate(f"{val:.2f}", (V, val + 0.08), fontsize=8.5, color=col, ha="center")
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(vocab_scales)
    ax_c.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_c.set_title("Panel C: Compression (Bytes / Token) vs Vocabulary Capacity", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_c.set_ylabel("Bytes per Token (higher is better)", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend()

    # Panel D: Active Vocabulary Utilization % vs Vocabulary Size
    ax_d = axes[1, 1]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        act_vals = [summary_scaling[V][m_name]["active_vocab_pct"] for V in vocab_scales]
        sz = 14 if "*" in mk else 8
        ax_d.plot(vocab_scales, act_vals, mk, color=col, label=m_name, linewidth=2.2, markersize=sz)
        for V, val in zip(vocab_scales, act_vals):
            ax_d.annotate(f"{val:.1f}%", (V, val + 1.0), fontsize=8.5, color=col, ha="center")
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(vocab_scales)
    ax_d.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_d.set_title("Panel D: Active Vocabulary Utilization % vs Vocabulary Capacity", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_d.set_ylabel("Active Vocabulary Coverage (%)", fontsize=10)
    ax_d.set_ylim(40, 100)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend()

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_eleven_scaling_frontier.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 11 scaling figure to {plot_path}")

    # Persist JSON
    output_data = {
        "summary_by_scale": summary_scaling,
        "all_records": [asdict(r) for r in all_records],
    }
    json_path = Path(__file__).resolve().parent / "phase_eleven_scaling_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Records] Saved Phase 11 scaling ledger to {json_path}")

    return output_data


if __name__ == "__main__":
    run_phase_eleven_scaling(vocab_scales=[8192, 16384, 32768, 65536], seeds=[101, 202, 303], num_docs=500)
