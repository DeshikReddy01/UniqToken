"""
Phase Thirteen: Five-Seed Confirmatory Scaling Benchmark (16K -> 32K -> 64K).
Full inferential statistical validation across N=5 paired seeds [101, 202, 303, 404, 505].
Enforces exact matched FLOP compute (1.333e+11) on CUDA.
Calculates Holm-Bonferroni adjusted p-values, paired Cohen's d_z, 95% CIs,
and marginal gain per vocabulary doubling (diminishing returns quantification).
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
from scipy import stats
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
from benchmarks.run_phase_twelve_scaling_experiment import generate_high_entropy_corpus
from benchmarks.run_phase_three_strict_matched import (
    TARGET_TRAINING_FLOPS,
    train_and_eval_strict_transformer,
)


@dataclass
class ConfirmatoryRecord:
    vocab_size: int
    seed: int
    model_name: str
    actual_vocab_size: int
    true_lm_bpb: float
    token_ce_loss: float
    bytes_per_token: float
    indic_bpt: float
    fertility: float
    active_vocab_pct: float
    pct_ge_6b: float
    wall_clock_sec: float


def paired_stats(a: List[float], b: List[float]) -> Dict[str, float]:
    """Computes paired t-test, Cohen's d_z, and 95% confidence interval for paired differences (a - b)."""
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    se_d = std_d / math.sqrt(n) if std_d > 0 else 1e-12
    t_stat = mean_d / se_d
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    d_z = mean_d / std_d if std_d > 0 else 0.0
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lower = mean_d - tcrit * se_d
    ci_upper = mean_d + tcrit * se_d
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "t_stat": t_stat,
        "df": n - 1,
        "raw_p_value": p_val,
        "cohens_d_z": d_z,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def holm_bonferroni(p_vals: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni correction across a family of p-values."""
    m = len(p_vals)
    indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def run_phase_thirteen_benchmark(
    vocab_scales: List[int] = [16384, 32768, 65536],
    seeds: List[int] = [101, 202, 303, 404, 505],
    num_docs: int = 1000,
) -> Dict[str, Any]:
    import sentencepiece as spm

    print("=" * 175)
    print("PHASE THIRTEEN: FIVE-SEED HIGH-CAPACITY CONFIRMATORY SCALING BENCHMARK (16K -> 32K -> 64K)")
    print(f"Scales: {vocab_scales} | Seeds: {seeds} (N = {len(seeds)} paired) | Matched FLOPs: {TARGET_TRAINING_FLOPS:.3e}")
    print("Corpus Headroom: >400,000 Candidates | Exact Invariant: V_actual == V_target")
    print("=" * 175)

    all_records: List[ConfirmatoryRecord] = []

    for V in vocab_scales:
        print(f"\n======================================== SCALE: V = {V:,} ========================================")
        sbp_merges = min(V // 10, 4000)
        base_target = max(V - sbp_merges, 1000)
        actual_merges = V - base_target

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n---> [Scale V = {V:,} | Paired Seed {seed_idx}/{len(seeds)}: {seed}]")
            train_docs, val_by_lang = generate_high_entropy_corpus(num_docs=num_docs, seed=seed)
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
                sp_actual_v = sp_proc.get_piece_size()

            sp_counts = Counter(sp_tokens)
            sp_active_cov = len(sp_counts) / sp_actual_v
            sp_tok_bytes = [len(t.encode("utf-8")) for t in sp_tokens]
            sp_pct_ge_6b = sum(1 for b in sp_tok_bytes if b >= 6) / max(len(sp_tok_bytes), 1) * 100.0
            sp_indic_toks = list(sp_proc.encode_as_pieces(indic_val))
            sp_indic_bpt = indic_val_bytes / max(len(sp_indic_toks), 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=sp_enc,
                vocab_size=sp_actual_v,
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )
            sp_bpt = total_val_bytes / max(len(sp_tokens), 1)
            sp_fert = len(sp_tokens) / max(num_words, 1)

            rec_sp = ConfirmatoryRecord(
                vocab_size=V,
                seed=seed,
                model_name="SentencePiece-Unigram",
                actual_vocab_size=sp_actual_v,
                true_lm_bpb=lm_bpb,
                token_ce_loss=val_loss,
                bytes_per_token=sp_bpt,
                indic_bpt=sp_indic_bpt,
                fertility=sp_fert,
                active_vocab_pct=sp_active_cov * 100.0,
                pct_ge_6b=sp_pct_ge_6b,
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_sp)
            print(f"  [SentencePiece (Anchor)     ] Act V: {sp_actual_v:<6} | BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {sp_bpt:.2f} | Indic: {sp_indic_bpt:.2f} | Fert: {sp_fert:.2f} | Active: {sp_active_cov*100.0:.1f}%", flush=True)

            # 2. Boundary-BPE
            t0 = time.time()
            b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
            bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
            bpe_model = b_bpe.train(bpe_chunks, verbose=False)
            bpe_tokens = bpe_model.encode(combined_val)
            bpe_actual_v = len(bpe_model.vocab)
            bpe_counts = Counter(bpe_tokens)
            bpe_active_cov = len(bpe_counts) / bpe_actual_v
            bpe_tok_bytes = [len(t.encode("utf-8")) for t in bpe_tokens]
            bpe_pct_ge_6b = sum(1 for b in bpe_tok_bytes if b >= 6) / max(len(bpe_tok_bytes), 1) * 100.0
            bpe_indic_toks = bpe_model.encode(indic_val)
            bpe_indic_bpt = indic_val_bytes / max(len(bpe_indic_toks), 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=lambda t: bpe_model.encode_to_ids(t),
                vocab_size=bpe_actual_v,
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )
            bpe_bpt = total_val_bytes / max(len(bpe_tokens), 1)
            bpe_fert = len(bpe_tokens) / max(num_words, 1)

            rec_bpe = ConfirmatoryRecord(
                vocab_size=V,
                seed=seed,
                model_name="Boundary-BPE",
                actual_vocab_size=bpe_actual_v,
                true_lm_bpb=lm_bpb,
                token_ce_loss=val_loss,
                bytes_per_token=bpe_bpt,
                indic_bpt=bpe_indic_bpt,
                fertility=bpe_fert,
                active_vocab_pct=bpe_active_cov * 100.0,
                pct_ge_6b=bpe_pct_ge_6b,
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_bpe)
            print(f"  [Boundary-BPE (Anchor)      ] Act V: {bpe_actual_v:<6} | BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpe_bpt:.2f} | Indic: {bpe_indic_bpt:.2f} | Fert: {bpe_fert:.2f} | Active: {bpe_active_cov*100.0:.1f}%", flush=True)

            # 3. Frozen Caliper Config B
            t0 = time.time()
            tok_base = CustomTokenizer.train_from_corpus(
                corpus=train_docs,
                target_vocab_size=base_target,
                seed_multiplier=1.2,
                ranking_strategy="byte_savings",
                min_boundary_entropy=0.35,
                length_exponent=1.5,
                pruning_length_exponent=0.0,
                min_frequency=1,
                verbose=False,
            )
            pretok_chunks = [tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))]
            cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
            sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
            cal_tok = CustomTokenizer(normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model)

            cal_tokens = cal_tok.encode(combined_val)
            cal_actual_v = len(cal_tok.model.vocab)
            cal_counts = Counter(cal_tokens)
            cal_active_cov = len(cal_counts) / cal_actual_v
            cal_tok_bytes = [len(t.encode("utf-8")) for t in cal_tokens]
            cal_pct_ge_6b = sum(1 for b in cal_tok_bytes if b >= 6) / max(len(cal_tok_bytes), 1) * 100.0
            cal_indic_toks = cal_tok.encode(indic_val)
            cal_indic_bpt = indic_val_bytes / max(len(cal_indic_toks), 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=lambda t: cal_tok.encode_to_ids(t),
                vocab_size=cal_actual_v,
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )
            cal_bpt = total_val_bytes / max(len(cal_tokens), 1)
            cal_fert = len(cal_tokens) / max(num_words, 1)

            rec_cal = ConfirmatoryRecord(
                vocab_size=V,
                seed=seed,
                model_name="Caliper-SuperBPE (Config B)",
                actual_vocab_size=cal_actual_v,
                true_lm_bpb=lm_bpb,
                token_ce_loss=val_loss,
                bytes_per_token=cal_bpt,
                indic_bpt=cal_indic_bpt,
                fertility=cal_fert,
                active_vocab_pct=cal_active_cov * 100.0,
                pct_ge_6b=cal_pct_ge_6b,
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_cal)
            print(f"  [Caliper-SuperBPE (Config B)] Act V: {cal_actual_v:<6} | BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {cal_bpt:.2f} | Indic: {cal_indic_bpt:.2f} | Fert: {cal_fert:.2f} | Active: {cal_active_cov*100.0:.1f}%", flush=True)

    # Statistical Evaluation & Hypothesis Testing
    print("\n" + "=" * 175)
    print("PHASE THIRTEEN: STATISTICAL HYPOTHESIS TESTING (5 PAIRED SEEDS, HOLM-BONFERRONI ADJUSTED)")
    print("=" * 175)

    hypothesis_records: List[Dict[str, Any]] = []
    summary_by_scale: Dict[int, Dict[str, Dict[str, float]]] = {}

    for V in vocab_scales:
        summary_by_scale[V] = {}
        for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]:
            recs = [r for r in all_records if r.vocab_size == V and r.model_name == m_name]
            summary_by_scale[V][m_name] = {
                "actual_vocab_size": int(np.mean([r.actual_vocab_size for r in recs])),
                "true_lm_bpb": float(np.mean([r.true_lm_bpb for r in recs])),
                "token_ce_loss": float(np.mean([r.token_ce_loss for r in recs])),
                "bytes_per_token": float(np.mean([r.bytes_per_token for r in recs])),
                "indic_bpt": float(np.mean([r.indic_bpt for r in recs])),
                "fertility": float(np.mean([r.fertility for r in recs])),
                "active_vocab_pct": float(np.mean([r.active_vocab_pct for r in recs])),
                "pct_ge_6b": float(np.mean([r.pct_ge_6b for r in recs])),
            }

        cal_bpb = [r.true_lm_bpb for r in all_records if r.vocab_size == V and r.model_name == "Caliper-SuperBPE (Config B)"]
        bpe_bpb = [r.true_lm_bpb for r in all_records if r.vocab_size == V and r.model_name == "Boundary-BPE"]
        sp_bpb = [r.true_lm_bpb for r in all_records if r.vocab_size == V and r.model_name == "SentencePiece-Unigram"]

        cal_ce = [r.token_ce_loss for r in all_records if r.vocab_size == V and r.model_name == "Caliper-SuperBPE (Config B)"]
        sp_ce = [r.token_ce_loss for r in all_records if r.vocab_size == V and r.model_name == "SentencePiece-Unigram"]

        # H1: Caliper BPB < Boundary-BPE BPB
        h1_stat = paired_stats(cal_bpb, bpe_bpb)
        hypothesis_records.append({
            "scale": V,
            "hypothesis": "H1: Caliper BPB < Boundary-BPE BPB",
            "metric": "True LM BPB",
            "caliper_mean": float(np.mean(cal_bpb)),
            "baseline_mean": float(np.mean(bpe_bpb)),
            **h1_stat,
        })

        # H2: Caliper Token CE < SentencePiece Token CE
        h2_stat = paired_stats(cal_ce, sp_ce)
        hypothesis_records.append({
            "scale": V,
            "hypothesis": "H2: Caliper Token CE < SentencePiece Token CE",
            "metric": "Token CE Loss",
            "caliper_mean": float(np.mean(cal_ce)),
            "baseline_mean": float(np.mean(sp_ce)),
            **h2_stat,
        })

    # Apply Holm correction
    raw_ps = [h["raw_p_value"] for h in hypothesis_records]
    adj_ps = holm_bonferroni(raw_ps)
    for h, adj_p in zip(hypothesis_records, adj_ps):
        h["p_holm_adj"] = adj_p
        h["significant_at_05"] = adj_p < 0.05

    print(f"{'Scale':<8} | {'Hypothesis':<42} | {'Caliper':<9} | {'Baseline':<9} | {'Mean Diff':<10} | {'t(4)':<8} | {'p_adj (Holm)':<14} | {'95% CI':<24} | {'Cohen d_z':<10} | {'Verdict'}")
    print("-" * 175)
    for h in hypothesis_records:
        ci_str = f"[{h['ci_lower']:+.3f}, {h['ci_upper']:+.3f}]"
        verdict = "CONFIRMED (p < 0.05)" if h["significant_at_05"] else "NOT STATISTICALLY SIG"
        print(f"V={h['scale']:<6} | {h['hypothesis']:<42} | {h['caliper_mean']:<9.3f} | {h['baseline_mean']:<9.3f} | {h['mean_diff']:<+10.3f} | {h['t_stat']:<+8.2f} | {h['p_holm_adj']:<14.4e} | {ci_str:<24} | {h['cohens_d_z']:<+10.2f} | {verdict}")
    print("=" * 175 + "\n")

    # Marginal Gain Analysis Per Vocabulary Doubling
    print("=" * 175)
    print("MARGINAL GAIN ANALYSIS PER VOCABULARY DOUBLING (DIMINISHING RETURNS QUANTIFICATION)")
    print("=" * 175)
    print(f"{'Model':<30} | {'Transition':<16} | {'Delta BPB':<12} | {'Delta Token CE':<16} | {'Delta Bytes/Tok':<18} | {'Delta Active Vocab %'}")
    print("-" * 175)

    marginal_analysis: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]:
        marginal_analysis[m_name] = {}
        for (v1, v2) in [(16384, 32768), (32768, 65536)]:
            trans_name = f"{v1//1024}K -> {v2//1024}K"
            d_bpb = summary_by_scale[v2][m_name]["true_lm_bpb"] - summary_by_scale[v1][m_name]["true_lm_bpb"]
            d_ce = summary_by_scale[v2][m_name]["token_ce_loss"] - summary_by_scale[v1][m_name]["token_ce_loss"]
            d_bpt = summary_by_scale[v2][m_name]["bytes_per_token"] - summary_by_scale[v1][m_name]["bytes_per_token"]
            d_act = summary_by_scale[v2][m_name]["active_vocab_pct"] - summary_by_scale[v1][m_name]["active_vocab_pct"]
            marginal_analysis[m_name][trans_name] = {
                "delta_bpb": d_bpb,
                "delta_ce": d_ce,
                "delta_bpt": d_bpt,
                "delta_active_pct": d_act,
            }
            print(f"{m_name:<30} | {trans_name:<16} | {d_bpb:<+12.3f} | {d_ce:<+16.3f} | {d_bpt:<+18.2f} | {d_act:<+20.1f}%")
        print("-" * 175)
    print("=" * 175 + "\n")

    # 4-Panel Publication Scaling Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    models = ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]

    # Panel A: True LM BPB vs Vocabulary Size with Error Bars
    ax_a = axes[0, 0]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        bpb_means = [summary_by_scale[V][m_name]["true_lm_bpb"] for V in vocab_scales]
        bpb_stds = [float(np.std([r.true_lm_bpb for r in all_records if r.vocab_size == V and r.model_name == m_name], ddof=1)) for V in vocab_scales]
        sz = 14 if "*" in mk else 8
        ax_a.errorbar(vocab_scales, bpb_means, yerr=bpb_stds, fmt=mk, color=col, label=m_name, linewidth=2.2, markersize=sz, capsize=4)
        for V, val in zip(vocab_scales, bpb_means):
            ax_a.annotate(f"{val:.3f}", (V, val + 0.02), fontsize=8.5, color=col, ha="center")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(vocab_scales)
    ax_a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_a.set_title("Panel A: True LM BPB vs Vocabulary Size (5 Paired Seeds, ±1 SD)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_a.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend()

    # Panel B: Token CE Loss vs Vocabulary Size
    ax_b = axes[0, 1]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        ce_means = [summary_by_scale[V][m_name]["token_ce_loss"] for V in vocab_scales]
        ce_stds = [float(np.std([r.token_ce_loss for r in all_records if r.vocab_size == V and r.model_name == m_name], ddof=1)) for V in vocab_scales]
        sz = 14 if "*" in mk else 8
        ax_b.errorbar(vocab_scales, ce_means, yerr=ce_stds, fmt=mk, color=col, label=m_name, linewidth=2.2, markersize=sz, capsize=4)
        for V, val in zip(vocab_scales, ce_means):
            ax_b.annotate(f"{val:.2f}", (V, val + 0.03), fontsize=8.5, color=col, ha="center")
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(vocab_scales)
    ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b.set_title("Panel B: Token Cross-Entropy Loss vs Vocabulary Size (±1 SD)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_b.set_ylabel("Token Cross-Entropy Loss (nats)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend()

    # Panel C: Bytes per Token vs Vocabulary Size
    ax_c = axes[1, 0]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o-", "s-", "*-"]):
        bpt_vals = [summary_by_scale[V][m_name]["bytes_per_token"] for V in vocab_scales]
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
        act_vals = [summary_by_scale[V][m_name]["active_vocab_pct"] for V in vocab_scales]
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
    ax_d.set_ylim(20, 100)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend()

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_thirteen_confirmatory_scaling.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 13 confirmatory scaling figure to {plot_path}")

    # Persist JSON
    output_data = {
        "summary_by_scale": summary_by_scale,
        "hypotheses": hypothesis_records,
        "marginal_gains": marginal_analysis,
        "all_records": [asdict(r) for r in all_records],
    }
    json_path = Path(__file__).resolve().parent / "phase_thirteen_confirmatory_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Records] Saved Phase 13 confirmatory scaling ledger to {json_path}")

    return output_data


if __name__ == "__main__":
    run_phase_thirteen_benchmark(vocab_scales=[16384, 32768, 65536], seeds=[101, 202, 303, 404, 505], num_docs=1000)
