"""
Phase Nine: Full Five-Seed Confirmatory Benchmark on Config B across 8K & 16K Scales.
Evaluates SentencePiece, Boundary-BPE, and Caliper Config B (Script-Aware + alpha=1.5)
across 5 paired seeds [101, 202, 303, 404, 505] under strict analytical FLOP compute.
Performs full statistical hypothesis testing: t(4), 95% CI, Cohen's d_z, Holm-Bonferroni correction,
and records per-script compression profiles.
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
import scipy.stats as stats
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
class RunRecord:
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
    script_bpt: Dict[str, float]
    wall_clock_sec: float


def run_phase_nine_confirmatory(
    vocab_scales: List[int] = [8192, 16384],
    seeds: List[int] = [101, 202, 303, 404, 505],
    num_docs: int = 500,
) -> Dict[str, Any]:
    import sentencepiece as spm

    print("=" * 175)
    print("PHASE NINE: FULL FIVE-SEED CONFIRMATORY BENCHMARK (CALIPER CONFIG B VS ANCHORS)")
    print(
        f"Scales: {vocab_scales} | Seeds: {seeds} (N = {len(seeds)} paired) | Matched FLOPs: {TARGET_TRAINING_FLOPS:.3e}"
    )
    print("Primary Hypotheses at 16K: H1: BPB(Caliper) < BPB(Boundary-BPE) | H2: CE(Caliper) < CE(SentencePiece)")
    print("=" * 175)

    all_records: List[RunRecord] = []
    languages = ["English", "Hindi", "Telugu", "Tamil", "Bengali", "Arabic", "Chinese", "Russian"]

    for V in vocab_scales:
        print(f"\n==================== VOCABULARY SCALE: V = {V:,} ====================")
        sbp_merges = min(V // 10, 1500)
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

            # 1. SentencePiece-Unigram Anchor
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

            sp_script_bpts = {
                l: len(val_by_lang[l].encode("utf-8")) / max(len(sp_proc.encode_as_pieces(val_by_lang[l])), 1)
                for l in languages
            }

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

            rec_sp = RunRecord(
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
                script_bpt=sp_script_bpts,
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_sp)
            print(
                f"  [SentencePiece (Anchor)     ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {sp_bpt:.2f} | Indic: {sp_indic_bpt:.2f} | Fert: {sp_fert:.2f} | Active: {sp_active_cov * 100.0:.1f}%",
                flush=True,
            )

            # 2. Boundary-BPE Anchor
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
            bpe_script_bpts = {
                l: len(val_by_lang[l].encode("utf-8")) / max(len(bpe_model.encode(val_by_lang[l])), 1)
                for l in languages
            }

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

            rec_bpe = RunRecord(
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
                script_bpt=bpe_script_bpts,
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_bpe)
            print(
                f"  [Boundary-BPE (Anchor)      ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpe_bpt:.2f} | Indic: {bpe_indic_bpt:.2f} | Fert: {bpe_fert:.2f} | Active: {bpe_active_cov * 100.0:.1f}%",
                flush=True,
            )

            # 3. Caliper Config B (Script-Aware + alpha=1.5)
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
            cal_script_bpts = {
                l: len(val_by_lang[l].encode("utf-8")) / max(len(cal_tok.encode(val_by_lang[l])), 1) for l in languages
            }

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

            rec_cal = RunRecord(
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
                script_bpt=cal_script_bpts,
                wall_clock_sec=time.time() - t0,
            )
            all_records.append(rec_cal)
            print(
                f"  [Caliper-SuperBPE (Config B)] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {cal_bpt:.2f} | Indic: {cal_indic_bpt:.2f} | Fert: {cal_fert:.2f} | Active: {cal_active_cov * 100.0:.1f}%",
                flush=True,
            )

    # Statistical Audits
    def compute_paired_stats(vec_a: List[float], vec_b: List[float]) -> Dict[str, float]:
        diffs = [a - b for a, b in zip(vec_a, vec_b)]
        d_bar = float(np.mean(diffs))
        s_d = float(np.std(diffs, ddof=1))
        n = len(diffs)
        t_stat = d_bar / (s_d / math.sqrt(n)) if s_d > 0 else 0.0
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
        ci_half = float(stats.t.ppf(0.975, df=n - 1) * (s_d / math.sqrt(n))) if s_d > 0 else 0.0
        dz = d_bar / s_d if s_d > 0 else 0.0
        return {
            "mean_diff": d_bar,
            "std_diff": s_d,
            "t_stat": t_stat,
            "p_raw": p_val,
            "ci_lower": d_bar - ci_half,
            "ci_upper": d_bar + ci_half,
            "cohens_dz": dz,
        }

    hypotheses = []
    for V in vocab_scales:
        v_records = [r for r in all_records if r.vocab_size == V]
        sp_bpb = [r.true_lm_bpb for r in v_records if r.model_name == "SentencePiece-Unigram"]
        sp_ce = [r.token_ce_loss for r in v_records if r.model_name == "SentencePiece-Unigram"]
        bpe_bpb = [r.true_lm_bpb for r in v_records if r.model_name == "Boundary-BPE"]
        bpe_ce = [r.token_ce_loss for r in v_records if r.model_name == "Boundary-BPE"]
        cal_bpb = [r.true_lm_bpb for r in v_records if r.model_name == "Caliper-SuperBPE (Config B)"]
        cal_ce = [r.token_ce_loss for r in v_records if r.model_name == "Caliper-SuperBPE (Config B)"]

        # H1: Caliper vs Boundary-BPE BPB
        st1 = compute_paired_stats(cal_bpb, bpe_bpb)
        hypotheses.append({"scale": V, "comparison": "Caliper vs Boundary-BPE", "metric": "True LM BPB", **st1})

        # H2: Caliper vs SentencePiece CE
        st2 = compute_paired_stats(cal_ce, sp_ce)
        hypotheses.append({"scale": V, "comparison": "Caliper vs SentencePiece", "metric": "Token CE Loss", **st2})

        # H3: Caliper vs SentencePiece BPB
        st3 = compute_paired_stats(cal_bpb, sp_bpb)
        hypotheses.append({"scale": V, "comparison": "Caliper vs SentencePiece", "metric": "True LM BPB", **st3})

        # H4: Caliper vs Boundary-BPE CE
        st4 = compute_paired_stats(cal_ce, bpe_ce)
        hypotheses.append({"scale": V, "comparison": "Caliper vs Boundary-BPE", "metric": "Token CE Loss", **st4})

    # Holm-Bonferroni Correction
    hypotheses.sort(key=lambda x: x["p_raw"])
    m = len(hypotheses)
    for k, hyp in enumerate(hypotheses):
        adj_p = min(1.0, hyp["p_raw"] * (m - k))
        hyp["p_adj"] = adj_p
        hyp["significant_05"] = bool(adj_p < 0.05)

    print("\n" + "=" * 175)
    print("PHASE NINE: FULL STATISTICAL HYPOTHESIS TESTING REPORT (HOLM-BONFERRONI ADJUSTED, N = 5 SEEDS)")
    print("=" * 175)
    print(
        f"{'Scale':<8} | {'Comparison':<28} | {'Metric':<16} | {'Mean Diff':<12} | {'t(4)':<8} | {'p (raw)':<12} | {'p (Holm)':<12} | {'95% CI':<24} | {'Cohen dz':<10} | {'Sig (p<0.05)'}"
    )
    print("-" * 175)

    for h in hypotheses:
        ci_str = f"[{h['ci_lower']:+.3f}, {h['ci_upper']:+.3f}]"
        sig_str = "YES (p<0.05)" if h["significant_05"] else "NO (p>=0.05)"
        print(
            f"V={h['scale']:<6} | {h['comparison']:<28} | {h['metric']:<16} | {h['mean_diff']:<+12.3f} | {h['t_stat']:<8.2f} | {h['p_raw']:<12.4e} | {h['p_adj']:<12.4e} | {ci_str:<24} | {h['cohens_dz']:<+10.2f} | {sig_str}"
        )
    print("=" * 175 + "\n")

    # Summary by scale & model
    print("=" * 175)
    print("PHASE NINE: AGGREGATE METRICS SUMMARY (MEAN ACROSS 5 PAIRED SEEDS)")
    print("=" * 175)
    print(
        f"{'Scale':<8} | {'Model':<30} | {'True LM BPB':<12} | {'Token CE':<10} | {'Bytes/Tok':<10} | {'Indic B/Tok':<12} | {'Fertility':<10} | {'Active Vocab %':<15} | {'>=6B %'}"
    )
    print("-" * 175)

    models = ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]
    summary_dict = {}
    for V in vocab_scales:
        summary_dict[V] = {}
        for m_name in models:
            recs = [r for r in all_records if r.vocab_size == V and r.model_name == m_name]
            bpb_m = float(np.mean([r.true_lm_bpb for r in recs]))
            ce_m = float(np.mean([r.token_ce_loss for r in recs]))
            bpt_m = float(np.mean([r.bytes_per_token for r in recs]))
            ind_m = float(np.mean([r.indic_bpt for r in recs]))
            fert_m = float(np.mean([r.fertility for r in recs]))
            act_m = float(np.mean([r.active_vocab_pct for r in recs]))
            ge6_m = float(np.mean([r.pct_ge_6b for r in recs]))

            summary_dict[V][m_name] = {
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
    print("=" * 175 + "\n")

    # Script-Level Summary at 16K Scale
    print("=" * 140)
    print("PHASE NINE: PER-SCRIPT COMPRESSION PROFILE AT 16K SCALE (MEAN ACROSS 5 SEEDS)")
    print("=" * 140)
    print(
        f"{'Language / Script':<20} | {'SentencePiece B/Tok':<22} | {'Boundary-BPE B/Tok':<22} | {'Caliper Config B B/Tok':<25} | {'Caliper vs SP Delta'}"
    )
    print("-" * 140)
    script_summary_16k = {}
    recs_16k = [r for r in all_records if r.vocab_size == 16384]
    for lang in languages:
        sp_b = float(np.mean([r.script_bpt[lang] for r in recs_16k if r.model_name == "SentencePiece-Unigram"]))
        bpe_b = float(np.mean([r.script_bpt[lang] for r in recs_16k if r.model_name == "Boundary-BPE"]))
        cal_b = float(np.mean([r.script_bpt[lang] for r in recs_16k if r.model_name == "Caliper-SuperBPE (Config B)"]))
        delta = cal_b - sp_b
        script_summary_16k[lang] = {"sp": sp_b, "boundary_bpe": bpe_b, "caliper": cal_b, "delta": delta}
        print(f"{lang:<20} | {sp_b:<22.2f} | {bpe_b:<22.2f} | {cal_b:<25.2f} | {delta:+6.2f} B/Tok")
    print("=" * 140 + "\n")

    # 4-Panel Publication Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel A: 8K Scale Pareto Frontier
    ax_a = axes[0, 0]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o", "s", "*"]):
        bpb = summary_dict[8192][m_name]["true_lm_bpb"]
        ce = summary_dict[8192][m_name]["token_ce_loss"]
        sz = 260 if "*" in mk else 160
        ax_a.scatter([bpb], [ce], color=col, marker=mk, s=sz, edgecolors="black", label=m_name, zorder=6)
        ax_a.annotate(
            f"{m_name.split()[0]}\nBPB: {bpb:.3f}, CE: {ce:.2f}", (bpb + 0.015, ce + 0.02), fontsize=8.5, color=col
        )
    ax_a.set_title("Panel A: 8,192 Scale Pareto Trade-off (N = 5 Seeds)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("True LM BPB (lower is better)", fontsize=10)
    ax_a.set_ylabel("Token Cross-Entropy Loss (lower is better)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend()

    # Panel B: 16K Scale Pareto Frontier
    ax_b = axes[0, 1]
    for m_name, col, mk in zip(models, ["#1f77b4", "#2ca02c", "#d62728"], ["o", "s", "*"]):
        bpb = summary_dict[16384][m_name]["true_lm_bpb"]
        ce = summary_dict[16384][m_name]["token_ce_loss"]
        sz = 260 if "*" in mk else 160
        ax_b.scatter([bpb], [ce], color=col, marker=mk, s=sz, edgecolors="black", label=m_name, zorder=6)
        ax_b.annotate(
            f"{m_name.split()[0]}\nBPB: {bpb:.3f}, CE: {ce:.2f}", (bpb + 0.015, ce + 0.02), fontsize=8.5, color=col
        )
    ax_b.set_title("Panel B: 16,384 Scale Pareto Trade-off (N = 5 Seeds)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("True LM BPB (lower is better)", fontsize=10)
    ax_b.set_ylabel("Token Cross-Entropy Loss (lower is better)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend()

    # Panel C: Per-Script Compression at 16K
    ax_c = axes[1, 0]
    x = np.arange(len(languages))
    width = 0.28
    sp_vals = [script_summary_16k[l]["sp"] for l in languages]
    bpe_vals = [script_summary_16k[l]["boundary_bpe"] for l in languages]
    cal_vals = [script_summary_16k[l]["caliper"] for l in languages]

    ax_c.bar(x - width, sp_vals, width, label="SentencePiece", color="#1f77b4")
    ax_c.bar(x, bpe_vals, width, label="Boundary-BPE", color="#2ca02c")
    ax_c.bar(x + width, cal_vals, width, label="Caliper Config B", color="#d62728")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(languages, rotation=25, ha="right", fontsize=9)
    ax_c.set_ylabel("Bytes per Token (higher is better)", fontsize=10)
    ax_c.set_title("Panel C: Per-Script Compression (Bytes/Token) at 16K Scale", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_c.legend()

    # Panel D: Paired True LM BPB Boxplot across Seeds at 16K
    ax_d = axes[1, 1]
    sp_bpb_seeds = [r.true_lm_bpb for r in recs_16k if r.model_name == "SentencePiece-Unigram"]
    bpe_bpb_seeds = [r.true_lm_bpb for r in recs_16k if r.model_name == "Boundary-BPE"]
    cal_bpb_seeds = [r.true_lm_bpb for r in recs_16k if r.model_name == "Caliper-SuperBPE (Config B)"]

    ax_d.boxplot(
        [sp_bpb_seeds, bpe_bpb_seeds, cal_bpb_seeds],
        labels=["SentencePiece", "Boundary-BPE", "Caliper Config B"],
        patch_artist=True,
    )
    ax_d.set_ylabel("True LM BPB", fontsize=10)
    ax_d.set_title("Panel D: 16K True LM BPB Distribution across 5 Paired Seeds", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_nine_confirmatory_tradeoff.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 9 confirmatory figure to {plot_path}")

    # Persist JSON
    output_data = {
        "summary_by_scale": summary_dict,
        "hypotheses_tested": hypotheses,
        "script_summary_16k": script_summary_16k,
        "all_records": [asdict(r) for r in all_records],
    }
    json_path = Path(__file__).resolve().parent / "phase_nine_confirmatory_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Records] Saved Phase 9 confirmatory ledger to {json_path}")

    return output_data


if __name__ == "__main__":
    run_phase_nine_confirmatory(vocab_scales=[8192, 16384], seeds=[101, 202, 303, 404, 505], num_docs=500)
