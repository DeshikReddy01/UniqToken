"""
Phase Five: Full 5-Seed Confirmatory Benchmark of Patched Caliper vs SentencePiece & Boundary-BPE.
Runs 30 GPU training runs across 5 paired seeds (101, 202, 303, 404, 505) at 8K and 16K scale.
Applies paired t-tests, 95% CI, Cohen's d_z, and Holm-Bonferroni multiple testing correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
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
from pre_tokenizer import RegexPreTokenizer
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.flop_counter import (
    FLOP_FORMULA_VERSION,
    compute_transformer_flops_per_step,
    plan_training_steps_for_target_flops,
)
from benchmarks.run_phase_three_strict_matched import (
    PAIRED_SEEDS,
    TARGET_TRAINING_FLOPS,
    generate_rich_multilingual_corpus,
    train_and_eval_strict_transformer,
)


@dataclass
class ConfirmatoryRecord:
    scale_target: int
    seed: int
    tokenizer_name: int | str
    actual_vocab_size: int
    vocab_sha256: str
    token_ce_loss: float
    true_lm_bpb: float
    bytes_per_token: float
    fertility: float
    active_vocab_pct: float
    tokens_ge_6b_pct: float
    p50_bytes: float
    p90_bytes: float
    training_steps: int
    target_flops: int
    actual_flops: int
    flop_error_pct: float
    param_count: int
    wall_clock_sec: float
    corpus_train_sha256: str
    corpus_val_sha256: str


def run_confirmatory_benchmark(
    scales: List[int] = [8192, 16384],
    seeds: List[int] = PAIRED_SEEDS,
    num_docs: int = 500,
) -> List[ConfirmatoryRecord]:
    import sentencepiece as spm

    print("=" * 145)
    print("PHASE FIVE: 5-SEED CONFIRMATORY BENCHMARK (PATCHED CALIPER VS SP & BOUNDARY-BPE)")
    print(f"Scales: {scales} | Seeds: {seeds} | Matched FLOPs: {TARGET_TRAINING_FLOPS:.3e} | Runs: {len(scales) * len(seeds) * 3}")
    print("=" * 145)

    records: List[ConfirmatoryRecord] = []
    total_evals = len(scales) * len(seeds) * 3
    eval_idx = 0
    t0_all = time.perf_counter()

    for V in scales:
        print(f"\n===================================================================================================")
        print(f"  VOCABULARY SCALE: V = {V:,} (5 Paired Seeds)")
        print(f"===================================================================================================")

        for seed in seeds:
            print(f"\n---> [Scale: {V:,} | Seed: {seed}] Generating corpus...")
            train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=num_docs, seed=seed)
            combined_val = "\n".join(val_by_lang.values())
            total_val_bytes = len(combined_val.encode("utf-8"))
            words = [w for w in combined_val.split() if w]
            num_words = len(words)

            train_bytes = "\n".join(train_docs).encode("utf-8")
            val_bytes = combined_val.encode("utf-8")
            train_sha = hashlib.sha256(train_bytes).hexdigest()
            val_sha = hashlib.sha256(val_bytes).hexdigest()

            # 1. Caliper (Patched Boundary)
            sbp_merges = min(V // 10, 1500)
            base_target = max(V - sbp_merges, 1000)
            actual_merges = V - base_target

            t0_step = time.perf_counter()
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
            caliper_sbp = CustomTokenizer(normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model)
            caliper_v = len(caliper_sbp.model.vocab)
            t_cal = time.perf_counter() - t0_step

            # 2. SentencePiece Unigram
            t0_step = time.perf_counter()
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
                sp_vocab = [sp_proc.id_to_piece(i) for i in range(sp_proc.get_piece_size())]
                sp_v = sp_proc.get_piece_size()
            t_sp = time.perf_counter() - t0_step

            # 3. Boundary-BPE
            t0_step = time.perf_counter()
            b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
            bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
            bpe_model = b_bpe.train(bpe_chunks, verbose=False)
            bpe_v = bpe_model.vocab_size
            t_bpe = time.perf_counter() - t0_step

            assert caliper_v == V, f"FATAL: Caliper vocab ({caliper_v}) != target ({V})"
            assert sp_v == V, f"FATAL: SentencePiece vocab ({sp_v}) != target ({V})"
            assert bpe_v == V, f"FATAL: Boundary-BPE vocab ({bpe_v}) != target ({V})"

            engines = [
                ("Caliper-SuperBPE (Patched)", caliper_v, list(caliper_sbp.model.vocab.keys()), lambda t: caliper_sbp.encode_to_ids(t), lambda t: caliper_sbp.encode(t)),
                ("SentencePiece-Unigram", sp_v, sp_vocab, lambda t: sp_proc.encode(t, out_type=int), lambda t: sp_proc.encode_as_pieces(t)),
                ("Boundary-BPE", bpe_v, list(bpe_model.vocab), lambda t: bpe_model.encode_to_ids(t), lambda t: bpe_model.encode(t)),
            ]

            for name, v_act, vocab_list, enc_fn, piece_fn in engines:
                eval_idx += 1
                v_sha = hashlib.sha256("".join(sorted(vocab_list)).encode("utf-8")).hexdigest()

                val_tokens = list(piece_fn(combined_val))
                tok_counts = Counter(val_tokens)
                active_cov = len(tok_counts) / v_act
                tok_bytes = [len(t.encode("utf-8")) for t in val_tokens]
                pct_ge_6b = sum(1 for b in tok_bytes if b >= 6) / max(len(tok_bytes), 1) * 100.0
                bpt = total_val_bytes / max(len(val_tokens), 1)
                fertility = len(val_tokens) / max(num_words, 1)

                val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                    enc_fn=enc_fn,
                    vocab_size=v_act,
                    train_texts=train_docs[:300],
                    val_text=combined_val,
                    total_val_bytes=total_val_bytes,
                    target_flops=TARGET_TRAINING_FLOPS,
                    seed=seed,
                )

                rec = ConfirmatoryRecord(
                    scale_target=V,
                    seed=seed,
                    tokenizer_name=name,
                    actual_vocab_size=v_act,
                    vocab_sha256=v_sha,
                    token_ce_loss=val_loss,
                    true_lm_bpb=lm_bpb,
                    bytes_per_token=bpt,
                    fertility=fertility,
                    active_vocab_pct=active_cov * 100.0,
                    tokens_ge_6b_pct=pct_ge_6b,
                    p50_bytes=float(np.percentile(tok_bytes, 50)),
                    p90_bytes=float(np.percentile(tok_bytes, 90)),
                    training_steps=steps,
                    target_flops=TARGET_TRAINING_FLOPS,
                    actual_flops=act_flops,
                    flop_error_pct=flop_err,
                    param_count=params,
                    wall_clock_sec=wall_clock,
                    corpus_train_sha256=train_sha,
                    corpus_val_sha256=val_sha,
                )
                records.append(rec)
                print(f"  [{eval_idx:>2}/{total_evals}] [{name:<28}] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {active_cov*100.0:.1f}% | >=6B: {pct_ge_6b:.1f}% | Time: {wall_clock:.2f}s", flush=True)

    # Save JSON records
    json_path = Path(__file__).resolve().parent / "phase_five_confirmatory_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    print(f"\n[Confirmatory Benchmark] Saved {len(records)} records to {json_path}")

    return records


def compute_paired_statistics(records: List[ConfirmatoryRecord]) -> None:
    scales = sorted(list(set(r.scale_target for r in records)))
    comparisons = [
        ("Caliper-SuperBPE (Patched)", "SentencePiece-Unigram"),
        ("Caliper-SuperBPE (Patched)", "Boundary-BPE"),
    ]
    metrics = ["token_ce_loss", "true_lm_bpb"]

    hypothesis_results = []

    for V in scales:
        scale_recs = [r for r in records if r.scale_target == V]
        for name_a, name_b in comparisons:
            recs_a = {r.seed: r for r in scale_recs if r.tokenizer_name == name_a}
            recs_b = {r.seed: r for r in scale_recs if r.tokenizer_name == name_b}
            seeds = sorted(list(set(recs_a.keys()) & set(recs_b.keys())))
            assert len(seeds) == 5, f"Expected 5 paired seeds, got {len(seeds)}"

            for metric in metrics:
                diffs = np.array([getattr(recs_a[s], metric) - getattr(recs_b[s], metric) for s in seeds])
                d_bar = float(np.mean(diffs))
                s_d = float(np.std(diffs, ddof=1))
                df = len(seeds) - 1
                t_stat = d_bar / (s_d / math.sqrt(len(seeds)))
                p_raw = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))

                # 95% CI
                t_crit = float(stats.t.ppf(0.975, df=df))
                se = s_d / math.sqrt(len(seeds))
                ci_low = d_bar - t_crit * se
                ci_high = d_bar + t_crit * se
                d_z = d_bar / s_d if s_d > 0 else 0.0

                label_metric = "Token CE Loss" if metric == "token_ce_loss" else "True LM BPB"
                comp_str = f"Caliper vs {'SentencePiece' if 'SentencePiece' in name_b else 'Boundary-BPE'}"

                hypothesis_results.append({
                    "scale": V,
                    "comp": comp_str,
                    "metric": label_metric,
                    "d_bar": d_bar,
                    "s_d": s_d,
                    "t_stat": t_stat,
                    "p_raw": p_raw,
                    "ci": (ci_low, ci_high),
                    "d_z": d_z,
                })

    # Holm-Bonferroni correction applied across the eight paired hypotheses
    hypothesis_results.sort(key=lambda x: x["p_raw"])
    m = len(hypothesis_results)
    for k, res in enumerate(hypothesis_results, 1):
        res["p_holm"] = min(res["p_raw"] * (m - k + 1), 1.0)
        res["is_sig"] = res["p_holm"] < 0.05

    # Enforce monotonicity of adjusted p-values
    for i in range(1, m):
        if hypothesis_results[i]["p_holm"] < hypothesis_results[i - 1]["p_holm"]:
            hypothesis_results[i]["p_holm"] = hypothesis_results[i - 1]["p_holm"]

    hypothesis_results.sort(key=lambda x: (x["scale"], x["comp"], x["metric"]))

    print("\n" + "=" * 145)
    print("PHASE FIVE: CONFIRMATORY STATISTICAL AUDIT (PAIRED T-TEST, 95% CI, COHEN'S d_z, HOLM CORRECTION)")
    print("=" * 145)
    print(f"{'Scale':<7} | {'Comparison':<27} | {'Metric':<14} | {'d_bar':>7} | {'s_d':>7} | {'t(4)':>7} | {'p-raw':>10} | {'p-Holm':>10} | {'95% CI':<21} | {'Cohen dz':>8} | {'Holm Sig'}")
    print("-" * 145)

    for r in hypothesis_results:
        ci_str = f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        sig_str = "YES (p<0.05)" if r["is_sig"] else "NO"
        print(f"{r['scale']:<7} | {r['comp']:<27} | {r['metric']:<14} | {r['d_bar']:>+7.3f} | {r['s_d']:>7.4f} | {r['t_stat']:>7.2f} | {r['p_raw']:>10.2e} | {r['p_holm']:>10.2e} | {ci_str:<21} | {r['d_z']:>+8.2f} | {sig_str}")
    print("=" * 145 + "\n")


def generate_confirmatory_plots(records: List[ConfirmatoryRecord]) -> None:
    scales = sorted(list(set(r.scale_target for r in records)))
    tokenizers = ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Patched)"]
    colors = {"SentencePiece-Unigram": "#1f77b4", "Boundary-BPE": "#2ca02c", "Caliper-SuperBPE (Patched)": "#d62728"}
    markers = {"SentencePiece-Unigram": "o", "Boundary-BPE": "s", "Caliper-SuperBPE (Patched)": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel A: True LM BPB vs Scale
    ax_a = axes[0, 0]
    for tok in tokenizers:
        means = [float(np.mean([r.true_lm_bpb for r in records if r.scale_target == V and r.tokenizer_name == tok])) for V in scales]
        stds = [float(np.std([r.true_lm_bpb for r in records if r.scale_target == V and r.tokenizer_name == tok], ddof=1)) for V in scales]
        ax_a.errorbar([f"{V//1024}K" for V in scales], means, yerr=stds, label=tok, color=colors[tok], marker=markers[tok], linewidth=2, capsize=4)
    ax_a.set_title("Panel A: True LM BPB (lower is better)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("True LM BPB (Bits / Byte)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(frameon=True)

    # Panel B: Token Cross-Entropy Loss
    ax_b = axes[0, 1]
    for tok in tokenizers:
        means = [float(np.mean([r.token_ce_loss for r in records if r.scale_target == V and r.tokenizer_name == tok])) for V in scales]
        stds = [float(np.std([r.token_ce_loss for r in records if r.scale_target == V and r.tokenizer_name == tok], ddof=1)) for V in scales]
        ax_b.errorbar([f"{V//1024}K" for V in scales], means, yerr=stds, label=tok, color=colors[tok], marker=markers[tok], linewidth=2, capsize=4)
    ax_b.set_title("Panel B: Token Cross-Entropy Loss (lower is better)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Validation Cross-Entropy (nats)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(frameon=True)

    # Panel C: Bytes per Token
    ax_c = axes[1, 0]
    for tok in tokenizers:
        means = [float(np.mean([r.bytes_per_token for r in records if r.scale_target == V and r.tokenizer_name == tok])) for V in scales]
        stds = [float(np.std([r.bytes_per_token for r in records if r.scale_target == V and r.tokenizer_name == tok], ddof=1)) for V in scales]
        ax_c.errorbar([f"{V//1024}K" for V in scales], means, yerr=stds, label=tok, color=colors[tok], marker=markers[tok], linewidth=2, capsize=4)
    ax_c.set_title("Panel C: Subword Compression (Bytes/Token) (higher is better)", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Bytes per Token", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(frameon=True)

    # Panel D: 2D Pareto Trade-off
    ax_d = axes[1, 1]
    for tok in tokenizers:
        for V in scales:
            bpb_m = float(np.mean([r.true_lm_bpb for r in records if r.scale_target == V and r.tokenizer_name == tok]))
            ce_m = float(np.mean([r.token_ce_loss for r in records if r.scale_target == V and r.tokenizer_name == tok]))
            lbl = f"{tok} ({V//1024}K)" if V == 16384 else None
            sz = 140 if V == 8192 else 200
            ax_d.scatter([bpb_m], [ce_m], color=colors[tok], marker=markers[tok], s=sz, edgecolors="black", label=lbl, zorder=5)
            ax_d.annotate(f"{V//1024}K", (bpb_m + 0.02, ce_m + 0.02), fontsize=8)

    ax_d.set_title("Panel D: 2D Pareto Frontier (Token CE vs True LM BPB)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("True LM BPB (lower is better)", fontsize=10)
    ax_d.set_ylabel("Token Cross-Entropy Loss (lower is better)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(frameon=True)

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_five_confirmatory_tradeoff.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved 4-panel confirmatory trade-off chart to {plot_path}")


if __name__ == "__main__":
    records = run_confirmatory_benchmark(scales=[8192, 16384], seeds=PAIRED_SEEDS, num_docs=500)
    compute_paired_statistics(records)
    generate_confirmatory_plots(records)
