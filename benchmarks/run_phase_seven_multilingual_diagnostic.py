"""
Phase Seven: Multilingual Candidate-Generation & Vocabulary Allocation Diagnostic Audit.
Performs fine-grained per-script breakdown (Latin, Devanagari, Telugu, Tamil, Bengali, Arabic, Chinese, Russian),
computes length-binned EM candidate survival rates, and profiles 16K vocabulary allocation.
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
from benchmarks.flop_counter import plan_training_steps_for_target_flops
from benchmarks.run_phase_three_strict_matched import (
    TARGET_TRAINING_FLOPS,
    generate_rich_multilingual_corpus,
    train_and_eval_strict_transformer,
)


def classify_token_script(tok: str) -> str:
    """Classifies a token into its primary Unicode script category."""
    clean = tok.replace("\u2581", "").strip()
    if not clean:
        return "Space/Control"
    if clean.startswith("<|") and clean.endswith("|>"):
        return "Special"
    if clean.startswith("<0x") and clean.endswith(">"):
        return "ByteFallback"
    if clean.isdigit() or clean.startswith("0x") or clean.startswith("SYS_"):
        return "Numeric/ID"

    first_char = clean[0]
    name = unicodedata.name(first_char, "")
    if "LATIN" in name:
        return "Latin"
    elif "DEVANAGARI" in name:
        return "Devanagari"
    elif "TELUGU" in name:
        return "Telugu"
    elif "TAMIL" in name:
        return "Tamil"
    elif "BENGALI" in name:
        return "Bengali"
    elif "ARABIC" in name:
        return "Arabic"
    elif "CJK" in name or "IDEOGRAPH" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
        return "Chinese/CJK"
    elif "CYRILLIC" in name:
        return "Cyrillic"
    elif any(unicodedata.category(c).startswith("P") for c in clean):
        return "Punctuation"
    return "Other"


def run_phase_seven_audit(
    target_vocab: int = 16384,
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 500,
) -> Dict[str, Any]:
    import sentencepiece as spm

    print("=" * 165)
    print("PHASE SEVEN: MULTILINGUAL CANDIDATE-GENERATION & VOCABULARY ALLOCATION DIAGNOSTIC AUDIT")
    print(f"Scale: V = {target_vocab:,} | Seeds: {seeds} | Compute: Strict Matched FLOPs ({TARGET_TRAINING_FLOPS:.3e})")
    print("Goal: Isolate exact per-script compression mechanisms, candidate survival rates, and vocabulary allocation")
    print("=" * 165)

    all_seed_results = []
    survival_data_by_seed = []
    vocab_alloc_by_seed = []

    languages = ["English", "Hindi", "Telugu", "Tamil", "Bengali", "Arabic", "Chinese", "Russian"]

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n---> [Diagnostic Seed {seed_idx}/{len(seeds)}: {seed}]")
        train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=num_docs, seed=seed)
        combined_val = "\n".join(val_by_lang.values())
        total_val_bytes = len(combined_val.encode("utf-8"))

        # 1. Train SentencePiece Unigram
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
            sp_enc = lambda t: sp_proc.encode(t, out_type=int)
            sp_vocab = [sp_proc.id_to_piece(i) for i in range(sp_proc.get_piece_size())]

        # 2. Train Caliper (Space-Prefixed) with full intermediate candidate capture
        norm = Normalizer()
        pretok = RegexPreTokenizer()
        pretok_chunks = [c for d in train_docs for c in pretok.pre_tokenize(norm.normalize(d))]
        chunk_counts = Counter(pretok_chunks)

        sbp_merges = min(target_vocab // 10, 1500)
        base_target = max(target_vocab - sbp_merges, 1000)
        actual_merges = target_vocab - base_target

        seed_builder = SeedVocabularyBuilder(
            target_vocab_size=base_target,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            min_frequency=1,
        )
        seed_tokens = seed_builder.build_seed_vocab(chunk_counts)
        seed_token_strings = [t.token for t in seed_tokens]

        trainer = UnigramTrainer(
            target_vocab_size=base_target,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            pruning_length_exponent=0.0,
            min_frequency=1,
        )
        base_model = trainer.train(pretok_chunks, verbose=False)
        surviving_tokens = list(base_model.vocab.keys())

        cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
        sbp_model = cem.optimize(base_model, chunks=pretok_chunks)
        caliper_tok = CustomTokenizer(normalizer=norm, pre_tokenizer=pretok, model=sbp_model)
        caliper_vocab = list(caliper_tok.model.vocab.keys())

        # Measure Candidate Survival by Byte-Length Bins
        length_bins = {
            "1-3B": (1, 3),
            "4-5B": (4, 5),
            "6-8B": (6, 8),
            "9+B": (9, 999),
        }
        survival_stats = {}
        for bin_label, (min_b, max_b) in length_bins.items():
            seed_cnt = sum(1 for t in seed_token_strings if min_b <= len(t.encode("utf-8")) <= max_b)
            surv_cnt = sum(1 for t in surviving_tokens if min_b <= len(t.encode("utf-8")) <= max_b)
            rate = surv_cnt / max(seed_cnt, 1)
            survival_stats[bin_label] = {
                "generated": seed_cnt,
                "survived": surv_cnt,
                "survival_rate": rate,
            }
        survival_data_by_seed.append(survival_stats)

        # Measure 16K Vocabulary Allocation by Byte Length and Script
        cal_len_hist = Counter(len(t.encode("utf-8")) for t in caliper_vocab)
        sp_len_hist = Counter(len(t.encode("utf-8")) for t in sp_vocab)

        cal_script_hist = Counter(classify_token_script(t) for t in caliper_vocab)
        sp_script_hist = Counter(classify_token_script(t) for t in sp_vocab)

        vocab_alloc_by_seed.append({
            "caliper_lengths": dict(cal_len_hist),
            "sp_lengths": dict(sp_len_hist),
            "caliper_scripts": dict(cal_script_hist),
            "sp_scripts": dict(sp_script_hist),
        })

        # Train Strict Transformers for Whole Corpus
        sp_loss, sp_bpb, _, _, _, _, _ = train_and_eval_strict_transformer(
            enc_fn=sp_enc,
            vocab_size=len(sp_vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        cal_loss, cal_bpb, _, _, _, _, _ = train_and_eval_strict_transformer(
            enc_fn=lambda t: caliper_tok.encode_to_ids(t),
            vocab_size=len(caliper_vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )

        # Per-Language Breakdown
        lang_breakdown = {}
        for lang in languages:
            val_text = val_by_lang[lang]
            val_bytes = len(val_text.encode("utf-8"))
            words = [w for w in val_text.split() if w]
            n_words = len(words)

            # SP stats
            sp_toks = list(sp_proc.encode_as_pieces(val_text))
            sp_bpt = val_bytes / max(len(sp_toks), 1)
            sp_fert = len(sp_toks) / max(n_words, 1)
            sp_bytes_list = [len(t.encode("utf-8")) for t in sp_toks]
            sp_ge6_pct = sum(1 for b in sp_bytes_list if b >= 6) / max(len(sp_bytes_list), 1) * 100.0
            sp_p50 = float(np.percentile(sp_bytes_list, 50))
            sp_p90 = float(np.percentile(sp_bytes_list, 90))

            # Caliper stats
            cal_toks = caliper_tok.encode(val_text)
            cal_bpt = val_bytes / max(len(cal_toks), 1)
            cal_fert = len(cal_toks) / max(n_words, 1)
            cal_bytes_list = [len(t.encode("utf-8")) for t in cal_toks]
            cal_ge6_pct = sum(1 for b in cal_bytes_list if b >= 6) / max(len(cal_bytes_list), 1) * 100.0
            cal_p50 = float(np.percentile(cal_bytes_list, 50))
            cal_p90 = float(np.percentile(cal_bytes_list, 90))

            lang_breakdown[lang] = {
                "val_bytes": val_bytes,
                "words": n_words,
                "sp": {
                    "bpt": sp_bpt,
                    "fertility": sp_fert,
                    "p50_bytes": sp_p50,
                    "p90_bytes": sp_p90,
                    "ge6_pct": sp_ge6_pct,
                },
                "caliper": {
                    "bpt": cal_bpt,
                    "fertility": cal_fert,
                    "p50_bytes": cal_p50,
                    "p90_bytes": cal_p90,
                    "ge6_pct": cal_ge6_pct,
                },
                "bpt_gap": sp_bpt - cal_bpt,
                "fertility_gap": cal_fert - sp_fert,
            }

            print(f"  [{lang:<10}] SP B/Tok: {sp_bpt:5.2f} (Fert: {sp_fert:4.2f}, p50/90: {sp_p50:2.0f}/{sp_p90:2.0f}B) | Caliper B/Tok: {cal_bpt:5.2f} (Fert: {cal_fert:4.2f}, p50/90: {cal_p50:2.0f}/{cal_p90:2.0f}B) | Delta B/Tok: {sp_bpt - cal_bpt:+5.2f}", flush=True)

        all_seed_results.append({
            "seed": seed,
            "sp_overall": {"bpb": sp_bpb, "ce": sp_loss},
            "caliper_overall": {"bpb": cal_bpb, "ce": cal_loss},
            "lang_breakdown": lang_breakdown,
        })

    # Aggregation across seeds
    print("\n" + "=" * 165)
    print("PHASE SEVEN: PER-LANGUAGE MULTILINGUAL PROFILING SUMMARY (MEAN ACROSS 3 SEEDS AT 16K)")
    print("=" * 165)
    print(f"{'Language/Script':<16} | {'SP Bytes/Tok':<14} | {'Cal Bytes/Tok':<14} | {'Delta B/Tok':<12} | {'SP Fertility':<13} | {'Cal Fertility':<13} | {'SP >=6B %':<11} | {'Cal >=6B %':<11} | {'SP p50/90':<11} | {'Cal p50/90'}")
    print("-" * 165)

    summary_by_lang = {}
    for lang in languages:
        sp_bpt_m = float(np.mean([s["lang_breakdown"][lang]["sp"]["bpt"] for s in all_seed_results]))
        cal_bpt_m = float(np.mean([s["lang_breakdown"][lang]["caliper"]["bpt"] for s in all_seed_results]))
        bpt_gap_m = sp_bpt_m - cal_bpt_m

        sp_fert_m = float(np.mean([s["lang_breakdown"][lang]["sp"]["fertility"] for s in all_seed_results]))
        cal_fert_m = float(np.mean([s["lang_breakdown"][lang]["caliper"]["fertility"] for s in all_seed_results]))

        sp_ge6_m = float(np.mean([s["lang_breakdown"][lang]["sp"]["ge6_pct"] for s in all_seed_results]))
        cal_ge6_m = float(np.mean([s["lang_breakdown"][lang]["caliper"]["ge6_pct"] for s in all_seed_results]))

        sp_p50_m = float(np.mean([s["lang_breakdown"][lang]["sp"]["p50_bytes"] for s in all_seed_results]))
        sp_p90_m = float(np.mean([s["lang_breakdown"][lang]["sp"]["p90_bytes"] for s in all_seed_results]))
        cal_p50_m = float(np.mean([s["lang_breakdown"][lang]["caliper"]["p50_bytes"] for s in all_seed_results]))
        cal_p90_m = float(np.mean([s["lang_breakdown"][lang]["caliper"]["p90_bytes"] for s in all_seed_results]))

        summary_by_lang[lang] = {
            "sp_bpt": sp_bpt_m,
            "caliper_bpt": cal_bpt_m,
            "bpt_gap": bpt_gap_m,
            "sp_fertility": sp_fert_m,
            "caliper_fertility": cal_fert_m,
            "sp_ge6_pct": sp_ge6_m,
            "caliper_ge6_pct": cal_ge6_m,
            "sp_p50": sp_p50_m,
            "sp_p90": sp_p90_m,
            "caliper_p50": cal_p50_m,
            "caliper_p90": cal_p90_m,
        }

        sp_len_str = f"{sp_p50_m:.0f}/{sp_p90_m:.0f}B"
        cal_len_str = f"{cal_p50_m:.0f}/{cal_p90_m:.0f}B"
        print(f"{lang:<16} | {sp_bpt_m:<14.2f} | {cal_bpt_m:<14.2f} | {bpt_gap_m:<+12.2f} | {sp_fert_m:<13.2f} | {cal_fert_m:<13.2f} | {sp_ge6_m:<10.1f}% | {cal_ge6_m:<10.1f}% | {sp_len_str:<11} | {cal_len_str}")
    print("=" * 165 + "\n")

    # Aggregate Candidate Survival Rates
    print("=" * 120)
    print("PHASE SEVEN: CANDIDATE SURVIVAL RATE BY BYTE-LENGTH BINS (CALIPER EM PRUNING)")
    print("=" * 120)
    print(f"{'Length Bin':<15} | {'Candidates Generated (Seed)':<30} | {'Surviving in 16K Vocab':<25} | {'Survival Rate %'}")
    print("-" * 120)

    summary_survival = {}
    for bin_lbl in ["1-3B", "4-5B", "6-8B", "9+B"]:
        gen_m = float(np.mean([s[bin_lbl]["generated"] for s in survival_data_by_seed]))
        surv_m = float(np.mean([s[bin_lbl]["survived"] for s in survival_data_by_seed]))
        rate_m = float(np.mean([s[bin_lbl]["survival_rate"] for s in survival_data_by_seed])) * 100.0
        summary_survival[bin_lbl] = {
            "generated": gen_m,
            "survived": surv_m,
            "survival_rate_pct": rate_m,
        }
        print(f"{bin_lbl:<15} | {gen_m:<30.1f} | {surv_m:<25.1f} | {rate_m:<10.1f}%")
    print("=" * 120 + "\n")

    # Aggregate Vocabulary Allocation by Script
    all_scripts = ["Latin", "Devanagari", "Telugu", "Tamil", "Bengali", "Arabic", "Chinese/CJK", "Cyrillic", "Numeric/ID", "Punctuation", "Other", "Special", "ByteFallback"]
    summary_scripts = {}
    print("=" * 120)
    print("PHASE SEVEN: 16K VOCABULARY SCRIPT ALLOCATION (CALIPER VS SENTENCEPIECE)")
    print("=" * 120)
    print(f"{'Script Category':<20} | {'Caliper Token Count':<22} | {'Caliper Vocab %':<18} | {'SP Token Count':<18} | {'SP Vocab %'}")
    print("-" * 120)

    for sc in all_scripts:
        cal_cnt_m = float(np.mean([v["caliper_scripts"].get(sc, 0) for v in vocab_alloc_by_seed]))
        sp_cnt_m = float(np.mean([v["sp_scripts"].get(sc, 0) for v in vocab_alloc_by_seed]))
        cal_pct = cal_cnt_m / target_vocab * 100.0
        sp_pct = sp_cnt_m / target_vocab * 100.0
        summary_scripts[sc] = {
            "caliper_count": cal_cnt_m,
            "caliper_pct": cal_pct,
            "sp_count": sp_cnt_m,
            "sp_pct": sp_pct,
        }
        if cal_cnt_m > 0 or sp_cnt_m > 0:
            print(f"{sc:<20} | {cal_cnt_m:<22.1f} | {cal_pct:<17.1f}% | {sp_cnt_m:<18.1f} | {sp_pct:<10.1f}%")
    print("=" * 120 + "\n")

    # Plot 4-Panel Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # Panel A: Bytes/Token by Language
    ax_a = axes[0, 0]
    x = np.arange(len(languages))
    width = 0.35
    sp_bpts = [summary_by_lang[l]["sp_bpt"] for l in languages]
    cal_bpts = [summary_by_lang[l]["caliper_bpt"] for l in languages]
    ax_a.bar(x - width/2, sp_bpts, width, label="SentencePiece", color="#1f77b4")
    ax_a.bar(x + width/2, cal_bpts, width, label="Caliper (Space-Prefixed)", color="#d62728")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(languages, rotation=25, ha="right", fontsize=9)
    ax_a.set_ylabel("Bytes per Token (higher is better)", fontsize=10)
    ax_a.set_title("Panel A: Compression (Bytes/Token) by Language/Script", fontsize=11, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_a.legend()

    # Panel B: Fertility by Language
    ax_b = axes[0, 1]
    sp_ferts = [summary_by_lang[l]["sp_fertility"] for l in languages]
    cal_ferts = [summary_by_lang[l]["caliper_fertility"] for l in languages]
    ax_b.bar(x - width/2, sp_ferts, width, label="SentencePiece", color="#1f77b4")
    ax_b.bar(x + width/2, cal_ferts, width, label="Caliper (Space-Prefixed)", color="#d62728")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(languages, rotation=25, ha="right", fontsize=9)
    ax_b.set_ylabel("Fertility (Tokens per Word) (lower is better)", fontsize=10)
    ax_b.set_title("Panel B: Fertility (Tokens/Word) by Language/Script", fontsize=11, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_b.legend()

    # Panel C: Candidate Survival Rate by Length Bin
    ax_c = axes[1, 0]
    bins = ["1-3B", "4-5B", "6-8B", "9+B"]
    rates = [summary_survival[b]["survival_rate_pct"] for b in bins]
    gens = [summary_survival[b]["generated"] for b in bins]
    survs = [summary_survival[b]["survived"] for b in bins]

    x_c = np.arange(len(bins))
    ax_c.bar(x_c, rates, color="#9467bd", width=0.5, edgecolor="black")
    for i, r in enumerate(rates):
        ax_c.text(i, r + 2, f"{r:.1f}%\n({survs[i]:.0f}/{gens[i]:.0f})", ha="center", fontsize=8.5, fontweight="bold")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(bins, fontsize=10)
    ax_c.set_ylabel("EM Pruning Survival Rate (%)", fontsize=10)
    ax_c.set_ylim(0, 115)
    ax_c.set_title("Panel C: Caliper EM Candidate Survival Rate by Byte-Length", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel D: Vocabulary Length Distribution (Caliper vs SP)
    ax_d = axes[1, 1]
    lengths = list(range(1, 17))
    cal_len_dist = [np.mean([v["caliper_lengths"].get(L, 0) for v in vocab_alloc_by_seed]) for L in lengths]
    sp_len_dist = [np.mean([v["sp_lengths"].get(L, 0) for v in vocab_alloc_by_seed]) for L in lengths]

    ax_d.plot(lengths, sp_len_dist, marker="o", color="#1f77b4", label="SentencePiece", linewidth=2)
    ax_d.plot(lengths, cal_len_dist, marker="s", color="#d62728", label="Caliper (Space-Prefixed)", linewidth=2)
    ax_d.set_xticks(lengths)
    ax_d.set_xlabel("Token Byte Length", fontsize=10)
    ax_d.set_ylabel("Number of Vocabulary Entries (out of 16,384)", fontsize=10)
    ax_d.set_title("Panel D: 16K Vocabulary Byte-Length Distribution", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend()

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_seven_multilingual_diagnostic.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 7 diagnostic figure to {plot_path}")

    # Output full JSON
    audit_data = {
        "summary_by_language": summary_by_lang,
        "summary_survival_by_length": summary_survival,
        "summary_scripts": summary_scripts,
        "raw_seeds": all_seed_results,
    }
    json_path = Path(__file__).resolve().parent / "phase_seven_multilingual_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"[Audit] Saved Phase 7 structured audit to {json_path}")

    return audit_data


if __name__ == "__main__":
    run_phase_seven_audit(target_vocab=16384, seeds=[101, 202, 303], num_docs=500)
