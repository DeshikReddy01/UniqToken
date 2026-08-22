"""
Phase Three Statistical Audit, Pareto Trade-off Plotter, and Tokenizer Profiler.
Implements Steps 1, 2, 3, and 4 of the Research Roadmap.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from tokenizer import CustomTokenizer
from benchmarks.run_phase_three_strict_matched import generate_rich_multilingual_corpus

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT / "benchmarks"
RECORDS_FILE = BENCHMARKS_DIR / "phase_three_strict_matched_records.json"
FROZEN_FILE = BENCHMARKS_DIR / "phase_three_baseline_frozen.json"


def freeze_baseline() -> None:
    """Step 1: Freezes the current benchmark as the definitive Phase 3A reproducibility baseline."""
    if RECORDS_FILE.exists():
        shutil.copy(RECORDS_FILE, FROZEN_FILE)
        print(f"[Step 1] Baseline successfully frozen and saved to {FROZEN_FILE}")
    else:
        raise FileNotFoundError(f"Records file {RECORDS_FILE} not found.")


def audit_statistics():
    """Step 2: Audits paired statistics across all 5 seeds with Holm-Bonferroni multiple-comparison correction."""
    with open(FROZEN_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    data: Dict[int, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for r in records:
        scale = r["scale_target"]
        tok = r["tokenizer"]
        seed = r["seed"]
        if scale not in data:
            data[scale] = {}
        if tok not in data[scale]:
            data[scale][tok] = {}
        data[scale][tok][seed] = {
            "true_lm_bpb": r["true_lm_bpb"],
            "val_loss_nats": r["val_loss_nats"],
            "bytes_per_token": r["bytes_per_token"],
        }

    scales = sorted(data.keys())
    print("\n" + "=" * 135)
    print("STEP 2: STATISTICAL AUDIT (PAIRED T-TEST, 95% CI, COHEN'S d_z, HOLM-BONFERRONI MULTIPLE TESTING CORRECTION)")
    print("=" * 135)

    comparisons = []
    for V in scales:
        cal = data[V]["Caliper-SuperBPE"]
        sp = data[V]["SentencePiece-Unigram"]
        bpe = data[V]["Boundary-BPE"]
        seeds = sorted(cal.keys())

        deltas_sp_bpb = [cal[s]["true_lm_bpb"] - sp[s]["true_lm_bpb"] for s in seeds]
        deltas_bpe_bpb = [cal[s]["true_lm_bpb"] - bpe[s]["true_lm_bpb"] for s in seeds]
        deltas_sp_loss = [cal[s]["val_loss_nats"] - sp[s]["val_loss_nats"] for s in seeds]
        deltas_bpe_loss = [cal[s]["val_loss_nats"] - bpe[s]["val_loss_nats"] for s in seeds]

        comparisons.append((V, "Caliper vs SentencePiece", "True LM BPB", deltas_sp_bpb))
        comparisons.append((V, "Caliper vs Boundary-BPE", "True LM BPB", deltas_bpe_bpb))
        comparisons.append((V, "Caliper vs SentencePiece", "Token CE Loss", deltas_sp_loss))
        comparisons.append((V, "Caliper vs Boundary-BPE", "Token CE Loss", deltas_bpe_loss))

    results = []
    for scale, pair_name, metric, deltas in comparisons:
        arr = np.array(deltas)
        n = len(arr)
        df = n - 1
        d_bar = np.mean(arr)
        s_d = np.std(arr, ddof=1)
        se = s_d / math.sqrt(n)
        t_stat = d_bar / se if se > 1e-12 else 0.0
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
        t_crit = stats.t.ppf(0.975, df=df)
        ci_lower = d_bar - t_crit * se
        ci_upper = d_bar + t_crit * se
        cohen_dz = d_bar / s_d if s_d > 1e-12 else 0.0

        results.append(
            {
                "scale": scale,
                "comparison": pair_name,
                "metric": metric,
                "deltas": deltas,
                "d_bar": d_bar,
                "s_d": s_d,
                "df": df,
                "t_stat": t_stat,
                "p_val": p_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohen_dz": cohen_dz,
            }
        )

    results_sorted = sorted(results, key=lambda x: x["p_val"])
    m = len(results_sorted)
    for k, item in enumerate(results_sorted):
        holm_alpha = 0.05 / (m - k)
        adj_p = min(item["p_val"] * (m - k), 1.0)
        item["holm_alpha"] = holm_alpha
        item["adj_p_val"] = adj_p
        item["significant_holm"] = item["p_val"] <= holm_alpha

    print(
        f"{'Scale':<7} | {'Comparison':<26} | {'Metric':<14} | {'d_bar':>7} | {'s_d':>7} | {'t(4)':>7} | {'p-raw':>10} | {'p-Holm':>10} | {'95% CI':<19} | {'Cohen dz':>8} | {'Holm Sig'}"
    )
    print("-" * 135)
    for item in sorted(results_sorted, key=lambda x: (x["scale"], x["metric"], x["comparison"])):
        sig_str = "YES (p<0.05)" if item["significant_holm"] else "NO"
        ci_str = f"[{item['ci_lower']:+6.3f}, {item['ci_upper']:+6.3f}]"
        print(
            f"{item['scale']:<7} | {item['comparison']:<26} | {item['metric']:<14} | {item['d_bar']:+7.3f} | {item['s_d']:7.4f} | {item['t_stat']:7.2f} | {item['p_val']:10.2e} | {item['adj_p_val']:10.2e} | {ci_str:<19} | {item['cohen_dz']:+8.2f} | {sig_str}"
        )

    return data, results_sorted


def generate_tradeoff_plots(data: Dict) -> None:
    """Step 3: Creates publication-quality 4-panel Pareto trade-off figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    scales = [8192, 16384]
    colors = {
        "SentencePiece-Unigram": "#1f77b4",
        "Boundary-BPE": "#2ca02c",
        "Caliper-SuperBPE": "#d62728",
    }
    markers = {
        "SentencePiece-Unigram": "o",
        "Boundary-BPE": "s",
        "Caliper-SuperBPE": "^",
    }

    # Plot 1: True LM BPB
    ax1 = axes[0, 0]
    for tok in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE"]:
        means = [np.mean([data[V][tok][s]["true_lm_bpb"] for s in data[V][tok]]) for V in scales]
        stds = [np.std([data[V][tok][s]["true_lm_bpb"] for s in data[V][tok]]) for V in scales]
        ax1.errorbar(
            scales, means, yerr=stds, label=tok, color=colors[tok], marker=markers[tok], linewidth=2, capsize=5
        )
    ax1.set_title("(A) True LM BPB (Bits / Byte) $\\downarrow$ [Fixed FLOP Budget]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Vocabulary Size ($V$)", fontsize=10)
    ax1.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax1.set_xticks(scales)
    ax1.set_xticklabels(["8,192", "16,384"])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)

    # Plot 2: Token CE Loss
    ax2 = axes[0, 1]
    for tok in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE"]:
        means = [np.mean([data[V][tok][s]["val_loss_nats"] for s in data[V][tok]]) for V in scales]
        stds = [np.std([data[V][tok][s]["val_loss_nats"] for s in data[V][tok]]) for V in scales]
        ax2.errorbar(
            scales, means, yerr=stds, label=tok, color=colors[tok], marker=markers[tok], linewidth=2, capsize=5
        )
    ax2.set_title("(B) Token Cross-Entropy Loss (Nats) $\\downarrow$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Vocabulary Size ($V$)", fontsize=10)
    ax2.set_ylabel("Validation Loss (lower is better)", fontsize=10)
    ax2.set_xticks(scales)
    ax2.set_xticklabels(["8,192", "16,384"])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9)

    # Plot 3: Bytes per Token
    ax3 = axes[1, 0]
    for tok in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE"]:
        means = [np.mean([data[V][tok][s]["bytes_per_token"] for s in data[V][tok]]) for V in scales]
        stds = [np.std([data[V][tok][s]["bytes_per_token"] for s in data[V][tok]]) for V in scales]
        ax3.errorbar(
            scales, means, yerr=stds, label=tok, color=colors[tok], marker=markers[tok], linewidth=2, capsize=5
        )
    ax3.set_title("(C) Subword Compression (Bytes / Token) $\\uparrow$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Vocabulary Size ($V$)", fontsize=10)
    ax3.set_ylabel("Bytes per Token (higher is better)", fontsize=10)
    ax3.set_xticks(scales)
    ax3.set_xticklabels(["8,192", "16,384"])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(frameon=True, fontsize=9)

    # Plot 4: Pareto Trade-off Frontier
    ax4 = axes[1, 1]
    for V in scales:
        for tok in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE"]:
            ce_mean = np.mean([data[V][tok][s]["val_loss_nats"] for s in data[V][tok]])
            bpb_mean = np.mean([data[V][tok][s]["true_lm_bpb"] for s in data[V][tok]])
            ax4.scatter(bpb_mean, ce_mean, color=colors[tok], marker=markers[tok], s=120, edgecolors="black", zorder=5)
            ax4.annotate(f"{tok.split('-')[0]} ({V // 1024}K)", (bpb_mean + 0.02, ce_mean - 0.04), fontsize=8.5)

    ax4.set_title("(D) Pareto Frontier: Token CE vs True LM BPB", fontsize=11, fontweight="bold")
    ax4.set_xlabel("True LM BPB $\\rightarrow$ [Optimal: Bottom-Left]", fontsize=10)
    ax4.set_ylabel("Token Cross-Entropy Loss (Nats) $\\rightarrow$", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plot_path = BENCHMARKS_DIR / "phase_three_tradeoff_plot.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"\n[Step 3] Publication-quality 4-panel trade-off chart saved to {plot_path}")


def profile_tokenizers():
    """Step 4: Deep mechanistic profiling of vocabulary and segmentation token distribution."""
    import sentencepiece as spm

    print("\n" + "=" * 110)
    print("STEP 4: DEEP TOKENIZER MECHANISTIC PROFILING (16,384 VOCABULARY SCALE)")
    print("=" * 110)

    train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=500, seed=101)
    val_text = "\n".join(val_by_lang.values())
    total_val_bytes = len(val_text.encode("utf-8"))
    V = 16384

    # 1. Caliper
    sbp_merges = min(V // 10, 1500)
    base_target = max(V - sbp_merges, 1000)
    actual_merges = V - base_target
    tok_sbp_base = CustomTokenizer.train_from_corpus(
        train_docs,
        target_vocab_size=base_target,
        seed_multiplier=1.2,
        ranking_strategy="byte_savings",
        min_frequency=1,
        verbose=False,
    )
    pretok_chunks = [
        tok for d in train_docs for tok in tok_sbp_base.pre_tokenizer.pre_tokenize(tok_sbp_base.normalizer.normalize(d))
    ]
    cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(tok_sbp_base.model, chunks=pretok_chunks)
    caliper = CustomTokenizer(
        normalizer=tok_sbp_base.normalizer, pre_tokenizer=tok_sbp_base.pre_tokenizer, model=sbp_model
    )

    # 2. SP
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

    # 3. Boundary-BPE
    b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
    bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
    bpe_model = b_bpe.train(bpe_chunks, verbose=False)

    tokenizers = [
        ("SentencePiece-Unigram", list(sp_proc.encode_as_pieces(val_text)), list(range(sp_proc.get_piece_size()))),
        ("Boundary-BPE", bpe_model.encode(val_text), list(bpe_model.vocab)),
        ("Caliper-SuperBPE", caliper.encode(val_text), list(caliper.model.vocab.keys())),
    ]

    words = [w for w in val_text.split() if w]
    num_words = len(words)

    print(f"{'Metric':<42} | {'SentencePiece':<16} | {'Boundary-BPE':<16} | {'Caliper-SuperBPE':<16}")
    print("-" * 98)

    profiles = {}
    for name, tokens, vocab in tokenizers:
        tok_bytes = [len(t.encode("utf-8")) for t in tokens]
        num_toks = len(tokens)
        bytes_per_tok = total_val_bytes / max(num_toks, 1)
        fertility = num_toks / max(num_words, 1)

        single_chars = sum(1 for t in tokens if len(t) == 1) / max(num_toks, 1)
        byte_tokens = sum(1 for t in tokens if t.startswith("<0x") or len(t.encode("utf-8")) == 1) / max(num_toks, 1)

        from collections import Counter

        counts = Counter(tokens)
        probs = np.array(list(counts.values()), dtype=np.float64) / num_toks
        usage_entropy = -np.sum(probs * np.log2(probs + 1e-12))
        vocab_coverage = len(counts) / len(vocab)
        singletons = sum(1 for c in counts.values() if c == 1) / len(counts)

        p10 = np.percentile(tok_bytes, 10)
        p50 = np.percentile(tok_bytes, 50)
        p90 = np.percentile(tok_bytes, 90)
        max_b = max(tok_bytes)

        profiles[name] = {
            "total_tokens": num_toks,
            "bytes_per_token": bytes_per_tok,
            "fertility": fertility,
            "single_char_ratio": single_chars,
            "byte_token_ratio": byte_tokens,
            "usage_entropy": usage_entropy,
            "vocab_coverage": vocab_coverage,
            "singleton_ratio": singletons,
            "p10_bytes": p10,
            "p50_bytes": p50,
            "p90_bytes": p90,
            "max_bytes": max_b,
        }

    metrics_to_print = [
        ("Total Evaluation Tokens (lower = more compressed)", "total_tokens", "{:,}"),
        ("Bytes per Token (compression ratio) [Higher is better]", "bytes_per_token", "{:.2f}"),
        ("Fertility (Tokens per Word) [Lower is better]", "fertility", "{:.2f}"),
        ("Single-Character Token Ratio", "single_char_ratio", "{:.2%}"),
        ("Byte-Fallback / 1-Byte Token Ratio", "byte_token_ratio", "{:.2%}"),
        ("Token Usage Shannon Entropy H(T) [bits]", "usage_entropy", "{:.2f}"),
        ("Active Vocab Coverage on Val", "vocab_coverage", "{:.2%}"),
        ("Singleton Token Proportion in Corpus", "singleton_ratio", "{:.2%}"),
        ("Token Byte Length (p10 / p50 / p90 / max)", "length_str", "{}"),
    ]

    for label, key, fmt in metrics_to_print:
        if key == "length_str":
            sp_val = f"{profiles['SentencePiece-Unigram']['p10_bytes']:.0f}/{profiles['SentencePiece-Unigram']['p50_bytes']:.0f}/{profiles['SentencePiece-Unigram']['p90_bytes']:.0f}/{profiles['SentencePiece-Unigram']['max_bytes']:.0f}"
            bpe_val = f"{profiles['Boundary-BPE']['p10_bytes']:.0f}/{profiles['Boundary-BPE']['p50_bytes']:.0f}/{profiles['Boundary-BPE']['p90_bytes']:.0f}/{profiles['Boundary-BPE']['max_bytes']:.0f}"
            cal_val = f"{profiles['Caliper-SuperBPE']['p10_bytes']:.0f}/{profiles['Caliper-SuperBPE']['p50_bytes']:.0f}/{profiles['Caliper-SuperBPE']['p90_bytes']:.0f}/{profiles['Caliper-SuperBPE']['max_bytes']:.0f}"
        else:
            sp_val = fmt.format(profiles["SentencePiece-Unigram"][key])
            bpe_val = fmt.format(profiles["Boundary-BPE"][key])
            cal_val = fmt.format(profiles["Caliper-SuperBPE"][key])

        print(f"{label:<42} | {sp_val:<16} | {bpe_val:<16} | {cal_val:<16}")

    print("=" * 98 + "\n")
    return profiles


if __name__ == "__main__":
    freeze_baseline()
    data, stats_results = audit_statistics()
    generate_tradeoff_plots(data)
    profile_tokenizers()
