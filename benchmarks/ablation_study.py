"""
Controlled Ablation Study: Isolating Tokenizer Subsystem Contributions.

Ablations:
- Model A: Baseline Unigram (single-char CJK, char_savings ranking, no script balancing)
- Model B: CJK Multi-Character Mining Only (max length clamped to 4 ideographs)
- Model C: Prefix-Aware Script Detection Only
- Model D: Script-Stratified Seeding (equal proportional script allocation)
- Model E: Complete Balanced System (Script-Stratified + Bounded CJK + SuperBPE)
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Tuple

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cem_merger import CrossEntropyMerging
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.real_scale_32k_64k_evaluator import build_multilingual_dataset


@dataclass
class AblationResult:
    name: str
    vocab_size: int
    total_tokens: int
    total_bytes: int
    overall_tid_bpb: float
    bytes_per_tok: float
    script_allocations: Dict[str, int]
    per_lang_bpb: Dict[str, float]


def evaluate_model(name: str, tok: CustomTokenizer, val_by_lang: Dict[str, str], target_vocab: int) -> AblationResult:
    tot_tok = 0
    tot_bytes = 0
    lang_bpb: Dict[str, float] = {}
    log2_v = math.log2(max(target_vocab, 2))

    for lang, text in val_by_lang.items():
        raw_b = len(text.encode("utf-8"))
        token_ids = tok.encode_to_ids(text)
        num_tok = len(token_ids)

        tot_tok += num_tok
        tot_bytes += raw_b
        bpb = (log2_v * num_tok) / max(raw_b, 1)
        lang_bpb[lang] = round(bpb, 3)

    overall_bpb = (log2_v * tot_tok) / max(tot_bytes, 1)
    bpt = tot_bytes / max(tot_tok, 1)

    vocab_tokens = list(tok.model.vocab.keys())
    script_dist = Counter(SeedVocabularyBuilder._detect_script(t) for t in vocab_tokens)

    return AblationResult(
        name=name,
        vocab_size=target_vocab,
        total_tokens=tot_tok,
        total_bytes=tot_bytes,
        overall_tid_bpb=round(overall_bpb, 3),
        bytes_per_tok=round(bpt, 2),
        script_allocations=dict(script_dist),
        per_lang_bpb=lang_bpb,
    )


def run_controlled_ablations(target_vocab: int = 2000, multiplier: int = 40) -> List[AblationResult]:
    train_docs, val_by_lang = build_multilingual_dataset(multiplier=multiplier, seed=100)
    results: List[AblationResult] = []

    print("=" * 115)
    print(f"CONTROLLED TOKENIZER ABLATION SUITE (Target Vocab: {target_vocab:,}, Train Docs: {len(train_docs):,})")
    print("=" * 115)

    # 1. Ablation A: Baseline Unigram (char_savings, no script balancing)
    print("-> Training Ablation A: Baseline (char_savings, no script temp)...")
    tok_a = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="char_savings",
        script_balance_temperature=None,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_model("A: Baseline Unigram (char_savings)", tok_a, val_by_lang, target_vocab))

    # 2. Ablation B: Byte Savings Ranking Only (no script balancing)
    print("-> Training Ablation B: Byte Savings Ranking Only...")
    tok_b = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="byte_savings",
        script_balance_temperature=None,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_model("B: Byte Savings (No Script Temp)", tok_b, val_by_lang, target_vocab))

    # 3. Ablation C: Script Balanced Byte Savings (T=1.0)
    print("-> Training Ablation C: Script Balanced Byte Savings (T=1.0)...")
    tok_c = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="byte_savings",
        script_balance_temperature=1.0,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_model("C: Script Balanced Byte Savings (T=1.0)", tok_c, val_by_lang, target_vocab))

    # 4. Ablation D: Script Balanced with SuperBPE Merges
    print("-> Training Ablation D: Script Balanced SuperBPE (100 Merges)...")
    sbp_merges = 100
    base_target = target_vocab - sbp_merges
    base_d = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=base_target,
        ranking_strategy="byte_savings",
        script_balance_temperature=1.0,
        min_frequency=1,
        verbose=False,
    )
    pretok_chunks = []
    for doc in train_docs:
        norm = base_d.normalizer.normalize(doc)
        pretok_chunks.extend(base_d.pre_tokenizer.pre_tokenize(norm))
    cem = CrossEntropyMerging(max_merges=sbp_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(base_d.model, chunks=pretok_chunks)
    tok_d = CustomTokenizer(
        normalizer=base_d.normalizer,
        pre_tokenizer=base_d.pre_tokenizer,
        model=sbp_model,
    )
    results.append(evaluate_model("D: SuperBPE (100 Merges, T=1.0)", tok_d, val_by_lang, target_vocab))

    return results


def print_ablation_report(results: List[AblationResult]) -> None:
    print("\n" + "=" * 125)
    print("CONTROLLED ABLATION STUDY RESULTS")
    print("=" * 125)

    hdr = f"{'Ablation Variant':<40} | {'Tokens':<8} | {'B/Tok':<6} | {'TID-BPB':<9} | Script Breakdown (Latin / CJK / Indic / Cyr / Arab / Thai)"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sa = r.script_allocations
        script_summary = (
            f"L:{sa.get('latin', 0):<3} | CJK:{sa.get('cjk', 0):<3} | Ind:{sa.get('indic', 0):<3} | "
            f"Cyr:{sa.get('cyrillic', 0):<3} | Ar:{sa.get('arabic', 0):<3} | Th:{sa.get('thai', 0):<3}"
        )
        print(f"{r.name:<40} | {r.total_tokens:<8} | {r.bytes_per_tok:<6.2f} | {r.overall_tid_bpb:<9.3f} | {script_summary}")
    print("=" * 125)

    # Per-Language Comparison
    languages = list(results[0].per_lang_bpb.keys())
    print("\n" + "=" * 125)
    print("PER-LANGUAGE TID-BPB ACROSS ABLATIONS (LOWER IS BETTER)")
    print("=" * 125)

    p_hdr = f"{'Language':<12} | " + " | ".join(f"{r.name[:24]:<24}" for r in results)
    print(p_hdr)
    print("-" * 125)

    for lang in languages:
        cols = [f"{r.per_lang_bpb.get(lang, 0.0):>6.3f} BPB" for r in results]
        print(f"{lang:<12} | " + " | ".join(f"{c:<24}" for c in cols))
    print("=" * 125 + "\n")


if __name__ == "__main__":
    results = run_controlled_ablations(target_vocab=2000, multiplier=40)
    print_ablation_report(results)
