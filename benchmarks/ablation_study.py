"""
Comprehensive 5-Stage Ablation Study: Resolving Multilingual Allocation & CJK Bounding.

Ablations:
- Model A: Baseline char_savings (unconstrained n-grams, unstratified)
- Model B: Raw byte_savings (unconstrained n-grams)
- Model C: byte_savings + CJK length bounding (max CJK <= 4)
- Model D: Model C + Script-stratified quota allocation
- Model E: Model D + SuperBPE (Cross-Entropy Merging)
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cem_merger import CrossEntropyMerging
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.real_scale_32k_64k_evaluator import build_multilingual_dataset


@dataclass
class StageAblationResult:
    stage_id: str
    name: str
    vocab_size: int
    total_tokens: int
    total_bytes: int
    overall_tid_bpb: float
    bytes_per_tok: float
    tokens_per_byte: float
    fallback_rate_pct: float
    script_allocations: Dict[str, int]
    per_lang_bpb: Dict[str, float]


def evaluate_stage_model(
    stage_id: str,
    name: str,
    tok: CustomTokenizer,
    val_by_lang: Dict[str, str],
    target_vocab: int,
) -> StageAblationResult:
    tot_tok = 0
    tot_bytes = 0
    tot_fallback = 0
    lang_bpb: Dict[str, float] = {}
    log2_v = math.log2(max(target_vocab, 2))

    for lang, text in val_by_lang.items():
        raw_b = len(text.encode("utf-8"))
        tokens = tok.encode(text)
        token_ids = tok.encode_to_ids(text)

        num_tok = len(token_ids)
        tot_tok += num_tok
        tot_bytes += raw_b

        # Count byte fallback tokens (<0x..>)
        tot_fallback += sum(1 for t in tokens if t.startswith("<0x") and t.endswith(">"))

        bpb = (log2_v * num_tok) / max(raw_b, 1)
        lang_bpb[lang] = round(bpb, 3)

    overall_bpb = (log2_v * tot_tok) / max(tot_bytes, 1)
    bpt = tot_bytes / max(tot_tok, 1)
    tpb = tot_tok / max(tot_bytes, 1)
    fallback_pct = (tot_fallback / max(tot_tok, 1)) * 100.0

    vocab_tokens = list(tok.model.vocab.keys())
    script_dist = Counter(SeedVocabularyBuilder._detect_script(t) for t in vocab_tokens)

    return StageAblationResult(
        stage_id=stage_id,
        name=name,
        vocab_size=target_vocab,
        total_tokens=tot_tok,
        total_bytes=tot_bytes,
        overall_tid_bpb=round(overall_bpb, 3),
        bytes_per_tok=round(bpt, 2),
        tokens_per_byte=round(tpb, 4),
        fallback_rate_pct=round(fallback_pct, 2),
        script_allocations=dict(script_dist),
        per_lang_bpb=lang_bpb,
    )


def run_5_stage_ablation(target_vocab: int = 2000, multiplier: int = 40) -> List[StageAblationResult]:
    train_docs, val_by_lang = build_multilingual_dataset(multiplier=multiplier, seed=100)
    results: List[StageAblationResult] = []

    print("=" * 115)
    print(f"5-STAGE CONTROLLED MULTILINGUAL ABLATION STUDY (Target Vocab: {target_vocab:,})")
    print("=" * 115)

    # Stage A: Baseline char_savings
    print("-> [Stage A] Training Baseline char_savings...")
    tok_a = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="char_savings",
        script_balance_temperature=None,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_stage_model("A", "Baseline char_savings", tok_a, val_by_lang, target_vocab))

    # Stage B: Raw byte_savings (marginal subword gain)
    print("-> [Stage B] Training Raw byte_savings...")
    tok_b = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="byte_savings",
        script_balance_temperature=None,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_stage_model("B", "Raw byte_savings", tok_b, val_by_lang, target_vocab))

    # Stage C: byte_savings + CJK bounded length (active via _get_max_ngram_for_chunk)
    print("-> [Stage C] Training byte_savings + CJK length <= 4...")
    tok_c = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="byte_savings",
        script_balance_temperature=None,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_stage_model("C", "byte_savings + CJK<=4", tok_c, val_by_lang, target_vocab))

    # Stage D: Model C + Script-stratified quotas (active with script_balance_temperature=1.0)
    print("-> [Stage D] Training Model C + Script-Stratified Quotas...")
    tok_d = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="byte_savings",
        script_balance_temperature=1.0,
        min_frequency=1,
        verbose=False,
    )
    results.append(evaluate_stage_model("D", "Model C + Stratified Quotas", tok_d, val_by_lang, target_vocab))

    # Stage E: Model D + SuperBPE (Cross-Entropy Merging)
    print("-> [Stage E] Training Model D + SuperBPE Merges...")
    sbp_merges = 100
    base_target = target_vocab - sbp_merges
    base_e = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=base_target,
        ranking_strategy="byte_savings",
        script_balance_temperature=1.0,
        min_frequency=1,
        verbose=False,
    )
    pretok_chunks = []
    for doc in train_docs:
        norm = base_e.normalizer.normalize(doc)
        pretok_chunks.extend(base_e.pre_tokenizer.pre_tokenize(norm))
    cem = CrossEntropyMerging(max_merges=sbp_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(base_e.model, chunks=pretok_chunks)
    tok_e = CustomTokenizer(
        normalizer=base_e.normalizer,
        pre_tokenizer=base_e.pre_tokenizer,
        model=sbp_model,
    )
    results.append(evaluate_stage_model("E", "Model D + SuperBPE", tok_e, val_by_lang, target_vocab))

    return results


def print_stage_ablation_report(results: List[StageAblationResult]) -> None:
    print("\n" + "=" * 135)
    print("5-STAGE CONTROLLED ABLATION REPORT")
    print("=" * 135)

    hdr = f"{'Stage':<6} | {'Ablation Description':<28} | {'Tokens':<8} | {'TID-BPB':<9} | {'B/Tok':<6} | {'Tok/Byte':<9} | {'Fallback %':<11} | Script Breakdown (L / CJK / Ind / Cyr / Ar / Th)"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sa = r.script_allocations
        script_summary = (
            f"L:{sa.get('latin', 0):<3} | CJK:{sa.get('cjk', 0):<3} | Ind:{sa.get('indic', 0):<3} | "
            f"Cyr:{sa.get('cyrillic', 0):<3} | Ar:{sa.get('arabic', 0):<3} | Th:{sa.get('thai', 0):<3}"
        )
        print(
            f"{r.stage_id:<6} | {r.name:<28} | {r.total_tokens:<8} | {r.overall_tid_bpb:<9.3f} | "
            f"{r.bytes_per_tok:<6.2f} | {r.tokens_per_byte:<9.4f} | {r.fallback_rate_pct:<11.2f} | {script_summary}"
        )
    print("=" * 135)

    # Per-Language Comparison
    languages = list(results[0].per_lang_bpb.keys())
    print("\n" + "=" * 135)
    print("PER-LANGUAGE TID-BPB ACROSS ALL 5 ABLATION STAGES (LOWER IS BETTER)")
    print("=" * 135)

    p_hdr = f"{'Language':<12} | " + " | ".join(f"[{r.stage_id}] {r.name[:16]:<16}" for r in results)
    print(p_hdr)
    print("-" * 135)

    for lang in languages:
        cols = [f"{r.per_lang_bpb.get(lang, 0.0):>6.3f} BPB" for r in results]
        print(f"{lang:<12} | " + " | ".join(f"{c:<21}" for c in cols))
    print("=" * 135 + "\n")


if __name__ == "__main__":
    results = run_5_stage_ablation(target_vocab=2000, multiplier=40)
    print_stage_ablation_report(results)
