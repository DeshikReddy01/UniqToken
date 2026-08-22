"""
Controlled Matched-Budget Vocabulary Experiment & Granular Subsystem Breakdown.

Features:
1. Exactly Matched 1:1 Vocabulary Budgets across Caliper SuperBPE, Caliper Unigram, SentencePiece, BPE.
2. Specifically benchmarks V=2,717 (SentencePiece exact match) and V=1,000, 2,000, 4,000.
3. Deconstructs the 'Symbol' bucket into: Control (7), Byte Fallback (256), Numerics, Punctuation/Whitespace, and Text.
4. Evaluates Downstream Transformer Cross-Entropy, True LM BPB, and Hardware Throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
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

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.multilingual_scaling_sweep import (
    EXPANDED_LEXICAL_CORPUS,
    build_rich_multilingual_corpus,
    train_and_eval_transformer,
)


@dataclass
class GranularVocabDecomposition:
    total_vocab: int
    control_special: int
    byte_tokens: int
    numerics: int
    punctuation_whitespace: int
    language_subwords: Dict[str, int]


def deconstruct_vocabulary(vocab_keys: List[str]) -> GranularVocabDecomposition:
    control_count = 0
    byte_count = 0
    numeric_count = 0
    punct_ws_count = 0
    lang_subwords: Dict[str, int] = Counter()

    for tok in vocab_keys:
        if tok.startswith("<|") and tok.endswith("|>"):
            control_count += 1
        elif tok.startswith("<0x") and tok.endswith(">"):
            byte_count += 1
        else:
            # Strip metaspace prefix
            clean = tok.replace("\u2581", "").strip()
            if not clean:
                punct_ws_count += 1
            elif clean.isdigit():
                numeric_count += 1
            elif all(not ch.isalnum() for ch in clean):
                punct_ws_count += 1
            else:
                script = SeedVocabularyBuilder._detect_script(tok)
                lang_subwords[script] += 1

    return GranularVocabDecomposition(
        total_vocab=len(vocab_keys),
        control_special=control_count,
        byte_tokens=byte_count,
        numerics=numeric_count,
        punctuation_whitespace=punct_ws_count,
        language_subwords=dict(lang_subwords),
    )


@dataclass
class MatchedExperimentEntry:
    vocab_target: int
    engine_name: str
    actual_vocab: int
    vocab_hash: str
    tokens: int
    bytes_per_tok: float
    tid_bpb: float
    lm_loss: float
    true_lm_bpb: float
    decomp: GranularVocabDecomposition


def run_matched_experiment(
    scales: List[int] = [1000, 2000, 2717, 4000],
    num_samples_per_lang: int = 400,
    seed: int = 100,
) -> List[MatchedExperimentEntry]:
    train_docs, val_by_lang = build_rich_multilingual_corpus(num_samples_per_lang=num_samples_per_lang, seed=seed)
    combined_val_text = "\n".join(val_by_lang.values())
    total_val_bytes = len(combined_val_text.encode("utf-8"))

    results: List[MatchedExperimentEntry] = []

    print("=" * 125)
    print(f"CONTROLLED EXACTLY-MATCHED MULTILINGUAL VOCABULARY EXPERIMENT (Budgets: {scales})")
    print(f"Evaluation Buffer: {total_val_bytes:,} UTF-8 bytes across 12 languages")
    print("=" * 125)

    for V in scales:
        print(f"\n---> [Evaluating Matched Budget: {V:,} Tokens] Training all 4 engines...")

        # 1. Caliper (Unigram)
        tok_uni = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=V,
            ranking_strategy="byte_savings",
            script_balance_temperature=1.0,
            min_frequency=1,
            verbose=False,
        )

        # 2. Caliper (SuperBPE)
        sbp_merges = min(V // 10, 250)
        base_target = max(V - sbp_merges, 960) if V >= 1000 else V
        actual_merges = V - base_target

        if actual_merges > 0:
            tok_sbp_base = CustomTokenizer.train_from_corpus(
                corpus=train_docs,
                target_vocab_size=base_target,
                ranking_strategy="byte_savings",
                script_balance_temperature=1.0,
                min_frequency=1,
                verbose=False,
            )
            pretok_chunks = []
            for doc in train_docs:
                norm = tok_sbp_base.normalizer.normalize(doc)
                pretok_chunks.extend(tok_sbp_base.pre_tokenizer.pre_tokenize(norm))
            cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
            sbp_model = cem.optimize(tok_sbp_base.model, chunks=pretok_chunks)
            tok_sbp = CustomTokenizer(
                normalizer=tok_sbp_base.normalizer,
                pre_tokenizer=tok_sbp_base.pre_tokenizer,
                model=sbp_model,
            )
        else:
            tok_sbp = tok_uni

        # 3. SentencePiece (Unigram)
        sp_proc = None
        try:
            import sentencepiece as spm

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
        except Exception:
            pass

        # 4. Standard BPE
        bpe_tr = BPETrainer(target_vocab_size=V, byte_fallback=True)
        bpe_m = bpe_tr.train(train_docs, verbose=False)

        engines = [
            (
                "Caliper (SuperBPE)",
                tok_sbp.vocab_size,
                lambda t: tok_sbp.encode_to_ids(t),
                list(tok_sbp.model.vocab.keys()),
            ),
            (
                "Caliper (Unigram)",
                tok_uni.vocab_size,
                lambda t: tok_uni.encode_to_ids(t),
                list(tok_uni.model.vocab.keys()),
            ),
            (
                "Standard BPE",
                len(bpe_m.vocab),
                lambda t: [bpe_m.token_to_id.get(x, 0) for x in bpe_m.encode(t)],
                list(bpe_m.vocab),
            ),
        ]
        if sp_proc is not None:
            sp_vocab = [sp_proc.id_to_piece(i) for i in range(sp_proc.get_piece_size())]
            engines.append(
                (
                    "SentencePiece (Unigram)",
                    sp_proc.get_piece_size(),
                    lambda t: sp_proc.encode(t, out_type=int),
                    sp_vocab,
                )
            )

        for name, v_actual, enc_fn, vocab_list in engines:
            v_hash = hashlib.md5("".join(sorted(vocab_list)).encode("utf-8")).hexdigest()[:8]
            tot_tok = sum(len(enc_fn(text)) for text in val_by_lang.values())

            tid_bpb = (math.log2(max(v_actual, 2)) * tot_tok) / max(total_val_bytes, 1)
            bpt = total_val_bytes / max(tot_tok, 1)

            lm_loss, true_lm_bpb = train_and_eval_transformer(
                enc_fn=enc_fn,
                vocab_size=v_actual,
                train_texts=train_docs[:200],
                val_text=combined_val_text,
                total_val_bytes=total_val_bytes,
            )

            decomp = deconstruct_vocabulary(vocab_list)

            results.append(
                MatchedExperimentEntry(
                    vocab_target=V,
                    engine_name=name,
                    actual_vocab=v_actual,
                    vocab_hash=v_hash,
                    tokens=tot_tok,
                    bytes_per_tok=round(bpt, 2),
                    tid_bpb=round(tid_bpb, 3),
                    lm_loss=round(lm_loss, 3),
                    true_lm_bpb=round(true_lm_bpb, 3),
                    decomp=decomp,
                )
            )

    return results


def print_matched_experiment_report(results: List[MatchedExperimentEntry]) -> None:
    print("\n" + "=" * 135)
    print("CONTROLLED EXACTLY-MATCHED EXPERIMENT REPORT (TID-BPB, LM LOSS, AND TRUE LM BPB)")
    print("=" * 135)

    hdr = f"{'Vocab Target':<12} | {'Engine':<24} | {'Actual V':<8} | {'Hash':<8} | {'Tokens':<8} | {'B/Tok':<6} | {'TID-BPB':<9} | {'LM Loss':<9} | {'True LM BPB':<12}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        print(
            f"{r.vocab_target:<12,} | {r.engine_name:<24} | {r.actual_vocab:<8,} | {r.vocab_hash:<8} | "
            f"{r.tokens:<8,} | {r.bytes_per_tok:<6.2f} | {r.tid_bpb:<9.3f} | {r.lm_loss:<9.3f} | {r.true_lm_bpb:<12.3f}"
        )
    print("=" * 135)

    # Granular Vocabulary Decomposition
    print("\n" + "=" * 135)
    print("GRANULAR VOCABULARY DECOMPOSITION (DISSECTING THE 'SYMBOL' BUCKET)")
    print("=" * 135)
    caliper_entries = [r for r in results if r.engine_name == "Caliper (SuperBPE)"]
    for c in caliper_entries:
        d = c.decomp
        lang_sub_tot = sum(d.language_subwords.values())
        print(
            f"Scale {c.actual_vocab:>5,}: Special Control={d.control_special:>2}, Byte Fallback={d.byte_tokens:>3}, "
            f"Numerics={d.numerics:>3}, Punct/Whitespace={d.punctuation_whitespace:>3}, Language Subwords={lang_sub_tot:>4} "
            f"--> [Sum: {d.control_special + d.byte_tokens + d.numerics + d.punctuation_whitespace + lang_sub_tot:,} / {c.actual_vocab:,}]"
        )
    print("=" * 135 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched Budget Experiment")
    parser.add_argument("--scales", nargs="+", type=int, default=[1000, 2000, 2717, 4000], help="Vocabulary budgets")
    parser.add_argument("--samples", type=int, default=400, help="Samples per language")
    args = parser.parse_args()

    results = run_matched_experiment(scales=args.scales, num_samples_per_lang=args.samples)
    print_matched_experiment_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
