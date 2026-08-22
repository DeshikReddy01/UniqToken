"""
Step 5: Targeted Caliper Ablation Study (16,384 Scale across 3 Paired Seeds).
Identifies which architectural/algorithmic component of Caliper affects the True LM BPB vs Token CE trade-off.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from tokenizer import CustomTokenizer
from benchmarks.run_phase_three_strict_matched import (
    TARGET_TRAINING_FLOPS,
    generate_rich_multilingual_corpus,
    train_and_eval_strict_transformer,
)


def run_ablation_study(
    target_vocab: int = 16384,
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 500,
) -> None:
    import sentencepiece as spm

    print("=" * 125)
    print(f"STEP 5: TARGETED CALIPER ABLATION STUDY (V = {target_vocab:,} | SEEDS = {seeds})")
    print("Goal: Discover which component causes the BPB / compression penalty")
    print("=" * 125)

    variants = [
        "1. SentencePiece (Ref Anchor)",
        "2. Caliper Baseline",
        "3. Caliper (No Entropy Split)",
        "4. Caliper (No CEM / Pure Unigram)",
        "5. Caliper (Aggressive CEM: 3,500 merges)",
    ]

    results: Dict[str, Dict[int, Dict[str, float]]] = {v: {} for v in variants}

    for seed in seeds:
        print(f"\n---> Running Ablation Seed: {seed}...")
        train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=num_docs, seed=seed)
        combined_val = "\n".join(val_by_lang.values())
        total_val_bytes = len(combined_val.encode("utf-8"))

        # 1. SentencePiece Baseline
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
            sp_vocab_size = sp_proc.get_piece_size()

        # 2. Caliper Baseline (Entropy splitting + byte_savings + 1,500 CEM merges)
        sbp_merges = min(target_vocab // 10, 1500)
        base_target = max(target_vocab - sbp_merges, 1000)
        actual_merges = target_vocab - base_target

        tok_base = CustomTokenizer.train_from_corpus(
            train_docs,
            target_vocab_size=base_target,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            min_frequency=1,
            verbose=False,
        )
        pretok_chunks = [
            c for d in train_docs for c in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))
        ]
        cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
        sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
        cal_base = CustomTokenizer(
            normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model
        )

        # 3. Caliper (No Entropy Split)
        tok_no_ent = CustomTokenizer.train_from_corpus(
            train_docs,
            target_vocab_size=base_target,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=None,
            min_frequency=1,
            verbose=False,
        )
        pretok_no_ent = [
            c for d in train_docs for c in tok_no_ent.pre_tokenizer.pre_tokenize(tok_no_ent.normalizer.normalize(d))
        ]
        cem_no_ent = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
        sbp_no_ent = cem_no_ent.optimize(tok_no_ent.model, chunks=pretok_no_ent)
        cal_no_ent = CustomTokenizer(
            normalizer=tok_no_ent.normalizer, pre_tokenizer=tok_no_ent.pre_tokenizer, model=sbp_no_ent
        )

        # 4. Caliper (No CEM / Pure Unigram EM)
        cal_no_cem = CustomTokenizer.train_from_corpus(
            train_docs,
            target_vocab_size=target_vocab,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            min_frequency=1,
            verbose=False,
        )

        # 5. Caliper (Aggressive CEM: 3,500 merges)
        agg_merges = 3500
        agg_base_target = max(target_vocab - agg_merges, 1000)
        tok_agg = CustomTokenizer.train_from_corpus(
            train_docs,
            target_vocab_size=agg_base_target,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            min_frequency=1,
            verbose=False,
        )
        pretok_agg = [
            c for d in train_docs for c in tok_agg.pre_tokenizer.pre_tokenize(tok_agg.normalizer.normalize(d))
        ]
        cem_agg = CrossEntropyMerging(max_merges=agg_merges, cross_word=True, verbose=False)
        sbp_agg = cem_agg.optimize(tok_agg.model, chunks=pretok_agg)
        cal_agg = CustomTokenizer(normalizer=tok_agg.normalizer, pre_tokenizer=tok_agg.pre_tokenizer, model=sbp_agg)

        ablation_instances = [
            ("1. SentencePiece (Ref Anchor)", sp_vocab_size, sp_enc),
            ("2. Caliper Baseline", len(cal_base.model.vocab), lambda t: cal_base.encode_to_ids(t)),
            ("3. Caliper (No Entropy Split)", len(cal_no_ent.model.vocab), lambda t: cal_no_ent.encode_to_ids(t)),
            ("4. Caliper (No CEM / Pure Unigram)", len(cal_no_cem.model.vocab), lambda t: cal_no_cem.encode_to_ids(t)),
            ("5. Caliper (Aggressive CEM: 3,500 merges)", len(cal_agg.model.vocab), lambda t: cal_agg.encode_to_ids(t)),
        ]

        for name, v_act, enc_fn in ablation_instances:
            tot_tok = sum(len(enc_fn(text)) for text in val_by_lang.values())
            bpt = total_val_bytes / max(tot_tok, 1)

            val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
                enc_fn=enc_fn,
                vocab_size=v_act,
                train_texts=train_docs[:300],
                val_text=combined_val,
                total_val_bytes=total_val_bytes,
                target_flops=TARGET_TRAINING_FLOPS,
                seed=seed,
            )

            results[name][seed] = {
                "vocab_size": v_act,
                "val_loss": val_loss,
                "true_lm_bpb": lm_bpb,
                "bytes_per_token": bpt,
            }

            print(
                f"  [{name:<40}] V: {v_act:<6,} | CE Loss: {val_loss:.3f} | True LM BPB: {lm_bpb:.3f} | B/Tok: {bpt:.2f}"
            )

    # Summary Table
    print("\n" + "=" * 125)
    print("STEP 5: ABLATION SUMMARY REPORT (MEAN ACROSS 3 SEEDS AT 16K SCALE)")
    print("=" * 125)
    print(
        f"{'Variant':<42} | {'True LM BPB':<12} | {'Token CE Loss':<14} | {'Bytes / Token':<14} | {'Delta BPB vs Baseline'}"
    )
    print("-" * 125)

    base_bpb = np.mean([results["2. Caliper Baseline"][s]["true_lm_bpb"] for s in seeds])

    for v in variants:
        bpb_mean = np.mean([results[v][s]["true_lm_bpb"] for s in seeds])
        ce_mean = np.mean([results[v][s]["val_loss"] for s in seeds])
        bpt_mean = np.mean([results[v][s]["bytes_per_token"] for s in seeds])
        delta = bpb_mean - base_bpb

        delta_str = f"{delta:+.3f} BPB" if v != "2. Caliper Baseline" else "0.000 (Baseline)"
        print(f"{v:<42} | {bpb_mean:<12.3f} | {ce_mean:<14.3f} | {bpt_mean:<14.2f} | {delta_str}")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    run_ablation_study(target_vocab=16384, seeds=[101, 202, 303], num_docs=500)
