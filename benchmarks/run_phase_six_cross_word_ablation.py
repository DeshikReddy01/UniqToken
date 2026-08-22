"""
Phase Six: Cross-Word Multi-Gram Merging Ablation (Branches A, B, C, D vs Anchors).
Tests whether selective data-driven cross-word collocation merging bridges the BPB gap
(target BPB < 2.60 while CE <= 9.10) at V = 16,384 across 3 paired seeds (101, 202, 303).
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
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from pre_tokenizer import Normalizer, RegexPreTokenizer
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.flop_counter import (
    FLOP_FORMULA_VERSION,
    compute_transformer_flops_per_step,
    plan_training_steps_for_target_flops,
)
from benchmarks.run_phase_three_strict_matched import (
    TARGET_TRAINING_FLOPS,
    generate_rich_multilingual_corpus,
    train_and_eval_strict_transformer,
)


def is_cross_word_token(tok: str, sc: str = "\u2581") -> bool:
    """Predicate identifying multi-word tokens that span lexical boundaries."""
    if tok.count(sc) >= 2:
        return True
    if sc in tok and not tok.startswith(sc):
        return True
    return False


def run_phase_six_ablation(
    target_vocab: int = 16384,
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 500,
) -> Dict[str, Dict[str, float]]:
    import sentencepiece as spm

    print("=" * 165)
    print("PHASE SIX: CROSS-WORD MULTI-GRAM MERGING ABLATION (BRANCHES A, B, C, D VS ANCHORS)")
    print(f"Scale: V = {target_vocab:,} | Seeds: {seeds} | Matched Analytical FLOPs: {TARGET_TRAINING_FLOPS:.3e}")
    print("Goal: Evaluate whether selective PMI-guided collocations achieve BPB < 2.60 while CE <= 9.10")
    print("=" * 165)

    variant_names = [
        "Branch A (Frozen Base)",
        "Branch B (Space-Prefixed)",
        "Branch C (Restricted Cross)",
        "Branch D (Unrestricted Cross)",
        "SentencePiece (Anchor)",
        "Boundary-BPE (Anchor)",
    ]

    all_results: Dict[str, List[Dict[str, float]]] = {v: [] for v in variant_names}

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n---> [Paired Seed {seed_idx}/{len(seeds)}: {seed}]")
        train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=num_docs, seed=seed)
        combined_val = "\n".join(val_by_lang.values())
        total_val_bytes = len(combined_val.encode("utf-8"))
        words = [w for w in combined_val.split() if w]
        num_words = len(words)

        # 1. SentencePiece Anchor
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
            sp_tokens = list(sp_proc.encode_as_pieces(combined_val))
            sp_enc = lambda t: sp_proc.encode(t, out_type=int)
            sp_vocab_size = sp_proc.get_piece_size()

        sp_counts = Counter(sp_tokens)
        sp_active_cov = len(sp_counts) / sp_vocab_size
        sp_tok_bytes = [len(t.encode("utf-8")) for t in sp_tokens]
        sp_pct_ge_6b = sum(1 for b in sp_tok_bytes if b >= 6) / max(len(sp_tok_bytes), 1) * 100.0
        sp_cross_tokens = sum(1 for t in range(sp_vocab_size) if is_cross_word_token(sp_proc.id_to_piece(t)))
        sp_cross_val_bytes = sum(len(t.encode("utf-8")) for t in sp_tokens if is_cross_word_token(t))
        sp_cross_val_pct = sp_cross_val_bytes / max(total_val_bytes, 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=sp_enc,
            vocab_size=sp_vocab_size,
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(sp_tokens), 1)
        fertility = len(sp_tokens) / max(num_words, 1)

        all_results["SentencePiece (Anchor)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "fertility": fertility,
                "active_vocab_pct": sp_active_cov * 100.0,
                "p50_bytes": float(np.percentile(sp_tok_bytes, 50)),
                "p90_bytes": float(np.percentile(sp_tok_bytes, 90)),
                "pct_ge_6b": sp_pct_ge_6b,
                "cross_tokens_vocab": sp_cross_tokens,
                "val_bytes_cross_pct": sp_cross_val_pct,
            }
        )
        print(
            f"  [SentencePiece (Anchor)     ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {sp_active_cov * 100.0:.1f}% | >=6B: {sp_pct_ge_6b:.1f}% | CrossVocab: {sp_cross_tokens} | CrossVal: {sp_cross_val_pct:.1f}%",
            flush=True,
        )

        # 2. Boundary-BPE Anchor
        b_bpe = BPETrainer(target_vocab_size=target_vocab, byte_fallback=True)
        bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
        bpe_model = b_bpe.train(bpe_chunks, verbose=False)
        bpe_tokens = bpe_model.encode(combined_val)
        bpe_counts = Counter(bpe_tokens)
        bpe_active_cov = len(bpe_counts) / len(bpe_model.vocab)
        bpe_tok_bytes = [len(t.encode("utf-8")) for t in bpe_tokens]
        bpe_pct_ge_6b = sum(1 for b in bpe_tok_bytes if b >= 6) / max(len(bpe_tok_bytes), 1) * 100.0
        bpe_cross_tokens = sum(1 for t in bpe_model.vocab if is_cross_word_token(t))
        bpe_cross_val_bytes = sum(len(t.encode("utf-8")) for t in bpe_tokens if is_cross_word_token(t))
        bpe_cross_val_pct = bpe_cross_val_bytes / max(total_val_bytes, 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: bpe_model.encode_to_ids(t),
            vocab_size=len(bpe_model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(bpe_tokens), 1)
        fertility = len(bpe_tokens) / max(num_words, 1)

        all_results["Boundary-BPE (Anchor)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "fertility": fertility,
                "active_vocab_pct": bpe_active_cov * 100.0,
                "p50_bytes": float(np.percentile(bpe_tok_bytes, 50)),
                "p90_bytes": float(np.percentile(bpe_tok_bytes, 90)),
                "pct_ge_6b": bpe_pct_ge_6b,
                "cross_tokens_vocab": bpe_cross_tokens,
                "val_bytes_cross_pct": bpe_cross_val_pct,
            }
        )
        print(
            f"  [Boundary-BPE (Anchor)      ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {bpe_active_cov * 100.0:.1f}% | >=6B: {bpe_pct_ge_6b:.1f}% | CrossVocab: {bpe_cross_tokens} | CrossVal: {bpe_cross_val_pct:.1f}%",
            flush=True,
        )

        # 3. Branch A: Frozen Base Caliper (Unpatched space isolation)
        # We instantiate a pre-tokenizer with isolated word boundaries
        import re

        sc = "\u2581"
        esc = re.escape(sc)
        old_patterns = [
            r"<\|[^\s|]+\|>",
            r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)",
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            r"#\w+",
            r"@\w+",
            r"(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])",
            r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+",
            rf"[^\W\d_\s{esc}]+(?:['’][^\W\d_\s{esc}]+)*",
            r"\d+",
            rf"{esc}+",
            r"\s+",
            rf"[^\w\s{esc}]|_",
        ]
        old_pretok = RegexPreTokenizer()
        old_pretok.regex = re.compile("|".join(f"(?:{p})" for p in old_patterns))

        sbp_merges_a = min(target_vocab // 10, 1500)
        base_target_a = max(target_vocab - sbp_merges_a, 1000)
        actual_merges_a = target_vocab - base_target_a

        norm = Normalizer()
        raw_chunks_a = []
        for d in train_docs:
            raw_chunks_a.extend(old_pretok.pre_tokenize(norm.normalize(d)))

        from unigram_trainer import UnigramTrainer

        trainer_a = UnigramTrainer(
            target_vocab_size=base_target_a,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            pruning_length_exponent=0.0,
            min_frequency=1,
        )
        model_a = trainer_a.train(raw_chunks_a, verbose=False)
        cem_a = CrossEntropyMerging(max_merges=actual_merges_a, cross_word=True, verbose=False)
        sbp_model_a = cem_a.optimize(model_a, chunks=raw_chunks_a)
        tok_a = CustomTokenizer(normalizer=norm, pre_tokenizer=old_pretok, model=sbp_model_a)

        tokens_a = tok_a.encode(combined_val)
        counts_a = Counter(tokens_a)
        active_cov_a = len(counts_a) / len(tok_a.model.vocab)
        tok_bytes_a = [len(t.encode("utf-8")) for t in tokens_a]
        pct_ge_6b_a = sum(1 for b in tok_bytes_a if b >= 6) / max(len(tok_bytes_a), 1) * 100.0
        cross_tokens_a = sum(1 for t in tok_a.model.vocab if is_cross_word_token(t))
        cross_val_bytes_a = sum(len(t.encode("utf-8")) for t in tokens_a if is_cross_word_token(t))
        cross_val_pct_a = cross_val_bytes_a / max(total_val_bytes, 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: tok_a.encode_to_ids(t),
            vocab_size=len(tok_a.model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(tokens_a), 1)
        fertility = len(tokens_a) / max(num_words, 1)

        all_results["Branch A (Frozen Base)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "fertility": fertility,
                "active_vocab_pct": active_cov_a * 100.0,
                "p50_bytes": float(np.percentile(tok_bytes_a, 50)),
                "p90_bytes": float(np.percentile(tok_bytes_a, 90)),
                "pct_ge_6b": pct_ge_6b_a,
                "cross_tokens_vocab": cross_tokens_a,
                "val_bytes_cross_pct": cross_val_pct_a,
            }
        )
        print(
            f"  [Branch A (Frozen Base)     ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {active_cov_a * 100.0:.1f}% | >=6B: {pct_ge_6b_a:.1f}% | CrossVocab: {cross_tokens_a} | CrossVal: {cross_val_pct_a:.1f}%",
            flush=True,
        )

        # 4. Branch B: Space-Prefixed Caliper (Phase 5 Patched)
        sbp_merges_b = min(target_vocab // 10, 1500)
        base_target_b = max(target_vocab - sbp_merges_b, 1000)
        actual_merges_b = target_vocab - base_target_b

        tok_base_b = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=base_target_b,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            pruning_length_exponent=0.0,
            min_frequency=1,
            verbose=False,
        )
        pretok_chunks_b = [
            tok for d in train_docs for tok in tok_base_b.pre_tokenizer.pre_tokenize(tok_base_b.normalizer.normalize(d))
        ]
        cem_b = CrossEntropyMerging(max_merges=actual_merges_b, cross_word=True, verbose=False)
        sbp_model_b = cem_b.optimize(tok_base_b.model, chunks=pretok_chunks_b)
        tok_b = CustomTokenizer(
            normalizer=tok_base_b.normalizer, pre_tokenizer=tok_base_b.pre_tokenizer, model=sbp_model_b
        )

        tokens_b = tok_b.encode(combined_val)
        counts_b = Counter(tokens_b)
        active_cov_b = len(counts_b) / len(tok_b.model.vocab)
        tok_bytes_b = [len(t.encode("utf-8")) for t in tokens_b]
        pct_ge_6b_b = sum(1 for b in tok_bytes_b if b >= 6) / max(len(tok_bytes_b), 1) * 100.0
        cross_tokens_b = sum(1 for t in tok_b.model.vocab if is_cross_word_token(t))
        cross_val_bytes_b = sum(len(t.encode("utf-8")) for t in tokens_b if is_cross_word_token(t))
        cross_val_pct_b = cross_val_bytes_b / max(total_val_bytes, 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: tok_b.encode_to_ids(t),
            vocab_size=len(tok_b.model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(tokens_b), 1)
        fertility = len(tokens_b) / max(num_words, 1)

        all_results["Branch B (Space-Prefixed)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "fertility": fertility,
                "active_vocab_pct": active_cov_b * 100.0,
                "p50_bytes": float(np.percentile(tok_bytes_b, 50)),
                "p90_bytes": float(np.percentile(tok_bytes_b, 90)),
                "pct_ge_6b": pct_ge_6b_b,
                "cross_tokens_vocab": cross_tokens_b,
                "val_bytes_cross_pct": cross_val_pct_b,
            }
        )
        print(
            f"  [Branch B (Space-Prefixed)  ] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {active_cov_b * 100.0:.1f}% | >=6B: {pct_ge_6b_b:.1f}% | CrossVocab: {cross_tokens_b} | CrossVal: {cross_val_pct_b:.1f}%",
            flush=True,
        )

        # 5. Branch C: Restricted Cross-Word Merging (PMI >= 2.0 bits, N = 2000 merges)
        sbp_merges_c = 2000
        base_target_c = max(target_vocab - sbp_merges_c, 1000)
        actual_merges_c = target_vocab - base_target_c

        tok_base_c = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=base_target_c,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            pruning_length_exponent=0.0,
            min_frequency=1,
            verbose=False,
        )
        pretok_chunks_c = [
            tok for d in train_docs for tok in tok_base_c.pre_tokenizer.pre_tokenize(tok_base_c.normalizer.normalize(d))
        ]
        cem_c = CrossEntropyMerging(max_merges=actual_merges_c, cross_word=True, min_pmi=2.0, verbose=False)
        sbp_model_c = cem_c.optimize(tok_base_c.model, chunks=pretok_chunks_c)
        tok_c = CustomTokenizer(
            normalizer=tok_base_c.normalizer, pre_tokenizer=tok_base_c.pre_tokenizer, model=sbp_model_c
        )

        tokens_c = tok_c.encode(combined_val)
        counts_c = Counter(tokens_c)
        active_cov_c = len(counts_c) / len(tok_c.model.vocab)
        tok_bytes_c = [len(t.encode("utf-8")) for t in tokens_c]
        pct_ge_6b_c = sum(1 for b in tok_bytes_c if b >= 6) / max(len(tok_bytes_c), 1) * 100.0
        cross_tokens_c = sum(1 for t in tok_c.model.vocab if is_cross_word_token(t))
        cross_val_bytes_c = sum(len(t.encode("utf-8")) for t in tokens_c if is_cross_word_token(t))
        cross_val_pct_c = cross_val_bytes_c / max(total_val_bytes, 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: tok_c.encode_to_ids(t),
            vocab_size=len(tok_c.model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(tokens_c), 1)
        fertility = len(tokens_c) / max(num_words, 1)

        all_results["Branch C (Restricted Cross)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "fertility": fertility,
                "active_vocab_pct": active_cov_c * 100.0,
                "p50_bytes": float(np.percentile(tok_bytes_c, 50)),
                "p90_bytes": float(np.percentile(tok_bytes_c, 90)),
                "pct_ge_6b": pct_ge_6b_c,
                "cross_tokens_vocab": cross_tokens_c,
                "val_bytes_cross_pct": cross_val_pct_c,
            }
        )
        print(
            f"  [Branch C (Restricted Cross)] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {active_cov_c * 100.0:.1f}% | >=6B: {pct_ge_6b_c:.1f}% | CrossVocab: {cross_tokens_c} | CrossVal: {cross_val_pct_c:.1f}%",
            flush=True,
        )

        # 6. Branch D: Unrestricted Cross-Word Merging (N = 3500 merges, unconstrained)
        sbp_merges_d = 3500
        base_target_d = max(target_vocab - sbp_merges_d, 1000)
        actual_merges_d = target_vocab - base_target_d

        tok_base_d = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=base_target_d,
            seed_multiplier=1.2,
            ranking_strategy="byte_savings",
            min_boundary_entropy=0.5,
            length_exponent=1.0,
            pruning_length_exponent=0.0,
            min_frequency=1,
            verbose=False,
        )
        pretok_chunks_d = [
            tok for d in train_docs for tok in tok_base_d.pre_tokenizer.pre_tokenize(tok_base_d.normalizer.normalize(d))
        ]
        cem_d = CrossEntropyMerging(max_merges=actual_merges_d, cross_word=True, min_pmi=None, verbose=False)
        sbp_model_d = cem_d.optimize(tok_base_d.model, chunks=pretok_chunks_d)
        tok_d = CustomTokenizer(
            normalizer=tok_base_d.normalizer, pre_tokenizer=tok_base_d.pre_tokenizer, model=sbp_model_d
        )

        tokens_d = tok_d.encode(combined_val)
        counts_d = Counter(tokens_d)
        active_cov_d = len(counts_d) / len(tok_d.model.vocab)
        tok_bytes_d = [len(t.encode("utf-8")) for t in tokens_d]
        pct_ge_6b_d = sum(1 for b in tok_bytes_d if b >= 6) / max(len(tok_bytes_d), 1) * 100.0
        cross_tokens_d = sum(1 for t in tok_d.model.vocab if is_cross_word_token(t))
        cross_val_bytes_d = sum(len(t.encode("utf-8")) for t in tokens_d if is_cross_word_token(t))
        cross_val_pct_d = cross_val_bytes_d / max(total_val_bytes, 1) * 100.0

        val_loss, lm_bpb, steps, act_flops, flop_err, params, wall_clock = train_and_eval_strict_transformer(
            enc_fn=lambda t: tok_d.encode_to_ids(t),
            vocab_size=len(tok_d.model.vocab),
            train_texts=train_docs[:300],
            val_text=combined_val,
            total_val_bytes=total_val_bytes,
            target_flops=TARGET_TRAINING_FLOPS,
            seed=seed,
        )
        bpt = total_val_bytes / max(len(tokens_d), 1)
        fertility = len(tokens_d) / max(num_words, 1)

        all_results["Branch D (Unrestricted Cross)"].append(
            {
                "true_lm_bpb": lm_bpb,
                "val_loss": val_loss,
                "bytes_per_token": bpt,
                "fertility": fertility,
                "active_vocab_pct": active_cov_d * 100.0,
                "p50_bytes": float(np.percentile(tok_bytes_d, 50)),
                "p90_bytes": float(np.percentile(tok_bytes_d, 90)),
                "pct_ge_6b": pct_ge_6b_d,
                "cross_tokens_vocab": cross_tokens_d,
                "val_bytes_cross_pct": cross_val_pct_d,
            }
        )
        print(
            f"  [Branch D (Unrestricted Cross)] BPB: {lm_bpb:.3f} | CE: {val_loss:.3f} | B/Tok: {bpt:.2f} | Fert: {fertility:.2f} | Active: {active_cov_d * 100.0:.1f}% | >=6B: {pct_ge_6b_d:.1f}% | CrossVocab: {cross_tokens_d} | CrossVal: {cross_val_pct_d:.1f}%",
            flush=True,
        )

    print("\n" + "=" * 175)
    print("PHASE SIX: CROSS-WORD MULTI-GRAM ABLATION SUMMARY REPORT (MEAN ACROSS 3 SEEDS AT 16K SCALE)")
    print("=" * 175)
    print(
        f"{'Configuration':<30} | {'True LM BPB':<12} | {'Token CE':<10} | {'Bytes/Tok':<10} | {'Fertility':<10} | {'Active Vocab %':<15} | {'Cross-Word Vocab':<17} | {'Cross-Word Val %':<17} | {'Delta vs Base'}"
    )
    print("-" * 175)

    summary: Dict[str, Dict[str, float]] = {}
    base_bpb = float(np.mean([r["true_lm_bpb"] for r in all_results["Branch A (Frozen Base)"]]))

    for v in variant_names:
        bpb_m = float(np.mean([r["true_lm_bpb"] for r in all_results[v]]))
        ce_m = float(np.mean([r["val_loss"] for r in all_results[v]]))
        bpt_m = float(np.mean([r["bytes_per_token"] for r in all_results[v]]))
        fert_m = float(np.mean([r["fertility"] for r in all_results[v]]))
        act_m = float(np.mean([r["active_vocab_pct"] for r in all_results[v]]))
        cw_voc_m = float(np.mean([r["cross_tokens_vocab"] for r in all_results[v]]))
        cw_val_m = float(np.mean([r["val_bytes_cross_pct"] for r in all_results[v]]))
        delta = bpb_m - base_bpb

        summary[v] = {
            "true_lm_bpb": bpb_m,
            "val_loss": ce_m,
            "bytes_per_token": bpt_m,
            "fertility": fert_m,
            "active_vocab_pct": act_m,
            "cross_tokens_vocab": cw_voc_m,
            "val_bytes_cross_pct": cw_val_m,
            "delta_bpb": delta,
        }

        delta_str = f"{delta:+.3f} BPB" if v != "Branch A (Frozen Base)" else "0.000 (Base)"
        print(
            f"{v:<30} | {bpb_m:<12.3f} | {ce_m:<10.3f} | {bpt_m:<10.2f} | {fert_m:<10.2f} | {act_m:<15.1f}% | {cw_voc_m:<17.0f} | {cw_val_m:<17.1f}% | {delta_str}"
        )
    print("=" * 175 + "\n")

    # Generate Pareto Trajectory Plot
    plt.figure(figsize=(11, 8), dpi=300)
    sp_bpb = summary["SentencePiece (Anchor)"]["true_lm_bpb"]
    sp_ce = summary["SentencePiece (Anchor)"]["val_loss"]
    bpe_bpb = summary["Boundary-BPE (Anchor)"]["true_lm_bpb"]
    bpe_ce = summary["Boundary-BPE (Anchor)"]["val_loss"]

    plt.scatter(
        [sp_bpb],
        [sp_ce],
        color="#1f77b4",
        marker="o",
        s=180,
        edgecolors="black",
        label="SentencePiece (Anchor)",
        zorder=6,
    )
    plt.annotate(
        f"SentencePiece\nBPB: {sp_bpb:.3f}, CE: {sp_ce:.2f}", (sp_bpb + 0.02, sp_ce - 0.03), fontsize=9, color="#1f77b4"
    )

    plt.scatter(
        [bpe_bpb],
        [bpe_ce],
        color="#2ca02c",
        marker="s",
        s=180,
        edgecolors="black",
        label="Boundary-BPE (Anchor)",
        zorder=6,
    )
    plt.annotate(
        f"Boundary-BPE\nBPB: {bpe_bpb:.3f}, CE: {bpe_ce:.2f}",
        (bpe_bpb + 0.02, bpe_ce - 0.03),
        fontsize=9,
        color="#2ca02c",
    )

    colors_dict = {
        "Branch A (Frozen Base)": "gray",
        "Branch B (Space-Prefixed)": "#9467bd",
        "Branch C (Restricted Cross)": "#ff7f0e",
        "Branch D (Unrestricted Cross)": "#d62728",
    }
    markers_dict = {
        "Branch A (Frozen Base)": "x",
        "Branch B (Space-Prefixed)": "D",
        "Branch C (Restricted Cross)": "P",
        "Branch D (Unrestricted Cross)": "*",
    }

    pts = []
    for b in [
        "Branch A (Frozen Base)",
        "Branch B (Space-Prefixed)",
        "Branch C (Restricted Cross)",
        "Branch D (Unrestricted Cross)",
    ]:
        bpb = summary[b]["true_lm_bpb"]
        ce = summary[b]["val_loss"]
        pts.append((bpb, ce))
        sz = 260 if "D" in b else (200 if "C" in b else 150)
        plt.scatter(
            [bpb],
            [ce],
            color=colors_dict[b],
            marker=markers_dict[b],
            s=sz,
            edgecolors="black" if "D" in b or "C" in b else None,
            label=b,
            zorder=7,
        )
        plt.annotate(
            f"{b.split('(')[1].replace(')', '')}\nBPB: {bpb:.3f}, CE: {ce:.2f}",
            (bpb + 0.02, ce + 0.02),
            fontsize=8.5,
            color=colors_dict[b],
        )

    # Trajectory curve
    plt.plot(
        [p[0] for p in pts], [p[1] for p in pts], color="#ff7f0e", linestyle="--", linewidth=2, alpha=0.7, zorder=5
    )

    plt.title("Phase Six: Cross-Word Multi-Gram Merging Pareto Frontier (V = 16,384)", fontsize=12, fontweight="bold")
    plt.xlabel("True LM BPB (lower is better) -> [Optimal: Left]", fontsize=11)
    plt.ylabel("Token Cross-Entropy Loss (lower is better) -> [Optimal: Bottom]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")

    plot_path = Path(__file__).resolve().parent / "phase_six_cross_word_pareto.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 6 Pareto chart to {plot_path}")

    # Save summary JSON
    json_path = Path(__file__).resolve().parent / "phase_six_cross_word_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] Saved Phase 6 summary metrics to {json_path}")

    return summary


if __name__ == "__main__":
    run_phase_six_ablation(target_vocab=16384, seeds=[101, 202, 303], num_docs=500)
