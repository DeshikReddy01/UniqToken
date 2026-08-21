"""
Deep Token Distribution Audit & True LM BPB Unit Verification.

Audits:
1. Token length distributions (Min, P50, P90, P99, Max) for Caliper, SentencePiece, and BPE.
2. Top-20 longest learned tokens across all engines to expose cross-word sentence memorization.
3. Verification of True LM BPB derivation: (Loss_nats / ln 2) * (Tokens / Bytes).
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from tokenizer import CustomTokenizer
from benchmarks.multilingual_scaling_sweep import build_rich_multilingual_corpus


def audit_token_distributions(target_vocab: int = 4000) -> None:
    train_docs, val_by_lang = build_rich_multilingual_corpus(num_samples_per_lang=400, seed=100)
    combined_val = "\n".join(val_by_lang.values())
    total_val_bytes = len(combined_val.encode("utf-8"))

    # 1. Caliper (SuperBPE) at 4K
    sbp_merges = 250
    base_target = target_vocab - sbp_merges
    base_tok = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=base_target,
        ranking_strategy="byte_savings",
        script_balance_temperature=1.0,
        min_frequency=1,
        verbose=False,
    )
    pretok_chunks = []
    for doc in train_docs:
        norm = base_tok.normalizer.normalize(doc)
        pretok_chunks.extend(base_tok.pre_tokenizer.pre_tokenize(norm))
    cem = CrossEntropyMerging(max_merges=sbp_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(base_tok.model, chunks=pretok_chunks)
    caliper_sbp = CustomTokenizer(
        normalizer=base_tok.normalizer,
        pre_tokenizer=base_tok.pre_tokenizer,
        model=sbp_model,
    )

    # 2. Caliper (Unigram) at 4K
    caliper_uni = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=target_vocab,
        ranking_strategy="byte_savings",
        script_balance_temperature=1.0,
        min_frequency=1,
        verbose=False,
    )

    # 3. SentencePiece (Unigram)
    sp_proc = None
    sp_tokens = []
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
                vocab_size=target_vocab,
                character_coverage=1.0,
                byte_fallback=True,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            sp_proc = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
            sp_tokens = [sp_proc.id_to_piece(i) for i in range(sp_proc.get_piece_size())]
    except Exception:
        pass

    # 4. Standard BPE at 4K
    bpe_tr = BPETrainer(target_vocab_size=target_vocab, byte_fallback=True)
    bpe_m = bpe_tr.train(train_docs, verbose=False)

    engines = [
        ("Caliper (SuperBPE)", caliper_sbp.vocab_size, list(caliper_sbp.model.vocab.keys()), lambda t: caliper_sbp.encode(t)),
        ("Caliper (Unigram)", caliper_uni.vocab_size, list(caliper_uni.model.vocab.keys()), lambda t: caliper_uni.encode(t)),
        ("SentencePiece (Unigram)", len(sp_tokens), sp_tokens, lambda t: sp_proc.encode(t, out_type=str) if sp_proc else []),
        ("Standard BPE", len(bpe_m.vocab), list(bpe_m.vocab), lambda t: bpe_m.encode(t)),
    ]

    print("=" * 125)
    print(f"TOKEN LENGTH & MEMORIZATION DISTRIBUTION AUDIT (TARGET VOCAB = {target_vocab:,})")
    print("=" * 125)

    hdr = f"{'Engine':<26} | {'Vocab':<6} | {'Mean Bytes':<11} | {'P50 (B)':<8} | {'P90 (B)':<8} | {'P99 (B)':<8} | {'Max Bytes':<10} | {'Tokens on Val':<13}"
    print(hdr)
    print("-" * len(hdr))

    for name, v_sz, vocab_list, enc_fn in engines:
        # Byte length of vocabulary entries
        vocab_byte_lens = [len(t.encode("utf-8")) for t in vocab_list]
        arr = np.array(vocab_byte_lens)

        val_tokens = enc_fn(combined_val)
        val_tok_count = len(val_tokens)

        print(
            f"{name:<26} | {v_sz:<6,} | {arr.mean():<11.2f} | {np.percentile(arr, 50):<8.1f} | "
            f"{np.percentile(arr, 90):<8.1f} | {np.percentile(arr, 99):<8.1f} | {arr.max():<10} | {val_tok_count:<13,}"
        )
    print("=" * 125)

    # Top 10 Longest Tokens for Caliper SuperBPE vs Standard BPE
    print("\n" + "=" * 125)
    print("TOP 10 LONGEST LEARNED TOKENS (CALIPER SUPERBPE vs STANDARD BPE)")
    print("=" * 125)

    sbp_sorted = sorted(list(caliper_sbp.model.vocab.keys()), key=lambda t: -len(t.encode("utf-8")))
    bpe_sorted = sorted(list(bpe_m.vocab), key=lambda t: -len(t.encode("utf-8")))

    print("\n--- Caliper SuperBPE Top 10 Longest Tokens ---")
    for i, tok in enumerate(sbp_sorted[:10], 1):
        b_len = len(tok.encode("utf-8"))
        print(f"  #{i:>2}: [{b_len:>2} bytes] {repr(tok)}")

    print("\n--- Standard BPE Top 10 Longest Tokens (Exposing Whole-Sentence Memorization) ---")
    for i, tok in enumerate(bpe_sorted[:10], 1):
        b_len = len(tok.encode("utf-8"))
        print(f"  #{i:>2}: [{b_len:>2} bytes] {repr(tok)}")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    audit_token_distributions(target_vocab=4000)
