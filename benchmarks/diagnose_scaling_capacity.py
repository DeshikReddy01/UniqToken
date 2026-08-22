import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sentencepiece as spm
from tempfile import TemporaryDirectory
from collections import Counter
import torch
import torch.nn as nn

from bpe_trainer import BPETrainer
from pre_tokenizer import RegexPreTokenizer
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from cem_merger import CrossEntropyMerging
from benchmarks.run_phase_three_strict_matched import (
    generate_rich_multilingual_corpus,
    TARGET_TRAINING_FLOPS,
)
from benchmarks.flop_counter import plan_training_steps_for_target_flops


class MiniLM(nn.Module):
    def __init__(self, v_sz: int, d_m: int = 64):
        super().__init__()
        self.embed = nn.Embedding(v_sz, d_m)
        self.pos = nn.Parameter(torch.randn(1, 128, d_m) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_m, nhead=2, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_m, v_sz, bias=False)


train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=500, seed=101)
all_text = "\n".join(train_docs)
words = [w for w in all_text.split() if w]
unique_words = set(words)
print("=" * 120)
print("CORPUS CAPACITY DIAGNOSTIC (num_docs = 500):")
print(f"Total Words in Training Corpus  : {len(words):,}")
print(f"Unique Words in Training Corpus : {len(unique_words):,}")
print(f"Total Raw Bytes                 : {len(all_text.encode('utf-8')):,}")
print("=" * 120)

scales = [8192, 16384, 32768, 65536]
results = []

for V in scales:
    # 1. SentencePiece
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
        sp_actual_v = sp_proc.get_piece_size()

    # 2. Boundary-BPE
    b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
    bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
    bpe_model = b_bpe.train(bpe_chunks, verbose=False)
    bpe_actual_v = len(bpe_model.vocab)

    # 3. Caliper
    sbp_merges = min(V // 10, 3000)
    base_target = max(V - sbp_merges, 1000)
    actual_merges = V - base_target
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
    cal_base_v = len(tok_base.model.vocab)
    pretok_chunks = [
        tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))
    ]
    cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
    cal_actual_v = len(sbp_model.vocab)

    # FLOPs and Parameter accounting
    m_sp = MiniLM(sp_actual_v, 64)
    m_bpe = MiniLM(bpe_actual_v, 64)
    m_cal = MiniLM(cal_actual_v, 64)

    steps_sp, flops_sp, _, _ = plan_training_steps_for_target_flops(
        TARGET_TRAINING_FLOPS, sp_actual_v, 16, 128, 64, 2, 2, 128
    )
    steps_bpe, flops_bpe, _, _ = plan_training_steps_for_target_flops(
        TARGET_TRAINING_FLOPS, bpe_actual_v, 16, 128, 64, 2, 2, 128
    )
    steps_cal, flops_cal, _, _ = plan_training_steps_for_target_flops(
        TARGET_TRAINING_FLOPS, cal_actual_v, 16, 128, 64, 2, 2, 128
    )

    results.append(
        {
            "req_V": V,
            "sp_actual_V": sp_actual_v,
            "bpe_actual_V": bpe_actual_v,
            "cal_actual_V": cal_actual_v,
            "sp_params": sum(p.numel() for p in m_sp.parameters()),
            "bpe_params": sum(p.numel() for p in m_bpe.parameters()),
            "cal_params": sum(p.numel() for p in m_cal.parameters()),
            "sp_steps": steps_sp,
            "bpe_steps": steps_bpe,
            "cal_steps": steps_cal,
            "sp_flops": flops_sp,
            "bpe_flops": flops_bpe,
            "cal_flops": flops_cal,
        }
    )

print("\n" + "=" * 140)
print(
    f"{'Requested V':<12} | {'SP Act V':<10} | {'BPE Act V':<10} | {'Cal Act V':<10} | {'SP Params':<10} | {'BPE Params':<10} | {'Cal Params':<10} | {'SP Steps':<8} | {'Cal Steps':<8}"
)
print("-" * 140)
for r in results:
    print(
        f"V={r['req_V']:<10} | {r['sp_actual_V']:<10} | {r['bpe_actual_V']:<10} | {r['cal_actual_V']:<10} | {r['sp_params']:<10,} | {r['bpe_params']:<10,} | {r['cal_params']:<10,} | {r['sp_steps']:<8} | {r['cal_steps']:<8}"
    )
print("=" * 140)
