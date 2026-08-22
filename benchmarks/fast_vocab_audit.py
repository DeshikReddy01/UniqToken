import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sentencepiece as spm
from tempfile import TemporaryDirectory
from bpe_trainer import BPETrainer
from tokenizer import CustomTokenizer
from cem_merger import CrossEntropyMerging
from benchmarks.run_phase_three_strict_matched import generate_rich_multilingual_corpus
from benchmarks.flop_counter import plan_training_steps_for_target_flops

train_docs, val_by_lang = generate_rich_multilingual_corpus(num_docs=500, seed=101)

scales = [8192, 16384, 32768, 65536]

print("=" * 100)
print(f"{'Scale':<10} | {'SP Actual V':<15} | {'BPE Actual V':<15} | {'Caliper Actual V':<18}")
print("-" * 100)

for V in scales:
    # SP
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
        sp_act = sp_proc.get_piece_size()

    # BPE
    b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
    bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
    bpe_model = b_bpe.train(bpe_chunks, verbose=False)
    bpe_act = len(bpe_model.vocab)

    # Caliper
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
    pretok_chunks = [tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))]
    cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
    cal_act = len(sbp_model.vocab)

    print(f"V={V:<8} | {sp_act:<15} | {bpe_act:<15} | {cal_act:<18}")

print("=" * 100)
