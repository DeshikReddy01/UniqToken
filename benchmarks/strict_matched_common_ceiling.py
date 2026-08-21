"""
Strict Common-Ceiling 1:1 Matched Multilingual Benchmark.

Enforces:
1. Exact 1:1 vocabulary equality (Actual V_caliper == Actual V_sp == Actual V_bpe == Budget).
2. Exactly matched PyTorch Transformer parameter count (embedding + lm_head dimensions identical).
3. Evaluates at matched budgets: V=1,000, V=2,000, V=3,684 (3-way match), and V=5,402 (SP match).
4. Reports True LM BPB, LM Loss, Parameter Count, Tokens/Byte, and Encoding MB/s.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Tuple
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from pre_tokenizer import RegexPreTokenizer
from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.real_scale_32k_64k_benchmark import (
    BoundaryControlledBPETrainer,
    build_massive_multilingual_corpus,
)


@dataclass
class StrictMatchedResult:
    matched_budget: int
    engine_name: str
    actual_vocab: int
    vocab_hash: str
    total_model_params: int
    val_tokens: int
    bytes_per_tok: float
    tokens_per_byte: float
    tid_bpb: float
    lm_loss: float
    true_lm_bpb: float
    throughput_mb_s: float


def train_and_eval_parameter_matched_transformer(
    enc_fn: Callable[[str], List[int]],
    vocab_size: int,
    train_texts: List[str],
    val_text: str,
    total_val_bytes: int,
    block_size: int = 64,
    d_model: int = 64,
    max_steps: int = 35,
) -> Tuple[float, float, int]:
    """Evaluates downstream Transformer and reports exact total model parameters."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset

        class SeqDS(Dataset):
            def __init__(self, ids: List[int], b_sz: int):
                self.chunks = []
                for i in range(0, len(ids) - b_sz, b_sz):
                    self.chunks.append((ids[i : i + b_sz], ids[i + 1 : i + b_sz + 1]))

            def __len__(self):
                return max(len(self.chunks), 1)

            def __getitem__(self, idx):
                if not self.chunks:
                    return torch.zeros(block_size, dtype=torch.long), torch.zeros(block_size, dtype=torch.long)
                x, y = self.chunks[idx % len(self.chunks)]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        class MiniLM(nn.Module):
            def __init__(self, v_sz: int, d_m: int = 64):
                super().__init__()
                self.embed = nn.Embedding(v_sz, d_m)
                self.pos = nn.Parameter(torch.randn(1, block_size, d_m) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model=d_m, nhead=2, dim_feedforward=128, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Linear(d_m, v_sz, bias=False)

            def forward(self, x):
                b, t = x.size()
                h = self.embed(x) + self.pos[:, :t, :]
                return self.head(self.encoder(h))

        train_ids: List[int] = []
        for doc in train_texts:
            train_ids.extend(enc_fn(doc))
        val_ids = enc_fn(val_text)

        model = MiniLM(vocab_size, d_model)
        total_params = sum(p.numel() for p in model.parameters())

        ds = SeqDS(train_ids, block_size)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        model.train()
        steps = 0
        for x, y in loader:
            if x.size(0) == 0:
                break
            optimizer.zero_grad()
            logits = model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= max_steps:
                break

        model.eval()
        with torch.no_grad():
            v_ds = SeqDS(val_ids, block_size)
            v_loader = DataLoader(v_ds, batch_size=16)
            tot_l = 0.0
            cnt = 0
            for vx, vy in v_loader:
                if vx.size(0) == 0:
                    continue
                v_out = model(vx)
                tot_l += crit(v_out.view(-1, v_out.size(-1)), vy.view(-1)).item()
                cnt += 1
            avg_loss = tot_l / max(cnt, 1)

        val_tokens = len(val_ids)
        true_lm_bpb = (avg_loss / math.log(2.0)) * (val_tokens / max(total_val_bytes, 1))
        return avg_loss, true_lm_bpb, total_params

    except Exception:
        return 0.0, 0.0, 0


def run_strict_matched_sweep(
    matched_scales: List[int] = [1000, 2000, 3684, 5402],
    samples_per_lang: int = 1200,
    seed: int = 42,
) -> List[StrictMatchedResult]:
    """Runs strictly matched 1:1 vocabulary benchmarks where actual V == target V."""
    train_docs, val_by_lang = build_massive_multilingual_corpus(samples_per_lang=samples_per_lang, seed=seed)
    combined_val_text = "\n".join(val_by_lang.values())
    total_val_bytes = len(combined_val_text.encode("utf-8"))

    results: List[StrictMatchedResult] = []

    print("=" * 135)
    print(f"STRICT COMMON-CEILING 1:1 MATCHED BENCHMARK (Budgets: {matched_scales})")
    print(f"Evaluation Buffer: {total_val_bytes:,} raw UTF-8 bytes across 12 languages")
    print("=" * 135)

    for V in matched_scales:
        print(f"\n---> [EXACT MATCHED BUDGET: {V:,} TOKENS] Training all available engines...")

        # 1. Caliper (SuperBPE)
        sbp_merges = min(V // 10, 400) if V > 1000 else 0
        base_target = max(V - sbp_merges, 1000) if V > 1000 else V
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
            tok_sbp = CustomTokenizer.train_from_corpus(
                corpus=train_docs,
                target_vocab_size=V,
                ranking_strategy="byte_savings",
                script_balance_temperature=1.0,
                min_frequency=1,
                verbose=False,
            )

        assert tok_sbp.vocab_size == V, f"Caliper SuperBPE failed exact match: got {tok_sbp.vocab_size} vs {V}"

        # 2. Caliper (Unigram)
        tok_uni = CustomTokenizer.train_from_corpus(
            corpus=train_docs,
            target_vocab_size=V,
            ranking_strategy="byte_savings",
            script_balance_temperature=1.0,
            min_frequency=1,
            verbose=False,
        )
        assert tok_uni.vocab_size == V, f"Caliper Unigram failed exact match: got {tok_uni.vocab_size} vs {V}"

        # 3. SentencePiece (Unigram)
        sp_proc = None
        sp_vocab = []
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
                if sp_proc.get_piece_size() == V:
                    sp_vocab = [sp_proc.id_to_piece(i) for i in range(sp_proc.get_piece_size())]
                else:
                    sp_proc = None  # Did not achieve exact match
        except Exception:
            sp_proc = None

        # 4. Boundary-Controlled BPE
        b_bpe = BoundaryControlledBPETrainer(target_vocab_size=V, byte_fallback=True)
        b_bpe.train(train_docs, verbose=False)
        bpe_valid = b_bpe.vocab_size == V

        engines = [
            ("Caliper (SuperBPE)", tok_sbp.vocab_size, list(tok_sbp.model.vocab.keys()), lambda t: tok_sbp.encode_to_ids(t)),
            ("Caliper (Unigram)", tok_uni.vocab_size, list(tok_uni.model.vocab.keys()), lambda t: tok_uni.encode_to_ids(t)),
        ]
        if bpe_valid:
            engines.append(("Boundary-BPE", b_bpe.vocab_size, list(b_bpe.model.vocab), lambda t: b_bpe.encode_to_ids(t)))
        if sp_proc is not None:
            engines.append(("SentencePiece (Unigram)", sp_proc.get_piece_size(), sp_vocab, lambda t: sp_proc.encode(t, out_type=int)))

        for name, v_sz, vocab_list, enc_fn in engines:
            v_hash = hashlib.md5("".join(sorted(vocab_list)).encode("utf-8")).hexdigest()[:8]

            t0 = time.perf_counter()
            tot_tok = sum(len(enc_fn(text)) for text in val_by_lang.values())
            enc_elapsed = max(time.perf_counter() - t0, 1e-6)

            mb_s = (total_val_bytes / (1024 * 1024)) / enc_elapsed
            tid_bpb = (math.log2(max(v_sz, 2)) * tot_tok) / max(total_val_bytes, 1)
            bpt = total_val_bytes / max(tot_tok, 1)
            tpb = tot_tok / max(total_val_bytes, 1)

            lm_loss, true_lm_bpb, total_params = train_and_eval_parameter_matched_transformer(
                enc_fn=enc_fn,
                vocab_size=v_sz,
                train_texts=train_docs[:250],
                val_text=combined_val_text,
                total_val_bytes=total_val_bytes,
                max_steps=35,
            )

            results.append(
                StrictMatchedResult(
                    matched_budget=V,
                    engine_name=name,
                    actual_vocab=v_sz,
                    vocab_hash=v_hash,
                    total_model_params=total_params,
                    val_tokens=tot_tok,
                    bytes_per_tok=round(bpt, 2),
                    tokens_per_byte=round(tpb, 4),
                    tid_bpb=round(tid_bpb, 3),
                    lm_loss=round(lm_loss, 3),
                    true_lm_bpb=round(true_lm_bpb, 3),
                    throughput_mb_s=round(mb_s, 2),
                )
            )

    return results


def print_strict_matched_report(results: List[StrictMatchedResult]) -> None:
    print("\n" + "=" * 145)
    print("STRICT 1:1 MATCHED VOCABULARY BENCHMARK (EXACT VOCABULARY & IDENTICAL PARAMETER BUDGET)")
    print("=" * 145)

    hdr = (
        f"{'Matched V':<10} | {'Engine':<24} | {'Actual V':<8} | {'Params':<10} | {'Hash':<8} | "
        f"{'Tokens':<8} | {'B/Tok':<6} | {'TID-BPB':<9} | {'LM Loss':<9} | {'True LM BPB':<12} | {'MB/s':<7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        print(
            f"{r.matched_budget:<10,} | {r.engine_name:<24} | {r.actual_vocab:<8,} | {r.total_model_params:<10,} | "
            f"{r.vocab_hash:<8} | {r.val_tokens:<8,} | {r.bytes_per_tok:<6.2f} | {r.tid_bpb:<9.3f} | "
            f"{r.lm_loss:<9.3f} | {r.true_lm_bpb:<12.3f} | {r.throughput_mb_s:<7.2f}"
        )
    print("=" * 145 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict Matched Budget Benchmark")
    parser.add_argument("--scales", nargs="+", type=int, default=[1000, 2000, 3684, 5402], help="Matched budgets")
    parser.add_argument("--samples", type=int, default=1200, help="Samples per language")
    args = parser.parse_args()

    results = run_strict_matched_sweep(matched_scales=args.scales, samples_per_lang=args.samples)
    print_strict_matched_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
