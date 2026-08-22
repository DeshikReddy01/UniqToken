"""
Phase Fourteen: Factorial Tokenizer x Vocabulary x LM-Capacity Benchmark (Phase 14A Exploratory).
Evaluates 3 Vocabulary Capacities (16K, 32K, 64K) x 3 LM Architectures (Small, Medium, Large)
across 3 Tokenizers (SentencePiece, Boundary-BPE, Caliper Config B) across N=3 paired seeds [101, 202, 303].
Total: 81 matched FLOP runs (5.0e+12 analytical FLOPs on CUDA).
Tests whether the 32K->64K BPB plateau is a model capacity bottleneck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Tuple

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from pre_tokenizer import Normalizer, RegexPreTokenizer
from seed_builder import SeedToken, SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from unigram_trainer import UnigramTrainer
from benchmarks.run_phase_twelve_scaling_experiment import generate_high_entropy_corpus

TARGET_TRAINING_FLOPS = 5.0e12


@dataclass
class LMArchConfig:
    name: str
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    batch_size: int
    lr: float


LM_CONFIGS: Dict[str, LMArchConfig] = {
    "Small (4L-128d)": LMArchConfig(
        name="Small (4L-128d)",
        num_layers=4,
        d_model=128,
        num_heads=4,
        d_ff=512,
        batch_size=16,
        lr=1e-3,
    ),
    "Medium (6L-256d)": LMArchConfig(
        name="Medium (6L-256d)",
        num_layers=6,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        batch_size=16,
        lr=8e-4,
    ),
    "Large (8L-512d)": LMArchConfig(
        name="Large (8L-512d)",
        num_layers=8,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        batch_size=8,
        lr=5e-4,
    ),
}


@dataclass
class CapacityRecord:
    vocab_size: int
    lm_tier: str
    seed: int
    model_name: str
    actual_vocab_size: int
    num_layers: int
    d_model: int
    total_params: int
    non_embed_params: int
    training_steps: int
    tokens_processed: int
    true_lm_bpb: float
    token_ce_loss: float
    bytes_per_token: float
    indic_bpt: float
    fertility: float
    active_vocab_pct: float
    pct_ge_6b: float
    actual_flops: float
    wall_clock_sec: float


def calculate_analytical_flops_per_step(
    v_sz: int,
    cfg: LMArchConfig,
    seq_len: int = 64,
) -> Tuple[int, int, float]:
    """
    Computes analytical parameters and FLOPs per step:
    6 * P_non_embed * B * S + 12 * L * d_model * S^2 * B + 6 * V * d_model * B * S
    """
    d_m = cfg.d_model
    l_cnt = cfg.num_layers
    d_ff = cfg.d_ff
    b_sz = cfg.batch_size

    # Layer parameters: Attn (4 * d_m^2) + FFN (2 * d_m * d_ff) + 2 * LayerNorm (4 * d_m)
    params_per_layer = 4 * (d_m ** 2) + 2 * d_m * d_ff + 4 * d_m
    p_non_embed = l_cnt * params_per_layer + 2 * d_m
    p_embed = v_sz * d_m
    p_head = v_sz * d_m
    p_total = p_non_embed + p_embed + p_head

    # FLOPs per step (forward + backward = 6x params)
    flops_transformer = 6.0 * p_non_embed * b_sz * seq_len
    flops_attention_quad = 12.0 * l_cnt * d_m * (seq_len ** 2) * b_sz
    flops_embed_and_head = 6.0 * (p_embed + p_head) * b_sz * seq_len
    flops_per_step = flops_transformer + flops_attention_quad + flops_embed_and_head

    return p_total, p_non_embed, flops_per_step


class CausalMiniTransformer(nn.Module):
    def __init__(self, v_sz: int, cfg: LMArchConfig, block_size: int = 64):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(v_sz, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, block_size, cfg.d_model) * 0.02)
        
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, v_sz, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.size()
        causal_mask = torch.triu(torch.full((t, t), float('-inf'), device=x.device), diagonal=1)
        h = self.embed(x) + self.pos[:, :t, :]
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


def train_and_eval_capacity_transformer(
    enc_fn: Callable[[str], List[int]],
    vocab_size: int,
    cfg: LMArchConfig,
    train_texts: List[str],
    val_text: str,
    total_val_bytes: int,
    target_flops: float = TARGET_TRAINING_FLOPS,
    block_size: int = 64,
    seed: int = 42,
) -> Tuple[float, float, int, int, int, int, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    p_total, p_non_embed, flops_per_step = calculate_analytical_flops_per_step(vocab_size, cfg, block_size)
    steps = max(1, int(round(target_flops / flops_per_step)))
    actual_flops = steps * flops_per_step
    tokens_processed = steps * cfg.batch_size * block_size

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids: List[int] = []
    for doc in train_texts:
        train_ids.extend(enc_fn(doc))
    val_ids = enc_fn(val_text)

    model = CausalMiniTransformer(vocab_size, cfg, block_size).to(device)
    assert model.embed.num_embeddings == vocab_size
    assert model.head.out_features == vocab_size

    ds = SeqDS(train_ids, block_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    t_start = time.perf_counter()
    model.train()
    step_count = 0
    while step_count < steps:
        for x, y in loader:
            if x.size(0) == 0:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= steps:
                break

    model.eval()
    with torch.no_grad():
        v_ds = SeqDS(val_ids, block_size)
        v_loader = DataLoader(v_ds, batch_size=cfg.batch_size, shuffle=False)
        total_loss = 0.0
        n_tokens = 0
        for vx, vy in v_loader:
            if vx.size(0) == 0:
                continue
            vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
            logits = model(vx)
            loss = crit(logits.view(-1, logits.size(-1)), vy.view(-1))
            total_loss += loss.item() * vy.numel()
            n_tokens += vy.numel()

    val_ce_loss = total_loss / max(n_tokens, 1)
    val_tok_count = len(val_ids)
    lm_bpb = (val_ce_loss * val_tok_count) / (total_val_bytes * math.log(2))
    wall_clock = time.perf_counter() - t_start

    return val_ce_loss, lm_bpb, p_total, p_non_embed, steps, tokens_processed, actual_flops, wall_clock


def run_phase_fourteen_benchmark(
    vocab_scales: List[int] = [16384, 32768, 65536],
    lm_tiers: List[str] = ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"],
    seeds: List[int] = [101, 202, 303],
    num_docs: int = 1000,
) -> Dict[str, Any]:
    import sentencepiece as spm

    print("=" * 175)
    print("PHASE FOURTEEN: FACTORIAL TOKENIZER x VOCABULARY x LM-CAPACITY BENCHMARK (PHASE 14A EXPLORATORY)")
    print(f"Vocab Scales: {vocab_scales} | LM Tiers: {lm_tiers}")
    print(f"Seeds: {seeds} (N = {len(seeds)} paired) | Matched Training FLOPs: {TARGET_TRAINING_FLOPS:.3e} on CUDA")
    print(f"Factorial Design: 3 Vocab x 3 LM x 3 Tokenizers x {len(seeds)} Seeds = {len(vocab_scales)*len(lm_tiers)*3*len(seeds)} Total LM Runs")
    print("=" * 175)

    all_records: List[CapacityRecord] = []

    for V in vocab_scales:
        print(f"\n======================================== SCALE: V = {V:,} ========================================")
        sbp_merges = min(V // 10, 4000)
        base_target = max(V - sbp_merges, 1000)
        actual_merges = V - base_target

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n---> [Scale V = {V:,} | Paired Seed {seed_idx}/{len(seeds)}: {seed}]")
            train_docs, val_by_lang = generate_high_entropy_corpus(num_docs=num_docs, seed=seed)
            combined_val = "\n".join(val_by_lang.values())
            total_val_bytes = len(combined_val.encode("utf-8"))
            words = [w for w in combined_val.split() if w]
            num_words = len(words)

            indic_val = "\n".join([val_by_lang[l] for l in ["Hindi", "Telugu", "Tamil", "Bengali"]])
            indic_val_bytes = len(indic_val.encode("utf-8"))

            # Train all 3 tokenizers once per (V, seed)
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
                sp_tokens = list(sp_proc.encode_as_pieces(combined_val))
                sp_enc = lambda t: sp_proc.encode(t, out_type=int)
                sp_actual_v = sp_proc.get_piece_size()

            sp_counts = Counter(sp_tokens)
            sp_active_cov = len(sp_counts) / sp_actual_v
            sp_tok_bytes = [len(t.encode("utf-8")) for t in sp_tokens]
            sp_pct_ge_6b = sum(1 for b in sp_tok_bytes if b >= 6) / max(len(sp_tok_bytes), 1) * 100.0
            sp_indic_toks = list(sp_proc.encode_as_pieces(indic_val))
            sp_indic_bpt = indic_val_bytes / max(len(sp_indic_toks), 1)
            sp_bpt = total_val_bytes / max(len(sp_tokens), 1)
            sp_fert = len(sp_tokens) / max(num_words, 1)

            # 2. Boundary-BPE
            b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
            bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
            bpe_model = b_bpe.train(bpe_chunks, verbose=False)
            bpe_tokens = bpe_model.encode(combined_val)
            bpe_actual_v = len(bpe_model.vocab)
            bpe_counts = Counter(bpe_tokens)
            bpe_active_cov = len(bpe_counts) / bpe_actual_v
            bpe_tok_bytes = [len(t.encode("utf-8")) for t in bpe_tokens]
            bpe_pct_ge_6b = sum(1 for b in bpe_tok_bytes if b >= 6) / max(len(bpe_tok_bytes), 1) * 100.0
            bpe_indic_toks = bpe_model.encode(indic_val)
            bpe_indic_bpt = indic_val_bytes / max(len(bpe_indic_toks), 1)
            bpe_bpt = total_val_bytes / max(len(bpe_tokens), 1)
            bpe_fert = len(bpe_tokens) / max(num_words, 1)
            bpe_enc = lambda t: bpe_model.encode_to_ids(t)

            # 3. Caliper Config B
            tok_base = CustomTokenizer.train_from_corpus(
                corpus=train_docs,
                target_vocab_size=base_target,
                seed_multiplier=1.2,
                ranking_strategy="byte_savings",
                min_boundary_entropy=0.35,
                length_exponent=1.5,
                pruning_length_exponent=0.0,
                min_frequency=1,
                verbose=False,
            )
            pretok_chunks = [tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))]
            cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
            sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
            cal_tok = CustomTokenizer(normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model)
            cal_tokens = cal_tok.encode(combined_val)
            cal_actual_v = len(cal_tok.model.vocab)
            cal_counts = Counter(cal_tokens)
            cal_active_cov = len(cal_counts) / cal_actual_v
            cal_tok_bytes = [len(t.encode("utf-8")) for t in cal_tokens]
            cal_pct_ge_6b = sum(1 for b in cal_tok_bytes if b >= 6) / max(len(cal_tok_bytes), 1) * 100.0
            cal_indic_toks = cal_tok.encode(indic_val)
            cal_indic_bpt = indic_val_bytes / max(len(cal_indic_toks), 1)
            cal_bpt = total_val_bytes / max(len(cal_tokens), 1)
            cal_fert = len(cal_tokens) / max(num_words, 1)
            cal_enc = lambda t: cal_tok.encode_to_ids(t)

            # Sweep LM Tiers for each of the 3 tokenizers
            for lm_name in lm_tiers:
                cfg = LM_CONFIGS[lm_name]

                # Run SP
                ce_sp, bpb_sp, p_tot_sp, p_non_sp, st_sp, tok_sp, fl_sp, wc_sp = train_and_eval_capacity_transformer(
                    enc_fn=sp_enc,
                    vocab_size=sp_actual_v,
                    cfg=cfg,
                    train_texts=train_docs[:300],
                    val_text=combined_val,
                    total_val_bytes=total_val_bytes,
                    target_flops=TARGET_TRAINING_FLOPS,
                    seed=seed,
                )
                rec_sp = CapacityRecord(
                    vocab_size=V,
                    lm_tier=lm_name,
                    seed=seed,
                    model_name="SentencePiece-Unigram",
                    actual_vocab_size=sp_actual_v,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot_sp,
                    non_embed_params=p_non_sp,
                    training_steps=st_sp,
                    tokens_processed=tok_sp,
                    true_lm_bpb=bpb_sp,
                    token_ce_loss=ce_sp,
                    bytes_per_token=sp_bpt,
                    indic_bpt=sp_indic_bpt,
                    fertility=sp_fert,
                    active_vocab_pct=sp_active_cov * 100.0,
                    pct_ge_6b=sp_pct_ge_6b,
                    actual_flops=fl_sp,
                    wall_clock_sec=wc_sp,
                )
                all_records.append(rec_sp)

                # Run Boundary-BPE
                ce_bpe, bpb_bpe, p_tot_bpe, p_non_bpe, st_bpe, tok_bpe, fl_bpe, wc_bpe = train_and_eval_capacity_transformer(
                    enc_fn=bpe_enc,
                    vocab_size=bpe_actual_v,
                    cfg=cfg,
                    train_texts=train_docs[:300],
                    val_text=combined_val,
                    total_val_bytes=total_val_bytes,
                    target_flops=TARGET_TRAINING_FLOPS,
                    seed=seed,
                )
                rec_bpe = CapacityRecord(
                    vocab_size=V,
                    lm_tier=lm_name,
                    seed=seed,
                    model_name="Boundary-BPE",
                    actual_vocab_size=bpe_actual_v,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot_bpe,
                    non_embed_params=p_non_bpe,
                    training_steps=st_bpe,
                    tokens_processed=tok_bpe,
                    true_lm_bpb=bpb_bpe,
                    token_ce_loss=ce_bpe,
                    bytes_per_token=bpe_bpt,
                    indic_bpt=bpe_indic_bpt,
                    fertility=bpe_fert,
                    active_vocab_pct=bpe_active_cov * 100.0,
                    pct_ge_6b=bpe_pct_ge_6b,
                    actual_flops=fl_bpe,
                    wall_clock_sec=wc_bpe,
                )
                all_records.append(rec_bpe)

                # Run Caliper Config B
                ce_cal, bpb_cal, p_tot_cal, p_non_cal, st_cal, tok_cal, fl_cal, wc_cal = train_and_eval_capacity_transformer(
                    enc_fn=cal_enc,
                    vocab_size=cal_actual_v,
                    cfg=cfg,
                    train_texts=train_docs[:300],
                    val_text=combined_val,
                    total_val_bytes=total_val_bytes,
                    target_flops=TARGET_TRAINING_FLOPS,
                    seed=seed,
                )
                rec_cal = CapacityRecord(
                    vocab_size=V,
                    lm_tier=lm_name,
                    seed=seed,
                    model_name="Caliper-SuperBPE (Config B)",
                    actual_vocab_size=cal_actual_v,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot_cal,
                    non_embed_params=p_non_cal,
                    training_steps=st_cal,
                    tokens_processed=tok_cal,
                    true_lm_bpb=bpb_cal,
                    token_ce_loss=ce_cal,
                    bytes_per_token=cal_bpt,
                    indic_bpt=cal_indic_bpt,
                    fertility=cal_fert,
                    active_vocab_pct=cal_active_cov * 100.0,
                    pct_ge_6b=cal_pct_ge_6b,
                    actual_flops=fl_cal,
                    wall_clock_sec=wc_cal,
                )
                all_records.append(rec_cal)

                print(
                    f"  [{lm_name:<16}] SP BPB: {bpb_sp:.3f} | BPE BPB: {bpb_bpe:.3f} | CAL BPB: {bpb_cal:.3f} "
                    f"(Diff Cal-BPE: {bpb_cal-bpb_bpe:+.3f}) | Steps: SP={st_sp} BPE={st_bpe} CAL={st_cal}",
                    flush=True,
                )

    # Aggregation & Analysis
    print("\n" + "=" * 175)
    print("PHASE FOURTEEN: FACTORIAL SUMMARY TABLE (3 VOCAB x 3 LM TIERS x 3 TOKENIZERS)")
    print("=" * 175)

    summary_grid: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))

    for lm_name in lm_tiers:
        for V in vocab_scales:
            for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]:
                recs = [r for r in all_records if r.lm_tier == lm_name and r.vocab_size == V and r.model_name == m_name]
                summary_grid[lm_name][V][m_name] = {
                    "true_lm_bpb": float(np.mean([r.true_lm_bpb for r in recs])),
                    "token_ce_loss": float(np.mean([r.token_ce_loss for r in recs])),
                    "bytes_per_token": float(np.mean([r.bytes_per_token for r in recs])),
                    "indic_bpt": float(np.mean([r.indic_bpt for r in recs])),
                    "fertility": float(np.mean([r.fertility for r in recs])),
                    "active_vocab_pct": float(np.mean([r.active_vocab_pct for r in recs])),
                    "total_params": int(np.mean([r.total_params for r in recs])),
                    "training_steps": int(np.mean([r.training_steps for r in recs])),
                    "tokens_processed": int(np.mean([r.tokens_processed for r in recs])),
                }

    print(f"{'LM Tier':<18} | {'Scale':<8} | {'Tokenizer':<28} | {'Total Params':<14} | {'Steps':<8} | {'Tokens Proc':<12} | {'True BPB':<10} | {'Token CE':<10} | {'B/Tok':<8} | {'Active %'}")
    print("-" * 175)
    for lm_name in lm_tiers:
        for V in vocab_scales:
            for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]:
                st = summary_grid[lm_name][V][m_name]
                print(f"{lm_name:<18} | V={V:<6} | {m_name:<28} | {st['total_params']:<14,d} | {st['training_steps']:<8d} | {st['tokens_processed']:<12,d} | {st['true_lm_bpb']:<10.3f} | {st['token_ce_loss']:<10.3f} | {st['bytes_per_token']:<8.2f} | {st['active_vocab_pct']:<6.1f}%")
            print("-" * 175)

    # Hypothesis & Plateau Breakdown
    print("\n" + "=" * 175)
    print("PLATEAU BREAKDOWN: DELTA BPB (32K -> 64K) ACROSS LM ARCHITECTURES")
    print("=" * 175)
    print(f"{'LM Tier':<18} | {'Caliper 32K BPB':<16} | {'Caliper 64K BPB':<16} | {'Delta 32K->64K':<16} | {'SP 32K->64K Delta':<18} | {'BPE 32K->64K Delta'}")
    print("-" * 175)

    plateau_analysis: Dict[str, Dict[str, float]] = {}
    for lm_name in lm_tiers:
        c32 = summary_grid[lm_name][32768]["Caliper-SuperBPE (Config B)"]["true_lm_bpb"]
        c64 = summary_grid[lm_name][65536]["Caliper-SuperBPE (Config B)"]["true_lm_bpb"]
        d_cal = c64 - c32

        sp32 = summary_grid[lm_name][32768]["SentencePiece-Unigram"]["true_lm_bpb"]
        sp64 = summary_grid[lm_name][65536]["SentencePiece-Unigram"]["true_lm_bpb"]
        d_sp = sp64 - sp32

        bpe32 = summary_grid[lm_name][32768]["Boundary-BPE"]["true_lm_bpb"]
        bpe64 = summary_grid[lm_name][65536]["Boundary-BPE"]["true_lm_bpb"]
        d_bpe = bpe64 - bpe32

        plateau_analysis[lm_name] = {
            "caliper_32k": c32,
            "caliper_64k": c64,
            "delta_caliper": d_cal,
            "delta_sp": d_sp,
            "delta_bpe": d_bpe,
        }
        print(f"{lm_name:<18} | {c32:<16.3f} | {c64:<16.3f} | {d_cal:<+16.3f} | {d_sp:<+18.3f} | {d_bpe:<+18.3f}")
    print("=" * 175 + "\n")

    # Co-Design Tradeoff: Capacity-Normalized Analysis
    print("=" * 175)
    print("TOKENIZER-MODEL CO-DESIGN: CAPACITY-NORMALIZED EFFICIENCY (16K+Large vs 64K+Small)")
    print("=" * 175)
    c_16_large = summary_grid["Large (8L-512d)"][16384]["Caliper-SuperBPE (Config B)"]
    c_64_small = summary_grid["Small (4L-128d)"][65536]["Caliper-SuperBPE (Config B)"]
    print(f"Caliper 16K + Large LM (8L-512d) : BPB = {c_16_large['true_lm_bpb']:.3f} | Total Params = {c_16_large['total_params']:,} | Steps = {c_16_large['training_steps']}")
    print(f"Caliper 64K + Small LM (4L-128d) : BPB = {c_64_small['true_lm_bpb']:.3f} | Total Params = {c_64_small['total_params']:,} | Steps = {c_64_small['training_steps']}")
    print(f"--> Difference (16K+Large vs 64K+Small): {c_16_large['true_lm_bpb'] - c_64_small['true_lm_bpb']:+.3f} BPB")
    print("=" * 175 + "\n")

    # 4-Panel Publication Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
    colors = {"Small (4L-128d)": "#1f77b4", "Medium (6L-256d)": "#ff7f0e", "Large (8L-512d)": "#2ca02c"}
    markers = {"Small (4L-128d)": "o-", "Medium (6L-256d)": "s-", "Large (8L-512d)": "^-"}

    # Panel A: Caliper True LM BPB vs V for each LM Tier
    ax_a = axes[0, 0]
    for lm_name in lm_tiers:
        vals = [summary_grid[lm_name][V]["Caliper-SuperBPE (Config B)"]["true_lm_bpb"] for V in vocab_scales]
        ax_a.plot(vocab_scales, vals, markers[lm_name], color=colors[lm_name], label=lm_name, linewidth=2.2, markersize=8)
        for V, val in zip(vocab_scales, vals):
            ax_a.annotate(f"{val:.3f}", (V, val + 0.015), fontsize=8.5, color=colors[lm_name], ha="center")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(vocab_scales)
    ax_a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_a.set_title("Panel A: Caliper SuperBPE BPB vs Vocab Capacity Across LM Tiers", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_a.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend()

    # Panel B: SentencePiece-Unigram True LM BPB vs V for each LM Tier
    ax_b = axes[0, 1]
    for lm_name in lm_tiers:
        vals = [summary_grid[lm_name][V]["SentencePiece-Unigram"]["true_lm_bpb"] for V in vocab_scales]
        ax_b.plot(vocab_scales, vals, markers[lm_name], color=colors[lm_name], label=lm_name, linewidth=2.2, markersize=8)
        for V, val in zip(vocab_scales, vals):
            ax_b.annotate(f"{val:.3f}", (V, val + 0.015), fontsize=8.5, color=colors[lm_name], ha="center")
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(vocab_scales)
    ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b.set_title("Panel B: SentencePiece BPB vs Vocab Capacity Across LM Tiers", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_b.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend()

    # Panel C: Boundary-BPE True LM BPB vs V for each LM Tier
    ax_c = axes[1, 0]
    for lm_name in lm_tiers:
        vals = [summary_grid[lm_name][V]["Boundary-BPE"]["true_lm_bpb"] for V in vocab_scales]
        ax_c.plot(vocab_scales, vals, markers[lm_name], color=colors[lm_name], label=lm_name, linewidth=2.2, markersize=8)
        for V, val in zip(vocab_scales, vals):
            ax_c.annotate(f"{val:.3f}", (V, val + 0.015), fontsize=8.5, color=colors[lm_name], ha="center")
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(vocab_scales)
    ax_c.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_c.set_title("Panel C: Boundary-BPE BPB vs Vocab Capacity Across LM Tiers", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_c.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend()

    # Panel D: Direct 3-Way Model Comparison at Large LM vs Small LM
    ax_d = axes[1, 1]
    m_colors = {"SentencePiece-Unigram": "#1f77b4", "Boundary-BPE": "#2ca02c", "Caliper-SuperBPE (Config B)": "#d62728"}
    for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]:
        vals_large = [summary_grid["Large (8L-512d)"][V][m_name]["true_lm_bpb"] for V in vocab_scales]
        vals_small = [summary_grid["Small (4L-128d)"][V][m_name]["true_lm_bpb"] for V in vocab_scales]
        ax_d.plot(vocab_scales, vals_large, "^-", color=m_colors[m_name], label=f"{m_name} (Large LM)", linewidth=2.2, markersize=8)
        ax_d.plot(vocab_scales, vals_small, "o--", color=m_colors[m_name], label=f"{m_name} (Small LM)", linewidth=1.5, markersize=6, alpha=0.6)
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(vocab_scales)
    ax_d.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_d.set_title("Panel D: Large LM vs Small LM Scaling Frontier (All Models)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_d.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=7.5)

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_fourteen_capacity_scaling.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 14 factorial scaling figure to {plot_path}")

    # Persist JSON
    output_data = {
        "summary_grid": summary_grid,
        "plateau_analysis": plateau_analysis,
        "all_records": [asdict(r) for r in all_records],
    }
    json_path = Path(__file__).resolve().parent / "phase_fourteen_capacity_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Records] Saved Phase 14 factorial scaling ledger to {json_path}")

    return output_data


if __name__ == "__main__":
    run_phase_fourteen_benchmark(
        vocab_scales=[16384, 32768, 65536],
        lm_tiers=["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"],
        seeds=[101, 202, 303],
        num_docs=1000,
    )
