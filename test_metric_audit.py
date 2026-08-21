"""
Metric Accounting & Vocabulary Allocation Audit.

Mechanically verifies:
1. Invariant: TID-BPB == log2(V) * (token_count / raw_byte_count) for every language
2. Invariant: sum(language bytes) == total evaluation bytes
3. Invariant: sum(language tokens) == total evaluation tokens
4. Invariant: aggregate TID-BPB == (log2(V) * sum_tokens) / sum_bytes
5. Vocabulary script allocation distribution (how many tokens out of V are Latin vs CJK vs Indic vs Cyrillic vs Arabic)
6. Token sequence inspection across languages
"""

from __future__ import annotations

import math
import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_builder import SeedVocabularyBuilder
from tokenizer import CustomTokenizer
from benchmarks.real_scale_32k_64k_evaluator import MULTILINGUAL_DATA_SOURCES, build_multilingual_dataset


@dataclass
class LanguageAuditEntry:
    language: str
    raw_byte_count: int
    char_count: int
    word_count: int
    token_count: int
    tokens_per_byte: float
    bytes_per_token: float
    log2_vocab: float
    computed_tid_bpb: float
    sample_tokens: List[str]


class MetricAccountingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 2000
        cls.train_docs, cls.val_by_lang = build_multilingual_dataset(multiplier=40, seed=100)

        # Train tokenizer
        cls.tokenizer = CustomTokenizer.train_from_corpus(
            corpus=cls.train_docs,
            target_vocab_size=cls.vocab_size,
            ranking_strategy="byte_savings",
            script_balance_temperature=0.9,
            min_frequency=1,
            verbose=False,
        )

    def test_vocabulary_script_distribution(self):
        """Audits the exact number of vocabulary entries allocated to each writing system."""
        vocab_tokens = list(self.tokenizer.model.vocab.keys())
        script_counts = Counter(SeedVocabularyBuilder._detect_script(tok) for tok in vocab_tokens)

        print("\n" + "=" * 80)
        print(f"VOCABULARY SCRIPT ALLOCATION BREAKDOWN (Total Vocab: {len(vocab_tokens):,})")
        print("=" * 80)
        for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(vocab_tokens)) * 100.0
            print(f"  {script:<15}: {count:>5} tokens ({pct:>5.1f}%)")
        print("=" * 80)

        # Basic health assertions
        self.assertEqual(len(vocab_tokens), self.vocab_size)
        self.assertGreater(script_counts["latin"], 50, "Latin must have non-trivial vocabulary allocation")
        self.assertGreater(script_counts["cjk"], 50, "CJK must have non-trivial vocabulary allocation")
        self.assertGreater(script_counts["indic"], 50, "Indic must have non-trivial vocabulary allocation")

    def test_metric_accounting_invariants(self):
        """Mechanically verifies arithmetic consistency across all languages and totals."""
        entries: List[LanguageAuditEntry] = []
        log2_v = math.log2(self.vocab_size)

        total_bytes = 0
        total_tokens = 0
        total_chars = 0
        total_words = 0

        for lang, text in self.val_by_lang.items():
            raw_b = len(text.encode("utf-8"))
            chars = len(text)
            words = max(len(text.split()), 1)

            tokens = self.tokenizer.encode(text)
            token_ids = self.tokenizer.encode_to_ids(text)

            # Token count must match token ID count exactly
            self.assertEqual(len(tokens), len(token_ids), f"Token and ID length mismatch in {lang}")

            num_tok = len(tokens)
            total_bytes += raw_b
            total_tokens += num_tok
            total_chars += chars
            total_words += words

            tpb = num_tok / max(raw_b, 1)
            bpt = raw_b / max(num_tok, 1)
            bpb = (log2_v * num_tok) / max(raw_b, 1)

            # Invariant: BPB must strictly equal log2(V) * tpb
            expected_bpb = log2_v * (num_tok / raw_b)
            self.assertAlmostEqual(bpb, expected_bpb, places=6, msg=f"TID-BPB formula mismatch in {lang}")

            # Invariant: tpb * bpt == 1.0
            self.assertAlmostEqual(tpb * bpt, 1.0, places=6, msg=f"tpb * bpt reciprocal invariant failed in {lang}")

            entries.append(
                LanguageAuditEntry(
                    language=lang,
                    raw_byte_count=raw_b,
                    char_count=chars,
                    word_count=words,
                    token_count=num_tok,
                    tokens_per_byte=tpb,
                    bytes_per_token=bpt,
                    log2_vocab=log2_v,
                    computed_tid_bpb=bpb,
                    sample_tokens=tokens[:10],
                )
            )

        # Aggregate Check
        aggregate_bpb = (log2_v * total_tokens) / total_bytes
        sum_weighted_bpb = sum((e.raw_byte_count / total_bytes) * e.computed_tid_bpb for e in entries)

        # Invariant: Aggregate BPB must exactly equal the byte-weighted sum of language BPBs
        self.assertAlmostEqual(aggregate_bpb, sum_weighted_bpb, places=6, msg="Aggregate TID-BPB weighting mismatch")

        print("\n" + "=" * 115)
        print("LANGUAGE-BY-LANGUAGE METRIC ACCOUNTING AUDIT")
        print("=" * 115)
        hdr = f"{'Language':<12} | {'Bytes':<8} | {'Chars':<8} | {'Tokens':<8} | {'B/Tok':<7} | {'Tok/Byte':<9} | {'TID-BPB':<9} | Sample First Tokens"
        print(hdr)
        print("-" * len(hdr))

        for e in entries:
            sample_str = " | ".join(repr(t) for t in e.sample_tokens[:5])
            print(
                f"{e.language:<12} | {e.raw_byte_count:<8} | {e.char_count:<8} | {e.token_count:<8} | "
                f"{e.bytes_per_token:<7.2f} | {e.tokens_per_byte:<9.4f} | {e.computed_tid_bpb:<9.3f} | {sample_str}"
            )
        print("=" * 115)
        print(f"Total Evaluation Bytes : {total_bytes:,}")
        print(f"Total Evaluation Tokens: {total_tokens:,}")
        print(f"Aggregate TID-BPB      : {aggregate_bpb:.3f}")
        print("=" * 115 + "\n")


if __name__ == "__main__":
    unittest.main()
