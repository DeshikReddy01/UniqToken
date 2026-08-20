"""
Unit tests for Downstream Model Pretraining & BPB Benchmark.
"""

from __future__ import annotations

import math
import unittest
from benchmarks.train_toy_transformer import (
    create_tokenizers,
    train_toy_transformer,
    PRETRAINING_CORPUS,
)


class DownstreamTransformerTests(unittest.TestCase):
    def test_downstream_tokenizers_creation(self):
        tokenizers = create_tokenizers(target_vocab=500)
        self.assertIn("Caliper (Unigram)", tokenizers)
        self.assertIn("Caliper (SuperBPE)", tokenizers)
        self.assertIn("Standard BPE", tokenizers)

    def test_downstream_pretraining_step_and_bpb(self):
        tokenizers = create_tokenizers(target_vocab=500)
        tok = tokenizers["Caliper (SuperBPE)"]
        metrics = train_toy_transformer(
            tok,
            "Caliper (SuperBPE)",
            PRETRAINING_CORPUS[:4],
            steps=5,
            seq_len=24,
            batch_size=2,
            dim=32,
            heads=2,
            layers=1,
        )
        self.assertGreater(metrics.total_tokens, 0)
        self.assertGreater(metrics.compression_ratio, 0.0)
        self.assertGreater(metrics.final_loss, 0.0)
        self.assertTrue(math.isfinite(metrics.final_loss))
        self.assertTrue(math.isfinite(metrics.bits_per_byte))
        # BPB must be internally consistent: loss(nats/tok)*tokens / (bytes*ln2).
        expected_bpb = metrics.final_loss * metrics.total_tokens / (metrics.total_bytes * math.log(2.0))
        self.assertAlmostEqual(metrics.bits_per_byte, expected_bpb, places=6)


if __name__ == "__main__":
    unittest.main()
