from __future__ import annotations

"""Regression tests for external-audit findings.

Covers: BPE inter-word space corruption, batch special-token sanitization
bypass, CEM stale total_pairs ordering, and BPE strict decode of invalid IDs.
"""

import random
import unittest

from bpe_model import BPEModel  # noqa: F401  (import guard for typo checks)
from bpe_trainer import BPETrainer
from cem_merger import CrossEntropyMerging
from tokenizer import CustomTokenizer


def _train_unigram(vocab_size: int = 300) -> CustomTokenizer:
    return CustomTokenizer.train_from_corpus(
        ["hello world " * 40, "foo bar baz qux " * 40],
        target_vocab_size=vocab_size,
        verbose=False,
    )


class BPEWhitespaceTests(unittest.TestCase):
    def _train_bpe(self) -> BPEModel:
        # Corpus with NO literal space token: the trainer alphabet is built
        # from word-internal characters only.
        trainer = BPETrainer(target_vocab_size=280)
        return trainer.train(["lowest", "lower", "newest", "widest"] * 20)

    def test_inter_word_space_survives_roundtrip(self):
        model = self._train_bpe()
        self.assertNotIn(" ", model.vocab, "precondition: space is not a trained token")
        ids = model.encode_to_ids("a b")
        unk_id = model.token_to_id.get("<|unk|>", 0)
        self.assertNotIn(unk_id, ids, "inter-word space must not fall back to unk")
        self.assertEqual(model.decode(ids), "a b")

    def test_space_token_used_when_in_vocab(self):
        trainer = BPETrainer(target_vocab_size=300)
        model = trainer.train(["a b", "a b", "c d", "c d"] * 20)
        toks = model.encode("a b")
        self.assertIn(" ", toks, "when trained, the literal space should be used")

    def test_strict_decode_rejects_invalid_ids(self):
        model = self._train_bpe()
        max_id = max(model.id_to_token)
        with self.assertRaises(ValueError):
            model.decode([max_id + 123], strict=True)
        # lenient (default) still skips unknown IDs without raising
        self.assertEqual(model.decode([max_id + 123]), "")


class BatchSecurityParityTests(unittest.TestCase):
    def test_batch_matches_single_for_disallowed_specials(self):
        tok = _train_unigram()
        eos_id = tok.model.token_to_id.get("<|endoftext|>")
        if eos_id is None:
            self.skipTest("model has no <|endoftext|> token")
        texts = ["hello <|endoftext|> world"] * 3

        single_ids = [tok.encode_to_ids(t) for t in texts]
        batch_ids = tok.encode_to_ids_batch(texts)
        self.assertEqual(single_ids, batch_ids)
        for ids in batch_ids:
            self.assertNotIn(eos_id, ids, "disallowed special token leaked into batch IDs")

        batch_tokens = tok.encode_batch(texts)
        self.assertEqual(
            [tok.encode(t) for t in texts],
            batch_tokens,
        )
        self.assertNotIn("<|endoftext|>", [t for row in batch_tokens for t in row])

    def test_batch_tab_parity_with_single_encode(self):
        tok = _train_unigram()
        texts = ["a\tb", "x\ny\trz"] * 3
        self.assertEqual(
            [tok.encode_to_ids(t) for t in texts],
            tok.encode_to_ids_batch(texts),
            "batch fast path must match single-path normalization on tabs/newlines",
        )

    def test_batch_allowed_special_still_activates(self):
        tok = _train_unigram()
        eos_id = tok.model.token_to_id["<|endoftext|>"]
        texts = ["hello <|endoftext|>"] * 3
        for ids in tok.encode_to_ids_batch(texts, allowed_special="all"):
            self.assertIn(eos_id, ids)


class CEMOrderingTests(unittest.TestCase):
    def test_cem_deterministic_and_finds_merges(self):
        tok = _train_unigram(vocab_size=400)
        chunks = ["the quick brown fox", "hello world", "the quick fox", "hello there world"]
        cem = CrossEntropyMerging(max_merges=5)
        model_a = cem.optimize(tok.model, list(chunks))
        self.assertGreater(len(cem.merges), 0)

        cem2 = CrossEntropyMerging(max_merges=5)
        model_b = cem2.optimize(tok.model, list(chunks))
        self.assertEqual(
            [m[:3] for m in cem.merges],
            [m[:3] for m in cem2.merges],
            "merge order must be deterministic after the total_pairs ordering fix",
        )
        self.assertEqual(dict(model_a.vocab), dict(model_b.vocab))

    def test_cem_accepts_generator(self):
        tok = _train_unigram(vocab_size=400)
        cem = CrossEntropyMerging(max_merges=3)
        cem.optimize(tok.model, (c for c in ["hello world", "hello there"]))
        self.assertGreaterEqual(len(cem.merges), 0)


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
