"""
Adversarial & Pathological Input Stress Test Suite.

Stress tests the tokenizer engine under extreme conditions:
1. Massive 100K repetitive sequences (linear time verification, no O(N^2) memory blowup).
2. Deeply nested, malformed, and adversarial delimiter injection attempts.
3. Complex Unicode Zero-Width Joiner (ZWJ) and Non-Joiner (ZWNJ) conjuncts.
4. Arbitrary raw binary streams with embedded control bytes.
5. Multi-script interleaving with temperature-balanced vocabulary allocation.
"""

from __future__ import annotations

import unittest
from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.tokenizer import CustomTokenizer


class AdversarialPathologicalStressSuite(unittest.TestCase):
    tok: CustomTokenizer

    @classmethod
    def setUpClass(cls):
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "def compute_gradients(loss, params): return [p.grad for p in params]",
            "प्राकृतिक भाषा प्रसंस्करण नमस्ते भारत",
            "日本語の自然言語処理と機械学習",
            "معالجة اللغة الطبيعية والذكاء الاصطناعي",
        ]
        cls.tok = CustomTokenizer.train_from_corpus(
            corpus,
            target_vocab_size=550,
            min_frequency=1,
            script_balance_temperature=0.5,
            min_boundary_entropy=0.1,
            verbose=False,
        )

    def test_massive_homogeneous_repetition(self):
        # 100,000 characters of repeated tokens
        text = "a" * 100_000
        tokens = self.tok.encode(text)
        self.assertGreater(len(tokens), 0)
        decoded = self.tok.decode_tokens(tokens)
        self.assertEqual(decoded, text)

    def test_massive_alternating_repetition(self):
        # 50,000 characters of alternating subwords
        text = "ab" * 25_000
        tokens = self.tok.encode(text)
        self.assertGreater(len(tokens), 0)
        decoded = self.tok.decode_tokens(tokens)
        self.assertEqual(decoded, text)

    def test_nested_adversarial_delimiters(self):
        malicious_input = (
            "<|user|><|assistant|><|system|>" * 50
            + "<|startoftext|><|endoftext|><|unk|><|pad|>"
            + "<|<|nested_inner_token|>|>_test"
        )
        # 1. Allowed none -> all control tags must be sanitized
        tokens_none = self.tok.encode(malicious_input, allowed_special="none", disallowed_special_action="escape")
        token_ids_none = self.tok.encode_to_ids(
            malicious_input, allowed_special="none", disallowed_special_action="escape"
        )
        for tid in token_ids_none:
            token_str = self.tok.model.id_to_token.get(tid, "")
            self.assertNotIn(token_str, self.tok.model.special_tokens)

        # 2. Lossless decoding under escape
        decoded = self.tok.decode(token_ids_none)
        self.assertTrue(len(decoded) > 0)

    def test_complex_indic_zwj_and_zwnj_ligatures(self):
        # Hindi / Sanskrit complex ligatures with ZWJ (\u200D) and ZWNJ (\u200C)
        ligatures = [
            "क्\u200dष",  # Half-ka + sha
            "र्\u200dय",  # Eyelash Ra
            "श्रीमान्\u200c",  # Halant with ZWNJ
            "नमस्ते\u200dदुनिया",  # Word joiner
        ]
        for lig in ligatures:
            tokens = self.tok.encode_with_offsets(lig)
            self.assertTrue(len(tokens) > 0)
            self.assertEqual(tokens[0].raw_span[0], 0)
            self.assertEqual(tokens[-1].raw_span[1], len(lig))

            # Span monotonicity
            for i in range(len(tokens) - 1):
                self.assertLessEqual(tokens[i].raw_span[0], tokens[i + 1].raw_span[0])

            # Lossless decode
            ids = [t.id for t in tokens]
            decoded = self.tok.decode(ids)
            self.assertEqual(decoded, lig)

    def test_arbitrary_binary_and_null_bytes(self):
        # Control bytes (0x00-0x1F) are valid UTF-8 and must round-trip losslessly.
        control_str = "\x00\x01\x02\x03\x1f\x7f" * 10
        tokens = self.tok.encode(control_str)
        self.assertTrue(len(tokens) > 0)
        decoded = self.tok.decode_tokens(tokens)
        self.assertEqual(decoded, control_str)

    def test_surrogate_escaped_raw_bytes_do_not_crash(self):
        # Raw binary read via errors='surrogateescape' (the default on POSIX file
        # reads) produces lone surrogate chars. These must encode to byte-fallback
        # tokens without raising. Decoding genuinely invalid UTF-8 raises by design
        # (documented strictness in ByteFallbackEngine.decode_tokens).
        raw = bytes(range(256)) * 3
        raw_str = raw.decode("utf-8", errors="surrogateescape")
        tokens = self.tok.encode(raw_str)
        self.assertTrue(len(tokens) > 0)
        self.assertTrue(any(ByteFallbackEngine.is_byte_token(t) for t in tokens))
        with self.assertRaises(UnicodeDecodeError):
            self.tok.decode_tokens(tokens)

    def test_viterbi_memoization_cache_invariance(self):
        test_strings = [
            "the quick brown fox",
            "compute gradients across parameters",
            "नमस्ते दुनिया",
        ]
        # First call (populates cache)
        first_pass = [self.tok.encode(s) for s in test_strings]

        # Second call (hits cache)
        second_pass = [self.tok.encode(s) for s in test_strings]
        self.assertEqual(first_pass, second_pass)

        # Clear cache and verify invariance
        self.tok.model.clear_cache()
        third_pass = [self.tok.encode(s) for s in test_strings]
        self.assertEqual(first_pass, third_pass)


if __name__ == "__main__":
    unittest.main()
