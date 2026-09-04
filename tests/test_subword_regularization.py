"""Subword regularization tests for Issue #48.

Covers SuperBPE cross-word merge dropout and BPE dropout (Provilkov et al.,
2020) threaded through ``CustomTokenizer``, ``BPEModel`` and ``BatchCollator``
via ``dropout_prob``.
"""

from __future__ import annotations

import random
import unittest
from math import log
from unittest import mock

from uniqtoken.batch_collator import BatchCollator
from uniqtoken.bpe_model import BPEModel
from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel

DROPOUT_ERROR = "dropout_prob must be in range [0.0, 1.0)"

TEXT = "the quick the fox"
MERGED_TOKENS = ("the\u2581quick", "\u2581the", "\u2581fox")
UNMERGED_TOKENS = ("the", "\u2581quick", "\u2581the", "\u2581fox")


def _make_tokenizer() -> CustomTokenizer:
    """Tiny tokenizer whose vocabulary contains SuperBPE cross-word merges."""
    vocab = {
        "the": log(0.4),
        "\u2581quick": log(0.2),
        "the\u2581quick": log(0.25),
        "\u2581fox": log(0.15),
        "the\u2581fox": log(0.1),
        "\u2581jumps": log(0.05),
        "the\u2581jumps": log(0.05),
        "\u2581the": log(0.09),
    }
    token_to_id = {token: index for index, token in enumerate(sorted(vocab))}
    model = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token={index: token for token, index in token_to_id.items()},
        special_tokens=[],
        max_subword_len=16,
        byte_fallback=False,
    )
    return CustomTokenizer(
        normalizer=Normalizer(normalize_unicode=False),
        pre_tokenizer=RegexPreTokenizer(),
        model=model,
    )


def _make_bpe_model() -> BPEModel:
    """Tiny BPE model with merge chains so dropout is observable."""
    merges = {("t", "h"): 0, ("th", "e"): 1, ("q", "u"): 2, ("qu", "i"): 3, ("qui", "c"): 4, ("quic", "k"): 5}
    return BPEModel(
        vocab=set("thequick"),
        token_to_id={},
        id_to_token={},
        merges=merges,
        special_tokens=[],
        byte_fallback=False,
    )


def _make_two_pair_tokenizer() -> CustomTokenizer:
    """Tokenizer whose vocabulary admits two disjoint cross-word merges.

    For ``TEXT`` the base tokens are ``["the", "\\u2581quick", "\\u2581the",
    "\\u2581fox"]``, so both ``"the\\u2581quick"`` (positions 0-1) and
    ``"\\u2581the\\u2581fox"`` (positions 2-3) are valid merge candidates and
    the two pairs are independent of each other.
    """
    vocab = {
        "the": log(0.4),
        "\u2581quick": log(0.2),
        "the\u2581quick": log(0.25),
        "\u2581fox": log(0.15),
        "the\u2581fox": log(0.1),
        "\u2581jumps": log(0.05),
        "the\u2581jumps": log(0.05),
        "\u2581the": log(0.09),
        "\u2581the\u2581fox": log(0.1),
    }
    token_to_id = {token: index for index, token in enumerate(sorted(vocab))}
    model = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token={index: token for token, index in token_to_id.items()},
        special_tokens=[],
        max_subword_len=16,
        byte_fallback=False,
    )
    return CustomTokenizer(
        normalizer=Normalizer(normalize_unicode=False),
        pre_tokenizer=RegexPreTokenizer(),
        model=model,
    )


def _make_byte_fallback_tokenizer() -> CustomTokenizer:
    """Tokenizer with single-character pieces plus byte fallback for typos."""
    vocab = {
        "the": log(0.4),
        "\u2581quick": log(0.2),
        "the\u2581quick": log(0.25),
        "\u2581fox": log(0.15),
        "the\u2581fox": log(0.1),
        "\u2581the": log(0.09),
        "\u2581": log(0.05),
    }
    for char in "tehquickfoxz":
        vocab.setdefault(char, log(0.01))
    for byte_val in range(256):
        vocab.setdefault(ByteFallbackEngine.byte_to_token(byte_val), log(0.001))
    token_to_id = {token: index for index, token in enumerate(sorted(vocab))}
    model = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token={index: token for token, index in token_to_id.items()},
        special_tokens=[],
        max_subword_len=16,
        byte_fallback=True,
    )
    return CustomTokenizer(
        normalizer=Normalizer(normalize_unicode=False),
        pre_tokenizer=RegexPreTokenizer(),
        model=model,
    )


class SubwordRegularizationTestBase(unittest.TestCase):
    def setUp(self):
        self._rng_state = random.getstate()
        self.tokenizer = _make_tokenizer()
        self.collator = BatchCollator(self.tokenizer, padding_token="the", bos_token=None, eos_token=None)

    def tearDown(self):
        # Tests seed the process-global RNG; restore it so test order
        # cannot influence later stochastic tests.
        random.setstate(self._rng_state)


class DropoutDeterminismTests(SubwordRegularizationTestBase):
    def test_zero_dropout_matches_default_and_is_deterministic(self):
        baseline = tuple(self.tokenizer.encode(TEXT))
        self.assertEqual(baseline, MERGED_TOKENS)
        for _ in range(10):
            self.assertEqual(tuple(self.tokenizer.encode(TEXT, dropout_prob=0.0)), baseline)

        # Other entry points agree with the deterministic baseline.
        self.assertEqual(
            tuple(self.tokenizer.encode_to_ids(TEXT, dropout_prob=0.0)),
            tuple(self.tokenizer.encode_to_ids(TEXT)),
        )
        self.assertEqual(
            self.tokenizer.encode_batch([TEXT, TEXT], dropout_prob=0.0),
            self.tokenizer.encode_batch([TEXT, TEXT]),
        )
        self.assertEqual(
            self.tokenizer.encode_to_ids_batch([TEXT, TEXT], dropout_prob=0.0),
            self.tokenizer.encode_to_ids_batch([TEXT, TEXT]),
        )
        self.assertEqual(
            self.tokenizer.encode_with_offsets(TEXT, dropout_prob=0.0),
            self.tokenizer.encode_with_offsets(TEXT),
        )

    def test_bpe_model_zero_dropout_is_deterministic(self):
        model = _make_bpe_model()
        baseline = model.encode("the quick")
        for _ in range(10):
            self.assertEqual(model.encode("the quick", dropout_prob=0.0), baseline)


class DropoutValidationTests(SubwordRegularizationTestBase):
    def test_invalid_dropout_prob_raises_value_error(self):
        for bad in (-0.1, 1.0, 1.5, "0.5", None, [0.2]):
            for call in (
                lambda p=bad: self.tokenizer.encode(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_to_ids(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.sample(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.sample_to_ids(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_with_offsets(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_batch([TEXT], dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_to_ids_batch([TEXT], dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_with_offsets_batch([TEXT], dropout_prob=p),
                lambda p=bad: self.collator.batch_encode([TEXT], padding=False, dropout_prob=p),
                lambda p=bad: _make_bpe_model().encode("the quick", dropout_prob=p),
            ):
                with self.assertRaises(ValueError) as ctx:
                    call()
                self.assertIn(DROPOUT_ERROR, str(ctx.exception))

    def test_invalid_dropout_prob_raises_even_for_empty_inputs(self):
        # Validation must run before the empty-text / empty-batch fast
        # paths: invalid values may never succeed silently.
        for bad in (-0.1, 1.0, "0.5"):
            for call in (
                lambda p=bad: self.tokenizer.encode("", dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_to_ids("", dropout_prob=p),
                lambda p=bad: self.tokenizer.sample("", dropout_prob=p),
                lambda p=bad: self.tokenizer.sample_to_ids("", dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_with_offsets("", dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_batch([], dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_to_ids_batch([], dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_with_offsets_batch([], dropout_prob=p),
                lambda p=bad: self.collator.batch_encode([], dropout_prob=p),
                lambda p=bad: _make_bpe_model().encode("", dropout_prob=p),
            ):
                with self.assertRaises(ValueError) as ctx:
                    call()
                self.assertIn(DROPOUT_ERROR, str(ctx.exception))

    def test_boolean_dropout_prob_is_rejected(self):
        # bool is an int subclass: False would silently act as 0.0 and
        # True as 1.0, so both must be rejected explicitly.
        for bad in (False, True):
            for call in (
                lambda p=bad: self.tokenizer.encode(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_to_ids(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.sample(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_with_offsets(TEXT, dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_batch([TEXT], dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_to_ids_batch([TEXT], dropout_prob=p),
                lambda p=bad: self.tokenizer.encode_with_offsets_batch([TEXT], dropout_prob=p),
                lambda p=bad: self.collator.batch_encode([TEXT], dropout_prob=p),
                lambda p=bad: _make_bpe_model().encode("the quick", dropout_prob=p),
            ):
                with self.assertRaises(ValueError) as ctx:
                    call()
                self.assertIn(DROPOUT_ERROR, str(ctx.exception))


class SuperBPEDropoutTests(SubwordRegularizationTestBase):
    def test_superbpe_merge_dropout_is_stochastic(self):
        random.seed(42)
        seen = set()
        for _ in range(50):
            seen.add(tuple(self.tokenizer.encode(TEXT, dropout_prob=0.8)))
        # Both outcomes (merge applied / skipped) must occur across trials.
        self.assertGreaterEqual(len(seen), 2)
        self.assertIn(MERGED_TOKENS, seen)
        self.assertIn(UNMERGED_TOKENS, seen)


class BPEDropoutTests(SubwordRegularizationTestBase):
    def test_bpe_dropout_is_stochastic_and_text_preserving(self):
        model = _make_bpe_model()
        random.seed(42)
        seen = set()
        for _ in range(50):
            tokens = model.encode("the quick", dropout_prob=0.8)
            seen.add(tuple(tokens))
            # Dropped merges must never lose or reorder text: word
            # concatenation is invariant under dropout.
            first_word = "".join(tokens[: tokens.index(" ")])
            second_word = "".join(tokens[tokens.index(" ") + 1 :])
            self.assertEqual(first_word, "the")
            self.assertEqual(second_word, "quick")
        self.assertGreaterEqual(len(seen), 2)


class DropoutRoundtripTests(SubwordRegularizationTestBase):
    def test_roundtrip_is_lossless_over_random_dropout_runs(self):
        random.seed(7)
        for _ in range(100):
            tokens = self.tokenizer.encode(TEXT, dropout_prob=0.5)
            self.assertEqual(self.tokenizer.decode_tokens(tokens), TEXT)


class OffsetSpanTilingTests(SubwordRegularizationTestBase):
    def test_offset_spans_tile_text_without_gaps_or_overlaps(self):
        random.seed(3)
        for _ in range(25):
            tokens = self.tokenizer.encode_with_offsets(TEXT, dropout_prob=0.7)
            spans = [token.raw_span for token in tokens]
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], len(TEXT))
            for prev, cur in zip(spans, spans[1:]):
                self.assertLess(prev[0], prev[1])
                self.assertEqual(cur[0], prev[1], "gap or overlap between consecutive spans")
            for token in tokens:
                self.assertIn(token.text, self.tokenizer.model.vocab)


class BatchCollatorDropoutTests(SubwordRegularizationTestBase):
    def test_batch_collator_propagates_dropout_prob(self):
        self.assertEqual(
            self.collator.batch_encode([TEXT], padding=False, dropout_prob=0.0).tokens[0],
            list(MERGED_TOKENS),
        )
        random.seed(11)
        seen = set()
        for _ in range(20):
            encoded = self.collator.batch_encode([TEXT], padding=False, dropout_prob=0.3)
            seen.add(tuple(encoded.tokens[0]))
        # With dropout 0.3 on the single cross-word candidate, both the
        # merged and unmerged outcomes must appear across 20 trials.
        self.assertGreaterEqual(len(seen), 2)
        self.assertIn(MERGED_TOKENS, seen)
        self.assertIn(UNMERGED_TOKENS, seen)


TWO_PAIR_ALL_MERGED = ("the\u2581quick", "\u2581the\u2581fox")
TWO_PAIR_FIRST_DROPPED = ("the", "\u2581quick", "\u2581the\u2581fox")


class MergeBoundaryBlockingTests(unittest.TestCase):
    """Regression tests: a dropout-dropped merge boundary is never retried.

    With two disjoint cross-word candidates, dropping the first while the
    second merges forces a second merge pass. The dropped boundary must stay
    blocked there instead of being re-rolled (the pre-fix bug), so the final
    segmentation must not depend on unrelated merges elsewhere.
    """

    def setUp(self):
        self.tokenizer = _make_two_pair_tokenizer()

    def _encode_with_controlled_dropout(self, encode_fn):
        # First draw 0.0 < 0.5 -> drop the first candidate; second draw
        # 0.99 -> merge the second candidate. Under the pre-fix behavior a
        # third draw (the forbidden retry) would also be consumed.
        with mock.patch("random.random", side_effect=[0.0, 0.99]) as rng:
            result = encode_fn(dropout_prob=0.5)
        self.assertEqual(rng.call_count, 2, "a dropped merge boundary was re-rolled")
        return result

    def test_dropped_boundary_is_not_retried_after_unrelated_merge(self):
        tokens = self._encode_with_controlled_dropout(
            lambda dropout_prob: self.tokenizer.encode(TEXT, dropout_prob=dropout_prob)
        )
        self.assertEqual(tuple(tokens), TWO_PAIR_FIRST_DROPPED)

    def test_dropped_boundary_is_not_retried_on_span_path(self):
        tokens = self._encode_with_controlled_dropout(
            lambda dropout_prob: self.tokenizer.encode_with_offsets(TEXT, dropout_prob=dropout_prob)
        )
        self.assertEqual(tuple(token.text for token in tokens), TWO_PAIR_FIRST_DROPPED)
        spans = [token.raw_span for token in tokens]
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(TEXT))
        for prev, cur in zip(spans, spans[1:]):
            self.assertEqual(cur[0], prev[1], "gap or overlap between consecutive spans")

    def test_dropped_boundary_roundtrip_and_batch_consistency(self):
        tokens = self._encode_with_controlled_dropout(
            lambda dropout_prob: self.tokenizer.encode(TEXT, dropout_prob=dropout_prob)
        )
        self.assertEqual(self.tokenizer.decode_tokens(tokens), TEXT)
        with mock.patch("random.random", side_effect=[0.0, 0.99, 0.99, 0.99]):
            batch_tokens = self.tokenizer.encode_batch([TEXT], dropout_prob=0.5)[0]
        self.assertEqual(tuple(batch_tokens), TWO_PAIR_FIRST_DROPPED)

    def test_two_pair_deterministic_baseline_is_unchanged(self):
        expected = TWO_PAIR_ALL_MERGED
        self.assertEqual(tuple(self.tokenizer.encode(TEXT)), expected)
        for _ in range(10):
            self.assertEqual(tuple(self.tokenizer.encode(TEXT, dropout_prob=0.0)), expected)


class DropoutDiversityTests(unittest.TestCase):
    def setUp(self):
        self._rng_state = random.getstate()

    def tearDown(self):
        random.setstate(self._rng_state)

    def test_dropout_produces_multiple_valid_segmentations_over_1000_runs(self):
        tokenizer = _make_two_pair_tokenizer()
        random.seed(101)
        seen = set()
        for _ in range(1000):
            tokens = tokenizer.encode(TEXT, dropout_prob=0.5)
            seen.add(tuple(tokens))
            self.assertEqual(tokenizer.decode_tokens(tokens), TEXT)
        # Four outcomes are possible (each of two disjoint candidates may
        # merge or be dropped), each with probability 0.25 at p=0.5, so at
        # least three distinct segmentations must appear across 1000 runs.
        # Deliberately no exact-distribution assertion (CI must not flake).
        self.assertGreaterEqual(len(seen), 3)
        self.assertIn(TWO_PAIR_ALL_MERGED, seen)
        self.assertIn(TWO_PAIR_FIRST_DROPPED, seen)


class TypoRobustnessTests(unittest.TestCase):
    """Typo-injected text must stay lossless and behave sensibly under dropout."""

    def setUp(self):
        self._rng_state = random.getstate()

    def tearDown(self):
        random.setstate(self._rng_state)

    TYPO_TEXTS = (
        "teh quick the fox",  # transposition
        "the quik teh fxo",  # missing / swapped letters
        "the quick the foxz",  # trailing garbage
        "Teh quick THE fox!",  # case change + punctuation (byte fallback)
    )

    def test_typo_text_roundtrips_under_dropout(self):
        tokenizer = _make_byte_fallback_tokenizer()
        random.seed(23)
        for text in self.TYPO_TEXTS:
            baseline = tokenizer.encode(text)
            self.assertEqual(tokenizer.decode_tokens(baseline), text)
            seen = set()
            for _ in range(40):
                tokens = tokenizer.encode(text, dropout_prob=0.5)
                seen.add(tuple(tokens))
                self.assertEqual(tokenizer.decode_tokens(tokens), text)
                for token in tokens:
                    self.assertIn(token, tokenizer.model.vocab)

    def test_typo_word_is_segmented_into_finer_pieces(self):
        tokenizer = _make_byte_fallback_tokenizer()
        self.assertGreater(len(tokenizer.encode("teh")), len(tokenizer.encode("the")))

    def test_unknown_character_falls_back_to_bytes_under_dropout(self):
        tokenizer = _make_byte_fallback_tokenizer()
        random.seed(23)
        for _ in range(10):
            tokens = tokenizer.encode("Teh", dropout_prob=0.5)
            self.assertTrue(any(ByteFallbackEngine.is_byte_token(t) for t in tokens))
            self.assertEqual(tokenizer.decode_tokens(tokens), "Teh")


if __name__ == "__main__":
    unittest.main()
