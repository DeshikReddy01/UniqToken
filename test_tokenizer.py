import unittest
from math import log
from tempfile import TemporaryDirectory

from byte_codec import ByteFallbackEngine
from batch_collator import BatchCollator
from pre_tokenizer import Normalizer, RegexPreTokenizer
from tokenizer import CustomTokenizer
from unigram_trainer import UnigramModel
from unigram_lattice import UnigramLattice
from unigram_trainer import UnigramTrainer
from vocab_adapter import VocabularyAdapter


class NormalizerTests(unittest.TestCase):
    def test_nfkc_composes_across_codepoints_and_preserves_raw_span(self):
        raw = "A\u030A"
        normalized, alignment = Normalizer().normalize_with_alignment(raw)

        self.assertEqual(normalized, "\u00C5")
        tokens = RegexPreTokenizer().pre_tokenize_with_offsets(normalized, alignment)
        self.assertEqual(tokens[0].raw_span, (0, 2))

    def test_whitespace_options_are_applied(self):
        normalized, alignment = Normalizer(
            collapse_whitespaces=True,
            strip_whitespace=True,
        ).normalize_with_alignment("  a\t  b  ")

        self.assertEqual(normalized, "a\u2581b")
        self.assertEqual(alignment[1], (3, 6))

    def test_rejects_misaligned_offset_map(self):
        with self.assertRaises(ValueError):
            RegexPreTokenizer().pre_tokenize_with_offsets("abc", [(0, 1)])

    def test_escapes_literal_metaspace_and_escape_prefix(self):
        raw = "x\u2581y\uE000z"
        normalizer = Normalizer()
        normalized, alignment = normalizer.normalize_with_alignment(raw)

        self.assertEqual(normalized, "x\uE000\uE001y\uE000\uE000z")
        self.assertEqual(normalizer.restore_escaped_metaspace(normalized), raw)
        self.assertEqual(alignment[1:3], [(1, 2), (1, 2)])


class ByteFallbackTests(unittest.TestCase):
    def test_byte_decoding_preserves_literal_metaspace(self):
        tokens = ByteFallbackEngine.char_to_byte_tokens("\u2581")
        self.assertEqual(ByteFallbackEngine.decode_tokens(tokens), "\u2581")

    def test_invalid_byte_sequence_is_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            ByteFallbackEngine.decode_tokens(["<0xFF>"])

    def test_subwords_still_decode_metaspace(self):
        self.assertEqual(ByteFallbackEngine.decode_tokens(["hello\u2581world"]), "hello world")


class CustomTokenizerTests(unittest.TestCase):
    def setUp(self):
        vocab = {"tok": log(0.5), "en": log(0.3), "ize": log(0.2)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        self.model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=[],
            max_subword_len=3,
            byte_fallback=False,
        )
        self.tokenizer = CustomTokenizer(
            normalizer=Normalizer(normalize_unicode=False),
            pre_tokenizer=RegexPreTokenizer(),
            model=self.model,
        )

    def test_subword_offsets_are_exact(self):
        tokens = self.tokenizer.encode_with_offsets("tokenize")
        self.assertEqual(
            [(token.text, token.raw_span) for token in tokens],
            [("tok", (0, 3)), ("en", (3, 5)), ("ize", (5, 8))],
        )

    def test_save_load_preserves_lattice_settings(self):
        custom_tokenizer = CustomTokenizer(
            normalizer=Normalizer(
                lowercase=True,
                normalize_unicode=False,
                collapse_whitespaces=True,
            ),
            pre_tokenizer=RegexPreTokenizer(
                split_digits=True,
                split_punctuation=False,
                keep_special_tokens=False,
                special_token_pattern=r"\[\[[^\]]+\]\]",
            ),
            model=self.model,
        )
        with TemporaryDirectory() as directory:
            custom_tokenizer.save(directory)
            loaded = CustomTokenizer.load(directory)

        self.assertEqual(loaded.model.max_subword_len, 3)
        self.assertFalse(loaded.model.byte_fallback)
        self.assertTrue(loaded.normalizer.lowercase)
        self.assertFalse(loaded.normalizer.normalize_unicode)
        self.assertTrue(loaded.normalizer.collapse_whitespaces)
        self.assertTrue(loaded.pre_tokenizer.split_digits)
        self.assertFalse(loaded.pre_tokenizer.split_punctuation)
        self.assertFalse(loaded.pre_tokenizer.keep_special_tokens)
        self.assertEqual(loaded.pre_tokenizer.special_token_pattern, r"\[\[[^\]]+\]\]")

    def test_vocabulary_adapter_preserves_model_settings_and_ids(self):
        updated = VocabularyAdapter.expand_vocabulary(
            self.tokenizer,
            ["tokenized tokenized"],
            num_new_tokens=1,
            min_frequency=1,
            max_ngram_length=8,
            verbose=False,
        )

        self.assertEqual(updated.model.max_subword_len, 3)
        self.assertFalse(updated.model.byte_fallback)
        self.assertEqual(updated.model.token_to_id["tok"], self.model.token_to_id["tok"])
        self.assertEqual(updated.model.token_to_id["en"], self.model.token_to_id["en"])


class LatticeTests(unittest.TestCase):
    def test_rejects_invalid_sampling_temperature(self):
        lattice = UnigramLattice("ab", {"a": log(0.5), "b": log(0.5)}, byte_fallback=False)
        with self.assertRaises(ValueError):
            lattice.sample(alpha=0)
        with self.assertRaises(ValueError):
            lattice.sample(alpha=float("nan"))

    def test_forward_backward_rejects_disconnected_lattice(self):
        lattice = UnigramLattice("z", {}, byte_fallback=False)
        with self.assertRaises(RuntimeError):
            lattice.forward_backward()

    def test_rejects_invalid_lattice_length(self):
        with self.assertRaises(ValueError):
            UnigramLattice("a", {"a": log(1.0)}, max_subword_len=0)


class TrainerValidationTests(unittest.TestCase):
    def test_rejects_invalid_training_configuration(self):
        with self.assertRaises(ValueError):
            UnigramTrainer(prune_rate=0)
        with self.assertRaises(ValueError):
            UnigramTrainer(em_sub_iterations=0)
        with self.assertRaises(ValueError):
            UnigramTrainer(max_ngram_length=0)


class BatchCollatorTests(unittest.TestCase):
    def setUp(self):
        vocab = {
            "a": log(0.2),
            "<|pad|>": log(0.2),
            "<|bos|>": log(0.2),
            "<|eos|>": log(0.2),
            "<|unk|>": log(0.2),
        }
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
            byte_fallback=False,
        )
        self.collator = BatchCollator(
            CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        )

    def test_padding_keeps_tokens_aligned_with_ids(self):
        batch = self.collator.batch_encode(["a", "aa"], max_length=5, truncation=True)
        self.assertEqual(batch.tokens[0], ["<|bos|>", "a", "<|eos|>", "<|pad|>", "<|pad|>"])
        self.assertEqual([len(row) for row in batch.input_ids], [5, 5])
        self.assertEqual([len(row) for row in batch.tokens], [5, 5])
        self.assertEqual(batch.attention_mask[0], [1, 1, 1, 0, 0])

    def test_rejects_overlong_sequence_without_truncation(self):
        with self.assertRaises(ValueError):
            self.collator.batch_encode(["aa"], max_length=2, truncation=False)


if __name__ == "__main__":
    unittest.main()
