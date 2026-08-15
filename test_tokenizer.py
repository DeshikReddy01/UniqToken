import unittest

from byte_codec import ByteFallbackEngine
from pre_tokenizer import Normalizer, RegexPreTokenizer


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


class ByteFallbackTests(unittest.TestCase):
    def test_byte_decoding_preserves_literal_metaspace(self):
        tokens = ByteFallbackEngine.char_to_byte_tokens("\u2581")
        self.assertEqual(ByteFallbackEngine.decode_tokens(tokens), "\u2581")

    def test_invalid_byte_sequence_is_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            ByteFallbackEngine.decode_tokens(["<0xFF>"])

    def test_subwords_still_decode_metaspace(self):
        self.assertEqual(ByteFallbackEngine.decode_tokens(["hello\u2581world"]), "hello world")


if __name__ == "__main__":
    unittest.main()
