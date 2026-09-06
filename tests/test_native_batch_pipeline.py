"""
Unit and Integration Tests for Native Rust Batch Encoding Pipeline (Issue #42).

Tests exact parity between native fused batch execution and Python reference paths,
multilingual coverage, BatchCollator integration, and fallback behaviors.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from uniqtoken import BatchCollator, CustomTokenizer


class NativeBatchPipelineTests(unittest.TestCase):
    tokenizer: CustomTokenizer
    collator: BatchCollator
    test_sentences: list[str]

    @classmethod
    def setUpClass(cls) -> None:
        corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "High performance native Rust batch tokenization eliminates FFI overhead.",
            "def compute_gradients(weights: list[float], loss: float) -> list[float]:",
            "x = 0xDEADBEEF + 0b101010 * 123456",
            "कृत्रिम बुद्धिमत्ता और प्राकृतिक भाषा प्रसंस्करण",
            "యూనిక్‌టోకెన్ అధిక పనితీరు టోకనైజర్",
            "自然语言处理与多语言深度学习分词技术",
            "自然言語処理と機械学習モデルの最適化",
            "التعلم العميق ومعالجة اللغات الطبيعية بدقة عالية",
            "Emojis and symbols: 🚀 🌟 🤖 ✨ 💯 🌍",
        ]
        cls.tokenizer = CustomTokenizer.train_from_corpus(
            corpus=corpus * 20,
            target_vocab_size=500,
            ranking_strategy="char_savings",
            verbose=False,
        )
        cls.collator = BatchCollator(cls.tokenizer)
        cls.test_sentences = corpus * 5

    def test_encode_to_ids_batch_parity(self) -> None:
        """Batch encode_to_ids_batch must produce identical IDs to sequential encode_to_ids."""
        sequential_ids = [self.tokenizer.encode_to_ids(text) for text in self.test_sentences]
        batch_ids = self.tokenizer.encode_to_ids_batch(self.test_sentences)
        self.assertEqual(len(sequential_ids), len(batch_ids))
        for idx, (seq_id, b_id) in enumerate(zip(sequential_ids, batch_ids)):
            self.assertEqual(
                seq_id,
                b_id,
                f"Mismatch at sentence {idx}: sequential={seq_id} != batch={b_id}",
            )

    def test_encode_batch_tokens_parity(self) -> None:
        """Batch encode_batch must produce identical token lists to sequential encode."""
        sequential_tokens = [self.tokenizer.encode(text) for text in self.test_sentences]
        batch_tokens = self.tokenizer.encode_batch(self.test_sentences)
        self.assertEqual(len(sequential_tokens), len(batch_tokens))
        for idx, (seq_tok, b_tok) in enumerate(zip(sequential_tokens, batch_tokens)):
            self.assertEqual(
                seq_tok,
                b_tok,
                f"Mismatch at sentence {idx}: sequential={seq_tok} != batch={b_tok}",
            )

    def test_batch_sizes_scaling(self) -> None:
        """Batch encode works reliably across single-item, boundary, and large batch sizes."""
        for size in (1, 15, 32, 64, 128, 256):
            subset = self.test_sentences[:size]
            expected = [self.tokenizer.encode_to_ids(t) for t in subset]
            actual = self.tokenizer.encode_to_ids_batch(subset)
            self.assertEqual(actual, expected, f"Failed for batch size {size}")

    def test_empty_batch_handling(self) -> None:
        """Empty texts sequence returns empty list."""
        self.assertEqual(self.tokenizer.encode_to_ids_batch([]), [])
        self.assertEqual(self.tokenizer.encode_batch([]), [])

    def test_special_tokens_fallback_and_handling(self) -> None:
        """Texts with special tokens preserve correct tokenization."""
        special_texts = [
            "hello <|unk|> world",
            "simple text without specials",
            "<|pad|> test <|eos|>",
        ]
        expected = [self.tokenizer.encode_to_ids(t, allowed_special="all") for t in special_texts]
        actual = self.tokenizer.encode_to_ids_batch(special_texts, allowed_special="all")
        self.assertEqual(actual, expected)

    def test_batch_collator_integration(self) -> None:
        """BatchCollator properly uses batch fast-path and produces aligned tensors."""
        enc = self.collator.batch_encode(
            self.test_sentences[:10],
            max_length=64,
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        self.assertEqual(len(enc.input_ids), 10)
        self.assertEqual(len(enc.attention_mask), 10)
        self.assertEqual(len(enc.tokens), 10)
        for row in enc.input_ids:
            self.assertEqual(len(row), 64)
        for mask in enc.attention_mask:
            self.assertEqual(len(mask), 64)
            self.assertTrue(all(v in (0, 1) for v in mask))

    def test_mock_native_ids_batch_dispatch(self) -> None:
        """Mock test verifying _encode_ids_native_batch is called directly in encode_to_ids_batch."""
        mock_core = MagicMock()
        mock_core.rust_encode_text_native_ids_batch.return_value = [[1, 2, 3], [4, 5]]
        with patch("uniqtoken.tokenizer._native_core", mock_core):
            with patch.object(self.tokenizer, "_native_pipeline_kwargs", return_value={"space_char": " "}):
                with patch.object(self.tokenizer.model, "_get_rust_trie", return_value=MagicMock()):
                    res = self.tokenizer.encode_to_ids_batch(["text1", "text2"])
                    self.assertEqual(res, [[1, 2, 3], [4, 5]])
                    mock_core.rust_encode_text_native_ids_batch.assert_called_once()

    def test_mock_native_ids_single_dispatch(self) -> None:
        """Mock test verifying single encode_to_ids calls rust_encode_text_native_ids."""
        mock_core = MagicMock()
        mock_core.rust_encode_text_native_ids.return_value = [10, 20, 30]
        with patch("uniqtoken.tokenizer._native_core", mock_core):
            with patch.object(self.tokenizer, "_native_pipeline_kwargs", return_value={"space_char": " "}):
                with patch.object(self.tokenizer.model, "_get_rust_trie", return_value=MagicMock()):
                    res = self.tokenizer.encode_to_ids("hello world")
                    self.assertEqual(res, [10, 20, 30])
                    mock_core.rust_encode_text_native_ids.assert_called_once()


if __name__ == "__main__":
    unittest.main()
