"""
Unit and Integration Tests for UniqToken CLI.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import uniqtoken.cli as cli


class CLITests(unittest.TestCase):
    def test_cli_train_encode_decode_roundtrip(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "model"
            ids_file = tmp / "ids.json"
            decoded_file = tmp / "decoded.txt"

            corpus_content = "\n".join(
                [
                    "the quick brown fox jumps over the lazy dog",
                    "machine learning and natural language processing",
                    "custom tokenizer training verification",
                ]
            )
            corpus_file.write_text(corpus_content, encoding="utf-8")

            # 1. Train
            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--ranking-strategy",
                    "pmi",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

            # 2. Encode to IDs (JSON)
            ret = cli.main(
                [
                    "encode",
                    "--model",
                    str(model_dir),
                    "--input",
                    "the quick brown fox",
                    "--to-ids",
                    "--json",
                    "--out",
                    str(ids_file),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue(ids_file.exists())
            ids_data = json.loads(ids_file.read_text(encoding="utf-8"))
            self.assertIsInstance(ids_data, list)
            self.assertGreater(len(ids_data), 0)

            # 3. Decode
            ret = cli.main(
                [
                    "decode",
                    "--model",
                    str(model_dir),
                    "--input",
                    str(ids_file),
                    "--out",
                    str(decoded_file),
                ]
            )
            self.assertEqual(ret, 0)
            decoded_text = decoded_file.read_text(encoding="utf-8")
            self.assertEqual(decoded_text, "the quick brown fox")

    def test_cli_decode_empty_ids_writes_empty_file(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "model"
            decoded_file = tmp / "decoded.txt"
            corpus_file.write_text("test corpus", encoding="utf-8")
            self.assertEqual(
                cli.main(
                    [
                        "train",
                        "--corpus",
                        str(corpus_file),
                        "--vocab-size",
                        "320",
                        "--out",
                        str(model_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli.main(
                    [
                        "decode",
                        "--model",
                        str(model_dir),
                        "--input",
                        "[]",
                        "--out",
                        str(decoded_file),
                    ]
                ),
                0,
            )
            self.assertEqual(decoded_file.read_text(encoding="utf-8"), "")

    def test_cli_encode_with_metrics_and_offsets(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "model"
            metrics_file = tmp / "metrics.json"
            offsets_file = tmp / "offsets.json"

            corpus_file.write_text("the quick brown fox jumps over the lazy dog\n", encoding="utf-8")

            cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--out",
                    str(model_dir),
                ]
            )

            # Encode with metrics
            ret = cli.main(
                [
                    "encode",
                    "--model",
                    str(model_dir),
                    "--input",
                    "the quick brown fox",
                    "--with-metrics",
                    "--out",
                    str(metrics_file),
                ]
            )
            self.assertEqual(ret, 0)
            report = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.assertIn("num_tokens", report)
            self.assertIn("bytes_per_token", report)
            self.assertIn("byte_fallback_rate", report)

            # Encode with offsets
            ret = cli.main(
                [
                    "encode",
                    "--model",
                    str(model_dir),
                    "--input",
                    "the quick brown fox",
                    "--with-offsets",
                    "--out",
                    str(offsets_file),
                ]
            )
            self.assertEqual(ret, 0)
            offsets = json.loads(offsets_file.read_text(encoding="utf-8"))
            self.assertIsInstance(offsets, list)
            self.assertIn("start", offsets[0])
            self.assertIn("end", offsets[0])

    def test_cli_superbpe_train(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "sbp_model"

            corpus_file.write_text("hello world from caliper superbpe compression\n", encoding="utf-8")

            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--superbpe-merges",
                    "5",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

    def test_cli_train_defaults_ranking_strategy_to_char_savings(self):
        parser = cli.build_parser()
        args = parser.parse_args(["train", "--corpus", "dummy.txt", "--out", "dummy_dir"])
        self.assertEqual(args.ranking_strategy, "char_savings")

    def test_cli_train_with_preset_and_digit_chunking(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "code_corpus.txt"
            model_dir = tmp / "code_model"

            corpus_file.write_text("x = 0xDEADBEEF + 123456\ny = 0b1010\n", encoding="utf-8")

            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--preset",
                    "code",
                    "--digit-chunk-size",
                    "3",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

    def test_cli_downstream_eval_smoke(self):
        ret = cli.main(["eval-downstream", "--smoke-test"])
        self.assertEqual(ret, 0)

    def test_cli_train_sample_corpus_with_progress_and_no_progress(self):
        sample_path = Path(__file__).parent / "sample.txt"
        self.assertTrue(sample_path.exists(), "tests/sample.txt fixture must exist")

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "sample_model"
            # 1. Train with progress on sample corpus
            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(sample_path),
                    "--vocab-size",
                    "500",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

            # 2. Train with --no-progress flag
            no_prog_dir = Path(tmp_dir) / "no_prog_model"
            ret_no_prog = cli.main(
                [
                    "train",
                    "--corpus",
                    str(sample_path),
                    "--vocab-size",
                    "500",
                    "--no-progress",
                    "--out",
                    str(no_prog_dir),
                ]
            )
            self.assertEqual(ret_no_prog, 0)
            self.assertTrue((no_prog_dir / "tokenizer.json").exists())


if __name__ == "__main__":
    unittest.main()
