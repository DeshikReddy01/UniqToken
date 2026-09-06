"""
Unit and integration tests for StreamingChunkCounter / DiskChunkCounter.
Validates binary serialization, k-way min-heap tournament merge, cascade multi-pass merge,
vocabulary parity with in-memory Counter for Unigram and BPE, memory boundedness, and cleanup.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import tracemalloc
import unittest
from collections import Counter
from pathlib import Path

from uniqtoken import BPETrainer, CustomTokenizer, DiskChunkCounter, StreamingChunkCounter, UnigramTrainer


class StreamingCounterTests(unittest.TestCase):
    """Test suite for disk-backed out-of-core chunk counter."""

    def test_basic_accumulation_and_mapping_protocol(self):
        """Verify add, update, __getitem__, __contains__, len, total, most_common."""
        with StreamingChunkCounter(chunk_size_bytes=1024 * 1024) as counter:
            counter.add("apple", 2)
            counter.add("banana", 5)
            counter.update(["apple", "cherry", "banana"])
            counter["date"] = 3

            self.assertFalse(counter._finalized)
            counter.finalize()
            self.assertTrue(counter._finalized)

            self.assertEqual(counter["apple"], 3)
            self.assertEqual(counter["banana"], 6)
            self.assertEqual(counter["cherry"], 1)
            self.assertEqual(counter["date"], 3)
            self.assertEqual(counter["missing"], 0)  # Counter semantics

            self.assertIn("apple", counter)
            self.assertNotIn("missing", counter)
            self.assertEqual(counter.get("banana"), 6)
            self.assertEqual(counter.get("missing", -1), -1)

            self.assertEqual(len(counter), 4)
            self.assertEqual(counter.total(), 13)

            # Items in lexicographical order
            items = list(counter.items())
            self.assertEqual(
                items,
                [
                    ("apple", 3),
                    ("banana", 6),
                    ("cherry", 1),
                    ("date", 3),
                ],
            )

            # most_common
            top2 = counter.most_common(2)
            self.assertTrue(top2 == [("banana", 6), ("apple", 3)] or top2 == [("banana", 6), ("date", 3)])

    def test_spilling_and_k_way_merge(self):
        """Verify spilling with tiny chunk size and external k-way min-heap merge."""
        # Using tiny 100-byte buffer to force frequent flushes to disk
        with StreamingChunkCounter(chunk_size_bytes=100) as counter:
            corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] * 20
            counter.update(corpus)

            # Confirm multiple runs were created
            self.assertGreater(len(counter._run_files), 1)

            counter.finalize()

            # Compare directly with standard Python Counter
            expected = Counter(corpus)
            self.assertEqual(len(counter), len(expected))
            self.assertEqual(counter.total(), sum(expected.values()))

            for token, count in expected.items():
                self.assertEqual(counter[token], count)

            # Verify sorted iteration matches sorted expected items
            self.assertEqual(list(counter.items()), sorted(expected.items()))

    def test_cascade_multi_pass_merge(self):
        """Verify cascading merge when number of runs exceeds max_open_runs."""
        # Set max_open_runs=4 and tiny chunk_size_bytes to produce ~20 runs
        with StreamingChunkCounter(chunk_size_bytes=60, max_open_runs=4) as counter:
            words = [f"word_{i:04d}" for i in range(100)] * 3
            counter.update(words)

            # Before finalize, many runs exist
            self.assertGreater(len(counter._run_files), 10)

            counter.finalize()

            # After finalize, all merged into exactly 1 consolidated run
            self.assertEqual(len(counter._run_files), 1)
            self.assertIsNotNone(counter._merged_file)
            assert counter._merged_file is not None
            self.assertTrue(os.path.exists(counter._merged_file))

            expected = Counter(words)
            self.assertEqual(len(counter), len(expected))
            self.assertEqual(counter.total(), sum(expected.values()))
            for w, c in expected.items():
                self.assertEqual(counter[w], c)

    def test_unicode_and_special_characters(self):
        """Verify non-ASCII, multi-byte UTF-8, emoji, spaces, and punctuation."""
        tokens = [
            "hello",
            "こんにちは",
            "مرحبا",
            "🎉🚀✨",
            "слово",
            "español",
            "\u2581metaspace",
            "<|special|>",
            "line\nbreak",
        ] * 5

        with DiskChunkCounter(chunk_size_bytes=80) as counter:
            counter.update(tokens)
            counter.finalize()

            expected = Counter(tokens)
            self.assertEqual(len(counter), len(expected))
            for t, cnt in expected.items():
                self.assertEqual(counter[t], cnt)
            self.assertEqual(list(counter.items()), sorted(expected.items()))

    def test_repeated_iteration_stability(self):
        """Verify that items() can be iterated repeatedly (simulating EM rounds)."""
        with StreamingChunkCounter(chunk_size_bytes=120) as counter:
            data = ["alpha", "beta", "gamma", "delta", "epsilon"] * 10
            counter.update(data)
            counter.finalize()

            first_pass = list(counter.items())
            for _ in range(10):
                self.assertEqual(list(counter.items()), first_pass)
                self.assertEqual(list(counter.keys()), [k for k, _ in first_pass])
                self.assertEqual(list(counter.values()), [v for _, v in first_pass])

    def test_empty_counter(self):
        """Verify edge case of empty counter."""
        with StreamingChunkCounter() as counter:
            counter.finalize()
            self.assertEqual(len(counter), 0)
            self.assertEqual(counter.total(), 0)
            self.assertEqual(list(counter.items()), [])
            self.assertEqual(counter["anything"], 0)
            self.assertNotIn("anything", counter)

    def test_cleanup_on_close_and_context_manager(self):
        """Verify temp directory is completely removed on close() and with block exit."""
        counter = StreamingChunkCounter(chunk_size_bytes=100)
        counter.update(["foo", "bar", "baz"] * 20)
        counter.finalize()
        dir_path = counter.dir_path
        self.assertTrue(os.path.exists(dir_path))

        counter.close()
        self.assertFalse(os.path.exists(dir_path))

        # Double close is safe
        counter.close()

        # Context manager test
        with StreamingChunkCounter() as c:
            p = c.dir_path
            self.assertTrue(os.path.exists(p))
        self.assertFalse(os.path.exists(p))

    def test_unigram_parity_with_in_memory_counter(self):
        """100% vocabulary parity between in-memory Counter and StreamingChunkCounter in UnigramTrainer."""
        corpus = [
            "UniqToken provides high performance tokenization for large scale language models.",
            "Deterministic subword vocabularies ensure reproducibility across training runs.",
            "External disk-backed chunk counting enables out-of-core scaling to terabyte datasets.",
            "Multilingual empirical benchmarks evaluate compression efficiency and byte fallback rate.",
        ] * 5

        # 1. Train with standard in-memory Counter
        trainer_mem = UnigramTrainer(
            target_vocab_size=320,
            seed_multiplier=2.0,
            min_frequency=1,
            streaming=False,
            show_progress=False,
        )
        model_mem = trainer_mem.train(corpus, verbose=False)

        # 2. Train with StreamingChunkCounter (tiny chunk_size_bytes to force multiple disk runs)
        trainer_disk = UnigramTrainer(
            target_vocab_size=320,
            seed_multiplier=2.0,
            min_frequency=1,
            streaming=True,
            chunk_size_bytes=200,  # Spill frequently to disk
            show_progress=False,
        )
        model_disk = trainer_disk.train(corpus, verbose=False)

        # 3. Assert 100% identical vocabulary and token IDs
        self.assertEqual(model_mem.token_to_id, model_disk.token_to_id)
        self.assertEqual(model_mem.id_to_token, model_disk.id_to_token)
        self.assertEqual(len(model_mem.vocab), len(model_disk.vocab))
        for tok, log_p in model_mem.vocab.items():
            self.assertIn(tok, model_disk.vocab)
            self.assertAlmostEqual(log_p, model_disk.vocab[tok], places=5)

    def test_bpe_parity_with_in_memory_counter(self):
        """100% vocabulary and merge parity between in-memory Counter and StreamingChunkCounter in BPETrainer."""
        words = ["low", "lower", "newest", "widest", "high", "higher", "highest"] * 10

        # 1. In-memory Counter
        bpe_mem = BPETrainer(target_vocab_size=300, num_merges=20)
        model_mem = bpe_mem.train(words, verbose=False)

        # 2. StreamingChunkCounter
        with StreamingChunkCounter(chunk_size_bytes=100) as counter:
            counter.update(words)
            bpe_disk = BPETrainer(target_vocab_size=300, num_merges=20)
            model_disk = bpe_disk.train(counter, verbose=False)

        # 3. Assert 100% identical merges and vocab
        self.assertEqual(model_mem.merges, model_disk.merges)
        self.assertEqual(model_mem.vocab, model_disk.vocab)

    def test_custom_tokenizer_train_from_corpus_streaming(self):
        """Verify CustomTokenizer.train_from_corpus with streaming=True."""
        corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require deterministic tokenization.",
        ] * 8

        tok_standard = CustomTokenizer.train_from_corpus(
            corpus=corpus,
            target_vocab_size=320,
            streaming=False,
            verbose=False,
        )

        tok_streaming = CustomTokenizer.train_from_corpus(
            corpus=corpus,
            target_vocab_size=320,
            streaming=True,
            chunk_size_bytes=250,
            verbose=False,
        )

        self.assertEqual(tok_standard.model.token_to_id, tok_streaming.model.token_to_id)
        test_sentence = "The quick brown fox jumps over machine learning."
        self.assertEqual(tok_standard.encode(test_sentence), tok_streaming.encode(test_sentence))

    def test_memory_bounded_streaming(self):
        """Verify that StreamingChunkCounter keeps peak RAM bounded under repeated flushes."""
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        with StreamingChunkCounter(chunk_size_bytes=10 * 1024) as counter:
            # Stream 100,000 items (which would be ~5MB+ in RAM as raw strings)
            def _generator():
                for i in range(100_000):
                    yield f"synthetic_chunk_{i % 5000:05d}"

            counter.update(_generator())
            counter.finalize()

            # Check total and unique
            self.assertEqual(counter.total(), 100_000)
            self.assertEqual(len(counter), 5000)

            # Iterate 3 times
            for _ in range(3):
                count = sum(1 for _ in counter.items())
                self.assertEqual(count, 5000)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_diff = sum(stat.size_diff for stat in stats)
        # Peak memory difference should be well bounded (< 15MB)
        self.assertLess(total_diff, 15 * 1024 * 1024)

    def test_prefinalize_reads_observe_totals(self):
        """Reads must never return buffer-only partials after runs spill.

        Regression test: __getitem__/get/__contains__ used to read only the
        in-memory buffer when unfinalized, silently undercounting spilled
        keys. Reads now finalize first, matching __len__/iteration.
        """
        with StreamingChunkCounter(chunk_size_bytes=1024) as counter:
            for i in range(3000):
                counter.add(f"chunk_{i % 100:03d}")
            # Force spills so most counts live on disk, not just the buffer.
            self.assertGreaterEqual(len(counter._run_files), 1)
            counter.update(["chunk_000"] * 5)
            expected = 30 + 5  # 3000/100 base occurrences plus the top-up
            # No explicit finalize() call: reads must observe totals anyway.
            self.assertEqual(counter["chunk_000"], expected)
            self.assertEqual(counter.get("chunk_000"), expected)
            self.assertIn("chunk_000", counter)
            self.assertNotIn("no_such_chunk", counter)
            self.assertTrue(counter._finalized)

    def test_use_after_close_raises(self):
        """Operations after close() fail loudly instead of crashing on missing files."""
        counter = StreamingChunkCounter(chunk_size_bytes=1024)
        counter.add("hello")
        counter.close()
        for op in (
            lambda: counter.add("world"),
            lambda: counter.update(["world"]),
            lambda: counter.__setitem__("world", 1),
            lambda: counter.finalize(),
            lambda: counter["hello"],
            lambda: counter[None],  # type: ignore[index]
            lambda: counter.get("hello"),
            lambda: counter.get(None),  # type: ignore[arg-type]
            lambda: "hello" in counter,
            lambda: None in counter,
            lambda: len(counter),
            lambda: list(counter),
        ):
            with self.assertRaises(RuntimeError):
                op()
        counter.close()  # double close stays safe


if __name__ == "__main__":
    unittest.main()
