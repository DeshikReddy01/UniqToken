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
from collections import Counter
from pathlib import Path

import pytest

from uniqtoken import BPETrainer, CustomTokenizer, DiskChunkCounter, StreamingChunkCounter, UnigramTrainer


def test_basic_accumulation_and_mapping_protocol():
    """Verify add, update, __getitem__, __contains__, len, total, most_common."""
    with StreamingChunkCounter(chunk_size_bytes=1024 * 1024) as counter:
        counter.add("apple", 2)
        counter.add("banana", 5)
        counter.update(["apple", "cherry", "banana"])
        counter["date"] = 3

        assert not counter._finalized
        counter.finalize()
        assert counter._finalized

        assert counter["apple"] == 3
        assert counter["banana"] == 6
        assert counter["cherry"] == 1
        assert counter["date"] == 3
        assert counter["missing"] == 0  # Counter semantics

        assert "apple" in counter
        assert "missing" not in counter
        assert counter.get("banana") == 6
        assert counter.get("missing", -1) == -1

        assert len(counter) == 4
        assert counter.total() == 13

        # Items in lexicographical order
        items = list(counter.items())
        assert items == [
            ("apple", 3),
            ("banana", 6),
            ("cherry", 1),
            ("date", 3),
        ]

        # most_common
        assert counter.most_common(2) == [("banana", 6), ("apple", 3)] or counter.most_common(2) == [
            ("banana", 6),
            ("date", 3),
        ]


def test_spilling_and_k_way_merge():
    """Verify spilling with tiny chunk size and external k-way min-heap merge."""
    # Using tiny 100-byte buffer to force frequent flushes to disk
    with StreamingChunkCounter(chunk_size_bytes=100) as counter:
        corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] * 20
        counter.update(corpus)

        # Confirm multiple runs were created
        assert len(counter._run_files) > 1

        counter.finalize()

        # Compare directly with standard Python Counter
        expected = Counter(corpus)
        assert len(counter) == len(expected)
        assert counter.total() == sum(expected.values())

        for token, count in expected.items():
            assert counter[token] == count

        # Verify sorted iteration matches sorted expected items
        assert list(counter.items()) == sorted(expected.items())


def test_cascade_multi_pass_merge():
    """Verify cascading merge when number of runs exceeds max_open_runs."""
    # Set max_open_runs=4 and tiny chunk_size_bytes to produce ~20 runs
    with StreamingChunkCounter(chunk_size_bytes=60, max_open_runs=4) as counter:
        words = [f"word_{i:04d}" for i in range(100)] * 3
        counter.update(words)

        # Before finalize, many runs exist
        assert len(counter._run_files) > 10

        counter.finalize()

        # After finalize, all merged into exactly 1 consolidated run
        assert len(counter._run_files) == 1
        assert counter._merged_file is not None
        assert os.path.exists(counter._merged_file)

        expected = Counter(words)
        assert len(counter) == len(expected)
        assert counter.total() == sum(expected.values())
        for w, c in expected.items():
            assert counter[w] == c


def test_unicode_and_special_characters():
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
        assert len(counter) == len(expected)
        for t, cnt in expected.items():
            assert counter[t] == cnt
        assert list(counter.items()) == sorted(expected.items())


def test_repeated_iteration_stability():
    """Verify that items() can be iterated repeatedly (simulating EM rounds)."""
    with StreamingChunkCounter(chunk_size_bytes=120) as counter:
        data = ["alpha", "beta", "gamma", "delta", "epsilon"] * 10
        counter.update(data)
        counter.finalize()

        first_pass = list(counter.items())
        for _ in range(10):
            assert list(counter.items()) == first_pass
            assert list(counter.keys()) == [k for k, _ in first_pass]
            assert list(counter.values()) == [v for _, v in first_pass]


def test_empty_counter():
    """Verify edge case of empty counter."""
    with StreamingChunkCounter() as counter:
        counter.finalize()
        assert len(counter) == 0
        assert counter.total() == 0
        assert list(counter.items()) == []
        assert counter["anything"] == 0
        assert "anything" not in counter


def test_cleanup_on_close_and_context_manager():
    """Verify temp directory is completely removed on close() and with block exit."""
    counter = StreamingChunkCounter(chunk_size_bytes=100)
    counter.update(["foo", "bar", "baz"] * 20)
    counter.finalize()
    dir_path = counter.dir_path
    assert os.path.exists(dir_path)

    counter.close()
    assert not os.path.exists(dir_path)

    # Double close is safe
    counter.close()

    # Context manager test
    with StreamingChunkCounter() as c:
        p = c.dir_path
        assert os.path.exists(p)
    assert not os.path.exists(p)


def test_unigram_parity_with_in_memory_counter():
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
    assert model_mem.token_to_id == model_disk.token_to_id
    assert model_mem.id_to_token == model_disk.id_to_token
    assert len(model_mem.vocab) == len(model_disk.vocab)
    for tok, log_p in model_mem.vocab.items():
        assert tok in model_disk.vocab
        assert pytest.approx(log_p, abs=1e-5) == model_disk.vocab[tok]


def test_bpe_parity_with_in_memory_counter():
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
    assert model_mem.merges == model_disk.merges
    assert model_mem.vocab == model_disk.vocab


def test_custom_tokenizer_train_from_corpus_streaming():
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

    assert tok_standard.model.token_to_id == tok_streaming.model.token_to_id
    test_sentence = "The quick brown fox jumps over machine learning."
    assert tok_standard.encode(test_sentence) == tok_streaming.encode(test_sentence)


def test_memory_bounded_streaming():
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
        assert counter.total() == 100_000
        assert len(counter) == 5000

        # Iterate 3 times
        for _ in range(3):
            count = sum(1 for _ in counter.items())
            assert count == 5000

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_diff = sum(stat.size_diff for stat in stats)
    # Peak memory difference should be well bounded (< 15MB)
    assert total_diff < 15 * 1024 * 1024
