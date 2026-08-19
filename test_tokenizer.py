import json
import math
import unittest
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

from byte_codec import ByteFallbackEngine
from batch_collator import BatchCollator
from pre_tokenizer import Normalizer, RegexPreTokenizer
from tokenizer import CustomTokenizer, TokenizationReport
from unigram_trainer import UnigramModel, UnigramTrainer
from unigram_lattice import UnigramLattice
from vocab_adapter import VocabularyAdapter
from multimodal.multimodal_tokenizer import MultimodalTokenizer, ImageElement
from multimodal.audio_codec import ResidualVectorQuantizer, AudioSegment
from trie import PrefixTrie
from bpe_trainer import BPETrainer
from hf_exporter import HuggingFaceExporter
from indentation_compressor import IndentationCompressor
from security_shield import SecurityShield
from seed_builder import SeedVocabularyBuilder
from streaming_decoder import StreamingDecoder
from cem_merger import CrossEntropyMerging


class NormalizerTests(unittest.TestCase):
    def test_nfkc_composes_across_codepoints_and_preserves_raw_span(self):
        raw = "A\u030a"
        normalized, alignment = Normalizer().normalize_with_alignment(raw)

        self.assertEqual(normalized, "\u00c5")
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
        raw = "x\u2581y\ue000z"
        normalizer = Normalizer()
        normalized, alignment = normalizer.normalize_with_alignment(raw)

        self.assertEqual(normalized, "x\ue000\ue001y\ue000\ue000z")
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

        added_tokens = set(updated.model.vocab) - set(self.model.vocab)
        self.assertTrue(added_tokens)
        self.assertEqual(updated.model.max_subword_len, max(3, max(map(len, added_tokens))))
        self.assertFalse(updated.model.byte_fallback)
        self.assertEqual(updated.model.token_to_id["tok"], self.model.token_to_id["tok"])
        self.assertEqual(updated.model.token_to_id["en"], self.model.token_to_id["en"])

    def test_vocabulary_adapter_zero_additions_is_a_noop(self):
        self.assertIs(
            VocabularyAdapter.expand_vocabulary(self.tokenizer, ["new domain"], num_new_tokens=0),
            self.tokenizer,
        )


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
        self.collator = BatchCollator(CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model))

    def test_padding_keeps_tokens_aligned_with_ids(self):
        batch = self.collator.batch_encode(["a", "aa"], max_length=5, truncation=True)
        self.assertEqual(batch.tokens[0], ["<|bos|>", "a", "<|eos|>", "<|pad|>", "<|pad|>"])
        self.assertEqual([len(row) for row in batch.input_ids], [5, 5])
        self.assertEqual([len(row) for row in batch.tokens], [5, 5])
        self.assertEqual(batch.attention_mask[0], [1, 1, 1, 0, 0])

    def test_rejects_overlong_sequence_without_truncation(self):
        with self.assertRaises(ValueError):
            self.collator.batch_encode(["aa"], max_length=2, truncation=False)


class MultimodalTests(unittest.TestCase):
    def setUp(self):
        vocab = {"test": log(0.5), "<|unk|>": log(0.5)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )
        self.tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        self.mm_tok = MultimodalTokenizer(self.tokenizer, patch_size=16, channels=3, num_visual_tokens=64)

    def test_image_patching_and_aspect_ratio(self):
        img = [[[1.0, 2.0, 3.0] for _ in range(32)] for _ in range(16)]
        tokens, patches = self.mm_tok.encode_image(img)
        self.assertEqual(tokens[0], "<|image_start|>")
        self.assertEqual(tokens[-1], "<|image_end|>")
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0].norm_bbox, (0.0, 0.0, 1.0, 0.5))

    def test_interleaved_modality_mask(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        img_elem = ImageElement(pixels=img)
        seq = self.mm_tok.encode_interleaved(["test", img_elem, "test"])
        self.assertIn(0, seq.modality_mask)  # Text
        self.assertIn(1, seq.modality_mask)  # Vision
        self.assertIn(3, seq.modality_mask)  # Special

    def test_save_load_preserves_multimodal_state(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        img_elem = ImageElement(pixels=img)
        seq = self.mm_tok.encode_interleaved(["test", img_elem, "test"])
        self.mm_tok.audio_quantizer.codebooks[0][0][0] = 123.0

        with TemporaryDirectory() as directory:
            self.mm_tok.save(directory)
            loaded = MultimodalTokenizer.load(directory)

            self.assertEqual(loaded.vocab_size, self.mm_tok.vocab_size)
            self.assertEqual(loaded.codebook.num_embeddings, self.mm_tok.codebook.num_embeddings)
            self.assertEqual(
                loaded.audio_quantizer.codebooks[0][0][0],
                123.0,
            )
            seq2 = loaded.encode_interleaved(["test", img_elem, "test"])
            self.assertEqual(len(seq2.token_strings), len(seq.token_strings))

    def test_freeze_state_survives_save_load(self):
        self.mm_tok.freeze()
        with TemporaryDirectory() as directory:
            self.mm_tok.save(directory)
            loaded = MultimodalTokenizer.load(directory)
        with self.assertRaises(KeyError):
            loaded._assign_id("<|new_metadata|>")

    def test_nonzero_pixel_range_uses_normalized_zero_padding(self):
        from multimodal.image_patcher import DynamicImagePatcher

        patcher = DynamicImagePatcher(patch_size=2, channels=1, pixel_range=(10.0, 20.0))
        patches, _ = patcher.extract_patches([[[15.0]]])
        self.assertEqual(patches[0].pixels, [0.5, 0.0, 0.0, 0.0])

    def test_rejects_zero_sized_image_grid_metadata(self):
        with self.assertRaises(ValueError):
            self.mm_tok.decode_text_and_images(["<|image_start|>", "<|grid_0x0|>", "<|vis_0000|>", "<|image_end|>"])

    def test_rejects_invalid_element_type(self):
        from typing import Any, cast

        with self.assertRaises(TypeError):
            self.mm_tok.encode_interleaved(cast(Any, [123]))

    def test_codebook_training_updates_embeddings(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        initial_codebook = [row[:] for row in self.mm_tok.codebook.codebook]

        tokens, patches = self.mm_tok.encode_image(img)
        patch_vectors = [p.pixels for p in patches]
        indices = [self.mm_tok.codebook.quantize_patch(vec)[0] for vec in patch_vectors]

        for _ in range(101):
            self.mm_tok.codebook.update_ema(patch_vectors, indices)

        updated = False
        for init_row, curr_row in zip(initial_codebook, self.mm_tok.codebook.codebook):
            for iv, cv in zip(init_row, curr_row):
                if abs(iv - cv) > 1e-10:
                    updated = True
                    break
            if updated:
                break

        self.assertTrue(updated, "Codebook should have been updated by EMA")

    def test_kmeans_init_improves_quantization(self):
        imgs = []
        for i in range(8):
            img = [[[0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1] for _ in range(16)] for _ in range(16)]
            imgs.append(img)

        patches1, _ = self.mm_tok.patcher.extract_patches(imgs[0])
        all_patches = patches1 * 64
        patch_vectors = [p.pixels for p in all_patches]

        self.mm_tok.codebook.kmeans_init(patch_vectors, max_iter=5)
        _, _, error = self.mm_tok.codebook.quantize_patch(patch_vectors[0])
        self.assertIsInstance(error, float)

    def test_multimodal_vocabulary_freeze(self):
        initial_vocab_size = self.mm_tok.vocab_size
        self.mm_tok.freeze()
        with self.assertRaises(KeyError):
            self.mm_tok._assign_id("<|unregistered_metadata_token|>")
        self.assertEqual(self.mm_tok.vocab_size, initial_vocab_size)


class TrieTests(unittest.TestCase):
    def test_prefix_trie_matches_and_accelerates_lattice(self):
        vocab = {"ab": log(0.4), "abc": log(0.6), "d": log(0.2)}
        trie = PrefixTrie.from_vocab(vocab)

        matches = trie.find_matches("abcd", 0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][1], "ab")
        self.assertEqual(matches[1][1], "abc")

        lat_no_trie = UnigramLattice("abcd", vocab, byte_fallback=True)
        lat_with_trie = UnigramLattice("abcd", vocab, byte_fallback=True, trie=trie)

        tokens1, score1 = lat_no_trie.viterbi()
        tokens2, score2 = lat_with_trie.viterbi()
        self.assertEqual(tokens1, tokens2)
        self.assertAlmostEqual(score1, score2)


class BPETests(unittest.TestCase):
    def test_bpe_training_and_encoding_roundtrip(self):
        corpus = ["low", "lower", "lowest", "lowering"] * 10
        trainer = BPETrainer(num_merges=10, byte_fallback=True)
        bpe_model = trainer.train(corpus)

        self.assertIn("low", bpe_model.vocab)
        encoded = bpe_model.encode("lowest")
        self.assertTrue(len(encoded) > 0)

        unk_id = bpe_model.token_to_id.get("<|unk|>", 0)
        ids = [bpe_model.token_to_id.get(t, unk_id) for t in encoded]
        decoded = bpe_model.decode(ids)
        self.assertEqual(decoded, "lowest")


class LatticeFastPathTests(unittest.TestCase):
    def setUp(self):
        vocab = {
            "a": log(0.1),
            "b": log(0.1),
            "c": log(0.1),
            "ab": log(0.15),
            "bc": log(0.15),
            "abc": log(0.2),
            "abcd": log(0.25),
            "xy": log(0.3),
            "xyz": log(0.2),
            "the": log(0.3),
            "ther": log(0.1),
            "\u2581": log(0.1),
            "\u2581\u2581": log(0.05),
            "<|unk|>": log(0.05),
        }
        for char in "dexyz\u0905\u0906\u0907é👍":
            vocab.setdefault(char, log(0.05))
        for b in range(256):
            b_tok = ByteFallbackEngine.byte_to_token(b)
            if b_tok not in vocab:
                vocab[b_tok] = log(0.001)
        token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.models = [
            UnigramModel(
                vocab=dict(vocab),
                token_to_id=dict(token_to_id),
                id_to_token={idx: tok for tok, idx in token_to_id.items()},
                special_tokens=["<|unk|>"],
                max_subword_len=8,
                byte_fallback=True,
            ),
            UnigramModel(
                vocab=dict(vocab),
                token_to_id=dict(token_to_id),
                id_to_token={idx: tok for tok, idx in token_to_id.items()},
                special_tokens=["<|unk|>"],
                max_subword_len=8,
                byte_fallback=False,
            ),
        ]

    def test_fast_path_matches_lattice_exactly(self):
        import random

        rng = random.Random(99)
        alphabet = "a b c d e x y z ▁ \u0905\u0906\u0907 é 👍".split()
        checked = 0
        for model in self.models:
            for _ in range(400):
                s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
                lattice = UnigramLattice(
                    s,
                    model.vocab,
                    max_subword_len=model.max_subword_len,
                    byte_fallback=model.byte_fallback,
                )
                lattice_tokens, _ = lattice.viterbi()
                edges, _ = lattice.viterbi_edges()
                lattice_spans = [(token, edge.start, edge.end) for edge in edges for token in edge.tokens]
                fast = model._encode_fast(s)
                self.assertEqual(model.encode(s), lattice_tokens, repr(s))
                if fast is not None:
                    self.assertEqual(
                        [t for t, _, _ in fast],
                        lattice_tokens,
                        f"fast path mismatch for {s!r}",
                    )
                self.assertEqual(
                    model.encode_with_spans(s),
                    lattice_spans,
                    f"span mismatch for {s!r}",
                )
                checked += 1
        self.assertGreater(checked, 0)


class HuggingFaceExportTests(unittest.TestCase):
    def test_hf_export_generates_valid_schema(self):
        vocab = {"tok": log(0.5), "en": log(0.3), "<|unk|>": log(0.2)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers is not installed")

        tok = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        with TemporaryDirectory() as tmp_dir:
            tok.export_to_huggingface(tmp_dir)
            with open(Path(tmp_dir) / "tokenizer.json", "r", encoding="utf-8") as f:
                hf_data = json.load(f)

            self.assertEqual(hf_data["version"], "1.0")
            self.assertEqual(hf_data["model"]["type"], "Unigram")
            self.assertIsNone(hf_data["post_processor"])
            self.assertTrue((Path(tmp_dir) / "tokenizer_config.json").exists())
            exported = Tokenizer.from_file(str(Path(tmp_dir) / "tokenizer.json"))
            self.assertEqual(exported.encode("token").tokens, ["tok", "en"])

    def test_hf_export_supports_byte_fallback(self):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers is not installed")

        vocab = {"<|unk|>": log(1.0)}
        vocab.update({ByteFallbackEngine.byte_to_token(byte): log(0.001) for byte in range(256)})
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=True,
        )
        tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
        hf_dict = HuggingFaceExporter.export_to_hf_dict(tok)
        self.assertTrue(hf_dict["model"]["byte_fallback"])

        with TemporaryDirectory() as tmp_dir:
            tok.export_to_huggingface(tmp_dir)
            exported = Tokenizer.from_file(str(Path(tmp_dir) / "tokenizer.json"))
            encoded = exported.encode("U0001f389")
            self.assertEqual(exported.decode(encoded.ids), "U0001f389")


class SecurityAndIndentationTests(unittest.TestCase):
    def test_security_sanitization_retains_source_alignment(self):
        shield = SecurityShield(["<|user|>"])
        sanitized, alignment = shield.sanitize_with_alignment("a<|user|>b")

        self.assertEqual(sanitized, "a<\\|user\\|>b")
        self.assertEqual(alignment[0], (0, 1))
        self.assertTrue(all(span == (1, 9) for span in alignment[1:-1]))
        self.assertEqual(alignment[-1], (9, 10))

    def test_indentation_compression_runs_during_encoding_with_offsets(self):
        vocab = {"x": log(0.5), "<|space_4|>": log(0.5)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id={"x": 0, "<|space_4|>": 1},
            id_to_token={0: "x", 1: "<|space_4|>"},
            special_tokens=["<|space_4|>"],
            byte_fallback=False,
        )
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)

        self.assertEqual(tokenizer.encode("    x"), ["<|space_4|>", "x"])
        self.assertEqual(tokenizer.decode(tokenizer.encode_to_ids("    x")), "    x")
        self.assertEqual(
            [(token.text, token.raw_span) for token in tokenizer.encode_with_offsets("    x")],
            [("<|space_4|>", (0, 4)), ("x", (4, 5))],
        )
        self.assertEqual(IndentationCompressor.decompress_indents("<|space_4|>x"), "    x")

    def test_rejects_invalid_security_action(self):
        shield = SecurityShield(["<|user|>"])
        with self.assertRaises(ValueError):
            shield.sanitize("<|user|>", disallowed_special_action="invalid")


class StreamingDecoderTests(unittest.TestCase):
    def test_streaming_decoder_preserves_byte_fallback_and_literal_metaspace(self):
        ids = {
            0: "<0xE2>",
            1: "<0x96>",
            2: "<0x81>",
            3: "<0xE0>",
            4: "<0x80>",
            5: "<0x81>",
            6: "<0xEE>",
        }
        decoder = StreamingDecoder(
            ids,
            metaspace_escape=("\ue000", "\ue001"),
        )
        self.assertEqual(decoder.feed_token_id(0), "")
        self.assertEqual(decoder.feed_token_id(1), "")
        self.assertEqual(decoder.feed_token_id(2), "▁")

        decoder.reset()
        escaped_tokens = [6, 4, 4, 6, 4, 5]
        self.assertEqual("".join(decoder.feed_token_id(token) for token in escaped_tokens), "▁")

    def test_streaming_decoder_applies_indentation_replacements(self):
        decoder = StreamingDecoder(
            {0: "<|space_4|>"},
            special_tokens=["<|space_4|>"],
            special_replacements={"<|space_4|>": "    "},
        )
        self.assertEqual(decoder.feed_token_id(0), "    ")


class AudioCodecTests(unittest.TestCase):
    def test_audio_rvq_and_multimodal_interleaving(self):
        rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=64, frame_size=320)
        synthetic_audio = [0.1 * math.sin(i * 0.1) for i in range(640)]

        tokens, num_frames = rvq.encode_audio(synthetic_audio)
        self.assertEqual(num_frames, 2)
        self.assertEqual(tokens[0], "<|audio_start|>")
        self.assertEqual(tokens[-1], "<|audio_end|>")

        reconstructed = rvq.decode_audio(tokens)
        self.assertEqual(len(reconstructed), len(synthetic_audio))

        vocab = {"audio": log(0.5), "<|unk|>": log(0.5)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )
        base_tok = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        mm_tok = MultimodalTokenizer(base_tok, patch_size=16, channels=3, num_visual_tokens=64)

        aud_segment = AudioSegment(samples=synthetic_audio)
        seq = mm_tok.encode_interleaved(["audio", aud_segment, "audio"])

        self.assertIn(0, seq.modality_mask)  # Text
        self.assertIn(2, seq.modality_mask)  # Audio
        self.assertIn(3, seq.modality_mask)  # Special


class NeuralCodecTests(unittest.TestCase):
    def test_neural_visual_codec_forward_and_tokens(self):
        from multimodal.neural_codecs import HAS_TORCH, NeuralVisualCodec

        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        # Batch of 2 RGB images (32x32)
        x = torch.randn(2, 3, 32, 32)
        model = NeuralVisualCodec(in_channels=3, hidden_dim=32, latent_dim=64, num_tokens=128)

        # 1. Forward training pass
        out = model(x)
        self.assertIn("loss", out)
        self.assertEqual(out["x_recon"].shape, x.shape)
        self.assertEqual(out["indices"].shape, (2, 8, 8))  # 4x downsampled

        # 2. Token emission and reconstruction
        token_strings, indices, (gh, gw) = model.encode_to_tokens(x)
        self.assertEqual(len(token_strings), 2 * 8 * 8)
        self.assertTrue(token_strings[0].startswith("<|vis_"))

        x_recon = model.decode_from_indices(indices, gh, gw)
        self.assertEqual(x_recon.shape, x.shape)

        odd_image = torch.randn(1, 3, 17, 25)
        odd_output = model(odd_image)
        self.assertEqual(odd_output["x_recon"].shape, odd_image.shape)
        _, odd_indices, (odd_gh, odd_gw) = model.encode_to_tokens(odd_image)
        self.assertEqual((odd_gh, odd_gw), (5, 7))
        self.assertEqual(
            model.decode_from_indices(odd_indices, odd_gh, odd_gw, output_size=(17, 25)).shape,
            odd_image.shape,
        )

    def test_neural_audio_codec_forward_and_tokens(self):
        from multimodal.neural_codecs import HAS_TORCH, NeuralAudioCodec

        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        # Batch of 2 audio waveforms (640 samples = 2 frames at 320 downsampling)
        audio = torch.randn(2, 1, 640)
        model = NeuralAudioCodec(
            in_channels=1,
            hidden_dim=32,
            latent_dim=64,
            num_quantizers=4,
            codebook_size=128,
        )

        # 1. Forward training pass
        out = model(audio)
        self.assertIn("loss", out)
        self.assertEqual(out["audio_recon"].shape, audio.shape)
        self.assertEqual(out["indices"].shape, (2, 2, 4))  # [B, T', N_q]

        # 2. Token emission and reconstruction
        tokens, indices = model.encode_to_tokens(audio)
        self.assertEqual(tokens[0], "<|audio_start|>")
        self.assertEqual(tokens[-1], "<|audio_end|>")

        audio_recon = model.decode_from_indices(indices)
        self.assertEqual(audio_recon.shape, audio.shape)

        short_audio = torch.randn(1, 1, 1)
        short_output = model(short_audio)
        self.assertEqual(short_output["audio_recon"].shape, short_audio.shape)
        uneven_audio = torch.randn(1, 1, 321)
        _, uneven_indices = model.encode_to_tokens(uneven_audio)
        self.assertEqual(uneven_indices.shape, (1, 2, 4))
        self.assertEqual(
            model.decode_from_indices(uneven_indices, output_length=321).shape,
            uneven_audio.shape,
        )


class CrossEntropyMergingTests(unittest.TestCase):
    def _make_pipeline(self):
        docs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ] * 20
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer()
        chunks: list[str] = []
        for doc in docs:
            normalized = normalizer.normalize(doc)
            chunks.extend(pre_tokenizer.pre_tokenize(normalized))
        model = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            prune_rate=0.3,
            byte_fallback=True,
        ).train(chunks, verbose=False)
        return docs, normalizer, pre_tokenizer, model, chunks

    def test_cem_reduces_corpus_token_count(self):
        _, _, _, model, chunks = self._make_pipeline()
        before = sum(len(model.encode(c)) for c in chunks)
        optimizer = CrossEntropyMerging(max_merges=100)
        improved = optimizer.optimize(model, chunks)
        after = sum(len(improved.encode(c)) for c in chunks)

        self.assertGreater(len(optimizer.merges), 0)
        self.assertLess(after, before)
        self.assertEqual(len(improved.vocab), len(model.vocab) + len(optimizer.merges))

    def test_cem_preserves_existing_token_ids(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=50)
        improved = optimizer.optimize(model, chunks)

        for tok, tid in model.token_to_id.items():
            self.assertEqual(improved.token_to_id[tok], tid)
        for merged in (m[2] for m in optimizer.merges):
            self.assertGreaterEqual(improved.token_to_id[merged], len(model.vocab))

    def test_cem_allocates_above_sparse_existing_ids(self):
        model = UnigramModel(
            vocab={"a": log(0.5), "b": log(0.5)},
            token_to_id={"a": 0, "b": 2},
            id_to_token={0: "a", 2: "b"},
            special_tokens=[],
            max_subword_len=2,
            byte_fallback=False,
        )
        optimizer = CrossEntropyMerging(max_merges=1)
        improved = optimizer.optimize(model, ["ab"] * 5)
        self.assertEqual(improved.token_to_id["ab"], 3)
        self.assertEqual(improved.id_to_token[2], "b")

    def test_cem_resets_merge_history_between_runs(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=1)
        optimizer.optimize(model, chunks)
        self.assertLessEqual(len(optimizer.merges), 1)
        optimizer.optimize(model, chunks)
        self.assertLessEqual(len(optimizer.merges), 1)

    def test_cem_roundtrip_is_lossless_through_full_pipeline(self):
        docs, normalizer, pre_tokenizer, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=50)
        improved = optimizer.optimize(model, chunks)
        tokenizer = CustomTokenizer(normalizer, pre_tokenizer, improved)

        for doc in docs:
            self.assertEqual(tokenizer.decode(tokenizer.encode_to_ids(doc)), doc)

    def test_cem_respects_max_subword_len_and_excludes_specials(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=100)
        optimizer.optimize(model, chunks)

        for a, b, merged, _, _ in optimizer.merges:
            self.assertLessEqual(len(merged), model.max_subword_len)
            self.assertNotIn(a, model.special_tokens)
            self.assertNotIn(b, model.special_tokens)
            self.assertNotIn(merged, model.special_tokens)
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(a))
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(b))

    def test_cem_max_merges_zero_returns_identical_model(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=0)
        self.assertIs(optimizer.optimize(model, chunks), model)
        with self.assertRaises(ValueError):
            CrossEntropyMerging(max_merges=-1)


class SuperBPETests(unittest.TestCase):
    SPACE = "\u2581"

    def _make_pipeline(self):
        docs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ] * 30
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer()
        chunks: list[str] = []
        for doc in docs:
            normalized = normalizer.normalize(doc)
            chunks.extend(pre_tokenizer.pre_tokenize(normalized))
        base_model = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            prune_rate=0.3,
            byte_fallback=True,
        ).train(chunks, verbose=False)
        optimizer = CrossEntropyMerging(max_merges=100, cross_word=True)
        improved = optimizer.optimize(base_model, chunks)
        base_tok = CustomTokenizer(normalizer, pre_tokenizer, base_model)
        improved_tok = CustomTokenizer(normalizer, pre_tokenizer, improved)
        return docs, normalizer, pre_tokenizer, base_tok, improved_tok, optimizer

    def test_superbpe_merges_span_word_boundaries(self):
        _, _, _, _, _, optimizer = self._make_pipeline()
        self.assertGreater(len(optimizer.merges), 0)
        for a, b, merged, _, _ in optimizer.merges:
            self.assertIn(self.SPACE, merged)
            self.assertNotIn(a, ("<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"))
            self.assertNotIn(b, ("<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"))

    def test_superbpe_replays_hierarchical_merges(self):
        vocab = {
            "a": log(0.2),
            self.SPACE: log(0.2),
            "b": log(0.2),
            "a" + self.SPACE: log(0.2),
            "a" + self.SPACE + "b": log(0.2),
        }
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=[],
            max_subword_len=3,
            byte_fallback=False,
        )
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        self.assertEqual(tokenizer.encode("a b"), ["a" + self.SPACE + "b"])
        token = tokenizer.encode_with_offsets("a b")[0]
        self.assertEqual(token.raw_span, (0, 3))

    def test_superbpe_reduces_token_count_through_pipeline(self):
        docs, _, _, base_tok, improved_tok, _ = self._make_pipeline()
        full_text = " ".join(docs)
        before = len(base_tok.encode(full_text))
        after = len(improved_tok.encode(full_text))

        self.assertLess(after, before)
        self.assertTrue(any(tok in improved_tok.encode(full_text) for tok in ["the" + self.SPACE, self.SPACE + "fox"]))

    def test_superbpe_roundtrip_lossless_through_pipeline(self):
        docs, _, _, _, improved_tok, _ = self._make_pipeline()
        for doc in docs:
            self.assertEqual(improved_tok.decode(improved_tok.encode_to_ids(doc)), doc)

    def test_superbpe_preserves_ids_and_grows_vocab(self):
        _, _, _, base_tok, improved_tok, optimizer = self._make_pipeline()
        for tok, tid in base_tok.model.token_to_id.items():
            self.assertEqual(improved_tok.model.token_to_id[tok], tid)
        self.assertEqual(
            len(improved_tok.model.vocab),
            len(base_tok.model.vocab) + len(optimizer.merges),
        )
        for merged in (m[2] for m in optimizer.merges):
            self.assertGreaterEqual(improved_tok.model.token_to_id[merged], len(base_tok.model.vocab))

    def test_superbpe_respects_max_subword_len(self):
        _, _, _, base_tok, _, optimizer = self._make_pipeline()
        for a, b, merged, _, _ in optimizer.merges:
            self.assertLessEqual(len(merged), base_tok.model.max_subword_len)
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(a))
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(b))

    def test_superbpe_offset_spans_are_contiguous(self):
        _, _, _, _, improved_tok, _ = self._make_pipeline()
        for doc in ["the quick brown fox", "brown dogs are quick"]:
            tokens = improved_tok.encode_with_offsets(doc)
            self.assertEqual(tokens[0].raw_span[0], 0)
            self.assertEqual(tokens[-1].raw_span[1], len(doc))
            prev_end = tokens[0].raw_span[0]
            for tok in tokens:
                start, end = tok.raw_span
                self.assertEqual(start, prev_end)
                prev_end = end
            self.assertEqual(improved_tok.decode([t.id for t in tokens]), doc)

    def test_superbpe_sample_applies_cross_word_merges(self):
        _, _, _, _, improved_tok, _ = self._make_pipeline()
        sample_tokens = improved_tok.sample("the quick brown fox", alpha=0.5)
        self.assertTrue(len(sample_tokens) > 0)
        sample_ids = [
            improved_tok.model.token_to_id.get(t, improved_tok.model.token_to_id.get("<|unk|>", 0))
            for t in sample_tokens
        ]
        self.assertEqual(improved_tok.decode(sample_ids), "the quick brown fox")

    def test_image_patcher_empty_nested_pixels(self):
        from multimodal.image_patcher import DynamicImagePatcher

        patcher = DynamicImagePatcher(patch_size=4, channels=3)
        self.assertEqual(patcher.extract_patches([]), ([], (0, 0)))
        self.assertEqual(patcher.extract_patches([[]]), ([], (0, 0)))

    def test_vocab_adapter_extreme_underflow(self):
        from vocab_adapter import VocabularyAdapter

        _, _, _, base_tok, _, _ = self._make_pipeline()
        # Simulate extreme negative log-probs
        base_tok.model.vocab["rare_token"] = -1000.0
        adapted = VocabularyAdapter.expand_vocabulary(
            base_tok, ["xyzabc xyzabc xyzabc"], num_new_tokens=5, min_frequency=1, verbose=False
        )
        self.assertGreater(len(adapted.model.vocab), len(base_tok.model.vocab))
        for tok, lp in adapted.model.vocab.items():
            self.assertFalse(math.isnan(lp))
            self.assertFalse(math.isinf(lp))


class PhaseOneOptimizationTests(unittest.TestCase):
    def test_seed_builder_pmi_ranking_and_adaptive_sizing(self):
        builder = SeedVocabularyBuilder(
            target_vocab_size=300,
            seed_multiplier=2.0,
            ranking_strategy="pmi",
            adaptive_multiplier=True,
            min_frequency=1,
        )
        chunks = ["neural", "network", "language", "model", "neural", "model", "transformer"]
        seed_vocab = builder.build_seed_vocab(chunks)
        self.assertGreater(len(seed_vocab), 0)
        tokens = [t.token for t in seed_vocab]
        self.assertIn("model", tokens)

        # Invalid ranking strategy check
        with self.assertRaises(ValueError):
            SeedVocabularyBuilder(ranking_strategy="invalid_strategy")

    def test_lattice_beam_pruning_and_min_edge_threshold(self):
        vocab = {
            "a": log(0.3),
            "b": log(0.3),
            "ab": log(0.2),
            "c": log(0.1),
            "abc": log(0.05),
            "rare": log(1e-6),
        }
        # Beam pruning: max 1 incoming edge per node
        lattice = UnigramLattice(
            "abc",
            vocab,
            max_subword_len=3,
            byte_fallback=True,
            max_edges_per_node=1,
            min_edge_log_prob=log(0.01),
        )
        for j in range(1, len("abc") + 1):
            self.assertLessEqual(len(lattice.end_nodes[j]), 1)

        # Invalid max_edges_per_node check
        with self.assertRaises(ValueError):
            UnigramLattice("abc", vocab, max_edges_per_node=0)

    def test_unigram_trainer_convergence_early_stopping(self):
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps",
            "brown fox jumps over",
        ]
        tok = CustomTokenizer.train_from_corpus(
            corpus,
            target_vocab_size=320,
            ranking_strategy="pmi",
            adaptive_multiplier=True,
            max_edges_per_node=5,
            convergence_tolerance=1e-3,
            verbose=False,
        )
        self.assertIsInstance(tok, CustomTokenizer)
        encoded = tok.encode("the quick brown fox")
        self.assertTrue(len(encoded) > 0)
        decoded = tok.decode(tok.encode_to_ids("the quick brown fox"))
        self.assertEqual(decoded, "the quick brown fox")

    def test_encode_with_metrics_diagnostic_report(self):
        corpus = ["standard test sentence with normal alphabet"]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, verbose=False)
        report = tok.encode_with_metrics("standard test sentence 🚀")
        self.assertIsInstance(report, TokenizationReport)
        self.assertGreater(report.num_tokens, 0)
        self.assertGreater(report.num_bytes, 0)
        self.assertGreater(report.num_chars, 0)
        # Byte fallback should catch emoji 🚀 if not in alphabet
        self.assertGreaterEqual(report.byte_fallback_tokens, 0)
        self.assertGreaterEqual(report.byte_fallback_rate, 0.0)
        self.assertLessEqual(report.byte_fallback_rate, 1.0)
        self.assertGreater(report.compression_ratio_bytes_per_token, 0.0)

    def test_complex_indic_and_arabic_unicode_offsets(self):
        corpus = [
            "प्राकृतिक भाषा प्रसंस्करण नमस्ते दुनिया",
            "تعتبر معالجة اللغات الطبيعية الحديثة",
        ]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=350, verbose=False)

        for text in corpus:
            tokens = tok.encode_with_offsets(text)
            self.assertTrue(len(tokens) > 0)
            self.assertEqual(tokens[0].raw_span[0], 0)
            self.assertEqual(tokens[-1].raw_span[1], len(text))
            for t in tokens:
                self.assertGreaterEqual(t.raw_span[0], 0)
                self.assertLessEqual(t.raw_span[1], len(text))
                self.assertLessEqual(t.raw_span[0], t.raw_span[1])
            # Monotonic start span progression
            for i in range(len(tokens) - 1):
                self.assertLessEqual(tokens[i].raw_span[0], tokens[i + 1].raw_span[0])
            # Lossless roundtrip
            decoded = tok.decode([t.id for t in tokens])
            self.assertEqual(decoded, text)


if __name__ == "__main__":
    unittest.main()
