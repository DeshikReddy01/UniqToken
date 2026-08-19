<p align="center">
  <h1 align="center">Caliper</h1>
  <p align="center">
    <strong>Zero-dependency, high-precision Byte-Fallback Unigram &amp; Multimodal Tokenizer</strong>
  </p>
  <p align="center">
    Built from scratch in pure Python — with exact character-span tracking, multilingual Unicode protection, and three interchangeable subword algorithms.
  </p>
</p>

<p align="center">
  <a href="https://github.com/umran666/caliper/actions/workflows/ci.yml"><img src="https://github.com/umran666/caliper/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/umran666/caliper/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/dependencies-zero-brightgreen.svg" alt="Dependencies">
</p>

---

## Overview

Most production tokenizers lean on a compiled C++ or Rust backend (SentencePiece, HuggingFace `tokenizers`) and treat character-offset alignment, control-token injection defense, and vocabulary extension as afterthoughts. **Caliper** is a single, dependency-free Python package that treats all three as first-class design constraints, while implementing the same core algorithms — Unigram Language Model segmentation, Byte-Pair Encoding, and post-training vocabulary merging — that back today's production LLM tokenizers.

### Design Goals

| # | Production Failure Mode | Caliper's Response |
|:-:|:---|:---|
| 1 | **Out-of-vocabulary catastrophe** — rare Unicode, emoji, or foreign scripts silently collapse to `<unk>`, destroying information. | Strict **byte fallback**: any character outside the vocabulary decomposes into its raw UTF-8 bytes (`<0x00>`–`<0xFF>`), guaranteeing a **0% OOV rate** and exact, lossless roundtrip decoding. |
| 2 | **Span drift** — normalization (NFKC, case folding) changes string length, breaking the character offsets that NER, extractive QA, and citation systems depend on. | **Dual-offset tracking**: sanitization, indentation compression, normalization, and pre-tokenization each produce their own alignment, composed end-to-end by `_compose_alignment()`, so `encode_with_offsets()` returns a `Token.raw_span` pointing to the exact byte range in the original raw text. |
| 3 | **Digit and script clumping** — numbers and mixed scripts get fused into arbitrary tokens, hurting arithmetic reasoning and URL parsing. | A **10-pattern regex boundary layer** isolates URLs, emails, hashtags, emoji (including ZWJ sequences), CJK ideographs, and digit runs before subword segmentation ever runs. |
| 4 | **Deterministic brittleness** — a single fixed segmentation makes models fragile to typos and spelling variants. | **FFBS subword regularization** — Forward-Filtering Backward-Sampling over the segmentation lattice — samples stochastic alternative segmentations during training ([Kudo, 2018](#algorithms--base-papers)). |
| 5 | **Vocabulary freezing** — extending a trained vocabulary normally forces re-indexing, corrupting the model's existing embedding matrix. | **Non-destructive vocabulary growth**: both `VocabularyAdapter` and `CrossEntropyMerging` append new tokens at `id = len(old_vocab) + i`, leaving every existing token ID and embedding row untouched. |

---

## Features

<table>
<tr><td>

**Tokenization**
- Three trainable algorithms: Unigram LM (DAG + Viterbi + EM + FFBS), BPE, and CEM/SuperBPE vocabulary extension
- Byte-fallback codec for 0% OOV across all Unicode
- FFBS subword regularization for training-time augmentation
- PrefixTrie for O(L) single-pass lattice edge mining

</td><td>

**Alignment & Safety**
- Exact dual-offset span tracking (raw → normalized → token)
- SecurityShield: control-token injection / delimiter-hijacking defense
- Indic virama, Arabic harakat, Hebrew niqqud, Hangul jamo cluster protection
- CJK isolation, emoji ZWJ/variation-selector preservation

</td></tr>
<tr><td>

**Serving**
- StreamingDecoder with UTF-8 byte-buffer for real-time generation
- BatchCollator with padding, attention masks, BOS/EOS injection
- PyTorch tensor output via `to_torch()`
- HuggingFace-compatible export (`tokenizer.json` schema)

</td><td>

**Code & Domain**
- IndentationCompressor: reversible 2/4/8/16-space and tab compression
- Non-destructive online vocabulary expansion for domain adaptation
- SuperBPE whitespace-crossing merge mode ([Liu et al., 2025](#algorithms--base-papers))
- Save/load serialization with full config preservation

</td></tr>
</table>

---

## Installation

```bash
git clone https://github.com/umran666/caliper.git
cd caliper
pip install -e .
```

**Optional extras** (defined in [`pyproject.toml`](pyproject.toml)):

| Extra | Command | What it adds |
|:------|:--------|:-------------|
| PyTorch | `pip install -e ".[torch]"` | `torch>=2.0.0` — tensor output in `BatchCollator` |
| HuggingFace | `pip install -e ".[huggingface]"` | `tokenizers>=0.13.0`, `transformers>=4.30.0` — interop & export |
| Benchmarks | `pip install -e ".[bench]"` | `sentencepiece>=0.1.99`, `tokenizers>=0.13.0` — comparison baselines |
| Testing | `pip install -e ".[test]"` | `pytest>=7.0.0`, `coverage>=7.0.0`, `ruff>=0.4.0`, `mypy>=1.8.0` |
| Everything | `pip install -e ".[all]"` | All of the above |

---

## Quickstart

### Train a Unigram tokenizer

```python
from tokenizer import CustomTokenizer

corpus = [...]  # list of training documents

tok = CustomTokenizer.train_from_corpus(
    corpus,
    target_vocab_size=32_000,
    special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"],
    byte_fallback=True,
)

# Encode → decode roundtrip
ids = tok.encode_to_ids("fix in 2024 at https://site.com")
text = tok.decode(ids)
assert text == "fix in 2024 at https://site.com"

# Stochastic subword regularization (training-time augmentation)
sampled = tok.sample("hello world", alpha=0.5)

# Exact character-span offsets for every token
for token in tok.encode_with_offsets("fix in 2024"):
    print(f"{token.text!r:>12}  id={token.id:<5}  raw_span={token.raw_span}")
```

### Train a BPE tokenizer

```python
from bpe_trainer import BPETrainer

trainer = BPETrainer(target_vocab_size=32_000, byte_fallback=True)
model = trainer.train(chunks=corpus, verbose=True)

tokens = model.encode("tokenization")
text = model.decode(token_ids)
```

### Extend vocabulary with CEM / SuperBPE

```python
from cem_merger import CrossEntropyMerging

# Standard CEM: greedily add merges that minimize cross-entropy increase
cem = CrossEntropyMerging(max_merges=200, verbose=True)
extended = cem.optimize(tok.model, chunks=corpus)

# SuperBPE mode: only accept merges that cross whitespace boundaries
superbpe = CrossEntropyMerging(max_merges=200, cross_word=True)
superbpe_model = superbpe.optimize(tok.model, chunks=corpus)
```

### Export to HuggingFace format

```python
tok.export_to_huggingface("hf_export/")

# Then load with transformers:
# from transformers import AutoTokenizer
# hf_tok = AutoTokenizer.from_pretrained("hf_export/")
```

### Streaming decode

```python
decoder = tok.get_streaming_decoder()

output = ""
for token_id in generated_ids:  # one id at a time from an LLM
    output += decoder.feed_token_id(token_id)
output += decoder.flush()
```

### Sanitize untrusted input

```python
from security_shield import SecurityShield

shield = SecurityShield(special_tokens=["<|endoftext|>", "<|system|>", "<|user|>"])
safe = shield.sanitize(
    untrusted_input,
    allowed_special="none",  # or {"<|user|>"} to whitelist
    disallowed_special_action="escape",  # "escape" | "raise" | "ignore"
)
```

> **Note:** `CustomTokenizer` wires `SecurityShield.sanitize()` into every `encode()`, `sample()`, and `encode_with_offsets()` call automatically (defaults: `allowed_special="none"`, `disallowed_special_action="escape"`), so sanitization is not an opt-in step.

### Compress structured whitespace

```python
from indentation_compressor import IndentationCompressor

compact = IndentationCompressor.compress_indents(source_code)
restored = IndentationCompressor.decompress_indents(compact)
assert restored == source_code
```

### Save and load

```python
tok.save("saved_model/")
tok2 = CustomTokenizer.load("saved_model/")

assert tok2.encode_to_ids("test") == tok.encode_to_ids("test")
```

---

## Architecture

### End-to-End Pipeline

```mermaid
flowchart LR
    A["Raw Text"] --> B["SecurityShield<br/>sanitize + alignment"]
    B --> C["Normalizer<br/>NFKC + dual-offset"]
    C --> D["RegexPreTokenizer<br/>10 boundary patterns"]
    D --> E1["UnigramLattice<br/>DAG · Viterbi · FFBS"]
    D --> E2["BPEModel<br/>rank-based merges"]
    E1 --> F["CEM / SuperBPE<br/>vocabulary extension"]
    E1 --> G["Token IDs"]
    E2 --> G
    F --> G
    G --> H["BatchCollator<br/>pad · mask · BOS/EOS"]
    G --> I["StreamingDecoder<br/>byte-buffer aware"]
    H --> J["PyTorch Tensors"]
    I --> K["Decoded Text"]
```

### Project Structure

```
caliper/
├── tokenizer.py              # CustomTokenizer — unified facade
├── pre_tokenizer.py           # Normalizer + RegexPreTokenizer (10 patterns)
├── byte_codec.py              # ByteFallbackEngine — UTF-8 ↔ <0xHH> codec
├── trie.py                    # PrefixTrie — O(L) vocab prefix matching
│
├── seed_builder.py            # SeedVocabularyBuilder — initial ≈3× candidate vocab
├── unigram_lattice.py         # UnigramLattice — DAG, Viterbi, EM stats, FFBS sampling
├── unigram_trainer.py         # UnigramTrainer — EM + likelihood-pruning training loop
├── vocab_adapter.py           # VocabularyAdapter — non-destructive vocab expansion
├── cem_merger.py              # CrossEntropyMerging — CEM / SuperBPE extension
│
├── bpe_trainer.py             # BPETrainer — classic greedy pairwise-merge training
├── bpe_model.py               # BPEModel — rank-based merge inference (tiktoken-style)
│
├── batch_collator.py          # BatchCollator — padding, masks, BOS/EOS, to_torch()
├── streaming_decoder.py       # StreamingDecoder — incremental UTF-8-safe decode
├── hf_exporter.py             # HuggingFaceExporter — tokenizer.json + config export
│
├── security_shield.py         # SecurityShield — control-token injection defense
├── indentation_compressor.py  # IndentationCompressor — reversible whitespace codec
│
├── multimodal/
│   ├── multimodal_tokenizer.py  # MultimodalTokenizer — text + image + audio
│   ├── visual_codebook.py       # VisualCodebook — VQ codebook for image patches
│   ├── image_patcher.py         # ImagePatcher — grid-based patch extraction
│   ├── audio_codec.py           # ResidualVectorQuantizer — RVQ for audio
│   └── neural_codecs.py         # NeuralVisualCodec / NeuralAudioCodec (PyTorch)
│
├── benchmarks/
│   └── benchmark_suite.py     # TokenizerBenchmarkSuite — 7-axis evaluation
│
├── test_tokenizer.py          # 57 unit tests across 17 test classes
├── test_fuzz_properties.py    # 7 property-based fuzz tests
├── saved_model/               # Example serialized tokenizer artifact
├── pyproject.toml             # Package config, extras, ruff/mypy settings
└── .github/workflows/ci.yml  # CI: 3 OS × 4 Python versions = 12-cell matrix
```

### Module Dependency Graph

```mermaid
graph TD
    T["tokenizer.py<br/>CustomTokenizer"] --> N["pre_tokenizer.py<br/>Normalizer · RegexPreTokenizer"]
    T --> UL["unigram_lattice.py<br/>UnigramLattice"]
    T --> UT["unigram_trainer.py<br/>UnigramTrainer · UnigramModel"]
    T --> SS["security_shield.py<br/>SecurityShield"]
    T --> IC["indentation_compressor.py<br/>IndentationCompressor"]
    T --> SD["streaming_decoder.py<br/>StreamingDecoder"]
    T --> HF["hf_exporter.py<br/>HuggingFaceExporter"]

    UT --> UL
    UT --> SB["seed_builder.py<br/>SeedVocabularyBuilder"]
    UT --> BC["byte_codec.py<br/>ByteFallbackEngine"]
    UT --> TR["trie.py<br/>PrefixTrie"]
    UL --> BC
    UL --> TR

    CEM["cem_merger.py<br/>CrossEntropyMerging"] --> UT
    VA["vocab_adapter.py<br/>VocabularyAdapter"] --> UT

    BT["bpe_trainer.py<br/>BPETrainer"] --> BC
    BT --> N
    BM["bpe_model.py<br/>BPEModel"] --> BC

    MM["multimodal/<br/>MultimodalTokenizer"] --> T
```

---

## Algorithms & Base Papers

Caliper is an independent, from-scratch implementation. It does not wrap any paper's reference code. The algorithms are drawn from:

| Algorithm | Module(s) | Reference |
|:----------|:----------|:----------|
| Unigram LM segmentation (DAG, Viterbi, EM, FFBS sampling) | `unigram_lattice.py`, `unigram_trainer.py` | Taku Kudo. *"Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates."* ACL 2018. |
| Byte-Pair Encoding | `bpe_trainer.py`, `bpe_model.py` | Rico Sennrich, Barry Haddow, Alexandra Birch. *"Neural Machine Translation of Rare Words with Subword Units."* ACL 2016. |
| Cross-Entropy Merging (CEM) | `cem_merger.py` | Leonidas Gee, Leonardo Rigutini, Marco Ernandes, Andrea Zugarini. *"Multi-Word Tokenization for Sequence Compression."* EMNLP 2023 (arXiv:2402.09949). |
| SuperBPE ("Space Travel") | `cem_merger.py` (`cross_word=True`) | Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A. Smith, Yejin Choi. *"SuperBPE: Space Travel for Language Models."* COLM 2025 (arXiv:2503.13423). |

---

## Security Model

`SecurityShield` guards against control-token smuggling and delimiter hijacking — e.g., a user injecting a literal `<|endoftext|>` or `<|system|>` string to manipulate a model's context boundary.

| Policy | Behavior |
|:-------|:---------|
| `"escape"` | Neutralizes the control sequence in place (default) |
| `"raise"` | Raises `ValueError`, rejecting the input |
| `"ignore"` | Passes the sequence through unmodified |

The `allowed_special` parameter accepts `"all"`, `"none"`, or a specific `set` of control tokens to whitelist. Sanitization preserves character-alignment tracking via `sanitize_with_alignment()`.

`CustomTokenizer` integrates this automatically — every `encode()`, `sample()`, and `encode_with_offsets()` call runs through `SecurityShield.sanitize()` first.

---

## Testing & CI

### Test Suite

| Suite | Tests | Scope |
|:------|------:|:------|
| `test_tokenizer.py` | 57 | 17 test classes covering normalization, byte-fallback, encoding/decoding, lattice construction, training validation, batch collation, multimodal, trie, BPE, fast-path parity, HuggingFace export, security shield, indentation compression, streaming decode, audio codecs, neural codecs, CEM, and SuperBPE |
| `test_fuzz_properties.py` | 7 | Property-based fuzzing: roundtrip integrity, offset validity, Unicode resilience, determinism |
| **Total** | **64** | |

### CI Pipeline

The GitHub Actions [workflow](.github/workflows/ci.yml) runs on every push and PR across a **12-cell matrix** (3 OS × 4 Python versions):

| | Ubuntu | Windows | macOS |
|:---|:---:|:---:|:---:|
| Python 3.9 | ✓ | ✓ | ✓ |
| Python 3.10 | ✓ | ✓ | ✓ |
| Python 3.11 | ✓ | ✓ | ✓ |
| Python 3.12 | ✓ | ✓ | ✓ |

Each cell runs:
1. **Ruff** lint + format check
2. **Mypy** static type checking
3. **Full test suite** (unit + property fuzzing)
4. **Benchmark suite** smoke test
5. **Package build** verification (`python -m build`)

### Running locally

```bash
pip install -e ".[test]"

pytest                                          # all tests
ruff check . && ruff format --check .           # lint + format
mypy .                                          # type check
coverage run -m pytest && coverage report       # coverage
python benchmarks/benchmark_suite.py            # benchmarks
```

---

## Benchmarks

`TokenizerBenchmarkSuite` evaluates Caliper across **7 axes** on **6 multilingual corpora** (English prose, Python source, Hindi/Devanagari, Japanese/CJK, Arabic, arithmetic/math):

| Axis | Metric |
|:-----|:-------|
| **Compression ratio** | Bytes per token across scripts |
| **Morphological fertility** | Tokens per word |
| **Encode throughput** | KB/sec and tokens/sec |
| **Decode throughput** | KB/sec |
| **Offset overhead** | `encode_with_offsets` / `encode` time ratio |
| **Code compression** | Token savings from `IndentationCompressor` |
| **Architecture comparison** | Unigram vs. BPE head-to-head |

External baselines (with `[bench]` extra installed): **HuggingFace `tokenizers`** (Rust), **SentencePiece** (C++), and **tiktoken**.

```bash
pip install -e ".[bench]"
python benchmarks/benchmark_suite.py
```

---

## Multimodal

The `multimodal/` package extends Caliper to handle text, image, and audio inputs through a unified `MultimodalTokenizer`:

| Module | Purpose |
|:-------|:--------|
| `multimodal_tokenizer.py` | `MultimodalTokenizer` — unified text + image + audio tokenization with cross-modal token interleaving |
| `visual_codebook.py` | `VisualCodebook` — vector-quantized codebook for mapping image patches to discrete tokens |
| `image_patcher.py` | `ImagePatcher` — grid-based patch extraction from pixel arrays |
| `audio_codec.py` | `ResidualVectorQuantizer` — multi-layer residual VQ for audio waveform discretization |
| `neural_codecs.py` | `NeuralVisualCodec` / `NeuralAudioCodec` — PyTorch-based learned codecs (requires `[torch]` extra) |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install the dev toolchain:
   ```bash
   pip install -e ".[test]"
   ```
3. Keep new code within the `ruff` (line-length 120, target `py39`) and `mypy` configuration.
4. Add or update tests in `test_tokenizer.py` / `test_fuzz_properties.py` for any behavioral change.
5. Verify before opening a PR:
   ```bash
   pytest && ruff check . && mypy .
   ```

---

## License

Released under the [MIT License](LICENSE).

---

<p align="center">
  Maintained by <a href="https://github.com/umran666">@umran666</a>
</p>
