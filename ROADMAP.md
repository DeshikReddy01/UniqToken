# UniqToken Architecture & Contributor Roadmap

This document defines the architectural strategy and execution pipeline for UniqToken. It organizes open development into two non-overlapping engines and eight structured execution stages.

---

## 1. The Core Split: Two Engines, Two Products

UniqToken explicitly separates runtime acceleration of existing models from vocabulary design for new models:

`
                                 UNIQTOKEN
                                     │
            ┌────────────────────────┴────────────────────────┐
            │                                                 │
     [COMPATIBILITY ENGINE]                           [RESEARCH ENGINE]
      uniqtoken.compat                                 uniqtoken.train
            │                                                 │
   ┌────────┼────────┐                               ┌────────┼────────┐
tiktoken   HF       SPM                           Unigram    BPE    SuperBPE
   │        │        │                               │        │        │
   └────────┴────────┘                               └────────┴────────┘
            │                                                 │
   Same Vocab, Same IDs                             New Vocab, New IDs
   Same Segmentation                                Script-Aware Allocation
   Zero Token Count Delta                           Zero-Drift Dual Offsets
   Existing Weights Intact                          Pre-Training / Sovereign LLMs
            │                                                 │
            └────────────────────────┬────────────────────────┘
                                     │
                             CANONICAL RUST CORE
                            crates/uniqtoken_core
                                     │
                      ┌──────────────┼──────────────┐
                      │              │              │
                   Python          C-ABI           WASM
                (HF Adapter)  (llama.cpp/vLLM)  (Web Runtime)
`

### Engine Comparison

| Property | Compatibility Engine (uniqtoken.compat) | Research Engine (uniqtoken.train) |
|:---|:---|:---|
| **Goal** | Runtime acceleration, low latency, small memory footprint | High context efficiency, script fairness, exact offset tracking |
| **Model Weights** | Works with **existing** models (LLaMA-3, Qwen, Mistral) | Requires **new model training** or vocabulary adaptation |
| **Vocabulary & IDs** | **Identical** to target model (exact differential match) | **New** entropy-guided, script-balanced vocabulary |
| **Token Count** | **Zero change** (must match reference tokenizer byte-for-byte)| **Material reduction** on non-Latin scripts under matched budgets |
| **Target Audience** | Inference optimization engineers, vLLM / llama.cpp users | Pre-training teams, sovereign AI labs (Indic/Arabic/CJK) |

---

## 2. Master Execution Stages & Issue Ledger

Contributors and maintainers can pick up tasks from the structured queue below. Every issue is tracked live on the [GitHub Issue Tracker](https://github.com/umran666/UniqToken/issues).

### Stage 1: Architecture Split & Benchmark Foundations
- [ ] **[#49](https://github.com/umran666/UniqToken/issues/49)** [P0-critical] Refactor core into uniqtoken.compat and uniqtoken.train.
- [ ] **[#50](https://github.com/umran666/UniqToken/issues/50)** [P0-critical] Rebuild matched-budget (8k–128k) benchmark harness and invalidate legacy ledgers.

### Stage 2: Canonical Native Rust Engine
- [ ] **[#42](https://github.com/umran666/UniqToken/issues/42)** [P0-critical] Full native batch pipeline in Rust to eliminate PyO3 FFI boundary overhead.
- [ ] **[#41](https://github.com/umran666/UniqToken/issues/41)** [P0-critical] Implement UAX #29 grapheme cluster boundaries to prevent Indic/Thai glyph splitting.
- [x] **[#43](https://github.com/umran666/UniqToken/issues/43)** [P1-high] Restrict number clumping to 1–3 digits (\d{1,3}) for LLM arithmetic reasoning. (Resolved in commit `9b193e3`)
- [ ] **[#36](https://github.com/umran666/UniqToken/issues/36)** [P1-high] Memory-mapped binary model format (mmap) for sub-millisecond cold starts.

### Stage 3: Differential Compatibility & Exception Matrix
- [ ] **[#51](https://github.com/umran666/UniqToken/issues/51)** [P1-high] Automated differential test suite against 	iktoken and HuggingFace with documented exception matrix (COMPATIBILITY_EXCEPTIONS.md).
- [ ] **[#46](https://github.com/umran666/UniqToken/issues/46)** [P1-high] Property-based fuzz testing suite using Hypothesis and LibFuzzer.

### Stage 4: Standalone C-ABI & System Bindings
- [ ] **[#22](https://github.com/umran666/UniqToken/issues/22)** [P1-high] C-ABI shared library export (libuniqtoken.so / .dll / .dylib) and C/C++ header (uniqtoken.h).
- [ ] **[#33](https://github.com/umran666/UniqToken/issues/33)** [P1-high] Zero-copy PyBuffer borrowing in Rust batch encoder.

### Stage 5: Hugging Face Ecosystem Integration
- [ ] **[#45](https://github.com/umran666/UniqToken/issues/45)** [P1-high] Implement native uniqtoken.hf.UniqTokenizer matching PreTrainedTokenizerFast conventions with public compatibility matrix.
- [ ] **[#44](https://github.com/umran666/UniqToken/issues/44)** [P1-high] Jinja2 chat template engine and pply_chat_template API.
- [ ] **[#24](https://github.com/umran666/UniqToken/issues/24)** [P2-medium] Direct push_to_hub() publishing utility.

### Stage 6: Inference Serving Engine Hooks
- [ ] **[#52](https://github.com/umran666/UniqToken/issues/52)** [P1-high] GGUF vocabulary table loader and C++ tokenization hook for llama.cpp.
- [ ] **[#27](https://github.com/umran666/UniqToken/issues/27)** [P1-high] vLLM custom tokenizer backend plugin and streaming worker.

### Stage 7: Large-Scale Training Infrastructure
- [ ] **[#47](https://github.com/umran666/UniqToken/issues/47)** [P1-high] Disk-backed external chunk counter for TB-scale out-of-core training.
- [ ] **[#48](https://github.com/umran666/UniqToken/issues/48)** [P2-medium] SuperBPE Subword Regularization and BPE Dropout.

### Stage 8: Scientific Evaluation & Research Publication
- [ ] **[#23](https://github.com/umran666/UniqToken/issues/23)** [P2-medium] Expand evaluation corpora with low-resource African and Indic languages.
- [ ] **[#25](https://github.com/umran666/UniqToken/issues/25)** [P2-medium] Interactive terminal comparison CLI tool (uniqtoken compare).

---

## 3. How to Contribute

1. **Pick an issue from Stages 1–4 first**: These form the foundational substrate.
2. **Read the Acceptance Criteria**: Each issue specifies exact verification criteria and tests.
3. **Submit a Draft PR**: Tag the corresponding issue number (e.g. Fixes #41).
4. **Pass the CI Matrix**: All PRs must pass the 16-job matrix (Python 3.9–3.12 across Ubuntu, macOS, Windows with 0 clippy warnings and 0 ruff errors).
