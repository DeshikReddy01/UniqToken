# Caliper Core (Rust Acceleration Engine)

High-performance native Rust crate for the Caliper Tokenizer.

## Components
- `trie.rs`: Native PrefixTrie with fast AHashMap character branch indexing and common prefix search.
- `viterbi.rs`: Dynamic programming 1-best shortest-path Viterbi search and forward-backward EM posterior expectation aggregator in log-space.
- `lib.rs`: PyO3 C-extension binding exposing `caliper_core` to Python.

## Building Native Extension
To compile the native extension into the local environment:

```bash
# Using maturin
pip install maturin
maturin develop --release

# Or build wheel
maturin build --release
```

When compiled, Caliper's Python layer automatically detects `caliper_core` and delegates Trie matching and Viterbi dynamic programming to native Rust; otherwise, it seamlessly runs on pure Python without error.
