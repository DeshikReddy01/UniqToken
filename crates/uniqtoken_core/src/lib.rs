//! UniqToken Core: High-performance native Rust acceleration module for UniqToken.

pub mod normalizer;
pub mod pipeline;
pub mod rust_tokenizer;
pub mod seed;
pub mod trie;
pub mod viterbi;

use normalizer::{rust_normalize, rust_normalize_with_alignment};
use pipeline::{
    rust_encode_text_batch, rust_encode_text_native, rust_encode_text_native_batch,
    rust_pre_tokenize,
};
use pyo3::prelude::*;
use rust_tokenizer::rust_diagnostic_batch;
use rust_tokenizer::RustTokenizer;
use seed::rust_mine_ngrams;
use trie::RustPrefixTrie;
use viterbi::{
    rust_diagnostic_viterbi, rust_encode_ids_batch, rust_encode_tokens_batch,
    rust_forward_backward_expectations, rust_viterbi_decode, rust_viterbi_decode_batch, ViterbiSpan,
};

/// Native UniqToken Core Python Module
#[pymodule]
fn uniqtoken_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPrefixTrie>()?;
    m.add_class::<RustTokenizer>()?;
    m.add_class::<ViterbiSpan>()?;
    m.add_function(wrap_pyfunction!(rust_viterbi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_diagnostic_viterbi, m)?)?;
    m.add_function(wrap_pyfunction!(rust_viterbi_decode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_tokens_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_ids_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_text_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_text_native, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_text_native_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_forward_backward_expectations, m)?)?;
    m.add_function(wrap_pyfunction!(rust_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(rust_normalize_with_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mine_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rust_pre_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(rust_diagnostic_batch, m)?)?;
    Ok(())
}


