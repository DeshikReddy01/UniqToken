//! Caliper Core: High-performance native Rust acceleration module for Caliper Tokenizer.

pub mod trie;
pub mod viterbi;

use pyo3::prelude::*;
use trie::RustPrefixTrie;
use viterbi::{rust_forward_backward_expectations, rust_viterbi_decode, ViterbiSpan};

/// Native Caliper Core Python Module
#[pymodule]
fn caliper_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPrefixTrie>()?;
    m.add_class::<ViterbiSpan>()?;
    m.add_function(wrap_pyfunction!(rust_viterbi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_forward_backward_expectations, m)?)?;
    Ok(())
}
