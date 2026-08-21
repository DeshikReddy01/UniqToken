//! High-performance native end-to-end normalization, pre-tokenization, and batch encoding pipeline.

use crate::trie::RustPrefixTrie;
use crate::viterbi::rust_viterbi_decode;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

const DEFAULT_BYTE_LOG_P: f64 = -10.0;

static PRETOK_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_pretok_regex() -> &'static Regex {
    PRETOK_REGEX.get_or_init(|| {
        // High-speed unicode-aware word/punctuation/whitespace splitting regex
        Regex::new(r"<\|[^\s|]+\|>|\p{L}+(?:['’]\p{L}+)*|\p{N}+|[^\s\p{L}\p{N}\u{2581}]+|\u{2581}+|\s+").unwrap()
    })
}

/// Normalizes a single string directly in native Rust.
pub fn normalize_string_native(text: &str, space_char: char) -> String {
    let mut normalized = String::with_capacity(text.len() + 8);
    let nfkc: String = text.nfkc().collect();
    for ch in nfkc.chars() {
        if ch.is_whitespace() {
            normalized.push(space_char);
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

/// Normalizes and pre-tokenizes a string into discrete subword chunks natively in Rust.
pub fn pre_tokenize_native(text: &str, space_char: char) -> Vec<String> {
    let normalized = normalize_string_native(text, space_char);
    let re = get_pretok_regex();
    re.find_iter(&normalized)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Native end-to-end pipeline: raw texts -> normalize -> regex pre-tokenize -> Viterbi DAG -> token IDs.
/// Executes completely in parallel across all CPU cores with the Python GIL released.
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback=true, space_char=' '))]
pub fn rust_encode_text_batch(
    py: Python<'_>,
    texts: Vec<String>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    space_char: char,
) -> PyResult<Vec<Vec<u32>>> {
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|raw_text| {
                let chunks = pre_tokenize_native(raw_text, space_char);
                let mut sentence_ids: Vec<u32> = Vec::with_capacity(chunks.len() * 2);

                for chunk in chunks {
                    if let Ok(spans) = rust_viterbi_decode(&chunk, trie, byte_fallback, None) {
                        for span in spans {
                            if let Some(id) = span.token_id {
                                sentence_ids.push(id);
                            }
                        }
                    }
                }
                Ok(sentence_ids)
            })
            .collect()
    })
}
