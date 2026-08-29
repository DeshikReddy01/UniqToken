//! High-performance native end-to-end normalization, pre-tokenization, and batch encoding pipeline.

use crate::trie::RustPrefixTrie;
use crate::viterbi::rust_viterbi_decode;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

static PRETOK_REGEX: OnceLock<Regex> = OnceLock::new();
static PRETOK_FULL_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_pretok_regex() -> &'static Regex {
    PRETOK_REGEX.get_or_init(|| {
        // High-speed unicode-aware word/punctuation/whitespace splitting regex
        Regex::new(r"<\|[^\s|]+\|>|\p{L}+(?:['’]\p{L}+)*|\p{N}+|[^\s\p{L}\p{N}\u{2581}]+|\u{2581}+|\s+").unwrap()
    })
}

pub(crate) fn get_full_pretok_regex() -> &'static Regex {
    PRETOK_FULL_REGEX.get_or_init(|| {
        let escaped_space = regex::escape("\u{2581}");
        let special_token = r"<\|[^\s|]+\|>";
        let url = r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)";
        let email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+";
        let hashtag = format!(r"{}?#\w+", escaped_space);
        let mention = format!(r"{}?@\w+", escaped_space);
        let emoji = r"(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?(?:\u200D(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?)*";
        let cjk = format!(r"{}?[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+", escaped_space);
        let word = format!(r"{}?[^\W\d_\s{}]+(?:['’][^\W\d_\s{}]+)*", escaped_space, escaped_space, escaped_space);
        let hex_number = format!(r"{}?0[xX][0-9a-fA-F]+|{}?0[bB][01]+", escaped_space, escaped_space);
        let number = format!(r"{}?\d+", escaped_space);
        let space_marker = format!(r"{}+", escaped_space);
        let whitespace = r"\s+";
        let punctuation = format!(r"{}?[^\w\s{}]|{}?_", escaped_space, escaped_space, escaped_space);
        let patterns = vec![
            special_token.to_string(),
            url.to_string(),
            email.to_string(),
            hashtag,
            mention,
            emoji.to_string(),
            cjk,
            word,
            hex_number,
            number,
            space_marker,
            whitespace.to_string(),
            punctuation,
        ];
        let combined = patterns.into_iter().map(|p| format!("(?:{})", p)).collect::<Vec<_>>().join("|");
        Regex::new(&combined).unwrap()
    })
}

#[pyfunction]
pub fn rust_pre_tokenize(text: &str) -> Vec<String> {
    let re = get_full_pretok_regex();
    re.find_iter(text).map(|m| m.as_str().to_string()).collect()
}

/// Characters Python's `Normalizer` maps to whitespace replacements.
///
/// Mirrors `Normalizer.UNICODE_SPACES` plus the ASCII space; tabs, newlines,
/// and carriage returns are deliberately NOT mapped here so the pre-tokenizer
/// regex handles them exactly like the Python single-encode path.
fn is_python_unicode_space(ch: char) -> bool {
    matches!(
        ch,
        '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// Normalizes a single string directly in native Rust.
///
/// Semantics mirror Python's `Normalizer.normalize` + metaspace substitution:
/// NFKC first, then only the configured unicode-space set (and plain space)
/// become the metaspace character. Other whitespace is left for the
/// pre-tokenizer's `\s` handling to keep Rust/Python parity on tabs/newlines.
pub fn normalize_string_native(text: &str, space_char: char) -> String {
    let mut normalized = String::with_capacity(text.len() + 8);
    let nfkc: String = text.nfkc().collect();
    for ch in nfkc.chars() {
        if ch == ' ' || is_python_unicode_space(ch) {
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
            .enumerate()
            .map(|(idx, raw_text)| {
                let chunks = pre_tokenize_native(raw_text, space_char);
                let mut sentence_ids: Vec<u32> = Vec::with_capacity(chunks.len() * 2);

                for chunk in chunks {
                    match rust_viterbi_decode(&chunk, trie, byte_fallback, None) {
                        Ok(spans) => {
                            for span in spans {
                                if let Some(id) = span.token_id {
                                    sentence_ids.push(id);
                                }
                            }
                        }
                        // Never silently turn a disconnected lattice into an
                        // empty or partial sequence — propagate with context.
                        Err(err) => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "rust_encode_text_batch: Viterbi decode failed for input #{} (chunk {:?}): {}",
                                idx, chunk, err
                            )));
                        }
                    }
                }
                Ok(sentence_ids)
            })
            .collect()
    })
}
