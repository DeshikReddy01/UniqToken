//! High-performance native Prefix Trie implementation for subword matching.

use ahash::AHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Default, Clone)]
pub struct TrieNode {
    pub children: AHashMap<char, TrieNode>,
    pub token: Option<String>,
    pub token_id: Option<u32>,
    pub log_p: f64,
    pub is_terminal: bool,
}

#[pyclass]
#[derive(Default, Clone)]
pub struct RustPrefixTrie {
    root: TrieNode,
}

#[pymethods]
impl RustPrefixTrie {
    #[new]
    pub fn new() -> Self {
        Self {
            root: TrieNode::default(),
        }
    }

    /// Inserts a non-empty subword with a finite log probability.
    pub fn insert(&mut self, token: &str, log_p: f64, token_id: Option<u32>) -> PyResult<()> {
        if token.is_empty() {
            return Err(PyValueError::new_err("token must not be empty"));
        }
        if !log_p.is_finite() {
            return Err(PyValueError::new_err("log_p must be finite"));
        }

        let mut curr = &mut self.root;
        for ch in token.chars() {
            curr = curr.children.entry(ch).or_default();
        }
        curr.is_terminal = true;
        curr.token = Some(token.to_string());
        curr.token_id = token_id;
        curr.log_p = log_p;
        Ok(())
    }

    /// Finds all matching prefixes for a slice of text starting at position 0.
    /// Returns tuples of (token, token_id, log_p, char_length).
    pub fn common_prefix_search(&self, text: &str) -> Vec<(String, Option<u32>, f64, usize)> {
        let mut results = Vec::with_capacity(8);
        let mut curr = &self.root;
        let mut char_count = 0;

        for ch in text.chars() {
            if let Some(next_node) = curr.children.get(&ch) {
                curr = next_node;
                char_count += 1;
                if curr.is_terminal {
                    if let Some(ref tok) = curr.token {
                        results.push((tok.clone(), curr.token_id, curr.log_p, char_count));
                    }
                }
            } else {
                break;
            }
        }

        results
    }

    /// Checks whether the exact token exists in the Trie.
    pub fn contains(&self, token: &str) -> bool {
        self.exact_metadata(token).is_some()
    }
}

impl RustPrefixTrie {
    pub(crate) fn common_prefix_search_chars(
        &self,
        chars: &[char],
        start: usize,
    ) -> Vec<(String, Option<u32>, f64, usize)> {
        let mut results = Vec::with_capacity(8);
        let mut current = &self.root;

        for (offset, ch) in chars[start..].iter().enumerate() {
            let Some(next) = current.children.get(ch) else {
                break;
            };
            current = next;
            if current.is_terminal {
                if let Some(token) = &current.token {
                    results.push((token.clone(), current.token_id, current.log_p, offset + 1));
                }
            }
        }
        results
    }

    pub(crate) fn exact_metadata(&self, token: &str) -> Option<(Option<u32>, f64)> {
        let mut current = &self.root;
        for ch in token.chars() {
            current = current.children.get(&ch)?;
        }
        current.is_terminal.then_some((current.token_id, current.log_p))
    }
}
