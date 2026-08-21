//! Native Viterbi decoding and forward-backward expectation algorithms.

use crate::trie::RustPrefixTrie;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use rayon::prelude::*;

const DEFAULT_BYTE_LOG_P: f64 = -10.0;

#[derive(Clone, Debug)]
struct TokenPiece {
    token: String,
    token_id: Option<u32>,
}

#[derive(Clone, Debug)]
struct Edge {
    prev_node: usize,
    pieces: Vec<TokenPiece>,
    log_p: f64,
    length: usize,
}

#[derive(Clone, Debug)]
struct Node {
    best_score: f64,
    best_edge: Option<Edge>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ViterbiSpan {
    #[pyo3(get)]
    pub token: String,
    #[pyo3(get)]
    pub token_id: Option<u32>,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

fn viterbi_decode_chars(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    let n = chars.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut incoming: Vec<Vec<Edge>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        let matches = trie.common_prefix_search_chars(chars, i);
        if matches.is_empty() && byte_fallback {
            let mut pieces = Vec::new();
            let mut edge_log_p = 0.0;
            let mut encoded_buf = [0_u8; 4];
            for byte in chars[i].encode_utf8(&mut encoded_buf).as_bytes() {
                let token = format!("<0x{byte:02X}>");
                let (token_id, log_p) = trie
                    .exact_metadata(&token)
                    .unwrap_or((None, DEFAULT_BYTE_LOG_P));
                pieces.push(TokenPiece { token, token_id });
                edge_log_p += log_p;
            }
            incoming[i + 1].push(Edge {
                prev_node: i,
                pieces,
                log_p: edge_log_p,
                length: 1,
            });
        } else {
            for (token, token_id, log_p, char_len) in matches {
                let end = i + char_len;
                if end <= n {
                    incoming[end].push(Edge {
                        prev_node: i,
                        pieces: vec![TokenPiece { token, token_id }],
                        log_p,
                        length: char_len,
                    });
                }
            }
        }
    }

    if let Some(limit) = max_edges_per_node {
        for edges in incoming.iter_mut().skip(1) {
            if edges.len() > limit {
                edges.sort_by(|left, right| right.log_p.total_cmp(&left.log_p));
                edges.truncate(limit);
            }
        }
    }

    let mut nodes: Vec<Node> = (0..=n)
        .map(|_| Node {
            best_score: f64::NEG_INFINITY,
            best_edge: None,
        })
        .collect();
    nodes[0].best_score = 0.0;

    for end in 1..=n {
        for edge in &incoming[end] {
            let previous_score = nodes[edge.prev_node].best_score;
            if previous_score == f64::NEG_INFINITY {
                continue;
            }
            let score = previous_score + edge.log_p;
            if score > nodes[end].best_score {
                nodes[end].best_score = score;
                nodes[end].best_edge = Some(edge.clone());
            }
        }
    }

    if nodes[n].best_edge.is_none() {
        return Err(format!(
            "lattice disconnected at character index {n}; enable byte fallback or provide complete vocabulary coverage"
        ));
    }

    let mut spans = Vec::new();
    let mut end = n;
    while end > 0 {
        let edge = nodes[end].best_edge.as_ref().ok_or_else(|| {
            format!("lattice backpointer missing at character index {end}")
        })?;
        for piece in edge.pieces.iter().rev() {
            spans.push(ViterbiSpan {
                token: piece.token.clone(),
                token_id: piece.token_id,
                start: edge.prev_node,
                end,
            });
        }
        end = edge.prev_node;
    }

    spans.reverse();
    Ok(spans)
}

fn viterbi_ids_chars(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<u32>, String> {
    let spans = viterbi_decode_chars(chars, trie, byte_fallback, max_edges_per_node)?;
    let mut ids = Vec::with_capacity(spans.len());
    for s in spans {
        if let Some(id) = s.token_id {
            ids.push(id);
        }
    }
    Ok(ids)
}

/// Computes the most probable segmentation using Python character offsets.
#[pyfunction]
#[pyo3(signature = (text, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_viterbi_decode(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> PyResult<Vec<ViterbiSpan>> {
    if matches!(max_edges_per_node, Some(0)) {
        return Err(PyValueError::new_err(
            "max_edges_per_node must be greater than zero",
        ));
    }
    let chars: Vec<char> = text.chars().collect();
    viterbi_decode_chars(&chars, trie, byte_fallback, max_edges_per_node)
        .map_err(PyValueError::new_err)
}

/// Computes most probable segmentations for a batch of strings concurrently using Rayon (releases GIL).
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_viterbi_decode_batch(
    py: Python<'_>,
    texts: Vec<String>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> PyResult<Vec<Vec<ViterbiSpan>>> {
    if matches!(max_edges_per_node, Some(0)) {
        return Err(PyValueError::new_err(
            "max_edges_per_node must be greater than zero",
        ));
    }
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|text| {
                let chars: Vec<char> = text.chars().collect();
                viterbi_decode_chars(&chars, trie, byte_fallback, max_edges_per_node)
                    .map_err(PyValueError::new_err)
            })
            .collect()
    })
}

/// Batch encodes strings directly to token integer IDs using parallel Rayon workers (releases GIL).
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_encode_ids_batch(
    py: Python<'_>,
    texts: Vec<String>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> PyResult<Vec<Vec<u32>>> {
    if matches!(max_edges_per_node, Some(0)) {
        return Err(PyValueError::new_err(
            "max_edges_per_node must be greater than zero",
        ));
    }
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|text| {
                let chars: Vec<char> = text.chars().collect();
                viterbi_ids_chars(&chars, trie, byte_fallback, max_edges_per_node)
                    .map_err(PyValueError::new_err)
            })
            .collect()
    })
}


/// Forward-backward statistics for text fully covered by the supplied trie.
#[pyfunction]
pub fn rust_forward_backward_expectations(
    text: &str,
    trie: &RustPrefixTrie,
    freq: f64,
) -> PyResult<(HashMap<String, f64>, f64)> {
    if !freq.is_finite() || freq < 0.0 {
        return Err(PyValueError::new_err(
            "freq must be finite and non-negative",
        ));
    }

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 || freq == 0.0 {
        return Ok((HashMap::new(), 0.0));
    }

    let mut all_edges: Vec<Edge> = Vec::new();
    for i in 0..n {
        for (token, token_id, log_p, char_len) in trie.common_prefix_search_chars(&chars, i) {
            if i + char_len <= n {
                all_edges.push(Edge {
                    prev_node: i,
                    pieces: vec![TokenPiece { token, token_id }],
                    log_p,
                    length: char_len,
                });
            }
        }
    }

    let mut alpha = vec![f64::NEG_INFINITY; n + 1];
    alpha[0] = 0.0;
    for edge in &all_edges {
        let previous = alpha[edge.prev_node];
        if previous != f64::NEG_INFINITY {
            let end = edge.prev_node + edge.length;
            alpha[end] = log_add(alpha[end], previous + edge.log_p);
        }
    }

    let total_log_z = alpha[n];
    if total_log_z == f64::NEG_INFINITY {
        return Err(PyValueError::new_err(
            "lattice is disconnected; forward-backward requires complete trie coverage",
        ));
    }

    let mut beta = vec![f64::NEG_INFINITY; n + 1];
    beta[n] = 0.0;
    for edge in all_edges.iter().rev() {
        let end = edge.prev_node + edge.length;
        if beta[end] != f64::NEG_INFINITY {
            beta[edge.prev_node] = log_add(beta[edge.prev_node], beta[end] + edge.log_p);
        }
    }

    let mut expected_counts = HashMap::new();
    for edge in &all_edges {
        let end = edge.prev_node + edge.length;
        if alpha[edge.prev_node] == f64::NEG_INFINITY || beta[end] == f64::NEG_INFINITY {
            continue;
        }
        let posterior = (alpha[edge.prev_node] + edge.log_p + beta[end] - total_log_z).exp();
        let token = &edge.pieces[0].token;
        *expected_counts.entry(token.clone()).or_insert(0.0) += posterior * freq;
    }

    Ok((expected_counts, total_log_z * freq))
}

#[inline]
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a == f64::INFINITY || b == f64::INFINITY {
        f64::INFINITY
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_add_handles_infinities() {
        assert_eq!(log_add(f64::NEG_INFINITY, -2.0), -2.0);
        assert_eq!(log_add(f64::INFINITY, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn multibyte_fallback_emits_every_byte_with_one_character_span() {
        let trie = RustPrefixTrie::new();
        let spans = rust_viterbi_decode("éa", &trie, true, None).unwrap();
        let tokens: Vec<&str> = spans.iter().map(|span| span.token.as_str()).collect();
        assert_eq!(tokens, vec!["<0xC3>", "<0xA9>", "<0x61>"]);
        assert_eq!((spans[0].start, spans[0].end), (0, 1));
        assert_eq!((spans[1].start, spans[1].end), (0, 1));
        assert_eq!((spans[2].start, spans[2].end), (1, 2));
    }

    #[test]
    fn pruning_keeps_stable_backpointers() {
        let mut trie = RustPrefixTrie::new();
        trie.insert("a", -1.0, Some(1)).unwrap();
        trie.insert("ba", -0.1, Some(2)).unwrap();
        trie.insert("ab", -0.2, Some(3)).unwrap();
        trie.insert("b", -1.0, Some(4)).unwrap();
        let spans = rust_viterbi_decode("aba", &trie, false, Some(1)).unwrap();
        assert!(!spans.is_empty());
        assert_eq!(spans.last().unwrap().end, 3);
    }

    #[test]
    fn rejects_zero_beam_size() {
        let trie = RustPrefixTrie::new();
        assert!(rust_viterbi_decode("a", &trie, true, Some(0)).is_err());
    }
}
