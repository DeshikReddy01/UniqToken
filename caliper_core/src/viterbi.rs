//! Native high-speed Viterbi dynamic programming and forward-backward EM expectation algorithms.

use crate::trie::RustPrefixTrie;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Edge {
    pub prev_node: usize,
    pub token: String,
    pub token_id: Option<u32>,
    pub log_p: f64,
    pub length: usize,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub best_score: f64,
    pub best_edge_idx: Option<usize>,
    pub incoming_edges: Vec<Edge>,
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

/// Computes the 1-best shortest path Viterbi segmentation natively in Rust.
#[pyfunction]
pub fn rust_viterbi_decode(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> PyResult<Vec<ViterbiSpan>> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut nodes: Vec<Node> = (0..=n)
        .map(|_| Node {
            best_score: f64::NEG_INFINITY,
            best_edge_idx: None,
            incoming_edges: Vec::with_capacity(8),
        })
        .collect();

    nodes[0].best_score = 0.0;

    for i in 0..n {
        let current_score = nodes[i].best_score;
        if current_score == f64::NEG_INFINITY {
            continue;
        }

        let slice_str: String = chars[i..].iter().collect();
        let matches = trie.common_prefix_search(&slice_str);

        let mut has_match = false;
        for (tok, tid, log_p, char_len) in matches {
            let next_i = i + char_len;
            if next_i <= n {
                has_match = true;
                let edge = Edge {
                    prev_node: i,
                    token: tok,
                    token_id: tid,
                    log_p,
                    length: char_len,
                };
                let target = &mut nodes[next_i];
                let score = current_score + log_p;
                if score > target.best_score {
                    target.best_score = score;
                    target.best_edge_idx = Some(target.incoming_edges.len());
                }
                target.incoming_edges.push(edge);
            }
        }

        // Byte fallback if no prefix matched
        if !has_match && byte_fallback {
            let ch = chars[i];
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            let mut acc_node = i;
            let mut fallback_score = current_score;

            for &b in encoded.as_bytes() {
                let next_node = if acc_node == i && encoded.len() == 1 {
                    i + 1
                } else {
                    acc_node + 1
                };
                let byte_token = format!("<0x{:02X}>", b);
                let byte_log_p = -15.0; // uniform fallback penalty
                let edge = Edge {
                    prev_node: acc_node,
                    token: byte_token,
                    token_id: None,
                    log_p: byte_log_p,
                    length: 1,
                };

                if next_node <= n {
                    let target = &mut nodes[next_node];
                    let score = fallback_score + byte_log_p;
                    if score > target.best_score {
                        target.best_score = score;
                        target.best_edge_idx = Some(target.incoming_edges.len());
                    }
                    target.incoming_edges.push(edge);
                }
                fallback_score += byte_log_p;
                acc_node = next_node;
            }
        }

        // Optional beam pruning on node incoming edges
        if let Some(k) = max_edges_per_node {
            if nodes[i + 1].incoming_edges.len() > k {
                nodes[i + 1]
                    .incoming_edges
                    .sort_by(|a, b| b.log_p.partial_cmp(&a.log_p).unwrap());
                nodes[i + 1].incoming_edges.truncate(k);
            }
        }
    }

    // Backtrack 1-best path
    let mut spans = Vec::new();
    let mut curr_idx = n;

    while curr_idx > 0 {
        let node = &nodes[curr_idx];
        if let Some(edge_idx) = node.best_edge_idx {
            let edge = &node.incoming_edges[edge_idx];
            let start = edge.prev_node;
            let end = curr_idx;
            spans.push(ViterbiSpan {
                token: edge.token.clone(),
                token_id: edge.token_id,
                start,
                end,
            });
            curr_idx = start;
        } else {
            // Disconnected path fallback
            let start = curr_idx.saturating_sub(1);
            let ch = chars[start];
            spans.push(ViterbiSpan {
                token: ch.to_string(),
                token_id: None,
                start,
                end: curr_idx,
            });
            curr_idx = start;
        }
    }

    spans.reverse();
    Ok(spans)
}

/// Forward-Backward expectation statistics accumulator for Unigram EM iterations.
#[pyfunction]
pub fn rust_forward_backward_expectations(
    text: &str,
    trie: &RustPrefixTrie,
    freq: f64,
) -> PyResult<(HashMap<String, f64>, f64)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Ok((HashMap::new(), 0.0));
    }

    // Build lattice edges
    let mut all_edges: Vec<Edge> = Vec::new();
    for i in 0..n {
        let slice_str: String = chars[i..].iter().collect();
        let matches = trie.common_prefix_search(&slice_str);
        for (tok, tid, log_p, char_len) in matches {
            let next_i = i + char_len;
            if next_i <= n {
                all_edges.push(Edge {
                    prev_node: i,
                    token: tok,
                    token_id: tid,
                    log_p,
                    length: char_len,
                });
            }
        }
    }

    // Alpha (Forward) Pass in log-space
    let mut alpha = vec![f64::NEG_INFINITY; n + 1];
    alpha[0] = 0.0;
    for edge in &all_edges {
        let prev_a = alpha[edge.prev_node];
        if prev_a != f64::NEG_INFINITY {
            let target_node = edge.prev_node + edge.length;
            let val = prev_a + edge.log_p;
            alpha[target_node] = log_add(alpha[target_node], val);
        }
    }

    let total_log_z = alpha[n];
    if total_log_z == f64::NEG_INFINITY {
        return Ok((HashMap::new(), f64::NEG_INFINITY));
    }

    // Beta (Backward) Pass in log-space
    let mut beta = vec![f64::NEG_INFINITY; n + 1];
    beta[n] = 0.0;
    for edge in all_edges.iter().rev() {
        let target_node = edge.prev_node + edge.length;
        let next_b = beta[target_node];
        if next_b != f64::NEG_INFINITY {
            let val = next_b + edge.log_p;
            beta[edge.prev_node] = log_add(beta[edge.prev_node], val);
        }
    }

    // Accumulate posterior expected counts
    let mut expected_counts = HashMap::new();
    for edge in &all_edges {
        let start = edge.prev_node;
        let end = start + edge.length;
        let a = alpha[start];
        let b = beta[end];
        if a != f64::NEG_INFINITY && b != f64::NEG_INFINITY {
            let posterior_log_p = a + edge.log_p + b - total_log_z;
            let posterior_p = posterior_log_p.exp();
            let count = expected_counts.entry(edge.token.clone()).or_insert(0.0);
            *count += posterior_p * freq;
        }
    }

    Ok((expected_counts, total_log_z * freq))
}

#[inline]
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}
