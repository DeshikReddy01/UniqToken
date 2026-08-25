use crate::normalizer::rust_normalize;
use crate::trie::RustPrefixTrie;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct RustTokenizer {
    trie: RustPrefixTrie,
    space_char: char,
    byte_fallback: bool,
}

#[pymethods]
impl RustTokenizer {
    #[new]
    #[pyo3(signature = (vocab=None, space_char='\u{2581}', byte_fallback=true))]
    fn new(vocab: Option<Vec<(String, f64, u32)>>, space_char: char, byte_fallback: bool) -> Self {
        let mut trie = RustPrefixTrie::new();
        if let Some(v) = vocab {
            for (tok, logp, id) in v {
                let _ = trie.insert(&tok, logp, Some(id));
            }
        }
        Self { trie, space_char, byte_fallback }
    }

    fn from_vocab(&mut self, vocab: Vec<(String, f64, u32)>) -> PyResult<()> {
        self.trie = RustPrefixTrie::new();
        for (tok, logp, id) in vocab {
            self.trie.insert(&tok, logp, Some(id))?;
        }
        Ok(())
    }

    fn encode(&self, text: &str) -> PyResult<Vec<String>> {
        // ponytail: single-pass norm→regex→viterbi, no Vec<String> chunks
        let norm = rust_normalize(text, self.space_char, true, true, false, false, false, false)?;
        let re = crate::pipeline::get_full_pretok_regex();
        let mut out = Vec::new();
        for m in re.find_iter(&norm) {
            let chunk = m.as_str();
            let chars: Vec<char> = chunk.chars().collect();
            let spans = crate::viterbi::viterbi_decode_chars(&chars, &self.trie, self.byte_fallback, None)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            for s in spans {
                out.push(s.token);
            }
        }
        Ok(out)
    }

    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<String>>> {
        py.allow_threads(|| {
            texts.par_iter().map(|text| {
                let norm = rust_normalize(text, self.space_char, true, true, false, false, false, false).unwrap();
                let re = crate::pipeline::get_full_pretok_regex();
                let mut out = Vec::new();
                for m in re.find_iter(&norm) {
                    let chunk = m.as_str();
                    let chars: Vec<char> = chunk.chars().collect();
                    let spans = crate::viterbi::viterbi_decode_chars(&chars, &self.trie, self.byte_fallback, None)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
                    for s in spans {
                        out.push(s.token);
                    }
                }
                Ok(out)
            }).collect()
        })
    }

    fn encode_ids(&self, text: &str) -> PyResult<Vec<u32>> {
        let norm = rust_normalize(text, self.space_char, true, true, false, false, false, false)?;
        let re = crate::pipeline::get_full_pretok_regex();
        let mut out = Vec::new();
        for m in re.find_iter(&norm) {
            let chunk = m.as_str();
            let chars: Vec<char> = chunk.chars().collect();
            let spans = crate::viterbi::viterbi_decode_chars(&chars, &self.trie, self.byte_fallback, None)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            for s in spans {
                if let Some(id) = s.token_id {
                    out.push(id);
                }
            }
        }
        Ok(out)
    }

    fn encode_ids_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        py.allow_threads(|| {
            texts.par_iter().map(|text| {
                let norm = rust_normalize(text, self.space_char, true, true, false, false, false, false).unwrap();
                let re = crate::pipeline::get_full_pretok_regex();
                let mut out = Vec::new();
                for m in re.find_iter(&norm) {
                    let chunk = m.as_str();
                    let chars: Vec<char> = chunk.chars().collect();
                    let spans = crate::viterbi::viterbi_decode_chars(&chars, &self.trie, self.byte_fallback, None)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
                    for s in spans {
                        if let Some(id) = s.token_id {
                            out.push(id);
                        }
                    }
                }
                Ok(out)
            }).collect()
        })
    }
}

fn rust_encode_tokens_batch_inner(
    chunks: &[String],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> PyResult<Vec<Vec<String>>> {
    use crate::viterbi::viterbi_decode_chars;
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let chars: Vec<char> = chunk.chars().collect();
        let spans = viterbi_decode_chars(&chars, trie, byte_fallback, None).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        out.push(spans.into_iter().map(|s| s.token).collect());
    }
    Ok(out)
}

fn rust_encode_ids_batch_inner(
    chunks: &[String],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> PyResult<Vec<Vec<u32>>> {
    use crate::viterbi::viterbi_decode_chars;
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let chars: Vec<char> = chunk.chars().collect();
        let spans = viterbi_decode_chars(&chars, trie, byte_fallback, None).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let ids: Vec<u32> = spans.into_iter().filter_map(|s| s.token_id).collect();
        out.push(ids);
    }
    Ok(out)
}

#[pyfunction]
pub fn rust_diagnostic_batch(
    py: Python<'_>,
    texts: Vec<String>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> PyResult<HashMap<String, f64>> {
    use std::collections::HashMap;
    use std::time::Instant;
    let start_total = Instant::now();
    let mut t_norm = 0.0;
    let mut t_pre = 0.0;
    let mut t_viterbi = 0.0;
    let mut t_alloc = 0.0;
    let mut total_chunks = 0usize;
    let mut total_tokens = 0usize;
    let mut total_edges = 0usize;
    let mut total_states = 0usize;
    py.allow_threads(|| {
        let re = crate::pipeline::get_full_pretok_regex();
        for text in &texts {
            let t0 = Instant::now();
            let norm = crate::normalizer::rust_normalize(text, '\u{2581}', true, true, false, false, false, false).unwrap();
            t_norm += t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let chunks: Vec<&str> = re.find_iter(&norm).map(|m| m.as_str()).collect();
            t_pre += t0.elapsed().as_secs_f64();
            total_chunks += chunks.len();
            for chunk in chunks {
                let chars: Vec<char> = chunk.chars().collect();
                let t0 = Instant::now();
                let spans = crate::viterbi::viterbi_decode_chars(&chars, trie, byte_fallback, None).unwrap();
                t_viterbi += t0.elapsed().as_secs_f64();
                total_tokens += spans.len();
                // edges/states approx: states = chars.len()+1, edges = spans.len() + fallback
                total_edges += spans.len();
                total_states += chars.len() + 1;
                let t0 = Instant::now();
                let _: Vec<String> = spans.into_iter().map(|s| s.token).collect();
                t_alloc += t0.elapsed().as_secs_f64();
            }
        }
    });
    let total = start_total.elapsed().as_secs_f64();
    let mut out = HashMap::new();
    out.insert("total".to_string(), total);
    out.insert("norm".to_string(), t_norm);
    out.insert("pre".to_string(), t_pre);
    out.insert("viterbi".to_string(), t_viterbi);
    out.insert("alloc".to_string(), t_alloc);
    out.insert("chunks".to_string(), total_chunks as f64);
    out.insert("tokens".to_string(), total_tokens as f64);
    out.insert("edges".to_string(), total_edges as f64);
    out.insert("states".to_string(), total_states as f64);
    Ok(out)
}
