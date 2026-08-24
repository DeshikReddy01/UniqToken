use ahash::AHashMap;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

fn detect_script(token: &str) -> &'static str {
    for ch in token.chars() {
        let cp = ch as u32;
        if cp == 0x2581 || cp == 0x0020 || cp == 0x0009 || cp == 0x000A || cp == 0x000D || (0xE000 <= cp && cp <= 0xF8FF) {
            continue;
        }
        if (0x0041 <= cp && cp <= 0x005A) || (0x0061 <= cp && cp <= 0x007A) || (0x00C0 <= cp && cp <= 0x024F) {
            return "latin";
        } else if 0x0900 <= cp && cp <= 0x097F {
            return "devanagari";
        } else if 0x0C00 <= cp && cp <= 0x0C7F {
            return "telugu";
        } else if 0x0B80 <= cp && cp <= 0x0BFF {
            return "tamil";
        } else if 0x0980 <= cp && cp <= 0x09FF {
            return "bengali";
        } else if 0x0900 <= cp && cp <= 0x0D7F {
            return "indic_other";
        } else if (0x4E00 <= cp && cp <= 0x9FFF) || (0x3400 <= cp && cp <= 0x4DBF) || (0x3040 <= cp && cp <= 0x30FF) || (0xAC00 <= cp && cp <= 0xD7AF) {
            return "cjk";
        } else if (0x0600 <= cp && cp <= 0x06FF) || (0x0750 <= cp && cp <= 0x077F) {
            return "arabic";
        } else if (0x0400 <= cp && cp <= 0x04FF) || (0x0500 <= cp && cp <= 0x052F) {
            return "cyrillic";
        } else if 0x0E00 <= cp && cp <= 0x0E7F {
            return "thai";
        } else if ch.is_ascii_digit() || token.starts_with("0x") || token.starts_with("SYS_") {
            return "numeric";
        }
        return "symbol";
    }
    "symbol"
}

fn max_ngram_for_chunk(chunk: &str, default_max: usize) -> usize {
    if detect_script(chunk) == "cjk" {
        std::cmp::min(default_max, 4)
    } else {
        default_max
    }
}

#[pyfunction]
#[pyo3(signature = (chunk_counts, max_ngram_length, special_tokens=None))]
pub fn rust_mine_ngrams(
    chunk_counts: HashMap<String, usize>,
    max_ngram_length: usize,
    special_tokens: Option<HashSet<String>>,
) -> PyResult<HashMap<String, usize>> {
    let specials = special_tokens.unwrap_or_default();
    let mut ngram_counts: AHashMap<String, usize> = AHashMap::with_capacity(chunk_counts.len() * 8);
    for (chunk, freq) in chunk_counts {
        if specials.contains(&chunk) {
            continue;
        }
        if chunk.starts_with("<|") && chunk.ends_with("|>") {
            continue;
        }
        let chars: Vec<char> = chunk.chars().collect();
        let clen = chars.len();
        let max_len = max_ngram_for_chunk(&chunk, max_ngram_length);
        for start in 0..clen {
            let mut end_limit = clen + 1;
            let ml = start + max_len + 1;
            if ml < end_limit {
                end_limit = ml;
            }
            for end in (start + 1)..end_limit {
                let piece: String = chars[start..end].iter().collect();
                *ngram_counts.entry(piece).or_insert(0) += freq;
            }
        }
    }
    Ok(ngram_counts.into_iter().collect())
}
