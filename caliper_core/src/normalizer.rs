use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use unicode_normalization::UnicodeNormalization;

const ESCAPE_PREFIX: char = '\u{E000}';
const ESCAPED_METASPACE: char = '\u{E001}';

static COMBINING_CACHE: OnceLock<Mutex<HashMap<u32, bool>>> = OnceLock::new();

fn is_combining(c: char) -> bool {
    if (c as u32) < 128 {
        return false;
    }
    let cache = COMBINING_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cache.lock().unwrap();
        if let Some(&v) = map.get(&(c as u32)) {
            return v;
        }
    }
    let v = Python::with_gil(|py| {
        let unicodedata = py.import("unicodedata").unwrap();
        let comb: u32 = unicodedata
            .getattr("combining")
            .unwrap()
            .call1((c.to_string(),))
            .unwrap()
            .extract()
            .unwrap();
        comb != 0
    });
    cache.lock().unwrap().insert(c as u32, v);
    v
}

fn punct_map(c: char) -> Option<&'static str> {
    match c {
        '\u{201C}' | '\u{201D}' | '\u{201E}' => Some("\""),
        '\u{2018}' | '\u{2019}' | '\u{201A}' => Some("'"),
        '\u{2014}' | '\u{2013}' | '\u{2212}' => Some("-"),
        '\u{2026}' => Some("..."),
        _ => None,
    }
}

fn is_unicode_space(c: char) -> bool {
    matches!(c,
        '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

#[pyfunction]
#[pyo3(signature = (text, space_char='\u{2581}', normalize_unicode=true, normalize_unicode_spaces=true, normalize_punctuation=false, lowercase=false, collapse_whitespaces=false, strip_whitespace=false))]
pub fn rust_normalize(
    text: &str,
    space_char: char,
    normalize_unicode: bool,
    normalize_unicode_spaces: bool,
    normalize_punctuation: bool,
    lowercase: bool,
    collapse_whitespaces: bool,
    strip_whitespace: bool,
) -> PyResult<String> {
    // token-only path — no alignment, ~1.33× faster than with_alignment for ASCII
    let mut s = if normalize_unicode {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut out = String::with_capacity(n + 8);
        let mut i = 0usize;
        while i < n {
            let mut end = i + 1;
            while end < n && is_combining(chars[end]) {
                end += 1;
            }
            let slice: String = chars[i..end].iter().collect();
            let norm = if slice.is_ascii() {
                slice
            } else {
                Python::with_gil(|py| {
                    let unicodedata = py.import("unicodedata").unwrap();
                    unicodedata
                        .getattr("normalize")
                        .unwrap()
                        .call1(("NFKC", slice))
                        .unwrap()
                        .extract()
                        .unwrap()
                })
            };
            out.push_str(&norm);
            i = end;
        }
        out
    } else {
        text.to_string()
    };
    if normalize_unicode_spaces {
        s = s.chars().map(|c| if is_unicode_space(c) { ' ' } else { c }).collect();
    }
    if normalize_punctuation {
        let mut t = String::with_capacity(s.len() + 4);
        for ch in s.chars() {
            if let Some(m) = punct_map(ch) {
                t.push_str(m);
            } else {
                t.push(ch);
            }
        }
        s = t;
    }
    if lowercase {
        s = s.to_lowercase();
    }
    if collapse_whitespaces {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == ' ' || ch == '\t' {
                out.push(' ');
                while matches!(chars.peek(), Some(' ') | Some('\t')) {
                    chars.next();
                }
            } else {
                out.push(ch);
            }
        }
        s = out;
    }
    if strip_whitespace {
        s = s.trim().to_string();
    }
    // metaspace escape
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        if ch == ESCAPE_PREFIX {
            out.push(ESCAPE_PREFIX);
            out.push(ESCAPE_PREFIX);
        } else if ch == space_char {
            out.push(ESCAPE_PREFIX);
            out.push(ESCAPED_METASPACE);
        } else if ch == ' ' {
            out.push(space_char);
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, space_char='\u{2581}', normalize_unicode=true, normalize_unicode_spaces=true, normalize_punctuation=false, lowercase=false, collapse_whitespaces=false, strip_whitespace=false))]
pub fn rust_normalize_with_alignment(
    text: &str,
    space_char: char,
    normalize_unicode: bool,
    normalize_unicode_spaces: bool,
    normalize_punctuation: bool,
    lowercase: bool,
    collapse_whitespaces: bool,
    strip_whitespace: bool,
) -> PyResult<(String, Vec<(usize, usize)>)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut units: Vec<(char, (usize, usize))> = Vec::with_capacity(n + 8);

    if normalize_unicode {
        let mut i = 0usize;
        while i < n {
            let mut end = i + 1;
            while end < n && is_combining(chars[end]) {
                end += 1;
            }
            let slice: String = chars[i..end].iter().collect();
            // fast path ASCII -> NFKC is identity
            let normalized: String = if slice.is_ascii() {
                slice
            } else {
                Python::with_gil(|py| {
                    let unicodedata = py.import("unicodedata").unwrap();
                    unicodedata
                        .getattr("normalize")
                        .unwrap()
                        .call1(("NFKC", slice))
                        .unwrap()
                        .extract()
                        .unwrap()
                })
            };
            let span = (i, end);
            for ch in normalized.chars() {
                units.push((ch, span));
            }
            i = end;
        }
    } else {
        for (i, &ch) in chars.iter().enumerate() {
            units.push((ch, (i, i + 1)));
        }
    }

    if normalize_unicode_spaces {
        for (ch, _) in units.iter_mut() {
            if is_unicode_space(*ch) {
                *ch = ' ';
            }
        }
    }

    if normalize_punctuation {
        let mut translated: Vec<(char, (usize, usize))> = Vec::with_capacity(units.len() + 4);
        for (ch, span) in units {
            if let Some(mapped) = punct_map(ch) {
                for c in mapped.chars() {
                    translated.push((c, span));
                }
            } else {
                translated.push((ch, span));
            }
        }
        units = translated;
    }

    if lowercase {
        let mut lowered: Vec<(char, (usize, usize))> = Vec::with_capacity(units.len() + 4);
        for (ch, span) in units {
            let lower = ch.to_lowercase().collect::<Vec<char>>();
            if lower.len() == 1 {
                lowered.push((lower[0], span));
            } else {
                for c in lower {
                    lowered.push((c, span));
                }
            }
        }
        units = lowered;
    }

    if collapse_whitespaces {
        let mut collapsed: Vec<(char, (usize, usize))> = Vec::with_capacity(units.len());
        let mut i = 0usize;
        while i < units.len() {
            let (ch, span) = units[i];
            if ch != ' ' && ch != '\t' {
                collapsed.push((ch, span));
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < units.len() && (units[end].0 == ' ' || units[end].0 == '\t') {
                end += 1;
            }
            // span from first to last in run
            let last_span = units[end - 1].1;
            collapsed.push((' ', (span.0, last_span.1)));
            i = end;
        }
        units = collapsed;
    }

    if strip_whitespace {
        let mut start = 0usize;
        let mut end = units.len();
        while start < end && units[start].0.is_whitespace() {
            start += 1;
        }
        while end > start && units[end - 1].0.is_whitespace() {
            end -= 1;
        }
        units = units[start..end].to_vec();
    }

    // metaspace escape
    let mut escaped: Vec<(char, (usize, usize))> = Vec::with_capacity(units.len() + 4);
    for (ch, span) in units {
        if ch == ESCAPE_PREFIX {
            escaped.push((ESCAPE_PREFIX, span));
            escaped.push((ESCAPE_PREFIX, span));
        } else if ch == space_char {
            escaped.push((ESCAPE_PREFIX, span));
            escaped.push((ESCAPED_METASPACE, span));
        } else if ch == ' ' {
            escaped.push((space_char, span));
        } else {
            escaped.push((ch, span));
        }
    }

    let out: String = escaped.iter().map(|(c, _)| *c).collect();
    let align: Vec<(usize, usize)> = escaped.into_iter().map(|(_, s)| s).collect();
    Ok((out, align))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nfkc_ligature() {
        let (s, a) = rust_normalize_with_alignment("ﬁ", '\u{2581}', true, true, false, false, false, false).unwrap();
        assert_eq!(s, "fi");
        assert_eq!(a, vec![(0,1),(0,1)]);
    }
    #[test]
    fn combining() {
        let (s, a) = rust_normalize_with_alignment("A\u{030A}", '\u{2581}', true, true, false, false, false, false).unwrap();
        assert_eq!(s, "\u{00C5}");
        assert_eq!(a, vec![(0,2)]);
    }
}
