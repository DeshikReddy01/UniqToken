from typing import Dict, List, Optional, Sequence, Tuple

class RustPrefixTrie:
    def __init__(self, items: Optional[List[Tuple[str, Optional[int], float]]] = None) -> None: ...
    def insert(self, token: str, token_id: Optional[int], score: float) -> None: ...
    def common_prefix_search_chars(self, chars: Sequence[str], start_idx: int = 0) -> List[Tuple[str, Optional[int], float, int]]: ...
    def exact_metadata(self, token: str) -> Optional[Tuple[Optional[int], float]]: ...
    def __len__(self) -> int: ...

class ViterbiSpan:
    token: str
    token_id: Optional[int]
    start: int
    end: int

def rust_viterbi_decode(
    text: str,
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[ViterbiSpan]: ...

def rust_viterbi_decode_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[List[ViterbiSpan]]: ...

def rust_encode_tokens_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[List[str]]: ...

def rust_encode_ids_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[List[int]]: ...

def rust_encode_text_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    space_char: str = " ",
) -> List[List[int]]: ...

def rust_forward_backward_expectations(
    trie: RustPrefixTrie,
    texts: Sequence[str],
    frequencies: Sequence[int],
    vocab_size: int,
    byte_fallback: bool = True,
) -> Tuple[List[float], float]: ...

def rust_normalize(
    text: str,
    space_char: str = " ",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> str: ...

def rust_normalize_with_alignment(
    text: str,
    space_char: str = " ",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> Tuple[str, List[Tuple[int, int]]]: ...

def rust_mine_ngrams(
    texts: Sequence[str],
    min_len: int = 1,
    max_len: int = 16,
    min_freq: int = 2,
) -> Dict[str, int]: ...

def rust_pre_tokenize(
    text: str,
) -> List[str]: ...
