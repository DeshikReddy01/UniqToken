from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

from byte_codec import ByteFallbackEngine


class BPEModel:
    """
    Byte-Pair Encoding (BPE) Subword Model.

    Implements iterative pair merging based on learned merge priority ranks
    (Tiktoken / GPT-4 / BPE standard). The inference path uses a
    rank-priority heap over adjacent pairs, giving O(len * log len) per
    word instead of the naive O(merges * len).
    """

    def __init__(
        self,
        vocab: Set[str],
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        merges: Dict[Tuple[str, str], int],
        special_tokens: Optional[List[str]] = None,
        byte_fallback: bool = True,
    ):
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.merges = merges
        self.special_tokens = list(special_tokens or [])
        self.byte_fallback = byte_fallback
        if "<|unk|>" in token_to_id:
            self._unk_token: Optional[str] = "<|unk|>"
        elif self.special_tokens:
            self._unk_token = self.special_tokens[0]
        else:
            self._unk_token = None
        if " " in token_to_id and " " in vocab:
            self._space_token = " "
        elif byte_fallback:
            self._space_token = ByteFallbackEngine.byte_to_token(32)
        else:
            self._space_token = " "

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        return set(zip(word[:-1], word[1:]))

    def _build_symbols(self, word: str) -> List[str]:
        if not word:
            return []
        symbols: List[str] = []
        for char in word:
            if char in self.vocab:
                symbols.append(char)
            elif self.byte_fallback:
                symbols.extend(ByteFallbackEngine.char_to_byte_tokens(char))
            elif self._unk_token is not None:
                symbols.append(self._unk_token)
            else:
                symbols.append(char)
        return symbols

    def _encode_word_heap(self, symbols: List[str]) -> List[str]:
        """Rank-priority BPE encode using a min-heap of adjacent pairs.

        Each heap entry is ``(rank, counter, left, right)``. Popping the
        smallest-rank pair applies a merge, and the two neighbouring
        pairs (left-of-merged and merged-of-right) are pushed with their
        cached ranks. The ``counter`` makes heap entries unique without
        hashing the strings on every comparison. Positions whose merge
        already happened are removed lazily: a stale entry is detected
        by comparing the popped rank against the live ``self.merges`` value
        for the popped pair, which is a fast dict lookup.
        """
        if len(symbols) <= 1:
            return list(symbols)

        ranks = self.merges
        if not ranks:
            return list(symbols)

        syms: List[str] = list(symbols)
        heap: List[Tuple[int, int, str, str]] = []
        counter = 0
        for i in range(len(syms) - 1):
            pair = (syms[i], syms[i + 1])
            rank = ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, counter, pair[0], pair[1]))
                counter += 1

        merge_stamp = 0
        while heap and merge_stamp < len(ranks):
            rank, _, left, right = heapq.heappop(heap)
            live = ranks.get((left, right), -1)
            if rank != live:
                continue
            merged = left + right
            new_syms: List[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == left and syms[i + 1] == right:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
            merge_stamp += 1
            for k in range(len(syms) - 1):
                pr = ranks.get((syms[k], syms[k + 1]))
                if pr is not None:
                    heapq.heappush(heap, (pr, counter, syms[k], syms[k + 1]))
                    counter += 1
        return syms

    def _encode_word(self, word: str) -> List[str]:
        symbols = self._build_symbols(word)
        if len(symbols) <= 1:
            return symbols
        return self._encode_word_heap(symbols)

    def encode(self, text: str) -> List[str]:
        """
        Segments text by applying BPE merges on whitespace-delimited word tokens.
        """
        if not text:
            return []

        words = text.split(" ")
        tokens: List[str] = []
        for idx, word in enumerate(words):
            if idx > 0:
                tokens.append(self._space_token)
            tokens.extend(self._encode_word(word))
        return tokens

    def encode_to_ids(self, text: str) -> List[int]:
        tokens = self.encode(text)
        unk_id = self.token_to_id.get("<|unk|>", 0)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, token_ids: List[int], space_char: str = "\u2581", strict: bool = False) -> str:
        """
        Decodes integer token IDs back to a human-readable string.

        strict=False (default, lenient): unknown IDs are skipped silently.
        strict=True: raises ValueError naming the first invalid ID.
        """
        tokens: List[str] = []
        for t in token_ids:
            tok = self.id_to_token.get(t)
            if tok is None:
                if strict:
                    raise ValueError(f"token id {t} is not in the model vocabulary")
                continue  # lenient: unknown IDs contribute nothing
            tokens.append(tok)
        return ByteFallbackEngine.decode_tokens(tokens, space_char=space_char)
