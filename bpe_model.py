from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from byte_codec import ByteFallbackEngine


class BPEModel:
    """
    Byte-Pair Encoding (BPE) Subword Model.

    Implements iterative pair merging based on learned merge priority ranks
    (Tiktoken / GPT-4 / BPE standard).
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
        # Resolve the OOV token from the actual vocab; a hardcoded "<|unk|>"
        # could be absent from custom special-token configurations.
        if "<|unk|>" in token_to_id:
            self._unk_token: Optional[str] = "<|unk|>"
        elif self.special_tokens:
            self._unk_token = self.special_tokens[0]
        else:
            self._unk_token = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        return set(zip(word[:-1], word[1:]))

    def _encode_word(self, word: str) -> List[str]:
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
                # No unk configured — keep the raw character so it maps through
                # unk_id at the ID stage instead of emitting an out-of-vocab token.
                symbols.append(char)

        if len(symbols) <= 1:
            return symbols

        while len(symbols) > 1:
            pairs = self._get_pairs(symbols)
            best_pair = min(
                pairs,
                key=lambda p: self.merges.get(p, float("inf")),
            )

            if best_pair not in self.merges:
                break

            first, second = best_pair
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

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
                tokens.append(" ")
            tokens.extend(self._encode_word(word))
        return tokens

    def encode_to_ids(self, text: str) -> List[int]:
        tokens = self.encode(text)
        unk_id = self.token_to_id.get("<|unk|>", 0)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, token_ids: List[int], space_char: str = "\u2581") -> str:
        """Decodes integer token IDs back to a human-readable string."""
        tokens = [self.id_to_token.get(t, "") for t in token_ids]
        return ByteFallbackEngine.decode_tokens(tokens, space_char=space_char)
