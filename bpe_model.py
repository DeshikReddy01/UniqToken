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

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        return set(zip(word[:-1], word[1:]))

    def encode(self, text: str) -> List[str]:
        """
        Segments text by greedily applying BPE merges in rank order.
        """
        if not text:
            return []

        # Start with base characters
        symbols: List[str] = []
        for char in text:
            if char in self.vocab:
                symbols.append(char)
            elif self.byte_fallback:
                symbols.extend(ByteFallbackEngine.char_to_byte_tokens(char))
            else:
                symbols.append("<|unk|>")

        if len(symbols) <= 1:
            return symbols

        # ponytail: greedy O(n^2) merges; heap if latency matters (vocab>5k or seq>1k)
        while len(symbols) > 1:
            pairs = self._get_pairs(symbols)
            # Find the pair with the lowest merge rank
            best_pair = min(
                pairs,
                key=lambda p: self.merges.get(p, float("inf")),
            )

            if best_pair not in self.merges:
                break  # No more valid merges

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

    def decode(self, token_ids: List[int], space_char: str = "\u2581") -> str:
        """Decodes integer token IDs back to a human-readable string."""
        tokens = [self.id_to_token.get(t, "") for t in token_ids]
        return ByteFallbackEngine.decode_tokens(tokens, space_char=space_char)
