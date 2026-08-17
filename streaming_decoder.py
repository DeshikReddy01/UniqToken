from __future__ import annotations

import re
from typing import Dict, List, Optional


class StreamingDecoder:
    """
    Real-Time Incremental Streaming Decoder with UTF-8 Byte Buffering.
    
    Solves the 'Replacement Character / Glitch UI' flaw during token-by-token LLM generation.
    When multi-byte UTF-8 characters (emojis, CJK, Indic, or raw bytes) arrive split across
    individual token emissions, this decoder buffers incomplete byte fragments and yields
    only complete, valid human-readable text deltas.
    """

    BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

    def __init__(
        self,
        id_to_token: Dict[int, str],
        space_char: str = "\u2581",
        skip_special_tokens: bool = True,
        special_tokens: Optional[List[str]] = None,
    ):
        self.id_to_token = id_to_token
        self.space_char = space_char
        self.skip_special_tokens = skip_special_tokens
        self.special_tokens = set(special_tokens or [])

        self.byte_buffer = bytearray()
        self.prefix_has_space = False

    def reset(self) -> None:
        """Resets internal state for a new stream."""
        self.byte_buffer.clear()
        self.prefix_has_space = False

    def feed_token_id(self, token_id: int) -> str:
        """
        Accepts a single token ID from the LLM generator and returns the decoded string delta.
        Returns an empty string "" if the token is an incomplete multi-byte fragment.
        """
        token = self.id_to_token.get(token_id, "<|unk|>")

        # Handle special control tokens
        if self.skip_special_tokens and (token in self.special_tokens or (token.startswith("<|") and token.endswith("|>"))):
            return ""

        # Check if token is a raw byte fallback token <0xXX>
        match = self.BYTE_TOKEN_PATTERN.match(token)
        if match:
            byte_val = int(match.group(1), 16)
            self.byte_buffer.append(byte_val)
            # Try to decode buffer
            return self._flush_buffer_if_valid()

        # Token is a regular subword string
        # First flush any pending byte buffer
        flushed_bytes = self._force_flush_buffer()

        # Replace metaspace marker with actual space
        subword_text = token.replace(self.space_char, " ")
        return flushed_bytes + subword_text

    def _flush_buffer_if_valid(self) -> str:
        """Attempts to decode accumulated bytes. If valid UTF-8, returns text and clears buffer."""
        if not self.byte_buffer:
            return ""
        try:
            decoded = self.byte_buffer.decode("utf-8")
            self.byte_buffer.clear()
            return decoded.replace(self.space_char, " ")
        except UnicodeDecodeError:
            # Bytes are incomplete (e.g. only 2 of 4 bytes received so far); keep waiting
            return ""

    def _force_flush_buffer(self) -> str:
        """Forces decoding of remaining buffer with error replacement if stream ends or subword arrives."""
        if not self.byte_buffer:
            return ""
        try:
            decoded = self.byte_buffer.decode("utf-8", errors="replace")
        finally:
            self.byte_buffer.clear()
        return decoded.replace(self.space_char, " ")

    def flush(self) -> str:
        """Final flush called when LLM reaches EOS."""
        return self._force_flush_buffer()
