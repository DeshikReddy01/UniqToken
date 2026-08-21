from __future__ import annotations

import codecs
import re
from typing import Dict, List, Optional, Tuple


class StreamingDecoder:
    """
    Incremental Streaming Decoder with UTF-8 Byte Accumulation.

    Prevents replacement character (U+FFFD) decoding errors during token-by-token generation.
    Buffers fragmented multi-byte UTF-8 sequences (e.g. multi-byte codepoints split across
    discrete token emissions) and emits only structurally valid Unicode text deltas.
    """

    BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

    def __init__(
        self,
        id_to_token: Dict[int, str],
        space_char: str = "\u2581",
        skip_special_tokens: bool = True,
        special_tokens: Optional[List[str]] = None,
        special_replacements: Optional[Dict[str, str]] = None,
        metaspace_escape: Optional[Tuple[str, str]] = None,
    ):
        self.id_to_token = id_to_token
        self.space_char = space_char
        self.skip_special_tokens = skip_special_tokens
        self.special_tokens = set(special_tokens or [])
        self.special_replacements = dict(special_replacements or {})
        self.metaspace_escape = metaspace_escape

        # Incremental UTF-8 decoder: emits every complete codepoint as soon as
        # it is formed and retains only a genuinely incomplete trailing partial
        # in its internal buffer. This means a completed leading byte (e.g.
        # ASCII 'a') is never held back waiting on a later fragmented multi-byte
        # start, and a partial at end-of-stream is handled in flush().
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._pending_escape = ""

    def reset(self) -> None:
        """Resets the internal byte accumulator."""
        self._utf8_decoder.reset()
        self._pending_escape = ""

    def _emit_text(self, text: str) -> str:
        """Restore literal metaspaces while preserving incomplete escape pairs."""
        if self.metaspace_escape is None:
            return text

        prefix, escaped_metaspace = self.metaspace_escape
        text = self._pending_escape + text
        self._pending_escape = ""
        output: List[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char != prefix:
                output.append(char)
                index += 1
                continue
            if index + 1 == len(text):
                self._pending_escape = prefix
                break

            escaped = text[index + 1]
            if escaped == prefix:
                output.append(prefix)
                index += 2
            elif escaped == escaped_metaspace:
                output.append(self.space_char)
                index += 2
            else:
                output.append(prefix)
                index += 1
        return "".join(output)

    def feed_token_id(self, token_id: int) -> str:
        """
        Processes a single token identifier and returns the decoded string delta.
        Returns an empty string if the incoming token represents an incomplete byte sequence.
        """
        token = self.id_to_token.get(token_id, "<|unk|>")

        if token in self.special_replacements:
            return self._force_flush_buffer() + self._emit_text(self.special_replacements[token])
        if self.skip_special_tokens and (
            token in self.special_tokens or (token.startswith("<|") and token.endswith("|>"))
        ):
            return ""

        match = self.BYTE_TOKEN_PATTERN.match(token)
        if match:
            byte_val = int(match.group(1), 16)
            # Feed one byte; the incremental decoder returns any complete
            # text and retains a genuinely incomplete trailing partial. A
            # strict decoder raises only on an *invalid* (not merely
            # incomplete) sequence; substitute U+FFFD for that single byte
            # and reset the incremental state so the next byte starts clean.
            byte = bytes([byte_val])
            try:
                raw = self._utf8_decoder.decode(byte, final=False)
            except UnicodeDecodeError:
                # The new byte may be valid by itself (for example ASCII after
                # a truncated multibyte prefix), so replace the pending invalid
                # sequence and retry the current byte instead of dropping it.
                self._utf8_decoder.reset()
                try:
                    raw = "�" + self._utf8_decoder.decode(byte, final=False)
                except UnicodeDecodeError:
                    self._utf8_decoder.reset()
                    raw = "��"
            return self._emit_text(raw) if raw else ""

        flushed_bytes = self._force_flush_buffer()
        subword_text = token.replace(self.space_char, " ")
        return flushed_bytes + self._emit_text(subword_text)

    def _force_flush_buffer(self) -> str:
        """Forces decoding of any buffered bytes with substitution on sequence termination."""
        try:
            raw = self._utf8_decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raw = "�"
        # Re-arm the incremental decoder so subsequent tokens can keep streaming.
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        return self._emit_text(raw)

    def flush(self) -> str:
        """Final flush called at sequence termination."""
        output = self._force_flush_buffer()
        if self._pending_escape:
            output += self._pending_escape
            self._pending_escape = ""
        return output
