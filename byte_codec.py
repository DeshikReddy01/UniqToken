import re
from typing import List


class ByteFallbackEngine:
    """
    Executable Byte-Fallback codec.

    Bridges the gap between raw UTF-8 bytes and token representations (<0x00> through <0xFF>).
    Guarantees executable OOV recovery and lossless roundtripping for valid UTF-8
    byte sequences produced by ``char_to_byte_tokens``.
    """

    BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

    @classmethod
    def is_byte_token(cls, token: str) -> bool:
        return bool(cls.BYTE_TOKEN_PATTERN.match(token))

    @classmethod
    def byte_to_token(cls, byte_val: int) -> str:
        if not 0 <= byte_val <= 255:
            raise ValueError(f"Byte value must be in range 0-255, got {byte_val}")
        return f"<0x{byte_val:02X}>"

    @classmethod
    def token_to_byte(cls, token: str) -> int:
        match = cls.BYTE_TOKEN_PATTERN.match(token)
        if not match:
            raise ValueError(f"Token {token!r} is not a valid byte fallback token")
        return int(match.group(1), 16)

    @classmethod
    def char_to_byte_tokens(cls, char_or_str: str) -> List[str]:
        """
        Converts an un-embedded or OOV string into a sequence of byte tokens.
        Uses ``surrogatepass`` so raw binary decoded with Python's
        ``errors='surrogateescape'`` (common for POSIX file reads) maps its lone
        surrogate chars to the exact original bytes instead of raising.
        """
        raw_bytes = char_or_str.encode("utf-8", errors="surrogatepass")
        return [cls.byte_to_token(b) for b in raw_bytes]

    @classmethod
    def decode_tokens(cls, tokens: List[str], space_char: str = "\u2581") -> str:
        """
        Reconstructs a human-readable string from a stream of subwords and byte fallback tokens.
        Accumulates adjacent byte tokens and decodes them as UTF-8 sequences.
        Invalid byte sequences raise UnicodeDecodeError rather than silently
        replacing data. Metaspace conversion applies only to learned subwords,
        never to text reconstructed from byte fallback tokens.
        """
        output_segments: List[str] = []
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                output_segments.append(byte_buffer.decode("utf-8"))
                byte_buffer.clear()

        for tok in tokens:
            if cls.is_byte_token(tok):
                byte_val = cls.token_to_byte(tok)
                byte_buffer.append(byte_val)
            else:
                flush_bytes()
                output_segments.append(tok.replace(space_char, " "))

        flush_bytes()
        return "".join(output_segments)
