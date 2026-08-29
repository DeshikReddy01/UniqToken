from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Set, Tuple, Union


RawSpan = Tuple[int, int]


class IndentationCompressor:
    """
    Indentation & Whitespace Run-Length Compressor.

    Compresses recurring whitespace blocks (2, 4, 8, 16 spaces) and tab characters
    into discrete structural tokens, reducing token sequence lengths in structured code.
    """

    INDENT_MAP = [
        (16, "<|space_16|>"),
        (8, "<|space_8|>"),
        (4, "<|space_4|>"),
        (2, "<|space_2|>"),
    ]

    INDENT_SPECIAL_TOKENS = [
        "<|space_2|>",
        "<|space_4|>",
        "<|space_8|>",
        "<|space_16|>",
        "<|tab|>",
    ]

    @classmethod
    def compress_indents(
        cls, text: str, vocab: Optional[Union[Set[str], Sequence[str], Mapping[str, Any]]] = None
    ) -> str:
        """
        Replaces structured indentation whitespace with deterministic tokens.
        """
        compressed, _ = cls.compress_indents_with_alignment(text, vocab=vocab)
        return compressed

    @classmethod
    def compress_indents_with_alignment(
        cls, text: str, vocab: Optional[Union[Set[str], Sequence[str], Mapping[str, Any]]] = None
    ) -> Tuple[str, List[RawSpan]]:
        """Compress whitespace while retaining the span of every emitted character."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        allowed_map = cls.INDENT_MAP
        vocab_set = None if vocab is None else set(vocab)
        if vocab_set is not None:
            allowed_map = [(count, token) for count, token in cls.INDENT_MAP if token in vocab_set]

        output: List[str] = []
        alignment: List[RawSpan] = []
        index = 0
        while index < len(text):
            if text[index] == "\t":
                token = "<|tab|>"
                if vocab_set is None or token in vocab_set:
                    output.append(token)
                    alignment.extend([(index, index + 1)] * len(token))
                else:
                    output.append("\t")
                    alignment.append((index, index + 1))
                index += 1
                continue

            if text[index] != " ":
                output.append(text[index])
                alignment.append((index, index + 1))
                index += 1
                continue

            run_end = index
            while run_end < len(text) and text[run_end] == " ":
                run_end += 1

            cursor = index
            remaining = run_end - index
            for count, token in allowed_map:
                while remaining >= count:
                    output.append(token)
                    alignment.extend([(cursor, cursor + count)] * len(token))
                    cursor += count
                    remaining -= count
            while remaining:
                output.append(" ")
                alignment.append((cursor, cursor + 1))
                cursor += 1
                remaining -= 1
            index = run_end

        return "".join(output), alignment

    @classmethod
    def decompress_indents(cls, text: str) -> str:
        """
        Restores structural indentation tokens back to exact whitespace characters.
        """
        text = text.replace("<|tab|>", "\t")
        for count, token in cls.INDENT_MAP:
            text = text.replace(token, " " * count)
        return text
