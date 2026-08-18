from __future__ import annotations

import re
from typing import List, Set, Tuple, Union


RawSpan = Tuple[int, int]


class SecurityShield:
    """
    Control Token Sanitizer & Delimiter Injection Guard.

    Prevents out-of-band control token smuggling and delimiter hijacking.
    Neutralizes unauthorized control sequences (e.g., '<|endoftext|>', '<|system|>')
    in untrusted string inputs via escaping, exception raising, or deletion.
    """

    def __init__(
        self,
        special_tokens: List[str],
        special_token_pattern: str = r"<\|[^\s|]+\|>",
    ):
        self.special_tokens = set(special_tokens)
        self.pattern = re.compile(special_token_pattern)

    def sanitize(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",  # "escape" | "raise" | "ignore"
    ) -> str:
        """
        Sanitizes input text according to the authorized control token policy.

        - allowed_special="all": preserves all matching control sequences as active tokens.
        - allowed_special="none": disallows and sanitizes all control sequences.
        - allowed_special={"<|user|>"}: whitelists specified control sequences only.
        """
        sanitized, _ = self.sanitize_with_alignment(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        return sanitized

    def sanitize_with_alignment(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> Tuple[str, List[RawSpan]]:
        """Sanitize text and map every output character back to the source text."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if disallowed_special_action not in {"escape", "raise", "ignore"}:
            raise ValueError("disallowed_special_action must be 'escape', 'raise', or 'ignore'")

        if allowed_special == "all":
            allowed_set = self.special_tokens
        elif allowed_special == "none" or not allowed_special:
            allowed_set = set()
        elif isinstance(allowed_special, str):
            allowed_set = {allowed_special}
        else:
            allowed_set = set(allowed_special)

        output: List[str] = []
        alignment: List[RawSpan] = []
        cursor = 0

        def append_literal(start: int, end: int) -> None:
            output.append(text[start:end])
            alignment.extend((index, index + 1) for index in range(start, end))

        for match in self.pattern.finditer(text):
            append_literal(cursor, match.start())
            token = match.group(0)
            if token in allowed_set:
                output.append(token)
                alignment.extend((index, index + 1) for index in range(match.start(), match.end()))
            elif disallowed_special_action == "raise":
                raise ValueError(
                    f"Security Exception: Input contains unauthorized control token {token!r}. "
                    f"Whitelist via allowed_special={{{token!r}}} if intentional."
                )
            elif disallowed_special_action == "escape":
                escaped = f"<\\|{token[2:-2]}\\|>"
                output.append(escaped)
                alignment.extend([(match.start(), match.end())] * len(escaped))
            cursor = match.end()

        append_literal(cursor, len(text))
        return "".join(output), alignment
