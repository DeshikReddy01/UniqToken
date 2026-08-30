from __future__ import annotations

import re
import difflib
import unicodedata
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

    @staticmethod
    def _canonicalize_with_alignment(text: str) -> Tuple[str, List[RawSpan]]:
        """Return NFKC text and conservative source spans for every output char."""
        canonical = unicodedata.normalize("NFKC", text)
        if canonical == text:
            return canonical, [(i, i + 1) for i in range(len(text))]

        alignment: List[RawSpan] = []
        matcher = difflib.SequenceMatcher(a=text, b=canonical, autojunk=False)
        for tag, source_start, source_end, output_start, output_end in matcher.get_opcodes():
            if tag == "equal":
                alignment.extend((i, i + 1) for i in range(source_start, source_end))
                continue
            source_span = (source_start, source_end)
            if source_start == source_end:
                anchor = min(source_start, len(text))
                source_span = (anchor, anchor)
            alignment.extend([source_span] * (output_end - output_start))
        return canonical, alignment

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
          (Policy spellings "all"/"none" are matched case-insensitively so a typo
          like "ALL" can never silently act as a whitelist of that literal.)

        NOTE: the escape form ``<\\|token\\|>`` is not injective — input that already
        contains that literal spelling is indistinguishable from an escaped control
        token. Do not feed pre-escaped text through sanitize twice.
        """
        # ponytail: fast path without alignment — avoids 120k tuple allocs per 240 texts; with_alignment kept exact
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if disallowed_special_action not in {"escape", "raise", "ignore"}:
            raise ValueError("disallowed_special_action must be 'escape', 'raise', or 'ignore'")
        if isinstance(allowed_special, str):
            policy = allowed_special.strip().lower()
            if policy == "all":
                allowed_set = self.special_tokens
            elif policy in ("none", ""):
                allowed_set = set()
            else:
                allowed_set = {allowed_special}
        elif allowed_special:
            allowed_set = set(allowed_special)
        else:
            allowed_set = set()

        canonical, canonical_alignment = self._canonicalize_with_alignment(text)
        output: List[str] = []
        raw_cursor = 0
        for match in self.pattern.finditer(canonical):
            source_spans = canonical_alignment[match.start() : match.end()]
            raw_start = min((span[0] for span in source_spans), default=raw_cursor)
            raw_end = max((span[1] for span in source_spans), default=raw_start)
            output.append(text[raw_cursor:raw_start])
            token = match.group(0)
            if token in allowed_set:
                output.append(token)
            elif disallowed_special_action == "raise":
                raise ValueError(
                    f"Security Exception: Input contains unauthorized control token {token!r}. "
                    f"Whitelist via allowed_special={{{token!r}}} if intentional."
                )
            elif disallowed_special_action == "escape":
                output.append(f"<\\|{token[2:-2]}\\|>")
            elif disallowed_special_action == "ignore":
                pass
            raw_cursor = raw_end
        output.append(text[raw_cursor:])
        return "".join(output)

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

        if isinstance(allowed_special, str):
            policy = allowed_special.strip().lower()
            if policy == "all":
                allowed_set = self.special_tokens
            elif policy in ("none", ""):
                allowed_set = set()
            else:
                allowed_set = {allowed_special}
        elif allowed_special:
            allowed_set = set(allowed_special)
        else:
            allowed_set = set()

        raw_text = text
        text, canonical_alignment = self._canonicalize_with_alignment(text)
        output: List[str] = []
        alignment: List[RawSpan] = []
        raw_cursor = 0

        def append_literal(start: int, end: int) -> None:
            output.append(raw_text[start:end])
            alignment.extend((index, index + 1) for index in range(start, end))

        for match in self.pattern.finditer(text):
            source_spans = canonical_alignment[match.start() : match.end()]
            raw_start = min((span[0] for span in source_spans), default=raw_cursor)
            raw_end = max((span[1] for span in source_spans), default=raw_start)
            append_literal(raw_cursor, raw_start)
            token = match.group(0)
            if token in allowed_set:
                output.append(token)
                alignment.extend(canonical_alignment[match.start() : match.end()])
            elif disallowed_special_action == "raise":
                raise ValueError(
                    f"Security Exception: Input contains unauthorized control token {token!r}. "
                    f"Whitelist via allowed_special={{{token!r}}} if intentional."
                )
            elif disallowed_special_action == "escape":
                escaped = f"<\\|{token[2:-2]}\\|>"
                output.append(escaped)
                source_span = canonical_alignment[match.start() : match.end()]
                if source_span:
                    raw_span = (min(s[0] for s in source_span), max(s[1] for s in source_span))
                else:
                    raw_span = (match.start(), match.end())
                alignment.extend([raw_span] * len(escaped))
            elif disallowed_special_action == "ignore":
                pass  # delete token; alignment already excludes it
            raw_cursor = raw_end

        append_literal(raw_cursor, len(raw_text))
        return "".join(output), alignment
