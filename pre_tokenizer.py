import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


RawSpan = Tuple[int, int]
AlignmentEntry = Union[int, RawSpan]


@dataclass(frozen=True)
class PreToken:
    """
    Represents an atomic chunk of text with dual offset spans:
    - norm_span: Character slice in the normalized string.
    - raw_span: Exact slice in the original source string (for QA/NER).
    """
    text: str
    norm_span: Tuple[int, int]
    raw_span: Tuple[int, int]


class Normalizer:
    """
    Standardizes raw text before tokenization with exact source-to-normalized offset tracking.
    """

    PUNCT_MAP = str.maketrans({
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "—": "-",
        "–": "-",
        "−": "-",
        "…": "...",
    })

    # Explicit mapping for common non-standard Unicode whitespaces
    UNICODE_SPACES = " \u00A0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"
    _ESCAPE_PREFIX = "\uE000"
    _ESCAPED_METASPACE = "\uE001"

    def __init__(
        self,
        space_char: str = "\u2581",
        lowercase: bool = False,
        normalize_unicode: bool = True,
        normalize_punctuation: bool = False,
        normalize_unicode_spaces: bool = True,
        collapse_whitespaces: bool = False,
        strip_whitespace: bool = False,
    ):
        if len(space_char) != 1 or space_char.isspace():
            raise ValueError("space_char must be a single, non-whitespace character")
        if space_char in {self._ESCAPE_PREFIX, self._ESCAPED_METASPACE}:
            raise ValueError("space_char uses a reserved normalization escape character")
        self.space_char = space_char
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.normalize_punctuation = normalize_punctuation
        self.normalize_unicode_spaces = normalize_unicode_spaces
        self.collapse_whitespaces = collapse_whitespaces
        self.strip_whitespace = strip_whitespace

    def normalize(self, text: str) -> str:
        norm_text, _ = self.normalize_with_alignment(text)
        return norm_text

    def restore_escaped_metaspace(self, text: str) -> str:
        """Restore literal metaspaces and escape prefixes after token decoding."""
        output: List[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == self._ESCAPE_PREFIX and index + 1 < len(text):
                escaped = text[index + 1]
                if escaped == self._ESCAPE_PREFIX:
                    output.append(self._ESCAPE_PREFIX)
                    index += 2
                    continue
                if escaped == self._ESCAPED_METASPACE:
                    output.append(self.space_char)
                    index += 2
                    continue
            output.append(char)
            index += 1
        return "".join(output)

    @staticmethod
    def _is_hangul_jamo(char: str) -> bool:
        codepoint = ord(char)
        return (
            0x1100 <= codepoint <= 0x11FF
            or 0xA960 <= codepoint <= 0xA97C
            or 0xD7B0 <= codepoint <= 0xD7FB
        )

    def _normalize_nfkc_with_alignment(self, text: str) -> Tuple[List[str], List[RawSpan]]:
        """Apply NFKC by normalization-safe clusters and retain source spans."""
        normalized_chars: List[str] = []
        alignment_map: List[RawSpan] = []
        cluster_start = 0

        def flush_cluster(end: int) -> None:
            cluster = text[cluster_start:end]
            normalized = unicodedata.normalize("NFKC", cluster)
            raw_span = (cluster_start, end)
            normalized_chars.extend(normalized)
            alignment_map.extend([raw_span] * len(normalized))

        for index, char in enumerate(text):
            normalized_char = unicodedata.normalize("NFKC", char)
            starts_with_combining_mark = (
                bool(normalized_char)
                and unicodedata.combining(normalized_char[0]) != 0
            )
            continues_cluster = (
                index > cluster_start
                and (
                    unicodedata.combining(char) != 0
                    or starts_with_combining_mark
                    or (
                        self._is_hangul_jamo(text[index - 1])
                        and self._is_hangul_jamo(char)
                    )
                )
            )
            if index > cluster_start and not continues_cluster:
                flush_cluster(index)
                cluster_start = index

        if text:
            flush_cluster(len(text))

        # The cluster strategy handles the normal cases while this fallback keeps
        # normalization correct for rare Unicode sequences with unusual boundaries.
        full_normalized = unicodedata.normalize("NFKC", text)
        if "".join(normalized_chars) != full_normalized:
            return list(full_normalized), [(0, len(text))] * len(full_normalized)

        return normalized_chars, alignment_map

    @staticmethod
    def _replace_characters(
        chars: Sequence[str],
        spans: Sequence[RawSpan],
        transform
    ) -> Tuple[List[str], List[RawSpan]]:
        output_chars: List[str] = []
        output_spans: List[RawSpan] = []
        for char, span in zip(chars, spans):
            replacement = transform(char)
            output_chars.extend(replacement)
            output_spans.extend([span] * len(replacement))
        return output_chars, output_spans

    def normalize_with_alignment(self, text: str) -> Tuple[str, List[RawSpan]]:
        """
        Normalizes text and returns an alignment map where:
        alignment_map[norm_char_idx] -> (raw_start, raw_end) in the original text.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        if self.normalize_unicode:
            normalized_chars, alignment_map = self._normalize_nfkc_with_alignment(text)
        else:
            normalized_chars = list(text)
            alignment_map = [(index, index + 1) for index in range(len(text))]

        if self.normalize_unicode_spaces:
            normalized_chars, alignment_map = self._replace_characters(
                normalized_chars,
                alignment_map,
                lambda char: " " if char in self.UNICODE_SPACES else char,
            )

        if self.normalize_punctuation:
            normalized_chars, alignment_map = self._replace_characters(
                normalized_chars,
                alignment_map,
                lambda char: char.translate(self.PUNCT_MAP),
            )

        if self.lowercase:
            lowered_by_char, lowered_spans = self._replace_characters(
                normalized_chars,
                alignment_map,
                lambda char: char.lower(),
            )
            normalized_chars = list("".join(normalized_chars).lower())
            if len(normalized_chars) != len(lowered_by_char):
                raise ValueError("lowercase normalization produced an unsupported alignment change")
            alignment_map = lowered_spans

        if self.collapse_whitespaces:
            collapsed_chars: List[str] = []
            collapsed_spans: List[RawSpan] = []
            index = 0
            while index < len(normalized_chars):
                char = normalized_chars[index]
                if char not in {" ", "\t"}:
                    collapsed_chars.append(char)
                    collapsed_spans.append(alignment_map[index])
                    index += 1
                    continue

                end = index + 1
                while end < len(normalized_chars) and normalized_chars[end] in {" ", "\t"}:
                    end += 1
                collapsed_chars.append(" ")
                collapsed_spans.append(
                    (
                        min(span[0] for span in alignment_map[index:end]),
                        max(span[1] for span in alignment_map[index:end]),
                    )
                )
                index = end
            normalized_chars, alignment_map = collapsed_chars, collapsed_spans

        if self.strip_whitespace:
            start = 0
            end = len(normalized_chars)
            while start < end and normalized_chars[start].isspace():
                start += 1
            while end > start and normalized_chars[end - 1].isspace():
                end -= 1
            normalized_chars = normalized_chars[start:end]
            alignment_map = alignment_map[start:end]

        def encode_metaspace(char: str) -> str:
            if char == self._ESCAPE_PREFIX:
                return self._ESCAPE_PREFIX + self._ESCAPE_PREFIX
            if char == self.space_char:
                return self._ESCAPE_PREFIX + self._ESCAPED_METASPACE
            if char == " ":
                return self.space_char
            return char

        normalized_chars, alignment_map = self._replace_characters(
            normalized_chars,
            alignment_map,
            encode_metaspace,
        )
        return "".join(normalized_chars), alignment_map


class RegexPreTokenizer:
    """
    Dual-Offset Regex Pre-Tokenizer with Multilingual Combining Mark & RFC URL Support.
    """

    # Combining Marks: Indic (Devanagari, Bengali, Tamil, etc.), Arabic, Hebrew, Thai, Latin Diacritics
    COMBINING_MARKS = (
        r"\u0300-\u036F\u0590-\u08FF\u0900-\u0DFF\u0E00-\u0E7F"
        r"\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F"
    )

    def __init__(
        self,
        space_char: str = "\u2581",
        split_digits: bool = False,
        split_punctuation: bool = True,
        keep_special_tokens: bool = True,
        special_token_pattern: str = r"<\|[^\s|]+\|>",
    ):
        self.space_char = space_char
        self.split_digits = split_digits
        self.split_punctuation = split_punctuation
        self.keep_special_tokens = keep_special_tokens
        self.special_token_pattern = special_token_pattern

        escaped_space = re.escape(self.space_char)

        # 1. Special Control Tokens (<|user|>, <|endoftext|>, etc.)
        special_token = special_token_pattern if self.keep_special_tokens else r"(?!x)x"

        # 2. URLs: RFC-compliant matching that strictly terminates on alphanumeric/slash (excludes trailing . , ; ! ? )
        url = (
            r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}"
            r"\.[a-zA-Z0-9()]{1,6}\b"
            r"(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*[a-zA-Z0-9/])?"
        )

        # 3. Emails
        email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

        # 4. Social Tags
        hashtag = r"#\w+"
        mention = r"@\w+"

        # 5. Emoji with full ZWJ (\u200D), skin tone modifiers, and variation selectors
        emoji = (
            r"(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])"
            r"(?:[\uFE0E\uFE0F])?"
            r"(?:[\U0001F3FB-\U0001F3FF])?"
            r"(?:\u200D(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?)*"
        )

        # 6. CJK and East Asian Scripts (Individual characters to prevent monolithic block tokens)
        cjk = r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]"

        # 7. Multilingual Alphabetic Words (Includes Unicode combining marks / viramas / matras)
        word = (
            rf"(?:[^\W\d_\s{escaped_space}]|[{self.COMBINING_MARKS}])+"
            rf"(?:['’](?:[^\W\d_\s{escaped_space}]|[{self.COMBINING_MARKS}])+)*"
        )

        # 8. Numbers (Single digit if split_digits=True, or continuous digit chunks if False)
        number = r"\d" if self.split_digits else r"\d+"

        # 9. Metaspaces & Whitespace sequences
        space_marker = rf"{escaped_space}+"
        whitespace = r"\s+"

        # 10. Punctuation & Symbols (including underscore '_')
        if self.split_punctuation:
            punctuation = rf"[^\w\s{escaped_space}]|_"
        else:
            punctuation = rf"[^\w\s{escaped_space}]+|_+"

        self.patterns = [
            special_token,
            url,
            email,
            hashtag,
            mention,
            emoji,
            cjk,
            word,
            number,
            space_marker,
            whitespace,
            punctuation,
        ]

        combined_pattern = "|".join(f"(?:{p})" for p in self.patterns)
        self.regex = re.compile(combined_pattern)

    def pre_tokenize_with_offsets(
        self,
        normalized_text: str,
        alignment_map: Optional[Sequence[AlignmentEntry]] = None,
    ) -> List[PreToken]:
        """
        Emits PreToken instances with both normalized and source raw character spans.
        """
        if not isinstance(normalized_text, str):
            raise TypeError(f"normalized_text must be a string, got {type(normalized_text).__name__}")
        if alignment_map is not None and len(alignment_map) != len(normalized_text):
            raise ValueError(
                "alignment_map must have one entry for every normalized character"
            )

        tokens: List[PreToken] = []

        for match in self.regex.finditer(normalized_text):
            norm_start, norm_end = match.start(), match.end()

            if alignment_map is not None:
                token_spans: List[RawSpan] = []
                for entry in alignment_map[norm_start:norm_end]:
                    if isinstance(entry, int):
                        token_spans.append((entry, entry + 1))
                    else:
                        token_spans.append(entry)
                raw_start = min(span[0] for span in token_spans)
                raw_end = max(span[1] for span in token_spans)
            else:
                raw_start, raw_end = norm_start, norm_end

            tokens.append(
                PreToken(
                    text=match.group(0),
                    norm_span=(norm_start, norm_end),
                    raw_span=(raw_start, raw_end),
                )
            )

        return tokens

    def pre_tokenize(self, text: str) -> List[str]:
        return [match.group(0) for match in self.regex.finditer(text)]


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    normalizer = Normalizer()
    pre_tokenizer = RegexPreTokenizer(split_digits=False)

    samples = [
        "ﬁx the bug in 2024 (visit https://example.com.)",
        "नमस्ते दुनिया and شكراً",
        "Cost is ½ price for Ａpple"
    ]

    for sample in samples:
        norm, alignment = normalizer.normalize_with_alignment(sample)
        tokens = pre_tokenizer.pre_tokenize_with_offsets(norm, alignment)
        print("=" * 65)
        print(f"RAW TEXT   : {sample!r}")
        print(f"NORMALIZED : {norm!r}")
        for t in tokens:
            raw_slice = sample[t.raw_span[0]:t.raw_span[1]]
            print(f"  {t.text!r:<15} NormSpan={t.norm_span} RawSpan={t.raw_span} RawSlice={raw_slice!r}")
