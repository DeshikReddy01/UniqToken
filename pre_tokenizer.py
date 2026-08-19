import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class PreToken:
    """An atomic normalized chunk with normalized and original-text spans."""

    text: str
    start: int
    end: int
    raw_span: Tuple[int, int]

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def norm_span(self) -> Tuple[int, int]:
        return self.span


class Normalizer:
    """
    Standardizes raw text before tokenization.

    NOTE ON REVERSIBILITY:
    - Normalization with NFKC is *canonical*, not byte-exact lossless.
      Compatibility characters (e.g. 'ﬁ' -> 'fi', '²' -> '2') are intentionally transformed.
    - If exact raw string reconstruction is required, disable NFKC (`normalize_unicode=False`).
    """

    PUNCT_MAP = str.maketrans(
        {
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
        }
    )

    UNICODE_SPACES = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")
    _ESCAPE_PREFIX = "\ue000"
    _ESCAPED_METASPACE = "\ue001"

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
        if not isinstance(space_char, str) or len(space_char) != 1:
            raise ValueError("space_char must be exactly one character")
        if space_char in {self._ESCAPE_PREFIX, self._ESCAPED_METASPACE}:
            raise ValueError("space_char conflicts with reserved metaspace escape characters")
        self.space_char = space_char
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.normalize_punctuation = normalize_punctuation
        self.normalize_unicode_spaces = normalize_unicode_spaces
        self.collapse_whitespaces = collapse_whitespaces
        self.strip_whitespace = strip_whitespace

    @staticmethod
    def _expand(value: str, span: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        return [(char, span) for char in value]

    def normalize_with_alignment(self, text: str) -> Tuple[str, List[Tuple[int, int]]]:
        """Normalizes text and maps every output character to its raw source span."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        units: List[Tuple[str, Tuple[int, int]]] = [(char, (i, i + 1)) for i, char in enumerate(text)]

        if self.normalize_unicode:
            normalized: List[Tuple[str, Tuple[int, int]]] = []
            i = 0
            while i < len(text):
                end = i + 1
                while end < len(text) and unicodedata.combining(text[end]):
                    end += 1
                value = unicodedata.normalize("NFKC", text[i:end])
                normalized.extend(self._expand(value, (i, end)))
                i = end
            units = normalized

        if self.normalize_unicode_spaces:
            units = [(" " if self.UNICODE_SPACES.fullmatch(char) else char, span) for char, span in units]

        if self.normalize_punctuation:
            translated: List[Tuple[str, Tuple[int, int]]] = []
            for char, span in units:
                translated.extend(self._expand(char.translate(self.PUNCT_MAP), span))
            units = translated

        if self.lowercase:
            lowered: List[Tuple[str, Tuple[int, int]]] = []
            for char, span in units:
                lowered.extend(self._expand(char.lower(), span))
            units = lowered

        if self.collapse_whitespaces:
            collapsed: List[Tuple[str, Tuple[int, int]]] = []
            i = 0
            while i < len(units):
                char, span = units[i]
                if char not in {" ", "\t"}:
                    collapsed.append((char, span))
                    i += 1
                    continue
                end = i + 1
                while end < len(units) and units[end][0] in {" ", "\t"}:
                    end += 1
                collapsed.append((" ", (span[0], units[end - 1][1][1])))
                i = end
            units = collapsed

        if self.strip_whitespace:
            start = 0
            end = len(units)
            while start < end and units[start][0].isspace():
                start += 1
            while end > start and units[end - 1][0].isspace():
                end -= 1
            units = units[start:end]

        escaped: List[Tuple[str, Tuple[int, int]]] = []
        for char, span in units:
            if char == self._ESCAPE_PREFIX:
                escaped.extend(self._expand(self._ESCAPE_PREFIX * 2, span))
            elif char == self.space_char:
                escaped.extend(self._expand(self._ESCAPE_PREFIX + self._ESCAPED_METASPACE, span))
            elif char == " ":
                escaped.append((self.space_char, span))
            else:
                escaped.append((char, span))

        return "".join(char for char, _ in escaped), [span for _, span in escaped]

    def normalize(self, text: str) -> str:
        return self.normalize_with_alignment(text)[0]

    def restore_escaped_metaspace(self, text: str) -> str:
        """Restores literal metaspace and escape-prefix characters after decoding."""
        restored: List[str] = []
        i = 0
        while i < len(text):
            if text[i] != self._ESCAPE_PREFIX or i + 1 >= len(text):
                restored.append(text[i])
                i += 1
                continue
            marker = text[i + 1]
            if marker == self._ESCAPED_METASPACE:
                restored.append(self.space_char)
                i += 2
            elif marker == self._ESCAPE_PREFIX:
                restored.append(self._ESCAPE_PREFIX)
                i += 2
            else:
                restored.append(text[i])
                i += 1
        return "".join(restored)


class RegexPreTokenizer:
    """
    Offset-preserving, regex-based Pre-Tokenizer.

    Uses compiled C-level regex iteration (`finditer`) to slice input text into
    atomic chunks while preserving character spans for downstream tasks.
    """

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
        special_token = special_token_pattern if self.keep_special_tokens else r"(?!x)x"
        url = r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"

        email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

        hashtag = r"#\w+"
        mention = r"@\w+"

        emoji = (
            r"(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])"
            r"(?:[\uFE0E\uFE0F])?"
            r"(?:[\U0001F3FB-\U0001F3FF])?"
            r"(?:\u200D(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?)*"
        )

        cjk = r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]"

        word = rf"[^\W\d_\s{escaped_space}]+(?:['’][^\W\d_\s{escaped_space}]+)*"

        number = r"\d" if self.split_digits else r"\d+"

        space_marker = rf"{escaped_space}+"
        whitespace = r"\s+"

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

    def pre_tokenize_iter(
        self,
        text: str,
        alignment: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> Iterator[PreToken]:
        """Yields chunks with normalized and raw-text offsets."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if alignment is not None and len(alignment) != len(text):
            raise ValueError("alignment length must match normalized text length")

        for match in self.regex.finditer(text):
            start, end = match.span()
            if alignment is None:
                raw_span = (start, end)
            else:
                source_spans = [
                    entry if isinstance(entry, tuple) else (entry, entry + 1) for entry in alignment[start:end]
                ]
                if not source_spans:
                    continue
                raw_span = (
                    min(span[0] for span in source_spans),
                    max(span[1] for span in source_spans),
                )
            yield PreToken(text=match.group(0), start=start, end=end, raw_span=raw_span)

    def pre_tokenize(self, text: str) -> List[str]:
        """
        Returns a flat list of pre-tokenized chunk strings.
        """
        return [pt.text for pt in self.pre_tokenize_iter(text)]

    def pre_tokenize_with_offsets(
        self,
        text: str,
        alignment: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[PreToken]:
        """Returns chunks with normalized and, when supplied, original spans."""
        return list(self.pre_tokenize_iter(text, alignment))

    def explain(self, text: str) -> None:
        """
        Diagnostic display showing how the text is sliced into chunks with character offsets.
        """
        tokens = self.pre_tokenize_with_offsets(text)
        print(f"\nInput: {text!r}")
        print("Tokens with Spans:")
        for idx, tok in enumerate(tokens):
            print(f"  {idx:>3}: {tok.text!r:<20} Span: {tok.span}")
        print(f"Total Chunks: {len(tokens)}\n")


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    normalizer = Normalizer()
    pre_tokenizer = RegexPreTokenizer(split_digits=False)

    samples = [
        "def compute_sum(a: int, b: int) -> int:\n    return a + b  # 100% precision",
        "Cost is $1,499.99 for iPhone 15 Pro (visit https://apple.com, or email dev@apple.com)!",
        "Emoji test: 👨‍👩‍👧‍👦 family and 👍🏽 thumbs up",
        "我喜欢自然语言处理 and नमस्ते दुनिया",
        "<|user|> Calculate 1.5e-10 + 42 = ? <|endoftext|>",
    ]

    for sample in samples:
        norm = normalizer.normalize(sample)
        tokens = pre_tokenizer.pre_tokenize_with_offsets(norm)
        print("=" * 70)
        print(f"ORIGINAL : {sample}")
        print(f"NORMALIZED: {norm}")
        print(f"CHUNKS   : {[t.text for t in tokens]}")
        print(f"OFFSETS  : {[t.span for t in tokens[:5]]} ... (total: {len(tokens)})")
