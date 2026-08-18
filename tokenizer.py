from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union

from indentation_compressor import IndentationCompressor
from pre_tokenizer import Normalizer, RegexPreTokenizer
from security_shield import SecurityShield
from streaming_decoder import StreamingDecoder
from unigram_trainer import UnigramModel, UnigramTrainer


@dataclass(frozen=True)
class Token:
    """
    Final Token emitted by the Tokenizer with token string, integer ID, and raw text offsets.
    """

    text: str
    id: int
    raw_span: Tuple[int, int]


class CustomTokenizer:
    """
    Production-Grade Byte-Fallback Unigram Custom Tokenizer.
    """

    def __init__(
        self,
        normalizer: Normalizer,
        pre_tokenizer: RegexPreTokenizer,
        model: UnigramModel,
    ):
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.model = model
        self.security = SecurityShield(special_tokens=self.model.special_tokens)
        self._cross_word_set: Optional[frozenset[str]] = None

    @staticmethod
    def _span(entry: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        return entry if isinstance(entry, tuple) else (entry, entry + 1)

    @classmethod
    def _compose_alignment(
        cls,
        inner_alignment: Sequence[Union[int, Tuple[int, int]]],
        outer_alignment: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        composed: List[Tuple[int, int]] = []
        for entry in inner_alignment:
            start, end = cls._span(entry)
            source_spans = outer_alignment[start:end]
            if not source_spans:
                raise ValueError("alignment contains an empty source span")
            composed.append(
                (
                    min(span[0] for span in source_spans),
                    max(span[1] for span in source_spans),
                )
            )
        return composed

    def _cross_word_tokens(self) -> frozenset[str]:
        """Vocab tokens containing the space char (SuperBPE spanning tokens).

        These can never be emitted by the per-chunk Unigram lattice (a chunk is
        ``"the"`` or ``"\u2581"``, never ``"the\u2581quick"``), so they are only
        reachable through the post-encode merge pass.
        """
        if self._cross_word_set is None:
            sc = self.normalizer.space_char
            self._cross_word_set = frozenset(t for t in self.model.vocab if sc in t and t.strip(sc))
        return self._cross_word_set

    def _apply_cross_word_merges(self, tokens: List[str]) -> List[str]:
        """Deterministic greedy pass that fuses adjacent tokens into SuperBPE
        spanning tokens (e.g. ``the`` + ``\u2581`` + ``quick`` -> ``the\u2581quick``)."""
        cross = self._cross_word_tokens()
        if not cross:
            return tokens
        merged: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i + 1 < n and tokens[i] + tokens[i + 1] in cross:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _apply_cross_word_merges_with_spans(self, tokens: List[Token]) -> List[Token]:
        cross = self._cross_word_tokens()
        if not cross:
            return tokens
        merged: List[Token] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i + 1 < n and tokens[i].text + tokens[i + 1].text in cross:
                a, b = tokens[i], tokens[i + 1]
                text = a.text + b.text
                merged.append(
                    Token(
                        text=text,
                        id=self.model.token_to_id.get(text, a.id),
                        raw_span=(a.raw_span[0], b.raw_span[1]),
                    )
                )
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _prepare_text(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]],
        disallowed_special_action: str,
    ) -> str:
        sanitized = self.security.sanitize(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        if self._indent_compression_enabled:
            return IndentationCompressor.compress_indents(sanitized)
        return sanitized

    def _prepare_text_with_alignment(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]],
        disallowed_special_action: str,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        sanitized, sanitized_alignment = self.security.sanitize_with_alignment(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        if not self._indent_compression_enabled:
            return sanitized, sanitized_alignment

        compressed, compressed_alignment = IndentationCompressor.compress_indents_with_alignment(sanitized)
        return compressed, self._compose_alignment(compressed_alignment, sanitized_alignment)

    @classmethod
    def train_from_corpus(
        cls,
        corpus: List[str],
        target_vocab_size: int = 8000,
        seed_multiplier: float = 3.0,
        max_ngram_length: int = 16,
        min_frequency: int = 2,
        byte_fallback: bool = True,
        split_digits: bool = False,
        special_tokens: Optional[List[str]] = None,
        compress_indents: bool = False,
        verbose: bool = True,
    ) -> CustomTokenizer:
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer(split_digits=split_digits)

        combined_special = list(special_tokens or [])
        if compress_indents:
            for it in IndentationCompressor.INDENT_SPECIAL_TOKENS:
                if it not in combined_special:
                    combined_special.append(it)

        chunks: List[str] = []
        for doc in corpus:
            if compress_indents:
                doc = IndentationCompressor.compress_indents(doc)
            norm = normalizer.normalize(doc)
            chunks.extend(pre_tokenizer.pre_tokenize(norm))

        trainer = UnigramTrainer(
            target_vocab_size=target_vocab_size,
            seed_multiplier=seed_multiplier,
            max_ngram_length=max_ngram_length,
            min_frequency=min_frequency,
            byte_fallback=byte_fallback,
            special_tokens=combined_special if combined_special else None,
        )

        model = trainer.train(chunks, verbose=verbose)
        return cls(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=model)

    def encode(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> List[str]:
        if not text:
            return []

        sanitized_text = self._prepare_text(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )

        norm = self.normalizer.normalize(sanitized_text)
        chunks = self.pre_tokenizer.pre_tokenize(norm)

        all_tokens: List[str] = []
        for chunk in chunks:
            if chunk in self.model.special_tokens:
                all_tokens.append(chunk)
            else:
                all_tokens.extend(self.model.encode(chunk))

        return self._apply_cross_word_merges(all_tokens)

    def sample(
        self,
        text: str,
        alpha: float = 0.5,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> List[str]:
        if not text:
            return []

        sanitized_text = self._prepare_text(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        norm = self.normalizer.normalize(sanitized_text)
        chunks = self.pre_tokenizer.pre_tokenize(norm)

        all_tokens: List[str] = []
        for chunk in chunks:
            if chunk in self.model.special_tokens:
                all_tokens.append(chunk)
            else:
                all_tokens.extend(self.model.sample(chunk, alpha=alpha))

        return self._apply_cross_word_merges(all_tokens)

    def encode_to_ids(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> List[int]:
        tokens = self.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        unk_id = self.model.token_to_id.get("<|unk|>", 0)
        return [self.model.token_to_id.get(t, unk_id) for t in tokens]

    def sample_to_ids(
        self,
        text: str,
        alpha: float = 0.5,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> List[int]:
        tokens = self.sample(
            text,
            alpha=alpha,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        unk_id = self.model.token_to_id.get("<|unk|>", 0)
        return [self.model.token_to_id.get(t, unk_id) for t in tokens]

    def encode_with_offsets(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> List[Token]:
        if not text:
            return []

        prepared_text, prepared_alignment = self._prepare_text_with_alignment(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        norm, normalization_alignment = self.normalizer.normalize_with_alignment(prepared_text)
        alignment = self._compose_alignment(normalization_alignment, prepared_alignment)
        pre_tokens = self.pre_tokenizer.pre_tokenize_with_offsets(norm, alignment)

        result: List[Token] = []
        unk_id = self.model.token_to_id.get("<|unk|>", 0)

        for pt in pre_tokens:
            chunk = pt.text
            if chunk in self.model.special_tokens:
                t_id = self.model.token_to_id.get(chunk, unk_id)
                result.append(Token(text=chunk, id=t_id, raw_span=pt.raw_span))
            else:
                for st, start, end in self.model.encode_with_spans(chunk):
                    t_id = self.model.token_to_id.get(st, unk_id)
                    norm_start = pt.norm_span[0] + start
                    norm_end = pt.norm_span[0] + end
                    source_spans = alignment[norm_start:norm_end]
                    raw_span = (
                        min(span[0] for span in source_spans),
                        max(span[1] for span in source_spans),
                    )
                    result.append(Token(text=st, id=t_id, raw_span=raw_span))

        return self._apply_cross_word_merges_with_spans(result)

    @property
    def _indent_compression_enabled(self) -> bool:
        return any(tok in self.model.special_tokens for tok in IndentationCompressor.INDENT_SPECIAL_TOKENS)

    def decode(self, token_ids: List[int]) -> str:
        decoded = self.model.decode(token_ids, space_char=self.normalizer.space_char)
        if self._indent_compression_enabled:
            decoded = IndentationCompressor.decompress_indents(decoded)
        return self.normalizer.restore_escaped_metaspace(decoded)

    def get_streaming_decoder(self, skip_special_tokens: bool = True) -> StreamingDecoder:
        indent_replacements = {}
        if self._indent_compression_enabled:
            indent_replacements = {token: " " * count for count, token in IndentationCompressor.INDENT_MAP}
            indent_replacements["<|tab|>"] = "\t"
        return StreamingDecoder(
            id_to_token=self.model.id_to_token,
            space_char=self.normalizer.space_char,
            skip_special_tokens=skip_special_tokens,
            special_tokens=self.model.special_tokens,
            special_replacements=indent_replacements,
            metaspace_escape=(
                self.normalizer._ESCAPE_PREFIX,
                self.normalizer._ESCAPED_METASPACE,
            ),
        )

    def save(self, directory: Union[str, Path]) -> None:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        config = {
            "vocab": self.model.vocab,
            "token_to_id": self.model.token_to_id,
            "special_tokens": self.model.special_tokens,
            "space_char": self.normalizer.space_char,
            "split_digits": self.pre_tokenizer.split_digits,
            "max_subword_len": self.model.max_subword_len,
            "byte_fallback": self.model.byte_fallback,
            "normalizer": {
                "space_char": self.normalizer.space_char,
                "lowercase": self.normalizer.lowercase,
                "normalize_unicode": self.normalizer.normalize_unicode,
                "normalize_punctuation": self.normalizer.normalize_punctuation,
                "normalize_unicode_spaces": self.normalizer.normalize_unicode_spaces,
                "collapse_whitespaces": self.normalizer.collapse_whitespaces,
                "strip_whitespace": self.normalizer.strip_whitespace,
            },
            "pre_tokenizer": {
                "space_char": self.pre_tokenizer.space_char,
                "split_digits": self.pre_tokenizer.split_digits,
                "split_punctuation": self.pre_tokenizer.split_punctuation,
                "keep_special_tokens": self.pre_tokenizer.keep_special_tokens,
                "special_token_pattern": self.pre_tokenizer.special_token_pattern,
            },
        }

        with open(dir_path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> CustomTokenizer:
        dir_path = Path(directory)
        with open(dir_path / "tokenizer.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        vocab = config["vocab"]
        token_to_id = config["token_to_id"]
        id_to_token = {int(v): k for k, v in token_to_id.items()}
        special_tokens = config["special_tokens"]

        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=special_tokens,
            max_subword_len=config.get("max_subword_len", 16),
            byte_fallback=config.get("byte_fallback", True),
        )

        normalizer_config = config.get("normalizer", {})
        pre_tokenizer_config = config.get("pre_tokenizer", {})

        normalizer = Normalizer(
            space_char=normalizer_config.get("space_char", config.get("space_char", "\u2581")),
            lowercase=normalizer_config.get("lowercase", False),
            normalize_unicode=normalizer_config.get("normalize_unicode", True),
            normalize_punctuation=normalizer_config.get("normalize_punctuation", False),
            normalize_unicode_spaces=normalizer_config.get("normalize_unicode_spaces", True),
            collapse_whitespaces=normalizer_config.get("collapse_whitespaces", False),
            strip_whitespace=normalizer_config.get("strip_whitespace", False),
        )
        pre_tokenizer = RegexPreTokenizer(
            space_char=pre_tokenizer_config.get("space_char", config.get("space_char", "\u2581")),
            split_digits=pre_tokenizer_config.get("split_digits", config.get("split_digits", False)),
            split_punctuation=pre_tokenizer_config.get("split_punctuation", True),
            keep_special_tokens=pre_tokenizer_config.get("keep_special_tokens", True),
            special_token_pattern=pre_tokenizer_config.get("special_token_pattern", r"<\|[^\s|]+\|>"),
        )

        return cls(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=model)

    def export_to_huggingface(self, directory: Union[str, Path]) -> None:
        """
        Exports the tokenizer to canonical HuggingFace tokenizer.json and tokenizer_config.json schema.
        """
        from hf_exporter import HuggingFaceExporter

        HuggingFaceExporter.save_hf_pretrained(self, directory)
