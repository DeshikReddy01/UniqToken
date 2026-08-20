from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union

from bpe_model import BPEModel
from indentation_compressor import IndentationCompressor
from pre_tokenizer import Normalizer, RegexPreTokenizer
from security_shield import SecurityShield
from seed_builder import SeedVocabularyBuilder
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


@dataclass(frozen=True)
class TokenizationReport:
    """
    Detailed runtime diagnostic metrics emitted during tokenization.
    """

    tokens: List[str]
    token_ids: List[int]
    token_spans: List[Tuple[int, int]]
    num_tokens: int
    num_bytes: int
    num_chars: int
    byte_fallback_tokens: int
    byte_fallback_rate: float
    compression_ratio_bytes_per_token: float
    avg_token_length: float


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

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

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
        """Greedily fuses adjacent tokens until no SuperBPE merge remains."""
        cross = self._cross_word_tokens()
        if not cross:
            return tokens
        current = tokens
        while True:
            merged: List[str] = []
            changed = False
            i = 0
            while i < len(current):
                if i + 1 < len(current) and current[i] + current[i + 1] in cross:
                    merged.append(current[i] + current[i + 1])
                    i += 2
                    changed = True
                else:
                    merged.append(current[i])
                    i += 1
            if not changed:
                return merged
            current = merged

    def _apply_cross_word_merges_with_spans(self, tokens: List[Token]) -> List[Token]:
        cross = self._cross_word_tokens()
        if not cross:
            return tokens
        current = tokens
        while True:
            merged: List[Token] = []
            changed = False
            i = 0
            while i < len(current):
                if i + 1 < len(current) and current[i].text + current[i + 1].text in cross:
                    a, b = current[i], current[i + 1]
                    text = a.text + b.text
                    merged.append(
                        Token(
                            text=text,
                            id=self.model.token_to_id[text],
                            raw_span=(a.raw_span[0], b.raw_span[1]),
                        )
                    )
                    i += 2
                    changed = True
                else:
                    merged.append(current[i])
                    i += 1
            if not changed:
                return merged
            current = merged

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
        ranking_strategy: str = "char_savings",
        adaptive_multiplier: bool = False,
        max_edges_per_node: Optional[int] = None,
        min_edge_log_prob: Optional[float] = None,
        convergence_tolerance: float = 1e-4,
        script_balance_temperature: Optional[float] = None,
        min_boundary_entropy: Optional[float] = None,
        verbose: bool = True,
    ) -> CustomTokenizer:
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer(split_digits=split_digits)

        combined_special = (
            list(SeedVocabularyBuilder.DEFAULT_SPECIAL_TOKENS) if special_tokens is None else list(special_tokens)
        )
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

        if not chunks:
            raise ValueError(
                "Empty corpus: no pre-tokenized chunks were produced. "
                "Provide non-empty text (and disable compress_indents if it "
                "reduces everything to whitespace)."
            )

        trainer = UnigramTrainer(
            target_vocab_size=target_vocab_size,
            seed_multiplier=seed_multiplier,
            max_ngram_length=max_ngram_length,
            min_frequency=min_frequency,
            byte_fallback=byte_fallback,
            special_tokens=combined_special if combined_special else None,
            ranking_strategy=ranking_strategy,
            adaptive_multiplier=adaptive_multiplier,
            max_edges_per_node=max_edges_per_node,
            min_edge_log_prob=min_edge_log_prob,
            convergence_tolerance=convergence_tolerance,
            script_balance_temperature=script_balance_temperature,
            min_boundary_entropy=min_boundary_entropy,
        )

        model = trainer.train(chunks, verbose=verbose)
        return cls(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=model)

    def encode(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
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
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
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

    def encode_batch(
        self,
        texts: Sequence[str],
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        num_workers: Optional[int] = None,
    ) -> List[List[str]]:
        """Encodes a sequence of texts, parallelizing across workers when batch is large."""
        if not texts:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")
        if len(texts) <= 64 or num_workers == 1:
            return [
                self.encode(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special_action=disallowed_special_action,
                )
                for t in texts
            ]

        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda t: self.encode(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special_action=disallowed_special_action,
                    ),
                    texts,
                )
            )

    def encode_to_ids_batch(
        self,
        texts: Sequence[str],
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        num_workers: Optional[int] = None,
    ) -> List[List[int]]:
        """Encodes a sequence of texts to token IDs, parallelizing across workers when batch is large."""
        if not texts:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")
        if len(texts) <= 64 or num_workers == 1:
            return [
                self.encode_to_ids(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special_action=disallowed_special_action,
                )
                for t in texts
            ]

        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda t: self.encode_to_ids(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special_action=disallowed_special_action,
                    ),
                    texts,
                )
            )

    def encode_with_offsets_batch(
        self,
        texts: Sequence[str],
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        num_workers: Optional[int] = None,
    ) -> List[List[Token]]:
        """Encodes a sequence of texts with exact spans, parallelizing across workers when batch is large."""
        if not texts:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")
        if len(texts) <= 64 or num_workers == 1:
            return [
                self.encode_with_offsets(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special_action=disallowed_special_action,
                )
                for t in texts
            ]

        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda t: self.encode_with_offsets(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special_action=disallowed_special_action,
                    ),
                    texts,
                )
            )

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
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
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

    def encode_with_metrics(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> TokenizationReport:
        """
        Encodes text and computes runtime diagnostic metrics (byte-fallback rate, compression ratio).
        """
        tokens_with_offsets = self.encode_with_offsets(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        tokens = [t.text for t in tokens_with_offsets]
        token_ids = [t.id for t in tokens_with_offsets]
        token_spans = [t.raw_span for t in tokens_with_offsets]
        raw_bytes = text.encode("utf-8")
        num_tokens = len(tokens)
        num_bytes = len(raw_bytes)
        num_chars = len(text)

        byte_fallback_tokens = sum(1 for t in tokens if t.startswith("<0x") and t.endswith(">") and len(t) == 6)
        byte_fallback_rate = (byte_fallback_tokens / num_tokens) if num_tokens > 0 else 0.0
        compression_ratio = (num_bytes / num_tokens) if num_tokens > 0 else 0.0
        avg_token_len = (sum(len(t) for t in tokens) / num_tokens) if num_tokens > 0 else 0.0

        return TokenizationReport(
            tokens=tokens,
            token_ids=token_ids,
            token_spans=token_spans,
            num_tokens=num_tokens,
            num_bytes=num_bytes,
            num_chars=num_chars,
            byte_fallback_tokens=byte_fallback_tokens,
            byte_fallback_rate=byte_fallback_rate,
            compression_ratio_bytes_per_token=compression_ratio,
            avg_token_length=avg_token_len,
        )

    @property
    def _indent_compression_enabled(self) -> bool:
        return any(tok in self.model.special_tokens for tok in IndentationCompressor.INDENT_SPECIAL_TOKENS)

    def decode(self, token_ids: List[int]) -> str:
        decoded = self.model.decode(token_ids, space_char=self.normalizer.space_char)
        if self._indent_compression_enabled:
            decoded = IndentationCompressor.decompress_indents(decoded)
        return self.normalizer.restore_escaped_metaspace(decoded)

    def decode_tokens(self, tokens: Sequence[str]) -> str:
        """Decodes a list of token strings directly back to the original text string."""
        unk_id = self.model.token_to_id.get("<|unk|>", 0)
        token_ids = [self.model.token_to_id.get(t, unk_id) for t in tokens]
        return self.decode(token_ids)

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
