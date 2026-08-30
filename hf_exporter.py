from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

from pre_tokenizer import Normalizer
from tokenizer import CustomTokenizer


class HuggingFaceExporter:
    """
    HuggingFace 'tokenizers' Standard Schema Exporter (v1.0).

    Serializes a CustomTokenizer into the canonical HuggingFace tokenizer.json schema,
    enabling direct integration with transformers.AutoTokenizer.from_pretrained().
    """

    @staticmethod
    def export_to_hf_dict(tokenizer: CustomTokenizer) -> Dict[str, Any]:
        """
        Converts internal model parameters into a HuggingFace v1.0 JSON-compliant dictionary.
        """
        model = tokenizer.model
        normalizer = tokenizer.normalizer

        # 1. Build added_tokens (special tokens)
        added_tokens: List[Dict[str, Any]] = []
        special_set = set(model.special_tokens)
        for token_str, token_id in model.token_to_id.items():
            if token_str in special_set or (token_str.startswith("<|") and token_str.endswith("|>")):
                added_tokens.append(
                    {
                        "id": token_id,
                        "content": token_str,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                )

        # HF Unigram IDs are array positions. Reject sparse or duplicate IDs
        # instead of emitting a tokenizer whose IDs decode to different tokens.
        sorted_tokens = sorted(model.token_to_id.items(), key=lambda item: item[1])
        ids = [token_id for _, token_id in sorted_tokens]
        if ids != list(range(len(ids))):
            raise ValueError("HuggingFace Unigram export requires contiguous token IDs starting at 0")
        vocab_list = [[tok, model.vocab.get(tok, -10.0)] for tok, _ in sorted_tokens]

        unk_id = model.token_to_id.get(model.unk_token)
        if unk_id is None:
            raise ValueError(
                "HuggingFace Unigram export requires the configured unknown token "
                f"{model.unk_token!r} in the vocabulary"
            )

        # Build a normalizer Sequence mirroring Normalizer.normalize's order:
        # NFKC -> unicode-space map -> punctuation map -> lowercase ->
        # whitespace collapse -> strip.
        normalizers: List[Dict[str, Any]] = []
        if normalizer.normalize_unicode:
            normalizers.append({"type": "NFKC"})
        if normalizer.normalize_unicode_spaces:
            normalizers.append(
                {
                    "type": "Replace",
                    "pattern": {"Regex": "[\\u00A0\\u1680\\u2000-\\u200A\\u202F\\u205F\\u3000]"},
                    "content": " ",
                }
            )
        if normalizer.normalize_punctuation:
            for code, repl in Normalizer.PUNCT_MAP.items():
                normalizers.append({"type": "Replace", "pattern": {"String": chr(code)}, "content": repl})
        if normalizer.lowercase:
            normalizers.append({"type": "Lowercase"})
        if normalizer.collapse_whitespaces:
            normalizers.append({"type": "Replace", "pattern": {"Regex": "[ \\t]+"}, "content": " "})
        if normalizer.strip_whitespace:
            normalizers.append({"type": "Strip"})

        pre = tokenizer.pre_tokenizer
        hf_pre_tokenizers: List[Dict[str, Any]] = []
        if pre.split_digits:
            hf_pre_tokenizers.append({"type": "Digits", "individual_numbers": True})
        hf_pre_tokenizers.append(
            {
                "type": "Metaspace",
                "replacement": normalizer.space_char,
                # UniqToken replaces existing whitespace but does not prepend a
                # metaspace token to text with no leading whitespace.
                "prepend_scheme": "never",
                "split": True,
            }
        )
        if pre.split_punctuation:
            # HF variant enum is capitalized
            hf_pre_tokenizers.append({"type": "Punctuation", "behavior": "Isolated"})
        hf_pre_tokenizer: Dict[str, Any] = (
            hf_pre_tokenizers[0]
            if len(hf_pre_tokenizers) == 1
            else {"type": "Sequence", "pretokenizers": hf_pre_tokenizers}
        )

        unrepresentable = []
        if pre.hex_literals:
            unrepresentable.append("hex_literals")
        if pre.digit_chunk_size is not None:
            unrepresentable.append("digit_chunk_size")
        if pre.preset is not None:
            unrepresentable.append("preset")
        if not pre.keep_special_tokens:
            unrepresentable.append("keep_special_tokens=False")
        if normalizer.casefold:
            unrepresentable.append("casefold")
        if unrepresentable:
            warnings.warn(
                "HuggingFace export cannot fully represent this UniqToken pre-tokenizer "
                f"configuration ({', '.join(unrepresentable)}); the exported tokenizer "
                "may tokenize differently from the source model.",
                stacklevel=2,
            )

        hf_schema: Dict[str, Any] = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": {"type": "Sequence", "normalizers": normalizers},
            "pre_tokenizer": hf_pre_tokenizer,
            "post_processor": None,
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "ByteFallback"},
                    {
                        "type": "Metaspace",
                        "replacement": normalizer.space_char,
                        "prepend_scheme": "never",
                        "split": True,
                    },
                ],
            },
            "model": {
                "type": "Unigram",
                "unk_id": unk_id,
                "byte_fallback": model.byte_fallback,
                "vocab": vocab_list,
            },
        }

        return hf_schema

    @classmethod
    def save_hf_pretrained(cls, tokenizer: CustomTokenizer, output_dir: Union[str, Path]) -> None:
        """
        Saves tokenizer.json and tokenizer_config.json into output_dir.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        hf_json = cls.export_to_hf_dict(tokenizer)
        with open(out_path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(hf_json, f, ensure_ascii=False, indent=2)

        config_json = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_type": "unigram",
            "unk_token": tokenizer.model.unk_token,
            "bos_token": "<|bos|>" if "<|bos|>" in tokenizer.model.token_to_id else None,
            "eos_token": "<|eos|>" if "<|eos|>" in tokenizer.model.token_to_id else None,
            "pad_token": "<|pad|>" if "<|pad|>" in tokenizer.model.token_to_id else None,
        }

        with open(out_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config_json, f, ensure_ascii=False, indent=2)
