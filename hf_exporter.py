from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

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

        # 2. Build Unigram vocabulary list [[token, score], ...] sorted by ID
        sorted_tokens = sorted(model.token_to_id.items(), key=lambda item: item[1])
        vocab_list = [[tok, model.vocab.get(tok, -10.0)] for tok, _ in sorted_tokens]

        unk_id = model.token_to_id.get("<|unk|>", 0)

        hf_schema: Dict[str, Any] = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": {
                "type": "Sequence",
                "normalizers": [{"type": "NFKC"}] if normalizer.normalize_unicode else [],
            },
            "pre_tokenizer": {
                "type": "Metaspace",
                "replacement": normalizer.space_char,
                # Caliper replaces existing whitespace but does not prepend a
                # metaspace token to text with no leading whitespace.
                "prepend_scheme": "never",
                "split": True,
            },
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
            "unk_token": "<|unk|>",
            "bos_token": "<|bos|>" if "<|bos|>" in tokenizer.model.token_to_id else None,
            "eos_token": "<|eos|>" if "<|eos|>" in tokenizer.model.token_to_id else None,
            "pad_token": "<|pad|>" if "<|pad|>" in tokenizer.model.token_to_id else None,
        }

        with open(out_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config_json, f, ensure_ascii=False, indent=2)
