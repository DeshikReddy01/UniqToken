from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tokenizer import CustomTokenizer


@dataclass
class BatchEncoding:
    """
    Container for batch-encoded sequences ready for Transformer model consumption.
    """

    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    tokens: List[List[str]]
    raw_spans: Optional[List[List[Tuple[int, int]]]] = None

    def to_dict(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    def to_torch(self):
        """
        Converts batch to PyTorch tensors if PyTorch is installed.
        """
        try:
            import torch

            return {
                "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
            }
        except ImportError:
            raise ImportError("PyTorch is not installed. Install torch to use to_torch().")


class BatchCollator:
    """
    Handles batch tokenization, padding, truncation, and attention mask generation.
    """

    def __init__(
        self,
        tokenizer: CustomTokenizer,
        padding_token: str = "<|pad|>",
        bos_token: Optional[str] = "<|bos|>",
        eos_token: Optional[str] = "<|eos|>",
    ):
        self.tokenizer = tokenizer
        self.pad_token = padding_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.pad_id = self.tokenizer.model.token_to_id.get(padding_token)
        self.bos_id = self.tokenizer.model.token_to_id.get(bos_token) if bos_token else None
        self.eos_id = self.tokenizer.model.token_to_id.get(eos_token) if eos_token else None

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = False,
        add_special_tokens: bool = True,
        sample: bool = False,
        alpha: float = 0.5,
    ) -> BatchEncoding:
        """
        Batch encodes a list of texts into aligned 2D token ID matrices and attention masks.
        """
        if max_length is not None and max_length < 0:
            raise ValueError("max_length must not be negative")
        if padding and self.pad_id is None:
            raise ValueError(f"padding token {self.pad_token!r} is not in the vocabulary")

        batch_ids: List[List[int]] = []
        batch_tokens: List[List[str]] = []
        unk_id = self.tokenizer.model.token_to_id.get("<|unk|>", 0)

        # 1. Check if native Rust Rayon batch acceleration is available
        use_rust_batch = False
        if not sample and hasattr(self.tokenizer.model, "_get_rust_trie"):
            try:
                import caliper_core

                rust_trie = self.tokenizer.model._get_rust_trie()
                if rust_trie is not None and len(texts) > 1:
                    # Preprocess / normalize chunks
                    norm_texts = [self.tokenizer.normalizer.normalize(t) for t in texts]
                    raw_spans_batch = caliper_core.rust_viterbi_decode_batch(
                        norm_texts,
                        rust_trie,
                        self.tokenizer.model.byte_fallback,
                    )
                    for spans in raw_spans_batch:
                        tokens = [s.token for s in spans]
                        ids = [s.token_id if s.token_id is not None else unk_id for s in spans]

                        if add_special_tokens:
                            if self.bos_id is not None and self.bos_token:
                                ids = [self.bos_id] + ids
                                tokens = [self.bos_token] + tokens
                            if self.eos_id is not None and self.eos_token:
                                ids = ids + [self.eos_id]
                                tokens = tokens + [self.eos_token]

                        if truncation and max_length is not None:
                            ids = ids[:max_length]
                            tokens = tokens[:max_length]

                        batch_ids.append(ids)
                        batch_tokens.append(tokens)
                    use_rust_batch = True
            except (ImportError, AttributeError, ValueError):
                use_rust_batch = False

        if not use_rust_batch:
            for text in texts:
                if sample:
                    tokens = self.tokenizer.sample(text, alpha=alpha)
                else:
                    tokens = self.tokenizer.encode(text)

                ids = [self.tokenizer.model.token_to_id.get(t, unk_id) for t in tokens]

                # Inject BOS / EOS if requested
                if add_special_tokens:
                    if self.bos_id is not None and self.bos_token:
                        ids = [self.bos_id] + ids
                        tokens = [self.bos_token] + tokens
                    if self.eos_id is not None and self.eos_token:
                        ids = ids + [self.eos_id]
                        tokens = tokens + [self.eos_token]

                # Truncation
                if truncation and max_length is not None:
                    ids = ids[:max_length]
                    tokens = tokens[:max_length]

                batch_ids.append(ids)
                batch_tokens.append(tokens)

        # Determine target sequence length
        if max_length is not None and padding:
            target_len = max_length
        elif padding:
            target_len = max(len(seq) for seq in batch_ids) if batch_ids else 0
        else:
            target_len = None

        if target_len is not None and any(len(sequence) > target_len for sequence in batch_ids):
            raise ValueError("a sequence exceeds max_length; enable truncation or increase max_length")

        # Build padded 2D matrices and attention masks
        padded_ids: List[List[int]] = []
        padded_tokens: List[List[str]] = []
        attention_masks: List[List[int]] = []

        pad_val = self.pad_id if self.pad_id is not None else self.tokenizer.model.token_to_id.get("<|unk|>", 0)
        for seq, tokens in zip(batch_ids, batch_tokens):
            seq_len = len(seq)
            if target_len is not None and seq_len < target_len:
                pad_len = target_len - seq_len
                padded_seq = seq + [pad_val] * pad_len
                padded_token_sequence = tokens + [self.pad_token] * pad_len
                mask = [1] * seq_len + [0] * pad_len
            else:
                padded_seq = list(seq)
                padded_token_sequence = list(tokens)
                mask = [1] * seq_len

            padded_ids.append(padded_seq)
            padded_tokens.append(padded_token_sequence)
            attention_masks.append(mask)

        return BatchEncoding(
            input_ids=padded_ids,
            attention_mask=attention_masks,
            tokens=padded_tokens,
        )
