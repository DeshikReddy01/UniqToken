from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bpe_model import BPEModel
from pre_tokenizer import Normalizer, RegexPreTokenizer
from tokenizer import CustomTokenizer
from unigram_trainer import UnigramModel

try:
    # ByteLevel/GPT-2 pre-tokenization needs \p{L}/\p{N}; same requirement as
    # the tiktoken adapter.
    import regex as _re
    from tiktoken_adapter import TIKTOKEN_PATTERNS
except ImportError:  # pragma: no cover
    _re = None
    TIKTOKEN_PATTERNS = {}

DEFAULT_SPECIAL_PATTERN = r"<\|[^\s|]+\|>"


def _warn_unsupported(component: str, detail: str) -> None:
    warnings.warn(
        f"HF importer: {component} ({detail}) has no exact Caliper equivalent; "
        "imported tokenizer may tokenize differently than the source.",
        stacklevel=3,
    )


def _map_normalizer(cfg: Any) -> Normalizer:
    """Maps an HF normalizer config onto Caliper's Normalizer (best effort)."""
    cfg = cfg or {}
    ntype = cfg.get("type")
    kwargs: Dict[str, Any] = {"space_char": "\u2581"}

    def walk(node: Any) -> None:
        if not node:
            return
        if node.get("type") == "Sequence":
            for child in node.get("normalizers", []):
                walk(child)
            return
        t = node.get("type")
        if t == "NFKC":
            kwargs["normalize_unicode"] = True
        elif t == "Lowercase":
            kwargs["lowercase"] = True
        elif t in ("NFC", "NFD", "NFKD", "StripAccents", "Replace", "Strip", "Prepend", "ByteLevel"):
            _warn_unsupported("normalizer", t)
        else:
            _warn_unsupported("normalizer", t or "unknown")

    walk(cfg)
    return Normalizer(**kwargs)


def _map_pre_tokenizer(cfg: Any, normalizer: Normalizer) -> RegexPreTokenizer:
    """Maps an HF pre-tokenizer config onto Caliper's RegexPreTokenizer (best effort)."""
    cfg = cfg or {}
    ptype = cfg.get("type")
    space_char = "\u2581"
    if ptype == "Metaspace":
        space_char = cfg.get("replacement", "\u2581") or "\u2581"
        prepend = cfg.get("prepend_scheme") or ("always" if cfg.get("add_prefix_space") else "never")
        if prepend != "never":
            _warn_unsupported("pre_tokenizer", f"Metaspace prepend_scheme={prepend!r} (Caliper never prepends)")
        if cfg.get("split") is False:
            _warn_unsupported("pre_tokenizer", "Metaspace split=False")
    elif ptype not in (None, "Metaspace"):
        _warn_unsupported("pre_tokenizer", ptype or "unknown")
    elif ptype is None:
        _warn_unsupported("pre_tokenizer", "none configured; Caliper's default regex will be used")

    return RegexPreTokenizer(space_char=space_char)


def import_hf_unigram(data: Dict[str, Any]) -> CustomTokenizer:
    """
    Imports an HF ``tokenizer.json`` (parsed dict) with a Unigram model into a
    Caliper :class:`CustomTokenizer`.

    Vocab scores and token IDs are preserved exactly. Normalizer/pre-tokenizer
    components are mapped best-effort; anything without an exact Caliper
    equivalent emits a warning. Requires ``byte_fallback`` or an unk token
    spelled ``<|unk|>`` for OOV characters (HF's custom unk strings are not
    wired into Caliper's lattice fallback).
    """
    model = data.get("model", {})
    if model.get("type") != "Unigram":
        raise ValueError(f"expected a Unigram model, got {model.get('type')!r}")

    vocab_list = model.get("vocab", [])
    if not vocab_list:
        raise ValueError("HF Unigram model has an empty vocab")

    vocab: Dict[str, float] = {}
    token_to_id: Dict[str, int] = {}
    id_to_token: Dict[int, str] = {}
    for idx, entry in enumerate(vocab_list):
        token, score = entry[0], float(entry[1])
        vocab[token] = score
        token_to_id[token] = idx
        id_to_token[idx] = token

    special_tokens: List[str] = []
    for added in data.get("added_tokens", []):
        token = added["content"]
        special_tokens.append(token)
        if token not in token_to_id:
            token_to_id[token] = added["id"]
            id_to_token[added["id"]] = token
            vocab.setdefault(token, -10.0)  # HF Unigram needs a score for every id

    byte_fallback = bool(model.get("byte_fallback", False))
    unk_id = model.get("unk_id")
    if not byte_fallback and (unk_id is None or id_to_token.get(unk_id) != "<|unk|>"):
        _warn_unsupported(
            "OOV fallback",
            f"byte_fallback=False and unk token {id_to_token.get(unk_id, '<none>')!r} is not '<|unk|>'",
        )

    normalizer = _map_normalizer(data.get("normalizer"))
    pre_tokenizer = _map_pre_tokenizer(data.get("pre_tokenizer"), normalizer)

    unigram = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        special_tokens=special_tokens,
        max_subword_len=max(max(len(t) for t in vocab), 1),
        byte_fallback=byte_fallback,
    )
    return CustomTokenizer(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=unigram)


def _bytes_to_unicode() -> Dict[int, str]:
    """Standard GPT-2 byte-to-unicode table (every byte -> printable char)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class HFByteLevelBPE:
    """
    Byte-level BPE encoding loaded from an HF ``tokenizer.json`` (GPT-2 style).

    Produces the SAME integer IDs as the equivalent HuggingFace ``tokenizers``
    Tokenizer (ByteLevel pre-tokenizer + BPE model + ByteLevel decoder) using
    tiktoken's GPT-2 split pattern. Requires the ``regex`` package.
    """

    def __init__(
        self,
        name: str,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_tokens: Optional[Dict[str, int]] = None,
        add_prefix_space: bool = False,
    ):
        if _re is None:
            raise ImportError("the 'regex' package is required for ByteLevel BPE import")
        self.name = name
        self.vocab = dict(vocab)
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = dict(special_tokens or {})
        self._id_to_special = {v: k for k, v in self.special_tokens.items()}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.add_prefix_space = add_prefix_space
        self._split_re = _re.compile(TIKTOKEN_PATTERNS["gpt2"])
        self.n_vocab = max(self.vocab.values(), default=-1) + 1

    @property
    def vocab_size(self) -> int:
        return self.n_vocab

    def _bpe(self, piece: str) -> List[str]:
        # ponytail: O(n^2) greedy lowest-rank merge — same result as HF's BPE,
        # heap version is an optimization not a semantic change.
        parts: List[str] = list(piece)
        while len(parts) > 1:
            best_rank: Optional[int] = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self.ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return parts

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        if self.add_prefix_space:
            text = " " + text
        for piece in self._split_re.findall(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for token in self._bpe(mapped):
                tid = self.vocab.get(token)
                if tid is None:
                    raise ValueError(f"no vocab id for token {token!r} in {self.name}")
                ids.append(tid)
        return ids

    def encode(self, text: str, allowed_special: Union[str, set] = "none") -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none" or not allowed_special:
            allowed = set()
        else:
            allowed = set(allowed_special)

        if not allowed:
            return self._encode_ordinary(text)

        import re as _stdlib_re

        special_re = _stdlib_re.compile(
            "(" + "|".join(_stdlib_re.escape(s) for s in sorted(allowed, key=len, reverse=True)) + ")"
        )
        ids: List[int] = []
        for segment in special_re.split(text):
            if segment in allowed:
                ids.append(self.special_tokens[segment])
            elif segment:
                ids.extend(self._encode_ordinary(segment))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        pieces: List[bytes] = []
        for tid in token_ids:
            if tid in self._id_to_special:
                pieces.append(self._id_to_special[tid].encode("utf-8"))
                continue
            token = next((t for t, i in self.vocab.items() if i == tid), None)
            if token is None:
                raise ValueError(f"unknown token id {tid} in {self.name}")
            pieces.append(bytes(self.byte_decoder[c] for c in token))
        return b"".join(pieces).decode("utf-8", errors="replace")


def import_hf_bpe(data: Dict[str, Any]) -> Union[HFByteLevelBPE, BPEModel]:
    """
    Imports an HF ``tokenizer.json`` BPE model.

    ByteLevel pre-tokenization (GPT-2 style) returns a fully functional
    :class:`HFByteLevelBPE` with exact-ID encode/decode. Non-byte-level BPE
    returns a Caliper :class:`BPEModel` carrying vocab/merges/IDs for data
    reuse — encode semantics depend on the source pre-tokenizer, which Caliper
    cannot reproduce in general.
    """
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE model, got {model.get('type')!r}")

    vocab: Dict[str, int] = dict(model.get("vocab", {}))
    merges: List[Tuple[str, str]] = []
    for entry in model.get("merges", []):
        if isinstance(entry, str):
            parts = entry.split(" ")
            if len(parts) != 2:
                raise ValueError(f"malformed merge entry: {entry!r}")
            merges.append((parts[0], parts[1]))
        else:
            merges.append((entry[0], entry[1]))

    special_tokens = {a["content"]: a["id"] for a in data.get("added_tokens", []) if a.get("special")}
    pt = data.get("pre_tokenizer") or {}
    pt_type = pt.get("type")
    byte_cfg: Dict[str, Any] = {}
    if pt_type == "Sequence":
        for child in pt.get("pretokenizers", []):
            if child.get("type") == "ByteLevel":
                byte_cfg = child
                pt_type = "ByteLevel"
                break

    if pt_type == "ByteLevel":
        return HFByteLevelBPE(
            name="hf_bpe",
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
            add_prefix_space=bool(byte_cfg.get("add_prefix_space", False)),
        )

    _warn_unsupported("pre_tokenizer", f"{pt_type!r} BPE import returns vocab/merges only")
    id_to_token = {i: t for t, i in vocab.items()}
    return BPEModel(
        vocab=set(vocab),
        token_to_id=vocab,
        id_to_token=id_to_token,
        merges={pair: i for i, pair in enumerate(merges)},
        special_tokens=list(special_tokens),
        byte_fallback=False,
    )


def import_hf_tokenizer(source: Union[str, Path, Dict[str, Any]]) -> Union[CustomTokenizer, HFByteLevelBPE, BPEModel]:
    """
    Imports an HF ``tokenizer.json`` file (or parsed dict), dispatching on the
    model type: Unigram -> CustomTokenizer, BPE -> HFByteLevelBPE/BPEModel.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "tokenizer.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    mtype = data.get("model", {}).get("type")
    if mtype == "Unigram":
        return import_hf_unigram(data)
    if mtype == "BPE":
        return import_hf_bpe(data)
    raise NotImplementedError(
        f"HF model type {mtype!r} is not supported (Caliper has no WordPiece engine); supported types: Unigram, BPE"
    )
