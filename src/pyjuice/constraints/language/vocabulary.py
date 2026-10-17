"""
The text every token id stands for: the vocabulary text-level constraints compile through.

A :class:`Vocabulary` maps each token id in ``[0, size)`` to the bytes it spells, or to no text at all (special
tokens, an unknown-token id, ids a model has beyond its tokenizer). The TEXT of a token sequence is its tokens' bytes
concatenated, exactly as they spell it: no leading space stripped and no space removed before punctuation (what some
tokenizers' ``decode`` does), so a constraint on a continuation sees the continuation's own text. Ids without text
never occur inside text; the end-of-sequence ids (:attr:`Vocabulary.eos`), which carry no text either, mark where it
ends.

Any tokenizer reduces to the constructor's list of byte strings (tiktoken, say:
``Vocabulary([enc.decode_single_token_bytes(v) for v in range(enc.n_vocab)], eos = [enc.eot_token])``);
:meth:`Vocabulary.from_tokenizer` reads a Hugging Face tokenizer without importing ``transformers`` or ``tokenizers``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from .base import Tokens, _as_tokens


def _gpt2_bytes_to_chars():
    """GPT-2's printable stand-ins for the 256 bytes (what byte-level BPE vocabularies are written in)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_OF_CHAR = _gpt2_bytes_to_chars()
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


class Vocabulary:
    """
    The bytes every token id spells (see the module docstring).

    :param token_bytes: the text of token ``v`` at ``token_bytes[v]``, ``None`` for an id with no text
    :type token_bytes: Sequence[Optional[bytes]]

    :param eos: the end-of-sequence ids; they carry no text (bytes given for them are ignored)
    :type eos: Iterable[int]

    :param size: the number of token ids, at least ``len(token_bytes)`` (the default); the ids past the list have no
        text -- e.g. a model's embedding rows beyond its tokenizer's
    :type size: Optional[int]
    """

    def __init__(self, token_bytes: Sequence[Optional[bytes]], eos: Iterable[int] = (), size: Optional[int] = None):
        n = len(token_bytes)
        size = n if size is None else int(size)
        if size < 1:
            raise ValueError(f"A vocabulary needs at least one token id, got size {size}.")
        if size < n:
            raise ValueError(f"`size` ({size}) is smaller than the number of tokens given ({n}).")
        eos = tuple(sorted({int(e) for e in eos}))
        outside = [e for e in eos if not 0 <= e < size]
        if outside:
            raise ValueError(f"End-of-sequence ids {outside} lie outside the vocabulary [0, {size}).")
        no_text = set(eos)
        has_text = np.zeros(size, dtype = bool)
        lengths = np.zeros(size, dtype = np.int64)
        parts = []
        for v, b in enumerate(token_bytes):
            if b is None or v in no_text:
                continue
            if not isinstance(b, (bytes, bytearray)):
                raise TypeError(f"Token {v}'s text must be bytes or None, got {type(b).__name__} "
                                f"(encode text with `.encode()`).")
            has_text[v] = True
            lengths[v] = len(b)
            parts.append(bytes(b))
        offsets = np.zeros(size + 1, dtype = np.int64)
        np.cumsum(lengths, out = offsets[1:])
        self._size = size
        self._eos = eos
        self._has_text = has_text
        self._offsets = offsets
        self._buffer = b"".join(parts)
        self._fingerprint = None

    # ---------------------------------------------------------------------------------------------
    # Reading tokenizers
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def from_tokenizer(cls, tokenizer, size: Optional[int] = None, eos: Optional[Iterable[int]] = None) -> "Vocabulary":
        """
        A Hugging Face tokenizer's vocabulary: a fast ``transformers`` tokenizer (read through its
        ``backend_tokenizer``) or a ``tokenizers.Tokenizer``, by how its decoder turns tokens into text:

        * byte-level BPE (GPT-2, Llama 3, Qwen, LLaDA, ...): every character of a token stands for one byte;
        * SentencePiece with byte fallback (Llama 2, Mistral, ...): ``▁`` is a space, ``<0xNN>`` the byte ``NN``;
        * SentencePiece without byte fallback (e.g. T5): ``▁`` is a space.

        Special tokens and the unknown token have no text; other added tokens have their content. Other decoders --
        WordPiece's, whose spaces depend on the neighbouring tokens, among them -- raise :class:`NotImplementedError`:
        build the vocabulary from the tokens' bytes with the constructor instead.

        :param tokenizer: the tokenizer
        :param size: the number of token ids (default: the tokenizer's, its added tokens included)
        :type size: Optional[int]

        :param eos: the end-of-sequence ids (default: the tokenizer's ``eos_token_id``, when it has one)
        :type eos: Optional[Iterable[int]]
        """
        backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
        if not (hasattr(backend, "to_str") and hasattr(backend, "get_vocab")):
            raise TypeError(f"Expected a Hugging Face fast tokenizer or a `tokenizers.Tokenizer`, got "
                            f"{type(tokenizer).__name__} (load it with `use_fast = True`, or build the Vocabulary "
                            f"from the tokens' bytes).")
        spec = json.loads(backend.to_str())
        read = _token_reader(spec.get("decoder"))
        pieces = backend.get_vocab(with_added_tokens = True)              # token string -> id
        num = max([backend.get_vocab_size(with_added_tokens = True)] + [v + 1 for v in pieces.values()])
        strings = [None] * num
        for s, v in pieces.items():
            strings[v] = s
        special = {int(a["id"]) for a in spec.get("added_tokens", []) if a.get("special")}
        special.update(int(v) for v in getattr(tokenizer, "all_special_ids", ()) or ())
        unk = _unknown_id(spec.get("model") or {}, pieces)
        if unk is not None:
            special.add(unk)
        added = {int(a["id"]): a["content"] for a in spec.get("added_tokens", []) if not a.get("special")}
        token_bytes = []
        for v, s in enumerate(strings):
            if s is None or v in special:
                token_bytes.append(None)
            elif v in added:
                token_bytes.append(added[v].encode("utf-8"))
            else:
                token_bytes.append(read(v, s))
        if eos is None:
            e = getattr(tokenizer, "eos_token_id", None)
            eos = () if e is None else (e,)
        return cls(token_bytes, eos = eos, size = size)

    # ---------------------------------------------------------------------------------------------
    # Tables
    # ---------------------------------------------------------------------------------------------

    @property
    def size(self) -> int:
        """The number of token ids (what a constraint over this vocabulary takes as its ``vocab_size``)."""
        return self._size

    def __len__(self) -> int:
        return self._size

    @property
    def eos(self) -> Tuple[int, ...]:
        """The end-of-sequence ids, ascending."""
        return self._eos

    @property
    def has_text(self) -> torch.Tensor:
        """Bool mask [size]: the ids with text (the others never occur inside text)."""
        return torch.from_numpy(self._has_text.copy())

    def token_bytes(self, token: int) -> Optional[bytes]:
        """The bytes ``token`` spells, ``None`` if it has no text."""
        v = self._check(token)
        if not self._has_text[v]:
            return None
        return self._buffer[self._offsets[v]:self._offsets[v + 1]]

    def decode(self, tokens: Tokens) -> bytes:
        """The text of a token sequence: its tokens' bytes, concatenated. Raises :class:`ValueError` on an id with no
        text (an end-of-sequence id, a special token, an id past the tokenizer's)."""
        out = []
        for t in _as_tokens(tokens):
            v = self._check(t)
            if not self._has_text[v]:
                kind = "an end-of-sequence id" if v in self._eos else "an id with no text"
                raise ValueError(f"Token {v} is {kind}; it does not occur inside text.")
            out.append(self._buffer[self._offsets[v]:self._offsets[v + 1]])
        return b"".join(out)

    def _check(self, token: int) -> int:
        v = int(token)
        if not 0 <= v < self._size:
            raise ValueError(f"Token id {v} is outside the vocabulary [0, {self._size}).")
        return v

    # ---------------------------------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------------------------------

    def fingerprint(self) -> str:
        """A stable hex digest of the vocabulary (same across processes); it also defines equality."""
        if self._fingerprint is None:
            h = hashlib.sha256()
            h.update(b"Vocabulary")
            h.update(np.array([self._size, len(self._eos)], dtype = np.int64).tobytes())
            h.update(np.asarray(self._eos, dtype = np.int64).tobytes())
            h.update(self._has_text.tobytes())
            h.update(self._offsets.tobytes())
            h.update(self._buffer)
            self._fingerprint = h.hexdigest()
        return self._fingerprint

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vocabulary) and self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __repr__(self) -> str:
        return f"Vocabulary(size={self._size}, with_text={int(self._has_text.sum())}, eos={self._eos})"


def _decoders_of(decoder) -> list:
    if not decoder:
        return []
    if decoder.get("type") == "Sequence":
        return [d for sub in decoder.get("decoders", []) for d in _decoders_of(sub)]
    return [decoder]


def _token_reader(decoder):
    """``read(v, s)``: the bytes token ``v`` (vocabulary string ``s``) spells, by the tokenizer's decoder."""
    parts = _decoders_of(decoder)
    types = [d.get("type") for d in parts]
    if "ByteLevel" in types and set(types) <= {"ByteLevel", "Fuse", "Strip"}:
        def read_byte_level(v, s):
            try:
                return bytes(_BYTE_OF_CHAR[ch] for ch in s)
            except KeyError as e:
                raise ValueError(f"Token {v} ({s!r}) is not written in GPT-2's byte map, which its byte-level "
                                 f"decoder reads (character {e.args[0]!r}).") from None
        return read_byte_level
    if types and set(types) <= {"Replace", "ByteFallback", "Fuse", "Strip", "Metaspace"}:
        space = None
        for d in parts:
            if d["type"] == "Metaspace":
                space = d.get("replacement", "▁")
            elif d["type"] == "Replace":
                pattern = d.get("pattern", {})
                if "String" not in pattern or d.get("content") != " " or len(pattern["String"]) != 1:
                    raise NotImplementedError(f"A Replace decoder other than a space marker's ({d}) is not "
                                              f"supported; build the Vocabulary from the tokens' bytes.")
                space = pattern["String"]
        if space is None:
            raise NotImplementedError(f"Tokenizers whose decoder is {types} are not supported; build the "
                                      f"Vocabulary from the tokens' bytes.")
        fallback = "ByteFallback" in types

        def read_sentencepiece(v, s):
            if fallback:
                m = _BYTE_TOKEN.fullmatch(s)
                if m:
                    return bytes([int(m.group(1), 16)])
            return s.replace(space, " ").encode("utf-8")
        return read_sentencepiece
    raise NotImplementedError(f"Tokenizers whose decoder is {types or 'missing'} are not supported (WordPiece's spaces, "
                              f"for one, depend on the neighbouring tokens); build the Vocabulary from the tokens' "
                              f"bytes.")


def _unknown_id(model: dict, pieces: dict) -> Optional[int]:
    """The id of the model's unknown token, if it has one."""
    if model.get("unk_id") is not None:                                # Unigram
        return int(model["unk_id"])
    unk = model.get("unk_token")                                       # BPE, WordPiece
    if unk is not None and unk in pieces:
        return int(pieces[unk])
    return None
