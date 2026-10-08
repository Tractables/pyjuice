"""
Tokenizer-dependent helpers for building constraints over text.

These read a Hugging Face tokenizer once and produce per-token tables; the automata themselves (in
:mod:`pyjuice.constraints.dfa`) only ever see token ids.
"""

from typing import Iterable, Optional

import torch

#: Ctrl-G's word separators.
DEFAULT_SEPARATORS = (" ", "\n", ",", ".", ":", ";", "\"", "/")


def token_strings(tokenizer, vocab_size: Optional[int] = None):
    """
    The text of every token as it appears mid-sequence (with its leading space, if any).

    Some tokenizers (e.g. Llama's SentencePiece) drop a token's leading space when it is decoded on its
    own, so each token is decoded after a special token and the special token's own text is removed --
    the approach Ctrl-G uses.
    """
    vocab_size = len(tokenizer) if vocab_size is None else vocab_size
    anchor = tokenizer.all_special_ids[0]
    anchor_text = tokenizer.decode([anchor])
    return [tokenizer.decode([anchor, v])[len(anchor_text):] for v in range(vocab_size)]


def word_token_kinds(tokenizer, vocab_size: Optional[int] = None,
                     separators: Iterable[str] = DEFAULT_SEPARATORS) -> torch.Tensor:
    """
    Per-token kinds for :meth:`DFA.word_count`: ``2 * (starts with a separator) + (contains a letter or
    digit)``. Special tokens (and tokens that decode to nothing) are kind 0. This reproduces Ctrl-G's
    ``WordCountBuilder`` classification.
    """
    return word_token_kinds_from_strings(token_strings(tokenizer, vocab_size),
                                         special_ids = tokenizer.all_special_ids, separators = separators)


def word_token_kinds_from_strings(strings, special_ids: Iterable[int] = (),
                                  separators: Iterable[str] = DEFAULT_SEPARATORS) -> torch.Tensor:
    """:func:`word_token_kinds` from the token texts directly (``strings[v]`` is token ``v``'s text)."""
    separators = set(separators)
    special = set(special_ids)
    kinds = torch.zeros(len(strings), dtype = torch.long)
    for v, text in enumerate(strings):
        if v in special or len(text) == 0:
            continue
        has_word_char = any(ch.isalpha() or ch.isdigit() for ch in text)
        kinds[v] = 2 * (text[0] in separators) + has_word_char
    return kinds
