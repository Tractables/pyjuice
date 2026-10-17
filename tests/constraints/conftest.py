"""Fixtures shared by every test of :mod:`pyjuice.constraints`."""
import pytest
import torch


@pytest.fixture(autouse = True)
def nan_fresh_storage(monkeypatch):
    """The dfa_exact backend's new storage filled with NaN (it is not filled in use, see ``state.FRESH_FILL``): a
    query reading a slot no kernel wrote gives NaN, and fails the test."""
    from pyjuice.constraints.backends.dfa_exact import state
    monkeypatch.setattr(state, "FRESH_FILL", float("nan"))


def word_kinds(strings, separators, special_ids = ()) -> torch.Tensor:
    """
    Every token's kind for :func:`word_count` from its text as it appears mid-sequence (``strings[v]``):
    ``2 * (starts with a separator) + (contains a letter or digit)``, 0 for the ``special_ids`` and tokens of no
    text -- Ctrl-G's ``WordCountBuilder`` classification.
    """
    separators = set(separators)
    special = set(special_ids)
    kinds = torch.zeros(len(strings), dtype = torch.long)
    for v, text in enumerate(strings):
        if v in special or len(text) == 0:
            continue
        has_word_char = any(ch.isalpha() or ch.isdigit() for ch in text)
        kinds[v] = 2 * (text[0] in separators) + has_word_char
    return kinds


def word_count(lo: int, hi: int, token_kind):
    """
    A :class:`~pyjuice.constraints.DFA` of the sequences of ``lo`` to ``hi`` words (inclusive), word boundaries read
    off every token's kind (:func:`word_kinds`)::

        token_kind[v] = 2 * (v starts with a separator) + (v contains a letter or digit)

    A word starts at a token with a letter or digit that begins with a separator or follows one; the sequence is
    taken as preceded by a separator -- Ctrl-G's ``WordCountBuilder``.

    :param token_kind: int tensor of size [vocab_size] with values in {0, 1, 2, 3}
    """
    from pyjuice.constraints import DFA
    token_kind = torch.as_tensor(token_kind, dtype = torch.long)
    if not 0 <= lo <= hi:
        raise ValueError(f"Need 0 <= lo <= hi, got lo={lo}, hi={hi}.")
    if token_kind.dim() != 1 or token_kind.min() < 0 or token_kind.max() > 3:
        raise ValueError("`token_kind` must be a 1-D tensor with values in {0, 1, 2, 3}.")
    # state 2k + s: k words so far, s = 1 if the previous token ended at a separator; 2(hi+1) = too many
    sink = 2 * (hi + 1)
    word = lambda k: 2 * (k + 1) if k + 1 <= hi else sink
    delta = []
    for k in range(hi + 1):
        # kinds: 0 = no sep / no word char, 1 = no sep / word char, 2 = sep / no word char, 3 = sep / word char
        delta.append([2 * k, 2 * k, 2 * k + 1, word(k)])            # s = 0: continue unless a separator
        delta.append([2 * k + 1, word(k), 2 * k + 1, word(k)])      # s = 1: any word char starts a word
    delta.append([sink] * 4)
    accept = [2 * k + s for k in range(lo, hi + 1) for s in (0, 1)]
    return DFA(token_kind.numel(), token_kind, delta, 1, accept).minimize()


@pytest.fixture(name = "word_kinds")
def word_kinds_fixture():
    """:func:`word_kinds`, for tests."""
    return word_kinds


@pytest.fixture(name = "word_count")
def word_count_fixture():
    """:func:`word_count`, for tests."""
    return word_count
