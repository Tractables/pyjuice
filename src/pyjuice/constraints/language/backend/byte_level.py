"""
Byte-level automata: the layer text-level constraints (:class:`~pyjuice.constraints.Regex`, ...) compile through, to token
automata (:mod:`.token_lift`).

A :class:`ByteAutomaton` is a complete DFA over the 256 byte values whose language is a set of texts, byte for byte
(valid UTF-8 when it comes from a regular expression, :mod:`.patterns`, every character set as sequences of UTF-8 byte
ranges: RE2's split), with one dead state, kept per byte class (the bytes no state tells apart).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .counters import NotCounted
from .dfa import _column_classes

MAX_CODE_POINT = 0x10FFFF


# -------------------------------------------------------------------------------------------------
# Byte automata
# -------------------------------------------------------------------------------------------------

class ByteAutomaton:
    """
    A complete DFA over bytes, kept per BYTE CLASS: the bytes no state tells apart share a column.

    :ivar table: int32 [num_states, num_classes], the successor of every (state, class)
    :ivar byte_class: int64 [256], every byte's class (classes numbered by their smallest byte)
    :ivar delta: int32 [num_states, 256], the successor of every (state, byte) -- made on first use
    :ivar initial: the initial state
    :ivar accept: bool [num_states]
    :ivar dead: the one state no accepting state is reachable from (it loops on every byte)
    :ivar minimal: whether it is minimal (made so by :func:`~.automata.minimize` and the algebra's operations)
    :ivar counted: None, or the repetitions left out of it (:mod:`~.counters`): then it is not the automaton of its texts,
        reading its table raises :class:`~.counters.NotCounted`, and only :func:`~.token_lift.to_tokens` knows how to go on

    Made from ``delta`` (the first argument), or from ``table`` and ``byte_class`` (keywords; ``delta`` None); the
    other form is made on first use.
    """

    def __init__(self, delta, initial: int, accept, dead: int, *, table = None, byte_class = None,
                 minimal: bool = False):
        if (delta is None) == (table is None):
            raise ValueError("A ByteAutomaton takes delta, or table and byte_class.")
        self._delta = None if delta is None else np.ascontiguousarray(delta, dtype = np.int32)
        self._table = None if table is None else np.ascontiguousarray(table, dtype = np.int32)
        self._byte_class = None if byte_class is None else np.ascontiguousarray(byte_class, dtype = np.int64)
        self.initial, self.dead = int(initial), int(dead)
        self.accept = np.asarray(accept, dtype = bool)
        self.minimal = bool(minimal)
        self.counted = None

    @property
    def delta(self) -> np.ndarray:
        if self.counted:
            raise NotCounted
        if self._delta is None:
            self._delta = np.ascontiguousarray(self._table[:, self._byte_class])
        return self._delta

    def _classes(self):
        self._byte_class, first = _column_classes(self._delta)
        self._table = np.ascontiguousarray(self._delta[:, first])

    @property
    def table(self) -> np.ndarray:
        if self.counted:
            raise NotCounted
        if self._table is None:
            self._classes()
        return self._table

    @property
    def _kept_table(self) -> np.ndarray:
        """:attr:`table`, of an automaton with repetitions left out too."""
        if self._table is None:
            self._classes()
        return self._table

    @property
    def byte_class(self) -> np.ndarray:
        if self._table is None:
            self._classes()
        return self._byte_class

    @property
    def num_states(self) -> int:
        return (self._table if self._table is not None else self._delta).shape[0]

    def accepts(self, data: bytes) -> bool:
        table, byte_class = self.table, self.byte_class
        q = self.initial
        for b in data:
            q = int(table[q, byte_class[b]])
        return bool(self.accept[q])


def _encode(cp: int) -> bytes:
    return chr(cp).encode("utf-8")


def utf8_sequences(lo: int, hi: int) -> List[List[Tuple[int, int]]]:
    """The code points ``lo..hi`` (surrogates left out) as sequences of byte ranges: every UTF-8 encoding of one of
    them is matched by exactly one sequence, byte ``k`` within its ``k``-th range."""
    out, stack = [], [(lo, hi)]
    while stack:
        lo, hi = stack.pop()
        if lo > hi:
            continue
        if lo <= 0xDFFF and hi >= 0xD800:                                 # surrogates are no characters
            stack.append((lo, 0xD7FF))
            stack.append((0xE000, hi))
            continue
        split = False
        for m in (0x7F, 0x7FF, 0xFFFF):                                   # one encoded length per range
            if lo <= m < hi:
                stack.append((lo, m))
                stack.append((m + 1, hi))
                split = True
                break
        if split:
            continue
        if hi <= 0x7F:
            out.append([(lo, hi)])
            continue
        n = len(_encode(lo))
        for i in range(1, n):                                             # whole continuation bytes per sequence
            m = (1 << (6 * i)) - 1
            if lo & ~m != hi & ~m:
                if lo & m != 0:
                    stack.append((lo, lo | m))
                    stack.append(((lo | m) + 1, hi))
                    split = True
                    break
                if hi & m != m:
                    stack.append((lo, (hi & ~m) - 1))
                    stack.append((hi & ~m, hi))
                    split = True
                    break
        if not split:
            out.append(list(zip(_encode(lo), _encode(hi))))
    return out


def complement_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Every code point outside ascending, disjoint ``ranges``."""
    out, lo = [], 0
    for a, b in ranges:
        if a > lo:
            out.append((lo, a - 1))
        lo = b + 1
    if lo <= MAX_CODE_POINT:
        out.append((lo, MAX_CODE_POINT))
    return out
