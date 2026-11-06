"""
A regular expression, as Python's own parser reads it, to a byte automaton: one automaton with empty moves over the
UTF-8 bytes of its characters, built in one pass over the parse (:class:`_Nfa`, a repetition's copies at once), then
determinized and minimized once (:func:`.automata.determinized`). The characters a piece of the pattern matches are
those ``re.fullmatch(pattern, text, re.ASCII)`` does: ASCII categories, ASCII case folding under ``(?i)``, a dot without
``(?s)`` anything but a newline; surrogates are no characters. What it reads is what
:func:`~pyjuice.constraints.language.regex._unsupported` lets through.
"""

from __future__ import annotations

import array
import re
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

try:
    from re import _constants as C, _parser as sre_parse               # Python >= 3.11
except ImportError:                                                     # pragma: no cover
    import sre_constants as C
    import sre_parse

from .byte_level import ByteAutomaton, MAX_CODE_POINT, complement_ranges, utf8_sequences

Ranges = Tuple[Tuple[int, int], ...]

_DIGIT, _SPACE = ((0x30, 0x39),), ((0x09, 0x0D), (0x20, 0x20))
_WORD = ((0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A))
_CATEGORIES = {C.CATEGORY_DIGIT: _DIGIT, C.CATEGORY_NOT_DIGIT: tuple(complement_ranges(_DIGIT)),
               C.CATEGORY_SPACE: _SPACE, C.CATEGORY_NOT_SPACE: tuple(complement_ranges(_SPACE)),
               C.CATEGORY_WORD: _WORD, C.CATEGORY_NOT_WORD: tuple(complement_ranges(_WORD))}
_NOT_NEWLINE = ((0, 0x09), (0x0B, MAX_CODE_POINT))
_EVERYTHING = ((0, MAX_CODE_POINT),)


def _view(a: array.array, start: int = 0) -> np.ndarray:
    """int32 ``a[start:]`` (no copy)."""
    return np.frombuffer(a, dtype = np.int32, offset = 4 * start) if len(a) > start else np.empty(0, dtype = np.int32)


def parse(pattern: str):
    """Python's parse of ``pattern``, as ``re.compile(pattern, re.ASCII)`` reads it (raises what it does)."""
    return sre_parse.parse(pattern, re.ASCII)


def _merged(ranges: Sequence[Tuple[int, int]]) -> Ranges:
    """Inclusive ranges as maximal ones, ascending."""
    out: List[List[int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return tuple((a, b) for a, b in out)


def _folded(ranges: Ranges) -> Ranges:
    """``ranges`` and the other case of every ASCII letter among them."""
    more = list(ranges)
    for a, b in ranges:
        for lo, hi, shift in ((0x41, 0x5A, 0x20), (0x61, 0x7A, -0x20)):
            x, y = max(a, lo), min(b, hi)
            if x <= y:
                more.append((x + shift, y + shift))
    return _merged(more)


@lru_cache(maxsize = 4096)
def _sequences(ranges: Ranges):
    """Every UTF-8 encoding of the characters ``ranges`` as sequences of inclusive byte ranges: ``(one, more)`` -- those
    of one byte, those of several."""
    one, more = [], []
    for lo, hi in ranges:
        for seq in utf8_sequences(lo, hi):
            (one if len(seq) == 1 else more).append(tuple(seq))
    return tuple(s[0] for s in one), tuple(more)


def _characters(op, av, ignorecase: bool, dotall: bool) -> Ranges:
    """The characters one item of a parse matches."""
    if op is C.ANY:
        return _EVERYTHING if dotall else _NOT_NEWLINE
    if op is C.LITERAL or op is C.NOT_LITERAL:
        own = ((av, av),)
        own = _folded(own) if ignorecase else own
        return own if op is C.LITERAL else tuple(complement_ranges(own))
    parts, negated = [], False                                         # op is C.IN
    for item, value in av:
        if item is C.NEGATE:
            negated = True
        elif item is C.LITERAL:
            parts.append((value, value))
        elif item is C.RANGE:
            parts.append(value)
        else:
            parts.extend(_CATEGORIES[value])
    own = _merged(parts)
    own = _folded(own) if ignorecase else own
    return tuple(complement_ranges(own)) if negated else own


class _Nfa:
    """
    The automaton with empty moves, as it is written: moves on inclusive byte ranges and empty moves, as flat arrays. A
    piece is written from a state, and :meth:`items` returns the state it ends in; every path from one to the other reads
    one of the piece's texts, and nothing enters the state it starts from, so the same state may start pieces in turn and
    a piece's copy can be written by shifting its arrays (:meth:`repeat`).
    """

    def __init__(self):
        self.num = 1
        self.src, self.lo, self.hi, self.dst = array.array("i"), array.array("i"), array.array("i"), array.array("i")
        self.eps_src, self.eps_dst = array.array("i"), array.array("i")

    def fresh(self) -> int:
        self.num += 1
        return self.num - 1

    def move(self, q: int, lo: int, hi: int, r: int):
        self.src.append(q)
        self.lo.append(lo)
        self.hi.append(hi)
        self.dst.append(r)

    def empty(self, q: int, r: int):
        self.eps_src.append(q)
        self.eps_dst.append(r)

    def characters(self, q: int, ranges: Ranges) -> int:
        """One character of ``ranges`` from ``q``."""
        one, more = _sequences(ranges)
        r = self.fresh()
        for lo, hi in one:
            self.move(q, lo, hi, r)
        for seq in more:
            u = q
            for lo, hi in seq[:-1]:
                v = self.fresh()
                self.move(u, lo, hi, v)
                u = v
            self.move(u, seq[-1][0], seq[-1][1], r)
        return r

    def items(self, items, q: int, ignorecase: bool, dotall: bool) -> int:
        """A parse's items in turn, from ``q``."""
        for op, av in items:
            if op is C.LITERAL and av < 0x80 and not ignorecase:
                r = self.fresh()
                self.move(q, av, av, r)
                q = r
            elif op is C.LITERAL or op is C.NOT_LITERAL or op is C.IN or op is C.ANY:
                q = self.characters(q, _characters(op, av, ignorecase, dotall))
            elif op is C.SUBPATTERN:
                add = av[1]
                q = self.items(av[3], q, ignorecase or bool(add & re.IGNORECASE), dotall or bool(add & re.DOTALL))
            elif op is C.BRANCH:
                exits = [self.items(b, q, ignorecase, dotall) for b in av[1]]
                r = self.fresh()
                for e in exits:
                    self.empty(e, r)
                q = r
            else:                                                       # MAX_REPEAT, MIN_REPEAT
                q = self.repeat(av[2], av[0], av[1], q, ignorecase, dotall)
        return q

    def repeat(self, body, lo: int, hi: int, q: int, ignorecase: bool, dotall: bool) -> int:
        """``body`` ``lo`` to ``hi`` times from ``q`` (a ``hi`` of ``MAXREPEAT``: no bound)."""
        if hi == C.MAXREPEAT:
            if lo > 1:
                q = self.repeat(body, lo - 1, lo - 1, q, ignorecase, dotall)
            head = self.fresh()
            self.empty(q, head)
            e = self.items(body, head, ignorecase, dotall)
            if e == head:                                               # (the body writes nothing)
                return head
            self.empty(e, head)
            return e if lo >= 1 else head
        if hi == 0:
            return q
        n0, m0, e0 = self.num, len(self.src), len(self.eps_src)
        e = self.items(body, q, ignorecase, dotall)
        states = self.num - n0
        if states == 0:                                                 # (the body writes nothing)
            return q
        exits = [e]
        if hi > 1:
            exits = self._copies(q, e, n0, m0, e0, states, hi - 1)
        if hi == lo:
            return exits[-1]
        r = self.fresh()
        for x in exits[max(lo - 1, 0):]:
            self.empty(x, r)
        if lo == 0:
            self.empty(q, r)
        return r

    def _copies(self, q: int, e: int, n0: int, m0: int, e0: int, states: int, k: int) -> List[int]:
        """``k`` more copies of what was just written from ``q`` (its states from ``n0``, its moves from ``m0`` and
        ``e0``) ending in ``e``, each from where the one before it ends: every copy's last state, the first's first."""
        shift = (np.arange(1, k + 1, dtype = np.int64) * states)[:, None]

        def copied(arr, start, sources):
            part = _view(arr, start).astype(np.int64)
            out = part[None, :] + shift
            if sources:                                                 # (a copy starts where the one before it ended)
                at = part == q
                out[:, at] = e - states + shift
            return out.ravel().astype(np.int32).tobytes()

        src, dst = copied(self.src, m0, True), copied(self.dst, m0, False)
        lo, hi = np.tile(_view(self.lo, m0), k).tobytes(), np.tile(_view(self.hi, m0), k).tobytes()
        eps_src, eps_dst = copied(self.eps_src, e0, True), copied(self.eps_dst, e0, False)
        self.src.frombytes(src)
        self.dst.frombytes(dst)
        self.lo.frombytes(lo)
        self.hi.frombytes(hi)
        self.eps_src.frombytes(eps_src)
        self.eps_dst.frombytes(eps_dst)
        self.num += k * states
        return [e + i * states for i in range(k + 1)]


def byte_automaton(tree) -> ByteAutomaton:
    """The minimal byte automaton (canonical form, :mod:`.automata`) of the texts a parse (:func:`parse`) matches."""
    from .automata import determinized
    flags = tree.state.flags
    nfa = _Nfa()
    end = nfa.items(tree, 0, bool(flags & re.IGNORECASE), bool(flags & re.DOTALL))
    return determinized(nfa.num, 0, end, *[_view(a) for a in (nfa.src, nfa.lo, nfa.hi, nfa.dst, nfa.eps_src, nfa.eps_dst)])


def characters_automaton(ranges: Sequence[Tuple[int, int]]) -> ByteAutomaton:
    """The minimal byte automaton of one character among the inclusive code point ``ranges``."""
    from .automata import determinized
    nfa = _Nfa()
    end = nfa.characters(0, _merged(ranges))
    return determinized(nfa.num, 0, end, *[_view(a) for a in (nfa.src, nfa.lo, nfa.hi, nfa.dst, nfa.eps_src, nfa.eps_dst)])
