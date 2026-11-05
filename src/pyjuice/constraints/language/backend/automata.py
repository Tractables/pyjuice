"""
An algebra of byte automata: the sets of texts (byte strings) :class:`~.byte_level.ByteAutomaton` accepts, built from
literals, characters and regular expressions and combined by concatenation, union, repetition, intersection and
difference -- the operations a schema compiles through (:class:`~pyjuice.constraints.JSONSchema`).

Every operation returns the MINIMAL automaton of its result in canonical form (:func:`minimize`): live states numbered
in breadth-first order from the initial state, every state's successors visited by increasing byte, then the one dead
state. Two automata accept the same texts iff their canonical forms are equal (:func:`equivalent`).

Concatenation, union and repetition are carried out lazily (:class:`Expr`): an expression of them is evaluated on
first use, all at once -- its automata side by side as one automaton with empty moves (each keeps its byte
transitions: a state has at most one successor per byte; the empty moves join them), determinized once
(:func:`_subsets`, the subset construction) and minimized once. Intersection and difference walk their operands'
reachable state tuples (:func:`~.dfa.product_bfs`). Both work over the bytes no operand tells apart (byte classes).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from .byte_level import ByteAutomaton
from .counters import Wired, kept_copies, runs_of
from .dfa import _column_classes, minimal_moves, moves_of_table, product_bfs


# -------------------------------------------------------------------------------------------------
# Canonical minimal form
# -------------------------------------------------------------------------------------------------

def _joint(parts: Sequence[ByteAutomaton]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``(byte_class, columns)``: the bytes none of ``parts`` (one each, whatever their copies) tells apart -- the coarsest
    common refinement of their classes, numbered by their smallest byte -- and the class every part has for each of them."""
    if len(parts) == 1:
        return parts[0].byte_class, [np.arange(parts[0]._kept_table.shape[1], dtype = np.int64)]
    classes = np.ascontiguousarray(np.stack([a.byte_class for a in parts]))
    byte_class, first = _column_classes(classes)
    return byte_class, list(classes[:, first])


def _from_moves(ptr: np.ndarray, cls: np.ndarray, dst: np.ndarray, initial: int, accept: np.ndarray, C: int,
                byte_class: np.ndarray) -> ByteAutomaton:
    """The canonical minimal automaton of moves as CSR per state over byte classes (:func:`~.dfa.minimal_moves`; the
    classes numbered by their smallest byte: its breadth-first numbering, the same over any classes numbered so, reads
    bytes in order), and the classes it tells apart."""
    return _from_minimal(*minimal_moves(ptr, cls, dst, accept, initial, C)[:2], byte_class)


def _from_minimal(out: np.ndarray, accepting: np.ndarray, byte_class: np.ndarray) -> ByteAutomaton:
    column_class, first = _column_classes(out)
    return ByteAutomaton(None, 0, accepting, accepting.size - 1, table = out if first.size == out.shape[1] else out[:, first],
                         byte_class = column_class[byte_class], minimal = True)


def _from_classes(table: np.ndarray, initial: int, accept: np.ndarray, byte_class: np.ndarray) -> ByteAutomaton:
    """:func:`_from_moves` of a complete int32 table [K, classes]."""
    ptr, cls, dst = moves_of_table(table, accept)
    return _from_moves(ptr, cls, dst, initial, accept, table.shape[1], byte_class)


def force(a) -> ByteAutomaton:
    """The automaton of ``a``: itself, or an expression's (evaluated once, see :class:`Expr`)."""
    return a.force() if isinstance(a, Expr) else a


def minimize(a) -> ByteAutomaton:
    """The minimal automaton accepting ``a``'s texts, in canonical form (see the module docstring)."""
    a = force(a)
    if a.minimal:                                               # (made by this module: canonical already)
        return a
    return _from_classes(a.table, a.initial, a.accept, a.byte_class)


def equivalent(a: ByteAutomaton, b: ByteAutomaton) -> bool:
    """Whether two automata accept the same texts."""
    a, b = minimize(a), minimize(b)
    return a.initial == b.initial and np.array_equal(a.accept, b.accept) and np.array_equal(a.table, b.table) and \
        np.array_equal(a.byte_class, b.byte_class)


def is_empty(a) -> bool:
    """Whether ``a`` accepts no text."""
    return not bool(minimize(a).accept.any())


# -------------------------------------------------------------------------------------------------
# Leaves
# -------------------------------------------------------------------------------------------------

def nothing() -> ByteAutomaton:
    """No text."""
    return ByteAutomaton(None, 0, np.zeros(1, dtype = bool), 0, table = np.zeros((1, 1), dtype = np.int32),
                         byte_class = np.zeros(256, dtype = np.int64), minimal = True)


def epsilon() -> ByteAutomaton:
    """The empty text alone."""
    return literal(b"")


@lru_cache(maxsize = 4096)
def literal(data: bytes) -> ByteAutomaton:
    """The text ``data`` alone (any bytes); the texts read last are kept (an automaton is not written to)."""
    n = len(data)
    delta = np.full((n + 2, 256), n + 1, dtype = np.int32)
    for i, b in enumerate(data):
        delta[i, b] = i + 1
    accept = np.zeros(n + 2, dtype = bool)
    accept[n] = True
    byte_class, first = _column_classes(delta)                  # (kept: the classes alone, not a row of 256 per byte)
    return ByteAutomaton(None, 0, accept, n + 1, table = delta[:, first], byte_class = byte_class)


def byte_ranges(ranges: Sequence[Tuple[int, int]]) -> ByteAutomaton:
    """One byte in the inclusive ``ranges``."""
    delta = np.full((3, 256), 2, dtype = np.int32)
    for lo, hi in ranges:
        delta[0, lo:hi + 1] = 1
    return minimize(ByteAutomaton(delta, 0, np.array([False, True, False]), 2))


def char_ranges(ranges: Sequence[Tuple[int, int]]) -> ByteAutomaton:
    """One character -- its UTF-8 encoding -- among the inclusive code point ``ranges`` (surrogates are none)."""
    from .patterns import characters_automaton
    return characters_automaton(ranges)


@lru_cache(maxsize = 1024)
def regex(pattern: str) -> ByteAutomaton:
    """The texts matching ``pattern`` in full, as :class:`~pyjuice.constraints.Regex` reads it (the same checks); the
    patterns read last are kept."""
    from ..regex import _checked
    from .patterns import byte_automaton
    return byte_automaton(_checked(pattern))


# -------------------------------------------------------------------------------------------------
# Concatenation, union, repetition: one automaton with empty moves, determinized
# -------------------------------------------------------------------------------------------------

@njit(cache = True)
def _closure(buf, num, eps_ptr, eps_dst, stamp, mark):
    """``buf[:num]`` (states stamped ``mark``) and every state empty moves reach from them, stamped too: the new
    count."""
    head = 0
    while head < num:
        u = buf[head]
        head += 1
        for k in range(eps_ptr[u], eps_ptr[u + 1]):
            v = eps_dst[k]
            if stamp[v] != mark:
                stamp[v] = mark
                buf[num] = v
                num += 1
    return num


@njit(cache = True)
def _sorted_states(buf, k):
    """``buf[:k]`` ascending, in place (the sets of a subset construction are small, as a rule)."""
    if k > 32:
        buf[:k] = np.sort(buf[:k])
        return
    for i in range(1, k):
        x = buf[i]
        j = i
        while j > 0 and buf[j - 1] > x:
            buf[j] = buf[j - 1]
            j -= 1
        buf[j] = x


@njit(cache = True)
def _kept_set(buf, k, i, at, pool, acc, accept, slot_set, slot_hash, one):
    """
    The number among the sets kept of the set of states ``buf[:k]``, closed under empty moves: the one found, or ``i``, a new
    one, then kept (the states of set ``j`` are ``pool[at[j]:at[j + 1]]``). A set of one state is found by ``one[state]``, a
    larger one by its hash, in the open addressing of ``slot_set`` and ``slot_hash``. The caller has made room.
    """
    if k == 1:
        j = one[buf[0]]
        if j >= 0:
            return j
        one[buf[0]] = i
    else:
        _sorted_states(buf, k)
        h = np.int64(1469598103934665603)
        for x in range(k):
            h = (h ^ buf[x]) * np.int64(1099511628211)
        mask = slot_set.shape[0] - 1
        slot = h & mask
        while True:
            j = slot_set[slot]
            if j < 0:
                break
            if slot_hash[slot] == h and at[j + 1] - at[j] == k:
                found = True
                for x in range(k):
                    if pool[at[j] + x] != buf[x]:
                        found = False
                        break
                if found:
                    return j
            slot = (slot + 1) & mask
        slot_set[slot] = i
        slot_hash[slot] = h
    end = at[i]
    for x in range(k):
        pool[end + x] = buf[x]
        if accept[buf[x]]:
            acc[i] = True
    at[i + 1] = end + k
    return i


@njit(cache = True)
def _grown_slots(slot_set, slot_hash):
    """The open addressing of :func:`_kept_set`, twice as large."""
    size = 2 * slot_set.shape[0]
    more_set = np.full(size, -1, dtype = np.int32)
    more_hash = np.empty(size, dtype = np.int64)
    for s in range(slot_set.shape[0]):
        j = slot_set[s]
        if j >= 0:
            at = slot_hash[s] & (size - 1)
            while more_set[at] >= 0:
                at = (at + 1) & (size - 1)
            more_set[at], more_hash[at] = j, slot_hash[s]
    return more_set, more_hash


@njit(cache = True)
def _longer(a, size):
    """A copy of ``a`` of ``size`` entries, those past its own not set."""
    out = np.empty(size, dtype = a.dtype)
    out[:a.shape[0]] = a
    return out


#: of a state in the subset construction: on its own a set not reached yet, one to be read, one read, or not a set on its own
_NOT_REACHED, _TO_READ, _READ, _IN_SETS = 0, 1, 2, 3


@njit(cache = True)
def _subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps_dst, accept, start, C):
    """
    The subset construction of an automaton with empty moves: its moves as CSR per state (``edge_ptr`` [N + 1], the class
    ``edge_class`` of every move and its successor ``edge_dst``), its empty moves too (``eps_ptr`` [N + 1], ``eps_dst``),
    ``accept`` bool [N], the initial state ``start``, over ``C`` classes. Returns ``(ptr, cls, dst, accepting, initial, state, pool, at,
    kept)``: the moves of the sets reachable from the initial one, to sets but the empty one, as CSR, classes ascending, and bool
    for every number (those of no reachable set: their rows lead nowhere that is read); then what the sets are: every state's
    standing (:data:`_READ`: a set of its own, reached), and the states ``pool[at[k]:at[k + 1]]`` of the ``kept`` others.

    Nearly all states of automata joined by empty moves are sets of their own: without empty moves, with one move on a
    class at the most, classes ascending, they move as they are. Such a state IS a set, numbered as the state, its row its
    moves, read where it stands, in the order of their numbers but for those found behind (rows in order, far fewer misses than
    breadth-first through the columns of an automaton's parts; a move to another of them is only a mark). The other sets are
    numbered past the states, as they are found: a set that one state and its empty moves make is remembered by that state
    (``one``), and is hashed only when first found. The numbers mean nothing: :func:`~.dfa.minimal_moves` makes them canonical.
    """
    N = edge_ptr.shape[0] - 1
    E = edge_ptr[N]
    state = np.zeros(N, dtype = np.uint8)
    for u in range(N):
        if eps_ptr[u + 1] > eps_ptr[u]:
            state[u] = _IN_SETS
        else:
            for e in range(edge_ptr[u] + 1, edge_ptr[u + 1]):
                if edge_class[e] <= edge_class[e - 1]:
                    state[u] = _IN_SETS
                    break
    cls = np.empty(E + 1024, dtype = np.int32)                  # the states' rows, then the other sets', as they are read
    dst = np.empty(E + 1024, dtype = np.int32)
    cls[:E] = edge_class
    dst[:E] = edge_dst
    moves = E
    ptr = np.empty(N + 1025, dtype = np.int64)
    ptr[:N + 1] = edge_ptr
    stamp = np.full(N, -1, dtype = np.int64)
    buf = np.empty(N, dtype = np.int32)
    one = np.full(N, -1, dtype = np.int32)                      # the kept set of a state and all empty moves reach from it
    pool = np.empty(max(1024, 2 * N), dtype = np.int32)         # the kept sets' sorted states back to back
    at = np.zeros(1025, dtype = np.int64)
    acc = np.zeros(1024, dtype = np.bool_)
    slot_set = np.full(4096, -1, dtype = np.int32)              # open addressing: the kept set of a hash
    slot_hash = np.empty(4096, dtype = np.int64)
    kept, hashed, head = 0, 0, 0
    count = np.zeros(C, dtype = np.int32)
    where = np.empty(C, dtype = np.int32)
    moved = np.empty(1024, dtype = np.int32)                    # a set's successors per class
    stack = np.empty(N + 1, dtype = np.int32)                   # the states found behind the sweep
    top, sweep, mark = 0, 0, 0
    if state[start] == _NOT_REACHED:
        state[start] = _TO_READ
        initial = start
    else:
        buf[0] = start
        stamp[start] = mark
        k = _closure(buf, 1, eps_ptr, eps_dst, stamp, mark)
        if k > pool.shape[0]:
            pool = np.empty(2 * k, dtype = np.int32)
        one[start] = _kept_set(buf, k, 0, at, pool, acc, accept, slot_set, slot_hash, one)
        kept, hashed = 1, 1
        initial = N
    while True:
        if head < kept:                                         # a kept set: its moves by class
            a0, a1 = at[head], at[head + 1]
            total = 0
            for i in range(a0, a1):
                u = pool[i]
                for e in range(edge_ptr[u], edge_ptr[u + 1]):
                    count[edge_class[e]] += 1
                total += edge_ptr[u + 1] - edge_ptr[u]
            if total > moved.shape[0]:
                moved = np.empty(2 * total, dtype = np.int32)
            place = 0
            for c in range(C):
                where[c] = place
                place += count[c]
            for i in range(a0, a1):
                u = pool[i]
                for e in range(edge_ptr[u], edge_ptr[u + 1]):
                    c = edge_class[e]
                    moved[where[c]] = edge_dst[e]
                    where[c] += 1
            if moves + total > cls.shape[0]:
                cls = _longer(cls, 2 * (moves + total))
                dst = _longer(dst, 2 * (moves + total))
            if kept + total + 1 > at.shape[0]:                  # (a new set a move, at the most)
                at = _longer(at, 2 * (kept + total + 1))
                more = np.zeros(at.shape[0], dtype = np.bool_)
                more[:kept] = acc[:kept]
                acc = more
            if N + kept + total + 1 > ptr.shape[0]:
                ptr = _longer(ptr, 2 * (N + kept + total + 1))
            while 4 * (hashed + total) > slot_set.shape[0]:
                slot_set, slot_hash = _grown_slots(slot_set, slot_hash)
            last_at, last_n, last_j = 0, 0, 0
            for c in range(C):
                n = count[c]
                if n == 0:
                    continue
                count[c] = 0
                b = where[c] - n
                cls[moves] = c
                j = -1
                t = moved[b]
                if n == 1:                                      # alone: its target says it
                    r = state[t]
                    if r != _IN_SETS:
                        if r == _NOT_REACHED:
                            state[t] = _TO_READ
                            if t < sweep:
                                stack[top] = t
                                top += 1
                        j = t
                    else:
                        i = one[t]
                        if i >= 0:
                            j = N + i
                if j < 0:
                    same = n == last_n
                    if same:
                        for i in range(n):
                            if moved[b + i] != moved[last_at + i]:
                                same = False
                                break
                    if same:
                        j = last_j
                    else:
                        mark += 1
                        k = 0
                        for i in range(b, b + n):
                            x = moved[i]
                            if stamp[x] != mark:
                                stamp[x] = mark
                                buf[k] = x
                                k += 1
                        k = _closure(buf, k, eps_ptr, eps_dst, stamp, mark)
                        if k == 1 and state[buf[0]] != _IN_SETS:
                            j = buf[0]
                            if state[j] == _NOT_REACHED:
                                state[j] = _TO_READ
                                if j < sweep:
                                    stack[top] = j
                                    top += 1
                        else:
                            if at[kept] + k > pool.shape[0]:
                                pool = _longer(pool, 2 * (at[kept] + k))
                            i = _kept_set(buf, k, kept, at, pool, acc, accept, slot_set, slot_hash, one)
                            if i == kept:
                                kept += 1
                                hashed += 1
                            if n == 1:
                                one[t] = i
                            j = N + i
                dst[moves] = j
                moves += 1
                last_at, last_n, last_j = b, n, j
            head += 1
            ptr[N + head] = moves
            continue
        if top > 0:                                             # a state behind the sweep
            top -= 1
            u = stack[top]
        else:
            while sweep < N and state[sweep] != _TO_READ:
                sweep += 1
            if sweep == N:
                break
            u = sweep
            sweep += 1
        state[u] = _READ
        for e in range(edge_ptr[u], edge_ptr[u + 1]):           # a state's row: as it is but for what is not on its own
            t = edge_dst[e]
            r = state[t]
            if r == _NOT_REACHED:
                state[t] = _TO_READ
                if t < sweep:
                    stack[top] = t
                    top += 1
            elif r == _IN_SETS:
                i = one[t]
                if i < 0:
                    if kept + 2 > at.shape[0]:
                        at = _longer(at, 2 * (kept + 2))
                        more = np.zeros(at.shape[0], dtype = np.bool_)
                        more[:kept] = acc[:kept]
                        acc = more
                    if N + kept + 2 > ptr.shape[0]:
                        ptr = _longer(ptr, 2 * (N + kept + 2))
                    if 4 * (hashed + 1) > slot_set.shape[0]:
                        slot_set, slot_hash = _grown_slots(slot_set, slot_hash)
                    mark += 1
                    buf[0] = t
                    stamp[t] = mark
                    k = _closure(buf, 1, eps_ptr, eps_dst, stamp, mark)
                    if at[kept] + k > pool.shape[0]:
                        pool = _longer(pool, 2 * (at[kept] + k))
                    i = _kept_set(buf, k, kept, at, pool, acc, accept, slot_set, slot_hash, one)
                    if i == kept:
                        kept += 1
                        hashed += 1
                    one[t] = i
                dst[e] = N + i
    accepting = np.empty(N + kept, dtype = np.bool_)
    accepting[:N] = accept
    accepting[N:] = acc[:kept]
    return ptr[:N + kept + 1], cls[:moves], dst[:moves], accepting, initial, state, pool, at, kept


@njit(cache = True)
def _moves_of_parts(tables, table_at, states, dead, cols, part, first, N):
    """
    The moves of automata side by side, as CSR over ``N`` states: ``(edge_ptr, edge_class, edge_dst)``. Distinct automata
    ``u``: a table of ``states[u]`` rows at ``tables[table_at[u]:table_at[u + 1]]``, its ``dead`` state without moves and no
    move into it, the class ``cols[u, j]`` it has for joint class ``j``. Its copies ``i``: ``part[i]`` the automaton, ``first[i]``
    the number of its first state. Every automaton's moves are read once.
    """
    U = states.shape[0]
    J = cols.shape[1]
    state_at = np.zeros(U + 1, dtype = np.int64)                # (one automaton's states after another's)
    for u in range(U):
        state_at[u + 1] = state_at[u] + states[u]
    ptr = np.zeros(state_at[U] + 1, dtype = np.int64)
    for u in range(U):
        at, K, C = table_at[u], states[u], (table_at[u + 1] - table_at[u]) // max(states[u], 1)
        for q in range(K):
            n = 0
            for j in range(J):
                if tables[at + q * C + cols[u, j]] != dead[u]:
                    n += 1
            ptr[state_at[u] + q + 1] = n
    for x in range(state_at[U]):
        ptr[x + 1] += ptr[x]
    own_class = np.empty(ptr[state_at[U]], dtype = np.int32)
    own_dst = np.empty(ptr[state_at[U]], dtype = np.int32)
    fill = 0
    for u in range(U):
        at, K, C = table_at[u], states[u], (table_at[u + 1] - table_at[u]) // max(states[u], 1)
        for q in range(K):
            for j in range(J):
                t = tables[at + q * C + cols[u, j]]
                if t != dead[u]:
                    own_class[fill], own_dst[fill] = j, t
                    fill += 1
    edge_ptr = np.zeros(N + 1, dtype = np.int64)
    total = 0
    for i in range(part.shape[0]):
        u = part[i]
        base = state_at[u]
        for q in range(states[u]):
            edge_ptr[first[i] + q + 1] = ptr[base + q + 1] - ptr[base + q]
            total += ptr[base + q + 1] - ptr[base + q]
    for x in range(N):
        edge_ptr[x + 1] += edge_ptr[x]
    edge_class = np.empty(total, dtype = np.int32)
    edge_dst = np.empty(total, dtype = np.int32)
    for i in range(part.shape[0]):
        u = part[i]
        base = state_at[u]
        at = edge_ptr[first[i]]
        for k in range(ptr[base], ptr[base + states[u]]):
            edge_class[at] = own_class[k]
            edge_dst[at] = own_dst[k] + first[i]
            at += 1
    return edge_ptr, edge_class, edge_dst


def _csr(src: np.ndarray, dst: np.ndarray, total: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(ptr, values)``: the pairs by ``src``, stably."""
    ptr = np.zeros(total + 1, dtype = np.int64)
    ptr[1:] = np.bincount(src, minlength = total)
    return np.cumsum(ptr), dst[np.argsort(src, kind = "stable")]


def determinized(num: int, start: int, end: int, src, lo, hi, dst, eps_src, eps_dst) -> ByteAutomaton:
    """The canonical minimal automaton of ``num`` states, moves on the inclusive byte ranges ``lo[i]`` to ``hi[i]`` from
    ``src[i]`` to ``dst[i]``, empty moves from ``eps_src[i]`` to ``eps_dst[i]``, from ``start`` to the accepting ``end``."""
    cuts = np.zeros(257, dtype = bool)
    cuts[0] = True
    cuts[lo] = True
    cuts[hi + 1] = True
    byte_class = np.cumsum(cuts[:256]) - 1
    edge_ptr, edge_class, edge_dst = _range_moves(num, src, byte_class[lo], byte_class[hi], dst)
    accept = np.zeros(num, dtype = np.bool_)
    accept[end] = True
    eps_ptr, eps = _csr(eps_src, eps_dst, num)
    C = int(byte_class[-1]) + 1
    ptr, cls, moved, acc, initial = _subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps, accept, start, C)[:5]
    return _from_moves(ptr, cls, moved, initial, acc, C, byte_class)


@njit(cache = True)
def _range_moves(num, src, first, last, dst):
    """The moves from ``src[i]`` to ``dst[i]`` on the classes ``first[i]`` to ``last[i]`` as CSR per state."""
    edge_ptr = np.zeros(num + 1, dtype = np.int64)
    for i in range(src.shape[0]):
        edge_ptr[src[i] + 1] += last[i] - first[i] + 1
    for q in range(num):
        edge_ptr[q + 1] += edge_ptr[q]
    edge_class = np.empty(edge_ptr[num], dtype = np.int32)
    edge_dst = np.empty(edge_ptr[num], dtype = np.int32)
    fill = edge_ptr[:num].copy()
    for i in range(src.shape[0]):
        at = fill[src[i]]
        for c in range(first[i], last[i] + 1):
            edge_class[at] = c
            edge_dst[at] = dst[i]
            at += 1
        fill[src[i]] = at
    return edge_ptr, edge_class, edge_dst


class _Joined:
    """Automata side by side, joined by empty moves, then determinized (:meth:`build`)."""

    def __init__(self):
        self.parts: List[ByteAutomaton] = []
        self.offsets: List[int] = []
        self.size = 0
        self.extra = 0                                          # fresh states, numbered after the parts'
        self.eps: List[Tuple[int, int]] = []
        self.accepting: List[int] = []
        self.wired: List[Wired] = []                            # (the repetitions wired short: see :mod:`.counters`)

    def add(self, a: ByteAutomaton) -> int:
        """A copy of ``a``: the number of its first state (its states follow in order)."""
        self.parts.append(a)
        self.offsets.append(self.size)
        self.size += a.num_states
        return self.size - a.num_states

    def fresh(self) -> int:
        """A fresh state without byte transitions (numbered once every part is in: a negative placeholder)."""
        self.extra += 1
        return -self.extra

    def wire(self, x) -> Tuple[int, List[int]]:
        """``x`` -- an automaton, or an expression not evaluated yet, its parts wired in turn -- added: ``(entry,
        accepting states)``. A repeated part is evaluated first (its copies then the minimal automaton's)."""
        if isinstance(x, Expr) and x._forced is not None:
            x = x._forced
        if not isinstance(x, Expr):
            off = self.add(x)
            return off + x.initial, [off + int(q) for q in np.nonzero(x.accept)[0]]
        if x.kind == "seq":
            entry, prev = self.wire(x.parts[0])
            for part in x.parts[1:]:                            # every accepting state enters the next part
                e, acc = self.wire(part)
                self.eps += [(q, e) for q in prev]
                prev = acc
            return entry, prev
        if x.kind == "list":                                    # one copy of the item, the separator back to it
            entry, accepting = self.wire(x.parts[0])
            e, acc = self.wire(x.parts[1])
            self.eps += [(q, e) for q in accepting] + [(q, entry) for q in acc]
            return entry, accepting
        if x.kind == "members":
            return self._wire_members(*x.parts)
        start = self.fresh()
        if x.kind == "alt":
            accepting = []
            for part in x.parts:
                e, acc = self.wire(part)
                self.eps.append((start, e))
                accepting += acc
            return start, accepting
        lo, hi = x.lo, x.hi                                     # "rep"
        copies = max(lo, 1) if hi is None else hi
        part = force(x.parts[0]) if copies > 1 or hi is None else x.parts[0]
        short = kept_copies(part, lo, copies) if copies > 1 else None
        if short is not None:
            lo, copies, wired = short
            self.wired.append(wired)
        accepting = [start] if lo == 0 else []
        prev = [start]                                          # the states a next copy is entered from
        for i in range(copies):
            e, acc = self.wire(part)
            if short is not None:
                wired.offsets.append(e - part.initial)
                wired.parts.append(len(self.parts) - 1)
            self.eps += [(p, e) for p in prev]
            prev = acc
            if i + 1 >= lo:
                accepting += acc
            if hi is None and i == copies - 1:                  # the last copy loops
                self.eps += [(p, e) for p in acc]
        return start, accepting

    def _wire_members(self, opening, closing, items, empty: bool) -> Tuple[int, List[int]]:
        """Two tracks of states between the members -- nothing written yet, or something (the next member then follows
        its ``later`` part, else its ``first``) -- through one copy of every value (see :func:`members`)."""
        k = len(items)
        none, some = [self.fresh() for _ in range(k + 1)], [self.fresh() for _ in range(k + 1)]

        def enter(x, sources, targets):
            entry, accepting = self.wire(x)
            self.eps.extend((u, entry) for u in sources)
            self.eps.extend((q, t) for q in accepting for t in targets)
            return entry, accepting

        start, _ = enter(opening, [], [none[0]])
        for i, (first, later, value, required) in enumerate(items):
            v, done = self.wire(value)
            enter(first, [none[i]], [v])
            enter(later, [some[i]], [v])
            self.eps.extend((q, some[i + 1]) for q in done)
            if not required:
                self.eps += [(none[i], none[i + 1]), (some[i], some[i + 1])]
        _, accepting = enter(closing, [some[k]] + ([none[k]] if empty else []), [])
        return start, accepting

    def build(self, start: int) -> ByteAutomaton:
        total = self.size + self.extra
        distinct = {}                                           # (an automaton's copies: read once)
        part = np.fromiter((distinct.setdefault(id(a), len(distinct)) for a in self.parts), dtype = np.int64, count = len(self.parts))
        own = list({id(a): a for a in self.parts}.values())
        byte_class, cols = _joint(own)
        tables = np.concatenate([a._kept_table.ravel() for a in own])
        table_at = np.zeros(len(own) + 1, dtype = np.int64)
        np.cumsum([a._kept_table.size for a in own], out = table_at[1:])
        edge_ptr, edge_class, edge_dst = _moves_of_parts(tables, table_at, np.asarray([a.num_states for a in own], dtype = np.int64),
                                                         np.asarray([a.dead for a in own], dtype = np.int64), np.stack(cols).astype(np.int64),
                                                         part, np.asarray(self.offsets, dtype = np.int64), total)
        accept = np.zeros(total, dtype = np.bool_)
        accepting = np.asarray(self.accepting, dtype = np.int64)
        accept[np.where(accepting >= 0, accepting, self.size - 1 - accepting)] = True
        pairs = np.asarray(self.eps, dtype = np.int64).reshape(-1, 2)
        pairs = np.where(pairs >= 0, pairs, self.size - 1 - pairs)             # (placeholders after the parts)
        eps_ptr, eps = _csr(pairs[:, 0], pairs[:, 1].astype(np.int32), total)
        C = int(byte_class.max()) + 1
        ptr, cls, moved, acc, initial, state, pool, at, kept = _subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps, accept,
                                                                        start if start >= 0 else self.size - 1 - start, C)
        if not self.wired and not any(a.counted for a in own):
            return _from_moves(ptr, cls, moved, initial, acc, C, byte_class)
        out, accepting_, block, place = minimal_moves(ptr, cls, moved, acc, initial, C)
        number = np.full(block.size, -1, dtype = np.int32)
        live = block >= 0
        number[live] = place[block[live]]
        a = _from_minimal(out, accepting_, byte_class)
        a.counted = runs_of(self.wired, self.parts, self.offsets, total, ptr, cls, moved, acc, state == _READ, pool, at, kept, number)
        return a


class Expr:
    """
    A concatenation (``kind`` "seq"), union ("alt"), repetition ("rep", ``lo`` to ``hi`` times; ``hi`` None: no bound),
    list ("list", :func:`list_of`) or object's members ("members", :func:`members`) of automata or other expressions, not
    carried out yet: evaluated on first use -- read as an automaton (its attributes are its automaton's) or forced
    (:func:`force`) -- all at once, the whole expression one automaton with empty moves, determinized and minimized once.
    What is reused is best forced once, else every use wires it again.
    """

    __slots__ = ("kind", "parts", "lo", "hi", "_forced")

    def __init__(self, kind: str, parts, lo: int = 0, hi: Optional[int] = None):
        self.kind, self.parts, self.lo, self.hi = kind, tuple(parts), lo, hi
        self._forced: Optional[ByteAutomaton] = None

    def force(self) -> ByteAutomaton:
        if self._forced is None:
            j = _Joined()
            entry, j.accepting = j.wire(self)
            self._forced = j.build(entry)
        return self._forced

    def __getattr__(self, name):
        return getattr(self.force(), name)


def concat(*parts):
    """The texts ``u_1 + ... + u_k``, ``u_i`` accepted by ``parts[i]`` (none: the empty text alone)."""
    if not parts:
        return epsilon()
    return parts[0] if len(parts) == 1 else Expr("seq", parts)


def union(*parts):
    """The texts some of ``parts`` accepts (none: no text)."""
    if not parts:
        return nothing()
    return parts[0] if len(parts) == 1 else Expr("alt", parts)


def optional(a):
    """``a``'s texts and the empty text."""
    return union(a, epsilon())


def star(a):
    """Concatenations of any number of ``a``'s texts (the empty text among them)."""
    return repeat(a, 0, None)


def list_of(item, separator):
    """``item`` or more, ``separator`` between them: ``item (separator item)*`` through one copy of ``item``."""
    return Expr("list", (item, separator))


def members(opening, closing, items, empty: bool):
    """``opening``, then the members ``items`` -- ``(first, later, value, required)``: a member's name read first, or after
    another member, and its value -- in order, the ones not required may be left out, then ``closing``; every one left
    out only if ``empty``. Through one copy of every value."""
    return Expr("members", (opening, closing, tuple(items), empty))


def repeat(a, lo: int, hi: Optional[int]):
    """Concatenations of ``lo`` to ``hi`` of ``a``'s texts (``hi`` None: no bound)."""
    if lo < 0 or (hi is not None and hi < lo):
        raise ValueError(f"repeat: bounds {lo}..{hi} are not 0 <= lo <= hi.")
    if hi == 0:
        return epsilon()
    return Expr("rep", [a], lo, hi)


# -------------------------------------------------------------------------------------------------
# Intersection and difference: products
# -------------------------------------------------------------------------------------------------

def _product(parts: Sequence[ByteAutomaton], mode: str) -> ByteAutomaton:
    byte_class, cols = _joint(parts)
    tables = [a.table[:, c].astype(np.int64) for a, c in zip(parts, cols)]
    flat = [t.reshape(-1) for t in tables]
    offsets = np.zeros(len(parts) + 1, dtype = np.int64)
    offsets[1:] = np.cumsum([f.size for f in flat])
    Ks = np.array([a.num_states for a in parts], dtype = np.int64)
    child = np.repeat(np.arange(tables[0].shape[1], dtype = np.int64)[:, None], len(parts), axis = 1)
    rows, keys = product_bfs(np.concatenate(flat), offsets, Ks, child,
                             np.array([a.initial for a in parts], dtype = np.int64))
    accept = np.ones(keys.size, dtype = bool)
    rest = keys.copy()
    for i in range(len(parts) - 1, -1, -1):
        a = parts[i].accept[rest % Ks[i]]
        accept &= ~a if (mode == "difference" and i > 0) else a
        rest //= Ks[i]
    return _from_classes(rows.astype(np.int32), 0, accept, byte_class)


def intersect(*parts) -> ByteAutomaton:
    """The texts every one of ``parts`` accepts (two at a time, each product minimal)."""
    if not parts:
        raise ValueError("intersect: no automaton (every text is no regular set of texts here).")
    out = minimize(parts[0])
    for a in parts[1:]:
        out = _product([out, minimize(a)], "intersection")
    return out


def difference(a, b) -> ByteAutomaton:
    """The texts ``a`` accepts and ``b`` does not."""
    return _product([minimize(a), minimize(b)], "difference")
