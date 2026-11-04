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
from .dfa import _column_classes, minimal_moves, moves_of_table, product_bfs


# -------------------------------------------------------------------------------------------------
# Canonical minimal form
# -------------------------------------------------------------------------------------------------

def _joint(parts: Sequence[ByteAutomaton]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``(byte_class, columns)``: the bytes none of ``parts`` tells apart -- the coarsest common refinement of their
    classes, numbered by their smallest byte -- and the class every part has for each of them."""
    if len(parts) == 1:
        return parts[0].byte_class, [np.arange(parts[0].table.shape[1], dtype = np.int64)]
    classes = np.ascontiguousarray(np.stack([a.byte_class for a in parts]))
    byte_class, first = _column_classes(classes)
    return byte_class, list(classes[:, first])


def _from_moves(ptr: np.ndarray, cls: np.ndarray, dst: np.ndarray, initial: int, accept: np.ndarray, C: int,
                byte_class: np.ndarray) -> ByteAutomaton:
    """The canonical minimal automaton of moves as CSR per state over byte classes (:func:`~.dfa.minimal_moves`; the
    classes numbered by their smallest byte: its breadth-first numbering, the same over any classes numbered so, reads
    bytes in order), and the classes it tells apart."""
    out, accepting = minimal_moves(ptr, cls, dst, accept, initial, C)
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


def literal(data: bytes) -> ByteAutomaton:
    """The text ``data`` alone (any bytes)."""
    n = len(data)
    delta = np.full((n + 2, 256), n + 1, dtype = np.int32)
    for i, b in enumerate(data):
        delta[i, b] = i + 1
    accept = np.zeros(n + 2, dtype = bool)
    accept[n] = True
    return ByteAutomaton(delta, 0, accept, n + 1)


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
def _subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps_dst, accept, start, C):
    """
    The subset construction of an automaton with empty moves: its moves as CSR per state (``edge_ptr`` [N + 1], the class
    ``edge_class`` of every move and its successor ``edge_dst``), its empty moves too (``eps_ptr`` [N + 1], ``eps_dst``),
    ``accept`` bool [N], the initial state ``start``, over ``C`` classes. Returns ``(ptr, cls, dst, accepting)``: over the
    reachable sets of states (set 0 the empty one, without moves, set 1 the initial one), their moves to sets but the
    empty one as CSR, classes ascending, and bool [M]. Classes in a row whose moves are the same lead to the same set.
    """
    N = edge_ptr.shape[0] - 1
    stamp = np.full(N, -1, dtype = np.int64)
    buf = np.empty(N, dtype = np.int32)
    cap = 1024
    pool = np.empty(max(4 * cap, 2 * N), dtype = np.int32)      # the sets' sorted states back to back
    ptr = np.zeros(cap + 1, dtype = np.int64)
    hashes = np.empty(cap, dtype = np.int64)
    move_ptr = np.zeros(cap + 1, dtype = np.int64)
    move_class = np.empty(8 * cap, dtype = np.int32)
    move_dst = np.empty(8 * cap, dtype = np.int32)
    moves = 0
    acc = np.zeros(cap, dtype = np.bool_)
    slots_size = 4 * cap
    slots = np.full(slots_size, -1, dtype = np.int32)           # open addressing: the set of a hash
    count = np.zeros(C, dtype = np.int32)
    where = np.empty(C, dtype = np.int32)
    moved = np.empty(1024, dtype = np.int32)                    # a set's successors per class
    num, mark = 2, 0                                            # set 0: empty (ptr[1] == ptr[0])
    buf[0] = start
    stamp[start] = mark
    m = _closure(buf, 1, eps_ptr, eps_dst, stamp, mark)
    _sorted_states(buf, m)
    pool[:m] = buf[:m]
    ptr[2] = m
    h = np.int64(1469598103934665603)
    for i in range(m):
        x = buf[i]
        h = (h ^ x) * np.int64(1099511628211)
        acc[1] = acc[1] or accept[x]
    hashes[1] = h
    slots[h & (slots_size - 1)] = 1
    head = 1
    while head < num:
        a0, a1 = ptr[head], ptr[head + 1]
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
        last_at, last_n, last_j = 0, 0, 0
        for c in range(C):
            n = count[c]
            if n == 0:
                continue
            count[c] = 0
            at = where[c] - n
            same = n == last_n
            if same:
                for i in range(n):
                    if moved[at + i] != moved[last_at + i]:
                        same = False
                        break
            if moves == move_class.shape[0]:
                more_class = np.empty(2 * moves, dtype = np.int32)
                more_class[:moves] = move_class
                move_class = more_class
                more_dst = np.empty(2 * moves, dtype = np.int32)
                more_dst[:moves] = move_dst
                move_dst = more_dst
            move_class[moves] = c
            if same:
                move_dst[moves] = last_j
                moves += 1
                continue
            mark += 1
            k = 0
            for i in range(at, at + n):
                t = moved[i]
                if stamp[t] != mark:
                    stamp[t] = mark
                    buf[k] = t
                    k += 1
            k = _closure(buf, k, eps_ptr, eps_dst, stamp, mark)
            _sorted_states(buf, k)
            h = np.int64(1469598103934665603)
            for i in range(k):
                h = (h ^ buf[i]) * np.int64(1099511628211)
            slot = h & (slots_size - 1)
            while True:
                j = slots[slot]
                if j < 0:
                    break
                if hashes[j] == h and ptr[j + 1] - ptr[j] == k:
                    found = True
                    for i in range(k):
                        if pool[ptr[j] + i] != buf[i]:
                            found = False
                            break
                    if found:
                        break
                slot = (slot + 1) & (slots_size - 1)
            if j < 0:                                           # a new set
                if num == cap:
                    cap *= 2
                    more_ptr = np.zeros(cap + 1, dtype = np.int64)
                    more_ptr[:num + 1] = move_ptr[:num + 1]
                    move_ptr = more_ptr
                    more = np.zeros(cap, dtype = np.bool_)
                    more[:num] = acc[:num]
                    acc = more
                    more_hashes = np.empty(cap, dtype = np.int64)
                    more_hashes[:num] = hashes[:num]
                    hashes = more_hashes
                    p = np.zeros(cap + 1, dtype = np.int64)
                    p[:num + 1] = ptr[:num + 1]
                    ptr = p
                    slots_size = 4 * cap
                    slots = np.full(slots_size, -1, dtype = np.int32)
                    for q in range(1, num):
                        s = hashes[q] & (slots_size - 1)
                        while slots[s] >= 0:
                            s = (s + 1) & (slots_size - 1)
                        slots[s] = q
                    slot = h & (slots_size - 1)
                    while slots[slot] >= 0:
                        slot = (slot + 1) & (slots_size - 1)
                end = ptr[num]
                if end + k > pool.shape[0]:
                    more_pool = np.empty(2 * (end + k), dtype = np.int32)
                    more_pool[:end] = pool[:end]
                    pool = more_pool
                pool[end:end + k] = buf[:k]
                ptr[num + 1] = end + k
                for i in range(k):
                    if accept[buf[i]]:
                        acc[num] = True
                        break
                j = num
                hashes[j] = h
                slots[slot] = j
                num += 1
            move_dst[moves] = j
            moves += 1
            last_at, last_n, last_j = at, n, j
        head += 1
        move_ptr[head] = moves
    return move_ptr[:num + 1], move_class[:moves], move_dst[:moves], acc[:num]


@njit(cache = True)
def _count_moves(table, dead, cols):
    """int64 [states]: the moves of every state (its dead state's none, nor those into it), over the joint classes
    ``cols`` reads."""
    K = table.shape[0]
    C = cols.shape[0]
    out = np.zeros(K, dtype = np.int64)
    for q in range(K):
        n = 0
        for j in range(C):
            if table[q, cols[j]] != dead:
                n += 1
        out[q] = n
    return out


@njit(cache = True)
def _fill_moves(table, dead, cols, first, edge_ptr, edge_class, edge_dst):
    """:func:`_count_moves`' moves in their places (``first``: the number of the table's first state)."""
    K = table.shape[0]
    C = cols.shape[0]
    for q in range(K):
        at = edge_ptr[first + q]
        for j in range(C):
            t = table[q, cols[j]]
            if t != dead:
                edge_class[at] = j
                edge_dst[at] = t + first
                at += 1


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
    ptr, cls, moved, acc = _subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps, accept, start, C)
    return _from_moves(ptr, cls, moved, 1, acc, C, byte_class)


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
        accepting = [start] if lo == 0 else []
        prev = [start]                                          # the states a next copy is entered from
        for i in range(copies):
            e, acc = self.wire(part)
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
        num = lambda q: q if q >= 0 else self.size - 1 - q    # placeholders after the parts
        byte_class, cols = _joint(self.parts)
        edge_ptr = np.zeros(total + 1, dtype = np.int64)
        edge_ptr[1:self.size + 1] = np.concatenate([_count_moves(a.table, a.dead, c) for a, c in zip(self.parts, cols)])
        np.cumsum(edge_ptr, out = edge_ptr)
        edge_class = np.empty(int(edge_ptr[-1]), dtype = np.int32)
        edge_dst = np.empty(int(edge_ptr[-1]), dtype = np.int32)
        for a, c, first in zip(self.parts, cols, self.offsets):
            _fill_moves(a.table, a.dead, c, first, edge_ptr, edge_class, edge_dst)
        accept = np.zeros(total, dtype = np.bool_)
        accept[[num(q) for q in self.accepting]] = True
        src = np.array([num(u) for u, _ in self.eps], dtype = np.int64)
        dst = np.array([num(v) for _, v in self.eps], dtype = np.int32)
        eps_ptr, eps = _csr(src, dst, total)
        C = int(byte_class.max()) + 1
        ptr, cls, moved, acc = _subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps, accept, num(start), C)
        return _from_moves(ptr, cls, moved, 1, acc, C, byte_class)


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
