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

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from .byte_level import ByteAutomaton, _coreachable
from .dfa import _column_classes, bfs_order, hopcroft, product_bfs


# -------------------------------------------------------------------------------------------------
# Canonical minimal form
# -------------------------------------------------------------------------------------------------

def _joint(parts: Sequence[ByteAutomaton]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``(byte_class, tables)``: the bytes none of ``parts`` tells apart -- the coarsest common refinement of their
    classes, numbered by their smallest byte -- and every part's table over them (int64)."""
    if len(parts) == 1:
        return parts[0].byte_class, [parts[0].table.astype(np.int64)]
    byte_class, first = _column_classes(np.ascontiguousarray(np.stack([a.byte_class for a in parts])))
    return byte_class, [a.table[:, a.byte_class[first]].astype(np.int64) for a in parts]


def _from_classes(table: np.ndarray, initial: int, accept: np.ndarray, byte_class: np.ndarray) -> ByteAutomaton:
    """The canonical minimal automaton of a complete table over byte classes (int64 [K, classes], the classes numbered
    by their smallest byte): reachable states, Hopcroft's partition, breadth-first numbering (the same over any classes
    numbered so: bytes in order), the dead state last, and the classes the minimal table tells apart."""
    reach = bfs_order(table, initial)
    remap = np.full(table.shape[0], -1, dtype = np.int64)
    remap[reach] = np.arange(reach.size)
    d = remap[table[reach]]
    acc = accept[reach]
    block, num_blocks = hopcroft(d, acc)
    rep = np.empty(num_blocks, dtype = np.int64)
    rep[block] = np.arange(reach.size)                          # any member represents its block
    d, acc, start = block[d[rep]], acc[rep], int(block[0])
    order = bfs_order(d, start)
    remap = np.empty(num_blocks, dtype = np.int64)
    remap[order] = np.arange(num_blocks)
    d, acc = remap[d[order]], acc[order]
    live = _coreachable(d, np.ascontiguousarray(acc))           # (the one dead state last)
    keep = np.nonzero(live)[0]
    dead = keep.size
    remap = np.full(num_blocks, dead, dtype = np.int64)
    remap[keep] = np.arange(keep.size)
    out = np.full((keep.size + 1, d.shape[1]), dead, dtype = np.int64)
    out[:keep.size] = remap[d[keep]]
    accepting = np.zeros(keep.size + 1, dtype = bool)
    accepting[:keep.size] = acc[keep]
    column_class, first = _column_classes(out)                  # (the classes the minimal automaton tells apart)
    return ByteAutomaton(None, int(remap[0]), accepting, dead, table = out[:, first],
                         byte_class = column_class[byte_class], minimal = True)


def force(a) -> ByteAutomaton:
    """The automaton of ``a``: itself, or an expression's (evaluated once, see :class:`Expr`)."""
    return a.force() if isinstance(a, Expr) else a


def minimize(a) -> ByteAutomaton:
    """The minimal automaton accepting ``a``'s texts, in canonical form (see the module docstring)."""
    a = force(a)
    if a.minimal:                                               # (made by this module: canonical already)
        return a
    return _from_classes(a.table.astype(np.int64), a.initial, a.accept, a.byte_class)


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
    return minimize(ByteAutomaton.from_char_automaton([0, 1], 0, [1], {0: {"c": 1}}, {"c": list(ranges)}))


def regex(pattern: str) -> ByteAutomaton:
    """The texts matching ``pattern`` in full, as :class:`~pyjuice.constraints.Regex` reads it (the same checks)."""
    from ..regex import _checked, _pattern_bytes
    _, parsed = _checked(pattern)
    return minimize(_pattern_bytes(pattern, parsed))


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
def _subsets(rows, eps_ptr, eps_dst, accept, start):
    """
    The subset construction of an automaton with empty moves: ``rows`` int64 [N, classes] every state's successor
    per class (-1: none, at most one), its empty moves as CSR (``eps_ptr`` [N + 1], ``eps_dst``), ``accept`` bool [N],
    the initial state ``start``. Returns ``(table, accepting)``: int64 [M, classes] over the reachable sets of states
    (set 0 the empty one, set 1 the initial one) and bool [M].
    """
    N, C = rows.shape
    stamp = np.full(N, -1, dtype = np.int64)
    buf = np.empty(N, dtype = np.int64)
    cap = 1024
    pool = np.empty(max(4 * cap, 2 * N), dtype = np.int64)      # the sets' sorted states back to back
    ptr = np.zeros(cap + 1, dtype = np.int64)
    chain = np.full(cap, -1, dtype = np.int64)                  # the next set of the same hash
    table = np.zeros((cap, C), dtype = np.int64)
    acc = np.zeros(cap, dtype = np.bool_)
    first = dict()                                              # hash -> its first set
    num, mark = 2, 0                                            # set 0: empty (ptr[1] == ptr[0])
    buf[0] = start
    stamp[start] = mark
    m = _closure(buf, 1, eps_ptr, eps_dst, stamp, mark)
    s = np.sort(buf[:m])
    pool[:m] = s
    ptr[2] = m
    h = np.int64(1469598103934665603)
    for x in s:
        h = (h ^ x) * np.int64(1099511628211)
        acc[1] = acc[1] or accept[x]
    first[h] = 1
    head = 1
    while head < num:
        a0, a1 = ptr[head], ptr[head + 1]
        for c in range(C):
            mark += 1
            k = 0
            for i in range(a0, a1):
                t = rows[pool[i], c]
                if t >= 0 and stamp[t] != mark:
                    stamp[t] = mark
                    buf[k] = t
                    k += 1
            if k == 0:
                table[head, c] = 0
                continue
            k = _closure(buf, k, eps_ptr, eps_dst, stamp, mark)
            s = np.sort(buf[:k])
            h = np.int64(1469598103934665603)
            for x in s:
                h = (h ^ x) * np.int64(1099511628211)
            j = first[h] if h in first else -1
            last = -1
            while j >= 0:                                       # the sets of this hash: the same one?
                if ptr[j + 1] - ptr[j] == k:
                    same = True
                    for i in range(k):
                        if pool[ptr[j] + i] != s[i]:
                            same = False
                            break
                    if same:
                        break
                last = j
                j = chain[j]
            if j < 0:                                           # a new set
                if num == cap:
                    cap *= 2
                    bigger = np.zeros((cap, C), dtype = np.int64)
                    bigger[:num] = table[:num]
                    table = bigger
                    more = np.zeros(cap, dtype = np.bool_)
                    more[:num] = acc[:num]
                    acc = more
                    ch = np.full(cap, -1, dtype = np.int64)
                    ch[:num] = chain[:num]
                    chain = ch
                    p = np.zeros(cap + 1, dtype = np.int64)
                    p[:num + 1] = ptr[:num + 1]
                    ptr = p
                end = ptr[num]
                if end + k > pool.size:
                    more_pool = np.empty(2 * (end + k), dtype = np.int64)
                    more_pool[:end] = pool[:end]
                    pool = more_pool
                pool[end:end + k] = s
                ptr[num + 1] = end + k
                for x in s:
                    if accept[x]:
                        acc[num] = True
                        break
                j = num
                if last >= 0:
                    chain[last] = j
                else:
                    first[h] = j
                num += 1
            table[head, c] = j
        head += 1
    return table[:num], acc[:num]


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

    def build(self, start: int) -> ByteAutomaton:
        total = self.size + self.extra
        num = lambda q: q if q >= 0 else self.size - 1 - q    # placeholders after the parts
        byte_class, tables = _joint(self.parts)
        sizes = np.array([a.num_states for a in self.parts], dtype = np.int64)
        offs = np.repeat(np.array(self.offsets, dtype = np.int64), sizes)[:, None]
        deads = np.repeat(np.array([a.dead for a in self.parts], dtype = np.int64), sizes)[:, None]
        d = np.concatenate(tables)
        rows = np.full((total, d.shape[1]), -1, dtype = np.int64)
        rows[:self.size] = np.where(d == deads, -1, d + offs)       # (a part's dead state: no transition)
        accept = np.zeros(total, dtype = bool)
        for q in self.accepting:
            accept[num(q)] = True
        src = np.array([num(u) for u, _ in self.eps], dtype = np.int64)
        dst = np.array([num(v) for _, v in self.eps], dtype = np.int64)
        order = np.argsort(src, kind = "stable")
        eps_ptr = np.zeros(total + 1, dtype = np.int64)
        np.add.at(eps_ptr, src + 1, 1)
        eps_ptr = np.cumsum(eps_ptr)
        table, acc = _subsets(rows, eps_ptr, dst[order], accept, num(start))
        return _from_classes(table, 1, acc, byte_class)


class Expr:
    """
    A concatenation (``kind`` "seq"), union ("alt") or repetition ("rep", ``lo`` to ``hi`` times; ``hi`` None: no
    bound) of automata or other expressions, not carried out yet: evaluated on first use -- read as an automaton (its
    attributes are its automaton's) or forced (:func:`force`) -- all at once, the whole expression one automaton with
    empty moves, determinized and minimized once. What is reused is best forced once, else every use wires it again.
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
    byte_class, tables = _joint(parts)
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
    return _from_classes(rows, 0, accept, byte_class)


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
