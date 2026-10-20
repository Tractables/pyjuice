"""
An algebra of byte automata: the sets of texts (byte strings) :class:`~.byte_level.ByteAutomaton` accepts, built from
literals, characters and regular expressions and combined by concatenation, union, repetition, intersection and
difference -- the operations a schema compiles through (:class:`~pyjuice.constraints.JSONSchema`).

Every operation returns the MINIMAL automaton of its result in canonical form (:func:`minimize`): live states numbered
in breadth-first order from the initial state, every state's successors visited by increasing byte, then the one dead
state. Two automata accept the same texts iff their canonical forms are equal (:func:`equivalent`).

Concatenation, union and repetition put their operands' tables side by side as one automaton with empty moves (each
operand's states keep their byte transitions: a state has at most one successor per byte; the empty moves join them)
and determinize it once (:func:`_subsets`, the subset construction); intersection and difference walk the operands'
reachable state tuples (:func:`~.dfa.product_bfs`). Both work over the bytes no operand tells apart (byte classes).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

from .byte_level import ByteAutomaton, _coreachable
from .dfa import bfs_order, hopcroft, product_bfs


# -------------------------------------------------------------------------------------------------
# Canonical minimal form
# -------------------------------------------------------------------------------------------------

@njit(cache = True)
def _column_classes(stacked):
    """The columns of ``stacked`` ([rows, 256]) grouped by equal content: ``(byte_class, first)`` as
    :func:`_byte_classes`. One pass hashing every column (rows in order), then each byte's column compared with the
    first column of its hash's classes."""
    R = stacked.shape[0]
    h = np.full(256, np.uint64(1469598103934665603))
    prime = np.uint64(1099511628211)
    for r in range(R):
        for b in range(256):
            h[b] = (h[b] ^ np.uint64(stacked[r, b] + 1)) * prime
    byte_class = np.empty(256, dtype = np.int64)
    first = np.empty(256, dtype = np.int64)
    num = 0
    for b in range(256):
        found = -1
        for c in range(num):
            rep = first[c]
            if h[rep] == h[b]:
                same = True
                for r in range(R):
                    if stacked[r, rep] != stacked[r, b]:
                        same = False
                        break
                if same:
                    found = c
                    break
        if found < 0:
            first[num] = b
            found = num
            num += 1
        byte_class[b] = found
    return byte_class, first[:num].copy()


def _byte_classes(tables: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """``(byte_class, first)``: the bytes no table's columns tell apart, as int64 [256] classes numbered by their
    smallest byte, and every class's smallest byte (int64 [classes], ascending)."""
    stacked = tables[0] if len(tables) == 1 else np.concatenate([t.reshape(t.shape[0], 256) for t in tables])
    return _column_classes(np.ascontiguousarray(stacked))


def _from_classes(table: np.ndarray, initial: int, accept: np.ndarray, byte_class: np.ndarray) -> ByteAutomaton:
    """The canonical minimal automaton of a complete table over byte classes (int64 [K, classes], the classes numbered
    by their smallest byte): reachable states, Hopcroft's partition, breadth-first numbering, the dead state last."""
    reach = bfs_order(table, initial)
    remap = np.full(table.shape[0], -1, dtype = np.int64)
    remap[reach] = np.arange(reach.size)
    d = remap[table[reach]]
    acc = accept[reach]
    block, num_blocks = hopcroft(d, acc)
    rep = np.empty(num_blocks, dtype = np.int64)
    rep[block] = np.arange(reach.size)                          # any member represents its block
    d, acc, start = block[d[rep]], acc[rep], int(block[0])
    order = bfs_order(d, start)                                 # (classes by smallest byte: bytes in order)
    remap = np.empty(num_blocks, dtype = np.int64)
    remap[order] = np.arange(num_blocks)
    d, acc = remap[d[order]], acc[order]
    live = _coreachable(d, np.ascontiguousarray(acc))           # (on the classes; the one dead state last)
    keep = np.nonzero(live)[0]
    dead = keep.size
    remap = np.full(num_blocks, dead, dtype = np.int64)
    remap[keep] = np.arange(keep.size)
    table = np.full((keep.size + 1, d.shape[1]), dead, dtype = np.int64)
    table[:keep.size] = remap[d[keep]]
    accepting = np.zeros(keep.size + 1, dtype = bool)
    accepting[:keep.size] = acc[keep]
    return ByteAutomaton(np.ascontiguousarray(table[:, byte_class], dtype = np.int32), int(remap[0]), accepting, dead)


def minimize(a: ByteAutomaton) -> ByteAutomaton:
    """The minimal automaton accepting ``a``'s texts, in canonical form (see the module docstring)."""
    byte_class, first = _byte_classes([a.delta])
    return _from_classes(a.delta[:, first].astype(np.int64), a.initial, a.accept, byte_class)


def equivalent(a: ByteAutomaton, b: ByteAutomaton) -> bool:
    """Whether two automata accept the same texts."""
    a, b = minimize(a), minimize(b)
    return a.initial == b.initial and np.array_equal(a.accept, b.accept) and np.array_equal(a.delta, b.delta)


def is_empty(a: ByteAutomaton) -> bool:
    """Whether ``a`` accepts no text."""
    return not bool(a.accept.any())


# -------------------------------------------------------------------------------------------------
# Leaves
# -------------------------------------------------------------------------------------------------

def nothing() -> ByteAutomaton:
    """No text."""
    return ByteAutomaton(np.zeros((1, 256), dtype = np.int32), 0, np.zeros(1, dtype = bool), 0)


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

    def build(self, start: int) -> ByteAutomaton:
        total = self.size + self.extra
        num = lambda q: q if q >= 0 else self.size - 1 - q    # placeholders after the parts
        stacked = np.ascontiguousarray(np.concatenate([a.delta for a in self.parts]))
        byte_class, first = _column_classes(stacked)
        sizes = np.array([a.num_states for a in self.parts], dtype = np.int64)
        offs = np.repeat(np.array(self.offsets, dtype = np.int64), sizes)[:, None]
        deads = np.repeat(np.array([a.dead for a in self.parts], dtype = np.int64), sizes)[:, None]
        rows = np.full((total, first.size), -1, dtype = np.int64)
        d = stacked[:, first].astype(np.int64)
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


def concat(*parts: ByteAutomaton) -> ByteAutomaton:
    """The texts ``u_1 + ... + u_k``, ``u_i`` accepted by ``parts[i]`` (none: the empty text alone)."""
    if not parts:
        return epsilon()
    j = _Joined()
    firsts = [j.add(a) for a in parts]
    for i in range(len(parts) - 1):                             # every accepting state enters the next part
        a, off, nxt = parts[i], firsts[i], firsts[i + 1] + parts[i + 1].initial
        j.eps += [(off + int(q), nxt) for q in np.nonzero(a.accept)[0]]
    j.accepting = [firsts[-1] + int(q) for q in np.nonzero(parts[-1].accept)[0]]
    return j.build(firsts[0] + parts[0].initial)


def union(*parts: ByteAutomaton) -> ByteAutomaton:
    """The texts some of ``parts`` accepts (none: no text)."""
    if not parts:
        return nothing()
    j = _Joined()
    start = j.fresh()
    for a in parts:
        off = j.add(a)
        j.eps.append((start, off + a.initial))
        j.accepting += [off + int(q) for q in np.nonzero(a.accept)[0]]
    return j.build(start)


def optional(a: ByteAutomaton) -> ByteAutomaton:
    """``a``'s texts and the empty text."""
    return union(a, epsilon())


def star(a: ByteAutomaton) -> ByteAutomaton:
    """Concatenations of any number of ``a``'s texts (the empty text among them)."""
    return repeat(a, 0, None)


def repeat(a: ByteAutomaton, lo: int, hi: Optional[int]) -> ByteAutomaton:
    """Concatenations of ``lo`` to ``hi`` of ``a``'s texts (``hi`` None: no bound)."""
    if lo < 0 or (hi is not None and hi < lo):
        raise ValueError(f"repeat: bounds {lo}..{hi} are not 0 <= lo <= hi.")
    if hi == 0:
        return epsilon()
    j = _Joined()
    start = j.fresh()
    if lo == 0:
        j.accepting.append(start)
    copies = max(lo, 1) if hi is None else hi
    prev = [start]                                              # the states a next copy is entered from
    for i in range(copies):
        off = j.add(a)
        j.eps += [(p, off + a.initial) for p in prev]
        prev = [off + int(q) for q in np.nonzero(a.accept)[0]]
        if i + 1 >= lo:
            j.accepting += prev
        if hi is None and i == copies - 1:                      # the last copy loops
            j.eps += [(p, off + a.initial) for p in prev]
    return j.build(start)


# -------------------------------------------------------------------------------------------------
# Intersection and difference: products
# -------------------------------------------------------------------------------------------------

def _product(parts: Sequence[ByteAutomaton], mode: str) -> ByteAutomaton:
    byte_class, first = _byte_classes([a.delta for a in parts])
    flat = [a.delta[:, first].astype(np.int64).reshape(-1) for a in parts]
    offsets = np.zeros(len(parts) + 1, dtype = np.int64)
    offsets[1:] = np.cumsum([f.size for f in flat])
    Ks = np.array([a.num_states for a in parts], dtype = np.int64)
    child = np.repeat(np.arange(first.size, dtype = np.int64)[:, None], len(parts), axis = 1)
    rows, keys = product_bfs(np.concatenate(flat), offsets, Ks, child,
                             np.array([a.initial for a in parts], dtype = np.int64))
    accept = np.ones(keys.size, dtype = bool)
    rest = keys.copy()
    for i in range(len(parts) - 1, -1, -1):
        a = parts[i].accept[rest % Ks[i]]
        accept &= ~a if (mode == "difference" and i > 0) else a
        rest //= Ks[i]
    return _from_classes(rows, 0, accept, byte_class)


def intersect(*parts: ByteAutomaton) -> ByteAutomaton:
    """The texts every one of ``parts`` accepts (two at a time, each product minimal)."""
    if not parts:
        raise ValueError("intersect: no automaton (every text is no regular set of texts here).")
    out = minimize(parts[0])
    for a in parts[1:]:
        out = _product([out, minimize(a)], "intersection")
    return out


def difference(a: ByteAutomaton, b: ByteAutomaton) -> ByteAutomaton:
    """The texts ``a`` accepts and ``b`` does not."""
    return _product([minimize(a), minimize(b)], "difference")
