"""
Byte-level automata, and the token automata they induce through a vocabulary: the layer text-level constraints
(:class:`~pyjuice.constraints.Regex`, ...) compile through.

A :class:`ByteAutomaton` is a complete DFA over the 256 byte values whose language is a set of texts, byte for byte
(valid UTF-8 when it comes from a regular expression, :mod:`.patterns`, every character set as sequences of UTF-8 byte
ranges: RE2's split), with one dead state, kept per byte class (the bytes no state tells apart).

:func:`to_tokens` turns it into a :class:`~pyjuice.constraints.DFA` over a
:class:`~pyjuice.constraints.Vocabulary`'s ids: a token sequence is accepted when its text -- the tokens' bytes,
concatenated -- is, whichever tokens spell it (tokens that split a character included); under the end-of-sequence
padding convention the text may be followed by end-of-sequence ids, and only by them. From every state reached at a
token boundary, in breadth-first order from the initial state, a depth-first walk of the vocabulary's byte trie stops
at the dead state, so the work is the number of live (state, trie node) pairs rather than states times vocabulary. The
trie walked has its nodes merged: texts spelling the same byte classes lead to the same states from every state, so
they are one node, and what no state reads is left out (:func:`_merged_trie`).
Tokens leading to the same state from every reached state share a class: two independent 64-bit random linear hashes
of every merged node's successors, summed during the walks, then one representative token per class.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from numba import get_num_threads, njit, prange

from ..dfa import DFA
from ..vocabulary import Vocabulary
from .dfa import _column_classes, bfs_order

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

    @property
    def delta(self) -> np.ndarray:
        if self._delta is None:
            self._delta = np.ascontiguousarray(self._table[:, self._byte_class])
        return self._delta

    def _classes(self):
        self._byte_class, first = _column_classes(self._delta)
        self._table = np.ascontiguousarray(self._delta[:, first])

    @property
    def table(self) -> np.ndarray:
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


# -------------------------------------------------------------------------------------------------
# Token automata
# -------------------------------------------------------------------------------------------------

@njit(cache = True)
def _trie_tables(data, offsets, tokens):
    """The byte trie of ``tokens`` (ids; ``data[offsets[v]:offsets[v + 1]]`` the bytes of ``v``, none of them empty), its
    nodes numbered breadth-first, every node's children together and by increasing byte: ``(first_child, node_byte,
    end_ptr, end_ids)``: a node's children are the nodes ``first_child[u]`` to ``first_child[u + 1] - 1``, ``node_byte``
    the byte into every node, the tokens ending at it CSR (ascending)."""
    most = data.shape[0] + 1                                    # (every byte a node, at most)
    child = np.empty(most, dtype = np.int32)                    # as they appear: first child, next sibling, byte
    sibling = np.empty(most, dtype = np.int32)
    byte = np.empty(most, dtype = np.uint8)
    end_node = np.empty(tokens.shape[0], dtype = np.int32)
    root = np.full(256, -1, dtype = np.int32)
    child[0] = -1
    num = 1
    for i in range(tokens.shape[0]):
        v = tokens[i]
        u = 0
        for k in range(offsets[v], offsets[v + 1]):
            b = data[k]
            if u == 0:
                w = root[b]
            else:
                w = child[u]
                while w >= 0 and byte[w] != b:
                    w = sibling[w]
            if w < 0:
                w = num
                num += 1
                byte[w], child[w] = b, -1
                sibling[w], child[u] = child[u], w
                if u == 0:
                    root[b] = w
            u = w
        end_node[i] = u
    # breadth-first: every node's children (by increasing byte) get the next numbers
    number = np.empty(num, dtype = np.int32)
    order = np.empty(num, dtype = np.int32)
    first_child = np.empty(num + 1, dtype = np.int32)
    node_byte = np.empty(num, dtype = np.uint8)
    number[0], order[0], node_byte[0] = 0, 0, 0
    last = 1
    for i in range(num):
        first_child[i] = last
        at = last
        w = child[order[i]]
        while w >= 0:                                           # (insertion: a node has few children, but for the root)
            j = at
            while j > last and node_byte[j - 1] > byte[w]:
                node_byte[j], order[j] = node_byte[j - 1], order[j - 1]
                j -= 1
            node_byte[j], order[j] = byte[w], w
            at += 1
            w = sibling[w]
        last = at
    first_child[num] = last
    for i in range(num):
        number[order[i]] = i
    end_ptr = np.zeros(num + 1, dtype = np.int32)
    for i in range(tokens.shape[0]):
        end_ptr[number[end_node[i]] + 1] += 1
    for i in range(num):
        end_ptr[i + 1] += end_ptr[i]
    end_ids = np.empty(tokens.shape[0], dtype = np.int32)
    fill = end_ptr[:num].copy()
    for i in range(tokens.shape[0]):
        u = number[end_node[i]]
        end_ids[fill[u]] = tokens[i]
        fill[u] += 1
    return first_child, node_byte, end_ptr, end_ids


class _Trie:
    """The byte trie of a vocabulary's tokens with nonempty text, breadth-first: :func:`_trie_tables`."""

    def __init__(self, vocabulary: Vocabulary):
        offsets, has_text = vocabulary._offsets, vocabulary._has_text
        self.data = np.frombuffer(vocabulary._buffer, dtype = np.uint8)
        tokens = np.nonzero(has_text & (offsets[1:] > offsets[:-1]))[0]
        self.first_child, self.node_byte, self.end_ptr, self.end_ids = _trie_tables(self.data, offsets, tokens)


def _trie(vocabulary: Vocabulary) -> _Trie:
    """The vocabulary's byte trie, built on first use and kept with it."""
    if vocabulary._byte_trie is None:
        vocabulary._byte_trie = _Trie(vocabulary)
    return vocabulary._byte_trie


@njit(cache = True)
def _merged_trie(first_child, node_byte, end_ptr, end_ids, byte_class, read, V):
    """
    The trie as far as the automaton can tell: its nodes whose texts spell the same byte classes as one node (from any
    state, such texts lead to the same state), edges on a class ``read`` by no state left out. Breadth-first, each node
    holding its trie nodes, at most once per trie node: ``(ptr, edge_class, edge_node, token_node, holds)``: every merged
    node's edges (CSR), the merged node of every token (node 0, the root, where no token ends, if no walk gets there)
    and whether tokens end at it.
    """
    N = first_child.shape[0] - 1
    C = read.shape[0]
    members = np.empty(N, dtype = np.int32)
    first = np.empty(N + 1, dtype = np.int32)
    ptr = np.empty(N + 1, dtype = np.int32)
    edge_class = np.empty(N, dtype = np.int32)
    edge_node = np.empty(N, dtype = np.int32)
    holds = np.empty(N, dtype = np.bool_)
    token_node = np.zeros(V, dtype = np.int32)
    count = np.zeros(C, dtype = np.int32)
    place = np.empty(C, dtype = np.int32)
    touched = np.empty(C, dtype = np.int32)
    members[0], first[0], first[1] = 0, 0, 1
    num, edges, m = 1, 0, 0
    while m < num:
        ptr[m] = edges
        a, b = first[m], first[m + 1]
        held = False
        if b == a + 1 and first_child[members[a] + 1] <= first_child[members[a]] + 1:   # (most nodes: a chain)
            u = members[a]
            for k in range(end_ptr[u], end_ptr[u + 1]):
                token_node[end_ids[k]] = m
                held = True
            holds[m] = held
            w = first_child[u]
            if w < first_child[u + 1] and read[byte_class[node_byte[w]]]:
                end = first[num]
                members[end] = w
                first[num + 1] = end + 1
                edge_class[edges], edge_node[edges] = byte_class[node_byte[w]], num
                edges += 1
                num += 1
            m += 1
            continue
        t = 0
        for i in range(a, b):
            u = members[i]
            for k in range(end_ptr[u], end_ptr[u + 1]):
                token_node[end_ids[k]] = m
                held = True
            for w in range(first_child[u], first_child[u + 1]):
                c = byte_class[node_byte[w]]
                if read[c]:
                    if count[c] == 0:
                        touched[t] = c
                        t += 1
                    count[c] += 1
        holds[m] = held
        end = first[num]
        for j in range(t):
            c = touched[j]
            place[c] = end
            end += count[c]
            count[c] = 0
            first[num + 1] = end
            edge_class[edges], edge_node[edges] = c, num
            edges += 1
            num += 1
        for i in range(a, b):
            u = members[i]
            for w in range(first_child[u], first_child[u + 1]):
                c = byte_class[node_byte[w]]
                if read[c]:
                    members[place[c]] = w
                    place[c] += 1
        m += 1
    ptr[num] = edges
    return ptr[:num + 1], edge_class[:edges], edge_node[:edges], token_node, holds[:num]


@njit(cache = True)
def _classes_read(table, dead):
    """bool [classes]: the classes on which some live state moves to a live state."""
    S, C = table.shape
    read = np.zeros(C, dtype = np.bool_)
    left = C
    for q in range(S):
        if q != dead:
            for c in range(C):
                if not read[c] and table[q, c] != dead:
                    read[c] = True
                    left -= 1
            if left == 0:
                break
    return read


@njit(cache = True)
def _walk(table, dead, initial, ptr, edge_class, edge_node, holds, r1, r2, r3, r4, budget):
    """The states reached at token boundaries (breadth-first from ``initial``) and, per merged node, two hashes of its
    successors from them: from every reached state a depth-first walk of the merged trie (:func:`_merged_trie`) through
    live states -- as far as ``budget`` edges go: ``(h1, h2, states, walked)``, the first ``walked`` of them walked (every
    state a walk found, if it is all of them). The hashes tell apart the nodes these states tell apart, by the state they
    lead to, whether tokens end at them or not."""
    S = table.shape[0]
    M = ptr.shape[0] - 1
    h1 = np.zeros(M + 2, dtype = np.uint64)
    h2 = np.zeros(M + 2, dtype = np.uint64)
    order = np.empty(S, dtype = np.int64)
    seen = np.zeros(S, dtype = np.bool_)
    order[0] = initial
    seen[initial] = True
    num, head = 1, 0
    stack_node = np.empty(M + 1, dtype = np.int32)
    stack_q = np.empty(M + 1, dtype = np.int32)
    steps = 0
    while head < num and steps < budget:
        s = order[head]
        head += 1
        a1, b1, a2, b2 = r1[s], r2[s], r3[s], r4[s]
        stack_node[0] = 0
        stack_q[0] = s
        top = 1
        while top > 0:
            top -= 1
            u = stack_node[top]
            q = stack_q[top]
            steps += ptr[u + 1] - ptr[u]
            for e in range(ptr[u], ptr[u + 1]):
                q2 = table[q, edge_class[e]]
                if q2 == dead:
                    continue
                w = edge_node[e]
                x = np.uint64(q2 + 1)
                h1[w] += a1 * x + b1
                h2[w] += a2 * x + b2
                if holds[w] and not seen[q2]:                          # a token ends here: a boundary state
                    seen[q2] = True
                    order[num] = q2
                    num += 1
                stack_node[top] = w
                stack_q[top] = q2
                top += 1
    return h1, h2, order[:num], head


@njit(cache = True)
def _walks(table, dead, states, ptr, edge_class, edge_node, r1, r2, r3, r4, h1, h2, budget):
    """:func:`_walk`'s hashes, added to ``h1`` and ``h2``, from ``states`` in turn, as far as ``budget`` edges go: how many
    were walked, one at least."""
    M = ptr.shape[0] - 1
    stack_node = np.empty(M + 1, dtype = np.int32)
    stack_q = np.empty(M + 1, dtype = np.int32)
    steps, i = 0, 0
    while i < states.shape[0] and (i == 0 or steps < budget):
        s = states[i]
        i += 1
        a1, b1, a2, b2 = r1[s], r2[s], r3[s], r4[s]
        stack_node[0] = 0
        stack_q[0] = s
        top = 1
        while top > 0:
            top -= 1
            u = stack_node[top]
            q = stack_q[top]
            steps += ptr[u + 1] - ptr[u]
            for e in range(ptr[u], ptr[u + 1]):
                q2 = table[q, edge_class[e]]
                if q2 == dead:
                    continue
                w = edge_node[e]
                x = np.uint64(q2 + 1)
                h1[w] += a1 * x + b1
                h2[w] += a2 * x + b2
                stack_node[top] = w
                stack_q[top] = q2
                top += 1
    return i


@njit(parallel = True, cache = True)
def _walks_parallel(table, dead, states, ptr, edge_class, edge_node, r1, r2, r3, r4, P, slots):
    """:func:`_walks` of all ``states``, no budget, in parallel: ``P`` walkers take every ``P``-th state, each summing into
    its own hashes -- ``slots`` of them, the merged nodes and two more --, then summed."""
    S = states.shape[0]
    h1 = np.zeros((P, slots), dtype = np.uint64)
    h2 = np.zeros((P, slots), dtype = np.uint64)
    for c in prange(P):
        stack_node = np.empty(slots, dtype = np.int32)
        stack_q = np.empty(slots, dtype = np.int32)
        for i in range(c, S, P):
            s = states[i]
            a1, b1, a2, b2 = r1[s], r2[s], r3[s], r4[s]
            stack_node[0] = 0
            stack_q[0] = s
            top = 1
            while top > 0:
                top -= 1
                u = stack_node[top]
                q = stack_q[top]
                for e in range(ptr[u], ptr[u + 1]):
                    q2 = table[q, edge_class[e]]
                    if q2 == dead:
                        continue
                    w = edge_node[e]
                    x = np.uint64(q2 + 1)
                    h1[c, w] += a1 * x + b1
                    h2[c, w] += a2 * x + b2
                    stack_node[top] = w
                    stack_q[top] = q2
                    top += 1
    o1 = np.zeros(slots, dtype = np.uint64)
    o2 = np.zeros(slots, dtype = np.uint64)
    for c in range(P):
        for t in range(slots):
            o1[t] += h1[c, t]
            o2[t] += h2[c, t]
    return o1, o2


@njit(cache = True)
def _quotient(h1, h2, ptr, edge_class, edge_node):
    """
    The merged trie with the nodes of the same hashes as one (no state walked tells them apart; the root stays alone):
    ``(gptr, gclass, gnode, into, zero)``: its edges, one per (node, class), as CSR, the edges into every node, and the
    node of those no walk reached (-1: none). What the walks have told apart a child of it keeps: the nodes of a node's
    children on a class are one, as the same states lead to the same states.
    """
    M = ptr.shape[0] - 1
    E = edge_node.shape[0]
    cap = 16
    while cap < 2 * M:
        cap *= 2
    slots = np.full(cap, -1, dtype = np.int32)
    of = np.empty(M, dtype = np.int32)
    rep = np.empty(M, dtype = np.int32)
    of[0], rep[0] = 0, 0
    G = 1
    zero = -1
    for u in range(1, M):
        at = np.int64((h1[u] ^ (h2[u] >> np.uint64(17))) & np.uint64(cap - 1))
        while True:
            g = slots[at]
            if g < 0:
                g = G
                G += 1
                slots[at] = g
                rep[g] = u
                if h1[u] == 0 and h2[u] == 0:
                    zero = g
                break
            if h1[rep[g]] == h1[u] and h2[rep[g]] == h2[u]:
                break
            at = (at + 1) & (cap - 1)
        of[u] = g
    cap = 16
    while cap < 2 * E:
        cap *= 2
    keys = np.full(cap, np.int64(-1))
    src = np.empty(E, dtype = np.int32)
    gclass = np.empty(E, dtype = np.int32)
    gnode = np.empty(E, dtype = np.int32)
    C = 0
    for e in range(E):
        C = max(C, edge_class[e] + 1)
    num = 0
    gptr = np.zeros(G + 1, dtype = np.int64)
    for u in range(M):
        g = of[u]
        for e in range(ptr[u], ptr[u + 1]):
            key = np.int64(g) * C + edge_class[e]
            at = (key * np.int64(0x9E3779B97F4A7C1)) & (cap - 1)
            while keys[at] >= 0 and keys[at] != key:
                at = (at + 1) & (cap - 1)
            if keys[at] < 0:
                keys[at] = key
                src[num], gclass[num], gnode[num] = g, edge_class[e], of[edge_node[e]]
                gptr[g + 1] += 1
                num += 1
    for g in range(G):
        gptr[g + 1] += gptr[g]
    fill = gptr[:G].copy()
    out_class = np.empty(num, dtype = np.int32)
    out_node = np.empty(num, dtype = np.int32)
    into = np.zeros(G, dtype = np.int32)
    for i in range(num):
        g = src[i]
        out_class[fill[g]], out_node[fill[g]] = gclass[i], gnode[i]
        fill[g] += 1
        into[gnode[i]] += 1
    return gptr, out_class, out_node, into, zero


@njit(cache = True)
def _check_part(table, dead, states, first, step, gptr, gclass, gnode, into, zero, bad):
    """``bad[i]``, every ``step``-th of ``states`` from ``first``, unless the state leads every trie node of a node of the
    quotient (:func:`_quotient`) to one state, or nowhere: it then tells no two of them apart. A walk of the quotient from
    the root gives every node it reaches a state: another edge into that node must say the same, and every edge into it
    must be walked: from a node reached, on a class that does not die. Then by induction over a trie node's depth it leads
    to its quotient node's state. Through a node no walk reached (``zero``) it fails: that says nowhere for the tokens."""
    G = gptr.shape[0] - 1
    value = np.empty(G, dtype = np.int32)
    stamp = np.full(G, -1, dtype = np.int64)
    queue = np.empty(G, dtype = np.int32)
    for i in range(first, states.shape[0], step):
        stamp[0] = i
        value[0] = states[i]
        queue[0] = 0
        num, head = 1, 0
        edges, needed = 0, 0
        ok = True
        while head < num and ok:
            g = queue[head]
            head += 1
            q = value[g]
            for e in range(gptr[g], gptr[g + 1]):
                q2 = table[q, gclass[e]]
                if q2 == dead:
                    continue
                w = gnode[e]
                edges += 1
                if stamp[w] != i:
                    stamp[w] = i
                    value[w] = q2
                    queue[num] = w
                    num += 1
                    needed += into[w]
                    if w == zero:
                        ok = False
                        break
                elif value[w] != q2:
                    ok = False
                    break
        bad[i] = not ok or edges != needed


@njit(parallel = True, cache = True)
def _check_parallel(table, dead, states, gptr, gclass, gnode, into, zero, bad, P):
    """:func:`_check_part` of all ``states``: ``P`` parts."""
    for c in prange(P):
        _check_part(table, dead, states, c, P, gptr, gclass, gnode, into, zero, bad)


def _not_told_apart(table: np.ndarray, dead: int, states: np.ndarray, quotient) -> np.ndarray:
    """Those of ``states`` that tell apart two trie nodes of a node of the quotient (:func:`_check_part`)."""
    bad = np.empty(states.size, dtype = bool)
    threads = get_num_threads()
    if states.size >= SERIAL_CHECKS and threads > 1:
        _check_parallel(table, dead, states, *quotient, bad, min(threads, states.size))
    else:
        _check_part(table, dead, states, 0, 1, *quotient, bad)
    return states[bad]


def _walk_the_rest(table: np.ndarray, dead: int, rest: np.ndarray, ptr: np.ndarray, edge_class: np.ndarray, edge_node: np.ndarray,
                   r: Sequence[np.ndarray], h1: np.ndarray, h2: np.ndarray) -> None:
    """
    The token classes as if walks from ``rest`` were added to the hashes ``h1`` and ``h2``, with next to no walks: of two
    hundred thousand states a few hundred tell apart all a walk from every one would (a state in a long string reads its
    tokens like the next). The nodes the hashes so far tell apart are a far smaller graph (:func:`_quotient`). Every state
    is checked on it (:func:`_check_part`), and walks only if it tells apart what the hashes do not; from then on the graph
    is that much larger. The states that have passed tell nothing apart whatever the graph comes to be: what they cannot
    tell apart stays together or is told apart by others.
    """
    M = ptr.size - 1
    quotient = _quotient(h1, h2, ptr, edge_class, edge_node)
    budget, at, batch = M, 0, FIRST_CHECKS
    while at < rest.size:
        left = rest[at:at + batch]
        at += left.size
        batch *= 2
        while True:
            left = _not_told_apart(table, dead, left, quotient)
            if left.size == 0:
                break
            if budget < WALKS_IN_THREADS or get_num_threads() == 1:
                walked = _walks(table, dead, left, ptr, edge_class, edge_node, r[0], r[1], r[2], r[3], h1, h2, budget)
                left = left[walked:]
                budget *= 2
            else:                                                       # (states that do tell much apart: all at once)
                a1, a2 = _walks_parallel(table, dead, left, ptr, edge_class, edge_node, r[0], r[1], r[2], r[3],
                                         min(get_num_threads(), left.size), M + 2)
                with np.errstate(over = "ignore"):
                    h1 += a1
                    h2 += a2
                left = left[:0]
            quotient = _quotient(h1, h2, ptr, edge_class, edge_node)
            if left.size == 0:
                break


@njit(cache = True)
def _token_classes(h1, h2, token_node):
    """The tokens whose merged nodes have the same hashes as one class, the classes numbered by their smallest token:
    ``(token_class, reps, hashed)``: int64 [V], every class's smallest token, and whether it has a hash (a class without
    leads nowhere from any state)."""
    V = token_node.shape[0]
    M = h1.shape[0]
    cap = 16
    while cap < 2 * M:
        cap *= 2
    slots = np.full(cap, -1, dtype = np.int32)
    node_class = np.full(M, -1, dtype = np.int32)
    token_class = np.empty(V, dtype = np.int64)
    reps = np.empty(M, dtype = np.int64)
    rep_node = np.empty(M, dtype = np.int32)
    num = 0
    for t in range(V):
        m = token_node[t]
        c = node_class[m]
        if c < 0:
            at = np.int64((h1[m] ^ (h2[m] >> np.uint64(17))) & np.uint64(cap - 1))
            while True:
                c = slots[at]
                if c < 0:
                    c = num
                    slots[at] = c
                    reps[c], rep_node[c] = t, m
                    num += 1
                    break
                if h1[rep_node[c]] == h1[m] and h2[rep_node[c]] == h2[m]:
                    break
                at = (at + 1) & (cap - 1)
            node_class[m] = c
        token_class[t] = c
    hashed = np.empty(num, dtype = np.bool_)
    for c in range(num):
        hashed[c] = h1[rep_node[c]] != 0 or h2[rep_node[c]] != 0
    return token_class, reps[:num], hashed


def _byte_complete(trie: "_Trie") -> bool:
    """Whether every byte is a token's whole text: then every state a byte text reaches is reached at a token
    boundary, and states a byte text tells apart are told apart by tokens."""
    first, last = trie.first_child[0], trie.first_child[1]
    if last - first != 256:
        return False
    return bool((trie.end_ptr[first + 1:last + 1] > trie.end_ptr[first:last]).all())


@njit(cache = True)
def _rows(table, dead, states, reps, kinds, offsets, data, byte_class, remap, out, lo, hi):
    """Rows ``lo`` to ``hi`` of ``out`` [len(states), len(reps)]: the successor of every state on every representative
    token, by ``remap`` -- kind 0 its bytes (``data``), 1 empty text (the state itself), 2 nowhere (the dead state)."""
    C = reps.shape[0]
    gone = remap[dead]
    for i in range(lo, hi):
        s = states[i]
        for c in range(C):
            kind = kinds[c]
            if kind == 2:
                out[i, c] = gone
            elif kind == 1:
                out[i, c] = remap[s]
            else:
                t = reps[c]
                q = s
                for k in range(offsets[t], offsets[t + 1]):
                    q = table[q, byte_class[data[k]]]
                    if q == dead:
                        break
                out[i, c] = remap[q]


@njit(parallel = True, cache = True)
def _rows_parallel(table, dead, states, reps, kinds, offsets, data, byte_class, remap, out, P):
    """:func:`_rows` in ``P`` parts."""
    R = states.shape[0]
    for c in prange(P):
        _rows(table, dead, states, reps, kinds, offsets, data, byte_class, remap, out, c * R // P, (c + 1) * R // P)


#: so many edges of the merged trie are walked first, or a table of so many entries stays on one thread: waking a team is
#: milliseconds on a busy machine
SERIAL_STEPS = 1 << 19
NO_BUDGET = 1 << 62
#: states checked on the quotient of the merged trie (:func:`_walk_the_rest`): at first so many, then twice as many a time,
#: on one thread below :data:`SERIAL_CHECKS`; states failing are walked, so many edges in threads at the most at a time
FIRST_CHECKS = 256
SERIAL_CHECKS = 1 << 12
WALKS_IN_THREADS = 1 << 24


def to_tokens(automaton: ByteAutomaton, vocabulary: Vocabulary, eos_padding: bool) -> DFA:
    """
    The minimal :class:`~pyjuice.constraints.DFA` over ``vocabulary``'s ids of the token sequences whose text the
    byte automaton accepts (see the module docstring) -- with ``eos_padding``, followed by any number of
    end-of-sequence ids and nothing else.
    """
    V, S = vocabulary.size, automaton.num_states
    trie = _trie(vocabulary)
    table, byte_class = automaton.table, automaton.byte_class
    dead_byte, start = automaton.dead, automaton.initial
    rng = np.random.default_rng(0)
    r = [rng.integers(1, 2 ** 63, size = S, dtype = np.uint64) for _ in range(4)]
    r[0] |= np.uint64(1)
    r[2] |= np.uint64(1)
    ptr, edge_class, edge_node, token_node, holds = _merged_trie(trie.first_child, trie.node_byte, trie.end_ptr, trie.end_ids,
                                                                 byte_class, _classes_read(table, dead_byte), V)
    M = ptr.size - 1
    complete = automaton.minimal and start != dead_byte and _byte_complete(trie)
    h1, h2, states, walked = _walk(table, dead_byte, start, ptr, edge_class, edge_node, holds, r[0], r[1], r[2], r[3],
                                   SERIAL_STEPS if complete else NO_BUDGET)
    if walked < states.size:        # the states reached at token boundaries: every one a byte text reaches
        left = np.ones(S, dtype = bool)
        left[states[:walked]] = False
        states = bfs_order(table, start)
        states = states[states != dead_byte]
        _walk_the_rest(table, dead_byte, states[left[states]], ptr, edge_class, edge_node, r, h1, h2)
    offsets, has_text = vocabulary._offsets, vocabulary._has_text
    empty = np.nonzero(has_text & (offsets[1:] == offsets[:-1]))[0]
    if empty.size:                                                      # empty text: every state to itself
        with np.errstate(over = "ignore"):
            x = (states + 1).astype(np.uint64)
            h1[M] = np.sum(r[0][states] * x + r[1][states], dtype = np.uint64)
            h2[M] = np.sum(r[2][states] * x + r[3][states], dtype = np.uint64)
        token_node[empty] = M
    eos = np.asarray(vocabulary.eos, dtype = np.int64)
    if eos_padding and eos.size:                                        # a class of their own
        h1[M + 1] = h2[M + 1] = np.uint64(2 ** 64 - 1)
        token_node[eos] = M + 1
    token_class, reps, hashed = _token_classes(h1, h2, token_node)
    kinds = np.where(hashed & has_text[reps], np.where(offsets[reps + 1] == offsets[reps], 1, 0), 2).astype(np.int64)
    K, C = states.size, reps.size
    remap = np.full(S, K, dtype = np.int64)                             # unreached (the byte dead state among them)
    remap[states] = np.arange(K)
    dead, pad = K, K + 1
    out = np.empty((K + 2, C), dtype = np.int64)
    out[K:] = dead
    if K * C >= SERIAL_STEPS and get_num_threads() > 1:
        _rows_parallel(table, dead_byte, states, reps, kinds, offsets, trie.data, byte_class, remap, out, get_num_threads())
    else:
        _rows(table, dead_byte, states, reps, kinds, offsets, trie.data, byte_class, remap, out, 0, K)
    accept = np.zeros(K + 2, dtype = bool)
    accept[:K] = automaton.accept[states]
    if eos_padding and eos.size:
        ce = int(token_class[eos[0]])                                   # (no text: nowhere, so far)
        accepting = np.nonzero(accept)[0]
        if complete:                # a state accepting the empty text alone is the padding's (else it is a new one)
            alone = accepting[(out[accepting] == dead).all(axis = 1)]
            pad = int(alone[0]) if alone.size else pad
        out[accepting, ce] = pad
        out[pad, ce] = pad
        accept[pad] = True
    tc, table, kept = torch.from_numpy(token_class), torch.from_numpy(out), torch.from_numpy(accept)
    # minimal byte automaton, every byte a token: the states are told apart by tokens as by bytes -- minimal already, and
    # so the tokens of every class (numbered by their smallest) are told apart
    return DFA._of_classes(V, tc, table, 0, kept)._renumbered() if complete else DFA(V, tc, table, 0, kept).minimize()


def widened(dfa: DFA, vocab_size: int) -> DFA:
    """``dfa`` over the larger vocabulary ``0..vocab_size-1``, the new ids leading to a dead state from every state
    (one is added if the automaton has none)."""
    if vocab_size == dfa.vocab_size:
        return dfa
    delta, accept, tc = dfa.delta, dfa.accept, dfa.token_class
    live = dfa.live_states()
    if bool(live.all()):
        sink = delta.size(0)
        delta = torch.cat([delta, torch.full((1, delta.size(1)), sink, dtype = delta.dtype)], 0)
        accept = torch.cat([accept, torch.zeros(1, dtype = torch.bool)])
    else:
        sink = int(torch.nonzero(~live)[0])
    delta = torch.cat([delta, torch.full((delta.size(0), 1), sink, dtype = delta.dtype)], 1)
    tc = torch.cat([tc, torch.full((vocab_size - dfa.vocab_size,), delta.size(1) - 1, dtype = tc.dtype)])
    return DFA(vocab_size, tc, delta, dfa.initial, accept)
