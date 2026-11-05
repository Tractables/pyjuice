"""
A byte automaton (:class:`~.byte_level.ByteAutomaton`) as a token automaton: :func:`to_tokens` turns it into a
:class:`~pyjuice.constraints.DFA` over a
:class:`~pyjuice.constraints.Vocabulary`'s ids: a token sequence is accepted when its text -- the tokens' bytes,
concatenated -- is, whichever tokens spell it (tokens that split a character included); under the end-of-sequence
padding convention the text may be followed by end-of-sequence ids, and only by them. From every state reached at a
token boundary, in breadth-first order from the initial state, a depth-first walk of the vocabulary's byte trie stops
at the dead state, and takes a subtree a level at a time where every byte below a node leads a state to one successor
(:func:`_walk_one`), so the work is far below the number of live (state, trie node) pairs.
Tokens leading to the same state from every reached state share a class: two independent 64-bit random linear hashes
of every trie node's successors, summed during the walks, then one representative token per class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from numba import get_num_threads, njit, prange

from ..dfa import DFA
from ..vocabulary import Vocabulary
from .behaviours import behaviours_of, classes_of_behaviours, rows_of_behaviours
from .counters import NotCounted, expanded_moves, states_put_back
from .dfa import bfs_order
from .byte_level import ByteAutomaton
from .threads import few_threads


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


@njit(cache = True)
def _bytes_below(first_child, node_byte):
    """uint64 [nodes, 4]: the bytes of every edge below a node, as bits (a child is numbered past its parent)."""
    M = first_child.shape[0] - 1
    below = np.zeros((M, 4), dtype = np.uint64)
    for u in range(M - 1, -1, -1):
        for w in range(first_child[u], first_child[u + 1]):
            b = node_byte[w]
            for k in range(4):
                below[u, k] |= below[w, k]
            below[u, b >> 6] |= np.uint64(1) << np.uint64(b & 63)
    return below


@njit(cache = True)
def _ends(end_ptr, end_ids, V, empty):
    """
    Where :func:`_token_classes` reads the hashes of tokens, in order: the root (nothing is added to it: tokens without text are
    there), the nodes tokens end at, the place past the trie's nodes for tokens of empty text (``empty``) and the one past that for
    the ends of sequence under padding. ``(end_nodes, token_end)``: those places, and every token's among them.
    """
    M = end_ptr.shape[0] - 1
    E = 3
    for u in range(1, M):
        if end_ptr[u + 1] > end_ptr[u]:
            E += 1
    end_nodes = np.empty(E, dtype = np.int32)
    token_end = np.zeros(V, dtype = np.int32)
    end_nodes[0] = 0
    n = 1
    for u in range(1, M):
        if end_ptr[u + 1] > end_ptr[u]:
            end_nodes[n] = u
            for k in range(end_ptr[u], end_ptr[u + 1]):
                token_end[end_ids[k]] = n
            n += 1
    end_nodes[n], end_nodes[n + 1] = M, M + 1
    for i in range(empty.shape[0]):
        token_end[empty[i]] = n
    return end_nodes, token_end


@njit(cache = True)
def _parents(first_child):
    """Every node's parent (the root's: itself)."""
    M = first_child.shape[0] - 1
    parent = np.zeros(M, dtype = np.int32)
    for u in range(M):
        for w in range(first_child[u], first_child[u + 1]):
            parent[w] = u
    return parent


class _Trie:
    """The byte trie of a vocabulary's tokens with nonempty text, breadth-first (:func:`_trie_tables`), and what walks of it read
    whatever the automaton: the bytes below every node (:func:`_bytes_below`), where tokens end (:func:`_ends`)."""

    def __init__(self, vocabulary: Vocabulary):
        offsets, has_text = vocabulary._offsets, vocabulary._has_text
        self.data = np.frombuffer(vocabulary._buffer, dtype = np.uint8)
        tokens = np.nonzero(has_text & (offsets[1:] > offsets[:-1]))[0]
        self.first_child, self.node_byte, self.end_ptr, self.end_ids = _trie_tables(self.data, offsets, tokens)
        self.below = _bytes_below(self.first_child, self.node_byte)
        self.ends_before = self.end_ptr
        self.empty = np.nonzero(has_text & (offsets[1:] == offsets[:-1]))[0]
        self.end_nodes, self.token_end = _ends(self.end_ptr, self.end_ids, vocabulary.size, self.empty)
        self.parent = _parents(self.first_child)
        self._places = {}

    def token_ends(self, V: int, eos: tuple) -> np.ndarray:
        """Every token's place among :attr:`end_nodes`, of ``V`` ids (more than the trie was made for: they have no text), the ids ``eos``
        at the last place."""
        places = self._places.get((V, eos))
        if places is None:
            places = np.zeros(V, dtype = np.int32)
            places[:self.token_end.size] = self.token_end
            places[list(eos)] = self.end_nodes.size - 1
            self._places[V, eos] = places
        return places


def _trie(vocabulary: Vocabulary) -> _Trie:
    """The vocabulary's byte trie, built on first use and kept with it."""
    if vocabulary._byte_trie is None:
        vocabulary._byte_trie = _Trie(vocabulary)
    return vocabulary._byte_trie


#: a node with more children than this has those a state lives on found by their bytes, if the state leads nowhere on most
MANY_CHILDREN = 64
#: the lowest bit of a word, by a de Bruijn sequence
_DE_BRUIJN = np.uint64(0x03f79d71b4cb0a89)
_TRAILING_ZEROS = np.zeros(64, dtype = np.int64)
for _bit in range(64):
    _TRAILING_ZEROS[(((1 << _bit) * int(_DE_BRUIJN)) & (2 ** 64 - 1)) >> 58] = _bit


@njit(cache = True)
def _bytes_of_classes(byte_class, C):
    """uint64 [C, 4]: the bytes of every class, as bits."""
    out = np.zeros((C, 4), dtype = np.uint64)
    for b in range(256):
        out[byte_class[b], b >> 6] |= np.uint64(1) << np.uint64(b & 63)
    return out


@njit(cache = True)
def _usual_move(table, class_bytes, q, dead, usual, same, count):
    """Keep ``usual[q]``, the successor most classes lead ``q`` to (``count`` zeros), and ``same[q]``, the bytes of those classes as bits."""
    C = table.shape[1]
    row = table[q]
    live = 0
    for c in range(C):
        live += row[c] != dead
    if 2 * live <= C:                                           # (nearly all states: the dead state)
        same[q, :] = ~np.uint64(0)
        for c in range(C):
            if row[c] != dead:
                for k in range(4):
                    same[q, k] &= ~class_bytes[c, k]
        usual[q] = dead
        return
    best, most = row[0], 0
    for c in range(C):
        x = row[c]
        count[x] += 1
        if count[x] > most:
            most, best = count[x], x
    for c in range(C):
        count[row[c]] = 0
    for c in range(C):
        if row[c] == best:
            for k in range(4):
                same[q, k] |= class_bytes[c, k]
    usual[q] = best


@njit(cache = True)
def _usual_moves(table, class_bytes, dead, usual, same, count):
    """:func:`_usual_move` of every state not known yet (threads walking only read them)."""
    for q in range(table.shape[0]):
        if usual[q] < 0:
            _usual_move(table, class_bytes, q, dead, usual, same, count)


@njit(cache = True)
def _into_child(w, q2, a1, b1, a2, b2, h1, h2, first_child, stack_node, stack_q, top, find, ends_before, seen, order, num):
    """A walk at state ``q2`` at the node ``w``: its hashes, the state if a token ends here, the node to go on from if it has children."""
    x = np.uint64(q2 + 1)
    h1[w] += a1 * x + b1
    h2[w] += a2 * x + b2
    if find and ends_before[w + 1] > ends_before[w] and not seen[q2]:
        seen[q2] = True
        order[num] = q2
        num += 1
    if first_child[w + 1] > first_child[w]:
        stack_node[top] = w
        stack_q[top] = q2
        top += 1
    return top, num


@njit(cache = True)
def _walk_one(s, a1, b1, a2, b2, table, dead, byte_class, first_child, node_byte, below, class_bytes, usual, same, count, h1, h2, d1, d2, stack_node, stack_q,
              find, ends_before, seen, order, num):
    """
    The hashes of a depth-first walk of the trie from state ``s``, through live states, added: to ``h1`` and ``h2``, or by ranges
    of nodes to ``d1`` and ``d2`` (:func:`_add_ranges`), the same numbers. Most of a walk is not walked: where every byte below a
    node leads a state to one successor (:func:`_usual_move`: a string's characters to the string, nearly all to the dead state), that
    successor holds for the whole next level of the subtree, and so on level by level as long as it holds: a level of a breadth-first
    numbered trie is a range. With ``find``, every state a token ends at (``ends_before``, by node, how many nodes before it tokens end at) is kept in ``order`` if not ``seen``. Returns
    ``(steps, num)``.
    """
    stack_node[0], stack_q[0] = 0, s
    top = 1
    steps = 0
    while top > 0:
        top -= 1
        u, q = stack_node[top], stack_q[top]
        if usual[q] < 0:
            _usual_move(table, class_bytes, q, dead, usual, same, count)
        steps += 1
        inside = True
        for k in range(4):
            if below[u, k] & ~same[q, k] != 0:
                inside = False
                break
        if not inside:
            e0, e1 = first_child[u], first_child[u + 1]
            if usual[q] == dead and e1 - e0 > MANY_CHILDREN:    # (of so many, those that live: found by their bytes)
                for k in range(4):
                    bits = ~same[q, k]
                    while bits != np.uint64(0):
                        low = bits & (~bits + np.uint64(1))
                        bits ^= low
                        b = 64 * k + _TRAILING_ZEROS[np.int64((low * _DE_BRUIJN) >> np.uint64(58))]
                        lo, hi = e0, e1
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if node_byte[mid] < b:
                                lo = mid + 1
                            else:
                                hi = mid
                        steps += 1
                        if lo < e1 and node_byte[lo] == b:
                            top, num = _into_child(lo, table[q, byte_class[b]], a1, b1, a2, b2, h1, h2, first_child, stack_node, stack_q, top, find, ends_before, seen,
                                                   order, num)
                continue
            steps += e1 - e0
            for w in range(e0, e1):
                q2 = table[q, byte_class[node_byte[w]]]
                if q2 != dead:
                    top, num = _into_child(w, q2, a1, b1, a2, b2, h1, h2, first_child, stack_node, stack_q, top, find, ends_before, seen, order, num)
            continue
        t = usual[q]
        first, last = u, u
        while t != dead:
            e0, e1 = first_child[first], first_child[last + 1]
            if e1 == e0:
                break
            first, last = e0, e1 - 1                                    # the next level: all of it at state t
            x = np.uint64(t + 1)
            y1, y2 = a1 * x + b1, a2 * x + b2
            d1[first] += y1
            d1[last + 1] -= y1
            d2[first] += y2
            d2[last + 1] -= y2
            if find and not seen[t] and ends_before[last + 1] > ends_before[first]:
                seen[t] = True
                order[num] = t
                num += 1
            steps += 1
            if usual[t] < 0:
                _usual_move(table, class_bytes, t, dead, usual, same, count)
            inside = True
            for k in range(4):
                if below[u, k] & ~same[t, k] != 0:
                    inside = False
                    break
            if not inside:                                              # no longer: the level's nodes walk on from t
                for w in range(first, last + 1):
                    if first_child[w + 1] > first_child[w]:
                        stack_node[top] = w
                        stack_q[top] = t
                        top += 1
                break
            t = usual[t]
    return steps, num


@njit(cache = True)
def _add_ranges(h, d):
    """Add to the hashes ``h`` what the differences ``d`` say."""
    total = np.uint64(0)
    for i in range(d.shape[0]):
        total += d[i]
        h[i] += total


@njit(cache = True)
def _walk(table, dead, initial, byte_class, first_child, node_byte, ends_before, r1, r2, r3, r4, below, class_bytes, usual, same, count, budget):
    """The states reached at token boundaries (breadth-first from ``initial``) and, per trie node, two hashes of its
    successors from them: from every reached state a walk (:func:`_walk_one`) -- as far as ``budget`` steps go: ``(h1, h2,
    states, walked)``, the first ``walked`` of them walked (every state a walk found, if it is all of them). The hashes
    tell apart the nodes these states tell apart, by the state they lead to, whether tokens end at them or not."""
    S = table.shape[0]
    M = first_child.shape[0] - 1
    h1 = np.zeros(M + 2, dtype = np.uint64)
    h2 = np.zeros(M + 2, dtype = np.uint64)
    d1 = np.zeros(M + 2, dtype = np.uint64)
    d2 = np.zeros(M + 2, dtype = np.uint64)
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
        done, num = _walk_one(s, r1[s], r2[s], r3[s], r4[s], table, dead, byte_class, first_child, node_byte, below, class_bytes, usual, same, count, h1, h2, d1, d2,
                              stack_node, stack_q, True, ends_before, seen, order, num)
        steps += done
    _add_ranges(h1, d1)
    _add_ranges(h2, d2)
    return h1, h2, order[:num], head


@njit(cache = True)
def _walks(table, dead, states, byte_class, first_child, node_byte, r1, r2, r3, r4, below, class_bytes, usual, same, count, h1, h2, budget):
    """:func:`_walk`'s hashes, added to ``h1`` and ``h2``, from ``states`` in turn, as far as ``budget`` steps go: how many
    were walked, one at least."""
    M = first_child.shape[0] - 1
    d1 = np.zeros(h1.shape[0], dtype = np.uint64)
    d2 = np.zeros(h1.shape[0], dtype = np.uint64)
    stack_node = np.empty(M + 1, dtype = np.int32)
    stack_q = np.empty(M + 1, dtype = np.int32)
    no_ends = np.zeros(1, dtype = np.int32)
    no_seen = np.zeros(1, dtype = np.bool_)
    no_order = np.zeros(1, dtype = np.int64)
    steps, i = 0, 0
    while i < states.shape[0] and (i == 0 or steps < budget):
        s = states[i]
        i += 1
        done, _ = _walk_one(s, r1[s], r2[s], r3[s], r4[s], table, dead, byte_class, first_child, node_byte, below, class_bytes, usual, same, count, h1, h2, d1, d2,
                            stack_node, stack_q, False, no_ends, no_seen, no_order, 0)
        steps += done
    _add_ranges(h1, d1)
    _add_ranges(h2, d2)
    return i


@njit(parallel = True, cache = True)
def _walks_parallel(table, dead, states, byte_class, first_child, node_byte, r1, r2, r3, r4, below, class_bytes, usual, same, P, slots):
    """:func:`_walks` of all ``states``, no budget, in parallel, every state's :func:`_usual_move` known: ``P`` walkers take every
    ``P``-th state, each summing into its own hashes -- ``slots`` of them, the trie's nodes and two more --, then summed."""
    S = states.shape[0]
    h1 = np.zeros((P, slots), dtype = np.uint64)
    h2 = np.zeros((P, slots), dtype = np.uint64)
    no_count = np.zeros(1, dtype = np.int32)
    no_ends = np.zeros(1, dtype = np.int32)
    no_seen = np.zeros(1, dtype = np.bool_)
    no_order = np.zeros(1, dtype = np.int64)
    for c in prange(P):
        stack_node = np.empty(slots, dtype = np.int32)
        stack_q = np.empty(slots, dtype = np.int32)
        d1 = np.zeros(slots, dtype = np.uint64)
        d2 = np.zeros(slots, dtype = np.uint64)
        for i in range(c, S, P):
            s = states[i]
            _walk_one(s, r1[s], r2[s], r3[s], r4[s], table, dead, byte_class, first_child, node_byte, below, class_bytes, usual, same, no_count, h1[c], h2[c], d1, d2,
                      stack_node, stack_q, False, no_ends, no_seen, no_order, 0)
        _add_ranges(h1[c], d1)
        _add_ranges(h2[c], d2)
    o1 = np.zeros(slots, dtype = np.uint64)
    o2 = np.zeros(slots, dtype = np.uint64)
    for c in range(P):
        for t in range(slots):
            o1[t] += h1[c, t]
            o2[t] += h2[c, t]
    return o1, o2


@njit(cache = True)
def _more_slots(h1, h2, rep, first, num, cap):
    """The places of nodes ``rep[first:num]``, each with hashes of its own, in a table of ``cap`` slots."""
    slots = np.full(cap, -1, dtype = np.int32)
    for k in range(first, num):
        n = rep[k]
        at = np.int64((h1[n] ^ (h2[n] >> np.uint64(17))) & np.uint64(cap - 1))
        while slots[at] >= 0:
            at = (at + 1) & (cap - 1)
        slots[at] = k
    return slots


@njit(cache = True)
def _quotient(h1, h2, first_child, node_byte, byte_class):
    """
    The trie with the nodes of the same hashes as one (no state walked tells them apart; the root stays alone):
    ``(gptr, gclass, gnode, into, zero)``: its edges, one per (node, class), as CSR, the edges into every node, and the
    node of those no walk reached (-1: none: nothing goes on from it). What the walks have told apart a child of it keeps: the
    nodes of a node's children on a class are one, as the same states lead to the same states.
    """
    M = first_child.shape[0] - 1
    cap = 4096
    slots = np.full(cap, -1, dtype = np.int32)
    of = np.empty(M, dtype = np.int32)
    rep = np.empty(cap // 2, dtype = np.int32)
    of[0], rep[0] = 0, 0
    G = 1
    zero = -1
    p1, p2, pg = np.uint64(0), np.uint64(0), -1
    for u in range(1, M):
        x1, x2 = h1[u], h2[u]
        if x1 == p1 and x2 == p2 and pg >= 0:                   # (nodes side by side are mostly alike)
            of[u] = pg
            continue
        at = np.int64((x1 ^ (x2 >> np.uint64(17))) & np.uint64(cap - 1))
        while True:
            g = slots[at]
            if g < 0:
                g = G
                G += 1
                slots[at] = g
                rep[g] = u
                if x1 == 0 and x2 == 0:
                    zero = g
                if 2 * G >= cap:
                    cap *= 2
                    slots = _more_slots(h1, h2, rep, 1, G, cap)
                    more = np.empty(cap // 2, dtype = np.int32)
                    more[:G] = rep[:G]
                    rep = more
                break
            if h1[rep[g]] == x1 and h2[rep[g]] == x2:
                break
            at = (at + 1) & (cap - 1)
        p1, p2, pg = x1, x2, g
        of[u] = g
    E = 0
    for u in range(M):
        if of[u] != zero:
            E += first_child[u + 1] - first_child[u]
    have = np.zeros((G, 4), dtype = np.uint64)                  # (every class is below 256: a byte's)
    src = np.empty(E, dtype = np.int32)
    gclass = np.empty(E, dtype = np.int32)
    gnode = np.empty(E, dtype = np.int32)
    num = 0
    gptr = np.zeros(G + 1, dtype = np.int64)
    for u in range(M):
        g = of[u]
        if g == zero:
            continue
        for w in range(first_child[u], first_child[u + 1]):
            c = byte_class[node_byte[w]]
            bit = np.uint64(1) << np.uint64(c & 63)
            if have[g, c >> 6] & bit == 0:
                have[g, c >> 6] |= bit
                src[num], gclass[num], gnode[num] = g, c, of[w]
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


def _walk_the_rest(table: np.ndarray, dead: int, rest: np.ndarray, byte_class: np.ndarray, trie: "_Trie", r: Sequence[np.ndarray], walking: tuple, h1: np.ndarray,
                   h2: np.ndarray) -> None:
    """
    The token classes as if walks from ``rest`` were added to the hashes ``h1`` and ``h2``, with next to no walks: of two
    hundred thousand states a few hundred tell apart all a walk from every one would (a state in a long string reads its
    tokens like the next). The nodes the hashes so far tell apart are a far smaller graph (:func:`_quotient`). Every state
    is checked on it (:func:`_check_part`), and walks only if it tells apart what the hashes do not; from then on the graph
    is that much larger. The states that have passed tell nothing apart whatever the graph comes to be: what they cannot
    tell apart stays together or is told apart by others.
    """
    first_child, node_byte = trie.first_child, trie.node_byte
    M = first_child.size - 1
    quotient = _quotient(h1, h2, first_child, node_byte, byte_class)
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
                walked = _walks(table, dead, left, byte_class, first_child, node_byte, r[0], r[1], r[2], r[3], trie.below, *walking, h1, h2, budget)
                left = left[walked:]
                budget *= 2
            else:                                                       # (states that do tell much apart: all at once)
                _usual_moves(table, walking[0], dead, *walking[1:])
                a1, a2 = _walks_parallel(table, dead, left, byte_class, first_child, node_byte, r[0], r[1], r[2], r[3], trie.below, *walking[:3],
                                         min(get_num_threads(), left.size), M + 2)
                with np.errstate(over = "ignore"):
                    h1 += a1
                    h2 += a2
                left = left[:0]
            if left.size == 0 and at >= rest.size:
                return
            quotient = _quotient(h1, h2, first_child, node_byte, byte_class)
            if left.size == 0:
                break


@njit(cache = True)
def _token_classes(h1, h2, end_nodes, token_end):
    """The tokens of the same hashes as one class, the classes numbered by their smallest token: ``(token_class, reps, hashed)``:
    int64 [V], every class's smallest token, and whether it has a hash (a class without leads nowhere from any state). The hashes
    are read at ``end_nodes``, in order, and such a node is mostly like the one before; ``token_end`` is every token's place among them."""
    E = end_nodes.shape[0]
    V = token_end.shape[0]
    of_end = np.empty(E, dtype = np.int32)
    cap = 1024
    slots = np.full(cap, -1, dtype = np.int32)
    rep_node = np.empty(cap // 2, dtype = np.int32)
    num = 0
    p1, p2, pc = np.uint64(0), np.uint64(0), -1
    for i in range(E):
        m = end_nodes[i]
        x1, x2 = h1[m], h2[m]
        if x1 == p1 and x2 == p2 and pc >= 0:
            of_end[i] = pc
            continue
        at = np.int64((x1 ^ (x2 >> np.uint64(17))) & np.uint64(cap - 1))
        while True:
            c = slots[at]
            if c < 0:
                c = num
                slots[at] = c
                rep_node[c] = m
                num += 1
                if 2 * num >= cap:
                    cap *= 2
                    slots = _more_slots(h1, h2, rep_node, 0, num, cap)
                    more = np.empty(cap // 2, dtype = np.int32)
                    more[:num] = rep_node[:num]
                    rep_node = more
                break
            if h1[rep_node[c]] == x1 and h2[rep_node[c]] == x2:
                break
            at = (at + 1) & (cap - 1)
        p1, p2, pc = x1, x2, c
        of_end[i] = c
    rank = np.full(num, -1, dtype = np.int32)
    reps = np.empty(num, dtype = np.int64)
    token_class = np.empty(V, dtype = np.int64)
    classes = 0
    for t in range(V):
        p = of_end[token_end[t]]
        c = rank[p]
        if c < 0:
            c = classes
            rank[p] = c
            reps[c] = t
            classes += 1
        token_class[t] = c
    hashed = np.empty(classes, dtype = np.bool_)
    for p in range(num):
        if rank[p] >= 0:
            hashed[rank[p]] = h1[rep_node[p]] != 0 or h2[rep_node[p]] != 0
    return token_class, reps[:classes], hashed


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


@njit(cache = True)
def _rep_trie(first_child, node_byte, byte_class, parent, rep_node):
    """The trie as far as the representative tokens go, ``rep_node`` the node of every class's (-1: none): ``(gptr, gclass,
    gnode, class_at)``, its edges as CSR, on classes, and the class ending at every node (-1: none). The root is node 0, the others
    numbered as they are found."""
    M = first_child.shape[0] - 1
    keep = np.zeros(M, dtype = np.bool_)
    keep[0] = True
    kept = np.empty(M, dtype = np.int32)
    kept[0] = 0
    G = 1
    for c in range(rep_node.shape[0]):
        m = rep_node[c]
        if m >= 0:
            while not keep[m]:
                keep[m] = True
                kept[G] = m
                G += 1
                m = parent[m]
    number = np.empty(M, dtype = np.int32)
    edges = 0
    for g in range(G):
        m = kept[g]
        number[m] = g
        for w in range(first_child[m], first_child[m + 1]):
            edges += keep[w]
    class_at = np.full(G, -1, dtype = np.int32)
    for c in range(rep_node.shape[0]):
        if rep_node[c] >= 0:
            class_at[number[rep_node[c]]] = c
    gptr = np.zeros(G + 1, dtype = np.int32)
    gclass = np.empty(edges, dtype = np.int32)
    gnode = np.empty(edges, dtype = np.int32)
    edges = 0
    for g in range(G):
        m = kept[g]
        for w in range(first_child[m], first_child[m + 1]):
            if keep[w]:
                gclass[edges], gnode[edges] = byte_class[node_byte[w]], number[w]
                edges += 1
        gptr[g + 1] = edges
    return gptr, gclass, gnode, class_at


@njit(cache = True)
def _sparse_rows(table, dead, states, remap, gptr, gclass, gnode, class_at, itself, accepting, end_class, K, C, moves, row_at, row_len, lo, hi):
    """
    Rows ``lo`` to ``hi`` of the token automaton as moves but those to the dead state, classes ascending: ``moves[row_at[i]:
    row_at[i] + row_len[i]]``, ``(class << 32) | state``, from place ``lo * C`` on, room for all. From every state a walk of
    :func:`_rep_trie` through live states; classes of empty text lead to the state itself (``itself``), and with
    ``end_class`` >= 0, of an accepting state to the state ``K + 1`` for now, the padding's.
    """
    G = gptr.shape[0] - 1
    stack_node = np.empty(G + 1, dtype = np.int32)
    stack_q = np.empty(G + 1, dtype = np.int32)
    at = lo * C
    for i in range(lo, hi):
        row_at[i] = at
        n = 0
        stack_node[0] = 0
        stack_q[0] = states[i]
        top = 1
        while top > 0:
            top -= 1
            u = stack_node[top]
            q = stack_q[top]
            for e in range(gptr[u], gptr[u + 1]):
                q2 = table[q, gclass[e]]
                if q2 == dead:
                    continue
                w = gnode[e]
                c = class_at[w]
                if c >= 0 and remap[q2] != K:
                    moves[at + n] = (np.int64(c) << 32) | remap[q2]
                    n += 1
                if gptr[w + 1] > gptr[w]:
                    stack_node[top] = w
                    stack_q[top] = q2
                    top += 1
        for j in range(itself.shape[0]):
            moves[at + n] = (itself[j] << 32) | i
            n += 1
        if end_class >= 0 and accepting[i]:
            moves[at + n] = (end_class << 32) | (K + 1)
            n += 1
        if n > 32:
            moves[at:at + n] = np.sort(moves[at:at + n])
        else:
            for a in range(at + 1, at + n):
                x = moves[a]
                b = a
                while b > at and moves[b - 1] > x:
                    moves[b] = moves[b - 1]
                    b -= 1
                moves[b] = x
        row_len[i] = n
        at += n


@njit(parallel = True, cache = True)
def _sparse_rows_parallel(table, dead, states, remap, gptr, gclass, gnode, class_at, itself, accepting, end_class, K, C, moves, row_at, row_len, R, P):
    """:func:`_sparse_rows` of the first ``R`` rows in ``P`` parts."""
    for p in prange(P):
        _sparse_rows(table, dead, states, remap, gptr, gclass, gnode, class_at, itself, accepting, end_class, K, C, moves, row_at, row_len,
                     p * R // P, (p + 1) * R // P)


@njit(cache = True)
def _sparse_order(moves, row_at, row_len, first_at, first_moves, K, C, pad):
    """:func:`~.dfa.bfs_order` of the rows of :func:`_sparse_rows` from state 0 (the dead state ``K`` where a row has its first
    gap, the padding's state ``K + 1`` being ``pad``): ``(order, place)``, the states and every state's place, -1 for the rest. Of a
    row only those of its moves are read that ``first_moves[first_at[state]]`` counts and lists, if any: the first to each state."""
    T = K + 2
    order = np.empty(T, dtype = np.int64)
    place = np.full(T, -1, dtype = np.int64)
    order[0] = 0
    place[0] = 0
    num, head = 1, 0
    mask = (np.int64(1) << 32) - 1
    while head < num:
        q = order[head]
        head += 1
        at, n = row_at[q], row_len[q]
        if place[K] < 0:                                        # (the dead state's place: until it has one)
            expected = 0
            for k in range(at, at + n):
                c = moves[k] >> 32
                if c > expected and place[K] < 0:
                    place[K] = num
                    order[num] = K
                    num += 1
                expected = c + 1
                t = moves[k] & mask
                t = pad if t == K + 1 else t
                if place[t] < 0:
                    place[t] = num
                    order[num] = t
                    num += 1
            if expected < C and place[K] < 0:
                place[K] = num
                order[num] = K
                num += 1
        elif first_at[q] >= 0:
            f = first_at[q]
            for j in range(f + 1, f + 1 + first_moves[f]):
                t = moves[at + first_moves[j]] & mask
                t = pad if t == K + 1 else t
                if place[t] < 0:
                    place[t] = num
                    order[num] = t
                    num += 1
        else:
            for k in range(at, at + n):
                t = moves[k] & mask
                t = pad if t == K + 1 else t
                if place[t] < 0:
                    place[t] = num
                    order[num] = t
                    num += 1
    return order[:num], place


@njit(cache = True)
def _sparse_table(moves, row_at, row_len, order, place, K, pad, out, lo, hi):
    """Rows ``lo`` to ``hi`` of the table of the states ``order``, numbered by their ``place``."""
    C = out.shape[1]
    gone = place[K]
    mask = (np.int64(1) << 32) - 1
    for i in range(lo, hi):
        q = order[i]
        for c in range(C):
            out[i, c] = gone
        for k in range(row_at[q], row_at[q] + row_len[q]):
            t = moves[k] & mask
            out[i, moves[k] >> 32] = place[pad if t == K + 1 else t]


@njit(parallel = True, cache = True)
def _sparse_table_parallel(moves, row_at, row_len, order, place, K, pad, out, P):
    """:func:`_sparse_table` in ``P`` parts."""
    R = order.shape[0]
    for p in prange(P):
        _sparse_table(moves, row_at, row_len, order, place, K, pad, out, p * R // P, (p + 1) * R // P)


_NO_FIRST_MOVES = (np.empty(0, dtype = np.int32), np.zeros(1, dtype = np.int32))


def _rows_by_walks(automaton: ByteAutomaton, states: np.ndarray, remap: np.ndarray, trie: "_Trie", reps: np.ndarray, kinds: np.ndarray, accept: np.ndarray,
                   end_class: int, K_full: int) -> tuple:
    """The moves of the first ``states.size`` states of a token automaton minimal already, by a walk of the representatives' trie from every state."""
    K, C = states.size, reps.size
    rep_node = np.where(kinds == 0, trie.end_nodes[trie.token_end[np.minimum(reps, trie.token_end.size - 1)]], -1).astype(np.int64)
    gptr, gclass, gnode, class_at = _rep_trie(trie.first_child, trie.node_byte, automaton.byte_class, trie.parent, rep_node)
    itself = np.nonzero(kinds == 1)[0].astype(np.int64)
    moves = np.empty(K * C + 1, dtype = np.int64)               # (room for all, where a part starts; only what is written is touched)
    row_at, row_len = np.zeros(K_full + 2, dtype = np.int64), np.zeros(K_full + 2, dtype = np.int64)
    args = (automaton._kept_table, automaton.dead, states, remap, gptr, gclass, gnode, class_at, itself, accept, end_class, K_full, C, moves, row_at, row_len)
    threads = get_num_threads()
    if K * C >= THREADS_ENTRIES and threads > 1:
        _sparse_rows_parallel(*args, K, min(threads, K))
    else:
        _sparse_rows(*args, 0, K)
    return moves, row_at, row_len


def _table_of_rows(automaton: ByteAutomaton, remap: np.ndarray, K: int, K_full: int, C: int, moves: np.ndarray, row_at: np.ndarray, row_len: np.ndarray, accept: np.ndarray,
                   end_class: int) -> tuple:
    """
    The table in canonical numbering of a token automaton minimal already, and its accepting states: what
    ``DFA._of_classes(...)._renumbered()`` makes of the dense table of every state's every class, without it: nearly all of
    that table is the dead state. From the moves of the first ``K`` of its ``K_full`` states (the others: the copies an automaton has left
    out, its :attr:`~.byte_level.ByteAutomaton.counted`, put back here), the breadth-first order over them, then the table once, as it
    will stand.
    """
    threads = get_num_threads()
    first_at, first_moves = _NO_FIRST_MOVES
    if automaton.counted:
        moves, first_at, first_moves = expanded_moves(automaton.counted, remap, K, K_full, moves, row_at, row_len, accept, threads)
    pad = K_full + 1
    if end_class >= 0:                                          # a state accepting the empty text alone is the padding's
        alone = np.nonzero(accept[:K_full] & (row_len[:K_full] == 1))[0]
        if alone.size:
            pad = int(alone[0])
        moves[-1] = (end_class << 32) | (K_full + 1)
        row_at[K_full + 1], row_len[K_full + 1] = moves.size - 1, 1
        accept[pad] = True
    if first_at is _NO_FIRST_MOVES[0]:
        first_at = np.full(K_full + 2, -1, dtype = np.int32)
    order, place = _sparse_order(moves, row_at, row_len, first_at, first_moves, K_full, C, pad)
    out = np.empty((order.size, C), dtype = np.int64)
    if K_full * C >= THREADS_ENTRIES and threads > 1:
        _sparse_table_parallel(moves, row_at, row_len, order, place, K_full, pad, out, min(threads, K_full))
    else:
        _sparse_table(moves, row_at, row_len, order, place, K_full, pad, out, 0, order.size)
    return out, accept[order]


#: an automaton of at most so many entries has its tokens' classes and rows from the behaviours of the trie's nodes, reading at most so many
#: moves for them (:mod:`.behaviours`; they read a few times its entries: past that, the walks and their checks, in threads, cost less)
BEHAVIOUR_ENTRIES = 1 << 18
BEHAVIOUR_WORK = 1 << 21
#: so many steps of walks of the trie come first; a table of so many entries stays on one thread
SERIAL_STEPS = 1 << 18
NO_BUDGET = 1 << 62
#: the table of a token automaton of at least so many entries, minimal already, is made from its moves (:func:`_table_of_rows`,
#: the dense table of its rows costing no more below it), in threads from :data:`THREADS_ENTRIES` on
SPARSE_ENTRIES = 1 << 13
THREADS_ENTRIES = 1 << 18
#: states checked on the quotient of the trie (:func:`_walk_the_rest`): at first so many, then twice as many a time,
#: on one thread below :data:`SERIAL_CHECKS`; states failing are walked, so many edges in threads at the most at a time
FIRST_CHECKS = 1 << 12
SERIAL_CHECKS = 1 << 8
WALKS_IN_THREADS = 1 << 24


def _random_words(S: int) -> list:
    """Four arrays of ``S`` random words for the hashes, the first and third odd; those of few states kept (a generator takes time)."""
    global _WORDS
    if S > _WORDS.shape[1]:
        words = np.random.default_rng(0).integers(1, 2 ** 63, size = (4, max(S, _KEPT_WORDS)), dtype = np.uint64)
        words[0] |= np.uint64(1)
        words[2] |= np.uint64(1)
        if S > _KEPT_WORDS:
            return list(words)
        _WORDS = words
    return list(_WORDS[:, :S])


_KEPT_WORDS = 1 << 12
_WORDS = np.empty((4, 0), dtype = np.uint64)


@few_threads
def to_tokens(automaton: ByteAutomaton, vocabulary: Vocabulary, eos_padding: bool) -> DFA:
    """
    The minimal :class:`~pyjuice.constraints.DFA` over ``vocabulary``'s ids of the token sequences whose text the
    byte automaton accepts (see the module docstring) -- with ``eos_padding``, followed by any number of
    end-of-sequence ids and nothing else.
    """
    V, S = vocabulary.size, automaton.num_states
    trie = _trie(vocabulary)
    table, byte_class = automaton._kept_table, automaton.byte_class
    dead_byte, start = automaton.dead, automaton.initial
    complete = automaton.minimal and start != dead_byte and _byte_complete(trie)
    if automaton.counted and not complete:
        raise NotCounted
    eos = np.asarray(vocabulary.eos, dtype = np.int64)
    padded = eos_padding and eos.size > 0
    token_end = trie.token_ends(V, vocabulary.eos if eos_padding else ())
    if complete and S * table.shape[1] <= BEHAVIOUR_ENTRIES:
        T, behaviour_of, at, source, target = behaviours_of(table, dead_byte, byte_class, trie.parent, trie.node_byte, BEHAVIOUR_WORK)
        if T >= 0:
            token_class, reps, behaviour = classes_of_behaviours(behaviour_of, T, trie.end_nodes, token_end)
            states = np.arange(S, dtype = np.int64)
            states[start], states[0] = 0, start
            states = states[states != dead_byte]
            K, C = states.size, reps.size
            K_full = K + states_put_back(automaton.counted) if automaton.counted else K
            remap = np.full(S, K_full, dtype = np.int64)
            remap[states] = np.arange(K)
            accept = np.zeros(K_full + 2, dtype = bool)
            accept[:K] = automaton.accept[states]
            end_class = int(token_class[eos[0]]) if padded else -1
            rows = rows_of_behaviours(behaviour, T, at, source, target, remap, accept, end_class, K, K_full)
            out, kept = _table_of_rows(automaton, remap, K, K_full, C, *rows, accept, end_class)
            return DFA._of_classes(V, torch.from_numpy(token_class), torch.from_numpy(out), 0, torch.from_numpy(kept))
    r = _random_words(S)
    M = trie.first_child.size - 1
    walking = (_bytes_of_classes(byte_class, table.shape[1]), np.full(S, -1, dtype = np.int32), np.zeros((S, 4), dtype = np.uint64), np.zeros(S, dtype = np.int32))
    h1, h2, states, walked = _walk(table, dead_byte, start, byte_class, trie.first_child, trie.node_byte, trie.ends_before, r[0], r[1], r[2], r[3], trie.below, *walking,
                                   SERIAL_STEPS if complete else NO_BUDGET)
    if walked < states.size:        # the states reached at token boundaries: every one a byte text reaches
        left = np.ones(S, dtype = bool)
        left[states[:walked]] = False
        states = bfs_order(table, start)
        states = states[states != dead_byte]
        _walk_the_rest(table, dead_byte, states[left[states]], byte_class, trie, r, walking, h1, h2)
    offsets, has_text = vocabulary._offsets, vocabulary._has_text
    if trie.empty.size:                                                 # empty text: every state to itself
        with np.errstate(over = "ignore"):
            x = (states + 1).astype(np.uint64)
            h1[M] = np.sum(r[0][states] * x + r[1][states], dtype = np.uint64)
            h2[M] = np.sum(r[2][states] * x + r[3][states], dtype = np.uint64)
    if padded:                                                          # a class of their own
        h1[M + 1] = h2[M + 1] = np.uint64(2 ** 64 - 1)
    token_class, reps, hashed = _token_classes(h1, h2, trie.end_nodes, token_end)
    kinds = np.where(hashed & has_text[reps], np.where(offsets[reps + 1] == offsets[reps], 1, 0), 2).astype(np.int64)
    K, C = states.size, reps.size
    K_full = K + states_put_back(automaton.counted) if automaton.counted else K
    remap = np.full(S, K_full, dtype = np.int64)                        # unreached (the byte dead state among them)
    remap[states] = np.arange(K)
    if complete and (automaton.counted or K * C >= SPARSE_ENTRIES):
        accept = np.zeros(K_full + 2, dtype = bool)
        accept[:K] = automaton.accept[states]
        end_class = int(token_class[eos[0]]) if padded else -1
        rows = _rows_by_walks(automaton, states, remap, trie, reps, kinds, accept, end_class, K_full)
        out, kept = _table_of_rows(automaton, remap, K, K_full, C, *rows, accept, end_class)
        return DFA._of_classes(V, torch.from_numpy(token_class), torch.from_numpy(out), 0, torch.from_numpy(kept))
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
