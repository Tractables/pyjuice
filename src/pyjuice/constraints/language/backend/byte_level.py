"""
Byte-level automata, and the token automata they induce through a vocabulary: the layer text-level constraints
(:class:`~pyjuice.constraints.Regex`, ...) compile through.

A :class:`ByteAutomaton` is a complete DFA over the 256 byte values whose language is a set of texts, byte for byte
(valid UTF-8 when it comes from a character automaton), with one dead state.
:meth:`ByteAutomaton.from_char_automaton` builds one from an automaton over characters, every character set as code
point ranges and every range as sequences of UTF-8 byte ranges (RE2's split), then subset construction.

:func:`to_tokens` turns it into a :class:`~pyjuice.constraints.DFA` over a
:class:`~pyjuice.constraints.Vocabulary`'s ids: a token sequence is accepted when its text -- the tokens' bytes,
concatenated -- is, whichever tokens spell it (tokens that split a character included); under the end-of-sequence
padding convention the text may be followed by end-of-sequence ids, and only by them. From every state reached at a
token boundary, in breadth-first order from the initial state, a depth-first walk of the vocabulary's byte trie stops
at the dead state, so the work is the number of live (state, trie node) pairs rather than states times vocabulary.
Tokens leading to the same state from every reached state share a class: two independent 64-bit random linear hashes
of every token's successors, summed during the walks, then one representative token per class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from numba import njit

from ..dfa import DFA
from ..vocabulary import Vocabulary

MAX_CODE_POINT = 0x10FFFF


# -------------------------------------------------------------------------------------------------
# Byte automata
# -------------------------------------------------------------------------------------------------

@dataclass(frozen = True)
class ByteAutomaton:
    """
    A complete DFA over bytes.

    :ivar delta: int32 [num_states, 256], the successor of every (state, byte)
    :ivar initial: the initial state
    :ivar accept: bool [num_states]
    :ivar dead: the one state no accepting state is reachable from (it loops on every byte)
    """

    delta: np.ndarray
    initial: int
    accept: np.ndarray
    dead: int

    @property
    def num_states(self) -> int:
        return self.delta.shape[0]

    def accepts(self, data: bytes) -> bool:
        q = self.initial
        for b in data:
            q = int(self.delta[q, b])
        return bool(self.accept[q])

    @classmethod
    def from_char_automaton(cls, states: Iterable, initial, finals: Iterable,
                            transitions: Mapping[object, Mapping[object, object]],
                            symbol_ranges: Mapping[object, Sequence[Tuple[int, int]]]) -> "ByteAutomaton":
        """
        From an automaton over characters: ``transitions[q][symbol]`` the successor of ``q`` on any character of
        ``symbol`` (missing: rejection), ``symbol_ranges[symbol]`` the symbol's characters as inclusive code point
        ranges. Surrogates are no characters; every text accepted is valid UTF-8.
        """
        states = list(states)
        sid = {q: i for i, q in enumerate(states)}
        # the byte NFA: nodes 0..len(states)-1 are the character states, chains of the UTF-8 sequences get fresh ids
        edges: List[List[Tuple[int, int, int]]] = [[] for _ in states]
        sequences = {}
        for q, outs in transitions.items():
            for symbol, q2 in outs.items():
                for lo, hi in symbol_ranges.get(symbol, ()):
                    if (lo, hi) not in sequences:
                        sequences[lo, hi] = utf8_sequences(lo, hi)
                    for seq in sequences[lo, hi]:
                        cur = sid[q]
                        for k, (a, b) in enumerate(seq):
                            if k == len(seq) - 1:
                                nxt = sid[q2]
                            else:
                                edges.append([])
                                nxt = len(edges) - 1
                            edges[cur].append((a, b, nxt))
                            cur = nxt
        # byte classes: the bytes no range boundary separates
        cuts = np.zeros(257, dtype = bool)
        cuts[0] = True
        for es in edges:
            for a, b, _ in es:
                cuts[a] = True
                cuts[b + 1] = True
        byte_class = np.cumsum(cuts[:256]) - 1
        num_classes = int(byte_class[-1]) + 1
        node_succ = []
        for es in edges:
            m: Dict[int, set] = {}
            for a, b, nxt in es:
                for c in range(byte_class[a], byte_class[b] + 1):
                    m.setdefault(c, set()).add(nxt)
            node_succ.append(m)
        accepting = {sid[q] for q in finals}
        start = frozenset([sid[initial]])
        ids, order, rows = {start: 0}, [start], []
        i = 0
        while i < len(order):
            S = order[i]
            i += 1
            row = []
            for c in range(num_classes):
                T = set()
                for u in S:
                    T |= node_succ[u].get(c, set())
                T = frozenset(T)
                if T not in ids:
                    ids[T] = len(order)
                    order.append(T)
                row.append(ids[T])
            rows.append(row)
        delta = np.asarray(rows, dtype = np.int32)[:, byte_class]
        accept = np.array([bool(S & accepting) for S in order])
        return _trimmed(delta, 0, accept)


def _trimmed(delta: np.ndarray, initial: int, accept: np.ndarray) -> ByteAutomaton:
    """Every state that cannot reach acceptance merged into one dead state (the last)."""
    live = accept.copy()
    while True:
        nxt = live | live[delta].any(axis = 1)
        if (nxt == live).all():
            break
        live = nxt
    keep = np.nonzero(live)[0]
    dead = keep.size
    remap = np.full(delta.shape[0], dead, dtype = np.int32)
    remap[keep] = np.arange(keep.size, dtype = np.int32)
    out = np.full((keep.size + 1, 256), dead, dtype = np.int32)
    out[:keep.size] = remap[delta[keep]]
    acc = np.zeros(keep.size + 1, dtype = bool)
    acc[:keep.size] = accept[keep]
    return ByteAutomaton(out, int(remap[initial]), acc, dead)


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


def code_point_ranges(chars: Iterable[str]) -> List[Tuple[int, int]]:
    """Characters as maximal inclusive code point ranges, ascending."""
    out: List[List[int]] = []
    for c in sorted({ord(ch) for ch in chars}):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [(a, b) for a, b in out]


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

class _Trie:
    """The byte trie of a vocabulary's tokens with nonempty text: children as CSR (sorted by byte), the tokens ending
    at every node as CSR."""

    def __init__(self, vocabulary: Vocabulary):
        buffer, offsets, has_text = vocabulary._buffer, vocabulary._offsets, vocabulary._has_text
        children: List[Dict[int, int]] = [{}]
        ends: List[List[int]] = [[]]
        for v in np.nonzero(has_text & (offsets[1:] > offsets[:-1]))[0].tolist():
            u = 0
            for b in buffer[offsets[v]:offsets[v + 1]]:
                w = children[u].get(b)
                if w is None:
                    children.append({})
                    ends.append([])
                    w = children[u][b] = len(children) - 1
                u = w
            ends[u].append(v)
        self.child_ptr = np.zeros(len(children) + 1, dtype = np.int64)
        np.cumsum([len(c) for c in children], out = self.child_ptr[1:])
        self.child_byte = np.empty(int(self.child_ptr[-1]), dtype = np.int64)
        self.child_node = np.empty(int(self.child_ptr[-1]), dtype = np.int64)
        for u, c in enumerate(children):
            for k, (b, w) in enumerate(sorted(c.items())):
                self.child_byte[self.child_ptr[u] + k] = b
                self.child_node[self.child_ptr[u] + k] = w
        self.end_ptr = np.zeros(len(ends) + 1, dtype = np.int64)
        np.cumsum([len(e) for e in ends], out = self.end_ptr[1:])
        self.end_ids = np.array([v for e in ends for v in e], dtype = np.int64)
        self.data = np.frombuffer(buffer, dtype = np.uint8).astype(np.int64)


def _trie(vocabulary: Vocabulary) -> _Trie:
    """The vocabulary's byte trie, built on first use and kept with it."""
    if vocabulary._byte_trie is None:
        vocabulary._byte_trie = _Trie(vocabulary)
    return vocabulary._byte_trie


@njit(cache = True)
def _walk(delta, dead, initial, child_ptr, child_byte, child_node, end_ptr, end_ids, V, r1, r2, r3, r4):
    """The states reached at token boundaries (breadth-first from ``initial``) and, per token, two hashes of its
    successors from them: from every reached state a depth-first walk of the trie through live states."""
    S = delta.shape[0]
    h1 = np.zeros(V, dtype = np.uint64)
    h2 = np.zeros(V, dtype = np.uint64)
    order = np.empty(S, dtype = np.int64)
    seen = np.zeros(S, dtype = np.bool_)
    order[0] = initial
    seen[initial] = True
    num, head = 1, 0
    N = child_ptr.shape[0] - 1
    stack_node = np.empty(N + 1, dtype = np.int64)
    stack_q = np.empty(N + 1, dtype = np.int64)
    while head < num:
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
            for e in range(child_ptr[u], child_ptr[u + 1]):
                q2 = delta[q, child_byte[e]]
                if q2 == dead:
                    continue
                w = child_node[e]
                if end_ptr[w] < end_ptr[w + 1]:                        # a token ends here: a boundary state
                    x = np.uint64(q2 + 1)
                    for k in range(end_ptr[w], end_ptr[w + 1]):
                        t = end_ids[k]
                        h1[t] += a1 * x + b1
                        h2[t] += a2 * x + b2
                    if not seen[q2]:
                        seen[q2] = True
                        order[num] = q2
                        num += 1
                stack_node[top] = w
                stack_q[top] = q2
                top += 1
    return h1, h2, order[:num]


@njit(cache = True)
def _columns(delta, dead, states, reps, kinds, offsets, data):
    """[len(states), len(reps)]: the successor of every state on every representative token -- kind 0 its bytes,
    1 empty text (the state itself), 2 no text (the dead state)."""
    R, C = states.shape[0], reps.shape[0]
    out = np.empty((R, C), dtype = np.int64)
    for c in range(C):
        t, kind = reps[c], kinds[c]
        for i in range(R):
            q = states[i]
            if kind == 2:
                q = dead
            elif kind == 0:
                for k in range(offsets[t], offsets[t + 1]):
                    q = delta[q, data[k]]
                    if q == dead:
                        break
            out[i, c] = q
    return out


def to_tokens(automaton: ByteAutomaton, vocabulary: Vocabulary, eos_padding: bool) -> DFA:
    """
    The minimal :class:`~pyjuice.constraints.DFA` over ``vocabulary``'s ids of the token sequences whose text the
    byte automaton accepts (see the module docstring) -- with ``eos_padding``, followed by any number of
    end-of-sequence ids and nothing else.
    """
    V, S = vocabulary.size, automaton.num_states
    trie = _trie(vocabulary)
    rng = np.random.default_rng(0)
    r = [rng.integers(1, 2 ** 63, size = S, dtype = np.uint64) for _ in range(4)]
    r[0] |= np.uint64(1)
    r[2] |= np.uint64(1)
    delta = automaton.delta.astype(np.int64)
    h1, h2, states = _walk(delta, automaton.dead, automaton.initial, trie.child_ptr, trie.child_byte,
                           trie.child_node, trie.end_ptr, trie.end_ids, V, r[0], r[1], r[2], r[3])
    offsets, has_text = vocabulary._offsets, vocabulary._has_text
    empty = np.nonzero(has_text & (offsets[1:] == offsets[:-1]))[0]
    if empty.size:                                                      # empty text: every state to itself
        with np.errstate(over = "ignore"):
            x = (states + 1).astype(np.uint64)
            h1[empty] = np.sum(r[0][states] * x + r[1][states], dtype = np.uint64)
            h2[empty] = np.sum(r[2][states] * x + r[3][states], dtype = np.uint64)
    eos = np.asarray(vocabulary.eos, dtype = np.int64)
    key = np.stack([h1, h2], axis = 1)
    if eos_padding and eos.size:
        key[eos] = np.uint64(2 ** 64 - 1)                               # a class of their own
    _, reps, token_class = np.unique(key, axis = 0, return_index = True, return_inverse = True)
    token_class = token_class.reshape(-1)
    kinds = np.where(~has_text[reps], 2, np.where(offsets[reps + 1] == offsets[reps], 1, 0)).astype(np.int64)
    rows = _columns(delta, automaton.dead, states, reps.astype(np.int64), kinds, offsets, trie.data)
    K = states.size
    remap = np.full(S, K, dtype = np.int64)                             # unreached (the byte dead state among them)
    remap[states] = np.arange(K)
    dead, pad = K, K + 1
    table = np.full((K + 2, reps.size), dead, dtype = np.int64)
    table[:K] = remap[rows]
    accept = np.zeros(K + 2, dtype = bool)
    accept[:K] = automaton.accept[states]
    if eos_padding and eos.size:
        ce = int(token_class[eos[0]])
        table[:, ce] = dead
        table[np.nonzero(accept)[0], ce] = pad
        table[pad, ce] = pad
        accept[pad] = True
    return DFA(V, torch.from_numpy(token_class.astype(np.int64)), torch.from_numpy(table), 0,
               torch.from_numpy(accept)).minimize()


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
