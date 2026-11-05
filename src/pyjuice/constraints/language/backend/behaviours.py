"""
What every token does to an automaton's states, by what its prefixes do. A node of the vocabulary's trie stands for a text, and that
text does one thing to the states: it leads each to one state, or nowhere. Tokens are one class exactly if they do the same, and an automaton's
tokens do few different things: 63 to the 25 states of ``.*(cat|dog|park).*``, among 98,024 nodes. A node's behaviour is its parent's and a
byte's class, so that once the two have been met, a node is one lookup; and only then does a behaviour have to be worked out, from the live
moves of its parent's (:func:`behaviours_of`). Exact: behaviours are compared whole. From them, the rows of the token automaton
come without a walk (:func:`rows_of_behaviours`).
"""

import numpy as np
from numba import njit


@njit(cache = True)
def _known_steps(parent, node_byte, byte_class, step, behaviour_of, w):
    """The behaviours of the nodes from ``w`` on (numbered breadth-first: a parent before its children), as far as ``step`` knows them:
    the first node it does not, or the number of nodes."""
    M = parent.shape[0]
    while w < M:
        n = step[behaviour_of[parent[w]], byte_class[node_byte[w]]]
        if n < 0:
            return w
        behaviour_of[w] = n
        w += 1
    return w


@njit(cache = True)
def _hash_of(s, t, h):
    return (h ^ (np.uint64(s) * np.uint64(0x100000001B3) + np.uint64(t))) * np.uint64(0x9E3779B97F4A7C15)


@njit(cache = True)
def _more_slots(keys, T, cap):
    """The behaviours 1 to ``T - 1`` in a table of ``cap`` slots (0: none there)."""
    slots = np.zeros(cap, dtype = np.int32)
    for g in range(1, T):
        at = np.int64((keys[g] ^ (keys[g] >> np.uint64(29))) & np.uint64(cap - 1))
        while slots[at] != 0:
            at = (at + 1) & (cap - 1)
        slots[at] = g
    return slots


@njit(cache = True)
def behaviours_of(table, dead, byte_class, parent, node_byte, most_work):
    """
    Every node's behaviour on the states of a complete ``table``: ``(T, behaviour_of, at, source, target)``. There are ``T``; behaviour 0
    leads nowhere from any state, behaviour 1 is the root's, every live state to itself; behaviour ``b`` leads ``source[e]`` to
    ``target[e]``, ``e`` from ``at[b]`` to ``at[b + 1]``, sources ascending, those it leads nowhere left out. ``T`` is -1 if working them out
    takes more than ``most_work`` moves read.
    """
    S, C = table.shape
    M = parent.shape[0]
    behaviour_of = np.empty(M, dtype = np.int32)
    behaviour_of[0] = 1
    most = 256
    step = np.full((most, C), -1, dtype = np.int32)
    step[0, :] = 0
    at = np.empty(most + 1, dtype = np.int64)
    keys = np.empty(most, dtype = np.uint64)
    room = max(4 * S, 4096)
    source = np.empty(room, dtype = np.int32)
    target = np.empty(room, dtype = np.int32)
    top = 0
    h = np.uint64(0)
    for s in range(S):
        if s != dead:
            source[top], target[top] = s, s
            top += 1
            h = _hash_of(s, s, h)
    at[0], at[1], at[2] = 0, 0, top
    keys[1] = h
    T = 2
    cap = 1024
    slots = _more_slots(keys, T, cap)
    work = 0
    w = 1
    while True:
        w = _known_steps(parent, node_byte, byte_class, step, behaviour_of, w)
        if w == M:
            return T, behaviour_of, at[:T + 1], source[:top], target[:top]
        b = behaviour_of[parent[w]]
        c = byte_class[node_byte[w]]
        first, last = at[b], at[b + 1]
        work += last - first
        if work > most_work:
            return -1, behaviour_of, at[:T + 1], source[:top], target[:top]
        if top + last - first > room:
            room = 2 * (top + last - first)
            more_s, more_t = np.empty(room, dtype = np.int32), np.empty(room, dtype = np.int32)
            more_s[:top], more_t[:top] = source[:top], target[:top]
            source, target = more_s, more_t
        made = top
        h = np.uint64(0)
        for e in range(first, last):
            t = table[target[e], c]
            if t != dead:
                source[top], target[top] = source[e], t
                top += 1
                h = _hash_of(source[e], t, h)
        size = top - made
        n = 0
        if size > 0:
            where = np.int64((h ^ (h >> np.uint64(29))) & np.uint64(cap - 1))
            n = -1
            while True:
                g = slots[where]
                if g == 0:
                    break
                if keys[g] == h and at[g + 1] - at[g] == size:
                    alike = True
                    o = at[g]
                    for e in range(size):
                        if source[o + e] != source[made + e] or target[o + e] != target[made + e]:
                            alike = False
                            break
                    if alike:
                        n = g
                        break
                where = (where + 1) & (cap - 1)
            if n > 0:
                top = made                                      # (met before)
            else:
                if T == most:
                    most *= 2
                    more_step = np.full((most, C), -1, dtype = np.int32)
                    more_step[:T] = step[:T]
                    step = more_step
                    more_at = np.empty(most + 1, dtype = np.int64)
                    more_at[:T + 1] = at[:T + 1]
                    at = more_at
                    more_keys = np.empty(most, dtype = np.uint64)
                    more_keys[:T] = keys[:T]
                    keys = more_keys
                n = T
                T += 1
                slots[where] = n
                keys[n] = h
                at[n + 1] = top
                if 2 * T >= cap:
                    cap *= 2
                    slots = _more_slots(keys, T, cap)
        step[b, c] = n


@njit(cache = True)
def classes_of_behaviours(behaviour_of, T, end_nodes, token_end):
    """
    The tokens of one behaviour as one class, the classes numbered by their smallest token: ``(token_class, reps, behaviour)``: int64 [V],
    every class's smallest token and its behaviour. ``end_nodes`` and ``token_end`` as :func:`~.token_lift._ends` makes them: tokens without
    text at its first place lead nowhere, tokens of empty text at the last place but one do nothing, and the ends of sequence at the last
    are a class of their own (behaviour ``T``).
    """
    E = end_nodes.shape[0]
    V = token_end.shape[0]
    of_end = np.empty(E, dtype = np.int32)
    of_end[0] = 0
    for i in range(1, E - 2):
        of_end[i] = behaviour_of[end_nodes[i]]
    of_end[E - 2], of_end[E - 1] = 1, T
    rank = np.full(T + 1, -1, dtype = np.int32)
    reps = np.empty(T + 1, dtype = np.int64)
    behaviour = np.empty(T + 1, dtype = np.int64)
    token_class = np.empty(V, dtype = np.int64)
    classes = 0
    for t in range(V):
        b = of_end[token_end[t]]
        c = rank[b]
        if c < 0:
            c = classes
            rank[b] = c
            reps[c], behaviour[c] = t, b
            classes += 1
        token_class[t] = c
    return token_class, reps[:classes], behaviour[:classes]


@njit(cache = True)
def rows_of_behaviours(behaviour, T, at, source, target, remap, accept, end_class, K, K_full):
    """
    The token automaton's moves but those to the dead state, per state, classes ascending: ``(moves, row_at, row_len)``, as
    :func:`~.token_lift._sparse_rows` makes them, with room for one row more; a class that ends a sequence leads an accepting state to the
    state ``K_full + 1`` for now, the padding's. ``remap``: the number among the ``K`` of every state of the table; ``accept`` by those.
    """
    row_len = np.zeros(K_full + 2, dtype = np.int64)
    C = behaviour.shape[0]
    for c in range(C):
        b = behaviour[c]
        if 0 < b < T:
            for e in range(at[b], at[b + 1]):
                row_len[remap[source[e]]] += 1
    if end_class >= 0:
        for i in range(K):
            row_len[i] += accept[i]
    row_at = np.zeros(K_full + 2, dtype = np.int64)
    total = 0
    for i in range(K):
        row_at[i] = total
        total += row_len[i]
    moves = np.empty(total + 1, dtype = np.int64)
    fill = row_at[:K].copy()
    for c in range(C):
        b = behaviour[c]
        tag = np.int64(c) << 32
        if 0 < b < T:
            for e in range(at[b], at[b + 1]):
                i = remap[source[e]]
                moves[fill[i]] = tag | remap[target[e]]
                fill[i] += 1
        elif c == end_class:
            for i in range(K):
                if accept[i]:
                    moves[fill[i]] = tag | (K_full + 1)
                    fill[i] += 1
    return moves, row_at, row_len
