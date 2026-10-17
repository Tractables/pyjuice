"""
Numba kernels of the DFA algorithms in :mod:`pyjuice.constraints.language.dfa`: the breadth-first order of the
reachable states (:func:`bfs_order`), Hopcroft's minimization (:func:`hopcroft`), and the breadth-first product of
several DFAs over their joint classes (:func:`product_bfs`). Tables are int64 ``[states, classes]``, complete.
"""

import numpy as np
from numba import njit


@njit(cache = True)
def bfs_order(delta, initial):
    """The states reachable from ``initial``, in breadth-first order, every state's successors by class in increasing
    order: int64 [reachable]."""
    K, C = delta.shape
    seen = np.zeros(K, dtype = np.bool_)
    order = np.empty(K, dtype = np.int64)
    order[0] = initial
    seen[initial] = True
    num, head = 1, 0
    while head < num:
        q = order[head]
        head += 1
        for c in range(C):
            t = delta[q, c]
            if not seen[t]:
                seen[t] = True
                order[num] = t
                num += 1
    return order[:num]


@njit(cache = True)
def hopcroft(delta, accept):
    """
    The coarsest partition of the states that keeps accepting and rejecting states apart and is respected by every
    class's transitions -- Hopcroft's algorithm (every state reachable, the table complete): the block of every state
    (int64 [K], blocks numbered in no particular order) and the number of blocks.
    """
    K, C = delta.shape
    # the predecessors of every (class, state), as CSR
    ptr = np.zeros(C * K + 1, dtype = np.int64)
    for q in range(K):
        for c in range(C):
            ptr[c * K + delta[q, c] + 1] += 1
    for i in range(C * K):
        ptr[i + 1] += ptr[i]
    pred = np.empty(K * C, dtype = np.int64)
    fill = ptr[:-1].copy()
    for q in range(K):
        for c in range(C):
            t = c * K + delta[q, c]
            pred[fill[t]] = q
            fill[t] += 1
    # blocks as contiguous ranges of `elems`; `loc` the position of every state in it
    elems = np.empty(K, dtype = np.int64)
    loc = np.empty(K, dtype = np.int64)
    block = np.empty(K, dtype = np.int64)
    start = np.zeros(K + 1, dtype = np.int64)
    size = np.zeros(K + 1, dtype = np.int64)
    num_accept = 0
    for q in range(K):
        if accept[q]:
            num_accept += 1
    num_blocks = 0
    first = np.array([0, num_accept], dtype = np.int64)
    fill2 = first.copy()
    for q in range(K):
        side = 0 if accept[q] else 1
        elems[fill2[side]] = q
        loc[q] = fill2[side]
        fill2[side] += 1
    for side in range(2):
        n = num_accept if side == 0 else K - num_accept
        if n > 0:
            start[num_blocks], size[num_blocks] = first[side], n
            for i in range(first[side], first[side] + n):
                block[elems[i]] = num_blocks
            num_blocks += 1
    # splitters (block, class): every class with the smaller initial block
    wl_block = np.empty(K * C + C, dtype = np.int64)
    wl_class = np.empty(K * C + C, dtype = np.int64)
    queued = np.zeros(K * C + C, dtype = np.bool_)
    top = 0
    smaller = 0 if num_blocks == 1 or size[0] <= size[1] else 1
    for c in range(C):
        wl_block[top], wl_class[top] = smaller, c
        queued[smaller * C + c] = True
        top += 1
    marked = np.zeros(K + 1, dtype = np.int64)                 # per block: its marked states sit at its front
    touched = np.empty(K + 1, dtype = np.int64)
    splitter = np.empty(K, dtype = np.int64)
    while top > 0:
        top -= 1
        B, c = wl_block[top], wl_class[top]
        queued[B * C + c] = False
        n = size[B]
        for i in range(n):                                       # (B may itself split below: its states first)
            splitter[i] = elems[start[B] + i]
        num_touched = 0
        for i in range(n):
            t = splitter[i]
            for k in range(ptr[c * K + t], ptr[c * K + t + 1]):
                p = pred[k]
                b = block[p]
                front = start[b] + marked[b]
                pos = loc[p]
                if pos >= front:                                 # not marked yet: move it to the front
                    if marked[b] == 0:
                        touched[num_touched] = b
                        num_touched += 1
                    other = elems[front]
                    elems[front], loc[p] = p, front
                    elems[pos], loc[other] = other, pos
                    marked[b] += 1
        for j in range(num_touched):
            b = touched[j]
            m = marked[b]
            marked[b] = 0
            if m == size[b]:
                continue
            new = num_blocks                                     # the marked part becomes a block of its own
            num_blocks += 1
            start[new], size[new] = start[b], m
            start[b] += m
            size[b] -= m
            for i in range(start[new], start[new] + m):
                block[elems[i]] = new
            for cc in range(C):
                if queued[b * C + cc]:
                    s = new
                else:
                    s = new if size[new] <= size[b] else b
                wl_block[top], wl_class[top] = s, cc
                queued[s * C + cc] = True
                top += 1
    return block, num_blocks


@njit(cache = True)
def product_bfs(deltas, offsets, num_states, child_class, initial):
    """
    The product of several DFAs over their joint classes, its states the tuples reachable from the initial one, in
    breadth-first order: ``(rows, keys)`` -- int64 [states, joint classes] the successors, int64 [states] every state's
    tuple as a mixed-radix number (the components' state counts ``num_states`` the radices; their product must fit).
    ``deltas`` the components' flattened tables back to back, ``offsets`` where each starts; ``child_class`` int64
    [joint classes, components] the class every joint class has in every component.
    """
    k = num_states.shape[0]
    Cj = child_class.shape[0]
    key = 0
    for i in range(k):
        key = key * num_states[i] + initial[i]
    ids = dict()
    ids[key] = 0
    cap = 1024
    keys = np.empty(cap, dtype = np.int64)
    rows = np.empty((cap, Cj), dtype = np.int64)
    keys[0] = key
    num, head = 1, 0
    cur = np.empty(k, dtype = np.int64)
    while head < num:
        key = keys[head]
        for i in range(k - 1, -1, -1):
            cur[i] = key % num_states[i]
            key //= num_states[i]
        for c in range(Cj):
            nk = 0
            for i in range(k):
                C_i = (offsets[i + 1] - offsets[i]) // num_states[i]
                nk = nk * num_states[i] + deltas[offsets[i] + cur[i] * C_i + child_class[c, i]]
            j = ids.get(nk, -1)
            if j < 0:
                if num == cap:                                   # grow
                    cap *= 2
                    bigger = np.empty(cap, dtype = np.int64)
                    bigger[:num] = keys[:num]
                    keys = bigger
                    more = np.empty((cap, Cj), dtype = np.int64)
                    more[:num] = rows[:num]
                    rows = more
                j = num
                ids[nk] = j
                keys[num] = nk
                num += 1
            rows[head, c] = j
        head += 1
    return rows[:num], keys[:num]
