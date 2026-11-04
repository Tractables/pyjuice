"""
Numba kernels of the DFA algorithms in :mod:`pyjuice.constraints.language.dfa`: the breadth-first order of the
reachable states (:func:`bfs_order`), Hopcroft's minimization over moves (:func:`hopcroft`, :func:`minimal_moves`), the
breadth-first product of several DFAs over their joint classes (:func:`product_bfs`), and the columns of a table grouped
by equal content (:func:`_column_classes`). Tables are int64 ``[states, classes]``, complete.
"""

import numpy as np
from numba import get_num_threads, njit, prange


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
def _renumbered_rows(delta, order, remap, out, lo, hi):
    C = delta.shape[1]
    for i in range(lo, hi):
        q = order[i]
        for c in range(C):
            out[i, c] = remap[delta[q, c]]


@njit(parallel = True, cache = True)
def _renumbered_parallel(delta, order, remap, out, P):
    R = order.shape[0]
    for p in prange(P):
        _renumbered_rows(delta, order, remap, out, p * R // P, (p + 1) * R // P)


#: a table of at least this many entries is renumbered in threads (waking a team is milliseconds on a busy machine)
PARALLEL_ENTRIES = 1 << 24


def renumbered(delta: np.ndarray, order: np.ndarray) -> np.ndarray:
    """The table of the states ``order`` (every successor among them), numbered by their place there."""
    remap = np.empty(delta.shape[0], dtype = np.int64)
    remap[order] = np.arange(order.size)
    out = np.empty((order.size, delta.shape[1]), dtype = np.int64)
    if out.size >= PARALLEL_ENTRIES and get_num_threads() > 1:
        _renumbered_parallel(delta, order, remap, out, get_num_threads())
    else:
        _renumbered_rows(delta, order, remap, out, 0, order.size)
    return out


@njit(cache = True)
def moves_of_table(delta, accept):
    """A complete table's moves as CSR per state, classes ascending: ``(ptr, cls, dst)``, without those into a state that
    accepts nothing and only loops -- nearly all of a table's, as a rule."""
    K, C = delta.shape
    sink = np.zeros(K, dtype = np.bool_)
    for q in range(K):
        if not accept[q]:
            loops = True
            for c in range(C):
                if delta[q, c] != q:
                    loops = False
                    break
            sink[q] = loops
    ptr = np.zeros(K + 1, dtype = np.int64)
    for q in range(K):
        n = 0
        for c in range(C):
            if not sink[delta[q, c]]:
                n += 1
        ptr[q + 1] = ptr[q] + n
    cls = np.empty(ptr[K], dtype = np.int32)
    dst = np.empty(ptr[K], dtype = np.int32)
    at = 0
    for q in range(K):
        for c in range(C):
            t = delta[q, c]
            if not sink[t]:
                cls[at], dst[at] = c, t
                at += 1
    return ptr, cls, dst


@njit(cache = True)
def _coarsest(into_ptr, into_class, into_src, accept, C):
    """
    The coarsest partition of the states that keeps accepting and rejecting states apart and is respected by every
    class's moves: Hopcroft's algorithm over moves, not over states times classes. ``into_class`` / ``into_src`` from
    ``into_ptr[t]`` to ``into_ptr[t + 1]`` are the moves into state ``t``. A state without a move on a class goes to a
    state outside the partition, of which no block needs splitting by: a partition respected by the moves into every
    other block is respected by those. Returns ``(block, num_blocks)``, int32 [states], blocks in no particular order.

    A block is a splitter once, all its classes at a time, when it is made: of a block split the smaller part is a new
    block, the other keeps its number and its place among the splitters, if it has one. A state's moves are thus read
    at most ``log2(states) + 1`` times.
    """
    L = accept.shape[0]
    elems = np.empty(L, dtype = np.int32)                       # blocks as ranges of `elems`; `loc`: a state's place
    loc = np.empty(L, dtype = np.int32)
    block = np.empty(L, dtype = np.int32)
    start = np.zeros(L + 1, dtype = np.int32)
    size = np.zeros(L + 1, dtype = np.int32)
    num_accept = 0
    for q in range(L):
        if accept[q]:
            num_accept += 1
    on_accept, on_rest = 0, num_accept
    other_block = 1 if num_accept > 0 else 0
    for q in range(L):
        if accept[q]:
            elems[on_accept], loc[q], block[q] = q, on_accept, 0
            on_accept += 1
        else:
            elems[on_rest], loc[q], block[q] = q, on_rest, other_block
            on_rest += 1
    num_blocks = 0
    if num_accept > 0:
        size[0] = num_accept
        num_blocks = 1
    if num_accept < L:
        start[num_blocks], size[num_blocks] = num_accept, L - num_accept
        num_blocks += 1
    stack = np.empty(L + 1, dtype = np.int32)
    for b in range(num_blocks):
        stack[b] = b
    top = num_blocks
    moved = np.empty(into_ptr[L], dtype = np.int32)             # the moves into a block, by class: their sources
    count = np.zeros(C, dtype = np.int32)
    where = np.empty(C, dtype = np.int32)
    classes = np.empty(C, dtype = np.int32)
    marked = np.zeros(L + 1, dtype = np.int32)                  # per block: its marked states sit at its front
    touched = np.empty(L + 1, dtype = np.int32)
    while top > 0:
        top -= 1
        B = stack[top]
        first, last = start[B], start[B] + size[B]
        num_classes = 0
        for i in range(first, last):
            t = elems[i]
            for k in range(into_ptr[t], into_ptr[t + 1]):
                c = into_class[k]
                if count[c] == 0:
                    classes[num_classes] = c
                    num_classes += 1
                count[c] += 1
        place = 0
        for i in range(num_classes):
            c = classes[i]
            where[c] = place
            place += count[c]
            count[c] = 0
        for i in range(first, last):
            t = elems[i]
            for k in range(into_ptr[t], into_ptr[t + 1]):
                c = into_class[k]
                moved[where[c]] = into_src[k]
                where[c] += 1
        begin = 0
        for i in range(num_classes):                            # (B may split on the way: its moves are copied)
            end = where[classes[i]]
            num_touched = 0
            for k in range(begin, end):
                p = moved[k]
                b = block[p]
                front = start[b] + marked[b]
                pos = loc[p]
                if pos >= front:                                # not marked yet: move it to the front
                    if marked[b] == 0:
                        touched[num_touched] = b
                        num_touched += 1
                    other = elems[front]
                    elems[front], loc[p] = p, front
                    elems[pos], loc[other] = other, pos
                    marked[b] += 1
            begin = end
            for j in range(num_touched):
                b = touched[j]
                m = marked[b]
                marked[b] = 0
                if m == size[b]:
                    continue
                new = num_blocks
                num_blocks += 1
                if 2 * m <= size[b]:
                    start[new], size[new] = start[b], m
                    start[b] += m
                    size[b] -= m
                else:
                    start[new], size[new] = start[b] + m, size[b] - m
                    size[b] = m
                for x in range(start[new], start[new] + size[new]):
                    block[elems[x]] = new
                stack[top] = new
                top += 1
    return block, num_blocks


@njit(cache = True)
def _live_partition(ptr, cls, dst, accept, among, C):
    """The states of ``among`` an accepting state is reachable from, and the partition :func:`_coarsest` of them, by the
    moves ``(ptr, cls, dst)`` among them: ``(number, block, num_blocks)``, ``number`` int32 [states] a live state's place
    in ``block``, -1 for the others."""
    K = ptr.shape[0] - 1
    back = np.zeros(K + 1, dtype = np.int64)                    # the moves into every state: by a walk back from the
    for q in range(K):                                          # accepting states
        if among[q]:
            for e in range(ptr[q], ptr[q + 1]):
                back[dst[e] + 1] += 1
    for t in range(K):
        back[t + 1] += back[t]
    src = np.empty(back[K], dtype = np.int32)
    fill = back[:K].copy()
    for q in range(K):
        if among[q]:
            for e in range(ptr[q], ptr[q + 1]):
                t = dst[e]
                src[fill[t]] = q
                fill[t] += 1
    live = np.empty(K, dtype = np.int32)
    number = np.full(K, -1, dtype = np.int32)
    L = 0
    for q in range(K):
        if among[q] and accept[q]:
            number[q] = 0
            live[L] = q
            L += 1
    head = 0
    while head < L:
        t = live[head]
        head += 1
        for k in range(back[t], back[t + 1]):
            q = src[k]
            if number[q] < 0:
                number[q] = 0
                live[L] = q
                L += 1
    L = 0
    for q in range(K):
        if number[q] == 0:
            number[q] = L
            L += 1
    into_ptr = np.zeros(L + 1, dtype = np.int64)
    accepting = np.empty(L, dtype = np.bool_)
    for q in range(K):
        i = number[q]
        if i >= 0:
            accepting[i] = accept[q]
            for e in range(ptr[q], ptr[q + 1]):
                j = number[dst[e]]
                if j >= 0:
                    into_ptr[j + 1] += 1
    for t in range(L):
        into_ptr[t + 1] += into_ptr[t]
    into_class = np.empty(into_ptr[L], dtype = np.int32)
    into_src = np.empty(into_ptr[L], dtype = np.int32)
    fill = into_ptr[:L].copy()
    for q in range(K):
        i = number[q]
        if i >= 0:
            for e in range(ptr[q], ptr[q + 1]):
                j = number[dst[e]]
                if j >= 0:
                    into_class[fill[j]], into_src[fill[j]] = cls[e], i
                    fill[j] += 1
    block, num_blocks = _coarsest(into_ptr, into_class, into_src, accepting, C)
    return number, block, num_blocks


@njit(cache = True)
def hopcroft(delta, accept):
    """
    The coarsest partition of the states that keeps accepting and rejecting states apart and is respected by every
    class's transitions (every state reachable, the table complete): the block of every state (int64 [K], blocks numbered
    in no particular order) and the number of blocks. The states nothing accepted is reachable from are one block, the
    last; the moves into them are not looked at (:func:`_coarsest`).
    """
    K, C = delta.shape
    ptr, cls, dst = moves_of_table(delta, accept)
    number, block, num_blocks = _live_partition(ptr, cls, dst, accept, np.ones(K, dtype = np.bool_), C)
    out = np.full(K, num_blocks, dtype = np.int64)
    dead = False
    for q in range(K):
        if number[q] >= 0:
            out[q] = block[number[q]]
        else:
            dead = True
    return out, num_blocks + 1 if dead else num_blocks


@njit(cache = True)
def minimal_moves(ptr, cls, dst, accept, initial, C):
    """
    The minimal automaton of the moves ``(ptr, cls, dst)`` as CSR per state, classes ascending -- a class a state has no
    move on goes to a state from which nothing is accepted -- in canonical form: ``(out, accepting)``, int32 [states, C]
    and bool [states], the live states numbered in breadth-first order from the initial one (state 0), the one dead state
    last. Its time goes with the moves, not with states times classes (but for ``out``, filled once).
    """
    K = ptr.shape[0] - 1
    among = np.zeros(K, dtype = np.bool_)
    queue = np.empty(K, dtype = np.int32)
    queue[0] = initial
    among[initial] = True
    num, head = 1, 0
    while head < num:
        q = queue[head]
        head += 1
        for e in range(ptr[q], ptr[q + 1]):
            t = dst[e]
            if not among[t]:
                among[t] = True
                queue[num] = t
                num += 1
    number, block, m = _live_partition(ptr, cls, dst, accept, among, C)
    if number[initial] < 0:                                     # no text: the dead state alone
        return np.zeros((1, C), dtype = np.int32), np.zeros(1, dtype = np.bool_)
    rep = np.empty(m, dtype = np.int32)                         # any member represents its block
    for q in range(K):
        if number[q] >= 0:
            rep[block[number[q]]] = q
    place = np.full(m, -1, dtype = np.int32)
    order = np.empty(m, dtype = np.int32)
    order[0] = block[number[initial]]
    place[order[0]] = 0
    num, head = 1, 0
    out = np.full((m + 1, C), m, dtype = np.int32)
    accepting = np.zeros(m + 1, dtype = np.bool_)
    while head < num:
        r = rep[order[head]]
        accepting[head] = accept[r]
        for e in range(ptr[r], ptr[r + 1]):
            i = number[dst[e]]
            if i >= 0:
                b = block[i]
                if place[b] < 0:
                    place[b] = num
                    order[num] = b
                    num += 1
                out[head, cls[e]] = place[b]
        head += 1
    return out, accepting


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


@njit(cache = True)
def _column_classes(stacked):
    """The columns of ``stacked`` ([rows, columns]) grouped by equal content: ``(column_class, first)`` -- int64
    [columns] every column's group, the groups numbered by their first column, and every group's first column
    (ascending). One pass hashing every column (rows in order), then each column compared with the first column of
    each group of its hash."""
    R, W = stacked.shape
    h = np.full(W, np.uint64(1469598103934665603))
    prime = np.uint64(1099511628211)
    for r in range(R):
        for b in range(W):
            h[b] = (h[b] ^ np.uint64(stacked[r, b] + 1)) * prime
    column_class = np.empty(W, dtype = np.int64)
    first = np.empty(W, dtype = np.int64)
    num = 0
    for b in range(W):
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
        column_class[b] = found
    return column_class, first[:num].copy()
