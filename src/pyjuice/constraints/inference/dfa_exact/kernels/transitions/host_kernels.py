"""
The compiled loops of the tables on the host (:mod:`.host`): every table from the live entries of the distinct boundaries, ``(g, q, c, s)``:
class ``c`` leads column ``q`` of group ``g`` to column ``s`` of the next boundary, by group, column, then class (:func:`_entries`). All orders are counting sorts of them.
"""

import numpy as np
from numba import njit


@njit(cache = True)
def _boundaries(state_id, n):
    """``(firsts, group)``: the boundaries of the same states with the next of the same states, not a sequence end but for the
    last, as one group, numbered by their first boundary (every boundary's group, every group's first)."""
    W = state_id.shape[1]
    cap = 16
    while cap < 2 * (n + 1):
        cap *= 2
    slots = np.full(cap, -1, dtype = np.int32)
    ids = np.empty(n + 1, dtype = np.int32)
    reps = np.empty(n + 1, dtype = np.int32)
    num = 0
    for t in range(n + 1):
        h = np.uint64(1469598103934665603)
        for j in range(W):
            h = (h ^ np.uint64(state_id[t, j] + 2)) * np.uint64(1099511628211)
        at = np.int64(h & np.uint64(cap - 1))
        while True:
            k = slots[at]
            if k < 0:
                slots[at] = num
                reps[num] = t
                ids[t] = num
                num += 1
                break
            r = reps[k]
            same = True
            for j in range(W):
                if state_id[r, j] != state_id[t, j]:
                    same = False
                    break
            if same:
                ids[t] = k
                break
            at = (at + 1) & (cap - 1)
    pairs = np.full(cap, np.int64(-1))
    numbers = np.empty(cap, dtype = np.int32)
    group = np.empty(n, dtype = np.int64)
    firsts = np.empty(n, dtype = np.int64)
    G = 0
    for t in range(n):
        if t + 1 == n:                                          # (one of its kind)
            group[t] = G
            firsts[G] = t
            G += 1
            break
        key = np.int64(ids[t]) * (n + 2) + ids[t + 1]
        at = (key * np.int64(0x9E3779B97F4A7C1)) & (cap - 1)
        while pairs[at] >= 0 and pairs[at] != key:
            at = (at + 1) & (cap - 1)
        if pairs[at] < 0:
            pairs[at] = key
            numbers[at] = G
            firsts[G] = t
            G += 1
        group[t] = numbers[at]
    return firsts[:G], group


@njit(cache = True)
def _moves(delta, col_of_state):
    """``(ptr, cls, succ)``: the moves of every state between states some boundary has, as CSR, classes ascending."""
    K, C = delta.shape
    rows = col_of_state.shape[0]
    active = np.zeros(K, dtype = np.bool_)
    for t in range(rows):
        for s in range(K):
            if col_of_state[t, s] >= 0:
                active[s] = True
    ptr = np.zeros(K + 1, dtype = np.int64)
    for s in range(K):
        n = 0
        if active[s]:
            for c in range(C):
                if active[delta[s, c]]:
                    n += 1
        ptr[s + 1] = ptr[s] + n
    cls = np.empty(ptr[K], dtype = np.int32)
    succ = np.empty(ptr[K], dtype = np.int32)
    at = 0
    for s in range(K):
        if active[s]:
            for c in range(C):
                t = delta[s, c]
                if active[t]:
                    cls[at], succ[at] = c, t
                    at += 1
    return ptr, cls, succ


@njit(cache = True)
def _num_entries(firsts, width, state_id, ptr):
    """At most so many live entries: every column's moves, at the groups' boundaries."""
    out = 0
    for g in range(firsts.shape[0]):
        t = firsts[g]
        for q in range(width[t]):
            s = state_id[t, q]
            out += ptr[s + 1] - ptr[s]
    return out


@njit(cache = True)
def _entries(firsts, width, state_id, col_of_state, ptr, cls, succ, most):
    """The live entries of the groups, by group, column, then class: ``(g, q, c, s)``, int32 [E]."""
    g_out = np.empty(most, dtype = np.int32)
    q_out = np.empty(most, dtype = np.int32)
    c_out = np.empty(most, dtype = np.int32)
    s_out = np.empty(most, dtype = np.int32)
    E = 0
    for g in range(firsts.shape[0]):
        t = firsts[g]
        for q in range(width[t]):
            st = state_id[t, q]
            for e in range(ptr[st], ptr[st + 1]):
                s = col_of_state[t + 1, succ[e]]
                if s >= 0:
                    g_out[E], q_out[E], c_out[E], s_out[E] = g, q, cls[e], s
                    E += 1
    return g_out[:E], q_out[:E], c_out[:E], s_out[:E]


@njit(cache = True)
def _group_table(g, q, c, s, G, W, C):
    """int32 [G, W, C]: every (group, column, class)'s column at the next boundary, -1 where it leads nowhere."""
    table = np.full((G, W, C), -1, dtype = np.int32)
    for e in range(g.shape[0]):
        table[g[e], q[e], c[e]] = s[e]
    return table


@njit(cache = True)
def _class_columns(g, c, G, C):
    """int64 [G, C]: the columns where every class leads somewhere."""
    out = np.zeros((G, C), dtype = np.int64)
    for e in range(g.shape[0]):
        out[g[e], c[e]] += 1
    return out


@njit(cache = True)
def _below(g, x, G, width):
    """int32 [G, width + 1]: per group, how many of its ``x`` are below every value (``g`` ascending)."""
    out = np.zeros((G, width + 1), dtype = np.int32)
    for e in range(g.shape[0]):
        out[g[e], x[e] + 1] += 1
    for k in range(G):
        for v in range(width):
            out[k, v + 1] += out[k, v]
    return out


@njit(cache = True)
def _group_bases(g, G):
    """int64 [G + 1]: where every group starts among entries by group."""
    out = np.zeros(G + 1, dtype = np.int64)
    for e in range(g.shape[0]):
        out[g[e] + 1] += 1
    for k in range(G):
        out[k + 1] += out[k]
    return out


@njit(cache = True)
def _by_second(g, x, G, width):
    """int64 [entries]: the entries, by group, reordered by ``x`` within their group, stably."""
    N = g.shape[0]
    bases = _group_bases(g, G)
    counts = _below(g, x, G, width)
    place = np.empty((G, width), dtype = np.int64)
    for k in range(G):
        for v in range(width):
            place[k, v] = bases[k] + counts[k, v]
    out = np.empty(N, dtype = np.int64)
    for e in range(N):
        k, v = g[e], x[e]
        out[place[k, v]] = e
        place[k, v] += 1
    return out


@njit(cache = True)
def _by_class_tables(g, q, c, s, G, C):
    """The entries by class: ``(rows, columns, successors, order, most)``: int32 [G, C + 1] the entries of a group below every class,
    those of every (group, class) by column, int64 [G, C] a group's classes by their columns, most first, the most of any."""
    rows = _below(g, c, G, C)
    at = _by_second(g, c, G, C)
    order = np.empty((G, C), dtype = np.int64)
    most = 0
    lengths = np.empty(C, dtype = np.int64)
    for k in range(G):
        for v in range(C):
            lengths[v] = rows[k, v] - rows[k, v + 1]            # (negated: most first)
            most = max(most, rows[k, v + 1] - rows[k, v])
        order[k] = np.argsort(lengths, kind = "mergesort")
    return rows, q[at], s[at], order, most


@njit(cache = True)
def _pairs(g, q, s, G, W):
    """The distinct successors of every (group, column), ascending: ``(pair_g, pair_q, pair_s, x)``, ``x`` int32 [entries]: every
    entry's pair's number within its group."""
    E = g.shape[0]
    pair_g = np.empty(E, dtype = np.int32)
    pair_q = np.empty(E, dtype = np.int32)
    pair_s = np.empty(E, dtype = np.int32)
    x = np.empty(E, dtype = np.int32)
    stamp = np.full(W + 1, -1, dtype = np.int64)
    place = np.empty(W + 1, dtype = np.int32)
    P, e, k, base = 0, 0, -1, 0
    while e < E:
        if g[e] != k:
            k = g[e]
            base = P
        column, start, end = q[e], P, e
        while end < E and g[end] == k and q[end] == column:
            v = s[end]
            if stamp[v] != e:
                stamp[v] = e
                pair_s[P] = v
                P += 1
            end += 1
        if P - start > 32:
            pair_s[start:P] = np.sort(pair_s[start:P])
        else:
            for i in range(start + 1, P):
                v = pair_s[i]
                j = i
                while j > start and pair_s[j - 1] > v:
                    pair_s[j] = pair_s[j - 1]
                    j -= 1
                pair_s[j] = v
        for i in range(start, P):
            pair_g[i], pair_q[i] = k, column
            place[pair_s[i]] = i - base
        for f in range(e, end):
            x[f] = place[s[f]]
        e = end
    return pair_g[:P], pair_q[:P], pair_s[:P], x


@njit(cache = True)
def _most_of_columns(row, ends_here):
    """``(most, inner)``: the most of any column of ``row`` [G, W + 1] (below every column), and of those not at a sequence end (1 at least)."""
    most, inner = 0, 1
    for k in range(row.shape[0]):
        for v in range(row.shape[1] - 1):
            m = row[k, v + 1] - row[k, v]
            most = max(most, m)
            if not ends_here[k]:
                inner = max(inner, m)
    return most, inner


@njit(cache = True)
def _mix(v, salt):
    """:func:`~.signatures._mix`, of the bits of an int64."""
    z = np.uint64(v) + (np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(salt))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache = True)
def _powers(longest, salt):
    """:func:`~.signatures._segment_hash`'s powers."""
    out = np.empty(max(longest, 1), dtype = np.uint64)
    base = np.uint64(0x100000001B3 + 2 * salt)
    power = np.uint64(1)
    for k in range(out.shape[0]):
        out[k] = power
        power = power * base
    return out


@njit(cache = True)
def _column_signatures(firsts, width, state_id, n, pair_base, pair_row, pair_s, enum_base, enum_row, enum_e, enum_pair, tok_index, tok_ptr,
                       enumerated, pair_bits, enum_bits, longest_pairs, longest_enum):
    """
    The signatures of a group's columns (:func:`~.signatures.signature_masses`; alike boundaries, alike columns, so no boundary but a
    group's first is looked at): ``(row_first, sig, first, ok)``: a column ``(g, q)``'s row is ``row_first[g] + q``, ``sig`` int64 its
    signature, ``first`` every signature's first row, ``ok`` whether every row's operands are its first's (no hash collision). The
    rows numbered as the device's: by the sizes, then the hashes, as signed values.
    """
    G = firsts.shape[0]
    row_first = np.zeros(G + 1, dtype = np.int64)
    for g in range(G):
        row_first[g + 1] = row_first[g] + width[firsts[g]]
    R = row_first[G]
    key = np.empty(R, dtype = np.int64)
    h1 = np.empty(R, dtype = np.int64)
    h3 = np.zeros(R, dtype = np.int64)
    pw1 = _powers(longest_pairs, 1)
    pw3 = _powers(longest_enum, 3)
    for g in range(G):
        t = firsts[g]
        end = t + 1 == n
        for q in range(width[t]):
            a, b = pair_row[g, q], pair_row[g, q + 1]
            count = b - a
            h = np.uint64(0)
            for k in range(count):
                v = np.int64(-7) if end else state_id[t + 1, pair_s[pair_base[g] + a + k]]
                h += _mix(v, 1) * pw1[k]
            r = row_first[g] + q
            h1[r] = np.int64(h)
            nenum = 0
            if enumerated:
                c, d = enum_row[g, q], enum_row[g, q + 1]
                nenum = d - c
                h = np.uint64(0)
                for k in range(nenum):
                    tok = np.int64(tok_index[tok_ptr[t] + enum_e[enum_base[g] + c + k]])
                    h += _mix(tok * 65536 + (enum_pair[enum_base[g] + c + k] - a), 3) * pw3[k]
                h3[r] = np.int64(h)
            key[r] = ((((state_id[t, q] << pair_bits) | count) << enum_bits | nenum) << 1) | (1 if end else 0)
    # the distinct rows, by a table of their hashes (few, as a rule), and their order, by sorting them alone
    cap = 16
    while cap < 2 * R:
        cap *= 2
    slots = np.full(cap, -1, dtype = np.int32)
    distinct = np.empty(R, dtype = np.int64)
    first_of = np.empty(R, dtype = np.int64)
    D = 0
    for r in range(R):
        at = np.int64((np.uint64(key[r]) * np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(h1[r]) ^ (np.uint64(h3[r]) >> np.uint64(7))) & np.uint64(cap - 1))
        while True:
            d = slots[at]
            if d < 0:
                d = D
                slots[at] = d
                first_of[d] = r
                D += 1
                break
            o = first_of[d]
            if key[o] == key[r] and h1[o] == h1[r] and h3[o] == h3[r]:
                break
            at = (at + 1) & (cap - 1)
        distinct[r] = d
    rows = first_of[:D]
    order = np.argsort(h3[rows], kind = "mergesort")
    order = order[np.argsort(h1[rows[order]], kind = "mergesort")]
    order = order[np.argsort(key[rows[order]], kind = "mergesort")]
    number = np.empty(D, dtype = np.int64)
    first = np.empty(D, dtype = np.int64)
    for i in range(D):
        number[order[i]] = i
        first[i] = rows[order[i]]
    sig = np.empty(R, dtype = np.int64)
    for r in range(R):
        sig[r] = number[distinct[r]]
    # exact: every column's operands are its first's
    group_of = np.empty(R, dtype = np.int64)
    for g in range(G):
        for r in range(row_first[g], row_first[g + 1]):
            group_of[r] = g
    ok = True
    for g in range(G):
        t = firsts[g]
        if not ok:
            break
        for q in range(width[t]):
            r = row_first[g] + q
            o = first[sig[r]]
            if o == r:
                continue
            g2 = group_of[o]
            t2, q2 = firsts[g2], o - row_first[g2]
            a, a2 = pair_row[g, q], pair_row[g2, q2]
            for k in range(pair_row[g, q + 1] - a):
                if t + 1 < n and state_id[t + 1, pair_s[pair_base[g] + a + k]] != state_id[t2 + 1, pair_s[pair_base[g2] + a2 + k]]:
                    ok = False
            if enumerated:
                c, c2 = enum_row[g, q], enum_row[g2, q2]
                for k in range(enum_row[g, q + 1] - c):
                    i, i2 = enum_base[g] + c + k, enum_base[g2] + c2 + k
                    if tok_index[tok_ptr[t] + enum_e[i]] != tok_index[tok_ptr[t2] + enum_e[i2]] or enum_pair[i] - a != enum_pair[i2] - a2:
                        ok = False
    return row_first, sig, first, ok


@njit(cache = True)
def _signature_representatives(group, class_of, width, row_first, sig, sig_off, pair_row, num_classes, S, G):
    """Every (class of positions, signature)'s first column at a position of the class: ``(position, column, shift)``, by position, then
    column, the shift its first pair's slot less its first pair's number. A position of a class and group seen before has none."""
    n = group.shape[0]
    seen = np.zeros(max(num_classes * S, 1), dtype = np.bool_)
    visited = np.zeros(max(num_classes * G, 1), dtype = np.bool_)
    most = max(num_classes * S, 1)
    rep_t = np.empty(most, dtype = np.int64)
    rep_q = np.empty(most, dtype = np.int32)
    rep_shift = np.empty(most, dtype = np.int64)
    m = 0
    for t in range(n):
        cl = class_of[t]
        if cl < 0:
            continue
        g = group[t]
        if visited[cl * G + g]:
            continue
        visited[cl * G + g] = True
        for q in range(width[t]):
            sg = sig[row_first[g] + q]
            if not seen[cl * S + sg]:
                seen[cl * S + sg] = True
                rep_t[m], rep_q[m], rep_shift[m] = t, q, sig_off[sg] - pair_row[g, q]
                m += 1
    return rep_t[:m], rep_q[:m], rep_shift[:m]


@njit(cache = True)
def _signature_slots(entry_x, pred_x, firsts, width, row_first, sig, sig_off, pair_base, pair_row):
    """The entries' and the predecessors' mass indices: their pair's signature slot."""
    exs = entry_x.copy()
    pxs = pred_x.copy()
    for g in range(firsts.shape[0]):
        for q in range(width[firsts[g]]):
            off = sig_off[sig[row_first[g] + q]]
            a = pair_row[g, q]
            for k in range(pair_row[g, q + 1] - a):
                exs[pair_base[g] + a + k] = off + k
        for i in range(pair_base[g], pair_base[g + 1]):
            pxs[i] = exs[pair_base[g] + pred_x[i]]
    return exs, pxs
