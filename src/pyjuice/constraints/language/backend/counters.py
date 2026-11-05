"""
Bounded repetition, done once: ``maxLength`` and ``maxItems`` make most of the states of a schema's automaton, all of them
copies. ``{"maxItems": 100}`` of objects of two strings of ``"maxLength": 1024`` is a text of 487 characters and an automaton of 2
million states: 99 copies of one of 20 thousand, itself 2048 copies of one of 12. A copy in the middle of such a chain differs from
the next only by where it stands.

So the chain is wired short: as many copies as a token can span and a few more (:func:`kept_copies`), and the short automaton goes
through everything an automaton goes through, cheaply; only then the copies left out are put back, in the token automaton's moves
(:func:`expanded_moves`). What makes that exact:

* the subset construction is a function of the moves, so where the moves repeat and two neighbouring copies' sets are the same but
  for the shift, every further copy's are too: the check is :func:`_window`, on the sets themselves;
* the states of a further copy are told apart as those of the copy after it, and from every other state, if a copy's entry is alone
  in its block and nothing but the copy before leads to it (:func:`_window` checks, in the short automaton's partition). If they
  were one with a state ``y`` of a later copy, then ``y`` and the words out of the copy would lead to that entry, which nothing later does;
* a token spans at most ``L`` copies (:func:`_span`), so the walks from a copy with more than ``L`` copies after it see only copies
  alike. Such a walk stays in the chain, and then leads as far from every copy, or leaves it, and from there on the copy it left no
  longer matters: it leads to the same state from all of them, which may be in this chain again, counted from its start. The moves
  from two neighbouring copies tell which (:func:`_forward_moves`), and so tokens have the same classes in both automata.

Where anything does not hold, :class:`NotCounted` is raised and the schema goes the long way, which every operation that reads a
whole automaton (products, a test of a text) raises too: the short automaton is for :func:`~.token_lift.to_tokens` alone.
"""

from contextvars import ContextVar
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import get_num_threads, njit, prange


class NotCounted(Exception):
    """This schema's automaton cannot be made from a short one: the long way, then."""


class Reach:
    """What :func:`kept_copies` reads of a vocabulary: how far a token can go. Being in :data:`COUNTING` is what turns the
    short chains on, for a compile."""

    def __init__(self, vocabulary, data: np.ndarray):
        self.offsets = vocabulary._offsets
        self.data = data
        self.spans: Dict[int, Tuple[object, int]] = {}

    @property
    def longest(self) -> int:
        """The bytes of the longest token."""
        return int((self.offsets[1:] - self.offsets[:-1]).max())


COUNTING: ContextVar[Optional[Reach]] = ContextVar("counting", default = None)

#: copies left out of a chain at the least, for its short form to be worth a check (never fewer than a token spans, see `kept_copies`)
MIN_LEFT_OUT = 16
#: copies of a chain kept past the ones a token spans from the template: the first copy before it, and the last three after it (the
#: moves of the copy after the template are read too, as far as a token spans; and the copy after those is only checked)
KEPT_AROUND = 4


class Run:
    """
    A chain of copies, its middle left out: ``table`` int32 [window copies, states of a copy], those states in copy after copy, in one
    order: the first copy the template, ``extra`` copies to put after it. ``children`` the runs inside each window copy, in one
    order (their own copies are put back first, and a window copy's states are then those too).
    """

    __slots__ = ("table", "extra", "children", "level", "first_new", "states", "moves", "window")

    def __init__(self, table: np.ndarray, extra: int, children: List[List["Run"]]):
        self.table, self.extra, self.children = table, extra, children
        self.level = 1 + max((c.level for row in children for c in row), default = -1)
        self.first_new, self.states, self.moves, self.window = 0, 0, 0, None

    def at(self, offset: int, number: np.ndarray, held: np.ndarray, total: np.ndarray) -> "Run":
        """
        The same run in an automaton this one's is a part of, from its number ``offset`` on: ``number`` its numbers, ``held`` whether a
        set holds a state, ``total`` how many sets have every number. A set holding a state past the template stands somewhere no copy
        put back says (a state of the template, as a state before it, moves on into them); and a state one with another there may
        not be, where the copies are all there.
        """
        states = self.table + offset
        table = number[states]
        if (table < 0).any() or held[states[1:]].any() or (total[table] != 1).any():
            raise NotCounted
        return Run(table.astype(np.int32), self.extra, [[c.at(offset, number, held, total) for c in row] for row in self.children])


class Wired:
    """A repetition as :meth:`~.automata._Joined.wire` wires it short: what :func:`runs_of` needs to find its runs."""

    __slots__ = ("part", "regions", "offsets", "parts")

    def __init__(self, part, regions: List[Tuple[int, int, int]]):
        self.part = part
        self.regions = regions                                  # (the template's copy, copies in the window, copies left out)
        self.offsets: List[int] = []
        self.parts: List[int] = []


@njit(cache = True)
def _shortest(table, initial, accept, dead):
    """The bytes of the shortest text accepted, -1 if there is none, and whether accepting states lead nowhere."""
    S, C = table.shape
    alone = True
    for q in range(S):
        if accept[q]:
            for c in range(C):
                if table[q, c] != dead:
                    alone = False
                    break
    seen = np.zeros(S, dtype = np.bool_)
    now = np.empty(S, dtype = np.int32)
    next_ = np.empty(S, dtype = np.int32)
    seen[initial] = True
    now[0] = initial
    n, depth = 1, 0
    while n > 0:
        m = 0
        for i in range(n):
            q = now[i]
            if accept[q]:
                return depth, alone
            for c in range(C):
                t = table[q, c]
                if not seen[t] and t != dead:
                    seen[t] = True
                    next_[m] = t
                    m += 1
        now, next_ = next_, now
        n = m
        depth += 1
    return -1, alone


@njit(cache = True)
def _most_passes(table, byte_class, initial, accept, dead, data, offsets):
    """The most times the text of one token, from some byte of it on, goes all the way through an automaton whose accepting
    states lead nowhere, starting again at each end (from the end of a token back: from a byte on, once and as many as from where
    that ends)."""
    most = 0
    passes = np.zeros(1024, dtype = np.int32)
    for v in range(offsets.shape[0] - 1):
        start, end = offsets[v], offsets[v + 1]
        if end - start >= passes.shape[0]:
            passes = np.zeros(2 * (end - start), dtype = np.int32)
        passes[end - start] = 0
        for p in range(end - 1, start - 1, -1):
            q = initial
            n = 0
            for k in range(p, end):
                q = table[q, byte_class[data[k]]]
                if q == dead:
                    break
                if accept[q]:
                    n = 1 + passes[k + 1 - start]
                    break
            passes[p - start] = n
            if n > most:
                most = n
    return most


def _span(part, reach: Reach) -> int:
    """The most copies of ``part`` in a row a token can leave, -1 if a copy can be empty: every copy it leaves but the first it
    went all the way through."""
    known = reach.spans.get(id(part))
    if known is not None:
        return known[1]
    table = part._kept_table
    shortest, alone = _shortest(table, part.initial, part.accept, part.dead)
    if shortest <= 0:
        span = -1
    elif alone:
        span = 1 + _most_passes(table, part.byte_class, part.initial, part.accept, part.dead, reach.data, reach.offsets)
    else:
        span = (reach.longest - 1) // shortest + 1
    reach.spans[id(part)] = (part, span)                        # (kept with it: an id stays its while it is here)
    return span


def kept_copies(part, lo: int, copies: int) -> Optional[Tuple[int, int, Wired]]:
    """
    How much of ``copies`` copies of ``part``, each after the ``lo``-th accepting, to wire: ``(lo, copies, wired)``, or None for all
    of them. A chain has two stretches of alike copies: those after which the text cannot end yet, and those after which it can (the
    last of those is special, as is the first of each; a chain with no bound has one copy after which it can, looping). Of each only
    what a token spans from the template is kept, and around it :data:`KEPT_AROUND`.
    """
    reach = COUNTING.get()
    if reach is None or copies < 2:
        return None
    span = _span(part, reach)
    if span < 0:
        return None
    keep = span + KEPT_AROUND
    enough = keep + max(span + 1, MIN_LEFT_OUT)
    before = max(lo - 1, 0)                                     # (copies the text cannot end after)
    after = copies - before
    regions: List[Tuple[int, int, int]] = []
    kept_before, kept_after = before, after
    if before >= enough:
        kept_before = keep
        regions.append((1, span + 2, before - keep))
    if after >= enough:
        kept_after = keep
        regions.append((kept_before + 1, span + 2, after - keep))
    if not regions:
        return None
    return kept_before + 1 if lo >= 1 else 0, kept_before + kept_after, Wired(part, regions)


@njit(cache = True)
def _window(N, first, size, copies, t, G, ptr, cls, dst, accepting, read, pool, at, kept, number, total, K_final):
    """
    The states of the copies ``t`` to ``t + G - 1`` of a chain of ``copies`` copies of ``size`` states each from ``first`` on, in one
    order: int32 [G, states of a copy] of their numbers among the minimal automaton's (``number`` of every set, ``total`` how many
    sets are of every number) -- or an empty table, if they are not alike or are told apart otherwise than their neighbours are.
    A set is of the last copy it holds a state of.
    """
    fail = np.zeros((0, 0), dtype = np.int32)
    end = first + copies * size
    top = np.full(kept, -1, dtype = np.int32)
    for k in range(kept):
        for i in range(at[k], at[k + 1]):
            x = pool[i]
            if first <= x < end:
                c = (x - first) // size
                if c > top[k]:
                    top[k] = c
    # the sets of the copies, in one order: those of the template, then each one's next
    of_copy = np.zeros(copies + 2, dtype = np.int64)
    for k in range(kept):
        if top[k] >= 0:
            of_copy[top[k] + 1] += 1
    for c in range(copies + 1):
        of_copy[c + 1] += of_copy[c]
    by_copy = np.empty(of_copy[copies + 1], dtype = np.int32)
    fill = of_copy[:copies + 1].copy()
    for k in range(kept):
        if top[k] >= 0:
            by_copy[fill[top[k]]] = k
            fill[top[k]] += 1
    sets = of_copy[t + 1] - of_copy[t]
    for g in range(1, G):
        if of_copy[t + g + 1] - of_copy[t + g] != sets:
            return fail
    own = 0
    for q in range(size):
        if read[first + t * size + q]:
            own += 1
    u = own + sets
    ids = np.empty((G + 1, u), dtype = np.int32)
    locals_ = np.empty(own, dtype = np.int32)
    j = 0
    for q in range(size):
        if read[first + t * size + q]:
            locals_[j] = q
            j += 1
    for g in range(G + 1):
        c = t + g
        for q in range(size):                                   # (a state on its own: the same ones in every copy)
            if read[first + c * size + q] != read[first + t * size + q]:
                return fail
        for j in range(own):
            ids[g, j] = first + c * size + locals_[j]
    stamp = np.full(N, -1, dtype = np.int64)
    for a in range(sets):
        k = by_copy[of_copy[t] + a]
        ids[0, own + a] = N + k
        prev = k
        for g in range(1, G + 1):
            c = t + g
            found = -1
            for b in range(of_copy[c], of_copy[c + 1]):
                k2 = by_copy[b]
                if at[k2 + 1] - at[k2] != at[prev + 1] - at[prev]:
                    continue
                mark = np.int64(g) * kept + k2 + 1
                for i in range(at[k2], at[k2 + 1]):
                    stamp[pool[i]] = mark
                same = True
                for i in range(at[prev], at[prev + 1]):
                    x = pool[i]
                    y = x + size if first <= x < end else x
                    if y >= N or stamp[y] != mark:
                        same = False
                        break
                if same:
                    found = k2
                    break
            if found < 0:
                return fail
            ids[g, own + a] = N + found
            prev = found
    # every copy but the last moves as the next does: to itself, to the next copy, or to what is not in the chain
    where = np.zeros(N + kept, dtype = np.int64)
    for g in range(G + 1):
        for l in range(u):
            where[ids[g, l]] = g * u + l + 1
    for g in range(G - 1):
        for l in range(u):
            d, d2 = ids[g, l], ids[g + 1, l]
            if accepting[d] != accepting[d2] or ptr[d + 1] - ptr[d] != ptr[d2 + 1] - ptr[d2]:
                return fail
            for i in range(ptr[d + 1] - ptr[d]):
                e, e2 = ptr[d] + i, ptr[d2] + i
                if cls[e] != cls[e2]:
                    return fail
                y = dst[e]
                w = where[y]
                if w > 0:
                    gy = (w - 1) // u
                    if gy != g and gy != g + 1:
                        return fail
                    if dst[e2] != ids[gy + 1, (w - 1) % u]:
                        return fail
                else:
                    inside = (first <= y < end) if y < N else top[y - N] >= 0
                    if inside or dst[e2] != y:
                        return fail
    # nothing later leads to a copy's entry; it is one state; no number is shared but as the template's are
    is_entry = np.zeros(N + kept, dtype = np.bool_)
    for l in range(u):
        d = ids[0, l]
        for e in range(ptr[d], ptr[d + 1]):
            w = where[dst[e]]
            if w > 0 and (w - 1) // u == 1:
                is_entry[dst[e]] = True
    if G > 1:
        for l in range(u):
            d = ids[1, l]
            for e in range(ptr[d], ptr[d + 1]):
                if is_entry[dst[e]]:
                    return fail
            if is_entry[d] and (number[d] < 0 or total[number[d]] != 1):
                return fail
    rep = np.empty(u, dtype = np.int32)
    seen = np.full(K_final + 1, -1, dtype = np.int32)
    keep = 0
    for l in range(u):
        f = number[ids[0, l]]
        if f < 0:
            rep[l] = -1
        elif seen[f] < 0:
            seen[f] = l
            rep[l] = l
            keep += 1
        else:
            rep[l] = seen[f]
    owner = np.full(K_final + 1, -1, dtype = np.int32)
    inside_copy = np.zeros(K_final + 1, dtype = np.int32)
    table = np.empty((G, keep), dtype = np.int32)
    for g in range(G):
        at_keep = 0
        for l in range(u):
            f = number[ids[g, l]]
            if rep[l] < 0:
                if f >= 0:
                    return fail
                continue
            if f < 0 or f != number[ids[g, rep[l]]]:
                return fail
            inside_copy[f] += 1
            if rep[l] == l:
                if owner[f] >= 0:                               # (two states of a copy told apart in the template are not
                    return fail                                 # here, or a state of another copy is one with it)
                owner[f] = g
                table[g, at_keep] = f
                at_keep += 1
        for l in range(u):
            f = number[ids[g, l]]
            if rep[l] >= 0 and total[f] != inside_copy[f]:
                return fail
    return table


def runs_of(wired: List[Wired], parts: list, offsets: List[int], N: int, ptr, cls, dst, accepting, read, pool, at, kept: int, number: np.ndarray) -> List[Run]:
    """The runs of an automaton just made short, the outermost ones (:class:`Run`), in its numbers ``number`` of every set."""
    known = number[number >= 0]
    K_final = int(known.max()) + 1 if known.size else 0
    total = np.bincount(known, minlength = K_final + 1).astype(np.int64)
    held = np.zeros(N, dtype = np.bool_)
    held[pool[:at[kept]]] = True
    in_window: Dict[int, Tuple[int, int]] = {}
    found: List[Run] = []
    children_of: List[List[List[Run]]] = []
    tables: List[Tuple[np.ndarray, int]] = []
    for w in wired:
        first, size = w.offsets[0], w.part.num_states
        if any(o != first + i * size for i, o in enumerate(w.offsets)):
            raise NotCounted
        for t, G, extra in w.regions:
            table = _window(N, first, size, len(w.offsets), t, G, ptr, cls, dst, accepting, read, pool, at, kept, number, total, K_final)
            if table.size == 0:
                raise NotCounted
            tables.append((table, extra))
            children_of.append([[] for _ in range(G)])
            for g in range(G):
                in_window[w.parts[t + g]] = (len(tables) - 1, g)
    roots: List[Run] = []
    for i, (a, o) in enumerate(zip(parts, offsets)):
        if not a.counted:
            continue
        mine = [r.at(o, number, held, total) for r in a.counted]
        if i in in_window:
            index, g = in_window[i]
            children_of[index][g] = mine
        else:
            roots += mine
    for (table, extra), children in zip(tables, children_of):
        found.append(Run(table, extra, children))
    return roots + found


# -------------------------------------------------------------------------------------------------
# The copies put back: in the moves of a token automaton
# -------------------------------------------------------------------------------------------------

_LOW = (np.int64(1) << 32) - 1


@njit(cache = True)
def _forward_moves(moves, row_at, row_len, window, where_g, where_l, where_r, run, forward):
    """
    Which moves of a chain's template stay in the chain, marked in ``forward``: those whose class leads one copy further from the next
    copy. Any other leads to the same state from both, the walk having left the chain: it does from every copy, its target counted
    from what it went through and not from the copy it left. False if a move is neither (a token spans more than was thought).
    """
    for l in range(window.shape[1]):
        s, s1 = window[0, l], window[1, l]
        n = row_len[s]
        if row_len[s1] != n:
            return False
        a, a1 = row_at[s], row_at[s1]
        for i in range(n):
            m, m1 = moves[a + i], moves[a1 + i]
            if m >> 32 != m1 >> 32:
                return False
            y, y1 = m & _LOW, m1 & _LOW
            on = where_r[y] == run and where_r[y1] == run and where_g[y1] == where_g[y] + 1 and where_l[y1] == where_l[y]
            if not on and y1 != y:
                return False
            forward[a + i] = on
    return True


@njit(cache = True)
def _first_moves(moves, row_at, row_len, forward, template, seen_on, seen_out, mark, first_moves, at, copy_first):
    """
    What the breadth-first order reads of a row of a copy of ``template``: the first of its moves to every target, on through the chain
    or not -- two of them alike lead to one state from every copy. Rows have some ten targets, on far more classes. Written to
    ``first_moves`` from ``at`` on: how many, and which of the row; where in ``copy_first``. Returns ``(at, mark)``.
    """
    for l in range(template.shape[0]):
        s = template[l]
        a = row_at[s]
        copy_first[l] = at
        count_at = at
        at += 1
        mark += 1
        for i in range(row_len[s]):
            y = moves[a + i] & _LOW
            if forward[a + i]:
                if seen_on[y] != mark:
                    seen_on[y] = mark
                    first_moves[at] = i
                    at += 1
            elif seen_out[y] != mark:
                seen_out[y] = mark
                first_moves[at] = i
                at += 1
        first_moves[count_at] = at - count_at - 1
    return at, mark


@njit(cache = True)
def _copied_rows(moves, row_at, row_len, first_at, copy_first, forward, template, window, where_g, where_l, where_r, first_new_of, u_of, extra, first_new, u, moves_from,
                 moves_of_copy, row_start, k0, k1):
    """The rows of copies ``k0`` to ``k1 - 1`` (from 1) put after the template, its moves where it stands: a move on through the chain,
    ``g`` copies, leads ``g`` copies on from here, past the last copy put back into the next ones of the window; a move that left it leads
    where it does from the template, its copies past the template now among those put back."""
    for k in range(k0, k1):
        base = moves_from + (k - 1) * moves_of_copy
        new = first_new + (k - 1) * u
        for l in range(u):
            s = template[l]
            n = row_len[s]
            a = row_at[s]
            b = base + row_start[l]
            row_at[new + l] = b
            row_len[new + l] = n
            first_at[new + l] = copy_first[l]
            for i in range(n):
                m = moves[a + i]
                y = m & _LOW
                g = where_g[y]
                if g >= 0:
                    l2 = where_l[y]
                    if forward[a + i]:
                        q = k + g
                        t = first_new + (q - 1) * u + l2 if q <= extra else window[q - extra, l2]
                        m = m - y + t
                    elif g >= 1:
                        r = where_r[y]
                        m = m - y + first_new_of[r] + (g - 1) * u_of[r] + l2
                moves[b + i] = m


@njit(parallel = True, cache = True)
def _copied_rows_parallel(moves, row_at, row_len, first_at, copy_first, forward, template, window, where_g, where_l, where_r, first_new_of, u_of, extra, first_new, u,
                          moves_from, moves_of_copy, row_start, P):
    for p in prange(P):
        _copied_rows(moves, row_at, row_len, first_at, copy_first, forward, template, window, where_g, where_l, where_r, first_new_of, u_of, extra, first_new, u,
                     moves_from, moves_of_copy, row_start, 1 + p * extra // P, 1 + (p + 1) * extra // P)


@njit(cache = True)
def _led_on(moves, row_at, row_len, forward, template_of, where_g, where_l, where_r, first_new, u_of, past, lo, hi):
    """
    The rows ``lo`` to ``hi - 1`` of states before the copies put back, those of a window's copies past the template (``past``, first:
    it reads the templates' moves as they were) or the others: a move past a template now leads into the copies put back. But for a
    move of a state past the template that the template's, of its class, makes on through the chain: that leads on as it did.
    """
    for s in range(lo, hi):
        gs = where_g[s]
        if (gs >= 1) != past:
            continue
        a = row_at[s]
        a0, e0 = np.int64(0), np.int64(0)
        if past:
            a0 = row_at[template_of[s]]
            e0 = a0 + row_len[template_of[s]]
        for i in range(row_len[s]):
            m = moves[a + i]
            y = m & _LOW
            g = where_g[y]
            if g >= 1:
                r = where_r[y]
                if past and r == where_r[s]:
                    c = m >> 32
                    while a0 < e0 and moves[a0] >> 32 < c:
                        a0 += 1
                    if a0 < e0 and moves[a0] >> 32 == c and forward[a0]:
                        y0 = moves[a0] & _LOW
                        if where_g[y0] + gs == g and where_l[y0] == where_l[y]:
                            continue
                moves[a + i] = m - y + first_new[r] + (g - 1) * u_of[r] + where_l[y]


@njit(parallel = True, cache = True)
def _led_on_parallel(moves, row_at, row_len, forward, template_of, where_g, where_l, where_r, first_new, u_of, past, hi, P):
    for p in prange(P):
        _led_on(moves, row_at, row_len, forward, template_of, where_g, where_l, where_r, first_new, u_of, past, p * hi // P, (p + 1) * hi // P)


@njit(cache = True)
def _compacted(moves0, row_at, row_len, K, moves):
    """The rows of ``moves0``, wherever they are, one after the other from the start of ``moves``."""
    at = 0
    for s in range(K):
        a, n = row_at[s], row_len[s]
        for i in range(n):
            moves[at + i] = moves0[a + i]
        row_at[s] = at
        at += n
    return at


@njit(cache = True)
def _mark_window(window, number, where_g, where_l, where_r, template_of):
    """Where every state of a run's window stands. False if one stands in two runs already."""
    for g in range(window.shape[0]):
        for l in range(window.shape[1]):
            s = window[g, l]
            if where_g[s] >= 0:
                return False
            where_g[s], where_l[s], where_r[s] = g, l, number
            template_of[s] = window[0, l]
    return True


@njit(cache = True)
def _unmark_window(window, where_g, where_r, first_at):
    """The states of a run's window no longer stand in it, and two moves of one past the template, alike so far, may no longer be."""
    for g in range(window.shape[0]):
        for l in range(window.shape[1]):
            s = window[g, l]
            where_g[s], where_r[s] = -1, -1
            if g >= 1:
                first_at[s] = -1


@njit(cache = True)
def _accepting_copies(accept, template, first_new, extra):
    u = template.shape[0]
    for k in range(extra):
        for l in range(u):
            accept[first_new + k * u + l] = accept[template[l]]


def _all_runs(roots: List[Run]) -> List[Run]:
    out, stack = [], list(roots)
    while stack:
        r = stack.pop()
        out.append(r)
        for row in r.children:
            stack.extend(row)
    return out


def states_put_back(roots: List[Run]) -> int:
    """How many states the copies left out are."""
    total = 0
    for r in sorted(_all_runs(roots), key = lambda r: r.level):
        sizes = {sum(c.states * c.extra for c in row) for row in r.children}
        if len(sizes) != 1:
            raise NotCounted
        r.states = r.table.shape[1] + sizes.pop()
        total += r.states * r.extra
    return total


def expanded_moves(roots: List[Run], remap: np.ndarray, K: int, K_full: int, moves0: np.ndarray, row_at: np.ndarray, row_len: np.ndarray, accept: np.ndarray,
                   threads: int) -> np.ndarray:
    """
    The moves of the whole token automaton from the short one's, ``K`` of its ``K_full`` states: ``moves0``, ``row_at`` and ``row_len`` as
    :func:`~.token_lift._sparse_rows` leaves them for the first ``K``, ``remap`` its states of the byte automaton's. The inner runs first;
    a window copy's states are then its own and those put back inside it. Fills ``row_at``, ``row_len`` and ``accept`` for all of them and
    returns the moves, with room for one row more, and what the breadth-first order reads of the rows: ``(moves, first_at, first_moves)``
    (:func:`_first_moves`; ``first_at`` -1: all of the row).
    """
    runs = sorted(_all_runs(roots), key = lambda r: r.level)
    for r in runs:
        r.table = remap[r.table].astype(np.int32)
        if (r.table >= K).any():
            raise NotCounted
        of_copy = int(row_len[r.table[0]].sum()) + sum(c.moves * c.extra for c in r.children[0])
        r.moves = of_copy
    used = int(row_len[:K].sum())
    total = used + sum(r.moves * r.extra for r in runs)
    moves = np.empty(total + 1, dtype = np.int64)
    at = _compacted(moves0, row_at, row_len, K, moves)
    forward = np.zeros(total + 1, dtype = np.bool_)
    first_at = np.empty(K_full + 2, dtype = np.int32)
    first_at[:K] = -1
    first_at[K_full:] = -1
    first_moves = np.empty(sum(r.moves + r.states for r in runs) + 1, dtype = np.int32)
    first_end, mark = 0, 0
    seen_on = np.zeros(K_full + 2, dtype = np.int32)
    seen_out = np.zeros(K_full + 2, dtype = np.int32)
    where_g = np.full(K_full + 2, -1, dtype = np.int32)
    where_l = np.empty(K_full + 2, dtype = np.int32)
    where_r = np.full(K_full + 2, -1, dtype = np.int32)
    template_of = np.empty(K_full + 2, dtype = np.int32)
    first_new_of = np.zeros(len(runs), dtype = np.int64)
    u_of = np.zeros(len(runs), dtype = np.int64)
    next_id = K
    level = 0
    i = 0
    while i < len(runs):
        level = runs[i].level
        j = i
        while j < len(runs) and runs[j].level == level:
            j += 1
        before = next_id
        for number, r in enumerate(runs[i:j]):
            window = np.empty((r.table.shape[0], r.states), dtype = np.int32)
            for g in range(r.table.shape[0]):
                parts = [r.table[g]] + [np.arange(c.first_new, c.first_new + c.states * c.extra, dtype = np.int32) for c in r.children[g]]
                window[g] = np.concatenate(parts) if len(parts) > 1 else parts[0]
            r.window = window
            r.first_new = next_id
            next_id += r.states * r.extra
            if not _mark_window(window, number, where_g, where_l, where_r, template_of):
                raise NotCounted                                # (a state in two chains)
            first_new_of[number] = r.first_new
            u_of[number] = r.states
        for number, r in enumerate(runs[i:j]):
            if not _forward_moves(moves, row_at, row_len, r.window, where_g, where_l, where_r, number, forward):
                raise NotCounted
        for r in runs[i:j]:
            template = r.window[0]
            row_start = np.zeros(r.states, dtype = np.int64)
            row_start[1:] = np.cumsum(row_len[template])[:-1]
            copy_first = np.empty(r.states, dtype = np.int32)
            first_end, mark = _first_moves(moves, row_at, row_len, forward, template, seen_on, seen_out, mark, first_moves, first_end, copy_first)
            tail = (moves, row_at, row_len, first_at, copy_first, forward, template, r.window, where_g, where_l, where_r, first_new_of, u_of, r.extra, r.first_new, r.states, at,
                    r.moves, row_start)
            if r.moves * r.extra >= PARALLEL_MOVES and threads > 1:
                _copied_rows_parallel(*tail, min(threads, r.extra))
            else:
                _copied_rows(*tail, 1, r.extra + 1)
            at += r.moves * r.extra
            _accepting_copies(accept, template, r.first_new, r.extra)
        reading = (moves, row_at, row_len, forward, template_of, where_g, where_l, where_r, first_new_of, u_of)
        for past in (True, False):                              # (in this order: see `_led_on`)
            if at >= PARALLEL_MOVES and threads > 1:
                _led_on_parallel(*reading, past, before, min(threads, before))
            else:
                _led_on(*reading, past, 0, before)
        for r in runs[i:j]:
            _unmark_window(r.window, where_g, where_r, first_at)
        i = j
    if next_id != K_full or at != total:
        raise NotCounted
    return moves, first_at, first_moves


#: so many moves to write are worth a team
PARALLEL_MOVES = 1 << 18
