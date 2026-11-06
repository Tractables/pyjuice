"""
The tables of a layout's transitions, built once per constraint: the boundaries that are alike, every boundary's entries, grouped by successor
or by class, the enumerated tokens.
"""

import dataclasses
import weakref
from functools import cached_property
from typing import Optional, Tuple

import numpy as np
import torch
from .forms import EnumeratedTokens, TransitionTables
from .host import HostTables, host_tables


def _group_codes(counts: torch.Tensor, etok: torch.Tensor, stored: torch.Tensor) -> tuple:
    """Every class's code at every group of alike boundaries, from how many columns it leads somewhere from there (``counts`` int64 [G, C]; ``etok`` and ``stored``, see
    :class:`~pyjuice.constraints.distributions.TokenClasses`): ``(code, live, count)``, the codes [G, C], the enumerated classes live [G, C], how many of them [G]."""
    live = (counts > 0) & (etok >= 0)                                    # [G, C], enumerated classes live
    count = live.sum(dim = 1)
    e = torch.where(live, torch.cumsum(live, dim = 1) - 1, count[:, None])    # (dead: the count, past the last)
    return torch.where(etok >= 0, -1 - e, stored), live, count


def enumerated_tokens(tr, classes) -> EnumeratedTokens:
    """The :class:`EnumeratedTokens` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`, on
    the device) under ``classes`` (a :class:`~pyjuice.constraints.distributions.TokenClasses`)."""
    if tr.n > 0 and isinstance(boundary_groups(tr), HostTables):
        return boundary_groups(tr).enumerated_tokens(classes)
    n, C, dev = tr.n, tr.C, tr.device
    stored = classes.stored_of_class.to(dev).long()
    etok = classes.enumerated_token.to(dev).long()
    code = stored.expand(n, C).clone() if n > 0 else torch.zeros(0, C, dtype = torch.long, device = dev)
    ptr = torch.zeros(n + 1, dtype = torch.long)
    zero = torch.zeros(1, dtype = torch.long, device = dev)
    tok, index = zero, zero
    swept = None
    if classes.num_enumerated > 0 and n > 0:
        groups = boundary_groups(tr)
        # (what the codes need of the moves is what the grouped tables do, so one pass: :func:`transition_tables` finds it)
        swept = groups.sweep(True, lambda counts, k, end: _group_codes(counts, etok, stored)[0], from_counts = True)
        place = torch.cumsum(etok >= 0, dim = 0) - 1                         # an enumerated class's among them
        codes, live, count = _group_codes(groups.class_columns, etok, stored)
        code = codes[groups.at]
        ptr[1:] = torch.cumsum(count[groups.at], dim = 0).cpu()
        g, c = torch.nonzero(live, as_tuple = True)                          # by group, then class
        first = torch.cumsum(count, dim = 0) - count
        lengths = count[groups.at]
        ids = torch.repeat_interleave(first[groups.at], lengths) + torch.arange(int(lengths.sum()), device = dev) - \
            torch.repeat_interleave(torch.cumsum(lengths, dim = 0) - lengths, lengths)    # every boundary's classes
        tok = torch.cat([etok[c[ids]], zero])
        index = torch.cat([place[c[ids]], zero])
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    out = EnumeratedTokens(ptr = i32(ptr), tok = i32(tok), index = i32(index), code = i32(code), num = int(ptr[-1]),
                           ptr_host = ptr.numpy(), on = classes.num_enumerated > 0, column_rows = swept and swept["enum_rows"],
                           column_sizes = swept and swept["esizes"])
    if swept is not None:
        groups.swept = (out, swept)
    return out


#: Distinct boundaries' transitions are built in batches of at most this many of the automaton's transitions
BOUNDARY_BATCH = 1 << 26
#: A batch's entries are kept for the next table when they are at most this many
BOUNDARY_ENTRIES_KEPT = 1 << 22
#: The most entries the tables can point at: their pointers are int32, what the kernels read
TABLE_ENTRIES_MOST = (1 << 31) - 1
#: A batch's columns of every class are summed from its live pairs' class sets (:attr:`BoundaryGroups.class_sets`) where that takes at most this many terms; else read from its moves
CLASS_SET_TERMS = 1 << 28
#: How many hashes of the class sets are tried: one collides with about 2^-62 for two sets
CLASS_SET_TRIES = 4


def _ranges(starts: torch.Tensor, counts: torch.Tensor, total: int) -> torch.Tensor:
    """``starts[i], starts[i] + 1, ...``, ``counts[i]`` of them, one range after another, ``total`` in all, of the type of ``starts``."""
    return torch.repeat_interleave(starts - (torch.cumsum(counts, dim = 0) - counts).to(starts.dtype), counts) + \
        torch.arange(total, dtype = starts.dtype, device = starts.device)


def _rows_of(g: torch.Tensor, x: torch.Tensor, counts: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """int64 [groups, width + 1]: per group, how many there are below every value, ``counts[i]`` at the value ``x[i]`` of the group ``g[i]``, each pair once."""
    rows = torch.zeros(groups, width + 1, dtype = torch.int64, device = g.device)
    rows[g, x + 1] = counts
    return torch.cumsum(rows, dim = 1)


def _rows_below(g: torch.Tensor, x: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """int64 [groups, width + 1]: per group, how many of its ``x`` are below every value, in any order."""
    rows = torch.zeros(groups, width + 1, dtype = torch.int64, device = g.device)
    rows[:, 1:] = torch.bincount(g * width + x, minlength = groups * width).view(groups, width)
    return torch.cumsum(rows, dim = 1)


def _sort_key(key: torch.Tensor, domain: int) -> torch.Tensor:
    """``key``, below ``domain``, in the smallest integer type: a stable sort takes int16 five times as fast as int64, int32 twice and a half."""
    return key.to(torch.int16 if domain <= 1 << 15 else torch.int32 if domain <= 1 << 31 else torch.int64)


def _pointers(rows: list, counts: list, at: torch.Tensor, n: int, width: int, dev) -> torch.Tensor:
    """[n, width] int32: every boundary's pointer row, its segment's relative row past the segments before."""
    if not rows:
        return torch.zeros(n, width, dtype = torch.int32, device = dev)
    if sum(counts) > TABLE_ENTRIES_MOST:
        raise RuntimeError(f"{sum(counts)} transition entries are more than the tables' int32 pointers can reach: this automaton is too wide at this length.")
    bases = torch.tensor([0] + list(np.cumsum(counts)[:-1]), dtype = torch.int64, device = dev)
    return (torch.cat(rows)[at] + bases[at][:, None]).to(torch.int32).contiguous()


def _joined(parts: list, total: int, dev) -> torch.Tensor:
    """int32: the segments one after another (a 0 if there are none), let go of on the way."""
    out = torch.cat(parts).to(torch.int32).contiguous() if total > 0 else torch.zeros(1, dtype = torch.int32, device = dev)
    parts.clear()
    return out


def _boundary_weights(W: int, dev) -> torch.Tensor:
    """int64 [W]: what a hash of a boundary's states multiplies them by."""
    return torch.from_numpy(np.random.default_rng(0).integers(1, 1 << 62, W) | 1).to(dev)


def _set_weights(C: int, seed: int, dev) -> torch.Tensor:
    """int64 [C]: what a hash of a set of classes adds up of them."""
    return torch.from_numpy(np.random.default_rng(seed).integers(1, 1 << 62, C) | 1).to(dev)


class BoundaryGroups:
    """
    The boundaries of a layout's transitions that are alike (:func:`boundary_groups`) -- a boundary's transitions depend
    on its possible states and the next boundary's alone, and on its being the sequence end -- and their live entries,
    from a sparse form of the automaton: of its [K, C] table, only the transitions between states some boundary has.

    :ivar firsts: [G] the first boundary of every group (Python ints, ascending)
    :ivar group: [n] every boundary's group (Python ints), and ``at``, the same on the device
    :ivar batches: ``(first group, end group)`` of the batches they are built in (:data:`BOUNDARY_BATCH`)
    :ivar class_columns: int64 [G, C] on the device: the columns where every class leads somewhere (built when first read)
    :ivar most: :func:`transition_stats`' most, once known
    :ivar by_class: the transition tables' ``max_cls`` and ``cls_order``, once known: both forms' (:func:`transition_tables`)
    :ivar swept: ``(enumerated tokens, a pass's tables)``: what :func:`enumerated_tokens`' pass over the moves left for :func:`transition_tables`
    """

    def __init__(self, tr):
        n, dev = tr.n, tr.device
        self.tr = weakref.proxy(tr)                                          # (it keeps them: no cycle)
        sid = tr.state_id
        hashed = ((sid + 2) * _boundary_weights(sid.size(1), dev)).sum(dim = 1)   # (a sum of products, wrapping)
        _, boundary = torch.unique(hashed, return_inverse = True)            # boundaries of the same states, ...
        first = torch.full((int(boundary.max()) + 1,), n + 1, dtype = torch.long, device = dev)
        first.scatter_reduce_(0, boundary, torch.arange(n + 1, device = dev), "amin")
        if not torch.equal(sid, sid[first[boundary]]):                       # ... so far by a hash: exact now
            digests = {}
            boundary = torch.tensor([digests.setdefault(sid[t].cpu().numpy().tobytes(), len(digests)) for t in range(n + 1)],
                                    device = dev)
        ids = boundary.cpu().tolist()
        number, group, firsts = {}, [], []
        for t in range(n):
            key = (ids[t], ids[t + 1], t + 1 == n)
            at = number.get(key)
            if at is None:
                at = number[key] = len(firsts)
                firsts.append(t)
            group.append(at)
        self.firsts, self.group = firsts, group
        self.at = torch.tensor(group, dtype = torch.long, device = dev)
        self.most = None
        self.by_class = None
        self.swept = None
        self._class_columns = None
        self._kept = {}
        # the automaton's transitions between states some boundary has, a CSR by state
        active = (tr.col_of_state >= 0).any(dim = 0)
        rows = torch.nonzero(active)[:, 0]
        table = tr.delta[rows]
        at, cls = torch.nonzero(active.index_select(0, table.ravel()).view_as(table), as_tuple = True)
        self.state, self.cls = rows[at], cls.to(torch.int32)
        self.succ = table[at, cls]
        self.count = torch.bincount(self.state, minlength = tr.K)
        self.start = torch.cumsum(self.count, dim = 0) - self.count
        # ... at most so many of them a batch
        self.batches = []
        if firsts:
            first_t = torch.tensor(firsts, dtype = torch.long, device = dev)
            of_group = tr.state_id[first_t]
            sizes = (self.count[of_group.clamp(min = 0)] * (of_group >= 0)).sum(dim = 1).tolist()
            k, total = 0, 0
            for g, size in enumerate(sizes):
                if g > k and total + size > BOUNDARY_BATCH:
                    self.batches.append((k, g))
                    k, total = g, 0
                total += size
            self.batches.append((k, len(firsts)))

    def entries(self, k: int, end: int):
        """``(g, q, c, s)``, int64: the live entries of groups ``k .. end`` by group (from ``k``), column, then class:
        class ``c`` leads column ``q`` to column ``s`` of the next boundary."""
        kept = self._kept.get((k, end))
        if kept is not None:
            return kept
        tr = self.tr
        firsts = torch.tensor(self.firsts[k:end], dtype = torch.long, device = tr.device)
        W = max(max(tr.width[t] for t in self.firsts[k:end]), 1)
        sid = tr.state_id[firsts, :W]
        g, q = torch.nonzero(sid >= 0, as_tuple = True)
        states = sid[g, q]
        counts = self.count[states]
        at = _ranges(self.start[states], counts, int(counts.sum()))
        g, q = torch.repeat_interleave(g, counts), torch.repeat_interleave(q, counts)
        s = tr.col_of_state[firsts[g] + 1, self.succ[at]]
        live = s >= 0
        at = at[live]
        out = g[live], q[live], self.cls[at].long(), s[live]
        if out[0].numel() <= BOUNDARY_ENTRIES_KEPT:
            self._kept[(k, end)] = out
        return out

    @cached_property
    def pairs(self) -> tuple:
        """Every state's distinct successors in the order of the columns, which holds at every boundary, so that a boundary's pairs need no sort:
        ``(count, start, succ, of_move)``, how many a state has, where they start, the successors, and every move's among its state's (int32)."""
        tr = self.tr
        K, dev = tr.K, tr.device
        rank = torch.empty(K, dtype = torch.int64, device = dev)
        rank[tr.order] = torch.arange(K, device = dev)
        keys, of_move = torch.unique(self.state * K + rank.index_select(0, self.succ), return_inverse = True)
        state = torch.div(keys, K, rounding_mode = "floor")
        count = torch.bincount(state, minlength = K)
        start = torch.cumsum(count, dim = 0) - count
        of_move = (of_move - start[self.state]).to(torch.int32)
        self.state = None                                                    # (what it was kept for)
        return count, start, tr.order[keys - state * K], of_move

    @cached_property
    def class_sets(self) -> tuple:
        """
        A state's pairs (:attr:`pairs`) lead by a set of classes each, and those sets are few however many the pairs: in two vocabularies' schemas and expressions, fewer than two
        a class, for up to 17 times as many moves as pairs. ``(of_pair, member, in_set, start, size)``: every pair's set (int32), and the sets' classes, each's ascending, one set's
        after another's (``member``, int32, ``in_set`` the set of each), where each starts and how many it has. Numbered by a hash, then read exactly: a collision only costs a second try.
        """
        tr = self.tr
        dev = tr.device
        _, p_start, p_succ, of_move = self.pairs
        P, moves = p_succ.numel(), of_move.numel()
        pair = torch.repeat_interleave(p_start, self.count) + of_move
        size = torch.bincount(pair, minlength = P)
        first = torch.cumsum(size, dim = 0) - size
        classes = self.cls.index_select(0, torch.argsort(_sort_key(pair, P), stable = True))    # (a pair's moves: by class)
        for seed in range(CLASS_SET_TRIES):
            hashed = torch.zeros(P, dtype = torch.int64, device = dev).index_add_(0, pair, _set_weights(tr.C, seed, dev).index_select(0, self.cls))
            distinct, of_pair = torch.unique(hashed, return_inverse = True)
            S = distinct.numel()
            one = torch.full((S,), P, dtype = torch.int64, device = dev).scatter_reduce_(0, of_pair, torch.arange(P, device = dev), "amin")
            in_one = size.index_select(0, one)
            start = torch.cumsum(in_one, dim = 0) - in_one
            member = classes.index_select(0, _ranges(first.index_select(0, one), in_one, int(in_one.sum())))
            if torch.equal(size, in_one.index_select(0, of_pair)) and torch.equal(classes, member.index_select(0, _ranges(start.index_select(0, of_pair), size, moves))):
                return of_pair.to(torch.int32 if S < 1 << 31 else torch.int64), member, torch.repeat_interleave(torch.arange(S, device = dev), in_one), start, in_one
        raise RuntimeError("the pairs' sets of classes did not read back as their hashes said")

    def _candidates(self, k: int, end: int) -> tuple:
        """Of the groups ``k .. end``: their first boundaries; their columns, by group and column (``g``, ``q``), and how many distinct successors the state of each
        has; those successors, a column's after another's: where in :attr:`pairs`' they are (``at``), of which column (``of``, int32) and which column they are at the next
        boundary (``s``, int32, -1 none)."""
        tr = self.tr
        p_count, p_start, p_succ, _ = self.pairs
        firsts = torch.tensor(self.firsts[k:end], dtype = torch.long, device = tr.device)
        sid = tr.state_id[firsts, :max(max(tr.width[t] for t in self.firsts[k:end]), 1)]
        g, q = torch.nonzero(sid >= 0, as_tuple = True)
        states = sid[g, q]
        count = p_count[states]
        of = torch.repeat_interleave(torch.arange(states.numel(), dtype = torch.int32, device = tr.device), count)
        at = _ranges(p_start[states], count, of.numel())
        s = tr.col_of_state.reshape(-1)[((firsts[g] + 1) * tr.K)[of] + p_succ[at]].to(torch.int32)
        return firsts, g, q, states, count, at, of, s

    def live_pairs(self, k: int, end: int) -> tuple:
        """``(g, q, s)``, int64: the (column, successor) pairs of groups ``k .. end`` by group (from ``k``), column, then successor (a sequence end's as they are)."""
        _, g, q, _, _, _, of, s = self._candidates(k, end)
        live = s >= 0
        of = of[live]
        return g[of], q[of], s[live].long()

    def _live_moves(self, k: int, end: int, g: torch.Tensor, states: torch.Tensor, pair_count: torch.Tensor, s: torch.Tensor) -> tuple:
        """Of a batch (:meth:`_candidates`), its live moves by group, column, then class: of which column (``of_m``, int32), which candidate is their pair (``pair``), where they
        lead (``ms``, a sequence end's as it is), their class (``c``), their group from ``k`` and class in one (``class_key``)."""
        C, dev = self.tr.C, self.tr.device
        of_move = self.pairs[3]
        ints = torch.int32 if of_move.numel() < 1 << 31 else torch.int64
        first_pair = (torch.cumsum(pair_count, dim = 0) - pair_count).to(ints)
        count = self.count[states]
        of_m = torch.repeat_interleave(torch.arange(states.numel(), dtype = torch.int32, device = dev), count)
        at = _ranges(self.start[states].to(ints), count, of_m.numel())
        pair = first_pair.index_select(0, of_m) + of_move.index_select(0, at)
        ms = s.index_select(0, pair)
        live = ms >= 0
        of_m, at, pair, ms = of_m[live], at[live], pair[live], ms[live]
        c = self.cls.index_select(0, at)
        return of_m, pair, ms, c, g.to(ints if (end - k) * C < 1 << 31 else torch.int64).index_select(0, of_m) * C + c

    @staticmethod
    def _pairs_kept(of: torch.Tensor, s: torch.Tensor, at_end: Optional[torch.Tensor]) -> torch.Tensor:
        """Which of a batch's candidates are pairs: the live ones, but at a sequence end, of one column, only the first live of a column (``at_end``, if any is, of its columns)."""
        kept = s >= 0
        if at_end is not None:
            there = torch.nonzero(kept & at_end.index_select(0, of))[:, 0]
            kept[there[1:][of.index_select(0, there[1:]) == of.index_select(0, there[:-1])]] = False
        return kept

    def sweep(self, grouped: bool, codes = None, from_counts: bool = False) -> dict:
        """
        One pass over every batch of groups, for what the tables hold (:func:`transition_tables`) but the predecessors, the lists by class and the enumerated entries (:meth:`predecessors`,
        :meth:`class_lists`, :meth:`enumerated_lists`, what few queries read, each a pass of its own): nothing of a batch is kept but its tables' segments, int32. The pairs need no sort
        (:attr:`pairs`). How many columns a class leads somewhere from, and how many enumerated entries a column has, come from the live pairs' class sets (:attr:`class_sets`):
        the ungrouped tables, one entry a move, read the moves. ``codes(counts, k, end)``: the classes' codes [end - k, C] (``counts``, those columns, if ``from_counts``), None
        without any: enumerated or not, a class is so at every group.
        """
        tr = self.tr
        C, W, n, dev = tr.C, tr.W, tr.n, tr.device
        counting = from_counts or self._class_columns is None
        out = dict(succ = [], x = [], cls = [], entry_rows = [], enum_rows = [], sizes = [], esizes = [], pairs = [], max_pairs = [], max_enum = [], coded = codes is not None)
        counted = []
        reading = counting or codes is not None or not grouped

        def with_entries(es, ex, rows):
            """A batch's entries by group, column, then successor or class."""
            out["entry_rows"].append(rows)
            out["succ"].append(es.to(torch.int32)); out["x"].append(ex.to(torch.int32))
            out["sizes"].append(rows[:, W])

        for k, end in self.batches:
            Gb = end - k
            _, g, q, states, pair_count, at, of, s = self._candidates(k, end)
            cols = states.numel()
            at_end = g == Gb - 1 if self.firsts[end - 1] + 1 == n else None  # (a sequence end, the last group: one column there; one pair a column)
            of_m = ms = c = class_key = counts = code = of_l = sets = None
            if reading and grouped and Gb * self.class_sets[1].numel() <= CLASS_SET_TERMS:
                of_pair, member, in_set, _, in_each = self.class_sets
                S = in_each.numel()
                live = s >= 0
                of_l, sets = of[live], of_pair.index_select(0, at[live])
                if counting:
                    pairs_in = torch.bincount(g.to(torch.int32).index_select(0, of_l) * S + sets, minlength = Gb * S).view(Gb, S)
                    counts = torch.zeros(Gb, C, dtype = torch.int64, device = dev).index_add_(1, member, pairs_in.index_select(1, in_set))
                    del pairs_in
                del live
            elif reading:
                of_m, _, ms, c, class_key = self._live_moves(k, end, g, states, pair_count, s)
                if counting:
                    counts = torch.bincount(class_key, minlength = Gb * C).view(Gb, C)
            del at
            if counting:
                counted.append(counts)
            if codes is not None:                                            # the code of every class at its group, of every move's where those are read
                code = codes(counts, k, end)
                code = code[0] if of_l is not None else code.reshape(-1).to(torch.int32).index_select(0, class_key)
            if grouped:
                kept = self._pairs_kept(of, s, at_end)
                of_k = of[kept]
                on_col = torch.bincount(of_k, minlength = cols)
                rows = _rows_of(g, q, on_col, Gb, W)
                found = rows[:, W]
                if cols:
                    out["max_pairs"].append(on_col.max())
                eg, es = g.index_select(0, of_k), s[kept]
                if at_end is not None:
                    es = torch.where(at_end.index_select(0, of_k), torch.zeros_like(es), es)
                with_entries(es, torch.arange(eg.numel(), device = dev) - (torch.cumsum(found, dim = 0) - found).index_select(0, eg), rows)
                out["pairs"].append(found)
                del of_k, eg, es, kept
                if code is not None:                                         # how many enumerated entries a column has
                    if of_l is not None:
                        enumerated = torch.bincount(in_set[code.index_select(0, member) < 0], minlength = S)
                        on_col = torch.zeros(cols, dtype = torch.int64, device = dev).index_add_(0, of_l, enumerated.index_select(0, sets))
                    else:
                        on_col = torch.bincount(of_m[code < 0], minlength = cols)
                    rows = _rows_of(g, q, on_col, Gb, W)
                    out["enum_rows"].append(rows.to(torch.int32))
                    out["esizes"].append(rows[:, W].clone())
                    if cols:
                        out["max_enum"].append(on_col.max())
                    del on_col
            else:
                if at_end is not None:
                    ms = torch.where(at_end.index_select(0, of_m), torch.zeros_like(ms), ms)
                with_entries(ms, c if code is None else code, _rows_of(g, q, torch.bincount(of_m, minlength = cols), Gb, W))
                out["cls"].append(c)
            del g, q, states, of, s, of_m, ms, c, class_key, pair_count, code, counts, of_l, sets
        for name in ("sizes", "esizes", "pairs"):                            # (each read once: every read waits for the device)
            out[name] = torch.cat(out[name]).tolist() if out[name] else []
        for name in ("max_pairs", "max_enum"):
            out[name] = int(torch.stack(out[name]).max()) if out[name] else 0
        if grouped:
            self.most = max(out["max_pairs"], 1)                             # (a sequence end's column has one pair: its statistic and this are one)
        if counting:
            self._class_columns = torch.cat(counted) if counted else torch.zeros(0, C, dtype = torch.int64, device = dev)
        if self.by_class is None:                                            # (a boundary's classes by their columns, most first)
            counts = self.class_columns
            self.by_class = dict(max_cls = int(counts.max()) if counts.numel() else 0,
                                 cls_order = torch.argsort(-counts, dim = 1, stable = True)[self.at].to(torch.int32).contiguous())
        return out

    def predecessors(self, tt: TransitionTables) -> tuple:
        """``(pred_ptr, pred_col, pred_x, pred_class)`` (:class:`TransitionTables`) of the tables ``tt``, from their entries as they lie, by group, column, then successor or class: those
        leading to every next column, by group, successor, then what they were, a stable sort on the smallest keys that fit. No move read."""
        tr = self.tr
        W, n, dev = tr.W, tr.n, tr.device
        ptr = tt.entry_ptr.view(n, W + 1)
        firsts = torch.tensor(self.firsts, dtype = torch.long, device = dev)
        sizes = (ptr[firsts, W] - ptr[firsts, 0]).tolist()
        in_entries = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64).tolist()
        columns = torch.arange(W, dtype = torch.int32, device = dev)
        rows, cols, xs, classes = [], [], [], []
        for k, end in self.batches:
            Gb = end - k
            a, b = in_entries[k], in_entries[end]
            row = ptr[firsts[k:end]]
            ints = torch.int32 if Gb * (W + 1) < 1 << 31 else torch.int64
            eg = torch.repeat_interleave(torch.arange(Gb, dtype = ints, device = dev), (row[:, W] - row[:, 0]).long())
            eq = torch.repeat_interleave(columns.repeat(Gb), (row[:, 1:] - row[:, :-1]).reshape(-1).long())
            es = tt.entry_succ[a:b].to(ints)
            by_succ = torch.argsort(_sort_key(eg * (W + 1) + es, Gb * (W + 1)), stable = True)
            rows.append(_rows_below(eg, es, Gb, W))
            cols.append(eq.index_select(0, by_succ))
            xs.append(tt.entry_x[a:b].index_select(0, by_succ))
            if tt.entry_class is not None:
                classes.append(tt.entry_class[a:b].index_select(0, by_succ))
            del row, eg, eq, es, by_succ
        total = in_entries[-1]
        return (_pointers(rows, sizes, self.at, n, W + 1, dev), _joined(cols, total, dev), _joined(xs, total, dev),
                _joined(classes, total, dev) if tt.entry_class is not None else None)

    def class_lists(self) -> tuple:
        """``(cls_ptr, cls_col, cls_succ)`` (:class:`TransitionTables`), a pass of their own: every live pair's entries, one a class of its set (:attr:`class_sets`), by group, class,
        then column, a sort on the smallest keys that fit, for the class weights of a conditional. Not kept here, with a layout that a cache may keep. Past
        :data:`TABLE_ENTRIES_MOST` entries, an error here, and only here, before any of the pass."""
        tr = self.tr
        C, n, dev = tr.C, tr.n, tr.device
        counts = self.class_columns
        rows = torch.zeros(counts.size(0), C + 1, dtype = torch.int64, device = dev)
        rows[:, 1:] = torch.cumsum(counts, dim = 1)
        sizes = counts.sum(dim = 1).tolist()
        total = sum(sizes)
        ptr = _pointers([rows], sizes, self.at, n, C + 1, dev)
        of_pair, member, _, start, size = self.class_sets
        start = start.to(torch.int32 if member.numel() < 1 << 31 else torch.int64)
        cols, succs = [], []
        for k, end in self.batches:
            _, g, q, _, _, at, of, s = self._candidates(k, end)
            live = s >= 0
            of_l, s, sets = of[live], s[live], of_pair.index_select(0, at[live])
            del at, of, live
            if self.firsts[end - 1] + 1 == n:                                # (a sequence end, the last group: one column)
                s = torch.where(g.index_select(0, of_l) == end - k - 1, torch.zeros_like(s), s)
            in_set = size.index_select(0, sets)
            entries = int(in_set.sum())
            key = torch.repeat_interleave(g.to(torch.int32 if (end - k) * C < 1 << 31 else torch.int64).index_select(0, of_l) * C, in_set) + \
                member.index_select(0, _ranges(start.index_select(0, sets), in_set, entries))
            by_cls = torch.argsort(_sort_key(key, (end - k) * C), stable = True)
            cols.append(torch.repeat_interleave(q.to(torch.int32).index_select(0, of_l), in_set).index_select(0, by_cls))
            succs.append(torch.repeat_interleave(s, in_set).index_select(0, by_cls))
            del g, q, s, of_l, sets, in_set, key, by_cls
        return ptr, _joined(cols, total, dev), _joined(succs, total, dev)

    def enumerated_counts(self, enum: EnumeratedTokens, t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """int64: how many enumerated entries the columns ``q`` of the boundaries ``t`` have under ``enum``, which needs no lists (``enum.column_rows``)."""
        rows, g = torch.cat(enum.column_rows), self.at.index_select(0, t)
        return (rows[g, q + 1] - rows[g, q]).long()                          # (differences of int32: right past 2^31 too)

    def enumerated_pointers(self, enum: EnumeratedTokens) -> torch.Tensor:
        """``enum_ptr`` (:class:`TransitionTables`) under ``enum``; past :data:`TABLE_ENTRIES_MOST` entries, an error here, and only here."""
        tr = self.tr
        return _pointers(enum.column_rows, enum.column_sizes, self.at, tr.n, tr.W + 1, tr.device)

    def enumerated_lists(self, code: torch.Tensor) -> tuple:
        """``(enum_e, enum_pair)`` (:class:`TransitionTables`) of every column of every group under the codes ``code`` (int64 [n, C]): a pass of their own."""
        tr = self.tr
        n, dev = tr.n, tr.device
        ints = torch.int32 if self.pairs[3].numel() < 1 << 31 else torch.int64
        es, pairs = [], []
        total = 0
        for k, end in self.batches:
            firsts, g, q, states, pair_count, _, of, s = self._candidates(k, end)
            cols = states.numel()
            at_end = g == end - k - 1 if self.firsts[end - 1] + 1 == n else None
            of_m, pair, _, _, class_key = self._live_moves(k, end, g, states, pair_count, s)
            kept = self._pairs_kept(of, s, at_end)
            number = torch.cumsum(kept, dim = 0, dtype = ints) - 1           # a pair's number in the batch
            found = torch.zeros(end - k, dtype = torch.int64, device = dev).index_add_(0, g, torch.bincount(of[kept], minlength = cols))
            base = (torch.cumsum(found, dim = 0) - found).index_select(0, g).to(ints)
            v = code[firsts].reshape(-1).to(torch.int32).index_select(0, class_key)
            en = v < 0
            of_e, place = of_m[en], pair[en]
            if at_end is not None:                                           # (a sequence end's column has one pair: the last of its numbers is that one's)
                first_pair = (torch.cumsum(pair_count, dim = 0) - pair_count).to(ints)
                place = torch.where(at_end.index_select(0, of_e), first_pair.index_select(0, of_e) + pair_count.index_select(0, of_e).to(ints) - 1, place)
            es.append(-1 - v[en])
            pairs.append((number.index_select(0, place) - base.index_select(0, of_e)).to(torch.int32))
            total += of_e.numel()
            del firsts, g, q, states, pair_count, of, s, of_m, pair, class_key, kept, number, found, base, v, en, of_e, place
        return _joined(es, total, dev), _joined(pairs, total, dev)

    def enumerated_of(self, t: torch.Tensor, q: torch.Tensor, first_pair: torch.Tensor, code: torch.Tensor) -> tuple:
        """
        The enumerated entries of the columns ``q`` of the boundaries ``t`` alone (int64, by boundary then column; ``first_pair``, the number of each one's first pair at its
        boundary) under the codes ``code`` (int64 [n, C]): ``(ptr, e, pair)``, as :class:`TransitionTables`' ``enum_ptr``, ``enum_e``, ``enum_pair`` read for those columns, the
        others having none (``ptr`` [n, W + 1]), each column's in the order of the whole table's, so that what reads them does the same.
        """
        tr = self.tr
        n, W, K, dev = tr.n, tr.W, tr.K, tr.device
        p_count, p_start, p_succ, of_move = self.pairs
        states = tr.state_id[t, q]
        cols = states.numel()
        column = torch.arange(cols, dtype = torch.int32, device = dev)
        # the columns' pairs: which are live at the next boundary, how many live before each
        count = p_count[states]
        of = torch.repeat_interleave(column, count)
        s = tr.col_of_state.reshape(-1)[((t + 1) * K)[of] + p_succ[_ranges(p_start[states], count, of.numel())]]
        live_before = torch.zeros(of.numel() + 1, dtype = torch.int64, device = dev)
        live_before[1:] = torch.cumsum(s >= 0, dim = 0)
        first_of = torch.cumsum(count, dim = 0) - count
        # their moves of enumerated classes that are live: the classes of a state's in order
        count = self.count[states]
        of_m = torch.repeat_interleave(column, count)
        at = _ranges(self.start[states], count, of_m.numel())
        c = self.cls.index_select(0, at).long()
        en = code[0][c] < 0
        of_m, at, c = of_m[en], at[en], c[en]
        pair = first_of.index_select(0, of_m) + of_move.index_select(0, at).long()
        en = s.index_select(0, pair) >= 0
        of_m, pair, c = of_m[en], pair[en], c[en]
        at_t = t.index_select(0, of_m)
        within = live_before.index_select(0, pair) - live_before.index_select(0, first_of.index_select(0, of_m))
        within = torch.where(at_t + 1 == n, torch.zeros_like(within), within)     # (a sequence end's column has one pair)
        ptr = torch.zeros(n, W + 1, dtype = torch.int64, device = dev)
        ptr[t, q + 1] = torch.bincount(of_m, minlength = cols)
        return (torch.cumsum(ptr.view(-1), dim = 0).view(n, W + 1).to(torch.int32).contiguous(), (-1 - code[at_t, c]).to(torch.int32).contiguous(),
                (first_pair.index_select(0, of_m) + within).to(torch.int32).contiguous())

    @property
    def class_columns(self) -> torch.Tensor:
        if self._class_columns is None:
            C, dev = self.tr.C, self.tr.device
            parts = []
            for k, end in self.batches:
                g, _, c, _ = self.entries(k, end)
                parts.append(torch.bincount(g * C + c, minlength = (end - k) * C).view(end - k, C))
            self._class_columns = torch.cat(parts) if parts else torch.zeros(0, C, dtype = torch.int64, device = dev)
        return self._class_columns


def boundary_groups(tr):
    """The :class:`BoundaryGroups` of a layout's transitions, kept with them -- or their :class:`~.host.HostTables`, for few transitions:
    then the tables below are built there."""
    kept = tr.__dict__.get("_boundary_groups")
    if kept is None:
        kept = host_tables(tr)
        kept = tr.__dict__["_boundary_groups"] = BoundaryGroups(tr) if kept is None else kept
    return kept


def next_col_table(tr) -> torch.Tensor:
    """int32 [n, W, C] on the device: every boundary's column at the next boundary of every (column, class), -1 where
    that is inactive or past its columns."""
    groups = boundary_groups(tr)
    if isinstance(groups, HostTables):
        return groups.next_col_table()
    out = torch.full((tr.n, tr.W, tr.C), -1, dtype = torch.int32, device = tr.device)
    for k, end in groups.batches:
        g, q, c, s = groups.entries(k, end)
        W = int(q.max()) + 1 if q.numel() else 1
        table = torch.full((end - k, W, tr.C), -1, dtype = torch.int32, device = tr.device)
        table[g, q, c] = s.to(torch.int32)
        ts = torch.nonzero((groups.at >= k) & (groups.at < end)).flatten()
        out[ts, :W] = table[groups.at[ts] - k]
    return out


def transition_stats(tr) -> Tuple[int, int, int]:
    """``(live, columns, most)`` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`): how many
    (column, class) pairs of the boundaries lead somewhere, how many columns the boundaries have, and the most distinct
    columns one column reaches in one token at any boundary but the last (a sequence end being one column; at least
    1)."""
    groups = boundary_groups(tr)
    n, W = tr.n, tr.W
    if n == 0:
        return 0, 0, 1
    if isinstance(groups, HostTables):
        return groups.transition_stats()
    boundaries = np.bincount(groups.group, minlength = len(groups.firsts))
    live = int((groups.class_columns.sum(dim = 1).cpu().numpy() * boundaries).sum())
    if groups.most is None:
        most = 1
        for k, end in groups.batches:
            g, q, _, s = groups.entries(k, end)
            keep = torch.as_tensor(groups.firsts[k:end], device = g.device)[g] + 1 < n
            pairs = torch.unique(((g * (W + 1) + q) * (W + 1) + s)[keep])
            if pairs.numel():
                most = max(most, int(torch.bincount(pairs // (W + 1)).max()))
        groups.most = most
    return live, sum(tr.width[:n]), groups.most


def transition_tables(tr, grouped: bool, enum: Optional[EnumeratedTokens] = None, lists: bool = True) -> TransitionTables:
    """
    The :class:`TransitionTables` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`, on the
    device); ``enum`` the :class:`EnumeratedTokens` (None: nothing enumerated, every class's code the class itself).

    A boundary's entries depend on its possible states and the next boundary's alone (its columns, their successors'
    columns, the classes live there -- an enumerated token's code counts from the boundary's first), and on its being
    the sequence end. Boundaries alike SHARE their entries: one segment, built once, every one of them pointing into it
    (``entry_ptr[t]``, ``pred_ptr[t]``, ``cls_ptr[t]``, ``enum_ptr[t]``). Decoding at a length well past the shortest
    accepted string, most boundaries are alike: a JSON Schema automaton of 24k states has 81 distinct of 1024. The
    distinct boundaries are built together, in batches (:data:`BOUNDARY_BATCH`).

    ``lists`` False, on the device: neither the predecessors (``pred_ptr``, ``pred_col``, ``pred_x``, ``pred_class``), nor the lists by class (``cls_ptr``, ``cls_col``, ``cls_succ``), nor
    the enumerated entries (``enum_ptr``, ``enum_e``, ``enum_pair``): None. They are most of the tables and few queries read them: a top-down pass, or a forward with ``block @ input``
    steps, the first (:meth:`~...compiled.program.Program.pred_lists`), a conditional's class weights the second (:meth:`~...compiled.program.Program.class_lists`), and masses by column
    signature those of few columns of the last (:meth:`~...compiled.program.Program.enum_lists`, :meth:`BoundaryGroups.enumerated_of`).
    """
    if isinstance(boundary_groups(tr), HostTables):
        return boundary_groups(tr).transition_tables(grouped, enum)
    n, W, dev = tr.n, tr.W, tr.device
    groups = boundary_groups(tr)
    swept, groups.swept = groups.swept, None
    code = enum.code.long() if enum is not None and enum.on else None
    if grouped and swept is not None and swept[0] is enum:
        out = swept[1]                                                       # (made with these codes)
    else:
        del swept
        firsts = torch.tensor(groups.firsts, dtype = torch.long, device = dev)
        out = groups.sweep(grouped, None if code is None else (lambda counts, k, end: code[firsts[k:end]]))
    sizes, esizes, at, group = out["sizes"], out["esizes"], groups.at, groups.group
    base = sum(sizes)
    per_boundary = np.concatenate([[0], np.cumsum([sizes[g] for g in group])]).astype(np.int64)
    pointers = lambda rows, counts: _pointers(rows, counts, at, n, W + 1, dev)
    none = torch.zeros(1, dtype = torch.int32, device = dev)
    enum_ptr, enum_e, enum_pair = None, None, None
    if grouped:
        if not out["coded"]:                                                 # (nothing enumerated: none)
            enum_ptr, enum_e, enum_pair = pointers([], []), none, none
        elif lists:
            enum_ptr = pointers(out["enum_rows"], esizes)
            enum_e, enum_pair = groups.enumerated_lists(code)
    cls_ptr, cls_col, cls_succ = groups.class_lists() if lists else (None, None, None)
    tables = TransitionTables(grouped = grouped, entry_ptr = pointers(out["entry_rows"], sizes), entry_succ = _joined(out["succ"], base, dev),
                              entry_x = _joined(out["x"], base, dev), pred_ptr = None, pred_col = None,
                              pred_x = None, num_pairs = [out["pairs"][g] for g in group] if grouped else [],
                              max_pairs = out["max_pairs"], enum_ptr = enum_ptr, enum_e = enum_e, enum_pair = enum_pair, max_enum = out["max_enum"],
                              cls_ptr = cls_ptr, cls_col = cls_col, cls_succ = cls_succ, **groups.by_class, entry_class = None if grouped else _joined(out["cls"], base, dev),
                              boundary_base = torch.from_numpy(per_boundary).to(dev), num_entries = int(per_boundary[-1]))
    if lists:
        pred_ptr, pred_col, pred_x, pred_class = groups.predecessors(tables)
        tables = dataclasses.replace(tables, pred_ptr = pred_ptr, pred_col = pred_col, pred_x = pred_x, pred_class = pred_class)
    return tables
