"""
The tables of a layout's transitions, built once per constraint: the boundaries that are alike, every boundary's entries, grouped by successor
or by class, the enumerated tokens.
"""

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
                           ptr_host = ptr.numpy(), on = classes.num_enumerated > 0)
    if swept is not None:
        groups.swept = (out, swept)
    return out


#: Distinct boundaries' transitions are built in batches of at most this many of the automaton's transitions
BOUNDARY_BATCH = 1 << 26
#: A batch's entries are kept for the next table when they are at most this many
BOUNDARY_ENTRIES_KEPT = 1 << 22
#: The most entries the tables can point at: their pointers are int32, what the kernels read
TABLE_ENTRIES_MOST = (1 << 31) - 1


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
    :ivar by_class: the transition tables' fields of the entries by class, once known: both forms' (:func:`transition_tables`)
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

    def _candidates(self, k: int, end: int) -> tuple:
        """Of the groups ``k .. end``: their first boundaries; their columns, by group and column (``g``, ``q``), and how many distinct successors the state of each
        has; those successors, a column's after another's: of which column (``of``, int32) and which column they are at the next boundary (``s``, int32, -1 none)."""
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
        return firsts, g, q, states, count, of, s

    def live_pairs(self, k: int, end: int) -> tuple:
        """``(g, q, s)``, int64: the (column, successor) pairs of groups ``k .. end`` by group (from ``k``), column, then successor (a sequence end's as they are)."""
        _, g, q, _, _, of, s = self._candidates(k, end)
        live = s >= 0
        of = of[live]
        return g[of], q[of], s[live].long()

    def sweep(self, grouped: bool, codes = None, from_counts: bool = False) -> dict:
        """
        One pass over the moves of every batch of groups, for what the tables hold (:func:`transition_tables`): nothing of a batch is kept but its tables' segments, int32, and
        nothing here sorts them but what the order of the pairs (:attr:`pairs`) does not give: by successor, and by class. The tables by class come once, and the columns of the
        classes. ``codes(counts, k, end)``: the classes' codes [end - k, C] (``counts``, the columns where every class leads somewhere, if ``from_counts``), None without any.
        """
        tr = self.tr
        C, W, n, dev = tr.C, tr.W, tr.n, tr.device
        of_move = self.pairs[3]
        ints = torch.int32 if of_move.numel() < 1 << 31 else torch.int64
        starts = self.start.to(ints)
        making = self.by_class is None
        counting = from_counts or making or self._class_columns is None
        out = dict(succ = [], x = [], col = [], px = [], e = [], pair = [], cls = [], pcls = [], entry_rows = [], pred_rows = [], enum_rows = [],
                   sizes = [], esizes = [], pairs = [], max_pairs = 0, max_enum = 0, coded = codes is not None)
        counted, cls_rows, csizes, ccols, csuccs, orders = [], [], [], [], [], []
        max_cls = 0

        def with_predecessors(eg, eq, es, ex, rows):
            """A batch's entries by group, column, then successor or class, and those leading to every next column: by group, successor, then what they were."""
            Gb = rows.size(0)
            by_succ = torch.argsort(_sort_key(eg * (W + 1) + es, Gb * (W + 1)), stable = True)
            out["entry_rows"].append(rows)
            out["pred_rows"].append(_rows_below(eg, es, Gb, W))
            out["succ"].append(es.to(torch.int32)); out["x"].append(ex.to(torch.int32))
            out["col"].append(eq.index_select(0, by_succ).to(torch.int32)); out["px"].append(ex.index_select(0, by_succ).to(torch.int32))
            out["sizes"] += rows[:, W].tolist()
            return by_succ

        for k, end in self.batches:
            Gb = end - k
            firsts, g, q, states, pair_count, of, s = self._candidates(k, end)
            cols = states.numel()
            ends_here = firsts + 1 == n
            at_end = ends_here[g] if ends_here.any() else None               # (a sequence end: one column there; one pair a column)
            first_pair = (torch.cumsum(pair_count, dim = 0) - pair_count).to(ints)
            column = torch.arange(cols, dtype = torch.int32, device = dev)
            # every live move: its column, its class, its pair, where it leads
            count = self.count[states]
            of_m = torch.repeat_interleave(column, count)
            at = _ranges(starts[states], count, of_m.numel())
            pair = first_pair.index_select(0, of_m) + of_move.index_select(0, at)
            ms = s.index_select(0, pair)
            live = ms >= 0
            of_m, at, pair, ms = of_m[live], at[live], pair[live], ms[live]
            c = self.cls.index_select(0, at)
            del at, live, count
            class_key = g.to(ints if Gb * C < 1 << 31 else torch.int64).index_select(0, of_m) * C + c
            counts = None
            if counting:
                counts = torch.bincount(class_key, minlength = Gb * C).view(Gb, C)
                counted.append(counts)
            # the code of every move's class at its group
            code = None if codes is None else codes(counts, k, end).reshape(-1).to(torch.int32).index_select(0, class_key)
            if grouped:
                kept = s >= 0
                if at_end is not None:                                       # (its first live)
                    there = torch.nonzero(kept & at_end.index_select(0, of))[:, 0]
                    kept[there[1:][of.index_select(0, there[1:]) == of.index_select(0, there[:-1])]] = False
                number = torch.cumsum(kept, dim = 0, dtype = ints) - 1       # a pair's number in the batch
                of_k = of[kept]
                on_col = torch.bincount(of_k, minlength = cols)
                rows = _rows_of(g, q, on_col, Gb, W)
                found = rows[:, W]
                base = torch.cumsum(found, dim = 0) - found
                if cols:
                    out["max_pairs"] = max(out["max_pairs"], int(on_col.max()))
                eg, eq, es = g.index_select(0, of_k), q.index_select(0, of_k), s[kept]
                if at_end is not None:
                    es = torch.where(at_end.index_select(0, of_k), torch.zeros_like(es), es)
                with_predecessors(eg, eq, es, torch.arange(eg.numel(), device = dev) - base.index_select(0, eg), rows)
                out["pairs"] += found.tolist()
                del of_k, eg, eq, es, kept
                if code is not None:                                         # the enumerated classes' live entries
                    en = code < 0
                    of_e, place = of_m[en], pair[en]
                    if at_end is not None:                                   # (a sequence end's column has one pair: the last of its numbers is that one's)
                        place = torch.where(at_end.index_select(0, of_e), first_pair.index_select(0, of_e) + pair_count.index_select(0, of_e).to(ints) - 1, place)
                    on_col = torch.bincount(of_e, minlength = cols)
                    rows = _rows_of(g, q, on_col, Gb, W)
                    out["enum_rows"].append(rows)
                    out["e"].append(-1 - code[en])
                    out["pair"].append((number.index_select(0, place) - base.index_select(0, g).to(ints).index_select(0, of_e)).to(torch.int32))
                    out["esizes"] += rows[:, W].tolist()
                    if of_e.numel():
                        out["max_enum"] = max(out["max_enum"], int(on_col.max()))
                    del en, of_e, place
                del number
            if at_end is not None:
                ms = torch.where(at_end.index_select(0, of_m), torch.zeros_like(ms), ms)
            if not grouped:
                eg, eq = g.index_select(0, of_m), q.index_select(0, of_m)
                by_succ = with_predecessors(eg, eq, ms, c if code is None else code, _rows_of(g, q, torch.bincount(of_m, minlength = cols), Gb, W))
                out["cls"].append(c); out["pcls"].append(c.index_select(0, by_succ))
                del eg, eq, by_succ
            if making:                                                       # by group, class, then column
                rows = torch.zeros(Gb, C + 1, dtype = torch.int64, device = dev)
                rows[:, 1:] = torch.cumsum(counts, dim = 1)
                cls_rows.append(rows)
                by_cls = torch.argsort(_sort_key(class_key, Gb * C), stable = True)
                ccols.append(q.to(torch.int32).index_select(0, of_m.index_select(0, by_cls)))
                csuccs.append(ms.index_select(0, by_cls))
                csizes += counts.sum(dim = 1).tolist()
                orders.append(torch.argsort(-counts, dim = 1, stable = True))
                max_cls = max(max_cls, int(counts.max()))
                del by_cls
            del firsts, g, q, states, of, s, of_m, pair, ms, c, class_key, first_pair, pair_count, column, code, counts
        if grouped:
            self.most = max(out["max_pairs"], 1)                             # (a sequence end's column has one pair: its statistic and this are one)
        if counting:
            self._class_columns = torch.cat(counted) if counted else torch.zeros(0, C, dtype = torch.int64, device = dev)
        if making:
            total = sum(csizes)
            self.by_class = dict(cls_ptr = _pointers(cls_rows, csizes, self.at, n, C + 1, dev), cls_col = _joined(ccols, total, dev), cls_succ = _joined(csuccs, total, dev),
                                 max_cls = max_cls, cls_order = torch.cat(orders)[self.at].to(torch.int32).contiguous() if orders else
                                 torch.arange(C, device = dev).repeat(n, 1).to(torch.int32).contiguous())
        return out

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


def transition_tables(tr, grouped: bool, enum: Optional[EnumeratedTokens] = None) -> TransitionTables:
    """
    The :class:`TransitionTables` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`, on the
    device); ``enum`` the :class:`EnumeratedTokens` (None: nothing enumerated, every class's code the class itself).

    A boundary's entries depend on its possible states and the next boundary's alone (its columns, their successors'
    columns, the classes live there -- an enumerated token's code counts from the boundary's first), and on its being
    the sequence end. Boundaries alike SHARE their entries: one segment, built once, every one of them pointing into it
    (``entry_ptr[t]``, ``pred_ptr[t]``, ``cls_ptr[t]``, ``enum_ptr[t]``). Decoding at a length well past the shortest
    accepted string, most boundaries are alike: a JSON Schema automaton of 24k states has 81 distinct of 1024. The
    distinct boundaries are built together, in batches (:data:`BOUNDARY_BATCH`).
    """
    if isinstance(boundary_groups(tr), HostTables):
        return boundary_groups(tr).transition_tables(grouped, enum)
    n, W, dev = tr.n, tr.W, tr.device
    groups = boundary_groups(tr)
    swept, groups.swept = groups.swept, None
    if grouped and swept is not None and swept[0] is enum:
        out = swept[1]                                                       # (made with these codes)
    else:
        del swept
        code = enum.code.long() if enum is not None and enum.on else None
        firsts = torch.tensor(groups.firsts, dtype = torch.long, device = dev)
        out = groups.sweep(grouped, None if code is None else (lambda counts, k, end: code[firsts[k:end]]))
    sizes, esizes, at, group = out["sizes"], out["esizes"], groups.at, groups.group
    base, ebase = sum(sizes), sum(esizes)
    enumerated = grouped and out["coded"]
    per_boundary = np.concatenate([[0], np.cumsum([sizes[g] for g in group])]).astype(np.int64)
    pointers = lambda rows, counts: _pointers(rows, counts, at, n, W + 1, dev)
    return TransitionTables(grouped = grouped, entry_ptr = pointers(out["entry_rows"], sizes), entry_succ = _joined(out["succ"], base, dev),
                            entry_x = _joined(out["x"], base, dev), pred_ptr = pointers(out["pred_rows"], sizes), pred_col = _joined(out["col"], base, dev),
                            pred_x = _joined(out["px"], base, dev), num_pairs = [out["pairs"][g] for g in group] if grouped else [],
                            max_pairs = out["max_pairs"],
                            enum_ptr = pointers(out["enum_rows"], esizes) if enumerated else (pointers([], []) if grouped else None),
                            enum_e = _joined(out["e"], ebase, dev) if grouped else None,
                            enum_pair = _joined(out["pair"], ebase, dev) if grouped else None, max_enum = out["max_enum"],
                            **groups.by_class, entry_class = None if grouped else _joined(out["cls"], base, dev),
                            pred_class = None if grouped else _joined(out["pcls"], base, dev),
                            boundary_base = torch.from_numpy(per_boundary).to(dev), num_entries = int(per_boundary[-1]))
