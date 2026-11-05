"""
The tables of a layout's transitions, built once per constraint: the boundaries that are alike, every boundary's entries, grouped by successor
or by class, the enumerated tokens.
"""

import weakref
from typing import Optional, Tuple

import numpy as np
import torch
from .forms import EnumeratedTokens, TransitionTables
from .host import HostTables, host_tables


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
    if classes.num_enumerated > 0 and n > 0:
        groups = boundary_groups(tr)
        place = torch.cumsum(etok >= 0, dim = 0) - 1                         # an enumerated class's among them
        live = (groups.class_columns > 0) & (etok >= 0)                      # [G, C], enumerated classes live
        count = live.sum(dim = 1)
        e = torch.where(live, torch.cumsum(live, dim = 1) - 1, count[:, None])    # (dead: the count, past the last)
        code = torch.where(etok >= 0, -1 - e, stored)[groups.at]
        ptr[1:] = torch.cumsum(count[groups.at], dim = 0).cpu()
        g, c = torch.nonzero(live, as_tuple = True)                          # by group, then class
        first = torch.cumsum(count, dim = 0) - count
        lengths = count[groups.at]
        ids = torch.repeat_interleave(first[groups.at], lengths) + torch.arange(int(lengths.sum()), device = dev) - \
            torch.repeat_interleave(torch.cumsum(lengths, dim = 0) - lengths, lengths)    # every boundary's classes
        tok = torch.cat([etok[c[ids]], zero])
        index = torch.cat([place[c[ids]], zero])
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    return EnumeratedTokens(ptr = i32(ptr), tok = i32(tok), index = i32(index), code = i32(code), num = int(ptr[-1]),
                            ptr_host = ptr.numpy())


#: Distinct boundaries' transitions are built in batches of at most this many of the automaton's transitions
BOUNDARY_BATCH = 1 << 26
#: A batch's entries are kept for the next table when they are at most this many
BOUNDARY_ENTRIES_KEPT = 1 << 22


def _below(keys: torch.Tensor, size: int) -> torch.Tensor:
    """int64 [size + 1]: how many of the ascending ``keys`` are below each value."""
    return torch.searchsorted(keys, torch.arange(size + 1, device = keys.device))


def _counts(keys: torch.Tensor, size: int) -> torch.Tensor:
    """int64 [size]: how many of the ascending ``keys`` are each value."""
    below = _below(keys, size)
    return below[1:] - below[:-1]


def _rows_below(g: torch.Tensor, x: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """int64 [groups, width + 1]: per group, how many of its ``x`` are below every value (``g``, ``x`` ascending
    together)."""
    below = _below(g * width + x, groups * width)
    at = torch.arange(groups, device = g.device)[:, None] * width + torch.arange(width + 1, device = g.device)[None, :]
    return below[at] - below[at[:, :1]]


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
        self._class_columns = None
        self._kept = {}
        self.holding = False                                                 # (see `release`)
        # the automaton's transitions between states some boundary has, a CSR by state
        active = (tr.col_of_state >= 0).any(dim = 0)
        keep = active[tr.delta]
        keep[~active] = False
        self.state, self.cls = torch.nonzero(keep, as_tuple = True)
        self.succ = tr.delta[self.state, self.cls]
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
        at = torch.repeat_interleave(self.start[states], counts) + torch.arange(int(counts.sum()), device = tr.device) - \
            torch.repeat_interleave(torch.cumsum(counts, dim = 0) - counts, counts)
        g, q = torch.repeat_interleave(g, counts), torch.repeat_interleave(q, counts)
        s = tr.col_of_state[firsts[g] + 1, self.succ[at]]
        live = s >= 0
        at = at[live]
        out = g[live], q[live], self.cls[at], s[live]
        if out[0].numel() <= BOUNDARY_ENTRIES_KEPT or self.holding:
            self._kept[(k, end)] = out
        return out

    def release(self) -> None:
        """A program holds every batch's entries while it builds its tables, each reading them (``holding``, set by it):
        what :data:`BOUNDARY_ENTRIES_KEPT` would not keep goes now."""
        self.holding = False
        self._kept = {key: out for key, out in self._kept.items() if out[0].numel() <= BOUNDARY_ENTRIES_KEPT}

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
    n, W, C, dev = tr.n, tr.W, tr.C, tr.device
    code = enum.code.long() if enum is not None and enum.num > 0 else None
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    cat = lambda xs, total: i32(torch.cat(xs)) if total > 0 else torch.zeros(1, dtype = torch.int32, device = dev)
    groups = boundary_groups(tr)
    firsts, group, at = groups.firsts, groups.group, groups.at
    G = len(firsts)

    succs, xs, cols, pxs, ees, epairs, ccols, csuccs, clss, pclss = ([] for _ in range(10))
    entry_rows, pred_rows, enum_rows, cls_rows, orders = ([] for _ in range(5))
    sizes, csizes, esizes, pairs = [], [], [], []
    max_pairs = max_enum = max_cls = 0
    most = 1
    by_class = groups.by_class                                       # (the entries by class: the same for both forms)
    for k, end in groups.batches:
        Gb = end - k
        g, q, c, s = groups.entries(k, end)                                  # by group, column, then class
        ends_here = torch.as_tensor(firsts[k:end], device = dev) + 1 == n
        if ends_here.any():                                                  # a sequence end: one column
            s = torch.where(ends_here[g], torch.zeros_like(s), s)
        if by_class is None:
            by_cls = torch.argsort(g * C + c, stable = True)                 # by group, class, then column
            gc, cc, qc = g[by_cls], c[by_cls], q[by_cls]
            classes = _rows_below(gc, cc, Gb, C)
            lengths = classes[:, 1:] - classes[:, :-1]
            cls_rows.append(classes)
            ccols.append(qc); csuccs.append(s[by_cls])
            csizes += _counts(g, Gb).tolist()
            orders.append(torch.argsort(-lengths, dim = 1, stable = True))
            max_cls = max(max_cls, int(lengths.max()))
        if grouped:
            key, inverse = torch.unique((g * (W + 1) + q) * (W + 1) + s, return_inverse = True)   # by group, column, successor
            eg, rest = key // ((W + 1) * (W + 1)), key % ((W + 1) * (W + 1))
            eq, es = rest // (W + 1), rest % (W + 1)
            found = _counts(eg, Gb)
            base = torch.cumsum(found, dim = 0) - found
            ex = torch.arange(key.numel(), device = dev) - base[eg]
            pairs += found.tolist()
            if key.numel():
                column = _rows_below(eg, eq, Gb, W + 1)
                column = column[:, 1:] - column[:, :-1]
                max_pairs = max(max_pairs, int(column.max()))
                inner = column[~ends_here]                                   # (not at a sequence end: a statistic's)
                if inner.numel():
                    most = max(most, int(inner.max()))
            if code is not None:                                             # the enumerated classes' live entries
                codes = code[torch.as_tensor(firsts[k:end], device = dev)]
                en = codes[g, c] < 0
                enum_rows.append(_rows_below(g[en], q[en], Gb, W))
                ees.append(-1 - codes[g[en], c[en]]); epairs.append(inverse[en] - base[g[en]])
                esizes += _counts(g[en], Gb).tolist()
                if en.any():
                    max_enum = max(max_enum, int(_counts(g[en] * (W + 1) + q[en], Gb * (W + 1)).max()))
        else:
            eg, eq, es, ex = g, q, s, (c if code is None else code[torch.as_tensor(firsts[k:end], device = dev)][g, c])
            clss.append(c)
        entry_rows.append(_rows_below(eg, eq, Gb, W))
        by_succ = torch.argsort((eg * (W + 1) + es) * (W + 1) + eq, stable = True)
        pred_rows.append(_rows_below(eg[by_succ], es[by_succ], Gb, W))
        succs.append(es); xs.append(ex); cols.append(eq[by_succ]); pxs.append(ex[by_succ])
        if not grouped:
            pclss.append(c[by_succ])
        sizes += _counts(eg, Gb).tolist()
    if grouped:
        groups.most = most

    def pointers(rows, counts):
        """[n, width] int32: every boundary's pointer row, its segment's relative row past the segments before."""
        if not rows:
            return torch.zeros(n, W + 1, dtype = torch.int32, device = dev)
        bases = torch.tensor([0] + list(np.cumsum(counts)[:-1]), dtype = torch.int64, device = dev)
        return i32(torch.cat(rows)[at] + bases[at][:, None])

    base, ebase = sum(sizes), sum(esizes)
    enumerated = grouped and code is not None
    per_boundary = np.concatenate([[0], np.cumsum([sizes[g] for g in group])]).astype(np.int64)
    if by_class is None:
        by_class = groups.by_class = dict(
            cls_ptr = pointers(cls_rows, csizes) if G else torch.zeros(n, C + 1, dtype = torch.int32, device = dev),
            cls_col = cat(ccols, sum(csizes)), cls_succ = cat(csuccs, sum(csizes)), max_cls = max_cls,
            cls_order = i32(torch.cat(orders)[at]) if G else i32(torch.arange(C, device = dev).repeat(n, 1)))
    return TransitionTables(grouped = grouped, entry_ptr = pointers(entry_rows, sizes), entry_succ = cat(succs, base),
                            entry_x = cat(xs, base), pred_ptr = pointers(pred_rows, sizes), pred_col = cat(cols, base),
                            pred_x = cat(pxs, base), num_pairs = [pairs[g] for g in group] if grouped else [],
                            max_pairs = max_pairs,
                            enum_ptr = pointers(enum_rows, esizes) if enumerated else (pointers([], []) if grouped else None),
                            enum_e = cat(ees, ebase) if grouped else None,
                            enum_pair = cat(epairs, ebase) if grouped else None, max_enum = max_enum,
                            **by_class, entry_class = None if grouped else cat(clss, base),
                            pred_class = None if grouped else cat(pclss, base),
                            boundary_base = torch.from_numpy(per_boundary).to(dev), num_entries = int(per_boundary[-1]))
