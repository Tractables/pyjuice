"""
Transition masses: for a missing token at a position, a column and one of its successors, the log-mass of the token
classes that lead there, from the class masses (:func:`transition_masses`). The products' ``input @ block`` and
``block @ input`` steps read them, so a missing token costs a gather per successor, not per class (see
:mod:`.prod`). Their tables (:class:`TransitionTables`) are built once per backend.

ENUMERATED CLASSES (see :class:`~pyjuice.constraints.distributions.TokenClasses`) have no class mass: where a kernel
read the mass of class ``c`` at row ``r`` it reads a CODE instead (:class:`EnumeratedTokens`): the class's stored
column, or ``-1 - e`` for the ``e``-th enumerated token live at the boundary, whose mass is its token's own
probability under the row's parameter row (``eprobs[prow[r], index]``, the enumerated tokens' probabilities, by
parameter row: a step's lanes read along one row, see :meth:`~..state.State.enumerated_probs`) times its per-sample
weight (``enum_logw``, see :class:`~..state.ClassMasses`) over the row's ``Z``.
"""

import math
import weakref
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ....distributions.categorical import _exact_dot
from .common import _next_col_of, grid_chunks



@dataclass(frozen = True)
class EnumeratedTokens:
    """
    The enumerated classes' tokens that are live at every boundary (lead somewhere from some active column), and
    every class's CODE at every boundary: its stored column, or ``-1 - e`` for the ``e``-th of the boundary's live
    enumerated tokens (a class dead at the boundary, enumerated: -1 - the count, matching no entry).

    :ivar ptr: [n + 1] int32, the boundary's tokens are ``tok[ptr[t] .. ptr[t + 1]]``; ``ptr_host`` the same on the host
    :ivar tok: [num + 1] int32, the tokens, then a 0 (kernels read a boundary's first token unmasked for a stored
        entry: past the last boundary's, this one)
    :ivar index: [num + 1] int32, every token's place among the automaton's enumerated tokens
        (``TokenClasses.enumerated_tokens``, in class order) -- its column of the enumerated tokens' probabilities
        (:meth:`~..state.State.enumerated_probs`) --, then a 0
    :ivar code: [n, C] int32
    :ivar num: the tokens over all boundaries (0: nothing enumerated; every code a stored column, the class itself)
    """

    ptr: torch.Tensor
    tok: torch.Tensor
    index: torch.Tensor
    code: torch.Tensor
    num: int
    ptr_host: np.ndarray


def enumerated_tokens(tr, classes) -> EnumeratedTokens:
    """The :class:`EnumeratedTokens` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`, on
    the device) under ``classes`` (a :class:`~pyjuice.constraints.distributions.TokenClasses`)."""
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


@dataclass(frozen = True)
class TransitionTables:
    """
    Every boundary's transitions as entries (column, successor, mass), the successor of a sequence end being its one
    column. Ungrouped, one entry per (column, class) that leads somewhere, its mass the class's mass -- read by the
    class's code (:class:`EnumeratedTokens`). Grouped, one per (column, successor) pair, its mass the pair's
    transition mass: the mass of every class that leads there (:func:`transition_masses`) -- a missing token then
    costs a gather per successor, not per class.

    :ivar grouped: which of the two
    :ivar entry_ptr: [n, W + 1] int32, the entries of column ``q`` at ``t``: ``entry_ptr[t, q] .. entry_ptr[t, q + 1]``
        (global numbers; ``entry_ptr[t, 0]``: the boundary's first)
    :ivar entry_succ, entry_x: [entries] int32, every entry's successor column and where its mass is: its class's
        code, or its pair's number within the boundary
    :ivar pred_ptr, pred_col, pred_x: the entries leading to column ``j`` at ``t + 1``, as a CSR (``pred_ptr``
        [n, W + 1]) into their columns and their mass indices
    :ivar num_pairs: grouped only, pairs per boundary (Python ints)
    :ivar max_pairs: grouped only, the most pairs of any column
    :ivar enum_ptr, enum_e, enum_pair: grouped only, the enumerated classes' live entries of every column, as a CSR
        (``enum_ptr`` [n, W + 1] int32, global numbers): the token's number ``e`` at its boundary and the pair it leads
        to; ``max_enum`` the most of any column
    :ivar cls_ptr, cls_col, cls_succ: the same (column, class) entries BY CLASS: the columns where class ``c`` leads
        somewhere at ``t``, ascending, ``cls_col[cls_ptr[t, c] .. cls_ptr[t, c + 1]]`` (``cls_ptr`` [n, C + 1] int32,
        global numbers), and their successors (a sequence end's: its one column) -- the class weights' walk over a
        class's columns (:func:`~.score._class_weight_kernel`); ``max_cls`` the most of any (boundary, class);
        ``cls_order`` [n, C] int32, every boundary's classes by their columns, most first (the walk's lanes take
        classes in this order, so that a program's classes have lists of a length)
    :ivar entry_class, pred_class: ungrouped only, [entries] int32: every entry's class, in the entry and the
        predecessor order -- by which a session's per-sample inputs rule entries out (the ``input @ block`` entry loops'
        :class:`~.live.SuccessorLists`, the CUDA top-down walk's terms)
    :ivar boundary_base, num_entries: boundaries alike share their entries (see :func:`transition_tables`), so a
        structure with a value PER BOUNDARY AND ENTRY -- a sample's successor lists -- counts the boundaries' entries
        one after the other: ``boundary_base`` int64 [n + 1] on the device, boundary ``t``'s entries are
        ``boundary_base[t] .. boundary_base[t + 1]`` there (its entry ``e`` at ``boundary_base[t] + e - entry_ptr[t, 0]``),
        ``num_entries`` their total, ``boundary_base[n]``
    """

    grouped: bool
    entry_ptr: torch.Tensor
    entry_succ: torch.Tensor
    entry_x: torch.Tensor
    pred_ptr: torch.Tensor
    pred_col: torch.Tensor
    pred_x: torch.Tensor
    num_pairs: list
    max_pairs: int
    enum_ptr: Optional[torch.Tensor] = None
    enum_e: Optional[torch.Tensor] = None
    enum_pair: Optional[torch.Tensor] = None
    max_enum: int = 0
    cls_ptr: Optional[torch.Tensor] = None
    cls_col: Optional[torch.Tensor] = None
    cls_succ: Optional[torch.Tensor] = None
    max_cls: int = 0
    cls_order: Optional[torch.Tensor] = None
    entry_class: Optional[torch.Tensor] = None
    pred_class: Optional[torch.Tensor] = None
    boundary_base: Optional[torch.Tensor] = None
    num_entries: int = 0


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


def boundary_groups(tr) -> BoundaryGroups:
    """The :class:`BoundaryGroups` of a layout's transitions, kept with them."""
    kept = tr.__dict__.get("_boundary_groups")
    if kept is None:
        kept = tr.__dict__["_boundary_groups"] = BoundaryGroups(tr)
    return kept


def next_col_table(tr) -> torch.Tensor:
    """int32 [n, W, C] on the device: every boundary's column at the next boundary of every (column, class), -1 where
    that is inactive or past its columns."""
    groups = boundary_groups(tr)
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


#: :func:`transition_masses`' tiles: the most rows (input steps at one position) and columns per program, classes per
#: product, warps. Rows and columns shrink until there are enough programs (they never change a result); the pairs
#: per product follow the widest column, up to 64.
#: :func:`_transition_kernel`'s tiles: TS rows a program (MEASURED on an HMM-4096 x hard_0 session at the pair tile 8:
#: 64 rows 320 ms a launch, 128 120, 256 285 -- a column's class successors and one-hot are per program, so more rows
#: amortize them until the registers run out; the rows' sums are the same whatever the tile), QT columns, TC classes
TRANSITION_TILES = dict(TS = 128, QT = 4, TC = 16, warps = 4, programs = 1024)
#: :func:`_transition_kernel`'s pair tile: None, the boundaries' mean pairs a column rounded up to a power of two
#: within [8, 64] -- a column's pairs are walked ``TP`` at a time, each pass over its masses and class successors
#: again, and the one-hot dot is ``TP`` wide, so a tile much wider than the columns' pairs is work on zeros (MEASURED
#: on an HMM-4096 x hard_0 session, 4.4 pairs a column: TP 64 321 ms a launch, 32 199, 16 178, 8 166, the same bits);
#: an int forces one (tests).
TRANSITION_PAIR_TILE = None


def transition_job(rows: torch.Tensor, dev):
    """
    The steps whose transition masses one :func:`transition_masses` call builds: ``rows`` [N, 3] (input row,
    position, the step's rank among its position's -- which block of the boundary's pairs, or of a sample's entries
    there, its masses are), as ``(rows, tiles, TS)`` on ``dev``, the rows ordered by position and ``tiles``
    [tiles, 3] (first row, number of rows, position) cutting them into tiles of at most ``TS`` rows at one position
    (fewer when there are few rows, so that there are enough programs).
    """
    TS = job_tile(rows.size(0))
    rows = rows.to(dev)
    rows = rows[torch.argsort(rows[:, 1], stable = True)]
    pos, counts = torch.unique_consecutive(rows[:, 1], return_counts = True)
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    return i32(rows), i32(job_tiles(pos, counts, TS)), TS


def job_tile(rows: int) -> int:
    """:func:`transition_job`'s most rows a tile, for so many rows."""
    return min(TRANSITION_TILES["TS"], max(16, host_next_power_of_2(rows) // 32))


def job_tiles(pos: torch.Tensor, counts: torch.Tensor, TS: int) -> torch.Tensor:
    """int64 [tiles, 3], ``(first row, number of rows, position)``: the rows of the positions ``pos`` (``counts`` of each,
    one position's after another's) cut into tiles of at most ``TS``."""
    per = (counts + TS - 1) // TS
    owner = torch.repeat_interleave(torch.arange(pos.numel(), device = pos.device), per)
    within = torch.arange(owner.numel(), device = pos.device) - (torch.cumsum(per, dim = 0) - per)[owner]
    first = (torch.cumsum(counts, dim = 0) - counts)[owner] + within * TS
    return torch.stack([first, (counts[owner] - within * TS).clamp(max = TS), pos[owner]], dim = 1)


@triton_jit
def _transition_kernel(class_mars, mass_row, mass_stride, trans, rows, tiles, delta, state_id, col_of_state,
                       entry_succ, entry_ptr, class_of_stored, enum_ptr, enum_e, enum_pair, tok_ptr, eidx, eprobs,
                       prow, lognorm, enum_logw, width, W, C, K, n, Cs, num_enum, logw_stride, input_start,
                       num_qtiles, pair_off, pairs, pid0, TS: tl.constexpr,
                       QT: tl.constexpr, TP: tl.constexpr, TC: tl.constexpr, TE: tl.constexpr, SAMPLE: tl.constexpr,
                       ENUM: tl.constexpr, LOGW: tl.constexpr):
    """
    One program per (tile of rows at one position, ``QT`` columns) -- ``SAMPLE``: and sample, for class masses of each
    sample's own (rows ``mass_row + b * mass_stride``; a row the same for every sample, stride 0, is built for sample
    0 alone; one set of masses is written either way): per column, its pairs ``TP`` at a time, as the product
    of the rows' class masses -- linear, scaled by 2^64, in column order over the ``Cs`` stored classes
    (``class_of_stored``: every column's class) -- with the 0/1 table of which of the pairs every class leads to: the
    class's successor column (the automaton's transitions ``delta`` through the boundaries' column maps ``state_id``
    and ``col_of_state``, a sequence end's being 0) against the pairs' (``entry_succ``, the boundary's entries being
    its pairs by column and successor). A class mass is a sum of fp32 probabilities, so 0 or at least 2^-149: scaled, a normal
    float, which tensor cores do not flush. ``ENUM``: then the column's enumerated entries (``enum_ptr`` / ``enum_e``
    / ``enum_pair``, see :class:`TransitionTables`), ``TE`` at a time, each the mass of its token: its probability
    under the row's parameter row (``eprobs[prow[row], eidx[e]]``, ``num_enum`` the table's row stride) times --
    ``LOGW`` -- its weight ``exp(enum_logw[b, e] - lognorm[row])`` (``logw_stride`` floats per sample). The log of the
    sum is the transition mass.
    """
    pid = pid0 + tl.program_id(0)
    tile = pid // num_qtiles
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    rk = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)   # the step's rank at t: its
    off = tl.load(pair_off + t) + rk * tl.load(pairs + t)                                # block of the boundary's pairs
    row = tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    b = tl.program_id(1).to(tl.int64)
    if SAMPLE:
        st = tl.load(mass_stride + (u - input_start), mask = rmask, other = 0).to(tl.int64)
        rmask = rmask & ((b == 0) | (st > 0))
        row = row + b * st
    masses = class_mars + row * Cs
    if ENUM:
        pr = tl.load(prow + row, mask = rmask, other = 0).to(tl.int64)
        if LOGW:
            lz = tl.load(lognorm + row, mask = rmask, other = 0.0)
            lw0 = enum_logw + b * logw_stride + tl.load(tok_ptr + t)
        tk0 = eidx + tl.load(tok_ptr + t)
    base = tl.load(entry_ptr + t * (W + 1))                                  # the boundary's first pair
    jj = tl.arange(0, TP)
    ee = tl.arange(0, TE)
    for qi in range(QT):
        q = (pid % num_qtiles) * QT + qi
        if q < tl.load(width + t):
            g_end = tl.load(entry_ptr + t * (W + 1) + q + 1) - base
            sq = tl.load(state_id + t * W + q)                                                   # the column's state
            for g0 in range(tl.load(entry_ptr + t * (W + 1) + q) - base, g_end, TP):
                acc = tl.zeros([TS, TP], dtype = tl.float32)
                sg = tl.load(entry_succ + base + g0 + jj, mask = g0 + jj < g_end, other = -2)   # the pairs' successors
                for c0 in range(0, Cs, TC):
                    c = c0 + tl.arange(0, TC)
                    cmask = c < Cs
                    x = tl.exp(tl.load(masses[:, None] + c[None, :], mask = rmask[:, None] & cmask[None, :],
                                       other = float("-inf")) + 44.3614195558365)                # x 2^64
                    cls = tl.load(class_of_stored + c, mask = cmask, other = 0)
                    sc = _next_col_of(delta, col_of_state, sq, cls, cmask, t, K, C)             # the classes' successors
                    sc = tl.where((t + 1 == n) & (sc >= 0), 0, sc)
                    acc += _exact_dot(x, tl.where(sc[:, None] == sg[None, :], 1.0, 0.0))
                if ENUM:                                                     # the column's enumerated entries
                    e_end = tl.load(enum_ptr + t * (W + 1) + q + 1)
                    for e0 in range(tl.load(enum_ptr + t * (W + 1) + q), e_end, TE):
                        em = e0 + ee < e_end
                        e = tl.load(enum_e + e0 + ee, mask = em, other = 0).to(tl.int64)
                        g = tl.load(tk0 + e, mask = em, other = 0).to(tl.int64)
                        x = tl.load(eprobs + pr[:, None] * num_enum + g[None, :], mask = rmask[:, None] & em[None, :],
                                    other = 0.0) * 18446744073709551616.0                        # x 2^64
                        if LOGW:
                            x = x * tl.exp(tl.load(lw0 + e, mask = em, other = float("-inf"))[None, :] - lz[:, None])
                        g = tl.load(enum_pair + e0 + ee, mask = em, other = -1) - g0
                        acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                tl.store(trans + off[:, None] + (g0 + jj)[None, :], tl.log(acc) - 44.3614195558365,
                         mask = rmask[:, None] & (g0 + jj < g_end)[None, :])


def enumerated_args(prog, masses) -> dict:
    """The keyword arguments a kernel reading class masses by code takes (see :class:`EnumeratedTokens` and
    :class:`~..state.ClassMasses`): ``code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, Cs, num_enum,
    logw_stride, ENUM, LOGW`` -- ``eidx`` the boundaries' tokens' columns of ``eprobs``, the enumerated tokens'
    probabilities by parameter row (``num_enum`` columns); unread placeholders where nothing is enumerated (``ENUM``
    False) or the masses are no sample's own (``LOGW`` False)."""
    en = prog.enum
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    return dict(code = en.code, tok_ptr = en.ptr, eidx = en.index, eprobs = eprobs, prow = masses.prow,
                lognorm = masses.lognorm if logw else masses.table,
                enum_logw = masses.enum_logw if logw else masses.table, Cs = masses.table.size(1),
                num_enum = eprobs.size(1), logw_stride = enum_logw_stride(masses), ENUM = enum, LOGW = logw)


def enum_logw_stride(masses) -> int:
    """How far apart the samples' rows of ``masses.enum_logw`` are: 0 when there is one row for every sample."""
    w = masses.enum_logw
    return 0 if w is None or w.size(0) == 1 else w.size(1)


def transition_masses(trans, job, masses, prog):
    """
    Write the transition masses of every step of ``job`` (from :func:`transition_job`): for each column ``q`` at the
    input's position ``t`` and each successor ``s``, ``log sum_{c: next_col[t, q, c] = s} exp(mass_r(c))`` (``r``:
    input ``u``'s row of the class masses ``masses``, a :class:`~..state.ClassMasses`, a stored class's mass from
    its table and an enumerated class's from its token; one set of masses for every sample -- a sample's own are
    :func:`sample_transition_masses`'), at ``trans[offset + p]`` for the pair's number ``p`` within the boundary,
    ``offset`` the step's block (``prog.pair_off[t] + rank * pairs[t]``).
    """
    rows, tiles, TS = job
    tt, cfg, en = prog.trans, TRANSITION_TILES, prog.enum
    W, C = prog.W, prog.C
    QT = cfg["QT"]
    while QT > 1 and tiles.size(0) * host_cdiv(max(prog.width), QT) < cfg["programs"]:
        QT //= 2
    num_qtiles = host_cdiv(max(prog.width), QT)
    if TRANSITION_PAIR_TILE is None:
        mean_pairs = sum(tt.num_pairs) / max(1, sum(int(w) for w in prog.width[:prog.n]))
        TP = min(64, max(8, host_next_power_of_2(max(1, math.ceil(mean_pairs)))))
    else:
        TP = TRANSITION_PAIR_TILE
    sample = masses.batch > 1
    stride = masses.stride if sample else masses.row
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _transition_kernel[(size, masses.batch)](
            masses.table, masses.row, stride, trans, rows, tiles, prog.delta, prog.state_id, prog.col_of_state,
            tt.entry_succ, tt.entry_ptr, prog.token_classes.class_of_stored32, tt.enum_ptr if enum else tt.entry_ptr,
            tt.enum_e if enum else tt.entry_ptr,
            tt.enum_pair if enum else tt.entry_ptr, en.ptr, en.index, eprobs, masses.prow,
            masses.lognorm if logw else masses.table, masses.enum_logw if logw else masses.table, prog.width_t, W, C,
            prog.K, prog.n, masses.table.size(1), eprobs.size(1), enum_logw_stride(masses), prog.input_start, num_qtiles,
            prog.pair_off, prog.pairs_t, first,
            TS = TS, QT = QT, TP = TP, TC = cfg["TC"], TE = min(64, max(16, host_next_power_of_2(max(tt.max_enum, 1)))),
            SAMPLE = sample, ENUM = enum, LOGW = logw, num_warps = cfg["warps"])


@dataclass
class SampleMasses:
    """
    Transition masses of each sample's own, over the sample's successor entries alone (:class:`~.live.SampleSuccessors`:
    the (column, successor) pairs some class the sample allows leads along) -- a session's soft evidence rules most
    classes out, so a sample's entries are few against the boundary's pairs. Sample ``b``'s masses of an input step at
    position ``t`` with rank ``r`` among the position's steps are the ``count[b, t]`` floats at ``base[b, t] + r *
    count[b, t]``, one per entry of its list at the boundary, in the list's order: the program entry ``g``'s at
    ``rank[b, g]`` (-1: none, see :attr:`~.live.SampleSuccessors.rank`).

    :ivar base: int64 [Be, n]; :ivar count: int32 [Be, n]; :ivar rank, ptr: the lists'
    :ivar cls_ptr, xcode, wsel: every entry of the sample's lists (its number there, see ``ptr``) and the classes the
        sample allows that lead along it, ``cls_ptr[b, e] .. cls_ptr[b, e + 1]`` of the flat int32 ``xcode`` /
        ``wsel`` (``cls_ptr`` int32 [Be, E + 1], one sample's after another's) -- what an entry's mass is the log-sum
        over (a session's inputs allow few of the hundreds of classes leading along a JSON automaton's entry): every
        class's code at the entry's boundary, resolved -- the stored column (>= 0), or for an enumerated class ``-1 -
        j``, ``j`` its token's column of ``enum_probs`` -- and its token's place among the enumerated tokens live at
        any boundary (its ``enum_logw`` column; -1 for a stored class)
    :ivar size: the floats of them all; :ivar Be: the samples with lists of their own (1: one set for every sample)
    :ivar version: the class masses' they were laid out for
    """

    base: torch.Tensor
    count: torch.Tensor
    rank: torch.Tensor
    ptr: torch.Tensor
    size: int
    Be: int
    version: int
    cls_ptr: Optional[torch.Tensor] = None                     # (built by sample_classes when the query takes the
    xcode: Optional[torch.Tensor] = None                       # layout: a session over the budget never pays for them)
    wsel: Optional[torch.Tensor] = None


def sample_masses(masses, succ, prog) -> SampleMasses:
    """The :class:`SampleMasses` layout of ``masses`` (a :class:`~..state.ClassMasses`) over the successor lists
    ``succ`` (its :class:`~.live.SampleSuccessors`) for ``prog``'s input steps: one wait for the device (the size);
    the entries' classes come with :func:`sample_classes`."""
    Be, n, W = succ.ptr.size(0), prog.n, prog.W
    ptr = succ.ptr.view(Be, n, W + 1)
    width = prog.width_t[:n].long().view(1, n, 1).expand(Be, n, 1)
    count = (ptr.gather(2, width)[..., 0] - ptr[:, :, 0]).contiguous()               # the sample's entries at t
    sizes = count.long() * prog.pos_steps[None, :]                                    # floats per (sample, position)
    base = (torch.cumsum(sizes.reshape(-1), dim = 0) - sizes.reshape(-1)).view(Be, n)
    return SampleMasses(base = base, count = count, rank = succ.rank, ptr = succ.ptr, size = int(sizes.sum()),
                        Be = Be, version = masses.version)


def _pair_tables(prog):
    """Of every (entry, class) pair of :func:`~.live.entry_classes`, built once per program: its class's code at its
    boundary resolved (``SampleMasses.xcode``) and its enumerated token's place (``SampleMasses.wsel``), int32 [P]."""
    kept = getattr(prog, "entry_class_tables", None)
    if kept is None:
        from .live import entry_classes
        n, C, dev, en = prog.n, prog.C, prog.dev, prog.enum
        key, first = entry_classes(prog)
        E = first.numel() - 1
        g = torch.repeat_interleave(torch.arange(E, device = dev), first[1:] - first[:-1])    # every pair's entry
        tb = torch.searchsorted(prog.succ.boundary_base[1:n + 1], g, right = True)        # ... its boundary
        c = key - tb * C                                                                  # ... its class
        x = en.code.view(-1)[key].long() if en.num > 0 else c                             # ... its code there
        e = (-1 - x).clamp(min = 0)
        at = en.ptr.long()[tb] + e if en.num > 0 else e
        enum = x < 0
        xcode = torch.where(enum, -1 - en.index.long()[at.clamp(max = en.index.numel() - 1)], x) if en.num > 0 else x
        wsel = torch.where(enum, at, -1)
        kept = prog.entry_class_tables = (xcode.to(torch.int32), wsel.to(torch.int32))
    return kept


def sample_classes(layout: SampleMasses, succ, prog) -> None:
    """Give ``layout`` every entry's allowed classes (``cls_ptr``, and their codes ``xcode`` / ``wsel``): the
    program's (entry, class) pairs (:func:`~.live.entry_classes`) whose class the sample allows
    (``succ.allowed``), by the entry's number in the sample's lists -- the samples :data:`~.live.SUPPORT_CHUNK` pairs
    at a time: the pairs by program entry, the sample's entries in program order
    (:attr:`~.live.SampleSuccessors.entry`), so an entry's classes end where the allowed pairs up to its program
    entry's last do. A wait for the device per chunk (its pairs' number)."""
    from .live import SUPPORT_CHUNK, entry_classes
    Be, n, C, dev = layout.Be, prog.n, prog.C, prog.dev
    key, first = entry_classes(prog)                                                  # (entry, class) pairs by entry
    xcode, wsel = _pair_tables(prog)
    E, P = succ.rank.size(1), key.numel()
    allowed = succ.allowed.view(Be, n * C)
    cls_ptr = torch.empty(Be, E + 1, dtype = torch.int32, device = dev)
    j = torch.arange(E, device = dev)
    base = torch.zeros(1, dtype = torch.int64, device = dev)                          # the samples' before
    picked, total = [], 0
    step = max(1, SUPPORT_CHUNK // max(P, 1))
    for b0 in range(0, Be, step):
        bs = min(step, Be - b0)
        keep = allowed[b0:b0 + bs, key] != 0                                          # the sample allows the class
        hits = torch.zeros(bs * P + 1, dtype = torch.int64, device = dev)             # the chunk's allowed pairs before
        hits[1:] = torch.cumsum(keep.view(-1), dim = 0)                               # each (one scan over the chunk:
        hits = hits.as_strided((bs, P + 1), (P, 1))                                   # a scan along rows runs a row a
        num = (succ.rank[b0:b0 + bs] >= 0).sum(dim = 1)                               # block); the sample's entries
        ends = hits.gather(1, first[succ.entry[b0:b0 + bs, :E].long() + 1])           # ... each one's last pair's
        ends = torch.where(j[None, :] < num[:, None], ends, hits[:, P:])
        cls_ptr[b0:b0 + bs, 0] = base + hits[:, 0]                                    # (samples one after another)
        cls_ptr[b0:b0 + bs, 1:] = base + ends
        base = base + hits[-1, P]
        picked.append(torch.nonzero(keep.view(-1)).squeeze(1) % P)                   # (in entry order: by entry)
        total += picked[-1].numel()
    assert total < 1 << 31, total                                                     # (int32 pointers, as the tables')
    pair = torch.cat(picked) if total else torch.zeros(1, dtype = torch.int64, device = dev)
    layout.cls_ptr = cls_ptr
    layout.xcode, layout.wsel = xcode[pair], wsel[pair]
