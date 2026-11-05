"""
The tables of a layout's transitions (:mod:`.tables`) on the host, for an automaton with few transitions between its active
states. A new constraint's tables are small then, and a few dozen array operations on the device take longer than the work they
do: a launch and a wait each. The same tables, value for value, in compiled loops and one copy to the device
(``transition_tables_on_the_host_test.py``); past :data:`HOST_ENTRIES`, the device's array operations, which scale.
"""

from functools import cached_property
from typing import Optional, Tuple

import numpy as np
import torch

from ..common import one_copy
from .forms import EnumeratedTokens, TransitionTables
from .host_kernels import (_below, _boundaries, _by_class_tables, _by_second, _class_columns, _entries, _group_bases,
                           _group_table, _most_of_columns, _moves, _num_entries, _pairs)


#: the tables of a layout are built on the device past so many live entries of its distinct boundaries and rows of its per-boundary
#: tables (entries + n x (W + C)). MEASURED, the one-time cost of a new automaton, host against device: HMM-128/4096, 4,000 of them 2.5
#: against 6.4 ms, 56,000 4.2 against 7.8, 69,000 5.7 against 8.9, 84,000 7.2 against 7.8; HMM-512/1024, 150,000 to 200,000 6.6-8.2 against
#: 8.8-11.6; on no shape slower past the noise
HOST_ENTRIES = 1 << 18
#: ... or past so many (state, class) pairs, where finding its entries alone costs as much as the rest (4 ms at 194,000)
HOST_MOVES = 1 << 16
#: a table over every boundary of at most so many entries is laid out on the host; a larger one, from its distinct rows, on the device
HOST_TABLE_ENTRIES = 1 << 20


def _ones_row(a: np.ndarray) -> np.ndarray:
    """A table of nothing is one zero: a pointer to read."""
    return a if a.size else np.zeros(1, dtype = np.int32)


class HostTables:
    """
    What :class:`~.tables.BoundaryGroups` says of a layout's transitions, on the host, and the tables built from it. Offers the
    groups' fields the others read (``firsts``, ``group``, ``batches``, ``most``, ``holding``, :meth:`release`).

    :ivar g, q, c, s: the live entries, :func:`_entries`
    :ivar G: the groups
    """

    def __init__(self, tr, firsts, group, entries):
        self.n, self.W, self.C, self.dev = tr.n, tr.W, tr.C, tr.device
        self.tr_width = np.asarray(tr.width, dtype = np.int64)
        self.firsts, self.group_host = firsts.tolist(), group
        self.firsts_host = firsts
        self.group = group.tolist()
        self.G = firsts.size
        self.batches = [(0, self.G)] if self.G else []
        self.g, self.q, self.c, self.s = entries
        self.most = None
        self.holding = False
        self.host = tr.host
        ends_here = firsts + 1 == self.n
        self.ends_here = ends_here
        self.s_end = np.where(ends_here[self.g], 0, self.s).astype(np.int32) if ends_here.any() else self.s
        self.by_class = None
        self.pairs_host = None                                  # the grouped tables, for the signature layout
        self._enum_codes = None
        self.enum_lists = None                                  # the enumerated tokens' classes per boundary, on the host

    def release(self) -> None:
        self.holding = False

    @cached_property
    def at(self) -> torch.Tensor:
        """Every boundary's group on the device."""
        return torch.from_numpy(self.group_host).to(self.dev)

    @property
    def class_columns(self) -> torch.Tensor:
        return torch.from_numpy(self.class_columns_host()).to(self.dev)

    def class_columns_host(self) -> np.ndarray:
        kept = self.__dict__.get("_class_columns")
        if kept is None:
            kept = self.__dict__["_class_columns"] = _class_columns(self.g, self.c, self.G, self.C)
        return kept

    def entries(self, k: int, end: int):
        """``(g, q, c, s)`` of groups ``k .. end`` on the device, int64 (:meth:`~.tables.BoundaryGroups.entries`)."""
        keep = (self.g >= k) & (self.g < end)
        long = lambda a: torch.from_numpy(a[keep].astype(np.int64)).to(self.dev)
        return torch.from_numpy((self.g[keep] - k).astype(np.int64)).to(self.dev), long(self.q), long(self.c), long(self.s)

    def next_col_table(self) -> torch.Tensor:
        """:func:`~.tables.next_col_table`, on the host, or for a large table from its distinct rows on the device."""
        n, W, C, G = self.n, self.W, self.C, self.G
        if n * W * C <= HOST_TABLE_ENTRIES:
            return one_copy([_group_table(self.g, self.q, self.c, self.s, G, W, C)[self.group_host]], self.dev)[0]
        if G * W * C <= HOST_TABLE_ENTRIES:
            table = one_copy([_group_table(self.g, self.q, self.c, self.s, G, W, C)], self.dev)[0]
        else:
            g, q, c, s = self.entries(0, G)
            table = torch.full((G, W, C), -1, dtype = torch.int32, device = self.dev)
            table[g, q, c] = s.to(torch.int32)
        return table[self.at]

    def enumerated_tokens(self, classes) -> EnumeratedTokens:
        """:func:`~.tables.enumerated_tokens`, on the host: what the token classes say of them, as they do."""
        n, C = self.n, self.C
        stored = getattr(classes, "stored_host", None)
        etok = getattr(classes, "enumerated_host", None)
        if stored is None:
            stored, etok = classes.stored_of_class.cpu().numpy(), classes.enumerated_token.cpu().numpy()
        group = self.group_host
        code_rows = np.broadcast_to(stored, (self.G, C)).astype(np.int32) if self.G else np.zeros((0, C), dtype = np.int32)
        ptr = np.zeros(n + 1, dtype = np.int64)
        tok, index = np.zeros(1, dtype = np.int32), np.zeros(1, dtype = np.int32)
        if classes.num_enumerated > 0:
            place = np.cumsum(etok >= 0) - 1                    # an enumerated class's among them
            live = (self.class_columns_host() > 0) & (etok >= 0)
            count = live.sum(axis = 1)
            e = np.where(live, np.cumsum(live, axis = 1) - 1, count[:, None])
            code_rows = np.where(etok >= 0, -1 - e, stored).astype(np.int32)
            ptr[1:] = np.cumsum(count[group])
            lists_g, lists_c = np.nonzero(live)
            first = np.cumsum(count) - count
            lengths = count[group]
            total = int(lengths.sum())
            ids = np.repeat(first[group], lengths) + np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
            tok = np.concatenate([etok[lists_c[ids]], [0]]).astype(np.int32)
            index = np.concatenate([place[lists_c[ids]], [0]]).astype(np.int32)
        self._enum_codes = code_rows
        self.enum_lists = index, ptr
        code = code_rows[group]
        ptr_t, tok_t, index_t, code_t = one_copy([ptr.astype(np.int32), tok, index, code], self.dev)
        return EnumeratedTokens(ptr = ptr_t, tok = tok_t, index = index_t, code = code_t, num = int(ptr[-1]), ptr_host = ptr,
                                on = classes.num_enumerated > 0)

    def _pointers(self, rows: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """int32 [n, width]: every boundary's pointer row, its segment's relative row past the segments before."""
        bases = np.cumsum(counts) - counts
        return (rows + bases[:, None].astype(np.int32))[self.group_host]

    def transition_tables(self, grouped: bool, enum: Optional[EnumeratedTokens]) -> TransitionTables:
        """:func:`~.tables.transition_tables`, on the host."""
        n, W, C, G = self.n, self.W, self.C, self.G
        g, q, c, s = self.g, self.q, self.c, self.s_end
        codes = None
        if enum is not None and enum.on:
            codes = self._enum_codes if self._enum_codes is not None else enum.code.cpu().numpy()[self.firsts_host]
        arrays = []
        if self.by_class is None:
            rows, by_col, by_succ_, order, most_cls = _by_class_tables(g, q, c, s, G, C)
            sizes_c = np.diff(_group_bases(g, G))
            arrays += [self._pointers(rows, sizes_c), _ones_row(by_col), _ones_row(by_succ_), order[self.group_host].astype(np.int32)]
        enum_arrays = []
        max_pairs = max_enum = 0
        if grouped:
            pg, pq, ps, x = _pairs(g, q, s, G, W)
            found = np.diff(_group_bases(pg, G))
            column = _below(pg, pq, G, W)
            max_pairs, inner = _most_of_columns(column, self.ends_here)
            self.most = int(inner)
            eg, eq, es = pg, pq, ps
            ex = np.arange(pg.size, dtype = np.int32) - _group_bases(pg, G)[:-1].astype(np.int32)[pg]
            num_pairs = found[self.group_host].tolist()
            if codes is not None:
                en = codes[g, c] < 0
                ng, nq = g[en], q[en]
                sizes_e = np.diff(_group_bases(ng, G))
                enum_row = self._pointers(_below(ng, nq, G, W), sizes_e)
                enum_arrays = [enum_row, _ones_row((-1 - codes[ng, c[en]]).astype(np.int32)), _ones_row(x[en])]
                if ng.size:
                    max_enum = int(_most_of_columns(_below(ng, nq, G, W), np.zeros(G, dtype = np.bool_))[0])
            else:
                enum_arrays = [np.zeros((n, W + 1), dtype = np.int32), np.zeros(1, dtype = np.int32), np.zeros(1, dtype = np.int32)]
        else:
            eg, eq, es = g, q, s
            ex = c if codes is None else codes[g, c].astype(np.int32)
            num_pairs = []
        sizes = np.diff(_group_bases(eg, G))
        entry_row = self._pointers(_below(eg, eq, G, W), sizes)
        by_succ = _by_second(eg, es, G, W)
        pred_row = self._pointers(_below(eg[by_succ], es[by_succ], G, W), sizes)
        arrays += [entry_row, _ones_row(es), _ones_row(ex), pred_row, _ones_row(eq[by_succ]), _ones_row(ex[by_succ])] + enum_arrays
        if not grouped:
            arrays += [_ones_row(c), _ones_row(c[by_succ])]
        out = one_copy(arrays, self.dev)
        if grouped:
            self.pairs_host = dict(base = _group_bases(pg, G), row = _below(pg, pq, G, W), s = ps, x = ex, pred_x = ex[by_succ],
                                   enum = None if codes is None else dict(base = _group_bases(ng, G), row = _below(ng, nq, G, W),
                                                                          e = (-1 - codes[ng, c[en]]).astype(np.int32), pair = x[en]))
        if self.by_class is None:
            cls_ptr, cls_col, cls_succ, cls_order = out[:4]
            out = out[4:]
            self.by_class = dict(cls_ptr = cls_ptr, cls_col = cls_col, cls_succ = cls_succ, max_cls = int(most_cls), cls_order = cls_order)
        entry_ptr, entry_succ, entry_x, pred_ptr, pred_col, pred_x = out[:6]
        out = out[6:]
        per_boundary = np.concatenate([[0], np.cumsum(sizes[self.group_host])]).astype(np.int64)
        kw = dict(enum_ptr = None, enum_e = None, enum_pair = None) if not grouped else dict(enum_ptr = out[0], enum_e = out[1], enum_pair = out[2])
        return TransitionTables(grouped = grouped, entry_ptr = entry_ptr, entry_succ = entry_succ, entry_x = entry_x, pred_ptr = pred_ptr,
                                pred_col = pred_col, pred_x = pred_x, num_pairs = num_pairs, max_pairs = int(max_pairs), max_enum = max_enum,
                                **kw, **self.by_class, entry_class = None if grouped else out[0], pred_class = None if grouped else out[1],
                                boundary_base = torch.from_numpy(per_boundary).to(self.dev), num_entries = int(per_boundary[-1]))

    def sweep_tables(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(ptr, cols)`` of :class:`~..reachability.SweepTables`: every group's predecessors, by successor, then column."""
        G, W = self.G, self.W
        pg, pq, ps, _ = _pairs(self.g, self.q, self.s, G, W)
        rows = _below(pg, ps, G, W).astype(np.int64) + _group_bases(pg, G)[:-1, None]
        ptr = np.concatenate([rows.reshape(-1), [pg.size]]).astype(np.int64)
        return torch.from_numpy(ptr).to(self.dev), one_copy([pq[_by_second(pg, ps, G, W)]], self.dev)[0]

    def transition_stats(self) -> Tuple[int, int, int]:
        """:func:`~.tables.transition_stats`, on the host."""
        boundaries = np.bincount(self.group_host, minlength = self.G)
        live = int((np.diff(_group_bases(self.g, self.G)) * boundaries).sum())
        if self.most is None:
            pg, pq, _, _ = _pairs(self.g, self.q, self.s, self.G, self.W)
            self.most = int(_most_of_columns(_below(pg, pq, self.G, self.W), self.ends_here)[1])
        return live, int(self.tr_width[:self.n].sum()), self.most


def host_tables(tr) -> Optional[HostTables]:
    """The :class:`HostTables` of a layout's transitions ``tr`` if they are few and the layout has its host tables, else None;
    decided once, kept with them."""
    kept = tr.__dict__.get("_host_tables", False)
    if kept is False:
        kept = tr.__dict__["_host_tables"] = _build(tr)
    return kept


def _build(tr) -> Optional[HostTables]:
    if tr.host is None or tr.n == 0 or tr.n * (tr.W + tr.C) > HOST_ENTRIES or tr.K * tr.C > HOST_MOVES:
        return None
    delta, state_id, col_of_state = tr.host
    firsts, group = _boundaries(state_id, tr.n)
    ptr, cls, succ = _moves(delta, col_of_state)
    width = np.asarray(tr.width, dtype = np.int64)
    most = _num_entries(firsts, width, state_id, ptr)
    if most + tr.n * (tr.W + tr.C) > HOST_ENTRIES:
        return None
    return HostTables(tr, firsts, group, _entries(firsts, width, state_id, col_of_state, ptr, cls, succ, most))
