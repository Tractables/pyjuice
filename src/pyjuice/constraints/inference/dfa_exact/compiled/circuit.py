"""
What a backend's tables take from the circuit alone, whatever the constraint, kept with the compiled circuit: a new
constraint on a live circuit builds only what its automaton changes.

* :class:`CircuitRows` (at compilation, host): the rows and boundaries the lifted passes read for every node;
* :class:`CircuitTables` (at the first query, one per device): every product's folds into steps, every sum layer's dense
  runs and fused parts, the interval numbers, and what query plans read about them;
* :class:`Assembly` (one per pattern of which intervals keep every pair): those steps grouped into launches on the
  device. A launch takes the steps of one contraction whose intervals keep every pair alike, so the grouping is the only
  thing here an automaton can change, and automata mostly agree on it (on an HMM every interval is a prefix or a suffix,
  which keeps every pair, whatever the automaton).

What is left for a backend -- columns, slots, the transitions and the positions of the pairs -- is read off its layout in
a few array operations per launch (:class:`~.program.Program`, :func:`~.top_down_program.top_down_program`).

Nothing here is written after it is built, and queries only read it, so that backends share it. It lives as long as the
circuit (a :class:`weakref.WeakKeyDictionary`), which it does not keep alive; a circuit moved to another device builds the
device's tables again.
"""

from __future__ import annotations

import bisect
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from pyjuice.nodes.nodes import compilations

from ....errors import ConstraintCompileError
from ..external_params import block_scale
from ..kernels.common import IDENTITY, SUM, TEMP, OneCopy, settled
from ..kernels.prod import Launch
from ..kernels.sum import dense_groups
from ..kernels.transitions import transition_job

#: (input row, position) columns of the steps whose input's transitions they read
TRANS_COLS = {"input_block": (1, 2), "block_input": (4, 5)}

#: the patterns of intervals a circuit keeps assembled, the last used first out
ASSEMBLIES = 8


# -------------------------------------------------------------------------------------------------
# At compilation: the rows
# -------------------------------------------------------------------------------------------------

class CircuitRows:
    """
    The rows and boundaries the lifted passes read for every node of a compiled circuit (host; see
    :func:`~.tables.build_pc_tables`). Rows are taken from the PC's own layers (every layer numbers its node groups
    consecutively, in the order of its ``nodes`` list).

    :ivar numbered: every node group and its first row, to check against the node group's record of it
    :ivar product_rows: per product layer, ``(out_rows [R], child_rows [R, k], boundaries [R, k + 1])`` int64
    :ivar node_ivs: the intervals of every sum and product node group, each once, and ``node_ends`` those, int64
    :ivar sum_groups: per sum node group, in row order, ``(first row, end row, a, b)``, its interval ``[a, b)``;
        ``sum_rows`` int64 [groups, 2] their rows, ``sum_iv`` int64 [groups] their intervals' numbers in ``node_ivs``
    :ivar product_groups: per product layer group, ``(first row, end row, intervals of its node groups, each once)``;
        ``group_iv`` int64 their numbers in ``node_ivs``, a group's from ``group_ptr``
    :ivar input_range, root_rows: the rows of all input nodes, and of the roots
    """

    def __init__(self, structure, pc):
        info_of = {info.ns: info for info in structure.nodes}
        first_row, numbered = {}, []
        self._checked_at = None

        def number(layer, row):
            for ns in layer.nodes:
                first_row[ns] = row
                numbered.append((ns, row, row + ns.num_nodes))
                row += ns.num_nodes

        for layer in pc.input_layer_group:
            number(layer, layer._output_ind_range[0])
        for lg in pc.inner_layer_groups:
            for layer in lg.layers:
                number(layer, layer._layer_nid_range[0])
        self.numbered = numbered
        self.check()
        interval = lambda ns: (info_of[ns].scope_runs[0][0], info_of[ns].scope_runs[0][1] + 1)
        ivs = {}
        for info in structure.nodes:
            if info.kind != "input":
                (a, b), = info.scope_runs
                ivs[(a, b + 1)] = True
        self.node_ivs = list(ivs)
        self.product_rows = []
        for lg in pc.inner_layer_groups:
            if not lg.is_prod():
                continue
            for layer in lg.layers:
                parts = []
                for ns in layer.nodes:
                    (a, b), = info_of[ns].scope_runs
                    starts = [(info_of[cs].scope_runs[0][0], k) for k, cs in enumerate(ns.chs)]
                    starts.sort(key = lambda start: start[0])                 # (stable: as the node group lists them)
                    edges = ns.edge_ids.cpu().numpy()
                    child_rows = [first_row[ns.chs[k]] + _product_child_index(ns, k, edges) for _, k in starts]
                    parts.append((first_row[ns], ns.num_nodes, child_rows, [s for s, _ in starts] + [b + 1]))
                self.product_rows.append(_concat_padded(parts))
        self.sum_groups = sorted((first_row[ns], first_row[ns] + ns.num_nodes, *interval(ns))
                                 for lg in pc.inner_layer_groups if lg.is_sum() for layer in lg.layers for ns in layer.nodes)
        number = {iv: k for k, iv in enumerate(self.node_ivs)}
        self.sum_rows = np.asarray([(first, end) for first, end, _, _ in self.sum_groups], dtype = np.int64).reshape(-1, 2)
        self.sum_iv = np.asarray([number[(a, b)] for _, _, a, b in self.sum_groups], dtype = np.int64)
        self.product_groups = []
        for lg in pc.inner_layer_groups:
            if lg.is_prod():
                rows = [layer._layer_nid_range for layer in lg.layers]
                self.product_groups.append((min(r[0] for r in rows), max(r[1] for r in rows),
                                            list({interval(ns): True for layer in lg.layers for ns in layer.nodes})))
        self.node_ends = np.asarray(self.node_ivs, dtype = np.int64).reshape(-1, 2)
        self.group_iv = np.asarray([number[iv] for _, _, ivs in self.product_groups for iv in ivs], dtype = np.int64)
        self.group_ptr = np.cumsum([0] + [len(ivs) for _, _, ivs in self.product_groups], dtype = np.int64)
        ranges = [layer._output_ind_range for layer in pc.input_layer_group]
        self.input_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))
        self.root_rows = tuple(pc._root_node_range)
        self._on_device = (None, None)

    def check(self):
        """Refuses a circuit whose node groups no longer describe it: their record of their rows describes the circuit
        their graph was compiled into LAST, so a graph compiled again into another circuit is not read with this one's
        rows. Node groups' records change when a circuit is compiled: read again only if one was since the last look (they
        may be tens of thousands, and this runs for every constraint)."""
        epoch = compilations()
        if self._checked_at == epoch:
            return
        stale = sum(tuple(getattr(ns, "_output_ind_range", ())) != (first, end) for ns, first, end in self.numbered)
        if not stale:
            self._checked_at = epoch
        if stale:
            raise ConstraintCompileError(
                f"The PC's node groups no longer describe this circuit: {stale} of them record other rows than "
                f"the circuit's layers. This happens when the same node graph has been compiled again into "
                f"another circuit. Compile the constraint against the circuit compiled last, or compile a fresh "
                f"copy of the nodes.")

    def on_device(self, dev) -> list:
        """:attr:`product_rows` as int32 tensors on ``dev``, kept (the last device's)."""
        dev = torch.device(dev)
        if self._on_device[0] != dev:
            self._on_device = (dev, [tuple(t.to(dev, torch.int32) for t in rows) for rows in self.product_rows])
        return self._on_device[1]


def _product_child_index(ns, k: int, edges: np.ndarray) -> np.ndarray:
    """[num_nodes] the node of child ``k`` (within that child's group) that each product node multiplies (``edges``:
    ``ns.edge_ids``)."""
    if ns.is_block_sparse():
        # a block's nodes multiply the same offsets of the child's blocks
        return np.repeat(edges[:, k], ns.block_size) * ns.chs[k].block_size + np.tile(np.arange(ns.block_size), edges.shape[0])
    return edges[:, k]


def _concat_padded(parts):
    """The (out [R], child_rows [R, k], bounds [R, k + 1]) tensors of a product layer's node groups (``parts``: first row,
    nodes, per child its rows, boundaries), padded to the largest k with -1."""
    R = sum(p[1] for p in parts)
    k = max(len(p[2]) for p in parts)
    out = np.empty(R, dtype = np.int64)
    child = np.full((R, k), -1, dtype = np.int64)
    bounds = np.full((R, k + 1), -1, dtype = np.int64)
    at = 0
    for first, nodes, rows, bd in parts:
        out[at:at + nodes] = np.arange(first, first + nodes)
        for j, row in enumerate(rows):
            child[at:at + nodes, j] = row
        bounds[at:at + nodes, :len(bd)] = bd
        at += nodes
    return tuple(torch.from_numpy(t) for t in (out, child, bounds))


_ROWS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_TABLES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def circuit_rows(structure, pc) -> CircuitRows:
    """The :class:`CircuitRows` of a compiled circuit, kept with it; checked again at every call (see
    :meth:`CircuitRows.check`)."""
    rows = _ROWS.get(pc)
    if rows is None:
        rows = _ROWS[pc] = CircuitRows(structure, pc)
    else:
        rows.check()
    return rows


def circuit_tables(pc, structure, dev, **switches) -> "CircuitTables":
    """The :class:`CircuitTables` of a compiled circuit on ``dev`` under ``switches`` (what the tables depend on besides
    the circuit: :class:`CircuitTables`' keyword arguments), kept with it -- the last device's and switches' only."""
    key = (torch.device(dev), dense_groups_switch(), tuple(sorted(switches.items())))
    kept = _TABLES.get(pc)
    if kept is None or kept[0] != key:
        kept = _TABLES[pc] = (key, CircuitTables(pc, structure, circuit_rows(structure, pc), torch.device(dev), **switches))
    return kept[1]


def dense_groups_switch() -> int:
    """The dense path's smallest node block (:data:`~..kernels.sum.DENSE_MIN_BLOCK`): which sum parts are dense."""
    from ..kernels import sum as lifted_sum
    return lifted_sum.dense.DENSE_MIN_BLOCK


# -------------------------------------------------------------------------------------------------
# At the first query: the steps
# -------------------------------------------------------------------------------------------------

class Emit(NamedTuple):
    """The steps of one contraction of the products of a node group: ``fields`` [R, step fields] int64 (see
    :class:`~..kernels.prod.Launch`; NumPy), the products' interval (``product``, its number in ``CircuitTables.sum_ivs``), the
    outputs' interval ``[entry, exit)``, the operands' whose keeping every pair the kernels must know, and every interval
    its steps read or write (``reads``, the first its output; a ``block @ block`` step's, its boundary triple's)."""

    stage: int
    form: str
    kinds: tuple
    fields: np.ndarray
    product: int
    entry: int
    exit: int
    operands: tuple
    reads: tuple


class FoldLayer(NamedTuple):
    """One product layer's products folded into binary steps (:func:`fold_layer`): ``emits``, the stages' numbers,
    the scratch rows the layer needs and the intervals of what they hold."""

    emits: List[Emit]
    stages: List[int]
    num_temps: int
    temp_ivs: List[Tuple[int, int]]


def fold_groups(child: np.ndarray, bounds: np.ndarray, input_end: int) -> dict:
    """The rows of a product layer's table (``child`` [R, k], ``bounds`` [R, k + 1], -1 padded) that fold alike, in the
    order they first occur: ``{(whether each child is an input, the boundaries): [(first row, end row)]}`` -- the rows of
    a node group are a run, so the runs are few, whatever the rows."""
    real = child >= 0
    kinds = real.astype(np.int8) + (real & (child < input_end))              # 0 padding, 1 a sum, 2 an input
    change = np.ones(child.shape[0], dtype = bool)
    change[1:] = (kinds[1:] != kinds[:-1]).any(axis = 1) | (bounds[1:] != bounds[:-1]).any(axis = 1)
    starts = np.nonzero(change)[0].tolist()
    groups = {}
    for s, e in zip(starts, starts[1:] + [child.shape[0]]):
        k = int(real[s].sum())
        groups.setdefault((tuple((kinds[s, :k] == 2).tolist()), tuple(bounds[s, :k + 1].tolist())), []).append((s, e))
    return groups


def fold_layer(rows, firsts: np.ndarray, input_end: int, iv_number: dict) -> FoldLayer:
    """
    One product layer's table, compiled into steps. Rows that share their children's kinds and boundaries fold the same
    way, right to left: ``c_0 @ (c_1 @ (... @ c_{k-1}))``, every intermediate block in a scratch row, the last step
    writing the layer's ``element_mars`` row. Each step is one contraction of :mod:`..kernels.prod`, chosen by its
    operands; a stage holds the steps that read only earlier stages. (In NumPy: thousands of small array operations a
    circuit, which on the host a thread team of torch's would run many times slower on a busy machine.)
    """
    out_rows, child_rows, bounds = (t.numpy() for t in rows)
    groups = fold_groups(child_rows, bounds, input_end)
    emits = []
    num_temps, temp_ivs = 0, {}

    def block(rows_m):
        return ("block", SUM, rows_m, np.searchsorted(firsts, rows_m, side = "right") - 1)

    def scratch(R, entry, exit):
        nonlocal num_temps
        first = num_temps
        num_temps += R
        temp_ivs[(entry, exit)] = True
        return ("block", TEMP, np.arange(first, first + R, dtype = np.int64), np.zeros(R, dtype = np.int64))

    def emit(stage, form, kinds, fields, entry, exit, operands = (), reads = ()):
        emits.append(Emit(stage, form, kinds, np.stack(fields, axis = 1), product, entry, exit, tuple(operands),
                          ((entry, exit), *reads)))

    for (is_input, bd), runs in groups.items():
        k = len(is_input)
        product = iv_number[(bd[0], bd[k])]                          # the products' interval (their sums')
        if len(runs) == 1:                                           # (a node group: one run of rows)
            kids, outs = child_rows[runs[0][0]:runs[0][1]], out_rows[runs[0][0]:runs[0][1]]
        else:
            kids = np.concatenate([child_rows[s:e] for s, e in runs])
            outs = np.concatenate([out_rows[s:e] for s, e in runs])
        R = outs.size
        zeros = np.zeros(R, dtype = np.int64)
        full = lambda v: np.full(R, v, dtype = np.int64)
        if k == 1:
            if is_input[0]:                                          # the input's own block
                emit(0, "input_block", (IDENTITY, False), [outs, kids[:, 0], full(bd[0]), zeros, zeros,
                                                           full(bd[1]), zeros], bd[0], bd[1])
            else:                                                    # a product over one sum: its block as is
                _, kind, l_rows, l_regs = block(kids[:, 0])
                emit(0, "copy", (kind, False), [outs, l_rows, l_regs, full(bd[0]), full(bd[1])], bd[0], bd[1])
            continue
        acc = ("input", kids[:, k - 1], bd[k - 1]) if is_input[k - 1] else block(kids[:, k - 1])
        stage = 0
        for m in range(k - 2, -1, -1):
            dest = ("block", None, outs, zeros) if m == 0 else scratch(R, bd[m], bd[k])
            out_temp = m != 0
            if is_input[m]:
                if acc[0] == "input":                                # an input next to an input: materialize it
                    tmp = scratch(R, acc[2], bd[k])
                    emit(stage, "input_block", (IDENTITY, True), [tmp[2], acc[1], full(acc[2]), zeros, zeros,
                                                                  full(bd[k]), zeros], acc[2], bd[k])
                    stage += 1
                    acc = tmp
                emit(stage, "input_block", (acc[1], out_temp), [dest[2], kids[:, m], full(bd[m]), acc[2], acc[3],
                                                                full(bd[k]), zeros], bd[m], bd[k],
                     [(bd[m] + 1, bd[k])], [(bd[m] + 1, bd[k])])
            else:
                _, l_kind, l_rows, l_regs = block(kids[:, m])
                if acc[0] == "input":
                    emit(stage, "block_input", (l_kind, out_temp), [dest[2], l_rows, l_regs, full(bd[m]), acc[1],
                                                                    full(acc[2]), zeros], bd[m], bd[k],
                         [(bd[m], acc[2])], [(bd[m], acc[2])])
                else:
                    emit(stage, "block_block", (l_kind, acc[1], out_temp),
                         [dest[2], l_rows, l_regs, acc[2], acc[3], full(bd[m]), full(bd[m + 1]), full(bd[k])],
                         bd[m], bd[k], (), [(bd[m], bd[m + 1]), (bd[m + 1], bd[k])])
            stage += 1
            acc = dest if m != 0 else acc
    return FoldLayer(emits, sorted({e.stage for e in emits}), num_temps, list(temp_ivs))


def _stack_key(blocks, E: int, block_size: int):
    """The key of a dense run's :func:`~.sum.stacked_weights` -- ``(E, block size, first weights in row
    order)`` -- when two or more node blocks occupy consecutive rows (one product can write them all), else None."""
    nids = [nid for _, nid, _ in blocks]
    if len(blocks) < 2 or any(b - a != block_size for a, b in zip(nids, nids[1:])):
        return None
    return (E, block_size, tuple(p0 for _, _, p0 in blocks))


def _split_aliased(fused, alias_row, alias_reg, elem_first: int, dev):
    """A fused sum part's edges split in two: the children read in element_mars (left-packed, padded with element
    row 0, below every region) and the children that copy a sum, read in the sum's own row -- under a gate, each with
    its gate cell (``idx``, ``a_idx``) and child place (``lc``, ``a_lc``)."""
    nids, cids, pids, nb_reg, max_slots, _, gate = fused
    c, p = cids.long(), pids.long()
    inside = c >= elem_first
    element = (c - elem_first).clamp(min = 0, max = alias_row.numel() - 1)
    row = torch.where(inside, alias_row.to(dev)[element].long(), -1)
    is_alias = inside & (row >= 0)
    if not is_alias.any():
        return fused

    def pack(keep, *cols, pad):
        order = torch.argsort((~keep).to(torch.int8), dim = 1, stable = True)
        k = max(1, int(keep.sum(dim = 1).max()))                       # never an empty table
        out = []
        for col, v in zip(cols, pad):
            g = torch.gather(col, 1, order)[:, :k]
            out.append(torch.where(torch.gather(keep, 1, order)[:, :k], g, torch.full_like(g, v)))
        return [t.to(dev, torch.int32).contiguous() for t in out]

    reg = torch.where(is_alias, alias_reg.to(dev)[element].long(), 0)
    if gate is None:
        ce, pe = pack(~is_alias, c, p, pad = (0, 0))
        ar, ag, ap = pack(is_alias, row, reg, p, pad = (-1, 0, 0))
        return nids, ce, pe, nb_reg, max_slots, (ar, ag, ap), None
    g = gate["idx"].long()
    ce, pe, ge = pack(~is_alias, c, p, g, pad = (0, 0, 0))
    ar, ag, ap, ga, la = pack(is_alias, row, reg, p, g, block_scale.child_places(gate, c), pad = (-1, 0, 0, 0, 0))
    gate = dict(gate, idx = ge, a_idx = ga, a_lc = la, dense_run = None,
                uniform = block_scale.uniform_chunks(ge, ce >= elem_first),
                a_uniform = block_scale.uniform_chunks(ga, ar >= 0))
    return nids, ce, pe, nb_reg, max_slots, (ar, ag, ap), gate


class CircuitTables:
    """
    What :class:`~.program.Program` reads about a circuit on a device, whatever the automaton.

    :ivar attrs: the Program's attributes that are the circuit's alone
    :ivar steps: the layers in ``pc.inner_layer_groups`` order: ``("prod", product group, [FoldLayer])``, ``("sum",
        product group, [(block size, parts)])``, ``parts`` ``[(dense runs, E, fused)]`` as the Program has them, but for
        a fused part's ``max_slots``: the distinct sum regions of its node blocks
    :ivar flag_ivs: the intervals whose keeping every pair groups the steps into launches
    :ivar of: int32 [(n + 1) * (n + 1)] (host), the number of interval ``[a, b)`` at ``a * (n + 1) + b`` (-1 if unused)
    :ivar assemblies: pattern of those intervals -> its :class:`Assembly`
    """

    def __init__(self, pc, structure, rows: CircuitRows, dev, alias_copies: bool, input_block_groups: tuple):
        self.dev, self.n, self.rows = dev, structure.num_vars, rows
        self.input_block_groups = input_block_groups
        n = self.n
        input_start, input_end = rows.input_range
        self.input_end = input_end
        groups = rows.sum_groups
        firsts = [first for first, _, _, _ in groups]
        firsts_np = np.asarray(firsts, dtype = np.int64)
        region_of = lambda row: bisect.bisect_right(firsts, row) - 1
        reg_ivs = [(a, b) for _, _, a, b in groups]
        sum_ivs = sorted(set(reg_ivs))
        iv_number = {iv: k for k, iv in enumerate(sum_ivs)}
        iv_bounds = np.array(sum_ivs, dtype = np.int64).reshape(-1, 2)
        a, b = iv_bounds[:, 0], iv_bounds[:, 1]
        reg_iv_host = [iv_number[iv] for iv in reg_ivs]
        reg_iv_np = np.asarray(reg_iv_host, dtype = np.int64)
        operands = _operand_table(rows.product_rows, firsts_np, n, input_end, iv_number, len(sum_ivs))
        iv_children = [np.unique(reg_iv_np[regions]) for regions in operands]
        iv_depth = np.zeros(len(sum_ivs), dtype = np.int64)
        for k in np.argsort(iv_bounds[:, 1] - iv_bounds[:, 0], kind = "stable"):     # children first
            if iv_children[k].size:
                iv_depth[k] = 1 + int(iv_depth[iv_children[k]].max())
        reg_rows = np.asarray([end - first for first, end, _, _ in groups], dtype = np.int64)

        # the layers: the products' folds, and the sums' dense runs and fused parts
        self.steps, stack_uses = [], {}
        gate_cells = block_scale.cells(pc)
        prod_index = -1
        prod_layers = iter(rows.product_rows)
        for lg in pc.inner_layer_groups:
            if lg.is_prod():
                prod_index += 1
                layers = [fold_layer(next(prod_layers), firsts_np, input_end, iv_number) for _ in lg.layers]
                self.steps.append(("prod", prod_index, layers))
            else:
                elem_first = rows.product_groups[prod_index][0]
                layers = []
                for layer in lg.layers:
                    parts = []
                    gated = any(gl is layer for gl in gate_cells.layers)
                    for nids, cids, pids in zip(layer.partitioned_nids, layer.partitioned_cids, layer.partitioned_pids):
                        regs = np.searchsorted(firsts_np, nids.cpu().numpy(), side = "right") - 1
                        if not gated:
                            found, rest = dense_groups(nids, cids, pids, layer.block_size, elem_first)
                        else:
                            found, rest = {}, torch.arange(nids.numel(), device = nids.device)
                        dense = {}                                      # (region, first child row) -> node blocks
                        for c0, blocks in found.items():
                            for k, nid, p0 in blocks:
                                dense.setdefault((int(regs[k]), c0), []).append((k, nid, p0))
                        E = cids.size(1)
                        for rc, blocks in dense.items():                # and the key of their stacked weights
                            blocks.sort(key = lambda block: block[1])
                            key = _stack_key(blocks, E, layer.block_size)
                            dense[rc] = (blocks, key)
                            if key is not None:
                                stack_uses[key] = stack_uses.get(key, 0) + 1
                        fused = None
                        if rest.numel() > 0:
                            rest_regs = regs[rest.cpu().numpy()]
                            gate = None
                            nb_reg = torch.from_numpy(rest_regs.astype(np.int64)).to(dev)
                            # (every node block: the layer's own tables, no copy of a [K, E])
                            rn, rc, rp = (t.data if rest.numel() == nids.numel() else t[rest] for t in (nids, cids, pids))
                            if gated:
                                gate = block_scale.edge_tables(gate_cells, layer, rn, rc, elem_first, dev)
                                gate["dense_run"] = block_scale.dense_run(rn, rc, nb_reg, layer.block_size, elem_first)
                            fused = (rn.contiguous(), rc.contiguous(), rp.contiguous(), nb_reg, np.unique(rest_regs), None, gate)
                        parts.append((dense, cids.size(1), fused))
                    layers.append((layer.block_size, parts))
                self.steps.append(("sum", prod_index, layers))

        # a one-child product over a sum is a copy of the sum's block: the sum layer that reads it reads the sum's
        # row instead, unless a dense node block reads it -- the dense path needs its children's rows contiguous --
        # and the copy is not made. ``alias`` records, per product layer group, every element row's (row within its
        # sum region, region), row -1 where the element is not aliased; the fused sum parts carry their aliased
        # children as a separate edge list
        alias = {}
        for k in range(len(self.steps) - 1) if alias_copies else ():
            kind, prod_index, layers = self.steps[k]
            if kind != "prod" or self.steps[k + 1][:2] != ("sum", prod_index):
                continue
            first, end, _ = rows.product_groups[prod_index]
            dense_rows = np.zeros(end, dtype = bool)
            for _, parts in self.steps[k + 1][2]:
                for dense, E, _ in parts:
                    for (_, c0) in dense:
                        dense_rows[c0:c0 + E] = True
            alias_row = np.full(end - first, -1, dtype = np.int32)
            alias_reg = np.zeros(end - first, dtype = np.int32)
            for li, layer in enumerate(layers):
                emits = []
                for e in layer.emits:
                    if e.form != "copy" or e.kinds != (SUM, False):
                        emits.append(e)
                        continue
                    st = e.fields
                    keep = dense_rows[st[:, 0]]
                    regs = st[~keep, 2]
                    alias_row[st[~keep, 0] - first] = st[~keep, 1] - firsts_np[regs]
                    alias_reg[st[~keep, 0] - first] = regs
                    if keep.any():
                        emits.append(e._replace(fields = st[keep]))
                layers[li] = layer._replace(emits = emits)
            if (alias_row >= 0).any():
                alias_row, alias_reg = torch.from_numpy(alias_row), torch.from_numpy(alias_reg)
                alias[prod_index] = (alias_row, alias_reg)
                for _, parts in self.steps[k + 1][2]:
                    for pi, (dense, E, fused) in enumerate(parts):
                        if fused is not None:
                            parts[pi] = (dense, E, _split_aliased(fused, alias_row, alias_reg, first, dev))

        # where every interval a product reads or writes keeps its slots; the intervals themselves (host, and their
        # ends on the device), and per product layer group the intervals of its products, for compact storage (see
        # :mod:`..kernels.compact`)
        fold_layers = [(pi, layer) for kind, pi, layers in self.steps if kind == "prod" for layer in layers]
        ivs = set(sum_ivs)
        flags = {}
        for _, layer in fold_layers:
            for e in layer.emits:
                ivs.update(e.reads)
                if e.form in TRANS_COLS:
                    flags[(e.entry, e.exit)] = True
                    flags.update((iv, True) for iv in e.operands)
        all_ivs = sorted(ivs)
        self.flag_ivs = list(flags)
        iv_index = {iv: k for k, iv in enumerate(all_ivs)}
        self.of = np.full((n + 1) * (n + 1), -1, dtype = np.int32)
        for k, (x, y) in enumerate(all_ivs):
            self.of[x * (n + 1) + y] = k
        iv_ends_np = np.asarray(all_ivs, dtype = np.int64).reshape(-1, 2)
        reg_iv_all = np.asarray([iv_index[sum_ivs[k]] for k in reg_iv_host], dtype = np.int64)
        elem_ivs = {}
        for kind, prod_index, layers in self.steps:
            if kind == "prod":
                products = {e.product for layer in layers for e in layer.emits}
                elem_ivs[prod_index] = np.asarray(sorted(iv_index[sum_ivs[k]] for k in products), dtype = np.int64)
        # ... as one table, a row per group padded with one past the last interval (one without lanes, see
        # :func:`~..kernels.compact._group_widths`), and every group's rows and slots: a query's widths of every
        # group's rows in a few array operations
        groups_p = rows.product_groups
        elem_iv_pad = np.full((len(groups_p), max([1, *map(len, elem_ivs.values())])), len(all_ivs), dtype = np.int64)
        for pi, values in elem_ivs.items():
            elem_iv_pad[pi, :len(values)] = values
        root_first, root_end = pc._root_node_range
        self.attrs = dict(
            n = n, dev = dev, input_start = input_start, input_end = input_end, firsts = firsts, reg_rows = reg_rows,
            reg_first = torch.tensor(firsts, dtype = torch.long, device = dev), sum_ivs = sum_ivs, iv_number = iv_number,
            iv_bounds = iv_bounds, iv_inside = (a[None, :] >= a[:, None]) & (b[None, :] <= b[:, None]),   # [k, m]: m in k
            reg_iv_host = reg_iv_host, reg_iv_np = reg_iv_np,
            reg_iv = torch.tensor(reg_iv_host, dtype = torch.long, device = dev), operands = operands,
            iv_regions = np.split(np.argsort(reg_iv_np, kind = "stable"),
                                  np.cumsum(np.bincount(reg_iv_np, minlength = len(sum_ivs)))[:-1]),
            iv_children = iv_children, iv_depth = iv_depth,
            no_evidence = torch.full((1, n), -1, dtype = torch.int32, device = dev),   # a sample with nothing observed
            stack_uses = stack_uses, gate_cells = gate_cells, root_region = region_of(root_first),
            root_rows = (root_first, root_end), alias = alias,
            num_sms = torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 1,
            all_ivs = all_ivs, iv_index = iv_index, iv_ends_np = iv_ends_np,
            iv_ends = torch.from_numpy(iv_ends_np).to(dev, torch.int32), reg_iv_all = reg_iv_all, elem_ivs = elem_ivs,
            elem_iv_pad = elem_iv_pad,
            elem_rows = np.asarray([end - first for first, end, _ in groups_p], dtype = np.int64))
        self.of_t = torch.from_numpy(self.of).to(dev)
        self.assemblies: "OrderedDict[bytes, Assembly]" = OrderedDict()

    def assembly(self, reach) -> "Assembly":
        """The :class:`Assembly` of the pattern ``reach`` makes of :attr:`flag_ivs`: built once, the last used kept."""
        full = {iv: reach.keeps_every_pair(*iv) for iv in self.flag_ivs}
        pattern = np.asarray(list(full.values()), dtype = bool).tobytes()
        kept = self.assemblies.get(pattern)
        if kept is None:
            kept = self.assemblies[pattern] = Assembly(self, full)
            if len(self.assemblies) > ASSEMBLIES:
                self.assemblies.popitem(last = False)
        else:
            self.assemblies.move_to_end(pattern)
        return kept


def _operand_table(product_rows, firsts: np.ndarray, n: int, input_end: int, iv_number: dict, num_ivs: int) -> list:
    """Per sum interval (``sum_ivs``' numbers: a product's scope is its parent sum's), the sum regions, ascending (int64),
    where a product over it has a child -- whose per-sample inside values computing the product, or its children's
    top-down values, reads (see :func:`~..queries.backend.plans.read_regions`)."""
    found = []
    for _, child_rows, bounds in product_rows:
        child, bnd = child_rows.numpy(), bounds.numpy()
        k = (child >= 0).sum(axis = 1)
        ends, inverse = np.unique(bnd[:, 0] * (n + 1) + bnd[np.arange(k.size), k], return_inverse = True)
        iv = np.array([iv_number.get((int(e) // (n + 1), int(e) % (n + 1)), -1) for e in ends],
                      dtype = np.int64)[inverse.reshape(-1)]
        r, c = np.nonzero((child >= input_end) & (iv >= 0)[:, None])        # every sum child of a product
        found.append(iv[r] * firsts.size + np.searchsorted(firsts, child[r, c], side = "right") - 1)
    keys = np.unique(np.concatenate(found)) if found else np.zeros(0, dtype = np.int64)
    return np.split(keys % firsts.size, np.searchsorted(keys // firsts.size, np.arange(1, num_ivs)))


# -------------------------------------------------------------------------------------------------
# The launches of a pattern
# -------------------------------------------------------------------------------------------------

class Assembly:
    """
    A circuit's steps grouped into launches, under one pattern of which intervals keep every pair.

    :ivar layers: per product layer group (by number), per layer, ``(stages of Launch, num_temps, the intervals of what
        its scratch rows hold, by their numbers)``
    :ivar launches: every :class:`~..kernels.prod.Launch`, by its ``number``: in :attr:`layers`' order
    :ivar steps: the Program's steps: its product layers ``(stages, num_temps, layer's number)``, its sum layers the
        circuit's
    :ivar temp_iv, temp_ptr: int64: the intervals of what every product layer's scratch rows hold (CSR, by its number)
    :ivar part_a, part_b, part_iv, part_ptr: int64: the ends and numbers (``CircuitTables``' ``all_ivs``) of the launches'
        ``ends``, one launch's after another's in :attr:`layers`' order, launch ``k``'s from ``part_ptr[k]``
    :ivar plans: the query plans of its programs, as they are whatever the automaton (:mod:`..queries.backend.plans`)
    :ivar triple_of, skip_triples: every ``block @ block`` step's boundary triple, in the order the skip masks are built
    :ivar pos_steps: int64 [n] on the device, the input steps per position
    :ivar steps_host: the same on the host
    :ivar trans_job: the steps :func:`~..kernels.transitions.transition_masses` builds for, or None
    :ivar score, top_down, forward_shape, position_classes: built for the first program that asks
        (:func:`~..kernels.score.score_tables`, :func:`~.top_down_program.top_down_template`,
        :func:`~..queries.backend.planners._forward_shape`, :func:`~..kernels.transition_signatures.position_classes`)
    """

    def __init__(self, circuit: CircuitTables, full: Dict[Tuple[int, int], bool]):
        self.circuit = circuit
        n, dev = circuit.n, circuit.dev
        hosts = {}                                     # (product group, layer) -> stages of launches on the host
        for kind, pi, layers in circuit.steps:
            if kind != "prod":
                continue
            for li, layer in enumerate(layers):
                stages = {s: {} for s in layer.stages}
                for e in layer.emits:
                    kinds = e.kinds
                    if e.form in TRANS_COLS:           # which of the output's and operand's intervals keep every pair
                        kinds += (tuple(full[iv] for iv in ((e.entry, e.exit), *e.operands)),)
                    stages[e.stage].setdefault((e.form, kinds), []).append(e)
                hosts[(pi, li)] = [[[form, kinds, _cat_fields(parts), parts] for (form, kinds), parts in stages[s].items()]
                                   for s in layer.stages]
        launches = [launch for stages in hosts.values() for launches_ in stages for launch in launches_]

        # which (output tile, shared chunk) pairs every block @ block step can skip: one mask per boundary triple
        triples = {}
        for form, _, _, parts in launches:
            if form == "block_block":
                for e in parts:
                    triples.setdefault((e.entry, e.reads[1][1], e.exit), len(triples))
        self.triple_of, self.skip_triples = triples, list(triples)
        t = np.asarray(self.skip_triples, dtype = np.int64).reshape(-1, 3)        # (as :func:`~..kernels.prod.skip_masks_for` reads them)
        self.skip_table = np.stack([t[:, 0], t[:, 1], 0 * t[:, 0], t[:, 1], t[:, 2], 0 * t[:, 0]], axis = 1)

        # every input @ block / block @ input step's rank among its position's: which block of the boundary's masses
        # are its own (field 6, set here once)
        ins = [launch for launch in launches if launch[0] in TRANS_COLS]
        positions = [launch[2][:, TRANS_COLS[launch[0]][1]] for launch in ins]
        tcat = np.concatenate(positions) if positions else np.zeros(0, dtype = np.int64)
        steps_t = np.bincount(tcat, minlength = n).astype(np.int64)      # input steps per position
        order = np.argsort(tcat, kind = "stable")
        rank = np.empty_like(tcat)
        rank[order] = np.arange(tcat.size) - (np.cumsum(steps_t) - steps_t)[tcat[order]]
        rows, k = [], 0
        for launch, pos in zip(ins, positions):
            st = launch[2].copy()
            st[:, 6] = rank[k:k + pos.size]
            k += pos.size
            rows.append(st[:, [*TRANS_COLS[launch[0]], 6]])
            launch[2] = st
        self.steps_host = torch.from_numpy(steps_t)
        self.pos_steps = self.steps_host.to(dev)
        self.trans_job = transition_job(torch.from_numpy(np.concatenate(rows)), dev) if rows else None
        self.trans_rows_host = None if self.trans_job is None else self.trans_job[0].cpu().numpy()
        self.layers = {}
        number = circuit.attrs["iv_index"]
        tables = OneCopy()
        for kind, pi, layers in circuit.steps:
            if kind == "prod":
                self.layers[pi] = [([[self._launch(tables, *launch) for launch in launches_] for launches_ in hosts[(pi, li)]],
                                    layer.num_temps, np.asarray([number[iv] for iv in layer.temp_ivs], dtype = np.int64))
                                   for li, layer in enumerate(layers)]
        tables.upload(dev)
        self.launches = [t for layers in self.layers.values() for stages, _, _ in layers for launches_ in stages
                         for t in launches_]
        for k, t in enumerate(self.launches):
            t.steps, t.number = settled(t.steps), k
        ends = [iv for t in self.launches for iv in t.ends]
        counts = [len(t.ends) for t in self.launches]
        temps, self.steps = [], []
        for kind, pi, layers in circuit.steps:
            if kind == "prod":
                self.steps.append((kind, pi, []))
                for stages, num_temps, temp_ivs in self.layers.get(pi, ()):
                    self.steps[-1][2].append((stages, num_temps, len(temps)))
                    temps.append(temp_ivs)
            else:
                self.steps.append((kind, pi, layers))
        self.temp_iv = np.concatenate(temps) if temps else np.zeros(0, dtype = np.int64)
        self.temp_ptr = np.cumsum([0] + [t.size for t in temps], dtype = np.int64)
        self.plans = OrderedDict()
        number = circuit.attrs["iv_index"]
        self.part_a = np.asarray([a for a, _ in ends], dtype = np.int64)
        self.part_b = np.asarray([b for _, b in ends], dtype = np.int64)
        self.part_iv = np.asarray([number[iv] for iv in ends], dtype = np.int64)
        self.part_ptr = np.cumsum([0] + counts, dtype = np.int64)
        self.score = None
        self.top_down = None
        self.forward_shape = None
        self.position_classes = None

    def _launch(self, tables: OneCopy, form: str, kinds: tuple, rows: np.ndarray, parts: list) -> Launch:
        """The :class:`~..kernels.prod.Launch` of a stage's steps (``rows`` [S, fields] int64, of the emits ``parts``); a
        query plan runs a subset of them through index lists (:class:`~..kernels.prod.Cut`)."""
        circuit = self.circuit
        S = rows.shape[0]
        to_device = lambda a: tables.add(a.astype(np.int32))
        counts = [e.fields.shape[0] for e in parts]
        iv = np.repeat(np.asarray([e.product for e in parts], dtype = np.int64), counts)
        ends = list({(e.entry, e.exit): True for e in parts})
        part_of = None
        if len(ends) > 1:
            number = {end: k for k, end in enumerate(ends)}
            part_of = np.repeat(np.asarray([number[(e.entry, e.exit)] for e in parts], dtype = np.int64), counts)
        groups_host = None
        if form == "block_block":
            triples = np.asarray([self.triple_of[(e.entry, e.reads[1][1], e.exit)] for e in parts], dtype = np.int32)
            steps = (to_device(rows), to_device(np.repeat(triples, counts)))
        elif form in TRANS_COLS:
            groups = None                                               # input @ block steps, several to a program
            chunks = [(0, S, None)]                                     # (every step's masses at hand: one chunk)
            # (the step groups of the class-by-class kernels, built whatever the program's transitions: a query's
            # are its own, see `transitions`)
            sizes_g = circuit.input_block_groups
            if form == "input_block" and kinds[0] != IDENTITY and sizes_g and sizes_g[0] > 1:
                order, firsts = _step_groups(rows, [G for G in sizes_g if G > 1])
                rows, iv = rows[order], iv[order]
                part_of = None if part_of is None else part_of[order]
                groups = {G: to_device(f) for G, f in firsts.items()}
                groups_host = firsts
            steps = (to_device(rows), chunks, groups)
        else:
            steps = to_device(rows)
        index = circuit.attrs["iv_index"]                               # the outputs' intervals
        out_ivs = np.unique(np.asarray([index[iv_] for iv_ in ends], dtype = np.int64))
        return Launch(form, kinds, steps, torch.from_numpy(rows), torch.from_numpy(iv), ends,
                      None if part_of is None else torch.from_numpy(part_of), groups_host, out_ivs)


def _cat_fields(parts: List[Emit]) -> np.ndarray:
    return parts[0].fields if len(parts) == 1 else np.concatenate([e.fields for e in parts])


def _step_groups(st: np.ndarray, sizes):
    """An ``input @ block`` launch's steps (host rows, see :data:`~.prod.INPUT_BLOCK_FIELDS`): ``order``, which
    sorts them by their position and block end, and per group size ``G`` of ``sizes``, ``firsts`` [groups + 1]: the
    first (sorted) step of every group of at most ``G`` consecutive steps that share both, then the end."""
    key = st[:, 2] * (int(st[:, 5].max()) + 1) + st[:, 5]
    order = np.argsort(key, kind = "stable")
    key = key[order]
    k = np.arange(st.shape[0], dtype = np.int64)
    change = np.ones(st.shape[0], dtype = bool)
    change[1:] = key[1:] != key[:-1]
    run_first = np.maximum.accumulate(np.where(change, k, 0))               # the first step of each step's run
    end = np.asarray([st.shape[0]], dtype = np.int64)
    return order, {G: np.concatenate([np.flatnonzero((k - run_first) % G == 0), end]) for G in sizes}


