"""
``juice.decoder(pc, **inputs, **options)``: a session of queries of one circuit -- :meth:`Decoder.marginal` and
:meth:`Decoder.conditional`, with the arguments and results of :func:`pyjuice.queries.marginal` and
:func:`pyjuice.queries.conditional` -- that keeps, between its calls, the values the evidence left unchanged and
computes only what a call's changes reach (the plan of :mod:`.plan`)::

    with juice.decoder(pc) as decoder:
        for t in range(n):
            probs = decoder.conditional(x, missing_mask = x_missing, target_vars = [t])
            ...

Values live in the circuit's own buffers (``node_mars``, ``element_mars``, ``node_flows``, ``element_flows``);
the session adds one, ``node_out``, the log-outsides its conditionals computed. Any other forward or backward of the
circuit during the session overwrites those buffers: the session notices (the circuit's ``_num_runs``) and starts
over, as it does when the batch size changes. The per-sample inputs -- ``sum_external_params`` and the keywords the
input distributions read (``categorical_evidence_logp``, ...) -- are given when the session is made and hold for all
of it; external parameters are staged once, into a buffer of the session's own. The session opens a
:func:`pyjuice.fast_inference` scope (its options pass through): the circuit's parameters must not change within it.

A call runs its plan by the rows of its groups: an input layer by the runs of its nodes the call recomputes (where its
distribution allows it), an inner layer by the runs of its partitions' rows (`partition_runs`) that its plan's groups
hold -- and the products those read -- from a size on (`_GROUP_RUNS_MIN_WORK`), by whole layers below; a layer whose
groups the plan does not touch launches nothing. Its backward fills (-inf) only the flows its pass adds to. With
``early_stop`` (the default), a conditional starts its top-down pass from stored outsides, and a marginal may sum over
a cut instead of going up to the root: results agree with the queries' to float rounding. Without it, a marginal is
the root's value and a conditional's pass starts at the root from its inside, as a full backward does: marginals are
the queries', to the bit, and conditionals too up to the atomics their kernels accumulate with.

Cuts read stored outsides in absolute terms, and outsides are only as exact as the backward that computed them: at a
reduced precision (TF32 / bf16 dots, the external-parameter backwards) a backward's flows leak mass with depth --
MEASURED on a chain of 24 steps, ~6e-4 per layer, cut marginals 1.5e-2 off at the bottom, against 1.9e-6 at
``precision = "fp32"``. A conditional normalizes that out; a cut cannot. So marginals take cuts only on a circuit
compiled with ``precision = "fp32"`` and without external-parameter layers (whose backwards have no fp32 form), and
go up to the root otherwise.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Optional, Sequence

import numpy as np
import torch

from pyjuice.layer import ExternalParamsSumLayer
from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_KWARG
from pyjuice.utils.fast_inference import fast_inference

from .. import distributions
from .plan import INPUT, PROD, SUM, Planner


FAST_INFERENCE_OPTIONS = ("allow_param_copy", "allow_bf16_params", "allow_cudagraph")

# The session's own slab for staged external parameters (`TensorCircuit._stage_external_params`)
_STAGING = "decoder_external_params"


def decoder(pc, early_stop: bool = True, **options) -> "Decoder":
    """A decoder session over ``pc`` (see :mod:`pyjuice.queries.decoder.session`), used as a context manager.

    :param early_stop: let marginals sum over cuts and conditionals start from stored outsides (results agree with the
        queries' to float rounding); False: results are the queries', to the bit
    :param options: the per-sample inputs (``sum_external_params``, the input distributions' keywords), held for the
        whole session, and :func:`pyjuice.fast_inference`'s options
    """
    return Decoder(pc, early_stop = early_stop, **options)


def _input_keywords(pc) -> set:
    names = {EXTERNAL_PARAMS_KWARG}
    for layer in pc.input_layer_group:
        names.update(k for k in layer.dist.call_kwargs if not k.endswith("_grad"))
    return names


class _Layout:
    """Where the planner's groups are in the compiled circuit, as arrays -- over the groups:

    * ``lo`` / ``hi`` [G]: every sum and input group's node rows (-1 for a product: its values live in
      ``element_mars``, which every product layer rewrites);
    * ``layer_of`` [G]: the layer holding the group;

    and over the layers, numbered bottom-up (the input layers, then the layers of every inner layer group in order):

    * ``kind`` / ``group`` / ``index``: the layer's kind (INPUT / PROD / SUM), its inner layer group (-1 for an input
      layer) and its index there;
    * ``child_prods[l]``: for a sum layer, the layers of the product layer group below that hold its groups' children;
    * ``gated[l]``: an external-parameter sum layer; ``nid_range[l]``: a sum layer's node rows."""

    def __init__(self, pc, planner: Planner):
        s = planner.structure
        G = len(s.groups)
        index = {id(ns): g for g, ns in enumerate(s.groups)}
        self.lo = np.full([G], -1, dtype = np.int64)
        self.hi = np.full([G], -1, dtype = np.int64)
        for g, ns in enumerate(s.groups):
            if s.kind[g] != PROD:
                self.lo[g], self.hi[g] = ns._output_ind_range

        self.layer_of = np.full([G], -1, dtype = np.int64)
        kind, group, layer_index, layers = [], [], [], []
        for k, lg in [(-1, pc.input_layer_group)] + list(enumerate(pc.inner_layer_groups)):
            for i, layer in enumerate(lg):
                for ns in layer.nodes:
                    assert id(ns) in index, "a compiled layer holds a node group the circuit's graph does not"
                    self.layer_of[index[id(ns)]] = len(layers)
                kind.append(INPUT if k < 0 else (SUM if lg.is_sum() else PROD))
                group.append(k)
                layer_index.append(i)
                layers.append(layer)
        assert (self.layer_of >= 0).all(), "the circuit's graph and its compiled layers disagree"
        self.kind, self.group, self.index = (np.array(a, dtype = np.int64) for a in (kind, group, layer_index))

        self.gated = np.array([isinstance(layer, ExternalParamsSumLayer) for layer in layers])
        self.has_external = bool(self.gated.any())
        self.nid_range = [layer.nid_range if kind[l] == SUM else None for l, layer in enumerate(layers)]
        self.child_prods = [[] for _ in layers]
        for g in np.flatnonzero(s.kind == SUM):
            l = self.layer_of[g]
            for c in s.ch_idx[s.ch_ptr[g]:s.ch_ptr[g + 1]]:
                lc = int(self.layer_of[c])
                if self.kind[lc] == PROD and lc not in self.child_prods[l]:
                    self.child_prods[l].append(lc)
        for l in range(len(layers)):
            self.child_prods[l].sort()

        # Input layers: where their nodes start, how many they are, and whether their forward can compute some only
        self.input_start = np.zeros([len(layers)], dtype = np.int64)
        self.input_size = np.zeros([len(layers)], dtype = np.int64)
        self.runs_ok = np.zeros([len(layers)], dtype = bool)
        for l, layer in enumerate(layers):
            if kind[l] == INPUT:
                start, end = layer._output_ind_range
                self.input_start[l], self.input_size[l] = start, end - start
                self.runs_ok[l] = distributions.takes_node_runs(layer)

        # Group runs: where every group's rows lie in the partition tables of the inner layers' passes (`_RowIndex`)
        self.is_sum = s.kind == SUM
        self.ch_ptr, self.ch_idx = s.ch_ptr, s.ch_idx
        elo, ehi = np.full([G], -1, dtype = np.int64), np.full([G], -1, dtype = np.int64)
        for g, ns in enumerate(s.groups):                            # products: their element rows
            if s.kind[g] == PROD:
                elo[g], ehi[g] = ns._output_ind_range

        # The rows of the flow buffers outside every group -- the dummy rows the tables' padding points at, whose
        # flows a pass reads (a product's backward adds its padded parents' element flows): -inf, as a full backward
        # leaves them, before every pass
        prods = s.kind == PROD
        self.outside_nodes = _complement(self.lo[~prods], self.hi[~prods], pc.num_nodes)
        self.outside_elements = _complement(elo[prods], ehi[prods], pc.num_elements)
        self.is_prod = s.kind == PROD
        self.modules = layers
        self.elo, self.ehi = elo, ehi
        self.layer_groups = [np.flatnonzero(self.layer_of == l) for l in range(len(layers))]
        self.layer_children = [np.unique(np.concatenate([s.ch_idx[s.ch_ptr[g]:s.ch_ptr[g + 1]] for g in gs]))
                               if kind[l] != INPUT else gs[:0] for l, gs in enumerate(self.layer_groups)]
        self.num_edges = np.array([0 if kind[l] == INPUT else layer.num_edges for l, layer in enumerate(layers)],
                                  dtype = np.int64)
        # Per kind of inner pass, the layers whose tables have a partition of more than one row (their shapes: nothing
        # read back) -- those a pass could compute some rows of (`row_index`)
        self.splittable = {}
        for name in ("sum_fw", "sum_bk", "prod_fw", "prod_bk"):
            want = SUM if name.startswith("sum") else PROD
            self.splittable[name] = [l for l, layer in enumerate(layers) if kind[l] == want and not self.gated[l]
                                     and any(t.numel() > 1 for t in _pass_tables(layer, name, host = False))]
        self._entries = {}

    def row_index(self, layers_by_pass: dict) -> dict:
        """Per kind of inner pass, the `_RowIndex` of the layers ``layers_by_pass[name]`` (ascending): every layer's
        entries made once -- its tables read back -- and kept; a layer whose rows are not one group's each left out."""
        out = {}
        for name, ls in layers_by_pass.items():
            entries, sizes, groups = [], {}, {}
            for l in ls:
                found = self._entries.get((name, l))
                if found is None:
                    found = self._entries[(name, l)] = self._layer_entries(name, l)
                if found:
                    entries += found[0]
                    sizes.update(found[1])
                    groups.update(found[2])
            out[name] = _RowIndex(entries, sizes, groups)
        return out

    def _layer_entries(self, name: str, l: int):
        """``(entries, sizes, groups)`` of layer ``l``'s tables of a pass (`_RowIndex`), or () if a row lies in no
        single group."""
        # The groups the pass writes the rows of: the layer's own (forward), its groups' children (backward)
        gs = self.layer_groups[l] if name.endswith("fw") else self.layer_children[l]
        lo, hi = (self.elo[gs], self.ehi[gs]) if name in ("sum_bk", "prod_fw") else (self.lo[gs], self.hi[gs])
        tables = _pass_tables(self.modules[l], name)
        found = [_table_entries(starts, width, lo, hi, gs) for starts, width in tables]
        if any(f is None for f in found):
            return ()
        entries, sizes, groups = [], {}, {}
        for p, f in enumerate(found):
            sizes[(l, p)] = len(tables[p][0])
            groups[(l, p)] = np.array(sorted({g for g, _, _ in f}), dtype = np.int64)
            entries += [(g, l, p, a, b) for g, a, b in f]
        return entries, sizes, groups

    def layers(self, groups, kind):
        """The layers of ``kind`` holding any of ``groups``, ascending."""
        ls = np.unique(self.layer_of[groups])
        return ls[self.kind[ls] == kind]

    def by_group(self, ls):
        """``ls`` (ascending layers) split into runs of one inner layer group each, in order."""
        if ls.size == 0:
            return []
        return np.split(ls, np.flatnonzero(np.diff(self.group[ls])) + 1)

    def children(self, ls):
        """The product layers holding the children of sum layers ``ls`` (one layer group's), ascending."""
        if len(ls) == 1:
            return self.child_prods[ls[0]]
        return sorted(set(c for l in ls for c in self.child_prods[l]))


class _Rows:
    """A call's sets of node rows -- each the rows of some sum and input groups, in the order of the groups given --
    prepared on the host and uploaded in one copy. A set lying in one run of rows is a slice, with nothing to upload;
    any other an index tensor, expanded on the device from its runs. A set with ``segments`` (one per group) gives with
    its rows every row's segment. ``meta``: what the call notes of its sets."""

    def __init__(self, layout: _Layout):
        self.L = layout
        self.ranges = {}                         # name -> (lo [R], hi [R], segment [R] or None)
        self.sets = {}
        self.meta = {}

    def add_ranges(self, name, lo, hi) -> None:
        """A set given as row ranges ``[lo, hi)`` (ascending, disjoint) rather than groups."""
        if lo.size > 0:
            self.ranges[name] = (lo, hi, None)

    def add(self, name, groups, segments = None) -> None:
        lo, hi = self.L.lo[groups], self.L.hi[groups]
        if lo.size == 0:
            return
        if segments is None:                     # ranges that continue one another merged into runs, in order
            start = np.ones([lo.size], dtype = bool)
            start[1:] = lo[1:] != hi[:-1]
            first = np.flatnonzero(start)
            lo, hi = lo[first], hi[np.append(first[1:], lo.size) - 1]
        self.ranges[name] = (lo, hi, segments)

    def upload(self, device) -> None:
        self.sets = {}
        multi = []
        for name, (lo, hi, seg) in self.ranges.items():
            if lo.size == 1 and seg is None:
                self.sets[name] = slice(int(lo[0]), int(hi[0]))
            else:
                multi.append(name)
        if not multi:
            return
        # A set of up to `_HOST_ROWS` rows goes up whole, expanded on the host; a larger one as its runs, expanded on
        # the device (its rows would cost more to upload)
        parts, plan = [], []
        for name in multi:
            lo, hi, seg = self.ranges[name]
            lens = hi - lo
            total = int(lens.sum())
            starts = lo - (np.cumsum(lens) - lens)
            if total <= _HOST_ROWS:
                parts.append(np.repeat(starts, lens) + np.arange(total))
                if seg is not None:
                    parts.append(np.repeat(seg, lens))
                plan.append((name, True, lo.size, total, seg is not None))
            else:
                parts += [starts, lens] + ([] if seg is None else [seg])
                plan.append((name, False, lo.size, total, seg is not None))
        packed = torch.from_numpy(np.concatenate(parts).astype(np.int64)).to(device)
        pos = 0
        for name, host, R, total, segmented in plan:
            if host:
                rows = packed[pos:pos + total]
                pos += total
                if segmented:
                    rows = (rows, packed[pos:pos + total])
                    pos += total
            else:
                starts, lens = packed[pos:pos + R], packed[pos + R:pos + 2 * R]
                pos += 2 * R
                rows = torch.repeat_interleave(starts, lens, output_size = total) + torch.arange(total, device = device)
                if segmented:
                    rows = (rows, torch.repeat_interleave(packed[pos:pos + R], lens, output_size = total))
                    pos += R
            self.sets[name] = rows

    def get(self, name):
        return self.sets.get(name)


# The most rows of a set `_Rows` uploads expanded on the host (beyond, its runs, expanded on the device)
_HOST_ROWS = 1 << 16

# The most runs of an input layer's nodes a call recomputes on their own (beyond, the whole layer, in one launch)
_MAX_INPUT_RUNS = 4

# Restricting work to what a call needs costs host time; below these sizes the work spared costs less. MEASURED
# (brezel, HMM 256 x 256, B 16, left to right): finding a call's input runs and launching them took ~20 us more than the
# whole layer, and its fills ~60 us more than filling both flow buffers -- the GPU time of a whole input layer of
# ~2^21 node values, and of filling ~2^24 floats. Input runs: from `_RUNS_MIN_VALUES` node values (nodes x B) of an
# input layer; fills restricted: from `_FILL_MIN_VALUES` flows in the two buffers.
_RUNS_MIN_VALUES = 1 << 21
_FILL_MIN_VALUES = 1 << 24

# A flow buffer whose rows to fill are more than this fraction of it is filled whole (one launch)
_FILL_WHOLE_FRACTION = 0.5


# The most row ranges of a flow buffer a pass fills one by one (beyond, through one index)
_MAX_FILL_SLICES = 4


def _union(lo: np.ndarray, hi: np.ndarray):
    """The union of row ranges ``[lo, hi)``, as ascending disjoint ``(lo, hi)``."""
    if lo.size == 0:
        return lo, hi
    order = np.argsort(lo, kind = "stable")
    lo, hi = lo[order], np.maximum.accumulate(hi[order])
    start = np.ones([lo.size], dtype = bool)
    start[1:] = lo[1:] > hi[:-1]
    first = np.flatnonzero(start)
    return lo[first], hi[np.append(first[1:], lo.size) - 1]


def _complement(lo: np.ndarray, hi: np.ndarray, n: int):
    """The rows of ``[0, n)`` outside the ranges ``[lo, hi)``, as ``[(a, b)]``."""
    lo, hi = _union(lo, hi)
    starts, ends = np.r_[0, hi], np.r_[lo, n]
    keep = ends > starts
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _pass_tables(layer, name, host: bool = True):
    """``[(row starts, row width)]``, per partition of a pass of ``layer`` (`_Layout.row_index`'s ``name``): the first
    node (element) each row of its tables writes, on the host, and how many it writes -- or (``host`` False) the
    tables of row starts themselves, on the device."""
    tables = layer.partitioned_nids if name in ("sum_fw", "prod_fw") else \
             layer.partitioned_chids if name == "sum_bk" else layer.partitioned_u_cids
    if not host:
        return list(tables)
    widths = layer.cs_block_sizes if name == "sum_bk" else [layer.block_size] * len(tables)
    return [(t.cpu().numpy().astype(np.int64), int(w)) for t, w in zip(tables, widths)]


def _table_entries(starts, width: int, lo: np.ndarray, hi: np.ndarray, gs: np.ndarray):
    """``[(group, first row, end row)]``: a partition table's rows (row ``r`` writing ``[starts[r], starts[r] +
    width)``) in runs of one group each -- group ``gs[i]`` holds rows ``[lo[i], hi[i])``. None if the rows do not
    ascend or one lies in no single group (the layer's pass then runs whole)."""
    if starts.size == 0:
        return []
    if (np.diff(starts) <= 0).any():
        return None
    order = np.argsort(lo, kind = "stable")
    lo, hi, gs = lo[order], hi[order], gs[order]
    k = np.searchsorted(lo, starts, side = "right") - 1
    if (k < 0).any() or (starts + width > hi[np.maximum(k, 0)]).any():
        return None
    g = gs[k]
    first = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    end = np.append(first[1:], starts.size)
    return list(zip(g[first].tolist(), first.tolist(), end.tolist()))


class _RowIndex:
    """For one kind of inner pass -- ``sum_fw`` / ``prod_fw`` (a layer's groups), ``sum_bk`` / ``prod_bk`` (their
    children's flows) --, where the groups' rows lie in the partition tables of the layers that can compute some of
    their rows: entries ``(group, layer, partition, first row, end row)`` ascending by layer, partition and row;
    ``sizes[(layer, partition)]``, its rows; ``groups[(layer, partition)]``, the groups with rows in it; ``parts[layer]``,
    its partitions."""

    def __init__(self, entries, sizes, groups):
        e = np.array(entries, dtype = np.int64).reshape(-1, 5)
        self.group, self.layer, self.part, self.lo, self.hi = (np.ascontiguousarray(e[:, i]) for i in range(5))
        self.sizes, self.groups = sizes, groups
        self.layers = np.unique(self.layer)
        self.parts = {}
        for l, p in sizes:
            self.parts[l] = max(self.parts.get(l, 0), p + 1)

    def runs(self, need: np.ndarray) -> dict:
        """``{(layer, partition): [(start, end)]}``: the rows of the groups ``need`` marks ([G] bool), in runs."""
        idx = np.flatnonzero(need[self.group])
        if idx.size == 0:
            return {}
        lay, part, lo, hi = self.layer[idx], self.part[idx], self.lo[idx], self.hi[idx]
        first = np.flatnonzero(np.r_[True, (lay[1:] != lay[:-1]) | (part[1:] != part[:-1]) | (lo[1:] != hi[:-1])])
        last = np.append(first[1:], idx.size) - 1
        out = {}
        for l, p, a, b in zip(lay[first].tolist(), part[first].tolist(), lo[first].tolist(), hi[last].tolist()):
            out.setdefault((l, p), []).append((a, b))
        return out


# Group runs: a layer's pass computes only the rows a call needs (`partition_runs`) from this much work (edges x B):
# below, finding and launching them costs more host time than the GPU time they spare
_GROUP_RUNS_MIN_WORK = 1 << 26

# The most runs of a partition's rows a pass launches one by one; beyond, or past this fraction of its rows, the whole
# partition (one launch)
_MAX_PARTITION_RUNS = 4
_PARTITION_WHOLE_FRACTION = 0.5


def _get(t: torch.Tensor, rows):
    return t[rows]


def _copy(t: torch.Tensor, rows) -> torch.Tensor:
    """``t``'s ``rows``, copied (a slice of ``t`` is a view of it)."""
    return t[rows].clone() if isinstance(rows, slice) else t[rows]


def _put(t: torch.Tensor, rows, value) -> None:
    if isinstance(rows, slice):
        t[rows] = value
    elif isinstance(value, torch.Tensor):
        t.index_copy_(0, rows, value)
    else:
        t.index_fill_(0, rows, value)


class Decoder:
    """A decoder session; see :func:`decoder`."""

    def __init__(self, pc, early_stop: bool = True, **options):
        self.pc = pc
        # Cuts and stored outsides need one marginal per sample: a circuit of several root nodes has one per root
        self.early_stop = early_stop and pc.num_root_nodes == 1
        self._fast_inference = {k: options.pop(k) for k in FAST_INFERENCE_OPTIONS if k in options}
        allowed = _input_keywords(pc)
        unknown = sorted(k for k in options if k not in allowed)
        if unknown:
            raise TypeError(f"{', '.join(f'`{k}`' for k in unknown)}: not a per-sample input of this circuit (those "
                            f"are {', '.join(f'`{k}`' for k in sorted(allowed))}) nor an option of the session.")
        self._inputs = {k: v for k, v in options.items() if v is not None}
        self._scope = None
        self._used = False
        self._planner = None
        self._layout = None
        self._batch = None                       # the batch size the session's state is for (None: no state)

    # ---------------------------------------------------------------------------------------------
    # The session
    # ---------------------------------------------------------------------------------------------

    def __enter__(self) -> "Decoder":
        if self._used:
            raise RuntimeError("A decoder is opened once; make another with `juice.decoder(pc)`.")
        if self.pc.__dict__.get("_open_decoder") is not None:
            raise RuntimeError("This circuit already has an open decoder; close it first.")
        self._used = True
        self._scope = ExitStack()
        self._scope.enter_context(fast_inference(**self._fast_inference))
        self.pc.__dict__["_open_decoder"] = self
        return self

    def __exit__(self, *exc) -> bool:
        try:
            self._scope.close()
        finally:
            self._scope, self._batch, self._node_out, self._previous = None, None, None, None
            self.pc.__dict__["_open_decoder"] = None
        return False

    def reset(self) -> None:
        """Forget what the session kept: the next call computes its whole answer."""
        self._check_open()
        self._batch = None

    def _check_open(self):
        if self._scope is None:
            raise RuntimeError("Use the decoder inside its `with` block: `with juice.decoder(pc) as decoder: ...`.")

    def _sync(self, batch_size: int) -> None:
        """Make the session's state that of ``batch_size``, starting over if it is not -- or if another pass of the
        circuit ran since the session's last call."""
        pc = self.pc
        if self._planner is None:
            self._planner = Planner(pc.root_ns)
            self._layout = _Layout(pc, self._planner)
        if self._batch == batch_size and getattr(pc, "_num_runs", 0) == self._runs:
            return

        pc._init_buffer(name = "node_mars", shape = (pc.num_nodes, batch_size), set_value = 0.0)
        pc._init_buffer(name = "element_mars", shape = (pc.num_elements, batch_size), set_value = -float("inf"))
        self._node_out = torch.full([pc.num_nodes, batch_size], -float("inf"), device = pc.node_mars.device)
        self._node_out[pc._root_node_range[0]:pc._root_node_range[1], :] = 0.0
        kwargs = dict(self._inputs)
        pc._check_external_params_kwargs(kwargs)
        pc._stage_external_params(kwargs, batch_size, name = _STAGING)
        self._kwargs = kwargs
        self._planner.reset()
        self._previous = None
        self._log_p = None                       # [B]: log p of the last call's evidence, the next seeds' scale
        self._batch = batch_size
        self._runs = getattr(pc, "_num_runs", 0)

        # Group runs at this batch size: per kind of inner pass, the layers that compute only the rows a call needs
        # (`_GROUP_RUNS_MIN_WORK`; their row index made only then), and which partitions of a sum layer's forward can
        # (`SumLayer.forward_runs_ok`)
        L = self._layout
        big = L.num_edges * batch_size >= _GROUP_RUNS_MIN_WORK
        self._row_index = L.row_index({name: [l for l in ls if big[l]] for name, ls in L.splittable.items()})
        self._runs_on = {}
        for name, ix in self._row_index.items():
            on = np.zeros([big.size], dtype = bool)
            on[ix.layers] = True
            self._runs_on[name] = on
        alg = pc.default_propagation_alg
        self._fw_ok = {}
        for l in np.flatnonzero(self._runs_on["sum_fw"]).tolist():
            k = int(L.group[l])
            self._fw_ok[l] = pc.inner_layer_groups[k][int(L.index[l])].forward_runs_ok(
                batch_size, alg if isinstance(alg, str) else alg[k], 1.0, pc.node_mars.is_cuda)
        self._group_runs = bool(any(on.any() for on in self._runs_on.values()))

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``log p(e)`` of every sample, as :func:`pyjuice.queries.marginal`: ``[B, root nodes]``."""
        self._check_open()
        pc = self.pc
        miss = self._prepare(data, missing_mask)
        # Cuts read stored outsides in absolute terms: only where a backward computes them exactly (see the module)
        cuts = self.early_stop and pc.precision == "fp32" and not self._layout.has_external
        plan = self._planner.marginal(cuts = cuts)
        kwargs = dict(self._kwargs, missing_mask = missing_mask)
        rows = _Rows(self._layout)
        rows.add("cut", plan.cut)
        rows.upload(pc.node_mars.device)
        self._forward(plan.recompute, data, kwargs)
        if plan.cut.size == 0:
            lls = pc.node_mars[pc._root_node_range[0]:pc._root_node_range[1], :].permute(1, 0).clone()
        else:
            cut = rows.get("cut")
            lls = torch.logsumexp(_get(pc.node_mars, cut) + _get(self._node_out, cut), dim = 0)[:, None]
        if self.early_stop:
            self._log_p = lls[:, 0].clone()
        self._planner.commit(plan)
        self._remember(data, miss)
        return lls

    def conditional(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                    target_vars: Optional[Sequence[int]] = None) -> torch.Tensor:
        """``P(x_t = v | e)`` of every target, as :func:`pyjuice.queries.conditional`."""
        self._check_open()
        pc = self.pc
        miss = self._prepare(data, missing_mask)
        targets = list(range(pc.num_vars)) if target_vars is None else list(target_vars)
        # Stored outsides seed the pass once the scale of its flows is known (`_backward`); before, the pass starts at
        # the root from its inside, as a full backward does
        reuse = self.early_stop and self._log_p is not None
        plan = self._planner.conditional(targets, reuse = reuse)
        kwargs = dict(self._kwargs, missing_mask = missing_mask, target_vars = target_vars)
        rows = self._conditional_rows(plan, targets, reuse)
        self._forward(plan.recompute, data, kwargs)
        if reuse:
            scale = self._log_p
        else:
            scale = pc.node_mars[pc._root_node_range[0]:pc._root_node_range[1]].logsumexp(dim = 0)
        self._backward(plan, kwargs, scale, reuse, rows)

        if self.early_stop:
            visited = rows.get("visited")
            if visited is not None:
                _put(self._node_out, visited,
                     _get(pc.node_flows, visited) - _get(pc.node_mars, visited) + scale[None, :])
            log_p = self._normalize_target_flows(rows, len(targets))
            if log_p is not None:
                self._log_p = log_p + scale

        out = distributions.backward(pc.input_layer_group[0], data, pc.node_flows, pc.node_mars,
                                     **dict(kwargs, logspace_flows = True))
        self._planner.commit(plan, outsides = self.early_stop)
        self._remember(data, miss)
        return out

    # ---------------------------------------------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------------------------------------------

    def _prepare(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Sync the session to the call's batch and tell the planner which variables the call changed. Returns the
        call's missing mask, [B, V]."""
        B, V = data.size(0), data.size(1)
        self._sync(B)
        if missing_mask is None:
            miss = torch.zeros([B, V], dtype = torch.bool, device = data.device)
        elif missing_mask.dim() == 1:
            miss = missing_mask[None, :].expand(B, V)
        else:
            miss = missing_mask
        if self._previous is None or self._previous[0].shape != data.shape:
            changed = np.arange(V)
        else:
            x0, m0 = self._previous
            differ = data != x0
            if differ.dim() == 3:                                    # soft evidence: a variable's whole slice
                differ = differ.any(dim = 2)
            changed = np.flatnonzero(((miss != m0) | (~miss & differ)).any(dim = 0).cpu().numpy())
        self._planner.observe(changed)
        return miss

    def _remember(self, data: torch.Tensor, miss: torch.Tensor) -> None:
        """Keep a copy of the call's evidence (never the caller's tensors), in the session's own buffers."""
        prev = self._previous
        if prev is not None and prev[0].shape == data.shape and prev[0].dtype == data.dtype:
            prev[0].copy_(data)
            prev[1].copy_(miss)
        else:
            self._previous = (data.clone(), miss.clone())

    def _conditional_rows(self, plan, targets: Sequence[int], reuse: bool) -> _Rows:
        """The row sets a conditional reads and writes, uploaded at once: per layer group, the sum seeds whose outsides
        start the pass there (the root's not, when the pass starts from its inside); the input seeds; and with early
        stop, the visited groups (whose outsides are stored) and the targets' input groups (one segment per target)."""
        s, L = self._planner.structure, self._layout
        rows = _Rows(L)
        seeds = plan.seeds
        sums = seeds[s.kind[seeds] == SUM]
        if not reuse:
            sums = sums[sums != s.root]
        ks = L.group[L.layer_of[sums]]
        for k in np.unique(ks):
            rows.add(("seeds", int(k)), sums[ks == k])
        rows.add("input_seeds", seeds[s.kind[seeds] == INPUT])

        # The layers the pass runs (for `_backward` too), the rows they compute (`_backward_plan`), and the flows to fill
        # (-inf) before it: the visited groups' node flows, which the product layers add to -- every parent of a visited
        # group is visited or a seed, so whatever a pass reads unwritten (whole partitions and layers read their other
        # rows too) feeds only groups it does not visit, and nothing reads those groups' flows --, an external-parameter
        # layer's own, whose backward may read all of them, and the dummy rows outside every group, which padding reads
        # (`_Layout.outside_nodes` / `outside_elements`); of the element flows only those. Small buffers are filled whole
        # (`_FILL_MIN_VALUES`, `_FILL_WHOLE_FRACTION`); a few ranges one launch each, more through an index uploaded with
        # the call's other rows.
        sum_layers = L.layers(np.concatenate([seeds, plan.visited]), SUM)
        prod_layers = L.layers(plan.visited, PROD)
        rows.meta["sum_layers"], rows.meta["prod_layers"] = sum_layers, prod_layers
        rows.meta["bk"] = self._backward_plan(plan, sum_layers, prod_layers)
        rows.meta["fill_nodes"] = rows.meta["fill_elements"] = "all"
        if (self.pc.num_nodes + self.pc.num_elements) * self.pc.node_mars.size(1) >= _FILL_MIN_VALUES:
            rows.meta["fill_elements"] = L.outside_elements
            vis = plan.visited[~L.is_prod[plan.visited]]
            other = [tuple(L.nid_range[l]) for l in sum_layers[L.gated[sum_layers]].tolist()] + L.outside_nodes
            lo, hi = _union(np.concatenate([L.lo[vis], np.array([a for a, _ in other], dtype = np.int64)]),
                            np.concatenate([L.hi[vis], np.array([b for _, b in other], dtype = np.int64)]))
            if int((hi - lo).sum()) > _FILL_WHOLE_FRACTION * self.pc.num_nodes:
                pass                                                 # most of the buffer: fill all of it
            elif lo.size <= _MAX_FILL_SLICES:
                rows.meta["fill_nodes"] = list(zip(lo.tolist(), hi.tolist()))
            else:
                rows.meta["fill_nodes"] = None
                rows.add_ranges("fill_nodes", lo, hi)
        if self.early_stop:
            rows.add("visited", plan.visited[s.kind[plan.visited] != PROD])
            t = np.asarray(targets, dtype = np.int64)
            counts = s.inp_ptr[t + 1] - s.inp_ptr[t]
            groups = s.inp_idx[np.arange(int(counts.sum())) + np.repeat(s.inp_ptr[t] - (np.cumsum(counts) - counts), counts)]
            if t.size == 1:
                rows.add("targets", groups)
            else:
                seg = np.repeat(np.arange(t.size), counts)
                rows.add("targets", groups, segments = seg)
                per = np.bincount(seg, weights = L.hi[groups] - L.lo[groups], minlength = t.size)
                if per.size > 0 and per[0] > 0 and (per == per[0]).all():
                    rows.meta["target_rows"] = int(per[0])                # every target that many rows
        rows.upload(self.pc.node_mars.device)
        return rows

    def _normalize_target_flows(self, rows: _Rows, num_targets: int) -> Optional[torch.Tensor]:
        """Make the targets' flows a full backward's. The pass ran with ``log p`` taken as its scale (an estimate),
        so a node's flow is ``out * in / exp(scale)`` rather than ``out * in / p``; a target's input nodes are a cut
        (every induced tree has one leaf of the variable), so their flows sum to ``p / exp(scale)`` -- divided out per
        target and sample. The input layers' conditionals normalize with a small additive guard (``1e-12``), which
        flows of another scale would not survive. Returns that ``log (p / exp(scale))``, of the first target, [B]."""
        r = rows.get("targets")
        if r is None:
            return None
        nf = self.pc.node_flows
        if num_targets == 1:
            # One segment: its maximum and its result broadcast, not gathered row by row (the same values)
            vals = _get(nf, r)
            seg = torch.zeros([vals.size(0)], dtype = torch.long, device = vals.device)
            m = vals.amax(dim = 0, keepdim = True)
            m = torch.where(torch.isfinite(m), m, 0.0)
            tot = torch.zeros_like(m).index_add_(0, seg, torch.exp(vals - m))
            log_p = torch.where(tot > 0, m + torch.log(tot), 0.0)            # evidence of probability 0: as it is
            _put(nf, r, vals - log_p)
            return log_p[0]
        rows_t, seg = r
        vals = nf[rows_t]
        B = vals.size(1)
        if rows.meta.get("target_rows") is not None:
            m = vals.view(num_targets, -1, B).amax(dim = 1)                  # every target's rows together, in order
        else:
            m = torch.full([num_targets, B], -float("inf"), dtype = vals.dtype, device = vals.device)
            m = m.scatter_reduce(0, seg[:, None].expand(-1, B), vals, reduce = "amax")
        m = torch.where(torch.isfinite(m), m, 0.0)
        tot = torch.zeros([num_targets, B], dtype = vals.dtype, device = vals.device) \
                   .index_add_(0, seg, torch.exp(vals - m[seg]))
        log_p = torch.where(tot > 0, m + torch.log(tot), 0.0)
        _put(nf, rows_t, vals - log_p[seg])
        return log_p[0]

    # ---------------------------------------------------------------------------------------------
    # Execution, by whole layers
    # ---------------------------------------------------------------------------------------------

    def _forward(self, recompute: np.ndarray, data: torch.Tensor, kwargs: dict) -> None:
        """Recompute the insides of the groups ``recompute``: the layers holding any of them (a sum layer with the
        layers of the product layer group below that hold its children), bottom-up -- with group runs, only the rows
        of those groups and of the products they read (`_forward_plan`)."""
        pc, L = self.pc, self._layout
        if recompute.size == 0:
            return
        for l in L.layers(recompute, INPUT):
            runs = self._input_runs(l, recompute)
            distributions.forward(pc.input_layer_group[L.index[l]], data.clone() if data.dim() == 3 else data,
                                  pc.node_mars, **(kwargs if runs is None else dict(kwargs, node_runs = runs)))

        alg = pc.default_propagation_alg
        sum_opts = dict(kwargs, **pc.propagation_alg_kwargs, precision = pc.precision, pflow_temperature = 1.0)
        sum_layers = L.layers(recompute, SUM)
        if not self._group_runs:                                     # whole layers, as before group runs
            for ls in L.by_group(sum_layers):
                k = int(L.group[ls[0]])
                lg, below = pc.inner_layer_groups[k], pc.inner_layer_groups[k - 1]
                for p in L.children(ls):
                    below[L.index[p]].forward(pc.node_mars, pc.element_mars)
                layer_alg = alg if isinstance(alg, str) else alg[k]
                for l in ls:
                    lg[L.index[l]].forward(pc.node_mars, pc.element_mars, pc.params, propagation_alg = layer_alg, **sum_opts)
            return
        sum_runs, prod_runs = self._forward_plan(recompute, sum_layers)
        for ls in L.by_group(sum_layers):
            k = int(L.group[ls[0]])
            lg, below = pc.inner_layer_groups[k], pc.inner_layer_groups[k - 1]
            for p, runs in prod_runs.get(k, []):
                below[L.index[p]].forward(pc.node_mars, pc.element_mars, partition_runs = runs)
            layer_alg = alg if isinstance(alg, str) else alg[k]
            for l in ls:
                runs = sum_runs.get(int(l))
                opts = sum_opts if runs is None else dict(sum_opts, partition_runs = runs)
                lg[L.index[l]].forward(pc.node_mars, pc.element_mars, pc.params, propagation_alg = layer_alg, **opts)

    @staticmethod
    def _whole(runs, size: int) -> bool:
        """Whether a partition computes all of its rows rather than ``runs`` of them: too many runs, or most rows."""
        return len(runs) > _MAX_PARTITION_RUNS or sum(b - a for a, b in runs) > _PARTITION_WHOLE_FRACTION * size

    def _children(self, need: np.ndarray) -> np.ndarray:
        """[G] bool: the children of the groups ``need`` marks."""
        L = self._layout
        gs = np.flatnonzero(need)
        out = np.zeros([need.size], dtype = bool)
        if gs.size > 0:
            lens = L.ch_ptr[gs + 1] - L.ch_ptr[gs]
            pos = np.repeat(L.ch_ptr[gs] - (np.cumsum(lens) - lens), lens) + np.arange(int(lens.sum()))
            out[L.ch_idx[pos]] = True
        return out

    def _partition_runs(self, name: str, l: int, found: dict, whole: Optional[np.ndarray] = None):
        """Layer ``l``'s `partition_runs` for a pass (`_Layout.row_index`'s ``name``) from the runs ``found``: a
        partition of many runs or most of its rows whole (None) -- its groups marked in ``whole`` when given."""
        index = self._row_index[name]
        parts = []
        for p in range(index.parts[l]):
            runs = found.get((l, p), [])
            if runs and self._whole(runs, index.sizes[(l, p)]):
                if whole is not None:
                    whole[index.groups[(l, p)]] = True
                runs = None
            parts.append(runs)
        return parts

    def _forward_plan(self, recompute: np.ndarray, sum_layers: np.ndarray):
        """The rows a call's forward computes: ``({sum layer: partition_runs}, {layer group: [(product layer,
        partition_runs or None)]})``. A sum layer computes its recomputed groups' rows -- a partition of many runs, or
        of most of its rows, or one its forward computes whole only (`SumLayer.forward_runs_ok`), whole, as a layer
        that takes no runs is; the product layers below compute the children of every sum that is computed (keyed
        by the group of the sums)."""
        L = self._layout
        R = np.zeros([L.layer_of.size], dtype = bool)
        R[recompute] = True
        sums_computed = R & L.is_sum                 # the recomputed sums, and every one a whole partition holds
        on, index = self._runs_on["sum_fw"], self._row_index["sum_fw"]
        found = index.runs(R) if on.any() else {}
        sum_runs = {}
        for l in sum_layers.tolist():
            if not on[l]:
                sums_computed[L.layer_groups[l]] = True
                continue
            parts = self._partition_runs("sum_fw", l, found, whole = sums_computed)
            for p, ok in enumerate(self._fw_ok[l]):
                if not ok and parts[p] is not None and len(parts[p]) > 0:
                    sums_computed[index.groups[(l, p)]] = True
                    parts[p] = None
            sum_runs[l] = parts
        return sum_runs, self._product_runs(self._children(sums_computed))

    def _product_runs(self, need: np.ndarray) -> dict:
        """``{sum layer group: [(product layer, partition_runs or None)]}``: the product layers holding the products
        ``need`` marks, computing their rows (whole, a layer that takes no runs), keyed by the layer group above."""
        L = self._layout
        on = self._runs_on["prod_fw"]
        found = self._row_index["prod_fw"].runs(need) if on.any() else {}
        out = {}
        for p in L.layers(np.flatnonzero(need), PROD).tolist():
            runs = self._partition_runs("prod_fw", p, found) if on[p] else None
            out.setdefault(int(L.group[p]) + 1, []).append((p, runs))
        return out

    def _backward_plan(self, plan, sum_layers: np.ndarray, prod_layers: np.ndarray):
        """The rows a conditional's backward computes, or None without group runs: ``{"sum": {sum layer:
        partition_runs}, "prod_fw": {layer group: [(product layer, partition_runs or None)]}, "prod_bk": {product
        layer: partition_runs}}`` -- a sum layer's element flows of the visited products (whole partitions and layers
        as in the forward); the products whose values those read, recomputed below it: the visited ones and every
        child of a whole partition or layer; a product layer's flows of its visited children."""
        if not self._group_runs:
            return None
        L = self._layout
        V = np.zeros([L.layer_of.size], dtype = bool)
        V[plan.visited] = True
        products = V & L.is_prod                     # the products whose values the sums' backward reads
        on = self._runs_on["sum_bk"]
        found = self._row_index["sum_bk"].runs(V & L.is_prod) if on.any() else {}
        sum_runs = {}
        for l in sum_layers.tolist():
            if on[l]:
                sum_runs[l] = self._partition_runs("sum_bk", l, found, whole = products)
            else:
                products[L.layer_children[l]] = True
        on = self._runs_on["prod_bk"]
        found = self._row_index["prod_bk"].runs(V & ~L.is_prod) if on.any() else {}
        prod_bk = {l: self._partition_runs("prod_bk", l, found) for l in prod_layers.tolist() if on[l]}
        return {"sum": sum_runs, "prod_fw": self._product_runs(products), "prod_bk": prod_bk}

    def _input_runs(self, l: int, recompute: np.ndarray):
        """The runs of input layer ``l``'s nodes that the call recomputes (layer-local ``(start, end)``), or None for the
        whole layer: a layer whose forward computes all of its nodes, more than `_MAX_INPUT_RUNS` runs, or runs covering
        most of the layer (one launch over it costs no more)."""
        L = self._layout
        if not L.runs_ok[l] or L.input_size[l] * self.pc.node_mars.size(1) < _RUNS_MIN_VALUES:
            return None
        gs = recompute[L.layer_of[recompute] == l]
        if gs.size == 1:                                             # (one group: no union to take)
            a, b = int(L.lo[gs[0]] - L.input_start[l]), int(L.hi[gs[0]] - L.input_start[l])
            return [(a, b)] if 2 * (b - a) <= L.input_size[l] else None
        lo, hi = _union(L.lo[gs] - L.input_start[l], L.hi[gs] - L.input_start[l])
        if lo.size > _MAX_INPUT_RUNS or 2 * int((hi - lo).sum()) > L.input_size[l]:
            return None
        return list(zip(lo.tolist(), hi.tolist()))

    def _backward(self, plan, kwargs: dict, scale: torch.Tensor, reuse: bool, rows: _Rows) -> None:
        """The top-down pass of a conditional, in log-space flows: over the layers holding visited groups or seeds.
        With ``reuse``, every seed starts from its stored outside: for its layer's backward, its ``node_mars`` is set to
        ``scale - out`` and its flow to 0, so that the flows below it are ``out * in / exp(scale)`` -- a full backward's
        ``out * in / p`` with ``log p`` taken as ``scale`` (``[B]``, an estimate of it; any value is exact, and one
        near ``log p`` keeps the flows at the magnitudes the kernels see in a full backward). Without, the pass starts
        at the root from its inside (``scale`` is then ``log p`` itself), as a full backward does. An
        external-parameter layer's backward shifts its ``node_mars`` in place; the layer's rows are put back
        afterwards, bit for bit."""
        pc, L = self.pc, self._layout
        s = self._planner.structure
        B = pc.node_mars.size(1)
        for name, size, fill in (("node_flows", pc.num_nodes, "fill_nodes"),
                                 ("element_flows", pc.num_elements, "fill_elements")):    # (see `_conditional_rows`)
            ranges = rows.meta[fill]
            if ranges == "all":
                pc._init_buffer(name = name, shape = (size, B), set_value = -float("inf"))
                continue
            pc._init_buffer(name = name, shape = (size, B))
            if ranges is None:
                _put(pc.__dict__[name], rows.get(fill), -float("inf"))
            else:
                for a, b in ranges:
                    pc.__dict__[name][a:b] = -float("inf")
        nf, nm = pc.node_flows, pc.node_mars

        root = s.root
        if not reuse and bool((plan.seeds == root).any()):
            nf[L.lo[root]:L.hi[root]] = 0.0                          # the root's flow, its inside kept

        bk_opts = dict(kwargs, param_flows = None, denom_param_flows = None, allow_modify_flows = False,
                       logspace_flows = True, negate_pflows = False, precision = pc.precision,
                       pflow_temperature = 1.0, temper_eflow = False)
        # Every layer group holding work, top-down: a product group's layers with visited groups, a sum group's with
        # visited groups or seeds
        work = {}
        for ls in L.by_group(rows.meta["sum_layers"]):
            work[int(L.group[ls[0]])] = ls
        for ls in L.by_group(rows.meta["prod_layers"]):
            work[int(L.group[ls[0]])] = ls
        bk = rows.meta["bk"]                                         # the rows to compute (None: whole layers)
        for k in sorted(work, reverse = True):
            lg, ls = pc.inner_layer_groups[k], work[k]
            if lg.is_prod():
                if bk is None:
                    for l in ls:
                        lg[L.index[l]].backward(nf, pc.element_flows, logspace_flows = True)
                else:
                    for l in ls:
                        lg[L.index[l]].backward(nf, pc.element_flows, logspace_flows = True,
                                                partition_runs = bk["prod_bk"].get(int(l)))
                continue
            saved = [(slice(*L.nid_range[l]), nm[slice(*L.nid_range[l])].clone()) for l in ls if L.gated[l]]
            seeds = rows.get(("seeds", k))
            if seeds is not None:
                saved.append((seeds, _copy(nm, seeds)))
                _put(nm, seeds, scale[None, :] - _get(self._node_out, seeds))
                _put(nf, seeds, 0.0)
            below = pc.inner_layer_groups[k - 1]
            if bk is None:                                           # whole layers, as before group runs
                for p in L.children(ls):
                    below[L.index[p]].forward(nm, pc.element_mars)
                for l in ls:
                    lg[L.index[l]].backward(nf, pc.element_flows, nm, pc.element_mars, pc.params,
                                            propagation_alg = "LL", **bk_opts)
            else:
                for p, runs in bk["prod_fw"].get(k, []):
                    below[L.index[p]].forward(nm, pc.element_mars, partition_runs = runs)
                for l in ls:
                    runs = bk["sum"].get(int(l))
                    opts = bk_opts if runs is None else dict(bk_opts, partition_runs = runs)
                    lg[L.index[l]].backward(nf, pc.element_flows, nm, pc.element_mars, pc.params,
                                            propagation_alg = "LL", **opts)
            for r, val in reversed(saved):
                _put(nm, r, val)

        # Input groups among the seeds -- targets whose outsides are stored -- last: a visited product above one (a
        # parent of another target) adds to its flow, which the stored outside already accounts for
        seeds = rows.get("input_seeds")
        if seeds is not None:
            _put(nf, seeds, _get(self._node_out, seeds) + _get(nm, seeds) - scale[None, :])
