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

A call runs its plan by whole layers -- an input layer whose distribution allows it, by the runs of its nodes the call
recomputes -- and a layer whose groups the plan does not touch launches nothing; its backward fills (-inf) only the
flows its pass reads before writing them. With
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
        # The node flows a backward through a layer reads or adds to before the pass wrote them, as row ranges [a, b): a
        # sum layer's own rows, a product layer's children's
        self.fill_nodes = [[] for _ in layers]
        for l, layer in enumerate(layers):
            if kind[l] == INPUT:
                start, end = layer._output_ind_range
                self.input_start[l], self.input_size[l] = start, end - start
                self.runs_ok[l] = distributions.takes_node_runs(layer)
            elif kind[l] == SUM:
                self.fill_nodes[l] = [tuple(layer.nid_range)]
            else:
                gs = np.flatnonzero((self.layer_of == l) & (s.kind == PROD))
                ch = np.concatenate([s.ch_idx[s.ch_ptr[g]:s.ch_ptr[g + 1]] for g in gs])
                lo, hi = _union(self.lo[ch], self.hi[ch])
                self.fill_nodes[l] = list(zip(lo.tolist(), hi.tolist()))

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


def _merge(ranges):
    """The union of a few row ranges ``[(a, b)]``, ascending and disjoint (plain Python: a call has a handful)."""
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


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

        # The layers the pass runs (for `_backward` too), and the flows to fill (-inf) before it: those its layers read
        # or add to before writing them (`_Layout.fill_nodes`). What else a pass reads unwritten feeds only groups it
        # does not visit -- every parent of a visited group is visited or a seed, so a visited product's element flow
        # comes from sum layers the pass runs -- and nothing reads those groups' flows: element flows and the rest of the
        # node flows need no fill. Small buffers are filled whole (`_FILL_MIN_VALUES`, `_FILL_WHOLE_FRACTION`); a few
        # ranges one launch each, more through an index uploaded with the call's other rows.
        sum_layers = L.layers(np.concatenate([seeds, plan.visited]), SUM)
        prod_layers = L.layers(plan.visited, PROD)
        rows.meta["sum_layers"], rows.meta["prod_layers"] = sum_layers, prod_layers
        rows.meta["fill_nodes"] = rows.meta["fill_elements"] = "all"
        if (self.pc.num_nodes + self.pc.num_elements) * self.pc.node_mars.size(1) >= _FILL_MIN_VALUES:
            rows.meta["fill_elements"] = []
            ranges = _merge([r for l in sum_layers.tolist() + prod_layers.tolist() for r in L.fill_nodes[l]])
            if sum(b - a for a, b in ranges) > _FILL_WHOLE_FRACTION * self.pc.num_nodes:
                pass                                                 # most of the buffer: fill all of it
            elif len(ranges) <= _MAX_FILL_SLICES:
                rows.meta["fill_nodes"] = ranges
            else:
                rows.meta["fill_nodes"] = None
                rows.add_ranges("fill_nodes", np.array([a for a, _ in ranges]), np.array([b for _, b in ranges]))
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
        layers of the product layer group below that hold its children), bottom-up."""
        pc, L = self.pc, self._layout
        if recompute.size == 0:
            return
        for l in L.layers(recompute, INPUT):
            runs = self._input_runs(l, recompute)
            distributions.forward(pc.input_layer_group[L.index[l]], data.clone() if data.dim() == 3 else data,
                                  pc.node_mars, **(kwargs if runs is None else dict(kwargs, node_runs = runs)))

        alg = pc.default_propagation_alg
        sum_opts = dict(kwargs, **pc.propagation_alg_kwargs, precision = pc.precision, pflow_temperature = 1.0)
        for ls in L.by_group(L.layers(recompute, SUM)):
            k = int(L.group[ls[0]])
            lg, below = pc.inner_layer_groups[k], pc.inner_layer_groups[k - 1]
            for p in L.children(ls):
                below[L.index[p]].forward(pc.node_mars, pc.element_mars)
            layer_alg = alg if isinstance(alg, str) else alg[k]
            for l in ls:
                lg[L.index[l]].forward(pc.node_mars, pc.element_mars, pc.params, propagation_alg = layer_alg, **sum_opts)

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
        for k in sorted(work, reverse = True):
            lg, ls = pc.inner_layer_groups[k], work[k]
            if lg.is_prod():
                for l in ls:
                    lg[L.index[l]].backward(nf, pc.element_flows, logspace_flows = True)
                continue
            saved = [(slice(*L.nid_range[l]), nm[slice(*L.nid_range[l])].clone()) for l in ls if L.gated[l]]
            seeds = rows.get(("seeds", k))
            if seeds is not None:
                saved.append((seeds, _copy(nm, seeds)))
                _put(nm, seeds, scale[None, :] - _get(self._node_out, seeds))
                _put(nf, seeds, 0.0)
            below = pc.inner_layer_groups[k - 1]
            for p in L.children(ls):
                below[L.index[p]].forward(nm, pc.element_mars)
            for l in ls:
                lg[L.index[l]].backward(nf, pc.element_flows, nm, pc.element_mars, pc.params, propagation_alg = "LL",
                                        **bk_opts)
            for r, val in reversed(saved):
                _put(nm, r, val)

        # Input groups among the seeds -- targets whose outsides are stored -- last: a visited product above one (a
        # parent of another target) adds to its flow, which the stored outside already accounts for
        seeds = rows.get("input_seeds")
        if seeds is not None:
            _put(nf, seeds, _get(self._node_out, seeds) + _get(nm, seeds) - scale[None, :])
