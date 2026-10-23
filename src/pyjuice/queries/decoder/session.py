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

A call runs its plan by whole layers: a layer whose groups the plan does not touch launches nothing. With
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
    """Where the planner's groups are in the compiled circuit: every sum and input group's node rows, and every layer's
    groups -- per input layer, per layer of every inner layer group -- with, for a sum layer, the layers of the product
    layer group below it that hold its children."""

    def __init__(self, pc, planner: Planner):
        s = planner.structure
        index = {id(ns): g for g, ns in enumerate(s.groups)}
        self.rows = [ns._output_ind_range if s.kind[g] != PROD else None for g, ns in enumerate(s.groups)]

        def groups_of(layer):
            gs = set()
            for ns in layer.nodes:
                assert id(ns) in index, "a compiled layer holds a node group the circuit's graph does not"
                gs.add(index[id(ns)])
            return gs

        self.input_groups = [groups_of(layer) for layer in pc.input_layer_group]
        self.inner_groups = [[groups_of(layer) for layer in lg] for lg in pc.inner_layer_groups]
        mapped = set().union(*self.input_groups, *[g for lg in self.inner_groups for g in lg])
        assert mapped == set(range(len(s.groups))), "the circuit's graph and its compiled layers disagree"

        self.has_external = any(isinstance(layer, ExternalParamsSumLayer) for lg in pc.inner_layer_groups
                                for layer in lg)
        self.child_prods = []                    # [k][i]: layers of layer group k - 1 holding layer i's children
        self.gated = []
        for k, lg in enumerate(pc.inner_layer_groups):
            per_layer, gated = [], []
            for i, layer in enumerate(lg):
                if lg.is_sum():
                    children = set()
                    for g in self.inner_groups[k][i]:
                        children.update(s.ch_idx[s.ch_ptr[g]:s.ch_ptr[g + 1]].tolist())
                    per_layer.append([j for j, below in enumerate(self.inner_groups[k - 1]) if below & children])
                    gated.append(isinstance(layer, ExternalParamsSumLayer))
                else:
                    per_layer.append([])
                    gated.append(False)
            self.child_prods.append(per_layer)
            self.gated.append(gated)


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
        self._forward(plan.recompute, data, kwargs)
        if plan.cut.size == 0:
            lls = pc.node_mars[pc._root_node_range[0]:pc._root_node_range[1], :].permute(1, 0).clone()
        else:
            rows = self._rows(plan.cut)
            lls = torch.logsumexp(pc.node_mars[rows] + self._node_out[rows], dim = 0)[:, None]
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
        self._forward(plan.recompute, data, kwargs)
        if reuse:
            scale = self._log_p
        else:
            scale = pc.node_mars[pc._root_node_range[0]:pc._root_node_range[1]].logsumexp(dim = 0)
        self._backward(plan, kwargs, scale, reuse)

        if self.early_stop:
            s = self._planner.structure
            visited = [g for g in plan.visited.tolist() if s.kind[g] != PROD]
            if visited:
                rows = self._rows(visited)
                self._node_out[rows] = pc.node_flows[rows] - pc.node_mars[rows] + scale[None, :]
            log_p = self._normalize_target_flows(targets)
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
            changed = torch.nonzero(((miss != m0) | (~miss & differ)).any(dim = 0)).flatten().cpu().numpy()
        self._planner.observe(changed)
        return miss

    def _remember(self, data: torch.Tensor, miss: torch.Tensor) -> None:
        self._previous = (data.clone(), miss.clone())                # never the caller's tensors

    def _normalize_target_flows(self, targets: Sequence[int]) -> Optional[torch.Tensor]:
        """Make the targets' flows a full backward's. The pass ran with ``log p`` taken as its scale (an estimate),
        so a node's flow is ``out * in / exp(scale)`` rather than ``out * in / p``; a target's input nodes are a cut
        (every induced tree has one leaf of the variable), so their flows sum to ``p / exp(scale)`` -- divided out per
        target and sample. The input layers' conditionals normalize with a small additive guard (``1e-12``), which
        flows of another scale would not survive. Returns that ``log (p / exp(scale))``, of the first target, [B]."""
        s, L = self._planner.structure, self._layout
        rows, seg = [], []
        for i, t in enumerate(targets):
            for g in s.input_groups([t]).tolist():
                a, b = L.rows[g]
                rows.append(torch.arange(a, b))
                seg.append(torch.full([b - a], i, dtype = torch.long))
        if not rows:
            return None
        device = self.pc.node_flows.device
        rows, seg = torch.cat(rows).to(device), torch.cat(seg).to(device)
        nf = self.pc.node_flows
        vals = nf[rows]
        B = vals.size(1)
        m = torch.full([len(targets), B], -float("inf"), dtype = vals.dtype, device = device)
        m = m.scatter_reduce(0, seg[:, None].expand(-1, B), vals, reduce = "amax")
        m = torch.where(torch.isfinite(m), m, 0.0)
        tot = torch.zeros([len(targets), B], dtype = vals.dtype, device = device).index_add_(0, seg, torch.exp(vals - m[seg]))
        log_p = torch.where(tot > 0, m + torch.log(tot), 0.0)                # evidence of probability 0: as it is
        nf[rows] = vals - log_p[seg]
        return log_p[0]

    def _rows(self, groups) -> torch.Tensor:
        """The node rows of sum and input ``groups``, as one index tensor."""
        ranges = [self._layout.rows[g] for g in groups]
        return torch.cat([torch.arange(a, b) for a, b in ranges]).to(self.pc.node_mars.device)

    # ---------------------------------------------------------------------------------------------
    # Execution, by whole layers
    # ---------------------------------------------------------------------------------------------

    def _forward(self, recompute: np.ndarray, data: torch.Tensor, kwargs: dict) -> None:
        """Recompute the insides of the groups ``recompute``: the layers holding any of them (a sum layer with the
        layers of the product layer group below that hold its children), bottom-up."""
        pc, L = self.pc, self._layout
        todo = set(recompute.tolist())
        if not todo:
            return
        for i, layer in enumerate(pc.input_layer_group):
            if L.input_groups[i] & todo:
                distributions.forward(layer, data.clone() if data.dim() == 3 else data, pc.node_mars, **kwargs)

        alg = pc.default_propagation_alg
        sum_opts = dict(kwargs, **pc.propagation_alg_kwargs, precision = pc.precision, pflow_temperature = 1.0)
        for k, lg in enumerate(pc.inner_layer_groups):
            if not lg.is_sum():
                continue
            active = [i for i in range(len(lg)) if L.inner_groups[k][i] & todo]
            if not active:
                continue
            below = pc.inner_layer_groups[k - 1]
            for j in sorted(set(j for i in active for j in L.child_prods[k][i])):
                below[j].forward(pc.node_mars, pc.element_mars)
            layer_alg = alg if isinstance(alg, str) else alg[k]
            for i in active:
                lg[i].forward(pc.node_mars, pc.element_mars, pc.params, propagation_alg = layer_alg, **sum_opts)

    def _backward(self, plan, kwargs: dict, scale: torch.Tensor, reuse: bool) -> None:
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
        pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, B), set_value = -float("inf"))
        pc._init_buffer(name = "element_flows", shape = (pc.num_elements, B), set_value = -float("inf"))
        nf, nm = pc.node_flows, pc.node_mars

        seeds, visited = set(plan.seeds.tolist()), set(plan.visited.tolist())
        root = s.root
        if root in seeds and not reuse:
            a, b = L.rows[root]
            nf[a:b] = 0.0                                            # the root's flow, its inside kept

        bk_opts = dict(kwargs, param_flows = None, denom_param_flows = None, allow_modify_flows = False,
                       logspace_flows = True, negate_pflows = False, precision = pc.precision,
                       pflow_temperature = 1.0, temper_eflow = False)
        work = seeds | visited
        for k in range(len(pc.inner_layer_groups) - 1, -1, -1):
            lg = pc.inner_layer_groups[k]
            if lg.is_prod():
                for j in range(len(lg)):
                    if L.inner_groups[k][j] & visited:
                        lg[j].backward(nf, pc.element_flows, logspace_flows = True)
                continue
            active = [i for i in range(len(lg)) if L.inner_groups[k][i] & work]
            if not active:
                continue
            saved = []
            for i in active:
                if L.gated[k][i]:
                    a, b = lg[i].nid_range
                    saved.append((a, b, nm[a:b].clone()))
                for g in L.inner_groups[k][i] & seeds:
                    if g == root and not reuse:
                        continue
                    a, b = L.rows[g]
                    if not L.gated[k][i]:
                        saved.append((a, b, nm[a:b].clone()))
                    nm[a:b] = scale[None, :] - self._node_out[a:b]
                    nf[a:b] = 0.0
            below = pc.inner_layer_groups[k - 1]
            for j in sorted(set(j for i in active for j in L.child_prods[k][i])):
                below[j].forward(nm, pc.element_mars)
            for i in active:
                lg[i].backward(nf, pc.element_flows, nm, pc.element_mars, pc.params, propagation_alg = "LL", **bk_opts)
            for a, b, val in reversed(saved):
                nm[a:b] = val

        # Input groups among the seeds -- targets whose outsides are stored -- last: a visited product above one (a
        # parent of another target) adds to its flow, which the stored outside already accounts for
        for g in seeds:
            if s.kind[g] == INPUT:
                a, b = L.rows[g]
                nf[a:b] = self._node_out[a:b] + nm[a:b] - scale[None, :]
