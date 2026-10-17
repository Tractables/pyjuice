"""
The runtime state of a :class:`~.backend.DFAExactBackend`: the buffers its queries run in, and what it derives from
the PC's parameters and keeps across queries.

Constrained queries run in buffers of their own, laid out for the lifted values (see
:func:`~.tables.buffer_layout`) and allocated on the first query.

The input nodes' token-class masses depend only on the parameters, so inside :func:`pyjuice.fast_inference` (whose
contract is that parameters do not change) they are computed by the first query and kept until the scope exits;
outside it, every query computes them. So do the input-independent inside values -- those of the nodes whose scope no
sample of a query observes (see :mod:`~.plans`), held once in ``prior_mars`` and read by every sample -- each
kept from the query that first computes it. Likewise the stacked weights of dense sum node blocks that read the same
children (one matrix product instead of one per block, on products narrow enough for it to pay): inside a scope that
allows parameter copies, each is copied once and kept until the scope exits; outside it, a query copies only the
weights it reads more than once (tied ones) and drops them when it returns. So do the token-major copies of the input
parameters that conditionals score against (:meth:`State.token_table`): kept by a scope that allows parameter copies,
else built by every query.

A decoder's per-sample inputs (e.g. soft evidence, see :meth:`~pyjuice.constraints.ConstrainedCircuit.decoder`) give
the input nodes that read them class masses of their own per sample: built by the session's first query, from the
inputs' distributions' functions (:mod:`pyjuice.constraints.distributions`), and kept with the others until the
session's scope exits (:meth:`State.masses`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from pyjuice.utils.fast_inference import bf16_params_allowed, is_active, param_copies_allowed, register_layer

from ...distributions import lookup
from .compiled.tables import _align, buffer_layout


@dataclass(eq = False)
class ExternalInputs:
    """
    A decoder session's per-sample inputs, checked against the PC (:meth:`~.backend.DFAExactBackend.external_inputs`).

    :ivar inputs: keyword -> value, as ``pc(x, ...)`` takes them (the input layers' alone)
    :ivar batch: their batch size: every query's, or 1 (every row reads the same); ``evidence_batch`` the input
        layers' inputs'
    :ivar layers: the input layers that read them (indices into the PC's input layer group)
    :ivar variables: bool [n] (host): the variables whose input nodes read them
    :ivar free: None, or bool [batch or 1, n] on the device: the positions they leave free -- marginalized under the
        inputs' weights whatever ``data`` holds there (a soft evidence's value mask False), which a query must not
        mark missing besides; ``free_host`` the same on the host
    :ivar gates: None, or the gates of the PC's gated sum nodes (``sum_external_params``, a
        :class:`.external_params.block_scale.SessionGates`)
    :ivar observed: bool [n] (host): the positions whose values differ per sample whatever is observed -- the variables
        of evidence of a batch, the scopes of gates of a batch -- which a query plans as observed
    :ivar scopes: the intervals ``(a, b)`` of the gated nodes' scopes when their gates differ per sample: the top-down
        values of a node of such a scope read them (its parents' are gated), so a query computes them per sample
    """

    inputs: Dict[str, Any]
    batch: int
    layers: tuple
    variables: np.ndarray
    free: Optional[torch.Tensor] = None
    free_host: Optional[np.ndarray] = None
    gates: Any = None
    observed: Optional[np.ndarray] = None
    scopes: tuple = ()
    evidence_batch: int = 1


@dataclass
class ClassMasses:
    """
    The class masses a query's input nodes read (:meth:`State.masses`): input node ``u`` of sample ``b`` reads row
    ``row[u - input_start] + b * stride[u - input_start]`` of ``table`` ([rows, num_classes], log).

    :ivar table, row, stride: ``row``, ``stride`` int32 on the device; ``stride`` None for the same rows for every
        sample
    :ivar batch: 1, or the samples that have rows of their own (``stride``)
    :ivar version: which computation of class masses ``table`` is (what is derived from them, e.g. transition masses,
        follows it)
    :ivar sample: per input layer (index in the PC's input layer group) that reads per-sample inputs, its
        :class:`~pyjuice.constraints.distributions.SampleInputs`
    :ivar lognorm: None, or under per-sample inputs fp32 [rows of ``table``]: every row's ``log Z`` (0 for rows of no
        input's): the rows are the class masses of a token distribution ``q = p w / Z``
    :ivar token_logw, token_row, token_stride, token_width: under per-sample inputs, the tokens' weights ``log w``:
        node ``u`` of sample ``b`` weighs token ``v < token_width`` ``token_logw[(token_row[u - input_start] + b *
        token_stride) * token_width + v]`` (fp32, flat), the others nothing; ``token_row`` int64 on the device (-1: a
        node no input reweights), and on the host as ``token_row_host``
    :ivar token_ids: None, or (top-k) int32, flat as ``token_logw``: the tokens the weights are of, ascending in every
        row of ``token_width`` (see :func:`~pyjuice.constraints.distributions.find_token`) -- token ``token_ids[i]``
        weighs ``token_logw[i]``, the others nothing
    :ivar successors: None until a query asks, then ``(lists,)``: the successor lists its liveness sweeps follow where
        a token is missing (:func:`~.kernels.live.live_successors`; ``lists`` None for every class's)
    """

    table: torch.Tensor
    row: torch.Tensor
    stride: Optional[torch.Tensor]
    batch: int
    version: int
    sample: Dict[int, Any]
    lognorm: Optional[torch.Tensor] = None
    token_logw: Optional[torch.Tensor] = None
    token_row: Optional[torch.Tensor] = None
    token_row_host: Optional[np.ndarray] = None
    token_stride: int = 0
    token_width: int = 0
    token_ids: Optional[torch.Tensor] = None
    successors: Optional[tuple] = None


#: What a storage holds when it is allocated (:meth:`State.storage_view`): None, nothing in particular -- every slot a
#: query reads is written first --, or a value it is filled with (the tests fill NaN, so that a read of a slot nothing
#: wrote shows).
FRESH_FILL: Optional[float] = None


def fresh_storage(numel: int, device) -> torch.Tensor:
    """``numel`` floats of new storage on ``device``, as :data:`FRESH_FILL` says."""
    out = torch.empty(max(int(numel), 1), device = device)
    if FRESH_FILL is not None:
        out.fill_(FRESH_FILL)
    return out


class State:
    """
    The buffers and the parameter-derived values of one :class:`~.backend.DFAExactBackend` (``backend.state``). It
    lives on the backend's device; moving the backend starts a new one.

    :ivar class_mars_version: counts the computations of the class masses (what is derived from them follows)
    :ivar epoch: counts the uses of the buffers (:meth:`buffers`): a decoder that keeps values in them between its
        queries reuses them only if nothing else used the buffers in between
    """

    def __init__(self, backend):
        self._backend = backend
        self._device = backend.device
        self._storage = {}                          # buffer name -> backing storage (see `storage_view`)
        self._layouts = {}                          # batch size -> `buffer_layout` and its device tensors
        self._class_rows = None                     # where the input nodes' class masses are (see `class_mass_rows`)
        self._class_mars = None                     # the input nodes' class masses (see `class_masses`)
        self._class_mars_kept = False               # kept by a `pyjuice.fast_inference` scope
        self.class_mars_version = 0
        self._class_mars_at = 0                     # the version `_class_mars` is
        self._session = None                        # (inputs, their class masses) of a decoder session (`masses`)
        self._kept_stacks = None                    # stacked dense weights kept by a scope (see `weight_stacks`)
        self._prior_valid = None                    # which input-independent values prior_mars holds (`prior_validity`)
        self._prior_version = None                  # the class masses they were computed with
        self._prior_kept = False                    # kept by a `pyjuice.fast_inference` scope
        self._prior_flows_state = None              # the input-independent top-down values (see `prior_flows`)
        self._token_tables = None                   # token-major input parameters kept by a scope (`token_table`)
        self.bf16_tokens = False                    # a decoder session's: token tables in bfloat16 ("auto": at tf32)
        self.epoch = 0

    # ---------------------------------------------------------------------------------------------
    # Buffers
    # ---------------------------------------------------------------------------------------------

    def buffers(self, batch_size: int, inputs_only: bool = False, layout: Optional[dict] = None) -> Dict[str, Any]:
        """
        The lifted buffers for a batch of ``batch_size``, cut from storage this state owns
        (:func:`~.tables.buffer_layout` says where every region sits, or ``layout``: a query's compact one, see
        :attr:`~.kernels.compact.CompactLayout.mars`); ``inputs_only``: ``node_mars`` holds the input rows alone, for
        a query with no per-sample inside value of a sum region (its storage given up if larger).

        Each buffer has one backing storage, reused while it is large enough and at most 4 times larger than needed
        (so a loop whose batch size varies does not reallocate, and keeps its addresses), not filled when allocated
        (:func:`fresh_storage`): every slot a kernel reads is written first, and padding is never read.

        :returns: a dict with ``node_mars`` and ``element_mars`` (flat), ``input_mars`` (a contiguous
            ``[input_end, batch_size]`` view, which an input layer writes as it writes a :class:`TensorCircuit`'s
            ``node_mars``) and ``layout`` (the :func:`buffer_layout` dict, plus ``sum_offsets`` / ``sum_widths`` as
            int64 tensors on the device for the kernels)
        """
        B = int(batch_size)
        self.epoch += 1
        layout = self.batch_layout(B) if layout is None else layout
        input_end = self._backend.input_range[1]
        node_mars = self.storage_view("node_mars", input_end * B if inputs_only else layout["node_size"])
        element_mars = self.storage_view("element_mars", layout["element_size"])
        return dict(node_mars = node_mars, element_mars = element_mars,
                    input_mars = node_mars[:input_end * B].view(input_end, B), layout = layout)

    def batch_layout(self, B: int) -> Dict[str, Any]:
        """The :func:`buffer_layout` of a batch of ``B``, with ``sum_offsets_t`` / ``sum_widths_t`` on the device."""
        layout = self._layouts.get(B)
        if layout is None:
            be = self._backend
            layout = buffer_layout(be.input_range, be.sum_regions, be.element_regions, B)
            layout["sum_offsets_t"] = torch.tensor(layout["sum_offsets"], dtype = torch.int64, device = self._device)
            layout["sum_widths_t"] = torch.tensor(layout["sum_widths"], dtype = torch.int64, device = self._device)
            self._layouts[B] = layout
        return layout

    def flows(self, batch_size: int, bufs: Dict[str, Any], compact = None) -> Dict[str, Any]:
        """
        The top-down buffers of a batch (see :mod:`~.queries.top_down`): ``node_flows`` (laid out as ``node_mars``,
        storage of its own, allocated on first use) and ``element_flows``, which is ``element_mars``'s storage -- the
        top-down pass never reads the product insides there. ``compact``: the query's
        :class:`~.kernels.compact.CompactLayout`, which lays both out (``element_flows`` in storage of its own where
        ``element_mars``'s is too small: a compact row holds every pair of live columns, joined or not).
        """
        if compact is None:
            return dict(node_flows = self.storage_view("node_flows", bufs["layout"]["node_size"]),
                        element_flows = bufs["element_mars"])
        elements = bufs["element_mars"]
        if compact.element_size > elements.numel():
            elements = self.storage_view("element_flows", compact.element_size)
        return dict(node_flows = self.storage_view("node_flows", compact.size), element_flows = elements)

    def prior_buffer(self) -> Dict[str, Any]:
        """
        ``prior_mars``: the input-independent inside values (see :mod:`~.plans`), laid out as ``node_mars`` for a
        batch of one, in storage of its own (it outlives queries of any batch size), allocated on first use. A dict
        with ``mars`` (flat), and every sum region's ``offset`` and ``width`` there (Python ints) and ``offset_t`` /
        ``width_t`` (int64 on the device).
        """
        layout = self.batch_layout(1)
        return dict(mars = self.storage_view("prior_mars", layout["node_size"]), offset = layout["sum_offsets"],
                    width = layout["sum_widths"], offset_t = layout["sum_offsets_t"], width_t = layout["sum_widths_t"])

    def prior_flows(self, regions) -> Dict[str, Any]:
        """
        ``prior_flows``: the input-independent top-down values (see :mod:`~.plans`), for a batch of one, in
        storage of its own laid out APPEND-ONLY: a sum region (node) gets its rows, filled with -inf, the first time a
        query keeps its values once (``regions``, bool per sum region), and keeps them, so the storage holds only the
        regions ever kept once and never more than one sample's ``node_flows``. Its values are kept as
        ``prior_mars``'s (see :meth:`prior_validity`). A dict with ``flows`` (flat), every region's ``offset`` and
        ``width`` (Python ints; offset -1 where it has no rows yet) and ``offset_t`` / ``width_t`` (int64 on the
        device), and ``valid`` (bool per sum region on the host: the regions whose values it holds).
        """
        sum_regions = self._backend.sum_regions
        pf = self._prior_flows_state
        if pf is None:
            R = len(sum_regions)
            pf = self._prior_flows_state = dict(flows = torch.empty(0, device = self._device), offset = [-1] * R,
                                                width = [0] * R, size = 0, valid = np.zeros(R, dtype = bool))
        new = [r for r in np.nonzero(regions)[0].tolist() if pf["offset"][r] < 0]
        if new or "offset_t" not in pf:
            start = pf["size"]
            for r in new:
                first, end, slots = sum_regions[r]
                pf["offset"][r], pf["width"][r] = pf["size"], _align(slots)
                pf["size"] += (end - first) * pf["width"][r]
            if pf["size"] > pf["flows"].numel():         # grow by half again, keeping what is there
                flows = torch.empty(max(pf["size"], pf["flows"].numel() * 3 // 2), device = self._device)
                flows[:pf["flows"].numel()] = pf["flows"]
                pf["flows"] = flows
            pf["flows"][start:pf["size"]] = float("-inf")
            pf["offset_t"] = torch.tensor([max(o, 0) for o in pf["offset"]], dtype = torch.int64, device = self._device)
            pf["width_t"] = torch.tensor(pf["width"], dtype = torch.int64, device = self._device)
        return pf

    def storage(self, name: str) -> Optional[torch.Tensor]:
        """The whole storage named ``name`` (see :meth:`storage_view`) as it is, or None if it was never allocated."""
        return self._storage.get(name)

    def replace_storage(self, name: str, storage: torch.Tensor) -> Optional[torch.Tensor]:
        """Make ``storage`` (flat floats on the device) the storage named ``name`` (see :meth:`storage_view`); returns
        the one it replaces (None if none). As every slot a query reads is written first, it need not be filled."""
        old = self._storage.get(name)
        self._storage[name] = storage
        return old

    def storage_view(self, name: str, numel: int) -> torch.Tensor:
        """The first ``numel`` floats of the storage named ``name`` (see :meth:`buffers`)."""
        storage = self._storage.get(name)
        if storage is None or storage.numel() < numel or storage.numel() > 4 * max(numel, 1):
            storage = self._storage[name] = None                                    # free the old one first
            storage = self._storage[name] = fresh_storage(numel, self._device)
        return storage[:numel]

    # ---------------------------------------------------------------------------------------------
    # Derived from the parameters
    # ---------------------------------------------------------------------------------------------

    def prior_validity(self, version: Optional[int] = None):
        """
        ``(valid, kept)`` for a query starting now, reading the class masses of computation ``version``
        (:attr:`ClassMasses.version`): which inside values ``prior_mars`` holds (bool on the host, one per sum interval
        of the lifted Program, ``Program.sum_ivs``, set by the queries that compute them), and whether they outlive the
        query. They depend on the parameters and the class masses only, so they are kept as the class masses are:
        inside :func:`pyjuice.fast_inference` until the scope exits, while queries read the same masses (a decoder
        session's per-sample inputs give masses of their own); outside it, every query starts with none.
        """
        if self._prior_valid is None or not self._prior_kept or self._prior_version != version:
            self._prior_valid = torch.zeros(len(self._backend.program().sum_ivs), dtype = torch.bool)
            self._prior_version = version
            if self._prior_flows_state is not None:
                self._prior_flows_state["valid"][:] = False
            if is_active():
                self._prior_kept = True
                register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return self._prior_valid, self._prior_kept

    def class_mass_rows(self):
        """
        Where the class masses of every input node are: ``(firsts, num_rows, mass_row)``, every input layer's
        class-mass table (see :mod:`pyjuice.constraints.distributions`) being rows ``firsts[k] ..`` of the
        ``num_rows`` rows of :meth:`class_masses`, and input node ``u`` reading row ``mass_row[u - input_start]``
        (int32, on the device). Built on first use: it depends only on the PC's structure.
        """
        if self._class_rows is None:
            input_start, input_end = self._backend.input_range
            mass_row = torch.zeros(input_end - input_start, dtype = torch.int32, device = self._device)
            firsts, num_rows = [], 0
            for layer in self._backend.pc.input_layer_group:
                rows = lookup(layer.dist).class_mass_rows(layer)
                first, end = layer._output_ind_range
                mass_row[first - input_start:end - input_start] = (num_rows + rows).to(torch.int32)
                firsts.append(num_rows)
                num_rows += int(rows.max()) + 1
            self._class_rows = (firsts, num_rows, mass_row)
        return self._class_rows

    def class_masses(self) -> torch.Tensor:
        """
        ``[num_rows, num_classes]`` (see :meth:`class_mass_rows`): the log-mass of every token class for every
        input layer's class-mass row, by its distribution's functions in :mod:`pyjuice.constraints.distributions`.
        Computed on every query, except inside :func:`pyjuice.fast_inference`: there by the first query, then kept
        until the scope exits.
        """
        if self._class_mars_kept:
            return self._class_mars
        be = self._backend
        firsts, num_rows, _ = self.class_mass_rows()
        if self._class_mars is None:
            self._class_mars = torch.empty(num_rows, be.num_classes, device = self._device)
        for layer, first, end in zip(be.pc.input_layer_group, firsts, firsts[1:] + [num_rows]):
            lookup(layer.dist).class_masses(layer, be.token_classes, out = self._class_mars[first:end])
        self.class_mars_version += 1
        self._class_mars_at = self.class_mars_version
        if is_active():
            self._class_mars_kept = True
            register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return self._class_mars

    def masses(self, external: Optional[ExternalInputs] = None) -> ClassMasses:
        """
        The class masses of a query's input nodes (:class:`ClassMasses`): :meth:`class_masses`, the same for every
        sample -- and under a decoder session's per-sample ``external`` inputs, those the inputs give the nodes that
        read them, per sample (their distributions' ``sample_inputs``), in a table of the session's: built by its
        first query (the other nodes' rows copied in), kept until its scope exits.
        """
        table = self.class_masses()
        _, _, mass_row = self.class_mass_rows()
        if external is None:
            return ClassMasses(table, mass_row, None, 1, self._class_mars_at, {})
        if self._session is not None and self._session[0] is external and self._session[1] == self._class_mars_at:
            return self._session[2]
        if not external.layers:                     # gates alone: the class masses, as a computation of the session's
            self.class_mars_version += 1            # own (what is kept once under its gates is its own)
            masses = ClassMasses(table, mass_row, None, 1, self.class_mars_version, {})
            self._session = (external, self._class_mars_at, masses)
            if is_active():
                register_layer(self)                                # its exit calls `_release_fast_inference_params`
            return masses
        be, Be, C = self._backend, external.evidence_batch, self._backend.num_classes
        input_start = be.input_range[0]
        layers = [be.pc.input_layer_group[k] for k in external.layers]
        sizes = [lookup(layer.dist).sample_rows(layer, external.inputs) for layer in layers]
        full = torch.empty(table.size(0) + Be * sum(sizes), C, dtype = table.dtype, device = self._device)
        full[:table.size(0)] = table                                    # the other nodes' rows first
        row, stride = mass_row.clone(), torch.zeros_like(mass_row)
        lognorm = torch.zeros(full.size(0), dtype = torch.float32, device = self._device)
        token_row = torch.full(mass_row.shape, -1, dtype = torch.int64, device = self._device)
        num_rows, sample, logw, ids, num_wrows = table.size(0), {}, [], [], 0
        for k, layer, rows_s in zip(external.layers, layers, sizes):
            out = full[num_rows:num_rows + Be * rows_s].view(Be, rows_s, C)
            si = lookup(layer.dist).sample_inputs(layer, be.token_classes, external.inputs, out)
            first, end = layer._output_ind_range
            row[first - input_start:end - input_start] = (num_rows + si.rows).to(torch.int32)
            if Be > 1:
                stride[first - input_start:end - input_start] = rows_s
            lognorm[num_rows:num_rows + Be * rows_s] = si.logz.reshape(-1)
            token_row[first - input_start:end - input_start] = num_wrows + si.token_rows
            logw.append(si.token_logw)
            ids.append(si.token_ids)
            num_rows, num_wrows = num_rows + Be * rows_s, num_wrows + si.token_logw.size(1)
            sample[k] = si
        if len({w.size(2) for w in logw}) > 1 or len({i is None for i in ids}) > 1:
            raise NotImplementedError("Per-sample inputs weighing different numbers of tokens in different input "
                                      "layers (or listing the tokens in some only) are not supported.")
        cat = lambda ts: (ts[0] if len(ts) == 1 else torch.cat(ts, dim = 1)).contiguous().view(-1)
        self.class_mars_version += 1
        masses = ClassMasses(full, row, stride if Be > 1 else None, Be, self.class_mars_version, sample,
                             lognorm = lognorm, token_logw = cat(logw), token_row = token_row,
                             token_row_host = token_row.cpu().numpy(), token_stride = num_wrows if Be > 1 else 0,
                             token_width = logw[0].size(2), token_ids = None if ids[0] is None else cat(ids))
        self._session = (external, self._class_mars_at, masses)
        if is_active():
            register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return masses

    def weight_stacks(self):
        """
        ``(stacks, kept)``: where a query finds and leaves the stacked weights of its dense sum node blocks (stack
        key -> :func:`~.sum.stacked_weights`, see :class:`~.program.Program`), and whether they outlive the
        query. Inside a :func:`pyjuice.fast_inference` scope that allows parameter copies, the scope's: kept until it
        exits, so the query stacks every run it can (see ``STACK_MAX_COLUMNS`` there) -- and the CUDA fused sums keep
        their weights in fragment order there too (:func:`~.kernels.sum._cuda_fragments`); else an empty dict of the
        query's own, which stacks only reused weights (and, a ``precision = "bf16"`` query, holds its bf16 weights).
        """
        if self._kept_stacks is not None:
            return self._kept_stacks, True
        if param_copies_allowed():
            self._kept_stacks = {}
            register_layer(self)                                    # its exit calls `_release_fast_inference_params`
            return self._kept_stacks, True
        return {}, False

    def token_table(self, group: int, rows: np.ndarray, precision: str = "fp32") -> torch.Tensor:
        """
        ``[V, len(rows)]``: the probability of every token under the class-mass rows ``rows`` (sorted, see
        :meth:`class_mass_rows`), token-major -- the table row group ``group`` of the conditionals is scored against
        (:mod:`~.queries.conditional`) -- from every input layer's distribution's ``token_probs``. Built by every query,
        except inside a :func:`pyjuice.fast_inference` scope that allows parameter copies: there by the first query,
        then kept until the scope exits; in bfloat16 when the scope allows bf16 parameter copies, or in a decoder
        session that asks for it (``bf16_tokens``, see :class:`~.queries.decoder.DFAExactDecoder`; "auto": for a
        query whose ``precision`` is "tf32" or "bf16").
        """
        bf16 = bf16_params_allowed() or self.bf16_tokens is True or \
            (self.bf16_tokens == "auto" and precision in ("tf32", "bf16"))
        dtype = torch.bfloat16 if bf16 else torch.float32
        key = (group, dtype)
        if self._token_tables is not None and key in self._token_tables:
            return self._token_tables[key]
        firsts, num_rows, _ = self.class_mass_rows()
        layers = list(self._backend.pc.input_layer_group)
        V = lookup(layers[0].dist).num_values(layers[0].dist)
        table = torch.empty(V, len(rows), dtype = dtype, device = self._device)
        for layer, first, end in zip(layers, firsts, firsts[1:] + [num_rows]):
            k0, k1 = (int(k) for k in np.searchsorted(rows, [first, end]))
            if k1 > k0:                                             # the layer's rows: consecutive columns
                lookup(layer.dist).token_probs(layer, rows[k0:k1] - first, table[:, k0:k1])
        if param_copies_allowed():
            if self._token_tables is None:
                self._token_tables = {}
                register_layer(self)                                # its exit calls `_release_fast_inference_params`
            self._token_tables[key] = table
        return table

    def _release_fast_inference_params(self, poison: bool = False):
        """The :func:`pyjuice.fast_inference` scope that kept the class masses, the input-independent values, the
        stacked weights or the token tables exited: parameters may change from now on (``poison``: fill the released
        masses and values with NaN, as the scope does for its layers' copies; the stacked weights and token tables are
        dropped). Safe to call again."""
        self._class_mars_kept = False
        self._kept_stacks = None
        self._token_tables = None
        self._prior_kept = False
        self._session = None
        if poison and self._class_mars is not None:
            self._class_mars.fill_(float("nan"))
        if poison and "prior_mars" in self._storage:
            self._storage["prior_mars"].fill_(float("nan"))
        if poison and self._prior_flows_state is not None:      # rows given again (filled with -inf) on next use
            R = len(self._backend.sum_regions)
            self._prior_flows_state["flows"].fill_(float("nan"))
            self._prior_flows_state = dict(self._prior_flows_state, offset = [-1] * R, width = [0] * R, size = 0)
            self._prior_flows_state.pop("offset_t")
