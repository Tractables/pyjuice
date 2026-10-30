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
        of evidence of a batch, the scopes of gates of a batch -- which a query plans as observed, with those
        :meth:`observed_under` adds
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
    _observed_under: Optional[tuple] = None

    def observed_under(self, prog, masses: "ClassMasses") -> np.ndarray:
        """
        bool [n] (host): the positions a query under ``masses`` (the session's :class:`ClassMasses`) plans as
        observed: :attr:`observed`, and under evidence of one row for all, the variables where it rules classes out
        (:func:`~.kernels.live.class_support`, computed once per masses for the queries' pruning): their values pruned
        per sample cost less than kept once over every column (a block kept once is computed without pruning, see
        :mod:`.queries.backend.plans`) -- on a JSON Schema automaton of 24k states, a thousandth.
        """
        if self.evidence_batch != 1 or not self.layers:
            return self.observed
        if self._observed_under is None or self._observed_under[0] != masses.version:
            from .kernels.live import live_successors
            live_successors(prog, masses)                                   # (the support, once per masses)
            out = self.observed
            if masses.support is not None:
                out = out | ((~masses.support.all(dim = 2)).any(dim = 0).cpu().numpy() & self.variables)
            self._observed_under = (masses.version, out)
        return self._observed_under[1]


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
        a token is missing (:func:`~.kernels.live.live_successors`; ``lists`` None for every class's); ``support``
        then the classes the inputs allow per sample and position (:func:`~.kernels.live.class_support`, None when
        they rule none out)
    :ivar prow: int32 [rows of ``table``]: every row's parameter row (a row of :meth:`State.enumerated_probs`):
        itself for the shared rows, the pair's for a per-sample row
    :ivar enum_probs: when the automaton has enumerated classes (see
        :class:`~pyjuice.constraints.distributions.TokenClasses`), fp32 [parameter rows, enumerated tokens]: every
        enumerated token's probability under every parameter row (:meth:`State.enumerated_probs`) -- an enumerated
        class's mass at a row is its token's there, under per-sample inputs times ``exp(enum_logw - lognorm)``; else
        None
    :ivar enum_logw: under per-sample inputs with enumerated classes, fp32 [batch, enumerated tokens live at any
        boundary] (:class:`~.kernels.transitions.EnumeratedTokens`): ``log w`` of every such token per sample at its
        boundary (0 at a position no input reweights, -inf for a token the inputs rule out); else None
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
    support: Optional[torch.Tensor] = None
    prow: Optional[torch.Tensor] = None
    enum_probs: Optional[torch.Tensor] = None
    enum_logw: Optional[torch.Tensor] = None


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


#: Of the device's free memory when the values kept once are first stored, the share they may take when the layout
#: of every region kept once does not fit that memory (then :class:`PriorPool` segments them; when it fits, they take
#: it all, as they did without a pool); a decoder session's ``memory_budget`` sets the bytes instead.
PRIOR_MEMORY_SHARE = 0.6


def _bytes(n: int) -> str:
    """``n`` bytes, in the unit that reads best."""
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} bytes"


class PriorPool:
    """
    The storage of the inside values kept once (``prior_mars``, see :mod:`~.queries.backend.plans`), under a memory
    budget. Every sum region kept once has an EXTENT of the pool while its values are held: a query gets extents for
    the values it computes (:meth:`allocate`), and when the pool is full, the values of an interval no query needs
    now are EVICTED -- their validity withdrawn (:meth:`State.prior_validity`), so that a later query reading them
    computes them again from the values still held below them. Which interval: the one read longest ago, among those
    that are no CHECKPOINT first -- the intervals whose depth (the longest chain of product steps to the inputs,
    ``Program.iv_depth``) is a multiple of the ``stride``, the smallest for which the checkpoints and a segment (the
    values of ``stride`` consecutive depths) fit the pool together: no value is then recomputed from more than
    ``stride`` steps below (left to right, a segment is computed once more as decoding enters it: about one forward
    pass more a session). Holes are merged as extents are freed, and the live extents moved to the front when none
    fits.

    A query whose values do not fit beside the checkpoints computes them in PIECES (:func:`~.queries.backend.forward.
    kept_values`), the lowest depths first (:meth:`piece`). Decoding that reads every value at every call (every
    position a target) would evict and recompute endlessly under a smaller pool: the pool refuses a query whose
    values do not fit it at all.

    The budget is the full layout when that fits the device's free memory (else :data:`PRIOR_MEMORY_SHARE` of it):
    then every region kept once has its extent, nothing is ever evicted, and the values are as without a pool.

    :param state: the :class:`State`
    :param budget: bytes, or None: by :data:`PRIOR_MEMORY_SHARE`
    """

    def __init__(self, state: "State", budget: Optional[int] = None):
        self.state = state
        self.budget = budget
        self.buffer: Optional[torch.Tensor] = None
        self.extents: Dict[int, tuple] = {}           # region -> (offset, floats)
        self.free: list = []                          # (offset, floats) of the holes, by offset
        self.stamp: Dict[int, int] = {}               # interval -> the query that last read or wrote its values
        self.queries = 0
        self.stride = 1
        self.evictions = 0
        self.reg_floats = None                        # a region's extent (floats, aligned)
        self.iv_floats = None                         # an interval's regions' extents together
        self.iv_regions = None                        # an interval's regions
        self._version = 0                             # counts changes of the extents
        self._table = None                            # (version, offsets host, offsets device)

    def reset(self):
        """Every value dropped (their validity withdrawn by the caller); the buffer kept."""
        self.extents, self.stamp = {}, {}
        self.free = [(0, self.buffer.numel())] if self.buffer is not None else []
        self._version += 1

    # ---------------------------------------------------------------------------------------------

    def _open(self):
        """The buffer, allocated on first use within the budget, and the checkpoint stride."""
        if self.buffer is not None:
            return
        prog, layout = self.state._backend.program(), self.state.batch_layout(1)
        full = layout["node_size"]
        self.reg_floats = np.asarray([int(_align(r)) for r in prog.reg_rows * np.asarray(layout["sum_widths"],
                                                                                        dtype = np.int64)])
        K = len(prog.sum_ivs)
        self.iv_floats = np.zeros(K, dtype = np.int64)
        np.add.at(self.iv_floats, prog.reg_iv_np, self.reg_floats)
        self.iv_regions = [np.nonzero(prog.reg_iv_np == k)[0] for k in range(K)]
        dev = self.state._device
        if self.budget is not None:
            budget = self.budget
        else:                                          # the memory this process can take: the device's free memory
            if dev.type == "cuda":                     # and the blocks its caching allocator holds unused
                free = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
            else:
                free = full * 4
            budget = full * 4 if full * 4 <= free else int(PRIOR_MEMORY_SHARE * free)
        floats = min(int(_align(max(budget // 4, 1))), int(_align(full)))
        self.buffer = fresh_storage(floats, dev)
        self.state._storage["prior_mars"] = self.buffer
        self.free = [(0, floats)]
        self.stride = 1
        if floats < full:                              # segmented: the stride by what the pool holds
            self.stride = self._stride(prog.iv_depth, floats)
        self._version += 1

    def _stride(self, depth: np.ndarray, floats: int) -> int:
        """
        The smallest stride whose checkpoints, a segment (any ``stride`` consecutive depths) and two more values fit
        ``floats`` together: a session walking the chain recomputes every value once at most. When none does (fewer
        than about 2 sqrt(D) values fit), the stride of the least estimated recomputation: every segment entered
        costs its values, and a segment the room beside the checkpoints does not hold costs about (S - room)^2 / 2
        room more as the walk steps through it -- against no checkpoints at all, D^2 / 2 P (the walk recomputes a
        pool's worth of the chain every P steps); D + 1 then (the leaves the checkpoints).
        """
        D = int(depth.max(initial = 0))
        level = np.zeros(D + 1, dtype = np.int64)       # the values' floats per depth
        np.add.at(level, depth, self.iv_floats)
        value = max(1, int(self.iv_floats.max()))
        best, best_cost = D + 1, D * D / (2 * max(1, floats / value))
        for S in range(1, D + 1):
            padded = np.concatenate([[0], np.cumsum(np.concatenate([level, np.zeros(S, dtype = np.int64)]))])
            segment = int((padded[S:] - padded[:-S]).max())
            room = floats - int(level[::S].sum()) - 2 * value
            if room >= segment:
                return S
            if room > 0:
                r = room / value
                cost = (D / S) * (S + max(0.0, S - r) ** 2 / (2 * r))
                if cost < best_cost:
                    best, best_cost = S, cost
        return best

    def _fit(self, floats: int) -> Optional[int]:
        """The offset of the first hole of ``floats`` or more, taken (first fit), or None."""
        for k, (off, size) in enumerate(self.free):
            if size >= floats:
                if size == floats:
                    del self.free[k]
                else:
                    self.free[k] = (off + floats, size - floats)
                return off
        return None

    def _release(self, off: int, floats: int):
        """A hole, merged with its neighbours."""
        self.free.append((off, floats))
        self.free.sort()
        merged = []
        for o, s in self.free:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        self.free = merged

    def _compact(self):
        """The live extents moved to the front, in order, one hole after them."""
        at = 0
        for r, (off, size) in sorted(self.extents.items(), key = lambda kv: kv[1][0]):
            if off != at:
                self.buffer[at:at + size] = self.buffer[off:off + size].clone()
                self.extents[r] = (at, size)
            at += size
        self.free = [(at, self.buffer.numel() - at)] if at < self.buffer.numel() else []
        self._version += 1

    def _take(self, floats: int) -> Optional[int]:
        """An extent of ``floats``: a hole, or the live extents compacted first when the holes hold it together."""
        off = self._fit(floats)
        if off is None and sum(size for _, size in self.free) >= floats:
            self._compact()
            off = self._fit(floats)
        return off

    def _evict(self, iv: int):
        """Interval ``iv``'s values dropped: its regions' extents freed, its validity withdrawn."""
        for r in self.iv_regions[iv]:
            if int(r) in self.extents:
                self._release(*self.extents.pop(int(r)))
        self.state._prior_valid[iv] = False
        self.stamp.pop(iv, None)
        self.evictions += 1
        self._version += 1

    def _held(self) -> np.ndarray:
        """bool [sum intervals]: those with extents."""
        out = np.zeros(len(self.iv_floats), dtype = bool)
        if self.extents:
            out[self.state._backend.program().reg_iv_np[list(self.extents)]] = True
        return out

    def _checkpoints(self) -> np.ndarray:
        return self.state._backend.program().iv_depth % self.stride == 0

    def _too_small(self, floats: int) -> RuntimeError:
        return RuntimeError(f"The values kept once that this query reads take {_bytes(floats * 4)}, more than the "
                            f"{_bytes(self.buffer.numel() * 4)} kept for them (the decoder's `memory_budget`, or "
                            f"{PRIOR_MEMORY_SHARE:.0%} of the free device memory when they were first stored, their "
                            f"total not fitting it). A query that reads them all (every position a target) needs them "
                            f"all held.")

    # ---------------------------------------------------------------------------------------------

    def holds(self, ivs: np.ndarray, checkpoints: bool = True) -> bool:
        """Whether the values of the intervals ``ivs`` (bool [sum intervals]) fit the pool -- beside the checkpoints
        it holds, unless ``checkpoints`` is False (then they may be evicted)."""
        self._open()
        keep = ivs | (self._held() & self._checkpoints()) if checkpoints else ivs
        return int(self.iv_floats[keep].sum()) <= self.buffer.numel()

    def piece(self, plan) -> Optional[np.ndarray]:
        """bool [sum intervals]: of the values ``plan`` computes, the ones to compute first when they do not all fit
        beside the ones it reads: the intervals of the lowest depths such that they, the held values they read, the
        held values the plan reads (kept through every piece: computed again otherwise) and the checkpoints fit the
        pool together -- without the checkpoints when nothing fits beside them; None when nothing fits at all."""
        self._open()
        prog, valid = self.state._backend.program(), self.state._prior_valid.numpy()
        todo = plan.todo.numpy() & ~valid
        keep, depth = plan.prior_read & valid, prog.iv_depth
        for checkpoints in (True, False):
            best = None
            for d in np.unique(depth[todo]):
                piece = todo & (depth <= d)
                below = np.zeros_like(piece)
                for k in np.nonzero(piece)[0]:
                    below[prog.iv_children[k]] = True
                if not self.holds(piece | (below & valid) | keep, checkpoints):
                    break
                best = piece
            if best is not None:
                return best
        return None

    def allocate(self, plan) -> Dict[str, Any]:
        """Extents for the values ``plan`` computes (:attr:`~.queries.backend.plans.QueryPlan.todo`), the ones it
        reads (``prior_read``) kept meanwhile and the others evicted as needed; then :meth:`table`."""
        self._open()
        prog, valid = self.state._backend.program(), self.state._prior_valid
        self.queries += 1
        pinned = plan.prior_read
        for iv in np.nonzero(pinned)[0]:
            self.stamp[int(iv)] = self.queries
        todo = plan.todo.numpy()
        checkpoint = self._checkpoints()
        for r in np.nonzero(todo[prog.reg_iv_np])[0]:
            r = int(r)
            if r in self.extents:
                continue
            floats = int(self.reg_floats[r])
            off = self._take(floats)
            while off is None:
                held = [iv for iv in np.nonzero(self._held() & ~pinned)[0]]
                if not held:
                    raise self._too_small(int(self.iv_floats[pinned].sum()))
                held.sort(key = lambda iv: (bool(checkpoint[iv]), self.stamp.get(int(iv), -1)))
                self._evict(int(held[0]))
                off = self._take(floats)
            self.extents[r] = (off, floats)
            self._version += 1
        return self.table()

    def table(self) -> Dict[str, Any]:
        """Where the values kept once are: ``mars`` (the pool), every sum region's ``offset`` there (-1 for one
        without an extent) and ``width`` (Python ints), ``offset_t`` / ``width_t`` on the device."""
        self._open()
        layout = self.state.batch_layout(1)
        if self._table is None or self._table[0] != self._version:
            R = len(layout["sum_widths"])
            offsets = [self.extents[r][0] if r in self.extents else -1 for r in range(R)]
            self._table = (self._version, offsets, torch.tensor(offsets, dtype = torch.int64, device = self.state._device))
        return dict(mars = self.buffer, offset = self._table[1], width = layout["sum_widths"], offset_t = self._table[2],
                    width_t = layout["sum_widths_t"])


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
        self._prior_pool = None                     # where they are kept, under a budget (see `prior_buffer`)
        self._prior_budget = None                   # bytes, a decoder session's (None: PRIOR_MEMORY_SHARE)
        self._token_tables = None                   # token-major input parameters kept by a scope (`token_table`)
        self._enum_probs = None                     # the enumerated tokens', fp32, kept by a scope (`enumerated_probs`)
        self._prow = None                           # every shared class-mass row's own number (see `masses`)
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
        The top-down buffers of a batch (see :mod:`~.queries.backend.top_down`): ``node_flows`` (laid out as
        ``node_mars``, storage of its own, allocated on first use) and ``element_flows``, which is ``element_mars``'s
        storage -- the top-down pass never reads the product insides there. ``compact``: the query's
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

    def prior_buffer(self, plan = None) -> Dict[str, Any]:
        """
        ``prior_mars``: the input-independent inside values (see :mod:`~.plans`), a block per sum region kept once
        (as ``node_mars`` lays a batch of one out) in the :class:`PriorPool`'s storage, which outlives queries of any
        batch size. With a ``plan`` (a :class:`~.queries.backend.plans.QueryPlan`), first the pool's extents for the
        values it computes. A dict with ``mars`` (flat), and every sum region's ``offset`` (-1 for one not held)
        and ``width`` there (Python ints) and ``offset_t`` / ``width_t`` (int64 on the device).
        """
        pool = self.prior_pool()
        return pool.allocate(plan) if plan is not None else pool.table()

    def prior_pool(self) -> PriorPool:
        """The :class:`PriorPool` holding the values kept once, under the session's budget."""
        if self._prior_pool is None:
            self._prior_pool = PriorPool(self, self._prior_budget)
        return self._prior_pool

    def set_prior_budget(self, budget: Optional[int]):
        """The bytes the values kept once may take (None: all of them when they fit the free device memory when they
        are first stored, else :data:`PRIOR_MEMORY_SHARE` of it): a decoder session's ``memory_budget``. Another
        budget drops the values held."""
        if budget is not None and (not isinstance(budget, int) or budget <= 0):
            raise ValueError(f"`memory_budget` must be a positive number of bytes, got {budget!r}.")
        if budget != self._prior_budget:
            self._prior_budget = budget
            self._prior_pool = None
            self._storage["prior_mars"] = None
            if self._prior_valid is not None:
                self._prior_valid[:] = False

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
            if self._prior_pool is not None:
                self._prior_pool.reset()
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
            self._class_mars = torch.empty(num_rows, be.token_classes.num_stored, device = self._device)
        for layer, first, end in zip(be.pc.input_layer_group, firsts, firsts[1:] + [num_rows]):
            lookup(layer.dist).class_masses(layer, be.token_classes, out = self._class_mars[first:end])
        self.class_mars_version += 1
        self._class_mars_at = self.class_mars_version
        if is_active():
            self._class_mars_kept = True
            register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return self._class_mars

    def token_major(self) -> torch.Tensor:
        """
        ``[V, num_rows]`` fp32 (see :meth:`class_mass_rows`): the probability of every token under every class-mass
        row, token-major, from every input layer's distribution's ``token_probs`` -- what :meth:`enumerated_probs`
        takes its columns from. Built by every call.
        """
        firsts, num_rows, _ = self.class_mass_rows()
        layers = list(self._backend.pc.input_layer_group)
        V = lookup(layers[0].dist).num_values(layers[0].dist)
        table = torch.empty(V, num_rows, dtype = torch.float32, device = self._device)
        for layer, first, end in zip(layers, firsts, firsts[1:] + [num_rows]):
            lookup(layer.dist).token_probs(layer, np.arange(end - first), table[:, first:end])
        return table

    def enumerated_probs(self) -> torch.Tensor:
        """
        ``[num_rows, num_enumerated]`` fp32: the probability of every enumerated class's token
        (``TokenClasses.enumerated_tokens``, in class order) under every class-mass row (of :meth:`token_major`), by
        row -- what an enumerated class's mass is read from (see
        :class:`~pyjuice.constraints.distributions.TokenClasses`; the kernels of :mod:`~.kernels.transitions`), at
        a token's column :attr:`~.kernels.transitions.EnumeratedTokens.index`: the lanes of a step share their row,
        so they read along one row of the table, not a column apiece. Built by every query, except inside a
        :func:`pyjuice.fast_inference` scope that allows parameter copies: there by the first query, then kept until
        the scope exits (the token-major table it is taken from is not kept).
        """
        if self._enum_probs is not None:
            return self._enum_probs
        tokens = self._backend.token_classes.enumerated_tokens.long()
        layers = list(self._backend.pc.input_layer_group)
        if all(hasattr(lookup(layer.dist), "token_columns") for layer in layers):     # the tokens' columns alone
            firsts, num_rows, _ = self.class_mass_rows()
            table = torch.empty(num_rows, tokens.numel(), dtype = torch.float32, device = self._device)
            for layer, first, end in zip(layers, firsts, firsts[1:] + [num_rows]):
                lookup(layer.dist).token_columns(layer, np.arange(end - first), tokens, table[first:end])
        else:
            table = self.token_major()[tokens].t().contiguous()
        if param_copies_allowed():
            self._enum_probs = table
            register_layer(self)                                    # its exit calls `_release_fast_inference_params`
        return table

    def _shared_rows(self) -> torch.Tensor:
        """int32 [num_rows]: every shared class-mass row's own number (its row of :meth:`enumerated_probs`)."""
        if self._prow is None:
            self._prow = torch.arange(self.class_mass_rows()[1], dtype = torch.int32, device = self._device)
        return self._prow

    def masses(self, external: Optional[ExternalInputs] = None) -> ClassMasses:
        """
        The class masses of a query's input nodes (:class:`ClassMasses`): :meth:`class_masses`, the same for every
        sample -- and under a decoder session's per-sample ``external`` inputs, those the inputs give the nodes that
        read them, per sample (their distributions' ``sample_inputs``), in a table of the session's: built by its
        first query (the other nodes' rows copied in), kept until its scope exits. With enumerated classes, their
        tokens' probabilities (:meth:`enumerated_probs`) and per-sample weights besides.
        """
        table = self.class_masses()
        _, _, mass_row = self.class_mass_rows()
        enum = self._backend.token_classes.num_enumerated > 0
        eprobs = self.enumerated_probs() if enum else None
        if external is None:
            return ClassMasses(table, mass_row, None, 1, self._class_mars_at, {}, prow = self._shared_rows(),
                               enum_probs = eprobs)
        if self._session is not None and self._session[0] is external and self._session[1] == self._class_mars_at:
            return self._session[2]
        if not external.layers:                     # gates alone: the class masses, as a computation of the session's
            self.class_mars_version += 1            # own (what is kept once under its gates is its own)
            masses = ClassMasses(table, mass_row, None, 1, self.class_mars_version, {}, prow = self._shared_rows(),
                                 enum_probs = eprobs)
            self._session = (external, self._class_mars_at, masses)
            if is_active():
                register_layer(self)                                # its exit calls `_release_fast_inference_params`
            return masses
        be, Be, C = self._backend, external.evidence_batch, self._backend.token_classes.num_stored
        input_start = be.input_range[0]
        firsts = self.class_mass_rows()[0]
        layers = [be.pc.input_layer_group[k] for k in external.layers]
        sizes = [lookup(layer.dist).sample_rows(layer, external.inputs) for layer in layers]
        full = torch.empty(table.size(0) + Be * sum(sizes), C, dtype = table.dtype, device = self._device)
        full[:table.size(0)] = table                                    # the other nodes' rows first
        row, stride = mass_row.clone(), torch.zeros_like(mass_row)
        lognorm = torch.zeros(full.size(0), dtype = torch.float32, device = self._device)
        prow = torch.empty(full.size(0), dtype = torch.int32, device = self._device)
        prow[:table.size(0)] = self._shared_rows()
        token_row = torch.full(mass_row.shape, -1, dtype = torch.int64, device = self._device)
        enum_logw = self._enumerated_weights(Be) if enum else None
        num_rows, sample, logw, ids, num_wrows = table.size(0), {}, [], [], 0
        for k, layer, rows_s in zip(external.layers, layers, sizes):
            out = full[num_rows:num_rows + Be * rows_s].view(Be, rows_s, C)
            si = lookup(layer.dist).sample_inputs(layer, be.token_classes, external.inputs, out)
            first, end = layer._output_ind_range
            row[first - input_start:end - input_start] = (num_rows + si.rows).to(torch.int32)
            if Be > 1:
                stride[first - input_start:end - input_start] = rows_s
            lognorm[num_rows:num_rows + Be * rows_s] = si.logz.reshape(-1)
            prow[num_rows:num_rows + Be * rows_s] = (firsts[k] + si.pair_rows).to(torch.int32).repeat(Be)
            token_row[first - input_start:end - input_start] = num_wrows + si.token_rows
            logw.append(si.token_logw)
            ids.append(si.token_ids)
            if enum:
                self._enumerated_weights_of(layer, si, enum_logw)
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
                             token_width = logw[0].size(2), token_ids = None if ids[0] is None else cat(ids),
                             prow = prow, enum_probs = eprobs, enum_logw = enum_logw)
        self._session = (external, self._class_mars_at, masses)
        if is_active():
            register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return masses

    def _enumerated_weights(self, Be: int) -> torch.Tensor:
        """fp32 [Be, enumerated tokens live at any boundary + 1], 0 (``log 1``): what :meth:`_enumerated_weights_of`
        fills for the positions an input layer's per-sample inputs reweight (the last column padding, as
        :class:`~.kernels.transitions.EnumeratedTokens`' tokens are)."""
        return torch.zeros(Be, self._backend.program().enum.num + 1, dtype = torch.float32, device = self._device)

    def _enumerated_weights_of(self, layer, si, enum_logw: torch.Tensor):
        """Into ``enum_logw`` (:meth:`_enumerated_weights`), at the boundaries of the layer's variables: every
        enumerated token's ``log w`` per sample from the layer's
        :class:`~pyjuice.constraints.distributions.SampleInputs` -- its weight among the first ``K`` tokens' (dense
        inputs), or its candidate's (top-k; -inf for none)."""
        en = self._backend.program().enum
        if en.num == 0:
            return
        ptr = en.ptr_host
        vids = si.variables.cpu().numpy().astype(np.int64)
        counts = ptr[vids + 1] - ptr[vids]
        if counts.sum() == 0:
            return
        dev = self._device
        pos = torch.from_numpy(np.repeat(vids, counts)).to(dev)                 # every token's boundary
        at = torch.cat([torch.arange(int(ptr[v]), int(ptr[v + 1])) for v in vids]).to(dev)   # ... and its number
        lv = layer.var_idmapping.to(dev)[pos].long()                             # the evidence's row of the boundary
        tok = en.tok[at].long()
        Be, Vl, K = si.token_logw.shape
        bs = torch.arange(Be, device = dev)[:, None]
        if si.token_ids is None:
            w = torch.where((tok < K)[None, :], si.token_logw[bs, lv[None, :], tok.clamp(max = K - 1)[None, :]],
                            float("-inf"))
        else:                                       # every boundary's tokens as a padded row of queries: one search
            most = int(counts.max())                # per (sample, evidence row), no gather of the candidates per token
            place = torch.from_numpy(np.concatenate([np.arange(c) for c in counts])).to(dev)
            queries = torch.full((Vl, most), K, dtype = torch.long, device = dev)   # (K: past every candidate)
            queries[lv, place] = tok
            ids = si.token_ids.long()                                            # [Be, Vl, K], ascending rows
            k = torch.searchsorted(ids, queries[None].expand(Be, -1, -1)).clamp(max = K - 1)    # [Be, Vl, most]
            hit = ids.gather(2, k) == queries[None]
            w = torch.where(hit, si.token_logw.gather(2, k), float("-inf"))[:, lv, place]      # [Be, tokens]
        enum_logw[:, at] = w.expand(enum_logw.size(0), -1)

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
        self._enum_probs = None
        self._prior_kept = False
        self._session = None
        if poison and self._class_mars is not None:
            self._class_mars.fill_(float("nan"))
        if poison and self._storage.get("prior_mars") is not None:
            self._storage["prior_mars"].fill_(float("nan"))
        if poison and self._prior_flows_state is not None:      # rows given again (filled with -inf) on next use
            R = len(self._backend.sum_regions)
            self._prior_flows_state["flows"].fill_(float("nan"))
            self._prior_flows_state = dict(self._prior_flows_state, offset = [-1] * R, width = [0] * R, size = 0)
            self._prior_flows_state.pop("offset_t")
