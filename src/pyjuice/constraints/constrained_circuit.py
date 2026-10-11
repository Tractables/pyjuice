"""
The result of :func:`pyjuice.constraints.compile`: a PC under a constraint, ready for queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from pyjuice.utils.fast_inference import bf16_params_allowed, is_active, param_copies_allowed, register_layer

from .distributions import TokenClasses, lookup
from .language.base import Constraint
from .structure import SHAPES, PCStructure, analyze_structure
from .backends.lifted.plan import BoundaryLayout, _align, build_pc_tables, buffer_layout


class ConstrainedCircuit:
    """
    A PC under a constraint: the PC, the constraint, and everything compiled from the pair. Queries
    under the constraint are its methods.

    Everything it holds besides the PC depends only on the constraint and the PC's STRUCTURE, never on
    parameter values or evidence, so it stays valid while the PC's parameters change (e.g. CoDD's
    external parameters at every step). The PC itself is fixed: :attr:`pc` cannot be reassigned
    (:meth:`with_pc` puts the same constraint on another PC with the same structure), and it moves to
    another device together with everything compiled from the pair, through :meth:`to`. A PC moved on its
    own makes every query raise.

    Constrained queries run in buffers of their own, laid out for the lifted plan (see
    :func:`~pyjuice.constraints.backends.lifted.plan.buffer_layout`) and allocated on the first query, so
    creating a constrained circuit releases the PC's activation buffers
    (:meth:`TensorCircuit.free_activation_buffers`; its parameter flows are kept). The PC allocates them
    again on its next own pass.

    The input nodes' token-class masses depend only on the parameters, so inside :func:`pyjuice.fast_inference`
    (whose contract is that parameters do not change) they are computed by the first query and kept until the
    scope exits; outside it, every query computes them. So do the input-independent inside values -- those of the
    nodes whose scope no sample of a query observes (see :mod:`~.backends.lifted.query_plan`), held once in
    ``prior_mars`` and read by every sample -- each kept from the query that first computes it. Likewise the stacked
    weights of dense sum node blocks that read the same children (one matrix product instead of one per block, on
    products narrow enough for it to pay): inside a scope that allows parameter copies, each is copied once and kept
    until the scope exits; outside it, a query copies only the weights it reads more than once (tied ones) and drops
    them when it returns. So do the token-major copies of the input parameters that conditionals score against
    (:meth:`_token_table`): kept by a scope that allows parameter copies, else built by every query.

    Created by :func:`pyjuice.constraints.compile`; not meant to be constructed directly.

    :ivar pc: the PC queries run on
    :ivar constraint: the constraint
    :ivar structure: the PC's :class:`~pyjuice.constraints.structure.PCStructure`
    :ivar automaton: the constraint's automaton
    :ivar layout: the :class:`~pyjuice.constraints.backends.lifted.plan.BoundaryLayout` of the automaton
        over the PC's variables
    :ivar token_classes: the automaton's :class:`~pyjuice.constraints.distributions.TokenClasses`, on the PC's
        device, as the input distributions read them
    :ivar columns_per_sample: the most slots any sum or product node's block takes per sample
        (see :func:`~pyjuice.constraints.backends.lifted.plan.build_pc_tables`)
    :ivar product_rows: per product layer, the rows and boundaries the lifted products read
    :ivar input_range: the ``node_mars`` rows of all input nodes
    :ivar root_rows: the ``node_mars`` rows of the root nodes
    :ivar sum_regions: ``(first_row, end_row, slots)`` of every sum node group's region in ``node_mars``
    :ivar element_regions: ``(first_row, end_row, slots)`` of every product layer group in ``element_mars``
    :ivar compile_time_s: wall-clock seconds :func:`~pyjuice.constraints.compile` (or :meth:`with_pc`) took
    """

    #: the backend queries run on
    backend = "lifted"
    #: whether queries are exact (up to floating point)
    exact = True

    def __init__(self, pc, constraint: Constraint, structure: PCStructure, automaton,
                 layout: BoundaryLayout, tables: Dict[str, Any], compile_time_s: float,
                 token_classes: Optional[TokenClasses] = None):
        self._pc = pc
        self._device = pc.params.device             # where the tables live
        pc.free_activation_buffers()                # constrained queries use buffers of their own
        self.constraint = constraint
        self.structure = structure
        self.automaton = automaton
        self.layout = layout
        if token_classes is None or token_classes.token_class.device != self._device:
            token_classes = TokenClasses(layout.token_class.to(self._device), automaton.num_classes)
        self.token_classes = token_classes
        self.columns_per_sample = tables["columns_per_sample"]
        self.product_rows = tables["product_rows"]
        self.input_range = tables["input_range"]
        self.root_rows = tables["root_rows"]
        self.sum_regions = tables["sum_regions"]
        self.element_regions = tables["element_regions"]
        self.compile_time_s = compile_time_s
        self._storage = {}                          # buffer name -> backing storage (see `_buffers`)
        self._layouts = {}                          # batch size -> `buffer_layout` and its device tensors
        self._program = None                        # the lifted forward's tables (see `_lifted_program`)
        self._class_rows = None                     # where the input nodes' class masses are (see `_class_mass_rows`)
        self._class_mars = None                     # the input nodes' class masses (see `_class_masses`)
        self._class_mars_kept = False               # kept by a `pyjuice.fast_inference` scope
        self._class_mars_version = 0                # counts their computations (what is derived from them follows)
        self._kept_stacks = None                    # stacked dense weights kept by a scope (see `_weight_stacks`)
        self._prior_valid = None                    # which input-independent values prior_mars holds (`_prior_validity`)
        self._prior_kept = False                    # kept by a `pyjuice.fast_inference` scope
        self._prior_flows_state = None              # the input-independent top-down values (see `_prior_flows`)
        self._token_tables = None                   # token-major input parameters kept by a scope (`_token_table`)

    @property
    def pc(self):
        """The PC queries run on (fixed; see the class docstring)."""
        return self._pc

    def _check_pc_unchanged(self):
        """Refuse a query on a PC that moved on its own (one attribute comparison per query)."""
        if self._pc.params.device != self._device:
            raise RuntimeError(f"The PC moved from {self._device} to {self._pc.params.device} on its own, but the "
                               f"constrained circuit's tables are still on {self._device}. Move both with "
                               f"`cc.to(device)`, or compile the constraint again.")

    def to(self, device) -> "ConstrainedCircuit":
        """
        Move the PC and everything compiled from the pair to ``device`` (in place, like
        :meth:`TensorCircuit.to`), and return this constrained circuit.

        :param device: an int ordinal, a string such as ``"cuda:1"`` or ``"cpu"``, or a ``torch.device``
        """
        self._pc.to(device)
        device = self._pc.params.device
        self.product_rows = [tuple(t.to(device) for t in rows) for rows in self.product_rows]
        self.token_classes = self.token_classes.to(device)
        self._storage, self._layouts = {}, {}       # the buffers are allocated again on the new device
        self._class_rows, self._class_mars, self._class_mars_kept = None, None, False
        self._kept_stacks = None
        self._prior_valid, self._prior_kept = None, False
        self._prior_flows_state = None
        self._token_tables = None
        self._program = None                        # and the forward's tables built again there
        self._device = device
        return self

    # ---------------------------------------------------------------------------------------------
    # Buffers
    # ---------------------------------------------------------------------------------------------

    def _buffers(self, batch_size: int) -> Dict[str, Any]:
        """
        The lifted buffers for a batch of ``batch_size``, cut from storage this constrained circuit owns
        (:func:`~pyjuice.constraints.backends.lifted.plan.buffer_layout` says where every region sits).

        Each buffer has one backing storage, reused while it is large enough and at most 4 times larger
        than needed (so a loop whose batch size varies does not reallocate, and keeps its addresses), and
        filled with -inf only when allocated: every slot a kernel reads is written first, and padding is
        never read.

        :returns: a dict with ``node_mars`` and ``element_mars`` (flat), ``input_mars`` (a contiguous
            ``[input_end, batch_size]`` view, which an input layer writes as it writes a
            :class:`TensorCircuit`'s ``node_mars``) and ``layout`` (the :func:`buffer_layout` dict, plus
            ``sum_offsets`` / ``sum_widths`` as int64 tensors on the device for the kernels)
        """
        B = int(batch_size)
        layout = self._layout(B)
        node_mars = self._storage_view("node_mars", layout["node_size"])
        element_mars = self._storage_view("element_mars", layout["element_size"])
        return dict(node_mars = node_mars, element_mars = element_mars,
                    input_mars = node_mars[:self.input_range[1] * B].view(self.input_range[1], B), layout = layout)

    def _layout(self, B: int) -> Dict[str, Any]:
        """The :func:`buffer_layout` of a batch of ``B``, with ``sum_offsets_t`` / ``sum_widths_t`` on the device."""
        layout = self._layouts.get(B)
        if layout is None:
            layout = buffer_layout(self.input_range, self.sum_regions, self.element_regions, B)
            layout["sum_offsets_t"] = torch.tensor(layout["sum_offsets"], dtype = torch.int64, device = self._device)
            layout["sum_widths_t"] = torch.tensor(layout["sum_widths"], dtype = torch.int64, device = self._device)
            self._layouts[B] = layout
        return layout

    def _flows(self, batch_size: int, bufs: Dict[str, Any]) -> Dict[str, Any]:
        """
        The top-down buffers of a batch (see :mod:`~.backends.lifted.backward`): ``node_flows`` (laid out as
        ``node_mars``, storage of its own, allocated on first use) and ``element_flows``, which is ``element_mars``'s
        storage -- the top-down pass never reads the product insides there.
        """
        return dict(node_flows = self._storage_view("node_flows", bufs["layout"]["node_size"]),
                    element_flows = bufs["element_mars"])

    def _top_down_program(self):
        """The top-down pass's tables (:class:`~.backends.lifted.backward.TopDownProgram`), built on first use and
        kept with the forward's."""
        prog = self._lifted_program()
        if getattr(prog, "top_down", None) is None:
            from .backends.lifted.backward import top_down_program
            prog.top_down = top_down_program(self)
        return prog.top_down

    def _prior_buffer(self) -> Dict[str, Any]:
        """
        ``prior_mars``: the input-independent inside values (see :mod:`~.backends.lifted.query_plan`), laid out as
        ``node_mars`` for a batch of one, in storage of its own (it outlives queries of any batch size), allocated
        on first use. A dict with ``mars`` (flat), and every sum region's ``offset`` and ``width`` there (Python
        ints) and ``offset_t`` / ``width_t`` (int64 on the device).
        """
        layout = self._layout(1)
        return dict(mars = self._storage_view("prior_mars", layout["node_size"]), offset = layout["sum_offsets"],
                    width = layout["sum_widths"], offset_t = layout["sum_offsets_t"], width_t = layout["sum_widths_t"])

    def _prior_flows(self, regions) -> Dict[str, Any]:
        """
        ``prior_flows``: the input-independent top-down values (see :mod:`~.backends.lifted.query_plan`), for a batch
        of one, in storage of its own laid out APPEND-ONLY: a sum region (node) gets its rows, filled with -inf, the
        first time a query keeps its values once (``regions``, bool per sum region), and keeps them, so the storage
        holds only the regions ever kept once and never more than one sample's ``node_flows``. Its values are kept as
        ``prior_mars``'s (see :meth:`_prior_validity`). A dict with ``flows`` (flat), every region's ``offset`` and
        ``width`` (Python ints; offset -1 where it has no rows yet) and ``offset_t`` / ``width_t`` (int64 on the
        device), and ``valid`` (bool per sum region on the host: the regions whose values it holds).
        """
        pf = self._prior_flows_state
        if pf is None:
            R = len(self.sum_regions)
            pf = self._prior_flows_state = dict(flows = torch.empty(0, device = self._device), offset = [-1] * R,
                                                width = [0] * R, size = 0, valid = np.zeros(R, dtype = bool))
        new = [r for r in np.nonzero(regions)[0].tolist() if pf["offset"][r] < 0]
        if new or "offset_t" not in pf:
            start = pf["size"]
            for r in new:
                first, end, slots = self.sum_regions[r]
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

    def _prior_validity(self):
        """
        ``(valid, kept)`` for a query starting now: which inside values ``prior_mars`` holds (bool on the host, one
        per sum interval of the lifted Program, ``Program.sum_ivs``, set by the queries that compute them), and
        whether they outlive the query. They depend on the parameters only, so they are kept as the class masses are:
        inside :func:`pyjuice.fast_inference` until the scope exits; outside it, every query starts with none.
        """
        if self._prior_valid is None or not self._prior_kept:
            self._prior_valid = torch.zeros(len(self._lifted_program().sum_ivs), dtype = torch.bool)
            if self._prior_flows_state is not None:
                self._prior_flows_state["valid"][:] = False
            if is_active():
                self._prior_kept = True
                register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return self._prior_valid, self._prior_kept

    def _class_mass_rows(self):
        """
        Where the class masses of every input node are: ``(firsts, num_rows, mass_row)``, every input layer's
        class-mass table (see :mod:`pyjuice.constraints.distributions`) being rows ``firsts[k] ..`` of the
        ``num_rows`` rows of :meth:`_class_masses`, and input node ``u`` reading row ``mass_row[u - input_start]``
        (int32, on the device). Built on first use: it depends only on the PC's structure.
        """
        if self._class_rows is None:
            input_start, input_end = self.input_range
            mass_row = torch.zeros(input_end - input_start, dtype = torch.int32, device = self._device)
            firsts, num_rows = [], 0
            for layer in self._pc.input_layer_group:
                rows = lookup(layer.dist).class_mass_rows(layer)
                first, end = layer._output_ind_range
                mass_row[first - input_start:end - input_start] = (num_rows + rows).to(torch.int32)
                firsts.append(num_rows)
                num_rows += int(rows.max()) + 1
            self._class_rows = (firsts, num_rows, mass_row)
        return self._class_rows

    def _class_masses(self) -> torch.Tensor:
        """
        ``[num_rows, num_classes]`` (see :meth:`_class_mass_rows`): the log-mass of every token class for every
        input layer's class-mass row, by its distribution's functions in :mod:`pyjuice.constraints.distributions`.
        Computed on every query, except inside :func:`pyjuice.fast_inference`: there by the first query, then kept
        until the scope exits.
        """
        if self._class_mars_kept:
            return self._class_mars
        firsts, num_rows, _ = self._class_mass_rows()
        if self._class_mars is None:
            self._class_mars = torch.empty(num_rows, self.num_classes, device = self._device)
        for layer, first, end in zip(self._pc.input_layer_group, firsts, firsts[1:] + [num_rows]):
            lookup(layer.dist).class_masses(layer, self.token_classes, out = self._class_mars[first:end])
        self._class_mars_version += 1
        if is_active():
            self._class_mars_kept = True
            register_layer(self)                                # its exit calls `_release_fast_inference_params`
        return self._class_mars

    def _weight_stacks(self):
        """
        ``(stacks, kept)``: where a query finds and leaves the stacked weights of its dense sum node blocks
        (stack key -> :func:`~pyjuice.constraints.backends.lifted.kernels.sum.stacked_weights`, see
        :meth:`~pyjuice.constraints.backends.lifted.forward.Program`), and whether they outlive the query. Inside a
        :func:`pyjuice.fast_inference` scope that allows parameter copies, the scope's: kept until it exits, so the
        query stacks every run it can (see ``STACK_MAX_COLUMNS`` there); else an empty dict of the query's own, which
        stacks only reused weights.
        """
        if self._kept_stacks is not None:
            return self._kept_stacks, True
        if param_copies_allowed():
            self._kept_stacks = {}
            register_layer(self)                                    # its exit calls `_release_fast_inference_params`
            return self._kept_stacks, True
        return {}, False

    def _token_table(self, group: int, rows: np.ndarray) -> torch.Tensor:
        """
        ``[V, len(rows)]``: the probability of every token under the class-mass rows ``rows`` (sorted, see
        :meth:`_class_mass_rows`), token-major -- the table row group ``group`` of the conditionals is scored against
        (:mod:`~.backends.lifted.conditional`) -- from every input layer's distribution's ``token_probs``. Built by
        every query, except inside a :func:`pyjuice.fast_inference` scope that allows parameter copies: there by the
        first query, then kept until the scope exits; in bfloat16 when the scope allows bf16 parameter copies.
        """
        dtype = torch.bfloat16 if bf16_params_allowed() else torch.float32
        key = (group, dtype)
        if self._token_tables is not None and key in self._token_tables:
            return self._token_tables[key]
        firsts, num_rows, _ = self._class_mass_rows()
        layers = list(self._pc.input_layer_group)
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
        if poison and self._class_mars is not None:
            self._class_mars.fill_(float("nan"))
        if poison and "prior_mars" in self._storage:
            self._storage["prior_mars"].fill_(float("nan"))
        if poison and self._prior_flows_state is not None:      # rows given again (filled with -inf) on next use
            self._prior_flows_state["flows"].fill_(float("nan"))
            self._prior_flows_state = dict(self._prior_flows_state, offset = [-1] * len(self.sum_regions),
                                           width = [0] * len(self.sum_regions), size = 0)
            self._prior_flows_state.pop("offset_t")

    def _lifted_program(self):
        """The lifted forward's tables (:class:`~pyjuice.constraints.backends.lifted.forward.Program`), built on
        the first query and kept until the circuit moves."""
        if self._program is None:
            from .backends.lifted.forward import Program
            self._program = Program(self)
        return self._program

    def _storage_view(self, name: str, numel: int) -> torch.Tensor:
        storage = self._storage.get(name)
        if storage is None or storage.numel() < numel or storage.numel() > 4 * max(numel, 1):
            storage = self._storage[name] = None                                    # free the old one first
            storage = self._storage[name] = torch.full((max(numel, 1),), -float("inf"), device = self._device)
        return storage[:numel]

    def buffer_bytes(self, batch_size: int) -> int:
        """Bytes of the lifted buffers for a batch of ``batch_size`` and of the class masses (fp32, alignment
        included)."""
        layout = buffer_layout(self.input_range, self.sum_regions, self.element_regions, batch_size)
        return 4 * (layout["node_size"] + layout["element_size"] + self._class_mass_rows()[1] * self.num_classes)

    # ---------------------------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of positions: the PC's number of variables."""
        return self.structure.num_vars

    @property
    def satisfiable(self) -> bool:
        """Whether any assignment of the PC's variables satisfies the constraint. If not, every query
        returns probability zero."""
        return self.layout.satisfiable

    @property
    def width_per_boundary(self) -> torch.Tensor:
        """[n+1] number of automaton states that can occur at each boundary (columns per boundary)."""
        return self.layout.width

    @property
    def max_width(self) -> int:
        """The largest number of columns at any boundary (0 if the constraint is unsatisfiable)."""
        return int(self.layout.width.max())

    @property
    def num_states(self) -> int:
        return self.automaton.num_states

    @property
    def num_classes(self) -> int:
        return self.automaton.num_classes

    @property
    def shape_counts(self) -> Dict[str, int]:
        """Number of sum and product node groups per scope shape (see
        :data:`~pyjuice.constraints.structure.SHAPES`)."""
        counts = {shape: 0 for shape in SHAPES}
        for info in self.structure.nodes:
            if info.kind != "input":
                counts[info.shape] += 1
        return counts

    @property
    def bytes_per_sample(self) -> int:
        """
        Memory the lifted buffers need per sample, in fp32: every sum node group's block, the largest product
        layer group's blocks, and one log-probability per input row -- without the alignment padding and the
        class masses, which do not grow with the batch (:meth:`buffer_bytes` has the exact size).
        """
        sums = sum((end - first) * slots for first, end, slots in self.sum_regions)
        elements = max(((end - first) * slots for first, end, slots in self.element_regions), default = 0)
        return 4 * (sums + elements + self.input_range[1])

    def info(self) -> Dict[str, Any]:
        """A summary of the constrained circuit."""
        return dict(backend = self.backend, exact = self.exact, n = self.n, satisfiable = self.satisfiable,
                    num_states = self.num_states, num_classes = self.num_classes, max_width = self.max_width,
                    width_per_boundary = self.width_per_boundary.tolist(), shape_counts = self.shape_counts,
                    columns_per_sample = self.columns_per_sample, bytes_per_sample = self.bytes_per_sample,
                    compile_time_s = self.compile_time_s, constraint = self.constraint.info())

    def __repr__(self) -> str:
        return (f"ConstrainedCircuit(backend={self.backend}, n={self.n}, num_states={self.num_states}, "
                f"max_width={self.max_width}, columns_per_sample={self.columns_per_sample}, "
                f"satisfiable={self.satisfiable})")

    # ---------------------------------------------------------------------------------------------
    # Rebinding
    # ---------------------------------------------------------------------------------------------

    def with_pc(self, pc) -> "ConstrainedCircuit":
        """
        The same constraint on another PC with the same structure (e.g. a drafter and a verifier, or a
        copy of the PC on another device). The structure analysis is checked by signature and reused, as
        are the automaton and its layout; only the PC's rows are read again (one pass over its node
        groups), since nothing guarantees that two circuits number their nodes alike.

        :param pc: a compiled PC whose structure equals this one's (parameters may differ)
        :type pc: TensorCircuit
        """
        from .compiler import ConstraintCompileError, _check_pc

        t0 = time.perf_counter()
        _check_pc(pc)
        structure = analyze_structure(pc)
        if structure.signature != self.structure.signature:
            raise ConstraintCompileError("Cannot rebind the constrained circuit: the new PC's structure differs "
                                         "from the one the constraint was compiled against. Compile the "
                                         "constraint against the new PC with `pyjuice.constraints.compile` "
                                         "instead.")
        tables = build_pc_tables(structure, self.layout, pc)
        return ConstrainedCircuit(pc, self.constraint, structure, self.automaton, self.layout, tables,
                                  compile_time_s = time.perf_counter() - t0, token_classes = self.token_classes)

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                 precision: str = "fp32") -> torch.Tensor:
        """
        ``log p(C, e)`` for every sample: the log-probability that the PC generates a string that satisfies the
        constraint and agrees with the observed tokens.

        :param data: [B, n] token ids (ignored where missing)
        :type data: torch.Tensor

        :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
        :type missing_mask: Optional[torch.Tensor]

        :param precision: "fp32" (the default; fp32-level accuracy) or "tf32" for the sum layers' matrix
            products
        :type precision: str

        :returns: [B, number of root nodes], as :func:`pyjuice.queries.marginal`
        """
        self._check_pc_unchanged()
        from .backends.lifted.forward import marginal
        return marginal(self, data, missing_mask, precision = precision)

    def conditional(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                    targets: Optional[torch.Tensor] = None, precision: str = "fp32",
                    return_log_marginal: bool = False):
        """
        ``log p(x_t = v | e, C)``: the distribution of every target position's token given the observed tokens and the
        constraint, for every target (sample, position).

        :param data: [B, n] token ids (ignored where missing)
        :type data: torch.Tensor

        :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
        :type missing_mask: Optional[torch.Tensor]

        :param targets: None (every missing position of every sample), or a bool mask [n] or [B, n] of missing
            positions
        :type targets: Optional[torch.Tensor]

        :param precision: "fp32" (the default; fp32-level accuracy) or "tf32" for the matrix products; inside
            ``pyjuice.fast_inference(allow_bf16_params = True)`` the token probabilities are scored in bfloat16
        :type precision: str

        :param return_log_marginal: also return ``log p(C, e)`` of every row's sample
        :type return_log_marginal: bool

        :returns: [number of targets, V], one row per target in the order ``dense[targets]`` lists them (sample by
            sample, positions in order), -inf rows where ``p(C, e) = 0``; with ``return_log_marginal``, also
            [number of targets]
        """
        self._check_pc_unchanged()
        from .backends.lifted.conditional import conditional
        return conditional(self, data, missing_mask, targets, precision = precision,
                           return_log_marginal = return_log_marginal)

    def sample(self, *args, **kwargs):
        """Samples from the PC conditioned on the constraint (and evidence). Not implemented yet."""
        self._check_pc_unchanged()
        raise NotImplementedError("`sample` under a constraint is not implemented yet.")

    def decoder(self, *args, **kwargs):
        """Incremental (token-by-token) constrained decoding. Not implemented yet."""
        self._check_pc_unchanged()
        raise NotImplementedError("`decoder` under a constraint is not implemented yet.")
