"""
The result of :func:`pyjuice.constraints.compile`: a PC under a constraint, ready for queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from .distributions import TokenClasses
from .language.base import Constraint
from .structure import SHAPES, PCStructure, analyze_structure
from .backends.lifted.plan import BoundaryLayout, build_pc_tables, buffer_layout


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
            :class:`TensorCircuit`'s ``node_mars``), ``class_mars`` (``[num_input_rows, num_classes]``) and
            ``layout`` (the :func:`buffer_layout` dict, plus ``sum_offsets`` / ``sum_widths`` as int64
            tensors on the device for the kernels)
        """
        B = int(batch_size)
        layout = self._layouts.get(B)
        if layout is None:
            layout = buffer_layout(self.input_range, self.sum_regions, self.element_regions, self.num_classes, B)
            layout["sum_offsets_t"] = torch.tensor(layout["sum_offsets"], dtype = torch.int64, device = self._device)
            layout["sum_widths_t"] = torch.tensor(layout["sum_widths"], dtype = torch.int64, device = self._device)
            self._layouts[B] = layout
        node_mars = self._storage_view("node_mars", layout["node_size"])
        element_mars = self._storage_view("element_mars", layout["element_size"])
        input_start, input_end = self.input_range
        class_offset, num_input_rows = layout["class_offset"], input_end - input_start
        return dict(node_mars = node_mars, element_mars = element_mars,
                    input_mars = node_mars[:input_end * B].view(input_end, B),
                    class_mars = node_mars[class_offset:class_offset + num_input_rows * self.num_classes].view(
                        num_input_rows, self.num_classes),
                    layout = layout)

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
        """Bytes of the lifted buffers for a batch of ``batch_size`` (fp32, alignment included)."""
        layout = buffer_layout(self.input_range, self.sum_regions, self.element_regions, self.num_classes, batch_size)
        return 4 * (layout["node_size"] + layout["element_size"])

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

    def conditional(self, *args, **kwargs):
        """Per-variable distributions given the constraint (and evidence). Not implemented yet."""
        self._check_pc_unchanged()
        raise NotImplementedError("`conditional` under a constraint is not implemented yet.")

    def sample(self, *args, **kwargs):
        """Samples from the PC conditioned on the constraint (and evidence). Not implemented yet."""
        self._check_pc_unchanged()
        raise NotImplementedError("`sample` under a constraint is not implemented yet.")

    def decoder(self, *args, **kwargs):
        """Incremental (token-by-token) constrained decoding. Not implemented yet."""
        self._check_pc_unchanged()
        raise NotImplementedError("`decoder` under a constraint is not implemented yet.")
