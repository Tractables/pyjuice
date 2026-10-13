"""
:class:`DFAExactBackend`: the ``dfa_exact`` backend as :func:`pyjuice.constraints.compile` sees it -- which pairs of a
constraint and a PC it serves, what it compiles from them, and its queries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ... import distributions
from ...language.base import Constraint
from ...structure import PCStructure, _runs
from ..base import Backend
from .compiled.layout import BoundaryLayout, build_layout
from .compiled.tables import build_pc_tables, buffer_layout
from .state import State


class DFAExactBackend(Backend):
    """
    Exact queries under a constraint with a finite-state form (the ``automaton`` capability), on a PC whose node
    scopes are contiguous runs in variable order: the automaton's states become extra columns of the PC's node values
    (see the package docstring).

    Everything it holds besides the PC and :attr:`state` depends only on the constraint and the PC's STRUCTURE, never
    on parameter values or evidence. Building it releases the PC's activation buffers
    (:meth:`TensorCircuit.free_activation_buffers`; its parameter flows are kept): its queries run in buffers of
    their own (:class:`~.state.State`). The PC allocates them again on its next own pass.

    :ivar pc: the PC queries run on
    :ivar constraint: the constraint
    :ivar structure: the PC's :class:`~pyjuice.constraints.structure.PCStructure`
    :ivar automaton: the constraint's automaton
    :ivar layout: the :class:`~.layout.BoundaryLayout` of the automaton over the PC's variables
    :ivar token_classes: the automaton's :class:`~pyjuice.constraints.distributions.TokenClasses`, on the PC's
        device, as the input distributions read them
    :ivar columns_per_sample: the most slots any sum or product node's block takes per sample
        (see :func:`~.tables.build_pc_tables`)
    :ivar product_rows: per product layer, the rows and boundaries the lifted products read
    :ivar input_range: the ``node_mars`` rows of all input nodes
    :ivar root_rows: the ``node_mars`` rows of the root nodes
    :ivar sum_regions: ``(first_row, end_row, slots)`` of every sum node group's region in ``node_mars``
    :ivar element_regions: ``(first_row, end_row, slots)`` of every product layer group in ``element_mars``
    :ivar device: where the tables and :attr:`state` live
    :ivar state: the :class:`~.state.State` its queries run in
    """

    name = "dfa_exact"
    exact = True

    def __init__(self, pc, constraint: Constraint, structure: PCStructure, automaton, layout: BoundaryLayout,
                 tables: Dict[str, Any], token_classes: Optional[distributions.TokenClasses] = None):
        self.pc = pc
        self.device = pc.params.device
        pc.free_activation_buffers()                # queries use buffers of their own
        self.constraint = constraint
        self.structure = structure
        self.automaton = automaton
        self.layout = layout
        if token_classes is None or token_classes.token_class.device != self.device:
            token_classes = distributions.TokenClasses(layout.token_class.to(self.device), automaton.num_classes)
        self.token_classes = token_classes
        self.columns_per_sample = tables["columns_per_sample"]
        self.product_rows = tables["product_rows"]
        self.input_range = tables["input_range"]
        self.root_rows = tables["root_rows"]
        self.sum_regions = tables["sum_regions"]
        self.element_regions = tables["element_regions"]
        self._program = None                        # the forward's tables (see `program`)
        self.state = State(self)

    # ---------------------------------------------------------------------------------------------
    # Compiling
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def refusals(cls, constraint: Constraint, structure: PCStructure) -> List[str]:
        """Every reason this backend cannot serve the pair (empty if it can)."""
        reasons = []

        caps = constraint.capabilities()
        automaton = None
        if "automaton" in caps:
            automaton = constraint.automaton()
        else:
            reasons.append(f"the constraint ({type(constraint).__name__}) has no `automaton` capability, which "
                           f"exact lifting needs; it supports: {', '.join(sorted(caps))}")

        by_reason = defaultdict(list)
        for info in structure.nodes:
            why = _unsupported_reason(info.ns, info.kind)
            if why is not None:
                by_reason[why].append(info.ns)
        for why, group in by_reason.items():
            reasons.append(f"{_count(len(group), 'node group')} {'is' if len(group) == 1 else 'are'} not supported: "
                           f"{why}; e.g. over variables {_format_vars(group[0].scope.to_list())}")

        vars_by_values = defaultdict(list)                              # (distribution, values) -> variables
        for info in structure.nodes:
            module = distributions.lookup(info.ns.dist) if info.kind == "input" else None
            if module is not None:
                key = (type(info.ns.dist).__name__, module.num_values(info.ns.dist))
                vars_by_values[key].extend(info.ns.scope.to_list())
        for (name, num_values), vs in sorted(vars_by_values.items()):
            if num_values != constraint.vocab_size:
                reasons.append(f"vocabulary mismatch: the constraint reads tokens 0..{constraint.vocab_size - 1} "
                               f"(vocab_size {constraint.vocab_size}), but the {name} input nodes over variables "
                               f"{_format_vars(vs)} range over {num_values} values")

        fragmented = [info for info in structure.nodes if info.shape == "fragmented"]
        if fragmented:
            worst = max(fragmented, key = lambda info: info.num_runs)   # the first one with the most runs
            r = worst.num_runs
            msg = (f"the node group over variables {_format_runs(worst.scope_runs)} has {r} separate runs in "
                   f"variable order ({_count(len(fragmented), 'node group')} "
                   f"{'is' if len(fragmented) == 1 else 'are'} fragmented, with up to {r} runs). A constraint reads "
                   f"the PC's variables in index order 0..{structure.num_vars - 1}, and this version only supports "
                   f"PCs whose node scopes are contiguous in that order (e.g. HMMs, 1-D PD over a sequence). A node "
                   f"with r runs would need a state of size up to K^(2r)")
            if automaton is not None:
                K = automaton.num_states
                msg += (f"; with this constraint's K = {K} states and r = {r} that is {_format_big(K ** (2 * r))} "
                        f"per node")
            reasons.append(msg)

        return reasons

    @classmethod
    def build(cls, constraint: Constraint, pc, structure: PCStructure) -> "DFAExactBackend":
        automaton = constraint.automaton()
        layout = build_layout(automaton, structure.num_vars)
        return cls(pc, constraint, structure, automaton, layout, build_pc_tables(structure, layout, pc))

    def rebind(self, pc, structure: PCStructure) -> "DFAExactBackend":
        """The same constraint on another PC with the same structure: the automaton, its layout and token classes are
        reused; only the PC's rows are read again (nothing guarantees that two circuits number their nodes alike)."""
        return DFAExactBackend(pc, self.constraint, structure, self.automaton, self.layout,
                               build_pc_tables(structure, self.layout, pc), token_classes = self.token_classes)

    def to(self, device) -> None:
        device = torch.device(device)
        self.product_rows = [tuple(t.to(device) for t in rows) for rows in self.product_rows]
        self.token_classes = self.token_classes.to(device)
        self.device = device
        self._program = None                        # the forward's tables built again there
        self.state = State(self)                    # and the buffers allocated again

    def program(self):
        """The lifted forward's tables (:class:`~.program.Program`), built on the first query and kept until the
        backend moves."""
        if self._program is None:
            from .compiled.program import Program
            self._program = Program(self)
        return self._program

    def top_down_program(self):
        """The top-down pass's tables (:class:`~.top_down_program.TopDownProgram`), built on first use and kept with the
        forward's."""
        prog = self.program()
        if getattr(prog, "top_down", None) is None:
            from .compiled.top_down_program import top_down_program
            prog.top_down = top_down_program(self)
        return prog.top_down

    # ---------------------------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of positions: the PC's number of variables."""
        return self.structure.num_vars

    @property
    def satisfiable(self) -> bool:
        """Whether any assignment of the PC's variables satisfies the constraint. If not, every query returns
        probability zero."""
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
    def bytes_per_sample(self) -> int:
        """
        Memory the lifted buffers need per sample, in fp32: every sum node group's block, the largest product layer
        group's blocks, and one log-probability per input row -- without the alignment padding and the class masses,
        which do not grow with the batch (:meth:`buffer_bytes` has the exact size).
        """
        sums = sum((end - first) * slots for first, end, slots in self.sum_regions)
        elements = max(((end - first) * slots for first, end, slots in self.element_regions), default = 0)
        return 4 * (sums + elements + self.input_range[1])

    def buffer_bytes(self, batch_size: int) -> int:
        """Bytes of the lifted buffers for a batch of ``batch_size`` and of the class masses (fp32, alignment
        included)."""
        layout = buffer_layout(self.input_range, self.sum_regions, self.element_regions, batch_size)
        return 4 * (layout["node_size"] + layout["element_size"] + self.state.class_mass_rows()[1] * self.num_classes)

    def info(self) -> Dict[str, Any]:
        return dict(satisfiable = self.satisfiable, num_states = self.num_states, num_classes = self.num_classes,
                    max_width = self.max_width, width_per_boundary = self.width_per_boundary.tolist(),
                    columns_per_sample = self.columns_per_sample, bytes_per_sample = self.bytes_per_sample)

    def __repr__(self) -> str:
        return (f"{self.name}(num_states={self.num_states}, max_width={self.max_width}, "
                f"columns_per_sample={self.columns_per_sample}, satisfiable={self.satisfiable})")

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                 precision: str = "fp32", storage: str = "auto") -> torch.Tensor:
        """
        ``log p(C, e)`` for every sample (see :meth:`~pyjuice.constraints.ConstrainedCircuit.marginal`).

        :param precision: "fp32" (the default; fp32-level accuracy) or "tf32" for the sum layers' matrix products
        :param storage: "auto" (the default), "dense" or "compact" (see :meth:`conditional`; a marginal keeps inside
            values only)
        """
        from .queries.marginal import marginal
        return marginal(self, data, missing_mask, precision = precision, storage = storage)

    def conditional(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                    targets: Optional[torch.Tensor] = None, return_log_marginal: bool = False,
                    candidates: Optional[torch.Tensor] = None, precision: str = "fp32", storage: str = "auto"):
        """
        ``log p(x_t = v | e, C)`` for every target, of every token or of the ``candidates`` (see
        :meth:`~pyjuice.constraints.ConstrainedCircuit.conditional`).

        :param precision: "fp32" (the default; fp32-level accuracy) or "tf32" for the matrix products; inside
            ``pyjuice.fast_inference(allow_bf16_params = True)`` the token probabilities are scored in bfloat16
        :param storage: "auto" (the default), "dense" or "compact": keep the per-sample values of every pair of
            automaton states, or under evidence only of each sample's live ones (far less memory when the evidence pins
            the automaton down, as a decoded prefix does; more where it joins few of them); "auto" picks the one whose
            buffers are smaller, per query
        """
        from .queries.conditional import conditional
        return conditional(self, data, missing_mask, targets, precision = precision,
                           return_log_marginal = return_log_marginal, candidates = candidates, storage = storage)

    def decoder(self, cc, **options):
        """A decoding session over ``cc`` whose queries reuse what the previous one computed where the evidence did
        not change it (:class:`~.queries.decoder.DFAExactDecoder`)."""
        from .queries.decoder import DFAExactDecoder
        return DFAExactDecoder(cc, **{**self.decoder_defaults, **options})


def _unsupported_reason(ns, kind: str) -> Optional[str]:
    """Why this backend cannot serve a node group (None if it can)."""
    if kind == "input":
        if len(ns.scope) > 1:
            return f"input nodes over {len(ns.scope)} variables (only single-variable input nodes are supported)"
        if distributions.lookup(ns.dist) is None:
            supported = ", ".join(t.__name__ for t in distributions.SUPPORTED)
            return f"input distribution {type(ns.dist).__name__} (supported: {supported})"
        return None
    if kind == "sum" and getattr(ns, "external_params", None) is not None:
        return "sum nodes with external parameters (gated layers are not supported yet)"
    return None


def _count(k: int, noun: str) -> str:
    return f"{k} {noun}{'' if k == 1 else 's'}"


def _format_runs(runs: Sequence[Tuple[int, int]], max_runs: int = 8) -> str:
    parts = [str(a) if a == b else f"{a}..{b}" for a, b in runs[:max_runs]]
    if len(runs) > max_runs:
        parts.append("...")
    return "{" + ", ".join(parts) + "}"


def _format_vars(vs: Sequence[int]) -> str:
    return _format_runs(_runs(set(vs)))


def _format_big(x: int) -> str:
    return f"{x:,}" if x < 10 ** 9 else f"{float(x):.1e}"
