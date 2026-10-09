"""
The result of :func:`pyjuice.constraints.compile`: a PC under a constraint, ready for queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from .language.base import Constraint
from .structure import SHAPES, PCStructure, analyze_structure
from .backends.lifted.plan import BoundaryLayout


class ConstrainedCircuit:
    """
    A PC under a constraint: the PC, the constraint, and everything compiled from the pair. Queries
    under the constraint are its methods.

    Everything it holds besides the PC depends only on the constraint and the PC's STRUCTURE, never on
    parameter values or evidence, so it stays valid while the PC's parameters change (e.g. CoDD's
    external parameters at every step). :meth:`with_pc` binds the same plan to another PC with the same
    structure.

    Created by :func:`pyjuice.constraints.compile`; not meant to be constructed directly.

    :ivar pc: the PC queries run on
    :ivar constraint: the constraint
    :ivar structure: the PC's :class:`~pyjuice.constraints.structure.PCStructure`
    :ivar automaton: the constraint's automaton
    :ivar layout: the :class:`~pyjuice.constraints.backends.lifted.plan.BoundaryLayout` of the automaton
        over the PC's variables
    :ivar compile_time_s: wall-clock seconds :func:`~pyjuice.constraints.compile` (or :meth:`with_pc`) took
    """

    #: the backend queries run on
    backend = "lifted"
    #: whether queries are exact (up to floating point)
    exact = True

    def __init__(self, pc, constraint: Constraint, structure: PCStructure, automaton,
                 layout: BoundaryLayout, compile_time_s: float):
        self.pc = pc
        self.constraint = constraint
        self.structure = structure
        self.automaton = automaton
        self.layout = layout
        self.compile_time_s = compile_time_s

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
        Estimated node-value memory per sample under the lifted plan, in fp32 and padded to
        :attr:`max_width` columns ``W``: a sum or product node holds 1 value if its scope is the whole
        sequence, ``W`` for a prefix or suffix and ``W^2`` for an interval; an input node holds one mass
        per token class.
        """
        W, C = self.max_width, self.num_classes
        per_node = {"whole": 1, "prefix": W, "suffix": W, "interval": W * W}
        total = 0
        for info in self.structure.nodes:
            total += info.ns.num_nodes * (C if info.kind == "input" else per_node[info.shape])
        return 4 * total

    def info(self) -> Dict[str, Any]:
        """A summary of the constrained circuit."""
        return dict(backend = self.backend, exact = self.exact, n = self.n, satisfiable = self.satisfiable,
                    num_states = self.num_states, num_classes = self.num_classes, max_width = self.max_width,
                    width_per_boundary = self.width_per_boundary.tolist(), shape_counts = self.shape_counts,
                    bytes_per_sample = self.bytes_per_sample, compile_time_s = self.compile_time_s,
                    constraint = self.constraint.info())

    def __repr__(self) -> str:
        return (f"ConstrainedCircuit(backend={self.backend}, n={self.n}, num_states={self.num_states}, "
                f"max_width={self.max_width}, satisfiable={self.satisfiable})")

    # ---------------------------------------------------------------------------------------------
    # Rebinding
    # ---------------------------------------------------------------------------------------------

    def with_pc(self, pc) -> "ConstrainedCircuit":
        """
        The same plan bound to another PC with the same structure (e.g. a drafter and a verifier, or a
        copy of the PC on another device). Only the PCs' structural signatures are compared; nothing is
        rebuilt.

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
        return ConstrainedCircuit(pc, self.constraint, structure, self.automaton, self.layout,
                                  compile_time_s = time.perf_counter() - t0)

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, *args, **kwargs):
        """Probability of the constraint (and evidence). Not implemented yet."""
        raise NotImplementedError("`marginal` under a constraint arrives with the reference backend.")

    def conditional(self, *args, **kwargs):
        """Per-variable distributions given the constraint (and evidence). Not implemented yet."""
        raise NotImplementedError("`conditional` under a constraint arrives with the reference backend.")

    def sample(self, *args, **kwargs):
        """Samples from the PC conditioned on the constraint (and evidence). Not implemented yet."""
        raise NotImplementedError("`sample` under a constraint arrives with the reference backend.")

    def decoder(self, *args, **kwargs):
        """Incremental (token-by-token) constrained decoding. Not implemented yet."""
        raise NotImplementedError("`decoder` under a constraint arrives with the incremental backend.")
