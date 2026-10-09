"""
The result of :func:`pyjuice.constraints.compile`: a PC under a constraint, ready for queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from .language.base import Constraint
from .structure import SHAPES, PCStructure, analyze_structure
from .backends.lifted.plan import BoundaryLayout, build_pc_tables


class ConstrainedCircuit:
    """
    A PC under a constraint: the PC, the constraint, and everything compiled from the pair. Queries
    under the constraint are its methods.

    Everything it holds besides the PC depends only on the constraint and the PC's STRUCTURE, never on
    parameter values or evidence, so it stays valid while the PC's parameters change (e.g. CoDD's
    external parameters at every step). The PC itself is fixed: :attr:`pc` cannot be reassigned, and
    once the PC has been moved to another device every query raises -- compile again, or use
    :meth:`with_pc` to put the same constraint on another PC with the same structure.

    Created by :func:`pyjuice.constraints.compile`; not meant to be constructed directly.

    :ivar pc: the PC queries run on
    :ivar constraint: the constraint
    :ivar structure: the PC's :class:`~pyjuice.constraints.structure.PCStructure`
    :ivar automaton: the constraint's automaton
    :ivar layout: the :class:`~pyjuice.constraints.backends.lifted.plan.BoundaryLayout` of the automaton
        over the PC's variables
    :ivar columns_per_sample: S, the columns every sample takes in the PC's buffers during a lifted pass
        (see :func:`~pyjuice.constraints.backends.lifted.plan.build_pc_tables`)
    :ivar product_rows: per product layer and pattern, the rows and boundaries the lifted products read
    :ivar input_range: the ``node_mars`` rows of all input nodes
    :ivar root_rows: the ``node_mars`` rows of the root nodes
    :ivar compile_time_s: wall-clock seconds :func:`~pyjuice.constraints.compile` (or :meth:`with_pc`) took
    """

    #: the backend queries run on
    backend = "lifted"
    #: whether queries are exact (up to floating point)
    exact = True

    def __init__(self, pc, constraint: Constraint, structure: PCStructure, automaton,
                 layout: BoundaryLayout, tables: Dict[str, Any], compile_time_s: float):
        self._pc = pc
        self._device = pc.params.device             # where the tables live
        self.constraint = constraint
        self.structure = structure
        self.automaton = automaton
        self.layout = layout
        self.columns_per_sample = tables["columns_per_sample"]
        self.product_rows = tables["product_rows"]
        self.input_range = tables["input_range"]
        self.root_rows = tables["root_rows"]
        self.compile_time_s = compile_time_s

    @property
    def pc(self):
        """The PC queries run on (fixed; see the class docstring)."""
        return self._pc

    def _check_pc_unchanged(self):
        """Refuse a query on a PC that moved since compiling (one attribute comparison per query)."""
        if self._pc.params.device != self._device:
            raise RuntimeError(f"The PC moved from {self._device} to {self._pc.params.device} after the constraint "
                               f"was compiled against it. Compile the constraint again, or use `with_pc`.")

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
        Memory a lifted pass needs per sample, in fp32: the PC's ``node_mars`` and ``element_mars`` at
        :attr:`columns_per_sample` columns, plus one mass per input node and token class.
        """
        num_inputs = self.input_range[1] - self.input_range[0]
        buffers = self.columns_per_sample * (self.pc.num_nodes + self.pc.num_elements)
        return 4 * (buffers + num_inputs * self.num_classes)

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
                                  compile_time_s = time.perf_counter() - t0)

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, *args, **kwargs):
        """Probability of the constraint (and evidence). Not implemented yet."""
        self._check_pc_unchanged()
        raise NotImplementedError("`marginal` under a constraint is not implemented yet.")

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
