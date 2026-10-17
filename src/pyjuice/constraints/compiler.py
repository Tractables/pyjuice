"""
Compiling a constraint against a PC: find a backend that can serve the pair, and build it.

Compilation depends only on the constraint and the PC's STRUCTURE, never on parameter values or
evidence, so a PC whose parameters change every step is compiled once.
"""

from __future__ import annotations

import time
from typing import Optional

from pyjuice.model import TensorCircuit

from .backends import BACKENDS
from .constrained_circuit import ConstrainedCircuit
from .distributions import lookup
from .errors import ConstraintCompileError
from .language.base import Constraint
from .structure import analyze_structure


def compile(constraint: Constraint, pc: TensorCircuit, backend: Optional[str] = None) -> ConstrainedCircuit:
    """
    Compile ``constraint`` against ``pc`` for constrained queries.

    The constraint reads the PC's variables ``0, ..., n-1`` in index order. When the PC's inputs range over more
    values than the constraint's vocabulary has ids -- a model's embedding rows beyond its tokenizer's -- a
    constraint that can widen is compiled over the PC's vocabulary (a text-level constraint's new ids have no text;
    see ``Constraint._widened``), and is the circuit's ``constraint``. If no backend can serve the pair, a
    :class:`ConstraintCompileError` lists every reason of every backend tried, at once.

    :param constraint: the constraint
    :type constraint: Constraint

    :param pc: a compiled PC (from :func:`pyjuice.compile`)
    :type pc: TensorCircuit

    :param backend: the backend to use, by name (see :data:`~pyjuice.constraints.backends.BACKENDS`); ``None``
        picks the first that can serve the pair (today there is one, ``"dfa_exact"``: exact lifting of the
        constraint's automaton, for PCs whose node scopes are contiguous in variable order)
    :type backend: Optional[str]
    """
    t0 = time.perf_counter()
    if not isinstance(constraint, Constraint):
        raise TypeError(f"Expected a Constraint, got {type(constraint).__name__}.")
    _check_pc(pc)
    if backend is not None and backend not in BACKENDS:
        raise ConstraintCompileError(f"Backend {backend!r} is not available yet; known backends: "
                                     f"{', '.join(repr(b) for b in BACKENDS)}.")

    structure = analyze_structure(pc)
    constraint = _widened_to_pc(constraint, structure)
    refused = {}
    for cls in ([BACKENDS[backend]] if backend is not None else BACKENDS.values()):
        reasons = cls.refusals(constraint, structure)
        if not reasons:
            return ConstrainedCircuit(pc, constraint, structure, cls.build(constraint, pc, structure),
                                      compile_time_s = time.perf_counter() - t0)
        refused[cls.name] = reasons
    raise ConstraintCompileError("\n".join(f"Cannot compile the constraint against this PC (backend {name!r}):\n"
                                           + "\n".join(f"  - {r}" for r in reasons)
                                           for name, reasons in refused.items()))


def _widened_to_pc(constraint: Constraint, structure) -> Constraint:
    """``constraint`` over the PC's vocabulary when every input the backends read ranges over the same number of
    values, more than the constraint's vocabulary, and the constraint widens; else ``constraint`` itself (a mismatch
    left is the backends' to report)."""
    values = set()
    for info in structure.nodes:
        module = lookup(info.ns.dist) if info.kind == "input" else None
        if module is not None:
            values.add(module.num_values(info.ns.dist))
    if len(values) == 1:
        (num_values,) = values
        if num_values > constraint.vocab_size:
            wide = constraint._widened(num_values)
            if wide is not None:
                return wide
    return constraint


def _check_pc(pc):
    if not isinstance(pc, TensorCircuit):
        raise TypeError(f"Expected a compiled PC (TensorCircuit), got {type(pc).__name__}. "
                        f"Compile the circuit with `pyjuice.compile` first.")
