"""
Compiling a constraint against a PC: check that a backend can serve the pair, and build its plan.

Compilation depends only on the constraint and the PC's STRUCTURE, never on parameter values or
evidence, so a PC whose parameters change every step is compiled once.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from pyjuice.model import TensorCircuit
from pyjuice.nodes.distributions import Categorical

from .language.base import Constraint
from .structure import PCStructure, _runs, analyze_structure
from .backends.lifted.plan import build_layout
from .compiled import CompiledConstraint


#: Backends :func:`compile` knows.
BACKENDS = ("lifted",)


class ConstraintCompileError(ValueError):
    """Raised when a constraint cannot be compiled against a PC. The message lists every reason."""


def compile(constraint: Constraint, pc: TensorCircuit, backend: Optional[str] = None) -> CompiledConstraint:
    """
    Compile ``constraint`` against ``pc`` for constrained queries.

    The constraint reads the PC's variables ``0, ..., n-1`` in index order. If the pair cannot be
    served, a :class:`ConstraintCompileError` lists every reason at once.

    :param constraint: the constraint
    :type constraint: Constraint

    :param pc: a compiled PC (from :func:`pyjuice.compile`)
    :type pc: TensorCircuit

    :param backend: the backend to use; ``None`` picks one (today always ``"lifted"``: exact lifting of
        the constraint's automaton, for PCs whose node scopes are contiguous in variable order)
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
    reasons = _lifted_refusals(constraint, structure)
    if reasons:
        raise ConstraintCompileError("Cannot compile the constraint against this PC (backend 'lifted'):\n"
                                     + "\n".join(f"  - {r}" for r in reasons))

    automaton = constraint.automaton()
    layout = build_layout(automaton, structure.num_vars)
    return CompiledConstraint(pc, constraint, structure, automaton, layout,
                              compile_time_s = time.perf_counter() - t0)


def _check_pc(pc):
    if not isinstance(pc, TensorCircuit):
        raise TypeError(f"Expected a compiled PC (TensorCircuit), got {type(pc).__name__}. "
                        f"Compile the circuit with `pyjuice.compile` first.")


def _lifted_refusals(constraint: Constraint, structure: PCStructure) -> List[str]:
    """Every reason the lifted backend cannot serve the pair (empty if it can)."""
    reasons = []

    caps = constraint.capabilities()
    automaton = None
    if "automaton" in caps:
        automaton = constraint.automaton()
    else:
        reasons.append(f"the constraint ({type(constraint).__name__}) has no `automaton` capability, which "
                       f"exact lifting needs; it supports: {', '.join(sorted(caps))}")

    by_reason = defaultdict(list)
    for ns, why in structure.unsupported:
        by_reason[why].append(ns)
    for why, group in by_reason.items():
        reasons.append(f"{_count(len(group), 'node group')} {'is' if len(group) == 1 else 'are'} not supported: "
                       f"{why}; e.g. over variables {_format_vars(group[0].scope.to_list())}")

    vars_by_cats = defaultdict(list)
    for info in structure.nodes:
        if info.kind == "input" and isinstance(info.ns.dist, Categorical):
            vars_by_cats[info.ns.dist.num_cats].extend(info.ns.scope.to_list())
    for num_cats, vs in sorted(vars_by_cats.items()):
        if num_cats != constraint.vocab_size:
            reasons.append(f"vocabulary mismatch: the constraint reads tokens 0..{constraint.vocab_size - 1} "
                           f"(vocab_size {constraint.vocab_size}), but the Categorical input nodes over variables "
                           f"{_format_vars(vs)} have num_cats {num_cats}")

    fragmented = [info for info in structure.nodes if info.shape == "fragmented"]
    if fragmented:
        worst = max(fragmented, key = lambda info: info.num_runs)       # the first one with the most runs
        r = worst.num_runs
        msg = (f"the node group over variables {_format_runs(worst.scope_runs)} has {r} separate runs in "
               f"variable order ({_count(len(fragmented), 'node group')} {'is' if len(fragmented) == 1 else 'are'} "
               f"fragmented, with up to {r} runs). A constraint reads the PC's variables in index order "
               f"0..{structure.num_vars - 1}, and this version only supports PCs whose node scopes are "
               f"contiguous in that order (e.g. HMMs, 1-D PD over a sequence). A node with r runs would need "
               f"a state of size up to K^(2r)")
        if automaton is not None:
            K = automaton.num_states
            msg += f"; with this constraint's K = {K} states and r = {r} that is {_format_big(K ** (2 * r))} per node"
        reasons.append(msg)

    return reasons


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
