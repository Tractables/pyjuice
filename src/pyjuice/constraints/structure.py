"""
PC-side structural analysis for constrained inference, independent of any constraint.

A constraint reads the PC's variables 0, ..., n-1 in index order. How expensive it is to combine a
constraint with a PC exactly depends on how each node's scope sits in that order:

* a scope that is one contiguous run ``[a, b]`` is summarised by how it moves the constraint's state
  from boundary ``a`` (before reading ``x_a``) to boundary ``b + 1``;
* a run that ends at ``n - 1`` (a SUFFIX) only needs its entry state, since its exit state must be
  accepting; one that starts at ``0`` (a PREFIX) only needs its exit state; the root (``[0, n-1]``)
  is a single value;
* a scope with several runs (FRAGMENTED) needs an entry and exit state for every run.

:func:`analyze_structure` computes this for every node group of a PC, together with a structural
signature (a hash of the node graph that ignores parameter values) and the features the first version of
the constrained backends does not support. It only reports; refusing is the compiler's job.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from pyjuice.nodes import CircuitNodes
from pyjuice.nodes.distributions import Categorical

from .language.base import _hash_into


#: Shapes of a node's scope in index order (see the module docstring).
SHAPES = ("whole", "suffix", "prefix", "interval", "fragmented")


@dataclass(frozen = True)
class NodeInfo:
    """Structural facts about one node group (a :class:`~pyjuice.nodes.CircuitNodes`)."""

    ns: CircuitNodes
    kind: str                                   # "input", "sum" or "prod"
    scope_runs: Tuple[Tuple[int, int], ...]     # maximal contiguous runs (a, b), inclusive, sorted
    shape: str                                  # one of SHAPES

    @property
    def num_runs(self) -> int:
        return len(self.scope_runs)


@dataclass(frozen = True)
class PCStructure:
    """
    The result of :func:`analyze_structure`.

    :ivar num_vars: number of PC variables ``n``
    :ivar signature: hex digest of the node graph's structure (node kinds, block sizes, scopes, edges,
        distribution signatures and metadata) -- equal for PCs that differ only in parameter values
    :ivar nodes: one :class:`NodeInfo` per node group, children before parents
    :ivar right_linear: every sum and product node's scope is a suffix (or the whole sequence), as in an
        HMM built position by position
    :ivar contiguous: every node's scope is a single contiguous run
    :ivar max_runs: the largest number of runs of any scope
    :ivar unsupported: ``(node group, reason)`` for everything the first version of the constrained
        backends does not handle
    """

    num_vars: int
    signature: str
    nodes: Tuple[NodeInfo, ...]
    right_linear: bool
    contiguous: bool
    max_runs: int
    unsupported: Tuple[Tuple[CircuitNodes, str], ...]

    def node(self, ns: CircuitNodes) -> NodeInfo:
        """The :class:`NodeInfo` of a node group."""
        for info in self.nodes:
            if info.ns is ns:
                return info
        raise KeyError("Node group is not part of this PC.")


_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def analyze_structure(pc) -> PCStructure:
    """
    Analyse a PC's structure for constrained inference.

    The result is cached per compiled PC (a compiled circuit's structure cannot change); a
    :class:`~pyjuice.nodes.CircuitNodes` root is analysed afresh every call.

    :param pc: a compiled :class:`~pyjuice.TensorCircuit`, or the root of an uncompiled circuit
    :type pc: Union[TensorCircuit, CircuitNodes]
    """
    if isinstance(pc, CircuitNodes):
        return _analyze(pc)
    cached = _CACHE.get(pc)
    if cached is None:
        cached = _analyze(pc.root_ns)
        _CACHE[pc] = cached
    return cached


def _analyze(root_ns: CircuitNodes) -> PCStructure:
    order = list(root_ns)                                       # children before parents
    num_vars = max(max(ns.scope.to_list()) for ns in order) + 1 if order else 0

    nodes, unsupported = [], []
    for ns in order:
        kind = "input" if ns.is_input() else ("sum" if ns.is_sum() else "prod")
        runs = _runs(ns.scope.to_list())
        nodes.append(NodeInfo(ns = ns, kind = kind, scope_runs = runs, shape = _shape(runs, num_vars)))
        reason = _unsupported_reason(ns, kind)
        if reason is not None:
            unsupported.append((ns, reason))

    inner = [info for info in nodes if info.kind != "input"]
    return PCStructure(
        num_vars = num_vars,
        signature = _signature(order),
        nodes = tuple(nodes),
        right_linear = all(info.shape in ("suffix", "whole") for info in inner),
        contiguous = all(info.num_runs == 1 for info in nodes),
        max_runs = max((info.num_runs for info in nodes), default = 0),
        unsupported = tuple(unsupported),
    )


def _runs(vars_):
    """Maximal contiguous runs of a set of variable ids, as sorted inclusive (a, b) pairs."""
    vs = sorted(vars_)
    runs, start, prev = [], vs[0], vs[0]
    for v in vs[1:]:
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    return tuple(runs)


def _shape(runs, num_vars) -> str:
    if len(runs) > 1:
        return "fragmented"
    a, b = runs[0]
    if a == 0 and b == num_vars - 1:
        return "whole"
    if b == num_vars - 1:
        return "suffix"
    if a == 0:
        return "prefix"
    return "interval"


def _unsupported_reason(ns: CircuitNodes, kind: str) -> Optional[str]:
    if kind == "input":
        if len(ns.scope) > 1:
            return f"input nodes over {len(ns.scope)} variables (only single-variable leaves are supported)"
        if not isinstance(ns.dist, Categorical):
            return f"input distribution {type(ns.dist).__name__} (only Categorical is supported)"
        return None
    if kind == "sum" and getattr(ns, "external_params", None) is not None:
        return "sum nodes with external parameters (gated layers are not supported yet)"
    return None


def _signature(order) -> str:
    """Hash of the node graph's structure. Parameter values and parameter tying are excluded: neither
    changes which computations the constrained backends perform."""
    ids = {ns: i for i, ns in enumerate(order)}
    h = hashlib.sha256()
    for ns in order:
        kind = "input" if ns.is_input() else ("sum" if ns.is_sum() else "prod")
        item = [kind, ns.num_node_blocks, ns.block_size, sorted(ns.scope.to_list())]
        if kind == "input":
            item += [ns.dist.get_signature(), ns.dist.get_metadata()]
        else:
            item += [[ids[cs] for cs in ns.chs], ns.edge_ids]
            ext = getattr(ns, "external_params", None)
            item += [None if ext is None else ext.get_signature()]
        _hash_into(h, item)
    return h.hexdigest()
