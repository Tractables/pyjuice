from __future__ import annotations

from functools import reduce
from typing import Sequence, Union

from pyjuice.nodes import CircuitNodes
from pyjuice.utils import BitSet


def get_subsumed_scopes(root_ns, scopes: Union[Sequence[BitSet],Sequence[int],BitSet], type: str = "any"):
    """
    Get all scopes in the PC `root_ns` whose scope subsumes `scopes`. `type` could be "any" or "all".
    """
    if not isinstance(root_ns, CircuitNodes):
        root_ns = root_ns.root_nodes

    if isinstance(scopes, BitSet):
        scopes = [scopes]

    if isinstance(scopes[0], int):
        scopes = [BitSet.from_array([var]) for var in scopes]

    target_scopes = set()

    if type == "any":
        for ns in root_ns:
            if ns.scope in target_scopes:
                continue

            if any([ns.scope.contains_all(ref_scope) for ref_scope in scopes]):
                target_scopes.add(ns.scope)

    elif type == "all":
        ref_scope = reduce(lambda x, y: x | y, scopes)
        for ns in root_ns:
            if ns.scope in target_scopes:
                continue

            if ns.scope.contains_any(ref_scope):
                target_scopes.add(ns.scope)

    else:
        raise ValueError(f"Illegal type {type}.")

    return list(target_scopes)
