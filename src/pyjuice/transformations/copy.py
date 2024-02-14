from __future__ import annotations

from copy import deepcopy as pydeepcopy
from typing import Optional, Dict

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from pyjuice.utils import BitSet


def deepcopy(root_ns: CircuitNodes, tie_params: bool = False, 
             var_mapping: Optional[Dict[int,int]] = None) -> CircuitNodes:
    """
    Create a deepcopy of the input PC.

    :param root_ns: the input PC
    :type root_ns: CircuitNodes

    :param tie_params: whether to tie the parameters between the original PC and the copied PC (if tied, their parameters will always be the same)
    :type tie_params: bool

    :param var_mapping: a mapping dictionary between the variables of the original PC and the copied PC
    :type var_mapping: Optional[Dict[int,int]]

    :returns: a copied PC
    """

    old2new = dict()
    tied_ns_pairs = []

    def dfs(ns: CircuitNodes):
        if ns in old2new:
            return

        # Recursively traverse children
        if ns.is_sum() or ns.is_prod():
            for cs in ns.chs:
                dfs(cs)

        new_chs = [old2new[cs] for cs in ns.chs]

        if not tie_params and ns.is_tied():
            tied_ns_pairs.append((ns, ns.get_source_ns()))

        if ns.is_sum():
            if not tie_params:
                new_ns = SumNodes(
                    ns.num_node_blocks,
                    new_chs,
                    ns.edge_ids.clone(),
                    block_size = ns.block_size
                )
                params = ns.get_params()
                if params is not None:
                    new_ns.set_params(params.clone(), normalize = False)
            else:
                new_ns = ns.duplicate(*new_chs, tie_params = True)
            
        elif ns.is_prod():
            new_ns = ProdNodes(
                ns.num_node_blocks,
                new_chs,
                ns.edge_ids.clone(),
                block_size = ns.block_size
            )
            
        else:
            assert ns.is_input()

            # Map variable scope
            if var_mapping is not None:
                ns_scope = ns.scope
                scope = BitSet()
                for v in ns_scope:
                    assert v in var_mapping, f"Variable {v} not found in `var_mapping`."
                    scope.add(var_mapping[v])
            else:
                scope = pydeepcopy(ns.scope)

            if not tie_params:
                new_ns = InputNodes(
                    num_node_blocks = ns.num_node_blocks,
                    scope = pydeepcopy(scope),
                    dist = pydeepcopy(ns.dist),
                    block_size = ns.block_size
                )
                params = ns.get_params()
                if params is not None:
                    new_ns.set_params(params.clone(), normalize = False)
            else:
                new_ns = ns.duplicate(scope = scope, tie_params = True)

        old2new[ns] = new_ns

    dfs(root_ns)

    for ns, source_ns in tied_ns_pairs:
        new_ns = old2new[ns]
        new_source_ns = old2new[source_ns]

        new_ns._source_node = new_source_ns

    return old2new[root_ns]