from __future__ import annotations

import torch
from copy import deepcopy as pydeepcopy
from typing import Optional, Dict

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from pyjuice.utils import BitSet


def deepcopy(root_ns: CircuitNodes, tie_params: bool = False, max_block_size: Optional[int] = None,
             var_mapping: Optional[Dict[int,int]] = None) -> CircuitNodes:
    """
    Create a deepcopy of the input PC.

    :param root_ns: the input PC
    :type root_ns: CircuitNodes

    :param tie_params: whether to tie the parameters between the original PC and the copied PC (if tied, their parameters will always be the same)
    :type tie_params: bool

    :param max_block_size: the maximum block size of the copied PC
    :type max_block_size: Optional[int]

    :param var_mapping: a mapping dictionary between the variables of the original PC and the copied PC
    :type var_mapping: Optional[Dict[int,int]]

    :returns: a copied PC
    """

    assert not (max_block_size is not None and tie_params), "Could not change block size when `tie_params=True`."
    if max_block_size is not None:
        assert max_block_size > 0 and (max_block_size & (max_block_size - 1)) == 0, f"`max_block_size` must be a power of 2, but got `max_block_size={max_block_size}`."

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
                if max_block_size is None:
                    edge_ids = ns.edge_ids.clone()
                    block_size = ns.block_size
                    params = ns.get_params()
                else:
                    old_ch_blk_size = ns.chs[0].block_size
                    old_blk_size = ns.block_size

                    new_ch_blk_size = new_chs[0].block_size
                    new_blk_size = min(old_blk_size, max_block_size)

                    blk_factor = old_blk_size // new_blk_size
                    ch_blk_factor = old_ch_blk_size // new_ch_blk_size

                    edge_ids = torch.stack(
                        (ns.edge_ids[0,:][:,None,None].repeat(1, blk_factor, ch_blk_factor) * blk_factor + torch.arange(0, blk_factor)[None,:,None],
                         ns.edge_ids[1,:][:,None,None].repeat(1, blk_factor, ch_blk_factor) * ch_blk_factor + torch.arange(0, ch_blk_factor)[None,None,:]),
                        dim = 0
                    ).flatten(1, 3)
                    block_size = new_blk_size

                    params = ns.get_params()
                    if params is not None:
                        num_edges = params.size(0)
                        params = params.reshape(num_edges, blk_factor, new_blk_size, ch_blk_factor, new_ch_blk_size).permute(0, 1, 3, 2, 4).flatten(0, 2)

                new_ns = SumNodes(
                    ns.num_nodes // block_size,
                    new_chs,
                    edge_ids,
                    block_size = block_size
                )
                if params is not None:
                    new_ns.set_params(params.clone(), normalize = False)
            else:
                new_ns = ns.duplicate(*new_chs, tie_params = True)
            
        elif ns.is_prod():
            if max_block_size is None:
                edge_ids = ns.edge_ids.clone()
                block_size = ns.block_size
            else:
                old_ch_blk_size = ns.chs[0].block_size
                old_blk_size = ns.block_size

                new_ch_blk_size = new_chs[0].block_size
                new_blk_size = min(old_blk_size, max_block_size)

                if old_blk_size == new_blk_size and old_ch_blk_size == new_ch_blk_size:
                    edge_ids = ns.edge_ids.clone()
                    block_size = ns.block_size
                else:
                    blk_factor = old_blk_size // new_blk_size
                    ch_blk_factor = old_ch_blk_size // new_ch_blk_size

                    if blk_factor == ch_blk_factor:
                        edge_ids = ns.edge_ids.clone()
                        edge_ids = edge_ids[:,None,:].repeat(1, blk_factor, 1) * blk_factor + torch.arange(0, blk_factor)[None,:,None]
                        edge_ids = edge_ids.flatten(0, 1)
                        block_size = new_blk_size
                    else:
                        raise NotImplementedError()

            new_ns = ProdNodes(
                ns.num_nodes // block_size,
                new_chs,
                edge_ids,
                block_size = block_size
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

            if max_block_size is None:
                block_size = ns.block_size
            else:
                block_size = min(ns.block_size, max_block_size)

            if not tie_params:
                new_ns = InputNodes(
                    num_node_blocks = ns.num_nodes // block_size,
                    scope = pydeepcopy(scope),
                    dist = pydeepcopy(ns.dist),
                    block_size = block_size
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