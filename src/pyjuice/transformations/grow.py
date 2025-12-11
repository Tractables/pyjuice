from __future__ import annotations

import torch

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from pyjuice.nodes.methods import foldup_aggregate
from pyjuice.nodes import multiply, summate, inputs


def grow(root_ns: CircuitNodes, num_duplicates: int, perturbation: float = 0.0):
    """
    Grow a PC by creating duplicates of all nodes (except for the root node). See https://arxiv.org/pdf/2211.12551.

    :param root_ns: the input PC
    :type root_ns: CircuitNodes

    :param num_duplicates: the number of duplicates
    :type num_duplicates: int

    :param perturbation: parameter perturbation
    :type perturbation: float

    :returns: a duplicated PC
    """
    assert num_duplicates > 0, "`num_duplicates` should be greater than 0."

    old2new = dict()

    def aggr_fn(ns, ch_outputs):
        if ns.is_input():
            new_ns = inputs(
                var = ns.scope.to_list(),
                num_node_blocks = ns.num_node_blocks * (1 + num_duplicates),
                dist = ns.dist,
                params = ns.get_params().repeat(1 + num_duplicates) if ns.has_params() else None,
                block_size = ns.block_size
            )
        elif ns.is_prod():
            edge_ids = ns.edge_ids.clone()
            num_edges, num_chs = edge_ids.size()
            assert num_edges == ns.num_node_blocks
            new_edge_ids = torch.zeros([num_edges * (1 + num_duplicates), num_chs], dtype = torch.long)
            for i in range(num_duplicates + 1):
                offset = ns.num_node_blocks * i
                new_edge_ids[offset:offset+num_edges,:] = edge_ids + offset

            new_ns = multiply(
                *ch_outputs,
                edge_ids = new_edge_ids
            )
        else:
            assert ns.is_sum()

            edge_ids = ns.edge_ids.clone()
            num_edges = edge_ids.size(1)
            num_node_blocks = ns.num_node_blocks
            num_ch_node_blocks = sum(ms.num_node_blocks for ms in ns.chs)

            new_edge_ids = torch.zeros([2, num_edges * ((num_duplicates + 1) ** 2)], dtype = torch.long)
            idx = 0
            for i in range(num_duplicates + 1):
                offset_i = num_node_blocks * i
                for j in range(num_duplicates + 1):
                    offset_j = num_node_blocks * j
                    
                    new_edge_ids[0,idx * num_edges:(idx+1) * num_edges] = edge_ids[0,:] + offset_i
                    new_edge_ids[1,idx * num_edges:(idx+1) * num_edges] = edge_ids[1,:] + offset_j

                    idx += 1

            if ns.has_params():
                params = ns.get_params()[None,:,:,:].repeat((num_duplicates + 1) ** 2, 1, 1, 1)
                idx = 0
                for i in range(num_duplicates + 1):
                    for j in range(num_duplicates + 1):
                        if i > 0 or j > 0:
                            params[idx,:,:,:] = (params[idx,:,:,:].log() + \
                                perturbation * torch.rand_like(params[idx,:,:,:])).exp()

                        idx += 1
            else:
                params = None

            new_ns = summate(
                *ch_outputs, 
                edge_ids = edge_ids, 
                params = params, 
                num_node_blocks = num_node_blocks * (1 + num_duplicates),
                block_size = ns.block_size
            )

            if ns.is_tied():
                new_ns.set_source_ns(old2new[ns.get_source_ns()])

            old2new[ns] = new_ns

        return new_ns

    return foldup_aggregate(aggr_fn, root_ns)
