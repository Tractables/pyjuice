from __future__ import annotations

import torch

from functools import partial
from typing import Callable, Optional, Dict

from pyjuice.nodes import CircuitNodes


def block_diagonal_edge_constructor(ns0, *args, num_node_blocks: int = 0, block_size: int = 0, **kwargs):
    assert num_node_blocks > 0 and block_size > 0

    if block_size == 1:
        return None

    ch_block_size = ns0.block_size

    ch_num_node_blocks = ns0.num_node_blocks
    for ns in args:
        assert isinstance(ns, CircuitNodes)
        ch_num_node_blocks += ns.num_node_blocks

    assert num_node_blocks == ch_num_node_blocks, "There must be the same number of sum blocks as the number of total child blocks."

    edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1)

    return edge_ids


def block_sparse_rnd_blk_edge_constructor(ns0, *args, num_node_blocks: int = 0, block_size: int = 0, num_chs_per_block: int = 0, **kwargs):

    assert num_node_blocks > 0 and block_size > 0
    assert num_chs_per_block > 0

    if block_size == 1:
        return None

    ch_block_size = ns0.block_size

    ch_num_node_blocks = ns0.num_node_blocks
    for ns in args:
        assert isinstance(ns, CircuitNodes)
        ch_num_node_blocks += ns.num_node_blocks

    total_edges_needed = num_node_blocks * num_chs_per_block

    if total_edges_needed >= ch_num_node_blocks:
        guaranteed = torch.randperm(ch_num_node_blocks)
        remaining_count = total_edges_needed - ch_num_node_blocks
        
        # Fill the rest with random indices
        if remaining_count > 0:
            extra = torch.randint(0, ch_num_node_blocks, (remaining_count,))
            chs_indices = torch.cat([guaranteed, extra])
        else:
            chs_indices = guaranteed
            
        # Shuffle again so the "guaranteed" ones aren't always the first neighbors
        chs_indices = chs_indices[torch.randperm(total_edges_needed)]
        
    else:
        # Fallback: If we have fewer slots than children, we can't select everyone.
        # We just select a random subset without replacement.
        chs_indices = torch.randperm(ch_num_node_blocks)[:total_edges_needed]

    par_indices = torch.arange(0, num_node_blocks)[:,None].repeat(1, num_chs_per_block).reshape(num_node_blocks * num_chs_per_block)

    edge_ids = torch.stack((par_indices, chs_indices), dim = 0)

    return edge_ids
