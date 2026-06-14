from __future__ import annotations

import torch

from functools import partial
from typing import Callable, Optional, Dict

from pyjuice.nodes import CircuitNodes


def block_diagonal_edge_constructor(ns0, *args, num_node_blocks: int = 0, block_size: int = 0, **kwargs):
    """
    An edge constructor that connects sum and child node blocks in a one-to-one (block-diagonal)
    pattern: the i-th sum node block is connected only to the i-th child node block. It is meant to be
    passed as the `edge_ids` argument of :func:`~pyjuice.summate`, and requires the number of sum node
    blocks to equal the total number of child node blocks.

    :param ns0: the first child node group; additional child node groups are passed as positional arguments
    :type ns0: CircuitNodes

    :returns: an edge-id tensor of size [2, num_node_blocks], or `None` when `block_size == 1`
    """
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
    """
    An edge constructor that connects every sum node block to `num_chs_per_block` randomly chosen child
    node blocks, yielding a block-sparse (rather than fully-connected) sum layer. It is meant to be
    passed as the `edge_ids` argument of :func:`~pyjuice.summate`, and is useful for building large,
    sparsely-connected PCs. When enough edges are requested, every child block is guaranteed to be
    connected at least once.

    :param ns0: the first child node group; additional child node groups are passed as positional arguments
    :type ns0: CircuitNodes

    :param num_chs_per_block: number of child node blocks each sum node block connects to
    :type num_chs_per_block: int

    :returns: an edge-id tensor describing the sampled sum-to-child block connections, or `None` when `block_size == 1`
    """

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
