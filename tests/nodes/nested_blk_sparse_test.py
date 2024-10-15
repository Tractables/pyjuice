import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet

import pytest


@pytest.mark.slow
def nested_blk_sparse_test():

    device = torch.device("cuda:0")

    block_size = 16
    num_node_blocks = 16

    with juice.set_block_size(block_size):
        ni0 = juice.inputs(0, num_node_blocks, dist = dists.Categorical(num_cats = 10))
        ni1 = juice.inputs(1, num_node_blocks, dist = dists.Categorical(num_cats = 10))

        np = juice.multiply(ni0, ni1) # `num_node_blocks` blocks of nodes each of size `block_size`

        # Create a block-diagonal sum layer
        edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1) # Tensor([[0, 1, ...], [0, 1, ...]])
        ns0 = juice.summate(np, edge_ids = edge_ids)

        # Create a (pseudo) product layer that permutes the nodes
        permuted_edges = torch.arange(0, block_size * num_node_blocks).reshape(
            num_node_blocks, block_size
        ).permute(1, 0).reshape(
            block_size * num_node_blocks
        )[:,None]
        np0 = juice.multiply(ns0, edge_ids = permuted_edges, sparse_edges = True)

        # Create a second block-diagonal sum layer
        edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1) # Tensor([[0, 1, ...], [0, 1, ...]])
        ns1 = juice.summate(np0, edge_ids = edge_ids)

        # Create a (pseudo) passthrough product layer
        passthrough_edges = torch.arange(0, num_node_blocks)[:,None]
        np1 = juice.multiply(ns1, edge_ids = passthrough_edges, sparse_edges = False) # `sparse_edges = False` is default

        # Create a sum layer with one root node
        ns = juice.summate(np1, num_node_blocks = 1, block_size = 1) # Setting `block_size` here overrides the value suggested by `juice.set_block_size`

    pc = juice.compile(ns)
    pc.print_statistics()

    pc.to(device)

    data = torch.randint(0, 10, (64, 2)).to(device)

    lls = pc(data, propagation_alg = "LL")
    pc.backward(data, flows_memory = 1.0, allow_modify_flows = False,
                propagation_alg = "LL", logspace_flows = True)


if __name__ == "__main__":
    nested_blk_sparse_test()