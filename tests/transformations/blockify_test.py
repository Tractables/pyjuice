import pyjuice as juice
import torch
import numpy as np

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs, set_block_size
from pyjuice.transformations import deepcopy

import pytest


def test_block():
    
    with set_block_size(block_size = 2):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
        n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

        m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

        m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long), block_size = 1)

    n.init_parameters()

    for use_cuda in [True, False]:
        new_n = juice.blockify(n, use_cuda = use_cuda)
        new_m = new_n.chs[0]

        new_n1 = new_m.chs[0]
        new_n2 = new_m.chs[1]

        new_m1 = new_n1.chs[0]
        new_m2 = new_n2.chs[0]

        new_ni0 = new_m1.chs[0]
        new_ni1 = new_m1.chs[1]
        new_ni2 = new_m2.chs[0]
        new_ni3 = new_m2.chs[1]

        assert new_ni0.block_size == 4 and new_ni0.num_node_blocks == 1
        assert new_ni1.block_size == 4 and new_ni1.num_node_blocks == 1
        assert new_ni2.block_size == 4 and new_ni2.num_node_blocks == 1
        assert new_ni3.block_size == 4 and new_ni3.num_node_blocks == 1

        assert new_m1.num_node_blocks == 2 and new_m1.block_size == 4
        assert new_m1.is_sparse()
        assert torch.all(new_m1.edge_ids == torch.tensor([[0, 0], [1, 1], [0, 2], [1, 3], [2, 0], [3, 1], [2, 2], [3, 3]]))

        assert new_m2.num_node_blocks == 1 and new_m2.block_size == 4
        assert new_m2.is_block_sparse()
        assert torch.all(new_m2.edge_ids == torch.tensor([[0, 0]]))

        assert new_n1.num_node_blocks == 1 and new_n1.block_size == 4
        assert torch.all(new_n1.edge_ids == torch.tensor([[0, 0], [0, 1]]))
        assert torch.all(new_n1._params[0][0:2,0:2] == n1._params[0])
        assert torch.all(new_n1._params[0][0:2,2:4] == n1._params[1])
        assert torch.all(new_n1._params[0][2:4,0:2] == n1._params[4])
        assert torch.all(new_n1._params[0][2:4,2:4] == n1._params[5])
        assert torch.all(new_n1._params[1][0:2,0:2] == n1._params[2])
        assert torch.all(new_n1._params[1][0:2,2:4] == n1._params[3])
        assert torch.all(new_n1._params[1][2:4,0:2] == n1._params[6])
        assert torch.all(new_n1._params[1][2:4,2:4] == n1._params[7])

        assert new_n2.num_node_blocks == 1 and new_n2.block_size == 4
        assert torch.all(new_n2.edge_ids == torch.tensor([[0], [0]]))
        assert torch.all(new_n2._params[0][0:2,0:2] == n2._params[0])
        assert torch.all(new_n2._params[0][0:2,2:4] == n2._params[1])
        assert torch.all(new_n2._params[0][2:4,0:2] == n2._params[2])
        assert torch.all(new_n2._params[0][2:4,2:4] == n2._params[3])


def test_block_sparse_block():

    with set_block_size(block_size = 4):

        ni0 = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 4, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(1, num_node_blocks = 4, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)

        edge_ids = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], 
            [0, 1, 3, 4, 5, 6, 7, 1, 2, 3, 4, 6, 7, 4, 5, 6, 7, 5, 6, 7]
        ], dtype = torch.long)
        ns = summate(np0, np1, edge_ids = edge_ids)

    ns.init_parameters()

    for use_cuda in [True, False]:
        new_ns = juice.blockify(ns, use_cuda = use_cuda)

        new_np0 = new_ns.chs[0]
        new_np1 = new_ns.chs[1]

        new_ni0 = new_np0.chs[0]
        new_ni1 = new_np0.chs[1]
        new_ni2 = new_np1.chs[0]
        new_ni3 = new_np1.chs[1]

        assert new_ni0.block_size == 16 and new_ni0.num_node_blocks == 1
        assert new_ni1.block_size == 16 and new_ni1.num_node_blocks == 1
        assert new_ni2.block_size == 16 and new_ni2.num_node_blocks == 1
        assert new_ni3.block_size == 16 and new_ni3.num_node_blocks == 1

        assert new_np0.num_node_blocks == 1 and new_np0.block_size == 16
        assert new_np0.is_block_sparse()
        assert torch.all(new_np0.edge_ids == torch.tensor([[0, 0]]))

        assert new_np1.num_node_blocks == 1 and new_np1.block_size == 16
        assert new_np1.is_block_sparse()
        assert torch.all(new_np1.edge_ids == torch.tensor([[0, 0]]))

        assert new_ns.num_node_blocks == 2 and new_ns.block_size == 8
        assert new_ns.ch_block_size == 16
        assert torch.all(new_ns.edge_ids == torch.tensor([[0, 0, 1], [0, 1, 1]]))
        assert torch.all(new_ns._params[0][0:4,0:4] == ns._params[0])
        assert torch.all(new_ns._params[0][0:4,4:8] == ns._params[1])
        assert torch.all(new_ns._params[0][0:4,8:12] == 0.0)
        assert torch.all(new_ns._params[0][0:4,12:16] == ns._params[2])
        assert torch.all(new_ns._params[1][0:4,0:4] == ns._params[3])
        assert torch.all(new_ns._params[1][0:4,4:8] == ns._params[4])
        assert torch.all(new_ns._params[1][0:4,8:12] == ns._params[5])
        assert torch.all(new_ns._params[1][0:4,12:16] == ns._params[6])
        assert torch.all(new_ns._params[0][4:8,0:4] == 0.0)
        assert torch.all(new_ns._params[0][4:8,4:8] == ns._params[7])
        assert torch.all(new_ns._params[0][4:8,8:12] == ns._params[8])
        assert torch.all(new_ns._params[0][4:8,12:16] == ns._params[9])
        assert torch.all(new_ns._params[1][4:8,0:4] == ns._params[10])
        assert torch.all(new_ns._params[1][4:8,4:8] == 0.0)
        assert torch.all(new_ns._params[1][4:8,8:12] == ns._params[11])
        assert torch.all(new_ns._params[1][4:8,12:16] == ns._params[12])
        assert torch.all(new_ns._params[2][0:4,0:4] == ns._params[13])
        assert torch.all(new_ns._params[2][0:4,4:8] == ns._params[14])
        assert torch.all(new_ns._params[2][0:4,8:12] == ns._params[15])
        assert torch.all(new_ns._params[2][0:4,12:16] == ns._params[16])
        assert torch.all(new_ns._params[2][4:8,0:4] == 0.0)
        assert torch.all(new_ns._params[2][4:8,4:8] == ns._params[17])
        assert torch.all(new_ns._params[2][4:8,8:12] == ns._params[18])
        assert torch.all(new_ns._params[2][4:8,12:16] == ns._params[19])

        assert torch.all(new_ns._zero_param_mask[0][0:4,8:12])
        assert torch.all(new_ns._zero_param_mask[0][4:8,0:4])
        assert torch.all(new_ns._zero_param_mask[1][4:8,4:8])
        assert torch.all(new_ns._zero_param_mask[2][4:8,0:4])

        assert new_ns._zero_param_mask.long().sum() == 4 * 4 * 4


if __name__ == "__main__":
    test_block()
    test_block_sparse_block()
