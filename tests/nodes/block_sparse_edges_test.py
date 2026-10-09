"""
Block-sparse sum edges: `block_sparse_rnd_blk_edge_constructor` never connects a sum block to the same
child block twice, and the dense views of a sum's parameters and parameter flows
(`get_params(as_matrix = True)`, `get_param_flows(as_matrix = True)`) count every copy of an edge block
that does appear twice -- as the forward pass does.
"""
import itertools

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes.methods.edge_constructors import block_sparse_rnd_blk_edge_constructor


def child_groups(block_counts, block_size = 4):
    return [inputs(v, num_node_blocks = c, block_size = block_size, dist = dists.Categorical(num_cats = 3))
            for v, c in enumerate(block_counts)]


@pytest.mark.parametrize("num_node_blocks, block_counts, num_chs_per_block", [
    (4, [4], 2),          # as in an HMM with 4 node blocks per position
    (8, [4], 3),
    (16, [3, 5], 3),      # two child groups
    (3, [2], 4),          # more children requested than exist: every child block, once
    (1, [6], 6),
    (2, [10], 3),         # fewer slots than child blocks: a random subset, no block twice
])
def test_random_block_sparse_edges_are_distinct_per_sum_block(num_node_blocks, block_counts, num_chs_per_block):
    num_ch_blocks = sum(block_counts)
    k = min(num_chs_per_block, num_ch_blocks)
    for seed in range(20):
        torch.manual_seed(seed)
        edge_ids = block_sparse_rnd_blk_edge_constructor(*child_groups(block_counts), num_node_blocks = num_node_blocks,
                                                         block_size = 4, num_chs_per_block = num_chs_per_block)
        assert edge_ids.shape == (2, num_node_blocks * k)
        assert torch.equal(edge_ids[0], torch.arange(num_node_blocks).repeat_interleave(k))
        assert ((edge_ids[1] >= 0) & (edge_ids[1] < num_ch_blocks)).all()
        for blk in range(num_node_blocks):
            chs = edge_ids[1, edge_ids[0] == blk]
            assert chs.unique().numel() == k, (seed, blk, chs.tolist())          # no child block twice
        if num_node_blocks * k >= num_ch_blocks:
            assert edge_ids[1].unique().numel() == num_ch_blocks               # every child block connected
        else:
            assert edge_ids[1].unique().numel() == num_node_blocks * k         # no child block reused

    assert block_sparse_rnd_blk_edge_constructor(*child_groups([4], block_size = 1), num_node_blocks = 4,
                                                 block_size = 1, num_chs_per_block = 2) is None


def repeated_edge_pc():
    """A sum whose first node block connects to child block 1 twice."""
    x0 = inputs(0, num_node_blocks = 2, block_size = 2, dist = dists.Categorical(num_cats = 3))
    x1 = inputs(1, num_node_blocks = 2, block_size = 2, dist = dists.Categorical(num_cats = 3))
    prod = multiply(x0, x1)
    s = summate(prod, num_node_blocks = 2, block_size = 2, edge_ids = torch.tensor([[0, 0, 1], [1, 1, 0]]))
    root = summate(multiply(s), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    return root, s, x0, x1


def test_dense_parameters_count_every_copy_of_a_repeated_edge_block():
    root, s, x0, x1 = repeated_edge_pc()
    dense = s.get_params(as_matrix = True)                                     # [4 nodes, 4 child nodes]
    assert torch.allclose(dense.sum(dim = 1), torch.ones(4))                  # normalized, like the edges
    assert torch.allclose(dense[0:2, 2:4], s.get_params()[0] + s.get_params()[1])

    # the dense matrix reproduces the compiled circuit's probabilities
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    X = torch.tensor(list(itertools.product(range(3), repeat = 2)))
    theta0, theta1 = x0.get_params().reshape(4, 3), x1.get_params().reshape(4, 3)
    prods = theta0[:, X[:, 0]] * theta1[:, X[:, 1]]                            # [4 products, 9 strings]
    want = root.get_params(as_matrix = True) @ (dense @ prods)                 # [1, 9]
    assert torch.allclose(pc(X.to(pc.device))[:, 0].exp().cpu(), want[0], atol = 1e-6)
    assert torch.allclose(want.sum(), torch.tensor(1.0), atol = 1e-6)


def test_dense_parameter_flows_count_every_copy_of_a_repeated_edge_block():
    root, s, _, _ = repeated_edge_pc()
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    X = torch.tensor(list(itertools.product(range(3), repeat = 2)), device = pc.device)
    pc.init_param_flows(flows_memory = 0.0)
    pc(X)
    pc.backward(X, allow_modify_flows = False)
    pc.update_param_flows()
    flows = s.get_param_flows()                                                # [3 edge blocks, 2, 2]
    dense = s.get_param_flows(as_matrix = True)
    assert torch.allclose(dense.sum(), flows.sum())                            # nothing dropped
    assert torch.allclose(dense[0:2, 2:4], flows[0] + flows[1])
