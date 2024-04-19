import pyjuice as juice
import torch
import tempfile
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.io import load, save

import pytest


def test_io():
    num_node_blocks = 2
    block_size = 4

    with juice.set_block_size(block_size):
        i00 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
        i01 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
        i10 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
        i11 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
        
        m00 = multiply(i00, i10)
        m01 = multiply(i01, i11)

        n0 = summate(m00, m01, num_node_blocks = num_node_blocks)

    temp_file = tempfile.NamedTemporaryFile(suffix='.jpc')
    temp_file_name = temp_file.name
    save(temp_file_name, n0)

    n0_dup = load(temp_file_name)

    assert (n0.edge_ids == n0_dup.edge_ids).all()
    assert len(n0.chs) == len(n0_dup.chs)
    assert (n0.chs[0].edge_ids == n0_dup.chs[0].edge_ids).all()
    assert (n0.chs[1].edge_ids == n0_dup.chs[1].edge_ids).all()
    assert n0.chs[0].chs[0].scope == n0_dup.chs[0].chs[0].scope
    assert n0.chs[0].chs[1].scope == n0_dup.chs[0].chs[1].scope
    assert n0.chs[1].chs[0].scope == n0_dup.chs[1].chs[0].scope
    assert n0.chs[1].chs[1].scope == n0_dup.chs[1].chs[1].scope
    assert n0.chs[0].chs[0].dist.num_cats == n0_dup.chs[0].chs[0].dist.num_cats
    assert n0.chs[0].chs[1].dist.num_cats == n0_dup.chs[0].chs[1].dist.num_cats
    assert n0.chs[1].chs[0].dist.num_cats == n0_dup.chs[1].chs[0].dist.num_cats
    assert n0.chs[1].chs[1].dist.num_cats == n0_dup.chs[1].chs[1].dist.num_cats


def test_io_param():
    num_node_blocks = 2
    block_size = 4

    with juice.set_block_size(block_size):
        i00 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
        i01 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
        i10 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
        i11 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
        
        m00 = multiply(i00, i10)
        m01 = multiply(i01, i11)

        n0 = summate(m00, m01, num_node_blocks = num_node_blocks)

    n0.init_parameters()

    pc = juice.TensorCircuit(n0)

    temp_file = tempfile.NamedTemporaryFile(suffix='.jpc')
    temp_file_name = temp_file.name
    save(temp_file_name, pc)

    n0_dup = load(temp_file_name)

    assert torch.all(torch.abs(n0._params - n0_dup._params) < 1e-4)
    assert torch.all(torch.abs(n0.chs[0].chs[0]._params - n0_dup.chs[0].chs[0]._params) < 1e-4)
    assert torch.all(torch.abs(n0.chs[0].chs[1]._params - n0_dup.chs[0].chs[1]._params) < 1e-4)
    assert torch.all(torch.abs(n0.chs[1].chs[0]._params - n0_dup.chs[1].chs[0]._params) < 1e-4)
    assert torch.all(torch.abs(n0.chs[1].chs[1]._params - n0_dup.chs[1].chs[1]._params) < 1e-4)


if __name__ == "__main__":
    test_io()
    test_io_param()
