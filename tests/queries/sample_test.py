import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_sample():

    device = torch.device("cuda:0")

    with juice.set_block_size(block_size = 16):
        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
        n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

        m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

        ms = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        ns = summate(ms, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)
    pc.to(device)

    samples = juice.queries.sample(pc, num_samples = 16)

    assert ((samples >= 0) & (samples < 2)).all()


if __name__ == "__main__":
    test_sample()