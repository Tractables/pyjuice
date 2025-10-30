import pyjuice as juice
import torch
import tempfile

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs

import pytest


def test_external_dist_io():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.External())

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    pc = juice.compile(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 2]).to(device)
    external_soft_evi = torch.rand([16, 1, 2], dtype = torch.float32).to(device)

    lls = pc(data, external_soft_evi = external_soft_evi)

    temp_file = tempfile.NamedTemporaryFile(suffix='.jpc')
    temp_file_name = temp_file.name
    juice.save(temp_file_name, pc)

    new_ns = juice.load(temp_file_name)

    assert isinstance(new_ns.chs[0].chs[1].dist, juice.distributions.External)


if __name__ == "__main__":
    test_external_dist_io()