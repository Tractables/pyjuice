import pyjuice as juice
import torch
import numpy as np
import math

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_external_categorical_dist_sample():

    num_cats = 3
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 1, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 1, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    ns.init_parameters(perturbation = 16)
    ni0._params[0] = 0.02
    ni0._params[1] = 0.08
    ni0._params[2] = 0.9
    ni0._params[3:] = ni0._params[:3]
    ni1._params[0] = 0.02
    ni1._params[1] = 0.95
    ni1._params[2] = 0.03
    ni1._params[3:] = ni1._params[:3]

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    external_categorical_logps = torch.ones([128, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    samples = juice.queries.sample(pc, num_samples = 128, external_categorical_logps = external_categorical_logps)

    assert (samples[:,0] == 2).float().mean() > 0.7
    assert (samples[:,1] == 1).float().mean() > 0.7


if __name__ == "__main__":
    test_external_categorical_dist_sample()
