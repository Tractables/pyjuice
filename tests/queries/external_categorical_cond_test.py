import pyjuice as juice
import torch
import numpy as np
import torchvision
import time

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_ext_cat_soft_cond():

    num_cats = 128
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    external_categorical_value_mask = torch.zeros([16, 2], dtype = torch.bool, device = device)
    external_categorical_value_mask[8:,:] = True

    outputs = juice.queries.conditional(
        pc, data = data, 
        extern_product_categorical_mode = "unnormalized_ll",
        external_categorical_logps = external_categorical_logps,
        external_categorical_value_mask = external_categorical_value_mask
    )

    assert torch.all((outputs.sum(dim = 2) - 1.0).abs() < 1e-4)


if __name__ == "__main__":
    test_ext_cat_soft_cond()