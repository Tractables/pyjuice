import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs

import pytest


def counting_lvd_test():
    num_nodes = 2

    with juice.LVDistiller(backend = "counting"):
        i0 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
        i1 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
        i2 = inputs(2, num_nodes, dists.Categorical(num_cats = 5))
        i3 = inputs(3, num_nodes, dists.Categorical(num_cats = 5))

        m0 = multiply(i0, i1, lv_dataset = torch.tensor([0,0,1,1]))
        n0 = summate(m0, num_nodes = num_nodes)

        m1 = multiply(i2, i3, lv_dataset = torch.tensor([1,1,0,0]))
        n1 = summate(m1, num_nodes = num_nodes)

        m = multiply(n0, n1, lv_dataset = torch.tensor([0,0,1,1]))
        n = summate(m, num_nodes = 1)

    assert torch.abs(n0._params - torch.tensor([1.0, 0.0, 0.0, 1.0])).max() < 1e-6
    assert torch.abs(n1._params - torch.tensor([0.0, 1.0, 1.0, 0.0])).max() < 1e-6


if __name__ == "__main__":
    counting_lvd_test()