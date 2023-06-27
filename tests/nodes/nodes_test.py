import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs

import pytest


def nodes_test():
    num_nodes = 8

    n0 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    n1 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    n2 = inputs(2, num_nodes, dists.Categorical(num_cats = 5))
    assert n0.num_nodes == 8 and n1.num_nodes == 8 and n2.num_nodes == 8

    m = multiply(n0, n1, n2)
    assert m.num_nodes == 8
    assert m.scope == BitSet.from_array([0,1,2])
    n = summate(m, num_nodes = 1)
    assert n.num_nodes == 1


if __name__ == "__main__":
    nodes_test()