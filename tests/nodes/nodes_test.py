import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs

import pytest


def nodes_test():
    num_node_groups_candidates = [4, 8, 12]
    group_size_candidates = [1, 2, 4, 16]

    for num_node_groups in num_node_groups_candidates:
        for group_size in group_size_candidates:

            num_nodes = num_node_groups * group_size

            with juice.set_group_size(group_size):
                n0 = inputs(0, num_node_groups, dists.Categorical(num_cats = 5))
                n1 = inputs(1, num_node_groups, dists.Categorical(num_cats = 5))
                n2 = inputs(2, num_node_groups, dists.Categorical(num_cats = 5))

                assert n0.num_nodes == num_nodes and n1.num_nodes == num_nodes and n2.num_nodes == num_nodes

                m = multiply(n0, n1, n2)

                assert m.num_nodes == num_nodes
                assert m.scope == BitSet.from_array([0,1,2])
                assert m.num_edges == num_nodes * 3

                n = summate(m, num_node_groups = 1)

                assert n.num_nodes == group_size
                assert n.num_edges == num_node_groups * (group_size ** 2)


if __name__ == "__main__":
    nodes_test()