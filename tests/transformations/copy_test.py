import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.transformations import deepcopy

import pytest


def copy_test():
    num_nodes = 2

    i00 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i10 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i11 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    
    ms0 = multiply(i00, i10)
    ms1 = multiply(i00, i11)

    ns = summate(ms0, ms1, num_nodes = 1)

    ns.init_parameters()

    ## Copy without parameter tying ##

    new_ns = deepcopy(ns)

    assert ns.num_nodes == new_ns.num_nodes
    assert torch.all(ns.edge_ids == new_ns.edge_ids)
    assert torch.all(ns._params == new_ns._params)

    assert torch.all(ns.chs[0].edge_ids == new_ns.chs[0].edge_ids)
    assert torch.all(ns.chs[1].edge_ids == new_ns.chs[1].edge_ids)

    assert new_ns.chs[0].chs[0] == new_ns.chs[1].chs[0]
    assert torch.all((ns.chs[0].chs[0].get_params() - new_ns.chs[0].chs[0].get_params()).abs() < 1e-6)
    assert torch.all((ns.chs[0].chs[1].get_params() - new_ns.chs[0].chs[1].get_params()).abs() < 1e-6)
    assert torch.all((ns.chs[1].chs[1].get_params() - new_ns.chs[1].chs[1].get_params()).abs() < 1e-6)

    ## Copy with parameter tying ##

    new_ns = deepcopy(ns, tie_params = True, var_mapping = {0: 2, 1: 3})

    assert ns.num_nodes == new_ns.num_nodes
    assert torch.all(ns.edge_ids == new_ns.edge_ids)
    assert new_ns.get_source_ns() == ns

    assert torch.all(ns.chs[0].edge_ids == new_ns.chs[0].edge_ids)
    assert torch.all(ns.chs[1].edge_ids == new_ns.chs[1].edge_ids)

    assert new_ns.chs[0].chs[0].get_source_ns() == ns.chs[0].chs[0]
    assert new_ns.chs[0].chs[1].get_source_ns() == ns.chs[0].chs[1]
    assert new_ns.chs[1].chs[1].get_source_ns() == ns.chs[1].chs[1]
    assert tuple(new_ns.chs[0].chs[0].scope.to_list()) == (2,)
    assert tuple(new_ns.chs[0].chs[1].scope.to_list()) == (3,)
    assert tuple(new_ns.chs[1].chs[1].scope.to_list()) == (3,)


if __name__ == "__main__":
    copy_test()