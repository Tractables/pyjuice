import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_tensorcircuit_fns():

    ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls = pc(data)
    pc.backward(data, allow_modify_flows = False)

    nsid, neid = n2._output_ind_range
    n2_mars = pc.get_node_mars(n2)
    assert n2_mars.size(0) == neid - nsid
    assert n2_mars.size(1) == 16
    assert (n2_mars == pc.node_mars[nsid:neid,:]).all()

    n2_flows = pc.get_node_flows(n2)
    assert n2_flows.size(0) == neid - nsid
    assert n2_flows.size(1) == 16
    assert (n2_flows == pc.node_flows[nsid:neid,:]).all()

    nsid, neid = m2._output_ind_range
    m2_mars = pc.get_node_mars(m2)
    assert m2_mars.size(0) == neid - nsid
    assert m2_mars.size(1) == 16
    assert (m2_mars == pc.element_mars[nsid:neid,:]).all()

    m2_flows = pc.get_node_flows(m2)
    assert m2_flows.size(0) == neid - nsid
    assert m2_flows.size(1) == 16
    assert (m2_flows == pc.element_flows[nsid:neid,:]).all()


if __name__ == "__main__":
    test_tensorcircuit_fns()