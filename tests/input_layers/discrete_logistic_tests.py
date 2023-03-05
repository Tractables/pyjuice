import pyjuice as juice
import torch
import numpy as np

from pyjuice.graph import RegionGraph, InputRegionNode, InnerRegionNode, PartitionNode
from pyjuice.layer import DiscreteLogisticLayer
from pyjuice.model import ProbCircuit

import pytest

def test_discrete_logistic_layer():
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda:0")

    inputs = [InputRegionNode(scope = [i], num_nodes = 2, node_type = DiscreteLogisticLayer, input_range = [0, 1], bin_count = 256) for i in range(4)]

    layer = DiscreteLogisticLayer(0, inputs, 0)
    layer.to(device)

    data = torch.rand([4, 1]).to(device)
    node_mars = torch.zeros([8, 1]).to(device)
    node_flows = torch.ones([8, 1]).to(device)

    layer.init_param_flows()

    last_ll = -100000.0
    for _ in range(100):
        layer(data, node_mars)

        layer.backward(data, node_flows, node_mars)
        
        layer.mini_batch_em(step_size = 0.01)

        assert node_mars.mean() > last_ll - 0.001
        last_ll = node_mars.mean()


if __name__ == "__main__":
    test_discrete_logistic_layer()