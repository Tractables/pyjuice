import pyjuice as juice
import torch
import torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from pyjuice.graph import RegionGraph, InputRegionNode, InnerRegionNode, PartitionNode
from pyjuice.layer import CategoricalLayer
from pyjuice.model import ProbCircuit
from pyjuice.transformations import prune, duplicate

import warnings
import logging

import pytest

def test_prune_1():
    inputs = [InputRegionNode(scope = [i], num_nodes = 2, node_type = CategoricalLayer, num_cats = 2) for i in range(4)]
    part1 = PartitionNode([inputs[0], inputs[1]], num_nodes = 4, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int))
    sum1 = InnerRegionNode([part1], num_nodes=2, edge_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype=int))

    part2 = PartitionNode([inputs[2], inputs[3]], num_nodes = 2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype=int))
    sum2 = InnerRegionNode([part2], num_nodes=2, edge_ids=torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=int))

    rootp = PartitionNode([sum1, sum2], num_nodes = 2, edge_ids=torch.tensor([[0, 0], [1, 1]], dtype=int))
    root_sum = InnerRegionNode([rootp], num_nodes = 1, edge_ids=torch.tensor([[0, 0], [0, 1]], dtype=int))

    pc = ProbCircuit(root_sum)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    pc.init_param_flows()
    pc.param_flows[:] = 1.0
    pc.param_flows[[1, 3, 5, 14]] = 0.0

    params1 = pc.params.data.clone()
    input_params1 = pc.input_layers[0].params.data.clone()

    pc = prune(pc, 11 / 15)

    params2 = pc.params.data.clone()
    input_params2 = pc.input_layers[0].params.data.clone()

    assert torch.abs(params1[2] / (params1[2] + params1[4]) - params2[1]) < 1e-4
    assert torch.abs(params1[4] / (params1[2] + params1[4]) - params2[2]) < 1e-4

    assert torch.abs(params1[9] / (params1[9] + params1[10]) - params2[3]) < 1e-4
    assert torch.abs(params1[10] / (params1[9] + params1[10]) - params2[4]) < 1e-4

    assert torch.abs(input_params1[0] - input_params2[0]) < 1e-4
    assert torch.abs(input_params1[1] - input_params2[1]) < 1e-4
    assert torch.abs(input_params1[2] - input_params2[2]) < 1e-4
    assert torch.abs(input_params1[3] - input_params2[3]) < 1e-4

    assert torch.abs(input_params1[6] - input_params2[4]) < 1e-4
    assert torch.abs(input_params1[7] - input_params2[5]) < 1e-4


def test_prune_2():    
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 512

    inputs = [InputRegionNode(scope = [i], num_nodes = 2, node_type = CategoricalLayer, num_cats = 2) for i in range(4)]

    part1 = PartitionNode([inputs[0], inputs[1]], num_nodes = 4, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int))
    sum1 = InnerRegionNode([part1], num_nodes=2, edge_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype=int))

    part2 = PartitionNode([inputs[2], inputs[3]], num_nodes = 2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype=int))
    sum2 = InnerRegionNode([part2], num_nodes=2, edge_ids=torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=int))

    rootp = PartitionNode([sum1, sum2], num_nodes = 2, edge_ids=torch.tensor([[0, 0], [1, 1]], dtype=int))
    root_sum = InnerRegionNode([rootp], num_nodes = 1, edge_ids=torch.tensor([[0, 0], [0, 1]], dtype=int))

    pc = ProbCircuit(root_sum)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)
    lls1 = pc(data)
    pc = duplicate(pc, sigma = 0.0)
    lls2 = pc(data)
    assert torch.all(torch.abs(lls1 - lls2) < 1e-4)
