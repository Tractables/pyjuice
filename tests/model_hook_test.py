import pyjuice as juice
import torch
import numpy as np

from pyjuice.graph import RegionGraph, InputRegionNode, InnerRegionNode, PartitionNode
from pyjuice.layer import CategoricalLayer
from pyjuice.model import ProbCircuit

import pytest

def test_model_hook():    
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

    lls = pc(data)
    lls.mean().backward()
