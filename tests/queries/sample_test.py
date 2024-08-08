import pyjuice as juice
import torch
import torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_sample():

    device = torch.device("cuda:0")

    with juice.set_block_size(block_size = 16):
        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
        n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

        m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

        ms = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        ns = summate(ms, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)
    pc.to(device)

    samples = juice.queries.sample(pc, num_samples = 16)

    assert ((samples >= 0) & (samples < 2)).all()


def test_sample_correctness():

    device = torch.device("cuda:0")

    with juice.set_block_size(block_size = 1):
        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        np = multiply(ni0, ni1)

        ns = summate(np, num_node_blocks = 1)

    ns.init_parameters(perturbation = 1.0)

    ns._params[0,0,0] = 0.2
    ns._params[1,0,0] = 0.8

    ni0._params[0] = 0.999
    ni0._params[1] = 0.001
    ni0._params[2] = 0.001
    ni0._params[3] = 0.999

    ni1._params[0] = 0.001
    ni1._params[1] = 0.999
    ni1._params[2] = 0.999
    ni1._params[3] = 0.001

    pc = juice.TensorCircuit(ns)

    pc.to(device)

    samples = juice.queries.sample(pc, num_samples = 512)

    assert samples[:,0].float().mean() > 0.6
    assert samples[:,1].float().mean() < 0.4


def test_sample_hclt():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)
    test_data = test_dataset.data.reshape(10000, 28*28)

    num_features = train_data.size(1)

    train_loader = DataLoader(
        dataset = TensorDataset(train_data),
        batch_size = 512,
        shuffle = True,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = 512,
        shuffle = False,
        drop_last = True
    )

    ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = 128, 
        chunk_size = 32
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    samples = juice.queries.sample(pc, num_samples = 16)

    assert ((samples >= 0) & (samples < 256)).all()


def test_sample_hmm():
    
    device = torch.device("cuda:0")

    ns = juice.structures.HMM(seq_length = 32, num_latents = 256, num_emits = 100, homogeneous = True, block_size = 64)
    ns.init_parameters(perturbation = 2.0)

    pc = juice.compile(ns)
    pc.to(device)

    samples = juice.queries.sample(pc, num_samples = 16)

    assert ((samples >= 0) & (samples < 100)).all()


if __name__ == "__main__":
    torch.set_num_threads(4)
    test_sample()
    test_sample_correctness()
    test_sample_hclt()
    test_sample_hmm()