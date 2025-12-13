import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists
from functools import partial

import pytest


def test_blk_sparse_hclt():

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

    edge_constructor = partial(
        juice.nodes.edge_constructors.block_sparse_rnd_blk_edge_constructor,
        num_chs_per_block = 4
    )

    ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = 128, 
        chunk_size = 32,
        block_size = 16,
        sum_edge_ids_constructor = edge_constructor
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    assert True


if __name__ == "__main__":
    test_blk_sparse_hclt()
