import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists

import pytest


@pytest.mark.slow
def test_small_batch_size():
    train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, download = True)
    valid_dataset = torchvision.datasets.MNIST(root = "./data", train = False, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)
    valid_data = valid_dataset.data.reshape(10000, 28*28)

    train_loader = DataLoader(
        dataset = TensorDataset(train_data),
        batch_size = 512,
        shuffle = True,
        drop_last = True
    )
    valid_loader = DataLoader(
        dataset = TensorDataset(valid_data),
        batch_size = 512,
        shuffle = False,
        drop_last = True
    )

    device = torch.device("cuda:0")

    ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_latents = 128
    )
    pc = juice.compile(ns)
    pc.to(device)

    data = valid_data.data[0].unsqueeze(0).to(device, dtype = torch.long)
    lls = juice.queries.marginal(
        pc, data = data
    )

    assert True


if __name__ == "__main__":
    test_small_batch_size()
