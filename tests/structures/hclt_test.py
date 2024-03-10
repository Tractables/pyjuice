import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists

import pytest


def evaluate(pc, loader):
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        lls_total += lls.mean().detach().cpu().numpy().item()
    
    lls_total /= len(loader)
    return lls_total


def mini_batch_em_epoch(num_epochs, pc, optimizer, scheduler, train_loader, test_loader, device):
    for epoch in range(num_epochs):
        t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            lls = pc(x)
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            optimizer.step()
            scheduler.step()

        train_ll /= len(train_loader)

        t1 = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t2 = time.time()

        print(f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def full_batch_em_epoch(pc, train_loader, test_loader, device):
    t0 = time.time()
    train_ll = 0.0
    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x)
        lls.mean().backward()

        train_ll += lls.mean().detach().cpu().numpy().item()

    pc.mini_batch_em(step_size = 1.0, pseudocount = 0.1)

    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = evaluate(pc, loader=test_loader)
    t2 = time.time()
    print(f"[train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def test_hclt():

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

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1)
    scheduler = juice.optim.CircuitScheduler(
        optimizer, 
        method = "multi_linear", 
        lrs = [0.9, 0.1, 0.05], 
        milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]
    )

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(5, pc, optimizer, scheduler, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -785


@pytest.mark.slow
def test_small_hclt_full():

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
        num_latents = 32, 
        chunk_size = 32
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1)
    scheduler = juice.optim.CircuitScheduler(
        optimizer, 
        method = "multi_linear", 
        lrs = [0.9, 0.1, 0.05], 
        milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]
    )

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(350, pc, optimizer, scheduler, train_loader, test_loader, device)
    full_batch_em_epoch(pc, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -660


@pytest.mark.slow
def test_large_hclt_full():

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

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1)
    scheduler = juice.optim.CircuitScheduler(
        optimizer, 
        method = "multi_linear", 
        lrs = [0.9, 0.1, 0.05], 
        milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]
    )

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(350, pc, optimizer, scheduler, train_loader, test_loader, device)
    full_batch_em_epoch(pc, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -640


def test_hclt_logistic():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28).float()
    test_data = test_dataset.data.reshape(10000, 28*28).float()

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
        chunk_size = 32,
        input_node_type = dists.DiscreteLogistic,
        input_node_params = {"val_range": (-1.0, 1.0), "num_cats": 256}
    )
    ns.init_parameters(perturbation = 4.0)
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1)
    scheduler = juice.optim.CircuitScheduler(
        optimizer, 
        method = "multi_linear", 
        lrs = [0.9, 0.1, 0.05], 
        milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]
    )

    mini_batch_em_epoch(20, pc, optimizer, scheduler, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -995.0


if __name__ == "__main__":
    # torch.manual_seed(3289)
    test_hclt()
    test_small_hclt_full()
    test_large_hclt_full()
    test_hclt_logistic()
