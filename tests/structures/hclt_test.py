import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists

import pytest


def _multi_linear_step_size(step, lrs, milestone_steps):
    if step >= milestone_steps[-1]:
        return lrs[-1]
    idx = sum(1 for ms in milestone_steps if ms < step)
    if idx == 0:
        return lrs[0]
    return lrs[idx - 1] + (lrs[idx] - lrs[idx - 1]) * (step - milestone_steps[idx - 1]) / \
           (milestone_steps[idx] - milestone_steps[idx - 1])


def _scheduled_step_size(minibatch_idx, base_lr, lrs, milestone_steps):
    # Faithfully reproduce the original `optimizer.step(); scheduler.step()` ordering: the very first
    # minibatch ran before any `scheduler.step()`, so it used the optimizer's constructor `lr`; every
    # subsequent minibatch used the multi-linear value at `scheduler.step_count = minibatch_idx - 1`.
    if minibatch_idx == 0:
        return base_lr
    return _multi_linear_step_size(minibatch_idx - 1, lrs, milestone_steps)


def evaluate(pc, loader):
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        lls_total += lls.mean().detach().cpu().numpy().item()
    
    lls_total /= len(loader)
    return lls_total


def mini_batch_em_epoch(num_epochs, pc, optimizer, base_lr, lrs, milestone_steps, train_loader, test_loader, device, logspace_flows = False):
    for epoch in range(num_epochs):
        t0 = time.time()
        train_ll = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)

            lls = pc(x)
            if not logspace_flows:
                lls.mean().backward()
            else:
                pc.backward(x, allow_modify_flows = False, logspace_flows = True)

            train_ll += lls.mean().detach().cpu().numpy().item()

            step_count = epoch * len(train_loader) + batch_idx
            optimizer.step(step_size = _scheduled_step_size(step_count, base_lr, lrs, milestone_steps))

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

    optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    lrs = [0.9, 0.1, 0.05]
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(5, pc, optimizer, 0.1, lrs, milestone_steps, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -785


def test_hclt_logspace_flows():

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

    optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    lrs = [0.9, 0.1, 0.05]
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]

    # for batch in train_loader:
    #     x = batch[0].to(device)

    #     lls = pc(x, record_cudagraph = True)
    #     lls.mean().backward()
    #     break

    mini_batch_em_epoch(5, pc, optimizer, 0.1, lrs, milestone_steps, train_loader, test_loader, device, logspace_flows = True)

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

    optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    lrs = [0.9, 0.1, 0.05]
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(350, pc, optimizer, 0.1, lrs, milestone_steps, train_loader, test_loader, device)
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

    optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    lrs = [0.9, 0.1, 0.05]
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(350, pc, optimizer, 0.1, lrs, milestone_steps, train_loader, test_loader, device)
    full_batch_em_epoch(pc, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -640


def test_hclt_logistic():

    device = torch.device("cuda:0")

    # Seed for determinism: this test has a tight LL threshold with little margin, so without a fixed
    # seed the unseeded parameter init + shuffled loader make it RNG-flaky around the threshold.
    torch.manual_seed(42)

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

    optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    lrs = [0.9, 0.1, 0.05]
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]

    mini_batch_em_epoch(20, pc, optimizer, 0.1, lrs, milestone_steps, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -996.0


if __name__ == "__main__":
    torch.set_num_threads(4)
    # torch.manual_seed(3289)
    test_hclt()
    test_hclt_logspace_flows()
    test_small_hclt_full()
    # test_large_hclt_full()
    test_hclt_logistic()
