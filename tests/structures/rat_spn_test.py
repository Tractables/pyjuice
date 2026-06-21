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


def _multi_linear_step_size(step, lrs, milestone_steps):
    if step >= milestone_steps[-1]:
        return lrs[-1]
    idx = sum(1 for ms in milestone_steps if ms < step)
    if idx == 0:
        return lrs[0]
    return lrs[idx - 1] + (lrs[idx] - lrs[idx - 1]) * (step - milestone_steps[idx - 1]) / \
           (milestone_steps[idx] - milestone_steps[idx - 1])


def mini_batch_em_epoch(num_epochs, pc, optimizer, train_loader, test_loader, device, base_lr, lrs, milestone_steps):
    for epoch in range(num_epochs):
        t0 = time.time()
        train_ll = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)

            lls = pc(x)
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            # Reproduce the original (optimizer.step() then scheduler.step()) ordering: the very
            # first minibatch used the optimizer's initial lr, and minibatch m (m >= 1) used the
            # scheduler value computed at step_count = m - 1.
            minibatch = epoch * len(train_loader) + batch_idx
            if minibatch == 0:
                step_size = base_lr
            else:
                step_size = _multi_linear_step_size(minibatch - 1, lrs, milestone_steps)
            optimizer.step(step_size = step_size)

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


def test_rat_spn():

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

    ns = juice.structures.RAT_SPN(
        num_vars = 28 * 28,
        num_latents = 64, 
        depth = 5,
        num_repetitions = 4,
        num_pieces = 2
    )
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    base_lr = 0.1
    optimizer = juice.optim.MiniBatchEM(pc, step_size = base_lr, pseudocount = 0.1)
    lrs = [0.9, 0.1, 0.05]
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()
        break

    mini_batch_em_epoch(20, pc, optimizer, train_loader, test_loader, device, base_lr, lrs, milestone_steps)

    test_ll = evaluate(pc, test_loader)

    assert test_ll > -1020


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(3289)
    test_rat_spn()
