import math
import matplotlib.pyplot as plt
import numpy as np
import pyjuice as juice
import pyjuice.nodes.distributions as dists
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import pytest


def gen_data(sample_size: int = 10_000, dim: int = 1, cond: float = None, shift: float = None) -> tuple[torch.tensor]:
    """Generates simple analytical dataset.

    Args:
        sample_size: Number of sample points. Defaults to 10_000.
        dim: Dimensions of problem. Defaults to 1.

    Returns:
        tuple[torch.tensor]: (theta, x)
    """
    shift = shift if shift else [0] * dim
    if cond:
        theta = torch.tensor([cond] * sample_size).reshape(-1, 1)
    else:
        theta = torch.rand((sample_size, dim)) + torch.tensor([shift])
    noise = torch.randn((sample_size, dim)) * 0.1

    x = theta + 0.4 * torch.sin(2 * math.pi * theta) + noise

    return theta, x


def _multi_linear_step_size(step, lrs, milestone_steps):
    if step >= milestone_steps[-1]:
        return lrs[-1]
    idx = sum(1 for ms in milestone_steps if ms < step)
    if idx == 0:
        return lrs[0]
    return lrs[idx - 1] + (lrs[idx] - lrs[idx - 1]) * (step - milestone_steps[idx - 1]) / \
           (milestone_steps[idx] - milestone_steps[idx - 1])


def train(pc, train_loader, device, num_epochs = 100):

    optimizer = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    lrs = [0.1, 0.0001]
    milestone_steps = [0, len(train_loader) * num_epochs]

    for epoch in range(1, num_epochs + 1):
        train_ll = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)

            lls = pc(x)
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            # Faithfully reproduce the old CircuitOptimizer + CircuitScheduler timing:
            # the original loop called optimizer.step() (using the lr from the *previous*
            # scheduler.step()) and then scheduler.step(); the scheduler's internal counter
            # therefore lagged the optimizer step by one, so step #k used schedule value k-2
            # (clamped at the lrs[0] plateau, which equals the constructor lr 0.1).
            step_count = (epoch - 1) * len(train_loader) + batch_idx - 1
            optimizer.step(step_size = _multi_linear_step_size(step_count, lrs, milestone_steps))

        train_ll /= len(train_loader)

        print(f"Train LL: {train_ll:.2f}")


def test_cond_sample():

    device = torch.device("cuda:0")

    sample_size = 1_000
    shift = 0.0
    dim = 1

    # generate training data
    theta, x = gen_data(sample_size, dim, shift=shift)

    # generate test data
    theta_test, x_test = gen_data(sample_size, dim, shift=shift)
    theta_test.shape, x_test.shape

    # prepare data set and data loader
    train_data = torch.cat((x, theta), dim=1)
    dataset = TensorDataset(train_data)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    ns = juice.structures.HCLT(
        train_data.float().to(device),
        num_latents = 512,
        input_dist = dists.Gaussian(mu = 0.0, sigma = 0.5),
    )

    pc = juice.compile(ns)
    pc.to(device)

    train(pc, train_loader, device)

    uncond_samples = juice.queries.sample(pc, 10_000)

    theta_cond = 0.4

    # define the missing mask and evidence
    data = torch.tensor([0, theta_cond])[None,:].expand(256, -1).contiguous().to(device)
    missing_mask = torch.tensor([True, False]).to(device)

    # propagate the evidence through the model
    # note, this also computes the marginal
    lls = pc(data, missing_mask = missing_mask)

    samples_cond = juice.queries.sample(pc, conditional = True)

    assert samples_cond[:,0].mean() > 0.6 and samples_cond[:,0].mean() < 0.7
    assert samples_cond[:,0].std() > 0.075 and samples_cond[:,0].std() < 0.120

    # plt.figure(figsize=(10, 5))
    # plt.scatter(theta_test.cpu(), x_test.cpu(), s = 1, color="blue")
    # plt.scatter(uncond_samples[:, 1].cpu(), uncond_samples[:, 0].cpu(), s = 1, color="green")
    # plt.scatter(
    #     torch.tensor([theta_cond]).repeat(samples_cond.shape[0]),
    #     samples_cond[:, 0].cpu(),
    #     s = 5,
    #     color = "red",
    #     alpha = 0.05
    # )
    # plt.xlabel(r"$\theta$")
    # plt.ylabel("x")
    # plt.title("Samples from the model", fontsize=18)
    # plt.savefig("1.png")


if __name__ == "__main__":
    torch.manual_seed(2389)
    test_cond_sample()
