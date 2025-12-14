import pyjuice as juice
import torch
import random
import numpy as np

import pytest


def get_all_sequences(seq_length, num_cats):
    """
    Generates a matrix containing all possible sequences of a given length
    using categories from 0 to num_cats-1.
    
    Args:
        seq_length (int): The length of the sequences.
        num_cats (int): The number of possible categories (vocabulary size).
        
    Returns:
        torch.Tensor: A matrix of shape (num_cats^seq_length, seq_length).
    """
    # 1. Create a tensor of all available categories
    cats = torch.arange(num_cats)
    
    # 2. Create a list where the categories are repeated 'seq_length' times
    #    e.g., [tensor([0,1]), tensor([0,1]), tensor([0,1])]
    inputs = [cats] * seq_length
    
    # 3. Compute the Cartesian product to get all combinations
    #    We use the * operator to unpack the list as arguments
    combinations = torch.cartesian_prod(*inputs)
    
    return combinations


def make_peaked_distribution(shape, target_indices=None, noise_level=0.05):
    """
    Creates a probability distribution where most mass is on a target index,
    but small random noise exists elsewhere.
    """
    # 1. Start with small random noise (uniform-ish)
    probs = torch.rand(shape)
    
    # 2. If we have specific targets to peak at, boost them significantly
    if target_indices is not None:
        # Create a mask for the target indices
        if len(shape) == 1:
            # For 1D vector (Gamma)
            probs[target_indices] += (1.0 / noise_level)
        else:
            # For 2D matrix (Alpha/Beta) - assumes target_indices corresponds to rows
            rows = torch.arange(shape[0])
            probs[rows, target_indices] += (1.0 / noise_level)
    else:
        # If no target specified, just pick the diagonal or 0-index to boost
        pass 

    # 3. Normalize to ensure they sum to 1.0 (valid probabilities)
    # If 1D, dim=0. If 2D, dim=1 (normalize rows)
    dim = len(shape) - 1
    probs = probs / probs.sum(dim=dim, keepdim=True)
    
    return probs


def test_sample_hmm_correctness():

    seed_value = 328993719
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    device = torch.device("cuda:0")

    seq_length = 4
    num_latents = 128
    num_emits = 4

    batch_size = 256

    gamma = make_peaked_distribution((num_latents,), target_indices = 0, noise_level = 0.005)

    diagonal_indices = torch.arange(num_latents)
    alpha = make_peaked_distribution(
        (num_latents, num_latents), 
        target_indices = diagonal_indices, 
        noise_level = 0.01
    )

    preferred_emissions = torch.arange(num_latents) % num_emits
    beta = make_peaked_distribution(
        (num_latents, num_emits), 
        target_indices = preferred_emissions, 
        noise_level = 0.02
    )

    ns = juice.structures.HMM(
        seq_length = seq_length, 
        num_latents = num_latents, 
        num_emits = num_emits, 
        homogeneous = True,
        alpha = alpha,
        beta = beta,
        gamma = gamma
    )

    pc = juice.compile(ns)
    pc.to(device)

    all_samples = get_all_sequences(seq_length, num_emits)

    entropy = 0.0
    for sid in range(0, all_samples.size(0), batch_size):
        eid = sid + batch_size
        if eid > all_samples.size(0):
            continue

        batch = all_samples[sid:eid,:].to(device)

        lls = pc(batch)
        entropy += -(lls.exp() * lls).sum().item()

    total_ll = 0.0
    count = 0
    for _ in range(200):
        samples = juice.queries.sample(pc, num_samples = batch_size)

        lls = pc(samples)
        total_ll += lls.sum().item()
        count += lls.size(0)

    sample_entropy = -total_ll / count

    assert abs(entropy - sample_entropy) < 0.01


if __name__ == "__main__":
    test_sample_hmm_correctness()
