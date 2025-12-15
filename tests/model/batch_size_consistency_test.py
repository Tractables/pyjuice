import pyjuice as juice
import torch
import random
import numpy as np

import pytest


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


def test_hmm_batch_size_consistency():

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

    ns = juice.structures.GeneralizedHMM(
        seq_length = seq_length, 
        num_latents = num_latents,
        homogeneous = True,
        input_dist = juice.distributions.ExternProductCategorical(num_cats = num_emits),
        alpha = alpha,
        beta = beta,
        gamma = gamma
    )
    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_emits, [1, seq_length]).to(device)

    external_categorical_logps = torch.rand([1, seq_length, num_emits], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    external_categorical_value_mask = torch.zeros([1, seq_length], dtype = torch.bool, device = device)
    external_categorical_value_mask[:,:3] = True

    ref_ll = None

    for batch_size in [1, 2, 4, 8, 16]:

        curr_data = data.repeat(batch_size, 1).contiguous()
        curr_external_categorical_logps = external_categorical_logps.repeat(batch_size, 1, 1).contiguous()
        curr_external_categorical_value_mask = external_categorical_value_mask.repeat(batch_size, 1).contiguous()

        lls = pc(curr_data, external_categorical_logps = curr_external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll",
                external_categorical_value_mask = curr_external_categorical_value_mask)

        if ref_ll is None:
            ref_ll = lls.mean().item()
        else:
            assert torch.all((lls - ref_ll).abs() < 1e-2)


if __name__ == "__main__":
    test_hmm_batch_size_consistency()
