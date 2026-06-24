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


def test_hmm_backward_small_batch():
    """
    Regression for the small-batch (`batch_size < 4`) sum-layer parameter-flow backward.

    A `batch_size < 4` routes the sum-layer backward to the *sparse* path
    (`SumLayer._backward_sparse_par_flows` -> `_bk_triton_sparse_par_kernel`). The kernel
    recovers the node block from the node-grid index via `nblock_id = pid_m // BLOCK_M`, so
    `BLOCK_M` must equal `block_size`. It used to be clamped to `min(2048 // num_edges,
    block_size)`, which is `< block_size` whenever `num_edges > 2048 / block_size` (here
    `block_size = num_edges = num_latents = 1024`). `nblock_id` then ran past `num_nblocks`,
    over-reading `nids` / `cids` -> a garbage child index -> an illegal `element_mars` access
    (a CUDA illegal-memory-access for `num_latents = 1024`; a silent flow corruption for
    smaller layers). The flows computed at `batch_size in {1, 2, 3}` (sparse path) must match
    the block-sparse path used at `batch_size >= 4`, and be invariant to the batch tiling.

    `force_use_fp32 = True` removes bf16 rounding so the cross-path comparison can be tight.
    """

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    seq_length = 4
    num_latents = 1024   # block_size = num_edges = 1024 -> BLOCK_M was clamped to 2 -> OOB
    num_cats = 6

    ns = juice.structures.GeneralizedHMM(
        seq_length = seq_length,
        num_latents = num_latents,
        homogeneous = True,
        input_dist = juice.distributions.Categorical(num_cats = num_cats)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    # A fixed pool of distinct samples; `n_pool` is divisible by every tested batch size.
    n_pool = 12
    data = torch.randint(0, num_cats, [n_pool, seq_length], device = device)

    def accumulate_param_flows(batch_size):
        # Sum the parameter flows over the whole pool, processed `batch_size` samples at a time.
        # `flows_memory = 0.0` zeros the accumulator on the first chunk, `1.0` accumulates after.
        for i, s in enumerate(range(0, n_pool, batch_size)):
            x = data[s:s + batch_size].contiguous()
            pc(x, force_use_fp32 = True)
            pc.backward(x, flows_memory = 0.0 if i == 0 else 1.0,
                        allow_modify_flows = False, force_use_fp32 = True)
        torch.cuda.synchronize()
        return pc.param_flows.clone()

    # Reference: batch_size = 6 (>= 4) uses the well-tested block-sparse backward path.
    ref = accumulate_param_flows(6)
    assert torch.isfinite(ref).all() and ref.abs().sum() > 0

    sparse_flows = {}
    for batch_size in [1, 2, 3]:
        # Each of these crashes (illegal memory access) before the fix.
        got = accumulate_param_flows(batch_size)
        sparse_flows[batch_size] = got
        assert torch.isfinite(got).all(), f"non-finite parameter flows at batch_size={batch_size}"
        # The sparse path must agree with the (trusted) block-sparse path.
        rel = (got - ref).abs().max() / (ref.abs().max() + 1e-12)
        assert rel < 2e-2, f"parameter flows at batch_size={batch_size} differ from the reference (relmax={rel})"

    # The sparse path must also be invariant to the batch tiling (BLOCK_M is batch-independent,
    # so the buggy index mapping would corrupt every tiling identically -- this catches a future
    # regression that crashes only for some layer shapes).
    for batch_size in [2, 3]:
        rel = (sparse_flows[batch_size] - sparse_flows[1]).abs().max() / (sparse_flows[1].abs().max() + 1e-12)
        assert rel < 1e-4, f"sparse parameter flows not tiling-invariant (batch {batch_size} vs 1, relmax={rel})"


if __name__ == "__main__":
    test_hmm_batch_size_consistency()
    test_hmm_backward_small_batch()
