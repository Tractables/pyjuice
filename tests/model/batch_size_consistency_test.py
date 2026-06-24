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


def test_sum_layer_backward_mode_and_fp32():
    """
    Regression for two sum-layer backward-dispatch fixes:

    1. The `mode=` override (forcing the sparse / block-sparse / pytorch backend) referenced a bare
       `STR2MODE` instead of `self.STR2MODE` -> NameError. `pc.forward` / `pc.backward` thread `mode=`
       down to `SumLayer._forward` / `_backward`, so forcing a backend must not raise.
    2. `force_use_fp32 = True` was silently dropped on the block-sparse *parameter*-flow backward
       (swallowed by `**kwargs`), while the element-flow backward honored it. It must now be accepted
       on the parameter-flow path and still produce correct (finite) parameter flows.
    """

    device = torch.device("cuda:0")
    torch.manual_seed(7)

    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 128, homogeneous = True,
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, 6, [16, 4], device = device)

    def param_flows(**kw):
        fwd_kw = {k: v for k, v in kw.items() if k in ("mode", "force_use_fp32")}
        pc(data, **fwd_kw)
        pc.backward(data, flows_memory = 0.0, allow_modify_flows = False, **kw)
        torch.cuda.synchronize()
        return pc.param_flows.clone()

    ref = param_flows()
    assert torch.isfinite(ref).all() and ref.abs().sum() > 0

    # (1) Forcing the sparse backend on both the forward and backward must not raise (the bare
    # `STR2MODE` reference did). `sparse` is valid for every layer shape, so it can be forced globally.
    got = param_flows(mode = "sparse")
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max() / (ref.abs().max() + 1e-12) < 2e-2

    # (2) `force_use_fp32` must be accepted on the parameter-flow path and stay correct.
    got = param_flows(force_use_fp32 = True)
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max() / (ref.abs().max() + 1e-12) < 2e-2


def test_small_batch_block_sparse_fast_path():
    """
    Regression for the small-batch (batch < 16) block-sparse fast path.

    For a large block size the sparse sum kernels leave the big node/edge dimensions un-tiled (one
    program per node block -> ~1 SM busy, >10x slower than the block-sparse kernels). Layers with
    `block_size >= 128` are therefore routed to the block-sparse forward / element-flow backward at
    small batch too (with a small-batch tiling heuristic that splits the node dimension for SM
    occupancy); the parameter-flow backward falls back to the sparse kernel (correct for any batch).
    The results must match the sparse path (forced here as the reference), for both the forward LL
    and the accumulated parameter flows.
    """

    device = torch.device("cuda:0")

    for num_latents in [128, 512]:   # block_size = num_latents >= 128 -> small-batch block-sparse path
        torch.manual_seed(num_latents)
        ns = juice.structures.GeneralizedHMM(
            seq_length = 4, num_latents = num_latents, homogeneous = True,
            input_dist = juice.distributions.Categorical(num_cats = 6)
        )
        ns.init_parameters(perturbation = 2.0)
        pc = juice.compile(ns)
        pc.to(device)

        for batch_size in [1, 2, 3, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            # Default routing -> small-batch block-sparse fast path.
            ll = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf = pc.param_flows.clone()

            # Reference: force the sparse kernels.
            ll_s = pc(data, mode = "sparse").clone()
            pc.backward(data, mode = "sparse", flows_memory = 0.0, allow_modify_flows = False)
            pf_s = pc.param_flows.clone()

            assert torch.isfinite(ll).all() and torch.isfinite(pf).all()
            assert (ll - ll_s).abs().max() < 1e-3, \
                f"forward LL mismatch (Nlat={num_latents}, batch={batch_size})"
            assert (pf - pf_s).abs().max() / (pf_s.abs().max() + 1e-12) < 1e-3, \
                f"parameter-flow mismatch (Nlat={num_latents}, batch={batch_size})"


def test_small_batch_forward_cuda_matches_triton():
    """
    The optional small-batch (batch < 16) CUDA forward kernel (block_size >= 128 large-block layers,
    a plain-CUDA 32-node-warp + edge-split online-logsumexp) must produce the same log-likelihoods as
    the Triton small-batch path it is autotuned against. Skipped if the CUDA kernel can't be built
    (no nvcc/ninja) -- then the dispatch transparently uses Triton anyway.
    """
    import pyjuice.layer.sum_layer as sl
    from pyjuice.layer.kernels import c as cuda_kernels

    if not (torch.cuda.is_available() and cuda_kernels.smallbatch_fw_is_available()):
        pytest.skip("small-batch CUDA forward kernel unavailable (no nvcc/ninja); Triton fallback used")

    device = torch.device("cuda:0")
    torch.manual_seed(123)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 512, homogeneous = True,   # block_size 512 >= 128 -> CUDA path
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    saved = sl.FORWARD_SUM_CUDA
    try:
        for batch_size in [1, 2, 3, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)
            sl.FORWARD_SUM_CUDA = True
            ll_cuda = pc(data).clone()
            sl.FORWARD_SUM_CUDA = False
            ll_triton = pc(data).clone()
            assert torch.isfinite(ll_cuda).all()
            assert (ll_cuda - ll_triton).abs().max() < 1e-3, \
                f"small-batch CUDA forward mismatch vs Triton at batch={batch_size}"
    finally:
        sl.FORWARD_SUM_CUDA = saved


def test_small_batch_ele_backward_cuda_matches_triton():
    """
    The optional small-batch (batch < 16) CUDA element-flow backward kernel (block_size >= 128 layers,
    a plain-CUDA warp-per-child fused online-logsumexp) must produce the same parameter flows as the
    Triton small-batch path (csmm2) it is autotuned against. Skipped if the CUDA kernel can't be built
    (no nvcc/ninja) -- then the dispatch transparently uses Triton anyway.
    """
    import pyjuice.layer.sum_layer as sl
    from pyjuice.layer.kernels import c as cuda_kernels

    if not (torch.cuda.is_available() and cuda_kernels.smallbatch_ele_is_available()):
        pytest.skip("small-batch CUDA ele backward kernel unavailable (no nvcc/ninja); Triton fallback used")

    device = torch.device("cuda:0")
    torch.manual_seed(321)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 512, homogeneous = True,   # block_size 512 >= 128 -> CUDA path
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    saved = sl.BACKWARD_ELE_FLOW_CUDA
    try:
        for batch_size in [1, 2, 3, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            sl.BACKWARD_ELE_FLOW_CUDA = True
            pc(data)
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_cuda = pc.param_flows.clone()

            sl.BACKWARD_ELE_FLOW_CUDA = False
            pc(data)
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_triton = pc.param_flows.clone()

            assert torch.isfinite(pf_cuda).all()
            assert (pf_cuda - pf_triton).abs().max() / (pf_triton.abs().max() + 1e-12) < 1e-3, \
                f"small-batch CUDA ele backward mismatch vs Triton at batch={batch_size}"

        # Confirm the small-batch CUDA ele dispatch was actually reached (an "sb" choice was cached),
        # so this test is not silently validating Triton-vs-Triton.
        reached = any(
            any(isinstance(k, tuple) and len(k) == 3 and k[2] == "sb"
                for k in getattr(layer, "_cached_bk_ele_choice", {}))
            for lg in pc.inner_layer_groups for layer in lg
            if type(layer).__name__ == "SumLayer"
        )
        assert reached, "small-batch CUDA ele dispatch was never reached (check the gate conditions)"
    finally:
        sl.BACKWARD_ELE_FLOW_CUDA = saved


def test_small_batch_prod_tiling_matches_untiled():
    """
    The small-batch (batch < 16) product-layer node-tile cap (`_SMALL_BATCH_PROD_TILE_M`) fans the
    2D prod kernel's node dimension across many programs (the default budget heuristic leaves one
    serial program per node-block -> ~1 SM busy at tiny batch). It is PURE TILING (the kernel walks
    BLOCK_M nodes serially), so the forward LL and parameter flows must be bit-identical to the
    un-capped path. ~22x faster prod kernel / ~1.9x faster small-batch fwd+bwd on an HMM.
    """
    import pyjuice.layer.prod_layer as pl

    device = torch.device("cuda:0")
    torch.manual_seed(202)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 512, homogeneous = True,
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    saved = pl._SMALL_BATCH_PROD_TILE_M
    try:
        for batch_size in [1, 2, 3, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            pl._SMALL_BATCH_PROD_TILE_M = 8           # default tiled path
            ll_tiled = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_tiled = pc.param_flows.clone()

            pl._SMALL_BATCH_PROD_TILE_M = 1 << 30     # effectively uncapped (old behavior)
            ll_ref = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_ref = pc.param_flows.clone()

            assert torch.equal(ll_tiled, ll_ref), f"prod-tiling forward LL not bit-identical at batch={batch_size}"
            assert torch.equal(pf_tiled, pf_ref), f"prod-tiling param flows not bit-identical at batch={batch_size}"
    finally:
        pl._SMALL_BATCH_PROD_TILE_M = saved


def test_small_batch_sparse_ele_tiling_matches_untiled():
    """
    The small-batch (batch < 16) node-tile cap (`_SMALL_BATCH_SPARSE_TILE_M`) for the SPARSE
    element-flow kernel splits each node-block across many programs (the sparse kernel otherwise sets
    BLOCK_M = cs_block_size -> one serial program per node-block, ~1 SM busy). This hits the layers
    that miss the block-sparse path (e.g. the HMM's block_size==1 passthrough layer). It is PURE
    TILING, so forward LL and parameter flows must be bit-identical to the un-tiled path. ~38x faster
    on the HMM's sparse-ele layer.
    """
    import pyjuice.layer.sum_layer as sl

    device = torch.device("cuda:0")
    torch.manual_seed(404)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 512, homogeneous = True,
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    saved = sl._SMALL_BATCH_SPARSE_TILE_M
    try:
        for batch_size in [1, 2, 3, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            sl._SMALL_BATCH_SPARSE_TILE_M = 8           # default tiled path
            ll_tiled = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_tiled = pc.param_flows.clone()

            sl._SMALL_BATCH_SPARSE_TILE_M = 1 << 30     # effectively uncapped (old behavior)
            ll_ref = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_ref = pc.param_flows.clone()

            assert torch.equal(ll_tiled, ll_ref), f"sparse-ele tiling forward LL not bit-identical at batch={batch_size}"
            assert torch.equal(pf_tiled, pf_ref), f"sparse-ele tiling param flows not bit-identical at batch={batch_size}"
    finally:
        sl._SMALL_BATCH_SPARSE_TILE_M = saved


def test_gap_batch_tiling_matches_untiled():
    """
    The "gap batch" regime (16 <= batch < 64) sits between the small-batch (<16) path and the
    >=64-aligned CUDA path; there the budget heuristics under-tile the product, sparse-ele AND
    block-sparse parameter-flow kernels (e.g. at batch=16 the par kernel got ~8 programs / 1 SM). The
    fixes extend the prod/sparse-ele node-tile caps through the gap and shrink the par kernel's
    output-column tile TILE_SIZE_K (bit-safe: TILE_SIZE_M -- the max-stabilization group -- is left
    unchanged). All are pure tiling, so forward LL and parameter flows must be bit-identical to the
    un-tiled path. ~40x faster par kernel / ~7x faster batch=16 fwd+bwd under CUDA graphs.
    """
    import pyjuice.layer.sum_layer as sl
    import pyjuice.layer.prod_layer as pl

    device = torch.device("cuda:0")
    torch.manual_seed(505)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 512, homogeneous = True,
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    saved = (pl._SMALL_BATCH_PROD_TILE_M, sl._SMALL_BATCH_SPARSE_TILE_M, sl._SMALL_BATCH_PAR_TILE_K)
    try:
        for batch_size in [16, 32]:   # the gap range (>=16, <64)
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            pl._SMALL_BATCH_PROD_TILE_M, sl._SMALL_BATCH_SPARSE_TILE_M, sl._SMALL_BATCH_PAR_TILE_K = 8, 8, 16
            ll_tiled = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_tiled = pc.param_flows.clone()

            pl._SMALL_BATCH_PROD_TILE_M, sl._SMALL_BATCH_SPARSE_TILE_M, sl._SMALL_BATCH_PAR_TILE_K = (1 << 30,) * 3
            ll_ref = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_ref = pc.param_flows.clone()

            assert torch.equal(ll_tiled, ll_ref), f"gap-batch tiling forward LL not bit-identical at batch={batch_size}"
            assert torch.equal(pf_tiled, pf_ref), f"gap-batch tiling param flows not bit-identical at batch={batch_size}"
    finally:
        pl._SMALL_BATCH_PROD_TILE_M, sl._SMALL_BATCH_SPARSE_TILE_M, sl._SMALL_BATCH_PAR_TILE_K = saved


if __name__ == "__main__":
    test_hmm_batch_size_consistency()
    test_hmm_backward_small_batch()
    test_sum_layer_backward_mode_and_fp32()
    test_small_batch_block_sparse_fast_path()
    test_small_batch_forward_cuda_matches_triton()
    test_small_batch_ele_backward_cuda_matches_triton()
    test_small_batch_prod_tiling_matches_untiled()
    test_small_batch_sparse_ele_tiling_matches_untiled()
    test_gap_batch_tiling_matches_untiled()
