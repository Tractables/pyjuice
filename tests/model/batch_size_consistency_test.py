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
    `block_size >= _SMALL_BATCH_MIN_BLOCK_SIZE` (=32 after re-profiling) are therefore routed to the
    block-sparse forward / element-flow backward at small batch too (with a small-batch tiling
    heuristic that splits the node dimension for SM occupancy); the parameter-flow backward falls
    back to the sparse kernel (correct for any batch). The results must match the sparse path (forced
    here as the reference), for both the forward LL and the accumulated parameter flows. The 32/64
    cases pin the lowered crossover (these were on the sparse path before).
    """

    device = torch.device("cuda:0")

    for num_latents in [32, 64, 128, 512]:   # block_size = num_latents >= 32 -> small-batch block-sparse path
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


def test_small_batch_par_backward_cuda_matches_triton():
    """
    The optional small-batch (batch < 16) CUDA parameter-flow backward kernel (a plain-CUDA node-warp
    with edges split across the grid -- coalesced node-contiguous param load/store, collision-free
    read-add-store, no atomics) must produce the same parameter flows as the Triton sparse par kernel
    it is autotuned against. Skipped if the CUDA kernel can't be built (no nvcc/ninja).
    """
    import pyjuice.layer.sum_layer as sl
    from pyjuice.layer.kernels import c as cuda_kernels

    if not (torch.cuda.is_available() and cuda_kernels.smallbatch_par_is_available()):
        pytest.skip("small-batch CUDA par backward kernel unavailable (no nvcc/ninja); Triton fallback used")

    device = torch.device("cuda:0")
    torch.manual_seed(606)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 512, homogeneous = True,   # block_size 512 (mult of 32) -> CUDA path
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    saved = sl.BACKWARD_PAR_FLOW_CUDA
    try:
        for batch_size in [1, 2, 3, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            sl.BACKWARD_PAR_FLOW_CUDA = True
            pc(data)
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_cuda = pc.param_flows.clone()

            sl.BACKWARD_PAR_FLOW_CUDA = False
            pc(data)
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_triton = pc.param_flows.clone()

            assert torch.isfinite(pf_cuda).all()
            assert (pf_cuda - pf_triton).abs().max() / (pf_triton.abs().max() + 1e-12) < 1e-3, \
                f"small-batch CUDA par backward mismatch vs Triton at batch={batch_size}"

        # Confirm the small-batch CUDA par dispatch was actually reached (a choice was cached).
        reached = any(
            len(getattr(layer, "_cached_bk_par_sparse_choice", {})) > 0
            for lg in pc.inner_layer_groups for layer in lg
            if type(layer).__name__ == "SumLayer"
        )
        assert reached, "small-batch CUDA par dispatch was never reached (check the gate conditions)"
    finally:
        sl.BACKWARD_PAR_FLOW_CUDA = saved


def _clear_sum_caches(pc):
    _names = ["_cached_fw_pcids", "_cached_fw_cuda", "_cached_fw_cuda_choice", "_cached_fw_sb",
              "_cached_bk_parids", "_cached_bk_ele_cuda", "_cached_bk_ele_choice", "_cached_bk_ele_sb",
              "_cached_bk_par_cuda", "_cached_bk_par_choice", "_cached_bk_par_sparse_choice",
              "_cached_bk_par_trim"]
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if type(layer).__name__ == "SumLayer":
                for nm in _names:
                    if hasattr(layer, nm):
                        getattr(layer, nm).clear()


def _trim_fires(pc):
    # the edge trim fires iff some partition's REAL max edge count (the last non-dummy cids column;
    # padding is a contiguous zero suffix, real children point to elements >= 1) is below the
    # pow2-padded width.
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if type(layer).__name__ == "SumLayer":
                for cids in layer.partitioned_cids:
                    if int((cids != 0).any(dim = 0).sum()) < int(cids.size(1)):
                        return True
    return False


def test_edge_trim_block_sparse_bit_identical():
    """
    The block-sparse edge-tile trim (`_BLOCK_SPARSE_EDGE_TRIM`) drops the fully-padded edge tiles that
    the pow2 padding adds. On the Triton path with a small block size (no bf16 tensor-core dot) it must
    be exactly bit-identical to the untrimmed result -- forward LL AND accumulated parameter flows.
    HCLT (block_size 32, fan-in 160 -> padded to 256) genuinely triggers the trim (asserted).

    Launch tuning is OFF for the duration: trimmed and untrimmed launches have different shape keys, so
    each is tuned on its own, and tile candidates agree only to reduction order (~1e-7, see
    `autotune.pick`). Bit-identity is a property of the trim at a FIXED configuration -- MEASURED: with
    tuning on, one run in five differed at batch 1; with it off, 6 of 6 identical.
    """
    import pyjuice.layer.sum_layer as sl
    from pyjuice.layer.kernels import autotune

    device = torch.device("cuda:0")
    torch.manual_seed(160)
    xs = torch.randint(0, 256, [400, 16]).float()
    ns = juice.structures.HCLT(xs, num_latents = 160)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    assert _trim_fires(pc), "edge trim did not fire on HCLT-160 (the test would be vacuous)"

    saved = (sl.FORWARD_SUM_CUDA, sl.BACKWARD_ELE_FLOW_CUDA, sl.BACKWARD_PAR_FLOW_CUDA, sl._BLOCK_SPARSE_EDGE_TRIM)
    saved_tune = autotune.ENABLED
    try:
        # CUDA off -> the exact Triton path (the CUDA kernels are only numerically equivalent).
        # batch 64 (>= _GAP_BATCH_MAX) exercises the BACKWARD_PAR_FLOW_TUNED path, which doubles
        # TILE_SIZE_K -- so the trim is checked against a changed tile size too.
        sl.FORWARD_SUM_CUDA = sl.BACKWARD_ELE_FLOW_CUDA = sl.BACKWARD_PAR_FLOW_CUDA = False
        autotune.ENABLED = False
        for batch_size in [1, 2, 8, 16, 64]:
            data = torch.randint(0, 256, [batch_size, 16], device = device)

            sl._BLOCK_SPARSE_EDGE_TRIM = True
            _clear_sum_caches(pc)
            ll_trim = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_trim = pc.param_flows.clone()

            sl._BLOCK_SPARSE_EDGE_TRIM = False
            _clear_sum_caches(pc)
            ll_full = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_full = pc.param_flows.clone()

            assert torch.equal(ll_trim, ll_full), f"edge-trim forward LL not bit-identical at batch={batch_size}"
            assert torch.equal(pf_trim, pf_full), f"edge-trim param flows not bit-identical at batch={batch_size}"
    finally:
        sl.FORWARD_SUM_CUDA, sl.BACKWARD_ELE_FLOW_CUDA, sl.BACKWARD_PAR_FLOW_CUDA, sl._BLOCK_SPARSE_EDGE_TRIM = saved
        autotune.ENABLED = saved_tune


def test_edge_trim_cuda_matches_sparse():
    """
    Guards the trim's interaction with the small-batch CUDA fast path: that kernel iterates
    `child = sb_ebase + edge` over `edge in [0, num_edges)` assuming global contiguity, so if the trim
    shrank the tile count without ALSO shrinking `num_edges` it would read PAST the real children (a
    silent OOB). On a globally-contiguous, non-pow2-fan-in model (HMM, block_size 128, fan-in 640 ->
    padded to 1024) the trimmed CUDA result must still match the exact sparse reference within the CUDA
    kernels' tolerance. Asserts both the trim fires AND the small-batch CUDA path is actually reached.
    Skipped if the CUDA kernels can't be built.
    """
    import pyjuice.layer.sum_layer as sl
    from pyjuice.layer.kernels import c as cuda_kernels

    if not (torch.cuda.is_available() and cuda_kernels.smallbatch_fw_is_available()):
        pytest.skip("small-batch CUDA kernels unavailable (no nvcc/ninja)")

    device = torch.device("cuda:0")
    torch.manual_seed(640)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 4, num_latents = 640, homogeneous = True,
        input_dist = juice.distributions.Categorical(num_cats = 6)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    assert _trim_fires(pc), "edge trim did not fire on HMM-640 (the test would be vacuous)"

    sums = [L for lg in pc.inner_layer_groups for L in lg if type(L).__name__ == "SumLayer"]
    saved = (sl.FORWARD_SUM_CUDA, sl.BACKWARD_ELE_FLOW_CUDA, sl.BACKWARD_PAR_FLOW_CUDA, sl._BLOCK_SPARSE_EDGE_TRIM)
    try:
        sl._BLOCK_SPARSE_EDGE_TRIM = True
        sb_cuda_reached = False
        for batch_size in [1, 2, 8]:
            data = torch.randint(0, 6, [batch_size, 4], device = device)

            # exact reference: the sparse kernels (no trim, no CUDA, no bf16 tensor-core dot)
            sl.FORWARD_SUM_CUDA = sl.BACKWARD_ELE_FLOW_CUDA = sl.BACKWARD_PAR_FLOW_CUDA = False
            _clear_sum_caches(pc)
            ll_ref = pc(data, mode = "sparse").clone()
            pc.backward(data, mode = "sparse", flows_memory = 0.0, allow_modify_flows = False)
            pf_ref = pc.param_flows.clone()

            # under test: the trimmed CUDA fast paths
            sl.FORWARD_SUM_CUDA = sl.BACKWARD_ELE_FLOW_CUDA = sl.BACKWARD_PAR_FLOW_CUDA = True
            _clear_sum_caches(pc)
            ll_cuda = pc(data).clone()
            pc.backward(data, flows_memory = 0.0, allow_modify_flows = False)
            pf_cuda = pc.param_flows.clone()
            sb_cuda_reached = sb_cuda_reached or any(v[2] for L in sums for v in L._cached_fw_sb.values())

            assert torch.isfinite(ll_cuda).all() and torch.isfinite(pf_cuda).all()
            assert (ll_cuda - ll_ref).abs().max() < 1e-2, \
                f"trimmed CUDA forward diverged from sparse at batch={batch_size}"
            assert (pf_cuda - pf_ref).abs().max() / (pf_ref.abs().max() + 1e-9) < 1e-2, \
                f"trimmed CUDA param flows diverged from sparse at batch={batch_size}"

        assert sb_cuda_reached, "small-batch CUDA path never reached (test does not guard the num_edges fix)"
    finally:
        sl.FORWARD_SUM_CUDA, sl.BACKWARD_ELE_FLOW_CUDA, sl.BACKWARD_PAR_FLOW_CUDA, sl._BLOCK_SPARSE_EDGE_TRIM = saved


def _partial_tile_pc(device, block_size = 32, num_node_blocks = 4, num_cats = 8):
    """A PC whose inner sum layer has `block_size = 32` / `num_edges = 128`, which makes the
    block-sparse parameter-flow launcher pick `TILE_SIZE_B = 64`. Returns `(pc, ns, layer)`."""
    from pyjuice.nodes import inputs, multiply, summate
    import pyjuice.nodes.distributions as dists

    torch.manual_seed(0)
    with juice.set_block_size(block_size):
        ins = [inputs(v, num_node_blocks = num_node_blocks,
                      dist = dists.Categorical(num_cats = num_cats)) for v in range(4)]
        s0 = summate(multiply(ins[0], ins[1]), num_node_blocks = num_node_blocks)
        s1 = summate(multiply(ins[2], ins[3]), num_node_blocks = num_node_blocks)
        ns = summate(multiply(s0, s1), num_node_blocks = num_node_blocks)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(device)

    layer = None
    for g in pc.inner_layer_groups:
        for l in g.layers:
            if hasattr(l, "partitioned_pfids") and getattr(l, "nodes", None) and ns in l.nodes:
                layer = l
    assert layer is not None
    return pc, ns, layer


def _node_flow_conservation(pc, ns, layer):
    """`max_n | sum_c F+[n,c] / sum_b f_b[n] - 1 |`, summed over every partition.

    A sum node's posterior flows split its own flow across its children exactly
    (`sum_c P(c|n,b) == 1`), so this is 0 up to floating point -- and it needs no reference
    implementation, which is what makes it a usable regression bar.

    Reads `node_flows` as the true log flow, which requires `allow_modify_flows = False`
    (otherwise the backward overwrites it in place with the `log f - log m` form).
    """
    bs = ns.block_size
    ar = torch.arange(bs, device = pc.param_flows.device)
    parts = range(len(layer.partitioned_pfids))
    gmin = min(int(layer.partitioned_nids[p].min()) for p in parts)
    n_nodes = ns.num_node_blocks * bs
    got = torch.zeros(n_nodes, device = pc.param_flows.device, dtype = torch.float64)
    want = torch.zeros_like(got)
    for p in parts:
        pfids, cids = layer.partitioned_pfids[p], layer.partitioned_cids[p]
        nids = layer.partitioned_nids[p].long()
        rows, E = pfids.shape
        idx = (pfids[:, None, :] + ar[None, :, None]).long()
        real = (cids != 0)[:, None, :].expand(rows, bs, E)
        loc = (nids[:, None] + ar[None, :] - gmin).reshape(-1)
        got.index_add_(0, loc, (pc.param_flows[idx].double() * real).sum(-1).reshape(-1))
        gid = (nids[:, None] + ar[None, :]).reshape(-1)
        want.index_add_(0, loc, pc.node_flows[gid].double().exp().sum(-1))
    keep = want.abs() > 1e-8
    assert bool(keep.any())
    return ((got - want).abs()[keep] / want.abs()[keep]).max().item()


def test_param_flows_include_the_final_partial_batch_tile():
    """
    Regression: the block-sparse parameter-flow backward silently DROPPED the trailing
    `batch_size % TILE_SIZE_B` samples.

    `SumLayer._backward_block_sparse_par_flows` contracted the batch into
    `B_NUM_TILES = batch_size // TILE_SIZE_B` tiles -- floor, not ceil -- so the kernel's batch
    loop never visited the final partial tile and those samples contributed no parameter flow at
    all. MEASURED on this layer (`TILE_SIZE_B = 64`): batch 100 lost 36% of the flow, batch 300
    lost 15%, batch 1000 lost 4%; on a `block_size = 8` layer batches of 33/40/48 lost ALL of it
    (floor gave zero tiles). The dispatch site only special-cased `batch < 16`, so every larger
    non-multiple stayed broken -- including the final partial batch of an ordinary epoch.

    It survived because it is invisible wherever `batch_size % TILE_SIZE_B == 0`: `TILE_SIZE_B`
    is a power of two, benchmark batch sizes are powers of two, and every batch size in the other
    tests here (1, 2, 3, 4, 6, 8, 16) is SMALLER than `TILE_SIZE_B`, i.e. one fully masked tile,
    which was always correct. So the batch sizes below are deliberately chosen to be multiples of
    NO plausible tile size (100, 300, 1000 are not multiples of 16, 32 or 64).

    Checked with the reference-free conservation identity rather than against another batch size,
    so the test cannot be satisfied by two paths being wrong in the same way.
    """
    device = torch.device("cuda:0")
    pc, ns, layer = _partial_tile_pc(device)

    torch.manual_seed(3)
    worst = {}
    for batch_size in [64, 100, 128, 300, 512, 1000]:
        x = torch.randint(0, 8, [batch_size, 4], device = device)
        pc(x)
        pc.backward(x, logspace_flows = True, flows_memory = 0.0, allow_modify_flows = False)
        worst[batch_size] = _node_flow_conservation(pc, ns, layer)

    # The fp32 accumulation floor on this layer is ~1e-3; the bug was 4%-36%, so 5e-3 separates
    # them cleanly without being sensitive to the (TF32) dot's precision.
    for batch_size, rel in worst.items():
        assert rel < 5e-3, \
            f"parameter flows do not conserve node flow at batch_size={batch_size} " \
            f"(relmax={rel:.4f}); the final partial batch tile is being dropped. All: {worst}"


def test_param_flows_invariant_to_batch_chunking():
    """
    The same pool of samples must give the same total parameter flows however it is CHUNKED --
    the user-facing form of the partial-tile bug above, and the thing that silently biased EM
    whenever the dataset size was not a multiple of the batch size.

    Every chunk size divides the pool, so the accumulated totals are comparable. Chunk sizes
    96/120/160/480 are NOT multiples of `TILE_SIZE_B = 64` and each dropped a different number of
    trailing samples before the fix, so they disagreed with each other and with the aligned 64.

    Compared with a tolerance, NOT bit-identity: the parameter-flow kernels accumulate through
    `tl.atomic_add`, whose ordering varies between runs, so this layer is not bitwise reproducible
    at larger batch (MEASURED: identical source, different sha1 at batch 256 and 512).
    """
    device = torch.device("cuda:0")
    pc, ns, layer = _partial_tile_pc(device)

    n_pool = 960
    torch.manual_seed(5)
    data = torch.randint(0, 8, [n_pool, 4], device = device)

    def accumulate(chunk):
        for i, s in enumerate(range(0, n_pool, chunk)):
            x = data[s:s + chunk].contiguous()
            pc(x)
            pc.backward(x, logspace_flows = True, allow_modify_flows = False,
                        flows_memory = 0.0 if i == 0 else 1.0)
        torch.cuda.synchronize()
        return pc.param_flows.clone()

    ref = accumulate(64)                      # 960 = 15 * 64, an exact number of tiles
    assert torch.isfinite(ref).all() and ref.abs().sum() > 0

    # PER-ELEMENT relative error, over the entries that carry real mass. Normalising by the GLOBAL
    # max instead (`(got-ref).max() / ref.max()`) hides the defect: the root layer's flows are ~1e3
    # while this layer's are ~1, so dropping half of a sum layer's flow showed up as 9e-3 and only
    # just cleared the bar. Elementwise, the same run reads ~0.4.
    def relerr(got):
        keep = ref.abs() > 1e-6 * ref.abs().max()
        return float(((got - ref).abs()[keep] / ref.abs()[keep]).max())

    # 120 and 240 are the discriminating chunk sizes: the large-batch launch tuning resets
    # `TILE_SIZE_B` to 32 whenever `batch_size % 32 == 0`, which accidentally re-aligns 96/160/480/960,
    # so those cannot see the bug. 120 % 64 = 56 and 240 % 64 = 48 keep a genuinely partial tile.
    for chunk in [96, 120, 160, 240, 480, 960]:
        assert n_pool % chunk == 0
        got = accumulate(chunk)
        assert torch.isfinite(got).all(), f"non-finite parameter flows at chunk={chunk}"
        rel = relerr(got)
        assert rel < 5e-3, \
            f"total parameter flows depend on the batch chunking (chunk={chunk} vs 64, relmax={rel:.4f})"


def _deep_hmm(device):
    torch.manual_seed(11)
    ns = juice.structures.GeneralizedHMM(
        seq_length = 32, num_latents = 256, homogeneous = True,   # 32 sum layers, block_size 256
        input_dist = juice.distributions.Categorical(num_cats = 16)
    )
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)
    return pc


@pytest.mark.parametrize("batch_size", [12, 64])
def test_flows_do_not_leak_with_depth(batch_size):
    """
    Regression: the sum layers' flow propagation was biased LOW by ~6e-4 per layer, so a deep
    circuit lost flow geometrically with depth -- 1% of the total at 32 layers, and an HMM's
    log-likelihood depended on the batch it was evaluated in (0.018 nats at 32 layers, batch 12).

    Cause: TF32 truncation. A TF32 `tl.dot` hands fp32 registers to the tensor core, which ignores
    the low 13 mantissa bits -- truncation, so for the all-positive operands here a bias, not noise.
    Two sites: the element-flow kernel's explicit `tl.dot` (batch >= 16), and -- in the paths written
    as fp32 (`csmm1` forward and the element-flow `TL_DOT = 0` branch, batch 9..15) -- Triton
    rewriting `tl.sum(a[:,:,None] * b[None,:,:], axis = 1)` into a TF32 dot by itself.

    Why the other tests here missed it: they use 4-layer HMMs (4 x 6e-4 is under their 1e-3 bar)
    and batches <= 8, which take the exact `csmm2` kernels.

    The instrument is pinned, because both of its free choices are made by TIMING: at batch < 16 the
    exact CUDA small-batch kernels may win the fork, and the launch-config tuner picks the tile shape.
    The tile matters: at batch 12 only the tuner's 16-row candidate is big enough for Triton's TF32
    rewrite -- the heuristic 8-row tile is exact -- and it is also the faster one, so the real tuner
    usually takes it. So the Triton paths are forced and the tuner always takes the LARGEST candidate.

    Reference-free: in an HMM each sum layer passes exactly one unit of flow per sample, so the
    total parameter flow of a one-shot backward must equal that of `batch_size` batch-1 backwards.
    """
    import pyjuice.layer.sum_layer as sl
    from pyjuice.layer.kernels import autotune

    device = torch.device("cuda:0")
    pc = _deep_hmm(device)
    torch.manual_seed(batch_size)
    x = torch.randint(0, 16, [batch_size, 32], device = device)

    saved = (sl.FORWARD_SUM_CUDA, sl.BACKWARD_ELE_FLOW_CUDA, autotune.ENABLED, autotune.best_of,
             dict(autotune._CACHE))
    try:
        sl.FORWARD_SUM_CUDA = sl.BACKWARD_ELE_FLOW_CUDA = False
        autotune.ENABLED = True
        autotune._CACHE.clear()
        autotune.best_of = lambda candidates, *a, **k: max(c for c, _ in candidates)

        def total(chunk):
            lls = []
            for i, s in enumerate(range(0, batch_size, chunk)):
                xs = x[s:s + chunk].contiguous()
                lls.append(pc(xs).view(-1).clone())
                pc.backward(xs, logspace_flows = True, allow_modify_flows = False,
                            flows_memory = 0.0 if i == 0 else 1.0)
            torch.cuda.synchronize()
            return pc.param_flows.double().sum().item(), torch.cat(lls).double()

        ref_flow, ref_ll = total(1)
        got_flow, got_ll = total(batch_size)
    finally:
        sl.FORWARD_SUM_CUDA, sl.BACKWARD_ELE_FLOW_CUDA, autotune.ENABLED, autotune.best_of, cache = saved
        autotune._CACHE.clear()
        autotune._CACHE.update(cache)

    # 32 sum layers x 1 unit per sample
    assert abs(ref_flow / batch_size - 32.0) < 1e-3, f"batch-1 reference itself is off: {ref_flow / batch_size}"

    rel = abs(got_flow / ref_flow - 1.0)
    assert rel < 3e-3, f"sum-layer flow leaks with depth at batch_size={batch_size}: {got_flow / batch_size:.5f} of 32 units/sample"

    # Only where the forward is meant to be fp32. From 16 up it is a bf16 dot BY DESIGN (rounded to
    # nearest, so unbiased), which on its own puts ~1.7e-3 between batch 64 and batch 1 here.
    if batch_size < 16:
        dll = (got_ll - ref_ll).abs().max().item()
        assert dll < 2e-3, f"log-likelihood depends on the batch size at batch_size={batch_size}: max |dLL| = {dll:.5f}"


def _sweep_circuit(kind, device):
    """Small circuits that each put a different kernel regime at a dispatch boundary."""
    import pyjuice.nodes.distributions as dists
    from pyjuice.nodes import inputs, multiply, summate

    torch.manual_seed(7)
    evidence = None
    if kind in ("hmm_untied", "hmm_tied"):           # block_size 64: the small-batch (< 16) kernels
        root = juice.structures.GeneralizedHMM(seq_length = 8, num_latents = 64, homogeneous = (kind == "hmm_tied"),
                                               input_dist = dists.Categorical(num_cats = 16))
        num_vars, num_cats = 8, 16
    elif kind == "few_children":                     # block-16 sums over 4 children: the forward reduces over 4
        with juice.set_block_size(1):
            np0 = multiply(inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = 6)),
                           inputs(1, num_node_blocks = 4, dist = dists.Categorical(num_cats = 6)))
        ns = summate(np0, num_node_blocks = 1, block_size = 16)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
        num_vars, num_cats = 2, 6
    elif kind == "softevi_few_cats":                 # a soft-evidence leaf over 4 categories, dense evidence
        root = juice.structures.GeneralizedHMM(seq_length = 4, num_latents = 32, homogeneous = True,
                                               input_dist = dists.SoftEvidenceCategorical(num_cats = 4, _dual_flow_backward = False))
        num_vars, num_cats = 4, 4
        evidence = True
    else:
        raise ValueError(kind)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(device)
    return pc, num_vars, num_cats, evidence


@pytest.mark.parametrize("tuned", [True, False])
@pytest.mark.parametrize("kind", ["hmm_untied", "hmm_tied", "few_children", "softevi_few_cats"])
def test_results_do_not_depend_on_the_batch_size(kind, tuned):
    """
    A sample's log-likelihood, and the parameter flows of a set of samples, cannot depend on how many
    samples share the call. Checked at every dispatch boundary (powers of two +-1, the small-batch
    thresholds) against a reference built from batch-1 calls, which take the sparse kernels.

    Every bug of this family found so far was invisible except at particular batch sizes: the dropped
    partial batch tile (any batch not a multiple of `TILE_SIZE_B`), TF32 flows leaking with depth (only
    from batch 9 up), and the fp32 broadcast-sum miscompile -- ln 2 per layer with fewer than 8 children
    and per soft-evidence variable with fewer than 8 categories at batch >= 16, double flows at batch 4.

    Two bars, for two kinds of bug: the TOTAL parameter flow (bias, which does not average out) to 1e-3,
    and every element to 5e-3 (a misrouted or miscounted element). Batch >= 16 runs the forward in bf16
    by design, hence the 3e-3 on the log-likelihood.

    Run with the launch tuner on (what users get) AND off (the heuristic tiles, `PYJUICE_AUTOTUNE=0`):
    whether a tile-shape-dependent bug shows can depend on which tile the timing picks. MEASURED: with
    the batch-4 double count re-introduced, the tuned HMM runs passed in one process and failed in the
    next, while the heuristic tiles failed every time.
    """
    from pyjuice.layer.kernels import autotune

    device = torch.device("cuda:0")
    saved = (autotune.ENABLED, dict(autotune._CACHE))
    autotune.ENABLED = tuned
    autotune._CACHE.clear()
    try:
        _check_batch_size_invariance(kind, device)
    finally:
        autotune.ENABLED = saved[0]
        autotune._CACHE.clear()
        autotune._CACHE.update(saved[1])


def _check_batch_size_invariance(kind, device):
    pc, num_vars, num_cats, evidence = _sweep_circuit(kind, device)

    sizes = [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 32, 33, 64, 65]
    n_max = max(sizes)
    torch.manual_seed(11)
    x = torch.randint(0, num_cats, (n_max, num_vars), device = device)
    ev = torch.log_softmax(torch.randn(n_max, num_vars, num_cats, device = device), dim = 2) if evidence else None

    def call(sl, first):
        kw = {} if ev is None else dict(categorical_evidence_logp = ev[sl].contiguous())
        lls = pc(x[sl].contiguous(), **kw).view(-1).clone()
        pc.backward(x[sl].contiguous(), logspace_flows = True, allow_modify_flows = False,
                    flows_memory = 0.0 if first else 1.0, **kw)
        return lls

    # Batch-1 reference, accumulated sample by sample; snapshot the flows at every size in the sweep.
    ref_lls, ref_flows = [], {}
    for i in range(n_max):
        ref_lls.append(call(slice(i, i + 1), i == 0))
        if i + 1 in sizes:
            torch.cuda.synchronize()
            ref_flows[i + 1] = pc.param_flows.double().clone()
    ref_lls = torch.cat(ref_lls).double()

    failures = []
    for n in sizes:
        lls = call(slice(0, n), True).double()
        torch.cuda.synchronize()
        flows, ref = pc.param_flows.double(), ref_flows[n]
        dll = (lls - ref_lls[:n]).abs().max().item()
        total = abs(flows.sum().item() / ref.sum().item() - 1.0)
        keep = ref.abs() > 1e-6 * ref.abs().max()
        elem = ((flows - ref).abs()[keep] / ref.abs()[keep]).max().item()
        if not (dll < 3e-3 and total < 1e-3 and elem < 5e-3):
            failures.append(f"batch {n}: max |dLL| {dll:.2e}, total flow off by {total:.2e}, worst element {elem:.2e}")
    assert not failures, f"{kind}: results depend on the batch size\n" + "\n".join(failures)


if __name__ == "__main__":
    test_hmm_batch_size_consistency()
    test_hmm_backward_small_batch()
    test_param_flows_include_the_final_partial_batch_tile()
    test_param_flows_invariant_to_batch_chunking()
    test_sum_layer_backward_mode_and_fp32()
    test_small_batch_block_sparse_fast_path()
    test_small_batch_forward_cuda_matches_triton()
    test_small_batch_ele_backward_cuda_matches_triton()
    test_small_batch_prod_tiling_matches_untiled()
    test_small_batch_sparse_ele_tiling_matches_untiled()
    test_gap_batch_tiling_matches_untiled()
    test_small_batch_par_backward_cuda_matches_triton()
