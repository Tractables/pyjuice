import itertools

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import prod as prod_kernels


# The product layer's autotuner picks the 2-d kernel's node tile `BLOCK_M` (and its batch tile) by timing, so any
# candidate may run on some GPU. A program walks its `BLOCK_M` nodes one after another and reduces each node's own
# edges, so the tile only groups nodes into programs: every candidate must give the default tile's bits, node tiles
# down to 1 included (offered for launches too small to hide their latency).


@pytest.mark.parametrize("block_size, num_edges", [(32, 1), (32, 2), (32, 8), (1024, 1), (1024, 2), (1024, 4)])
def test_every_node_tile_gives_the_default_tiles_bits(block_size, num_edges):
    device = torch.device("cuda:0")
    torch.manual_seed(block_size + num_edges)
    for batch_size, accum, lse in itertools.product((1, 7, 128, 300), (False, True), (False, True)):
        if lse and not accum:
            continue
        nblocks = 3 if block_size == 1024 else 40
        n_eles = nblocks * block_size * num_edges + 64
        element_vals = torch.randn(n_eles, batch_size, device = device) * 3
        element_vals[torch.rand_like(element_vals) < 0.05] = -float("inf")
        cids = torch.stack([torch.randperm(n_eles // block_size - 1, device = device)[:num_edges] * block_size
                            for _ in range(nblocks)])
        nids = torch.arange(nblocks, device = device) * block_size + 16
        n_nodes = nblocks * block_size + 32
        node_vals = torch.randn(n_nodes, batch_size, device = device)
        BLOCK_B = min(2048 // num_edges, 1 << (batch_size - 1).bit_length())

        def run(bm):
            out = node_vals.clone()
            prod_kernels._forward_backward_kernel_2d[(nblocks * block_size // bm, (batch_size + BLOCK_B - 1) // BLOCK_B)](
                node_vals_ptr = out, element_vals_ptr = element_vals, nids_ptr = nids, cids_ptr = cids,
                tot_n_nodes = n_nodes, tot_n_eles = n_eles, n_nblocks = nblocks, num_edges = num_edges,
                batch_size = batch_size, BLOCK_M = bm, BLOCK_B = BLOCK_B, block_size = block_size,
                accum = accum, prop_logsumexp = lse)
            return out

        ref = run(8)
        assert torch.isfinite(ref).any()
        for bm in (1, 2, 4, 16, 32):
            out = run(bm)
            assert torch.equal(torch.isnan(out), torch.isnan(ref)) and torch.equal(out.nan_to_num(), ref.nan_to_num()), \
                f"BLOCK_M {bm} vs 8: block {block_size}, edges {num_edges}, batch {batch_size}, accum {accum}, lse {lse}"


@pytest.mark.parametrize("batch_size", [16, 128])
def test_small_node_tiles_are_offered_and_compute_the_same_circuit(batch_size, monkeypatch):
    device = torch.device("cuda:0")
    forced = [0]
    offered = []
    real_pick = autotune.pick

    def pick(key, candidates, bench, *args, **kwargs):
        # not cached: every call asks again, so each pass below takes its forced config (the other sites' choices
        # are cached by the first pass and stay the same)
        if key[0] is prod_kernels._forward_backward_kernel_2d:
            offered.append(list(candidates))
            return candidates[min(forced[0], len(candidates) - 1)]
        return real_pick(key, candidates, bench, *args, **kwargs)

    monkeypatch.setattr(autotune, "pick", pick)
    # (and never cached: the layer asks `pick` only on a cache miss, and an earlier test of this worker on the same
    # product shape left its choice there -- every pass then took it, nothing was offered)
    real_cached = autotune.cached
    monkeypatch.setattr(autotune, "cached", lambda key: None if isinstance(key, tuple) and key
                        and key[0] is prod_kernels._forward_backward_kernel_2d else real_cached(key))
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 8, num_latents = 256, num_emits = 16, homogeneous = True, block_size = 64)
    pc = juice.compile(ns, verbose = False).to(device)
    data = torch.randint(0, 16, (batch_size, pc.num_vars), device = device)

    results = []
    c = 0
    while c == 0 or c < max(len(cands) for cands in results[0][2]):     # every config any product layer offers
        forced[0] = c
        offered.clear()
        lls = pc(data)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True)
        results.append((lls.clone(), pc.node_flows.clone(), [list(o) for o in offered]))
        c += 1

    tiles = {cfg[0] for cands in results[0][2] for cfg in cands}
    assert {1, 2, 4} <= tiles, f"the small node tiles are not on offer: {sorted(tiles)}"
    ref_lls, ref_nf = results[0][0], results[0][1]
    for c, (lls, nf, _) in enumerate(results[1:], start = 1):
        assert torch.equal(lls, ref_lls), f"batch {batch_size}, product config {c}: log-likelihoods differ"
        assert torch.equal(nf.nan_to_num(), ref_nf.nan_to_num()), f"batch {batch_size}, product config {c}: node flows differ"
