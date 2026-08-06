"""
Ancestral sampling through a sum layer, under the SHARED parameters.

One lane per selected sample. Each lane locates its node in the compiled `nids` / `cids` / `pids`
layout, then walks the node's edges drawing a child by inverse CDF. Unconditionally the edge weights
are the shared parameters; conditionally they are `theta * exp(element_mars - node_mars)`, with
`node_mars` doubling as the shift that keeps the terms near 1 and their sum near exactly 1.

A sum layer whose parameters are modified per sample -- an
:class:`~pyjuice.layer.ExternalParamsSumLayer` that was given external tensors -- is NOT served here.
It is handed to its own parameterization, which owns its kernel exactly as it owns the forward and
the backward; see :func:`pyjuice.nodes.external_params.ExternalSumParams.sample_layer`.
"""

from __future__ import annotations

import random
import triton
import triton.language as tl


@triton.jit
def sample_sum_layer_kernel(nids, cids, pids, node_mars, element_mars, mparams, node_samples, element_samples,
                            ind_target, ind_n, ind_b, seed, block_size: tl.constexpr, batch_size: tl.constexpr,
                            num_edges: tl.constexpr, num_samples: tl.constexpr, num_nblocks: tl.constexpr, BLOCK_S: tl.constexpr,
                            BLOCK_M: tl.constexpr, M_NUM_BLKS: tl.constexpr, BLOCK_K: tl.constexpr, K_NUM_BLKS: tl.constexpr,
                            conditional: tl.constexpr, do_calibration: tl.constexpr):

    pid_s = tl.program_id(0) # ID of size-`BLOCK_S` batches

    # Sample offsets and mask
    offs_sample = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    mask_sample = offs_sample < num_samples

    # Load node and batch ids
    node_sample_id = tl.load(ind_n + offs_sample, mask = mask_sample, other = 0)
    batch_id = tl.load(ind_b + offs_sample, mask = mask_sample, other = 0)
    node_id = tl.load(node_samples + node_sample_id * batch_size + batch_id)

    # Locate node ids in `nids`
    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_S], dtype = tl.int64)
    for i in range(M_NUM_BLKS):
        mask_nids = offs_nids < num_nblocks

        ref_nid = tl.load(nids + offs_nids, mask = mask_nids, other = 0)
        # `& mask_nids` because a padded slot reads as node id 0, which any node in the first block
        # would otherwise match -- selecting a row past the end of `nids`, whose `cids` / `pids` are
        # another layer's. Reachable whenever `num_nblocks` does not fill `BLOCK_M`.
        is_match = (node_id[:,None] >= ref_nid[None,:]) & \
                   (node_id[:,None] < ref_nid[None,:] + block_size) & mask_nids[None,:] # [BLOCK_S, BLOCK_M]

        match_local_id = tl.sum(is_match * (offs_nids[None,:] + 1), axis = 1)
        match_local_offset = tl.sum(is_match * (node_id[:,None] - ref_nid[None,:]), axis = 1)

        local_nids = tl.where(match_local_id > 0, match_local_id - 1, local_nids)
        local_nid_offs = tl.where(match_local_id > 0, match_local_offset, local_nid_offs)

        offs_nids += BLOCK_M

    # Update sample mask to filter out inactive ones
    mask_sample = mask_sample & (local_nids >= 0)

    # Sample random probabilities uniform between 0 and 1
    rnd_val = tl.rand(seed, offs_sample)

    if conditional:
        nmars = tl.load(node_mars + node_id * batch_size + batch_id, mask = mask_sample, other = 0.0) # [Block_B]

    # Calibration loop
    if do_calibration:
        sum_pars = tl.zeros([BLOCK_S], dtype = tl.float32)
        offs_child = tl.arange(0, BLOCK_K)
        mask_child = offs_child < num_edges
        for i in range(K_NUM_BLKS):

            # Load parameters
            param_id = tl.load(pids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
            epars = tl.load(mparams + param_id + local_nid_offs[None,:], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0) # [BLOCK_K, BLOCK_B]

            if conditional:
                # In this case, we use `param * cmar / nmar` as the "parameter"
                emars_id = tl.load(cids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
                emars = tl.load(element_mars + emars_id * batch_size + batch_id, mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0)

                epars = epars * tl.exp(emars - nmars[None,:]) # [BLOCK_K, BLOCK_B]

            sum_pars += tl.sum(epars, axis = 0)

            offs_child += BLOCK_K
            mask_child = offs_child < num_edges

        rnd_val *= sum_pars
        rnd_val -= 1e-12

    # Main loop over blocks of child nodes
    chids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    offs_child = tl.arange(0, BLOCK_K)
    mask_child = offs_child < num_edges
    for i in range(K_NUM_BLKS):

        # Load parameters
        param_id = tl.load(pids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
        epars = tl.load(mparams + param_id + local_nid_offs[None,:], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0) # [BLOCK_K, BLOCK_S]

        if conditional:
            # In this case, we use `param * cmar / nmar` as the "parameter"
            emars_id = tl.load(cids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
            emars = tl.load(element_mars + emars_id * batch_size + batch_id, mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0)

            epars = epars * tl.exp(emars - nmars[None,:]) # [BLOCK_K, BLOCK_B]

        cum_probs = tl.cumsum(epars, axis = 0) # [BLOCK_K, BLOCK_S]
        local_chids = tl.sum((rnd_val[None,:] >= cum_probs).to(tl.int64), axis = 0) # [BLOCK_K, BLOCK_S]

        is_overflow = (local_chids == BLOCK_K)
        rnd_val = tl.where(is_overflow, rnd_val - tl.sum(epars, axis = 0), rnd_val)

        chids = tl.where(is_overflow | (chids > -1), chids, local_chids + i * BLOCK_K)

        offs_child += BLOCK_K
        mask_child = (offs_child < num_edges)

    # Retrieve the global child ids and save them to `element_samples`
    global_chids = tl.load(cids + local_nids * num_edges + chids, mask = mask_sample, other = 0)
    target_id = tl.load(ind_target + offs_sample, mask = mask_sample, other = 0)

    tl.store(element_samples + target_id, global_chids, mask = mask_sample)


def sample_sum_layer(pc, layer, nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples,
                     ind_target, ind_n, ind_b, block_size, conditional, do_calibration = False):

    num_samples = ind_n.size(0)
    num_nblocks = nids.size(0)
    num_edges = cids.size(1)
    batch_size = node_samples.size(1)
    seed = random.randint(0, 2**31)

    # EVERY tile dimension is floored at 2. Triton's `TritonGPURemoveLayoutConversions` pass fails to
    # compile these kernels whenever a tile has a size-1 dimension that takes part in a reduction --
    # "PassManager::run failed", at compile time, so it is a hard crash rather than a slow path. It is
    # reachable from both ends and was: `BLOCK_M == 1` (a single node block) with `BLOCK_S == 32`
    # (`num_samples >= 4096`) broke every HMM-shaped chain, and `BLOCK_S == 1` (`num_samples < 256`)
    # broke `PD`-structured circuits at small batch, including every conditional draw on them. A
    # second, masked-off lane costs nothing and removes the whole class.
    BLOCK_K = max(2, min(512, triton.next_power_of_2(num_edges)))
    BLOCK_M = max(2, min(512, triton.next_power_of_2(num_nblocks)))
    BLOCK_S = max(2, min(2048 // BLOCK_K, 2048 // BLOCK_M,
                         max(triton.next_power_of_2(num_samples // 128), 1)))

    M_NUM_BLKS = triton.cdiv(num_nblocks, BLOCK_M)
    K_NUM_BLKS = triton.cdiv(num_edges, BLOCK_K)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    sample_sum_layer_kernel[grid](
        nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples,
        ind_target, ind_n, ind_b, seed, block_size, batch_size, num_edges, num_samples, num_nblocks,
        BLOCK_S, BLOCK_M, M_NUM_BLKS, BLOCK_K, K_NUM_BLKS, conditional, do_calibration
    )

    return None
