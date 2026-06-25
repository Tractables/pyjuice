"""
Backward pass: element flows, sparse kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``bk_ele_sparse._<kernel_name>[grid](...)``. To add a kernel for this case, define it
here as a module-level ``@triton_jit`` function -- no extra wiring needed.
"""

import triton
import triton.language as tl

# In the latest triton, math functions were shuffled around into different modules:
# https://github.com/openai/triton/pull/3172
if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math

from pyjuice.utils.kernel_launcher import triton_jit



# @triton.jit
@triton_jit
def _bk_triton_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, mparams, 
                                 chids, parids, parpids, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                 n_edge_blocks: tl.constexpr, allow_modify_flows: tl.constexpr, logspace_flows: tl.constexpr,
                                 BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                                 TILES_PER_BLOCK: tl.constexpr, propagation_alg_id: tl.constexpr, accumulate_ch_flows: tl.constexpr, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`BLOCK_M` nodes

    # Get inferred node block id + within-block tile id from `pid_m`. With `TILES_PER_BLOCK > 1`
    # (small-batch tiling) each node-block is split into `TILES_PER_BLOCK` tiles of `BLOCK_M` nodes, so
    # the node dimension fans across many programs instead of one serial program per block (~1 SM).
    # `TILES_PER_BLOCK == 1` (BLOCK_M == cs_block_size) reproduces the original un-tiled indexing.
    eleblock_id = pid_m // TILES_PER_BLOCK
    ntile_id = pid_m % TILES_PER_BLOCK

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_id = tl.load(local_ids + eleblock_id)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, n_edge_blocks * BLOCK_SIZE_K) # I.e., [0, num_edges)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids + eleblock_id * n_edge_blocks + offs_edge_gid)
    # Offset the per-node params by the within-block tile (`ntile_id`); the parents (`par_start`,
    # `edge_start` below) are shared across the whole node-block, so they are NOT tile-offset.
    epars_ptr = mparams + par_start + offs_edge_nid + ntile_id * BLOCK_M * BLOCK_SIZE_K # [num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize `node_flows` and `node_mars`
    edge_start = tl.load(parids + eleblock_id * n_edge_blocks + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:]
    nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:]
    if allow_modify_flows == 1:
        log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
    else:
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

    # Initialize pointers to `element_flows` and `element_mars` (tile-offset within the node-block)
    off_eleids = tl.load(chids + eleblock_id) + ntile_id * BLOCK_M
    eflows_ptr = element_flows + off_eleids * batch_size + offs_batch # [BLOCK_B]
    emars_ptr = element_mars + off_eleids * batch_size + offs_batch # [BLOCK_B]

    # Inner loop
    for i in range(0, BLOCK_M):
        epars = tl.load(epars_ptr) # [num_edges]
        emars = tl.load(emars_ptr, mask = mask_batch) # [BLOCK_B]

        if propagation_alg_id == 1:
            if allow_modify_flows:
                nflows = log_n_fdm

            lpars = tl.log(epars)
            eflows = tl.sum(tl.where(tl.abs(lpars[:,None] + emars[None,:] - nmars) < 1e-6, nflows, 0.0), axis = 0)

        else:
            lpars = tl.log(epars)
            if propagation_alg_id == 2:
                lpars *= alpha
                epars = tl.exp(lpars)

            if allow_modify_flows == 1:
                if propagation_alg_id == 0:
                    eflows = tl.sum(epars[:,None] * tl.exp(emars[None,:] + log_n_fdm), axis = 0)
                
                if propagation_alg_id == 2:
                    eflows = tl.sum(epars[:,None] * tl.exp(emars[None,:] * alpha + log_n_fdm), axis = 0)
            else:
                if logspace_flows:
                    if propagation_alg_id == 0:
                        elflows = nflows + lpars[:,None] + emars[None,:] - nmars
                    
                    if propagation_alg_id == 2:
                        elflows = nflows + lpars[:,None] + (emars[None,:] - nmars) * alpha

                    elflows_max = tl.max(elflows, axis = 0)
                    eflows = tl.log(tl.sum(tl.exp(elflows - elflows_max[None,:]), axis = 0)) + elflows_max
                    eflows = tl.where((elflows_max == -float("inf")) | (emars == -float("inf")), -float("inf"), eflows)

                else:
                    if propagation_alg_id == 0:
                        eflows = tl.sum(nflows * epars[:,None] * tl.exp(emars[None,:] - nmars), axis = 0)

                    if propagation_alg_id == 2:
                        eflows = tl.sum(nflows * epars[:,None] * tl.exp((emars[None,:] - nmars) * alpha), axis = 0)

        if accumulate_ch_flows:
            ori_eflows = tl.load(eflows_ptr, mask = mask_batch, other = 0.0)
            if logspace_flows:
                m = tl.maximum(eflows, ori_eflows)
                eflows = tl.where(m == -float("inf"), -float("inf"),
                                m + tl.log(tl.exp(eflows - m) + tl.exp(ori_eflows - m)))
            else:
                eflows += ori_eflows

        tl.store(eflows_ptr, eflows, mask = mask_batch)

        # Increment `epars_ptr`
        epars_ptr += BLOCK_SIZE_K

        # Increment `emars_ptr` and `eflows_ptr`
        emars_ptr += batch_size
        eflows_ptr += batch_size


# @triton.jit
@triton_jit
def _bk_triton_large_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, mparams, 
                                       chids, parids, parpids, local_ids, num_eles, pid_m_offset, 
                                       batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                       n_edge_blocks: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                       logspace_flows: tl.constexpr, BLOCK_B: tl.constexpr, 
                                       TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                                       propagation_alg_id: tl.constexpr, accumulate_ch_flows: tl.constexpr, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
    mask_m = offs_m < num_eles

    # Get inferred node block id from `pid_m`
    eleblock_ids = offs_m // BLOCK_SIZE_M
    tile_ids = offs_m % BLOCK_SIZE_M

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_ids = tl.load(local_ids + eleblock_ids)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, n_edge_blocks * BLOCK_SIZE_K) # I.e., [0, num_edges)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids + eleblock_ids[:,None] * n_edge_blocks + offs_edge_gid[None,:])
    epars = tl.load(mparams + par_start + offs_edge_nid[None,:], mask = mask_m[:,None]) # [TILE_SIZE_M, num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize `node_flows` and `node_mars`
    edge_start = tl.load(parids + eleblock_ids[:,None] * n_edge_blocks + offs_edge_gid[None,:]) # [TILE_SIZE_M, num_edges]
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid[None,:])[:,:,None] * batch_size + \
        offs_batch[None,None,:]
    nmars = tl.load(nmars_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid[None,:])[:,:,None] * batch_size + \
        offs_batch[None,None,:]
    if allow_modify_flows == 1:
        log_n_fdm = tl.load(nflows_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]
    else:
        nflows = tl.load(nflows_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]

    # Initialize pointers to `element_flows` and values `emars`
    off_eleids = tl.load(chids + eleblock_ids) # TILE_SIZE_M
    eflows_ptr = element_flows + off_eleids[:,None] * batch_size + offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]
    emars = tl.load(element_mars + off_eleids[:,None] * batch_size + offs_batch[None,:], 
                    mask = (mask_m[:,None] & mask_batch[None,:])) # [TILE_SIZE_M, BLOCK_B]

    # Compute eflows
    if propagation_alg_id == 1:
        if allow_modify_flows:
            nflows = log_n_fdm

        lpars = tl.log(epars)
        eflows = tl.sum(tl.where(tl.abs(lpars[:,:,None] + emars[:,None,:] - nmars) < 1e-6, nflows, 0.0), axis = 1)

    else:
        lpars = tl.log(epars)
        if propagation_alg_id == 2:
            lpars *= alpha
            epars = tl.exp(lpars)

        if allow_modify_flows == 1:
            if propagation_alg_id == 0:
                eflows = tl.sum(epars[:,:,None] * tl.exp(emars[:,None,:] + log_n_fdm), axis = 1)

            if propagation_alg_id == 2:
                eflows = tl.sum(epars[:,:,None] * tl.exp(emars[:,None,:] * alpha + log_n_fdm), axis = 1)
        else:
            if logspace_flows:
                if propagation_alg_id == 0:
                    elflows = nflows + lpars[:,:,None] + emars[:,None,:] - nmars

                if propagation_alg_id == 2:
                    elflows = nflows + lpars[:,:,None] + (emars[:,None,:] - nmars) * alpha

                elflows_max = tl.max(elflows, axis = 1)
                eflows = tl.log(tl.sum(tl.exp(elflows - elflows_max[:,None,:]), axis = 1)) + elflows_max
                eflows = tl.where((elflows_max == -float("inf")) | (emars == -float("inf")), -float("inf"), eflows)
            else:
                if propagation_alg_id == 0:
                    eflows = tl.sum(nflows * epars[:,:,None] * tl.exp(emars[:,None,:] - nmars), axis = 1)

                if propagation_alg_id == 2:
                    eflows = tl.sum(nflows * epars[:,:,None] * tl.exp((emars[:,None,:] - nmars) * alpha), axis = 0)

    if accumulate_ch_flows:
        ori_eflows = tl.load(eflows_ptr, mask = (mask_m[:,None] & mask_batch[None,:]), other = 0.0)
        if logspace_flows:
            m = tl.maximum(eflows, ori_eflows)
            eflows = tl.where(m == -float("inf"), -float("inf"),
                              m + tl.log(tl.exp(eflows - m) + tl.exp(ori_eflows - m)))
        else:
            eflows += ori_eflows

    tl.store(eflows_ptr, eflows, mask = (mask_m[:,None] & mask_batch[None,:]))


@triton_jit
def _bk_triton_sparse_tempered_ele_kernel(node_flows, element_flows, node_mars_tempered, element_mars, mparams, 
                                          chids, parids, parpids, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                          n_edge_blocks: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
                                          accumulate_ch_flows: tl.constexpr, eflow_temperature = 1.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`BLOCK_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_id = tl.load(local_ids + eleblock_id)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, n_edge_blocks * BLOCK_SIZE_K) # I.e., [0, num_edges)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids + eleblock_id * n_edge_blocks + offs_edge_gid)
    epars_ptr = mparams + par_start + offs_edge_nid # [num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize `node_flows` and `node_mars_tempered`
    edge_start = tl.load(parids + eleblock_id * n_edge_blocks + offs_edge_gid)
    nmars_tempered_ptr = node_mars_tempered + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:]
    nmars_tempered = tl.load(nmars_tempered_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:]
    nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

    # Initialize pointers to `element_flows` and `element_mars`
    off_eleids = tl.load(chids + eleblock_id)
    eflows_ptr = element_flows + off_eleids * batch_size + offs_batch # [BLOCK_B]
    emars_ptr = element_mars + off_eleids * batch_size + offs_batch # [BLOCK_B]

    # Inner loop
    for i in range(0, BLOCK_M):
        epars = tl.load(epars_ptr) # [num_edges]
        emars = tl.load(emars_ptr, mask = mask_batch) # [BLOCK_B]

        lpars = tl.log(epars)

        elflows = nflows + (lpars[:,None] + emars[None,:]) / eflow_temperature - nmars_tempered

        elflows_max = tl.max(elflows, axis = 0)
        eflows = tl.log(tl.sum(tl.exp(elflows - elflows_max[None,:]), axis = 0)) + elflows_max
        eflows = tl.where((elflows_max == -float("inf")) | (emars == -float("inf")), -float("inf"), eflows)

        if accumulate_ch_flows:
            ori_eflows = tl.load(eflows_ptr, mask = mask_batch, other = 0.0)
            m = tl.maximum(eflows, ori_eflows)
            eflows = tl.where(m == -float("inf"), -float("inf"),
                              m + tl.log(tl.exp(eflows - m) + tl.exp(ori_eflows - m)))

        tl.store(eflows_ptr, eflows, mask = mask_batch)

        # Increment `epars_ptr`
        epars_ptr += BLOCK_SIZE_K

        # Increment `emars_ptr` and `eflows_ptr`
        emars_ptr += batch_size
        eflows_ptr += batch_size


@triton_jit
def _bk_triton_large_sparse_tempered_ele_kernel(node_flows, element_flows, node_mars_tempered, element_mars, mparams, 
                                                chids, parids, parpids, local_ids, num_eles, pid_m_offset, 
                                                batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                                n_edge_blocks: tl.constexpr, BLOCK_B: tl.constexpr, 
                                                TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                                                accumulate_ch_flows: tl.constexpr, eflow_temperature = 1.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
    mask_m = offs_m < num_eles

    # Get inferred node block id from `pid_m`
    eleblock_ids = offs_m // BLOCK_SIZE_M
    tile_ids = offs_m % BLOCK_SIZE_M

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_ids = tl.load(local_ids + eleblock_ids)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, n_edge_blocks * BLOCK_SIZE_K) # I.e., [0, num_edges)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids + eleblock_ids[:,None] * n_edge_blocks + offs_edge_gid[None,:])
    epars = tl.load(mparams + par_start + offs_edge_nid[None,:], mask = mask_m[:,None]) # [TILE_SIZE_M, num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize `node_flows` and `node_mars_tempered`
    edge_start = tl.load(parids + eleblock_ids[:,None] * n_edge_blocks + offs_edge_gid[None,:]) # [TILE_SIZE_M, num_edges]
    nmars_tempered_ptr = node_mars_tempered + \
        (edge_start + offs_edge_nid[None,:])[:,:,None] * batch_size + \
        offs_batch[None,None,:]
    nmars_tempered = tl.load(nmars_tempered_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid[None,:])[:,:,None] * batch_size + \
        offs_batch[None,None,:]
    nflows = tl.load(nflows_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]

    # Initialize pointers to `element_flows` and values `emars`
    off_eleids = tl.load(chids + eleblock_ids) # TILE_SIZE_M
    eflows_ptr = element_flows + off_eleids[:,None] * batch_size + offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]
    emars = tl.load(element_mars + off_eleids[:,None] * batch_size + offs_batch[None,:], 
                    mask = (mask_m[:,None] & mask_batch[None,:])) # [TILE_SIZE_M, BLOCK_B]

    # Compute eflows
    lpars = tl.log(epars)
    
    elflows = nflows + (lpars[:,:,None] + emars[:,None,:]) / eflow_temperature - nmars_tempered

    elflows_max = tl.max(elflows, axis = 1)
    eflows = tl.log(tl.sum(tl.exp(elflows - elflows_max[:,None,:]), axis = 1)) + elflows_max
    eflows = tl.where((elflows_max == -float("inf")) | (emars == -float("inf")), -float("inf"), eflows)

    if accumulate_ch_flows:
        ori_eflows = tl.load(eflows_ptr, mask = (mask_m[:,None] & mask_batch[None,:]), other = 0.0)
        m = tl.maximum(eflows, ori_eflows)
        eflows = tl.where(m == -float("inf"), -float("inf"),
                          m + tl.log(tl.exp(eflows - m) + tl.exp(ori_eflows - m)))

    tl.store(eflows_ptr, eflows, mask = (mask_m[:,None] & mask_batch[None,:]))
