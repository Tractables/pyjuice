"""
Backward pass: element flows, block-sparse kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``bk_ele_bsparse._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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
def _bk_triton_block_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, mparams, 
                                       chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                       local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                       allow_modify_flows: tl.constexpr, logspace_flows: tl.constexpr, BLOCK_B: tl.constexpr, 
                                       TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr, 
                                       propagation_alg_id: tl.constexpr, accumulate_ch_flows: tl.constexpr, 
                                       allow_neg_flows: tl.constexpr, pid_m_offset = 0, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_id = tl.load(local_ids + eleblock_id)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = mparams + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars`
    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Initialize pointers to `element_mars` (only when using MPE propagation method)
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    if propagation_alg_id == 2:
        emars *= alpha

    # Inner loop
    if logspace_flows:
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")
    else:
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]

        if propagation_alg_id == 1:
            nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
            nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
            elpars = tl.log(tl.trans(epars)) # [TILE_SIZE_K, TILE_SIZE_M]

            eflows = tl.sum(tl.where(tl.abs(elpars[:,:,None] + emars[None,:,:] - nmars[:,None,:]) < 1e-6, nflows[:,None,:], 0.0), axis = 0)

            if logspace_flows:
                # logaddexp
                diff = acc - eflows
                acc = tl.where(
                    diff == 0, 
                    acc + 0.69314718055994530942, # log(2)
                    tl.where(
                        diff > 0,
                        acc + tlmath.log1p(tl.exp(-diff)),
                        eflows + tlmath.log1p(tl.exp(diff))
                    )
                )
            else:
                # sum
                acc += eflows
        else:

            if propagation_alg_id == 2:
                epars = tl.exp(tl.log(epars) * alpha)

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
                nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

                if logspace_flows:
                    if propagation_alg_id == 0:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)

                    if propagation_alg_id == 2:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars * alpha)
                elif allow_neg_flows:
                    if propagation_alg_id == 0:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars)
                    
                    if propagation_alg_id == 2:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars * alpha)
                else:
                    if propagation_alg_id == 0:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)
                    
                    if propagation_alg_id == 2:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars * alpha)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
            n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

            if allow_neg_flows:
                if TL_DOT == 1:
                    partial_flows = tl.dot(epars, n_fdm_sub * nflows)
                else:
                    partial_flows = tl.sum(epars[:,:,None] * n_fdm_sub[None,:,:] * nflows[None,:,:], axis = 1)
            else:
                if TL_DOT == 1:
                    partial_flows = tl.dot(epars, n_fdm_sub)
                else:
                    partial_flows = tl.sum(epars[:,:,None] * n_fdm_sub[None,:,:], axis = 1)

            if logspace_flows:
                partial_flows_max = emars + log_n_fdm_max
                acc = tl.where(partial_flows_max == -float("inf"),
                    acc,
                    tl.where(partial_flows_max > acc,
                        tl.log(partial_flows + tl.exp(acc - partial_flows_max) + 1e-32) + partial_flows_max,
                        tl.log(tl.exp(partial_flows_max - acc) * partial_flows + 1.0) + acc
                    )
                )
            else:
                acc += partial_flows * tl.exp(emars + log_n_fdm_max)

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[:,None] * batch_size
        nflows_ptr += parids_inc[:,None] * batch_size
        parids_inc_ptr += ptr_inc_step

    # Write back
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    if accumulate_ch_flows:
        ori_eflows = tl.load(element_flows + offs_elemfs, mask = mask_batch[None,:], other = 0.0)
        if logspace_flows:
            m = tl.maximum(acc, ori_eflows)
            acc = tl.where(m == -float("inf"), 
                -float("inf"),
                m + tl.log(tl.exp(acc - m) + tl.exp(ori_eflows - m))
            )
        else:
            acc += ori_eflows
    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


# @triton.jit
@triton_jit
def _bk_triton_block_sparse_ele_csmm2_kernel(node_flows, element_flows, node_mars, element_mars, mparams, 
                                             chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                             local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                             allow_modify_flows: tl.constexpr, logspace_flows: tl.constexpr, BLOCK_B: tl.constexpr, 
                                             TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                             BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr, 
                                             propagation_alg_id: tl.constexpr, accumulate_ch_flows: tl.constexpr, 
                                             allow_neg_flows: tl.constexpr, pid_m_offset = 0, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_id = tl.load(local_ids + eleblock_id)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = mparams + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars`
    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[None,:] * batch_size + \
        offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[None,:] * batch_size + \
        offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Initialize pointers to `element_mars`
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    if propagation_alg_id == 2:
        emars *= alpha

    # Inner loop
    if logspace_flows:
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")
    else:
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]

        if propagation_alg_id == 1:
            nflows = tl.load(nflows_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
            nmars = tl.load(nmars_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
            elpars = tl.log(tl.trans(epars)) # [TILE_SIZE_K, TILE_SIZE_M]

            eflows = tl.sum(tl.where(tl.abs(elpars[None,:,:] + tl.trans(emars)[:,None,:] - nmars[:,:,None]) < 1e-6, nflows[:,:,None], 0.0), axis = 1)
            eflows = tl.trans(eflows)

            if logspace_flows:
                # logaddexp
                diff = acc - eflows
                acc = tl.where(
                    diff == 0, 
                    acc + 0.69314718055994530942, # log(2)
                    tl.where(
                        diff > 0,
                        acc + tlmath.log1p(tl.exp(-diff)),
                        eflows + tlmath.log1p(tl.exp(diff))
                    )
                )
            else:
                # sum
                acc += eflows

        else:

            if propagation_alg_id == 2:
                epars = tl.exp(tl.log(epars) * alpha)

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
                nmars = tl.load(nmars_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]

                if logspace_flows:
                    if propagation_alg_id == 0:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)

                    if propagation_alg_id == 2:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars * alpha)
                elif allow_neg_flows:
                    if propagation_alg_id == 0:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars)
                    
                    if propagation_alg_id == 2:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars * alpha)
                else:
                    if propagation_alg_id == 0:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)
                    
                    if propagation_alg_id == 2:
                        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars * alpha)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 1)
            n_fdm_sub = tl.where(log_n_fdm_max[:,None] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[:,None]), 0.0)

            if allow_neg_flows:
                partial_flows = tl.sum(epars[:,:,None] * tl.trans(n_fdm_sub * nflows)[None,:,:], axis = 1)
            else:
                partial_flows = tl.sum(epars[:,:,None] * tl.trans(n_fdm_sub)[None,:,:], axis = 1)

            if logspace_flows:
                partial_flows_max = emars + log_n_fdm_max[None,:]
                acc = tl.where(partial_flows_max == -float("inf"),
                    acc,
                    tl.where(partial_flows_max > acc,
                        tl.log(partial_flows + tl.exp(acc - partial_flows_max) + 1e-32) + partial_flows_max,
                        tl.log(tl.exp(partial_flows_max - acc) * partial_flows + 1.0) + acc
                    )
                )
            else:
                acc += partial_flows * tl.exp(emars + log_n_fdm_max[None,:])

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[None,:] * batch_size
        nflows_ptr += parids_inc[None,:] * batch_size
        parids_inc_ptr += ptr_inc_step

    # Write back
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    if accumulate_ch_flows:
        ori_eflows = tl.load(element_flows + offs_elemfs, mask = mask_batch[None,:], other = 0.0)
        if logspace_flows:
            m = tl.maximum(acc, ori_eflows)
            acc = tl.where(m == -float("inf"), 
                -float("inf"),
                m + tl.log(tl.exp(acc - m) + tl.exp(ori_eflows - m))
            )
        else:
            acc += ori_eflows
    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


@triton_jit
def _bk_triton_block_sparse_tempered_ele_kernel(node_flows, element_flows, node_mars_tempered, element_mars, mparams, 
                                                chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                                local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                                BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr, 
                                                accumulate_ch_flows: tl.constexpr, pid_m_offset = 0, eflow_temperature = 1.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_id = tl.load(local_ids + eleblock_id)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = mparams + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars_tempered`
    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_tempered_ptr = node_mars_tempered + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Initialize pointers to `element_mars` (only when using MPE propagation method)
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tlmath.pow(epars, 1.0 / eflow_temperature)

        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        nmars_tempered = tl.load(nmars_tempered_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        log_n_fdm = tl.where(nmars_tempered == -float("inf"), -float("inf"), nflows - nmars_tempered)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
        n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

        if TL_DOT == 1:
            partial_flows = tl.dot(epars, n_fdm_sub)
        else:
            partial_flows = tl.sum(epars[:,:,None] * n_fdm_sub[None,:,:], axis = 1)

        partial_flows_max = emars / eflow_temperature + log_n_fdm_max
        acc = tl.where(partial_flows_max == -float("inf"),
            acc,
            tl.where(partial_flows_max > acc,
                tl.log(partial_flows + tl.exp(acc - partial_flows_max) + 1e-32) + partial_flows_max,
                tl.log(tl.exp(partial_flows_max - acc) * partial_flows + 1.0) + acc
            )
        )

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_tempered_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_tempered_ptr += parids_inc[:,None] * batch_size
        nflows_ptr += parids_inc[:,None] * batch_size
        parids_inc_ptr += ptr_inc_step

    # Write back
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    if accumulate_ch_flows:
        ori_eflows = tl.load(element_flows + offs_elemfs, mask = mask_batch[None,:], other = 0.0)
        m = tl.maximum(acc, ori_eflows)
        acc = tl.where(m == -float("inf"), 
            -float("inf"),
            m + tl.log(tl.exp(acc - m) + tl.exp(ori_eflows - m))
        )

    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


@triton_jit
def _bk_triton_block_sparse_tempered_ele_csmm2_kernel(node_flows, element_flows, node_mars_tempered, element_mars, mparams, 
                                                      chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                                      local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                                      BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr, 
                                                      accumulate_ch_flows: tl.constexpr, pid_m_offset = 0, eflow_temperature = 1.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        eleblock_id = tl.load(local_ids + eleblock_id)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = mparams + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars_tempered`
    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_tempered_ptr = node_mars_tempered + \
        (edge_start + offs_edge_nid)[None,:] * batch_size + \
        offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[None,:] * batch_size + \
        offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Initialize pointers to `element_mars`
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tlmath.pow(epars, 1.0 / eflow_temperature)

        nflows = tl.load(nflows_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
        nmars_tempered = tl.load(nmars_tempered_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]

        log_n_fdm = tl.where(nmars_tempered == -float("inf"), -float("inf"), nflows - nmars_tempered)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 1)
        n_fdm_sub = tl.where(log_n_fdm_max[:,None] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[:,None]), 0.0)

        partial_flows = tl.sum(epars[:,:,None] * tl.trans(n_fdm_sub)[None,:,:], axis = 1)

        partial_flows_max = emars / eflow_temperature + log_n_fdm_max[None,:]
        acc = tl.where(partial_flows_max == -float("inf"),
            acc,
            tl.where(partial_flows_max > acc,
                tl.log(partial_flows + tl.exp(acc - partial_flows_max) + 1e-32) + partial_flows_max,
                tl.log(tl.exp(partial_flows_max - acc) * partial_flows + 1.0) + acc
            )
        )

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_tempered_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_tempered_ptr += parids_inc[None,:] * batch_size
        nflows_ptr += parids_inc[None,:] * batch_size
        parids_inc_ptr += ptr_inc_step

    # Write back
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    if accumulate_ch_flows:
        ori_eflows = tl.load(element_flows + offs_elemfs, mask = mask_batch[None,:], other = 0.0)
        m = tl.maximum(acc, ori_eflows)
        acc = tl.where(m == -float("inf"), 
            -float("inf"),
            m + tl.log(tl.exp(acc - m) + tl.exp(ori_eflows - m))
        )

    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])
