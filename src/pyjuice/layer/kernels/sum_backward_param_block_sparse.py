"""
Backward pass: parameter flows, block-sparse kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``bk_par_bsparse._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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
def _bk_triton_block_sparse_par_kernel(node_flows, node_mars, element_mars, mparams, param_flows, nids, cids, pids, pfids,
                                       batch_size: tl.constexpr, num_edges: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                       logspace_flows: tl.constexpr, TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, 
                                       TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
                                       TL_DOT: tl.constexpr, propagation_alg_id: tl.constexpr, negate_pflows: tl.constexpr, 
                                       allow_neg_flows: tl.constexpr, pid_m_offset = 0, alpha = 0.0):

    pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Batch offsets and mask
    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

    # Initialize pointers to `node_flows` and `node_mars`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nflows_ptr = node_flows + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

    # Initialize `params` (only when using MPE propagation method)
    if propagation_alg_id == 1:
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(mparams + epars_offsets)
        elpars = tl.log(epars)

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]

        if propagation_alg_id == 1:
            nflows = tl.load(nflows_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]
            nmars = tl.load(nmars_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]

            cond = tl.abs(elpars[:,None,:] + emars[None,:,:] - nmars[:,:,None]) < 1e-6
            if logspace_flows:
                acc += tl.sum(tl.where(cond, tl.exp(nflows[:,:,None]), 0.0), axis = 1)
            else:
                acc += tl.sum(tl.where(cond, nflows[:,:,None], 0.0), axis = 1)

        else:

            nmars = tl.load(nmars_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:], other = -float("inf")) # [TILE_SIZE_M, TILE_SIZE_B]

                if propagation_alg_id == 2:
                    log_n_fdm += (alpha - 1.0) * nmars
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]
                
                if logspace_flows:
                    log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)
                elif allow_neg_flows:
                    log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars)
                else:
                    log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
            n_fdm_sub = tl.where(log_n_fdm_max[None,:] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[None,:]), 0.0)

            scaled_emars = tl.exp(emars + log_n_fdm_max[:,None])

            if allow_neg_flows:
                if TL_DOT == 1:
                    partial_flows = tl.dot(n_fdm_sub * nflows, scaled_emars)
                else:
                    partial_flows = tl.sum(n_fdm_sub[:,:,None] * nflows[:,:,None] * scaled_emars[None,:,:], axis = 1)
            else:
                if TL_DOT == 1:
                    partial_flows = tl.dot(n_fdm_sub, scaled_emars)
                else:
                    partial_flows = tl.sum(n_fdm_sub[:,:,None] * scaled_emars[None,:,:], axis = 1)

            acc += partial_flows

        # Increment `emars_ptr`, `nmars_ptr`, and `nflows_ptr`
        emars_ptr += TILE_SIZE_B
        nmars_ptr += TILE_SIZE_B
        nflows_ptr += TILE_SIZE_B

        # Update batch mask
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    # Initialize `params` (only when NOT using MPE propagation method)
    if propagation_alg_id != 1:
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(mparams + epars_offsets)

    if propagation_alg_id != 1:
        pflows = acc * epars
    else:
        pflows = acc

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    if negate_pflows:
        tl.atomic_add(param_flows + eparflows_offsets, -1.0 * pflows)
    else:
        tl.atomic_add(param_flows + eparflows_offsets, pflows)


# @triton.jit
@triton_jit
def _bk_triton_block_sparse_par_csmm2_kernel(node_flows, node_mars, element_mars, mparams, param_flows, nids, cids, pids, pfids,
                                             batch_size: tl.constexpr, num_edges: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                             logspace_flows: tl.constexpr, TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, 
                                             TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
                                             TL_DOT: tl.constexpr, propagation_alg_id: tl.constexpr, negate_pflows: tl.constexpr, 
                                             allow_neg_flows: tl.constexpr, pid_m_offset = 0, alpha = 0.0):

    pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Batch offsets and mask
    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

    # Initialize pointers to `node_flows` and `node_mars`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + (off_nids + offs_node[None,:]) * batch_size + offs_batch[:,None]
    nflows_ptr = node_flows + (off_nids + offs_node[None,:]) * batch_size + offs_batch[:,None]

    # Initialize `params` (only when using MPE propagation method)
    if propagation_alg_id == 1:
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(mparams + epars_offsets)
        elpars = tl.log(epars)

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)
    
    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]
        nmars = tl.load(nmars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]

        if propagation_alg_id == 1:
            nflows = tl.load(nflows_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]

            if logspace_flows:
                acc += tl.sum(tl.where(tl.abs(elpars[None,:,:] + emars[:,None,:] - nmars[:,:,None]) < 1e-6, tl.exp(nflows[:,:,None]), 0.0), axis = 0)
            else:
                acc += tl.sum(tl.where(tl.abs(elpars[None,:,:] + emars[:,None,:] - nmars[:,:,None]) < 1e-6, nflows[:,:,None], 0.0), axis = 0)

        else:

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[:,None], other = -float("inf")) # [TILE_SIZE_B, TILE_SIZE_M]

                if propagation_alg_id == 2:
                    log_n_fdm += (alpha - 1.0) * nmars
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]

                if logspace_flows:
                    log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)
                elif allow_neg_flows:
                    log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars)
                else:
                    log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 1)
            n_fdm_sub = tl.where(log_n_fdm_max[:,None] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[:,None]), 0.0)

            scaled_emars = tl.exp(emars + log_n_fdm_max[:,None])

            if allow_neg_flows:
                partial_flows = tl.sum(tl.trans(n_fdm_sub * nflows)[:,:,None] * scaled_emars[None,:,:], axis = 1)
            else:
                partial_flows = tl.sum(tl.trans(n_fdm_sub)[:,:,None] * scaled_emars[None,:,:], axis = 1)

            acc += partial_flows

        # Increment `emars_ptr`, `nmars_ptr`, and `nflows_ptr`
        emars_ptr += TILE_SIZE_B
        nmars_ptr += TILE_SIZE_B
        nflows_ptr += TILE_SIZE_B

        # Update batch mask
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    # Initialize `params` (only when NOT using MPE propagation method)
    if propagation_alg_id != 1:
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(mparams + epars_offsets)

    if propagation_alg_id != 1:
        pflows = acc * epars
    else:
        pflows = acc

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    if negate_pflows:
        tl.atomic_add(param_flows + eparflows_offsets, -1.0 * pflows)
    else:
        tl.atomic_add(param_flows + eparflows_offsets, pflows)


@triton_jit
def _bk_triton_block_sparse_tempered_par_kernel(node_flows, node_mars_tempered, element_mars, mparams, param_flows, nids, cids, pids, pfids,
                                                batch_size: tl.constexpr, num_edges: tl.constexpr, TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, 
                                                TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
                                                TL_DOT: tl.constexpr, negate_pflows: tl.constexpr, pid_m_offset = 0, pflow_temperature = 1.0):

    pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Batch offsets and mask
    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

    # Initialize pointers to `node_flows` and `node_mars_tempered`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + nblock_id)
    nmars_tempered_ptr = node_mars_tempered + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nflows_ptr = node_flows + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]

        nmars_tempered = tl.load(nmars_tempered_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]

        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]
            
        log_n_fdm = tl.where(nmars_tempered == -float("inf"), -float("inf"), nflows - nmars_tempered)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
        n_fdm_sub = tl.where(log_n_fdm_max[None,:] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[None,:]), 0.0)

        scaled_emars = tl.exp(emars / pflow_temperature + log_n_fdm_max[:,None])

        if TL_DOT == 1:
            partial_flows = tl.dot(n_fdm_sub, scaled_emars)
        else:
            partial_flows = tl.sum(n_fdm_sub[:,:,None] * scaled_emars[None,:,:], axis = 1)

        acc += partial_flows

        # Increment `emars_ptr`, `nmars_tempered_ptr`, and `nflows_ptr`
        emars_ptr += TILE_SIZE_B
        nmars_tempered_ptr += TILE_SIZE_B
        nflows_ptr += TILE_SIZE_B

        # Update batch mask
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    # Initialize `params` (only when NOT using MPE propagation method)
    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
    epars = tl.load(mparams + epars_offsets)

    pflows = acc * tlmath.pow(epars, 1.0 / pflow_temperature)

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    if negate_pflows:
        tl.atomic_add(param_flows + eparflows_offsets, -1.0 * pflows)
    else:
        tl.atomic_add(param_flows + eparflows_offsets, pflows)


@triton_jit
def _bk_triton_block_sparse_tempered_par_csmm2_kernel(node_flows, node_mars_tempered, element_mars, mparams, param_flows, nids, cids, pids, pfids,
                                                      batch_size: tl.constexpr, num_edges: tl.constexpr, TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, 
                                                      TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
                                                      TL_DOT: tl.constexpr, negate_pflows: tl.constexpr, pid_m_offset = 0, pflow_temperature = 1.0):

    pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Batch offsets and mask
    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

    # Initialize pointers to `node_flows` and `node_mars`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + nblock_id)
    nmars_tempered_ptr = node_mars_tempered + (off_nids + offs_node[None,:]) * batch_size + offs_batch[:,None]
    nflows_ptr = node_flows + (off_nids + offs_node[None,:]) * batch_size + offs_batch[:,None]

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)
    
    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]
        nmars_tempered = tl.load(nmars_tempered_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]

        nflows = tl.load(nflows_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]

        log_n_fdm = tl.where(nmars_tempered == -float("inf"), -float("inf"), nflows - nmars_tempered)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 1)
        n_fdm_sub = tl.where(log_n_fdm_max[:,None] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[:,None]), 0.0)

        scaled_emars = tl.exp(emars / pflow_temperature + log_n_fdm_max[:,None])

        partial_flows = tl.sum(tl.trans(n_fdm_sub)[:,:,None] * scaled_emars[None,:,:], axis = 1)

        acc += partial_flows

        # Increment `emars_ptr`, `nmars_ptr`, and `nflows_ptr`
        emars_ptr += TILE_SIZE_B
        nmars_tempered_ptr += TILE_SIZE_B
        nflows_ptr += TILE_SIZE_B

        # Update batch mask
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    # Initialize `params`
    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
    epars = tl.load(mparams + epars_offsets)

    pflows = acc * tlmath.pow(epars, 1.0 / pflow_temperature)

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    if negate_pflows:
        tl.atomic_add(param_flows + eparflows_offsets, -1.0 * pflows)
    else:
        tl.atomic_add(param_flows + eparflows_offsets, pflows)
