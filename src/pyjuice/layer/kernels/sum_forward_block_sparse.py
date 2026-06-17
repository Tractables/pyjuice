"""
Forward pass: block-sparse sum kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``fw_bsparse._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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
def _fw_triton_block_sparse_tlmm_kernel(node_mars, element_mars, mparams, nids, cids_start, cids_increment,
                                        pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                        BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                        TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, use_bf16: tl.constexpr,
                                        propagation_alg_id: tl.constexpr, pflow_tempered_enabled: tl.constexpr, 
                                        pid_m_offset = 0, alpha = 0.0, pflow_temperature = 1.0, node_mars_tempered = None):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Node offsets
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

    # Edge offsets
    offs_edge = tl.arange(0, TILE_SIZE_K)

    # Initialize pointers to `params`
    offs_estart = nblock_id * TILE_SIZE_K + offs_edge
    offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
    par_start = tl.load(pids_start + offs_estart)
    epars_ptr = mparams + \
        offs_node[:,None] + \
        par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    edge_start = tl.load(cids_start + offs_estart)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    if pflow_tempered_enabled:
        acc_tempered = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr)
        emars = tl.load(emars_ptr, mask = mask_batch[None,:])

        if propagation_alg_id == 1:
            # MPE propagation method
            lpars = tl.log(epars)
            nmars = tl.max(lpars[:,:,None] + emars[None,:,:], axis = 1)

            acc = tl.maximum(acc, nmars)

        else:

            if propagation_alg_id == 0:
                # LL propagation method
                emars_max = tl.max(emars, axis = 0)[None,:]
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

            if propagation_alg_id == 2:
                # GeneralLL propagation method

                emars_max = tl.max(emars, axis = 0)[None,:]
                # Compute p_i^{alpha} for every i
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp((emars - emars_max) * alpha), 0.0)
                # Compute w_i^{alpha} for every i
                epars = tl.exp(tl.log(epars) * alpha)

                # Also scale `emars_max`
                emars_max *= alpha

            if use_bf16 == 1:
                # Built-in matmul kernel of triton + float16
                epars_bf16 = epars.to(tl.bfloat16)
                emars_bf16 = emars_sub.to(tl.bfloat16)
                nmars = tl.dot(epars_bf16, emars_bf16).to(tl.float32)
            else:
                # Built-in matmul kernel of triton + float32
                nmars = tl.dot(epars, emars_sub)

            acc = tl.where(emars_max > acc,
                tl.log(nmars + tl.exp(acc - emars_max) + 1e-24) + emars_max,
                tl.where(acc != -float("inf"),
                    tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc,
                    -float("inf")
                )
            )

            if pflow_tempered_enabled:
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp((emars - emars_max) / pflow_temperature), 0.0)
                epars = tlmath.pow(epars, 1.0 / pflow_temperature)
                emars_max /= pflow_temperature

                if use_bf16 == 1:
                    # Built-in matmul kernel of triton + float16
                    epars_bf16 = epars.to(tl.bfloat16)
                    emars_bf16 = emars_sub.to(tl.bfloat16)
                    nmars = tl.dot(epars_bf16, emars_bf16).to(tl.float32)
                else:
                    # Built-in matmul kernel of triton + float32
                    nmars = tl.dot(epars, emars_sub)

                acc_tempered = tl.where(emars_max > acc_tempered,
                    tl.log(nmars + tl.exp(acc_tempered - emars_max) + 1e-24) + emars_max,
                    tl.where(acc_tempered != -float("inf"),
                        tl.log(tl.exp(emars_max - acc_tempered) * nmars + 1.0) + acc_tempered,
                        -float("inf")
                    )
                )

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment `emars_ptr`
        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[:,None] * batch_size
        cids_inc_ptr += TILE_SIZE_K

    if propagation_alg_id == 2:
        # Compute p_i^{1/alpha}
        acc *= (1.0 / alpha)

    # Write back
    off_nids = tl.load(nids + nblock_id)
    offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    if pflow_tempered_enabled:
        tl.store(node_mars_tempered + offs_nmars, acc_tempered, mask = mask_batch[None,:])


# @triton.jit
@triton_jit
def _fw_triton_block_sparse_csmm1_kernel(node_mars, element_mars, mparams, nids, cids_start, cids_increment,
                                        pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                        BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                        TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, use_bf16: tl.constexpr,
                                        propagation_alg_id: tl.constexpr, pflow_tempered_enabled: tl.constexpr, 
                                        pid_m_offset = 0, alpha = 0.0, pflow_temperature = 1.0, node_mars_tempered = None):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Node offsets
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

    # Edge offsets
    offs_edge = tl.arange(0, TILE_SIZE_K)

    # Initialize pointers to `params`
    offs_estart = nblock_id * TILE_SIZE_K + offs_edge
    offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
    par_start = tl.load(pids_start + offs_estart)
    epars_ptr = mparams + \
        offs_node[:,None] + \
        par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    edge_start = tl.load(cids_start + offs_estart)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    if pflow_tempered_enabled:
        acc_tempered = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr)
        emars = tl.load(emars_ptr, mask = mask_batch[None,:])

        if propagation_alg_id == 1:
            # MPE propagation method
            lpars = tl.log(epars)
            nmars = tl.max(lpars[:,:,None] + emars[None,:,:], axis = 1)

            acc = tl.maximum(acc, nmars)

        else:

            if propagation_alg_id == 0:
                # LL propagation method
                emars_max = tl.max(emars, axis = 0)[None,:]
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

            if propagation_alg_id == 2:
                # GeneralLL propagation method

                emars_max = tl.max(emars, axis = 0)[None,:]
                # Compute p_i^{alpha} for every i
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp((emars - emars_max) * alpha), 0.0)
                # Compute w_i^{alpha} for every i
                epars = tl.exp(tl.log(epars) * alpha)

                # Also scale `emars_max`
                emars_max *= alpha

            if use_bf16 == 1:
                # Simulated matmul kernel + bfloat16
                epars = epars.to(tl.bfloat16)
                emars_sub = emars_sub.to(tl.bfloat16)
                nmars = tl.sum(epars[:,:,None] * emars_sub[None,:,:], axis = 1).to(tl.float32)
            else:
                # Simulated matmul kernel + float32
                nmars = tl.sum(epars[:,:,None] * emars_sub[None,:,:], axis = 1)

            acc = tl.where(emars_max > acc,
                tl.log(nmars + tl.exp(acc - emars_max) + 1e-24) + emars_max,
                tl.where(acc != -float("inf"),
                    tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc,
                    -float("inf")
                )
            )

            if pflow_tempered_enabled:
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp((emars - emars_max) / pflow_temperature), 0.0)
                epars = tlmath.pow(epars, 1.0 / pflow_temperature)
                emars_max /= pflow_temperature

                if use_bf16 == 1:
                    # Simulated matmul kernel + bfloat16
                    epars = epars.to(tl.bfloat16)
                    emars_sub = emars_sub.to(tl.bfloat16)
                    nmars = tl.sum(epars[:,:,None] * emars_sub[None,:,:], axis = 1).to(tl.float32)
                else:
                    # Simulated matmul kernel + float32
                    nmars = tl.sum(epars[:,:,None] * emars_sub[None,:,:], axis = 1)

                acc_tempered = tl.where(emars_max > acc_tempered,
                    tl.log(nmars + tl.exp(acc_tempered - emars_max) + 1e-24) + emars_max,
                    tl.where(acc_tempered != -float("inf"),
                        tl.log(tl.exp(emars_max - acc_tempered) * nmars + 1.0) + acc_tempered,
                        -float("inf")
                    )
                )

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment `emars_ptr`
        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[:,None] * batch_size
        cids_inc_ptr += TILE_SIZE_K

    if propagation_alg_id == 2:
        # Compute p_i^{1/alpha}
        acc *= (1.0 / alpha)

    # Write back
    off_nids = tl.load(nids + nblock_id)
    offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    if pflow_tempered_enabled:
        tl.store(node_mars_tempered + offs_nmars, acc_tempered, mask = mask_batch[None,:])


# @triton.jit
@triton_jit
def _fw_triton_block_sparse_csmm2_kernel(node_mars, element_mars, mparams, nids, cids_start, cids_increment,
                                         pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                         BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                         TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, use_bf16: tl.constexpr,
                                         propagation_alg_id: tl.constexpr, pflow_tempered_enabled: tl.constexpr, 
                                         pid_m_offset = 0, alpha = 0.0, pflow_temperature = 1.0, node_mars_tempered = None):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Node offsets
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

    # Edge offsets
    offs_edge = tl.arange(0, TILE_SIZE_K)

    # Initialize pointers to `params`
    offs_estart = nblock_id * TILE_SIZE_K + offs_edge
    offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
    par_start = tl.load(pids_start + offs_estart)
    epars_ptr = mparams + \
        offs_node[:,None] + \
        par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    edge_start = tl.load(cids_start + offs_estart)
    emars_ptr = element_mars + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]

    # Batch increment pointers
    pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    if pflow_tempered_enabled:
        acc_tempered = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr)
        emars = tl.load(emars_ptr, mask = mask_batch[:,None])

        if propagation_alg_id == 1:
            # MPE propagation method
            lpars = tl.log(epars)
            nmars = tl.max(lpars[:,:,None] + tl.trans(emars)[None,:,:], axis = 1)

            acc = tl.maximum(acc, nmars)

        else:

            if propagation_alg_id == 0:
                # LL propagation method
                emars_max = tl.max(emars, axis = 1)
                emars_sub = tl.where(emars_max[:,None] != -float("inf"), tl.exp(emars - emars_max[:,None]), 0.0)

            if propagation_alg_id == 2:
                # GeneralLL propagation method

                emars_max = tl.max(emars, axis = 1)
                # Compute p_i^{alpha} for every i
                emars_sub = tl.where(emars_max[:,None] != -float("inf"), tl.exp((emars - emars_max[:,None]) * alpha), 0.0)
                # Compute w_i^{alpha} for every i
                epars = tl.exp(tl.log(epars) * alpha)

                # Also scale `emars_max`
                emars_max *= alpha

            # Simulated matmul kernel + float32
            nmars = tl.sum(epars[:,:,None] * tl.trans(emars_sub)[None,:,:], axis = 1)

            acc = tl.where(emars_max[None,:] > acc,
                tl.log(nmars + tl.exp(acc - emars_max[None,:]) + 1e-24) + emars_max[None,:],
                tl.where(acc != -float("inf"), 
                    tl.log(tl.exp(emars_max[None,:] - acc) * nmars + 1.0) + acc,
                    -float("inf")
                )
            )

            if pflow_tempered_enabled:
                emars_sub = tl.where(emars_max != -float("inf"), tl.exp((emars - emars_max) / pflow_temperature), 0.0)
                epars = tlmath.pow(epars, 1.0 / pflow_temperature)
                emars_max /= pflow_temperature

                # Simulated matmul kernel + float32
                nmars = tl.sum(epars[:,:,None] * tl.trans(emars_sub)[None,:,:], axis = 1)

                acc_tempered = tl.where(emars_max[None,:] > acc_tempered,
                    tl.log(nmars + tl.exp(acc_tempered - emars_max[None,:]) + 1e-24) + emars_max[None,:],
                    tl.where(acc_tempered != -float("inf"), 
                        tl.log(tl.exp(emars_max[None,:] - acc_tempered) * nmars + 1.0) + acc_tempered,
                        -float("inf")
                    )
                )

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment `emars_ptr`
        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[None,:] * batch_size
        cids_inc_ptr += TILE_SIZE_K

    if propagation_alg_id == 2:
        # Compute p_i^{1/alpha}
        acc *= (1.0 / alpha)

    # Write back
    off_nids = tl.load(nids + nblock_id)
    offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    if pflow_tempered_enabled:
        tl.store(node_mars_tempered + offs_nmars, acc_tempered, mask = mask_batch[None,:])
