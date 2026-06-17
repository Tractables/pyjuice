"""
Forward pass: sparse sum kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``fw_sparse._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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



@triton_jit
def _fw_triton_sparse_kernel(node_mars, element_mars, mparams, nids, cids, pids,
                             local_ids, batch_size, partial_eval: tl.constexpr, num_edges: tl.constexpr, 
                             BLOCK_B: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, propagation_alg_id: tl.constexpr, 
                             pflow_tempered_enabled: tl.constexpr, alpha = 0.0, node_mars_tempered = None,
                             pflow_temperature = 1.0):
    
    pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(axis = 1) # ID of size-`BLOCK_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, num_edges)
    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_ptr = mparams + par_start # [num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize and load edge mars
    edge_ids = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_ids[:,None] * batch_size + \
        offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

    # Compute max and subtract (only when using LL or GeneralLL propagation method)
    if propagation_alg_id == 0:
        emars_max = tl.max(emars, axis = 0)
        emars = tl.exp(emars - emars_max[None,:])
    
    if propagation_alg_id == 2:
        emars_max = tl.max(emars, axis = 0)
        emars = tl.exp((emars - emars_max[None,:]) * alpha)
        emars_max *= alpha

    # Initialize pointers to `node_mars`
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + \
        off_nids * batch_size + \
        offs_batch

    if pflow_tempered_enabled:
        nmars_tempered_ptr = node_mars_tempered + \
            off_nids * batch_size + \
            offs_batch

    # Inner loop
    for i in range(0, BLOCK_SIZE_M):
        epars = tl.load(epars_ptr)

        if propagation_alg_id == 0:
            nmars = tl.where(emars_max == -float("inf"), -float("inf"), tl.log(tl.sum(emars * epars[:,None], axis = 0)) + emars_max)

        if propagation_alg_id == 1:
            nmars = tl.max(emars + tl.log(epars)[:,None], axis = 0)

        if propagation_alg_id == 2:
            epars = tl.exp(tl.log(epars) * alpha)

            nmars = (tl.log(tl.sum(emars * epars[:,None], axis = 0)) + emars_max) * (1.0 / alpha)

        tl.store(nmars_ptr, nmars, mask = mask_batch)

        if pflow_tempered_enabled:
            nmars_tempered = tl.log(tl.sum(tlmath.pow(emars * epars[:,None], 1.0 / pflow_temperature), axis = 0)) + emars_max / pflow_temperature

            tl.store(nmars_tempered_ptr, nmars_tempered, mask = mask_batch)

        # Increment `epars_ptr`
        epars_ptr += 1

        # Increment `nmars_ptr`
        nmars_ptr += batch_size

        if pflow_tempered_enabled:
            nmars_tempered_ptr += batch_size


# @triton.jit
@triton_jit
def _fw_triton_large_sparse_kernel(node_mars, element_mars, mparams, nids, cids, pids, local_ids, batch_size, 
                                   num_nodes, pid_m_offset, partial_eval: tl.constexpr, num_edges: tl.constexpr, BLOCK_B: tl.constexpr, 
                                   TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, propagation_alg_id: tl.constexpr, 
                                   pflow_tempered_enabled: tl.constexpr, alpha = 0.0, node_mars_tempered = None,
                                   pflow_temperature = 1.0):

    pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(axis = 1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
    mask_m = offs_m < num_nodes

    # Get inferred node block id from `pid_m`
    nblock_ids = offs_m // BLOCK_SIZE_M
    tile_ids = offs_m % BLOCK_SIZE_M

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_ids = tl.load(local_ids + nblock_ids, mask = mask_m)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, num_edges)
    par_start = tl.load(pids + nblock_ids[:,None] * num_edges + offs_edge[None,:], mask = mask_m[:,None]) # [TILE_SIZE_M, num_edges]
    epars = tl.load(mparams + tile_ids[:,None] * BLOCK_SIZE_M + par_start, mask = mask_m[:,None], other = 0.0) # [TILE_SIZE_M, num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize and load edge mars
    edge_ids = tl.load(cids + nblock_ids[:,None] * num_edges + offs_edge[None,:]) # [TILE_SIZE_M, num_edges]
    emars_ptr = element_mars + \
        edge_ids[:,:,None] * batch_size + \
        offs_batch[None,None,:] # [TILE_SIZE_M, num_edges, BLOCK_B]
    emars = tl.load(emars_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:]), other = 0.0) # [TILE_SIZE_M, num_edges, BLOCK_B]

    # Compute max and subtract (only when using LL or GeneralLL propagation method)
    if propagation_alg_id == 0:
        emars_max = tl.max(emars, axis = 1)
        emars = tl.exp(emars - emars_max[:,None,:])

    if propagation_alg_id == 2:
        emars_max = tl.max(emars, axis = 1)
        emars = tl.exp((emars - emars_max[:,None,:]) * alpha)
        emars_max *= alpha

    # Compute sum node marginals
    if propagation_alg_id == 0:
        nmars = tl.where(emars_max == -float("inf"), -float("inf"), tl.log(tl.sum(emars * epars[:,:,None], axis = 1)) + emars_max)

    if propagation_alg_id == 1:
        nmars = tl.max(emars + tl.log(epars)[:,:,None], axis = 1)

    if propagation_alg_id == 2:
        epars = tl.exp(tl.log(epars) * alpha)

        nmars = (tl.log(tl.sum(emars * epars[:,:,None], axis = 1)) + emars_max) * (1.0 / alpha)

    # Initialize pointers to `node_mars`
    off_nids = tl.load(nids + nblock_ids) # [TILE_SIZE_M]
    nmars_ptr = node_mars + \
        (off_nids + tile_ids)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]

    tl.store(nmars_ptr, nmars, mask = (mask_m[:,None] & mask_batch[None,:]))

    if pflow_tempered_enabled:
        nmars_tempered = tl.log(tl.sum(tlmath.pow(emars * epars[:,None], 1.0 / pflow_temperature), axis = 0)) + emars_max / pflow_temperature

        nmars_ptr = node_mars_tempered + \
            (off_nids + tile_ids)[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]

        tl.store(nmars_tempered_ptr, nmars_tempered, mask = (mask_m[:,None] & mask_batch[None,:]))
