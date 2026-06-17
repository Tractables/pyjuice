"""
Backward pass: parameter flows, sparse kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``bk_par_sparse._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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
def _bk_triton_sparse_par_kernel(node_flows, node_mars, element_mars, mparams, param_flows, nids, cids, pids, pfids,
                                 pid_m_offset, num_edges: tl.constexpr, batch_size: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                 logspace_flows: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr, 
                                 TILE_SIZE_B: tl.constexpr, B_NUM_BLOCKS: tl.constexpr, propagation_alg_id: tl.constexpr, 
                                 negate_pflows: tl.constexpr, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` samples
    pid_e = tl.program_id(1) # ID of size-`BLOCK_K` edges
    pid_m = tl.program_id(2) + pid_m_offset # ID of size-`BLOCK_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // BLOCK_M
    tile_id = pid_m % BLOCK_M

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, BLOCK_K) + pid_e * BLOCK_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [BLOCK_K, BLOCK_B]

    # Initialize pointers to `node_flows` and `node_mars`
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]
    nflows_ptr = node_flows + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]

    if propagation_alg_id == 1:
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_ptr = mparams + par_start + tile_id
        epars = tl.load(epars_ptr) # [BLOCK_K]
        elpars = tl.log(epars)

    # Inner loop
    acc = tl.zeros([BLOCK_K], dtype = tl.float32)

    for b in range(0, B_NUM_BLOCKS):
        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B + b * BLOCK_B
        mask_batch = offs_batch < batch_size

        emars = tl.load(emars_ptr, mask = mask_batch[None,:], other = -float("inf")) # [BLOCK_K, BLOCK_B]

        if propagation_alg_id == 1:
            nmars = tl.load(nmars_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]
            nflows = tl.load(nflows_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]

            if logspace_flows:
                acc += tl.sum(tl.where(tl.abs(elpars[:,None] + emars - nmars[None,:]) < 1e-6, tl.exp(nflows[None,:]), 0.0), axis = 1)
            else:
                acc += tl.sum(tl.where(tl.abs(elpars[:,None] + emars - nmars[None,:]) < 1e-6, nflows[None,:], 0.0), axis = 1)

        else:

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch, other = -float("inf")) # [BLOCK_B]

                if propagation_alg_id == 0:
                    pflows = tl.sum(tl.exp(emars + log_n_fdm[None,:]), axis = 1)
                
                if propagation_alg_id == 2:
                    nmars = tl.load(nmars_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]
                    pflows = tl.sum(tl.exp(emars + log_n_fdm[None,:] + (alpha - 1.0) * nmars[None,:]), axis = 1)
            else:
                nmars = tl.load(nmars_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]
                nflows = tl.load(nflows_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]

                if logspace_flows:
                    plflows = nflows[None,:] + emars - nmars[None,:]
                    plflows = tl.where(nmars[None,:] == -float("inf"),
                        nflows[None,:],
                        plflows
                    )
                    plflows_max = tl.max(plflows, axis = 1)
                    pflows = tl.where(plflows_max == -float("inf"),
                        0.0,
                        tl.exp(tl.log(tl.sum(tl.exp(plflows - plflows_max[:,None]), axis = 1)) + plflows_max)
                    )
                else:
                    pflows = tl.sum(nflows[None,:] * tl.exp(emars - nmars[None,:]), axis = 1)

            acc += pflows

        # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
        emars_ptr += BLOCK_B
        nmars_ptr += BLOCK_B
        nflows_ptr += BLOCK_B

    if propagation_alg_id != 1:
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_ptr = mparams + par_start + tile_id
        epars = tl.load(epars_ptr) # [BLOCK_K]

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_ptr = param_flows + parflow_start + tile_id

    if propagation_alg_id != 1:
        curr_pflows = acc * epars
    else:
        curr_pflows = acc

    if negate_pflows:
        tl.atomic_add(eparflows_ptr, -1.0 * curr_pflows)
    else:
        tl.atomic_add(eparflows_ptr, curr_pflows)


@triton_jit
def _bk_triton_sparse_tempered_par_kernel(node_flows, node_mars_tempered, element_mars, mparams, param_flows, nids, cids, pids, pfids,
                                          pid_m_offset, num_edges: tl.constexpr, batch_size: tl.constexpr, BLOCK_M: tl.constexpr, 
                                          BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr, TILE_SIZE_B: tl.constexpr, B_NUM_BLOCKS: tl.constexpr, 
                                          negate_pflows: tl.constexpr, pflow_temperature = 1.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` samples
    pid_e = tl.program_id(1) # ID of size-`BLOCK_K` edges
    pid_m = tl.program_id(2) + pid_m_offset # ID of size-`BLOCK_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // BLOCK_M
    tile_id = pid_m % BLOCK_M

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, BLOCK_K) + pid_e * BLOCK_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [BLOCK_K, BLOCK_B]

    # Initialize pointers to `node_flows` and `node_mars_tempered`
    off_nids = tl.load(nids + nblock_id)
    nmars_tempered_ptr = node_mars_tempered + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]
    nflows_ptr = node_flows + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]

    # Inner loop
    acc = tl.zeros([BLOCK_K], dtype = tl.float32)

    for b in range(0, B_NUM_BLOCKS):
        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B + b * BLOCK_B
        mask_batch = offs_batch < batch_size

        emars = tl.load(emars_ptr, mask = mask_batch[None,:], other = -float("inf")) # [BLOCK_K, BLOCK_B]

        nmars_tempered = tl.load(nmars_tempered_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]
        nflows = tl.load(nflows_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]

        plflows = nflows[None,:] + emars / pflow_temperature - nmars_tempered[None,:]
        plflows = tl.where(nmars_tempered[None,:] == -float("inf"),
            nflows[None,:],
            plflows
        )
        plflows_max = tl.max(plflows, axis = 1)
        pflows = tl.where(plflows_max == -float("inf"),
            0.0,
            tl.exp(tl.log(tl.sum(tl.exp(plflows - plflows_max[:,None]), axis = 1)) + plflows_max)
        )

        acc += pflows

        # Increment `emars_ptr`, `nmars_tempered_ptr`, and `nflows_ptr`
        emars_ptr += BLOCK_B
        nmars_tempered_ptr += BLOCK_B
        nflows_ptr += BLOCK_B

    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_ptr = mparams + par_start + tile_id
    epars = tl.load(epars_ptr) # [BLOCK_K]

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_ptr = param_flows + parflow_start + tile_id

    curr_pflows = acc * tlmath.pow(epars, 1.0 / pflow_temperature)

    if negate_pflows:
        tl.atomic_add(eparflows_ptr, -1.0 * curr_pflows)
    else:
        tl.atomic_add(eparflows_ptr, curr_pflows)
