"""
Forward/backward kernels for product layers.

Triton kernels factored out of ``prod_layer.py``. Invoked from there as
``kernels._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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
def _forward_backward_kernel_3d(node_vals_ptr, element_vals_ptr, local_ids_ptr, nids_ptr, cids_ptr, tot_n_nodes, tot_n_eles, n_nblocks,
                                num_edges: tl.constexpr, batch_size, BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr, 
                                block_size: tl.constexpr, accum: tl.constexpr, partial_eval: tl.constexpr, prop_logsumexp: tl.constexpr):
    """
    This kernel implements the function with 3d tensors. However, it only work with `triton==2.0.0`.
    """
    
    pid_m = tl.program_id(axis = 0) # ID of size-`BLOCK_M` nodes
    pid_b = tl.program_id(axis = 1) # ID of size-`BLOCK_B` batches

    if block_size >= BLOCK_M:

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (block_size // BLOCK_M)
        ntile_id = pid_m % (block_size // BLOCK_M)

        # For partial evaluation
        if partial_eval:
            nblock_id = tl.load(local_ids_ptr + nblock_id)

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B 
        mask_batch = offs_batch < batch_size

        # Get the block start ids for the children
        # To make the triton compiler happy, we reload every index `BLOCK_M` times
        offs_ne = tl.arange(0, num_edges * BLOCK_M) // BLOCK_M
        offs_ne = tl.view(offs_ne, (BLOCK_M, num_edges))
        offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_ne) # [BLOCK_M, num_edges]

        # Get the edge values from child nodes
        block_nids = tl.arange(0, BLOCK_M) + ntile_id * BLOCK_M
        offs_evals = offs_egstart + block_nids[:,None]
        evals = tl.load(element_vals_ptr + offs_evals[None,:,:] * batch_size + offs_batch[:,None,None], mask = mask_batch[:,None,None])

        if prop_logsumexp:
            # Take the logsumexp of the child nodes' values
            evals_max = tl.max(evals, axis = 2)
            nvals = tl.log(tl.sum(tl.exp(evals - evals_max[:,:,None]), axis = 2)) + evals_max
        else:
            # Take the sum of the child nodes' values
            nvals = tl.sum(evals, axis = 2)

        # Node ids to `node_vals_ptr`
        nblock_start = tl.load(nids_ptr + nblock_id)
        offs_nvals = (nblock_start + block_nids[None,:]) * batch_size + offs_batch[:,None]

        # Accumulate the `node_vals` if required
        if accum:
            node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch[:,None], other = 0)
            
            if prop_logsumexp:
                # logaddexp
                diff = nvals - node_vals
                nvals = tl.where(
                    diff == 0, 
                    nvals + 0.69314718055994530942, # log(2)
                    tl.where(
                        diff > 0,
                        nvals + tlmath.log1p(tl.exp(-diff)),
                        node_vals + tlmath.log1p(tl.exp(diff))
                    )
                )
            else:
                # sum
                nvals += node_vals

        tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch[:,None])

    else:

        # Node offsets and mask
        offs_node = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
        mask_node = offs_node < n_nblocks * block_size

        # Inferred block ids
        nblock_ids = offs_node // block_size

        # For partial evaluation
        if partial_eval:
            nblock_ids = tl.load(local_ids_ptr + nblock_ids, mask = mask_node)

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B 
        mask_batch = offs_batch < batch_size

        # Get the block start ids for the children
        offs_ne = tl.arange(0, num_edges * BLOCK_M) // BLOCK_M
        offs_ne = tl.view(offs_ne, (BLOCK_M, num_edges))
        offs_egstart = tl.load(cids_ptr + nblock_ids[:,None] * num_edges + offs_ne, mask = mask_node[:,None]) # [BLOCK_M, num_edges]

        # Get the edge values from child nodes
        block_nids = (offs_node % block_size)
        offs_evals = offs_egstart + block_nids[:,None]
        evals = tl.load(element_vals_ptr + offs_evals[None,:,:] * batch_size + offs_batch[:,None,None], mask = (mask_batch[:,None,None] & mask_node[None,:,None]))

        if prop_logsumexp:
            # Take the logsumexp of the child nodes' values
            evals_max = tl.max(evals, axis = 2)
            nvals = tl.log(tl.sum(tl.exp(evals - evals_max[:,:,None]), axis = 2)) + evals_max
        else:
            # Take the sum of the child nodes' values
            nvals = tl.sum(evals, axis = 2)

        # Node ids to `node_vals_ptr`
        nblock_start = tl.load(nids_ptr + nblock_ids[None,:])
        offs_nvals = (nblock_start + block_nids[None,:]) * batch_size + offs_batch[:,None]

        # Accumulate the `node_vals` if required
        if accum:
            node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch[:,None], other = 0)
            
            if prop_logsumexp:
                # logaddexp
                diff = node_vals - nvals
                nvals = tl.where(
                    nvals == -float("inf"), 
                    node_vals,
                    tl.where(
                        diff > 0,
                        node_vals + tlmath.log1p(tl.exp(-diff)),
                        nvals + tlmath.log1p(tl.exp(diff))
                    )
                )
            else:
                # sum
                nvals += node_vals

        tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch[:,None])


# @triton.jit
@triton_jit
def _forward_backward_kernel_2d(node_vals_ptr, element_vals_ptr, local_ids_ptr, nids_ptr, cids_ptr, tot_n_nodes, tot_n_eles, n_nblocks,
                                num_edges: tl.constexpr, batch_size, BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr, 
                                block_size: tl.constexpr, accum: tl.constexpr, partial_eval: tl.constexpr, prop_logsumexp: tl.constexpr):
    """
    This kernel implements the function with 2d tensors. It works for all `triton` versions.
    """

    pid_m = tl.program_id(axis = 0) # ID of size-`BLOCK_M` nodes
    pid_b = tl.program_id(axis = 1) # ID of size-`BLOCK_B` batches

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (block_size // BLOCK_M)
    ntile_id = pid_m % (block_size // BLOCK_M)

    # For partial evaluation
    if partial_eval:
        nblock_id = tl.load(local_ids_ptr + nblock_id)

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B # [BLOCK_B]
    mask_batch = offs_batch < batch_size

    # Get the block start ids for the children
    offs_edge = tl.arange(0, num_edges)
    offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_edge) # [num_edges]

    # Base ptr for ch values
    offs_evals = (offs_egstart[:,None] + ntile_id * BLOCK_M) * batch_size + offs_batch[None,:] # [num_edges, BLOCK_B]

    # Base ptr for par values
    nblock_start = tl.load(nids_ptr + nblock_id)
    offs_nvals = (nblock_start + ntile_id * BLOCK_M) * batch_size + offs_batch # [BLOCK_B]

    # Inner loop
    for i in range(0, BLOCK_M):
        evals = tl.load(element_vals_ptr + offs_evals, mask = mask_batch[None,:], other = 0)
        
        if prop_logsumexp:
            # Take the logsumexp of the child nodes' values
            evals_max = tl.max(evals, axis = 0)
            nvals = tl.where(evals_max != -float("inf"), tl.log(tl.sum(tl.exp(evals - evals_max[None,:]), axis = 0)) + evals_max, -float("inf"))
        else:
            # Take the sum of the child nodes' values
            nvals = tl.sum(evals, axis = 0)

        # Accumulate the `node_vals` if required
        if accum:
            node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch)

            if prop_logsumexp:
                # logaddexp
                diff = node_vals - nvals
                nvals = tl.where(
                    nvals == -float("inf"), 
                    node_vals,
                    tl.where(
                        diff > 0,
                        node_vals + tlmath.log1p(tl.exp(-diff)),
                        nvals + tlmath.log1p(tl.exp(diff))
                    )
                )
            else:
                # sum
                nvals += node_vals

        tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch)

        offs_nvals += batch_size
        offs_evals += batch_size


# @triton.jit
@triton_jit
def _forward_backward_kernel_large(node_vals_ptr, element_vals_ptr, local_ids_ptr, nids_ptr, cids_ptr, tot_n_nodes, tot_n_eles, n_nblocks,
                                   num_edges: tl.constexpr, batch_size, BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr, 
                                   N_NUM_BLKS: tl.constexpr, block_size: tl.constexpr, accum: tl.constexpr, partial_eval: tl.constexpr,
                                   prop_logsumexp: tl.constexpr):
    """
    This kernel implements the function with 2d tensors. It is designed for nodes with many edges.
    """
    
    pid_m = tl.program_id(axis = 0) # ID of size-`1` nodes
    pid_b = tl.program_id(axis = 1) # ID of size-`BLOCK_B` batches

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // block_size
    ntile_id = pid_m % block_size

    # For partial evaluation
    if partial_eval:
        nblock_id = tl.load(local_ids_ptr + nblock_id)

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B # [BLOCK_B]
    mask_batch = offs_batch < batch_size

    # Get the block start ids for the children
    offs_edge = tl.arange(0, BLOCK_N)
    mask_edge = (offs_edge < num_edges)
    offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_edge) # [BLOCK_N]

    # Base ptr for ch values
    offs_evals = (offs_egstart[:,None] + ntile_id) * batch_size + offs_batch[None,:] # [BLOCK_N, BLOCK_B]

    # Base ptr for par values
    nblock_start = tl.load(nids_ptr + nblock_id)
    offs_nvals = (nblock_start + ntile_id) * batch_size + offs_batch # [BLOCK_B]

    # Prepare buffer
    if prop_logsumexp:
        nvals = tl.zeros([BLOCK_B], dtype = tl.float32) - float("inf")
    else:
        nvals = tl.zeros([BLOCK_B], dtype = tl.float32)

    # Inner loop
    for i in range(0, N_NUM_BLKS):
        evals = tl.load(element_vals_ptr + offs_evals, mask = (mask_edge[:,None] & mask_batch[None,:]), other = 0)
        
        if prop_logsumexp:
            # Take the logsumexp of the child nodes' values
            evals_max = tl.max(evals, axis = 0)
            nvals_sub = tl.where(evals_max != -float("inf"), tl.sum(tl.exp(evals - evals_max[None,:]), axis = 0), 0.0)
            nvals = tl.where(evals_max > nvals,
                tl.log(nvals_sub + tl.exp(nvals - evals_max) + 1e-24) + evals_max,
                tl.log(tl.exp(evals_max - nvals) * nvals_sub + 1.0) + nvals
            )
        else:
            # Take the sum of the child nodes' values
            nvals += tl.sum(evals, axis = 0)

        offs_edge += BLOCK_N
        mask_edge = (offs_edge < num_edges)

        # Re-compute the ch value ids
        offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_edge) # [BLOCK_N]
        offs_evals = (offs_egstart[:,None] + ntile_id) * batch_size + offs_batch[None,:] # [BLOCK_N, BLOCK_B]

    # Accumulate the `node_vals` if required
    if accum:
        node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch)
        
        if prop_logsumexp:
            # logaddexp
            diff = node_vals - nvals
            nvals = tl.where(
                nvals == -float("inf"), 
                node_vals,
                tl.where(
                    diff > 0,
                    node_vals + tlmath.log1p(tl.exp(-diff)),
                    nvals + tlmath.log1p(tl.exp(diff))
                )
            )
        else:
            # sum
            nvals += node_vals

    tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch)
