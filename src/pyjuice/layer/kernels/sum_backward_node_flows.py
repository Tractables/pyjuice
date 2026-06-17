"""
Backward pass: node-flow modification kernels.

Triton kernels factored out of ``sum_layer.py``. Invoked from there as
``bk_nflows._<kernel_name>[grid](...)``. To add a kernel for this case, define it
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
def _bk_triton_modify_flow_kernel(node_flows, node_mars, local_ids, nids, batch_size: tl.constexpr, partial_eval: tl.constexpr, 
                                  BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, propagation_alg_id: tl.constexpr, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` examples
    pid_m = tl.program_id(1) # ID of size-`BLOCK_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // BLOCK_M)
    tile_id = pid_m % (BLOCK_SIZE_M // BLOCK_M)

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_flows` and `node_mars`
    offs_node = tl.arange(0, BLOCK_M) + tile_id * BLOCK_M
    off_nids = tl.load(nids + nblock_id)
    offs_nmfs = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

    nmars = tl.load(node_mars + offs_nmfs, mask = mask_batch[None,:])
    nflows = tl.load(node_flows + offs_nmfs, mask = mask_batch[None,:])

    if propagation_alg_id == 0:
        lflows = tl.log(nflows)
        uflows = tl.where(nmars != -float("inf"), lflows - nmars, -float("inf"))

    if propagation_alg_id == 1:
        uflows = nflows

    if propagation_alg_id == 2:
        lflows = tl.log(nflows)
        uflows = tl.where(nmars != -float("inf"), lflows - nmars * alpha, -float("inf"))

    tl.store(node_flows + offs_nmfs, uflows, mask = mask_batch[None,:])


# @triton.jit
@triton_jit
def _bk_triton_large_modify_flow_kernel(node_flows, node_mars, local_ids, nids, num_nodes, batch_size: tl.constexpr, partial_eval: tl.constexpr, 
                                        BLOCK_B: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, propagation_alg_id: tl.constexpr, 
                                        pid_m_offset = 0, alpha = 0.0):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` examples
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
    mask_m = offs_m < num_nodes

    # Get inferred node block id from `pid_m`
    nblock_ids = offs_m // BLOCK_SIZE_M
    tile_ids = offs_m % BLOCK_SIZE_M

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_ids = tl.load(local_ids + nblock_ids)

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_flows` and `node_mars`
    off_nids = tl.load(nids + nblock_ids, mask = mask_m) # [TILE_SIZE_M]
    offs_nmfs = (off_nids + tile_ids)[:,None] * batch_size + offs_batch[None,:]

    nmars = tl.load(node_mars + offs_nmfs, mask = (mask_m[:,None] & mask_batch[None,:]))
    nflows = tl.load(node_flows + offs_nmfs, mask = (mask_m[:,None] & mask_batch[None,:]))

    if propagation_alg_id == 0:
        lflows = tl.log(nflows)
        uflows = tl.where(nmars != -float("inf"), lflows - nmars, -float("inf"))

    if propagation_alg_id == 1:
        uflows = nflows

    if propagation_alg_id == 2:
        lflows = tl.log(nflows)
        uflows = tl.where(nmars != -float("inf"), lflows - nmars * alpha, -float("inf"))

    tl.store(node_flows + offs_nmfs, uflows, mask = (mask_m[:,None] & mask_batch[None,:]))
