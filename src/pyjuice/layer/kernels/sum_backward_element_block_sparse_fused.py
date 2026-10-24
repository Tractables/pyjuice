"""
Backward pass: the block-sparse element-flow kernel of the ``tlmm`` regime with the product layer below fused in.

A copy of `sum_backward_element_block_sparse._bk_triton_block_sparse_ele_kernel` for LL propagation (log-space or
linear flows, `allow_modify_flows` / `allow_neg_flows` / `accumulate_ch_flows` as there) that forms its child
elements' values as the sum of their products' child node values instead of reading ``element_mars`` -- so a caller
whose element values are stale (a decoder session) need not recompute the products before the backward. The children
come from the product layer group's table ``[n, K_CH]`` of node rows per element row ``first_row .. first_row + n``
(`pyjuice.layer.fused_children.element_children`), padded with the dummy node (row 0 of ``node_mars``, value 0);
the values are the product kernel's to the bit for products of up to two children, to fp32 rounding beyond
(`_children_sum`).

Reached only through `SumLayer.backward(..., compute_children = table)`: the kernels training runs are untouched.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.layer.kernels import _round_to_tf32
from .sum_forward_block_sparse_fused import _children_sum


# @triton.jit
@triton_jit
def _bk_triton_block_sparse_ele_fused_kernel(node_flows, element_flows, node_mars, mparams, chids, parids_start,
                                             parids_increment, parpids_start, parpids_increment, ele_children,
                                             ele_first_row, batch_size: tl.constexpr, ptr_inc_step: tl.constexpr,
                                             allow_modify_flows: tl.constexpr, logspace_flows: tl.constexpr,
                                             BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                             TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                                             BLOCK_SIZE_K: tl.constexpr, K_CH: tl.constexpr, TL_DOT: tl.constexpr,
                                             accumulate_ch_flows: tl.constexpr, allow_neg_flows: tl.constexpr,
                                             pid_m_offset = 0, DOT_TF32X3: tl.constexpr = False):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

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

    # The element values of this tile's rows: the sum of their children's node values (the kernel this copies reads
    # them from `element_mars`)
    off_eleids = tl.load(chids + eleblock_id)
    ele_rows = off_eleids + offs_ele # [TILE_SIZE_M], int32 as every id here (64-bit addresses spill registers)
    emars = _children_sum(node_mars, ele_children + (ele_rows - ele_first_row) * K_CH, ele_rows != 0, offs_batch,
                          mask_batch, batch_size, K_CH)
    emars = tl.where(ele_rows[:,None] == 0, -float("inf"), emars)

    # Inner loop
    if logspace_flows:
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")
    else:
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]

        if allow_modify_flows == 1:
            log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        else:
            nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
            nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

            if logspace_flows:
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)
            elif allow_neg_flows:
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), -nmars)
            else:
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
        n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

        if allow_neg_flows:
            if TL_DOT == 1:
                if DOT_TF32X3:
                    partial_flows = tl.dot(epars, n_fdm_sub * nflows, input_precision = "tf32x3")
                else:
                    partial_flows = tl.dot(_round_to_tf32(epars), _round_to_tf32(n_fdm_sub * nflows))
            else:
                partial_flows = tl.sum(tl.trans(epars)[:,:,None] * (n_fdm_sub * nflows)[:,None,:], axis = 0)
        else:
            if TL_DOT == 1:
                if DOT_TF32X3:
                    partial_flows = tl.dot(epars, n_fdm_sub, input_precision = "tf32x3")
                else:
                    partial_flows = tl.dot(_round_to_tf32(epars), _round_to_tf32(n_fdm_sub))
            else:
                # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                partial_flows = tl.sum(tl.trans(epars)[:,:,None] * n_fdm_sub[:,None,:], axis = 0)

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
