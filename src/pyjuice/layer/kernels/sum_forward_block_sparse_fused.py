"""
Forward pass: the block-sparse sum kernel of the ``tlmm`` regime with the product layer below fused in.

A copy of `sum_forward_block_sparse._fw_triton_block_sparse_tlmm_kernel` for LL propagation without tempering that
reads an element's value as the sum of its product's child node values -- what the product layers' forward writes
into ``element_mars`` -- so that a caller whose element values are stale (a decoder session, which recomputes only
the rows a call reaches) need not launch those forwards first. The children come from the product layer group's
table ``[n, K_CH]`` of node rows per element row ``first_row .. first_row + n`` (`pyjuice.layer.fused_children.
element_children`), padded with the dummy node (row 0 of ``node_mars``, value 0); the dummy element (row 0, which
the sum tables' padding points at) stays -inf by its row id and is not looked up. The values are the product kernel's
to the bit for products of up to two children, to fp32 rounding beyond (`_children_sum`). Nothing is written to
``element_mars``: in this regime nothing reads it (the fused backward forms the values the same way), and the caller
computes the products' values itself where a pass does.

Reached only through `SumLayer.forward(..., compute_children = table)`: the kernels training runs are untouched.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.layer.kernels import _round_to_tf32


@triton.jit
def _child(node_mars, child_ptrs, j: tl.constexpr, row_mask, offs_batch, mask_batch, batch_size: tl.constexpr):
    """``[rows, BLOCK_B]``: the ``j``-th child's node values, 0 where the table holds no child (row 0, the padding --
    that load is masked off, not made) or the row itself is masked."""
    c = tl.load(child_ptrs + j, mask = row_mask, other = 0)
    return tl.load(node_mars + c[:,None] * batch_size + offs_batch[None,:], mask = (c > 0)[:,None] & mask_batch[None,:],
                   other = 0.0)


@triton.jit
def _children_sum(node_mars, child_ptrs, row_mask, offs_batch, mask_batch, batch_size: tl.constexpr, K_CH: tl.constexpr):
    """``[rows, BLOCK_B]``: the sum over the ``K_CH`` children at ``child_ptrs`` (``[rows]`` pointers into a table's
    rows; a row with ``row_mask`` False, the dummy element, reads nothing) of their node values. Added by recursive
    halving -- ``(c0 + c2) + (c1 + c3)`` for four -- the order `tl.sum` reduces the product kernel's children in at
    most widths, so the value is the product kernel's to the bit for products of up to two children (one add) and to
    fp32 rounding beyond (the order `tl.sum` takes depends on its layout). Each add follows its operands' loads, so
    that few child tiles are live at once (MEASURED: eight loads then the adds ran 2-7% slower)."""
    if K_CH == 1:
        return _child(node_mars, child_ptrs, 0, row_mask, offs_batch, mask_batch, batch_size)
    if K_CH == 2:
        return _child(node_mars, child_ptrs, 0, row_mask, offs_batch, mask_batch, batch_size) + \
               _child(node_mars, child_ptrs, 1, row_mask, offs_batch, mask_batch, batch_size)
    if K_CH == 4:
        a = _child(node_mars, child_ptrs, 0, row_mask, offs_batch, mask_batch, batch_size) + \
            _child(node_mars, child_ptrs, 2, row_mask, offs_batch, mask_batch, batch_size)
        b = _child(node_mars, child_ptrs, 1, row_mask, offs_batch, mask_batch, batch_size) + \
            _child(node_mars, child_ptrs, 3, row_mask, offs_batch, mask_batch, batch_size)
        return a + b
    a = _child(node_mars, child_ptrs, 0, row_mask, offs_batch, mask_batch, batch_size) + \
        _child(node_mars, child_ptrs, 4, row_mask, offs_batch, mask_batch, batch_size)
    b = _child(node_mars, child_ptrs, 2, row_mask, offs_batch, mask_batch, batch_size) + \
        _child(node_mars, child_ptrs, 6, row_mask, offs_batch, mask_batch, batch_size)
    a = a + b
    b = _child(node_mars, child_ptrs, 1, row_mask, offs_batch, mask_batch, batch_size) + \
        _child(node_mars, child_ptrs, 5, row_mask, offs_batch, mask_batch, batch_size)
    c = _child(node_mars, child_ptrs, 3, row_mask, offs_batch, mask_batch, batch_size) + \
        _child(node_mars, child_ptrs, 7, row_mask, offs_batch, mask_batch, batch_size)
    acc = a + (b + c)
    for j in range(8, K_CH):                                        # beyond eight: in order
        acc += _child(node_mars, child_ptrs, j, row_mask, offs_batch, mask_batch, batch_size)
    return acc


# @triton.jit
@triton_jit
def _fw_triton_block_sparse_tlmm_fused_kernel(node_mars, mparams, nids, cids_start, cids_increment,
                                              pids_start, pids_increment, ele_children, ele_first_row, batch_size: tl.constexpr,
                                              BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                              TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, K_CH: tl.constexpr,
                                              use_bf16: tl.constexpr, pid_m_offset = 0, fp32_dot: tl.constexpr = False):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

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

    # The element rows of this tile's edges (the kernel this copies keeps a pointer into `element_mars`; here the rows
    # themselves, which index the children table) -- int32, as every id here: 64-bit addresses spilled registers
    edge_rows = tl.load(cids_start + offs_estart) # [TILE_SIZE_K]

    # Batch increment pointers
    pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr).to(tl.float32)

        # The element values: the sum of the children's node values (`_children_sum`), the dummy element -inf
        emars = _children_sum(node_mars, ele_children + (edge_rows - ele_first_row) * K_CH, edge_rows != 0, offs_batch,
                              mask_batch, batch_size, K_CH)
        emars = tl.where(edge_rows[:,None] == 0, -float("inf"), emars)

        # LL propagation method
        emars_max = tl.max(emars, axis = 0)[None,:]
        emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

        if use_bf16 == 1:
            # Built-in matmul kernel of triton + float16
            epars_bf16 = epars.to(tl.bfloat16)
            emars_bf16 = emars_sub.to(tl.bfloat16)
            nmars = tl.dot(epars_bf16, emars_bf16).to(tl.float32)
        elif fp32_dot:
            nmars = tl.dot(epars, emars_sub, input_precision = "tf32x3")
        else:
            # Built-in matmul kernel of triton + float32 (a TF32 dot: see `_round_to_tf32`)
            nmars = tl.dot(_round_to_tf32(epars), _round_to_tf32(emars_sub))

        acc = tl.where(emars_max > acc,
            tl.log(nmars + tl.exp(acc - emars_max) + 1e-24) + emars_max,
            tl.where(acc != -float("inf"),
                tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc,
                -float("inf")
            )
        )

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment the element rows
        cids_inc = tl.load(cids_inc_ptr)
        edge_rows += cids_inc
        cids_inc_ptr += TILE_SIZE_K

    # Write back
    off_nids = tl.load(nids + nblock_id)
    offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])
