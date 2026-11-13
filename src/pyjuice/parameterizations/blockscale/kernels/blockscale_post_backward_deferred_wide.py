"""
The deferred post-backward's gradient (`blockscale_post_backward_deferred`'s `_bs_post_deferred_grad_kernel`) with a
program over several gate tiles of its node block: that kernel's arithmetic tile by tile, its tiles' loads once.

There a program takes one gate tile (`TILE_G` of a node block's gates) and reads the block's `log Z` and node flows once
per gate tile. Here a program holds `NGT` gate tiles at once -- 3-D tiles [gate tile, ...], each gate tile's slice what
that kernel computes for it: its own shift `mx` (the max over ITS gates), its products, its
`[TILE_G, TILE_M] x [TILE_M, TILE_B]` tf32x3 dot -- a batched `tl.dot`, MEASURED each batch element's 2-D dot to the bit
(Triton 3.7, every tile shape here) -- added in that kernel's order over the node tiles. `TILE_B` is the dots' N
dimension, and each sample's shift is its own: a free dimension, narrower here to keep the 3-D tiles in registers. The
same bits; a node tile's `log Z` and flows loaded once for `NGT` gate tiles, the node values shifted by the programs of
the first ones. The dot form only (`USE_DOT`: the plan's tiles at least 16), and the gradient wanted (the shift alone is
the other kernel's).
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from .blockscale_post_backward_deferred import _mul_rn


@triton_jit
def _bs_post_deferred_grad_wide_kernel(node_flows, node_mars, lz_ptrs, sg_ptrs, ext, gate_ptrs, nids_ptrs, ext_bases,
                                       grad_ext, batch_size: tl.constexpr, n_gates: tl.constexpr,
                                       N_CHILD_GATES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                                       gate_stride: tl.constexpr, ROWS: tl.constexpr, TILE_M: tl.constexpr,
                                       TILE_G: tl.constexpr, TILE_B: tl.constexpr, B_TILES_PER_PROG: tl.constexpr,
                                       NGT: tl.constexpr, NIDS_I64: tl.constexpr, GATE_I64: tl.constexpr,
                                       GRAD_ATOMIC: tl.constexpr):
    pid_y = tl.program_id(0)                       # (layer, node block)
    pid_s = tl.program_id(1)                       # batch slice
    pid_k = tl.program_id(2)                       # its gate tiles: NGT from pid_k * NGT

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_M
    mem = pid_y // ROWS
    nb = pid_y % ROWS
    log_z = tl.load(lz_ptrs + mem).to(tl.pointer_type(tl.float32))
    if NIDS_I64:
        nids = tl.load(nids_ptrs + mem).to(tl.pointer_type(tl.int64))
    else:
        nids = tl.load(nids_ptrs + mem).to(tl.pointer_type(tl.int32))
    if GATE_I64:
        gate = tl.load(gate_ptrs + mem).to(tl.pointer_type(tl.int64))
    else:
        gate = tl.load(gate_ptrs + mem).to(tl.pointer_type(tl.int32))
    ext_base = tl.load(ext_bases + mem)
    nstart = tl.load(nids + nb)

    # Every gate tile's gate rows, [NGT, TILE_G]: as the per-layer kernel bounds them by the gate table's width
    offs_g = (pid_k * NGT + tl.arange(0, NGT))[:,None] * TILE_G + tl.arange(0, TILE_G)[None,:]
    mask_g = offs_g < n_gates
    j = offs_g // N_CHILD_GATES
    gbase = tl.load(gate + nb * gate_stride + j, mask = mask_g & (j < gate_stride), other = -1)
    grow = gbase + ext_base + offs_g % N_CHILD_GATES
    ghas = mask_g & (gbase >= 0)
    sigma = tl.load(sg_ptrs + mem).to(tl.pointer_type(tl.float32))

    offs_b = pid_s * (B_TILES_PER_PROG * TILE_B) + tl.arange(0, TILE_B)
    for _ in range(B_TILES_PER_PROG):
        mask_b = offs_b < batch_size
        gb_mask = ghas[:,:,None] & mask_b[None,None,:]
        lphi = tl.load(ext + grow[:,:,None] * batch_size + offs_b[None,None,:], mask = gb_mask,
                       other = -float("inf"))                                                   # [NGT, G, B]
        mx = tl.max(lphi, axis = 1)                                                             # (each tile's own)
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        p = tl.where(gb_mask, tl.exp(lphi - mx[:,None,:]), 0.0)                                 # [NGT, G, B]
        tot = tl.zeros([NGT, TILE_G, TILE_B], dtype = tl.float32)

        for t in range(M_TILES):
            offs_node = t * TILE_M + tl.arange(0, TILE_M)
            nrows = (nstart + offs_node).to(tl.int64)
            zrows = nb.to(tl.int64) * BLOCK_SIZE_M + offs_node
            lz = tl.load(log_z + zrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:], other = 0.0)

            if pid_k == 0:                         # (the shift once a node: the first gate tiles' program)
                nm_ptr = node_mars + nrows[:,None] * batch_size + offs_b[None,:]
                tl.store(nm_ptr, tl.load(nm_ptr, mask = mask_b[None,:], other = 0.0) - lz, mask = mask_b[None,:])

            sg = tl.load(sigma + zrows[None,:,None] * n_gates + offs_g[:,None,:], mask = mask_g[:,None,:],
                         other = 0.0)                                                           # [NGT, M, G]
            nf = tl.load(node_flows + nrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:],
                         other = -float("inf"))
            v = tl.where(mask_b[None,None,:], tl.exp(nf[None,:,:] - lz[None,:,:] + mx[:,None,:]), 0.0)  # [NGT, M, B]
            out = tl.dot(tl.permute(sg, (0, 2, 1)), v, input_precision = "tf32x3")             # [NGT, G, B]
            tot += _mul_rn(-p, out)

        gptr = grad_ext + grow[:,:,None] * batch_size + offs_b[None,None,:]
        if GRAD_ATOMIC:                            # (another program of the launch adds into these rows too)
            tl.atomic_add(gptr, tot, mask = gb_mask)
        else:
            tl.store(gptr, tl.load(gptr, mask = gb_mask, other = 0.0) + tot, mask = gb_mask)
        offs_b += TILE_B
