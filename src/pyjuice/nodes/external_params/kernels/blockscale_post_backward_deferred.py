"""
`_bs_post_backward_kernel`'s work for every gated layer of a backward pass at once, after its last layer (the
deferred post-backward, `pyjuice.nodes.external_params.blockscale.post_deferred`): the log-Z half of the gate
gradient and the `-log Z` shift in one launch over (layer, node block, gate tile, batch slice), the denominator flow
`W` in one launch per `W` over (node tile, gate tile).

Per layer the post-backward is a few microseconds of work spread over a grid of short programs, the gradient summed
over a node block's tiles through stored shares and a ticket (MEASURED, a tied gated HMM-4096 at B 128: 25 us a
layer). Over all the layers at once there are programs enough for the loops the per-layer kernel could not afford:

  * the gradient: a program per (layer, node block, gate tile, batch slice) walks the block's node tiles itself and
    adds the shares in tile order -- what the last ticket of the per-layer kernel adds, so the same bits, with no
    shares stored and no tickets. Where two of the launch's node blocks write one gate row -- copies of a node whose
    gates are tied (`tie_external`): in several layers, or in one -- they add atomically (`GRAD_ATOMIC`), in no
    fixed order, as a layer's own kernel does where its node blocks share rows: a load and a store there kept one
    program's terms (MEASURED, a tied chain of three: the tied gradient off the sum of the copies' by 2.9-7.7 times its
    largest entry);
  * `W`: a program per (node tile, gate tile) walks the layers in the pass's order and their batch slices in order,
    adding each slice's sum as the per-layer kernel did -- one writer, so deterministic: the per-layer kernel's bits
    where a layer is one batch slice, and where it is several (its slices then add atomically, in no fixed order)
    the bits of one order those adds can take.

Every tile computes what the per-layer kernel's computes -- the same tile shapes (`TILE_M` / `TILE_G` / `TILE_B`, the
per-layer plan's), the same shift `mx` over the gate tile, the same three-pass TF32 dots -- read through per-layer
tables of pointers (`log Z`, `sigma`, the gate table, the node block rows) and `ext` bases.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton.jit
def _mul_rn(a, b):
    """``a * b`` rounded on its own: the per-layer kernel stores its shares `-p * out` before it adds them, so the add
    here must not fuse with the product into one FMA (its single rounding is other bits)."""
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype = tl.float32, is_pure = True,
                                     pack = 1)


@triton_jit
def _bs_post_deferred_grad_kernel(node_flows, node_mars, lz_ptrs, sg_ptrs, ext, gate_ptrs, nids_ptrs, ext_bases,
                                  grad_ext, batch_size: tl.constexpr, n_gates: tl.constexpr,
                                  N_CHILD_GATES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, gate_stride: tl.constexpr,
                                  ROWS: tl.constexpr, TILE_M: tl.constexpr, TILE_G: tl.constexpr, TILE_B: tl.constexpr,
                                  B_TILES_PER_PROG: tl.constexpr, WANT_GRAD: tl.constexpr, USE_DOT: tl.constexpr,
                                  NIDS_I64: tl.constexpr, GATE_I64: tl.constexpr, GRAD_ATOMIC: tl.constexpr):
    pid_y = tl.program_id(0)                       # (layer, node block)
    pid_g = tl.program_id(1)                       # gate tile
    pid_s = tl.program_id(2)                       # batch slice

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

    # The gate rows in `ext`, as the per-layer kernel bounds them by the gate table's width
    offs_g = pid_g * TILE_G + tl.arange(0, TILE_G)
    mask_g = offs_g < n_gates
    j = offs_g // N_CHILD_GATES
    gbase = tl.load(gate + nb * gate_stride + j, mask = mask_g & (j < gate_stride), other = -1)
    grow = gbase + ext_base + offs_g % N_CHILD_GATES
    ghas = mask_g & (gbase >= 0)
    if WANT_GRAD:
        sigma = tl.load(sg_ptrs + mem).to(tl.pointer_type(tl.float32))

    offs_b = pid_s * (B_TILES_PER_PROG * TILE_B) + tl.arange(0, TILE_B)
    for _ in range(B_TILES_PER_PROG):
        mask_b = offs_b < batch_size
        if WANT_GRAD:
            lphi = tl.load(ext + grow[:,None] * batch_size + offs_b[None,:], mask = ghas[:,None] & mask_b[None,:],
                           other = -float("inf"))                                               # [G, B]
            mx = tl.max(lphi, axis = 0)
            mx = tl.where(mx == -float("inf"), 0.0, mx)
            p = tl.where(ghas[:,None] & mask_b[None,:], tl.exp(lphi - mx[None,:]), 0.0)         # [G, B]
            tot = tl.zeros([TILE_G, TILE_B], dtype = tl.float32)

        for t in range(M_TILES):
            offs_node = t * TILE_M + tl.arange(0, TILE_M)
            nrows = (nstart + offs_node).to(tl.int64)
            zrows = nb.to(tl.int64) * BLOCK_SIZE_M + offs_node
            lz = tl.load(log_z + zrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:], other = 0.0)

            if pid_g == 0:
                nm_ptr = node_mars + nrows[:,None] * batch_size + offs_b[None,:]
                tl.store(nm_ptr, tl.load(nm_ptr, mask = mask_b[None,:], other = 0.0) - lz, mask = mask_b[None,:])

            if WANT_GRAD:
                sg = tl.load(sigma + zrows[:,None] * n_gates + offs_g[None,:], mask = mask_g[None,:], other = 0.0)
                nf = tl.load(node_flows + nrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:],
                             other = -float("inf"))
                v = tl.where(mask_b[None,:], tl.exp(nf - lz + mx[None,:]), 0.0)                 # [M, B]
                if USE_DOT:
                    out = tl.dot(tl.trans(sg), v, input_precision = "tf32x3")                   # [G, B]
                else:
                    # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                    out = tl.sum(sg[:,:,None] * v[:,None,:], axis = 0)
                tot += _mul_rn(-p, out)

        if WANT_GRAD:
            gptr = grad_ext + grow[:,None] * batch_size + offs_b[None,:]
            gmask = ghas[:,None] & mask_b[None,:]
            if GRAD_ATOMIC:                        # (another program of the launch adds into these rows too)
                tl.atomic_add(gptr, tot, mask = gmask)
            else:
                tl.store(gptr, tl.load(gptr, mask = gmask, other = 0.0) + tot, mask = gmask)
        offs_b += TILE_B


@triton_jit
def _bs_post_deferred_w_kernel(node_flows, lz_ptrs, ext, gate_ptrs, nids_ptrs, ext_bases, w_ptrs, w_off,
                               batch_size: tl.constexpr, n_gates: tl.constexpr, N_CHILD_GATES: tl.constexpr,
                               BLOCK_SIZE_M: tl.constexpr, gate_stride: tl.constexpr, TILE_M: tl.constexpr,
                               TILE_G: tl.constexpr, TILE_B: tl.constexpr, B_TILES_PER_PROG: tl.constexpr,
                               N_SLICES: tl.constexpr, USE_DOT: tl.constexpr, NIDS_I64: tl.constexpr,
                               GATE_I64: tl.constexpr):
    pid_y = tl.program_id(0)                       # (node block, node tile): rows of `W`
    pid_g = tl.program_id(1)                       # gate tile
    pid_w = tl.program_id(2)                       # which `W`: its layers are rows w_off[pid_w] .. w_off[pid_w + 1]

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_M
    nb = pid_y // M_TILES
    offs_node = (pid_y % M_TILES) * TILE_M + tl.arange(0, TILE_M)
    zrows = nb.to(tl.int64) * BLOCK_SIZE_M + offs_node
    offs_g = pid_g * TILE_G + tl.arange(0, TILE_G)
    mask_g = offs_g < n_gates
    j = offs_g // N_CHILD_GATES

    W = tl.load(w_ptrs + pid_w).to(tl.pointer_type(tl.float32))
    wptr = W + zrows[:,None] * n_gates + offs_g[None,:]
    run = tl.load(wptr, mask = mask_g[None,:], other = 0.0)
    for mem in range(tl.load(w_off + pid_w), tl.load(w_off + pid_w + 1)):     # (the layers in the pass's order)
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
        nrows = (tl.load(nids + nb) + offs_node).to(tl.int64)
        gbase = tl.load(gate + nb * gate_stride + j, mask = mask_g & (j < gate_stride), other = -1)
        grow = gbase + ext_base + offs_g % N_CHILD_GATES
        ghas = mask_g & (gbase >= 0)

        for s in range(N_SLICES):                  # (each slice's sum added as the per-layer kernel adds it)
            acc = tl.zeros([TILE_M, TILE_G], dtype = tl.float32)
            offs_b = s * (B_TILES_PER_PROG * TILE_B) + tl.arange(0, TILE_B)
            for _ in range(B_TILES_PER_PROG):
                mask_b = offs_b < batch_size
                lz = tl.load(log_z + zrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:], other = 0.0)
                nf = tl.load(node_flows + nrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:],
                             other = -float("inf"))
                lphi = tl.load(ext + grow[:,None] * batch_size + offs_b[None,:], mask = ghas[:,None] & mask_b[None,:],
                               other = -float("inf"))                                           # [G, B]
                mx = tl.max(lphi, axis = 0)
                mx = tl.where(mx == -float("inf"), 0.0, mx)
                v = tl.where(mask_b[None,:], tl.exp(nf - lz + mx[None,:]), 0.0)                 # [M, B]
                p = tl.where(ghas[:,None] & mask_b[None,:], tl.exp(lphi - mx[None,:]), 0.0)     # [G, B]
                if USE_DOT:
                    acc += tl.dot(v, tl.trans(p), input_precision = "tf32x3")                   # [M, G]
                else:
                    acc += tl.sum(tl.trans(v)[:,:,None] * tl.trans(p)[:,None,:], axis = 0)
                offs_b += TILE_B
            run += acc
    tl.store(wptr, run, mask = mask_g[None,:])
