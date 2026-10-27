"""
What `BlockScaleSumParams` does after a layer's flows, as ONE launch per partition: the log-Z half of the gate
gradient, the denominator flow of `apply_z_correction`, and the `-log Z` shift that gives `node_mars` back.

They ran as three launches (`_bs_triton_phigrad_logz_kernel`, `_bs_triton_denom_w_kernel`, `shift_logz`), each a
few microseconds of work behind a launch -- MEASURED on a tied gated HMM-4096 at B 128, 9.7 + 12 + 3.8 us a step
against ~1-2 us of traffic each -- and the first two build the SAME operands from the same loads:

    v[n, b] = exp(node_flows[n,b] - log Z[n,b] + mx[b]),     p[g, b] = exp(log phi[g,b] - mx[b])

with `mx[b]` the largest `log phi` over the tile's gates, a shift that cancels exactly in both products (the
bounds are `_bs_triton_denom_w_kernel`'s). The gradient contracts them over the nodes, the denominator over the
batch:

    d LL / d log phi[g, b] -= p[g,b] * sum_n sigma[n,g] v[n,b]
    W[n, g]                += sum_b v[n,b] p[g,b]                       (atomic when batch slices share one)

and the shift, `node_mars[n, b] -= log Z[n, b]`, is an elementwise pass over the same (node, sample) tile, done by
the programs of the first gate tile so each entry is written once. It reads only `log Z`, never what the two
contractions read from `node_mars` -- they read none of it -- so it may run in the same program.

THE GRADIENT IS DETERMINISTIC, as the log-Z kernel's was (one program per node block, so one writer of each entry):
a gradient entry sums over the node block's rows, which several node tiles hold, and their atomic adds landed in an
order that varies from run to run -- two equal backwards differed in the last bit. Where a block is several tiles
(`GRAD_PARTIAL`), each STORES its share (`partial`, `[node blocks, tiles, gates, batch]`) and takes a ticket from its
group's counter (the programs of one node block, gate tile and batch slice); the program holding the last ticket adds
the shares in tile order -- the one writer of each entry -- and sets the counter back to 0 for the next launch. The
tickets are an acq_rel atomic after a CTA barrier, the shares read past L1 (`.cg`). A second launch for the sum
instead: MEASURED ~12 us a layer more (its grid is the gradient's, a few programs, each walking every tile); the node
tiles looping inside one program per block instead: 3-10x slower (a block's walk is latency-bound at any tile).
`W` keeps an atomic add across batch slices, as `_bs_triton_denom_w_kernel` did across its.

Not bit-identical to the three launches: the gate tile (and with it `mx`) and the batch slicing can differ, which
reorders the sums -- agreement to rounding. The shift is: one add, as in `shift_logz`.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def _bs_post_backward_kernel(node_flows, node_mars, log_z, sigma, ext, gate, nids, grad_ext, W, partial, counter,
                             batch_size: tl.constexpr, n_gates: tl.constexpr, N_CHILD_GATES: tl.constexpr,
                             BLOCK_SIZE_M: tl.constexpr, gate_stride: tl.constexpr, ext_base,
                             TILE_M: tl.constexpr, TILE_G: tl.constexpr, TILE_B: tl.constexpr,
                             B_TILES_PER_PROG: tl.constexpr, WANT_GRAD: tl.constexpr, WANT_W: tl.constexpr,
                             W_ATOMIC: tl.constexpr, USE_DOT: tl.constexpr, GRAD_IEEE: tl.constexpr,
                             W_IEEE: tl.constexpr, GRAD_PARTIAL: tl.constexpr):
    pid_y = tl.program_id(0)                       # (node block, node tile): grid-x, past the 65535 of grid-y
    pid_g = tl.program_id(1)                       # gate tile
    pid_s = tl.program_id(2)                       # batch slice

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_M
    nb = pid_y // M_TILES
    offs_node = (pid_y % M_TILES) * TILE_M + tl.arange(0, TILE_M)
    # rows of `node_flows` / `node_mars` (the circuit's), and of `log Z`, `sigma` and `W` (the partition's)
    nrows = (tl.load(nids + nb) + offs_node).to(tl.int64)
    zrows = nb.to(tl.int64) * BLOCK_SIZE_M + offs_node

    # A gate's row in `ext`: the gate table's entry for its edge block (bounded by the table's width, as in
    # `_bs_triton_phigrad_logz_kernel`: `n_gates` comes from the power-of-two padded edge count) plus its place
    offs_g = pid_g * TILE_G + tl.arange(0, TILE_G)
    mask_g = offs_g < n_gates
    j = offs_g // N_CHILD_GATES
    gbase = tl.load(gate + nb * gate_stride + j, mask = mask_g & (j < gate_stride), other = -1)
    grow = gbase + ext_base + offs_g % N_CHILD_GATES
    ghas = mask_g & (gbase >= 0)

    if WANT_GRAD:
        sg = tl.load(sigma + zrows[:,None] * n_gates + offs_g[None,:], mask = mask_g[None,:], other = 0.0)  # [M, G]
    if WANT_W:
        acc = tl.zeros([TILE_M, TILE_G], dtype = tl.float32)

    offs_b = pid_s * (B_TILES_PER_PROG * TILE_B) + tl.arange(0, TILE_B)
    for _ in range(B_TILES_PER_PROG):
        mask_b = offs_b < batch_size
        lz = tl.load(log_z + zrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:], other = 0.0)

        if pid_g == 0:
            nm_ptr = node_mars + nrows[:,None] * batch_size + offs_b[None,:]
            tl.store(nm_ptr, tl.load(nm_ptr, mask = mask_b[None,:], other = 0.0) - lz, mask = mask_b[None,:])

        if WANT_GRAD or WANT_W:
            nf = tl.load(node_flows + nrows[:,None] * batch_size + offs_b[None,:], mask = mask_b[None,:],
                         other = -float("inf"))
            lphi = tl.load(ext + grow[:,None] * batch_size + offs_b[None,:], mask = ghas[:,None] & mask_b[None,:],
                           other = -float("inf"))                                               # [G, B]
            mx = tl.max(lphi, axis = 0)
            mx = tl.where(mx == -float("inf"), 0.0, mx)
            v = tl.where(mask_b[None,:], tl.exp(nf - lz + mx[None,:]), 0.0)                     # [M, B]
            p = tl.where(ghas[:,None] & mask_b[None,:], tl.exp(lphi - mx[None,:]), 0.0)         # [G, B]

            if WANT_GRAD:
                if USE_DOT:
                    if GRAD_IEEE == 2:
                        out = tl.dot(tl.trans(sg), v, input_precision = "tf32x3")               # [G, B]
                    elif GRAD_IEEE:
                        out = tl.dot(tl.trans(sg), v, input_precision = "ieee")
                    else:
                        out = tl.dot(tl.trans(sg), v)
                else:
                    # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                    out = tl.sum(sg[:,:,None] * v[:,None,:], axis = 0)
                if GRAD_PARTIAL:
                    # this node tile's share, at [node block, tile, gate, sample]: every (gate < n_gates, sample)
                    # entry is written, so the sum below reads no stale scratch
                    pptr = partial + (pid_y.to(tl.int64) * n_gates + offs_g[:,None]) * batch_size + offs_b[None,:]
                    tl.store(pptr, -p * out, mask = mask_g[:,None] & mask_b[None,:])
                else:
                    tl.atomic_add(grad_ext + grow[:,None] * batch_size + offs_b[None,:], -p * out,
                                  mask = ghas[:,None] & mask_b[None,:])

            if WANT_W:
                if USE_DOT:
                    if W_IEEE == 2:
                        acc += tl.dot(v, tl.trans(p), input_precision = "tf32x3")               # [M, G]
                    elif W_IEEE:
                        acc += tl.dot(v, tl.trans(p), input_precision = "ieee")
                    else:
                        acc += tl.dot(v, tl.trans(p))
                else:
                    acc += tl.sum(tl.trans(v)[:,:,None] * tl.trans(p)[:,None,:], axis = 0)

        offs_b += TILE_B

    if WANT_GRAD and GRAD_PARTIAL:
        tl.debug_barrier()                         # every thread's shares stored before the ticket releases them
        group = (nb * tl.num_programs(1) + pid_g) * tl.num_programs(2) + pid_s
        ticket = tl.atomic_add(counter + group, 1, sem = "acq_rel")
        if ticket == M_TILES - 1:
            offs_b = pid_s * (B_TILES_PER_PROG * TILE_B) + tl.arange(0, TILE_B)
            for _ in range(B_TILES_PER_PROG):
                mask_b = offs_b < batch_size
                pmask = mask_g[:,None] & mask_b[None,:]
                pptr = partial + (nb.to(tl.int64) * M_TILES * n_gates + offs_g[:,None]) * batch_size + offs_b[None,:]
                tot = tl.zeros([TILE_G, TILE_B], dtype = tl.float32)
                if M_TILES <= 64:
                    for t in tl.static_range(M_TILES):
                        tot += tl.load(pptr + t * n_gates * batch_size, mask = pmask, other = 0.0,
                                       cache_modifier = ".cg")
                else:
                    for t in range(M_TILES):
                        tot += tl.load(pptr + t * n_gates * batch_size, mask = pmask, other = 0.0,
                                       cache_modifier = ".cg")
                gptr = grad_ext + grow[:,None] * batch_size + offs_b[None,:]
                gmask = ghas[:,None] & mask_b[None,:]
                tl.store(gptr, tl.load(gptr, mask = gmask, other = 0.0) + tot, mask = gmask)
                offs_b += TILE_B
            tl.atomic_xchg(counter + group, 0)

    if WANT_W:
        # ACCUMULATE, never store (`W` is the circuit's persistent denominator buffer); atomic only where batch
        # slices hold partial sums of the same entries
        wptr = W + zrows[:,None] * n_gates + offs_g[None,:]
        if W_ATOMIC:
            tl.atomic_add(wptr, acc, mask = mask_g[None,:])
        else:
            tl.store(wptr, tl.load(wptr, mask = mask_g[None,:], other = 0.0) + acc, mask = mask_g[None,:])

