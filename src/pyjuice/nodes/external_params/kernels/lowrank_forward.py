from __future__ import annotations

import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton.jit
def _logaddexp(a, b):
    """
    Max-stabilized `log(exp(a) + exp(b))`, defined when both arguments are `-inf`.

    The shift is forced to 0 when the max is `-inf`, so the differences stay `-inf` (giving 0 after
    `exp`) instead of becoming `-inf - -inf = nan`. That case is not exotic here: it is exactly the
    identity initialization, where the whole correction vanishes.
    """
    m = tl.maximum(a, b)
    m = tl.where(m == float("-inf"), 0.0, m)

    return m + tl.log(tl.exp(a - m) + tl.exp(b - m))


@triton.jit
def _logsumexp(x, axis: tl.constexpr):
    """Max-stabilized `logsumexp` along `axis`, defined when a whole slice is `-inf`."""
    m = tl.max(x, axis = axis)
    m = tl.where(m == float("-inf"), 0.0, m)

    return m + tl.log(tl.sum(tl.exp(x - tl.expand_dims(m, axis)), axis = axis))


@triton_jit
def _fw_lowrank_kernel(node_mars_ptr, element_mars_ptr, ext_ptr, nids_ptr, cids_ptr, xu_ptr, xv_ptr,
                       batch_size, num_edges, ext_base, BLOCK_SIZE: tl.constexpr, CH_BLOCK_SIZE: tl.constexpr,
                       RANK: tl.constexpr, MAX_N_EBLKS: tl.constexpr, TILE_M: tl.constexpr,
                       TILE_B: tl.constexpr, TILE_C: tl.constexpr):
    """
    Turn the shared-parameter node values into the effective ones under a per-sample low-rank
    correction, in a single pass.

    On entry `node_mars` holds `log S1`, the value under the shared parameters alone; on exit it holds

        log( S1 + S2 ) - log Z,     S2 = sum_{e,r} exp(V) * W,     Z = 1 + sum_{e,r} exp(V) * A

    with `W = sum_c exp(U) * child` and `A = sum_c exp(U)`. `A` is the normalizer's inner term and is
    accumulated in the same `c` loop as `W`, from the same loads -- so normalizing on the fly costs one
    extra accumulator and no extra memory traffic.

    :note: this kernel is READ-MODIFY-WRITE on `node_mars` -- it reads `log S1` and overwrites the same
           slots -- so running it twice on the same buffer applies the correction twice. That is safe
           inside `ExternalParamsSumLayer.forward`, which re-runs the shared forward (rewriting
           `log S1`) beforehand, but it means a launch-config autotuner must NOT simply time repeated
           launches: it has to either time the whole `layer.forward` or run into a scratch `node_mars`,
           the way the sum layer's own CUDA autotuners use a scratch buffer.

    Grid is `(node block row, batch tile, node tile)`. `W` and `A` do not depend on the node index, so
    they are recomputed once per node tile: the overhead relative to the shared sum kernel is
    `2*RANK/TILE_M + RANK/CH_BLOCK_SIZE`, i.e. large `TILE_M` is cheap and small `TILE_M` is not.

    That recomputation is a re-READ of `U`, not just arithmetic, and it is what bounds this variant: at
    `block_size = 1024, TILE_M = 32` it reads `U` 32 times, which on a 31-layer HMM is 4.2 GB and
    predicts 2.9 ms against 3.8 ms measured. It is nonetheless the best of the single-launch forms,
    because the alternatives lose more to occupancy than they save.
    """
    pid_r = tl.program_id(axis = 0)
    pid_b = tl.program_id(axis = 1)
    pid_m = tl.program_id(axis = 2)

    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)
    mask_b = offs_b < batch_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_rank = tl.arange(0, RANK)

    log_s2 = tl.full([TILE_M, TILE_B], float("-inf"), dtype = tl.float32)
    log_zt = tl.full([TILE_M, TILE_B], float("-inf"), dtype = tl.float32)

    for j in range(MAX_N_EBLKS):
        xu = tl.load(xu_ptr + pid_r * MAX_N_EBLKS + j)
        xv = tl.load(xv_ptr + pid_r * MAX_N_EBLKS + j)

        # Rows with fewer than `MAX_N_EBLKS` incident edge blocks are padded with -1; clamp so the
        # loads stay in bounds and discard the result instead
        is_edge_block = xu >= 0

        # `xu` / `xv` are relative to this layer's block of the staging buffer
        xu = tl.maximum(xu, 0) + ext_base
        xv = tl.maximum(xv, 0) + ext_base

        ## `log W` and `log A`, reducing over this edge block's children ##

        log_w = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)
        log_a = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)

        for c_start in range(0, CH_BLOCK_SIZE, TILE_C):
            offs_c = c_start + tl.arange(0, TILE_C)

            # Edge block `j` occupies compiled edge slots [j*CH_BLOCK_SIZE, (j+1)*CH_BLOCK_SIZE)
            cids = tl.load(cids_ptr + pid_r * num_edges + j * CH_BLOCK_SIZE + offs_c)
            emars = tl.load(element_mars_ptr + cids[:,None] * batch_size + offs_b[None,:],
                            mask = mask_b[None,:], other = float("-inf"))                    # [TILE_C, TILE_B]

            u_offs = xu + offs_c[:,None] * RANK + offs_rank[None,:]                    # [TILE_C, RANK]
            u = tl.load(ext_ptr + u_offs[:,:,None] * batch_size + offs_b[None,None,:],
                        mask = mask_b[None,None,:], other = float("-inf"))                   # [TILE_C, RANK, TILE_B]

            log_w = _logaddexp(log_w, _logsumexp(u + emars[:,None,:], 0))
            log_a = _logaddexp(log_a, _logsumexp(u, 0))

        ## Combine with the node-side factors ##

        v_offs = xv + offs_m[:,None] * RANK + offs_rank[None,:]                        # [TILE_M, RANK]
        v = tl.load(ext_ptr + v_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:], other = float("-inf"))                       # [TILE_M, RANK, TILE_B]

        s2 = _logsumexp(v + log_w[None,:,:], 1)
        zt = _logsumexp(v + log_a[None,:,:], 1)

        log_s2 = _logaddexp(log_s2, tl.where(is_edge_block, s2, float("-inf")))
        log_zt = _logaddexp(log_zt, tl.where(is_edge_block, zt, float("-inf")))

    ## Combine with the shared term and normalize ##

    nid = tl.load(nids_ptr + pid_r)
    mars_ptr = node_mars_ptr + (nid + offs_m)[:,None] * batch_size + offs_b[None,:]

    log_s1 = tl.load(mars_ptr, mask = mask_b[None,:], other = 0.0)

    # The shared parameters contribute exactly 1 to the normalizer -- PyJuice keeps them
    # child-normalized -- so their log-contribution is 0
    log_z = _logaddexp(tl.zeros([TILE_M, TILE_B], dtype = tl.float32), log_zt)

    tl.store(mars_ptr, _logaddexp(log_s1, log_s2) - log_z, mask = mask_b[None,:])


@triton_jit
def _fw_lowrank_single_kernel(node_mars_ptr, element_mars_ptr, ext_ptr, nids_ptr, cids_ptr, xu_ptr, xv_ptr,
                              batch_size, num_edges, ext_base, BLOCK_SIZE: tl.constexpr,
                              CH_BLOCK_SIZE: tl.constexpr, RANK: tl.constexpr, TILE_M: tl.constexpr,
                              TILE_B: tl.constexpr, TILE_C: tl.constexpr):
    """
    The single-edge-block case, with `log W` / `log A` HOISTED out of the node dimension.

    `log W` and `log A` do not depend on the node index, so computing them once and then walking the
    node tiles *inside* the program removes the `BLOCK_SIZE / TILE_M` redundant recomputation that the
    general kernel pays. For a dense transition (`block_size = num_latents`, one edge block) that
    factor is large -- 32x at `block_size = 1024, TILE_M = 32` -- and it dominates everything else.

    The trade is parallelism: the grid loses its node dimension, leaving `rows x batch tiles`. MEASURED,
    that trade is catastrophic on the shape it was written for -- a dense 1024-wide transition with one
    node block gives 2-8 programs on 188 SMs, and this variant came in at 32x the shared forward, an
    order of magnitude WORSE than just paying the redundancy. It is kept only because it is the cleanest
    statement of the redundancy-free form; prefer `"split"`, which is redundancy-free AND parallel.

    Same read-modify-write caveat on `node_mars` as the general kernel.
    """
    pid_r = tl.program_id(axis = 0)
    pid_b = tl.program_id(axis = 1)

    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)
    mask_b = offs_b < batch_size

    offs_rank = tl.arange(0, RANK)

    xu = tl.load(xu_ptr + pid_r) + ext_base
    xv = tl.load(xv_ptr + pid_r) + ext_base

    ## `log W` / `log A` -- computed ONCE for this (node block, batch tile) ##

    log_w = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)
    log_a = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)

    for c_start in range(0, CH_BLOCK_SIZE, TILE_C):
        offs_c = c_start + tl.arange(0, TILE_C)

        cids = tl.load(cids_ptr + pid_r * num_edges + offs_c)
        emars = tl.load(element_mars_ptr + cids[:,None] * batch_size + offs_b[None,:],
                        mask = mask_b[None,:], other = float("-inf"))

        u_offs = xu + offs_c[:,None] * RANK + offs_rank[None,:]
        u = tl.load(ext_ptr + u_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:], other = float("-inf"))

        log_w = _logaddexp(log_w, _logsumexp(u + emars[:,None,:], 0))
        log_a = _logaddexp(log_a, _logsumexp(u, 0))

    ## Walk the node tiles, reusing them ##

    nid = tl.load(nids_ptr + pid_r)

    for m_start in range(0, BLOCK_SIZE, TILE_M):
        offs_m = m_start + tl.arange(0, TILE_M)

        v_offs = xv + offs_m[:,None] * RANK + offs_rank[None,:]
        v = tl.load(ext_ptr + v_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:], other = float("-inf"))

        log_s2 = _logsumexp(v + log_w[None,:,:], 1)
        log_zt = _logsumexp(v + log_a[None,:,:], 1)

        mars_ptr = node_mars_ptr + (nid + offs_m)[:,None] * batch_size + offs_b[None,:]
        log_s1 = tl.load(mars_ptr, mask = mask_b[None,:], other = 0.0)

        log_z = _logaddexp(tl.zeros([TILE_M, TILE_B], dtype = tl.float32), log_zt)

        tl.store(mars_ptr, _logaddexp(log_s1, log_s2) - log_z, mask = mask_b[None,:])


@triton_jit
def _fw_lowrank_wa_kernel(element_mars_ptr, ext_ptr, cids_ptr, xu_ptr, log_w_ptr, log_a_ptr,
                          batch_size, num_edges, num_eblks, ext_base, CH_BLOCK_SIZE: tl.constexpr,
                          RANK: tl.constexpr, TILE_B: tl.constexpr, TILE_C: tl.constexpr):
    """
    Phase 1 of the two-pass form: reduce `U` against the children into `log W` / `log A`.

    `log W` and `log A` are indexed by `(node block row, edge block, rank, batch)` -- NOT by the node
    index -- so computing them here once and staging them lets phase 2 keep a small `TILE_M`, and hence
    full parallelism, without recomputing them per node tile.

    They are tiny (`rows * E * rank * B`) next to `U` itself, so the extra round trip is nearly free
    compared to re-reading `U` `block_size / TILE_M` times.
    """
    pid_r = tl.program_id(axis = 0)
    pid_j = tl.program_id(axis = 1)
    pid_b = tl.program_id(axis = 2)

    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)
    mask_b = offs_b < batch_size

    offs_rank = tl.arange(0, RANK)

    xu = tl.load(xu_ptr + pid_r * num_eblks + pid_j) + ext_base

    log_w = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)
    log_a = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)

    for c_start in range(0, CH_BLOCK_SIZE, TILE_C):
        offs_c = c_start + tl.arange(0, TILE_C)

        cids = tl.load(cids_ptr + pid_r * num_edges + pid_j * CH_BLOCK_SIZE + offs_c)
        emars = tl.load(element_mars_ptr + cids[:,None] * batch_size + offs_b[None,:],
                        mask = mask_b[None,:], other = float("-inf"))

        u_offs = xu + offs_c[:,None] * RANK + offs_rank[None,:]
        u = tl.load(ext_ptr + u_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:], other = float("-inf"))

        log_w = _logaddexp(log_w, _logsumexp(u + emars[:,None,:], 0))
        log_a = _logaddexp(log_a, _logsumexp(u, 0))

    out_offs = ((pid_r * num_eblks + pid_j) * RANK + offs_rank)[:,None] * batch_size + offs_b[None,:]
    tl.store(log_w_ptr + out_offs, log_w, mask = mask_b[None,:])
    tl.store(log_a_ptr + out_offs, log_a, mask = mask_b[None,:])


@triton_jit
def _fw_lowrank_noop_kernel(node_mars_ptr, ext_ptr, nids_ptr, xv_ptr, pw_ptr, pa_ptr, batch_size,
                            num_eblks, n_ctiles, ext_base, RANK: tl.constexpr, TILE_M: tl.constexpr,
                            TILE_B: tl.constexpr):
    """
    DIAGNOSTIC ONLY: same parameter list, same grid, empty body.

    Launching this in place of the real kernels isolates the cost of *issuing* the launches -- Python
    binding, signature, the driver call -- from the cost of the work they do. Any wall-clock difference
    it produces is pure launch overhead by construction.
    """
    pass


@triton_jit
def _fw_lowrank_wa_partial_kernel(element_mars_ptr, ext_ptr, cids_ptr, xu_ptr, pw_ptr, pa_ptr,
                                  batch_size, num_edges, num_eblks, n_ctiles, ext_base,
                                  CH_BLOCK_SIZE: tl.constexpr, RANK: tl.constexpr,
                                  TILE_B: tl.constexpr, TILE_C: tl.constexpr):
    """
    Phase 1a: PARTIAL `log W` / `log A`, one program per child tile.

    The two-launch form starves on shapes where `num_node_blocks` and `num_edge_blocks` are both 1 -- a
    dense transition -- because the batch axis is then the only parallel axis, leaving `B / TILE_B`
    programs (2 for a 64-example batch, measured at 21 GB/s, ~1.4% of peak). The fix is to split the
    `c` reduction, which is the only axis with real width (`ch_block_size` of them).

    That also happens to be the best possible access pattern: `U` is stored `[E, Kc, rank, B]`, so one
    `c` tile is `TILE_C * rank * B` FULLY CONTIGUOUS elements. Each program streams one such run.

    Each program emits a partial reduction; :func:`_fw_lowrank_wa_reduce_kernel` combines them. The
    partials are `rank / TILE_C` of the size of `U`, so the extra round trip is minor next to reading
    `U` `block_size / TILE_M` times.
    """
    pid_re = tl.program_id(axis = 0)          # row * num_eblks + edge block
    pid_ct = tl.program_id(axis = 1)          # child tile
    pid_b = tl.program_id(axis = 2)

    row = pid_re // num_eblks
    j = pid_re % num_eblks

    offs_c = pid_ct * TILE_C + tl.arange(0, TILE_C)
    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)
    mask_b = offs_b < batch_size

    offs_rank = tl.arange(0, RANK)

    xu = tl.load(xu_ptr + pid_re) + ext_base

    cids = tl.load(cids_ptr + row * num_edges + j * CH_BLOCK_SIZE + offs_c)
    emars = tl.load(element_mars_ptr + cids[:,None] * batch_size + offs_b[None,:],
                    mask = mask_b[None,:], other = float("-inf"))

    u_offs = xu + offs_c[:,None] * RANK + offs_rank[None,:]
    u = tl.load(ext_ptr + u_offs[:,:,None] * batch_size + offs_b[None,None,:],
                mask = mask_b[None,None,:], other = float("-inf"))

    pw = _logsumexp(u + emars[:,None,:], 0)
    pa = _logsumexp(u, 0)

    out_offs = ((pid_re * RANK + offs_rank) * n_ctiles + pid_ct)[:,None] * batch_size + offs_b[None,:]
    tl.store(pw_ptr + out_offs, pw, mask = mask_b[None,:])
    tl.store(pa_ptr + out_offs, pa, mask = mask_b[None,:])


@triton_jit
def _fw_lowrank_wa_reduce_kernel(pw_ptr, pa_ptr, log_w_ptr, log_a_ptr, batch_size, n_ctiles,
                                 N_CTILES_POW2: tl.constexpr, TILE_B: tl.constexpr):
    """
    Phase 1b: combine the per-child-tile partials into `log W` / `log A`.

    One program per `(row, edge block, rank)` slot, so the parallel width is `rows * E * rank` -- 16x
    wider than the batch axis alone on a dense transition -- and the partials are laid out with batch
    innermost so each program reads `[n_ctiles, TILE_B]` contiguously per row.
    """
    pid_rer = tl.program_id(axis = 0)         # (row * num_eblks + edge block) * RANK + rank
    pid_b = tl.program_id(axis = 1)

    offs_ct = tl.arange(0, N_CTILES_POW2)
    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)

    mask = (offs_ct < n_ctiles)[:,None] & (offs_b < batch_size)[None,:]
    in_offs = (pid_rer * n_ctiles + offs_ct)[:,None] * batch_size + offs_b[None,:]

    pw = tl.load(pw_ptr + in_offs, mask = mask, other = float("-inf"))
    pa = tl.load(pa_ptr + in_offs, mask = mask, other = float("-inf"))

    out_offs = pid_rer * batch_size + offs_b
    tl.store(log_w_ptr + out_offs, _logsumexp(pw, 0), mask = offs_b < batch_size)
    tl.store(log_a_ptr + out_offs, _logsumexp(pa, 0), mask = offs_b < batch_size)


@triton_jit
def _fw_lowrank_combine_reduce_kernel(node_mars_ptr, ext_ptr, nids_ptr, xv_ptr, pw_ptr, pa_ptr,
                                      batch_size, num_eblks, n_ctiles, ext_base, RANK: tl.constexpr,
                                      TILE_M: tl.constexpr, TILE_B: tl.constexpr):
    """
    Phases 1b+2 fused: reduce the child-tile partials INSIDE the combine kernel.

    Saves one launch per layer. Every node tile now redoes the partial reduction, but the partials are
    `rank * n_ctiles * B` -- tens of KB, so L2-resident -- and re-reducing them costs a couple of
    microseconds of L2 traffic against ~8 us for a launch. That trade is a win exactly in the
    latency-bound regime (small `rank * batch`) and roughly neutral otherwise, so it does not replace
    the 3-launch form; :func:`fw_lowrank` picks between them.

    Keep `n_ctiles` modest (i.e. `TILE_C` large) here: it is a sequential loop in every program.
    """
    pid_r = tl.program_id(axis = 0)
    pid_m = tl.program_id(axis = 1)
    pid_b = tl.program_id(axis = 2)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)
    mask_b = offs_b < batch_size

    offs_rank = tl.arange(0, RANK)

    log_s2 = tl.full([TILE_M, TILE_B], float("-inf"), dtype = tl.float32)
    log_zt = tl.full([TILE_M, TILE_B], float("-inf"), dtype = tl.float32)

    for j in range(num_eblks):
        pid_re = pid_r * num_eblks + j

        log_w = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)
        log_a = tl.full([RANK, TILE_B], float("-inf"), dtype = tl.float32)

        for ct in range(n_ctiles):
            offs = ((pid_re * RANK + offs_rank) * n_ctiles + ct)[:,None] * batch_size + offs_b[None,:]
            log_w = _logaddexp(log_w, tl.load(pw_ptr + offs, mask = mask_b[None,:],
                                              other = float("-inf")))
            log_a = _logaddexp(log_a, tl.load(pa_ptr + offs, mask = mask_b[None,:],
                                              other = float("-inf")))

        xv = tl.load(xv_ptr + pid_re) + ext_base
        v_offs = xv + offs_m[:,None] * RANK + offs_rank[None,:]
        v = tl.load(ext_ptr + v_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:], other = float("-inf"))

        log_s2 = _logaddexp(log_s2, _logsumexp(v + log_w[None,:,:], 1))
        log_zt = _logaddexp(log_zt, _logsumexp(v + log_a[None,:,:], 1))

    nid = tl.load(nids_ptr + pid_r)
    mars_ptr = node_mars_ptr + (nid + offs_m)[:,None] * batch_size + offs_b[None,:]
    log_s1 = tl.load(mars_ptr, mask = mask_b[None,:], other = 0.0)

    log_z = _logaddexp(tl.zeros([TILE_M, TILE_B], dtype = tl.float32), log_zt)
    tl.store(mars_ptr, _logaddexp(log_s1, log_s2) - log_z, mask = mask_b[None,:])


@triton_jit
def _fw_lowrank_combine_kernel(node_mars_ptr, ext_ptr, nids_ptr, xv_ptr, log_w_ptr, log_a_ptr,
                               batch_size, num_eblks, ext_base, RANK: tl.constexpr,
                               TILE_M: tl.constexpr, TILE_B: tl.constexpr):
    """
    Phase 2: combine the staged `log W` / `log A` with `V` and rewrite `node_mars`.

    Because phase 1 removed the recomputation, `TILE_M` is free to be small here -- it now only trades
    against the `[TILE_M, RANK, TILE_B]` working tile, so it can be sized for parallelism and for
    keeping `TILE_B` large enough to coalesce the batch-innermost factor reads.
    """
    pid_r = tl.program_id(axis = 0)
    pid_m = tl.program_id(axis = 1)
    pid_b = tl.program_id(axis = 2)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_b = pid_b * TILE_B + tl.arange(0, TILE_B)
    mask_b = offs_b < batch_size

    offs_rank = tl.arange(0, RANK)

    log_s2 = tl.full([TILE_M, TILE_B], float("-inf"), dtype = tl.float32)
    log_zt = tl.full([TILE_M, TILE_B], float("-inf"), dtype = tl.float32)

    for j in range(num_eblks):
        wa_offs = ((pid_r * num_eblks + j) * RANK + offs_rank)[:,None] * batch_size + offs_b[None,:]
        log_w = tl.load(log_w_ptr + wa_offs, mask = mask_b[None,:], other = float("-inf"))
        log_a = tl.load(log_a_ptr + wa_offs, mask = mask_b[None,:], other = float("-inf"))

        xv = tl.load(xv_ptr + pid_r * num_eblks + j) + ext_base
        v_offs = xv + offs_m[:,None] * RANK + offs_rank[None,:]
        v = tl.load(ext_ptr + v_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:], other = float("-inf"))

        log_s2 = _logaddexp(log_s2, _logsumexp(v + log_w[None,:,:], 1))
        log_zt = _logaddexp(log_zt, _logsumexp(v + log_a[None,:,:], 1))

    nid = tl.load(nids_ptr + pid_r)
    mars_ptr = node_mars_ptr + (nid + offs_m)[:,None] * batch_size + offs_b[None,:]
    log_s1 = tl.load(mars_ptr, mask = mask_b[None,:], other = 0.0)

    log_z = _logaddexp(tl.zeros([TILE_M, TILE_B], dtype = tl.float32), log_zt)
    tl.store(mars_ptr, _logaddexp(log_s1, log_s2) - log_z, mask = mask_b[None,:])


def fw_lowrank(node_mars: torch.Tensor, element_mars: torch.Tensor, external_params: torch.Tensor,
               nids: torch.Tensor, cids: torch.Tensor, xu: torch.Tensor, xv: torch.Tensor,
               block_size: int, ch_block_size: int, rank: int, ext_base: int = 0,
               TILE_M: int = 32, TILE_B: int = 32, TILE_C: int = 16, TILE_BC: int = 0,
               variant: str = "grid", alloc = None):
    """
    Launch the low-rank forward correction for one forward partition.

    `variant` selects the launch structure -- all three compute the same quantity, and differ only in
    how they trade recomputing `log W` / `log A` against parallelism. See
    :class:`~pyjuice.nodes.external_params.LowRankSumParams`.
    """
    batch_size = node_mars.size(1)

    TILE_M = min(TILE_M, block_size)
    TILE_C = min(TILE_C, ch_block_size)

    # The `W`/`A` phases reduce over children and the combine phase does not, so they do not want the
    # same batch tile; `TILE_BC = 0` means "use TILE_B for both".
    TILE_BC = min(TILE_BC, batch_size) if TILE_BC else TILE_B

    if alloc is None:
        def alloc(name, numel):
            return torch.empty([numel], dtype = torch.float32, device = node_mars.device)

    if variant == "cuda":
        # ONE host call launches both phases; the grid arithmetic happens in C++. See the .cu for why
        # that matters on a deep chain (the correction is launch-bound, not bandwidth-bound, there).
        from .c import get_module

        mod = get_module()
        if mod is not None:
            rows, num_eblks = xu.size(0), xu.size(1)
            n_ctiles = triton.cdiv(ch_block_size, TILE_C)

            tb1 = max(1, min(TILE_B, batch_size))
            while rank * tb1 > 1024:
                tb1 //= 2
            tb2 = max(1, min(TILE_BC, batch_size))
            while TILE_M * tb2 > 1024:
                tb2 //= 2

            mod.lowrank_forward(
                node_mars, element_mars, external_params, nids, cids, xu, xv,
                alloc("pw", rows * num_eblks * rank * n_ctiles * batch_size),
                alloc("pa", rows * num_eblks * rank * n_ctiles * batch_size),
                block_size, ch_block_size, rank, ext_base, TILE_C, TILE_M, tb1, tb2,
            )
            return None

        variant = "split2"      # toolchain unavailable -> Triton

    if variant == "noop":
        # Two launches per layer, matching `split2`'s launch count and grids, with no work done
        rows, num_eblks = xu.size(0), xu.size(1)
        n_ctiles = triton.cdiv(ch_block_size, TILE_C)
        for g in ((rows * num_eblks, n_ctiles, triton.cdiv(batch_size, TILE_B)),
                  (rows, triton.cdiv(block_size, TILE_M), triton.cdiv(batch_size, TILE_BC))):
            _fw_lowrank_noop_kernel[g](
                node_mars_ptr = node_mars, ext_ptr = external_params, nids_ptr = nids, xv_ptr = xv,
                pw_ptr = node_mars, pa_ptr = node_mars, batch_size = batch_size,
                num_eblks = num_eblks, n_ctiles = n_ctiles, ext_base = ext_base,
                RANK = rank, TILE_M = TILE_M, TILE_B = TILE_BC,
            )
        return None

    if variant in ("scratch", "split", "split2"):
        rows, num_eblks = xu.size(0), xu.size(1)

        if variant != "split2":
            log_w = alloc("log_w", rows * num_eblks * rank * batch_size)
            log_a = alloc("log_a", rows * num_eblks * rank * batch_size)

        if variant in ("split", "split2"):
            n_ctiles = triton.cdiv(ch_block_size, TILE_C)

            pw = alloc("pw", rows * num_eblks * rank * n_ctiles * batch_size)
            pa = alloc("pa", rows * num_eblks * rank * n_ctiles * batch_size)

            _fw_lowrank_wa_partial_kernel[(rows * num_eblks, n_ctiles,
                                           triton.cdiv(batch_size, TILE_B))](
                element_mars_ptr = element_mars, ext_ptr = external_params, cids_ptr = cids,
                xu_ptr = xu, pw_ptr = pw, pa_ptr = pa, batch_size = batch_size,
                num_edges = cids.size(1), num_eblks = num_eblks, n_ctiles = n_ctiles,
                ext_base = ext_base, CH_BLOCK_SIZE = ch_block_size, RANK = rank,
                TILE_B = TILE_B, TILE_C = TILE_C,
            )
            if variant == "split2":
                # Fused reduce+combine: one launch instead of two
                _fw_lowrank_combine_reduce_kernel[(rows, triton.cdiv(block_size, TILE_M),
                                                   triton.cdiv(batch_size, TILE_BC))](
                    node_mars_ptr = node_mars, ext_ptr = external_params, nids_ptr = nids,
                    xv_ptr = xv, pw_ptr = pw, pa_ptr = pa, batch_size = batch_size,
                    num_eblks = num_eblks, n_ctiles = n_ctiles, ext_base = ext_base,
                    RANK = rank, TILE_M = TILE_M, TILE_B = TILE_BC,
                )
                return None

            _fw_lowrank_wa_reduce_kernel[(rows * num_eblks * rank,
                                          triton.cdiv(batch_size, TILE_B))](
                pw_ptr = pw, pa_ptr = pa, log_w_ptr = log_w, log_a_ptr = log_a,
                batch_size = batch_size, n_ctiles = n_ctiles,
                N_CTILES_POW2 = triton.next_power_of_2(n_ctiles), TILE_B = TILE_B,
            )
        else:
            _fw_lowrank_wa_kernel[(rows, num_eblks, triton.cdiv(batch_size, TILE_B))](
                element_mars_ptr = element_mars, ext_ptr = external_params, cids_ptr = cids,
                xu_ptr = xu, log_w_ptr = log_w, log_a_ptr = log_a, batch_size = batch_size,
                num_edges = cids.size(1), num_eblks = num_eblks, ext_base = ext_base,
                CH_BLOCK_SIZE = ch_block_size, RANK = rank,
                TILE_B = TILE_B, TILE_C = TILE_C,
            )

        _fw_lowrank_combine_kernel[(rows, triton.cdiv(block_size, TILE_M),
                                    triton.cdiv(batch_size, TILE_BC))](
            node_mars_ptr = node_mars, ext_ptr = external_params, nids_ptr = nids, xv_ptr = xv,
            log_w_ptr = log_w, log_a_ptr = log_a, batch_size = batch_size, num_eblks = num_eblks,
            ext_base = ext_base, RANK = rank, TILE_M = TILE_M, TILE_B = TILE_BC,
        )
        return None

    if variant == "hoist" and xu.size(1) == 1:
        # One edge block per node: hoist `log W` / `log A` out of the node dimension
        _fw_lowrank_single_kernel[(nids.size(0), triton.cdiv(batch_size, TILE_B))](
            node_mars_ptr = node_mars,
            element_mars_ptr = element_mars,
            ext_ptr = external_params,
            nids_ptr = nids,
            cids_ptr = cids,
            xu_ptr = xu,
            xv_ptr = xv,
            batch_size = batch_size,
            num_edges = cids.size(1),
            ext_base = ext_base,
            BLOCK_SIZE = block_size,
            CH_BLOCK_SIZE = ch_block_size,
            RANK = rank,
            TILE_M = TILE_M,
            TILE_B = TILE_B,
            TILE_C = TILE_C,
        )
        return None

    grid = (nids.size(0), triton.cdiv(batch_size, TILE_B), triton.cdiv(block_size, TILE_M))

    _fw_lowrank_kernel[grid](
        node_mars_ptr = node_mars,
        element_mars_ptr = element_mars,
        ext_ptr = external_params,
        nids_ptr = nids,
        cids_ptr = cids,
        xu_ptr = xu,
        xv_ptr = xv,
        batch_size = batch_size,
        num_edges = cids.size(1),
        ext_base = ext_base,
        BLOCK_SIZE = block_size,
        CH_BLOCK_SIZE = ch_block_size,
        RANK = rank,
        MAX_N_EBLKS = xu.size(1),
        TILE_M = TILE_M,
        TILE_B = TILE_B,
        TILE_C = TILE_C,
    )
