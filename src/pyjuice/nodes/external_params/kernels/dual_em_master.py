"""
The conditional dual-flow M-step (`kernels/dual_em.py`) for a circuit that stores its parameters below fp32 and
updates their fp32 master copy (`pyjuice.model.backend.params_master`), with the copy's passes fused into its own:

  * `_dual_em_cum_sync_kernel`, the first pass: `theta` is the STORED value where that is no longer the master copy's
    rounding (written since the update stored it), and there the stored value is taken into the copy --
    `sync_master`'s check and write, in the pass that reads the copy anyway;
  * `_dual_em_update_store_kernel`, the second: the update, its result also stored, rounded to nearest even, into the
    stored parameters -- the `copy_` after it.

Before, a `sync_master` pass over the range read both copies first and a `copy_` after the update read the copy again.
Fused, the first pass reads the stored values (2 bytes a parameter) and the second writes them (2), against the sync's
6 and the copy's 6 -- but only where the range's master copy does not fit the L2 is that a win: below, the sync leaves
the copy in the L2 for the first pass and the copy reads it back from there (MEASURED, a tied gated HMM, bf16, the range
~94% of the parameters: HMM-4096, a 64 MB copy of a 128 MB L2, its M-step 297 -> 309 us with the sync fused alone;
HMM-6144 (144 MB) 874 -> 776, HMM-8192 (256 MB) 1491 -> 1318). The caller decides (`ExternalSumParams._fold_master`).

The same arithmetic as `dual_em`'s kernels on the same `theta`, to the layout Triton gives the first pass's tile (its
reduction order: ~1e-7 apart). The stored parameters' range is indexed as the update's output, `stored[par -
out_base]` (the caller hands both `[ps:pe]`).
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def _dual_em_cum_sync_kernel(mparams, stored, param_flows, denom, cum, cids, pids, pfids, nids, row_map, kcount,
                             num_edges: tl.constexpr, n_groups: tl.constexpr,
                             TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                             BLOCK_SIZE_M: tl.constexpr, GROUP_CBS: tl.constexpr,
                             consts, cum_base, denom_base, out_base, DENOM_MODE: tl.constexpr):
    """`dual_em._dual_em_cum_kernel` with `theta` the stored value where that is not the master copy's rounding -- and
    there taken into the copy (`sync_master`'s write)."""
    pid_k = tl.program_id(0)
    pid_y = tl.program_id(1)

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    nblock_id = tl.load(row_map + pid_y // M_TILES)
    pid_m = pid_y % M_TILES
    offs_node = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offs_edge = pid_k * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)
    emask = offs_edge < num_edges

    cid = tl.load(cids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    real = emask & (cid != 0)

    par = tl.load(pids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    pf = tl.load(pfids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    m_ptr = mparams + par[None,:] + offs_node[:,None]
    m = tl.load(m_ptr, mask = real[None,:], other = 0.0).to(tl.float32)
    s = tl.load(stored + (par[None,:] + offs_node[:,None] - out_base), mask = real[None,:], other = 0.0)
    stale = s != m.to(stored.dtype.element_ty)                 # written since the copy was last stored
    theta = tl.where(stale, s.to(tl.float32), m)
    tl.store(m_ptr, theta, mask = real[None,:] & stale)        # (as `sync_master`: the update reads the copy)
    fp = tl.load(param_flows + pf[None,:] + offs_node[:,None], mask = real[None,:], other = 0.0)

    k = tl.load(kcount + nblock_id).to(tl.float32)
    pseudocount = tl.load(consts + 0)
    tiny_den = tl.load(consts + 3)       # 1e-38, the division guard

    if DENOM_MODE == 1:                  # DENOM_ROWGROUP
        gidx = offs_edge // GROUP_CBS
        w = tl.load(denom + (nblock_id * BLOCK_SIZE_M + offs_node)[:,None] * n_groups + gidx[None,:],
                    mask = real[None,:] & (gidx < n_groups)[None,:], other = 0.0)
        den = tl.maximum(theta * w + pseudocount * theta, tiny_den)
    else:                                # DENOM_DENSE
        fm = tl.load(denom + (pf[None,:] + offs_node[:,None] - denom_base),
                     mask = real[None,:], other = 0.0)
        den = tl.maximum(fm + pseudocount * theta, tiny_den)

    ratio = (fp + pseudocount / k) / den
    acc = tl.sum(tl.where(real[None,:], theta * ratio, 0.0), axis = 1)          # [M]

    nid = tl.load(nids + nblock_id)
    tl.atomic_add(cum + (nid + offs_node - cum_base), acc)


@triton_jit
def _dual_em_update_store_kernel(mparams, stored, param_flows, denom, cum, out, cids, pids, pfids, nids, row_map,
                                 kcount, num_edges: tl.constexpr, n_groups: tl.constexpr,
                                 TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_M: tl.constexpr, GROUP_CBS: tl.constexpr,
                                 consts, cum_base, denom_base, out_base, out_size,
                                 DENOM_MODE: tl.constexpr, KEEP_ZERO: tl.constexpr):
    """`dual_em._dual_em_update_kernel` (`OUT_BY_PAR`: `out` the master copy's `[ps:pe]`, in line with the stored
    values after `_dual_em_cum_sync_kernel`), the new value also stored, rounded, into `stored`."""
    pid_k = tl.program_id(0)
    pid_y = tl.program_id(1)

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    nblock_id = tl.load(row_map + pid_y // M_TILES)
    pid_m = pid_y % M_TILES
    offs_node = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offs_edge = pid_k * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)
    emask = offs_edge < num_edges

    cid = tl.load(cids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    real = emask & (cid != 0)

    par = tl.load(pids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    pf = tl.load(pfids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    theta = tl.load(mparams + par[None,:] + offs_node[:,None], mask = real[None,:], other = 0.0).to(tl.float32)
    fp = tl.load(param_flows + pf[None,:] + offs_node[:,None], mask = real[None,:], other = 0.0)

    k = tl.load(kcount + nblock_id).to(tl.float32)
    pseudocount = tl.load(consts + 0)
    step_size = tl.load(consts + 1)
    one_m_s = tl.load(consts + 2)        # 1 - step_size
    tiny_den = tl.load(consts + 3)       # 1e-38, the division guard
    tiny_par = tl.load(consts + 4)       # 1e-30, the momentum-underflow floor

    if DENOM_MODE == 1:                  # DENOM_ROWGROUP
        gidx = offs_edge // GROUP_CBS
        w = tl.load(denom + (nblock_id * BLOCK_SIZE_M + offs_node)[:,None] * n_groups + gidx[None,:],
                    mask = real[None,:] & (gidx < n_groups)[None,:], other = 0.0)
        den = tl.maximum(theta * w + pseudocount * theta, tiny_den)
    else:                                # DENOM_DENSE
        fm = tl.load(denom + (pf[None,:] + offs_node[:,None] - denom_base),
                     mask = real[None,:], other = 0.0)
        den = tl.maximum(fm + pseudocount * theta, tiny_den)

    ratio = (fp + pseudocount / k) / den

    nid = tl.load(nids + nblock_id)
    c = tl.load(cum + (nid + offs_node - cum_base))                            # [M]
    c = tl.maximum(one_m_s + step_size * c, tiny_den)

    new = theta * (one_m_s + step_size * ratio) / c[:,None]
    new = tl.maximum(new, tiny_par)                            # momentum-underflow guard
    if KEEP_ZERO:
        zero_thr = tl.load(consts + 5)                         # 1e-12, the `keep_zero_params` threshold
        new = tl.where(theta < zero_thr, tl.load(consts + 6), new)

    off = par[None,:] + offs_node[:,None] - out_base
    inside = real[None,:] & (off >= 0) & (off < out_size)
    off = tl.maximum(tl.minimum(off, out_size - 1), 0)
    tl.store(out + off, new, mask = inside)
    tl.store(stored + off, new.to(stored.dtype.element_ty), mask = inside)
