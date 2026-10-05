"""
The conditional dual-flow M-step, `theta <- normalize(theta * F+ / F-)`, as two shared Triton kernels.

Used by `ExternalSumParams._fused_em_correction` for EVERY parameterization that requests the
denominator flow. Nothing here is specific to one of them except HOW `F-` is formed, which is a
`DENOM_MODE` constexpr -- see below.

WHY `F-` IS FORMED IN-KERNEL RATHER THAN PASSED IN. The generic torch M-step has to materialize `F-`
into a param-flow-sized buffer and then builds `ratio`, `flow` and `new_theta` on top of it: MEASURED
at a 4 MB `ns`, 20.1 MB of transients against 4.1 MB here, i.e. ~5x the parameter range versus ~1x,
and 57.8 ms against 26.7 ms. Both costs scale linearly with the model, so a parameterization that
only needs the fast path for memory reasons still wants these kernels.

MODES. `F-[n,c]` is whatever the parameterization's own backward accumulated, expanded back to the
per-edge grid:

  * `DENOM_DENSE` -- it is already in param-flow layout, so `F-[n,c] = denom[pfid(n,c) - denom_base]`.
    The straightforward choice, and the one to use unless the statistic compresses.
  * `DENOM_ROWGROUP` -- it factors as `F-[n,c] = theta[n,c] * G[n, group(c)]` with
    `group(c) = c // GROUP_CBS`, i.e. a per-(node, group-of-children) factor. `BlockScaleSumParams`'
    gate space: `G` is `GROUP_CBS` times smaller than a param-flow mirror, which is what lets it be
    carried between backward calls rather than rebuilt.

A new form is a new mode here plus a `denom_kernel_spec` on the parameterization; the normalizer, the
update, the pseudocount split, the step-size interpolation and the `keep_zero_params` rule are shared
and stay in one place.

SCALARS COME FROM A DEVICE TABLE (`consts`), not from arguments. Triton types a Python float argument
as fp64, and so does a float LITERAL inside a kernel, either of which silently promotes the whole
`den` / `ratio` chain to DOUBLE precision -- measured at ~1.8x, and caught only because an earlier
single-kernel form made the accumulator loop-carried and Triton then refused the type change. Every
threshold is in the table for the same reason; `em_par_update_kernel` does this too. The offsets are
laid out in `_fused_em_correction`.

COORDINATES. Both passes work in the layer's COMPILED space -- `pids` / `pfids` / `cids` per
(row, edge slot), plus `offs_node` within the block -- so no new index derivation is introduced and
padded slots (`cids == 0`) are masked out as everywhere else. `row_map` selects the rows of the ONE
`ns` being updated, which is what keeps several `ns` in one layer separate.

`kcount[row]` is the number of children of each node of that row (the count of real edge slots), i.e.
the torch path's `eblk_per_nb[nb] * ch_block_size`. It sets the pseudocount split.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


DENOM_DENSE = 0
DENOM_ROWGROUP = 1


@triton_jit
def _dual_em_cum_kernel(mparams, param_flows, denom, cum, cids, pids, pfids, nids, row_map, kcount,
                        num_edges: tl.constexpr, n_groups: tl.constexpr,
                        TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr, GROUP_CBS: tl.constexpr,
                        consts, cum_base, denom_base, DENOM_MODE: tl.constexpr):
    """
    Pass 1: `cum[n] += sum_c theta[n,c] * ratio[n,c]`, the per-node normalizer of the dual M-step, with

        ratio[n,c] = (F+[n,c] + pseudocount / K[n]) / (F-[n,c] + pseudocount * theta[n,c])

    ATOMIC into `cum`, because a node's children may be split across edge tiles AND across forward
    partitions -- the normalizer has to sum over all of them before pass 2 divides by it. Indexed by
    GLOBAL node id minus `cum_base` for the same reason: the buffer has to be shared across partitions.
    """
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
    theta = tl.load(mparams + par[None,:] + offs_node[:,None], mask = real[None,:], other = 0.0)
    fp = tl.load(param_flows + pf[None,:] + offs_node[:,None], mask = real[None,:], other = 0.0)

    k = tl.load(kcount + nblock_id).to(tl.float32)
    pseudocount = tl.load(consts + 0)
    tiny_den = tl.load(consts + 3)       # 1e-38, the division guard

    # the MAP form: denominator `F- + pseudocount * theta`, not `+ pseudocount`
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
def _dual_em_update_kernel(mparams, param_flows, denom, cum, out, cids, pids, pfids, nids, row_map,
                           kcount, num_edges: tl.constexpr, n_groups: tl.constexpr,
                           TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr, GROUP_CBS: tl.constexpr,
                           consts, cum_base, denom_base, out_base, out_size,
                           DENOM_MODE: tl.constexpr, KEEP_ZERO: tl.constexpr,
                           OUT_BY_PAR: tl.constexpr = 0):
    """
    Pass 2: `out[.] = clamp(theta * ((1-s) + s*ratio) / ((1-s) + s*cum[n]), min = 1e-30)`.

    `ratio` is RECOMPUTED rather than stored. It is three flops over values this kernel has to read
    anyway, against a parameter-sized round trip to keep it -- and keeping it is what the eager path did.

    `OUT_BY_PAR` writes `params[par - out_base]`, i.e. back into the very parameter `theta` was read
    from, which is what lets the caller hand us `params[ps:pe]` and skip a separate write-back.
    Addressing by `pfid` instead would rely on the parameter and param-flow ranges inducing the same
    local order; they do, but there is no reason to depend on it when `par` is already in hand. The
    offset is masked and clamped either way.
    """
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
    theta = tl.load(mparams + par[None,:] + offs_node[:,None], mask = real[None,:], other = 0.0)
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

    if OUT_BY_PAR:
        off = par[None,:] + offs_node[:,None] - out_base
    else:
        off = pf[None,:] + offs_node[:,None] - out_base
    inside = real[None,:] & (off >= 0) & (off < out_size)
    tl.store(out + tl.maximum(tl.minimum(off, out_size - 1), 0), new, mask = inside)
