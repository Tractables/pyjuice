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
                       batch_size, num_edges, BLOCK_SIZE: tl.constexpr, CH_BLOCK_SIZE: tl.constexpr,
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

    Grid is `(node block row, batch tile, node tile)`. `W` and `A` do not depend on the node index, so
    they are recomputed once per node tile: the arithmetic overhead relative to the shared sum kernel
    is `2*RANK/TILE_M + RANK/CH_BLOCK_SIZE`, i.e. large `TILE_M` is cheap and small `TILE_M` is not.
    With a large batch the batch axis supplies the parallelism, so `TILE_M` can be kept large.
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
        xu = tl.maximum(xu, 0)
        xv = tl.maximum(xv, 0)

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


def fw_lowrank(node_mars: torch.Tensor, element_mars: torch.Tensor, external_params: torch.Tensor,
               nids: torch.Tensor, cids: torch.Tensor, xu: torch.Tensor, xv: torch.Tensor,
               block_size: int, ch_block_size: int, rank: int,
               TILE_M: int = 32, TILE_B: int = 32, TILE_C: int = 16):
    """
    Launch :func:`_fw_lowrank_kernel` for one forward partition.
    """
    batch_size = node_mars.size(1)

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
        BLOCK_SIZE = block_size,
        CH_BLOCK_SIZE = ch_block_size,
        RANK = rank,
        MAX_N_EBLKS = xu.size(1),
        TILE_M = min(TILE_M, block_size),
        TILE_B = TILE_B,
        TILE_C = min(TILE_C, ch_block_size),
    )
