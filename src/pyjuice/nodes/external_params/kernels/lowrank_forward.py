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

    Each program emits a partial reduction, which :func:`_fw_lowrank_combine_reduce_kernel` finishes.
    The partials are `rank / TILE_C` of the size of `U`, so the extra round trip is minor.
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

    # A PADDED edge block carries a -1 slot: `edge_ids` has no such edge, so no factor is stored for
    # it and it must contribute nothing. The offset is clamped so the address stays in bounds and the
    # load is masked off, which leaves `u` at `-inf` -- the identity of the log-sum-exp below, so both
    # partials come out `-inf` and the block drops out of the reduction.
    xu_raw = tl.load(xu_ptr + pid_re)
    live = xu_raw >= 0
    xu = tl.where(live, xu_raw, 0) + ext_base

    cids = tl.load(cids_ptr + row * num_edges + j * CH_BLOCK_SIZE + offs_c)
    emars = tl.load(element_mars_ptr + cids[:,None] * batch_size + offs_b[None,:],
                    mask = mask_b[None,:], other = float("-inf"))

    u_offs = xu + offs_c[:,None] * RANK + offs_rank[None,:]
    u = tl.load(ext_ptr + u_offs[:,:,None] * batch_size + offs_b[None,None,:],
                mask = mask_b[None,None,:] & live, other = float("-inf"))

    pw = _logsumexp(u + emars[:,None,:], 0)
    pa = _logsumexp(u, 0)

    out_offs = ((pid_re * RANK + offs_rank) * n_ctiles + pid_ct)[:,None] * batch_size + offs_b[None,:]
    tl.store(pw_ptr + out_offs, pw, mask = mask_b[None,:])
    tl.store(pa_ptr + out_offs, pa, mask = mask_b[None,:])


@triton_jit
def _fw_lowrank_combine_reduce_kernel(node_mars_ptr, ext_ptr, nids_ptr, xv_ptr, pw_ptr, pa_ptr,
                                      batch_size, num_eblks, n_ctiles, ext_base, RANK: tl.constexpr,
                                      TILE_M: tl.constexpr, TILE_B: tl.constexpr):
    """
    Phases 1b+2 fused: reduce the child-tile partials INSIDE the combine kernel.

    Saves one launch per layer. Every node tile redoes the partial reduction, but the partials are
    `rank * n_ctiles * B` -- tens of KB, so L2-resident -- and re-reducing them costs a couple of
    microseconds of L2 traffic against ~8 us for a launch.

    Keep `n_ctiles` modest (i.e. `TILE_C` large) here: it is a sequential loop in every program. The
    CUDA path does not have this trade-off -- it reduces once in its own kernel and is preferred.
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

        # Padded edge block -- see the note in the partial kernel. `log_w` / `log_a` are already
        # `-inf` here, so the accumulators would be unchanged anyway; masking `v` is what keeps the
        # read in bounds.
        xv_raw = tl.load(xv_ptr + pid_re)
        live = xv_raw >= 0
        xv = tl.where(live, xv_raw, 0) + ext_base
        v_offs = xv + offs_m[:,None] * RANK + offs_rank[None,:]
        v = tl.load(ext_ptr + v_offs[:,:,None] * batch_size + offs_b[None,None,:],
                    mask = mask_b[None,None,:] & live, other = float("-inf"))

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
               variant: str = "split2", alloc = None):
    """
    Launch the Triton low-rank forward correction for one forward partition.

    This is the FALLBACK path, used when the CUDA extension is unavailable; the CUDA implementation is
    both faster and the only one that stages `logW`/`logA`/`logZ` for the backward. Several other Triton
    launch structures were written and measured during development and all lost to this one -- a single
    fused kernel (redundant `U` re-reads), a node-loop-inside form (2-8 programs on 188 SMs, 32x the
    shared forward), and a three-launch split (an extra launch for a reduction this form absorbs) -- so
    only the winner is kept.

    :note: READ-MODIFY-WRITE on `node_mars`: it reads `log S1` and overwrites the same slots, so running
           it twice applies the correction twice. Safe inside `ExternalParamsSumLayer.forward`, which
           re-runs the shared forward first, but a launch-config autotuner must time the whole
           `layer.forward` (or use a scratch `node_mars`) rather than repeated bare launches.
    """
    batch_size = node_mars.size(1)

    TILE_M = min(TILE_M, block_size)
    TILE_C = min(TILE_C, ch_block_size)
    TILE_BC = min(TILE_BC, batch_size) if TILE_BC else TILE_B

    if alloc is None:
        def alloc(name, numel):
            return torch.empty([numel], dtype = torch.float32, device = node_mars.device)

    assert variant == "split2", f"Unsupported Triton variant {variant}."

    rows, num_eblks = xu.size(0), xu.size(1)
    n_ctiles = triton.cdiv(ch_block_size, TILE_C)
    numel = rows * num_eblks * rank * n_ctiles * batch_size

    pw = alloc("pw", numel)
    pa = alloc("pa", numel)

    _fw_lowrank_wa_partial_kernel[(rows * num_eblks, n_ctiles, triton.cdiv(batch_size, TILE_B))](
        element_mars_ptr = element_mars, ext_ptr = external_params, cids_ptr = cids, xu_ptr = xu,
        pw_ptr = pw, pa_ptr = pa, batch_size = batch_size, num_edges = cids.size(1),
        num_eblks = num_eblks, n_ctiles = n_ctiles, ext_base = ext_base,
        CH_BLOCK_SIZE = ch_block_size, RANK = rank, TILE_B = TILE_B, TILE_C = TILE_C,
    )
    _fw_lowrank_combine_reduce_kernel[(rows, triton.cdiv(block_size, TILE_M),
                                       triton.cdiv(batch_size, TILE_BC))](
        node_mars_ptr = node_mars, ext_ptr = external_params, nids_ptr = nids, xv_ptr = xv,
        pw_ptr = pw, pa_ptr = pa, batch_size = batch_size, num_eblks = num_eblks,
        n_ctiles = n_ctiles, ext_base = ext_base, RANK = rank, TILE_M = TILE_M, TILE_B = TILE_BC,
    )

    return None
