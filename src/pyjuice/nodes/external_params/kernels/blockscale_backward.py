"""
Triton element-flow backward for the per-block multiplicative gate (`BlockScaleSumParams`).

A fork of `pyjuice.layer.kernels.sum_backward_element_block_sparse._bk_triton_block_sparse_ele_kernel`,
restricted to the regime the gate is defined for (LL, log-space flows, no partial eval, no tempering,
`allow_modify_flows` / `allow_neg_flows` / `accumulate_ch_flows` off) and with ONE addition.

WHY THIS EXISTS ALONGSIDE THE CUDA FORKS. The gate has no Triton path at all otherwise, so a shape the
CuTe/TMA fork cannot serve had to raise -- and, worse, a shape it CAN serve was forced onto it even
where the ungated layer's own autotuner prefers Triton. At K=1024 / block_size=128 / batch=256 the
ungated backward runs Triton at ~32 us rather than its CuTe kernel at ~80 us; without this kernel the
gated backward had no way to follow it there.

WHAT THE GATE CHANGES. The standard kernel accumulates, per k-tile of parents,

    partial_flows     = dot(epars, exp(log_n_fdm - log_n_fdm_max))     [TILE_SIZE_M, BLOCK_B]
    partial_flows_max = emars + log_n_fdm_max
    acc               = logaddexp(acc, log(partial_flows) + partial_flows_max)

Every parent in a k-tile belongs to ONE parent node block (this kernel requires `ptr_inc_step == 1`,
i.e. a tile never straddles two blocks), and `phi` is constant over the parents of a block. So the
gate factors straight out of the contraction and becomes an add on the tile's exponent:

    partial_flows_max = emars + log_n_fdm_max + log phi[child gate, b]

`log phi = -inf` encodes "no gate here" (a padded edge block, whose parameters are zero anyway), and
the existing `partial_flows_max == -inf` branch already drops such a tile -- so absent gates need no
special case.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def _bs_triton_ele_kernel(node_flows, element_flows, node_mars, element_mars, mparams,
                          ext, gate, chids, parids_start, parids_increment,
                          parpids_start, parpids_increment,
                          batch_size: tl.constexpr, ptr_inc_step: tl.constexpr,
                          BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr,
                          K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                          TL_DOT: tl.constexpr, GATE_CBS: tl.constexpr,
                          gate_stride: tl.constexpr, ext_base, pid_m_offset = 0):

    pid_b = tl.program_id(0)                            # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) + pid_m_offset             # ID of size-`TILE_SIZE_M` nodes

    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = mparams + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:]             # [TILE_SIZE_M, TILE_SIZE_K]

    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + offs_batch[None,:]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + offs_batch[None,:]

    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:])   # [TILE_SIZE_M, BLOCK_B]

    # This tile's children map to gates by integer division; constant for the whole k loop.
    offs_gate = offs_ele // GATE_CBS                        # [TILE_SIZE_M]
    gate_ptr = gate + eleblock_id * gate_stride

    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr)                          # [TILE_SIZE_M, TILE_SIZE_K]

        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:])
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:])
        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
        n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

        if TL_DOT == 1:
            partial_flows = tl.dot(epars, n_fdm_sub)
        else:
            partial_flows = tl.sum(epars[:,:,None] * n_fdm_sub[None,:,:], axis = 1)

        # The gate of THIS k-tile's parent block, one value per (child gate, sample). `-1` means the
        # parent block and this child block are not connected; the `-inf` it produces drops the tile
        # in the accumulate below, which is what the ungated kernel computes for a padded tile too.
        gbase = tl.load(gate_ptr + k)
        lphi = tl.where(
            gbase >= 0,
            tl.load(ext + (gbase + ext_base + offs_gate)[:,None] * batch_size + offs_batch[None,:],
                    mask = mask_batch[None,:], other = 0.0),
            -float("inf")
        )                                                   # [TILE_SIZE_M, BLOCK_B]

        partial_flows_max = emars + log_n_fdm_max + lphi
        acc = tl.where(partial_flows_max == -float("inf"),
            acc,
            tl.where(partial_flows_max > acc,
                tl.log(partial_flows + tl.exp(acc - partial_flows_max) + 1e-32) + partial_flows_max,
                tl.log(tl.exp(partial_flows_max - acc) * partial_flows + 1.0) + acc
            )
        )

        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[:,None] * batch_size
        nflows_ptr += parids_inc[:,None] * batch_size
        parids_inc_ptr += ptr_inc_step

    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


@triton_jit
def _bs_triton_par_kernel(node_flows, node_mars, element_mars, mparams, param_flows,
                          ext, gate, nids, cids, pids, pfids,
                          batch_size: tl.constexpr, num_edges: tl.constexpr,
                          TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr,
                          TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, TL_DOT: tl.constexpr,
                          NODE_CBS: tl.constexpr, GATE_CBS: tl.constexpr,
                          gate_stride: tl.constexpr, ext_base, pid_m_offset = 0):
    """
    Gated parameter-flow backward, a fork of `_bk_triton_block_sparse_par_kernel_rmw`.

    The counterpart of `_bs_triton_ele_kernel`, and the reason it exists is the same: without it the
    param flows have no Triton path, so their supported regime is narrower than the element flows' --
    `num_edges % 128 == 0` (the CuTe kernel's edge subtile) and `block_size % 64 == 0`, which a
    64-state layer fails while its element flows are served fine.

    `phi` depends on the CONTRACTED index `b`, so unlike the element kernel it cannot be pulled out of
    the reduction. It folds onto `element_mars` instead -- `emars` here is already `[batch, edge]`,
    exactly the gate's shape -- which is where the CuTe fork puts it too, and leaves the
    `node_mars == -inf` branch untouched. `-inf` for an absent gate zeroes the edge through
    `exp(emars + ...)`, matching what the ungated kernel computes for a padded edge.

    Non-atomic read-add-store, so it inherits the parent's requirement that `pfids` be collision-free.
    """
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1) + pid_m_offset

    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + edge_start[None,:] * batch_size + offs_batch[:,None]

    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nflows_ptr = node_flows + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

    # Each edge's gate row. Constant across the batch loop, so resolved once: which child block the
    # edge falls in picks the row, and where it sits inside that block picks the gate.
    gbase = tl.load(gate + nblock_id * gate_stride + (offs_edge // NODE_CBS))
    grow = gbase + ext_base + ((offs_edge % NODE_CBS) // GATE_CBS)
    ghas = gbase >= 0

    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0)
        lphi = tl.load(ext + grow[None,:] * batch_size + offs_batch[:,None],
                       mask = mask_batch[:,None] & ghas[None,:], other = 0.0)
        emars = tl.where(ghas[None,:], emars + lphi, -float("inf"))

        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:], other = 0.0)
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:], other = 0.0)
        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
        n_fdm_sub = tl.where(log_n_fdm_max[None,:] != -float("inf"),
                             tl.exp(log_n_fdm - log_n_fdm_max[None,:]), 0.0)
        scaled_emars = tl.exp(emars + log_n_fdm_max[:,None])

        if TL_DOT == 1:
            acc += tl.dot(n_fdm_sub, scaled_emars)
        else:
            acc += tl.sum(n_fdm_sub[:,:,None] * scaled_emars[None,:,:], axis = 1)

        emars_ptr += TILE_SIZE_B
        nmars_ptr += TILE_SIZE_B
        nflows_ptr += TILE_SIZE_B
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars = tl.load(mparams + offs_node[:,None] + par_start[None,:])
    pflows = acc * epars

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    offsets = offs_node[:,None] + parflow_start[None,:]
    tl.store(param_flows + offsets, tl.load(param_flows + offsets) + pflows)
