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
                          parpids_start, parpids_increment, grad_ext,
                          batch_size: tl.constexpr, ptr_inc_step: tl.constexpr,
                          BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr,
                          K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                          TL_DOT: tl.constexpr, GATE_CBS: tl.constexpr,
                          gate_stride: tl.constexpr, ext_base, WRITE_GRAD: tl.constexpr = 0,
                          GRAD_ATOMIC: tl.constexpr = 1, pid_m_offset = 0):

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
    # k-tiles per PARENT BLOCK: the tiles that share one gate row.
    TPB: tl.constexpr = BLOCK_SIZE_K // TILE_SIZE_K if BLOCK_SIZE_K > TILE_SIZE_K else 1
    GROWS: tl.constexpr = TILE_SIZE_M // GATE_CBS if TILE_SIZE_M >= GATE_CBS else 1
    gacc = tl.zeros([GROWS, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr)                          # [TILE_SIZE_M, TILE_SIZE_K]

        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:])
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:])
        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
        n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

        # ---- ONE CONTRACTION PER PARENT GROUP ----
        #
        # `phi` is constant over the parents of ONE node block, which is what lets it leave the
        # contraction and become a shift of this tile's max. When `ptr_inc_step > 1` a k-tile spans
        # `ptr_inc_step` DIFFERENT parent blocks (`TILE_SIZE_K > BLOCK_SIZE_K`), so one shift no longer
        # describes the tile and the contraction has to split -- `phi` cannot be folded into the
        # `[K, B]` operand either, because it depends on the child gate as well (the M axis).
        #
        # This is why the layer used to be refused outright at `ptr_inc_step != 1`, which in practice
        # meant EVERY 32-wide gated sum layer (`ptr_inc_step = TILE_SIZE_K // block_size`).
        #
        # `ptr_inc_step` is a constexpr, so the common case compiles to exactly the single dot and
        # single merge it always did -- the split costs nothing where it is not needed. Everything
        # else in this kernel already handled several groups: `offs_edge_gid` indexes the group and
        # `par_start` / `edge_start` are already vectors over it.
        for g in tl.static_range(ptr_inc_step):
            if ptr_inc_step == 1:
                nsub_g = n_fdm_sub
            else:
                # Mask to this group's lanes rather than re-loading them: the operands are already in
                # registers, and a masked-out lane contributes an exact zero to the dot.
                nsub_g = tl.where((offs_edge_gid == g)[:,None], n_fdm_sub, 0.0)

            if TL_DOT == 1:
                partial_flows = tl.dot(epars, nsub_g)
            else:
                # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                partial_flows = tl.sum(tl.trans(epars)[:,:,None] * nsub_g[:,None,:], axis = 0)

            # The gate of THIS group's parent block, one value per (child gate, sample). `-1` means
            # the parent block and this child block are not connected; the `-inf` it produces drops
            # the group in the accumulate below, which is what the ungated kernel computes for a
            # padded tile too. The table carries one entry per (k-tile, group); at `ptr_inc_step == 1`
            # that is one per k-tile, exactly as before.
            gbase = tl.load(gate_ptr + k * ptr_inc_step + g)
            # `gbase >= 0` has to be in the LOAD MASK, not only in the `tl.where`: Triton evaluates
            # both arms, so the sentinel would otherwise be read at a negative row offset -- out of
            # bounds whenever `gbase + ext_base < 0`, which is every disconnected block in the first
            # slot.
            lphi = tl.where(
                gbase >= 0,
                tl.load(ext + (gbase + ext_base + offs_gate)[:,None] * batch_size + offs_batch[None,:],
                        mask = mask_batch[None,:] & (gbase >= 0), other = 0.0),
                -float("inf")
            )                                               # [TILE_SIZE_M, BLOCK_B]

            partial_flows_max = emars + log_n_fdm_max + lphi

            if WRITE_GRAD:
                contrib = partial_flows * tl.exp(partial_flows_max)
                if TILE_SIZE_M >= GATE_CBS:
                    gacc += tl.sum(tl.reshape(contrib, (TILE_SIZE_M // GATE_CBS, GATE_CBS, BLOCK_B)),
                                   axis = 1)
                else:
                    gacc += tl.sum(contrib, axis = 0)[None,:]

                # Emit once per parent block. At `ptr_inc_step == 1` that is every TPB k-tiles, the
                # in-register accumulation that made this affordable. At `ptr_inc_step > 1` a k-tile
                # is WIDER than a parent block, so `TPB` is 1 and every group emits its own row --
                # there is nothing to accumulate across.
                if (ptr_inc_step > 1) or ((k % TPB) == TPB - 1):
                    if TILE_SIZE_M >= GATE_CBS:
                        grow = gbase + ext_base + (tile_id * TILE_SIZE_M) // GATE_CBS \
                               + tl.arange(0, TILE_SIZE_M // GATE_CBS)
                    else:
                        grow = gbase + ext_base + (tile_id * TILE_SIZE_M) // GATE_CBS \
                               + tl.arange(0, 1)
                    if GRAD_ATOMIC:
                        tl.atomic_add(grad_ext + grow[:,None] * batch_size + offs_batch[None,:], gacc,
                                      mask = mask_batch[None,:] & (gbase >= 0))
                    else:
                        tl.store(grad_ext + grow[:,None] * batch_size + offs_batch[None,:], gacc,
                                 mask = mask_batch[None,:] & (gbase >= 0))
                    gacc = gacc * 0.0

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
                          gate_stride: tl.constexpr, ext_base, pid_m_offset = 0,
                          PADDED: tl.constexpr = 0, PF_ATOMIC: tl.constexpr = 0,
                          GATE_BS: tl.constexpr = 0, N_NODE_GATES: tl.constexpr = 1,
                          gate_nstride = None):
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

    THE WRITE IS NOT UNCONDITIONALLY SAFE, and the two flags say which hazard applies. Unlike the CuTe
    and small-batch param forks -- which the sum layer only dispatches to from BEHIND
    `_par_flow_collision_free` -- this kernel is the unconditional fallback (`_ext_bw_par_triton_hook`,
    `sum_layer.py`), so it is reached exactly when that predicate FAILS. Two things make it fail, and
    each needs its own flag:

    * `PF_ATOMIC` -- two programs genuinely accumulate into ONE `param_flows` slot. Ordinary parameter
      tying does this whenever two members of a tie group land in the same layer: compilation gives
      them one `_param_flow_range`, so their compiled `pfids` rows are identical. A read-add-store then
      loses one contributor's update. MEASURED on such a layer: 11 of 12 identical runs differed
      bitwise, spread 38% of the value -- while the same shape untied, and the same collision on an
      UNGATED layer (which correctly picks its atomic kernel at `sum_layer.py`'s par dispatch), are
      both bit-stable over 12 runs.
    * `PADDED` -- some compiled slots are padding. A padded slot carries `cids == 0`, `pids == 0` and
      `pfids == 0`; `param_flows` has no dummy prefix, so slot 0 is owned by a REAL edge block. The
      padded lane's `pflows` is already exactly zero, but the WRITE is not free: its read-add-store of
      `+0.0` racing a real program's discards the real update. Masking the lane out of the write --
      rather than tolerating it with an atomic -- also restores collision-freedom over the slots that
      ARE written, which is what lets a ragged untied layer keep the fast non-atomic path.

    The two are orthogonal and both default off, so a dense untied layer compiles to exactly the code
    this kernel had before either flag existed.
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
    #
    # The column is BOUNDED. `offs_edge` runs over the compiled `num_edges`, which is padded up to a
    # power of two, while the gate table is only `ext_max_n_eblks` columns wide -- so whenever the
    # widest row's edge-block count is not a power of two the column runs past the row and reads the
    # NEXT row's gate, which is a valid `>= 0` offset and therefore silently defeats `ghas`; on the
    # last row it reads past the tensor entirely. MEASURED on a uniform 3-edge-block layer at
    # `node_cbs = 32`: 4 columns indexed against a 3-wide table, rows 0-2 picking up gates 16/32/48
    # and row 3 running off the end. The CuTe fork guards exactly this (`j < gate_stride ? gt[j] : -1`)
    # and the small-batch fork host-checks it; this one did neither. Unreachable today only because the
    # forward refuses every layer with a padded column, so the bound goes in before that changes.
    offs_gcol = offs_edge // NODE_CBS
    gbase = tl.load(gate + nblock_id * gate_stride + offs_gcol,
                    mask = offs_gcol < gate_stride, other = -1)
    # The NODE-AXIS gate row, as in the forward. A SCALAR because the launcher keeps
    # `TILE_SIZE_M <= GATE_BS`, so one M tile lies inside a single node gate; `Ck` is per row because
    # two `ns` on a layer can differ. Compiled away entirely when there is one gate per node block.
    node_gate_off = 0
    if N_NODE_GATES > 1:
        node_gate_off = ((tile_id * TILE_SIZE_M) // GATE_BS) * tl.load(gate_nstride + nblock_id)
    grow = gbase + ext_base + node_gate_off + ((offs_edge % NODE_CBS) // GATE_CBS)
    ghas = gbase >= 0

    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0)
        lphi = tl.load(ext + grow[None,:] * batch_size + offs_batch[:,None],
                       mask = mask_batch[:,None] & ghas[None,:], other = 0.0)
        emars = tl.where(ghas[None,:], emars + lphi, -float("inf"))

        # `-inf` on a masked-out BATCH lane, not 0.0 -- the same requirement as the ungated
        # `sum_backward_param_block_sparse` kernels, and for the same reason. `B_NUM_TILES` is
        # `cdiv(batch, TILE_SIZE_B)`, so the last tile is partial whenever the batch does not divide
        # evenly; with `other = 0.0` such a lane gets `log_n_fdm = 0 - 0 = 0`, hence `n_fdm_sub = 1`
        # and `scaled_emars = exp(0) = 1`, and contributes a spurious 1 to the dot for EVERY padded
        # lane. MEASURED on this layer at `block_size = 8`, batch 96: the parameter flows came out
        # 15.7x too large. `-inf` makes `log_n_fdm = -inf`, so the column's max is `-inf`,
        # `n_fdm_sub = 0` and `scaled_emars = 0` -- an exact zero contribution.
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:], other = -float("inf"))
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:], other = 0.0)
        log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)

        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
        n_fdm_sub = tl.where(log_n_fdm_max[None,:] != -float("inf"),
                             tl.exp(log_n_fdm - log_n_fdm_max[None,:]), 0.0)
        scaled_emars = tl.exp(emars + log_n_fdm_max[:,None])

        if TL_DOT == 1:
            acc += tl.dot(n_fdm_sub, scaled_emars)
        else:
            # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
            acc += tl.sum(tl.trans(n_fdm_sub)[:,:,None] * scaled_emars[:,None,:], axis = 0)

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

    # `edge_start != 0` -- not `ghas` -- is the padding predicate. Real children are elements
    # `>= num_dummy_eles > 0` and padding is element 0, so this is exact by construction and, unlike
    # `ghas`, does not depend on the gate table being indexed correctly.
    if PADDED:
        wmask = (edge_start != 0)[None,:]
        if PF_ATOMIC:
            tl.atomic_add(param_flows + offsets, pflows, mask = wmask)
        else:
            tl.store(param_flows + offsets,
                     tl.load(param_flows + offsets, mask = wmask, other = 0.0) + pflows,
                     mask = wmask)
    else:
        if PF_ATOMIC:
            tl.atomic_add(param_flows + offsets, pflows)
        else:
            tl.store(param_flows + offsets, tl.load(param_flows + offsets) + pflows)


@triton_jit
def _bs_triton_phigrad_logz_kernel(node_flows, log_z, sigma, ext, gate, grad_ext, nids,
                                   batch_size: tl.constexpr, n_gates: tl.constexpr,
                                   N_CHILD_GATES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                                   BLOCK_B: tl.constexpr, GATE_TILE: tl.constexpr,
                                   USE_DOT: tl.constexpr, gate_stride: tl.constexpr, ext_base,
                                   TILE_SIZE_M: tl.constexpr = 0):
    """
    The log-Z half of `d LL / d log phi`:

        term2[g, b] = phi_b[g] * sum_n sigma[g,n] * exp(node_flows[n,b] - log Z[n,b])

    which is a MATMUL over the node axis, `sigma @ v`. Recognising that is the whole point: the first
    version ran one program per (gate, node block, batch tile) and re-loaded the entire
    [BLOCK_SIZE_M, BLOCK_B] block of `node_flows` and `log Z` for EVERY gate -- 128x redundant at
    gate_cbs=8 -- which made it 96-99% of the gradient's cost and scaled with 1/gate_cbs, batch and K,
    the three things that set the number of gates and the block size.

    Tiling the gate axis loads that block once per GATE_TILE gates instead of once per gate.

    `log Z` comes free from the forward's cache; `sigma` is recomputed only when `params` changes.
    """
    pid_g = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_b = tl.program_id(2)

    # The NODE axis is tiled, and grid-y carries `(node block, node tile)`. Holding the whole block
    # put `node_flows` and `log Z` -- `[BLOCK_SIZE_M, BLOCK_B]` each -- beyond shared memory once the
    # block got wide: at `block_size = 2048` EVERY `(GATE_TILE, BLOCK_B)` candidate was refused
    # (`Required: 139264, Hardware limit: 101376`), so a 2048-state HMM could not run its gate
    # gradient at all, with or without `apply_z_correction`.
    #
    # Safe because the contraction reduces over the NODE axis: each tile produces a PARTIAL `out`, and
    # the emission below is already `tl.atomic_add`. The shift `mx` is taken over `lphi`, which has no
    # node index, so every tile shares it and the partial sums are commensurate.
    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    pid_nb = pid_y // M_TILES
    pid_m = pid_y % M_TILES

    offs_g = pid_g * GATE_TILE + tl.arange(0, GATE_TILE)
    mask_g = offs_g < n_gates
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size
    offs_node = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)

    # loaded ONCE for the whole gate tile
    off_nids = tl.load(nids + pid_nb)
    nf = tl.load(node_flows + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:],
                 mask = mask_batch[None,:], other = -float("inf"))
    lz = tl.load(log_z + (pid_nb * BLOCK_SIZE_M + offs_node[:,None]) * batch_size + offs_batch[None,:],
                 mask = mask_batch[None,:], other = 0.0)
    j = offs_g // N_CHILD_GATES
    d = offs_g % N_CHILD_GATES
    # BOUNDED by the gate table's width, exactly as in `_bs_triton_par_kernel`. `n_gates` is derived
    # from the compiled `num_edges`, which is padded up to a power of two, while the table is only
    # `ext_max_n_eblks` wide -- so whenever the widest row's edge-block count is not a power of two
    # the column runs past the row and picks up the NEXT row's gate, which is a valid `>= 0` offset
    # and therefore survives the `gbase >= 0` masks below. MEASURED on a block-sparse ragged layer
    # (`n_eblks = 4`, `gate_stride = 3`): an out-of-bounds atomic 349 KB past the gradient buffer.
    # Unreachable until the forward began accepting layers whose rows differ in edge-block count.
    gbase = tl.load(gate + pid_nb * gate_stride + j, mask = mask_g & (j < gate_stride), other = -1)
    row = gbase + ext_base + d
    lphi = tl.load(ext + row[:,None] * batch_size + offs_batch[None,:],
                   mask = mask_g[:,None] & mask_batch[None,:] & (gbase >= 0)[:,None],
                   other = -float("inf"))

    # SHIFTED by the tile's largest gate, because the matmul cannot carry `phi` inside the exponent
    # the way a per-element form can (the [M, B] operand has no gate axis). Unshifted, `exp(nf - lz)`
    # underflows to exactly 0 for any gate past ~88 -- `log Z >= log phi` -- and the whole log-Z term
    # silently vanishes, which shows up as the zero-sum invariant going to 1.0 rather than 0. The
    # shift cancels exactly, and both halves are then bounded: `log Z >= mx + log sigma`, so the
    # exponent here is at most `-log sigma`, and `lphi - mx <= 0` below.
    mx = tl.max(lphi, axis = 0)[None,:]                                 # [1, B]
    mx = tl.where(mx == -float("inf"), 0.0, mx)
    v = tl.where(mask_batch[None,:], tl.exp(nf - lz + mx), 0.0)         # [M, B]

    sg = tl.load(sigma + (pid_nb * BLOCK_SIZE_M + offs_node[:,None]) * n_gates + offs_g[None,:],
                 mask = mask_g[None,:], other = 0.0)                    # [M, G]

    # A DOT, not a broadcast-sum: `tl.sum(sg[:,:,None] * v[:,None,:], axis=0)` materializes an
    # [M, G, B] intermediate -- half a million elements here -- and was the dominant cost even after
    # the redundant loads were tiled away. `sigma^T @ v` is the same contraction on tensor cores.
    #
    # Only where it pays. `tl.dot` is TF32, and at small batch this kernel is already cheap while the
    # rest of the gradient path is exact fp32 -- so trading that accuracy away there buys nothing.
    if USE_DOT:
        out = tl.dot(tl.trans(sg), v)                                   # [G, B]
    else:
        out = tl.sum(sg[:,:,None] * v[:,None,:], axis = 0)

    # SUBTRACTED: the gradient is (Ntilde term - logZ term) and the element kernel added the first.
    # `exp(lphi - mx)` undoes the shift applied to `v` above, and is bounded by 1.
    tl.atomic_add(grad_ext + row[:,None] * batch_size + offs_batch[None,:],
                  -tl.exp(lphi - mx) * out,
                  mask = mask_g[:,None] & mask_batch[None,:] & (gbase >= 0)[:,None])


@triton_jit
def _bs_triton_denom_w_kernel(node_flows, log_z, W, ext, gate, nids,
                              batch_size: tl.constexpr, n_gates: tl.constexpr,
                              TILE_SIZE_B: tl.constexpr, TILE_SIZE_G: tl.constexpr,
                              TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                              NODE_CBS: tl.constexpr, GATE_CBS: tl.constexpr,
                              gate_stride: tl.constexpr, ext_base,
                              B_TILES_PER_PROG: tl.constexpr, USE_DOT: tl.constexpr = 1,
                              DOT_IEEE: tl.constexpr = 1, W_ATOMIC: tl.constexpr = 0):
    """
    The whole of `F-`'s per-batch work: the contraction, done in GATE space.

        W[n, g] += sum_b u[n,b] * phi[g,b],      u[n,b] = exp(node_flows[n,b] - log Z[n,b])

    `theta[n,c]` has no batch index, so it leaves the batch sum entirely and

        F-[n,c] = sum_b f_b[n] * theta_b[n,c] = theta[n,c] * W[n, g(c)]

    `W` is therefore a SUFFICIENT accumulator for `F-`, and the only one the circuit stores:
    `_bs_triton_denom_scatter_kernel` rebuilds `F-` from it once per EM step. That is both the memory
    argument (`W` is `gate_cbs` times smaller than a per-edge `F-` -- MEASURED 240.0 MB -> 1.88 MB on a
    2048-state gated HMM, 18.5% of peak training memory) and the traffic one: the contraction depends
    only on the GATE, of which there are `num_edges / GATE_CBS`. The kernel this replaced contracted per
    EDGE tile, repeating the same contraction for all `gate_cbs` edges sharing a gate -- MEASURED at
    block_size 128 / 512 edges / batch 512, `node_flows` and `log Z` were re-read 16x, 32 MB of that
    kernel's 38 MB, leaving it ~9x above its traffic floor. Here they are read once per GATE tile, and
    the gate axis usually fits one tile (16 gates at that shape), so once in total.

    `g(c) = c // GATE_CBS` exactly, because `NODE_CBS` is a multiple of `GATE_CBS`:
    `(c // NODE_CBS) * (NODE_CBS // GATE_CBS) + (c % NODE_CBS) // GATE_CBS == c // GATE_CBS`.
    The scatter uses that identity to find a gate without consulting the gate table at all.

    THE SHIFT IS LOAD-BEARING. `phi` cannot ride inside the exponent, because the `[M, B]` operand has
    no gate axis -- so `exp(log phi)` would overflow for any router logit past ~88, and a router logit is
    unbounded. Each batch column is therefore shifted by the largest `log phi` over the tile's gates:

        u[n,b] = exp(node_flows - log Z + mx[b]),   p[b,g] = exp(log phi[g,b] - mx[b])

    The shift cancels exactly within each `W[n,g]`, so tiles need NOT agree on it, and both factors stay
    bounded: `p <= 1`, and `log Z >= mx + log sigma` gives `node_flows - log Z + mx <= node_flows - log
    sigma` (the bound `_bs_triton_phigrad_logz_kernel` also relies on). Where a tile's gates are all far
    below the global maximum `u` underflows to 0, which is correct -- the true contribution is then
    equally negligible.

    A masked-out batch lane contributes an exact zero (both operands are zeroed), so a partial final
    batch tile is safe; `B_NUM_TILES` is a `cdiv`.

    Validated against the torch reference `BlockScaleSumParams._accumulate_denom_torch` and, end to end,
    by a finite-difference check on `d LL / d log theta = F+ - F-`.
    """
    pid_y = tl.program_id(0)                       # (node block, node tile)
    pid_g = tl.program_id(1)                       # which gate tile
    pid_s = tl.program_id(2)                       # which slice of the batch reduction

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    nblock_id = pid_y // M_TILES
    pid_m = pid_y % M_TILES
    offs_node = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)

    offs_g = pid_g * TILE_SIZE_G + tl.arange(0, TILE_SIZE_G)
    gmask = offs_g < n_gates
    # The first edge carrying each gate is what locates it in the gate table.
    c_rep = offs_g * GATE_CBS
    gcol = c_rep // NODE_CBS
    gbase = tl.load(gate + nblock_id * gate_stride + gcol,
                    mask = gmask & (gcol < gate_stride), other = -1)
    grow = gbase + ext_base + (c_rep % NODE_CBS) // GATE_CBS
    ghas = (gbase >= 0) & gmask

    off_nids = tl.load(nids + nblock_id)
    offs_batch = pid_s * (B_TILES_PER_PROG * TILE_SIZE_B) + tl.arange(0, TILE_SIZE_B)
    nf_ptr = node_flows + (off_nids + offs_node[:,None].to(tl.int64)) * batch_size + offs_batch[None,:]
    lz_ptr = log_z + (nblock_id.to(tl.int64) * BLOCK_SIZE_M + offs_node[:,None]) * batch_size \
             + offs_batch[None,:]

    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_G], dtype = tl.float32)
    for _ in range(B_TILES_PER_PROG):
        mask_b = offs_batch < batch_size
        nf = tl.load(nf_ptr, mask = mask_b[None,:], other = -float("inf"))      # [M, B]
        lz = tl.load(lz_ptr, mask = mask_b[None,:], other = 0.0)                # [M, B]
        lphi = tl.load(ext + grow[None,:] * batch_size + offs_batch[:,None],
                       mask = mask_b[:,None] & ghas[None,:], other = -float("inf"))   # [B, G]

        mx = tl.max(lphi, axis = 1)                                             # [B]
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        u = tl.where(mask_b[None,:], tl.exp(nf - lz + mx[None,:]), 0.0)         # [M, B]
        p = tl.where(mask_b[:,None] & ghas[None,:], tl.exp(lphi - mx[:,None]), 0.0)   # [B, G]

        if USE_DOT:
            if DOT_IEEE:
                acc += tl.dot(u, p, input_precision = "ieee")
            else:
                acc += tl.dot(u, p)
        else:
            # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
            acc += tl.sum(tl.trans(u)[:,:,None] * p[:,None,:], axis = 0)

        offs_batch += TILE_SIZE_B
        nf_ptr += TILE_SIZE_B
        lz_ptr += TILE_SIZE_B

    wptr = W + (nblock_id * BLOCK_SIZE_M + offs_node)[:,None] * n_gates + offs_g[None,:]
    # ACCUMULATE, never store: `W` is the PC's persistent denominator buffer, so several mini-batches
    # may feed one EM step. Several batch slices also hold PARTIAL sums for the same `W[n,g]`, and only
    # then does the combine have to be atomic -- with one slice each `(n,g)` belongs to exactly one
    # program, so the cheaper read-add-write is safe.
    if W_ATOMIC:
        tl.atomic_add(wptr, acc, mask = gmask[None,:])
    else:
        tl.store(wptr, tl.load(wptr, mask = gmask[None,:], other = 0.0) + acc, mask = gmask[None,:])


@triton_jit
def _bs_triton_denom_scatter_kernel(W, mparams, out, cids, pids, pfids,
                                    num_edges: tl.constexpr, n_gates: tl.constexpr,
                                    TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                                    BLOCK_SIZE_M: tl.constexpr, GATE_CBS: tl.constexpr,
                                    pf_base, out_size, PF_ATOMIC: tl.constexpr = 0):
    """
    Reconstruct `F-[n,c] = theta[n,c] * W[n, g(c)]` and scatter it at the edge's `pfid`.

    Runs ONCE PER EM STEP (from `compute_em_correction`), not once per backward: the batch dependence
    lives entirely in `W`, so this has no batch axis at all -- it streams `theta`, gathers from the
    (tiny, cache-resident) `W`, and writes `out`. `g(c) = c // GATE_CBS` (see
    `_bs_triton_denom_w_kernel`), so the gate table is not consulted again and no gate arithmetic is
    repeated per edge.

    `out` is one `ns`'s slice of the param-flow space, and `pf_base` / `out_size` are its `pfid` range:
    the write goes to `pfid - pf_base`, masked to `[0, out_size)`. That is what keeps the peak at ONE
    node's worth of `F-` rather than the whole PC's, and it is also how a tie group is summed -- each
    member is scattered with its OWN `pf_base` into the same buffer, since the members' `pfids` are the
    same layout at different offsets.

    A padded edge (`cids == 0`, the dummy child) has `pfids == 0`, a slot a REAL edge owns, so it is
    masked out of the write rather than storing a zero into someone else's slot.
    """
    pid_k = tl.program_id(0)
    pid_y = tl.program_id(1)

    M_TILES: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    nblock_id = pid_y // M_TILES
    pid_m = pid_y % M_TILES
    offs_node = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offs_edge = pid_k * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)
    emask = offs_edge < num_edges

    cid = tl.load(cids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    real = emask & (cid != 0)

    gidx = offs_edge // GATE_CBS
    w = tl.load(W + (nblock_id * BLOCK_SIZE_M + offs_node)[:,None] * n_gates + gidx[None,:],
                mask = emask[None,:] & (gidx < n_gates)[None,:], other = 0.0)    # [M, K]

    par = tl.load(pids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    theta = tl.load(mparams + par[None,:] + offs_node[:,None], mask = emask[None,:], other = 0.0)

    pf = tl.load(pfids + nblock_id * num_edges + offs_edge, mask = emask, other = 0)
    # Rebase into `out` and mask to it. A partition may hold node blocks of SEVERAL `ns`, and only the
    # one being reconstructed belongs here; the others land outside `[0, out_size)` and are dropped.
    # The offset is clamped as well as masked, so an out-of-range lane never forms a wild address.
    off = pf[None,:] + offs_node[:,None] - pf_base
    inside = real[None,:] & (off >= 0) & (off < out_size)
    ptr = out + tl.maximum(tl.minimum(off, out_size - 1), 0)
    if PF_ATOMIC:
        tl.atomic_add(ptr, theta * w, mask = inside)
    else:
        tl.store(ptr, tl.load(ptr, mask = inside, other = 0.0) + theta * w, mask = inside)
