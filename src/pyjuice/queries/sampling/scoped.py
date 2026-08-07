"""
Ancestral sampling against a STRUCTURAL frontier layout (see `scope_plan.py`).

The shared-parameter kernels in `sum_layer.py` / `prod_layer.py` take a list of selected
`(frontier slot, sample)` pairs, which the driver has to discover per layer with a `torch.where`, a
compaction and a per-column cursor. Those kernels are 8% of the pass; discovering their arguments is
the other 92%. Here the layout comes from the circuit instead, and three things follow:

* **the row is a scalar per program.** Each program owns one frontier row and a tile of samples, so
  `node_samples[row, b]` is contiguous along `b` -- where the pair-list form gathers from a different
  row per lane;
* **the two product kernels become one.** `count_prod_nch` existed only to produce `ch_count` for the
  cursor. With each child writing to its own scope's row there is no cursor, so counting, the
  intermediate buffers and a whole launch per layer all disappear;
* **every shape is static**, which is what a CUDA graph needs and what a circuit that is not
  structured decomposable can never give the pair-list form.

An inactive row holds `-1`, matches no compiled node, and costs a masked-off lane.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(do_not_specialize = ["batch_size"])
def _scoped_sum_kernel(nids, cids, pids, node_mars, element_mars, mparams,
                       node_samples, element_samples, rows, erows, seed_ptr, batch_size,
                       block_size: tl.constexpr, num_edges: tl.constexpr,
                       num_nblocks: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr,
                       M_NUM_TILES: tl.constexpr, BLOCK_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                       CONDITIONAL: tl.constexpr, DO_CALIBRATION: tl.constexpr):
    """One program per (frontier row, tile of samples). Draws a child for every live sample of a row."""
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)

    row = tl.load(rows + pid_r)
    erow = tl.load(erows + pid_r)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size

    # `-1` where this scope is not on the sample's path, which then matches no compiled node
    node_id = tl.load(node_samples + row * batch_size + offs_b, mask = mask_b, other = -1)
    mask_lane = mask_b & (node_id >= 0)

    # ---- locate each node in this partition's compiled rows
    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_B], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_B], dtype = tl.int64)
    for i in range(M_NUM_TILES):
        mask_nids = offs_nids < num_nblocks

        ref_nid = tl.load(nids + offs_nids, mask = mask_nids, other = 0)
        is_match = (node_id[:,None] >= ref_nid[None,:]) & \
                   (node_id[:,None] < ref_nid[None,:] + block_size) & mask_nids[None,:]

        match_local_id = tl.sum(is_match * (offs_nids[None,:] + 1), axis = 1)
        match_local_offset = tl.sum(is_match * (node_id[:,None] - ref_nid[None,:]), axis = 1)

        local_nids = tl.where(match_local_id > 0, match_local_id - 1, local_nids)
        local_nid_offs = tl.where(match_local_id > 0, match_local_offset, local_nid_offs)

        offs_nids += BLOCK_M

    mask_lane = mask_lane & (local_nids >= 0)

    # The seed is LOADED, not passed as a scalar: CUDA-graph capture bakes scalar kernel arguments
    # in, so a captured pass seeded that way would redraw the identical sample on every replay. The
    # offset is unique per (row, sample), so rows do not share a stream.
    rnd_val = tl.rand(tl.load(seed_ptr), row * batch_size + offs_b)

    if CONDITIONAL:
        nmars = tl.load(node_mars + node_id * batch_size + offs_b, mask = mask_lane, other = 0.0)

    # ---- the normalizer, when the edge weights are not known to sum to one
    #
    # The walk below compares `rnd_val` drawn on `[0, 1)` against a running cumsum, which is only a
    # draw from the right distribution if the weights total one. They do for a circuit built the
    # usual way, and for any conditional pass (`theta * exp(emars - nmars)` sums to one by the
    # definition of `nmars`), so this costs a second trip through the gathers and is opt-in. Where
    # the totals are NOT one, skipping it biases the draw toward the early edges and sends the tail
    # into the `last_edge` fallback.
    if DO_CALIBRATION:
        sum_pars = tl.zeros([BLOCK_B], dtype = tl.float32)
        offs_cal = tl.arange(0, BLOCK_K)
        mask_cal = offs_cal < num_edges
        for i in range(K_NUM_TILES):
            m1 = mask_lane[None,:] & mask_cal[:,None]

            cal_par_id = tl.load(pids + local_nids[None,:] * num_edges + offs_cal[:,None], mask = m1, other = 0)
            cal_pars = tl.load(mparams + cal_par_id + local_nid_offs[None,:], mask = m1, other = 0.0)

            if CONDITIONAL:
                cal_ch_id = tl.load(cids + local_nids[None,:] * num_edges + offs_cal[:,None], mask = m1, other = 0)
                cal_mars = tl.load(element_mars + cal_ch_id * batch_size + offs_b[None,:], mask = m1, other = 0.0)
                cal_pars = cal_pars * tl.exp(cal_mars - nmars[None,:])

            sum_pars += tl.sum(cal_pars, axis = 0)

            offs_cal += BLOCK_K
            mask_cal = offs_cal < num_edges

        # The epsilon keeps `rnd_val == sum_pars` from walking off the end when the two agree exactly
        rnd_val = rnd_val * sum_pars - 1e-12

    # ---- inverse-CDF walk over the edges
    chids = tl.zeros([BLOCK_B], dtype = tl.int64) - 1
    last_edge = tl.zeros([BLOCK_B], dtype = tl.int64) - 1
    offs_edge = tl.arange(0, BLOCK_K)
    mask_edge = offs_edge < num_edges
    for i in range(K_NUM_TILES):
        m2 = mask_lane[None,:] & mask_edge[:,None]

        par_id = tl.load(pids + local_nids[None,:] * num_edges + offs_edge[:,None], mask = m2, other = 0)
        epars = tl.load(mparams + par_id + local_nid_offs[None,:], mask = m2, other = 0.0)

        if CONDITIONAL:
            ch_id = tl.load(cids + local_nids[None,:] * num_edges + offs_edge[:,None], mask = m2, other = 0)
            emars = tl.load(element_mars + ch_id * batch_size + offs_b[None,:], mask = m2, other = 0.0)
            epars = epars * tl.exp(emars - nmars[None,:])

        cum = tl.cumsum(epars, axis = 0)
        local_chids = tl.sum((rnd_val[None,:] >= cum).to(tl.int64), axis = 0)

        is_overflow = (local_chids == BLOCK_K)
        rnd_val = tl.where(is_overflow, rnd_val - tl.sum(epars, axis = 0), rnd_val)
        chids = tl.where(is_overflow | (chids > -1), chids, local_chids + i * BLOCK_K)

        last_edge = tl.maximum(last_edge,
                               tl.max(tl.where(epars > 0.0, offs_edge[:,None], -1), axis = 0).to(tl.int64))

        offs_edge += BLOCK_K
        mask_edge = offs_edge < num_edges

    # A walk that ran off the end falls back to the last edge carrying weight, rather than reading
    # `cids[row * num_edges - 1]` -- the previous row's last child
    chids = tl.where((chids < 0) | (chids >= num_edges), last_edge, chids)
    chids = tl.where(chids < 0, 0, chids)

    global_chids = tl.load(cids + local_nids * num_edges + chids, mask = mask_lane, other = 0)
    tl.store(element_samples + erow * batch_size + offs_b, global_chids, mask = mask_lane)


@triton.jit(do_not_specialize = ["batch_size"])
def _scoped_prod_kernel(nids, cids, crows, node_samples, element_samples, rows, batch_size,
                        block_size: tl.constexpr, num_edges: tl.constexpr,
                        num_nblocks: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr,
                        M_NUM_TILES: tl.constexpr, BLOCK_C: tl.constexpr, C_NUM_TILES: tl.constexpr):
    """
    Expand every live element of a row into its children, each written to its OWN scope's row.

    This is `count_prod_nch` and `sample_prod_layer` merged: the count existed only to drive a
    cursor, and `crows` replaces the cursor.
    """
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)

    row = tl.load(rows + pid_r)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size

    ele_id = tl.load(element_samples + row * batch_size + offs_b, mask = mask_b, other = -1)
    mask_lane = mask_b & (ele_id >= 0)

    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_B], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_B], dtype = tl.int64)
    for i in range(M_NUM_TILES):
        mask_nids = offs_nids < num_nblocks

        ref_nid = tl.load(nids + offs_nids, mask = mask_nids, other = 0)
        is_match = (ele_id[:,None] >= ref_nid[None,:]) & \
                   (ele_id[:,None] < ref_nid[None,:] + block_size) & mask_nids[None,:]

        match_local_id = tl.sum(is_match * (offs_nids[None,:] + 1), axis = 1)
        match_local_offset = tl.sum(is_match * (ele_id[:,None] - ref_nid[None,:]), axis = 1)

        local_nids = tl.where(match_local_id > 0, match_local_id - 1, local_nids)
        local_nid_offs = tl.where(match_local_id > 0, match_local_offset, local_nid_offs)

        offs_nids += BLOCK_M

    mask_lane = mask_lane & (local_nids >= 0)

    offs_child = tl.arange(0, BLOCK_C)
    mask_child = offs_child < num_edges
    for i in range(C_NUM_TILES):
        m2 = mask_lane[:,None] & mask_child[None,:]

        c_ids = tl.load(cids + local_nids[:,None] * num_edges + offs_child[None,:], mask = m2, other = 0)
        c_row = tl.load(crows + local_nids[:,None] * num_edges + offs_child[None,:], mask = m2, other = -1)

        # `crows` is `-1` for a padded slot, so the two tests agree by construction
        write = m2 & (c_ids > 0) & (c_row >= 0)
        tl.store(node_samples + c_row * batch_size + offs_b[:,None],
                 c_ids + local_nid_offs[:,None], mask = write)

        offs_child += BLOCK_C
        mask_child = offs_child < num_edges


def scoped_sum_layer(layer, nids, cids, pids, node_mars, element_mars, params,
                     node_samples, element_samples, rows, erows, seed_ptr, conditional,
                     do_calibration = False):
    num_rows = rows.numel()
    if num_rows == 0:
        return None

    batch_size = node_samples.size(1)
    num_edges = cids.size(1)
    num_nblocks = nids.size(0)

    BLOCK_K = max(2, min(512, triton.next_power_of_2(num_edges)))
    BLOCK_M = max(2, min(512, triton.next_power_of_2(num_nblocks)))
    BLOCK_B = max(2, min(2048 // BLOCK_K, 2048 // BLOCK_M, triton.next_power_of_2(batch_size)))

    _scoped_sum_kernel[(num_rows, triton.cdiv(batch_size, BLOCK_B))](
        nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples,
        rows, erows, seed_ptr, batch_size,
        layer.block_size, num_edges, num_nblocks, BLOCK_B, BLOCK_M,
        triton.cdiv(num_nblocks, BLOCK_M), BLOCK_K, triton.cdiv(num_edges, BLOCK_K),
        1 if conditional else 0, 1 if do_calibration else 0,
    )
    return None


def scoped_prod_layer(layer, nids, cids, crows, node_samples, element_samples, rows):
    num_rows = rows.numel()
    if num_rows == 0:
        return None

    batch_size = node_samples.size(1)
    num_edges = cids.size(1)
    num_nblocks = nids.size(0)

    BLOCK_C = max(2, min(128, triton.next_power_of_2(num_edges)))
    BLOCK_M = max(2, min(512, triton.next_power_of_2(num_nblocks)))
    BLOCK_B = max(2, min(2048 // BLOCK_C, 2048 // BLOCK_M, triton.next_power_of_2(batch_size)))

    _scoped_prod_kernel[(num_rows, triton.cdiv(batch_size, BLOCK_B))](
        nids, cids, crows, node_samples, element_samples, rows, batch_size,
        layer.block_size, num_edges, num_nblocks, BLOCK_B, BLOCK_M,
        triton.cdiv(num_nblocks, BLOCK_M), BLOCK_C, triton.cdiv(num_edges, BLOCK_C),
    )
    return None
