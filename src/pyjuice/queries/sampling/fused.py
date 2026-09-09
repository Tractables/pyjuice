"""
The whole conditional top-down pass in ONE kernel, for deep narrow circuits.

`scoped.py` launches, per level, a product-layer forward (to rebuild `element_mars`), a sum-layer
draw, two buffer clears and a product-layer expansion. On a circuit that is DEEP and NARROW that is
the entire cost: MEASURED on a 32-variable HMM at batch 1, the pass is ~128 launches of ~2.7 us each
inside a CUDA graph, against frontier buffers of [64, 1] and [1, 1]. Nothing is compute-bound; it is
all dispatch.

Fusing is possible because **samples are independent**. A program owns a tile of the batch and walks
every level for it, so the level-to-level dependency is sequential *within* a program and no
cross-program synchronisation is ever needed -- no cooperative launch, no grid sync.

Three things fall out of doing it in one program:

* `element_mars` is never materialised. The sum draw needs the marginal of each CANDIDATE child, and
  a product node's marginal is just the sum of its own children's `node_mars`, so it is computed for
  the ~`num_edges` candidates that are actually looked at instead of for the whole layer. That is
  what the per-level product-layer recompute existed to provide.
* `element_samples` disappears with it: the drawn element is expanded into its child node rows
  immediately, so it never has to be parked in a buffer.
* both clears disappear. `element_samples.fill_(-1)` existed so a scope that is not live this draw
  reads `-1` rather than a stale id from the previous group; with no buffer there is nothing to go
  stale. The `node_samples` row clear is folded in, in the same order the unfused pass does it
  (clear the row, then write the children -- which matters when a child row IS the row just read).

**This is a fast path, not a replacement.** It is gated on shape, and `scoped.py` remains the general
implementation. See :func:`fusion_applicability`: the gate is per-level WORK, not structured
decomposability -- `build_scope_plan` is size-static for any circuit, and the families that lose here
are the SHALLOW WIDE ones (a `PD` circuit is 14 groups with 132 node blocks and fan-in 1024 per
level, ~135k elements that one program would serialise and that 28 ordinary launches do in parallel).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# A level's tables are padded to the max over levels, so a circuit whose levels differ wildly would
# pay for the padding. These bound what the fast path will accept; beyond them it declines and the
# caller falls back.
_MAX_LEVELS = 512
_MAX_SUM_EDGES = 2048
_MAX_PROD_EDGES = 64
_MAX_NBLOCKS = 64
_MAX_ROWS = 16
#: Elements touched per level per sample, above which one program is the wrong shape for the job.
_MAX_LEVEL_WORK = 8192


class FusedPlan():
    """Per-level tables, padded to a common shape and indexed by level in the kernel."""

    __slots__ = ("num_levels", "sum_nids", "sum_cids", "sum_pids", "sum_rows", "sum_nblocks",
                 "sum_bsize", "prod_nids", "prod_cids", "prod_crows", "prod_nblocks", "prod_bsize",
                 "num_rows", "num_sum_edges", "num_prod_edges", "max_nblocks", "max_rows",
                 "root_row", "root_node")


def fusion_applicability(pc, plan):
    """
    Whether the fused pass can serve this circuit, and why not when it cannot.

    :returns: `(ok, reason)` -- `reason` is None when ok.
    """
    groups = pc.inner_layer_groups
    n = len(groups)
    if n == 0 or n % 2 != 0:
        return False, f"expected an even number of inner layer groups, found {n}"
    if n // 2 > _MAX_LEVELS:
        return False, f"{n // 2} levels exceeds the {_MAX_LEVELS} the fused tables are built for"

    # The walk pairs each sum group with the product group directly below it, which is the shape
    # `_scoped_top_down` assumes when it recomputes `inner_layer_groups[layer_id - 1]`.
    for idx in range(n - 1, -1, -1):
        want_sum = (idx % 2 == 1)
        if groups[idx].is_sum() != want_sum:
            return False, "inner layer groups do not alternate product/sum"

    max_se = max_pe = max_nb = max_rw = 0
    for idx in range(n - 1, 0, -2):
        sg, pg = groups[idx], groups[idx - 1]
        slayers, players = list(sg), list(pg)
        if len(slayers) != 1 or len(players) != 1:
            return False, "a layer group holds more than one layer"
        sl, pl = slayers[0], players[0]
        if sl.num_fw_partitions != 1 or pl.num_fw_partitions != 1:
            return False, "a layer is split across more than one partition"
        max_se = max(max_se, sl.partitioned_cids[0].size(1))
        max_pe = max(max_pe, pl.partitioned_cids[0].size(1))
        max_nb = max(max_nb, sl.partitioned_nids[0].size(0), pl.partitioned_nids[0].size(0))
        max_rw = max(max_rw, plan.sum_rows[id(sl)].numel())

    if max_se > _MAX_SUM_EDGES:
        return False, f"sum fan-in {max_se} exceeds {_MAX_SUM_EDGES}"
    if max_pe > _MAX_PROD_EDGES:
        return False, f"product fan-in {max_pe} exceeds {_MAX_PROD_EDGES}"
    if max_nb > _MAX_NBLOCKS:
        return False, f"{max_nb} node blocks per layer exceeds {_MAX_NBLOCKS}"
    if max_rw > _MAX_ROWS:
        return False, f"{max_rw} frontier rows per level exceeds {_MAX_ROWS}"

    # The gate that actually matters. One program serialises a level; that is a win only while the
    # level is small. A shallow wide circuit does more work per level than a program should hold, and
    # the ordinary per-layer launches parallelise it properly.
    work = max_rw * max_se
    if work > _MAX_LEVEL_WORK:
        return False, (f"{work} elements per level per sample exceeds {_MAX_LEVEL_WORK}; this "
                       f"circuit is too wide for a single program to serialise")
    return True, None


def build_fused_plan(pc, plan):
    """Flatten the per-level tables into padded, level-indexed tensors. `None` when not applicable."""
    ok, _ = fusion_applicability(pc, plan)
    if not ok:
        return None

    groups = pc.inner_layer_groups
    dev = pc.device
    idxs = list(range(len(groups) - 1, 0, -2))          # top-down: sum group indices
    L = len(idxs)

    sum_layers = [list(groups[i])[0] for i in idxs]
    prod_layers = [list(groups[i - 1])[0] for i in idxs]

    E_S = max(l.partitioned_cids[0].size(1) for l in sum_layers)
    E_P = max(l.partitioned_cids[0].size(1) for l in prod_layers)
    NB = max(max(l.partitioned_nids[0].size(0) for l in sum_layers),
             max(l.partitioned_nids[0].size(0) for l in prod_layers))
    R = max(plan.sum_rows[id(l)].numel() for l in sum_layers)

    def z(*shape, fill = 0):
        return torch.full(shape, fill, dtype = torch.long, device = dev)

    fp = FusedPlan()
    fp.num_levels = L
    fp.num_sum_edges, fp.num_prod_edges, fp.max_nblocks, fp.max_rows = E_S, E_P, NB, R
    fp.sum_nids, fp.sum_cids, fp.sum_pids = z(L, NB), z(L, NB, E_S), z(L, NB, E_S)
    fp.sum_rows = z(L, R, fill = -1)
    fp.sum_nblocks, fp.sum_bsize = z(L), z(L)
    fp.num_rows = z(L)
    fp.prod_nids, fp.prod_cids = z(L, NB), z(L, NB, E_P)
    # -1 is the padded-slot sentinel the unfused product kernel already uses
    fp.prod_crows = z(L, NB, E_P, fill = -1)
    fp.prod_nblocks, fp.prod_bsize = z(L), z(L)

    for i, (sl, pl) in enumerate(zip(sum_layers, prod_layers)):
        s_nids, s_cids, s_pids = sl.partitioned_nids[0], sl.partitioned_cids[0], sl.partitioned_pids[0]
        p_nids, p_cids = pl.partitioned_nids[0], pl.partitioned_cids[0]
        p_crows = plan.prod_crows[id(pl)][0]
        rows = plan.sum_rows[id(sl)]

        fp.sum_nids[i, :s_nids.size(0)] = s_nids
        fp.sum_cids[i, :s_cids.size(0), :s_cids.size(1)] = s_cids
        fp.sum_pids[i, :s_pids.size(0), :s_pids.size(1)] = s_pids
        fp.sum_nblocks[i] = s_nids.size(0)
        fp.sum_bsize[i] = sl.block_size
        fp.sum_rows[i, :rows.numel()] = rows
        fp.num_rows[i] = rows.numel()

        fp.prod_nids[i, :p_nids.size(0)] = p_nids
        fp.prod_cids[i, :p_cids.size(0), :p_cids.size(1)] = p_cids
        fp.prod_crows[i, :p_crows.size(0), :p_crows.size(1)] = p_crows
        fp.prod_nblocks[i] = p_nids.size(0)
        fp.prod_bsize[i] = pl.block_size

    fp.root_row = plan.root_row
    fp.root_node = pc.root_ns._output_ind_range[0]
    return fp


@triton.jit(do_not_specialize = ["batch_size"])
def _fused_top_down_kernel(
        sum_nids, sum_cids, sum_pids, sum_rows, sum_nblocks, sum_bsize, num_rows,
        prod_nids, prod_cids, prod_crows, prod_nblocks, prod_bsize,
        node_mars, mparams, node_samples, seed_ptr, batch_size,
        num_levels: tl.constexpr,
        E_S: tl.constexpr, E_P: tl.constexpr, NB: tl.constexpr, R: tl.constexpr,
        BLOCK_B: tl.constexpr):
    """
    One program per batch tile; it walks every level for its own samples.

    :note: the level-to-level handover goes through `node_samples` in GLOBAL memory, and the threads
           that write a level's child rows are not the threads that read them at the next level --
           the write is a `[BLOCK_B, E_P]` store and the read a `[BLOCK_B]` load, spread differently
           across the block. Triton barriers shared memory for its own reductions but promises
           nothing about a global-memory dependency between threads, so every handover needs an
           explicit `tl.debug_barrier()` (a `__syncthreads()`).

           This is not theoretical. Without them the pass is silently WRONG in a way that hides at
           batch 1 and appears as soon as there is more than one program: MEASURED at batch 4, two of
           the four columns came back entirely `-1`, because level 0 read the root row before the
           seeding store to it had landed. The frontier seeding is on the host for the same reason --
           a whole-buffer fill inside the kernel is the same cross-thread hazard, one barrier earlier.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size

    offs_e = tl.arange(0, E_S)
    offs_p = tl.arange(0, E_P)
    offs_n = tl.arange(0, NB)

    seed = tl.load(seed_ptr)

    for lvl in range(num_levels):
        s_nb = tl.load(sum_nblocks + lvl)
        s_bs = tl.load(sum_bsize + lvl)
        p_nb = tl.load(prod_nblocks + lvl)
        p_bs = tl.load(prod_bsize + lvl)
        n_rows = tl.load(num_rows + lvl)

        # Every node block of this level's two layers, loaded once for the whole level.
        s_ref = tl.load(sum_nids + lvl * NB + offs_n, mask = offs_n < s_nb, other = -1)
        p_ref = tl.load(prod_nids + lvl * NB + offs_n, mask = offs_n < p_nb, other = -1)

        for r in range(R):
            row = tl.load(sum_rows + lvl * R + r, mask = r < n_rows, other = -1)
            active = (row >= 0)

            node_id = tl.load(node_samples + row * batch_size + offs_b,
                              mask = mask_b & active, other = -1)
            lane = mask_b & active & (node_id >= 0)

            # ---- locate the sampled node in this sum layer's compiled blocks
            hit = (node_id[:, None] >= s_ref[None, :]) & \
                  (node_id[:, None] < s_ref[None, :] + s_bs) & (s_ref[None, :] >= 0)
            l_nid = tl.sum(hit * (offs_n[None, :] + 1), axis = 1) - 1
            l_off = tl.sum(hit * (node_id[:, None] - s_ref[None, :]), axis = 1)
            lane = lane & (l_nid >= 0)
            safe_nid = tl.where(l_nid >= 0, l_nid, 0)

            nmars = tl.load(node_mars + node_id * batch_size + offs_b, mask = lane, other = 0.0)

            # ---- the candidate edges, and each candidate child's marginal
            m2 = lane[:, None] & (offs_e[None, :] < E_S)
            base = lvl * NB * E_S + safe_nid[:, None] * E_S + offs_e[None, :]
            par_id = tl.load(sum_pids + base, mask = m2, other = 0)
            epars = tl.load(mparams + par_id + l_off[:, None], mask = m2, other = 0.0)
            ch_id = tl.load(sum_cids + base, mask = m2, other = -1)

            # A product node's marginal is the sum of its own children's `node_mars`, so it is
            # computed here for the candidates actually consulted rather than for the whole layer --
            # this is the per-level recompute, done inline and only where it is read.
            chit = (ch_id[:, :, None] >= p_ref[None, None, :]) & \
                   (ch_id[:, :, None] < p_ref[None, None, :] + p_bs) & (p_ref[None, None, :] >= 0)
            c_blk = tl.sum(chit * (offs_n[None, None, :] + 1), axis = 2) - 1
            c_off = tl.sum(chit * (ch_id[:, :, None] - p_ref[None, None, :]), axis = 2)
            valid = m2 & (ch_id >= 0) & (c_blk >= 0)
            safe_blk = tl.where(c_blk >= 0, c_blk, 0)

            emars = tl.zeros([BLOCK_B, E_S], dtype = tl.float32)
            for j in range(E_P):
                gch = tl.load(prod_cids + lvl * NB * E_P + safe_blk * E_P + j, mask = valid, other = -1)
                m3 = valid & (gch > 0)
                emars += tl.load(node_mars + (gch + c_off) * batch_size + offs_b[:, None],
                                 mask = m3, other = 0.0)

            epars = tl.where(valid, epars * tl.exp(emars - nmars[:, None]), 0.0)

            # ---- inverse-CDF walk, matching `_scoped_sum_kernel`: the same per-(row, sample) RNG
            # offset, the same fallback to the last edge carrying weight when the walk runs off the
            # end (rather than reading the previous row's last child).
            rnd = tl.rand(seed, row * batch_size + offs_b)
            cum = tl.cumsum(epars, axis = 1)
            pick = tl.sum((rnd[:, None] >= cum).to(tl.int64), axis = 1)
            last = tl.max(tl.where(epars > 0.0, offs_e[None, :], -1), axis = 1).to(tl.int64)
            pick = tl.where((pick < 0) | (pick >= E_S), last, pick)
            pick = tl.where(pick < 0, 0, pick)

            drawn = tl.load(sum_cids + lvl * NB * E_S + safe_nid * E_S + pick, mask = lane, other = -1)

            # ---- expand the drawn element into its children's own rows.
            # Clear the row FIRST, exactly as the unfused pass does (it clears the sum layer's rows
            # before the product group writes), so a child that lands on this very row survives.
            # The barrier is what makes "first" mean anything across threads.
            tl.store(node_samples + row * batch_size + offs_b, -1, mask = mask_b & active)
            tl.debug_barrier()

            dhit = (drawn[:, None] >= p_ref[None, :]) & \
                   (drawn[:, None] < p_ref[None, :] + p_bs) & (p_ref[None, :] >= 0)
            d_blk = tl.sum(dhit * (offs_n[None, :] + 1), axis = 1) - 1
            d_off = tl.sum(dhit * (drawn[:, None] - p_ref[None, :]), axis = 1)
            wlane = lane & (drawn >= 0) & (d_blk >= 0)
            safe_d = tl.where(d_blk >= 0, d_blk, 0)

            pbase = lvl * NB * E_P + safe_d[:, None] * E_P + offs_p[None, :]
            c_ids = tl.load(prod_cids + pbase, mask = wlane[:, None], other = 0)
            c_row = tl.load(prod_crows + pbase, mask = wlane[:, None], other = -1)
            write = wlane[:, None] & (c_ids > 0) & (c_row >= 0)
            tl.store(node_samples + c_row * batch_size + offs_b[:, None],
                     c_ids + d_off[:, None], mask = write)
            # The next level reads these rows from other threads -- see the note on the kernel.
            tl.debug_barrier()


def fused_top_down(pc, fp, node_samples, seed_ptr):
    """Run the whole conditional top-down pass: a fill, a root store, and one kernel."""
    batch_size = node_samples.size(1)
    E_S, E_P, NB = fp.num_sum_edges, fp.num_prod_edges, fp.max_nblocks

    # Seeding stays on the host. Doing it inside the kernel is a cross-thread global-memory hazard
    # against the level-0 read (see the kernel's note); as two ordinary ops it is two launches, which
    # against the ~128 the unfused pass needs is not where this pass's time goes.
    node_samples.fill_(-1)
    node_samples[fp.root_row, :] = fp.root_node

    # ONE SAMPLE PER PROGRAM. A program serialises every level, so the only parallelism this pass has
    # is across the batch -- widening the tile spends that parallelism to no purpose and shrinks the
    # grid. MEASURED at batch 16 on the CoDD HMM: a tile of 8 gives 2 programs and 1.086 ms, against
    # 0.171 ms for the same work at batch 1, i.e. the wide tile serialises what the grid should
    # spread. It also keeps the [BLOCK_B, E_S, NB] candidate predicate to one row of registers.
    BLOCK_B = 1
    grid = (batch_size,)

    _fused_top_down_kernel[grid](
        fp.sum_nids, fp.sum_cids, fp.sum_pids, fp.sum_rows, fp.sum_nblocks, fp.sum_bsize,
        fp.num_rows, fp.prod_nids, fp.prod_cids, fp.prod_crows, fp.prod_nblocks, fp.prod_bsize,
        pc.node_mars, pc.params, node_samples, seed_ptr, batch_size,
        fp.num_levels, E_S, E_P, NB, fp.max_rows, BLOCK_B,
    )
    return None
