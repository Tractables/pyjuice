"""
Ancestral sampling through a GATED sum layer (`BlockScaleSumParams`), in Triton.

This is the sampling counterpart of `kernels/blockscale_forward.py`, and it is a delta on the
shared-parameter sampler in `pyjuice/queries/sampling/sum_layer.py`: same lane-per-selected-node
layout, same linear scan to locate a node in the compiled `nids`, same `[BLOCK_K, BLOCK_S]` tiles and
the same inverse-CDF walk down the edge axis. Keep the two in step. What it adds is the gate and the
normalization the gate forces:

    P(c | n, b)  propto  theta[n,c] * exp( log phi[b, g(n,c)] )                        unconditional
    P(c | n, b)  propto  theta[n,c] * exp( log phi[b, g(n,c)] + element_mars[c, b] )   conditional

`Z_b[n]` divides every child of `n` equally and so cancels out of the draw. That is what makes this
kernel independent of the forward pass: it needs neither `node_mars` nor the cached `log Z`, which is
why an unconditional draw works with no forward pass having run at all.

Three things follow from the gate being a LOG gate with unbounded values (a router's logits), and
they are why this is not simply the plain kernel with one extra multiply:

  * the weights must be accumulated with a running max, exactly as the forward stabilizes `N` and
    `Z`. `exp(log phi)` alone overflows for `log phi > ~88` and the whole node's mass becomes `inf`
    or `nan`. `theta <= 1`, so shifting by `max_c(log phi + element_mars)` keeps every term at or
    below 1 and matches what the forward's own accumulators do;
  * the sum is therefore never 1, so the draw is ALWAYS calibrated -- there is no uncalibrated fast
    path to fall back on, unlike the plain kernel where `node_mars` makes the sum 1 by construction;
  * `r` is drawn against a sum computed by THIS kernel from the same terms it then walks, so
    `r < sum` holds by construction rather than by hope.

Like the portable forward, it reads `nids` / `cids` / `pids` per lane rather than deriving addresses
from a per-tile base, so it imposes no layout requirements at all: ragged, part-padded and
block-sparse topologies are all served without a special case. A padded edge slot carries the dummy
zero parameter and a `-1` gate offset, so its weight is exactly zero and it can never be drawn.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton.jit
def _gate_weights(cids, pids, element_mars, mparams, ext, gate,
                  local_nids, local_nid_offs, node_gate_off, batch_id, offs_edge,
                  mask_lane, mask_edge, batch_size, ext_base,
                  num_edges: tl.constexpr, NODE_CBS: tl.constexpr, GATE_CBS: tl.constexpr,
                  gate_stride: tl.constexpr, CONDITIONAL: tl.constexpr):
    """
    The `[BLOCK_K, BLOCK_S]` tile of `(theta, exponent)` for one block of edges.

    Factored out because BOTH passes have to weigh the edges by exactly the same rule -- the second
    pass walks a CDF against a sum the first one computed, and a formula that drifted between them
    would put `r` past the end of the walk. One definition makes that impossible rather than merely
    unlikely.

    `theta` stays in probability space and only the exponent is shifted, which is what the forward
    does too: `theta <= 1` for a normalized sum node, so the shifted product cannot overflow.
    """
    mask = mask_edge[:,None] & mask_lane[None,:]

    par_id = tl.load(pids + local_nids[None,:] * num_edges + offs_edge[:,None], mask = mask, other = 0)
    epars = tl.load(mparams + par_id + local_nid_offs[None,:], mask = mask, other = 0.0)

    # The gate row of each edge, addressed exactly as the forward addresses it: the compiled table
    # holds the start of the edge block's gate sub-grid, and the kernel adds only the two refinements
    # WITHIN the block -- `node_gate_off` rows down, `(k % NODE_CBS) // GATE_CBS` columns across.
    #
    # BOUNDED by the table's width: `num_edges` is padded up to a power of two while the table is
    # only `ext_max_n_eblks` wide, so an unbounded column reads the next row's gate -- a valid `>= 0`
    # offset that would survive the `ghas` test below.
    offs_gcol = offs_edge // NODE_CBS
    gbase = tl.load(gate + local_nids[None,:] * gate_stride + offs_gcol[:,None],
                    mask = mask & (offs_gcol[:,None] < gate_stride), other = -1)
    grow = gbase + ext_base + node_gate_off[None,:] + ((offs_edge % NODE_CBS) // GATE_CBS)[:,None]
    ghas = gbase >= 0

    lphi = tl.load(ext + grow * batch_size + batch_id[None,:], mask = ghas & mask, other = 0.0)

    # A padded edge block has no gate, and a masked lane or edge must not contribute either. Both are
    # `-inf` in the exponent, so their weight is exactly zero without a special case downstream.
    a = tl.where(ghas & mask, lphi, -float("inf"))

    if CONDITIONAL:
        ch_id = tl.load(cids + local_nids[None,:] * num_edges + offs_edge[:,None], mask = mask, other = 0)
        emars = tl.load(element_mars + ch_id * batch_size + batch_id[None,:], mask = mask,
                        other = -float("inf"))
        a = a + emars

    return epars, a


@triton_jit
def _bs_sample_kernel(nids, cids, pids, element_mars, mparams, ext, gate, gate_nstride,
                      node_samples, element_samples, ind_target, ind_n, ind_b,
                      seed, ext_base, num_sel, batch_size,
                      block_size: tl.constexpr, num_edges: tl.constexpr, num_nblocks: tl.constexpr,
                      BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, M_NUM_TILES: tl.constexpr,
                      BLOCK_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                      NODE_CBS: tl.constexpr, GATE_CBS: tl.constexpr, GATE_BS: tl.constexpr,
                      N_NODE_GATES: tl.constexpr, gate_stride: tl.constexpr,
                      CONDITIONAL: tl.constexpr):
    """
    One lane per selected (node, sample) pair; each draws one child of its node.

    `num_sel` and `batch_size` are RUNTIME arguments rather than the plain kernel's constexprs.
    Both change with the frontier and with the number of samples requested, and specializing on them
    buys nothing -- they are a bound and a stride, not something the compiler tiles on -- while
    costing a Triton compile per distinct value.
    """

    pid_s = tl.program_id(0)

    offs_sample = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    mask_lane = offs_sample < num_sel

    node_sample_id = tl.load(ind_n + offs_sample, mask = mask_lane, other = 0)
    batch_id = tl.load(ind_b + offs_sample, mask = mask_lane, other = 0)
    node_id = tl.load(node_samples + node_sample_id * batch_size + batch_id)

    # ---- locate each node's row in this partition's compiled layout
    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_S], dtype = tl.int64)
    for i in range(M_NUM_TILES):
        mask_nids = offs_nids < num_nblocks

        ref_nid = tl.load(nids + offs_nids, mask = mask_nids, other = 0)
        # `& mask_nids` because a padded slot reads as node id 0, which a node in the first block
        # would otherwise match -- selecting whichever row happens to sit past the end of `nids`.
        is_match = (node_id[:,None] >= ref_nid[None,:]) & \
                   (node_id[:,None] < ref_nid[None,:] + block_size) & mask_nids[None,:]

        match_local_id = tl.sum(is_match * (offs_nids[None,:] + 1), axis = 1)
        match_local_offset = tl.sum(is_match * (node_id[:,None] - ref_nid[None,:]), axis = 1)

        local_nids = tl.where(match_local_id > 0, match_local_id - 1, local_nids)
        local_nid_offs = tl.where(match_local_id > 0, match_local_offset, local_nid_offs)

        offs_nids += BLOCK_M

    # Nodes belonging to another partition drop out here
    mask_lane = mask_lane & (local_nids >= 0)

    # How many rows down the gate grid this node sits, when its block holds more than one gate.
    # `Ck` is per row rather than a constexpr because two `ns` sharing a layer may have different
    # child counts; with a single node gate the whole thing compiles away.
    node_gate_off = tl.zeros([BLOCK_S], dtype = tl.int64)
    if N_NODE_GATES > 1:
        node_gate_off = (local_nid_offs // GATE_BS) * tl.load(gate_nstride + local_nids,
                                                              mask = mask_lane, other = 0)

    rnd = tl.rand(seed, offs_sample)

    chids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    last_edge = tl.zeros([BLOCK_S], dtype = tl.int64) - 1

    if K_NUM_TILES == 1:
        # ---- SINGLE PASS. The whole edge axis fits one tile, so the weights stay in registers and
        # the normalizer costs no second trip through the gathers -- which are what this kernel
        # spends its time on. This is the common case: `BLOCK_K` reaches 512.
        offs_edge = tl.arange(0, BLOCK_K)
        mask_edge = offs_edge < num_edges

        epars, a = _gate_weights(cids, pids, element_mars, mparams, ext, gate,
                                 local_nids, local_nid_offs, node_gate_off, batch_id, offs_edge,
                                 mask_lane, mask_edge, batch_size, ext_base,
                                 num_edges, NODE_CBS, GATE_CBS, gate_stride, CONDITIONAL)

        mx = tl.max(a, axis = 0)
        # Guarded for the all-`-inf` lane, where the difference would be `inf - inf = NaN`
        w = tl.where(mx[None,:] == -float("inf"), 0.0, epars * tl.exp(a - mx[None,:]))

        r = rnd * tl.sum(w, axis = 0)
        chids = tl.sum((r[None,:] >= tl.cumsum(w, axis = 0)).to(tl.int64), axis = 0)
        last_edge = tl.max(tl.where(w > 0.0, offs_edge[:,None], -1), axis = 0).to(tl.int64)

    else:
        # ---- PASS 1: the normalizer, accumulated online against a running max, exactly as the
        # forward accumulates `N` and `Z`. The max is per LANE only -- it is a max over edges.
        mx = tl.zeros([BLOCK_S], dtype = tl.float32) - float("inf")
        acc = tl.zeros([BLOCK_S], dtype = tl.float32)

        for k in range(K_NUM_TILES):
            offs_edge = tl.arange(0, BLOCK_K) + k * BLOCK_K
            mask_edge = offs_edge < num_edges

            epars, a = _gate_weights(cids, pids, element_mars, mparams, ext, gate,
                                     local_nids, local_nid_offs, node_gate_off, batch_id, offs_edge,
                                     mask_lane, mask_edge, batch_size, ext_base,
                                     num_edges, NODE_CBS, GATE_CBS, gate_stride, CONDITIONAL)

            nmx = tl.maximum(mx, tl.max(a, axis = 0))
            rescale = tl.where(nmx == -float("inf"), 0.0, tl.exp(mx - nmx))
            w = tl.where(nmx[None,:] == -float("inf"), 0.0, epars * tl.exp(a - nmx[None,:]))

            acc = acc * rescale + tl.sum(w, axis = 0)
            mx = nmx

        # ---- PASS 2: the inverse-CDF walk, against the final shift so that every tile's weights are
        # on one scale and the per-tile carry below is a plain subtraction.
        r = rnd * acc

        for k in range(K_NUM_TILES):
            offs_edge = tl.arange(0, BLOCK_K) + k * BLOCK_K
            mask_edge = offs_edge < num_edges

            epars, a = _gate_weights(cids, pids, element_mars, mparams, ext, gate,
                                     local_nids, local_nid_offs, node_gate_off, batch_id, offs_edge,
                                     mask_lane, mask_edge, batch_size, ext_base,
                                     num_edges, NODE_CBS, GATE_CBS, gate_stride, CONDITIONAL)

            w = tl.where(mx[None,:] == -float("inf"), 0.0, epars * tl.exp(a - mx[None,:]))

            local_chids = tl.sum((r[None,:] >= tl.cumsum(w, axis = 0)).to(tl.int64), axis = 0)

            is_overflow = (local_chids == BLOCK_K)
            r = tl.where(is_overflow, r - tl.sum(w, axis = 0), r)
            chids = tl.where(is_overflow | (chids > -1), chids, local_chids + k * BLOCK_K)

            last_edge = tl.maximum(last_edge,
                                   tl.max(tl.where(w > 0.0, offs_edge[:,None], -1), axis = 0).to(tl.int64))

    # A walk that ran off the end falls back to the last edge carrying weight. `r < sum` holds in
    # exact arithmetic -- `rnd < 1` and the sum is built from these very terms -- but the two passes
    # accumulate in different orders, so `r == sum` is reachable in fp32 and would leave `chids` at
    # `-1`, whose `cids[row * num_edges - 1]` is the PREVIOUS row's last child: a valid-looking id
    # from the wrong node, silently corrupting the rest of the top-down pass.
    chids = tl.where((chids < 0) | (chids >= num_edges), last_edge, chids)
    chids = tl.where(chids < 0, 0, chids)

    global_chids = tl.load(cids + local_nids * num_edges + chids, mask = mask_lane, other = 0)
    target_id = tl.load(ind_target + offs_sample, mask = mask_lane, other = 0)

    tl.store(element_samples + target_id, global_chids, mask = mask_lane)


def bs_sample_sum_layer(nids, cids, pids, element_mars, params, ext, gate, gate_nstride,
                        node_samples, element_samples, ind_target, ind_n, ind_b, seed,
                        block_size: int, node_cbs: int, gate_cbs: int, gate_bs: int,
                        ext_base: int, conditional: bool) -> None:
    """
    Draw one child for every selected node of one forward partition of a gated sum layer.

    Tiling follows the shared-parameter sampler exactly (`BLOCK_K` x `BLOCK_S` bounded at 2048
    elements, `BLOCK_M` over the `nids` scan), so the two kernels stay comparable and a difference in
    behaviour is attributable to the gate rather than to the launch.
    """

    num_sel = ind_n.size(0)
    if num_sel == 0:
        return None

    num_nblocks = nids.size(0)
    num_edges = cids.size(1)
    batch_size = node_samples.size(1)

    BLOCK_K = min(512, triton.next_power_of_2(num_edges))
    BLOCK_M = min(512, triton.next_power_of_2(num_nblocks))
    BLOCK_S = min(2048 // BLOCK_K, 2048 // BLOCK_M, max(triton.next_power_of_2(num_sel // 128), 1))

    M_NUM_TILES = triton.cdiv(num_nblocks, BLOCK_M)
    K_NUM_TILES = triton.cdiv(num_edges, BLOCK_K)

    _bs_sample_kernel[(triton.cdiv(num_sel, BLOCK_S),)](
        nids = nids, cids = cids, pids = pids, element_mars = element_mars, mparams = params,
        ext = ext, gate = gate, gate_nstride = gate_nstride,
        node_samples = node_samples, element_samples = element_samples,
        ind_target = ind_target, ind_n = ind_n, ind_b = ind_b,
        seed = seed, ext_base = ext_base, num_sel = num_sel, batch_size = batch_size,
        block_size = block_size, num_edges = num_edges, num_nblocks = num_nblocks,
        BLOCK_S = BLOCK_S, BLOCK_M = BLOCK_M, M_NUM_TILES = M_NUM_TILES,
        BLOCK_K = BLOCK_K, K_NUM_TILES = K_NUM_TILES,
        NODE_CBS = node_cbs, GATE_CBS = gate_cbs, GATE_BS = gate_bs,
        N_NODE_GATES = block_size // gate_bs, gate_stride = gate.size(1),
        CONDITIONAL = (1 if conditional else 0),
    )

    return None
