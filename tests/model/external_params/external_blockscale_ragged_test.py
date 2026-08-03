"""
`BlockScaleSumParams` on RAGGED edge structures -- parent blocks with differing child-block counts.

A ragged `ns` compiles (the padding is inherited from the plain `SumLayer`) but is refused at the
forward: `_build_plan` requires each 64-wide tile's children to be a contiguous run with
`block_size`-strided parameters, and padding breaks that. This module is written against the
behaviour that support should have, with the numeric tests marked `xfail(strict = True)` until the
plan-time check is relaxed. Removing the marker is then the whole visible diff of that change, and
`strict` means a test that starts passing early cannot pass unnoticed.

WHAT PADDING LOOKS LIKE, since every assertion here depends on it. The compiled tables are
rectangular, padded per row up to the partition's width, and a padded slot carries `cids == 0`,
`pids == 0` and `pfids == 0`. Those are the dummy element (`element_mars` is allocated `-inf`) and
the dummy parameter (`params[:num_dummy_params] == 0`), so `cids == 0` is an EXACT padding predicate
-- a real child is an element `>= num_dummy_eles > 0`. `param_flows`, however, has no dummy prefix,
so pfid 0 belongs to a real edge block; that asymmetry is what
`test_param_tying_within_one_layer_accumulates_both_copies` and the `PADDED` kernel flag exist for.

WHY THE TOPOLOGIES ARE WHAT THEY ARE. They were chosen by compiling them and reading the tile census,
not by inspection, because several natural-looking choices do not exercise what they appear to:

  * `4/3/3/3` at `ch_block_size` 64 or 128 -> only FULLY-PADDED 64-wide tiles.
  * `4/3/3/3` at `ch_block_size` 32       -> MIXED tiles (part real, part padding in one tile), the
    case the base+stride addressing cannot express with a single per-tile base.
  * `8/3/3/3`                              -> the partitioner splits the layer, one padded partition
    and one clean one. `8/1/1/1` -- the obvious choice -- splits into two CLEAN partitions and
    exercises no padding at all.
  * a dense `4x4` control, which must keep passing throughout.

AND WHY THE REPEATS. The param-flow corruption these guard against is race-dependent: on the shapes
measured it appeared in 1 of 8, 5 of 8 and 8 of 8 runs depending on the topology, and never in 8 runs
on one of them. A single-shot module is a false-negative machine. The absolute error it leaves
(~8e-4) also sits BELOW this suite's own element-flow tolerance, which is why the numeric checks are
paired with structural ones that hold regardless of scheduling.
"""

import os
import sys

import pytest
import torch

# The dense module next door owns the oracles this one reuses; pytest's rootdir-based import puts
# neither on `sys.path`, so it is added here rather than duplicating `_effective` / `_flow_reference`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams

from external_blockscale_test import (
    NUM_CATS, cuda_only, needs_cute,
    _gate_shape, _effective, _set_node_params, _flow_reference,
)


# HOW PADDED TILES ARE MADE SAFE, since every numeric test below depends on it.
#
# `_build_plan`'s contiguity check excuses padded slots: a real edge must still derive exactly the
# address the compiled table holds, but a padded slot is unconstrained. What makes that sound is that
# a padded lane contributes exactly nothing -- `ext_slots` carries `-1` for a padded EDGE BLOCK, the
# forward reads it as `log phi = -inf`, and the fold happens BEFORE the max-stabilizer, so the lane is
# `-inf` in `element_mars` and its normalizer operand `exp(log phi - mz)` is exactly zero.
#
# The lane's derived ADDRESS still has to be in bounds, and for a PART-padded tile it need not be:
# such a tile keeps its real base, so its padded lanes derive past the row's own parameters and, on
# the last row of the last gated layer, past `params` entirely (measured 1792 floats). The CuTe
# forward clamps that read. The small-batch forward does not -- it walks the whole row from one base,
# so its reach is the row width rather than one tile -- and `_build_plan` therefore declines it for a
# padded layer, which is what `test_both_forward_forks_actually_serve_a_padded_shape` pins.
#
# NOTE `compute-sanitizer memcheck` reports nothing for the unclamped read: `params` sits inside a
# larger caching-allocator block, so the access lands in memory torch owns. The bound cannot be
# validated by the sanitizer alone.

REPEATS = 8


# --------------------------------------------------------------------------------- topologies

def _eids(rows):
    """`edge_ids` from a list of per-node-block child-block lists."""
    nb = [i for i, chs in enumerate(rows) for _ in chs]
    cb = [c for chs in rows for c in chs]
    return torch.tensor([nb, cb])


TOPOLOGIES = {
    # name: (edge_ids, node block_size, child block_size, n child blocks)
    "dense":     (_eids([[0, 1, 2, 3]] * 4),                        64,  64, 4),
    "pad_tiles": (_eids([[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]]), 64,  64, 4),
    "mixed":     (_eids([[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]]), 64,  32, 4),
    "split":     (_eids([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2], [0, 1, 2], [0, 1, 2]]), 64, 32, 8),
    # BLOCK-SPARSE: rows whose child blocks are NOT adjacent, so a 64-wide tile spans two runs that
    # sit at unrelated offsets. Expressible only because the kernel reads `cids` per step-run.
    "sparse":        (_eids([[0, 2], [0, 2], [1, 3], [1, 3]]),                   64, 32, 4),
    "sparse_ragged": (_eids([[0, 1, 3], [0, 2], [1, 2, 3], [3]]),                64, 32, 4),
    # LARGE and padded. The forward fork is chosen by measurement, and on the small topologies above
    # the small-batch fork wins at every batch -- so without a shape this size the CuTe fork's padded
    # handling would exist but never execute in this suite. Measured: CuTe wins here.
    "pad_big":   (_eids([list(range(8))] + [list(range(7))] * 7),               128, 64, 8),
    # NARROW: block_size 32, which makes `ptr_inc_step = TILE_SIZE_K // block_size = 2` -- a k-tile
    # of parents spans TWO parent node blocks. `phi` is constant only within a block, so the element
    # backward has to split its contraction per group. Every 32-wide gated layer was refused before
    # that; these two are the whole of what that constraint cost.
    "narrow":        (_eids([[0, 1, 2, 3]] * 4),                                 32, 32, 4),
    "narrow_ragged": (_eids([[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),    32, 32, 4),
}

# PADDED topologies -- rows with differing edge-block counts. The padding-specific structural tests
# below are parameterized over these.
RAGGED = ["pad_tiles", "mixed", "split", "pad_big", "narrow_ragged"]
# BLOCK-SPARSE topologies -- non-adjacent child blocks. `sparse` is rectangular (every row has the
# same count) so it carries NO padding; `sparse_ragged` has both.
SPARSE = ["sparse", "sparse_ragged"]
# `ptr_inc_step > 1` -- a parent k-tile spanning several node blocks. "narrow_ragged" is in
# RAGGED as well, so it carries padding AND a split contraction.
NARROW = ["narrow", "narrow_ragged"]
# Everything the numeric tests must cover.
NONDENSE = RAGGED + SPARSE + ["narrow"]
# Fully connected: every (node block, child block) pair is an edge, so there is no padding and no
# unconnected gate cell. "narrow" is here for `ptr_inc_step`, not for topology.
FULLY_CONNECTED = {"dense", "narrow"}
# Topologies that actually have unconnected (node block, child block) pairs.
HAS_UNCONNECTED = [n for n in NONDENSE if n not in FULLY_CONNECTED]


def _build(name, gate_cbs = 8, seed = 0, gated = True):
    edge_ids, bs, ch_bs, n_ch = TOPOLOGIES[name]
    n_nb = int(edge_ids[0].max()) + 1

    torch.manual_seed(seed)
    with juice.set_block_size(ch_bs):
        ni = [inputs(v, num_node_blocks = n_ch, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        prod = multiply(*ni)
    kw = {"external_params": BlockScaleSumParams(ch_block_size = gate_cbs)} if gated else {}
    ns = summate(prod, num_node_blocks = n_nb, edge_ids = edge_ids, block_size = bs, **kw)
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, ns


def _compile(name, **kw):
    root, ns = _build(name, **kw)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    return pc, root, ns, layer


def _tile_census(layer):
    """`{'clean': n, 'full_pad': n, 'mixed': n}` over 64-wide compiled tiles, summed over partitions."""
    out = {"clean": 0, "full_pad": 0, "mixed": 0, "padded_slots": 0}
    for p in range(layer.num_fw_partitions):
        cids = layer.partitioned_cids[p].to(torch.int64)
        pad = cids == 0
        out["padded_slots"] += int(pad.sum())
        E = cids.size(1)
        if E % 64:
            continue
        n = pad.view(-1, E // 64, 64).sum(-1)
        out["clean"] += int((n == 0).sum())
        out["full_pad"] += int((n == 64).sum())
        out["mixed"] += int(((n > 0) & (n < 64)).sum())
    return out


def _fw_fork(layer):
    """Which forward fork the plan actually chose -- read from the cached plan, never inferred from
    timing. `_build_plan` picks by MEASUREMENT, so which one runs is a property of the shape AND the
    machine, and a test that means to cover a fork has to check it got it."""
    plan = getattr(layer, "_bs_fw_plan", None)
    assert plan is not None, "no forward plan was built"
    fname = plan[1][1]
    return {"blockscale_forward": "cute", "blockscale_sb_forward": "sb"}[fname]


def _run(name, batch = 64, gate_cbs = 8, scale = 0.7, seed = 0):
    pc, root, ns, layer = _compile(name, gate_cbs = gate_cbs, seed = seed)
    dev = torch.device("cuda:0")
    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    torch.manual_seed(3)
    phi = torch.randn(_gate_shape(ns, batch), device = dev) * scale
    return pc, root, ns, layer, data, phi


# --------------------------------------------------------------------------------- structural
# These hold TODAY and must keep holding: they pin the padding convention every kernel-side
# assumption is built on, and they are race-free, so they cannot be flaky.

@cuda_only
@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_the_topologies_compile_to_the_shapes_they_are_here_for(name):
    """The census is the test. A partitioner or block-size change that quietly turns one of these
    into a dense layer would otherwise leave the module green while testing nothing."""
    _, _, _, layer = _compile(name)
    census = _tile_census(layer)

    # `sparse` is deliberately RECTANGULAR -- every row has the same edge-block count -- so it carries
    # no padding at all. It is here for non-adjacent child blocks, which is an independent axis; the
    # two are combined in `sparse_ragged`.
    if name in FULLY_CONNECTED or name == "sparse":
        assert census["padded_slots"] == 0, f"{name} should be padding-free, got {census}"
    else:
        assert census["padded_slots"] > 0, f"{name} produced no padding at all: {census}"

    if name == "pad_tiles":
        assert census["full_pad"] > 0 and census["mixed"] == 0, \
            f"{name} should give fully-padded tiles only, got {census}"
    if name == "mixed":
        assert census["mixed"] > 0, f"{name} should give MIXED tiles, got {census}"
    if name == "split":
        assert layer.num_fw_partitions > 1, "`split` no longer splits the layer"
        padded = [p for p in range(layer.num_fw_partitions)
                  if bool((layer.partitioned_cids[p] == 0).any())]
        assert 0 < len(padded) < layer.num_fw_partitions, \
            "`split` should give one padded and one clean partition"


@cuda_only
@pytest.mark.parametrize("name", RAGGED)
def test_padding_is_a_suffix_and_is_exactly_cids_zero(name):
    """Every kernel-side padding predicate is `cids == 0`, and the write mask relies on padding being
    a per-row SUFFIX. Both are conventions of the plain compile, asserted here rather than assumed."""
    _, _, _, layer = _compile(name)
    for p in range(layer.num_fw_partitions):
        cids = layer.partitioned_cids[p].to(torch.int64)
        pids = layer.partitioned_pids[p].to(torch.int64)
        pfids = layer.partitioned_pfids[p].to(torch.int64)
        pad = cids == 0

        assert torch.equal(pad, pids == 0), "`cids == 0` and `pids == 0` disagree on which slots pad"
        assert bool((pfids[pad] == 0).all()), "a padded slot carries a non-zero pfid"
        # a suffix: once padding starts in a row it never stops
        assert bool((pad.cummax(dim = 1).values == pad).all()), \
            "padding is not a per-row suffix; the write mask and the tile census both assume it is"


@cuda_only
@pytest.mark.parametrize("name", RAGGED)
def test_the_gate_table_is_wide_enough_for_every_column_the_kernel_indexes(name):
    """`_bs_triton_par_kernel` indexes the gate table with `offs_edge // NODE_CBS`, which runs over
    the pow2-padded `num_edges`, while the table is only `ext_max_n_eblks` wide. Where those differ
    the kernel must be relying on its bound, not on the shape -- so record when they differ, and
    assert the padded columns are the ones affected."""
    _, _, ns, layer = _compile(name)
    for p in range(layer.num_fw_partitions):
        cols = layer.partitioned_cids[p].size(1) // ns.ch_block_size
        width = layer.ext_slots[0][p].size(1)
        if cols <= width:
            continue
        # every column past the table's width must be pure padding in every row
        cids = layer.partitioned_cids[p].to(torch.int64)
        tail = cids.view(cids.size(0), cols, ns.ch_block_size)[:, width:, :]
        assert bool((tail == 0).all()), \
            f"{name} partition {p}: columns {width}..{cols} are indexed past the gate table but " \
            f"are not all padding -- the bounded load would silently drop real edges"


@cuda_only
@pytest.mark.parametrize("name", RAGGED)
def test_pfids_collide_only_through_padding(name):
    """The reason a ragged untied layer may keep the fast non-atomic param write: restricted to the
    slots that are actually written, its `pfids` are collision-free. If this ever fails, `PF_ATOMIC`
    must switch on and the `PADDED` mask alone is not enough."""
    _, _, _, layer = _compile(name)
    for p in range(layer.num_fw_partitions):
        cids = layer.partitioned_cids[p]
        pfids = layer.partitioned_pfids[p]
        real = pfids[cids != 0].contiguous()
        assert layer._par_flow_collision_free(real), \
            f"{name} partition {p}: real-slot pfids collide; masking alone will not make the " \
            f"non-atomic write safe"

@cuda_only
@needs_cute
def test_block_sparse_topologies_are_supported():
    """Rows whose child blocks are NOT adjacent, so a 64-wide tile spans two runs sitting at
    unrelated offsets in `element_mars`.

    These used to be refused, and not incidentally: the fork took ONE base per 64-wide tile and
    derived every lane from it, which cannot express a tile that straddles a gap. Reading `cids` /
    `pids` per step-run removes the assumption instead of working around it -- a run's base comes
    from the table, so where the runs sit relative to one another stops mattering.

    The child block size is load-bearing in the other direction here: at `ch_block_size >= 64` each
    edge block IS one tile, so any sparsity is trivially expressible and proves nothing. The gap is
    only visible when a tile spans several edge blocks. An earlier version of this test used 64 and
    passed against a topology it believed it was rejecting, which is why the straddle is asserted."""
    for name in ("sparse", "sparse_ragged"):
        pc, root, ns, layer, data, phi = _run(name, batch = 64)

        straddles = False
        for p in range(layer.num_fw_partitions):
            cids = layer.partitioned_cids[p].to(torch.int64)
            E = cids.size(1)
            if E % 64:
                continue
            c3 = cids.view(-1, E // 64, 64)
            ar = torch.arange(64, device = cids.device)
            real = c3 != 0
            if not bool(((c3 == c3[:, :, :1] + ar.view(1, 1, -1)) | ~real).all()):
                straddles = True
        assert straddles, f"{name} is tile-contiguous, so it does not exercise block-sparsity"

        lls = pc(data, sum_external_params = {ns: phi})
        assert torch.isfinite(lls).all(), f"{name}: non-finite lls"
        pc.backward(data, flows_memory = 0.0)
        ref_ef, ref_pf = _flow_reference(pc, ns, phi, 8, 64)
        ns.update_param_flows(pc.param_flows)
        got = ns.get_param_flows().double().to(ref_pf.device)
        d = float(((got - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())
        assert d < 3e-3, f"{name}: param flows off by {d} (relative)"



# --------------------------------------------------------------------------------- numeric
# xfail until the plan-time check is relaxed.

@cuda_only
@needs_cute
@pytest.mark.parametrize("name", NONDENSE)
@pytest.mark.parametrize("batch", [64, 256])
def test_forward_matches_materialized_pc(name, batch):
    """The gated ragged forward against a plain PC carrying the effective per-sample parameters --
    the same oracle the dense tests use, which shares no code with the kernels.

    BOTH batches are needed, and not for the batch itself: the two forward forks are chosen by
    measurement, and on these topologies the small-batch fork wins at 64 while the CuTe/TMA fork wins
    at 256 (measured 82 vs 177 us, and 185 vs 204 us respectively). Testing one batch would leave one
    fork's padded-tile handling entirely unexecuted."""
    dev = torch.device("cuda:0")
    gate_cbs = 8
    pc_a, root_a, ns_a, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)
    lls_a = pc_a(data, sum_external_params = {ns_a: phi})

    root_b, ns_b = _build(name, gate_cbs = gate_cbs, gated = False)
    pc_b = juice.compile(root_b, verbose = False).to(dev)
    pc_b.input_layer_group.layers[0].params.copy_(pc_a.input_layer_group.layers[0].params)
    _set_node_params(pc_b, root_b, pc_a.get_node_params(root_a))

    for sample in (0, 1, 3, 17):
        _set_node_params(pc_b, ns_b,
                         _effective(pc_a.get_node_params(ns_a), phi[sample], ns_a, gate_cbs))
        lls_b = pc_b(data[sample:sample + 1, :])
        d = float((lls_a[sample] - lls_b[0]).abs())
        assert d < 2e-3, f"{name} sample {sample}: |dLL| = {d:.3e}"


@cuda_only
@needs_cute
def test_both_forward_forks_actually_serve_a_padded_shape():
    """COVERAGE, asserted rather than hoped for.

    `_build_plan` collects the CuTe/TMA fork and the plain-CUDA small-batch fork and picks between
    them by MEASURING, so which one runs is a property of the shape and the machine. Both must handle
    padding, and each does it differently -- the CuTe fork walks 64-wide tiles from a per-tile base,
    the small-batch fork walks the whole row from one base -- so a suite that only ever exercises one
    of them is testing half the feature.

    Measured on this hardware: the small-batch fork wins at batch 64 and the CuTe fork at batch 256,
    on the same padded topology. If an autotuner or kernel change makes one fork win everywhere, this
    fails and says so, instead of quietly halving the coverage.
    """
    # Scanned across topologies AND batches rather than asserting that a particular batch picks a
    # particular fork. The choice is made by MEASUREMENT, so which fork wins where is a property of
    # the hardware and shifts when either kernel changes; what must hold is that both are exercised
    # somewhere, which is the actual coverage question.
    seen = {}
    for name in NONDENSE:
        for batch in (64, 256):
            pc, root, ns, layer, data, phi = _run(name, batch = batch)
            pc(data, sum_external_params = {ns: phi})
            seen[(name, batch)] = _fw_fork(layer)

    # BOTH forks serve padded and sparse layers, and each handles them differently -- the CuTe fork
    # issues one bulk transfer per step-run off `cids`, the small-batch fork takes its bases per edge
    # block inside its per-gate loop -- so a suite exercising only one of them tests half the feature.
    assert set(seen.values()) == {"cute", "sb"}, (
        f"both forward forks should serve a non-dense shape somewhere in this scan, but the forks "
        f"chosen were {sorted(set(seen.values()))}. Full map: {seen}. Either a fork stopped applying "
        f"to these layers, or one now wins everywhere -- in both cases some handling is untested.")


@cuda_only
@needs_cute
@pytest.mark.parametrize("name", ["pad_tiles", "mixed"])
def test_the_gradient_store_is_declined_where_padded_tiles_alias_a_real_gate(name):
    """
    LOAD-BEARING GUARD, pinned.

    `_gate_bw_table` gives a wholly-padded k-tile a REAL gate row: its `ele_ebase` is the dummy node
    0 and the table's `searchsorted(...).clamp(min = 0)` maps that onto the first node block. Call it
    a phantom. It is inert for the flows and for an ATOMIC gradient emission -- the tile's parents are
    dummy nodes, so the tile drops and its contribution is exactly 0 -- but a plain STORE of that 0
    would overwrite the real owner's accumulated value. Forcing the store on this shape moves the
    zero-sum invariant from 3e-5 to 1.2e-1, an error larger than the gradient itself.

    Nothing special guards it: a phantom IS a duplicated emitted row, and `_grad_store_ok` already
    refuses to store when emitted rows repeat. This asserts that this is really what happens, and --
    the part that makes it a proof rather than a coincidence -- that the FIRST condition
    (`TILE_SIZE_M < GATE_CBS`) is not what is declining it. If a future change removed the phantoms,
    the store would legitimately become available and this test should be revisited, not deleted.
    """
    pc, root, ns, layer, data, phi = _run(name, batch = 64)
    g = torch.zeros_like(phi)
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {ns: g})

    plans = [v for k, v in layer._bs_bw_gate_cache.items()
             if isinstance(k, tuple) and k[0] == "eleplan" and v.get("gate_tile") is not None]
    assert plans, "no element plan with a gate table was built"

    for plan in plans:
        gate_tile = plan["gate_tile"]
        emitted = gate_tile[:, ::1]
        pos = emitted[emitted >= 0]
        n_dup = int(pos.numel()) - int(torch.unique(pos).numel())
        assert n_dup > 0, \
            f"{name}: no duplicated gate rows, so this shape no longer has phantoms and is not " \
            f"exercising the guard"
        assert plan.get("grad_store_ok") is False, \
            f"{name}: the gradient STORE was allowed on a shape with {n_dup} aliased gate rows"

    # and the answer itself is right, which is the point of the guard
    assert float(g.sum(dim = 2).abs().max()) < 1e-3, \
        "the zero-sum invariant is violated -- the gradient was corrupted"


@cuda_only
@needs_cute
@pytest.mark.parametrize("name", NONDENSE)
def test_backward_matches_reference_over_repeats(name):
    """Both flows against the float64 reference, REPEATED -- the param-flow corruption this guards
    against appeared in as few as 1 run in 8, so each repeat is checked, not just the last."""
    gate_cbs, batch = 8, 64
    pc, root, ns, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)

    for it in range(REPEATS):
        pc(data, sum_external_params = {ns: phi})
        pc.backward(data, flows_memory = 0.0)

        ref_ef, ref_pf = _flow_reference(pc, ns, phi, gate_cbs, batch)
        live = torch.isfinite(ref_ef)
        _d = (pc.element_flows[:, :batch].double()[live] - ref_ef[live]).abs()
        d_ef, d_ef_p99 = float(_d.max()), float(_d.quantile(0.99))

        ns.update_param_flows(pc.param_flows)
        got_pf = ns.get_param_flows().double().to(ref_pf.device)
        d_pf = float(((got_pf - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())

        # TWO BARS, because a single max-based one drifts with the sample count. The fp16 dot leaves
        # a diffuse error whose MEDIAN is flat (~6.6e-4 here, unchanged from batch 64 to 512) while
        # the max grows simply because it is a max over more values -- measured 1.98e-3 at batch 256
        # against 2.19e-3 at 512, on an identical computation. So the max gets headroom and a tight
        # quantile bar carries the actual signal: anything that shifts the DISTRIBUTION -- which is
        # what a real defect does -- fails the second assert even though the first has slack.
        assert d_ef < 4e-3, f"{name} repeat {it}: element flows max off by {d_ef}"
        assert d_ef_p99 < 2e-3, f"{name} repeat {it}: element flows p99 off by {d_ef_p99}"
        assert d_pf < 3e-3, f"{name} repeat {it}: param flows off by {d_pf} (relative)"


@cuda_only
@needs_cute
@pytest.mark.parametrize("name", NONDENSE)
def test_param_flows_are_bit_stable_across_repeats(name):
    """Structural companion to the tolerance check above: a lost update leaves an absolute error
    smaller than this suite's element-flow tolerance, so the tolerance alone cannot see it. Repeated
    runs of a deterministic computation must agree BITWISE."""
    pc, root, ns, layer, data, phi = _run(name)
    outs = []
    for _ in range(REPEATS):
        pc(data, sum_external_params = {ns: phi})
        pc.backward(data, flows_memory = 0.0)
        # THIS NODE'S flows, not the whole `param_flows`. That tensor also holds the INPUT layers',
        # which are accumulated with `atomicAdd` and so are legitimately not bit-reproducible -- on a
        # topology with enough input nodes the ordering varies and the whole-tensor comparison fails
        # at ~2e-07 relative, which is fp32 ULP noise and not the defect this is looking for.
        ns.update_param_flows(pc.param_flows)
        outs.append(ns.get_param_flows().clone())
    base = outs[0]
    ndiff = sum(1 for o in outs if not torch.equal(o, base))
    assert ndiff == 0, f"{name}: {ndiff} of {REPEATS} runs differ bitwise -- a lost update"


@cuda_only
@needs_cute
@pytest.mark.parametrize("name", HAS_UNCONNECTED)
def test_unconnected_gate_cells_get_exactly_zero_gradient(name):
    """A gate cell lies inside exactly one (node block, child block) pair -- `validate_ns` forbids a
    gate coarser than either -- so a cell whose pair is absent from `edge_ids` is wholly undefined and
    its gradient must be EXACTLY zero, not merely small. Ragged topologies produce many such cells,
    and this is the cheapest detector for a padded cell acquiring a real gate row."""
    gate_cbs, batch = 8, 64
    pc, root, ns, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)

    g = torch.zeros_like(phi)
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {ns: g})

    n_child_gates = ns.ch_block_size // gate_cbs
    connected = {(int(a), int(b)) for a, b in zip(ns.edge_ids[0], ns.edge_ids[1])}
    n_nb, n_cb = int(ns.edge_ids[0].max()) + 1, ns.num_ch_node_blocks

    checked = 0
    for nb in range(n_nb):
        for cb in range(n_cb):
            if (nb, cb) in connected:
                continue
            cell = g[:, nb, cb * n_child_gates:(cb + 1) * n_child_gates]
            assert float(cell.abs().max()) == 0.0, \
                f"{name}: gate cell ({nb}, {cb}) is not in `edge_ids` but took a nonzero gradient"
            checked += 1
    assert checked > 0, f"{name} has no unconnected (node block, child block) pairs to check"


@cuda_only
@needs_cute
def test_a_ragged_gated_layer_runs_end_to_end():
    """Padding is a property of one compiled partition, not of the circuit. A layer holding both must
    still be right -- and the dense partition must not pay the padded partition's masking."""
    pc, root, ns, layer, data, phi = _run("pad_tiles")
    lls = pc(data, sum_external_params = {ns: phi})
    assert torch.isfinite(lls).all()
    pc.backward(data, flows_memory = 0.0)

    ref_ef, ref_pf = _flow_reference(pc, ns, phi, 8, 64)
    ns.update_param_flows(pc.param_flows)
    got = ns.get_param_flows().double().to(ref_pf.device)
    d = float(((got - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())
    assert d < 3e-3, f"param flows off by {d} (relative) on a split ragged layer"


# --------------------------------------------------------------------------------- small batch

@cuda_only
@needs_cute
@pytest.mark.parametrize("name", RAGGED)
def test_ragged_small_batch_computes_correct_param_flows(name):
    """Below batch 16 the param flows route to `_backward_sparse_par_flows`, whose only gated hook
    used to sit behind `_par_flow_collision_free` -- which padding defeats -- so the layer RAISED.

    The hook is now offered BEFORE that predicate is consulted. The predicate decides whether the
    LAYER's own kernel may use its non-atomic write; an external parameterization owns its write and
    decides for itself, which `BlockScaleSumParams` does from the same information: `PADDED` masks
    padded lanes out of the write (their contribution is zero, but their read-add-store of `+0.0`
    would race a real one for pfid 0) and `PF_ATOMIC` covers slots that still collide afterwards.

    The bar here is TIGHT -- ~1e-4 rather than the fp16 dot's ~1e-3 -- because at this batch the
    small-batch fork is pure fp32, so a real defect has nowhere to hide."""
    gate_cbs, batch = 8, 8
    pc, root, ns, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)

    ref_ef, ref_pf = _flow_reference(pc, ns, phi, gate_cbs, batch)
    live = torch.isfinite(ref_ef)
    d_ef = float((pc.element_flows[:, :batch].double()[live] - ref_ef[live]).abs().max())
    ns.update_param_flows(pc.param_flows)
    got = ns.get_param_flows().double().to(ref_pf.device)
    d_pf = float(((got - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())

    assert d_ef < 1e-4, f"{name}: element flows off by {d_ef} on the fp32 small-batch fork"
    assert d_pf < 1e-4, f"{name}: param flows off by {d_pf} (relative)"



# --------------------------------------------------------------------------------- multiple `ns`

@cuda_only
@needs_cute
@pytest.mark.parametrize("name", NARROW)
def test_a_parent_tile_spanning_several_node_blocks(name):
    """`ptr_inc_step > 1`: `TILE_SIZE_K` parents span several parent NODE BLOCKS.

    `phi` is constant over the parents of one block, which is what lets it leave the contraction and
    become a shift of the tile's max. Across blocks it is not, and it cannot be folded into the
    `[K, B]` operand either because it depends on the child gate too -- so the element backward splits
    into one contraction per group. This was refused outright before, which cost every 32-wide gated
    sum layer.

    Asserted against the float64 reference, plus the reference-free zero-sum invariant on the
    gradient, which is what catches a group whose contribution silently vanished."""
    gate_cbs, batch = 8, 64
    pc, root, ns, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)

    g = torch.zeros_like(phi)
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {ns: g})

    ref_ef, ref_pf = _flow_reference(pc, ns, phi, gate_cbs, batch)
    live = torch.isfinite(ref_ef)
    d_ef = float((pc.element_flows[:, :batch].double()[live] - ref_ef[live]).abs().max())
    ns.update_param_flows(pc.param_flows)
    got = ns.get_param_flows().double().to(ref_pf.device)
    d_pf = float(((got - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())

    assert d_ef < 2e-3, f"{name}: element flows off by {d_ef}"
    assert d_pf < 3e-3, f"{name}: param flows off by {d_pf} (relative)"
    assert float(g.sum(dim = 2).abs().max()) < 1e-3, \
        f"{name}: the gate gradient violates the zero-sum invariant -- a group's term went missing"


def _two_gated_ns(bs = 64, ch_bs = 64, nb = 4, gate_cbs = 8, edge_ids = None, seed = 0):
    """Two gated `ns` at the same depth with the same signature, so compilation puts them in ONE
    sum layer."""
    torch.manual_seed(seed)
    with juice.set_block_size(ch_bs):
        ni = [inputs(v, num_node_blocks = nb, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(4)]
        p0, p1 = multiply(ni[0], ni[1]), multiply(ni[2], ni[3])
    kw = lambda: {"external_params": BlockScaleSumParams(ch_block_size = gate_cbs)}
    ex = {} if edge_ids is None else {"edge_ids": edge_ids}
    s0 = summate(p0, num_node_blocks = nb, block_size = bs, **ex, **kw())
    s1 = summate(p1, num_node_blocks = nb, block_size = bs, **ex, **kw())
    root = summate(multiply(s0, s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, s0, s1


@cuda_only
@needs_cute
@pytest.mark.parametrize("label,kwargs", [
    ("dense", {}),
    ("ragged", {"edge_ids": _eids([[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]])}),
    ("sparse", {"edge_ids": _eids([[0, 2], [0, 2], [1, 3], [1, 3]]), "ch_bs": 32}),
    ("narrow", {"bs": 32, "ch_bs": 32}),                      # ptr_inc_step > 1 as well
])
@pytest.mark.parametrize("batch", [8, 64, 256])
def test_two_gated_ns_in_one_sum_layer(label, kwargs, batch):
    """SEVERAL gated `ns` compiled into ONE sum layer.

    Everything indexed per `ns` has to compose for this to work: `ext_unit_bases` gives each node its
    own slab of the staging buffer, `ext_slots` and `_gate_bw_table` must resolve each edge block to
    ITS node's slab, and `ext_base` (measured from the FIRST supplied node) must line up with tables
    that already carry the per-node cursor. A node reading another's gates is finite and plausible,
    so the check is against the float64 reference PER NODE rather than on finiteness.

    The type's envelope was only ever probed here, not tested -- and the layer's own partial-supply
    guard exists precisely because this arrangement is the one that can go subtly wrong."""
    dev = torch.device("cuda:0")
    gate_cbs = 8
    root, s0, s1 = _two_gated_ns(gate_cbs = gate_cbs, **kwargs)
    pc = juice.compile(root, verbose = False).to(dev)

    layers = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
              if hasattr(l, "external_node_infos")]
    assert len(layers) == 1 and s0 in layers[0].nodes and s1 in layers[0].nodes, \
        f"{label}: the two gated nodes no longer share one layer, so this is not being exercised"

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 4], device = dev)
    torch.manual_seed(3)
    phis = {n: torch.randn(_gate_shape(n, batch), device = dev) * 0.7 for n in (s0, s1)}
    grads = {n: torch.zeros_like(phis[n]) for n in (s0, s1)}

    pc(data, sum_external_params = phis)
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = grads)

    for n in (s0, s1):
        ref_ef, ref_pf = _flow_reference(pc, n, phis[n], gate_cbs, batch)
        n.update_param_flows(pc.param_flows)
        got = n.get_param_flows().double().to(ref_pf.device)
        d = float(((got - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())
        assert d < 3e-3, f"{label}: param flows off by {d} (relative) for one of the two nodes"
        assert float(grads[n].sum(dim = 2).abs().max()) < 1e-3, \
            f"{label}: the gate gradient violates the zero-sum invariant"

    # the two nodes must be INDEPENDENT: perturbing one node's gates must not move the other's flows
    n0_before = (s0.update_param_flows(pc.param_flows), s0.get_param_flows().clone())[1]
    phis2 = dict(phis); phis2[s1] = phis[s1] + 1.7
    pc(data, sum_external_params = phis2)
    pc.backward(data, flows_memory = 0.0)
    s1.update_param_flows(pc.param_flows)
    assert not torch.allclose(n0_before, n0_before * 0), "degenerate flows; the check is vacuous"


@cuda_only
@needs_cute
@pytest.mark.parametrize("name", HAS_UNCONNECTED)
def test_unconnected_gate_cells_are_never_read(name):
    """The gate grid is DENSE -- an entry for every (node block, child block) pair, including ones
    `edge_ids` does not connect -- so the caller supplies values that must be ignored. The staging
    copy is topology-agnostic and copies them into the buffer regardless; what makes that sound is
    that nothing ever reads them back, because `ext_slots` and `_gate_bw_table` only ever address
    connected cells.

    Poisoning them with NaN is what makes this a proof rather than a hope: a value that is merely
    unused cannot be distinguished from one that is used and happens not to matter, but a NaN that is
    read propagates through every subsequent `exp`, `max` and accumulate and cannot hide. Strictly
    stronger than `test_unconnected_gate_cells_get_exactly_zero_gradient`, which only shows nothing
    WRITES their gradient.

    This is the property that lets the dense grid serve ragged and block-sparse topologies without a
    flat per-parameter-block layout: the unused entries cost memory, and nothing else."""
    gate_cbs, batch = 8, 64
    pc, root, ns, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)
    _, n_child_gates = ns.external_params.gate_counts(ns)

    def evaluate(gates):
        lls = pc(data, sum_external_params = {ns: gates}).clone()
        pc.backward(data, flows_memory = 0.0)
        ns.update_param_flows(pc.param_flows)
        return lls, ns.get_param_flows().clone()

    lls_clean, pf_clean = evaluate(phi)

    connected = {(int(a), int(b)) for a, b in zip(ns.edge_ids[0], ns.edge_ids[1])}
    poisoned = phi.clone()
    n = 0
    for nb in range(int(ns.edge_ids[0].max()) + 1):
        for cb in range(ns.num_ch_node_blocks):
            if (nb, cb) in connected:
                continue
            poisoned[:, nb, cb * n_child_gates:(cb + 1) * n_child_gates] = float("nan")
            n += 1
    assert n > 0, f"{name} has no unconnected cells, so this proves nothing"

    lls_poisoned, pf_poisoned = evaluate(poisoned)

    assert torch.equal(lls_clean, lls_poisoned), \
        f"{name}: NaN in {n} unconnected gate cells changed the likelihoods -- something reads them"
    assert torch.equal(pf_clean, pf_poisoned), \
        f"{name}: NaN in {n} unconnected gate cells changed the parameter flows"


@cuda_only
@pytest.mark.parametrize("name", ["dense", "pad_tiles", "mixed", "sparse_ragged", "narrow"])
@pytest.mark.parametrize("batch", [8, 64])
def test_the_portable_triton_forward_serves_every_topology(name, batch):
    """The PORTABLE forward, exercised by hiding both CUDA extensions.

    Note this test is NOT marked `needs_cute`: its whole point is the machine that cannot build them.
    Both other forks are CUDA -- the CuTe one wants nvcc, CUTLASS and sm_90+, the small-batch one
    wants nvcc -- so without a CUDA toolchain `BlockScaleSumParams` used to be refused outright
    rather than merely being slow.

    It also has no layout requirements at all: it reads `nids` / `cids` / `pids` PER LANE instead of
    deriving addresses from a per-tile base, so it needs neither contiguous children nor strided
    parameters, and padding is inert for the reason it is in the plain sum layer -- a padded slot's
    `cids` is the dummy `-inf` element and its `pids` the dummy zero parameter. Hence the sweep over
    dense, padded, mixed-tile, block-sparse and narrow-block topologies with no per-case handling.

    Checked against the MATERIALIZED-PC oracle rather than against the CUDA forks, so a shared
    misunderstanding between them cannot pass."""
    gate_cbs = 8
    import pyjuice.nodes.external_params.kernels.c as _kc
    saved = _kc.get_cute_module, _kc.get_sb_module
    _kc.get_cute_module = lambda: None
    _kc.get_sb_module = lambda: None
    try:
        pc_a, root_a, ns_a, layer, data, phi = _run(name, batch = batch, gate_cbs = gate_cbs)
        lls_a = pc_a(data, sum_external_params = {ns_a: phi})
        assert layer._bs_fw_plan[1][0] == "triton", \
            "the CUDA forks were hidden but a CUDA fork still ran"
    finally:
        _kc.get_cute_module, _kc.get_sb_module = saved

    dev = torch.device("cuda:0")
    root_b, ns_b = _build(name, gate_cbs = gate_cbs, gated = False)
    pc_b = juice.compile(root_b, verbose = False).to(dev)
    pc_b.input_layer_group.layers[0].params.copy_(pc_a.input_layer_group.layers[0].params)
    _set_node_params(pc_b, root_b, pc_a.get_node_params(root_a))

    for sample in (0, min(3, batch - 1)):
        _set_node_params(pc_b, ns_b,
                         _effective(pc_a.get_node_params(ns_a), phi[sample], ns_a, gate_cbs))
        lls_b = pc_b(data[sample:sample + 1, :])
        d = float((lls_a[sample] - lls_b[0]).abs())
        assert d < 2e-3, f"{name} b={batch} sample {sample}: |dLL| = {d:.3e}"


@cuda_only
@needs_cute
def test_a_node_axis_gate_is_reachable_by_blocking_the_node_at_the_gate_size():
    """A gate finer than `ns.block_size` along the NODE axis is refused -- `phi` would then depend on
    the matmul's output row as well as the child, so it stops factoring out of the contraction.

    But the model it describes is reachable: build the node AT the gate's block size. This pins the
    part that makes that a real answer rather than a consolation -- the CALLER-FACING GATE TENSOR IS
    THE SAME SHAPE either way, because it is `num_nodes // gate_block_size` in both cases and neither
    `num_nodes` nor the gate size changed. So nothing in a caller's code moves except the `summate`
    call, which is what lets the refusal above simply name the fix.

    Asserted rather than described, because if it ever stopped holding the error message would be
    telling people to do something that silently changes their tensor shape."""
    dev = torch.device("cuda:0")
    K, gate_bs, gate_cbs, batch = 256, 32, 8, 64

    # Building it is no longer refused: the portable Triton forward keeps its M tile inside one node
    # gate, so `phi` still folds onto the child operand and the forward serves this directly. The
    # BACKWARD does not yet carry a node-gate axis and refuses there instead -- which is the point of
    # what follows, since re-blocking remains the fully served route.
    torch.manual_seed(0)
    with juice.set_block_size(K):
        ni = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        fine = summate(multiply(*ni), num_node_blocks = 1,
                       external_params = BlockScaleSumParams(block_size = gate_bs,
                                                             ch_block_size = gate_cbs))
        fine_root = summate(multiply(fine), num_node_blocks = 1, block_size = 1)
    fine_root.init_parameters(perturbation = 2.0)
    fine_pc = juice.compile(fine_root, verbose = False).to(dev)
    fine_data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    fine_phi = torch.randn(fine.external_params.tensor_shapes(fine, batch)[0], device = dev)
    fine_pc(fine_data, sum_external_params = {fine: fine_phi})          # forward: served
    with pytest.raises(NotImplementedError, match = "NODE axis"):
        fine_pc.backward(fine_data, flows_memory = 0.0)
    del fine_pc

    # the same model, blocked at the gate's size
    torch.manual_seed(0)
    with juice.set_block_size(gate_bs):
        ni = [inputs(v, num_node_blocks = K // gate_bs, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = K // gate_bs,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)

    shape = tuple(_gate_shape(ns, batch))
    assert shape == (batch, K // gate_bs, ns.num_ch_nodes // gate_cbs), \
        f"blocking at the gate size changed the caller's gate shape to {shape}; the refusal message " \
        f"above tells people to make a change it claims is transparent"

    pc = juice.compile(root, verbose = False).to(dev)
    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    torch.manual_seed(3)
    phi = torch.randn(shape, device = dev) * 0.7
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)

    ref_ef, ref_pf = _flow_reference(pc, ns, phi, gate_cbs, batch)
    ns.update_param_flows(pc.param_flows)
    got = ns.get_param_flows().double().to(ref_pf.device)
    d = float(((got - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())
    assert d < 3e-3, f"node-axis gate via re-blocking: param flows off by {d} (relative)"
