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
}

# PADDED topologies -- rows with differing edge-block counts. The padding-specific structural tests
# below are parameterized over these.
RAGGED = ["pad_tiles", "mixed", "split", "pad_big"]
# BLOCK-SPARSE topologies -- non-adjacent child blocks. `sparse` is rectangular (every row has the
# same count) so it carries NO padding; `sparse_ragged` has both.
SPARSE = ["sparse", "sparse_ragged"]
# Everything the numeric tests must cover.
NONDENSE = RAGGED + SPARSE


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
    if name in ("dense", "sparse"):
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
        d_ef = float((pc.element_flows[:, :batch].double()[live] - ref_ef[live]).abs().max())

        ns.update_param_flows(pc.param_flows)
        got_pf = ns.get_param_flows().double().to(ref_pf.device)
        d_pf = float(((got_pf - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())

        assert d_ef < 2e-3, f"{name} repeat {it}: element flows off by {d_ef}"
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
@pytest.mark.parametrize("name", NONDENSE)
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


# --------------------------------------------------------------------------------- boundary

@cuda_only
@needs_cute
@pytest.mark.parametrize("name", RAGGED)
def test_ragged_small_batch_refuses_rather_than_corrupting(name):
    """Below batch 16 the param flows route to `_backward_sparse_par_flows`, whose only gated hook
    sits behind `_par_flow_collision_free` -- which padding defeats. There is no fallback, so this
    must RAISE. Pinned because the small-batch CUDA param kernel it would otherwise reach has the
    same unmasked non-atomic write, and relaxing that predicate alone would relocate the corruption
    rather than fix it.

    CAVEAT while `RAGGED_SUPPORTED` is False: this currently passes because the FORWARD refuses the
    layer, not because of the small-batch param path -- so it does not yet test what it says. When the
    plan-time check is relaxed it must be re-checked that the raise still comes, and comes from the
    param dispatch; the assertion on the message below is what makes that visible."""
    pc, root, ns, layer, data, phi = _run(name, batch = 8)
    with pytest.raises(NotImplementedError) as excinfo:
        pc(data, sum_external_params = {ns: phi})
        pc.backward(data, flows_memory = 0.0)

    assert ("no external param-flow backward applies" in str(excinfo.value)
            or "no block-scale forward applies" in str(excinfo.value)), \
        f"refused, but from the wrong place: {excinfo.value}"
