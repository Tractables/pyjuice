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


# Flip to True when `_build_plan`'s contiguity check is relaxed to ignore padded slots; the numeric
# tests below then assert correctness instead of the refusal.
RAGGED_SUPPORTED = False

pending = pytest.mark.xfail(
    not RAGGED_SUPPORTED, strict = True,
    reason = "ragged edge structures are still refused at the forward (`_build_plan`)")

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
}

RAGGED = ["pad_tiles", "mixed", "split"]


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

    if name == "dense":
        assert census["padded_slots"] == 0, f"the control is not padding-free: {census}"
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
def test_block_sparse_topologies_are_still_refused():
    """Relaxing the check for PADDING must not accidentally admit genuine block-sparsity: a row whose
    child blocks are not adjacent has non-contiguous `cids` across REAL edges, which the base+stride
    addressing cannot express at all.

    The child block size matters here and is not incidental. Contiguity is checked WITHIN each 64-wide
    tile, so at `ch_block_size == 64` every edge block is exactly one tile and is trivially contiguous
    however sparse the topology is -- a first version of this test used 64 and passed against a
    topology it believed it was rejecting. Non-adjacency is only expressible to the kernel when a tile
    SPANS several edge blocks, i.e. `ch_block_size < 64`."""
    dev = torch.device("cuda:0")
    edge_ids = _eids([[0, 2], [0, 2], [1, 3], [1, 3]])
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = 4, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        prod = multiply(*ni)
    ns = summate(prod, num_node_blocks = 4, edge_ids = edge_ids, block_size = 64,
                 external_params = BlockScaleSumParams(ch_block_size = 8))
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    # The topology must actually be non-contiguous for real edges, or this proves nothing.
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    noncontig = False
    for p in range(layer.num_fw_partitions):
        cids = layer.partitioned_cids[p].to(torch.int64)
        E = cids.size(1)
        if E % 64:
            continue
        c3 = cids.view(-1, E // 64, 64)
        ar = torch.arange(64, device = cids.device)
        real = c3 != 0
        if not bool((((c3 == c3[:, :, :1] + ar.view(1, 1, -1)) | ~real)).all()):
            noncontig = True
    assert noncontig, \
        "this topology's real edges are still tile-contiguous, so it does not exercise block-sparsity"

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [64, 2], device = dev)
    phi = torch.randn(_gate_shape(ns, 64), device = dev)

    with pytest.raises(NotImplementedError):
        pc(data, sum_external_params = {ns: phi})


# --------------------------------------------------------------------------------- numeric
# xfail until the plan-time check is relaxed.

@cuda_only
@needs_cute
@pending
@pytest.mark.parametrize("name", RAGGED)
def test_forward_matches_materialized_pc(name):
    """The gated ragged forward against a plain PC carrying the effective per-sample parameters --
    the same oracle the dense tests use, which shares no code with the kernels."""
    dev = torch.device("cuda:0")
    gate_cbs, batch = 8, 64
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
@pending
@pytest.mark.parametrize("name", RAGGED)
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
@pending
@pytest.mark.parametrize("name", RAGGED)
def test_param_flows_are_bit_stable_across_repeats(name):
    """Structural companion to the tolerance check above: a lost update leaves an absolute error
    smaller than this suite's element-flow tolerance, so the tolerance alone cannot see it. Repeated
    runs of a deterministic computation must agree BITWISE."""
    pc, root, ns, layer, data, phi = _run(name)
    outs = []
    for _ in range(REPEATS):
        pc(data, sum_external_params = {ns: phi})
        pc.backward(data, flows_memory = 0.0)
        outs.append(pc.param_flows.clone())
    base = outs[0]
    ndiff = sum(1 for o in outs if not torch.equal(o, base))
    assert ndiff == 0, f"{name}: {ndiff} of {REPEATS} runs differ bitwise -- a lost update"


@cuda_only
@needs_cute
@pending
@pytest.mark.parametrize("name", RAGGED)
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
@pending
def test_a_dense_and_a_ragged_gated_layer_coexist():
    """Padding is a property of one compiled partition, not of the circuit. A layer holding both must
    still be right -- and the dense partition must not pay the padded partition's masking."""
    pc, root, ns, layer, data, phi = _run("split")
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

    expected = ("no block-scale forward applies" if not RAGGED_SUPPORTED
                else "no external param-flow backward applies")
    assert expected in str(excinfo.value), \
        f"refused, but from the wrong place: {excinfo.value}"
