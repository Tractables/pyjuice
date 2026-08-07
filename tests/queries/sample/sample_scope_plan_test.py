"""
The sampler's structural frontier layout (`queries/sampling/scope_plan.py`).

Today's top-down pass discovers its frontier layout at run time -- a `torch.where` to find each
layer's entries, a compaction to keep the buffer dense, a per-column cursor to hand out slots. That
bookkeeping is 92% of the pass's GPU time and ~80% of its wall time on a `PD` circuit; the sampling
kernels are 8%.

None of it depends on the draw. A frontier entry stands for a SCOPE, and which scopes a layer owns is
a property of the circuit, so the whole layout can be derived at compile time. These tests pin the
derivation, since everything downstream will address the frontier through it: a wrong row does not
crash, it writes one node's child into another node's slot.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.queries.sampling.scope_plan import build_scope_plan


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
NUM_CATS = 8


def _hmm(num_vars = 6, K = 32):
    with juice.set_block_size(K):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        for v in range(1, num_vars):
            ns = summate(multiply(ns, inputs(v, num_node_blocks = 1,
                                             dist = dists.Categorical(num_cats = NUM_CATS))),
                         num_node_blocks = 1)
        return summate(multiply(ns), num_node_blocks = 1, block_size = 1)


def _hclt(num_vars = 12):
    data = torch.randint(0, NUM_CATS, [256, num_vars]).float().to(torch.device("cuda:0"))
    return juice.structures.HCLT(data, num_bins = NUM_CATS, sigma = 0.5 / NUM_CATS,
                                 num_latents = 8, chunk_size = 8)


def _pd(num_vars = 16):
    """Not structured decomposable: several splits of one region. The case the whole layout exists
    for, since its plan cannot be cached."""
    return juice.structures.PD(data_shape = (num_vars,), num_latents = 16, split_intervals = (4,))


def _rat(num_vars = 16):
    return juice.structures.RAT_SPN(num_vars = num_vars, num_latents = 8, depth = 2,
                                    num_repetitions = 2, num_pieces = 2,
                                    input_dist = dists.Categorical(num_cats = NUM_CATS))


def _ragged():
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        m0 = multiply(i[0], i[1], edge_ids = torch.tensor([[0, 0], [1, 2], [2, 1]], dtype = torch.long))
        s0 = summate(m0, edge_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]], dtype = torch.long))
        s1 = summate(multiply(i[2], i[3]), num_node_blocks = 2)
        m2 = multiply(s0, s1, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        return summate(m2, num_node_blocks = 1, block_size = 1)


def _mixture():
    """
    Two sub-circuits over the SAME scopes at different block sizes, joined at a block-size-1 root.
    Their layers land in the same group, so a product group here owns each scope TWICE.

    REGRESSION, and the reason this file grew a sixth structure: none of the five above has a product
    group with more than one layer, so all of them passed while the layout was giving each (layer,
    scope) its own element row. That left `sum_erows` pointing at one layer's row while the other
    layer's kernel scanned its own, and the second branch's elements were written and never read --
    2942 of 4096 draws came back missing at least one variable, silently, as a zero.
    """
    def branch(block_size, num_node_blocks):
        i = [inputs(v, num_node_blocks = num_node_blocks, block_size = block_size,
                    dist = dists.Categorical(num_cats = 5)) for v in range(6)]
        pairs = [summate(multiply(i[a], i[b]), num_node_blocks = num_node_blocks,
                         block_size = block_size) for a, b in ((0, 1), (2, 3), (4, 5))]
        return summate(multiply(*pairs), num_node_blocks = 1, block_size = 1)

    return summate(multiply(branch(4, 2)), multiply(branch(2, 4)),
                   num_node_blocks = 1, block_size = 1)


def _sparse_mixture():
    """The same overlap reached through explicit block-sparse `edge_ids` rather than dense layers --
    it fails identically without the fix (2613 of 4096), so the defect is about the layout, not about
    how the edges were built."""
    def branch(block_size, num_node_blocks):
        i = [inputs(v, num_node_blocks = num_node_blocks, block_size = block_size,
                    dist = dists.Categorical(num_cats = 5)) for v in range(6)]
        pairs = []
        for a, b in ((0, 1), (2, 3), (4, 5)):
            m = multiply(i[a], i[b])
            edges = torch.tensor([[k % num_node_blocks for k in range(2 * num_node_blocks)],
                                  [k % num_node_blocks for k in range(2 * num_node_blocks)]],
                                 dtype = torch.long)
            pairs.append(summate(m, edge_ids = edges, num_node_blocks = num_node_blocks,
                                 block_size = block_size))
        return summate(multiply(*pairs), num_node_blocks = 1, block_size = 1)

    return summate(multiply(branch(4, 2)), multiply(branch(2, 4)),
                   num_node_blocks = 1, block_size = 1)


STRUCTURES = {"hmm": _hmm, "hclt": _hclt, "pd": _pd, "rat_spn": _rat, "ragged": _ragged,
              "mixture": _mixture, "sparse_mixture": _sparse_mixture}

# The structures whose product groups actually own a scope in more than one layer. Asserted rather
# than assumed in `test_the_overlapping_structures_really_do_overlap`: if a change to compilation
# merged those layers, the regression tests below would keep passing while testing nothing.
OVERLAPPING = ("mixture", "sparse_mixture")


@pytest.fixture(scope = "module")
def planned():
    out = {}
    for name, build in STRUCTURES.items():
        torch.manual_seed(0)
        ns = build()
        ns.init_parameters(perturbation = 2.0)
        pc = juice.compile(ns, verbose = False).to(torch.device("cuda:0"))
        juice.queries.sample(pc, num_samples = 8)          # populates `_num_nscopes` / `_num_escopes`
        out[name] = (pc, build_scope_plan(pc))
    return out


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_the_layout_needs_exactly_the_buffers_the_driver_allocates(planned, name):
    """A different discipline for using the frontier, not a new demand on it. If these ever diverge,
    the derivation has drifted from what the pass actually needs."""
    pc, plan = planned[name]

    assert plan.num_node_rows == pc._num_nscopes
    # `<=` for element rows, not `==`: the driver sizes `_num_escopes` by totalling every layer's
    # scopes, while a group's layers SHARE a row per scope, so an overlapping structure needs strictly
    # fewer. The direction that matters is that the layout never demands more than is allocated.
    assert plan.num_elem_rows <= pc._num_escopes


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_rows_are_unique_within_a_layer(planned, name):
    """Two scopes sharing a row would have them overwrite each other's selections."""
    pc, plan = planned[name]

    for rows in list(plan.sum_rows.values()) + list(plan.prod_rows.values()):
        assert len(set(rows.tolist())) == rows.numel()
        assert int(rows.min()) >= 0


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_an_element_row_means_the_same_scope_across_a_whole_group(planned, name):
    """
    Within a product group a row must be a FUNCTION OF THE SCOPE, not of the layer that owns it: the
    sum layer above has one destination per scope, so two layers owning that scope have to meet on
    one row and separate by `nids`. Both directions are pinned -- distinct scopes never share a row
    (they would overwrite each other), and one scope never spreads across two (half the writes would
    go unread, which is the defect `_mixture` exists for).
    """
    pc, plan = planned[name]

    for group in pc.inner_layer_groups:
        if not group.is_prod():
            continue

        row_of_scope, scope_of_row = {}, {}
        for layer in group:
            rows = plan.prod_rows[id(layer)].tolist()
            for scope, row in zip(layer.scopes, rows):
                key = tuple(sorted(scope))
                assert row_of_scope.setdefault(key, row) == row, \
                    "one scope was given two element rows -- one layer's writes are unreadable"
                assert scope_of_row.setdefault(row, key) == key, \
                    "two scopes were given one element row"


@cuda_only
@pytest.mark.parametrize("name", OVERLAPPING)
def test_the_overlapping_structures_really_do_overlap(planned, name):
    """The regression tests above are only worth their runtime if these structures still put one
    scope in several product layers of a group. Compilation changes could quietly merge them."""
    pc, plan = planned[name]

    worst = 0
    for group in pc.inner_layer_groups:
        if not group.is_prod():
            continue
        counts = {}
        for layer in group:
            for key in {tuple(sorted(scope)) for scope in layer.scopes}:
                counts[key] = counts.get(key, 0) + 1
        worst = max(worst, max(counts.values(), default = 0))

    assert worst > 1, f"'{name}' no longer has a scope owned by several product layers of one group"


@cuda_only
@pytest.mark.parametrize("name", OVERLAPPING)
def test_a_draw_over_overlapping_product_layers_reaches_every_variable(planned, name):
    """
    The behaviour the layout exists to produce, on the structures that used to break it.

    Checked as coverage rather than as a distribution: a lost branch leaves rows untouched, and an
    untouched row is returned as the buffer's `0` -- a perfectly plausible category. That is why this
    shipped silently, and why the frontier is read directly here.
    """
    pc, _ = planned[name]

    frontier = juice.queries.sample(pc, num_samples = 2048, _sample_input_ns = False)
    covered = (frontier != -1).sum(dim = 0)

    assert bool((covered == pc.num_vars).all()), \
        f"{int((covered < pc.num_vars).sum())} of 2048 draws are missing a variable"


@cuda_only
@pytest.mark.parametrize("name", OVERLAPPING)
def test_overlapping_draws_match_the_pair_list_pass(planned, name):
    """Coverage alone would accept a draw that reaches every variable from the wrong branch, so the
    marginals are compared against the independently-written pair-list pass."""
    pc, _ = planned[name]
    N = 60_000

    torch.manual_seed(1)
    scoped = juice.queries.sample(pc, num_samples = N).float()
    torch.manual_seed(1)
    pairs = juice.queries.sample(pc, num_samples = N, _use_scope_plan = False).float()

    se = ((scoped.var(dim = 0) + pairs.var(dim = 0)) / N).sqrt().clamp(min = 1e-9)
    z = float(((scoped.mean(dim = 0) - pairs.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"scoped and pair-list draws differ: max |z| = {z:.2f}"


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_every_sum_layer_knows_where_its_drawn_child_lands(planned, name):
    """A sum node's children share its scope, so each of its rows has an element row in the product
    group below. A `-1` here would mean a draw with nowhere to go."""
    pc, plan = planned[name]

    assert set(plan.sum_erows) == set(plan.sum_rows)
    for layer_id, erows in plan.sum_erows.items():
        assert erows.numel() == plan.sum_rows[layer_id].numel()
        assert int(erows.min()) >= 0
        assert int(erows.max()) < plan.num_elem_rows


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_child_slots_resolve_to_their_own_scopes_row(planned, name):
    """
    The table that makes a circuit whose decomposition varies with the draw addressable without a
    cursor. Every real child must resolve; every padded slot must not, or the pass would write the
    dummy element into a live row.
    """
    pc, plan = planned[name]

    checked = 0
    for group in pc.inner_layer_groups:
        if not group.is_prod():
            continue
        for layer in group:
            for partition_id, crows in enumerate(plan.prod_crows[id(layer)]):
                cids = layer.partitioned_cids[partition_id].cpu()
                assert crows.shape == cids.shape

                real = cids > 0
                assert bool((crows[real] >= 0).all()), "a real child did not resolve to a row"
                assert bool((crows[real] < plan.num_node_rows).all())
                assert bool((crows[~real] < 0).all()), "a padded slot was given a row"
                checked += int(real.sum())

    assert checked > 0, "no child slots were checked -- the structure has no product layers?"


# ------------------------------------------------------------- caches the sampler holds on the pc

def _cached(pc):
    return {name: pc.__dict__.get(name)
            for name in ("_sample_scope_plan", "_sample_scoped_states", "_sample_plans")}


@cuda_only
def test_the_caches_are_dropped_when_the_circuit_changes_device():
    """
    REGRESSION. The scope plan, the persistent frontier buffers and the recorded index plans are all
    cached ON THE CIRCUIT and all hold device tensors, and none of them recorded which device that
    was. After a `pc.to(other_device)` they would be handed to kernels launched against the new one.

    The DEVICE CHANGE is simulated rather than performed: this suite's `conftest` pins each worker to
    a single GPU before torch is imported, so an in-process `pc.to("cuda:1")` is not reachable here
    (`test_a_real_cross_device_move_rebuilds_the_plan` does it for real and is normally skipped).
    What is checked is the mechanism itself, in both directions -- caches are dropped when the
    recorded device no longer matches, and kept when it does, since dropping them unconditionally
    would also "pass" while quietly rebuilding the layout on every single call.
    """
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        root = summate(multiply(summate(multiply(i[0], i[1]), num_node_blocks = 2),
                                summate(multiply(i[2], i[3]), num_node_blocks = 2)),
                       num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))

    juice.queries.sample(pc, num_samples = 256, use_cudagraph = True)
    juice.queries.sample(pc, num_samples = 256, _use_scope_plan = False)
    warm = _cached(pc)
    assert all(value is not None for value in warm.values()), "the caches did not warm up"

    # same device -> everything is REUSED, or the fix costs a rebuild per call
    juice.queries.sample(pc, num_samples = 256, use_cudagraph = True)
    assert pc.__dict__["_sample_scope_plan"] is warm["_sample_scope_plan"]
    assert pc.__dict__["_sample_scoped_states"] is warm["_sample_scoped_states"]

    # the circuit now reports a different device -> every cache holding its tensors must go
    pc.__dict__["_sample_cache_device"] = torch.device("cuda:7")
    juice.queries.sample(pc, num_samples = 256)

    assert pc.__dict__["_sample_scope_plan"] is not warm["_sample_scope_plan"], \
        "the scope plan was reused across a device change"
    assert pc.__dict__.get("_sample_scoped_states") is not warm["_sample_scoped_states"], \
        "the pinned frontier buffers were reused across a device change"
    assert pc.__dict__.get("_sample_plans") is not warm["_sample_plans"], \
        "the recorded index plans were reused across a device change"
    assert pc.__dict__["_sample_cache_device"] == pc.device


@cuda_only
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "needs two GPUs")
def test_a_real_cross_device_move_rebuilds_the_plan():
    """
    The same thing performed rather than simulated. Normally SKIPPED -- see the note above -- so it
    is a manual check (`CUDA_VISIBLE_DEVICES=0,1 pytest -p no:cacheprovider <this file> -k real`),
    not something the suite's green tick stands behind.
    """
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        root = summate(multiply(summate(multiply(i[0], i[1]), num_node_blocks = 2),
                                summate(multiply(i[2], i[3]), num_node_blocks = 2)),
                       num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))

    juice.queries.sample(pc, num_samples = 256)                             # warms the scope plan
    juice.queries.sample(pc, num_samples = 256, _use_scope_plan = False)    # warms the index plans
    assert "_sample_scope_plan" in pc.__dict__ and "_sample_plans" in pc.__dict__

    pc = pc.to(torch.device("cuda:1"))

    for kwargs in ({}, {"_use_scope_plan": False}):
        samples = juice.queries.sample(pc, num_samples = 256, **kwargs)
        assert samples.device.index == 1

    plan = pc.__dict__["_sample_scope_plan"]
    rows = plan.sum_rows[next(iter(plan.sum_rows))]
    assert rows.device.index == 1, "the scope plan was reused from the old device"


@cuda_only
def test_calibration_takes_effect_on_the_default_path():
    """
    REGRESSION. `_do_calibration` rescales the uniform by the total edge weight, which is what makes
    the inverse-CDF walk a correct draw when a sum node's parameters do not sum to one. The default
    (scoped) path accepted the flag and ignored it.

    It is a no-op on a circuit built the usual way -- which is why this test DE-NORMALIZES one, and
    why it checks the flag changes the answer there before checking the answer is right. Otherwise a
    silently-ignored flag passes both halves.
    """
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4)) for v in range(4)]
        root = summate(multiply(summate(multiply(i[0], i[1]), num_node_blocks = 2),
                                summate(multiply(i[2], i[3]), num_node_blocks = 2)),
                       num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    N = 60_000

    def draw(**kwargs):
        torch.manual_seed(1)
        return juice.queries.sample(pc, num_samples = N, **kwargs).float()

    def max_z(a, b):
        se = ((a.var(dim = 0) + b.var(dim = 0)) / N).sqrt().clamp(min = 1e-9)
        return float(((a.mean(dim = 0) - b.mean(dim = 0)) / se).abs().max())

    assert max_z(draw(_do_calibration = True), draw()) < 5.0, \
        "calibration changed a NORMALIZED circuit's draw, where it should be a no-op"

    pc.params[1:] *= 0.35                       # every sum node's edges now total well under one

    calibrated = draw(_do_calibration = True)
    assert max_z(calibrated, draw()) > 5.0, \
        "calibration made no difference on an unnormalized circuit -- the flag is being ignored"
    z = max_z(calibrated, draw(_do_calibration = True, _use_scope_plan = False))
    assert z < 5.0, f"scoped and pair-list calibrated draws differ: max |z| = {z:.2f}"
