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


STRUCTURES = {"hmm": _hmm, "hclt": _hclt, "pd": _pd, "rat_spn": _rat, "ragged": _ragged}


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
    assert plan.num_elem_rows == pc._num_escopes


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
