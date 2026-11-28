"""
`LowRank(top_k)` sum layers handed the product layer group below (the circuit's `stream_children`: LowRank's
`stream_fusion_plan` and `stream_fusion_bk`, `TensorCircuit._stream_fusion_groups` and `_stream_fusion_bk_groups`):
their kernels read each child's value as its product's -- its two children's node values summed, the product layer's
add -- and store each child's flow as its product's two children's node flows -- the product layer's backward where
each child has one parent element --, so no product launch is made below them; their torch passes have the values
formed first and the flows written as the children's after. Against the same circuit never fusing: the values bit for
bit, whatever the masks, the map of rows, the parameters' sources and the path (the stacked passes, the unstacked
kernels, the torch passes); a query's answers and the gradients up to the backward kernels' order of adds; and the
product layers below a fused sum group never launched.
"""

import contextlib
import importlib.util
import os

import pytest
import torch

import pyjuice.layer.kernels.autotune as autotune
from pyjuice.layer.prod_layer import ProdLayer
from pyjuice.queries import conditional

# (the stack tests' circuits and batches, loaded by path: the suite's importlib mode puts no test directory on
# `sys.path`)
_spec = importlib.util.spec_from_file_location(
    "_lowrank_top_k_stack_for_fusion",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowrank_top_k_stack_test.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
H = S.H

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


@pytest.fixture
def fused_preferred(monkeypatch):
    """The circuit's fusion decisions -- the forward's and the backward's -- steered to the fused passes, after every
    candidate's trial ran as the tuner runs it (on the live buffers: what a trial leaves behind is what the pass then
    reads)."""
    real = autotune.pick

    def pick(key, cands, bench, *a, **k):
        if key[0] not in ("prod_stream_fusion", "prod_stream_fusion_bk"):
            return real(key, cands, bench, *a, **k)
        for c in cands:
            bench(c)
        return "fused"
    monkeypatch.setattr(autotune, "pick", pick)


def _fused(pc, state):
    """the product layer groups (by index) the circuit's decisions left to the sums above them, any call's (`state`:
    the forward's decisions, `_stream_fusion_`, or the backward's, `_stream_fusion_bk_`)"""
    got = set()
    for v in pc.__dict__.get(state, {}).values():
        if isinstance(v, frozenset):
            got |= v
    return got


@contextlib.contextmanager
def _products_run():
    """the product layers whose forward and backward ran: `(forwards, backwards)`, lists of the layers"""
    fw, bw = [], []
    real_f, real_b = ProdLayer.forward, ProdLayer.backward
    ProdLayer.forward = lambda self, *a, **k: (fw.append(self), real_f(self, *a, **k))[1]
    ProdLayer.backward = lambda self, *a, **k: (bw.append(self), real_b(self, *a, **k))[1]
    try:
        yield fw, bw
    finally:
        ProdLayer.forward, ProdLayer.backward = real_f, real_b


def _layers_of(pc, groups):
    return {id(layer) for k in groups for layer in pc.inner_layer_groups[k]}


@contextlib.contextmanager
def _path(name):
    """the passes a sum layer takes: the stacked ones ("stack"), the unstacked kernels ("kernels"), the torch passes
    ("torch")"""
    tkf, tlf, tks = S._modules()
    stacked, fused = tks.STACKED, tkf.TOP_K_FUSED
    tks.STACKED, tkf.TOP_K_FUSED = name == "stack", name != "torch"
    try:
        yield
    finally:
        tks.STACKED, tkf.TOP_K_FUSED = stacked, fused


def _pair(b, masked, seed, tied = True, rows = None):
    """two circuits made alike -- the first fusing where the decisions say, the second never -- and a call's arguments"""
    pc, kw, x, miss = S._case(b, masked, seed = seed, tied = tied, rows = rows, fuses = True)
    ref = S._case(b, masked, seed = seed, tied = tied, rows = rows)[0]
    return pc, ref, kw, x, miss


@pytest.mark.parametrize("path", ["stack", "kernels", "torch"])
@pytest.mark.parametrize("masked", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rows", [None, [0, 1, 1, 3, 0, 2]])
@pytest.mark.parametrize("tied", [True, False])
def test_the_fused_forward_is_the_unfused_one(path, masked, rows, tied, fused_preferred):
    """Forward after forward, the values bit for bit -- the first unfused in both (the layers settle), the second the
    decision's (its trials on the live buffers), the third with the product layers left to the sums: none of theirs
    launched below a fused sum group, every other one launched."""
    if path == "torch" and (masked != 0.5 or rows is not None or not tied):
        pytest.skip("the torch passes: one case")
    S._modules()
    pc, ref, kw, x, miss = _pair(4 if rows else 5, masked, seed = 7, tied = tied, rows = rows)
    with torch.no_grad(), _path(path):
        want = [ref(x, missing_mask = miss, **kw).clone() for _ in range(3)]
        got = [pc(x, missing_mask = miss, **kw).clone() for _ in range(2)]
        with _products_run() as (fw, _):
            got.append(pc(x, missing_mask = miss, **kw).clone())
    for g, w in zip(got, want):
        assert torch.equal(g, w), float((g - w).abs().max())
    fused = _fused(pc, "_stream_fusion_")
    assert len(fused) == S.T - 2, fused                            # (every sum over a product of two: all but the first)
    left = _layers_of(pc, fused)
    ran = {id(layer) for layer in fw}
    assert not ran & left
    assert ran == {id(layer) for grp in pc.inner_layer_groups if grp.is_prod() for layer in grp} - left


@pytest.mark.parametrize("path", ["stack", "kernels", "torch"])
@pytest.mark.parametrize("masked", [0.5, 1.0])
def test_the_fused_backward_is_the_unfused_one(path, masked, fused_preferred):
    """A conditional query (its forward keeps its stash for its backward: the stack's, or the unstacked forward's) and
    the gradients of the external tensors and the evidence through the circuit's backward, call after call: the
    unfused circuit's, up to the backward kernels' order of adds -- with no product layer's recompute nor backward run
    below a fused sum group in the last calls."""
    S._modules()
    pc, ref, kw, x, miss = _pair(4, masked, seed = 11)

    def run(c):
        probs = conditional(c, x, missing_mask = miss, **kw)
        ts = {g: tuple(t.clone().requires_grad_(True) for t in v) for g, v in kw["external_params"].items()}
        lpg = kw["categorical_evidence_logp"].clone().requires_grad_(True)
        ll = c(x, missing_mask = miss, external_params = ts, categorical_evidence_logp = lpg,
               soft_evidence_cat_ids = kw["soft_evidence_cat_ids"], differentiable = True).view(-1)
        ll.sum().backward()
        return [probs, ll.detach()] + [t.grad for v in ts.values() for t in v] + [lpg.grad]

    with _path(path):
        want = [run(ref) for _ in range(3)]
        got = [run(pc) for _ in range(2)]
        with _products_run() as (fw, bw):
            got.append(run(pc))
    for g, w in zip(got, want):
        assert torch.equal(g[1], w[1])                              # (the values: the forward's, bit for bit)
        assert H._rel(g[0], w[0]) < 1e-5, H._rel(g[0], w[0])        # (the conditional's answers)
        for k, (u, v) in enumerate(zip(g[2:], w[2:])):
            if masked < 1.0:
                assert H._rel(u, v) < 1e-5, (k, H._rel(u, v))
            else:                                                   # (every token summed out: each likelihood 1
                assert float((u - v).abs().max()) < 1e-6, k         #  whatever the tensors, its gradients 0 up to
                                                                    #  the roundoff, compared absolutely -- a
                                                                    #  residue of O(1) terms, ~1e-7 apart where the
                                                                    #  two circuits' plain layers each timed their own
                                                                    #  element backward's arm, whose bits differ
                                                                    #  (`_cached_bk_ele_choice`): MEASURED 2e-7)
    fused, fused_bk = _fused(pc, "_stream_fusion_"), _fused(pc, "_stream_fusion_bk_")
    assert len(fused) == S.T - 2 and fused_bk == fused, (fused, fused_bk)
    left = _layers_of(pc, fused_bk)
    assert not {id(layer) for layer in fw} & left and not {id(layer) for layer in bw} & left


@pytest.mark.parametrize("path", ["stack", "kernels", "torch"])
def test_the_fused_em_statistics_are_the_unfused_ones(path, fused_preferred):
    """A forward, then a backward with the EM statistics (the forward keeps no stash: the backward makes its operands
    again, the children's values read through the table), call after call: the node flows and the sums' and the leaves'
    statistics the unfused circuit's, up to the backward kernels' order of adds -- with no product layer's recompute nor
    backward run below a fused sum group in the last call."""
    S._modules()
    pc, ref, kw, x, miss = _pair(4, 0.5, seed = 13)

    def run(c):
        c.init_param_flows(flows_memory = 0.0)
        with torch.no_grad():
            c(x, missing_mask = miss, **kw)
        c.backward(x, missing_mask = miss, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                   allow_modify_flows = False, **kw)
        out = [c.node_flows.clone(), c.param_flows.clone(), c.input_layer_group[0].param_flows.clone()]
        if c.denom_param_flows is not None:
            out.append(c.denom_param_flows.clone())
        return out

    with _path(path):
        want = [run(ref) for _ in range(3)]
        got = [run(pc) for _ in range(2)]
        with _products_run() as (fw, bw):
            got.append(run(pc))
    for g, w in zip(got, want):
        fin = torch.isfinite(w[0])                                   # (the node flows, log space)
        assert bool((torch.isfinite(g[0]) == fin).all())
        assert float((g[0][fin] - w[0][fin]).abs().max()) < 1e-4, float((g[0][fin] - w[0][fin]).abs().max())
        assert len(g) == len(w) and len(g) > 2
        for k, (u, v) in enumerate(zip(g[1:], w[1:])):                 # (the sums', the leaves' statistics)
            assert H._rel(u, v) < 1e-5, (k, H._rel(u, v))
    fused_bk = _fused(pc, "_stream_fusion_bk_")
    assert len(fused_bk) == S.T - 2, fused_bk
    left = _layers_of(pc, fused_bk)
    assert not {id(layer) for layer in fw} & left and not {id(layer) for layer in bw} & left
