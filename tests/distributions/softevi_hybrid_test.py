"""The hybrid top-k passes of `SoftEvidenceCategorical` (the dense choice "hybrid"): the hot categories -- referenced by at
least 1 / `_HOT_DIV` of their block's slots -- take their part as dense products over every slot, the cold ones are
walked as before. The backward (`softevi/backward.py`'s `_hot_expected_flow`) against the "single" layout's list walk, a
whole backward: the same parameter flows and evidence gradient; the forward (`softevi/forward.py`'s
`_fw_launch_hybrid`) against the row form's: the same node values. The dense
path is forced on toy shapes as in `softevi_missing_mask_test.py` (the L2 figure shrunk, the performance call made), and
the hot set made non-empty (`_HOT_MIN`, `_HOT_DIV`)."""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes.distributions import c_kernels
from pyjuice.nodes.distributions.softevi import backward as _sbw, dense_index as _dense, forward as _sfw


def _setup(monkeypatch, homogeneous, dual_flow, B = 12, V = 5, L = 8, C = 96, K = 20, seed = 0):
    device = torch.device("cuda:0")
    monkeypatch.setattr(_dense._l2_bytes, "_cached", 512, raising = False)
    monkeypatch.setattr(_dense, "_dense_worth_it", lambda layer, kwargs: True)
    monkeypatch.setattr(_dense, "_HOT_MIN", 2)
    monkeypatch.setattr(_dense, "_HOT_DIV", 4)
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(
        seq_length = V, num_latents = L, homogeneous = homogeneous,
        input_dist = dists.SoftEvidenceCategorical(num_cats = C, _dual_flow_backward = dual_flow))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(device)
    layer = pc.input_layer_group[0]
    g = torch.Generator(device = device).manual_seed(seed + 1)
    data = torch.randint(0, C, (B, V), device = device, generator = g)
    lp = torch.log_softmax(3.0 * torch.randn(B, V, K, device = device, generator = g), dim = 2).contiguous()
    # K distinct candidates a slot, the observed token among them; a few categories in most lists (hot), most in few
    w = torch.ones(C, device = device)
    w[:8] = 200.0
    ids = torch.multinomial(w.expand(B * V, -1), K, replacement = False, generator = g).view(B, V, K)
    missing = ~(ids == data.unsqueeze(-1)).any(dim = -1)
    ids[:, :, -1] = torch.where(missing, data, ids[:, :, -1])
    return pc, layer, data, dict(categorical_evidence_logp = lp, soft_evidence_cat_ids = ids.contiguous())


def _backward(pc, layer, data, kw, choice, with_grad, mask = None):
    layer._dense_choice_override = choice
    try:
        pc.init_param_flows(flows_memory = 0.0)
        extra = {} if mask is None else {"missing_mask": mask}
        pc(data, **kw, **extra)
        grad = torch.zeros_like(kw["categorical_evidence_logp"]) if with_grad else None
        gkw = {} if grad is None else {"categorical_evidence_logp_grad": grad}
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, **kw, **gkw, **extra)
        return layer.param_flows.clone(), grad
    finally:
        layer._dense_choice_override = None


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("dual_flow,with_grad", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("masked,L", [(False, 8), (True, 8), (False, 64)])
def test_hybrid_backward_is_the_list_walks(homogeneous, dual_flow, with_grad, masked, L, monkeypatch):
    """Tied emissions (one block) and untied (a block a variable: the hot keys carry their block), the expected flow
    alone, with the evidence gradient, and the gradient alone (no dual flows: nothing written to the flows' second
    half); with a missing mask (its slots' ratios are zero: they add nothing on either side); latents in one tile of
    the hot flows' kernel and in two."""
    pc, layer, data, kw = _setup(monkeypatch, homogeneous, dual_flow, L = L)
    mask = None
    if masked:
        mask = torch.zeros(data.shape, dtype = torch.bool, device = data.device)
        mask[:, 1] = True
        mask[3, 2] = True
    ran = []
    real = _sbw._hot_expected_flow
    monkeypatch.setattr(_sbw, "_hot_expected_flow",
                        lambda kw_, index, pt: ran.append(index["hot"]["keys"].numel()) or real(kw_, index, pt))
    pf_h, g_h = _backward(pc, layer, data, kw, "hybrid", with_grad, mask)
    assert ran and ran[0] >= 2, "the hybrid backward did not take any category as hot"
    index = _dense._build_dense_index(layer, kw, ladder = False, cold = True)
    full = _dense._build_dense_index(layer, kw, ladder = False)
    n_cold = sum(int(s["ref_cnt"].sum()) for s in index["shards"])
    assert n_cold + index["hot"]["elem"].numel() == sum(int(s["ref_cnt"].sum()) for s in full["shards"])
    assert 0 < n_cold

    pf_s, g_s = _backward(pc, layer, data, kw, "single", with_grad, mask)
    assert torch.equal(pf_h != 0, pf_s != 0), "the hybrid and the walk write different flows"
    assert torch.allclose(pf_h, pf_s, rtol = 1e-5, atol = 1e-7), f"flows: max diff {(pf_h - pf_s).abs().max():.3e}"
    if with_grad:
        assert torch.allclose(g_h, g_s, rtol = 1e-5, atol = 1e-6), f"gradient: max diff {(g_h - g_s).abs().max():.3e}"


def test_hybrid_declines_without_hot_categories(monkeypatch):
    """Too few hot categories (`_HOT_MIN`): no cold index, the "hybrid" choice is not offered to the tuner, and forced
    it walks the whole lists -- the single layout's backward."""
    pc, layer, data, kw = _setup(monkeypatch, True, True)
    monkeypatch.setattr(_dense, "_HOT_MIN", 10 ** 6)
    assert _dense._build_dense_index(layer, kw, ladder = False, cold = True) is None
    ran = []
    monkeypatch.setattr(_sbw, "_hot_expected_flow", lambda *a: ran.append(1))
    pf_h, g_h = _backward(pc, layer, data, kw, "hybrid", True)
    pf_s, g_s = _backward(pc, layer, data, kw, "single", True)
    assert not ran
    assert torch.allclose(pf_h, pf_s, rtol = 1e-6, atol = 1e-9) and torch.allclose(g_h, g_s, rtol = 1e-6, atol = 1e-9)


def _forward(pc, layer, data, kw, choice, monkeypatch):
    monkeypatch.setattr(_sfw, "_fw_autotune", lambda k: choice)
    layer._dense_choice_override = "hybrid"
    try:
        pc(data, **kw)
        s, e = layer._output_ind_range
        return pc.node_mars[s:e].clone()
    finally:
        layer._dense_choice_override = None


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("L,K,chunk", [(8, 20, 8), (8, 13, 4), (64, 20, 64), (64, 13, 32)])
def test_hybrid_forward_is_the_row_forms(homogeneous, L, K, chunk, monkeypatch):
    """Tied and untied emissions, candidate counts off the cold sums' unroll, the latents in one chunk or two; Z (kept
    across steps) and the tables (fresh each launch) poisoned with NaN, then with a finite value: written wherever read."""
    if not c_kernels.softevi_rows_available():
        pytest.skip("the row-form CUDA extension is not available")
    pc, layer, data, kw = _setup(monkeypatch, homogeneous, True, L = L, K = K)
    # (the candidates reversed -- their order means nothing -- so the hot ones, drawn first, sit in the sums' tail)
    kw = {k: v.flip(2).contiguous() for k, v in kw.items()}
    rows = _forward(pc, layer, data, kw, ("rows", chunk), monkeypatch)
    cold = _dense._build_dense_index(layer, kw, ladder = False, cold = True)
    assert cold is not None and cold["rows"] is not None and int((cold["rows"]["elem_key"] < 0).sum()) > 0

    class _PoisonedEmpty:
        def __init__(self, value):
            self.value = value

        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *size, **k):
            out = torch.empty(*size, **k)
            return out.fill_(self.value) if out.is_floating_point() else out

    for value in (None, float("nan"), 3.0):
        with monkeypatch.context() as mp:
            if value is not None:
                layer._fw_z_buf.fill_(value)
                mp.setattr(_sfw, "torch", _PoisonedEmpty(value))
            hyb = _forward(pc, layer, data, kw, ("hybrid", chunk), mp)
        assert torch.isfinite(hyb).all(), f"poisoned with {value}: non-finite node values"
        assert torch.allclose(hyb, rows, rtol = 0, atol = 2e-5), \
            f"poisoned with {value}: hybrid vs row form {(hyb - rows).abs().max():.3e}"
