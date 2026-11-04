"""The row form of `SoftEvidenceCategorical`'s top-k forward (`c_kernels/softevi_rows.cu`, launched by `softevi/forward.py`'s
`_fw_launch_rows`): the referenced categories' parameters transposed into a table, then each slot's candidates' rows
summed into the local normalizer `Z`. It runs on the dense index, so -- as in `softevi_missing_mask_test.py` -- the L2
figure the dense path gates on is shrunk instead of building a multi-GB model, and the path is forced past its
performance call; the form itself is forced by stubbing `_fw_autotune`'s choice."""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes.distributions import c_kernels
from pyjuice.nodes.distributions.softevi import dense_index as _dense, forward as _sfw


def _needs_rows():
    if not c_kernels.softevi_rows_available():
        pytest.skip("the row-form CUDA extension is not available")


def _setup(monkeypatch, num_vars, num_latents, num_cats, K, B, homogeneous, seed = 0):
    device = torch.device("cuda:0")
    monkeypatch.setattr(_dense._l2_bytes, "_cached", 512, raising = False)
    monkeypatch.setattr(_dense, "_dense_worth_it", lambda layer, kwargs: True)
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(
        seq_length = num_vars, num_latents = num_latents, homogeneous = homogeneous,
        input_dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(device)
    layer = pc.input_layer_group[0]

    g = torch.Generator(device = device).manual_seed(seed + 1)
    data = torch.randint(0, num_cats, (B, num_vars), device = device, generator = g)
    lp = torch.log_softmax(3.0 * torch.randn(B, num_vars, K, device = device, generator = g), dim = 2).contiguous()
    # K distinct candidates a slot, the observed token among them; common categories in many lists (a low-id
    # half of the vocabulary drawn more often), as an LM's top-k
    w = torch.ones(num_cats, device = device)
    w[:num_cats // 4] = 20.0
    ids = torch.multinomial(w.expand(B * num_vars, -1), K, replacement = False, generator = g).view(B, num_vars, K)
    missing = ~(ids == data.unsqueeze(-1)).any(dim = -1)
    ids[:, :, -1] = torch.where(missing, data, ids[:, :, -1])
    kw = dict(categorical_evidence_logp = lp, soft_evidence_cat_ids = ids.contiguous())
    return pc, layer, data, kw


def _forward(pc, layer, data, kw, choice, monkeypatch):
    """The input layer's node_mars with the normalizer's form forced (`("rows", chunk)` or `(TL, swizzle)`)."""
    launched = []
    real = _sfw._fw_launch_choice
    monkeypatch.setattr(_sfw, "_fw_autotune", lambda k: choice)
    monkeypatch.setattr(_sfw, "_fw_launch_choice", lambda k, c: launched.append(c) or real(k, c))
    lls = pc(data, **kw).clone()
    s, e = layer._output_ind_range
    return lls, pc.node_mars[s:e].clone(), launched


def _reference(layer, data, kw):
    """float64 node_mars of the input layer: log beta[n, x] + log p(x) - log sum_k p_k beta[n, cat_k]."""
    lp, ids = kw["categorical_evidence_logp"].double(), kw["soft_evidence_cat_ids"]
    params = layer.params.double()
    vids = layer.vids.view(-1)
    lvids = layer.var_idmapping[vids]
    base = layer.s_pids.view(-1)
    x = data.t()[vids]                                                            # [nodes, B]
    lpn, idn = lp.permute(1, 0, 2)[lvids], ids.permute(1, 0, 2)[lvids]           # [nodes, B, K]
    Z = (lpn.exp() * params[base[:, None, None] + idn]).sum(2)
    lex = torch.where(idn == x[:, :, None], lpn, torch.full_like(lpn, -float("inf"))).amax(2)
    return params[base[:, None] + x].log() + lex - Z.log()


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("K", [13, 32])
@pytest.mark.parametrize("chunk", [8, 4])
def test_row_form_matches_float64_and_the_index_driven_form(homogeneous, K, chunk, monkeypatch):
    """The row form against float64 and the index-driven form, tied emissions (one parameter block) and untied (a block
    a variable: the table's keys carry the block), candidate counts not a multiple of the kernel's unroll, and the
    latents whole or in two chunks (two tables, two Z launches)."""
    _needs_rows()
    pc, layer, data, kw = _setup(monkeypatch, num_vars = 5, num_latents = 8, num_cats = 96, K = K, B = 6,
                                 homogeneous = homogeneous)
    ref = _reference(layer, data, kw)
    lls_r, nm_r, ran = _forward(pc, layer, data, kw, ("rows", chunk), monkeypatch)
    assert ran == [("rows", chunk)]
    lls_d, nm_d, _ = _forward(pc, layer, data, kw, (4, 0), monkeypatch)
    assert torch.isfinite(nm_r).all()
    assert (nm_r.double() - ref).abs().max() < 2e-5, f"row form vs float64 {(nm_r.double() - ref).abs().max():.3e}"
    assert torch.allclose(nm_r, nm_d, rtol = 0, atol = 2e-5)
    assert torch.allclose(lls_r, lls_d, rtol = 1e-6, atol = 1e-4)


def test_row_form_serves_slots_no_shared_tile_holds(monkeypatch):
    """Past the index-driven form's shared tile (`_fw_dense_fits`) the forward used to fall back to Triton's gather; the
    row form keeps no tile, so the forward stays on the index -- and is the only candidate there."""
    _needs_rows()
    pc, layer, data, kw = _setup(monkeypatch, num_vars = 4, num_latents = 8, num_cats = 200, K = 24, B = 40,
                                 homogeneous = True)
    monkeypatch.setattr(_dense._fw_smem_cap, "_cached", 64, raising = False)        # no tile fits
    assert not _dense._fw_dense_fits(layer, kw) and _dense._fw_use_dense(layer, kw)
    launched = []
    real = _sfw._fw_launch_choice
    monkeypatch.setattr(_sfw, "_fw_launch_choice", lambda k, c: launched.append(c) or real(k, c))
    layer._fw_tune_cache = {}
    pc(data, **kw)
    s, e = layer._output_ind_range
    nm = pc.node_mars[s:e].clone()
    assert launched and all(c[0] == "rows" for c in launched), launched
    assert (nm.double() - _reference(layer, data, kw)).abs().max() < 2e-5


def test_row_form_writes_every_value_it_reads(monkeypatch):
    """`Z` (reused across steps) and the table (fresh each launch) poisoned with NaN, then with a finite value: the row
    form writes every entry it reads, and the same bits each time (its sums run in candidate order, no atomics)."""
    _needs_rows()
    pc, layer, data, kw = _setup(monkeypatch, num_vars = 5, num_latents = 12, num_cats = 120, K = 17, B = 7,
                                 homogeneous = False)
    _, clean, _ = _forward(pc, layer, data, kw, ("rows", 4), monkeypatch)

    class _PoisonedEmpty:
        def __init__(self, value):
            self.value = value

        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *size, **k):
            out = torch.empty(*size, **k)
            return out.fill_(self.value) if out.is_floating_point() else out

    for value in (float("nan"), 3.0):
        layer._fw_z_buf.fill_(value)
        with monkeypatch.context() as mp:
            mp.setattr(_sfw, "torch", _PoisonedEmpty(value))
            _, nm, _ = _forward(pc, layer, data, kw, ("rows", 4), mp)
        assert torch.equal(nm, clean), f"poisoned with {value}: max diff {(nm - clean).abs().max():.3e}"


def test_row_form_needs_latents_in_fours(monkeypatch):
    """The row form reads and writes four latents at a time: with latents not in fours it is never offered -- the
    index-driven form serves the forward where its tile fits, Triton's gather where not -- and the forward is right
    either way."""
    pc, layer, data, kw = _setup(monkeypatch, num_vars = 4, num_latents = 6, num_cats = 80, K = 12, B = 5,
                                 homogeneous = True)
    assert not _dense._fw_rows_fits(layer, kw)
    index = _dense._build_dense_index(layer, kw)
    assert _dense._fw_rows_chunks(index, layer.params.device) == []
    ref = _reference(layer, data, kw)
    for cap in (None, 64):
        with monkeypatch.context() as mp:
            if cap is not None:
                mp.setattr(_dense._fw_smem_cap, "_cached", cap, raising = False)
            launched = []
            real = _sfw._fw_launch_choice
            mp.setattr(_sfw, "_fw_launch_choice", lambda k, c: launched.append(c) or real(k, c))
            layer._fw_tune_cache = {}
            pc(data, **kw)
        s, e = layer._output_ind_range
        assert not any(c[0] == "rows" for c in launched)
        assert (cap is None) == bool(launched)
        assert (pc.node_mars[s:e].double() - ref).abs().max() < 2e-5
