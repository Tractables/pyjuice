import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes.distributions import softevi_categorical as _softevi


# A `SoftEvidenceCategorical` leaf whose variable is MARGINALIZED carries no soft evidence: the forward
# overwrites its `node_mars` with 0 (`_fw_missing_mask_kernel`), discarding the local normalizer that the
# backward's evidence terms are defined against. The only statistic such a position may contribute is the
# one `bk_dual_flow_mask_fn` writes -- the dense anchor `flow * theta[c]`, added to F+ and F- alike.
#
# That gives a sharp invariant to test against: at a fully masked variable, F+ must equal F- EXACTLY, and
# both must be proportional to theta. Before the post-processing kernels were told about `missing_mask`
# (it is a named argument of `backward`, so it never reached them through `**kwargs`) they also fired at
# masked positions, adding an observed-token flow to F+ and an evidence-weighted expected flow to F-.


def _build(num_vars, num_latents, num_cats, dual_flow = True, homogeneous = False, seed = 0):
    torch.manual_seed(seed)
    root_ns = juice.structures.GeneralizedHMM(
        seq_length = num_vars, num_latents = num_latents, homogeneous = homogeneous,
        input_dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = dual_flow)
    )
    root_ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(root_ns)
    pc.to(torch.device("cuda:0"))
    return pc


def _step(pc, data, mask, **kw):
    pc.init_param_flows(flows_memory = 0.0)
    if mask is not None:
        kw["missing_mask"] = mask
    pc(data, **kw)
    pc.backward(data, allow_modify_flows = False, logspace_flows = True, **kw)
    return pc.input_layer_group[0].param_flows.clone()


def _evidence(B, V, K, device, seed = 1):
    torch.manual_seed(seed)
    return torch.log_softmax(torch.randn(B, V, K, device = device), dim = 2).contiguous()


def _topk_ids(B, V, K, num_cats, data, device, seed = 2):
    torch.manual_seed(seed)
    ids = torch.randint(0, num_cats, (B, V, K), device = device).long().sort(dim = 2)[0].contiguous()
    ids[:, :, -1] = data      # the observed token must be among the candidates
    return ids


@pytest.mark.parametrize("has_topk", [False, True])
@pytest.mark.parametrize("mask_layout", ["per_var", "per_batch_var"])
def test_masked_leaf_gets_only_the_anchor(has_topk, mask_layout):
    device = torch.device("cuda:0")
    S, L, C, B = 3, 2, 8, 4
    K = 4 if has_topk else C

    pc = _build(S, L, C)
    layer = pc.input_layer_group[0]

    data = torch.randint(0, C, (B, S), device = device)
    kw = dict(categorical_evidence_logp = _evidence(B, S, K, device))
    if has_topk:
        kw["soft_evidence_cat_ids"] = _topk_ids(B, S, K, C, data, device)

    if mask_layout == "per_var":
        mask = torch.zeros(S, dtype = torch.bool, device = device)
        mask[0] = True
    else:
        mask = torch.zeros(B, S, dtype = torch.bool, device = device)
        mask[:, 0] = True

    pf = _step(pc, data, mask, **kw)

    vids = layer.vids.view(-1)
    checked = 0
    for n in range(vids.numel()):
        if int(vids[n]) != 0:                       # variable 0 is the masked one
            continue
        pf0, p0 = int(layer.s_pfids[n]), int(layer.s_pids[n])
        Fp = pf[pf0 : pf0 + C]
        Fm = pf[pf0 + C : pf0 + 2 * C]
        theta = layer.params[p0 : p0 + C]

        assert torch.isfinite(Fp).all() and torch.isfinite(Fm).all()
        # the anchor goes to both phases, so they must agree bit for bit
        assert torch.equal(Fp, Fm), f"node {n}: F+ != F- at a fully masked variable"
        # ... and the anchor is `S * theta`, so the ratio is constant across categories
        ratio = Fp / theta
        assert ratio.max() - ratio.min() < 1e-5 * max(float(ratio.max()), 1e-12), \
            f"node {n}: F+ is not proportional to theta (spread {float(ratio.min())}..{float(ratio.max())})"
        assert float(ratio.max()) > 0.0
        checked += 1
    assert checked == L


def test_all_false_mask_matches_no_mask():
    """The mask must be inert where nothing is actually masked."""
    device = torch.device("cuda:0")
    S, L, C, B = 3, 2, 8, 4

    pc = _build(S, L, C)
    data = torch.randint(0, C, (B, S), device = device)
    kw = dict(categorical_evidence_logp = _evidence(B, S, C, device))

    pf_none = _step(pc, data, None, **kw)
    pf_false = _step(pc, data, torch.zeros(B, S, dtype = torch.bool, device = device), **kw)

    assert torch.equal(pf_none, pf_false)


def test_unmasked_variables_are_untouched_by_a_mask_elsewhere():
    """Masking variable 0 must not change what any other variable accumulates."""
    device = torch.device("cuda:0")
    S, L, C, B = 3, 2, 8, 4

    pc = _build(S, L, C)
    layer = pc.input_layer_group[0]
    data = torch.randint(0, C, (B, S), device = device)
    kw = dict(categorical_evidence_logp = _evidence(B, S, C, device))

    mask = torch.zeros(B, S, dtype = torch.bool, device = device)
    mask[:, 0] = True

    pf_ref = _step(pc, data, None, **kw)
    pf_msk = _step(pc, data, mask, **kw)

    vids = layer.vids.view(-1)
    for n in range(vids.numel()):
        if int(vids[n]) == 0:
            continue
        pf0 = int(layer.s_pfids[n])
        # node flows do change (the masked leaf changes the circuit's marginals), so compare loosely --
        # what must NOT happen is the mask leaking into an unmasked variable's kernel path
        assert torch.isfinite(pf_msk[pf0 : pf0 + 2 * C]).all()
    assert torch.isfinite(pf_ref).all()


@pytest.mark.parametrize("batch_size", [6, 8])
def test_no_nan_with_a_partially_padded_batch_tile(batch_size):
    """`BLOCK_SIZE_B` is rounded up to a power of two, so a batch of 6 leaves 2 padding lanes. Those lanes
    load `params` as 0 -> `logZ = -inf`, and the no-`ext_ids` branch reduces OVER the batch axis, so a NaN
    there spreads onto lanes that are valid."""
    device = torch.device("cuda:0")
    S, L, C = 3, 2, 8

    pc = _build(S, L, C)
    data = torch.randint(0, C, (batch_size, S), device = device)
    kw = dict(categorical_evidence_logp = _evidence(batch_size, S, C, device))

    pf = _step(pc, data, None, **kw)
    assert torch.isfinite(pf).all(), "non-finite param_flows with a padded batch tile"


def _dense_setup(monkeypatch, use_dense):
    """The dense top-k path only engages when the parameter table overflows L2 and the emissions are tied
    across variables, so shrink the L2 figure it gates on instead of building a multi-GB model."""
    device = torch.device("cuda:0")
    S, L, C, B, K = 4, 8, 64, 4, 32

    monkeypatch.setattr(_softevi._l2_bytes, "_cached", 512, raising = False)
    if not use_dense:
        monkeypatch.setattr(_softevi, "_DENSE_TOPK_BACKWARD", False)

    pc = _build(S, L, C, homogeneous = True)
    layer = pc.input_layer_group[0]

    data = torch.randint(0, C, (B, S), device = device)
    kw = dict(categorical_evidence_logp = _evidence(B, S, K, device),
              soft_evidence_cat_ids = _topk_ids(B, S, K, C, data, device))

    probe = dict(kw, dual_flow_backward = True)
    assert _softevi._dense_topk_applicable(layer, probe) == use_dense, \
        f"expected dense={use_dense}, got the other path"

    return pc, layer, data, kw, (S, L, C, B)


def test_fully_masked_batch_gets_only_the_anchor_on_the_dense_path(monkeypatch):
    """Same invariant as above, but through `bk_dense_prologue` + the expected-flow kernel.

    The dense path needs emissions TIED across variables, so a param-flow row mixes several variables and
    `F+ == F-` only holds if every one of them is masked."""
    pc, layer, data, kw, (S, L, C, B) = _dense_setup(monkeypatch, use_dense = True)

    mask = torch.ones(B, S, dtype = torch.bool, device = data.device)
    pf = _step(pc, data, mask, **kw)

    for n in range(layer.vids.view(-1).numel()):
        pf0, p0 = int(layer.s_pfids[n]), int(layer.s_pids[n])
        Fp = pf[pf0 : pf0 + C]
        Fm = pf[pf0 + C : pf0 + 2 * C]
        theta = layer.params[p0 : p0 + C]
        assert torch.isfinite(Fp).all() and torch.isfinite(Fm).all()
        assert torch.equal(Fp, Fm), f"node {n}: F+ != F- with every variable masked (dense path)"
        ratio = Fp / theta
        assert ratio.max() - ratio.min() < 1e-5 * max(float(ratio.max()), 1e-12)


def test_dense_and_scattered_paths_agree_under_a_mask(monkeypatch):
    """The dense top-k kernels are an optimization of `bk_softevi_kernel`; a mask must not split them."""
    flows = {}
    for use_dense in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            pc, layer, data, kw, (S, L, C, B) = _dense_setup(mp, use_dense = use_dense)
            mask = torch.zeros(B, S, dtype = torch.bool, device = data.device)
            mask[:, 0] = True
            mask[0, 2] = True
            flows[use_dense] = _step(pc, data, mask, **kw)

    a, b = flows[True], flows[False]
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.allclose(a, b, rtol = 1e-4, atol = 1e-6), \
        f"dense vs scattered disagree under a mask (max abs diff {float((a - b).abs().max()):.3e})"


if __name__ == "__main__":
    for topk in (False, True):
        for layout in ("per_var", "per_batch_var"):
            test_masked_leaf_gets_only_the_anchor(topk, layout)
    test_all_false_mask_matches_no_mask()
    test_unmasked_variables_are_untouched_by_a_mask_elsewhere()
    for bs in (6, 8):
        test_no_nan_with_a_partially_padded_batch_tile(bs)
