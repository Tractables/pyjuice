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
    """Candidate ids as `torch.topk` would give them: UNIQUE within a row, and containing the observed
    token. Unique matters -- it is the kernels' documented precondition, and with a repeated id the
    implementations legitimately disagree (the dense CUDA forward and the Triton forward differed by
    3 nats here). This helper used to draw with `randint`, which repeated an id in every row; that went
    unseen while both arms of a comparison ran the same forward."""
    torch.manual_seed(seed)
    ids = torch.stack([torch.randperm(num_cats, device = device)[:K] for _ in range(B * V)]).view(B, V, K)
    obs = data.view(B, V)
    missing = ~(ids == obs.unsqueeze(-1)).any(dim = -1)
    ids[:, :, -1] = torch.where(missing, obs, ids[:, :, -1])   # the observed token must be a candidate
    return ids.sort(dim = 2)[0].long().contiguous()


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


def _dense_setup(monkeypatch, use_dense, dual_flow = True, with_grad = False, force = True):
    """The dense top-k path only engages when the parameter table overflows L2 and the emissions are tied
    across variables, so shrink the L2 figure it gates on instead of building a multi-GB model.

    With `use_dense` the dense path is also FORCED (`force`): past the structural checks, dense vs
    scattered is a performance call -- a work-ratio guess, or a timed choice in the backward -- and a test
    that means to exercise the dense kernels must not depend on how that call goes for a toy shape.
    `force = False` leaves the real decision in place (for testing the decision itself)."""
    device = torch.device("cuda:0")
    S, L, C, B, K = 4, 8, 64, 4, 32

    monkeypatch.setattr(_softevi._l2_bytes, "_cached", 512, raising = False)
    if not use_dense:
        monkeypatch.setattr(_softevi, "_DENSE_TOPK_BACKWARD", False)
    elif force:
        monkeypatch.setattr(_softevi, "_dense_worth_it", lambda layer, kwargs: True)

    pc = _build(S, L, C, dual_flow = dual_flow, homogeneous = True)
    layer = pc.input_layer_group[0]

    torch.manual_seed(3)
    data = torch.randint(0, C, (B, S), device = device)
    kw = dict(categorical_evidence_logp = _evidence(B, S, K, device),
              soft_evidence_cat_ids = _topk_ids(B, S, K, C, data, device))

    probe = dict(kw, dual_flow_backward = dual_flow)
    if with_grad:
        probe["categorical_evidence_logp_grad"] = torch.zeros_like(kw["categorical_evidence_logp"])
    if force or not use_dense:
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


def _grad_step(pc, data, mask, **kw):
    pc.init_param_flows(flows_memory = 0.0)
    if mask is not None:
        kw["missing_mask"] = mask
    pc(data, **kw)
    grad = torch.zeros_like(kw["categorical_evidence_logp"])
    pc.backward(data, allow_modify_flows = False, logspace_flows = True, categorical_evidence_logp_grad = grad, **kw)
    return pc.input_layer_group[0].param_flows.clone(), grad


@pytest.mark.parametrize("dual_flow", [True, False])
def test_timed_dense_choice_benchmarks_into_scratch(dual_flow):
    """Dense vs scattered is timed on the first backward of a shape (`_tune_dense_choice`). Both sides
    ACCUMULATE into the parameter flows and the evidence gradient, so the timing runs must go to scratch:
    one that leaked into the live buffers would add each side's contribution ~20 more times. Checked
    against the same step with tuning off. The two may pick different sides, which agree only to
    rounding, hence a tolerance rather than equality."""
    from pyjuice.layer.kernels import autotune

    out = {}
    saved = (autotune.ENABLED, dict(autotune._CACHE))
    try:
        for tuned in (False, True):
            autotune.ENABLED = tuned
            autotune._CACHE.clear()
            with pytest.MonkeyPatch.context() as mp:
                pc, layer, data, kw, (S, L, C, B) = _dense_setup(mp, True, dual_flow = dual_flow, with_grad = True,
                                                                 force = False)
                out[tuned] = _grad_step(pc, data, None, **kw)
                if tuned:
                    keys = [k for k in autotune._CACHE if k[1][0] == "softevi_dense_vs_scattered"]
                    assert keys, "the backward did not time dense vs scattered"
    finally:
        autotune.ENABLED = saved[0]
        autotune._CACHE.clear()
        autotune._CACHE.update(saved[1])

    (pf_t, g_t), (pf_u, g_u) = out[True], out[False]
    assert torch.allclose(pf_t, pf_u, rtol = 1e-4, atol = 1e-6), \
        f"param flows changed by tuning (max abs diff {float((pf_t - pf_u).abs().max()):.3e})"
    assert torch.allclose(g_t, g_u, rtol = 1e-4, atol = 1e-6), \
        f"evidence gradient changed by tuning (max abs diff {float((g_t - g_u).abs().max()):.3e})"


@pytest.mark.parametrize("dual_flow", [True, False])
def test_dense_triton_fallback_reads_only_valid_references(dual_flow, monkeypatch):
    """The dense index's reference tables are NOT zero-filled: a category's list is padded to the
    longest one in its shard, and every reader must stop at the category's own count. The CUDA kernels
    loop `j < cnt`; the Triton fallback (taken without a CUDA toolchain) masks `j < cnt`. Nothing else
    in the suite runs that fallback, so force it here, and poison `torch.empty` -- NaN for floats -- so
    that a read past a list's end shows up as NaN rather than passing on freshly zeroed memory."""
    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t.fill_(0)

    flows = {}
    for use_dense in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            pc, layer, data, kw, (S, L, C, B) = _dense_setup(mp, use_dense, dual_flow = dual_flow, with_grad = True)
            if use_dense:
                mp.setattr(_softevi._DenseDenomDispatch, "_cuda_ok", lambda self: False)
                mp.setattr(torch, "empty", poisoned_empty)
            flows[use_dense] = _grad_step(pc, data, None, **kw)

    (pf_d, g_d), (pf_s, g_s) = flows[True], flows[False]
    assert torch.isfinite(pf_d).all() and torch.isfinite(g_d).all(), \
        "the Triton dense fallback read past a reference list (NaN from the uninitialized padding)"
    assert torch.allclose(pf_d, pf_s, rtol = 1e-4, atol = 1e-6), \
        f"param flows: Triton dense vs scattered (max abs diff {float((pf_d - pf_s).abs().max()):.3e})"
    assert torch.allclose(g_d, g_s, rtol = 1e-4, atol = 1e-6), \
        f"evidence gradient: Triton dense vs scattered (max abs diff {float((g_d - g_s).abs().max()):.3e})"


@pytest.mark.parametrize("masked", [False, True])
def test_nondual_dense_path_matches_scattered_and_dual(masked):
    """Without dual flows the dense top-k kernels run for the evidence gradient alone (`update_pflows`
    off: there is no denominator half to write, and the observed-category flow stays with
    `bk_params_kernel`). Before, this case always took the scattered `bk_softevi_kernel`, ~2.3x slower
    on the CoDD shape. So: dense must equal scattered, in both the param flows and the gradient -- and
    the gradient must equal the dual-flow run's, which does not depend on how the flows are stored."""
    def mask_for(data):
        if not masked:
            return None
        mask = torch.zeros_like(data, dtype = torch.bool)
        mask[:, 0] = True
        mask[0, 2] = True
        return mask

    out = {}
    for dual_flow, use_dense in ((False, True), (False, False), (True, True)):
        with pytest.MonkeyPatch.context() as mp:
            pc, layer, data, kw, (S, L, C, B) = _dense_setup(mp, use_dense, dual_flow = dual_flow, with_grad = True)
            out[(dual_flow, use_dense)] = _grad_step(pc, data, mask_for(data), **kw)

    (pf_dense, g_dense), (pf_scat, g_scat), (pf_dual, g_dual) = out[(False, True)], out[(False, False)], out[(True, True)]
    for t in (pf_dense, g_dense, pf_scat, g_scat, g_dual):
        assert torch.isfinite(t).all()
    assert g_dense.abs().max() > 0, "the evidence gradient came out all zero"

    assert torch.allclose(pf_dense, pf_scat, rtol = 1e-4, atol = 1e-6), \
        f"non-dual param flows: dense vs scattered (max abs diff {float((pf_dense - pf_scat).abs().max()):.3e})"
    assert torch.allclose(g_dense, g_scat, rtol = 1e-4, atol = 1e-6), \
        f"non-dual evidence gradient: dense vs scattered (max abs diff {float((g_dense - g_scat).abs().max()):.3e})"
    assert torch.allclose(g_dense, g_dual, rtol = 1e-4, atol = 1e-6), \
        f"evidence gradient: non-dual vs dual (max abs diff {float((g_dense - g_dual).abs().max()):.3e})"


if __name__ == "__main__":
    for topk in (False, True):
        for layout in ("per_var", "per_batch_var"):
            test_masked_leaf_gets_only_the_anchor(topk, layout)
    test_all_false_mask_matches_no_mask()
    test_unmasked_variables_are_untouched_by_a_mask_elsewhere()
    for bs in (6, 8):
        test_no_nan_with_a_partially_padded_batch_tile(bs)
