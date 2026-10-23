"""
`juice.queries.conditional` against brute force: P(x_t = v | e) from one forward pass per value v of the target -- log
p(x_t = v, e), normalized over v -- by `juice.queries.marginal`, which runs no backward pass and so no flow regime.
Covers every distribution the query serves (hard and soft data, some and all targets, every missing-mask form) and
circuits with external sum parameters, whose backward runs only with log-space flows.

SoftEvidenceCategorical: a target's answer is the posterior of what the forward marginalized there -- q = p w / Z where
it is missing, p w where a value mask frees it. Brute force observes each value under the same measure: without a
value mask (observed tokens weigh q), and at a free target (observed with the mask True: p w). A missing target under a
value mask has no such forward (an observed token weighs p w there, a missing one q), so it is checked against
`dfa_exact`'s conditionals under the constraint that accepts everything, which define it so.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams, LowRankSumParams


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEV = torch.device("cuda:0")
ATOL = 1e-5             # precision = "fp32"
AUTO_ATOL = 5e-5        # the default precision (bf16 dots where the tiles allow them): measured up to 3.8e-6


@pytest.fixture(params = ["kernel", "matmul"])
def categorical_path(request, monkeypatch):
    """Categorical's conditional backward through its kernel, or through `emission_sums` (taken from MATMUL_MIN_CATS
    categories; forced here) -- and that path, not the other, ran."""
    import pyjuice.queries.distributions.categorical as qc
    monkeypatch.setattr(qc, "MATMUL_MIN_CATS", 1 << 40 if request.param == "kernel" else 0)
    calls = []
    sums = qc.emission_sums
    monkeypatch.setattr(qc, "emission_sums", lambda *a, **k: calls.append(1) or sums(*a, **k))
    yield request.param
    assert bool(calls) == (request.param == "matmul")


def _brute_force(pc, data, missing, targets, num_cats, **kwargs):
    """[B, len(targets), num_cats]: for every target variable t (missing in every row), P(x_t = v | e) normalized from
    log p(x_t = v, e) -- `data` with t set to v (a one-hot slice for soft data), t observed."""
    rows = []
    for t in targets:
        lps = []
        for v in range(num_cats):
            data_v = data.clone()
            if data.dim() == 3:
                data_v[:, t, :] = 0.0
                data_v[:, t, v] = 1.0
            else:
                data_v[:, t] = v
            missing_v = missing.clone()
            missing_v[..., t] = False
            lps.append(juice.queries.marginal(pc, data = data_v, missing_mask = missing_v, **kwargs).view(-1))
        rows.append(torch.softmax(torch.stack(lps, dim = 1), dim = 1))
    return torch.stack(rows, dim = 1)


def _check(pc, data, missing, targets, num_cats, target_vars, atol = ATOL, **kwargs):
    got = juice.queries.conditional(pc, data = data.clone(), missing_mask = missing, target_vars = target_vars, **kwargs)
    ref = _brute_force(pc, data, missing, targets, num_cats, **kwargs)
    # `got` is [B, number of target variables, num_cats], the targets in `target_vars` order (all variables if None)
    rows = targets if target_vars is None else [target_vars.index(t) for t in targets]
    assert torch.allclose(got[:, rows, :], ref, atol = atol), float((got[:, rows, :] - ref).abs().max())


def _chain(dist_fn, seq_length, num_latents, external = None, block_size = 4, precision = "fp32", tied = False):
    """An HMM-like chain over `seq_length` variables; with `external` ("blockscale" / "lowrank") every transition
    carries those external sum parameters; `tied`: every variable's emissions are the first's."""
    with juice.set_block_size(block_size):
        first = chain = inputs(0, num_node_blocks = num_latents // block_size, dist = dist_fn())
        transitions = []
        for t in range(1, seq_length):
            if tied:
                emit = first.duplicate(t, tie_params = True)
            else:
                emit = inputs(t, num_node_blocks = num_latents // block_size, dist = dist_fn())
            ext = dict()
            if external == "blockscale":
                ext = dict(external_params = BlockScaleSumParams(ch_block_size = block_size))
            elif external == "lowrank":
                ext = dict(external_params = LowRankSumParams(rank = 4))
            chain = summate(multiply(chain, emit), num_node_blocks = num_latents // block_size, **ext)
            transitions.append(chain)
        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False, precision = precision).to(DEV), transitions


def _missing(B, V, targets, seed):
    g = torch.Generator(device = DEV).manual_seed(seed)
    missing = torch.rand(B, V, device = DEV, generator = g) < 0.3
    missing[:, targets] = True
    return missing


@cuda_only
@pytest.mark.parametrize("target_vars", [[2, 5], None])
@pytest.mark.parametrize("mask_dims", [1, 2])
def test_categorical_hard_data(target_vars, mask_dims, categorical_path):
    torch.manual_seed(0)
    V, C, B = 8, 5, 12
    pc = juice.compile(juice.structures.HMM(seq_length = V, num_latents = 16, num_emits = C), verbose = False,
                       precision = "fp32").to(DEV)
    targets = [2, 5]
    data = torch.randint(0, C, (B, V), device = DEV)
    if mask_dims == 2:
        missing = _missing(B, V, targets, 1)
    else:
        missing = torch.zeros(V, dtype = torch.bool, device = DEV)
        missing[[0, 2, 5, 6]] = True
    _check(pc, data, missing, targets, C, target_vars)


@cuda_only
def test_categorical_hclt(categorical_path):
    torch.manual_seed(0)
    V, C, B = 10, 6, 16
    x = torch.randint(0, C, (256, V))
    ns = juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Categorical(num_cats = C))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, precision = "fp32").to(DEV)
    targets = [0, 4, 9]
    data = torch.randint(0, C, (B, V), device = DEV)
    _check(pc, data, _missing(B, V, targets, 2), targets, C, targets)


@cuda_only
@pytest.mark.parametrize("tied", [True, False])
def test_categorical_paths_agree_at_a_large_vocabulary(tied, monkeypatch):
    """Where brute force would take thousands of forwards: the kernel and `emission_sums` (each against brute force
    above) give the same conditionals over 8192 categories, tied emissions or not."""
    import pyjuice.queries.distributions.categorical as qc
    torch.manual_seed(0)
    V, C, B = 6, 8192, 12
    pc = juice.compile(juice.structures.HMM(seq_length = V, num_latents = 16, num_emits = C, homogeneous = tied),
                       verbose = False, precision = "fp32").to(DEV)
    targets = [1, 4]
    data = torch.randint(0, C, (B, V), device = DEV)
    missing = _missing(B, V, targets, 11)
    got = {}
    for path, threshold in (("kernel", 1 << 40), ("matmul", 0)):
        monkeypatch.setattr(qc, "MATMUL_MIN_CATS", threshold)
        got[path] = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = targets)
    assert torch.allclose(got["matmul"], got["kernel"], atol = 1e-6, rtol = 1e-4), \
        float((got["matmul"] - got["kernel"]).abs().max())


@cuda_only
def test_categorical_soft_data(categorical_path):
    torch.manual_seed(0)
    V, C, B = 6, 4, 10
    pc = juice.compile(juice.structures.HMM(seq_length = V, num_latents = 8, num_emits = C), verbose = False,
                       precision = "fp32").to(DEV)
    targets = [1, 4]
    data = torch.rand(B, V, C, device = DEV)
    data /= data.sum(-1, keepdim = True)
    _check(pc, data, _missing(B, V, targets, 3), targets, C, targets)


@cuda_only
def test_discrete_logistic():
    torch.manual_seed(0)
    V, C, B = 5, 6, 12
    pc, _ = _chain(lambda: dists.DiscreteLogistic(num_cats = C, val_range = (0.0, 1.0)), V, 8)
    targets = [1, 3]
    data = torch.randint(0, C, (B, V), device = DEV)
    _check(pc, data, _missing(B, V, targets, 4), targets, C, targets)


@cuda_only
def test_extern_product_categorical():
    """The target is free (value mask False): its tokens weigh p(v) exp(E[v]); brute force observes each v (value
    mask True there), which weighs exactly that."""
    torch.manual_seed(0)
    V, C, B = 4, 8, 12
    pc, _ = _chain(lambda: dists.ExternProductCategorical(num_cats = C), V, 8)
    t = 2
    data = torch.randint(0, C, (B, V), device = DEV)
    logps = torch.rand(B, V, C, device = DEV)
    logps = (logps / logps.sum(-1, keepdim = True)).log()
    vmask = torch.ones(B, V, dtype = torch.bool, device = DEV)
    vmask[:, t] = False
    got = juice.queries.conditional(pc, data = data, extern_product_categorical_mode = "unnormalized_ll",
                                    external_categorical_logps = logps, external_categorical_value_mask = vmask)
    lps = []
    for v in range(C):
        data_v, vmask_v = data.clone(), vmask.clone()
        data_v[:, t], vmask_v[:, t] = v, True
        lps.append(juice.queries.marginal(pc, data = data_v, extern_product_categorical_mode = "unnormalized_ll",
                                          external_categorical_logps = logps,
                                          external_categorical_value_mask = vmask_v).view(-1))
    ref = torch.softmax(torch.stack(lps, dim = 1), dim = 1)
    assert torch.allclose(got[:, t, :], ref, atol = ATOL), float((got[:, t, :] - ref).abs().max())


@cuda_only
@pytest.mark.parametrize("external, block_size, B", [
    # BlockScale's backward refuses precision = "fp32", so it runs at the default precision
    ("blockscale", 4, 8), ("blockscale", 16, 16),
    # LowRank's backward runs from block_size 16 and batch 16
    ("lowrank", 16, 16), ("lowrank", 32, 24),
])
def test_external_sum_params(external, block_size, B, categorical_path):
    torch.manual_seed(0)
    V, C = 5, 6
    precision, atol = ("auto", AUTO_ATOL) if external == "blockscale" else ("fp32", ATOL)
    pc, transitions = _chain(lambda: dists.Categorical(num_cats = C), V, 2 * block_size, external = external,
                             block_size = block_size, precision = precision)
    g = torch.Generator(device = DEV).manual_seed(5)
    params = dict()
    for ns in transitions:
        shapes = ns.external_params.tensor_shapes(ns, B)
        tensors = tuple(torch.randn(s, device = DEV, generator = g) * 0.5 - 0.5 for s in shapes)
        params[ns] = tensors if len(tensors) > 1 else tensors[0]
    targets = [1, 3]
    data = torch.randint(0, C, (B, V), device = DEV)
    _check(pc, data, _missing(B, V, targets, 6), targets, C, targets, atol = atol, sum_external_params = params)


# ---- SoftEvidenceCategorical ----

SOFT_C = 6


def _soft_pc(model):
    """(pc, number of variables): a chain with tied or untied emissions, or an HCLT."""
    torch.manual_seed(0)
    if model in ("tied", "untied"):
        return _chain(lambda: dists.SoftEvidenceCategorical(num_cats = SOFT_C), 5, 8, tied = model == "tied")[0], 5
    V = 7
    x = torch.randint(0, SOFT_C, (256, V))
    ns = juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.SoftEvidenceCategorical(num_cats = SOFT_C))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, precision = "fp32").to(DEV), V


def _soft_evidence(B, V, K = SOFT_C, topk = None, seed = 0):
    """Dense log-weights [B, V, K] with some tokens weighing nothing (every position keeps one); or `topk` distinct
    candidates per sample and position, in no order."""
    g = torch.Generator(device = DEV).manual_seed(seed)
    if topk is not None:
        ids = torch.argsort(torch.rand(B, V, SOFT_C, device = DEV, generator = g), dim = 2)[..., :topk]
        return dict(categorical_evidence_logp = torch.randn(B, V, topk, device = DEV, generator = g) * 2,
                    soft_evidence_cat_ids = ids)
    E = torch.randn(B, V, K, device = DEV, generator = g) * 2
    E[torch.rand(E.shape, device = DEV, generator = g) < 0.2] = -float("inf")
    E[..., 0] = torch.where(torch.isinf(E).all(dim = -1), 0.0, E[..., 0])
    return dict(categorical_evidence_logp = E)


def _soft_data(inputs):
    """Tokens the evidence weighs: every position's first candidate, or its heaviest token."""
    if "soft_evidence_cat_ids" in inputs:
        return inputs["soft_evidence_cat_ids"][..., 0].clone()
    return inputs["categorical_evidence_logp"].argmax(dim = 2)


def _dfa_exact_conditional(pc, data, missing, targets, **inputs):
    """[B, len(targets), SOFT_C]: `dfa_exact`'s conditionals under the constraint that accepts everything (`targets`
    ascending)."""
    import pyjuice.constraints as jc
    cc = jc.compile(jc.DFA.anything(SOFT_C), pc)
    with cc.decoder(precision = "fp32", **inputs) as decoder:
        rows = decoder.conditional(data, missing, targets = torch.tensor(targets, device = DEV))
    return rows.exp().view(data.size(0), len(targets), SOFT_C)


@cuda_only
@pytest.mark.parametrize("evidence", ["dense", "dense_short", "topk"])
@pytest.mark.parametrize("model", ["tied", "untied", "hclt"])
def test_softevi_categorical(model, evidence):
    pc, V = _soft_pc(model)
    B, targets = 8, [1, 3]
    inputs = _soft_evidence(B, V, **dict(dense = dict(), dense_short = dict(K = SOFT_C - 2), topk = dict(topk = 3))[evidence])
    data, missing = _soft_data(inputs), _missing(B, V, targets, 7)
    got = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = targets, **inputs)

    # Dense evidence over K tokens weighs the others nothing; the library reads no weight past K, so brute force stops there
    K = inputs["categorical_evidence_logp"].size(2) if evidence == "dense_short" else SOFT_C
    ref = _brute_force(pc, data, missing, targets, K, **inputs)
    assert torch.allclose(got[..., :K], ref, atol = ATOL), float((got[..., :K] - ref).abs().max())
    assert (got[..., K:] == 0).all()
    if model != "hclt":                     # dfa_exact serves PCs whose scopes are contiguous in variable order
        want = _dfa_exact_conditional(pc, data, missing, targets, **inputs)
        assert torch.allclose(got, want, atol = ATOL), float((got - want).abs().max())


@cuda_only
@pytest.mark.parametrize("evidence", ["dense", "topk"])
def test_softevi_categorical_value_mask(evidence):
    """A free target (value mask False): brute force observing each value with the mask True; a missing one (q):
    `dfa_exact`."""
    pc, V = _soft_pc("tied")
    B = 8
    inputs = _soft_evidence(B, V, **(dict(topk = 3) if evidence == "topk" else dict()))
    data = _soft_data(inputs)
    vmask = torch.ones(B, V, dtype = torch.bool, device = DEV)
    vmask[:, 2] = False
    missing = torch.zeros(B, V, dtype = torch.bool, device = DEV)
    missing[:, 4] = True
    got = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = [2, 4],
                                    soft_evidence_value_mask = vmask, **inputs)
    lps = []
    for v in range(SOFT_C):
        data_v, vmask_v = data.clone(), vmask.clone()
        data_v[:, 2], vmask_v[:, 2] = v, True
        lps.append(juice.queries.marginal(pc, data = data_v, missing_mask = missing, soft_evidence_value_mask = vmask_v,
                                          **inputs).view(-1))
    ref = torch.softmax(torch.stack(lps, dim = 1), dim = 1)
    assert torch.allclose(got[:, 0, :], ref, atol = ATOL), float((got[:, 0, :] - ref).abs().max())
    want = _dfa_exact_conditional(pc, data, missing, [2, 4], soft_evidence_value_mask = vmask, **inputs)
    assert torch.allclose(got, want, atol = ATOL), float((got - want).abs().max())


@cuda_only
def test_softevi_categorical_gemm_normalizers(monkeypatch):
    """`log Z` through the dense forward's GEMM -- taken from 32768 categories, forced here -- against brute force,
    whose forwards (no value mask) never take it."""
    import pyjuice.nodes.distributions.softevi.forward as fw
    monkeypatch.setattr(fw, "_GEMM_MIN_CATS", 1)
    calls = []
    gemm = fw._fw_dense_gemm
    monkeypatch.setattr(fw, "_fw_dense_gemm", lambda *a, **k: calls.append(1) or gemm(*a, **k))
    pc, V = _soft_pc("tied")
    B, targets = 8, [0, 2, 4]
    inputs = _soft_evidence(B, V)
    data, missing = _soft_data(inputs), _missing(B, V, targets, 8)
    got = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = targets, **inputs)
    assert calls, "the GEMM forward did not run"
    ref = _brute_force(pc, data, missing, targets, SOFT_C, **inputs)
    assert torch.allclose(got, ref, atol = ATOL), float((got - ref).abs().max())


@cuda_only
def test_softevi_categorical_shared_evidence_and_all_targets():
    """Evidence every sample shares (expanded, batch stride 0) gives what its copy gives; `target_vars = None`
    (every variable) what the targets alone give."""
    pc, V = _soft_pc("untied")
    B, targets = 8, [1, 3]
    inputs = _soft_evidence(1, V)
    E = inputs["categorical_evidence_logp"]
    data, missing = _soft_data(dict(categorical_evidence_logp = E.expand(B, -1, -1))), _missing(B, V, targets, 9)
    got = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = targets,
                                    categorical_evidence_logp = E.expand(B, -1, -1))
    ref = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = targets,
                                    categorical_evidence_logp = E.expand(B, -1, -1).contiguous())
    assert torch.allclose(got, ref, atol = 1e-6)
    every = juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = None,
                                      categorical_evidence_logp = E.expand(B, -1, -1).contiguous())
    assert torch.allclose(every[:, targets, :], ref, atol = 1e-6)


@cuda_only
def test_softevi_categorical_marginal_and_refusals():
    """`marginal` is the circuit's own forward; a conditional without evidence, or asked for its gradient, is refused."""
    pc, V = _soft_pc("tied")
    B = 8
    inputs = _soft_evidence(B, V, topk = 3)
    data, missing = _soft_data(inputs), _missing(B, V, [1], 10)
    assert torch.equal(juice.queries.marginal(pc, data = data, missing_mask = missing, **inputs),
                       pc(data, missing_mask = missing, **inputs))
    with pytest.raises(ValueError, match = "categorical_evidence_logp"):
        juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = [1])
    with pytest.raises(ValueError, match = "categorical_evidence_logp_grad"):
        juice.queries.conditional(pc, data = data, missing_mask = missing, target_vars = [1],
                                  categorical_evidence_logp_grad = torch.zeros_like(inputs["categorical_evidence_logp"]),
                                  **inputs)


@cuda_only
@pytest.mark.parametrize("layout", ["shared", "shared_scattered", "private", "mixed", "unaligned", "absent_target",
                                    "chunked"])
def test_emission_sums_against_a_loop(layout, monkeypatch):
    """`emission_sums` -- both of its forms, the table viewed or gathered, chunked or not -- against the sum it
    stands for, node by node."""
    import pyjuice.queries.distributions.common as common
    g = torch.Generator(device = DEV).manual_seed(12)
    C, B, T, H = 7, 5, 4, 6
    offset = 3 if layout == "unaligned" else 0                     # parameters that do not start a row of the table
    if layout == "shared":
        rows = torch.arange(H, device = DEV).repeat(T)             # every target reads the same H rows
    elif layout == "shared_scattered":
        rows = (2 * torch.arange(H, device = DEV)).repeat(T)       # ... which are not a contiguous run
    elif layout in ("private", "chunked", "unaligned", "absent_target"):
        rows = torch.arange(T * H, device = DEV)                   # every target its own rows
    else:
        rows = torch.cat([torch.arange(H, device = DEV).repeat(2), torch.arange(H, 3 * H, device = DEV)])
    rows_of = torch.arange(T, device = DEV).repeat_interleave(H)
    if layout == "absent_target":
        keep = rows_of != 2                                        # target 2 has no node
        rows, rows_of = rows[keep], rows_of[keep]
    if layout == "chunked":
        monkeypatch.setattr(common, "_SUMS_CHUNK", H * C)          # one target per chunk
    perm = torch.randperm(rows.numel(), device = DEV, generator = g)            # nodes in no particular order
    rows, rows_of = rows[perm], rows_of[perm]
    n_rows = int(rows.max()) + 1
    params = torch.rand(offset + n_rows * C, device = DEV, generator = g)
    s_pids = offset + rows * C
    weights = torch.rand(rows.numel(), B, device = DEV, generator = g)

    got = common.emission_sums(params, s_pids, rows_of, weights, T, C)
    want = torch.zeros(B, T, C, device = DEV)
    for n in range(rows.numel()):
        want[:, rows_of[n], :] += weights[n][:, None] * params[s_pids[n]:s_pids[n] + C][None, :]
    assert torch.allclose(got, want, atol = 1e-5), float((got - want).abs().max())
