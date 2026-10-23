"""
`juice.queries.conditional` against brute force: P(x_t = v | e) from one forward pass per value v of the target -- log
p(x_t = v, e), normalized over v -- by `juice.queries.marginal`, which runs no backward pass and so no flow regime.
Covers every distribution the query serves (hard and soft data, some and all targets, every missing-mask form) and
circuits with external sum parameters, whose backward runs only with log-space flows.
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


def _chain(dist_fn, seq_length, num_latents, external = None, block_size = 4, precision = "fp32"):
    """An HMM-like chain over `seq_length` variables; with `external` ("blockscale" / "lowrank") every transition
    carries those external sum parameters."""
    with juice.set_block_size(block_size):
        chain = inputs(0, num_node_blocks = num_latents // block_size, dist = dist_fn())
        transitions = []
        for t in range(1, seq_length):
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
def test_categorical_hard_data(target_vars, mask_dims):
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
def test_categorical_hclt():
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
def test_categorical_soft_data():
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
def test_external_sum_params(external, block_size, B):
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
