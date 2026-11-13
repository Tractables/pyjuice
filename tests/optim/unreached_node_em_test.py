"""
A node no flow reached, at pseudocount 0, has nothing for the M-step to update it from: its flows sum to 0, and the
update divides 0 by 0. It keeps its parameters -- in every input distribution that has an update, here, and in the
sum layers (`tests/model/em_node_blocks_test.py`, every form; the conditional dual-flow M-step in
`tests/model/parameterizations/denom_param_flows_test.py`). Before, the update gave NaN, which the next forward
spread through the circuit (Categorical's EM, full-batch, with an input node whose flow had underflowed to 0).

Each distribution's update from flows where some nodes have none: those nodes' parameters the same bits as before
(the dual-flow SoftEvi update renormalizes: to fp32's rounding), and every other node's the same bits as the update
with those nodes' flows in place (a node's update is its own); and a whole circuit, a sub-circuit of which the root's
parameters do not reach, through `mini_batch_em`.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import InputLayer
from pyjuice.nodes import inputs, multiply, summate

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
DEV = torch.device("cuda:0")
N = 11                                         # nodes: a full tile of eight and a partial one
DEAD = [0, 4, 10]                              # the nodes no flow reaches


def _masked(mode, num_cats):
    if mode == "full_mask":
        mask = (torch.arange(num_cats)[None, :] + torch.arange(N)[:, None]) % 3 != 0
        return dists.MaskedCategorical(num_cats = num_cats, mask_mode = mode), mask.long()
    lo = torch.arange(N) % 2
    return dists.MaskedCategorical(num_cats = num_cats, mask_mode = mode), torch.stack([lo, lo + 3], dim = 1)


# (name, the distribution and the `inputs` keywords, the layer's own kernel or not, `keep_zero_params` read)
CASES = [
    ("categorical", lambda: (dists.Categorical(num_cats = 6), {}), True, True),
    ("categorical_template", lambda: (dists.Categorical(num_cats = 6), {}), False, True),
    ("categorical_300", lambda: (dists.Categorical(num_cats = 300), {}), True, True),
    ("extern_product", lambda: (dists.ExternProductCategorical(num_cats = 6), {}), True, True),
    ("extern_product_300", lambda: (dists.ExternProductCategorical(num_cats = 300), {}), True, True),
    ("soft_evidence", lambda: (dists.SoftEvidenceCategorical(num_cats = 6, _dual_flow_backward = False), {}),
     True, True),
    ("soft_evidence_300", lambda: (dists.SoftEvidenceCategorical(num_cats = 300, _dual_flow_backward = False), {}),
     True, True),
    ("soft_evidence_dual", lambda: (dists.SoftEvidenceCategorical(num_cats = 6, _dual_flow_backward = True), {}),
     True, True),
    ("soft_evidence_dual_300",
     lambda: (dists.SoftEvidenceCategorical(num_cats = 300, _dual_flow_backward = True), {}), True, True),
    ("masked_range", lambda: (lambda d, m: (d, {"mask": m}))(*_masked("range", 6)), True, False),
    ("masked_full_mask", lambda: (lambda d, m: (d, {"mask": m}))(*_masked("full_mask", 6)), True, False),
    ("masked_rev_range", lambda: (lambda d, m: (d, {"mask": m}))(*_masked("rev_range", 6)), True, False),
    ("bernoulli", lambda: (dists.Bernoulli(), {}), True, False),
]


def _layer(make):
    dist, kwargs = make()
    torch.manual_seed(0)
    nodes = inputs(0, num_nodes = N, block_size = 1, dist = dist, **kwargs)
    layer = InputLayer([nodes], pc_num_vars = 1)
    layer._init_parameters(perturbation = 2.0)
    layer.to(DEV)
    layer.init_param_flows(flows_memory = 0.0)
    return layer, dist


def _update(layer, params, flows, step_size, keep_zero_params, own_kernel, monkeypatch):
    with torch.no_grad():
        layer.params.copy_(params)
        layer.param_flows.copy_(flows)
    if not own_kernel:
        monkeypatch.setattr(layer, "em_layer_fn", None)
    layer.mini_batch_em(step_size = step_size, pseudocount = 0.0, keep_zero_params = keep_zero_params)
    torch.cuda.synchronize()
    monkeypatch.undo()
    return layer.params.detach().clone()


@cuda_only
@pytest.mark.parametrize("step_size", [1.0, 0.25])
@pytest.mark.parametrize("keep_zero_params", [False, True])
@pytest.mark.parametrize("name,make,own_kernel,reads_keep_zero", CASES, ids = [c[0] for c in CASES])
def test_an_input_node_no_flow_reached_keeps_its_parameters(name, make, own_kernel, reads_keep_zero,
                                                            keep_zero_params, step_size, monkeypatch):
    if keep_zero_params and not reads_keep_zero:
        pytest.skip("the distribution's update has no `keep_zero_params`")
    layer, dist = _layer(make)
    P, F = dist.num_parameters(), dist.num_param_flows()
    dual = name.startswith("soft_evidence_dual")
    K = getattr(dist, "num_cats", 2)
    params = layer.params.detach().clone()
    if keep_zero_params:                                   # some parameters exactly 0, in the nodes no flow reaches too
        p = params.view(N, P)
        p[:, 1] = 0.0
        p[:, :K] /= p[:, :K].sum(dim = 1, keepdim = True)
    g = torch.Generator(device = DEV).manual_seed(1)
    flows = (torch.rand(N * F, generator = g, device = DEV) + 0.1).view(N, F)
    if dual:
        flows[:, K:] += 1.0                                # (F- above F+: a dual flow's)
    unreached = flows.clone()
    unreached[DEAD] = 0.0
    dead = torch.zeros(N, P, dtype = torch.bool, device = DEV)
    dead[DEAD] = True
    dead = dead.view(-1)

    got = _update(layer, params, unreached.view(-1), step_size, keep_zero_params, own_kernel, monkeypatch)
    full = _update(layer, params, flows.view(-1), step_size, keep_zero_params, own_kernel, monkeypatch)
    assert torch.isfinite(got).all(), f"{name}: the update of a node no flow reached is not finite"
    if dual:                                               # (renormalized: `theta / sum(theta)`, an fp32 sum)
        assert torch.allclose(got[dead], params[dead], rtol = 2e-5, atol = 1e-29), \
            float((got[dead] - params[dead]).abs().max())
    else:
        assert torch.equal(got[dead], params[dead]), f"{name}: a node no flow reached moved"
    assert not torch.equal(got[~dead], params[~dead]), f"{name}: nothing was updated"
    assert torch.equal(got[~dead], full[~dead]), f"{name}: the other nodes' update changed"


@cuda_only
@pytest.mark.parametrize("num_cats", [6, 300])
@pytest.mark.parametrize("step_size", [1.0, 0.25])
def test_a_category_no_flow_reached_keeps_its_dual_flow_ratio(num_cats, step_size):
    """SoftEvi's dual-flow update multiplies a category's parameter by `F+ / F-`: where both are 0 (no flow reached
    the category) the ratio is 1, the category's parameter kept up to the node's renormalization -- before, the 0 / 0
    made the node's normalizer NaN, and every category of the node 1e-30 (the update's floor)."""
    dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)
    layer, _ = _layer(lambda: (dist, {}))
    params = layer.params.detach().clone()
    g = torch.Generator(device = DEV).manual_seed(2)
    fp = torch.rand(N, num_cats, generator = g, device = DEV) + 0.1
    fm = fp + torch.rand(N, num_cats, generator = g, device = DEV) + 0.5
    fp[:, 2], fm[:, 2] = 0.0, 0.0
    with torch.no_grad():
        layer.param_flows.copy_(torch.cat([fp, fm], dim = 1).view(-1))
    layer.mini_batch_em(step_size = step_size, pseudocount = 0.0)
    got = layer.params.detach().view(N, num_cats).double()
    t, p, m = params.view(N, num_cats).double(), fp.double(), fm.double()
    ratio = torch.where((p == 0) & (m == 0), torch.ones_like(p), p / m)
    cum = (1 - step_size) + step_size * (t * ratio).sum(dim = 1, keepdim = True)
    want = (t * ((1 - step_size) + step_size * ratio) / cum).clamp_min(1e-30)
    assert torch.isfinite(got).all()
    rel = float(((got - want).abs() / want).max())
    assert rel < 1e-5, f"the update is {rel:.2e} from the float64 reference"


def _two_decompositions(block_size):
    """A root over two decompositions of the same four variables, sharing nothing: the root's parameters to the
    second set to 0, nothing reaches it."""
    with juice.set_block_size(block_size):
        x = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        y = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        a = multiply(summate(multiply(x[0], x[1]), num_node_blocks = 2),
                     summate(multiply(x[2], x[3]), num_node_blocks = 2))
        b = multiply(summate(multiply(y[0], y[2]), num_node_blocks = 2),
                     summate(multiply(y[1], y[3]), num_node_blocks = 2))
        root = summate(a, b, num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, a, b, x, y


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids = ["fp32", "bf16"])
@pytest.mark.parametrize("block_size", [4, 32])
@pytest.mark.parametrize("step_size", [1.0, 0.25])
def test_a_sub_circuit_no_flow_reaches_keeps_its_parameters(dtype, block_size, step_size):
    """Through `mini_batch_em`: the sum and input nodes under the root's children it gives 0 keep their parameters
    (the same bits); everything finite; the rest updated."""
    torch.manual_seed(0)
    root, a, b, x, y = _two_decompositions(block_size)
    pc = juice.compile(root, verbose = False, param_dtype = dtype).to(DEV)
    rs, re = root._param_range
    child = (root.edge_ids[1].to(DEV)[:, None] * root.ch_block_size
             + torch.arange(root.ch_block_size, device = DEV)[None, :]).reshape(-1)
    with torch.no_grad():
        rp = pc.params.data[rs:re].float()
        rp[child >= a.num_nodes] = 0.0                     # (the root's children: `a`'s nodes, then `b`'s)
        pc.params.data[rs:re] = (rp / rp.sum()).to(dtype)
    data = torch.randint(0, 5, (256, 4), device = DEV)
    pc.init_param_flows(flows_memory = 0.0)
    pc(data)
    pc.backward(data, flows_memory = 1.0)

    sums_b = list(b.chs)                                   # (the second decomposition's two sums)
    inner, ins = pc.params.detach().clone(), [l.params.detach().clone() for l in pc.input_layer_group]
    pc.mini_batch_em(step_size = step_size, pseudocount = 0.0)
    assert torch.isfinite(pc.params).all() and all(torch.isfinite(l.params).all() for l in pc.input_layer_group)
    for ns in sums_b:
        lo, hi = ns._param_range
        assert torch.equal(pc.params.detach()[lo:hi], inner[lo:hi]), "a sum node no flow reached moved"
    for ns in y:
        layer = next(l for l in pc.input_layer_group if any(n is ns for n in l.nodes))
        lo, hi = ns._param_range
        before = ins[list(pc.input_layer_group).index(layer)]
        assert torch.equal(layer.params.detach()[lo:hi], before[lo:hi]), "an input node no flow reached moved"
    for ns in a.chs:
        lo, hi = ns._param_range
        assert not torch.equal(pc.params.detach()[lo:hi], inner[lo:hi])
    for ns in x:
        layer = next(l for l in pc.input_layer_group if any(n is ns for n in l.nodes))
        lo, hi = ns._param_range
        before = ins[list(pc.input_layer_group).index(layer)]
        assert not torch.equal(layer.params.detach()[lo:hi], before[lo:hi])
