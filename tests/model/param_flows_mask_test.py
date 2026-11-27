"""`TensorCircuit.backward(param_flows_mask = m)`: the parameter flows (and `_cum_flow`, the EM step's normalizer) of
the samples `m` holds -- those of the backward whose weights are 0 at the others (the same batch, the same kernels), and
of a backward of that sub-batch up to the kernels a batch size takes --, while the node flows stay every sample's --
those of the backward without a mask, to the bit. In log-space and linear flows (the latter also modified in place),
with weights and a missing mask (the input layers' marginalized flows); refused where a layer's parameterization does
not take one, and by a plain leaf asked for the evidence's gradients, which its kernels make from the same flows."""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.parameterizations import LowRank

DEV = torch.device("cuda:0")


def _circuits():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (256, 12))
    yield "hmm", juice.structures.HMM(seq_length = 8, num_latents = 16, num_emits = 5), 8
    yield "hclt", juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Categorical(num_cats = 5)), 12
    yield "gaussian", juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Gaussian(mu = 0.0, sigma = 1.0)), 12


def _pass(pc, x, kw, **bkw):
    pc.init_param_flows(flows_memory = 0.0)
    pc._cum_flow = 0.0
    with torch.no_grad():
        pc(x, **kw)
    pc.backward(x, compute_param_flows = True, **kw, **bkw)
    return (pc.node_flows.clone(), pc.param_flows.clone(), [l.param_flows.clone() for l in pc.input_layer_group],
            pc._cum_flow)


@pytest.mark.parametrize("flows", ["log", "linear", "linear_modified"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_masked_parameter_flows_are_the_sub_batch(flows, weighted, missing):
    for name, ns, V in _circuits():
        ns.init_parameters(perturbation = 2.0)
        pc = juice.compile(ns, verbose = False).to(DEV)
        B = 24
        g = torch.Generator().manual_seed(1)
        x = (torch.randint(0, 5, (B, V), generator = g) if name != "gaussian" else torch.randn(B, V, generator = g)).to(DEV)
        m = (torch.rand(B, generator = g) < 0.5).to(DEV)
        m[0], m[1] = True, False
        w = (torch.rand(B, generator = g) + 0.5).to(DEV) if weighted else None
        miss = (torch.rand(B, V, generator = g) < 0.3).to(DEV) if missing else None
        bkw = dict(logspace_flows = flows == "log", allow_modify_flows = flows == "linear_modified")
        kw = {} if miss is None else {"missing_mask": miss}
        sub = {} if miss is None else {"missing_mask": miss[m]}

        nf, pf, ipf, cum = _pass(pc, x, kw, ll_weights = w, param_flows_mask = m, **bkw)
        nf_all = _pass(pc, x, kw, ll_weights = w, **bkw)[0]
        _, pf_0, ipf_0, cum_0 = _pass(pc, x, kw, ll_weights = (1.0 if w is None else w) * m.float(), **bkw)
        _, pf_sub, ipf_sub, cum_sub = _pass(pc, x[m], sub, ll_weights = None if w is None else w[m], **bkw)

        assert torch.equal(nf, nf_all), name                                  # (every sample's node flows)
        for got, zero, part in ((pf, pf_0, pf_sub), *zip(ipf, ipf_0, ipf_sub)):
            assert torch.allclose(got, zero, rtol = 1e-6, atol = 1e-7), (name, float((got - zero).abs().max()))
            assert torch.allclose(got, part, rtol = 5e-3, atol = 1e-5), (name, float((got - part).abs().max()))
        assert abs(cum - cum_0) <= 1e-6 * abs(cum_0) and abs(cum - cum_sub) <= 1e-5 * abs(cum_sub), (name, cum, cum_sub)
        assert float(pf.abs().max()) > 0.0


def test_a_mask_is_refused_where_the_parameterization_does_not_take_it():
    with juice.set_block_size(4):
        leaf = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = 5))
        ns = summate(leaf, num_node_blocks = 1, parameterization = LowRank(rank = 2))
        root = summate(multiply(ns, inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = 5))),
                       num_node_blocks = 1, block_size = 1)
    pc = juice.compile(root).to(DEV)
    x = torch.randint(0, 5, (3, 2), device = DEV)
    m = torch.tensor([True, False, True], device = DEV)
    ext = {ns: (torch.randn(3, 4, 2, device = DEV), torch.randn(3, 2, 4, device = DEV))}
    with torch.no_grad():
        pc(x, external_params = ext)
    with pytest.raises(NotImplementedError, match = "param_flows_mask"):
        pc.backward(x, compute_param_flows = True, param_flows_mask = m, external_params = ext)
    # (nothing supplied for the layer: the plain sum layer, which takes it)
    _, pf, _, _ = _pass(pc, x, {}, param_flows_mask = m)
    _, pf_sub, _, _ = _pass(pc, x[m], {})
    assert torch.allclose(pf, pf_sub, rtol = 1e-4, atol = 1e-6)


def test_a_plain_leaf_refuses_a_mask_with_the_evidence_gradients():
    with juice.set_block_size(4):
        xs = [inputs(v, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = 6)) for v in range(2)]
        root = summate(multiply(*xs), num_node_blocks = 1, block_size = 1)
    pc = juice.compile(root).to(DEV)
    x = torch.randint(0, 6, (3, 2), device = DEV)
    lp = torch.log_softmax(torch.randn(3, 2, 6, device = DEV), dim = -1)
    with torch.no_grad():
        pc(x, categorical_evidence_logp = lp)
    with pytest.raises(NotImplementedError, match = "param_flows_mask"):
        pc.backward(x, compute_param_flows = True, param_flows_mask = torch.tensor([True, False, True], device = DEV),
                    categorical_evidence_logp = lp, categorical_evidence_logp_grad = torch.zeros_like(lp))


def test_the_normalizer_is_added_on_the_device_and_read_as_a_float():
    """`_cum_flow`: a backward's share added on the device (`_add_cum_flow`, no host synchronization), folded into the
    float where it is read -- its value and type for every reader, through sets, `+=`, numbers and another device's
    pending sum."""
    pc = juice.compile(juice.structures.HMM(seq_length = 4, num_latents = 4, num_emits = 3))
    pc._cum_flow = 2.0
    pc._add_cum_flow(torch.tensor(1.5, device = DEV))
    pc._add_cum_flow(torch.tensor([0.25], device = DEV, dtype = torch.float32))
    pc._add_cum_flow(3)
    assert isinstance(pc._cum_flow, float) and pc._cum_flow == 6.75
    pc._cum_flow += 1.0
    assert pc._cum_flow == 7.75
    pc._add_cum_flow(torch.tensor(1.0))                                         # (a pending sum on the CPU, then ...)
    pc._add_cum_flow(torch.tensor(1.0, device = DEV))                           # (... one on the GPU: folded first)
    assert pc._cum_flow == 9.75
    pc._add_cum_flow(torch.tensor(5.0, device = DEV))
    pc._cum_flow = 0.0                                                          # (a set drops what is pending)
    assert pc._cum_flow == 0.0
