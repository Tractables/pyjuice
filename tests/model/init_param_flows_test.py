"""
`TensorCircuit.init_param_flows(flows_memory)`: at 0 every flow buffer -- the sum layers', the denominator's (a
circuit requesting it) and every input layer's -- comes back +0.0, whatever it held (negative flows, NaN, inf); at any
other value below 1 the flows are scaled, to the bit; at 1 they are left alone.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams

DEV = torch.device("cuda:0")


def _circuit(gated):
    torch.manual_seed(0)
    with juice.set_block_size(4):
        ni = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(2)]
        kw = {"external_params": BlockScaleSumParams(ch_block_size = 2)} if gated else {}
        ns = summate(multiply(*ni), num_node_blocks = 2, **kw)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(DEV)


def _buffers(pc):
    out = [pc.param_flows] + [layer.param_flows for layer in pc.input_layer_group]
    if getattr(pc, "denom_param_flows", None) is not None:
        out.append(pc.denom_param_flows)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
@pytest.mark.parametrize("gated", [False, True])
def test_flows_memory_zero_clears_every_buffer(gated):
    pc = _circuit(gated)
    pc.init_param_flows(flows_memory = 0.0)
    for layer in pc.input_layer_group:
        layer.init_param_flows(flows_memory = 0.0)                       # (allocated)
    if gated:                                        # (a z-corrected gate requests the denominator flows)
        assert getattr(pc, "denom_param_flows", None) is not None

    def fill():
        for b in _buffers(pc):
            b.uniform_(-2.0, 2.0)
            b.view(-1)[:3] = torch.tensor([float("nan"), float("inf"), -float("inf")], device = DEV)

    fill()
    pc.init_param_flows(flows_memory = 0.0)
    for b in _buffers(pc):
        assert (b == 0).all() and not torch.signbit(b).any()             # +0.0 everywhere, NaN and inf cleared

    fill()
    for b in _buffers(pc):
        b.view(-1)[:3] = 1.0
    want = [b.clone() * 0.25 for b in _buffers(pc)]
    pc.init_param_flows(flows_memory = 0.25)
    assert all(torch.equal(b, w) for b, w in zip(_buffers(pc), want))   # scaled, to the bit

    want = [b.clone() for b in _buffers(pc)]
    pc.init_param_flows(flows_memory = 1.0)
    assert all(torch.equal(b, w) for b, w in zip(_buffers(pc), want))   # left alone
