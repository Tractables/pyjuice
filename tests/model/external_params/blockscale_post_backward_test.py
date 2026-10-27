"""
`_bs_post_backward_kernel` -- the log-Z half of the gate gradient, the denominator flow and the `-log Z` shift in one
launch per partition -- against the three launches it replaces (`PYJUICE_BLOCKSCALE_POST_FUSED=0`), on the same
circuit: `node_mars` after the backward to the bit (one add either way), the gradient and the denominator flows to
rounding (the tiles, and with them the sums' order, differ). The accuracy of both against float64 is the gradient and
denominator suites' (they run the fused kernel by default).

The shapes are the kernel's regimes: one node block and several, a ragged layer of two partitions (padded gate cells)
and one whose widest row is not a power of two (the gate table's bound), a node block of 8 (no `tl.dot`: the
broadcast-sum branch), two gate tiles (the shift by the first alone), batches that are not a multiple of the batch tile
(atomic batch slices), and each of the gradient alone, the denominator alone and both.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams
import pyjuice.nodes.external_params.blockscale.backward as bs_backward


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

SPLIT = torch.tensor([[0] * 8 + [1] * 3 + [2] * 3 + [3] * 3, list(range(8)) + [0, 1, 2] * 3])
# the widest row three child blocks, not a power of two: the compiled edge count is padded past the gate table's width
NP2 = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [0, 1, 2, 0, 1, 1, 2, 3]])

# name: (node blocks, node block size, child blocks, child block size, gate width, edge ids or None)
SHAPES = {
    "one_block":    (1, 128, 1, 128, 8, None),
    "four_blocks":  (4, 64, 4, 64, 16, None),
    "ragged_split": (4, 64, 8, 32, 8, SPLIT),
    "block_8":      (4, 8, 4, 8, 4, None),
    "many_gates":   (1, 64, 4, 64, 4, None),            # 64 gates: two gate tiles
    # 1024 gates, 32 gate tiles: the shift by the first alone. With two, both tiles' programs ran at once and read
    # `node_mars` before either wrote it, so a shift by every tile went unseen
    "gate_tiles":   (1, 64, 64, 64, 4, None),
    "ragged_np2":   (4, 64, 4, 32, 8, NP2),
}


def _compile(name, apply_z_correction):
    nb, bs, nc, cbs, gate_cbs, edge_ids = SHAPES[name]
    torch.manual_seed(0)
    with juice.set_block_size(cbs):
        ni = [inputs(v, num_node_blocks = nc, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        prod = multiply(*ni)
    kw = dict(edge_ids = edge_ids) if edge_ids is not None else {}
    ns = summate(prod, num_node_blocks = nb, block_size = bs,
                 external_params = BlockScaleSumParams(ch_block_size = gate_cbs,
                                                       apply_z_correction = apply_z_correction), **kw)
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    return pc, ns, layer


def _backward(pc, ns, data, phi, want_grad):
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0, compute_external_grads = want_grad)
    return (pc.node_mars.clone(), pc.get_external_params_grad(ns)[0].clone() if want_grad else None,
            pc.denom_param_flows.clone() if pc.denom_param_flows is not None else None)


@cuda_only
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("batch", [64, 100, 8])
@pytest.mark.parametrize("want_grad,apply_z_correction", [(True, True), (True, False), (False, True)])
def test_one_launch_gives_what_three_give(name, batch, want_grad, apply_z_correction, monkeypatch):
    if name == "block_8" and batch < 16:
        pytest.skip("a gated layer of block size 8 has no parameter-flow backward below batch 16 (the sum layer "
                    "refuses it), with either path")
    if name == "ragged_np2" and batch < 16:
        pytest.skip("below batch 16 the small-batch parameter-flow fork refuses a row of more edge blocks than the gate "
                    "table's width, with either path")
    pc, ns, layer = _compile(name, apply_z_correction)
    if name == "ragged_split":
        assert layer.num_fw_partitions == 2, "the ragged layer is no longer split in two"
    dev = torch.device("cuda:0")
    g = torch.Generator(device = "cpu").manual_seed(batch)
    data = torch.randint(0, NUM_CATS, [batch, 2], generator = g).to(dev)
    phi = (torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], generator = g) * 0.7).to(dev)

    monkeypatch.setattr(bs_backward, "POST_FUSED", False)
    nm3, grad3, w3 = _backward(pc, ns, data, phi, want_grad)
    monkeypatch.setattr(bs_backward, "POST_FUSED", True)
    nm1, grad1, w1 = _backward(pc, ns, data, phi, want_grad)
    assert layer.__dict__.get("_bs_post_plans"), "the fused kernel did not run"
    # the same from run to run, as the log-Z kernel's was: node tiles add their gradient in order, not atomically
    if want_grad:
        assert torch.equal(_backward(pc, ns, data, phi, want_grad)[1], grad1), \
            "two equal backwards gave different gradients"

    assert torch.equal(nm1, nm3), "`node_mars` after the backward differs from the three launches'"
    for got, ref, what in ((grad1, grad3, "the gate gradient"), (w1, w3, "the denominator flows")):
        if ref is None:
            assert got is None
            continue
        scale = float(ref.abs().max())
        assert scale > 0, f"{what}: nothing was written"
        d = float((got - ref).abs().max()) / scale
        assert d < 1e-5, f"{what} off the three launches' by {d} (relative to its largest entry)"
