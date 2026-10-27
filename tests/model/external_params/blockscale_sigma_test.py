"""
`_sigma`'s per-gate parameter sums from `blockscale_sigma.cu` against its torch gather (the fallback where the kernel
does not compile): fp32 and bf16 parameters, one node block and several, a ragged layer of two partitions (padded
edge slots), a node block of 8, gate widths from 4 to the whole edge block; then the cache: the same tensor until the
parameters change, a new one after.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams
import pyjuice.nodes.external_params.kernels.c as ext_c


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

SPLIT = torch.tensor([[0] * 8 + [1] * 3 + [2] * 3 + [3] * 3, list(range(8)) + [0, 1, 2] * 3])

# name: (node blocks, node block size, child blocks, child block size, gate width, edge ids or None)
SHAPES = {
    "one_block":    (1, 128, 1, 128, 8, None),
    "four_blocks":  (4, 64, 4, 64, 16, None),
    "ragged_split": (4, 64, 8, 32, 8, SPLIT),
    "block_8":      (4, 8, 4, 8, 4, None),
    "gate_4":       (2, 32, 4, 64, 4, None),
    "whole_block":  (2, 64, 2, 64, 64, None),
}


def _compile(name, dtype):
    nb, bs, nc, cbs, gate_cbs, edge_ids = SHAPES[name]
    torch.manual_seed(0)
    with juice.set_block_size(cbs):
        ni = [inputs(v, num_node_blocks = nc, dist = dists.Categorical(num_cats = 4)) for v in range(2)]
        prod = multiply(*ni)
    kw = dict(edge_ids = edge_ids) if edge_ids is not None else {}
    ns = summate(prod, num_node_blocks = nb, block_size = bs,
                 external_params = BlockScaleSumParams(ch_block_size = gate_cbs), **kw)
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = dtype).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    return pc, ns, layer, gate_cbs


@cuda_only
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_sums_as_the_gather(name, dtype, monkeypatch):
    assert ext_c.get_sigma_module() is not None, "blockscale_sigma.cu did not compile"
    pc, ns, layer, gate_cbs = _compile(name, dtype)
    if name == "ragged_split":
        assert layer.num_fw_partitions == 2, "the ragged layer is no longer split in two"
    bs = ns.external_params
    for pid in range(layer.num_fw_partitions):
        layer.__dict__.pop("_bs_sigma_cache", None)
        got, n_got = bs._sigma(layer, pc.params, pid, layer.block_size, gate_cbs)
        with monkeypatch.context() as m:
            m.setattr(ext_c, "get_sigma_module", lambda: None)
            layer.__dict__.pop("_bs_sigma_cache", None)
            ref, n_ref = bs._sigma(layer, pc.params, pid, layer.block_size, gate_cbs)
        assert n_got == n_ref and got.shape == ref.shape and got.dtype == torch.float32
        assert float(ref.abs().max()) > 0, "nothing summed"
        # the same sums in another order: fp32 rounding apart at most
        d = float(((got - ref).abs() / ref.abs().clamp(min = 1e-30)).max())
        assert d < 1e-5, f"partition {pid}: sigma off the gather by {d} (relative)"


@cuda_only
def test_cache_until_the_parameters_change():
    pc, ns, layer, gate_cbs = _compile("four_blocks", torch.bfloat16)
    bs = ns.external_params
    layer.__dict__.pop("_bs_sigma_cache", None)
    a, _ = bs._sigma(layer, pc.params, 0, layer.block_size, gate_cbs)
    b, _ = bs._sigma(layer, pc.params, 0, layer.block_size, gate_cbs)
    assert a is b, "an unchanged parameter tensor was summed again"
    with torch.no_grad():
        pc.params.mul_(2.0)
    c, _ = bs._sigma(layer, pc.params, 0, layer.block_size, gate_cbs)
    assert c is not a and torch.allclose(c, 2.0 * a, rtol = 1e-6), "the sums did not follow the parameters"
