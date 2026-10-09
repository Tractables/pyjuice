"""
The sum layer's forward precision modes (`pyjuice.layer.sum_layer.PRECISIONS`), against a float64 forward:

* "auto" keeps today's kernels;
* "tf32" never uses bf16 products; "fp32" gives fp32-level products (`tf32x3` in the Triton kernel);
* the removed `force_use_bf16` / `force_use_fp32` flags are refused when set and ignored when unset;
* an unknown mode is refused.
"""
import pytest
import torch

import pyjuice as juice

DEV = torch.device("cuda:0")


def hmm_sum_layer(latents, seed = 0):
    torch.manual_seed(seed)
    pc = juice.compile(juice.structures.HMM(seq_length = 3, num_latents = latents, num_emits = 16),
                       verbose = False).to(DEV)
    return pc, [l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg.layers][0]


def run(pc, layer, batch, seed = 0, **kw):
    """The layer's forward on random children; returns (its node rows, the float64 forward of the same)."""
    nids, cids, pids = layer.partitioned_nids[0], layer.partitioned_cids[0], layer.partitioned_pids[0]
    NB = cids.size(0)
    BS = layer.block_size
    first = int(cids[cids > 0].min())
    R = int(cids.max()) - first + 1                                 # child rows (a node block may read a subset)
    g = torch.Generator(device = DEV).manual_seed(seed)
    x = torch.randn(R, batch, device = DEV, generator = g) * 3 - 5
    node_mars = torch.zeros(pc.num_nodes, batch, device = DEV)
    element_mars = torch.full((pc.num_elements, batch), -float("inf"), device = DEV)
    element_mars[first:first + R] = x
    layer.forward(node_mars, element_mars, pc.params, **kw)
    W = torch.zeros(NB * BS, R, dtype = torch.float64, device = DEV)
    i = torch.arange(BS, device = DEV)
    for k in range(NB):
        W[k * BS + i[:, None], (cids[k] - first)[None, :]] += pc.params[pids[k][None, :] + i[:, None]].double()
    m = x.double().max(0, keepdim = True).values
    exact = torch.log(W @ torch.exp(x.double() - m)) + m
    n0 = int(nids.min())
    return node_mars[n0:n0 + NB * BS].double(), exact


#: max |log value - float64| per mode, with margin over the measured (5e-4 TF32, 3e-6 fp32 on a 4096-latent layer)
TOL = {"tf32": 2e-3, "fp32": 2e-5}


@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
def test_triton_path_accuracy(precision):
    pc, layer = hmm_sum_layer(64)
    got, exact = run(pc, layer, 48, precision = precision)
    err = (got - exact).abs().max()
    if precision == "auto":
        assert err <= 2e-2                                          # bf16 products
    else:
        assert err <= TOL[precision]


def test_removed_layer_flags():
    pc, layer = hmm_sum_layer(16)
    for name in ("force_use_bf16", "force_use_fp32"):
        with pytest.raises(TypeError, match = f"{name}.*precision"):   # set: refused
            run(pc, layer, 16, **{name: True})
        run(pc, layer, 16, **{name: False})                          # unset (their old default): ignored


def test_unknown_precision_is_refused():
    pc, layer = hmm_sum_layer(16)
    with pytest.raises(ValueError, match = "precision"):
        run(pc, layer, 16, precision = "bf16")
