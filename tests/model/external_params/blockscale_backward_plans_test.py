"""
What BlockScale's backward keeps between calls: per partition and batch, the launch plans of the log-Z gradient and of
the denominator flow. A circuit that ran at other batch sizes -- its plans built there -- must give what a fresh circuit
gives, on a layer of TWO partitions, whose plans must not stand in for each other.

Against a fresh circuit, not a reference: a stale plan launches at another batch (another grid, another `batch_size`),
an error of the gradient's own size, far past what two circuits choosing their forks by measurement differ by.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

# One node block of eight child blocks and three of three: the partitioner splits the layer in two
EDGE_IDS = torch.tensor([[0] * 8 + [1] * 3 + [2] * 3 + [3] * 3, list(range(8)) + [0, 1, 2] * 3])


def _compile(seed = 0):
    torch.manual_seed(seed)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = 8, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        prod = multiply(*ni)
    ns = summate(prod, num_node_blocks = 4, edge_ids = EDGE_IDS, block_size = 64,
                 external_params = BlockScaleSumParams(ch_block_size = 8))
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    assert layer.num_bk_partitions == 2 and layer.num_fw_partitions == 2, "the layer is no longer split in two"
    return pc, ns, layer


def _inputs(ns, batch):
    dev = torch.device("cuda:0")
    g = torch.Generator(device = "cpu").manual_seed(batch)
    data = torch.randint(0, NUM_CATS, [batch, 2], generator = g).to(dev)
    phi = (torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], generator = g) * 0.7).to(dev)
    return data, phi


def _outputs(pc, ns, data, phi):
    """The gate gradient, the denominator flows and the parameter flows of one backward."""
    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)
    return (pc.get_external_params_grad(ns)[0].clone(), pc.denom_param_flows.clone(), pc.param_flows.clone())


def _assert_close(got, ref, what):
    for g, r, name in zip(got, ref, ("the gate gradient", "the denominator flows", "the parameter flows")):
        d = float((g - r).abs().max() / r.abs().max().clamp_min(1e-30))
        assert d < 1e-3, f"{what}: {name} off a fresh circuit's by {d} (relative to its largest entry)"


@cuda_only
def test_a_circuit_run_at_other_batches_gives_a_fresh_circuits_backward():
    used, ns, _ = _compile()
    for batch in (64, 8, 128, 64, 8):
        data, phi = _inputs(ns, batch)
        got = _outputs(used, ns, data, phi)
        fresh, fns, _ = _compile()
        _assert_close(got, _outputs(fresh, fns, data, phi), f"batch {batch}")

