"""
`_sigma`'s per-gate parameter sums from `blockscale_sigma.cu` against its torch gather (the fallback where the kernel
does not compile): fp32 and bf16 parameters, one node block and several, a ragged layer of two partitions (padded
edge slots), a node block of 8, gate widths from 4 to the whole edge block; then the cache: the same tensor until the
parameters change, a new one after; and a tied group's layers sharing one sum (`_sigma_slot`): summed once a parameter
change for the group, untied layers their own, two circuits of the same nodes each their own parameters, a member
whose parameter ids differ not shared.
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


def _forget(bs, layer):
    """Every sum `_sigma` keeps for `layer`: its own and its tied group's."""
    for name in ("_bs_sigma_cache", "_bs_sigma_slot"):
        layer.__dict__.pop(name, None)
    bs.__dict__.pop("_bs_sigma_groups", None)


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
        _forget(bs, layer)
        got, n_got = bs._sigma(layer, pc.params, pid, layer.block_size, gate_cbs)
        with monkeypatch.context() as m:
            m.setattr(ext_c, "get_sigma_module", lambda: None)
            _forget(bs, layer)
            ref, n_ref = bs._sigma(layer, pc.params, pid, layer.block_size, gate_cbs)
        assert got is not ref
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


def _hmm(tied, dtype = torch.bfloat16, T = 6, seed = 0):
    """A gated HMM of `T - 1` transitions, all the first's copies (`tied`) or each its own."""
    torch.manual_seed(seed)
    with juice.set_block_size(256):
        ns = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4))
        src = None
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4))
            if tied and src is not None:
                ns = src.duplicate(multiply(ns, emit), tie_params = True)
            else:
                ns = summate(multiply(ns, emit), num_node_blocks = 2,
                             external_params = BlockScaleSumParams(ch_block_size = 16))
                if src is None:
                    src = ns
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root


def _gated(pc):
    return [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers if hasattr(l, "external_node_infos")]


def _counting(monkeypatch):
    """`[n]`: how many times `blockscale_sigma.cu` sums from now on."""
    real, n = ext_c.get_sigma_module(), [0]
    assert real is not None, "blockscale_sigma.cu did not compile"

    class Counting:
        def blockscale_sigma(self, *args):
            n[0] += 1
            return real.blockscale_sigma(*args)

    monkeypatch.setattr(ext_c, "get_sigma_module", lambda: Counting())
    return n


def _sums(pc):
    return [l.external_params._sigma(l, pc.params, pid, l.block_size, l.external_params.gate_sizes(l.nodes[0])[1])[0]
            for l in _gated(pc) for pid in range(l.num_fw_partitions)]


def _gather(pc, layer, pid, monkeypatch):
    """The torch gather's sums for `layer`'s partition `pid`, kept by nothing."""
    bs = layer.external_params
    saved = (dict(layer.__dict__.get("_bs_sigma_cache", {})), dict(layer.__dict__.get("_bs_sigma_slot", {})),
             bs.__dict__.get("_bs_sigma_groups"))
    with monkeypatch.context() as m:
        m.setattr(ext_c, "get_sigma_module", lambda: None)
        _forget(bs, layer)
        out = bs._sigma(layer, pc.params, pid, layer.block_size, bs.gate_sizes(layer.nodes[0])[1])[0]
    layer.__dict__["_bs_sigma_cache"], layer.__dict__["_bs_sigma_slot"] = saved[0], saved[1]
    if saved[2] is not None:
        bs.__dict__["_bs_sigma_groups"] = saved[2]
    return out


def _close(got, ref, what):
    d = float(((got - ref).abs() / ref.abs().clamp(min = 1e-30)).max())
    assert float(ref.abs().max()) > 0 and d < 1e-5, f"{what}: off the gather by {d} (relative)"


@cuda_only
def test_a_tied_groups_layers_share_one_sum(monkeypatch):
    pc = juice.compile(_hmm(tied = True), verbose = False, param_dtype = torch.bfloat16).to("cuda:0")
    layers = _gated(pc)
    assert len(layers) == 5 and all(l.num_fw_partitions == 1 for l in layers)
    n = _counting(monkeypatch)
    sums = _sums(pc)
    assert n[0] == 1, f"{n[0]} sums for one tied group"
    assert all(s is sums[0] for s in sums), "the group's layers hold different tensors"
    _close(sums[0], _gather(pc, layers[3], 0, monkeypatch), "the shared sum")
    with torch.no_grad():
        pc.params.mul_(3.0)                                  # (a parameter change: the group sums once again)
    again = _sums(pc)
    assert n[0] == 2 and all(s is again[0] for s in again) and again[0] is not sums[0]
    _close(again[0], _gather(pc, layers[2], 0, monkeypatch), "after the change")   # (bf16: not 3x the old exactly)
    x = torch.randint(0, 4, (32, 6), device = "cuda:0")      # (a training step's forward and backward: one sum)
    phi = {l.nodes[0]: torch.randn(l.external_params.tensor_shapes(l.nodes[0], 32)[0], device = "cuda:0")
           for l in layers}
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    pc.mini_batch_em(step_size = 0.1, pseudocount = 0.1)
    before = n[0]
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    assert n[0] == before + 1, f"a step after an EM update summed {n[0] - before} times"


@cuda_only
def test_untied_layers_keep_their_own(monkeypatch):
    pc = juice.compile(_hmm(tied = False), verbose = False, param_dtype = torch.bfloat16).to("cuda:0")
    layers = _gated(pc)
    n = _counting(monkeypatch)
    sums = _sums(pc)
    assert n[0] == len(layers) == 5 and len({id(s) for s in sums}) == 5
    for i, l in enumerate(layers):
        _close(sums[i], _gather(pc, l, 0, monkeypatch), f"layer {i}")


@cuda_only
def test_two_circuits_of_the_same_nodes_sum_their_own_parameters(monkeypatch):
    root = _hmm(tied = True)
    pc1 = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to("cuda:0")
    pc2 = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to("cuda:0")
    with torch.no_grad():
        pc2.params.mul_(2.0)
        pc1.params.mul_(1.0)                                  # (equal versions: only the tensors tell them apart)
    assert pc1.params._version == pc2.params._version
    for _ in range(2):                                        # (interleaved: each validates the shared slot)
        s1, s2 = _sums(pc1), _sums(pc2)
        _close(s1[0], _gather(pc1, _gated(pc1)[0], 0, monkeypatch), "the first circuit")
        _close(s2[-1], _gather(pc2, _gated(pc2)[-1], 0, monkeypatch), "the second circuit")
        _close(s2[0], 2.0 * s1[0], "the second against the first")


@cuda_only
def test_a_member_whose_parameter_ids_differ_is_not_shared(monkeypatch):
    pc = juice.compile(_hmm(tied = True), verbose = False, param_dtype = torch.bfloat16).to("cuda:0")
    layers = _gated(pc)
    bs = layers[0].external_params
    gate_cbs = bs.gate_sizes(layers[0].nodes[0])[1]
    first = bs._sigma(layers[0], pc.params, 0, layers[0].block_size, gate_cbs)[0]
    (slot,) = bs.__dict__["_bs_sigma_groups"].values()
    slot[0] = slot[0].flip(0).contiguous()                   # (the group's ids no longer the members')
    n = _counting(monkeypatch)
    other = bs._sigma(layers[1], pc.params, 0, layers[1].block_size, gate_cbs)[0]
    assert n[0] == 1 and other is not first and layers[1].__dict__["_bs_sigma_slot"][0] is False
    _close(other, _gather(pc, layers[1], 0, monkeypatch), "the unshared member")
