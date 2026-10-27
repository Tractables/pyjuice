"""
The input layer's mini-batch EM update in a kernel of its own (`pyjuice.layer.kernels.input_em`, `Categorical`'s
`em_layer_fn`) against the layer's template running `em_fn`: the same bits, over Categorical layers of several shapes
-- an HMM's tied emissions (few source nodes, the tied nodes' flows accumulated first), variables of mixed category
counts in one layer, 2 and 256 categories, many nodes, some parameters exactly zero -- with and without
`keep_zero_params`. Past 256 categories the template's other `em_fn` runs, and the kernel is not taken.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
import pyjuice.layer.kernels.input_em as input_em

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
DEV = torch.device("cuda:0")


def _hmm(num_cats = 64, latents = 256, T = 6):
    """An HMM whose emissions are all the first's (tied): 2 x `latents` source nodes of input."""
    with juice.set_block_size(latents // 2):
        ns = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = num_cats))
        src, esrc = None, None
        for t in range(1, T):
            emit = esrc.duplicate(t, tie_params = True) if esrc is not None else None
            if emit is None:
                emit = esrc = inputs(t, num_node_blocks = 2, dist = dists.Categorical(num_cats = num_cats))
            if src is None:
                ns = src = summate(multiply(ns, emit), num_node_blocks = 2)
            else:
                ns = src.duplicate(multiply(ns, emit), tie_params = True)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    return root, [num_cats] * T


def _mixed(cats = (3, 64, 17, 2, 64, 200), nodes = 32):
    """Variables of different category counts, all in one input layer."""
    ns = [inputs(v, num_node_blocks = 1, block_size = nodes, dist = dists.Categorical(num_cats = c))
          for v, c in enumerate(cats)]
    sums = [summate(multiply(ns[i], ns[i + 1]), num_node_blocks = 1, block_size = nodes)
            for i in range(0, len(ns), 2)]
    root = summate(multiply(*sums), num_node_blocks = 1, block_size = 1)
    return root, list(cats)


def _wide(num_cats, num_vars = 8, nodes = 64):
    ns = [inputs(v, num_node_blocks = 1, block_size = nodes, dist = dists.Categorical(num_cats = num_cats))
          for v in range(num_vars)]
    sums = [summate(multiply(ns[i], ns[i + 1]), num_node_blocks = 1, block_size = nodes)
            for i in range(0, num_vars, 2)]
    root = summate(multiply(*sums), num_node_blocks = 1, block_size = 1)
    return root, [num_cats] * num_vars


CASES = [
    ("tied_hmm", lambda: _hmm()),
    ("mixed_categories", lambda: _mixed()),
    ("two_categories", lambda: _wide(2)),
    ("256_categories", lambda: _wide(256)),
    ("many_nodes", lambda: _wide(64, num_vars = 32, nodes = 512)),       # 16384 source nodes
]


def _instance(builder, seed = 0):
    torch.manual_seed(seed)
    root, cats = builder()
    pc = juice.compile(root, verbose = False).to(DEV)
    # A few exactly-zero parameters (`keep_zero_params`' case; left unnormalized -- the update reads them as they are)
    for layer in pc.input_layer_group:
        with torch.no_grad():
            layer.params[torch.rand_like(layer.params) < 0.05] = 0.0
    x =torch.stack([torch.randint(0, c, (128,)) for c in cats], dim = 1).to(DEV)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    return pc


def _counted(monkeypatch):
    calls = [0]
    real = input_em.categorical_em

    def counted(*args, **kwargs):
        ran = real(*args, **kwargs)
        calls[0] += int(ran)
        return ran

    monkeypatch.setattr(input_em, "categorical_em", counted)
    return calls


def _both(layer, keep_zero_params, monkeypatch):
    """The layer's update by its own kernel and by the template, from the same parameters and flows."""
    params, flows = layer.params.clone(), layer.param_flows.clone()
    layer.mini_batch_em(step_size = 0.3, pseudocount = 0.1, keep_zero_params = keep_zero_params)
    torch.cuda.synchronize()
    own = layer.params.clone()
    with torch.no_grad():
        layer.params.copy_(params)
        layer.param_flows.copy_(flows)
    monkeypatch.setattr(layer, "em_layer_fn", None)
    layer.mini_batch_em(step_size = 0.3, pseudocount = 0.1, keep_zero_params = keep_zero_params)
    torch.cuda.synchronize()
    monkeypatch.undo()
    return own, layer.params.clone(), params


@cuda_only
@pytest.mark.parametrize("keep_zero_params", [False, True])
@pytest.mark.parametrize("name,builder", CASES, ids = [c[0] for c in CASES])
def test_its_own_kernel_gives_the_templates_bits(name, builder, keep_zero_params, monkeypatch):
    pc = _instance(builder)
    layers = list(pc.input_layer_group)
    assert layers
    for layer in layers:
        calls = _counted(monkeypatch)
        own, template, before = _both(layer, keep_zero_params, monkeypatch)
        assert calls[0] == 1, f"{name}: the layer's own kernel did not run"
        assert not torch.equal(own, before), f"{name}: nothing was updated"
        assert torch.equal(own.view(torch.int32), template.view(torch.int32)), \
            f"{name}: the update differs from the template's (max {float((own - template).abs().max()):.2e})"
        if keep_zero_params:
            assert torch.all(own[before == 0] == 0), f"{name}: a zero parameter moved"


@cuda_only
def test_past_256_categories_the_template_runs(monkeypatch):
    pc = _instance(lambda: _wide(300, num_vars = 4, nodes = 16))
    for layer in pc.input_layer_group:
        calls = _counted(monkeypatch)
        layer.mini_batch_em(step_size = 0.3, pseudocount = 0.1)
        torch.cuda.synchronize()
        assert calls[0] == 0
        monkeypatch.undo()
