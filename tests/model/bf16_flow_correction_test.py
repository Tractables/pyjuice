"""
EM on parameters stored in bf16 takes their flows to the fp32 master copy's values first (`params_master.flows_to_master`):
the backward makes every parameter flow with the stored values, so each flow's last factor is its parameter's rounding,
and EM on the flows as they are converges to the rounded circuit's fixed point (MEASURED, HCLT-128 on MNIST: -0.5 nats a
sample against fp32 training). Against an fp32 twin, every EM path: the bf16 circuit's update from a master copy off the
bf16 grid (`M`, stored `S` its rounding, a few values written from outside since -- the master copy's check takes those)
and the flows its own backward made (`F`) is the fp32 circuit's update from the values the check takes (`old`) and the
flows `F * old / S`: the node-block forms forced in turn, the per-block kernels (blocks below `em_blocks.MIN_BLOCK_SIZE`),
tied flows, keep-zero, and the conditional dual-flow M-step of BlockScale's z-correction with the master copy's passes
fused in, of their own, and in its generic torch form. With the flows as they are, the twins come apart: the test sees
the factor.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.model.backend import em_blocks

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
BF16 = torch.bfloat16
FORMS = [("split", em_blocks.BLOCK_N, 4), ("block", None, em_blocks.BLOCK_WARPS), ("one",) + em_blocks.ONE_PASS_TILE,
         ("one",) + em_blocks.ONE_PASS_TILE_MASTER]


def _twins(kind):
    """(fp32 pc, bf16 pc, V, C, gated) from the same nodes; `gated`: BlockScale transitions with the z-correction."""
    torch.manual_seed(0)
    gated = kind == "gated"
    if kind == "hclt":                                             # untied, node blocks of 64
        x = torch.randint(0, 16, (512, 24))
        ns = juice.structures.HCLT(x.float(), num_latents = 64, input_dist = dists.Categorical(num_cats = 16))
        V, C = 24, 16
    elif kind == "hmm":                                            # tied transitions, blocks of 128
        ns = juice.structures.HMM(seq_length = 12, num_latents = 128, num_emits = 32); V, C = 12, 32
    elif kind == "pd":                                             # blocks of 8: the per-block kernels
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 8, split_intervals = 2,
                                 input_dist = dists.Categorical(num_cats = 16)); V, C = 16, 16
    else:
        from pyjuice.nodes import inputs, multiply, summate
        from pyjuice.parameterizations import BlockScale
        with juice.set_block_size(32):
            chain = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = 8))
            for t in range(1, 6):
                emit = inputs(t, num_node_blocks = 4, dist = dists.Categorical(num_cats = 8))
                chain = summate(multiply(chain, emit), num_node_blocks = 4,
                                parameterization = BlockScale(ch_block_size = 32))
            ns = summate(multiply(chain), num_node_blocks = 1, block_size = 1)
        V, C = 6, 8
    ns.init_parameters(perturbation = 2.0)
    pc32 = juice.compile(ns, verbose = False).to(DEV)
    pc16 = juice.compile(ns, verbose = False, param_dtype = BF16).to(DEV)
    return pc32, pc16, V, C, gated


def _gates(pc, B):
    g = torch.Generator(device = DEV).manual_seed(2)
    gates = {}
    for ns in pc.root_ns:
        if ns.is_sum() and getattr(ns, "parameterization", None) is not None:
            shapes = ns.parameterization.tensor_shapes(ns, B)
            tensors = tuple(torch.randn(s, device = DEV, generator = g) * 0.5 - 0.5 for s in shapes)
            gates[ns] = tensors if len(tensors) > 1 else tensors[0]
    return gates


def _pid_of_pfid(pc):
    """Each parameter-flow slot's parameter (-1: none) -- a node's flow range and its parameter range (a tied node's
    its source's) in the same local order."""
    idx = torch.full([pc.num_param_flows], -1, dtype = torch.long)
    for ns in pc.root_ns:
        if ns.is_sum() and ns._param_flow_range is not None:
            (ps, pe), (pfs, pfe) = ns._param_range, ns._param_flow_range
            assert pe - ps == pfe - pfs
            idx[pfs:pfe] = torch.arange(ps, pe)
    return idx.to(DEV)


def _step(pc32, pc16, V, C, gated, step_size, pcount, kz, correct = True):
    """One EM step of the bf16 circuit from an off-grid master copy, and of the fp32 twin from what the update should
    start from; (bf16 master after, fp32 params after)."""
    B = 64
    g = torch.Generator(device = DEV).manual_seed(1)
    x = torch.randint(0, C, (B, V), device = DEV, generator = g)
    kw = dict(sum_external_params = _gates(pc16, B)) if gated else {}
    base = pc32.params.detach().clone()
    M = base * torch.exp(0.05 * torch.randn(base.shape, device = DEV, generator = g))       # off the bf16 grid
    if kz:
        M[torch.rand(M.shape, device = DEV, generator = g) < 0.1] = 0.0                   # zeros stay where they are
    S = M.to(BF16)
    # written from outside -- from a generator of its own: from `g`, a step without keep-zero drew it at the position the
    # keep-zero step's zeros took, and those zeros, kept, left every written value 0 (x1.5 changing nothing)
    gw = torch.Generator(device = DEV).manual_seed(3)
    written = torch.rand(M.shape, device = DEV, generator = gw) < 0.03
    S = torch.where(written, (S.float() * 1.5).to(BF16), S)
    old = torch.where(S != M.to(BF16), S.float(), M)                                         # the master copy's check
    assert bool((S != M.to(BF16)).any())
    pc16.params.data.copy_(S)
    pc16._params_master(sync = False).copy_(M)
    pc32.params.data.copy_(old)

    # the bf16 circuit's flows, made with the stored values; the twin's: those taken to `old`
    pc16.init_param_flows(flows_memory = 0.0)
    pc16(x, **kw)
    pc16.backward(x, flows_memory = 1.0, **kw)
    pid = _pid_of_pfid(pc16)
    s, o = S.float()[pid.clamp_min(0)], old[pid.clamp_min(0)]
    r = torch.where((pid < 0) | (s == 0) | (s == o), torch.ones_like(s), o / s) if correct else torch.ones_like(s)
    pc32.init_param_flows(flows_memory = 0.0)
    pc32.param_flows.copy_(pc16.param_flows * r)
    if gated:
        pc32.denom_param_flows.copy_(pc16.denom_param_flows)             # gate space: no parameter in it
    for l32, l16 in zip(pc32.input_layer_group, pc16.input_layer_group):
        l32.param_flows.copy_(l16.param_flows)

    for pc in (pc16, pc32):
        pc.mini_batch_em(step_size = step_size, pseudocount = pcount, keep_zero_params = kz)
    assert torch.equal(pc16.params, pc16.params_master.to(BF16))
    return pc16.params_master.detach().double(), pc32.params.detach().double()


def _close(a, b):
    return torch.allclose(a, b, atol = 1e-8, rtol = 1e-5)


@cuda_only
@pytest.mark.parametrize("form", FORMS, ids = lambda c: f"{c[0]}{c[1]}x{c[2]}")
@pytest.mark.parametrize("kind", ["hclt", "hmm", "pd"])
def test_em_takes_the_flows_to_the_master_copy(kind, form, monkeypatch):
    def forced(nch_bucket, nodes_bucket, has_master):                 # (the one-pass and block forms need a bucket)
        if form[0] == "block":
            return ("block", nodes_bucket, form[2]) if nodes_bucket is not None else ("split", em_blocks.BLOCK_N, 4)
        return form if form[0] != "one" or nch_bucket is not None else ("split", em_blocks.BLOCK_N, 4)
    monkeypatch.setattr(em_blocks, "_form", forced)
    pc32, pc16, V, C, gated = _twins(kind)
    # (pseudocount 0 before keep-zero: the zeros it keeps leave nodes without flow, and those are 0 / 0 -- NaN -- at
    # pseudocount 0 in either circuit, which the steps after would carry)
    for step_size, pcount, kz in ((1.0, 0.1, False), (0.1, 0.1, False), (0.3, 0.0, False), (0.5, 0.1, True)):
        m16, p32 = _step(pc32, pc16, V, C, gated, step_size, pcount, kz)
        assert _close(m16, p32), (step_size, pcount, kz, float(((m16 - p32).abs() / p32.abs().clamp_min(1e-12)).max()))
    # the flows as they are: another update (the test sees the factor)
    m16, p32 = _step(pc32, pc16, V, C, gated, 1.0, 0.1, False, correct = False)
    assert not _close(m16, p32)


@cuda_only
def test_anemones_top_down_term_takes_the_factor_too():
    """Anemone's update (`step_size_rescaling`) adds `(1 - step_size) * P_td * theta` to the flows, made from the stored
    values like the data's -- so the factor takes its last factor to the master copy's value as well, and it must be
    neither made from the copy and then taken (the factor twice) nor left as stored. Exactly comparable with an fp32 twin
    on ONE node set: the master copy off the bf16 grid in its range alone (on it elsewhere), so the top-down
    probabilities of its nodes -- products of its ancestors' parameters -- are the same in both circuits."""
    pc32, pc16, V, C, gated = _twins("hclt")
    target = max((ns for ns in pc16.root_ns if ns.is_sum() and not ns.is_tied() and ns is not pc16.root_ns),
                 key = lambda ns: ns._param_range[1] - ns._param_range[0])
    ps, pe = target._param_range
    g = torch.Generator(device = DEV).manual_seed(4)
    x = torch.randint(0, C, (64, V), device = DEV, generator = g)
    M = pc32.params.detach().to(BF16).float()                                      # on the grid ...
    M[ps:pe] = M[ps:pe] * torch.exp(0.05 * torch.randn(pe - ps, device = DEV, generator = g))   # ... but `target`
    S = M.to(BF16)
    assert bool((S[ps:pe].float() != M[ps:pe]).any()) and torch.equal(S[:ps].float(), M[:ps])

    def run(correct):
        pc16.params.data.copy_(S)
        pc16._params_master(sync = False).copy_(M)
        pc32.params.data.copy_(M)
        pc16.init_param_flows(flows_memory = 0.0)
        pc16(x)
        pc16.backward(x, flows_memory = 1.0)
        pid = _pid_of_pfid(pc16)
        s, o = S.float()[pid.clamp_min(0)], M[pid.clamp_min(0)]
        r = torch.where((pid < 0) | (s == 0) | (s == o), torch.ones_like(s), o / s) if correct else torch.ones_like(s)
        pc32(x)
        pc32.backward(x, flows_memory = 0.0)                       # (the top-down pass's buffers; its flows replaced)
        pc32.param_flows.copy_(pc16.param_flows * r)
        for l32, l16 in zip(pc32.input_layer_group, pc16.input_layer_group):
            l32.param_flows.copy_(l16.param_flows)
        pc32._cum_flow = pc16._cum_flow
        for pc in (pc16, pc32):
            pc.mini_batch_em(step_size = 0.4, pseudocount = 0.1, step_size_rescaling = True)
        assert torch.equal(pc16.params, pc16.params_master.to(BF16))
        return pc16.params_master[ps:pe].double(), pc32.params[ps:pe].detach().double()

    m16, p32 = run(correct = True)
    assert _close(m16, p32), float(((m16 - p32).abs() / p32.abs().clamp_min(1e-12)).max())
    m16, p32 = run(correct = False)
    assert not _close(m16, p32)


@cuda_only
@pytest.mark.parametrize("path", ["fused_passes", "passes_of_their_own", "torch"])
def test_dual_flow_em_takes_the_flows_to_the_master_copy(path, monkeypatch):
    """BlockScale's z-correction: the conditional M-step `theta <- normalize(theta * F+ / F-)` on `F+` taken to the
    master copy's values (`F- = theta * W` from the copy itself) -- the master copy's check in the update's own passes
    (a range past the L2: forced), in a pass of its own (the shared kernels' `ROUNDED`), and the generic torch form."""
    from pyjuice.parameterizations.sum_parameterization import SumParameterization
    monkeypatch.setattr(SumParameterization, "_fold_master", staticmethod(lambda params, n: path == "fused_passes"))
    if path == "torch":
        monkeypatch.setattr(SumParameterization, "_fused_em_correction", lambda self, *a, **k: None)
    pc32, pc16, V, C, gated = _twins("gated")
    assert len(pc16._denom_correction_nss()) > 0                         # (the conditional M-step runs)
    for step_size, pcount, kz in ((1.0, 0.1, False), (0.5, 0.1, False), (0.5, 0.1, True)):
        m16, p32 = _step(pc32, pc16, V, C, gated, step_size, pcount, kz)
        assert _close(m16, p32), (step_size, pcount, kz, float(((m16 - p32).abs() / p32.abs().clamp_min(1e-12)).max()))
    m16, p32 = _step(pc32, pc16, V, C, gated, 1.0, 0.1, False, correct = False)
    assert not _close(m16, p32)
