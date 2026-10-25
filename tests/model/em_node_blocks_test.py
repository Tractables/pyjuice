"""
Mini-batch EM's node-block update (`pyjuice.model.backend.em_blocks`) against float64 and against the per-block kernels
it replaces (`par_update.em_par_update`, `params_master.em_par_update_master`), from one state: every kernel form (the
two-pass reference and each one-pass tile, forced in turn), fp32 parameters and bf16 ones through their master copy,
node blocks from 1 to 128 nodes (the small ones the per-block kernels'), step sizes, pseudocounts, keep-zero, ranges
excluded (the corrected ones of the external-parameter M-step), and the update agreeing with itself run to run to the
bit (no atomics).
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.layer.kernels.autotune as autotune
from pyjuice.model.backend import em_blocks
from pyjuice.model.backend.par_update import em_par_update
from pyjuice.model.backend.params_master import em_par_update_master

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
CFGS = [("two", em_blocks.BLOCK_N, 4)] + [("one", w, nw) for w, nw in em_blocks.ONE_PASS_CFGS]


def _ref64(pc, params, flows, step_size, pseudocount, keep_zero, exclude = ()):
    """The update in float64 from the per-block tables (every entry: up to 32 children of one node)."""
    ps, pfs, bsz, bint, gnid, nchs, _, _ = pc.par_update_kwargs
    ar = torch.arange(32, device = DEV)
    m = ar[None, :] < bsz[:, None]
    for lo, hi in exclude:
        m &= ~((ps >= lo) & (ps < hi))[:, None]
    pidx = (ps[:, None] + ar[None, :] * bint[:, None])[m]
    fidx = (pfs[:, None] + ar[None, :] * bint[:, None])[m]
    node = gnid[:, None].expand(-1, 32)[m]
    nch = nchs[:, None].expand(-1, 32)[m].double()
    old = params.double()[pidx]
    f = flows.double()[fidx]
    if keep_zero:
        f = f + pseudocount / nch
    contrib = torch.where(old < 1e-12, 0.0, f) if keep_zero else f
    tot = torch.zeros([int(gnid.max()) + 1], dtype = torch.float64, device = DEV).index_add_(0, node, contrib)
    newp = f / tot[node] if keep_zero else (f + pseudocount / nch) / (tot[node] + pseudocount)
    upd = (1 - step_size) * old + step_size * newp
    if keep_zero:
        upd = torch.where(old < 1e-12, 0.0, upd)
    out = params.double().clone()
    out[pidx] = upd
    return out


def _circuit(kind):
    torch.manual_seed(0)
    if kind == "hclt":
        x = torch.randint(0, 32, (512, 24))
        ns = juice.structures.HCLT(x.float(), num_latents = 64, input_dist = dists.Categorical(num_cats = 32)); V, C = 24, 32
    elif kind == "hmm":
        ns = juice.structures.HMM(seq_length = 16, num_latents = 128, num_emits = 32); V, C = 16, 32
    else:
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 16, split_intervals = 2,
                                 input_dist = dists.Categorical(num_cats = 16)); V, C = 16, 16
    ns.init_parameters(perturbation = 2.0)
    return ns, V, C


def _state(ns, V, C, dtype):
    pc = juice.compile(ns, verbose = False, param_dtype = dtype).to(DEV)
    x = torch.randint(0, C, (64, V), device = DEV)
    pc.init_param_flows(flows_memory = 0.0)
    pc(x); pc.backward(x, flows_memory = 1.0)
    return pc, pc.param_flows.clone()


def _run(pc, flows, start, step_size, pcount, kz, exclude, path):
    """One update from ``start`` by ``path`` ("old": the per-block kernels; else em_blocks), the result in fp32 (the
    master copy below fp32)."""
    pc.params.data.copy_(start)
    master = start.float().clone() if start.dtype != torch.float32 else None
    if path == "old":
        kwargs = pc._par_update_kwargs_excluding(list(exclude))
        if master is None:
            em_par_update(pc.params.data, flows, kwargs, step_size = step_size, pseudocount = pcount, keep_zero_params = kz)
        else:
            em_par_update_master(master, pc.params.data, flows, kwargs, step_size, pcount, kz)
    else:
        em_blocks.em_update(pc, flows, step_size, pcount, kz, exclude = exclude, master = master)
    cum = pc.par_update_kwargs[6].clone()                                 # every node's total, as left
    if master is not None:
        assert torch.equal(pc.params.data, master.to(start.dtype)), "the stored values are not the master's rounding"
        return master.double(), cum
    return pc.params.detach().double().clone(), cum


@cuda_only
@pytest.mark.parametrize("cfg", CFGS, ids = lambda c: f"{c[0]}{c[1]}x{c[2]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids = ["fp32", "bf16"])
@pytest.mark.parametrize("kind", ["hclt", "hmm", "pd"])
def test_every_form_is_the_update(cfg, dtype, kind, monkeypatch):
    real = autotune.cached
    monkeypatch.setattr(autotune, "cached", lambda key: cfg if isinstance(key, tuple) and key[:1] == ("em_node_block",) else real(key))
    ns, V, C = _circuit(kind)
    pc, flows = _state(ns, V, C, dtype)
    sums = [n for n in pc.root_ns if n.is_sum() and not n.is_tied()]
    assert any(n.block_size >= em_blocks.MIN_BLOCK_SIZE for n in sums)    # (the node-block kernels run)
    for step_size, pcount, kz, excl in ((1.0, 0.1, False, ()), (0.1, 0.1, False, ()), (0.01, 0.0, False, ()),
                                        (0.5, 0.1, True, ()), (0.3, 0.1, False, (sums[-2]._param_range,))):
        start = pc.params.detach().clone()
        if kz:
            start[torch.rand(start.shape, device = DEV) < 0.1] = 0.0           # parameters at zero stay there
        want = _ref64(pc, start.float(), flows, step_size, pcount, kz, excl)
        old, cum_old = _run(pc, flows, start, step_size, pcount, kz, excl, "old")
        new, cum_new = _run(pc, flows, start, step_size, pcount, kz, excl, "new")
        again, _ = _run(pc, flows, start, step_size, pcount, kz, excl, "new")
        # the nodes' totals left in `par_update_kwargs`' buffer as the per-block kernels leave them (0 where excluded)
        assert torch.allclose(cum_new, cum_old, atol = 1e-6, rtol = 1e-5), float((cum_new - cum_old).abs().max())
        e_old, e_new = float((old - want).abs().max()), float((new - want).abs().max())
        assert e_new <= 2 * e_old + 1e-9, (step_size, pcount, kz, excl, e_old, e_new)
        assert torch.equal(new, again)                                     # one order of summation: to the bit
        if excl:                                                           # an excluded range is left as it was
            lo, hi = excl[0]
            assert torch.equal(new[lo:hi], start[lo:hi].double())


@cuda_only
def test_mini_batch_em_takes_it_and_the_switch_restores_the_per_block_kernels(monkeypatch):
    ns, V, C = _circuit("hmm")
    pc, flows = _state(ns, V, C, torch.float32)
    start = pc.params.detach().clone()
    calls = []
    real = em_blocks.em_update
    monkeypatch.setattr(em_blocks, "em_update", lambda *a, **k: calls.append(1) or real(*a, **k))
    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    new = pc.params.detach().clone()
    assert calls == [1]
    pc.params.data.copy_(start); pc.param_flows.copy_(flows)
    monkeypatch.setattr(em_blocks, "ENABLED", False)
    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert calls == [1]                                                    # the per-block kernels, as before
    assert torch.allclose(pc.params, new, atol = 1e-7, rtol = 1e-6), float((pc.params - new).abs().max())
