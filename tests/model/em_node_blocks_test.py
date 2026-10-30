"""
Mini-batch EM's node-block update (`pyjuice.model.backend.em_blocks`) against float64 and against the per-block kernels
it replaces (`par_update.em_par_update`, `params_master.em_par_update_master`), from one state: every kernel form (the
split kernels and both one-pass tiles, forced in turn), fp32 parameters and bf16 ones through their master copy,
node blocks from 1 to 128 nodes (the small ones the per-block kernels'), step sizes, pseudocounts, keep-zero, ranges
excluded (the corrected ones of the external-parameter M-step), the update agreeing with itself run to run to the bit
(no atomics), and its form a rule of the shape, never a measurement.
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
CFGS = [("split", em_blocks.BLOCK_N, 4), ("one",) + em_blocks.ONE_PASS_TILE, ("one",) + em_blocks.ONE_PASS_TILE_MASTER]


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
        ns = juice.structures.HCLT(x.float(), num_latents = 64, input_dist = dists.Categorical(num_cats = 32))
        V, C = 24, 32
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
            em_par_update(pc.params.data, flows, kwargs, step_size = step_size, pseudocount = pcount,
                          keep_zero_params = kz)
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
    monkeypatch.setattr(em_blocks, "_form",                           # (the one-pass forms need a children bucket)
                        lambda nch_bucket, has_master: cfg if cfg[0] != "one" or nch_bucket is not None
                        else ("split", em_blocks.BLOCK_N, 4))
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids = ["fp32", "bf16"])
def test_the_form_is_the_shapes_not_a_timing(dtype, monkeypatch):
    """The forms sum in different orders, so which one runs must not depend on a measurement: two processes updating
    the same values (DDP's ranks) would otherwise come out apart. The update never asks the tuner, and its form is
    `_form`'s rule."""
    ns, V, C = _circuit("hmm")
    pc, flows = _state(ns, V, C, dtype)                                    # (the passes tune their kernels)

    def no_tuning(*args, **kwargs):
        raise AssertionError("the EM update consulted the tuner")
    for name in ("pick", "choose", "best_of", "should_tune"):
        monkeypatch.setattr(autotune, name, no_tuning)
    launched = []
    real = em_blocks._launch
    monkeypatch.setattr(em_blocks, "_launch", lambda cfg, *a, **k: launched.append(cfg) or real(cfg, *a, **k))
    start = pc.params.detach().clone()
    master = start.float().clone() if dtype != torch.float32 else None
    em_blocks.em_update(pc, flows, 0.5, 0.1, False, master = master)
    want = ("one",) + (em_blocks.ONE_PASS_TILE_MASTER if master is not None else em_blocks.ONE_PASS_TILE)
    assert launched and all(c == want for c in launched), launched
    assert em_blocks._form(None, master is not None) == ("split", em_blocks.BLOCK_N, 4)


@cuda_only
def test_mini_batch_em_takes_it(monkeypatch):
    """`mini_batch_em` updates by node blocks, to what the per-block kernels give from the same state."""
    ns, V, C = _circuit("hmm")
    pc, _ = _state(ns, V, C, torch.float32)
    start = pc.params.detach().clone()
    calls = []
    real = em_blocks.em_update
    monkeypatch.setattr(em_blocks, "em_update", lambda *a, **k: calls.append(1) or real(*a, **k))
    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    new = pc.params.detach().clone()
    assert calls == [1]
    # (from the flows as `mini_batch_em` left them: it sums tied copies' flows in place before the update)
    ref, _ = _run(pc, pc.param_flows.clone(), start, 0.5, 0.1, False, (), "old")
    assert torch.allclose(new.double(), ref, atol = 1e-7, rtol = 1e-6), float((new.double() - ref).abs().max())
