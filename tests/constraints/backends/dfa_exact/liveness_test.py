"""
Evidence pruning (:mod:`pyjuice.constraints.backends.dfa_exact.kernels.live`): every sample's live columns against a
brute force over strings, and the marginal against the reference under every pattern of evidence -- none, all, a
prefix, a suffix, scattered, and a different one per sample -- on every kind of PC, with the buffers filled with NaN
before each call (a dead slot must still be written, -inf). A query's :class:`Liveness`: every (sample, boundary)'s
live columns listed, every interval's live count, and the lane order of every interval or of some only. Also: pruning
engages when tokens are observed, and does nothing when none is. And the marginal survives blocks of a wide dynamic
range (see ``test_marginals_survive_a_wide_dynamic_range``).
"""
import itertools
import math
import random

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.constraints.backends.dfa_exact.queries.marginal as M
import pyjuice.constraints.backends.dfa_exact.compiled.program as P
from pyjuice.constraints.backends.dfa_exact.kernels.live import liveness

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8,
         "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5}         # see conftest.py
PATTERNS = ["none", "all", "prefix", "suffix", "scattered", "mixed"]


def evidence(pattern, n, B = 6, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.ones(B, n, dtype = torch.bool)
    for b in range(B):
        p = PATTERNS[b % 5 + 1] if pattern == "mixed" else pattern
        if p == "all":
            missing[b] = False
        elif p == "prefix":
            missing[b, :n // 2] = False
        elif p == "suffix":
            missing[b, n // 2:] = False
        elif p == "scattered":
            missing[b] = torch.rand(n, generator = g) < 0.5
    return data, missing


def assert_close(got, want, n):
    assert got.shape == want.shape
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert torch.isfinite(got[fin]).all()
    tol = 1e-5 * want[fin].abs() + n * 1e-6
    assert ((got[fin] - want[fin]).abs() <= tol).all(), (got[fin] - want[fin]).abs().max()


@pytest.fixture(params = ["fused", "dense", "dense_pruned"])
def sum_path(request, monkeypatch):
    """The sum layers through the fused kernel, or the dense path wherever blocks qualify, with its live columns
    only (``dense_pruned``) or all of them -- the test batches are below the size dense pruning starts from."""
    import pyjuice.constraints.backends.dfa_exact.queries.forward as F
    from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", (lambda ncols, live: live < ncols) if request.param == "dense_pruned" else
                        (lambda ncols, live: False))
    return request.param


@pytest.fixture(params = ["classes", "grouped"])
def transitions(request, monkeypatch):
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 1 << 30 if request.param == "classes" else 0)
    return request.param


@pytest.fixture(params = ["cuda", "per_sample", "per_boundary"])
def sweeps(request, monkeypatch):
    """The liveness sweeps: the CUDA kernel's (a CTA per sample and sweep, the default where the extension builds), one
    Triton program per sample, or a launch per boundary (used from far wider automata than the test ones by
    default)."""
    from pyjuice.constraints.backends.dfa_exact.kernels import live
    from pyjuice.constraints.backends.dfa_exact.kernels.c import get_module
    if request.param == "cuda" and get_module() is None:
        pytest.skip("the dfa_exact CUDA extension is unavailable")
    monkeypatch.setattr(live, "LIVE_CUDA", request.param == "cuda")
    monkeypatch.setattr(live, "LIVE_STEPS_MIN_WIDTH", 0 if request.param == "per_boundary" else 1 << 30)
    return request.param


def marginal(cc, data, missing):
    """The marginal with the buffers filled with NaN first."""
    bufs = cc.backend.state.buffers(data.size(0))
    bufs["node_mars"].fill_(float("nan"))
    bufs["element_mars"].fill_(float("nan"))
    return cc.marginal(data, missing, storage = "dense")


def obs_classes(cc, data, missing):
    dev = cc.pc.device
    x = torch.where(missing, 0, data).to(dev)
    return torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).contiguous()


def brute_live(dfa, L, data, missing):
    """[B, n + 1, W] bool: column i at boundary t is reachable through the sample's evidence on [0, t) and can reach
    acceptance through its evidence on [t, n) (boundary n: the sequence end, column 0)."""
    B, n = data.shape
    K = dfa.num_states
    step = lambda q, v: dfa.run([v], state = q)
    out = torch.zeros(B, n + 1, L.max_width, dtype = torch.bool)
    for b in range(B):
        allowed = [range(V) if missing[b, t] else [int(data[b, t])] for t in range(n)]
        fwd = [{dfa.initial}]
        for t in range(n):
            fwd.append({step(q, v) for q in fwd[-1] for v in allowed[t]})
        bwd = [None] * n + [{q for q in range(K) if dfa.accept[q]}]
        for t in range(n - 1, -1, -1):
            bwd[t] = {q for q in range(K) if any(step(q, v) in bwd[t + 1] for v in allowed[t])}
        for t in range(n):
            for i, q in enumerate(L.state_id[t, :int(L.width[t])].tolist()):
                out[b, t, i] = q in fwd[t] and q in bwd[t]
        out[b, n, 0] = bool(fwd[n] & bwd[n])
    return out


def test_liveness_matches_brute_force(build_pc, sweeps):
    compared = 0
    for seed in range(6):
        rng = random.Random(seed)
        K, n = 7, 6
        dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0,
                                rng.sample(range(K), rng.randint(1, 3)))
        cc = jc.compile(dfa, build_pc("hmm", n, V))
        if not cc.backend.satisfiable:
            continue
        prog = cc.backend.program()
        for pattern in ("prefix", "suffix", "scattered", "mixed"):
            data, missing = evidence(pattern, n, B = 8, seed = seed)
            lv = liveness(prog, obs_classes(cc, data, missing), data.size(0))
            want = brute_live(dfa, cc.backend.layout, data, missing)
            got = lv.live.cpu() != 0
            for t in range(n):
                w = int(cc.backend.layout.width[t])
                assert torch.equal(got[:, t, :w], want[:, t, :w]), (seed, pattern, t)
            assert torch.equal(got[:, n, 0], want[:, n, 0]), (seed, pattern)
            compared += 1
    assert compared > 10


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("kind", list(KINDS))
def test_marginal_under_evidence_matches_the_reference(kind, pattern, sum_path, transitions, build_pc, reference):
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)),
                    build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    assert_close(marginal(cc, data, missing), reference.marginal(cc, data, missing), n)


@pytest.mark.parametrize("pattern", ["prefix", "suffix", "mixed"])
@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated", "hmm", "hand_left"])
def test_a_wide_automaton_under_evidence_matches_the_reference(kind, pattern, sum_path, sweeps, build_pc, reference):
    """A 48-state automaton: blocks span several tiles, so block @ block skips tiles and chunks without a live
    column, and the fused sums whole tiles of dead columns."""
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    assert_close(marginal(cc, data, missing), reference.marginal(cc, data, missing), n)


@pytest.mark.parametrize("kind", list(KINDS))
def test_per_boundary_liveness_equals_per_sample(kind, build_pc, monkeypatch):
    """The liveness sweeps as a launch per boundary, and the CUDA kernel's (where the extension builds), give the
    one-program-per-sample sweeps' live columns, and so the same live slots, lists and lane order, under every pattern
    of evidence (a 48-state automaton: boundaries of several tiles)."""
    from pyjuice.constraints.backends.dfa_exact.kernels import live
    from pyjuice.constraints.backends.dfa_exact.kernels.c import get_module
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    prog = cc.backend.program()
    widths = [int(w) for w in cc.backend.layout.width.tolist()[:n]] + [1]
    monkeypatch.setattr(live, "LIVE_STEP_TILE", 8)                          # several tiles per boundary
    assert max(widths) > 8
    launches, forward = [], live._live_forward_kernel
    monkeypatch.setattr(live, "_live_forward_kernel", type("Counting", (), {
        "__getitem__": lambda self, grid: launches.append(1) or forward[grid]})())
    for pattern in PATTERNS:
        data, missing = evidence(pattern, n)
        oc = obs_classes(cc, data, missing)
        got = {}
        arms = [("per_sample", 1 << 30, False), ("per_boundary", 0, False)]
        if get_module() is not None:
            arms.append(("cuda", 1 << 30, True))
        for name, min_width, cuda in arms:
            monkeypatch.setattr(live, "LIVE_STEPS_MIN_WIDTH", min_width)
            monkeypatch.setattr(live, "LIVE_CUDA", cuda)
            assert (live._cuda_sweep(prog) is not None) == cuda                # (the arm's path taken)
            launches.clear()
            got[name] = liveness(prog, oc, data.size(0))
            assert len(launches) == (n if name == "per_boundary" else 0)
        a = got["per_sample"]
        for name in got:
            if name == "per_sample":
                continue
            b = got[name]
            for t, w in enumerate(widths):
                assert torch.equal(a.live[:, t, :w], b.live[:, t, :w]), (name, pattern, t)
            assert all(torch.equal(x, y) for x, y in zip(listed(a), listed(b))), (name, pattern)
            every = np.ones(len(prog.sum_ivs), dtype = bool)
            oa, ob = a.order(every), b.order(every)
            assert torch.equal(oa.perm, ob.perm) and torch.equal(oa.rank, ob.rank) and torch.equal(oa.counts, ob.counts)


def listed(lv):
    """Every (sample, boundary) row's live columns as :meth:`Liveness.lists` lists them, on the host."""
    count, start, columns = (x.cpu() for x in lv.lists())
    return [columns[int(s):int(s) + int(c)] for s, c in zip(start, count)]


def interval_flags(cc, live, a, b):
    """bool [B * slots] (sample-major): the live slots of the interval ``[a, b)``, both ends live."""
    layout, n = cc.backend.layout, cc.backend.n
    pos = layout.reachability.positions(a, b)
    X = 1 if b == n else int(layout.width[b])
    pairs = torch.arange(int(layout.width[a]) * X) if pos is None else torch.nonzero(pos.flatten() >= 0).flatten()
    return ((live[:, a, pairs // X] != 0) & (live[:, b, pairs % X] != 0)).flatten()


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_permuted", "left_linear"])
def test_lane_order(kind, subset, build_pc):
    """The lane order of every sum interval, or of some (the others -1, their regions 0), against the interval's live
    slots (both ends live, sample-major): ``perm`` the live columns in order then the dead ones, ``rank`` their places
    among the live ones (-1 dead), the interval's and every region's live count, every region's first column -- with a
    batch whose intervals span several list tiles."""
    from pyjuice.constraints.backends.dfa_exact.kernels import live as L
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc.backend.program()
    data, missing = evidence("mixed", n, B = 512)
    B = data.size(0)
    lv = liveness(prog, obs_classes(cc, data, missing), B)
    K = len(prog.sum_ivs)
    chosen = np.random.default_rng(0).random(K) < 0.5 if subset else np.ones(K, dtype = bool)
    torch.full(((1 << 18) - 256,), -7, dtype = torch.int32, device = prog.dev)   # freed: no zeros left to reuse
    order = lv.order(chosen)
    assert lv.lanes is order
    live = lv.live.cpu()
    perm, rank = order.perm.cpu(), order.rank.cpu()
    assert max(B * s for s in prog.sum_iv_slots) > 2 * L.LIST_TILE
    for k, (a, b) in enumerate(prog.sum_ivs):
        if not chosen[k]:
            assert order.first[k] == -1 and order.index[k] == -1
            continue
        flag = interval_flags(cc, live, a, b)
        f, size = order.first[k], flag.numel()
        want = torch.cat([torch.nonzero(flag).flatten(), torch.nonzero(~flag).flatten()]).to(torch.int32)
        assert torch.equal(perm[f:f + size], want), k
        assert torch.equal(rank[f:f + size], torch.where(flag, torch.cumsum(flag, 0) - 1, -1).to(torch.int32)), k
        assert order.num_live(k) == int(flag.sum())
    reg_iv = np.asarray(prog.reg_iv_host)
    assert order.reg_live.cpu().tolist() == [order.num_live(k) if chosen[k] else 0 for k in reg_iv]
    assert order.reg_first.cpu().tolist() == [order.first[k] if chosen[k] else 0 for k in reg_iv]
    assert chosen.any() and (subset == (not chosen.all()))


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_permuted", "left_linear"])
def test_live_lists_and_counts(kind, build_pc):
    """Every (sample, boundary)'s live columns listed, ascending, below the boundary's width (``lists()``: each row
    where its columns would be were every one live), and every interval's live count (``num_live``, intervals keeping
    every pair or not), against the flags -- with samples whose evidence breaks the constraint (nothing live) and an
    odd batch."""
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc.backend.program()
    data, missing = evidence("mixed", n, B = 37)
    data[:4] = 1                                                  # 1 1 1 ... never contains "1 2"
    missing[:4] = False
    B = data.size(0)
    lv = liveness(prog, obs_classes(cc, data, missing), B)
    live = lv.live.cpu()
    assert lv.lists() is lv.lists()                                 # built once
    count, start, columns = (x.cpu() for x in lv.lists())
    rows = listed(lv)
    widths = [int(w) for w in cc.backend.layout.width.tolist()[:n]] + [1]
    assert columns.numel() == B * sum(widths)
    for b in range(B):
        for t, w in enumerate(widths):
            row = b * (n + 1) + t
            want = torch.nonzero(live[b, t, :w]).flatten().to(torch.int32)
            assert int(start[row]) == b * sum(widths) + sum(widths[:t]) and int(count[row]) == want.numel()
            assert torch.equal(rows[row], want), (b, t)
    assert int(count.view(B, n + 1)[:4].sum()) == 0                # the broken samples: nothing live
    assert lv.num_live == [int(interval_flags(cc, live, a, b).sum()) for a, b in prog.sum_ivs]
    layout = cc.backend.layout
    assert kind != "pd" or any(layout.reachability.positions(a, b) is not None for a, b in prog.sum_ivs)


def test_pruning_engages_only_under_evidence(build_pc, monkeypatch):
    """An observed prefix pins every sample to one column per boundary before it: the sums compute far fewer
    columns. With nothing observed, nothing is pruned (and nothing computed for it)."""
    import pyjuice.constraints.backends.dfa_exact.queries.marginal as M
    calls, real = [], M.liveness
    monkeypatch.setattr(M, "liveness", lambda *a: calls.append(1) or real(*a))
    n = 8
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    prog = cc.backend.program()
    data, missing = evidence("none", n)
    cc.marginal(data, missing, storage = "dense")
    assert calls == []
    cc.marginal(data, storage = "dense")                                            # everything observed
    assert calls == [1]
    data, missing = evidence("prefix", n)
    lv = liveness(prog, obs_classes(cc, data, missing), data.size(0))
    full = [data.size(0) * s for s in prog.sum_iv_slots]
    assert sum(lv.num_live) < 0.7 * sum(full)
    live = lv.live.cpu()
    for t in range(n // 2 + 1):                                  # through the observed prefix: one column each
        assert (live[:, t].sum(dim = 1) <= 1).all()


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated", "hand", "hmm"])
def test_marginals_survive_a_wide_dynamic_range(kind, storage, build_pc, reference, monkeypatch):
    """A block's entries can span far more than fp32's exponent. Here tokens 1 and 2 have probability e^-50 at every
    position under the constraint "contains 1 2": the satisfied state carries nearly all of a block's mass, a state
    still owing the keyword e^-100 of it. A sample whose observed prefix leaves it in such a state at a block's boundary
    needs exactly those small entries -- which block @ block's row and column shifts flush unless the tile is computed
    again exactly, and which the finite values of the sample's dead columns in a block kept once would flush as well,
    were they not masked. Against the reference in float64, kept once (inside fast_inference) and per sample, dense or
    compact (whose block @ block products compute again exactly one shared column at a time)."""
    n = KINDS[kind]
    pc = build_pc(kind, n, V)
    with torch.no_grad():
        for layer in pc.input_layer_group:                                 # P(0) ~ 1, P(1) = P(2) = e^-50 everywhere
            table = layer.params.view(-1, V)
            table[:, 1:] = math.exp(-50.0)
            table[:, 0] = 1.0 - 2 * math.exp(-50.0)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    t = n // 2                                                              # prefixes end at a boundary of a kept block
    data = torch.zeros(4, n, dtype = torch.long)
    data[2, t - 1] = 1                                                      # ... owing "2": state 1
    data[3, 0], data[3, 1] = 1, 2                                           # ... satisfied: state 2
    missing = torch.arange(n)[None, :] >= torch.tensor([0, t, t, t])[:, None]   # sample 0: nothing observed
    monkeypatch.setitem(reference.marginal.__globals__, "DTYPE", torch.float64)
    want = reference.marginal(cc, data, missing).float()
    assert torch.isfinite(want).all()                                        # every mass ~ e^-100, finite
    with juice.fast_inference():
        kept = cc.marginal(data, missing, storage = storage)
    real = M.prior_regions
    try:
        M.prior_regions = lambda prog, observed, B, kept, liveness: np.zeros(len(prog.sum_ivs), dtype = bool)
        per_sample = cc.marginal(data, missing, storage = storage)
    finally:
        M.prior_regions = real
    for got in (kept, per_sample):
        assert torch.isfinite(got).all(), got
        assert ((got - want).abs() <= 1e-5 * want.abs() + 1e-4).all(), (got - want)
