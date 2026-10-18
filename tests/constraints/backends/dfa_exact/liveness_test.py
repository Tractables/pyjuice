"""
Evidence pruning (:mod:`pyjuice.constraints.backends.dfa_exact.kernels.live`): every sample's live columns against a
brute force over strings, and the marginal against the reference under every pattern of evidence -- none, all, a
prefix, a suffix, scattered, and a different one per sample -- on every kind of PC, with the buffers filled with NaN
before each call (a dead slot must still be written, -inf). A query's :class:`Liveness`: every (sample, boundary)'s
live columns listed, every interval's live count, and the lane order of every interval or of some only. Also: pruning
engages when tokens are observed, and does nothing when none is. And the marginal survives blocks of a wide dynamic
range (see ``test_marginals_survive_a_wide_dynamic_range``). Where per-sample inputs rule classes out (top-k soft
evidence: a class without a candidate), the sweeps follow only the classes left: their live columns against the brute
force through those classes' tokens, every sample's successor lists against the automaton, and a session's queries
against the same ones through every class.
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
from pyjuice.constraints.backends.dfa_exact.kernels.live import liveness, sample_successors

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
    import pyjuice.constraints.backends.dfa_exact.queries.backend.forward as F
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


def brute_live(dfa, L, data, missing, classes = None, token_class = None):
    """[B, n + 1, W] bool: column i at boundary t is reachable through the sample's evidence on [0, t) and can reach
    acceptance through its evidence on [t, n) (boundary n: the sequence end, column 0) -- a missing token any, or
    (``classes`` [B, n, C] bool) one of a class the sample allows there (``token_class``: every token's)."""
    B, n = data.shape
    K = dfa.num_states
    step = lambda q, v: dfa.run([v], state = q)
    out = torch.zeros(B, n + 1, L.max_width, dtype = torch.bool)
    for b in range(B):
        allowed = [([v for v in range(V) if classes is None or classes[b, t, token_class[v]]] if missing[b, t] else
                    [int(data[b, t])]) for t in range(n)]
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


def random_dfas(count, tokens = V):
    """``count`` random 7-state automata over ``tokens`` tokens (one to three accepting states)."""
    for seed in range(count):
        rng = random.Random(seed)
        K = 7
        yield seed, jc.DFA.from_dense(tokens, [[rng.randrange(K) for _ in range(tokens)] for _ in range(K)], 0,
                                      rng.sample(range(K), rng.randint(1, 3)))


@pytest.mark.parametrize("shared", [False, True])
def test_liveness_through_sample_classes_matches_brute_force(build_pc, sweeps, shared):
    """Where a token is missing, only the classes a sample's inputs leave possible (:func:`sample_successors`, every
    sample's own lists or one set for all): the live columns against the brute force through those classes' tokens."""
    compared, n, B = 0, 6, 8
    for seed, dfa in random_dfas(6):
        cc = jc.compile(dfa, build_pc("hmm", n, V))
        if not cc.backend.satisfiable:
            continue
        prog = cc.backend.program()
        C = prog.next_col.size(2)
        token_class = cc.backend.token_classes.token_class.tolist()
        g = torch.Generator().manual_seed(seed)
        for pattern in ("none", "prefix", "scattered", "mixed"):
            data, missing = evidence(pattern, n, B = B, seed = seed)
            classes = torch.rand(1 if shared else B, n, C, generator = g) < 0.6
            lv = liveness(prog, obs_classes(cc, data, missing), B, sample_successors(prog, classes.to(prog.dev)))
            want = brute_live(dfa, cc.backend.layout, data, missing, classes.expand(B, n, C), token_class)
            got = lv.live.cpu() != 0
            for t in range(n):
                w = int(cc.backend.layout.width[t])
                assert torch.equal(got[:, t, :w], want[:, t, :w]), (seed, pattern, t)
            assert torch.equal(got[:, n, 0], want[:, n, 0]), (seed, pattern)
            compared += 1
    assert compared > 10


@pytest.mark.parametrize("tokens", [V, 70])
def test_sample_successors_are_the_allowed_classes_successors(tokens, build_pc):
    """A sample's successor lists: at every boundary and column, the distinct successors the classes allowed there
    lead to (at the last boundary the sequence end's one column), ascending -- with every class allowed, the program's
    own lists. Over 70 tokens, more classes than one word of class bits holds."""
    n, B = 6, 5
    for seed, dfa in random_dfas(4, tokens):
        cc = jc.compile(dfa, build_pc("hmm", n, tokens))
        if not cc.backend.satisfiable:
            continue
        prog = cc.backend.program()
        nc, width = prog.next_col.cpu(), prog.width
        W, C = nc.shape[1:]
        allowed = torch.rand(B, n, C, generator = torch.Generator().manual_seed(seed)) < 0.5
        lists = sample_successors(prog, allowed.to(prog.dev))
        assert lists.strides() == (n * (W + 1), lists.succ.size(1))
        ptr, succ = lists.ptr.cpu(), lists.succ.cpu()
        for b in range(B):
            assert ptr[b, 0, 0] == 0
            for t in range(n):
                assert t == 0 or ptr[b, t, 0] == ptr[b, t - 1, W]
                for q in range(W):
                    want = sorted({0 if t == n - 1 else int(nc[t, q, c]) for c in range(C)
                                   if allowed[b, t, c] and nc[t, q, c] >= 0}) if q < width[t] else []
                    assert succ[b, ptr[b, t, q]:ptr[b, t, q + 1]].tolist() == want, (seed, b, t, q)
        assert tokens == V or C > 62
        every = sample_successors(prog, torch.ones(1, n, C, dtype = torch.bool, device = prog.dev))
        E = prog.succ.entry_succ.numel()
        assert every.strides() == (0, 0)
        assert torch.equal(every.ptr[0], prog.succ.entry_ptr.view(n, W + 1))
        assert torch.equal(every.succ[0, :E], prog.succ.entry_succ)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_a_sessions_candidates_prune_its_queries(storage, shared, build_pc, monkeypatch):
    """Top-k soft evidence of 2 of the 3 tokens a position leaves classes without a candidate: a decoder session's
    sweeps follow only the classes left -- its candidates' (``class_support``), fewer live slots than through
    every class (``LIVE_SUPPORT`` off) -- and its marginals and conditionals are those through every class, -inf where
    theirs are."""
    import pyjuice.nodes.distributions as dists
    import pyjuice.constraints.backends.dfa_exact.queries.conditional as Q
    from pyjuice.constraints.backends.dfa_exact.kernels import live
    n, B = 6, 6
    pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical)
    cc = jc.compile(jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)), pc)
    dev = cc.pc.device
    g = torch.Generator().manual_seed(0)
    rows = 1 if shared else B
    inputs = dict(categorical_evidence_logp = (torch.randn(rows, n, 2, generator = g) * 2).to(dev),
                  soft_evidence_cat_ids = torch.argsort(torch.rand(rows, n, V, generator = g), dim = 2)[..., :2].to(dev))
    data, missing = evidence("mixed", n, B = B)
    data, missing = data.to(dev), missing.to(dev)
    live_counts, supports = [], []
    real_support = live.class_support
    monkeypatch.setattr(live, "class_support", lambda *a: (lambda c: supports.append(c) or c)(real_support(*a)))
    for module in (M, Q):                               # every query's live slots
        real = module.liveness
        monkeypatch.setattr(module, "liveness", lambda *a, real = real: (
            lambda lv: live_counts.append(sum(lv.num_live)) or lv)(real(*a)))
    out = {}
    for on in (True, False):
        monkeypatch.setattr(live, "LIVE_SUPPORT", on)
        live_counts.clear()
        with cc.decoder(storage = storage, **inputs) as decoder:
            out[on] = (decoder.marginal(data, missing), decoder.conditional(data, missing), sum(live_counts))
    assert_close(out[True][0], out[False][0], n)
    assert_close(out[True][1], out[False][1], n)
    assert torch.isfinite(out[True][0]).any()
    assert out[True][2] < out[False][2]
    token_class = cc.backend.token_classes.token_class
    want = torch.zeros(rows, n, cc.backend.num_classes, dtype = torch.bool, device = dev)
    want.scatter_(2, token_class[inputs["soft_evidence_cat_ids"]].long(), True)
    assert len(supports) == 1 and torch.equal(supports[0], want)           # (once a session)


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


# -------------------------------------------------------------------------------------------------
# Successor lists: a column's entries whose successor is live (the forward's input @ block entry loops)
# -------------------------------------------------------------------------------------------------

VW = 8                                                                  # tokens of the successor lists' automata


def wide_dfa(K = 24, seed = 0):
    """A random ``K``-state automaton over :data:`VW` tokens: columns of several entries, most leading to columns the
    evidence kills."""
    rng = random.Random(seed)
    return jc.DFA.from_dense(VW, [[rng.randrange(K) for _ in range(VW)] for _ in range(K)], 0, rng.sample(range(K), 6))


def entry_form(form, monkeypatch):
    """A missing token's transitions grouped by successor, or a column's live entries (the test automata are dense:
    the entry loop forced)."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0 if form == "grouped" else 1 << 30)
    monkeypatch.setattr(P, "ENTRY_LOOP_MAX_LIVE", 2)


@pytest.mark.parametrize("form", ["grouped", "entries"])
@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_successor_lists_match_a_brute_force(kind, form, build_pc, monkeypatch):
    """Every (sample, boundary, live column)'s successor list -- the column's entries of the Program's transition
    tables whose successor is live at the next boundary, ascending -- and every boundary's choice to take them (over
    the samples), against the liveness flags: an odd batch, a sample whose evidence breaks the constraint (nothing
    live)."""
    from pyjuice.constraints.backends.dfa_exact.kernels import live as L
    entry_form(form, monkeypatch)
    n = KINDS[kind]
    cc = jc.compile(wide_dfa(), build_pc(kind, n, VW))
    prog = cc.backend.program()
    g = torch.Generator().manual_seed(1)
    B = 7
    data = torch.randint(0, VW, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.6
    missing[0] = True
    lv = liveness(prog, obs_classes(cc, data, missing), B)
    live = lv.live.cpu() != 0
    ep = prog.trans.entry_ptr.cpu().long().view(n, -1)
    succ = prog.trans.entry_succ.cpu().long()
    lists, kept, total = {}, [0] * n, [0] * n
    for b in range(B):
        for t in range(n):
            for i in torch.nonzero(live[b, t, :int(prog.width[t])]).flatten().tolist():
                es = torch.arange(int(ep[t, i]), int(ep[t, i + 1]))
                lists[b, t, i] = want = es[live[b, t + 1, succ[es]]]
                kept[t], total[t] = kept[t] + want.numel(), total[t] + es.numel()
    ratios = sorted({kept[t] / total[t] for t in range(n) if total[t] > 0})
    assert lists and len(ratios) > 1
    monkeypatch.setattr(L, "SUCCESSOR_LIST_MAX_LIVE", (ratios[0] + ratios[1]) / 2)   # some take them, some do not
    sl = lv.successor_lists(cc.backend.state)
    assert lv.successor_lists(cc.backend.state) is sl                       # built once
    start, count, entries, use = (x.cpu().long() for x in (sl.start, sl.count, sl.entries, sl.use))
    E = sl.num_entries
    assert E == succ.numel() and entries.numel() == B * E
    for (b, t, i), want in lists.items():
        s0 = b * E + int(ep[t, 0]) + int(start[b, t, i])
        assert torch.equal(entries[s0:s0 + int(count[b, t, i])], want), (b, t, i)
    assert use.tolist() == [int(kept[t] <= L.SUCCESSOR_LIST_MAX_LIVE * total[t]) for t in range(n)]  # over samples
    assert 0 < int(use.sum()) < n
    assert sl.boundaries.tolist() == (use != 0).tolist()                     # (read back: launches compile by them)


@pytest.mark.parametrize("evidence_kind", ["none", "topk"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("form", ["grouped", "entries"])
@pytest.mark.parametrize("kind", ["hmm", "hmm_untied", "pd", "hand", "hand_left", "left_linear"])
def test_successor_lists_change_nothing(kind, form, storage, evidence_kind, build_pc, monkeypatch):
    """Marginals, conditionals and log marginals with the successor lists taken at every boundary (the kernels with
    the lists alone), at every other boundary (the kernels choosing per step), at none, and not built at all: the
    same bits -- the lists drop only terms through a dead successor, -inf (top-k per-sample inputs: the liveness also
    follows their candidate classes)."""
    import pyjuice.nodes.distributions as dists
    from pyjuice.constraints.backends.dfa_exact.kernels import live as L, prod as K
    from pyjuice.constraints.backends.dfa_exact.queries.backend import forward as F
    entry_form(form, monkeypatch)
    n = KINDS[kind]
    topk = evidence_kind == "topk"
    pc = build_pc(kind, n, VW, input_dist = dists.SoftEvidenceCategorical if topk else None)
    cc = jc.compile(wide_dfa(), pc)
    g = torch.Generator().manual_seed(2)
    B, dev = 5, cc.pc.device
    data = torch.randint(0, VW, (B, n), generator = g)
    missing = torch.ones(B, n, dtype = torch.bool)
    missing[1, :n // 2] = False                                             # a prefix, a suffix, scattered
    missing[2, n // 2:] = False
    missing[3] = torch.rand(n, generator = g) < 0.5
    missing[4, 1::2] = False
    inputs = {}
    if topk:
        ids = torch.argsort(torch.rand(B, n, VW, generator = g), dim = 2)[..., :3]
        data = torch.where(missing, data, ids[..., 0])
        inputs = dict(categorical_evidence_logp = (torch.randn(B, n, 3, generator = g) * 2).to(dev),
                      soft_evidence_cat_ids = ids.to(dev))
    data, missing = data.to(dev), missing.to(dev)
    real = L.Liveness.successor_lists

    def alternate(self, state):                                             # every other boundary takes them
        sl = real(self, state)
        sl.use.copy_((torch.arange(sl.use.numel(), device = sl.use.device) % 2 == 0).to(torch.int32))
        sl.boundaries = sl.use.cpu().numpy() != 0
        return sl

    class Spy:                                                              # the modes the entry loops compile
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                modes.add(kwargs["SUCC_LISTS"])
                return self.kernel[grid](*args, **kwargs)
            return launch

    for name in ("_input_block_kernel", "_input_block_group_kernel"):
        monkeypatch.setattr(K, name, Spy(getattr(K, name)))
    spans, products = K.Launch.input_boundaries, F.products
    out, seen = {}, {}
    for arm in ("always", "alternate", "never", "unbuilt"):
        modes = set()
        monkeypatch.setattr(K.Launch, "input_boundaries", spans)
        monkeypatch.setattr(F, "products", products)
        if arm == "unbuilt":
            monkeypatch.setattr(F, "products", lambda *args, succ_lists = None, **kwargs: products(*args, **kwargs))
        elif arm == "alternate":                                            # (every launch as if over every boundary:
            monkeypatch.setattr(L.Liveness, "successor_lists", alternate)   # the steps choose, the launches cannot)
            monkeypatch.setattr(K.Launch, "input_boundaries", property(lambda self: np.arange(n)))
            monkeypatch.setattr(L, "SUCCESSOR_LIST_MAX_LIVE", 2.0)
        else:
            monkeypatch.setattr(L.Liveness, "successor_lists", real)
            monkeypatch.setattr(L, "SUCCESSOR_LIST_MAX_LIVE", 2.0 if arm == "always" else -1.0)
        with cc.decoder(storage = storage, precision = "fp32", **inputs) as d:
            marg = d.marginal(data, missing)
            cond, logm = d.conditional(data, missing, return_log_marginal = True)
        out[arm], seen[arm] = (marg, cond, logm), modes
    assert torch.isfinite(out["always"][1]).any()
    assert 2 in seen["always"] and 1 in seen["alternate"] and seen["never"] == seen["unbuilt"] == {0}, seen
    for arm in ("alternate", "never", "unbuilt"):
        for got, want in zip(out["always"], out[arm]):
            assert torch.equal(got, want), arm
