"""
Evidence pruning (:mod:`pyjuice.constraints.backends.lifted.kernels.live`): every sample's live columns against a
brute force over strings, and the marginal against the reference under every pattern of evidence -- none, all, a
prefix, a suffix, scattered, and a different one per sample -- on every kind of PC, with the buffers filled with NaN
before each call (a dead slot must still be written, -inf). Also: pruning engages when tokens are observed, and
does nothing when none is.
"""
import itertools
import random

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted.kernels.live import prune

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
    import pyjuice.constraints.backends.lifted.forward as F
    from pyjuice.constraints.backends.lifted.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "DENSE_PRUNE_MIN_COLUMNS", 0 if request.param == "dense_pruned" else 1 << 30)
    return request.param


@pytest.fixture(params = ["classes", "grouped"])
def transitions(request, monkeypatch):
    import pyjuice.constraints.backends.lifted.forward as F
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 1 << 30 if request.param == "classes" else 0)
    return request.param


@pytest.fixture(params = ["per_sample", "per_boundary"])
def liveness(request, monkeypatch):
    """The liveness sweeps in one program per sample, or as a launch per boundary (used from far wider automata than
    the test ones by default)."""
    from pyjuice.constraints.backends.lifted.kernels import live
    monkeypatch.setattr(live, "LIVE_STEPS_MIN_WIDTH", 1 << 30 if request.param == "per_sample" else 0)
    return request.param


def marginal(cc, data, missing):
    """The marginal with the buffers filled with NaN first."""
    bufs = cc._buffers(data.size(0))
    bufs["node_mars"].fill_(float("nan"))
    bufs["element_mars"].fill_(float("nan"))
    return cc.marginal(data, missing)


def obs_classes(cc, data, missing):
    dev = cc.pc.device
    x = torch.where(missing, 0, data).to(dev)
    return torch.where(missing.to(dev), -1, cc.token_classes.token_class[x]).contiguous()


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


def test_liveness_matches_brute_force(build_pc, liveness):
    compared = 0
    for seed in range(6):
        rng = random.Random(seed)
        K, n = 7, 6
        dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0,
                                rng.sample(range(K), rng.randint(1, 3)))
        cc = jc.compile(dfa, build_pc("hmm", n, V))
        if not cc.satisfiable:
            continue
        prog = cc._lifted_program()
        for pattern in ("prefix", "suffix", "scattered", "mixed"):
            data, missing = evidence(pattern, n, B = 8, seed = seed)
            pruning = prune(prog, obs_classes(cc, data, missing), data.size(0))
            want = brute_live(dfa, cc.layout, data, missing)
            got = pruning.live.cpu() != 0
            for t in range(n):
                assert torch.equal(got[:, t, :int(cc.layout.width[t])], want[:, t, :int(cc.layout.width[t])]), \
                    (seed, pattern, t)
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
def test_a_wide_automaton_under_evidence_matches_the_reference(kind, pattern, sum_path, liveness, build_pc, reference):
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
    """The liveness sweeps as a launch per boundary give the one-program-per-sample sweeps' live columns, and so the
    same live slots, under every pattern of evidence (a 48-state automaton: boundaries of several tiles)."""
    from pyjuice.constraints.backends.lifted.kernels import live
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    prog = cc._lifted_program()
    widths = [int(w) for w in cc.layout.width.tolist()[:n]] + [1]
    monkeypatch.setattr(live, "LIVE_STEP_TILE", 8)                          # several tiles per boundary
    assert max(widths) > 8
    launches, forward = [], live._live_forward_kernel
    monkeypatch.setattr(live, "_live_forward_kernel", type("Counting", (), {
        "__getitem__": lambda self, grid: launches.append(1) or forward[grid]})())
    for pattern in PATTERNS:
        data, missing = evidence(pattern, n)
        oc = obs_classes(cc, data, missing)
        got = {}
        for name, min_width in (("per_sample", 1 << 30), ("per_boundary", 0)):
            monkeypatch.setattr(live, "LIVE_STEPS_MIN_WIDTH", min_width)
            launches.clear()
            got[name] = prune(prog, oc, data.size(0))
            assert len(launches) == (n if name == "per_boundary" else 0)
        a, b = got["per_sample"], got["per_boundary"]
        for t, w in enumerate(widths):
            assert torch.equal(a.live[:, t, :w], b.live[:, t, :w]), (pattern, t)
        assert torch.equal(a.perm, b.perm) and torch.equal(a.rank, b.rank) and torch.equal(a.counts, b.counts)


def test_pruning_engages_only_under_evidence(build_pc, monkeypatch):
    """An observed prefix pins every sample to one column per boundary before it: the sums compute far fewer
    columns. With nothing observed, nothing is pruned (and nothing computed for it)."""
    import pyjuice.constraints.backends.lifted.forward as F
    calls, pruned = [], F.prune
    monkeypatch.setattr(F, "prune", lambda *a: calls.append(1) or pruned(*a))
    n = 8
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    prog = cc._lifted_program()
    data, missing = evidence("none", n)
    cc.marginal(data, missing)
    assert calls == []
    cc.marginal(data)                                            # everything observed
    assert calls == [1]
    data, missing = evidence("prefix", n)
    pruning = prune(prog, obs_classes(cc, data, missing), data.size(0))
    full = [data.size(0) * s for s in prog.sum_iv_slots]
    assert sum(pruning.num_live) < 0.7 * sum(full)
    live = pruning.live.cpu()
    for t in range(n // 2 + 1):                                  # through the observed prefix: one column each
        assert (live[:, t].sum(dim = 1) <= 1).all()
