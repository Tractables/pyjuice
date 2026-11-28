"""
`juice.queries.conditional` and `juice.queries.sample` on circuits with `LowRank(top_k)` parameterizations -- the top_k
tests' HMM (`tests/model/parameterizations/lowrank_top_k_test.py`: transitions whose parameters read the token beside
them, soft-evidence leaves with a top_k tilt, external tensors a sample each) -- against their definitions by the
float64 HMM there, nothing of the queries' code shared with the references:

  * the conditional `[B, targets, V]`: at a missing target the exact posterior P(x_t = c | e) by brute force (each
    candidate observed in turn, normalized) -- where a top_k sum reads the token and at the last position, which none
    reads --; at an observed one the leaf's emission under the state's posterior, sum_i P(z_t = i | e) P(c | i) (plain
    `SoftEvidenceCategorical`'s convention, not a one-hot); each row summing to 1, nothing off the candidates;
  * the draws: unconditionally the exact joint over the candidate tuples (every tuple scored); conditioned on a forward
    with tokens observed, P(x_miss | x_obs) at the missing positions and the observed tokens themselves -- by a
    chi-square test (a false failure ~1e-6 a check, the cells expected fewer than 5 draws pooled), whose power shows
    against the draw of a leaf blind to the slot its sum drew.

Every path of each: the conditional's read-out by torch (`conditional_reference`: 4 nodes, no kernel state) and by
its kernel (128 nodes: the leaves' kernels' forward state; at top_k 64 the sums' kernels too); the draws by the walk of
a run of levels (CUDA, Triton), by the scoped pass's per-layer kernels and by torch (`sample_sums_reference`, forced),
at 4 and 128 nodes. And `external_params_rows`: samples reading one row of the tensors and of the evidence.
"""

import contextlib
import importlib.util
import itertools
import math
import os
import random

import pytest
import torch

from pyjuice.parameterizations.lowrank import top_k_queries, top_k_query_kernels
from pyjuice.queries import conditional, marginal, sample

try:
    from scipy.stats import chi2 as _chi2
except ImportError:                                              # (a bound of the same order instead: `_plausible`)
    _chi2 = None

# The HMM and its float64 definition: the top_k tests' own, loaded by path (the suite's importlib mode puts no test
# directory on `sys.path`), under a name of its own (its tests are not collected again here)
_spec = importlib.util.spec_from_file_location(
    "_lowrank_top_k_reference",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "model", "parameterizations",
                 "lowrank_top_k_test.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEV, V = H.DEV, H.V
DRAW_SEED = 0                    # (the draws' RNG: `random` -- the kernels' seeds -- and torch's)
DRAW_CASE = 1                    # (the draws' cases' seed: one where the slots move the joint enough for the check to
                                 # see a leaf blind to them -- noncentrality >= 180 at 20000 draws in every case, where
                                 # seed 0's is 91 at 128 nodes: `test_the_chi_square_rejects_draws_blind_to_the_slot`)
P_FALSE = 1e-6                   # (a chi-square check's false-failure rate)
NUM_DRAWS = 20000
_READOUT = {4: "reference", 128: "kernel"}


# ---------------------------------------------------------------------------------------------- the cases

def _case(N, C, K, b, seed, T = 4, masked = 0.5, scale = 0.7, V = V, r = 3, re = 2):
    """`(pc, tab, ext, ids, lp, x, miss)`: the HMM of `N` states and `T` positions, its tables, and a batch of `b`
    (`H._batch`: the external tensors, the candidates and their evidence, tokens among them, a mask)."""
    tab = H._tables(N, seed, V)
    pc, _, _ = H._hmm_pc(T, N, K, r, re, tab, V)
    return (pc, tab) + H._batch(b, T, N, C, K, r, re, seed + 1, masked = masked, scale = scale, V = V)


def _kw(ext, ids, lp, N, **extra):
    """The queries' keywords: the external tensors (`H._pc_ext`) and the soft evidence."""
    return dict(external_params = H._pc_ext(ext, ids.size(1), N), categorical_evidence_logp = lp.float(),
                soft_evidence_cat_ids = ids, **extra)


def _needs_kernels(N, sums = False):
    """Skips the 128-node cases where the kernels did not build (the torch passes would serve them unseen)."""
    from pyjuice.parameterizations.lowrank import top_k_fused, top_k_leaf_fused
    if N % 128 == 0 and (top_k_leaf_fused.module() is None or (sums and top_k_fused.top_k_module() is None)):
        pytest.skip("the CUDA modules did not build")


# Half the tokens missing (`masked = 0.5`), placed so the batch holds each kind: every token a sum reads missing with an
# observed one after it (where its slots move its posterior -- with none observed after it, they cancel) and with none,
# the last token (which no sum reads) missing and observed, every position observed somewhere
_HALF_MASKED = {4: [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]],
                3: [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]}


def _half_masked(T):
    return torch.tensor(_HALF_MASKED[T], dtype = torch.bool, device = DEV)


def _slots_matter(miss):
    """Whether some token a sum reads (all but the last) is missing with an observed one after it: where the slots
    move its posterior (with none observed after it, they cancel -- the slot-blind answer is then exact)."""
    observed_after = torch.flip(torch.cummax(torch.flip((~miss).int(), [1]), dim = 1).values, [1]).bool()[:, 1:]
    return bool((miss[:, :-1] & observed_after).any())


# ---------------------------------------------------------------------------------------------- the conditional

def _state_posterior(tab, ext, ids, lp, x, miss, K):
    """`[b, T, N]` P(z_t = i | e) of the float64 HMM: d log p(e) / d (a bias on each position's states), by autograd."""
    A, B, G, U, D = ext
    b, T, C = ids.shape
    beta = torch.zeros(b, T, tab["lT"].size(0), dtype = torch.float64, device = ids.device, requires_grad = True)
    slot = torch.arange(C, device = ids.device).clamp(max = K)
    lv = None
    for t in range(T - 1, -1, -1):
        e = H._emit_terms(H._leaf_logp(tab["lE"], lp[:, t], ids[:, t], U[:, t], D[:, t], K), ids[:, t], x[:, t],
                          miss[:, t])
        if lv is None:
            lv = torch.logsumexp(e, dim = 2)
        else:
            inner = torch.logsumexp(H._trans_logp(tab["lT"], A[:, t], B[:, t], G[:, t]) + lv[:, None, None, :], dim = 3)
            lv = torch.logsumexp(e + inner[:, :, slot], dim = 2)
        lv = lv + beta[:, t]
    torch.logsumexp(tab["lpi"][None, :] + lv, dim = 1).sum().backward()
    return beta.grad


def _definition(tab, ext, ids, lp, x, miss, K):
    """`[b, T, C]` the conditional at each position's candidates by its definition: where the token is missing, brute
    force -- each candidate observed in turn through the float64 HMM, normalized --; where it is observed, the leaf's
    emission under the state's posterior, sum_i P(z_t = i | e) P(c | i)."""
    b, T, C = ids.shape
    U, D = ext[3], ext[4]
    post = _state_posterior(tab, ext, ids, lp, x, miss, K)
    out = torch.zeros(b, T, C, dtype = torch.float64, device = ids.device)
    for t in range(T):
        lls = []
        for c in range(C):
            xc, mc = x.clone(), miss.clone()
            xc[:, t], mc[:, t] = ids[:, t, c], False
            lls.append(H._ref_hmm(tab, ext, ids, lp, xc, mc, K))
        brute = torch.softmax(torch.stack(lls, dim = 1), dim = 1)
        q = H._leaf_logp(tab["lE"], lp[:, t], ids[:, t], U[:, t], D[:, t], K).exp()            # [b, N, C]: P(c | i)
        out[:, t] = torch.where(miss[:, t, None], brute, torch.bmm(post[:, t, None, :], q)[:, 0])
    return out


def _assert_is_definition(probs, defn, ids, num_cats, atol, rtol = 0.0):
    """`probs [b, T, num_cats]` is `defn [b, T, C]` at the candidates' tokens (distinct ones) -- within atol + rtol
    times the probability --, exactly 0 at every other token, each row summing to 1."""
    b, T, C = ids.shape
    assert probs.shape == (b, T, num_cats) and probs.dtype == torch.float32, (probs.shape, probs.dtype)
    cand = torch.zeros(b, T, num_cats, dtype = torch.bool, device = probs.device).scatter_(2, ids, True)
    assert float(probs.masked_fill(cand, 0.0).abs().max()) == 0.0, "mass off the candidates"
    assert float((probs.sum(dim = 2) - 1).abs().max()) < 1e-5, probs.sum(dim = 2)
    err = (probs.double().gather(2, ids) - defn).abs()
    assert bool((err <= atol + rtol * defn.abs()).all()), \
        (float(err.max()), float((err / defn.abs().clamp_min(1e-30)).max()))


@pytest.fixture
def readout(monkeypatch):
    """The read-outs that served a test's conditionals, in order: "kernel" (`top_k_query_kernels.conditional` gave the
    answer), "kernel declined" (it returned None), "reference" (`top_k_queries.conditional_reference`)."""
    served = []
    kernel, reference = top_k_query_kernels.conditional, top_k_queries.conditional_reference

    def by_kernel(*args, **kwargs):
        out = kernel(*args, **kwargs)
        served.append("kernel declined" if out is None else "kernel")
        return out

    def by_reference(*args, **kwargs):
        served.append("reference")
        return reference(*args, **kwargs)

    monkeypatch.setattr(top_k_query_kernels, "conditional", by_kernel)
    monkeypatch.setattr(top_k_queries, "conditional_reference", by_reference)
    return served


@pytest.mark.parametrize("N", [4, 128])                          # (the torch read-out; the kernel's)
@pytest.mark.parametrize("C, K", [(5, 2), (4, 4), (3, 6)])       # (a shared slot; none; more slots than candidates)
@pytest.mark.parametrize("masked", [0.5, 1.0])                  # (`_HALF_MASKED`; every token missing)
def test_conditional_is_the_definition(N, C, K, masked, readout):
    """Every position a target: a missing one's answer the exact posterior of its token -- where a top_k sum reads it
    (the slots' posterior spread over their candidates) and at the last position (none reads it) --, an observed one's
    the leaf's emission under the state's posterior; nothing off the candidates, each row summing to 1. (Every token
    missing, none is observed after a token: its slots cancel, and the case checks the read-out's arithmetic alone.)"""
    _needs_kernels(N)
    T, b = 4, 6
    pc, tab, ext, ids, lp, x, _ = _case(N, C, K, b, 0, T = T)
    miss = _half_masked(T) if masked < 1.0 else torch.ones(b, T, dtype = torch.bool, device = DEV)
    probs = conditional(pc, x, missing_mask = miss, target_vars = list(range(T)), **_kw(ext, ids, lp, N))
    _assert_is_definition(probs, _definition(tab, ext, ids, lp, x, miss, K), ids, V, 1e-5)
    assert readout == [_READOUT[N]], readout


@pytest.mark.parametrize("C", [40, 70])                          # (fewer candidates than slots; a shared slot)
@pytest.mark.parametrize("grad", [False, True])                  # (the counts by the sums' backward; by their forward)
def test_conditional_at_the_kernels_shapes(C, grad, readout, monkeypatch):
    """At the kernels' shapes -- 128 nodes, top_k 64 --, where the sums' kernels write the slots' posterior counts the
    read-out takes (their forward, keeping them for a backward, where the call has grad on; else their backward), the
    tokens missing as `_HALF_MASKED`: the definition, up to the kernels' rounding -- relative, as fp16 logits' is
    (measured <= 1e-3 relative, 1e-4 absolute, at observed and missing tokens alike; the torch passes at these shapes
    ~1e-7)."""
    _needs_kernels(128, sums = True)
    from pyjuice.parameterizations.lowrank import top_k_backward
    sums = []
    terms = top_k_backward.terms
    monkeypatch.setattr(top_k_backward, "terms", lambda *a, **k: (sums.append(1), terms(*a, **k))[1])
    T, N, K, b, V2 = 3, 128, 64, 6, 80
    pc, tab, ext, ids, lp, x, _ = _case(N, C, K, b, 11, T = T, V = V2, r = 20, re = 8)
    miss = _half_masked(T)
    with torch.set_grad_enabled(grad):
        probs = conditional(pc, x, missing_mask = miss, **_kw(ext, ids, lp, N))
    _assert_is_definition(probs, _definition(tab, ext, ids, lp, x, miss, K), ids, V2, 2e-5, rtol = 1e-2)
    assert readout == ["kernel"] and len(sums) == T - 1, (readout, len(sums))      # (the sums' kernels, every one)


@pytest.mark.parametrize("N", [4, 128])
def test_conditional_answers_each_target_in_its_row(N, readout):
    """`target_vars` None: every variable in order, the definition; a subset in another order with a variable listed
    twice: each variable's answer in its row -- the twice-listed one's in its last, the earlier row zero (as plain
    `SoftEvidenceCategorical`'s) --, the answers of the full query."""
    _needs_kernels(N)
    T, C, K, b = 4, 5, 2, 6
    pc, tab, ext, ids, lp, x, miss = _case(N, C, K, b, 4, T = T)
    assert _slots_matter(miss)
    kw = _kw(ext, ids, lp, N)
    full = conditional(pc, x, missing_mask = miss, **kw)
    _assert_is_definition(full, _definition(tab, ext, ids, lp, x, miss, K), ids, V, 1e-5)
    got = conditional(pc, x, missing_mask = miss, target_vars = [T - 1, 0, T - 1, 1], **kw)
    assert got.shape == (b, 4, V), got.shape
    assert float(got[:, 0].abs().max()) == 0.0
    for row, v in ((1, 0), (2, T - 1), (3, 1)):
        assert torch.allclose(got[:, row], full[:, v], atol = 1e-6), (row, v)
    assert readout == [_READOUT[N]] * 2, readout


@pytest.mark.parametrize("N", [4, 128])
@pytest.mark.parametrize("evidence", ["rows", "samples"])        # (the evidence of the rows' batch; a sample's own)
def test_conditional_through_the_rows_map(N, evidence, readout):
    """`external_params_rows`: samples reading one row of the tensors (rows repeated, in any order) -- and of the
    evidence, or each with its own -- answer as the definition does for the tensors and the evidence they read."""
    _needs_kernels(N)
    T, C, K, R = 4, 5, 2, 3
    pc, tab, ext, ids, lp, _, _ = _case(N, C, K, R, 2, T = T)
    rows = torch.tensor([2, 0, 0, 1, 2, 2, 1], device = DEV)
    g = torch.Generator().manual_seed(3)
    pick = torch.randint(0, C, (rows.numel(), T), generator = g).to(DEV)
    x = ids[rows].gather(2, pick[..., None])[..., 0]
    miss = (torch.rand(rows.numel(), T, generator = g) < 0.5).to(DEV)
    assert _slots_matter(miss)
    ev = (ids, lp) if evidence == "rows" else (ids[rows], lp[rows])
    probs = conditional(pc, x, missing_mask = miss, **_kw(ext, *ev, N, external_params_rows = rows))
    defn = _definition(tab, tuple(t[rows] for t in ext), ids[rows], lp[rows], x, miss, K)
    _assert_is_definition(probs, defn, ids[rows], V, 1e-5)
    assert readout == [_READOUT[N]], readout


@pytest.mark.parametrize("N", [4, 128])
def test_conditional_refuses_a_value_mask(N):
    """`soft_evidence_value_mask` (a free position's p w) has no answer under the parameterization: refused."""
    _needs_kernels(N)
    pc, tab, ext, ids, lp, x, miss = _case(N, 5, 2, 6, 0)
    free = torch.zeros(x.shape, dtype = torch.bool, device = DEV)
    with pytest.raises(NotImplementedError, match = "soft_evidence_value_mask"):
        conditional(pc, x, missing_mask = miss, soft_evidence_value_mask = free, **_kw(ext, ids, lp, N))


@pytest.mark.parametrize("N", [4, 128])
def test_a_conditional_leaves_no_external_parameter_gradients(N):
    """The query's backward makes no gradients of the external tensors (`compute_external_grads = False`): those a
    plain backward staged (its default makes them) are gone after a conditional, and none are left."""
    _needs_kernels(N)
    pc, tab, ext, ids, lp, x, miss = _case(N, 5, 2, 6, 0)
    kw = _kw(ext, ids, lp, N)
    with torch.no_grad():
        pc(x, missing_mask = miss, **kw)
    pc.backward(x, missing_mask = miss, logspace_flows = True, allow_modify_flows = False, **kw)
    assert pc._staged_external_params_grad is not None
    conditional(pc, x, missing_mask = miss, **kw)
    assert pc._staged_external_params_grad is None


# ---------------------------------------------------------------------------------------------- the draws

def _seeded():
    random.seed(DRAW_SEED)
    torch.manual_seed(DRAW_SEED)


#: the draws' paths: the run of levels walked by one kernel -- CUDA (the default) or Triton --, the scoped pass's
#: per-layer Triton draws, torch (`_kernels_serve` refused)
DRAW_PATHS = ["walk", "walk-triton", "layers", "torch"]


@contextlib.contextmanager
def _draws_by(path):
    """The draws by `path` (`DRAW_PATHS`) -- and that path alone ran: the walk took its levels (by the CUDA module or
    not), or the per-layer draws ran (by their kernels, or by torch: `sample_sums_reference` and the leaves'
    `_leaf_logq` both called)."""
    called = set()

    def spy(name, fn, ret = False):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            if not ret or out:
                called.add(name)
            return out
        return wrapped

    with pytest.MonkeyPatch.context() as mp:
        for name in ("sample_sums_reference", "_leaf_logq", "sample_sums"):
            mp.setattr(top_k_queries, name, spy(name, getattr(top_k_queries, name)))
        mp.setattr(top_k_queries, "walk", spy("walk", top_k_queries.walk, ret = True))
        cuda_walk = top_k_queries._walk_module()
        if cuda_walk is not None:
            mp.setattr(top_k_queries, "_walk_module",
                       lambda: (called.add("cuda"), cuda_walk)[1] if top_k_queries.WALK_CUDA else None)
        if path == "walk-triton":
            mp.setattr(top_k_queries, "WALK_CUDA", False)
        if path == "layers":
            mp.setattr(top_k_queries, "WALK", False)
        if path == "torch":
            mp.setattr(top_k_queries, "_kernels_serve", lambda *ts: False)
        yield
    want = {"walk": {"walk", "cuda"} if cuda_walk is not None else {"walk"}, "walk-triton": {"walk"},
            "layers": {"sample_sums"}, "torch": {"sample_sums", "sample_sums_reference", "_leaf_logq"}}[path]
    assert called == want, (path, called)


def _repeated(ext, ids, lp, n, N):
    """The keywords of `n` samples, each with the tensors and the evidence of row 0 (a batch of `n`: the draw's)."""
    rep = lambda t: t[:1].expand(n, *t.shape[1:]).contiguous()
    return _kw(tuple(rep(t) for t in ext), rep(ids), rep(lp), N)


def _places(tokens, ids0):
    """`[n, T]` each drawn token's place among its position's candidates `ids0 [T, C]` (asserted to be one of them)."""
    hit = tokens[:, :, None] == ids0[None]
    assert bool(hit.any(dim = 2).all()), "a drawn token none of its position's candidates"
    return hit.int().argmax(dim = 2)


def _all_tuples(T, C):
    return torch.tensor(list(itertools.product(range(C), repeat = T)), device = DEV)


def _joint(tab, ext, ids, lp, K, observed = None):
    """`(tuples [m, T], log p [m])`: every tuple of candidate places of a batch of one (its tensors and evidence),
    scored by the float64 HMM; `observed` {position: place}: the tuples that agree with it alone (their softmax its
    conditional)."""
    T, C = ids.size(1), ids.size(2)
    tuples = _all_tuples(T, C)
    for t, p in (observed or {}).items():
        tuples = tuples[tuples[:, t] == p]
    m = tuples.size(0)
    rep = lambda t: t.expand(m, *t.shape[1:]).contiguous()
    x = rep(ids).gather(2, tuples[..., None])[..., 0]
    none_missing = torch.zeros(m, T, dtype = torch.bool, device = DEV)
    return tuples, H._ref_hmm(tab, tuple(rep(t) for t in ext), rep(ids), rep(lp), x, none_missing, K)


def _slot_blind_joint(tab, ext, ids, lp, K):
    """`(tuples, log p)` as `_joint` gives them, of a WRONG draw: each leaf's token from P(c | z_t) whatever slot the
    sum beside it drew -- the transitions the slots' mixture, sum_s P(s | z) theta(z' | z, s), and each token
    independent of the next state."""
    A, B, G, U, D = ext
    T, C = ids.size(1), ids.size(2)
    lq = [H._leaf_logp(tab["lE"], lp[:, t], ids[:, t], U[:, t], D[:, t], K)[0] for t in range(T)]     # log P(c | z)
    tuples = _all_tuples(T, C)
    a = tab["lpi"][None, :] + lq[0][:, tuples[:, 0]].t()                                             # [m, N]
    for t in range(1, T):
        q = lq[t - 1]
        own = torch.cat([q[:, :K], q.new_full((q.size(0), max(K - C, 0)), -float("inf"))], dim = 1)
        rest = torch.logsumexp(q[:, K:], dim = 1, keepdim = True) if C > K else \
            q.new_full((q.size(0), 1), -float("inf"))
        lth = H._trans_logp(tab["lT"], A[:, t - 1], B[:, t - 1], G[:, t - 1])[0]                       # [N, S, N]
        mix = torch.logsumexp(torch.cat([own, rest], dim = 1)[:, :, None] + lth, dim = 1)               # [N, N]
        a = torch.logsumexp(a[:, :, None] + mix[None], dim = 1) + lq[t][:, tuples[:, t]].t()
    return tuples, torch.logsumexp(a, dim = 1)


def _chi_square(places, tuples, log_p):
    """`(statistic, degrees of freedom)` of the draws' tuples of places `places [n, T']` against softmax(`log_p`) over
    the tuples `tuples [m, T']` -- the cells expected fewer than 5 draws pooled (with the smallest other cell where the
    pool itself is expected fewer) --, every draw asserted to be one of the tuples, of probability > 0."""
    n = places.size(0)
    base = max(int(tuples.max()), int(places.max())) + 1
    digits = base ** torch.arange(tuples.size(1) - 1, -1, -1, device = places.device)
    code = lambda q: (q.long() * digits).sum(dim = 1)
    obs = torch.bincount(code(places), minlength = base ** tuples.size(1))[code(tuples)].double()
    exp = n * torch.softmax(log_p.double(), dim = 0)
    assert int(obs.sum()) == n and float(obs[exp == 0].sum()) == 0.0, "draws off the distribution's support"
    big = exp >= 5
    o, e = obs[big].tolist(), exp[big].tolist()
    so, se = float(obs[~big].sum()), float(exp[~big].sum())
    if se > 0:
        if se >= 5 or not e:
            o.append(so)
            e.append(se)
        else:
            k = min(range(len(e)), key = e.__getitem__)
            o[k] += so
            e[k] += se
    return sum((oi - ei) ** 2 / ei for oi, ei in zip(o, e)), len(e) - 1


def _plausible(stat, dof):
    """Whether a chi-square statistic lies in the bulk of its law: its p-value above `P_FALSE` (without scipy, below
    dof + 8 sqrt(2 dof))."""
    if _chi2 is not None:
        return float(_chi2.sf(stat, dof)) > P_FALSE
    return stat < dof + 8.0 * math.sqrt(2.0 * dof)


def _assert_draws_follow(places, tuples, log_p, what):
    stat, dof = _chi_square(places, tuples, log_p)
    p = "" if _chi2 is None else f", p {float(_chi2.sf(stat, dof)):.1e}"
    assert dof >= 1 and _plausible(stat, dof), f"{what}: chi-square {stat:.1f} on {dof} degrees of freedom{p}"


@pytest.mark.parametrize("N", [4, 128])
@pytest.mark.parametrize("path", DRAW_PATHS)
@pytest.mark.parametrize("C, K", [(4, 2), (3, 4)])               # (two candidates sharing the rest slot; a slot each)
def test_unconditional_draws_follow_the_joint(N, path, C, K):
    """`sample(num_samples = n)` (the tensors and the evidence of row 0, n times): the tuples of tokens follow the exact
    joint -- the sum drawing its slot and child together, the leaf its candidate within that slot."""
    T, n = 3, NUM_DRAWS
    pc, tab, ext, ids, lp, _, _ = _case(N, C, K, 1, DRAW_CASE, T = T, scale = 1.5)
    _seeded()
    with _draws_by(path), torch.no_grad():
        s = sample(pc, num_samples = n, **_repeated(ext, ids, lp, n, N))
    assert s.shape == (n, T), s.shape
    tuples, log_p = _joint(tab, ext, ids, lp, K)
    _assert_draws_follow(_places(s, ids[0]), tuples, log_p, "the joint")


@pytest.mark.parametrize("N", [4, 128])
@pytest.mark.parametrize("path", DRAW_PATHS)
def test_conditional_draws_follow_the_posterior(N, path):
    """`marginal` then `sample(conditional = True)`: the missing tokens jointly follow P(x_miss | x_obs) of the exact
    joint and the observed ones are returned as they are -- none observed; one, its place in the rest slot of the sum
    reading it, in a slot of its own, or at the last position (no sum reads it); all."""
    _needs_kernels(N)
    from pyjuice.parameterizations.lowrank import top_k_leaf_fused
    T, C, K, n = 3, 4, 2, NUM_DRAWS
    pc, tab, ext, ids, lp, _, _ = _case(N, C, K, 1, DRAW_CASE, T = T, scale = 1.5)
    places0 = [3, 1, 2]                                           # (position 0's token in the rest slot, 1's its own)
    x0 = ids[0, torch.arange(T, device = DEV), torch.tensor(places0, device = DEV)]
    kw = _repeated(ext, ids, lp, n, N)
    _seeded()
    for observed in ([], [0], [1], [2], [0, 1, 2]):
        m0 = torch.ones(T, dtype = torch.bool, device = DEV)
        m0[observed] = False
        with _draws_by(path), torch.no_grad():
            marginal(pc, x0[None].expand(n, -1).contiguous(), missing_mask = m0[None].expand(n, -1).contiguous(), **kw)
            # (the tokens' places from the kernels' forward state at 128 nodes; from the torch forward's record at 4)
            assert (top_k_leaf_fused._state(pc.input_layer_group[0], n) is not None) == (N == 128)
            s = sample(pc, conditional = True, **kw)
        assert s.shape == (n, T) and bool((s[:, ~m0] == x0[~m0]).all()), observed
        if m0.any():
            tuples, log_p = _joint(tab, ext, ids, lp, K, observed = {t: places0[t] for t in observed})
            _assert_draws_follow(_places(s, ids[0])[:, m0], tuples[:, m0], log_p, f"observed {observed}")


@pytest.mark.parametrize("N", [4, 128])
@pytest.mark.parametrize("path", DRAW_PATHS)
@pytest.mark.parametrize("cond", [False, True])
def test_draws_through_the_rows_map(N, path, cond):
    """`external_params_rows`: samples reading one of two rows of the tensors and of the evidence (the rows' batch)
    -- each follows its row's joint, unconditionally and conditioned on a forward with a token observed (that one
    returned as it is)."""
    _needs_kernels(N)
    T, C, K, n = 3, 4, 2, NUM_DRAWS
    pc, tab, ext, ids, lp, x, _ = _case(N, C, K, 2, DRAW_CASE, T = T, scale = 1.5)
    rows = torch.arange(n, device = DEV) % 2
    kw = _kw(ext, ids, lp, N, external_params_rows = rows)
    m0 = torch.tensor([True, False, True], device = DEV)          # (the middle token observed, when conditioning)
    _seeded()
    with _draws_by(path), torch.no_grad():
        if cond:
            marginal(pc, x[rows], missing_mask = m0[None].expand(n, -1).contiguous(), **kw)
            s = sample(pc, conditional = True, **kw)
        else:
            s = sample(pc, num_samples = n, **kw)
    assert s.shape == (n, T), s.shape
    if cond:
        assert bool((s[:, 1] == x[rows, 1]).all())
    keep = m0 if cond else torch.ones(T, dtype = torch.bool, device = DEV)
    for r in range(2):
        observed = {1: int((ids[r, 1] == x[r, 1]).nonzero()[0, 0])} if cond else None
        tuples, log_p = _joint(tab, tuple(t[r:r + 1] for t in ext), ids[r:r + 1], lp[r:r + 1], K, observed)
        _assert_draws_follow(_places(s[rows == r], ids[r])[:, keep], tuples[:, keep], log_p, f"row {r}")


@pytest.mark.parametrize("N", [4, 128])
@pytest.mark.parametrize("C, K", [(4, 2), (3, 4)])
def test_the_chi_square_rejects_draws_blind_to_the_slot(N, C, K):
    """The draws' check has power, in each case of `test_unconditional_draws_follow_the_joint`: the draws it accepts
    against the exact joint, it rejects against the joint of a leaf that ignores the slot its sum drew (its token from
    P(c | z) whatever the slot) -- a joint that is the HMM's own where the slots move no transition (G = 0), so it is
    that draw's and not a slip of its program."""
    T, n = 3, NUM_DRAWS
    pc, tab, ext, ids, lp, _, _ = _case(N, C, K, 1, DRAW_CASE, T = T, scale = 1.5)
    flat = ext[:2] + (torch.zeros_like(ext[2]),) + ext[3:]
    assert torch.allclose(_slot_blind_joint(tab, flat, ids, lp, K)[1], _joint(tab, flat, ids, lp, K)[1], atol = 1e-9)
    _seeded()
    with _draws_by("walk"), torch.no_grad():
        s = sample(pc, num_samples = n, **_repeated(ext, ids, lp, n, N))
    places = _places(s, ids[0])
    tuples, log_p = _joint(tab, ext, ids, lp, K)
    _assert_draws_follow(places, tuples, log_p, "the joint")
    blind_tuples, blind = _slot_blind_joint(tab, ext, ids, lp, K)
    assert torch.equal(blind_tuples, tuples)
    tv = 0.5 * float((torch.softmax(log_p, 0) - torch.softmax(blind, 0)).abs().sum())
    stat, dof = _chi_square(places, tuples, blind)
    assert not _plausible(stat, dof), (stat, dof, tv)
