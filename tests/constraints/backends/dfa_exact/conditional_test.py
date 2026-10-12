"""
Conditionals under a constraint (:mod:`pyjuice.constraints.backends.dfa_exact.conditional`): ``cc.conditional`` against
the test reference's (``reference.conditional``) on every kind of PC in conftest.py's ``PC_KINDS``, under constraints
and every pattern of evidence, through every sum path; the class weights it scores (``dfa_class_flow``, which the
normalized rows cannot check: a factor common to every class cancels) against the reference's input nodes' top-down
blocks; ``log p(C, e)`` against the marginal; targets; evidence and constraints no string satisfies; the token tables'
lifetime, in fp32 and bf16; tf32; and an HMM's decoding through a pseudo cache (:class:`HMMDecodeCache`) giving the
rows a query computing everything gives, and (slow) its speed on an HMM4096.
"""
import time

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.backends.dfa_exact.queries import forward as F
from pyjuice.constraints.backends.dfa_exact.queries.conditional import class_weights, conditional
from pyjuice.constraints.backends.dfa_exact.kernels.score import score_tables
from pyjuice.constraints.distributions import TokenClasses
from pyjuice.constraints.evidence import candidate_tensor

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8,
         "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5,
         "hand_shared": 4, "hand_fold": 4}                                                      # see conftest.py
PATTERNS = ["none", "prefix", "suffix", "scattered", "mixed"]      # "all" observes everything: no target
CONSTRAINTS = {
    "contains": lambda: jc.DFA.contains([[1, 2]], V),
    "either": lambda: jc.DFA.contains([[2]], V) | jc.DFA.contains([[0, 0]], V),
    "avoids": lambda: jc.Not(jc.DFA.contains([[1, 1]], V)),    # two accepting states: a sequence end's columns
}


def evidence(pattern, n, B = 6, seed = 0):
    """As backward_test.py's: ``mixed`` cycles through every pattern, "all" (no target) included."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.ones(B, n, dtype = torch.bool)
    pats = ["all", "prefix", "suffix", "scattered", "none"]
    for b in range(B):
        p = pats[b % 5] if pattern == "mixed" else pattern
        if p == "all":
            missing[b] = False
        elif p == "prefix":
            missing[b, :n // 2] = False
        elif p == "suffix":
            missing[b, n // 2:] = False
        elif p == "scattered":
            missing[b] = torch.rand(n, generator = g) < 0.5
    return data, missing


@pytest.fixture(params = ["fused", "dense", "dense_unpruned"])
def sum_path(request, monkeypatch):
    """The fused sum kernels, or the dense path with its products on the live columns only or on every column."""
    from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", (lambda ncols, live: False) if request.param == "dense_unpruned" else
                        (lambda ncols, live: live < ncols))
    return request.param


def assert_rows_close(got, want, n, tol = None):
    """Log-probabilities: the same -inf entries, the finite ones within ``tol`` (default: fp32-level)."""
    assert got.shape == want.shape
    assert not torch.isnan(got).any()
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    bound = 1e-5 * want[fin].abs() + n * 1e-6 if tol is None else torch.full_like(want[fin], tol)
    assert ((got[fin] - want[fin]).abs() <= bound).all(), (got[fin] - want[fin]).abs().max()


def row_samples(targets):
    """The sample of every row (``dense[targets]`` order)."""
    return torch.nonzero(targets)[:, 0]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", list(CONSTRAINTS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_conditionals_match_the_reference(kind, name, pattern, sum_path, build_pc, reference):
    """Every missing position's row against the reference's; rows sum to 1 where finite; ``log p(C, e)`` is the
    marginal's."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    got, log_marginal = cc.conditional(data, missing, return_log_marginal = True)
    assert_rows_close(got, reference.conditional(cc, data, missing), n)
    fin = torch.isfinite(got).any(dim = 1)
    assert (torch.logsumexp(got[fin], dim = 1).abs() <= 1e-5).all()
    want = reference.marginal(cc, data, missing)[:, 0][row_samples(missing).to(got.device)]
    assert_rows_close(log_marginal, want, n)


@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("name", ["contains", "avoids"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_class_weights_match_the_reference(kind, name, pattern, sum_path, build_pc, reference):
    """The log class weights of every target position's rows, every sample whose token there is missing: ``A[c] =
    log sum_i exp(O_u[i, next_col[t, i, c]])`` of the reference's top-down blocks of the input nodes ``u`` over ``t``,
    log-summed over the nodes that read the same row of the class masses (tied ones)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    B, dev = data.size(0), cc.pc.params.device
    positions = missing.any(dim = 0).numpy()
    with torch.no_grad():
        groups, _ = class_weights(cc.backend, data, missing.to(dev), positions)
    prog = cc.backend.program()
    score = score_tables(prog)
    C = cc.backend.num_classes
    want = {}                                                                   # (t, column) -> [B, C]
    for ns, O in reference.top_down(cc, data, missing).items():
        t = ns.scope.to_list()[0] if ns.is_input() else None
        if t is None or not positions[t]:
            continue
        Wi = int(cc.backend.layout.width[t])
        nc = cc.backend.layout.next_col[t, :Wi].to(dev)                                 # [Wi, C]
        G = O[:, :, torch.arange(Wi, device = dev)[:, None], nc.clamp(min = 0)]
        A = torch.logsumexp(G.masked_fill(nc < 0, -float("inf")), dim = 2)      # [N, B, C]
        first = ns._output_ind_range[0]
        for j in range(ns.num_nodes):
            key = (t, int(score.col_of[first + j - prog.input_start]))
            want[key] = A[j] if key not in want else torch.logaddexp(want[key], A[j])
    checked = 0
    for g, pos, w in groups:
        for p, t in enumerate(pos.tolist()):
            rows = missing[:, t].to(dev)                                        # samples whose token is missing
            for k in range(w.size(2)):
                got = w[p * B:(p + 1) * B, :, k][rows]
                exp = want.get((t, k), torch.full((B, C), -float("inf"), device = dev))[rows]
                assert_rows_close(got, exp.float(), n)
                checked += 1
    assert checked > 0


def test_targets(build_pc, reference):
    """Targets as a [n] mask and a [B, n] subset of the missing positions: the reference's rows for them, in the
    order ``dense[targets]``; a target that is observed raises."""
    n = KINDS["hand_left"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand_left", n, V))
    data, missing = evidence("mixed", n)
    missing[:, 2] = True
    one = torch.zeros(n, dtype = torch.bool)
    one[2] = True
    assert_rows_close(cc.conditional(data, missing, targets = one),
                      reference.conditional(cc, data, missing, targets = one), n)
    g = torch.Generator().manual_seed(1)
    some = missing & (torch.rand(missing.shape, generator = g) < 0.5)
    assert_rows_close(cc.conditional(data, missing, targets = some),
                      reference.conditional(cc, data, missing, targets = some), n)
    with pytest.raises(ValueError):
        cc.conditional(data, missing, targets = ~missing)


def test_evidence_and_targets_in_every_form(build_pc):
    """``mask_id``, a prefix, and ``mask_id`` with ``missing_mask`` give what the explicit ``missing_mask`` gives,
    conditionals and marginal alike; targets as positions ([k] or [B, k] with padding) what their mask gives."""
    n = KINDS["hand_left"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand_left", n, V))
    data, missing = evidence("mixed", n)
    B = data.size(0)
    want = cc.conditional(data, missing)
    assert torch.equal(cc.conditional(torch.where(missing, -1, data)), want)
    assert torch.equal(cc.conditional(torch.where(missing, 7, data), mask_id = 7), want)
    some = missing & (torch.arange(n) % 2 == 0)
    assert torch.equal(cc.conditional(torch.where(some, -1, data), missing & ~some), want)
    assert torch.equal(cc.marginal(torch.where(missing, -1, data)), cc.marginal(data, missing))
    L = n // 2
    after = torch.zeros(B, n, dtype = torch.bool)
    after[:, L:] = True
    assert torch.equal(cc.conditional(data[:, :L]), cc.conditional(data, after))
    assert torch.equal(cc.marginal(data[:, :L]), cc.marginal(data, after))

    missing[:, 2] = True
    one = torch.zeros(n, dtype = torch.bool)
    one[2] = True
    assert torch.equal(cc.conditional(data, missing, targets = [2]), cc.conditional(data, missing, targets = one))
    pos = torch.full((B, 2), -1, dtype = torch.long)
    mask = torch.zeros(B, n, dtype = torch.bool)
    for b in range(B):
        idx = torch.nonzero(missing[b]).flatten()[:b % 3]                    # 0, 1 or 2 targets per sample
        pos[b, :len(idx)] = idx.flip(0)                                       # in any order
        mask[b, idx] = True
    assert torch.equal(cc.conditional(data, missing, targets = pos), cc.conditional(data, missing, targets = mask))


@pytest.mark.parametrize("name", ["contains", "avoids"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_candidates_are_columns_of_the_full_rows(kind, name, build_pc):
    """``candidates`` in every form -- shared, per sample, per sample and position, with repeats and padding (-1, and
    ids past the vocabulary) -- give the full rows' columns at them, -inf at padding; ``log p(C, e)`` is unchanged."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    B = data.size(0)
    full, log_marginal = cc.conditional(data, missing, return_log_marginal = True)
    rows = torch.nonzero(missing)
    g = torch.Generator().manual_seed(0)
    forms = {"shared": torch.tensor([2, 0, -1, 1, V + 2, 2]),
             "per sample": torch.randint(-1, V, (B, 5), generator = g),
             "per position": torch.randint(-1, V + 1, (B, n, 40), generator = g)}            # several tiles
    for form, cand in forms.items():
        got, got_marginal = cc.conditional(data, missing, return_log_marginal = True, candidates = cand)
        per_row = candidate_tensor(cand, B, n)[rows[:, 0], rows[:, 1]]
        ok = (per_row >= 0) & (per_row < V)
        want = torch.where(ok, full.cpu().gather(1, per_row.clamp(0, V - 1)), -float("inf"))
        assert_rows_close(got.cpu(), want, n)
        assert torch.equal(got_marginal, log_marginal), form


@pytest.mark.parametrize("tiles", ["small", "large"])
def test_candidates_over_many_rows_and_bf16_tables(tiles, build_pc, monkeypatch):
    """An HMM whose input nodes read 256 rows (the kernel's whole chunks of rows), in fp32 and with a bf16 token table
    inside fast_inference, with either of the kernel's tiles; and a constraint no string satisfies."""
    from pyjuice.constraints.backends.dfa_exact.kernels import score
    monkeypatch.setattr(score, "SCORE_CANDIDATE_PAIRS", 1 << 30 if tiles == "small" else 0)
    torch.manual_seed(0)
    n = 6
    pc = juice.compile(juice.structures.HMM(seq_length = n, num_latents = 256, num_emits = V),
                       verbose = False).to(torch.device("cuda:0"))
    cc = jc.compile(CONSTRAINTS["either"](), pc)
    data, missing = evidence("mixed", n)
    cand = torch.randint(-1, V, (data.size(0), n, 7))
    rows = torch.nonzero(missing)
    per_row = candidate_tensor(cand, data.size(0), n)[rows[:, 0], rows[:, 1]]
    ok = (per_row >= 0) & (per_row < V)
    for scope, tol in ((juice.fast_inference(), None), (juice.fast_inference(allow_bf16_params = True), 1e-2)):
        with scope:
            full = cc.conditional(data, missing).cpu()
            got = cc.conditional(data, missing, candidates = cand).cpu()
        assert_rows_close(got, torch.where(ok, full.gather(1, per_row.clamp(0, V - 1)), -float("inf")), n, tol = tol)
    never = jc.compile(jc.DFA.contains([[1, 2] * n], V), pc)
    got = never.conditional(data, missing, candidates = cand)
    assert got.shape == (int(missing.sum()), 7) and torch.isneginf(got).all()


def test_evidence_breaking_the_constraint(build_pc, reference):
    """A sample whose evidence no satisfying string agrees with gets rows of -inf and ``log p(C, e)`` -inf, the
    others in its batch their own rows; a constraint no string satisfies, rows of -inf."""
    n = KINDS["hmm"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hmm", n, V))
    data = torch.zeros(2, n, dtype = torch.long)
    missing = torch.zeros(2, n, dtype = torch.bool)
    missing[0, 2] = True                                                        # "1 2" impossible: one free token
    missing[1, 2:] = True
    got, log_marginal = cc.conditional(data, missing, return_log_marginal = True)
    assert torch.isneginf(got[0]).all() and torch.isneginf(log_marginal[0])
    assert_rows_close(got, reference.conditional(cc, data, missing), n)
    never = jc.compile(jc.DFA.contains([[1, 2] * n], V), build_pc("hmm", n, V))
    assert not never.backend.satisfiable
    assert torch.isneginf(never.conditional(data, missing)).all()


def test_token_tables_live_as_long_as_the_scope(build_pc, reference):
    """Inside fast_inference the token tables are built by the first query and kept (the same tensors, the same
    rows), and dropped when the scope exits; outside, nothing is kept, and a change of the parameters is read."""
    n = KINDS["hmm_untied"]
    pc = build_pc("hmm_untied", n, V)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    data, missing = evidence("mixed", n)
    with juice.fast_inference():
        a = cc.conditional(data, missing)
        kept = dict(cc.backend.state._token_tables)
        assert kept and all(t.dtype == torch.float32 for t in kept.values())
        b = cc.conditional(data, missing)
        assert all(cc.backend.state._token_tables[k] is t for k, t in kept.items())
        assert torch.equal(a, b)
    assert cc.backend.state._token_tables is None
    with torch.no_grad():
        layer = pc.input_layer_group[0]
        table = layer.params.view(-1, V)
        table[:] = torch.rand_like(table) + 0.1
        table /= table.sum(dim = 1, keepdim = True)
    assert_rows_close(cc.conditional(data, missing), reference.conditional(cc, data, missing), n)
    assert cc.backend.state._token_tables is None


def test_bf16_token_tables(build_pc, reference):
    """Inside fast_inference(allow_bf16_params = True) the tables are bfloat16: rows within bf16's rounding of the
    probabilities."""
    n = KINDS["pd"]
    cc = jc.compile(CONSTRAINTS["either"](), build_pc("pd", n, V))
    data, missing = evidence("mixed", n)
    with juice.fast_inference(allow_bf16_params = True):
        got = cc.conditional(data, missing)
        assert all(t.dtype == torch.bfloat16 for t in cc.backend.state._token_tables.values())
    assert_rows_close(got, reference.conditional(cc, data, missing), n, tol = 1e-2)


def test_tf32(build_pc, reference):
    n = KINDS["hand"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand", n, V))
    data, missing = evidence("mixed", n)
    assert_rows_close(cc.conditional(data, missing, precision = "tf32"), reference.conditional(cc, data, missing), n,
                      tol = 1e-2)


@pytest.mark.parametrize("kind", ["pd_prod_dominated", "hand_permuted", "hmm"])
def test_conditionals_are_deterministic(kind, build_pc):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    assert torch.equal(cc.conditional(data, missing), cc.conditional(data, missing))


def test_token_tables_are_the_parameters_transposed(build_pc):
    """``token_probs`` (consecutive rows: a view's copy; scattered ones: gathered) and every row group's table: the
    parameter rows, token-major."""
    from pyjuice.constraints.distributions import categorical
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand_permuted", 5, V))
    layer = cc.pc.input_layer_group[0]
    table = layer.params.view(-1, V)
    for rows in (np.arange(1, 4), np.asarray([0, 2, 3, table.size(0) - 1])):
        out = torch.empty(V, len(rows) + 2, device = table.device)
        categorical.token_probs(layer, rows, out[:, 1:-1])
        assert torch.equal(out[:, 1:-1], table[torch.from_numpy(rows).to(table.device)].t())
    score = score_tables(cc.backend.program())
    firsts, _, _ = cc.backend.state.class_mass_rows()
    for g, rows in enumerate(score.rows):
        want = table[torch.from_numpy(rows - firsts[0]).to(table.device)].t()
        assert torch.equal(cc.backend.state.token_table(g, rows), want)


def test_class_tiles():
    """Every token once, in class order, every tile holding one class's tokens (by id), padding -1."""
    g = torch.Generator().manual_seed(0)
    token_class = torch.randint(0, 7, (1000,), generator = g)
    token_class[token_class == 3] = 4                                           # a class without a token
    classes = TokenClasses(token_class, 7)
    for width in (16, 64):
        toks, tile_class = (x.cpu().long() for x in classes.class_tiles(width))
        tiles = toks.view(-1, width)
        assert tiles.size(0) == tile_class.numel()
        real = toks[toks >= 0]
        assert torch.equal(real.sort().values, torch.arange(1000))
        for tile, c in zip(tiles, tile_class.tolist()):
            ids = tile[tile >= 0]
            assert ids.numel() and (token_class[ids] == c).all() and (ids.diff() > 0).all()
            assert (tile[ids.numel():] == -1).all()


class HMMDecodeCache:
    """
    A pseudo caching engine, for tests only: the left-to-right decoding of an HMM-shaped circuit (sums over the
    suffixes ``[t, n)``) under a constraint, step ``t`` observing the tokens before ``t`` and asking for ``t``. A sum
    node's top-down value depends only on the evidence outside its scope, so observing ``x_{t-1}`` changes none of
    those whose scope holds ``t - 1``, every one of which step ``t - 1`` needed: step ``t`` holds them (``held``, see
    :mod:`pyjuice.constraints.backends.dfa_exact.query_plan`) and computes the one new layer. The buffers keep the
    values only for the same batch size and with no other query in between.
    """

    def __init__(self, cc):
        prog = cc.backend.program()
        self.cc = cc
        self.scope = np.asarray([prog.sum_ivs[prog.reg_iv_host[r]] for r in range(len(cc.backend.sum_regions))])
        self.last = None

    def step(self, data, t, precision = "fp32", cached = True):
        B, n = data.shape
        missing = torch.zeros(B, n, dtype = torch.bool)
        missing[:, t:] = True
        targets = torch.zeros(B, n, dtype = torch.bool)
        targets[:, t] = True
        held = None
        if cached and self.last == (B, t - 1):
            held = dict(top_down = (self.scope[:, 0] <= t - 1) & (t - 1 < self.scope[:, 1]))
        self.last = (B, t)
        return conditional(self.cc.backend, data, missing, targets, precision, held = held)


@pytest.mark.parametrize("kind", ["hmm", "hmm_untied", "hmm_block_sparse"])
def test_cached_decoding_gives_the_same_rows(kind, sum_path, build_pc, reference):
    """Every step of an HMM's decoding through :class:`HMMDecodeCache`: the rows of a query computing every value, bit
    for bit, and the reference's."""
    n = 8
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, _ = evidence("none", n, B = 4)
    cache = HMMDecodeCache(cc)
    with juice.fast_inference():
        cached = [cache.step(data, t).clone() for t in range(n)]           # one decoding, then the other
        fresh = [cache.step(data, t, cached = False).clone() for t in range(n)]
    for t in range(n):
        assert torch.equal(cached[t], fresh[t]), t
        missing = torch.zeros_like(data, dtype = torch.bool)
        missing[:, t:] = True
        targets = torch.zeros_like(missing)
        targets[:, t] = True
        assert_rows_close(cached[t], reference.conditional(cc, data, missing, targets = targets), n)


@pytest.mark.slow
def test_cached_decoding_speed(capsys):
    """An HMM4096 over 50257 tokens and 32 positions, under a 60-state, 10-class automaton (three keywords, all
    required -- the size of Ctrl-G's first example's), decoded step by step at batch 1, 16 and 128 through
    :class:`HMMDecodeCache`, inside fast_inference (fp32 and bf16 token tables) and without the cache: the cache's
    rows are the others', and its steps are faster. Prints the milliseconds per step."""
    n, Vb = 32, 50257
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = n, num_latents = 4096, num_emits = Vb)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to("cuda")
    dfa = jc.DFA.contains([[11, 12, 13]], Vb) & jc.DFA.contains([[21, 22, 23, 24]], Vb) & jc.DFA.contains([[31, 32]], Vb)
    cc = jc.compile(dfa, pc)
    report = []
    for B in (1, 16, 128):
        g = torch.Generator().manual_seed(B)
        data = torch.randint(0, Vb, (B, n), generator = g)
        for bf16 in (False, True):
            times = {}
            for cached in (True, False):
                cache = HMMDecodeCache(cc)
                with juice.fast_inference(allow_bf16_params = bf16):
                    for rep in range(2):                                        # the second pass: plans kept
                        ts = []
                        for t in range(n):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            rows = cache.step(data, t, cached = cached)
                            torch.cuda.synchronize()
                            ts.append(time.perf_counter() - t0)
                            if rep == 1 and t in (8, 24):
                                times.setdefault(("rows", t), []).append(rows)
                times[cached] = 1e3 * float(np.median(ts[8:]))
            for t in (8, 24):
                a, b = times[("rows", t)]
                assert_rows_close(a, b, n, tol = 1e-4)
            assert times[True] < times[False]
            report.append(f"B={B:4d} {'bf16' if bf16 else 'fp32'} table: cached {times[True]:.3f} ms/step, "
                          f"uncached {times[False]:.3f} ms/step")
    with capsys.disabled():
        print("\n" + "\n".join(report))
