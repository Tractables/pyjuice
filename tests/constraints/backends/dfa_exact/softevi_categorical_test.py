"""
SoftEvidenceCategorical input nodes under a constraint, with dense soft evidence (``categorical_evidence_logp``) bound
to a decoder session (:mod:`pyjuice.constraints.distributions.softevi_categorical`). Every token's emission is the
library's per-position product of experts ``q(v) = p(v) w(v) / Z``, a missing token's too, so the truth is pyjuice's
own forward: under the one-state constraint, ``pc(x, missing_mask, categorical_evidence_logp)`` itself; under any
other, the log-sum of ``pc(x)`` over the completions the constraint accepts. On every kind of PC of conftest.py, the
sum and transition paths, compact storage, evidence per row or shared, and the class masses on their own against a
float64 brute force (many classes, an empty one, few tokens of evidence). Evidence shaped like a language model's
(peaked, tokens drawn from it) against the reference's q, also at Codpula's 126464 tokens -- where uniform evidence is
the Categorical PC itself and one-hot evidence the constraint's verdict on its string -- and at Codpula's shape (slow).
Top-k evidence (``soft_evidence_cat_ids``) likewise: the library's forward, brute force (observed tokens that are not
candidates: impossible), the reference's conditionals, and at Codpula's vocabulary its top 1024 -- the dense evidence
that weighs every other token nothing, from the library's transposed table (a decoder's scope keeps one) or not. A
value mask (``soft_evidence_value_mask``): the library's unnormalized forward, brute force over the free positions,
the reference's conditionals at free and missing targets, Codpula's vocabulary, sessions.
"""
import itertools

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.constraints.distributions import TokenClasses, softevi_categorical

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 6, "pd_prod_dominated": 6,
         "pd_blockified": 6, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5, "hand_shared": 4}
CONSTRAINTS = {
    "anything": lambda: jc.DFA.anything(V),
    "contains": lambda: jc.DFA.contains([[1, 2]], V),
    "contains_either_and_not": lambda: jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)),
}
DEV = torch.device("cuda:0")


def soft_pc(build_pc, kind, n = None, **kwargs):
    return build_pc(kind, KINDS[kind] if n is None else n, V, input_dist = dists.SoftEvidenceCategorical, **kwargs)


def evidence(n, B, Be = None, K = V, seed = 0):
    """``(data, missing, E)``: row 0 all missing, the rest random; log-weights with some tokens weighing nothing (-inf),
    ``Be`` rows of them (default ``B``) over ``K`` tokens; observed tokens below ``K``."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, K, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0] = True
    E = torch.randn(B if Be is None else Be, n, K, generator = g) * 2
    E[torch.rand(E.shape, generator = g) < 0.2] = -float("inf")
    E[..., 0] = torch.where(torch.isinf(E).all(dim = -1), 0.0, E[..., 0])           # every position weighs a token
    return data.to(DEV), missing.to(DEV), E.to(DEV)


def topk_evidence(n, B, Be = None, K = 2, seed = 0):
    """``(data, missing, inputs)``: as :func:`evidence`, the evidence top-k -- ``K`` distinct candidates per sample and
    position (``soft_evidence_cat_ids``, in no order); fewer than all tokens: sample 1 observes a token that is not a
    candidate at position 0."""
    g = torch.Generator().manual_seed(seed)
    data, missing, _ = evidence(n, B, seed = seed)
    rows = B if Be is None else Be
    ids = torch.argsort(torch.rand(rows, n, V, generator = g), dim = 2)
    if K < V:
        missing[1, 0] = False
        data[1, 0] = int(ids[min(1, rows - 1), 0, K])
    ids = ids[..., :K]
    E = torch.randn(rows, n, K, generator = g) * 2
    return data, missing, dict(categorical_evidence_logp = E.to(DEV), soft_evidence_cat_ids = ids.to(DEV))


def brute_force(pc, constraint, data, missing, E):
    """log sum over the completions the constraint accepts of log q(x), from pyjuice's own forward (every token
    observed) under the evidence row of each sample (``E``: dense log-weights, or the inputs, as ``pc(x, ...)`` takes
    them)."""
    inputs = E if isinstance(E, dict) else dict(categorical_evidence_logp = E)
    out = []
    for b in range(data.size(0)):
        miss = missing[b].nonzero().view(-1).tolist()
        fills = list(itertools.product(range(V), repeat = len(miss)))
        xs = data[b].repeat(len(fills), 1)
        if miss:
            xs[:, miss] = torch.tensor(fills, device = DEV)
        xs = xs[torch.tensor([constraint.accepts(row) for row in xs.tolist()], device = DEV)]
        if xs.size(0) == 0:
            out.append(torch.tensor(-float("inf"), device = DEV))
            continue
        kw = {key: t[b if t.size(0) > 1 else 0][None].expand(xs.size(0), *t.shape[1:]).contiguous()
              for key, t in inputs.items()}
        out.append(torch.logsumexp(pc(xs, **kw).view(-1).double(), 0).float())
    return torch.stack(out)


def assert_close(got, want, n):
    got, want = got.view(-1), want.view(-1)
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), (got, want)
    fin = torch.isfinite(want)
    assert torch.isfinite(got[fin]).all()
    tol = 1e-5 * want[fin].abs() + n * 2e-6
    assert ((got[fin] - want[fin]).abs() <= tol).all(), (got[fin] - want[fin]).abs().max()


@pytest.fixture(params = ["fused", "dense"])
def sum_path(request, monkeypatch):
    """The sum layers through the fused kernel only, or the dense (cuBLAS) path wherever they qualify."""
    from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 if request.param == "dense" else 1 << 30)
    return request.param


@pytest.fixture(params = ["classes", "grouped", "input_groups"])
def transitions(request, monkeypatch):
    """A missing token's transitions one class at a time (one step per program, or ``input_groups``: several input
    steps per program, however narrow the launch), or grouped by successor through transition masses -- per sample,
    built for the session."""
    import pyjuice.constraints.backends.dfa_exact.compiled.program as P
    import pyjuice.constraints.backends.dfa_exact.kernels.prod as prod
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0 if request.param == "grouped" else 1 << 30)
    if request.param == "input_groups":
        monkeypatch.setattr(prod, "GROUP_MIN_LANES", 0)
    return request.param


# -------------------------------------------------------------------------------------------------
# Marginals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_the_one_state_constraint_is_the_pcs_own_marginal(kind, shared, build_pc):
    """Under ``DFA.anything``, the session's marginal is ``pc(x, missing_mask, categorical_evidence_logp)``: the
    observed tokens' values, and the missing ones' masses summing to one (evidence per row, or one row for all)."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing, E = evidence(n, 4, Be = 1 if shared else None)
    cc = jc.compile(jc.DFA.anything(V), pc)
    with cc.decoder(categorical_evidence_logp = E) as decoder:
        got = decoder.marginal(data, missing)
    want = pc(data, missing_mask = missing, categorical_evidence_logp = E.expand(4, -1, -1).contiguous())
    assert_close(got, want, n)


@pytest.mark.parametrize("constraint", ["contains", "contains_either_and_not"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_marginal_matches_a_brute_force(kind, constraint, sum_path, transitions, build_pc):
    """``log p(C, x_observed)`` under the evidence: the log-sum of pyjuice's own ``log q(x)`` over every completion the
    constraint accepts, every path of the sums and the transitions."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing, E = evidence(n, 3)
    c = CONSTRAINTS[constraint]()
    cc = jc.compile(c, pc)
    with cc.decoder(categorical_evidence_logp = E, storage = "dense") as decoder:
        got = decoder.marginal(data, missing)
    assert_close(got, brute_force(pc, c, data, missing, E), n)


@pytest.mark.parametrize("kind", list(KINDS))
def test_compact_storage_and_shared_evidence(kind, build_pc):
    """Compact storage gives the dense one's marginal; evidence of one row read by every row gives what the same row
    repeated per row gives, bit for bit (the same masses, read through the per-sample path or not)."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing, E = evidence(n, 5, Be = 1)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    got = {}
    for storage in ("dense", "compact"):
        for rows in ("shared", "per_row"):
            ev = E if rows == "shared" else E.expand(5, -1, -1).contiguous()
            with cc.decoder(categorical_evidence_logp = ev, storage = storage) as decoder:
                got[storage, rows] = decoder.marginal(data, missing)
    assert torch.equal(got["dense", "shared"], got["dense", "per_row"])
    assert torch.equal(got["compact", "shared"], got["compact", "per_row"])
    assert_close(got["compact", "shared"], got["dense", "shared"], n)
    assert_close(got["dense", "shared"], brute_force(pc, CONSTRAINTS["contains"](), data, missing, E), n)


@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_a_session_reuses_only_what_its_rows_inputs_allow(storage, topk, build_pc):
    """Left to right in one session (values reused where the evidence did not change), then the rows reordered as
    beams are (each continuing another row): every call is what a fresh session gives -- with evidence per row too,
    where a row taking another row's values would read that row's evidence; dense or top-k."""
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, _, inputs = topk_evidence(n, 4, K = V) if topk else evidence(n, 4)    # (every token a candidate: finite)
    inputs = inputs if topk else dict(categorical_evidence_logp = inputs)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    prefix = lambda t: (torch.arange(n, device = DEV)[None, :] >= t).expand(4, n)
    order = torch.tensor([2, 0, 1, 3], device = DEV)
    calls = [(data, prefix(t)) for t in range(n)] + [(data[order], prefix(n - 1))]
    for kw in (inputs, {key: t[:1] for key, t in inputs.items()}):
        with cc.decoder(storage = storage, **kw) as decoder:
            session = [decoder.marginal(x, m) for x, m in calls]
        for (x, m), got in zip(calls, session):
            with cc.decoder(storage = storage, **kw) as decoder:
                assert_close(got, decoder.marginal(x, m), n)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_evidence_every_row_reads_is_kept_once(kind, build_pc):
    """Evidence of one row for all: the values over positions no row observes are every sample's, kept once for the
    session (the brute force's value); evidence per row: they are each sample's own, none kept once over them."""
    from pyjuice.constraints.backends.dfa_exact.queries.marginal import marginal
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing, E = evidence(n, 3)
    free = torch.arange(n, device = DEV) >= n // 2                          # a suffix no row observes (left-linear:
    missing[:] = ~free if kind == "hand_left" else free                     # a prefix)
    c = CONSTRAINTS["contains"]()
    cc = jc.compile(c, pc)
    for ev in (E[:1], E):
        external = cc.backend.external_inputs(dict(categorical_evidence_logp = ev))
        with juice.fast_inference():
            report = {}
            got = marginal(cc.backend, data, missing, report = report, external = external)
        assert bool(report["plan"].prior) == (ev.size(0) == 1)
        assert_close(got, brute_force(pc, c, data, missing, ev), n)


def test_a_plain_query_inside_a_session(build_pc):
    """A query without the session's inputs in between (inside its scope): each reads the values kept once under its
    own class masses, never the other's."""
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, missing, E = evidence(n, 3)
    missing[:] = torch.arange(n, device = DEV) >= n // 2
    c = CONSTRAINTS["contains"]()
    cc = jc.compile(c, pc)
    plain = cc.marginal(data, missing)                                      # no evidence: w = 1, q = p
    with cc.decoder(categorical_evidence_logp = E[:1]) as decoder:
        first = decoder.marginal(data, missing)
        assert_close(cc.marginal(data, missing), plain, n)
        assert_close(decoder.marginal(data, missing), first, n)
    assert_close(first, brute_force(pc, c, data, missing, E[:1]), n)
    assert_close(plain, brute_force(pc, c, data, missing, torch.zeros_like(E[:1])), n)


def test_mixed_input_layers(build_pc):
    """Categorical and SoftEvidenceCategorical input layers in one PC: the evidence reaches the latter's variables
    only, the others keep their class masses (kept once where nothing per sample reaches them)."""
    torch.manual_seed(0)
    n = 5
    def dist(v):
        return dists.SoftEvidenceCategorical(num_cats = V) if v % 2 else dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = dist(v)) for v in range(n)]
    s = summate(x[n - 1], num_node_blocks = 2, block_size = 2)
    for v in range(n - 2, -1, -1):
        s = summate(multiply(x[v], s), num_node_blocks = 2 if v else 1, block_size = 2 if v else 1)
    s.init_parameters(perturbation = 2.0)
    pc = juice.compile(s, verbose = False).to(DEV)
    data, missing, E = evidence(n, 3)
    missing[1, :2] = False                                             # an observed prefix: kept once after it
    for c in (CONSTRAINTS["anything"](), CONSTRAINTS["contains_either_and_not"]()):
        cc = jc.compile(c, pc)
        with cc.decoder(categorical_evidence_logp = E) as decoder:
            got = decoder.marginal(data, missing)
        assert_close(got, brute_force(pc, c, data, missing, E), n)


# -------------------------------------------------------------------------------------------------
# Evidence as a language model gives it, and Codpula's vocabulary
# -------------------------------------------------------------------------------------------------

CODD_V = 126464                                                         # Codpula's (LLaDA's) vocabulary


def lm_evidence(B, n, V, hot = (), seed = 0):
    """Log-probabilities shaped like a language model's: the log-softmax, at temperature 0.7, of logits with a broad
    Gaussian floor and a few strongly preferred tokens per position -- and the tokens ``hot`` names ((position, token)
    pairs) strongly preferred too, so that the constraint's tokens matter."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, n, V, generator = g) * 2
    top = torch.randint(0, V, (B, n, 8), generator = g)
    logits.scatter_add_(2, top, torch.full(top.shape, 10.0))
    for t, v in hot:
        logits[:, t, v] += 12.0
    return torch.log_softmax(logits / 0.7, dim = -1).to(DEV)


def draw(E, seed = 0):
    """[B, n] tokens drawn from the evidence's distributions (an observed token is a likely one)."""
    g = torch.Generator(device = DEV).manual_seed(seed)
    B, n, V = E.shape
    return torch.multinomial(E.view(-1, V).exp(), 1, generator = g).view(B, n)


def masks(n, B, seed = 0):
    """The missing masks of decoding: a prefix observed (none, half, all but one), and any order."""
    g = torch.Generator().manual_seed(seed)
    out = [(torch.arange(n) >= t).expand(B, n) for t in (0, n // 2, n - 1)]
    out.append(torch.rand(B, n, generator = g) < 0.5)
    return [m.to(DEV) for m in out]


@pytest.mark.parametrize("kind", list(KINDS))
def test_language_model_evidence_against_the_reference(kind, build_pc, reference):
    """Evidence as a language model gives it (peaked, the constraint's tokens likely at some positions), tokens drawn
    from it where observed: the session's marginals, every decoding mask, against the float64 reference's q."""
    n = KINDS[kind]
    Vm = 50
    pc = build_pc(kind, n, Vm, input_dist = dists.SoftEvidenceCategorical)
    c = jc.DFA.contains([[7, 31]], Vm) & jc.Not(jc.DFA.contains([[44]], Vm))
    E = lm_evidence(3, n, Vm, hot = [(1, 7), (2, 31), (n - 1, 44)])
    cc = jc.compile(c, pc)
    with cc.decoder(categorical_evidence_logp = E) as decoder:
        for m in masks(n, 3):
            data = draw(E)
            want = reference.marginal(cc, data, m, dict(categorical_evidence_logp = E))
            assert_close(decoder.marginal(data, m), want, n)


CODD_CONSTRAINT = lambda: jc.DFA.contains([[50000, 100000]], CODD_V) & jc.Not(jc.DFA.contains([[126000]], CODD_V))


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_codpula_vocabulary(kind, build_pc, reference):
    """Codpula's 126464 tokens, three classes and one of nearly all of them: language-model evidence against the
    reference; uniform evidence is the Categorical PC itself (q = p); one-hot evidence is the constraint's own verdict
    on the evidence's string, whatever the PC (q a point mass): 0 if it accepts it, -inf if not."""
    n = KINDS[kind]
    pc = build_pc(kind, n, CODD_V, input_dist = dists.SoftEvidenceCategorical)
    c = CODD_CONSTRAINT()
    cc = jc.compile(c, pc)
    E = lm_evidence(2, n, CODD_V, hot = [(1, 50000), (2, 100000), (n - 2, 126000)])
    with cc.decoder(categorical_evidence_logp = E) as decoder:
        for m in masks(n, 2):
            data = draw(E)
            want = reference.marginal(cc, data, m, dict(categorical_evidence_logp = E))
            assert_close(decoder.marginal(data, m), want, n)

    plain = build_pc(kind, n, CODD_V)                                       # the same parameters, Categorical
    for a, b in zip(plain.input_layer_group, pc.input_layer_group):
        a.params.data.copy_(b.params.data)
    plain.params.data.copy_(pc.params.data)
    cc_plain = jc.compile(c, plain)
    data, m = draw(E), masks(n, 2)[3]
    with cc.decoder(categorical_evidence_logp = torch.zeros(1, n, CODD_V, device = DEV)) as decoder:
        assert_close(decoder.marginal(data, m), cc_plain.marginal(data, m), n)

    strings = torch.full((3, n), 7, device = DEV)                            # accepted, rejected twice
    strings[:, 1], strings[:, 2] = 50000, 100000
    strings[1, 2] = 99999
    strings[2, n - 1] = 126000
    one_hot = torch.full((3, n, CODD_V), -float("inf"), device = DEV).scatter_(2, strings[:, :, None], 0.0)
    with cc.decoder(categorical_evidence_logp = one_hot) as decoder:
        got = decoder.marginal(strings, torch.ones(3, n, dtype = torch.bool, device = DEV)).view(-1)
    assert got[0].abs() < 1e-5 and torch.isneginf(got[1:]).all(), got


def assert_rows_close(got, want, n):
    """Conditionals' rows (log-probabilities over the tokens): -inf where the reference's are, else close."""
    assert got.shape == want.shape
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert ((got[fin] - want[fin]).abs() <= 1e-4 + n * 2e-6 + 1e-5 * want[fin].abs()).all(), \
        (got[fin] - want[fin]).abs().max()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_conditional_matches_the_reference(kind, shared, sum_path, transitions, build_pc, reference):
    """``log p(x_t = v | e, C)`` under language-model evidence, every token of every missing position: the reference's
    (the target's own emission q = p w / Z), every path; the log marginal it returns is the marginal's; and the
    scores of candidates alone are the full rows' (a candidate past the vocabulary -inf)."""
    n = KINDS[kind]
    Vm = 50
    pc = build_pc(kind, n, Vm, input_dist = dists.SoftEvidenceCategorical)
    c = jc.DFA.contains([[7, 31]], Vm) & jc.Not(jc.DFA.contains([[44]], Vm))
    E = lm_evidence(1 if shared else 3, n, Vm, hot = [(1, 7), (2, 31), (n - 1, 44)])
    inputs = dict(categorical_evidence_logp = E)
    cc = jc.compile(c, pc)
    data = draw(E.expand(3, -1, -1).contiguous())
    m = masks(n, 3)[3]
    m[0] = True
    with cc.decoder(categorical_evidence_logp = E, storage = "dense", precision = "fp32") as decoder:
        got, lm = decoder.conditional(data, m, return_log_marginal = True)
        marg = decoder.marginal(data, m)
        cand = torch.randint(0, Vm + 3, (3, n, 6), device = DEV)
        got_c = decoder.conditional(data, m, candidates = cand)
    assert_rows_close(got, reference.conditional(cc, data, m, inputs = inputs), n)
    rows = torch.nonzero(m)
    assert_close(lm, marg.view(-1)[rows[:, 0]], n)
    want_c = torch.where(cand[rows[:, 0], rows[:, 1]] < Vm,
                         got.gather(1, cand[rows[:, 0], rows[:, 1]].clamp(max = Vm - 1)), -float("inf"))
    assert_rows_close(got_c, want_c, n)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("columns", ["per class", "in shares"])
def test_score_weights_paths(shared, columns, reference, monkeypatch):
    """The scoring's class weights under per-sample inputs (an HMM of 160 input nodes per position): one program per
    (column, class), or per column in shares of 64 rows -- the shares' partials, then their combination, the last
    share short -- the reference's conditionals either way."""
    from pyjuice.constraints.backends.dfa_exact.kernels import score
    if columns == "per class":
        monkeypatch.setattr(score, "SCORE_COLUMNS_MAX_CLASSES", 0)
    else:
        monkeypatch.setattr(score, "SCORE_COLUMNS_SHARE_ROWS", 1)
    shares, real = [], score._score_weights_columns_kernel

    class Recorded:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                shares.append(kwargs["SHARES"])
                return real[grid](*args, **kwargs)
            return launch

    monkeypatch.setattr(score, "_score_weights_columns_kernel", Recorded())
    n, Vm = 4, 50
    torch.manual_seed(0)
    ns = juice.structures.GeneralizedHMM(seq_length = n, num_latents = 160, block_size = 32,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = Vm))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(DEV)
    cc = jc.compile(jc.DFA.contains([[7, 31]], Vm) & jc.Not(jc.DFA.contains([[44]], Vm)), pc)
    E = lm_evidence(1 if shared else 3, n, Vm, hot = [(1, 7), (2, 31), (n - 1, 44)])
    data = draw(E.expand(3, -1, -1).contiguous())
    m = masks(n, 3)[3]
    m[0] = True
    with cc.decoder(categorical_evidence_logp = E, precision = "fp32") as decoder:
        got = decoder.conditional(data, m)
    assert_rows_close(got, reference.conditional(cc, data, m, inputs = dict(categorical_evidence_logp = E)), n)
    assert (max(shares) > 1) if columns == "in shares" else not shares, shares


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_codpula_vocabulary_conditionals(kind, build_pc, reference):
    """Conditionals at Codpula's 126464 tokens: language-model evidence against the reference (any order and prefix
    decoding in one session, compact storage); uniform evidence gives the Categorical PC's; one-hot evidence gives
    the point mass at its token where the constraint allows it."""
    n = KINDS[kind]
    pc = build_pc(kind, n, CODD_V, input_dist = dists.SoftEvidenceCategorical)
    c = CODD_CONSTRAINT()
    cc = jc.compile(c, pc)
    E = lm_evidence(2, n, CODD_V, hot = [(1, 50000), (2, 100000), (n - 2, 126000)])
    inputs = dict(categorical_evidence_logp = E)
    data = draw(E)
    with cc.decoder(categorical_evidence_logp = E, precision = "fp32") as decoder:
        for m in masks(n, 2)[:3] + [masks(n, 2)[3] | (torch.arange(n, device = DEV) == 1)]:
            if m.any():
                assert_rows_close(decoder.conditional(data, m), reference.conditional(cc, data, m, inputs = inputs), n)

    plain = build_pc(kind, n, CODD_V)
    for a, b in zip(plain.input_layer_group, pc.input_layer_group):
        a.params.data.copy_(b.params.data)
    plain.params.data.copy_(pc.params.data)
    m = masks(n, 2)[3] | (torch.arange(n, device = DEV) == 0)
    with cc.decoder(categorical_evidence_logp = torch.zeros(1, n, CODD_V, device = DEV), precision = "fp32") as decoder:
        assert_rows_close(decoder.conditional(data, m), jc.compile(c, plain).conditional(data, m, precision = "fp32"), n)

    strings = torch.full((1, n), 7, device = DEV)
    strings[0, 1], strings[0, 2] = 50000, 100000
    one_hot = torch.full((1, n, CODD_V), -float("inf"), device = DEV).scatter_(2, strings[:, :, None], 0.0)
    with cc.decoder(categorical_evidence_logp = one_hot, precision = "fp32") as decoder:
        got = decoder.conditional(strings, torch.ones(1, n, dtype = torch.bool, device = DEV))
    assert torch.equal(got.argmax(dim = 1), strings[0]) and (got.max(dim = 1).values.abs() < 1e-5).all()


def test_a_session_of_conditionals(build_pc):
    """Prefix decoding with conditionals in one session (values reused), evidence per row or shared, then rows moved:
    every call is a fresh session's."""
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, _, E = evidence(n, 4)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    prefix = lambda t: (torch.arange(n, device = DEV)[None, :] >= t).expand(4, n)
    order = torch.tensor([2, 0, 1, 3], device = DEV)
    calls = [(data, prefix(t)) for t in range(n)] + [(data[order], prefix(n - 1))]
    for ev in (E, E[:1]):
        with cc.decoder(categorical_evidence_logp = ev) as decoder:
            session = [decoder.conditional(x, m, targets = [min(n - 1, int(m[0].int().argmax()))]) for x, m in calls]
        for (x, m), got in zip(calls, session):
            with cc.decoder(categorical_evidence_logp = ev) as decoder:
                fresh = decoder.conditional(x, m, targets = [min(n - 1, int(m[0].int().argmax()))])
            assert_rows_close(got, fresh, n)


def test_scoring_columns_follow_the_session(build_pc, reference):
    """The scoring's per-sample columns: sessions of other inputs and batch sizes over the same circuit, each target
    position's conditionals against its own inputs (the reference's) -- the input nodes on the device kept on the
    program for every session, every position's weights' row kept per session (not taken again by its later calls)."""
    n, Vm = KINDS["hmm"], 50
    pc = build_pc("hmm", n, Vm, input_dist = dists.SoftEvidenceCategorical)
    cc = jc.compile(jc.DFA.contains([[7, 31]], Vm), pc)
    prog = cc.backend.program()
    nodes = None
    for B, rows, seed in ((3, 3, 0), (3, 3, 1), (2, 1, 2), (4, 4, 3)):
        E = lm_evidence(rows, n, Vm, hot = [(1, 7), (2, 31)], seed = seed)
        data = draw(E.expand(B, -1, -1).contiguous(), seed = seed)
        m = torch.zeros(B, n, dtype = torch.bool, device = DEV)
        m[:, 1:] = True
        inputs = dict(categorical_evidence_logp = E)
        got = []
        with cc.decoder(precision = "fp32", **inputs) as decoder:
            for t in (1, 2, 3):
                got.append(decoder.conditional(data, m, targets = [t]))
                if t == 1:
                    kept = dict(prog.sample_rows)
                    assert kept and (nodes is None or all(prog.sample_nodes[g] is v for g, v in nodes.items()))
                    nodes = dict(prog.sample_nodes)
            assert all(prog.sample_rows[g] is v for g, v in kept.items())          # (the session's)
            assert all(prog.sample_nodes[g] is v for g, v in nodes.items())        # (the program's)
        for t, row in zip((1, 2, 3), got):
            tg = torch.zeros_like(m)
            tg[:, t] = True
            assert_rows_close(row, reference.conditional(cc, data, m, targets = tg, inputs = inputs), n)


# -------------------------------------------------------------------------------------------------
# Top-k evidence (soft_evidence_cat_ids)
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_topk_the_one_state_constraint_is_the_pcs_own_marginal(kind, shared, build_pc):
    """Top-k evidence, two of the three tokens candidates: under ``DFA.anything`` the session's marginal is
    ``pc(x, missing_mask, categorical_evidence_logp, soft_evidence_cat_ids)`` -- -inf where an observed token is not
    a candidate."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing, inputs = topk_evidence(n, 4, Be = 1 if shared else None)
    cc = jc.compile(jc.DFA.anything(V), pc)
    with cc.decoder(**inputs) as decoder:
        got = decoder.marginal(data, missing)
    want = pc(data, missing_mask = missing, **{key: t.expand(4, -1, -1).contiguous() for key, t in inputs.items()})
    assert torch.isneginf(want).any() and torch.isfinite(want).any()
    assert_close(got, want, n)


@pytest.mark.parametrize("constraint", ["contains", "contains_either_and_not"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_topk_marginal_matches_a_brute_force(kind, constraint, build_pc):
    """``log p(C, x_observed)`` under top-k evidence (every token a candidate, or two of three): pyjuice's own ``log
    q(x)`` summed over the completions the constraint accepts, dense and compact storage, the candidates'
    probabilities from the library's transposed table (the decoder's scope keeps one) or the table itself."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    c = CONSTRAINTS[constraint]()
    cc = jc.compile(c, pc)
    for K in (V, 2):
        data, missing, inputs = topk_evidence(n, 3, K = K, seed = K)
        want = brute_force(pc, c, data, missing, inputs)
        for storage, copies in itertools.product(("dense", "compact"), (True, False)):
            with cc.decoder(storage = storage, allow_param_copy = copies, **inputs) as decoder:
                assert_close(decoder.marginal(data, missing), want, n)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_topk_conditional_matches_the_reference(kind, shared, build_pc, reference):
    """Conditionals under top-k language-model evidence (the 12 likeliest of 50 tokens), tokens drawn from it: the
    reference's, -inf at every token that is not a candidate; candidates alone score as the full rows (one past the
    vocabulary -inf); the log marginal returned is the marginal's."""
    n = KINDS[kind]
    Vm = 50
    pc = build_pc(kind, n, Vm, input_dist = dists.SoftEvidenceCategorical)
    c = jc.DFA.contains([[7, 31]], Vm) & jc.Not(jc.DFA.contains([[44]], Vm))
    E, ids = torch.topk(lm_evidence(1 if shared else 3, n, Vm, hot = [(1, 7), (2, 31), (n - 1, 44)]), 12, dim = 2)
    inputs = dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)
    dense = torch.full((E.size(0), n, Vm), -float("inf"), device = DEV).scatter_(2, ids, E)
    cc = jc.compile(c, pc)
    data = draw(dense.expand(3, -1, -1).contiguous())
    m = masks(n, 3)[3]
    m[0] = True
    with cc.decoder(precision = "fp32", **inputs) as decoder:
        got, lm = decoder.conditional(data, m, return_log_marginal = True)
        marg = decoder.marginal(data, m)
        cand = torch.randint(0, Vm + 3, (3, n, 9), device = DEV)
        cand[..., :4] = ids[..., :4]                                        # some of the candidates
        got_c = decoder.conditional(data, m, candidates = cand)
    want = reference.conditional(cc, data, m, inputs = inputs)
    assert_rows_close(got, want, n)
    rows = torch.nonzero(m)
    assert (torch.isfinite(want).sum(dim = 1) <= 12).all() and torch.isfinite(marg).any()
    assert_close(lm, marg.view(-1)[rows[:, 0]], n)
    want_c = torch.where(cand[rows[:, 0], rows[:, 1]] < Vm,
                         got.gather(1, cand[rows[:, 0], rows[:, 1]].clamp(max = Vm - 1)), -float("inf"))
    assert_rows_close(got_c, want_c, n)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_codpula_vocabulary_topk(kind, build_pc, reference):
    """Codpula's 126464 tokens and its top-k, the 1024 likeliest of a language model's: the reference's marginals and
    conditionals, which are the dense evidence's that weighs every other token nothing -- the candidates' probabilities
    read from the library's transposed table (the decoder's scope keeps one) or, without parameter copies, from the
    table itself; and a single candidate per position is the constraint's verdict on their string."""
    n = KINDS[kind]
    pc = build_pc(kind, n, CODD_V, input_dist = dists.SoftEvidenceCategorical)
    cc = jc.compile(CODD_CONSTRAINT(), pc)
    E, ids = torch.topk(lm_evidence(2, n, CODD_V, hot = [(1, 50000), (2, 100000), (n - 2, 126000)]), 1024, dim = 2)
    inputs = dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)
    dense = torch.full((2, n, CODD_V), -float("inf"), device = DEV).scatter_(2, ids, E)
    data = draw(dense)
    ms = [m for m in masks(n, 2)[:3] + [masks(n, 2)[3] | (torch.arange(n, device = DEV) == 1)] if m.any()]
    got = {}
    for copies in (True, False):
        for name, kw in (("topk", inputs), ("dense", dict(categorical_evidence_logp = dense))):
            with cc.decoder(allow_param_copy = copies, precision = "fp32", **kw) as decoder:
                got[copies, name] = [(decoder.marginal(data, m), decoder.conditional(data, m)) for m in ms]
                if name == "topk":
                    assert (pc.input_layer_group[0]._fast_inference_params is not None) == copies
    for k, m in enumerate(ms):
        want = (reference.marginal(cc, data, m, inputs), reference.conditional(cc, data, m, inputs = inputs))
        for (marg, cond) in (got[key][k] for key in got):
            assert_close(marg, want[0], n)
            assert_rows_close(cond, want[1], n)

    strings = torch.full((3, n, 1), 7, device = DEV)                         # accepted, rejected twice
    strings[:, 1], strings[:, 2] = 50000, 100000
    strings[1, 2] = 99999
    strings[2, n - 1] = 126000
    with cc.decoder(categorical_evidence_logp = torch.zeros(3, n, 1, device = DEV), soft_evidence_cat_ids = strings, precision = "fp32") \
            as decoder:
        got = decoder.marginal(strings[..., 0], torch.ones(3, n, dtype = torch.bool, device = DEV)).view(-1)
    assert got[0].abs() < 1e-5 and torch.isneginf(got[1:]).all(), got


# -------------------------------------------------------------------------------------------------
# A value mask (soft_evidence_value_mask)
# -------------------------------------------------------------------------------------------------

def masked_evidence(n, B, Be = None, topk = False, seed = 0):
    """``(data, missing, inputs)``: :func:`evidence` (or :func:`topk_evidence`, every token a candidate), with a value
    mask per sample, False at about a third of the positions -- free: marginalized under the weights -- and missing
    only where it is True; row 0 has free and missing positions both."""
    g = torch.Generator().manual_seed(seed + 100)
    data, missing, inputs = topk_evidence(n, B, Be, K = V, seed = seed) if topk else evidence(n, B, Be, seed = seed)
    inputs = inputs if topk else dict(categorical_evidence_logp = inputs)
    vm = torch.rand(B, n, generator = g) >= 0.35
    vm[0, 0], vm[0, 1] = False, True
    missing = missing & vm.to(DEV)
    missing[0, 1] = True
    return data, missing, dict(inputs, soft_evidence_value_mask = vm.to(DEV))


@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_value_mask_the_one_state_constraint_is_the_pcs_own_marginal(kind, shared, topk, build_pc):
    """Under ``DFA.anything``, the session's marginal is ``pc(x, missing_mask, ..., soft_evidence_value_mask)``: True,
    an observed token's ``log p(d) w(d)``; False, ``log Z``; missing, 0 -- evidence per row or one row for all (the
    mask per row: every row its own masses); the backend's own query, given the inputs, alike."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing, inputs = masked_evidence(n, 4, Be = 1 if shared else None, topk = topk)
    cc = jc.compile(jc.DFA.anything(V), pc)
    with cc.decoder(**inputs) as decoder:
        got = decoder.marginal(data, missing)
    want = pc(data, missing_mask = missing, **{key: t.expand(4, *t.shape[1:]).contiguous() for key, t in inputs.items()})
    assert_close(got, want, n)
    assert_close(cc.backend.marginal(data, missing, external = cc.backend.external_inputs(inputs)), want, n)


@pytest.mark.parametrize("constraint", ["contains", "contains_either_and_not"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_value_mask_matches_a_brute_force(kind, constraint, build_pc):
    """``log`` of the sum over the completions of the free positions the constraint accepts of the library's
    unnormalized ``p(x) w(x)`` (its forward with the mask True everywhere), dense and top-k evidence, dense and compact
    storage."""
    pc = soft_pc(build_pc, kind)
    n = KINDS[kind]
    c = CONSTRAINTS[constraint]()
    cc = jc.compile(c, pc)
    for topk in (False, True):
        data, _, inputs = masked_evidence(n, 3, topk = topk, seed = 3)
        free = ~inputs["soft_evidence_value_mask"]
        want = brute_force(pc, c, data, free, dict(inputs, soft_evidence_value_mask = ~free | True))
        for storage in ("dense", "compact"):
            with cc.decoder(storage = storage, **inputs) as decoder:
                assert_close(decoder.marginal(data), want, n)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_value_mask_conditional_matches_the_reference(kind, shared, build_pc, reference):
    """Marginals and conditionals under a value mask and language-model evidence (dense, and its top 12): the
    reference's, every target -- a free position's (its tokens weighing ``p w``) and a missing one's (``q``)."""
    n = KINDS[kind]
    Vm = 50
    pc = build_pc(kind, n, Vm, input_dist = dists.SoftEvidenceCategorical)
    c = jc.DFA.contains([[7, 31]], Vm) & jc.Not(jc.DFA.contains([[44]], Vm))
    lm = lm_evidence(1 if shared else 3, n, Vm, hot = [(1, 7), (2, 31), (n - 1, 44)])
    E, ids = torch.topk(lm, 12, dim = 2)
    dense = torch.full_like(lm, -float("inf")).scatter_(2, ids, E)
    cc = jc.compile(c, pc)
    data = draw(dense.expand(3, -1, -1).contiguous())
    vm = torch.ones(3, n, dtype = torch.bool, device = DEV)
    vm[:, 1], vm[1, n - 1], vm[2, 0] = False, False, False
    m = torch.zeros(3, n, dtype = torch.bool, device = DEV)
    m[:, 2], m[0, n - 1] = True, True
    for inputs in (dict(categorical_evidence_logp = lm), dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)):
        inputs = dict(inputs, soft_evidence_value_mask = vm)
        with cc.decoder(precision = "fp32", **inputs) as decoder:
            marg = decoder.marginal(data, m)
            got = decoder.conditional(data, m)
            got_t = decoder.conditional(data, targets = ~vm)                 # the free positions alone
        assert_close(marg, reference.marginal(cc, data, m, inputs), n)
        want = reference.conditional(cc, data, m, inputs = inputs)
        assert_rows_close(got, want, n)
        assert_rows_close(got_t, reference.conditional(cc, data, None, ~vm, inputs = inputs), n)
        external = cc.backend.external_inputs(inputs)                       # the backend's own query alike
        assert_rows_close(cc.backend.conditional(data, m, external = external, precision = "fp32"), want, n)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_codpula_vocabulary_value_mask(kind, build_pc, reference):
    """A value mask at Codpula's 126464 tokens, dense evidence and its top 1024: the reference's marginal and
    conditionals at the free positions."""
    n = KINDS[kind]
    pc = build_pc(kind, n, CODD_V, input_dist = dists.SoftEvidenceCategorical)
    cc = jc.compile(CODD_CONSTRAINT(), pc)
    lm = lm_evidence(2, n, CODD_V, hot = [(1, 50000), (2, 100000), (n - 2, 126000)])
    E, ids = torch.topk(lm, 1024, dim = 2)
    data = draw(torch.full_like(lm, -float("inf")).scatter_(2, ids, E))
    vm = torch.ones(2, n, dtype = torch.bool, device = DEV)
    vm[:, 1], vm[1, n - 2] = False, False
    for inputs in (dict(categorical_evidence_logp = lm), dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)):
        inputs = dict(inputs, soft_evidence_value_mask = vm)
        with cc.decoder(**inputs) as decoder:
            assert_close(decoder.marginal(data), reference.marginal(cc, data, None, inputs), n)
            got = decoder.conditional(data, targets = ~vm)
        assert_rows_close(got, reference.conditional(cc, data, None, ~vm, inputs = inputs), n)


def test_a_session_under_a_value_mask(build_pc):
    """Prefix decoding of the positions the value mask keeps (the free ones marginalized throughout), marginals and
    conditionals (every missing position: the free ones too), then the rows moved: every call is what a fresh session
    gives."""
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, _, inputs = masked_evidence(n, 4)
    vm = inputs["soft_evidence_value_mask"]
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    prefix = lambda t: (torch.arange(n, device = DEV)[None, :] >= t) & vm
    order = torch.tensor([2, 0, 1, 3], device = DEV)
    calls = [(data, prefix(t)) for t in range(n)] + [(data[order], prefix(n - 1))]
    query = lambda decoder, x, m: (decoder.marginal(x, m), decoder.conditional(x, m))
    for kw in (inputs, dict(inputs, categorical_evidence_logp = inputs["categorical_evidence_logp"][:1])):
        with cc.decoder(**kw) as decoder:
            session = [query(decoder, x, m) for x, m in calls]
        for (x, m), (marg, cond) in zip(calls, session):
            with cc.decoder(**kw) as decoder:
                fresh = query(decoder, x, m)
            assert_close(marg, fresh[0], n)
            assert_rows_close(cond, fresh[1], n)


def test_a_session_ignores_the_data_at_free_positions(build_pc, monkeypatch):
    """Whatever the data holds at the positions the value mask frees: calls that differ only there (and in a newly
    observed position) continue each other -- values are reused, and every call is a fresh session's."""
    import pyjuice.constraints.backends.dfa_exact.queries.decoder as session_queries
    held, real = [], session_queries.marginal
    monkeypatch.setattr(session_queries, "marginal", lambda *a, **kw: (held.append(kw["held"]), real(*a, **kw))[1])
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, _, inputs = masked_evidence(n, 4)
    vm = inputs["soft_evidence_value_mask"]
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    calls = []
    for t in range(n):
        x = torch.where(vm, data, t % V)                                    # another token at every free position
        calls.append((x, (torch.arange(n, device = DEV)[None, :] >= t) & vm))
    with cc.decoder(**inputs) as decoder:
        session = [decoder.marginal(x, m) for x, m in calls]
    assert any(h is not None and h.any() for h in held[1:n])
    for (x, m), got in zip(calls, session):
        with cc.decoder(**inputs) as decoder:
            assert_close(got, decoder.marginal(x, m), n)


@pytest.mark.slow
def test_codpula_shape(reference):
    """Codpula's PC: an HMM of 1024 latents over 32 positions and 126464 tokens; prefix decoding in a session under
    language-model evidence against the reference."""
    torch.manual_seed(0)
    n = 32
    ns = juice.structures.GeneralizedHMM(seq_length = n, num_latents = 1024,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = CODD_V))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(DEV)
    cc = jc.compile(CODD_CONSTRAINT(), pc)
    E = lm_evidence(1, n, CODD_V, hot = [(5, 50000), (6, 100000), (20, 126000)])
    data = draw(E)
    with cc.decoder(categorical_evidence_logp = E) as decoder:
        for t in range(n):
            m = (torch.arange(n, device = DEV) >= t)[None, :]
            got = decoder.marginal(data, m)
            if t in (0, 6, 21, n - 1):
                assert_close(got, reference.marginal(cc, data, m, dict(categorical_evidence_logp = E)), n)


# -------------------------------------------------------------------------------------------------
# The class masses on their own
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("evidence_kind", ["dense", "topk", "topk_transposed"])
@pytest.mark.parametrize("classes", ["one", "three_and_empty", "each_token"])
@pytest.mark.parametrize("Be", [1, 3])
@pytest.mark.parametrize("kind", ["hmm", "hmm_untied", "hand_permuted"])
def test_class_masses_match_a_float64_brute_force(kind, Be, classes, evidence_kind, build_pc, monkeypatch):
    """Every node's row, per sample: ``log sum_{v in c} p(v) w(v) - log Z`` against float64, over 150 tokens with
    evidence on 30 (the others weigh nothing): dense on the first 30, the tokens in many splits or in one (no partial
    budget, written straight); or top-k on 30 others, the candidates' probabilities from the parameter table or (in
    fast_inference) the library's transposed one, the classes 64 at a time (each token its own: three tiles)."""
    Vb = 150
    pc = build_pc(kind, 5 if kind != "hmm_untied" else 4, Vb, input_dist = dists.SoftEvidenceCategorical)
    layer = next(iter(pc.input_layer_group))
    g = torch.Generator().manual_seed(1)
    nv = int(torch.unique(layer.vids.view(-1)).numel())
    E = (torch.randn(Be, nv, 30, generator = g) * 3).to(DEV)
    C = {"one": 1, "three_and_empty": 4, "each_token": Vb}[classes]
    tc = torch.tensor([0, 2, 3])[torch.randint(0, 3, (Vb,), generator = g)] if classes == "three_and_empty" else \
        (torch.zeros(Vb, dtype = torch.long) if classes == "one" else torch.randperm(Vb, generator = g))
    tokens = TokenClasses(tc.to(DEV), C)                                    # (class 1 empty; each token: not in order)
    p = layer.params.view(-1, Vb).double()
    if evidence_kind == "dense":
        inputs = dict(categorical_evidence_logp = E)
        w = torch.cat([E.double().exp(), torch.zeros(Be, nv, Vb - 30, device = DEV, dtype = torch.float64)], dim = 2)
    else:
        ids = torch.argsort(torch.rand(Be, nv, Vb, generator = g), dim = 2)[..., :30].to(DEV)
        inputs = dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)
        w = torch.zeros(Be, nv, Vb, device = DEV, dtype = torch.float64).scatter_(2, ids, E.double().exp())
    for budget in ((1 << 26, 1) if evidence_kind == "dense" else (1 << 26,)):
        monkeypatch.setattr(softevi_categorical, "MASS_PARTIAL_BUDGET", budget)
        out = torch.full((Be, softevi_categorical.sample_rows(layer, inputs), C), float("nan"), device = DEV)
        with juice.fast_inference() if evidence_kind == "topk_transposed" else torch.no_grad():
            si = softevi_categorical.sample_inputs(layer, tokens, inputs, out)
            assert (softevi_categorical._transposed_table(layer, inputs) is not None) == \
                (evidence_kind == "topk_transposed")
        rows = softevi_categorical.class_mass_rows(layer)
        lvid = layer.var_idmapping[layer.vids.view(-1).long()]
        for b in range(Be):
            pw = p[rows] * w[b, lvid]                                       # [nodes, V]
            per_class = torch.zeros(pw.size(0), C, dtype = torch.float64, device = DEV).index_add_(1, tc.to(DEV), pw)
            want = per_class.log() - pw.sum(dim = 1, keepdim = True).log()
            got = si.masses[b, si.rows].double()
            assert torch.equal(torch.isneginf(got), torch.isneginf(want))
            fin = torch.isfinite(want)
            assert (got[fin] - want[fin]).abs().max() < 1e-5


# -------------------------------------------------------------------------------------------------
# What a session refuses
# -------------------------------------------------------------------------------------------------

def test_inputs_checked_and_refused(build_pc):
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, missing, E = evidence(n, 2)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    ids = torch.arange(V, device = DEV).expand(2, n, V).contiguous()
    with pytest.raises(ValueError, match = "needs `categorical_evidence_logp`"):
        cc.decoder(soft_evidence_cat_ids = ids)
    for bad in (ids[..., :2], ids.float(), ids.cpu()):
        with pytest.raises(ValueError, match = "soft_evidence_cat_ids"):
            cc.decoder(categorical_evidence_logp = E, soft_evidence_cat_ids = bad)
    for value in (0, -1, V):                                               # a duplicate, out of range twice
        bad = ids.clone()
        bad[1, n - 1, 1] = value
        with pytest.raises(ValueError, match = "distinct tokens"):
            cc.decoder(categorical_evidence_logp = E, soft_evidence_cat_ids = bad)
    with pytest.raises(ValueError, match = "needs `categorical_evidence_logp`"):
        cc.decoder(soft_evidence_value_mask = missing)
    for bad in (missing.float(), missing[:, :1], missing.expand(3, 2, n)):
        with pytest.raises(ValueError, match = "soft_evidence_value_mask"):
            cc.decoder(categorical_evidence_logp = E, soft_evidence_value_mask = bad)
    with cc.decoder(categorical_evidence_logp = E, soft_evidence_value_mask = ~missing) as decoder:
        with pytest.raises(ValueError, match = "one way"):                  # missing where the mask frees it
            decoder.marginal(data, missing)
        decoder.marginal(data, missing & ~missing)
    with pytest.raises(ValueError, match = "sum_external_params"):
        cc.decoder(sum_external_params = {})
    with pytest.raises(ValueError, match = "K <="):
        cc.decoder(categorical_evidence_logp = torch.zeros(2, n, V + 1, device = DEV))
    with pytest.raises(ValueError, match = r"\[batch"):
        cc.decoder(categorical_evidence_logp = torch.zeros(2, n, device = DEV))
    with cc.decoder(categorical_evidence_logp = E) as decoder:
        with pytest.raises(ValueError, match = "batch of 2"):
            decoder.marginal(data[:1], missing[:1])
        with pytest.raises(TypeError, match = "whole session"):
            decoder.marginal(data, missing, categorical_evidence_logp = E)
    plain = jc.compile(CONSTRAINTS["contains"](), build_pc("hmm", n, V))
    with pytest.raises(ValueError, match = "No input layer"):
        plain.decoder(categorical_evidence_logp = E)
