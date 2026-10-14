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


def brute_force(pc, constraint, data, missing, E):
    """log sum over the completions the constraint accepts of log q(x), from pyjuice's own forward (every token
    observed) under the evidence row of each sample."""
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
        Eb = E[b if E.size(0) > 1 else 0][None].expand(xs.size(0), -1, -1).contiguous()
        out.append(torch.logsumexp(pc(xs, categorical_evidence_logp = Eb).view(-1).double(), 0).float())
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


@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_a_session_reuses_only_what_its_rows_inputs_allow(storage, build_pc):
    """Left to right in one session (values reused where the evidence did not change), then the rows reordered as
    beams are (each continuing another row): every call is what a fresh session gives -- with evidence per row too,
    where a row taking another row's values would read that row's evidence."""
    pc = soft_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, _, E = evidence(n, 4)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    prefix = lambda t: (torch.arange(n, device = DEV)[None, :] >= t).expand(4, n)
    order = torch.tensor([2, 0, 1, 3], device = DEV)
    calls = [(data, prefix(t)) for t in range(n)] + [(data[order], prefix(n - 1))]
    for ev in (E, E[:1]):
        with cc.decoder(categorical_evidence_logp = ev, storage = storage) as decoder:
            session = [decoder.marginal(x, m) for x, m in calls]
        for (x, m), got in zip(calls, session):
            with cc.decoder(categorical_evidence_logp = ev, storage = storage) as decoder:
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

@pytest.mark.parametrize("classes", ["one", "three_and_empty", "each_token"])
@pytest.mark.parametrize("Be", [1, 3])
@pytest.mark.parametrize("kind", ["hmm", "hmm_untied", "hand_permuted"])
def test_class_masses_match_a_float64_brute_force(kind, Be, classes, build_pc, monkeypatch):
    """Every node's row, per sample: ``log sum_{v in c} p(v) w(v) - log Z`` against float64, over 40 tokens with
    evidence on the first 30 (the others weigh nothing); the tokens in many splits, or in one (no partial budget),
    written straight."""
    Vb = 40
    pc = build_pc(kind, 5 if kind != "hmm_untied" else 4, Vb, input_dist = dists.SoftEvidenceCategorical)
    layer = next(iter(pc.input_layer_group))
    g = torch.Generator().manual_seed(1)
    nv = int(torch.unique(layer.vids.view(-1)).numel())
    E = (torch.randn(Be, nv, 30, generator = g) * 3).to(DEV)
    C = {"one": 1, "three_and_empty": 4, "each_token": Vb}[classes]
    tc = torch.tensor([0, 2, 3])[torch.randint(0, 3, (Vb,), generator = g)] if classes == "three_and_empty" else \
        (torch.zeros(Vb, dtype = torch.long) if classes == "one" else torch.arange(Vb))   # (class 1 empty)
    tokens = TokenClasses(tc.to(DEV), C)
    p = layer.params.view(-1, Vb).double()
    w = torch.cat([E.double().exp(), torch.zeros(Be, nv, Vb - 30, device = DEV, dtype = torch.float64)], dim = 2)
    for budget in (1 << 26, 1):
        monkeypatch.setattr(softevi_categorical, "MASS_PARTIAL_BUDGET", budget)
        inputs = dict(categorical_evidence_logp = E)
        out = torch.full((Be, softevi_categorical.sample_rows(layer, inputs), C), float("nan"), device = DEV)
        si = softevi_categorical.sample_inputs(layer, tokens, inputs, out)
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
    ids = torch.zeros(2, n, V, dtype = torch.long, device = DEV)
    with pytest.raises(NotImplementedError, match = "soft_evidence_cat_ids"):
        cc.decoder(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)
    with pytest.raises(NotImplementedError, match = "soft_evidence_value_mask"):
        cc.decoder(categorical_evidence_logp = E, soft_evidence_value_mask = missing)
    with pytest.raises(NotImplementedError, match = "sum_external_params"):
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
        with pytest.raises(NotImplementedError, match = "Conditionals"):
            decoder.conditional(data, missing)
    plain = jc.compile(CONSTRAINTS["contains"](), build_pc("hmm", n, V))
    with pytest.raises(ValueError, match = "No input layer"):
        plain.decoder(categorical_evidence_logp = E)
