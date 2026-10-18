"""
Enumerated token classes (:class:`pyjuice.constraints.distributions.TokenClasses`): classes of one token have no
column in the class-mass tables; wherever such a class's mass is read -- the transition masses, the top-down pass
through ungrouped tables when the transitions do not fit their budget, the liveness support, the conditionals'
normalizer -- the token's own probability (and per-sample weight) is read instead. Enumeration is on when the
automaton's transitions are grouped and at least half its classes have one token (``ENUMERATE_MAX_TOKENS``,
``ENUMERATE_MIN_FRACTION``).

The split itself; the Categorical and SoftEvidenceCategorical class-mass writers (stored columns and ``log Z`` over
every token against float64); the transition masses against float64 sums over every pair's tokens; the liveness
support of a session; and the key check: marginals, conditionals and log marginals with enumeration are those without
it, on every kind of PC, dense and compact storage, transitions kept and rebuilt in chunks (the top-down then reads
class masses by code), under no, dense, top-k and value-masked evidence; bit-identical where nothing is enumerated
(an ungrouped automaton, or one without singletons, whatever the constant).
"""
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
import pyjuice.constraints.backends.dfa_exact.backend as BE
import pyjuice.constraints.backends.dfa_exact.compiled.program as P
from pyjuice.constraints.distributions import TokenClasses, categorical, softevi_categorical
from pyjuice.constraints.backends.dfa_exact.kernels import live, prod_top_down
from pyjuice.constraints.backends.dfa_exact.kernels.c import get_module

V = 12
KINDS = {"hmm": 6, "hmm_untied": 6, "left_linear": 6, "pd": 6, "hand_left": 5, "hand_permuted": 5}
DEV = torch.device("cuda:0")


def singleton_automaton(vocab = V):
    """Some of ``1 2``, ``3 4``, ``5 6`` appears, and ``0 0`` never: every token of the patterns its own class (from
    some state each leads somewhere else), the rest one class -- 7 singletons of 8 classes."""
    return jc.DFA.contains([[1, 2], [3, 4], [5, 6]], vocab) & ~jc.DFA.contains([[0, 0]], vocab)


def class_automaton(vocab = V, classes = 4, states = 5, seed = 0):
    """A random automaton over ``classes`` classes of ``vocab // classes`` tokens each: no singleton."""
    rng = random.Random(seed)
    tc = [v * classes // vocab for v in range(vocab)]
    table = [[rng.randrange(states) for _ in range(classes)] for _ in range(states)]
    return jc.DFA.from_dense(vocab, [[table[q][tc[v]] for v in range(vocab)] for q in range(states)], 0,
                             rng.sample(range(states), 2))


def evidence(n, B, kind, K = 6, seed = 0):
    """``(data, missing, inputs)``: random tokens, half of them missing (row 0 all missing); the per-sample inputs of
    ``kind``: none, dense log-weights with some tokens weighing nothing (``dense_narrow``: over the first 5 tokens
    only, the others weighing nothing, the observed tokens among the 5), top-k (``K`` candidates, the observed tokens
    among them), or dense with a value mask leaving some positions free (not marked missing besides)."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0] = True
    inputs = {}
    if kind in ("dense", "value_mask", "dense_narrow"):
        Kd = 5 if kind == "dense_narrow" else V
        E = torch.randn(B, n, Kd, generator = g) * 2
        E[torch.rand(E.shape, generator = g) < 0.2] = -float("inf")
        E[..., 0] = torch.where(torch.isinf(E).all(dim = -1), 0.0, E[..., 0])
        inputs["categorical_evidence_logp"] = E
        if kind == "dense_narrow":
            data = data % Kd
    if kind == "topk":
        ids = torch.argsort(torch.rand(B, n, V, generator = g), dim = 2)[..., :K]
        data = ids[..., 0]
        inputs = dict(categorical_evidence_logp = torch.randn(B, n, K, generator = g) * 2, soft_evidence_cat_ids = ids)
    if kind == "value_mask":
        vm = torch.rand(B, n, generator = g) < 0.7
        missing &= vm                                                        # a free position is not missing besides
        inputs["soft_evidence_value_mask"] = vm
    return data.to(DEV), missing.to(DEV), {k: t.to(DEV) for k, t in inputs.items()}


def assert_close(got, want, n, tol = 1e-5):
    got, want = got.reshape(-1), want.reshape(-1)
    assert got.shape == want.shape
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), (got, want)
    fin = torch.isfinite(want)
    assert torch.isfinite(got[fin]).all()
    assert ((got[fin] - want[fin]).abs() <= tol * want[fin].abs() + n * 2e-6).all(), (got[fin] - want[fin]).abs().max()


def queries(cc, data, missing, inputs, storage):
    with cc.decoder(storage = storage, precision = "fp32", **inputs) as decoder:
        marginal = decoder.marginal(data, missing)
        conditional, log_marginal = decoder.conditional(data, missing, return_log_marginal = True)
    return marginal, conditional, log_marginal


# -------------------------------------------------------------------------------------------------
# The split
# -------------------------------------------------------------------------------------------------

def test_the_split():
    tc = torch.tensor([3, 0, 1, 0, 2, 0, 4, 0])                              # classes 1-4 singletons, 0 four tokens
    classes = TokenClasses(tc, 6, enumerate_max = 1)                          # (class 5: no token, stored)
    assert classes.num_stored == 2 and classes.num_enumerated == 4
    assert classes.stored_of_class.tolist() == [0, -1, -1, -1, -1, 1]
    assert classes.class_of_stored.tolist() == [0, 5]
    assert classes.stored_class.tolist() == [-1, 0, -1, 0, -1, 0, -1, 0]
    assert classes.enumerated_token.tolist() == [-1, 2, 4, 0, 6, -1]
    assert classes.enumerated_tokens.tolist() == [2, 4, 0, 6]                # (in class order)
    by = classes.by_class(4)                                                 # the stored classes' tokens alone
    assert by.order.tolist() == [1, 3, 5, 7] and by.class_rank.tolist() == [0, -1]
    toks, tile_class = classes.class_tiles(4, stored = True)                 # the stored classes' tokens, in tiles
    assert toks.tolist() == [1, 3, 5, 7] and tile_class.tolist() == [0]
    for cl in (classes, TokenClasses(tc, 6)):                                # every class's (the soft-evidence masses)
        toks, tile_class = cl.class_tiles(4)
        assert toks.tolist() == [1, 3, 5, 7, 2, -1, -1, -1, 4, -1, -1, -1, 0, -1, -1, -1, 6, -1, -1, -1]
        assert tile_class.tolist() == [0, 1, 2, 3, 4]
    plain = TokenClasses(tc, 6)
    assert plain.num_stored == 6 and plain.num_enumerated == 0 and torch.equal(plain.stored_class, plain.token_class)
    assert torch.equal(plain.stored_of_class, torch.arange(6, dtype = torch.int32))
    assert plain.by_class(4).order.tolist() == TokenClasses(tc, 6, enumerate_max = 0).by_class(4).order.tolist()
    moved = classes.to(DEV)
    assert moved.enumerate_max == 1 and moved.num_stored == 2 and moved.stored_class.device.type == "cuda"
    with pytest.raises(NotImplementedError, match = "one token"):
        TokenClasses(tc, 6, enumerate_max = 2)


def test_enumeration_needs_enough_singletons(build_pc, monkeypatch):
    """Half the classes or more singletons: enumerated, transitions grouped or not; fewer: not."""
    pc = build_pc("hmm", 6, V)
    for ratio, dfa, want in [(1 << 30, singleton_automaton(), 1), (0, singleton_automaton(), 1),
                             (0, class_automaton(), 0), (0, jc.DFA.contains([[1], [2]], V) & class_automaton(), 0),
                             (1 << 30, jc.DFA.contains([[1], [2]], V) & class_automaton(), 0)]:
        monkeypatch.setattr(P, "GROUP_MIN_RATIO", ratio)
        cc = jc.compile(dfa, pc)
        assert cc.backend.token_classes.enumerate_max == want, (ratio, dfa)
        assert cc.backend.grouped == (ratio == 0)
        assert (cc.backend.info()["num_enumerated_classes"] > 0) == (want == 1)
    monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", 0)
    assert jc.compile(singleton_automaton(), pc).backend.token_classes.enumerate_max == 0


def test_the_codes_of_every_class_at_every_boundary(build_pc, monkeypatch):
    """The enumerated tokens live at every boundary, and every class's code there: a stored class's column, a live
    enumerated class's ``-1 - e`` (its token the e-th of the boundary's), a dead one's ``-1 - count`` (token 0 is
    forbidden everywhere); the ungrouped entries' codes are live ones; the token list ends in its padding."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 1 << 30)                       # ungrouped: tables of codes
    n = KINDS["hmm"]
    cc = jc.compile(singleton_automaton() & ~jc.DFA.contains([[0]], V), build_pc("hmm", n, V))
    classes = cc.backend.token_classes
    data, missing, inputs = evidence(n, 2, "none")
    cc.conditional(data, missing)
    prog = cc.backend.program()
    en, nc, W = prog.enum, cc.backend.layout.next_col, cc.backend.layout.width.tolist()
    ptr, tok, code = en.ptr_host, en.tok.cpu().tolist(), en.code.cpu()
    assert en.num == ptr[-1] and len(tok) == en.num + 1 and tok[-1] == 0
    dead = int(classes.token_class[0])
    assert classes.enumerated_token[dead] == 0
    for t in range(n):
        count = ptr[t + 1] - ptr[t]
        live = (nc[t, :W[t]] >= 0).any(dim = 0)
        for c in range(cc.backend.num_classes):
            x = int(code[t, c])
            if classes.stored_of_class[c] >= 0:
                assert x == int(classes.stored_of_class[c])
            elif live[c]:
                assert -1 - count < x < 0 and tok[ptr[t] - 1 - x] == int(classes.enumerated_token[c])
            else:
                assert x == -1 - count, (t, c, x)
        assert not live[dead]
    tt = prog.trans_top_down
    assert not tt.grouped
    ex, eptr = tt.entry_x.cpu(), tt.entry_ptr.cpu()
    for t in range(n):
        count = ptr[t + 1] - ptr[t]
        codes = ex[int(eptr[t, 0]):int(eptr[t, W[t]])]
        assert ((codes >= 0) & (codes < classes.num_stored) | (codes < 0) & (codes >= -count)).all(), t


# -------------------------------------------------------------------------------------------------
# The writers
# -------------------------------------------------------------------------------------------------

def test_categorical_class_masses_of_the_stored_classes_match_a_float64_brute_force(build_pc):
    """Both passes (natural order up to 64 stored classes, class order above): the stored classes' columns."""
    Vb = 600
    pc = build_pc("hmm", 4, Vb)
    g = torch.Generator().manual_seed(2)
    for stored, singletons in ((10, 500), (100, 400), (0, Vb)):           # (every stored class two tokens or more)
        tc = torch.cat([torch.arange(Vb - singletons) % max(stored, 1),
                        stored + torch.arange(singletons)])[torch.randperm(Vb, generator = g)]
        C = stored + singletons
        classes = TokenClasses(tc.to(DEV), C, enumerate_max = 1)
        assert classes.num_stored == stored and classes.num_enumerated == singletons
        for layer in pc.input_layer_group:
            probs = layer.params.double().view(-1, Vb)
            want = torch.zeros(probs.size(0), C, dtype = torch.float64, device = DEV).index_add_(1, tc.to(DEV), probs)
            want = want[:, classes.class_of_stored].log()
            got = categorical.class_masses(layer, classes)
            assert got.shape == (probs.size(0), stored)
            assert torch.equal(torch.isneginf(got), torch.isneginf(want))
            fin = torch.isfinite(want)
            assert stored == 0 or (got[fin].double() - want[fin]).abs().max() < 1e-5


@pytest.mark.parametrize("stored", [3, 70])
@pytest.mark.parametrize("evidence_kind", ["dense", "dense_split", "topk", "topk_transposed", "value_mask"])
def test_sample_inputs_stored_columns_and_log_z_match_a_float64_brute_force(evidence_kind, stored, build_pc,
                                                                            monkeypatch):
    """Per sample and row: the stored classes' ``log sum_{v in c} p(v) w(v) - log Z`` and ``log Z`` over every token,
    the enumerated classes' included -- the dense pass in one split and in many, top-k from the parameter table and
    the transposed one (3 stored classes, or 70: more than one tile of classes, the enumerated candidates sorted
    before the first), a value mask leaving rows unnormalized."""
    Vb, Be = 150, 3
    pc = build_pc("hmm", 5, Vb, input_dist = dists.SoftEvidenceCategorical)
    layer = next(iter(pc.input_layer_group))
    g = torch.Generator().manual_seed(1)
    nv = int(torch.unique(layer.vids.view(-1)).numel())
    singletons = Vb - 2 * stored
    tc = torch.cat([torch.arange(2 * stored) % stored, stored + torch.arange(singletons)])
    tc = tc[torch.randperm(Vb, generator = g)]
    C = stored + singletons
    classes = TokenClasses(tc.to(DEV), C, enumerate_max = 1)
    assert classes.num_stored == stored
    if evidence_kind.startswith("topk"):
        E = (torch.randn(Be, nv, 30, generator = g) * 3).to(DEV)
        ids = torch.argsort(torch.rand(Be, nv, Vb, generator = g), dim = 2)[..., :30].to(DEV)
        inputs = dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids)
        w = torch.zeros(Be, nv, Vb, device = DEV, dtype = torch.float64).scatter_(2, ids, E.double().exp())
    else:
        E = (torch.randn(Be, nv, Vb, generator = g) * 3).to(DEV)
        E[torch.rand(E.shape, generator = g) < 0.3] = -float("inf")
        inputs = dict(categorical_evidence_logp = E)
        w = E.double().exp()
    vm = None
    if evidence_kind == "value_mask":
        vm = (torch.rand(Be, nv, generator = g) < 0.5).to(DEV)
        inputs["soft_evidence_value_mask"] = vm
    monkeypatch.setattr(softevi_categorical, "MASS_PARTIAL_BUDGET", 1 if evidence_kind == "dense_split" else 1 << 26)
    out = torch.full((Be, softevi_categorical.sample_rows(layer, inputs), stored), float("nan"), device = DEV)
    with juice.fast_inference() if evidence_kind == "topk_transposed" else torch.no_grad():
        si = softevi_categorical.sample_inputs(layer, classes, inputs, out)
    rows = softevi_categorical.class_mass_rows(layer)
    lvid = layer.var_idmapping[layer.vids.view(-1).long()]
    p = layer.params.view(-1, Vb).double()
    for b in range(Be):
        pw = p[rows] * w[b, lvid]                                            # [nodes, V]
        per_class = torch.zeros(pw.size(0), C, dtype = torch.float64, device = DEV).index_add_(1, tc.to(DEV), pw)
        logz = pw.sum(dim = 1).log()
        norm = logz if vm is None else torch.where(vm[b, lvid], logz, 0.0)
        want = per_class[:, classes.class_of_stored].log() - norm[:, None]
        got = si.masses[b, si.rows].double()
        assert torch.equal(torch.isneginf(got), torch.isneginf(want))
        fin = torch.isfinite(want)
        assert (got[fin] - want[fin]).abs().max() < 1e-5
        got_z = si.logz[b, si.rows].double()
        want_z = torch.where(torch.isfinite(norm), norm, 0.0)
        assert (got_z - want_z).abs().max() < 1e-5


# -------------------------------------------------------------------------------------------------
# The readers
# -------------------------------------------------------------------------------------------------

def test_transition_masses_with_enumerated_classes_match_a_float64_brute_force(build_pc, monkeypatch):
    """Grouped transitions kept whole: every (step, column, successor)'s mass is the log-sum over the tokens leading
    there of ``p(v)`` -- and under dense evidence of a session, per sample, of ``q(v) = p(v) w(v) / Z``."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0)
    n, B = 6, 3
    dfa = singleton_automaton()
    for ev in ("none", "dense"):
        pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical if ev == "dense" else None)
        cc = jc.compile(dfa, pc)
        assert cc.backend.token_classes.num_enumerated == 7
        data, missing, inputs = evidence(n, B, ev)
        with cc.decoder(precision = "fp32", **inputs) as decoder:
            decoder.marginal(data, missing)
            prog = cc.backend.program()
            masses = cc.backend.state._session[2] if ev == "dense" else cc.backend.state.masses()
            T = prog.transition_buffer(masses.batch).double().cpu()
        assert prog.trans.grouped and prog.trans_persistent
        tt, width, nc = prog.trans, prog.width, cc.backend.layout.next_col
        ptr, succ = tt.entry_ptr.long().cpu(), tt.entry_succ.long().cpu()
        layer = next(iter(pc.input_layer_group))
        p = layer.params.double().view(-1, V).cpu()
        tc = cc.backend.token_classes.token_class.long().cpu()
        prow = masses.prow.long().cpu()
        row_of = masses.row.long().cpu()
        stride = masses.stride.long().cpu() if masses.stride is not None else torch.zeros_like(row_of)
        lognorm = masses.lognorm.double().cpu() if masses.lognorm is not None else None
        E = inputs["categorical_evidence_logp"].double().cpu() if ev == "dense" else None
        checked = 0
        for u, t, off in prog.trans_job[0].long().cpu().tolist():
            for b in range(masses.batch):
                r = int(row_of[u - prog.input_start] + b * stride[u - prog.input_start])
                q_tok = p[prow[r]].clone()
                if ev == "dense":
                    q_tok = q_tok * E[b, t].exp() / (q_tok * E[b, t].exp()).sum()
                    assert abs(float(lognorm[r]) - float((p[prow[r]] * E[b, t].exp()).sum().log())) < 1e-4
                for q in range(width[t]):
                    to = nc[t, q].long()
                    for g in range(int(ptr[t, q]), int(ptr[t, q + 1])):
                        classes = (to >= 0) if t + 1 == n else (to == succ[g])
                        want = q_tok[classes[tc]].sum().log()
                        got = T[b * prog.trans_size + off + g - int(ptr[t, 0])]
                        assert torch.equal(torch.isneginf(got), torch.isneginf(want)) and \
                            (torch.isneginf(want) or abs(float(got - want)) < 1e-5), (u, t, b, q, g)
                        checked += 1
        assert checked > 50


def test_a_sessions_support_follows_its_candidates(build_pc, monkeypatch):
    """Top-k evidence: a session's liveness support allows exactly the candidates' classes -- an enumerated class
    (one token) when its token is a candidate with a weight, the stored one when any of its tokens is."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0)
    n, B = 6, 4
    pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical)
    cc = jc.compile(singleton_automaton(), pc)
    assert cc.backend.token_classes.num_enumerated == 7
    data, missing, inputs = evidence(n, B, "topk", K = 3)
    inputs["categorical_evidence_logp"][1, 2, 0] = -float("inf")           # a candidate weighing nothing
    supports = []
    real = live.class_support
    monkeypatch.setattr(live, "class_support", lambda *a: (lambda c: supports.append(c) or c)(real(*a)))
    with cc.decoder(**inputs) as decoder:
        decoder.marginal(data, missing)
    C = cc.backend.num_classes
    cls = cc.backend.token_classes.token_class[inputs["soft_evidence_cat_ids"].long()].long()
    weighed = torch.isfinite(inputs["categorical_evidence_logp"])
    want = torch.zeros(B, n, C + 1, dtype = torch.bool, device = DEV)     # (an unweighed candidate: a spare column)
    want.scatter_(2, torch.where(weighed, cls, C), True)
    assert len(supports) == 1 and torch.equal(supports[0], want[:, :, :C])


# -------------------------------------------------------------------------------------------------
# Enumeration changes nothing
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("evidence_kind", ["none", "dense", "dense_narrow", "topk", "value_mask"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("transitions", ["grouped", "classes", "entries"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_enumerated_classes_change_nothing(kind, transitions, storage, evidence_kind, build_pc, monkeypatch):
    """Marginals, conditionals and log marginals with the singleton classes enumerated are those with every class
    stored -- transitions grouped (the enumerated classes in the transition masses) or ungrouped (the forward kernels
    over every class, or over a column's live entries; the top-down kernels read every entry's mass by code)."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0 if transitions == "grouped" else 1 << 30)
    monkeypatch.setattr(P, "ENTRY_LOOP_MAX_LIVE", 2 if transitions == "entries" else 0)
    n, B = KINDS[kind], 4
    pc = build_pc(kind, n, V, input_dist = dists.SoftEvidenceCategorical if evidence_kind != "none" else None)
    data, missing, inputs = evidence(n, B, evidence_kind)
    out = {}
    for enumerate_max in (0, 1):
        monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", enumerate_max)
        cc = jc.compile(singleton_automaton(), pc)
        assert cc.backend.token_classes.num_enumerated == (7 if enumerate_max else 0)
        out[enumerate_max] = queries(cc, data, missing, inputs, storage)
        assert cc.backend.grouped == (transitions == "grouped")
        assert cc.backend.program().entry_loop == (transitions == "entries")
        assert cc.backend.state.class_masses().size(1) == (1 if enumerate_max else 8)
    assert torch.isfinite(out[1][0]).any()
    for got, want in zip(out[1], out[0]):
        assert_close(got, want, n)


@pytest.mark.parametrize("evidence_kind", ["dense", "topk"])
def test_enumerated_classes_change_nothing_with_mixed_input_layers(evidence_kind, monkeypatch):
    """Categorical and SoftEvidenceCategorical input layers in one PC (variables alternating): the per-sample rows
    belong to the second layer of the class-mass table, the enumerated tokens' probabilities to its parameter rows."""
    from pyjuice.nodes import inputs, multiply, summate
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0)
    torch.manual_seed(0)
    n, B = 5, 3

    def dist(v):
        return dists.SoftEvidenceCategorical(num_cats = V) if v % 2 else dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = dist(v)) for v in range(n)]
    s = summate(x[n - 1], num_node_blocks = 2, block_size = 2)
    for v in range(n - 2, -1, -1):
        s = summate(multiply(x[v], s), num_node_blocks = 2 if v else 1, block_size = 2 if v else 1)
    s.init_parameters(perturbation = 2.0)
    pc = juice.compile(s, verbose = False).to(DEV)
    assert len(pc.input_layer_group) == 2
    data, missing, inputs_ = evidence(n, B, evidence_kind)
    out = {}
    for enumerate_max in (0, 1):
        monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", enumerate_max)
        cc = jc.compile(singleton_automaton(), pc)
        assert cc.backend.token_classes.num_enumerated == (7 if enumerate_max else 0)
        out[enumerate_max] = queries(cc, data, missing, inputs_, "dense")
        firsts = cc.backend.state.class_mass_rows()[0]
        assert firsts[1] > 0                                                   # the second layer's rows after the
    assert torch.isfinite(out[1][0]).any()                                     # first's
    for got, want in zip(out[1], out[0]):
        assert_close(got, want, n)


@pytest.mark.parametrize("evidence_kind", ["none", "topk"])
def test_nothing_enumerated_is_bit_identical(evidence_kind, build_pc, monkeypatch):
    """An automaton without singletons, grouped or not, or one with too few: the constant changes no code path, every
    output bit for bit the same."""
    n, B = 6, 4
    pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical if evidence_kind != "none" else None)
    data, missing, inputs = evidence(n, B, evidence_kind)
    few = jc.DFA.contains([[1], [2]], V) & class_automaton()
    for ratio, dfa in ((1 << 30, few), (0, class_automaton()), (1 << 30, class_automaton())):
        monkeypatch.setattr(P, "GROUP_MIN_RATIO", ratio)
        out = []
        for enumerate_max in (0, 1):
            monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", enumerate_max)
            cc = jc.compile(dfa, pc)
            assert cc.backend.token_classes.num_enumerated == 0 and cc.backend.program().enum.num == 0
            out.append(queries(cc, data, missing, inputs, "dense"))
        for got, want in zip(*out):
            assert torch.equal(got, want)


@pytest.mark.parametrize("evidence_kind", ["none", "dense", "topk"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("transitions", ["grouped", "entries"])
def test_every_class_enumerated(transitions, storage, evidence_kind, build_pc, monkeypatch):
    """Every token its own class: no class is stored (a class-mass table of no columns), every mass a token's --
    marginals, conditionals and log marginals those of every class stored."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 0 if transitions == "grouped" else 1 << 30)
    monkeypatch.setattr(P, "ENTRY_LOOP_MAX_LIVE", 2)
    n, B = 6, 4
    pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical if evidence_kind != "none" else None)
    data, missing, inputs = evidence(n, B, evidence_kind)
    dfa = class_automaton(classes = V)
    out = {}
    for enumerate_max in (0, 1):
        monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", enumerate_max)
        cc = jc.compile(dfa, pc)
        assert cc.backend.token_classes.num_stored == (0 if enumerate_max else V)
        assert cc.backend.grouped == (transitions == "grouped")
        out[enumerate_max] = queries(cc, data, missing, inputs, storage)
        assert cc.backend.state.class_masses().shape[1] == (0 if enumerate_max else V)
    assert torch.isfinite(out[1][0]).any()
    for got, want in zip(out[1], out[0]):
        assert_close(got, want, n)


@pytest.mark.parametrize("evidence_kind", ["none", "dense", "topk"])
@pytest.mark.parametrize("enumerate_max", [0, 1])
def test_candidates_of_enumerated_classes_are_columns_of_the_full_rows(enumerate_max, evidence_kind, build_pc,
                                                                      monkeypatch):
    """Candidates mixing enumerated and stored classes' tokens (some outside a top-k run, some repeated, some padding):
    every row's candidate scores are the columns of its full row -- the normalizer over every token with a weight,
    whichever tokens are asked for."""
    monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", enumerate_max)
    n, B = 6, 4
    pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical if evidence_kind != "none" else None)
    data, missing, inputs = evidence(n, B, evidence_kind)
    cc = jc.compile(singleton_automaton(), pc)
    assert cc.backend.token_classes.num_enumerated == (7 if enumerate_max else 0)
    g = torch.Generator().manual_seed(1)
    candidates = torch.randint(0, V, (B, n, 5), generator = g)
    candidates[:, :, 0] = 1                                                  # an enumerated class's token everywhere
    candidates[:, :, 1] = 11                                                 # a stored class's
    candidates[0, :, 2] = V                                                  # padding
    with cc.decoder(storage = "dense", precision = "fp32", **inputs) as decoder:
        full, logm = decoder.conditional(data, missing, return_log_marginal = True)
        got, logm_c = decoder.conditional(data, missing, candidates = candidates, return_log_marginal = True)
    assert torch.isfinite(full).any() and torch.equal(logm, logm_c)
    rows = torch.nonzero(missing.to(DEV))
    want = torch.full_like(got, -float("inf"))
    for r, (b, t) in enumerate(rows.tolist()):
        for j, v in enumerate(candidates[b, t].tolist()):
            if v < V:
                want[r, j] = full[r, v]
    assert_close(got, want, n)


@pytest.mark.parametrize("evidence_kind", ["none", "topk"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("automaton", ["singletons", "classes"])
def test_the_two_class_loops_give_the_same_bits(automaton, storage, evidence_kind, build_pc, monkeypatch):
    """Ungrouped transitions: the forward kernels over a column's live entries and over every class (the dead ones
    -inf, no-ops) take the same terms in class order -- marginals, conditionals and log marginals bit for bit, the
    singleton classes enumerated or none."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 1 << 30)
    n, B = 6, 4
    pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical if evidence_kind != "none" else None)
    data, missing, inputs = evidence(n, B, evidence_kind)
    dfa = singleton_automaton() if automaton == "singletons" else class_automaton()
    out = []
    for most in (0, 2):
        monkeypatch.setattr(P, "ENTRY_LOOP_MAX_LIVE", most)
        cc = jc.compile(dfa, pc)
        assert cc.backend.program().entry_loop == (most > 0) and not cc.backend.grouped
        assert cc.backend.token_classes.num_enumerated == (7 if automaton == "singletons" else 0)
        out.append(queries(cc, data, missing, inputs, storage))
    assert torch.isfinite(out[0][0]).any()
    for got, want in zip(*out):
        assert torch.equal(got, want)


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
@pytest.mark.parametrize("evidence_kind", ["none", "dense", "topk"])
@pytest.mark.parametrize("kind", ["hmm", "hand_left"])
def test_the_cuda_walk_reads_the_enumerated_masses(kind, evidence_kind, build_pc, monkeypatch):
    """Compact storage, the singleton classes enumerated: the ``input @ block`` CUDA walk (``c/walk.cu``, staging the
    stored columns and the boundary's enumerated tokens, its terms packed by code) takes every launch and gives the
    Triton kernel's conditionals bit for bit, shared class masses or each sample's own."""
    monkeypatch.setattr(P, "GROUP_MIN_RATIO", 1 << 30)                       # ungrouped: the walk reads by code
    n, B = KINDS[kind], 4
    pc = build_pc(kind, n, V, input_dist = dists.SoftEvidenceCategorical if evidence_kind != "none" else None)
    data, missing, inputs = evidence(n, B, evidence_kind)
    cc = jc.compile(singleton_automaton(), pc)
    assert cc.backend.token_classes.num_enumerated == 7 and not cc.backend.grouped
    taken, real = [], prod_top_down._input_block_cuda

    def record(*args):
        out = real(*args)
        if prod_top_down.INPUT_BLOCK_CUDA:
            taken.append(out is not None)
        return out

    monkeypatch.setattr(prod_top_down, "_input_block_cuda", record)
    masks = [missing] + [(torch.arange(n) >= t)[None, :].expand(B, n) for t in range(1, n)]   # (left to right)
    out = {}
    for cuda in (False, True):
        monkeypatch.setattr(prod_top_down, "INPUT_BLOCK_CUDA", cuda)
        with cc.decoder(storage = "compact", **inputs) as decoder:
            out[cuda] = [decoder.conditional(data, mk) for mk in masks]
    assert taken and all(taken)
    assert any(torch.isfinite(got).any() for got in out[True])
    for got, want in zip(out[True], out[False]):
        assert torch.equal(got, want)
