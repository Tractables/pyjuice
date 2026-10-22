"""
Conditionals under a constraint (:mod:`pyjuice.constraints.inference.dfa_exact.queries.conditional`): ``cc.conditional``
against the test reference's (``reference.conditional``) on every kind of PC in conftest.py's ``PC_KINDS``, under
constraints and every pattern of evidence, through every sum path; the class weights it scores (``dfa_class_flow``,
which the normalized rows cannot check: a factor common to every class cancels) against the reference's input nodes'
top-down blocks; ``log p(C, e)`` against the marginal; targets; evidence and constraints no string satisfies; the token
tables' lifetime, in fp32 and bf16; tf32. The class weights stored by the launch writing them first, the others
log-added to a fill only where needed. Decoding with values reused between queries is in decoder_test.py.
"""
import random
import sys

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels import compact as CP
from pyjuice.constraints.inference.dfa_exact.queries import conditional as QC
from pyjuice.constraints.inference.dfa_exact.queries.backend import forward as F
from pyjuice.constraints.inference.dfa_exact.queries import marginal as QM
from pyjuice.constraints.inference.dfa_exact.queries.conditional import class_weights
from pyjuice.constraints.inference.dfa_exact.kernels import score as S
from pyjuice.constraints.inference.dfa_exact.kernels.live import sample_successors
from pyjuice.constraints.inference.dfa_exact.kernels.score import score_tables
from pyjuice.constraints.inference.dfa_exact.queries.backend import plans as QP
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
    from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
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
    got, log_marginal = cc.conditional(data, missing, return_log_marginal = True, storage = "dense", precision = "fp32")
    assert_rows_close(got, reference.conditional(cc, data, missing), n)
    fin = torch.isfinite(got).any(dim = 1)
    assert (torch.logsumexp(got[fin], dim = 1).abs() <= 1e-5).all()
    want = reference.marginal(cc, data, missing)[:, 0][row_samples(missing).to(got.device)]
    assert_rows_close(log_marginal, want, n)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", list(CONSTRAINTS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_compact_storage_gives_the_same_rows(kind, name, pattern, sum_path, build_pc, reference):
    """``storage = "compact"`` (every sample's values, inside and top-down, over the pairs of its live columns only):
    the reference's rows, and the dense storage's -- bit for bit through the fused sums, but for block @ block products
    (whose compact products sum the live shared columns only, in other groups)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    got, log_marginal = cc.conditional(data, missing, return_log_marginal = True, storage = "compact", precision = "fp32")
    assert_rows_close(got, reference.conditional(cc, data, missing), n)
    dense, dense_marginal = cc.conditional(data, missing, return_log_marginal = True, storage = "dense", precision = "fp32")
    if sum_path == "fused" and kind in ["hmm", "hmm_untied", "hmm_block_sparse", "left_linear", "hand_left"]:
        assert torch.equal(got, dense) and torch.equal(log_marginal, dense_marginal)
    else:
        assert_rows_close(got, dense, n)


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_compact_keeps_no_inside_values_where_it_needs_none(kind, build_pc):
    """Inside fast_inference, every missing position of a prefix asked: on an HMM no per-sample inside value is needed
    (every per-sample scope holds a target; the others are kept once), and compact, node_mars holds the input rows
    alone; on a PD some are, and it holds every region's. The rows are dense's either way."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("prefix", n)
    B = data.size(0)
    with juice.fast_inference():
        dense = cc.conditional(data, missing, storage = "dense")
        full = cc.backend.state.storage("node_mars").numel()
        got = cc.conditional(data, missing, storage = "compact")
        held = cc.backend.state.storage("node_mars").numel()
    assert_rows_close(got, dense, n)
    inputs = cc.backend.input_range[1] * B
    assert held == (inputs if kind == "hmm" else full) and full > inputs


def test_storage_option(build_pc):
    """``storage`` is "auto", "dense" or "compact" (anything else raises); a marginal keeps no top-down value, and its
    compact inside values give dense's bit for bit (an HMM's fused sums compute each lane alone)."""
    n = KINDS["hmm"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hmm", n, V))
    data, missing = evidence("prefix", n)
    with pytest.raises(ValueError, match = "storage"):
        cc.conditional(data, missing, storage = "sparse")
    with pytest.raises(ValueError, match = "storage"):
        cc.marginal(data, missing, storage = "sparse")
    assert torch.equal(cc.marginal(data, missing, storage = "compact"), cc.marginal(data, missing, storage = "dense"))


def wide_automaton(K = 48, accepting = 12, seed = 0):
    rng = random.Random(seed)
    return jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0,
                             rng.sample(range(K), accepting))


@pytest.mark.parametrize("query", ["conditional", "marginal"])
@pytest.mark.parametrize("kind", ["pd", "hand_left"])
def test_values_held_in_the_other_storage_are_computed_again(kind, query, build_pc):
    """A query holding values left in the other storage (a decoder's, under "auto") computes them again: compact,
    held values without the ``carry`` that compact ones come with; dense, a carry. Its rows are those of a query holding
    nothing, and the report says so."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("prefix", n)
    reg_iv = cc.backend.program().reg_iv_np
    run = (lambda **kw: QC.conditional(cc.backend, data, missing, **kw)) if query == "conditional" else \
        (lambda **kw: QM.marginal(cc.backend, data, missing, **kw))
    for storage in ("compact", "dense"):
        fresh = {}
        want = run(storage = storage, report = fresh)
        plan = fresh["plan"]                              # hold every value the query computes per sample
        if query == "conditional":
            held = dict(inside = plan.sample[reg_iv], top_down = plan.top_down.sample)
            assert held["inside"].any() or held["top_down"].any()
        else:
            held = plan.sample[reg_iv]
            assert held.any()
        carry = None
        if storage == "dense":                            # a compact query's layout: the values said to be compact
            other = {}
            run(storage = "compact", report = other)
            inside = held["inside"] if query == "conditional" else held
            carry = dict(layout = other["compact"], inside = inside, regions = held["top_down"] if
                         query == "conditional" else np.zeros_like(inside), parents = torch.arange(data.size(0)).cuda())
        report = {}
        got = run(storage = storage, held = held, carry = carry, report = report)
        assert report["dropped"] and report["storage"] == storage
        assert torch.equal(got, want), storage


@pytest.mark.parametrize("query", ["conditional", "marginal"])
@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated", "hand", "hmm"])
def test_auto_storage_keeps_the_smaller_buffers(kind, query, build_pc):
    """``storage = "auto"`` (the default): a query keeps its values dense where its compact buffers would take more
    floats -- a 48-state automaton under any-order evidence observing one position per sample, whose live columns it
    joins in few pairs (not an HMM's) -- else compact; its rows those of the storage it picks, bit for bit, and the
    report says which."""
    n = KINDS[kind]
    cc = jc.compile(wide_automaton(), build_pc(kind, n, V))
    B = 6
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V, (B, n), generator = g)
    order = torch.stack([torch.randperm(n, generator = g) for _ in range(B)])
    run = (lambda **kw: QC.conditional(cc.backend, data, missing, **kw)) if query == "conditional" else \
        (lambda **kw: QM.marginal(cc.backend, data, missing, **kw))
    picked = set()
    for s in range(n):
        missing = torch.ones(B, n, dtype = torch.bool)
        missing.scatter_(1, order[:, :s], False)
        rows, reports = {}, {}
        for storage in ("dense", "compact"):
            reports[storage] = {}
            rows[storage] = run(report = reports[storage], storage = storage)
        floats = reports["compact"]["compact"].floats(query == "conditional")
        want = "compact" if floats <= CP.dense_floats(cc.backend, B, query == "conditional") else "dense"
        report = {}
        got = run(report = report)
        assert report["storage"] == want and (report["compact"] is None) == (want == "dense"), s
        assert torch.equal(got, rows[want]), s
        picked.add(want)
    assert picked == ({"compact"} if kind == "hmm" else {"dense", "compact"})


def dense_weights(groups):
    """Every row group's class weights from :func:`class_weights` (its rows, ``(group, positions, weights, row,
    rep)``) as ``(group, positions, [positions x B, classes, rows])``: -inf at a dead (column, class)."""
    out = []
    for g, pos, w, row, _ in groups:
        full = torch.full((*row.shape, w.size(1)), -float("inf"), device = w.device)
        live = row >= 0
        full[live] = w[row[live].long()]
        out.append((g, pos, full))
    return out


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
        groups = dense_weights(class_weights(cc.backend, data, missing.to(dev), positions, storage = "dense")[0])
    prog = cc.backend.program()
    score = score_tables(prog)
    C = cc.backend.num_classes
    want = {}                                                                   # (t, column) -> [B, C]
    for ns, O in reference.top_down(cc, data, missing).items():
        t = ns.scope.to_list()[0] if ns.is_input() else None
        if t is None or not positions[t]:
            continue
        Wi = int(cc.backend.layout.width[t])
        nc = cc.backend.layout.next_col_at(t).to(dev)                                   # [Wi, C]
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


def live_lists(prog, live, allowed, stored, t, b):
    """Brute force: the live classes of the column of position ``t`` and sample ``b``, in the boundary's class order,
    each with its key -- its live pairs (``next_col``: the column it leads ``i`` to, at the last position the sequence
    end's one column; a pair live at both ends, every pair where ``live`` is None) and whether it is stored -- a class
    the sample's ``allowed`` rules out having none."""
    nc, width, order, n = prog.layout.next_col_table(), prog.width, prog.trans.cls_order.cpu(), prog.n
    out = {}
    for c in order[t].tolist():
        pairs = tuple((i, 0 if t + 1 == n else int(nc[t, i, c])) for i in range(int(width[t])) if nc[t, i, c] >= 0)
        pairs = tuple((i, q) for i, q in pairs if live is None or (live[b, t, i] and live[b, t + 1, q]))
        if pairs and (allowed is None or allowed[b, t, c]):
            out[c] = (pairs, bool(stored[c] >= 0))
    return out


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("classes", ["every", "sample"])
@pytest.mark.parametrize("one_pair", [False, True])
@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left", "left_linear", "walks"])
def test_live_class_rows_match_a_brute_force(kind, storage, pattern, one_pair, classes, shared, build_pc, monkeypatch):
    """A query keeps its class weights in their live rows alone: a target column's (position ``t``, sample ``b``)
    class ``c`` is live exactly when a column live at ``t`` for the sample leads by ``c`` to one live at ``t + 1``
    (``next_col``; at the last position the sequence end's one column) -- every column live where the query prunes
    nothing --, its weights a sum over those live pairs. Shared rows (``CLASS_ROWS_SHARED``): the live classes of a
    column with the same live pairs and kind (stored or enumerated) share one row, their representative the first of
    them in the boundary's class order, a column's representatives on consecutive rows in that order; unshared, every
    live class a row of its own in that order. A row group's columns in order. Also reading a class's pairs one at a
    time (``one_pair``: every list walked over several tiles), and under a session's per-sample classes (``classes``,
    stood in for the session's): a class the sample rules out has none."""
    monkeypatch.setattr(S, "CLASS_ROWS_SHARED", shared)
    if one_pair:
        monkeypatch.setattr(S, "LIVE_CLASS_LANES", 1)
    n = {"walks": 6}.get(kind, KINDS.get(kind))
    cc = jc.compile(CONSTRAINTS["either"](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    B, dev = data.size(0), cc.pc.params.device
    allowed = None
    if classes == "sample":
        C = cc.backend.program().C
        allowed = torch.rand(B, n, C, generator = torch.Generator().manual_seed(3)) < 0.6
        monkeypatch.setattr(QC, "live_successors", lambda prog, masses: sample_successors(prog, allowed.to(prog.dev)))
    positions = missing.any(dim = 0).numpy()
    report = {}
    with torch.no_grad():
        groups, _ = class_weights(cc.backend, data, missing.to(dev), positions, storage = storage, report = report)
    prog = cc.backend.program()
    lv = report["liveness"]
    live = None if lv is None else lv.live.cpu() != 0
    stored = cc.backend.token_classes.stored_of_class.cpu()
    checked = merged = 0
    for g, pos, w, row, rep in groups:
        row, rows = row.cpu(), 0
        assert (rep is not None) == shared
        for p, t in enumerate(pos.tolist()):
            for b in range(B):
                keys = live_lists(prog, live, allowed, stored, t, b)
                want = list(keys)
                m = p * B + b
                assert [c for c in prog.trans.cls_order[t].tolist() if row[m, c] >= 0] == want, (g, t, b)
                if shared:
                    first = {}
                    for c in want:
                        first.setdefault(keys[c], c)
                    reps = list(dict.fromkeys(first[keys[c]] for c in want))         # in class order
                    assert [int(rep[m, c]) for c in want] == [first[keys[c]] for c in want], (g, t, b)
                    assert [int(row[m, c]) for c in want] == [rows + reps.index(first[keys[c]]) for c in want], (g, t, b)
                    assert all(int(rep[m, c]) == -1 for c in range(prog.C) if c not in keys), (g, t, b)
                    rows, merged = rows + len(reps), merged + len(want) - len(reps)
                else:
                    assert [int(row[m, c]) for c in want] == list(range(rows, rows + len(want))), (g, t, b)
                    rows += len(want)
                checked += 1
        assert w.size(0) == rows
    assert checked > 0 and (lv is not None or pattern == "none")
    assert merged > 0 or not shared or pattern == "none"    # (a sample past a 2: every class leads to acceptance)


@pytest.mark.parametrize("pattern", ["none", "mixed"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left", "left_linear"])
def test_shared_rows_are_exact_whatever_the_keys(kind, storage, pattern, build_pc, monkeypatch):
    """Classes share a row only when their live pairs and kinds are the same, whatever their keys: with every key the
    same (``CLASS_ROW_KEY_MASK`` 0: every live class of a column a candidate for the first) the rows are still checked
    exactly -- classes sharing a row have the same live pairs, and the first class of the column is shared by every
    class with its pairs --, and every class's weights are those it gets with rows of its own (unshared) and with the
    keys in full: the same -inf, the finite ones equal to their rounding (the walks' tiles follow the rows' number)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["either"](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    positions = missing.any(dim = 0).numpy()
    B, dev = data.size(0), cc.pc.params.device
    out = {}
    for name, shared, mask in (("unshared", False, S.CLASS_ROW_KEY_MASK), ("keys", True, S.CLASS_ROW_KEY_MASK),
                               ("collide", True, 0)):
        monkeypatch.setattr(S, "CLASS_ROWS_SHARED", shared)
        monkeypatch.setattr(S, "CLASS_ROW_KEY_MASK", mask)
        report = {}
        with torch.no_grad():
            groups, _ = class_weights(cc.backend, data, missing.to(dev), positions, storage = storage, report = report)
        out[name] = (groups, [w for _, _, w in dense_weights(groups)], report["liveness"])
    for name in ("keys", "collide"):
        for a, b in zip(out["unshared"][1], out[name][1]):
            assert torch.equal(torch.isneginf(a), torch.isneginf(b)), name
            fin = torch.isfinite(a)
            assert ((a[fin] - b[fin]).abs() <= 1e-5 * a[fin].abs() + n * 1e-6).all(), name
    prog = cc.backend.program()
    groups, _, lv = out["collide"]
    live = None if lv is None else lv.live.cpu() != 0
    stored = cc.backend.token_classes.stored_of_class.cpu()
    for g, pos, w, row, rep in groups:
        row, rep = row.cpu(), rep.cpu()
        for p, t in enumerate(pos.tolist()):
            for b in range(B):
                keys = live_lists(prog, live, None, stored, t, b)
                m = p * B + b
                for c in keys:
                    r = int(rep[m, c])
                    assert keys[r] == keys[c] and int(row[m, c]) == int(row[m, r]), (g, t, b, c)
                if keys:
                    c0 = next(iter(keys))
                    assert all(int(rep[m, c]) == c0 for c in keys if keys[c] == keys[c0]), (g, t, b)


@pytest.mark.parametrize("one_sample", [False, True])
@pytest.mark.parametrize("kind, storage", [("hmm", "dense"), ("hand_left", "dense"), ("left_linear", "dense"),
                                           ("pd", "dense"), ("hmm", "compact"), ("hand_left", "compact"),
                                           ("left_linear", "compact"), ("pd", "compact")])
def test_class_weights_walk_every_samples_own_live_columns(kind, storage, one_sample, build_pc, monkeypatch):
    """Samples sharing a class-weight program whose live lists at the target boundary differ in length -- every column
    (nothing observed), one (a prefix), none (evidence the constraint rules out) --: every sample's class weights are
    those it gets alone. Also with one sample per program (``one_sample``: every lane walks the same columns), and
    reading the products' top-down values compact."""
    if one_sample:
        monkeypatch.setattr(S, "CLASS_WEIGHT_LANES", 1)
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["either"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n, B = 12)
    data[3], missing[3] = 1, False                                      # "1 1 ..." contains neither "2" nor "0 0"
    missing[3, n // 2] = True
    positions = missing.any(dim = 0).numpy()
    dev = cc.pc.params.device
    with torch.no_grad():
        together = [w for _, _, w in dense_weights(class_weights(cc.backend, data, missing.to(dev), positions,
                                                                 storage = storage)[0])]
        alone = [[w for _, _, w in dense_weights(class_weights(cc.backend, data[b:b + 1], missing[b:b + 1].to(dev),
                                                               positions, storage = storage)[0])]
                 for b in range(data.size(0))]
    B = data.size(0)
    for g, w in enumerate(together):
        P = w.size(0) // B
        for b in range(B):
            got = w.view(P, B, *w.shape[1:])[:, b]
            want = alone[b][g]
            assert torch.equal(torch.isneginf(got), torch.isneginf(want)), (g, b)
            fin = torch.isfinite(want)
            assert ((got[fin] - want[fin]).abs() <= 1e-5 * want[fin].abs() + n * 1e-6).all(), (g, b)
    assert any(torch.isfinite(w).any() for w in together)


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("name", ["contains", "either", "avoids"])
@pytest.mark.parametrize("kind", ["hmm", "hmm_block_sparse", "hand_left", "left_linear", "pd"])
def test_the_two_class_weight_walks_give_the_same_bits(kind, name, pattern, storage, build_pc, monkeypatch):
    """The class weights walked over every class's own columns, and over the sample's columns (every one, or its live
    ones under evidence pruning): the same terms in the same order -- bit for bit; and the kernel's own choice is one
    of the two."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    positions = missing.any(dim = 0).numpy()
    dev = cc.pc.params.device
    out = {}
    for walk in (1, 2, 0):
        monkeypatch.setattr(S, "CLASS_WEIGHT_WALK", walk)
        with torch.no_grad():
            out[walk] = [w for _, _, w in dense_weights(class_weights(cc.backend, data, missing.to(dev), positions,
                                                                      storage = storage)[0])]
    assert any(torch.isfinite(w).any() for w in out[1])
    for a, b, c in zip(out[1], out[2], out[0]):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_targets(build_pc, reference):
    """Targets as a [n] mask and a [B, n] subset of the missing positions: the reference's rows for them, in the
    order ``dense[targets]``; a target that is observed raises."""
    n = KINDS["hand_left"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand_left", n, V))
    data, missing = evidence("mixed", n)
    missing[:, 2] = True
    one = torch.zeros(n, dtype = torch.bool)
    one[2] = True
    assert_rows_close(cc.conditional(data, missing, targets = one, precision = "fp32"),
                      reference.conditional(cc, data, missing, targets = one), n)
    g = torch.Generator().manual_seed(1)
    some = missing & (torch.rand(missing.shape, generator = g) < 0.5)
    assert_rows_close(cc.conditional(data, missing, targets = some, precision = "fp32"),
                      reference.conditional(cc, data, missing, targets = some), n)
    with pytest.raises(ValueError):
        cc.conditional(data, missing, targets = ~missing, precision = "fp32")


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


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_evidence_on_the_host_or_the_device(kind, build_pc):
    """The tokens and their classes built on the host (evidence there: packed in the query's one copy) or on the
    device (evidence there) give the same rows, bit for bit, under every pattern of evidence, and with candidates."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    dev = cc.pc.params.device
    for pattern in PATTERNS:
        data, missing = evidence(pattern, n)
        if not missing.any():
            continue
        assert torch.equal(cc.conditional(data, missing), cc.conditional(data.to(dev), missing.to(dev))), pattern
        cand = torch.randint(-1, V + 2, (data.size(0), n, 2), generator = torch.Generator().manual_seed(0))
        assert torch.equal(cc.conditional(data, missing, candidates = cand),
                           cc.conditional(data.to(dev), missing.to(dev), candidates = cand.to(dev))), pattern


@pytest.mark.parametrize("name", ["contains", "avoids"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_candidates_are_columns_of_the_full_rows(kind, name, build_pc):
    """``candidates`` in every form -- shared, per sample, per sample and position, with repeats and padding (-1, and
    ids past the vocabulary) -- give the full rows' columns at them, -inf at padding; ``log p(C, e)`` is unchanged."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    B = data.size(0)
    full, log_marginal = cc.conditional(data, missing, return_log_marginal = True, precision = "fp32")
    rows = torch.nonzero(missing)
    g = torch.Generator().manual_seed(0)
    forms = {"shared": torch.tensor([2, 0, -1, 1, V + 2, 2]),
             "per sample": torch.randint(-1, V, (B, 5), generator = g),
             "per position": torch.randint(-1, V + 1, (B, n, 40), generator = g)}            # several tiles
    for form, cand in forms.items():
        got, got_marginal = cc.conditional(data, missing, return_log_marginal = True, candidates = cand, precision = "fp32")
        per_row = candidate_tensor(cand, B, n)[rows[:, 0], rows[:, 1]]
        ok = (per_row >= 0) & (per_row < V)
        want = torch.where(ok, full.cpu().gather(1, per_row.clamp(0, V - 1)), -float("inf"))
        assert_rows_close(got.cpu(), want, n)
        assert torch.equal(got_marginal, log_marginal), form


@pytest.mark.parametrize("tiles", ["small", "large"])
def test_candidates_over_many_rows_and_bf16_tables(tiles, build_pc, monkeypatch):
    """An HMM whose input nodes read 256 rows (the kernel's whole chunks of rows), in fp32 and with a bf16 token table
    inside fast_inference, with either of the kernel's tiles; and a constraint no string satisfies."""
    from pyjuice.constraints.inference.dfa_exact.kernels import score
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
            full = cc.conditional(data, missing, precision = "fp32").cpu()
            got = cc.conditional(data, missing, candidates = cand, precision = "fp32").cpu()
        assert_rows_close(got, torch.where(ok, full.gather(1, per_row.clamp(0, V - 1)), -float("inf")), n, tol = tol)
    never = jc.compile(jc.DFA.contains([[1, 2] * n], V), pc)
    got = never.conditional(data, missing, candidates = cand, precision = "fp32")
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
    got, log_marginal = cc.conditional(data, missing, return_log_marginal = True, precision = "fp32")
    assert torch.isneginf(got[0]).all() and torch.isneginf(log_marginal[0])
    assert_rows_close(got, reference.conditional(cc, data, missing), n)
    never = jc.compile(jc.DFA.contains([[1, 2] * n], V), build_pc("hmm", n, V))
    assert not never.backend.satisfiable
    assert torch.isneginf(never.conditional(data, missing, precision = "fp32")).all()


def test_token_tables_live_as_long_as_the_scope(build_pc, reference):
    """Inside fast_inference the token tables are built by the first query and kept (the same tensors, the same
    rows), and dropped when the scope exits; outside, nothing is kept, and a change of the parameters is read."""
    n = KINDS["hmm_untied"]
    pc = build_pc("hmm_untied", n, V)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    data, missing = evidence("mixed", n)
    with juice.fast_inference():
        a = cc.conditional(data, missing, precision = "fp32")
        kept = dict(cc.backend.state._token_tables)
        assert kept and all(t.dtype == torch.float32 for t in kept.values())
        b = cc.conditional(data, missing, precision = "fp32")
        assert all(cc.backend.state._token_tables[k] is t for k, t in kept.items())
        assert torch.equal(a, b)
    assert cc.backend.state._token_tables is None
    with torch.no_grad():
        layer = pc.input_layer_group[0]
        table = layer.params.view(-1, V)
        table[:] = torch.rand_like(table) + 0.1
        table /= table.sum(dim = 1, keepdim = True)
    assert_rows_close(cc.conditional(data, missing, precision = "fp32"), reference.conditional(cc, data, missing), n)
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


def cw_weights(cc, data, missing, storage = "compact"):
    """The class weights of a query whose targets are its missing positions (cloned)."""
    positions = missing.any(dim = 0).numpy()
    with torch.no_grad():
        groups, _ = class_weights(cc.backend, data, missing.to(cc.pc.params.device), positions, storage = storage)
    return [w for _, _, w in dense_weights(groups)]


def forced_fill(real):
    """:func:`~...queries.backend.planners.class_weight_writes` as if every launch log-added to a fill."""
    def writes(prog, steps, positions):
        real(prog, steps, positions)
        for step in steps:
            if step is None or step[0] != "prod":
                continue
            for _, stages, _, _ in step[2]:
                for _, _, cws in stages:
                    cws[:] = [(la, item, False) for la, item, _ in cws]
        return True
    return writes


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed", "ruled_out"])
@pytest.mark.parametrize("kind", list(KINDS) + ["walks", "hand_dead"])
def test_class_weights_are_stored_once_or_added_to_a_fill(kind, pattern, storage, build_pc, monkeypatch):
    """A plan's class weight launches store the weights where they write each destination first and log-add them
    elsewhere, ``dfa_class_flow`` filled with -inf only where a launch adds to nothing yet or a weight is written by
    none (``cw_fill``, :func:`~...queries.backend.planners.class_weight_writes`; ``hand_dead``: steps without a top-down
    value): with the buffer poisoned (NaN) before every query, the weights are what log-adding every launch's to a fill
    gives, bit for bit -- also where the evidence leaves no sample a live column (``ruled_out``: every launch still
    writes)."""
    n = {"walks": 6, "hand_dead": 3}.get(kind, KINDS.get(kind))
    if pattern == "ruled_out":                                   # 0s around one missing token: never "1 2"
        data, missing = torch.zeros(4, n, dtype = torch.long), torch.zeros(4, n, dtype = torch.bool)
        missing[:, n // 2] = True
    else:
        data, missing = evidence(pattern, n)
    out = {}
    for force in (False, True):
        with monkeypatch.context() as m:
            if force:
                m.setattr(QP, "class_weight_writes", forced_fill(QP.class_weight_writes))
            cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
            view = cc.backend.state.storage_view

            def poisoned(name, numel):
                v = view(name, numel)
                if name == "dfa_class_flow":
                    v.fill_(float("nan"))
                return v

            m.setattr(cc.backend.state, "storage_view", poisoned)
            out[force] = cw_weights(cc, data, missing, storage = storage)
    for g, (got, want) in enumerate(zip(out[False], out[True])):
        assert not torch.isnan(want).any() and torch.equal(got.view(torch.int32), want.view(torch.int32)), g


@pytest.mark.parametrize("kind, fill, store", [("hmm", False, True), ("hmm_block_sparse", False, True),
                                               ("pd", False, True), ("hand", True, False), ("hand_dead", True, True)])
def test_class_weight_fill_where_needed(kind, fill, store, build_pc):
    """Where every input node is an operand of one product (an HMM, a PD), the class weight launches each write their
    destinations first and between them every one: they store, no fill; where some are operands of several
    (``hand``), the later launches log-add: the buffer filled; where some are operands of steps without a top-down
    value only (``hand_dead``), every launch stores but their weights are written by none: the buffer filled."""
    n = {"hand_dead": 3}.get(kind, KINDS.get(kind))
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    cw_weights(cc, data, missing)
    plans = [p.top_down for p in cc.backend.program().plans.values() if getattr(p, "top_down", None) is not None]
    stores = [s for p in plans for step, _, layers in p.steps if step == "prod" for _, stages, _, _ in layers
              for _, _, cws in stages for _, _, s in cws]
    assert plans and stores and all(p.cw_fill == fill for p in plans)
    assert all(stores) == store



@pytest.mark.slow
def test_transition_tables_past_two_billion_entries(reference):
    """An automaton of 128 live states and 46715 classes over 400 positions: ``next_col`` [n, W, C] holds 2.4e9
    entries, past what a 32-bit index reaches, so the kernels' offsets into it ``(t * W + i) * C`` must be 64-bit
    (they were not: an illegal memory access on JSON hard_0 from n = 768 on, in the class weights' sample walk; the
    forward walks' and the sweeps' offsets were the same). A two-latent HMM, the last 41 positions asked -- the ones
    whose offsets overflow -- with the prefix observed: the rows are the reference's (its indexing PyTorch's). Takes
    about 20 GB of GPU memory and 40 GB of host memory (the tables in int64 while they are built)."""
    V, S, n = 50257, 128, 400
    dev = torch.device("cuda")
    c = torch.arange(V)
    delta = torch.full((S + 1, V), S, dtype = torch.long)                   # S: the dead state
    for q in (c % S, (c // S) % S, (c // (S * S)) % S):                     # three states a class lives at,
        delta[q, c] = (q + 1 + c % 7) % S                                   # leading one of seven states on
    accept = torch.cat([torch.ones(S, dtype = torch.bool), torch.zeros(1, dtype = torch.bool)])
    dfa = jc.DFA(V, c, delta, 0, accept)
    assert dfa.num_classes > 40000 and n * S * dfa.num_classes > 2 ** 31
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = n, num_latents = 2, num_emits = V)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * n))       # (pyjuice initializes and compiles a structure recursively)
    try:
        ns.init_parameters(perturbation = 2.0)
        cc = jc.compile(dfa, juice.compile(ns, verbose = False).to(dev))
    finally:
        sys.setrecursionlimit(limit)
    assert int(cc.backend.layout.width.max()) == S
    g = torch.Generator(device = dev).manual_seed(1)
    data = dfa.sample(torch.zeros(1, n, V, device = dev), 1, generator = g)[0]
    missing = torch.arange(n, device = dev) >= n - 41
    got = cc.conditional(data, missing, precision = "fp32")
    # the rows are differences of log values near -4000 (400 positions of a 2-latent HMM over 50257 tokens), whose
    # fp32 ulp is 4.9e-4: up to 3 of them off the reference, at positions whose offsets overflow as at the others
    assert_rows_close(got, reference.conditional(cc, data, missing), n, tol = 4e-3)
    assert (torch.logsumexp(got, dim = 1).abs() <= 1e-4).all()


@pytest.mark.parametrize("sweep", ["cuda", "per_boundary", "per_sample"])
def test_transitions_through_the_automaton_give_the_tables_rows(sweep, build_pc, monkeypatch):
    """The kernels read a transition from the table over every boundary (``next_col``) or, past
    ``TRANSITION_TABLE_MAX_ENTRIES``, through the automaton's transitions and the boundaries' column maps: the same
    columns, so the same rows bit for bit -- conditionals and marginals, dense and compact storage, every sweep of
    the liveness, and a decoder session whose soft evidence prunes classes (its successor lists from the maps)."""
    import pyjuice.nodes.distributions as dists
    from pyjuice.constraints.inference.dfa_exact.compiled import program as P
    from pyjuice.constraints.inference.dfa_exact.kernels import live
    monkeypatch.setattr(live, "LIVE_CUDA", sweep == "cuda")
    monkeypatch.setattr(live, "LIVE_STEPS_MIN_WIDTH", 0 if sweep == "per_boundary" else 1 << 30)
    cases = [(kind, name, pattern) for kind in ("hmm", "pd") for name in ("contains", "either") for pattern in ("mixed",)]
    cases += [("hand_fold", "avoids", "scattered"), ("hmm", "contains", "none")]
    compiled = 0
    for kind, name, pattern in cases:
        n = KINDS[kind]
        pcs = [build_pc(kind, n, V) for _ in range(2)]
        data, missing = evidence(pattern, n)
        rows = []
        for pc, entries in zip(pcs, (P.TRANSITION_TABLE_MAX_ENTRIES, 0)):      # the table, then the maps
            monkeypatch.setattr(P, "TRANSITION_TABLE_MAX_ENTRIES", entries)
            cc = jc.compile(CONSTRAINTS[name](), pc)
            assert cc.backend.program().homogeneous == (entries == 0)
            rows.append([cc.conditional(data, missing, storage = s, precision = "fp32") for s in ("dense", "compact")]
                        + [cc.marginal(data, missing)])
            compiled += 1
        for a, b in zip(*rows):
            assert torch.equal(a, b), (kind, name, pattern)
    # a session's inputs: top-k soft evidence of 2 of the 3 tokens, the sweeps through the classes left
    n, B = KINDS["hmm"], 6
    outs = []
    for entries in (P.TRANSITION_TABLE_MAX_ENTRIES, 0):
        monkeypatch.setattr(P, "TRANSITION_TABLE_MAX_ENTRIES", entries)
        pc = build_pc("hmm", n, V, input_dist = dists.SoftEvidenceCategorical)
        cc = jc.compile(jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)), pc)
        dev = cc.pc.device
        gi = torch.Generator().manual_seed(1)
        inputs = dict(categorical_evidence_logp = (torch.randn(B, n, 2, generator = gi) * 2).to(dev),
                      soft_evidence_cat_ids = torch.argsort(torch.rand(B, n, V, generator = gi), dim = 2)[..., :2].to(dev))
        data, missing = evidence("mixed", n, B = B)
        with cc.decoder(**inputs) as decoder:
            outs.append([decoder.conditional(data, missing), decoder.marginal(data, missing)])
    for a, b in zip(*outs):
        assert torch.equal(a, b)
    assert compiled == 2 * len(cases)
