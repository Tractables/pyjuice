"""
The class weights through the pairs' class sets (kernels/score/set_weights.py) against the classes' own walks: a query's lists hold, for every class, exactly its live
pairs, set by set; the conditionals are the classes' own up to rounding on every kind of circuit, evidence and storage; a decoder session stays within strong_check's
bars of the float64 reference; where the rule takes them, and what falls back to the classes' walks, to the bit.
"""
import dataclasses
import importlib.util
import os
import types
from fractions import Fraction

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels import live as LV
from pyjuice.constraints.inference.dfa_exact.kernels import score as S
from pyjuice.constraints.inference.dfa_exact.kernels.score import set_weights as SW
from pyjuice.constraints.inference.dfa_exact.kernels.transitions import host as H
from pyjuice.constraints.inference.dfa_exact.kernels.transitions import tables as TT
from pyjuice.constraints.inference.dfa_exact.queries import conditional as CO


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strong = _load("strong_check_test")
signatures = _load("transition_signatures_test")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
CASES = [("hmm", 8), ("hmm_block_sparse", 8), ("hmm_untied", 12), ("pd", 8), ("pd_blockified", 8), ("hand", 5), ("hand_left", 5)]


def automaton(seed, V = 12):
    """Few successors a state, so that a pair is led by several classes and a class is in several sets; every state reached."""
    g = torch.Generator().manual_seed(seed)
    K, C = int(torch.randint(3, 10, (1,), generator = g)), int(torch.randint(3, 7, (1,), generator = g))
    delta = torch.randint(0, max(K // 2, 2), (K, C), generator = g)
    delta[:, 0] = torch.arange(K).roll(1)
    token_class = torch.cat([torch.arange(C), torch.randint(0, C, (V - C,), generator = g)])[torch.randperm(V, generator = g)]
    return jc.DFA(V, token_class, delta, 0, torch.randperm(K, generator = g)[:max(K // 2, 1)].tolist())


def evidence(dfa, n, V, B, seed):
    """Strings the automaton accepts, and what is missing of them: nothing observed, a prefix observed, positions of each sample's own."""
    x = dfa.sample(torch.zeros(1, n, V, device = DEV), B, generator = torch.Generator(device = DEV).manual_seed(seed))[0]
    g = torch.Generator().manual_seed(seed)
    scattered = torch.rand(B, n, generator = g) < 0.5
    scattered[:, int(torch.randint(0, n, (1,), generator = g))] = True
    prefix = torch.zeros(B, n, dtype = torch.bool)
    prefix[:, n // 2:] = True
    return x, {"none": torch.ones(B, n, dtype = torch.bool, device = DEV), "half": prefix.to(DEV), "scattered": scattered.to(DEV)}


def taken_by(monkeypatch):
    """Every query that went through the class sets, as it combines them."""
    taken = []
    combine = SW.combine_set_weights
    monkeypatch.setattr(CO, "combine_set_weights", lambda *a: (taken.append(1), combine(*a))[1])
    return taken


@cuda_only
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind, n", CASES)
def test_conditionals_through_the_class_sets_are_the_classes_own_up_to_rounding(kind, n, storage, build_pc, monkeypatch):
    V, B = 12, 3
    pc = build_pc(kind, n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)                              # (tables on the device: the sets are theirs)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)                   # (and however few the pairs)
    taken = taken_by(monkeypatch)
    compared, worst = 0, 0.0
    for seed in range(6):
        dfa = automaton(seed, V)
        cc = jc.compile(dfa, pc)
        if not cc.backend.layout.satisfiable:
            continue
        x, masks = evidence(dfa, n, V, B, seed)
        for name, missing in masks.items():
            monkeypatch.setattr(SW, "SET_WEIGHTS", False)
            want = CO.conditional(cc.backend, x, missing, precision = "fp32", storage = storage).clone()
            monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
            before = len(taken)
            got = CO.conditional(cc.backend, x, missing, precision = "fp32", storage = storage)
            assert len(taken) == before + 1, (seed, name)
            assert torch.equal(torch.isinf(got), torch.isinf(want)), (seed, name)
            finite = ~torch.isinf(want)
            if finite.any():
                worst = max(worst, float((got[finite] - want[finite]).abs().max()))
                compared += int(finite.sum())
    assert compared > 0 and worst <= 1e-4, (compared, worst)


@cuda_only
@pytest.mark.parametrize("storage, observed", [("dense", "none"), ("compact", "none"), ("dense", "scattered"), ("compact", "scattered")])
def test_a_classs_live_pairs_are_those_of_the_sets_it_is_in(storage, observed, build_pc, monkeypatch):
    """A query's lists against the classes' own: for every target column and class, the pairs of the sets the class is in that have a row there are the pairs of the
    class's list live at both ends, each once; a column's rows are its live sets, the longest list first; under no pruning a position's samples share their lists."""
    V, n, B = 12, 8, 2
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    calls = []
    lists = SW.set_weight_lists
    monkeypatch.setattr(CO, "set_weight_lists", lambda *a: (calls.append([a, lists(*a)]), calls[-1][1])[1])
    checked = 0
    for seed in range(4):
        dfa = automaton(seed, V)
        cc = jc.compile(dfa, pc)
        if not cc.backend.layout.satisfiable:
            continue
        x, masks = evidence(dfa, n, V, B, seed)
        calls.clear()
        CO.conditional(cc.backend, x, masks[observed], precision = "fp32", storage = storage)
        (prog, lv, col_pos, col_K, *_), sw = calls[0]
        assert sw is not None and (lv is None) == (observed == "none")       # (nothing observed: no liveness to read, whatever the storage)
        for chunk in (1, max(sw.pairs // 3, 1)):                             # (a group a chunk, or a few: the same lists)
            with monkeypatch.context() as m:
                m.setattr(SW, "SET_WEIGHT_CHUNK", chunk)
                other = lists(*calls[0][0])
            for f in dataclasses.fields(sw):
                a, b = getattr(sw, f.name), getattr(other, f.name)
                assert torch.equal(a, b) and a.dtype == b.dtype if isinstance(a, torch.Tensor) else a == b, (f.name, chunk)
        C, W = prog.C, prog.W
        cls_ptr, cls_col, cls_succ = prog.class_lists()
        cls_ptr = cls_ptr.view(n, C + 1)
        cs_ptr, cs_set, _ = SW._class_sets_of(prog)
        live = None if lv is None else lv.live.view(-1, n + 1, W)
        for m in range(sw.count.numel()):
            t, b, g = int(col_pos[m // B]), m % B, int(sw.grp[m])
            assert g == (m // B if lv is None else m)
            of_set, lengths = {}, {}
            for s in range(sw.S):
                rank = int(sw.rank[g, s])
                if rank >= 0:
                    row = int(sw.first[m]) + rank
                    a, e = int(sw.lptr[row]), int(sw.lptr[row + 1])
                    of_set[s] = list(zip(sw.pair_i[a:e].tolist(), sw.pair_q[a:e].tolist()))
                    lengths[rank] = e - a
            assert sorted(lengths) == list(range(int(sw.count[m]))) and all(lengths[r] >= lengths[r + 1] > 0 for r in range(len(lengths) - 1))
            assert int(sw.off[m + 1] - sw.off[m]) == len(lengths) * int(col_K[m])
            for c in range(C):
                a, e = int(cls_ptr[t, c]), int(cls_ptr[t, c + 1])
                want = sorted((i, q) for i, q in zip(cls_col[a:e].tolist(), cls_succ[a:e].tolist()) if live is None or (live[b, t, i] and live[b, t + 1, q]))
                got = sorted(pair for s in cs_set[int(cs_ptr[c]):int(cs_ptr[c + 1])].tolist() for pair in of_set.get(s, []))
                assert got == want, (seed, m, c)
                checked += len(want)
    assert checked > 0


def assert_same_rows(want, got, what):
    """Two :class:`ClassWeightRows`, field for field: a column's classes up to its count (past it, nothing is written)."""
    for name in ("count", "off", "row"):
        a, b = getattr(want, name), getattr(got, name)
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), (what, name)
    assert (want.rep is None) == (got.rep is None), what
    if want.rep is not None:
        assert want.rep.dtype == got.rep.dtype and torch.equal(want.rep, got.rep), (what, "rep")
    kept = torch.arange(want.cls.size(1), device = want.cls.device)[None, :] < want.count[:, None]
    assert want.cls.dtype == got.cls.dtype and torch.equal(want.cls[kept], got.cls[kept]), (what, "cls")
    assert want.starts == got.starts and want.most == got.most and want._groups == got._groups, (what, want.starts, got.starts, want.most, got.most)


@cuda_only
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind, n", CASES)
def test_rows_from_the_class_sets_are_the_rows_from_the_classes_lists(kind, n, storage, shared, build_pc, monkeypatch):
    """A query that has its lists lays its rows out from them, the classes' lists not read: the same rows field for field, a row shared by the classes of the same
    live pairs or not; under a session's per-sample classes too, one set for every sample or a sample's own. Targets before the last position: at the sequence end the
    rows are a refinement (:func:`test_rows_at_the_sequence_end_refine_the_classes_own`)."""
    V, B = 12, 3
    pc = build_pc(kind, n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    monkeypatch.setattr(S.class_rows, "CLASS_ROWS_SHARED", shared)
    made = []
    rows_of = SW.set_weight_rows
    monkeypatch.setattr(CO, "set_weight_rows", lambda *a: (made.append((a, rows_of(*a))), made[-1][1])[1])
    lists, lists_of = [], SW.set_weight_lists
    monkeypatch.setattr(CO, "set_weight_lists", lambda *a: (lists.append(1), lists_of(*a))[1])
    compared, merged, ruled_out = 0, 0, 0
    for seed in range(6):
        dfa = automaton(seed, V)
        cc = jc.compile(dfa, pc)
        if not cc.backend.layout.satisfiable:
            continue
        x, masks = evidence(dfa, n, V, B, seed)
        for name, missing in masks.items():
            made.clear()
            lists.clear()
            targets = missing.clone()
            targets[:, n - 1] = False
            if not targets.any():
                continue
            CO.conditional(cc.backend, x, missing, targets = targets, precision = "fp32", storage = storage)
            (prog, sw, lv, col_pos, col_K, col_first, group_first, batch, stored), got = made[0]
            assert got is not None and prog._class_lists is None and prog.succ.cls_col is None, "the classes' lists were read"
            assert len(lists) == 1 and len(made) == 1, "the query's lists, or its rows, made more than once"
            assert_same_rows(S.class_weight_rows(prog, lv, col_pos, col_K, col_first, group_first, batch, stored), got, (seed, name))
            compared += 1
            if got.rep is not None:
                merged += int(((got.rep >= 0) & (got.rep != torch.arange(prog.C, device = DEV)[None, :])).sum())
            if lv is None:
                prog._class_lists = None
                continue
            g = torch.Generator().manual_seed(seed)
            for samples in (1, batch):                                    # (classes ruled out: for every sample, or a sample's own)
                allowed = (torch.rand(samples, n, prog.C, generator = g) < 0.7).to(torch.int8).to(DEV).contiguous()
                under = types.SimpleNamespace(live = lv.live, allowed = allowed)
                want = S.class_weight_rows(prog, under, col_pos, col_K, col_first, group_first, batch, stored)
                assert_same_rows(want, rows_of(prog, sw, under, col_pos, col_K, col_first, group_first, batch, stored), (seed, name, samples))
                ruled_out += int((want.row < 0).sum() - (got.row < 0).sum())
            prog._class_lists = None                                      # (what this test's own comparison made: the next query is to make none)
    assert compared > 0 and ruled_out > 0 and (merged > 0) == shared, (compared, merged, ruled_out)


@cuda_only
@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_rows_at_the_sequence_end_refine_the_classes_own(storage, build_pc, monkeypatch):
    """At the last position every accepting successor is one column, so the classes' lists make one row of classes that reach different accepting states from the same
    columns; the sets keep them apart: the same live classes, never a row shared that the classes' lists keep apart, and before the last position the same rows."""
    V, n, B = 12, 8, 3
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    made = []
    rows_of = SW.set_weight_rows
    monkeypatch.setattr(CO, "set_weight_rows", lambda *a: (made.append((a, rows_of(*a))), made[-1][1])[1])
    at_end, apart = 0, 0
    for seed in range(8):
        dfa = automaton(seed, V)
        cc = jc.compile(dfa, pc)
        if not cc.backend.layout.satisfiable:
            continue
        x, masks = evidence(dfa, n, V, B, seed)
        for name, missing in masks.items():
            made.clear()
            CO.conditional(cc.backend, x, missing, precision = "fp32", storage = storage)
            (prog, sw, lv, col_pos, col_K, col_first, group_first, batch, stored), got = made[0]
            want = S.class_weight_rows(prog, lv, col_pos, col_K, col_first, group_first, batch, stored)
            prog._class_lists = None
            every = torch.arange(prog.C, device = DEV)
            for m in range(got.count.numel()):
                t = int(col_pos[m // batch])
                assert torch.equal(got.rep[m] >= 0, want.rep[m] >= 0), (seed, name, m)                     # (the same live classes)
                live = got.rep[m] >= 0
                assert torch.equal(got.row[m][live], got.row[m][got.rep[m][live].long()]), (seed, name, m)  # (a class's row is its representative's)
                if t + 1 < n:
                    assert torch.equal(got.rep[m], want.rep[m]) and got.count[m] == want.count[m], (seed, name, m)
                    continue
                at_end += 1
                apart += int(got.count[m] - want.count[m])
                assert got.count[m] >= want.count[m]
                assert torch.equal(want.rep[m][live], want.rep[m][got.rep[m][live].long()]), (seed, name, m)  # (shared by the sets: shared by the classes' lists)
                assert int(got.count[m]) == int((got.rep[m] == every).sum())
    assert at_end > 0 and apart > 0, (at_end, apart)                                                        # (not vacuous)


@cuda_only
@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_rows_from_the_class_sets_share_only_what_the_exact_check_passes(storage, build_pc, monkeypatch):
    """The hash of a class's sets with rows only proposes. With one that tells no two classes apart, every class is checked against the first live one of its column:
    the same live classes, no row shared that the classes' lists keep apart -- other sets, fewer of them, another kind --, some still shared, and the conditionals the
    classes' own up to rounding."""
    V, n, B = 12, 8, 3
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "_collection_keys", lambda live, sets_of, *a: torch.zeros_like(sets_of))
    made = []
    rows_of = SW.set_weight_rows
    monkeypatch.setattr(CO, "set_weight_rows", lambda *a: (made.append((a, rows_of(*a))), made[-1][1])[1])
    refused, shared = 0, 0
    for seed in range(8):
        dfa = automaton(seed, V)
        cc = jc.compile(dfa, pc)
        if not cc.backend.layout.satisfiable:
            continue
        x, masks = evidence(dfa, n, V, B, seed)
        for name, missing in masks.items():
            targets = missing.clone()
            targets[:, n - 1] = False                                     # (the sequence end: a refinement whatever the hash)
            if not targets.any():
                continue
            monkeypatch.setattr(SW, "SET_WEIGHTS", False)
            walked = CO.conditional(cc.backend, x, missing, targets = targets, precision = "fp32", storage = storage).clone()
            monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
            made.clear()
            out = CO.conditional(cc.backend, x, missing, targets = targets, precision = "fp32", storage = storage)
            finite = ~torch.isinf(walked)
            assert torch.equal(torch.isinf(out), ~finite) and float((out[finite] - walked[finite]).abs().max()) <= 1e-4, (seed, name)
            (prog, sw, lv, col_pos, col_K, col_first, group_first, batch, stored), got = made[0]
            want = S.class_weight_rows(prog, lv, col_pos, col_K, col_first, group_first, batch, stored)
            every = torch.arange(prog.C, device = DEV)[None, :]
            live = got.rep >= 0
            assert torch.equal(live, want.rep >= 0), (seed, name)                                             # (the same live classes)
            of_got = got.rep.long().clamp(min = 0)
            assert torch.equal(want.rep.gather(1, of_got)[live], want.rep[live]), (seed, name)                # (shared here: shared by the classes' lists)
            assert torch.equal(got.row.gather(1, of_got)[live], got.row[live]), (seed, name)                  # (a class's row is its representative's)
            assert torch.equal(got.count.long(), (live & (got.rep == every)).sum(dim = 1)), (seed, name)
            assert bool((got.count >= want.count).all()) and got.starts[-1] >= want.starts[-1], (seed, name)
            order = prog.succ.cls_order.view(n, prog.C).index_select(0, col_pos.repeat_interleave(batch)).long()
            place = torch.where(live.gather(1, order), every, prog.C).argmin(dim = 1)                         # (a column's first live class, in its order)
            lead = order.gather(1, place[:, None])
            some = live.any(dim = 1)
            assert torch.equal((got.rep == lead)[some], (want.rep == lead)[some]), (seed, name)               # (the first live class's row: all the classes' lists give it)
            refused += int((got.count - want.count).sum())
            shared += int((live & (got.rep != every)).sum())
    assert refused > 0 and shared > 0, (refused, shared)                                                      # (not vacuous)


@cuda_only
def test_rows_past_their_cells_come_from_the_classes_lists(build_pc, monkeypatch):
    """Tables too large to lay the rows out from the sets -- the (set, class) table an automaton has, or a query's by (target column, class) and (target column, entry)
    --, each to the last cell: the classes' lists then, the weights still through the sets, within rounding of the classes' walks."""
    V, n = 12, 8
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    dfa = automaton(1, V)
    cc = jc.compile(dfa, pc)
    x, masks = evidence(dfa, n, V, 3, 1)
    one = torch.zeros(1, n, dtype = torch.bool, device = DEV)
    one[0, n // 2] = True
    taken = taken_by(monkeypatch)
    made = []
    rows_of = SW.set_weight_rows
    monkeypatch.setattr(CO, "set_weight_rows", lambda *a: (made.append((a, rows_of(*a))), made[-1][1])[1])
    prog = cc.backend.program()
    bound_by = set()
    for data, missing in ((x[:1], one), (x, masks["none"])):               # (one target column; every column of a batch)
        monkeypatch.setattr(SW, "SET_WEIGHTS", False)
        want = CO.conditional(cc.backend, data, missing, precision = "fp32").clone()
        finite = ~torch.isinf(want)
        monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
        monkeypatch.setattr(SW, "SET_WEIGHT_ROW_CELLS", 1 << 26)
        CO.conditional(cc.backend, data, missing, precision = "fp32")
        sw = made[-1][0][1]
        table, query = sw.S * prog.C, sw.count.numel() * max(prog.C, SW._row_tables(prog)[0].numel(), sw.S)
        assert table != query
        bound_by.add(table > query)
        for cells, from_sets in ((max(table, query), True), (max(table, query) - 1, False), (min(table, query), False)):
            monkeypatch.setattr(SW, "SET_WEIGHT_ROW_CELLS", cells)
            prog._class_lists = None
            taken.clear()
            got = CO.conditional(cc.backend, data, missing, precision = "fp32")
            assert (made[-1][1] is not None) == from_sets and (prog._class_lists is None) == from_sets, (cells, table, query)
            assert taken and torch.equal(torch.isinf(got), ~finite) and float((got[finite] - want[finite]).abs().max()) <= 1e-4
    assert bound_by == {True, False}                                        # (each of the two bounds the one that binds, once)


@cuda_only
def test_an_automaton_the_cuda_kernels_share_takes_the_sets_where_they_decline(build_pc, monkeypatch):
    """Compact values under pruning on an automaton of few classes and columns are the CUDA kernels': not every query of it takes the sets, so its rows come from the
    classes' lists; a dense query, or one with no liveness, still goes through the sets by the rule, and a compact one is left as it was, to the bit."""
    V, n, B = 12, 8, 3
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_RATIO", 0.0)
    dfa = automaton(1, V)
    cc = jc.compile(dfa, pc)
    x, masks = evidence(dfa, n, V, B, 1)
    prog = cc.backend.program()
    assert SW.set_weights_apply(prog, None, None) and not SW.set_weights_always(prog)
    taken = taken_by(monkeypatch)
    made = []
    monkeypatch.setattr(CO, "set_weight_rows", lambda *a: made.append(a))
    through = 0
    for name, missing in masks.items():
        for storage in ("dense", "compact"):
            monkeypatch.setattr(SW, "SET_WEIGHTS", False)
            want = CO.conditional(cc.backend, x, missing, precision = "fp32", storage = storage).clone()
            monkeypatch.setattr(SW, "SET_WEIGHTS", True)
            taken.clear()
            got = CO.conditional(cc.backend, x, missing, precision = "fp32", storage = storage)
            assert bool(taken) == (storage == "dense"), (name, storage)
            finite = ~torch.isinf(want)
            if taken:
                through += 1
                assert torch.equal(torch.isinf(got), ~finite) and float((got[finite] - want[finite]).abs().max()) <= 1e-4, (name, storage)
            else:
                assert torch.equal(got, want), (name, storage)
    assert through == len(masks) and not made


@cuda_only
@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_the_hosts_count_takes_the_most_live_columns_of_a_position(storage, build_pc, monkeypatch):
    """What the host's count of a query's pairs goes by: per target position the most live columns of any sample there, read off the liveness -- fewer than the
    boundary has under evidence, and not one sample's --, every column where there is no liveness. Reading it builds the liveness' lists no second time."""
    V, n, B = 12, 8, 3
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    calls, most = [], SW.most_live_columns
    monkeypatch.setattr(CO, "most_live_columns", lambda *a: (calls.append((a, most(*a))), calls[-1][1])[1])
    launches = []

    class Counted:                                                          # (the kernel of the live columns' lists: every launch, and whether with the ranks)
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]
            return lambda *a, **kw: (launches.append(kw["RANK"]), launch(*a, **kw))[1]

    monkeypatch.setattr(LV, "_row_lists_kernel", Counted(LV._row_lists_kernel))
    pruned, uneven, not_the_first, every = 0, 0, 0, 0
    for seed in range(6):
        dfa = automaton(seed, V)
        cc = jc.compile(dfa, pc)
        if not cc.backend.layout.satisfiable:
            continue
        x, masks = evidence(dfa, n, V, B, seed)
        for name, missing in masks.items():
            calls.clear()
            launches.clear()
            CO.conditional(cc.backend, x, missing, precision = "fp32", storage = storage)
            (prog, lv, positions), got = calls[0]
            if storage == "compact":                                        # (the lists with their ranks, as the layout reads them: built once)
                assert launches == [True], (seed, name, launches)
            targets = [t for t in range(n) if positions[t]]
            assert len(calls) == 1 and len(got) == len(targets) > 0
            if lv is None:
                assert got == [prog.width[t] for t in targets]
                every += 1
                continue
            for t, g in zip(targets, got):
                counts = [int((lv.live[b, t, :prog.width[t]] != 0).sum()) for b in range(B)]
                assert g == max(counts), (seed, name, t)
                pruned += g < prog.width[t]
                uneven += max(counts) != min(counts)
                not_the_first += max(counts) != counts[0]
    assert pruned > 0 and uneven > 0 and not_the_first > 0 and (every > 0) == (storage == "dense"), (pruned, uneven, not_the_first, every)


@cuda_only
@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_an_automaton_of_no_lists_by_class_takes_the_sets_for_every_query(storage, build_pc, monkeypatch):
    """An automaton whose lists by class would have more entries than the tables' int32 pointers reach (to the last entry: here the bound the rule reads is
    lowered, and asking for those lists raises as it would): every query goes through the sets at the default rule -- one of a single target too, far below
    the least size of a query elsewhere, and one the CUDA kernels would take --, the lists by class never asked for; the answers the classes' walks' of the
    same automaton with pointers enough, up to rounding. With the sets off, the error as it was."""
    V, n, B = 12, 8, 3
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    dfa = automaton(1, V)
    x, masks = evidence(dfa, n, V, B, 1)
    one = torch.zeros(B, n, dtype = torch.bool, device = DEV)
    one[:, n // 2] = True
    queries = [(masks["half"], one), (masks["half"], None), (masks["none"], None), (masks["scattered"], None)]
    taken = taken_by(monkeypatch)
    cc = jc.compile(dfa, pc)
    prog = cc.backend.program()
    entries = int(TT.boundary_groups(prog._tr).class_columns.sum())
    assert not SW.no_lists_by_class(prog) and not SW.set_weights_always(prog)
    monkeypatch.setattr(SW, "SET_WEIGHTS", False)
    want = [CO.conditional(cc.backend, x, missing, targets = targets, precision = "fp32", storage = storage).clone() for missing, targets in queries]
    monkeypatch.setattr(SW, "SET_WEIGHTS", True)
    CO.conditional(cc.backend, x, masks["half"], targets = one, precision = "fp32", storage = storage)
    assert not taken                                                         # (with lists by class to walk: a query so small is theirs)
    def too_many(self):
        raise RuntimeError(f"{entries} transition entries are more than the tables' int32 pointers can reach")

    for most, none in ((entries, False), (entries - 1, True)):               # (to the last entry)
        monkeypatch.setattr(SW, "TT", types.SimpleNamespace(TABLE_ENTRIES_MOST = most))
        other = jc.compile(dfa, pc).backend
        prog = other.program()
        assert SW.no_lists_by_class(prog) == none
        assert SW.set_weights_always(prog) == none and SW.set_weights_apply(prog, object(), object()) == none and SW.set_weights_apply(prog, None, None) == none
        if not none:
            continue
        monkeypatch.setattr(TT.BoundaryGroups, "class_lists", too_many)
        for (missing, targets), walked in zip(queries, want):
            taken.clear()
            got = CO.conditional(other, x, missing, targets = targets, precision = "fp32", storage = storage)
            finite = ~torch.isinf(walked)
            assert taken and prog._class_lists is None, "a query fell back to the lists by class"
            assert torch.equal(torch.isinf(got), ~finite) and float((got[finite] - walked[finite]).abs().max()) <= 1e-4
        monkeypatch.setattr(SW, "SET_WEIGHTS", False)
        with pytest.raises(RuntimeError, match = "int32 pointers"):
            CO.conditional(other, x, masks["half"], targets = one, precision = "fp32", storage = storage)


@cuda_only
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("mode", ["l2r", "diffusion"])
def test_a_session_through_the_class_sets_is_within_the_bars_of_the_float64_reference(mode, storage, precision, monkeypatch):
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    taken = taken_by(monkeypatch)
    pc, dfa, data = signatures.long_hmm()
    cc = jc.compile(dfa, pc)
    masks = strong.schedule(mode, data.size(0), data.size(1))
    strong.check_against_reference(cc, strong.Float64Reference(pc, dfa, data.size(1)), data, masks, None, storage, precision)
    assert taken, "no query went through the class sets"


@cuda_only
def test_where_the_class_sets_apply(build_pc, monkeypatch):
    """By the automaton alone: never on the host's tables, nor a query the CUDA kernels may take; else by how many times its pairs the classes' lists hold, and how
    many entries a boundary. To the last entry."""
    pc = build_pc("hmm", 8, 12)
    with monkeypatch.context() as m:
        m.setattr(SW, "SET_WEIGHTS", "force")
        assert not SW.set_weights_apply(jc.compile(automaton(0), pc).backend.program(), None, None)       # (the host's tables: no sets, whatever is forced)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    prog = jc.compile(automaton(0), pc).backend.program()
    with monkeypatch.context() as m:
        m.setattr(SW, "SET_WEIGHTS", "force")
        assert SW.set_weights_apply(prog, None, None)
        m.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)   # (a CUDA graph captured: the lists are a query's own)
        assert not SW.set_weights_apply(prog, None, None)
    entries, pairs = prog.transition_stats[0], int(prog.trans_pairs.sum())
    assert entries > pairs > 0
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_ENTRIES", entries // prog.n)
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_RATIO", entries / pairs)
    assert SW.set_weights_apply(prog, None, None) and SW.set_weights_apply(prog, object(), None) and SW.set_weights_apply(prog, None, object())
    assert not SW.set_weights_apply(prog, object(), object())               # (compact values under pruning, few classes and columns: the CUDA kernels' query)
    with monkeypatch.context() as m:                                        # (boundaries too wide for a CTA's shared memory: those kernels decline, to the last column)
        m.setattr(SW, "CLASS_WEIGHT_CUDA_SMEM", 8 * prog.W)
        assert not SW.set_weights_apply(prog, object(), object())
        m.setattr(SW, "CLASS_WEIGHT_CUDA_SMEM", 8 * prog.W - 1)
        assert SW.set_weights_apply(prog, object(), object())
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_RATIO", entries / pairs * 1.001)
    assert not SW.set_weights_apply(prog, None, None)
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_RATIO", 1.0)
    monkeypatch.setattr(SW, "SET_WEIGHT_MIN_ENTRIES", entries // prog.n + 1)
    assert not SW.set_weights_apply(prog, None, None)
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    assert SW.set_weights_apply(prog, object(), object())
    monkeypatch.setattr(SW, "SET_WEIGHTS", False)
    assert not SW.set_weights_apply(prog, None, None)


@cuda_only
@pytest.mark.parametrize("bound, value", [("SET_WEIGHT_PAIRS_MOST", 0), ("SET_WEIGHT_RANKS_MOST", 0), ("SET_WEIGHT_PAIRS_LEAST", None)])
def test_lists_past_their_bounds_fall_back_to_the_classes_walks(bound, value, build_pc, monkeypatch):
    """Too many pairs, too large a rank table, or too few pairs to be worth the lists (to the last pair): the classes' own walks, to the bit, and nothing combined."""
    V, n, B = 12, 8, 2
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    monkeypatch.setattr(SW, "SET_WEIGHT_PAIRS_LEAST", 0)
    dfa = automaton(1, V)
    cc = jc.compile(dfa, pc)
    x, masks = evidence(dfa, n, V, B, 1)
    monkeypatch.setattr(SW, "SET_WEIGHTS", False)
    want = CO.conditional(cc.backend, x, masks["half"], precision = "fp32").clone()
    monkeypatch.setattr(SW, "SET_WEIGHTS", "force")
    taken = taken_by(monkeypatch)
    made, asked = [], []
    lists = SW.set_weight_lists
    monkeypatch.setattr(CO, "set_weight_lists", lambda *a: (asked.append(a), made.append(lists(*a)), made[-1])[2])
    assert not torch.isinf(want).all()
    CO.conditional(cc.backend, x, masks["half"], precision = "fp32")
    assert taken and made[-1].pairs > 0 and len(asked[-1]) == 6
    if value is None:
        args, pairs = asked[-1][:5], made[-1].pairs
        monkeypatch.setattr(SW, bound, pairs)                             # (as many pairs as the query has: still the sets)
        assert lists(*args) is not None
        monkeypatch.setattr(SW, bound, pairs + 1)
        assert lists(*args) is None
        monkeypatch.setattr(SW, bound, 1)                                 # (the host's count first: no live column, nothing asked of the device)
        assert lists(*args, [0]) is None and lists(*args, [1]) is not None
        taken.clear()
        CO.conditional(cc.backend, x, masks["half"], precision = "fp32")  # (and the query's own columns are what it counts)
        assert taken
        monkeypatch.setattr(SW, bound, pairs)                             # (as many by the host's count as the bound: counted on the device, and taken)
        assert lists(*args, [Fraction(pairs, args[4] * max(cc.backend.program().succ.max_pairs, 1))]) is not None
        value = 1 << 62
    if bound == "SET_WEIGHT_PAIRS_MOST":                                  # (past it, as many as a share of the memory left holds: all of these, or none)
        taken.clear()
        monkeypatch.setattr(SW, bound, value)
        CO.conditional(cc.backend, x, masks["half"], precision = "fp32")
        assert taken
        monkeypatch.setattr(SW, "SET_WEIGHT_MEMORY_SHARE", 0.0)
    taken.clear()
    monkeypatch.setattr(SW, bound, value)
    assert torch.equal(CO.conditional(cc.backend, x, masks["half"], precision = "fp32"), want) and not taken
