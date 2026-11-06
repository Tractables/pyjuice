"""
The class weights through the pairs' class sets (kernels/score/set_weights.py) against the classes' own walks: a query's lists hold, for every class, exactly its live
pairs, set by set; the conditionals are the classes' own up to rounding on every kind of circuit, evidence and storage; a decoder session stays within strong_check's
bars of the float64 reference; where the rule takes them, and what falls back to the classes' walks, to the bit.
"""
import dataclasses
import importlib.util
import os
from fractions import Fraction

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels.score import set_weights as SW
from pyjuice.constraints.inference.dfa_exact.kernels.transitions import host as H
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
