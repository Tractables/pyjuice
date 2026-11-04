"""
What backends share of a circuit (:mod:`pyjuice.constraints.inference.dfa_exact.compiled.circuit`): constraints compiled
one after the other on one circuit share its tables and answer, bit for bit, as on a circuit of their own, whatever
order they run in; the tables follow the switches they depend on and the circuit's device, and do not keep the circuit
alive. And the automaton's tables (:mod:`~pyjuice.constraints.inference.dfa_exact.kernels.transitions`) do not depend on
the batches they are built in.
"""
import gc
import os
import random
import subprocess
import sys
import weakref

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.compiled import program as P
from pyjuice.constraints.inference.dfa_exact.compiled.layout import Reachability, build_layout, interval_tables
from pyjuice.constraints.inference.dfa_exact.kernels.prod import skip_masks
from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
from pyjuice.constraints.inference.dfa_exact.kernels import transitions as TR

V = 6
#: a chain, a tree, a tree over several products per sum, and a hand-built circuit with folds: conftest.py's
KINDS = {"hmm": 6, "hmm_block_sparse": 6, "pd": 8, "pd_prod_dominated": 8, "hand_fold": 4, "walks": 6}


def automata():
    """Automata that keep every pair of an interval (or not) differently, of one state to tens of them."""
    g = torch.Generator().manual_seed(0)
    K = 24
    return {"everything": jc.DFA.from_dense(V, [[0] * V], 0, [0]),
            "contains": jc.DFA.contains([[1, 2]], V),
            "either": jc.DFA.contains([[2]], V) | jc.DFA.contains([[0, 0]], V),
            "random": jc.DFA.from_dense(V, torch.randint(0, K, (K, V), generator = g), 0, list(range(0, K, 2)))}


def answers(cc, n):
    """A query that reads the forward pass and one that reads the top-down pass, under some evidence."""
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, V, (4, n), generator = g).cuda()
    missing = (torch.rand(4, n, generator = g) < 0.6).cuda()
    return cc.marginal(data, missing).clone(), cc.conditional(data, missing, return_log_marginal = True)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    else:
        for x, y in zip(a, b):
            same(x, y)


@pytest.mark.parametrize("kind", list(KINDS))
def test_constraints_on_one_circuit_answer_as_on_their_own(kind, build_pc):
    """Compiled one after the other on one circuit, sharing its tables, and queried in turn (so that one's tables or
    state would show in the next's), against a circuit of their own, built alike: the same bits."""
    n = KINDS[kind]
    shared = build_pc(kind, n, V)
    named = automata()
    ccs = {name: jc.compile(dfa, shared) for name, dfa in named.items()}
    tables = {id(cc.backend.program().assembly.circuit) for cc in ccs.values()}
    assert len(tables) == 1, "the constraints on one circuit do not share its tables"
    got = {name: answers(cc, n) for name, cc in ccs.items()}
    got_again = {name: answers(ccs[name], n) for name in reversed(list(ccs))}
    for name, dfa in named.items():
        want = answers(jc.compile(dfa, build_pc(kind, n, V)), n)
        same(got[name], want)
        same(got_again[name], want)


def test_constraints_of_another_pattern_take_their_own_launches(build_pc):
    """On a tree, whether an interval keeps every pair groups its steps into launches, which automata disagree on: each
    pattern has its own, and a pattern seen before is reused."""
    n = KINDS["pd"]
    pc = build_pc("pd", n, V)
    g = torch.Generator().manual_seed(2)
    K = 24
    a = jc.compile(jc.DFA.from_dense(V, [[0] * V], 0, [0]), pc).backend.program()         # every pair kept
    b = jc.compile(jc.DFA.from_dense(V, torch.randint(0, K, (K, V), generator = g), 0, list(range(K))), pc).backend.program()
    c = jc.compile(jc.DFA.from_dense(V, [[0] * V], 0, [0]), pc).backend.program()
    assert a.assembly.circuit is b.assembly.circuit
    assert a.assembly is c.assembly
    assert a.assembly is not b.assembly and any(b.reach.positions(*iv) is not None for iv in b.assembly.circuit.flag_ivs)


def test_the_tables_follow_their_switches(build_pc, monkeypatch):
    """What they depend on besides the circuit: a change of it is never answered from what was kept."""
    pc = build_pc("hmm", KINDS["hmm"], V)
    dfa = jc.DFA.contains([[1, 2]], V)

    def groups():
        for kind, _, layers in jc.compile(dfa, pc).backend.program().steps:
            if kind == "prod":
                for stages, _, _ in layers:
                    for launches in stages:
                        for launch in launches:
                            if launch.groups_host:
                                return tuple(launch.groups_host)
        return ()

    default = groups()
    monkeypatch.setattr(P, "INPUT_BLOCK_GROUP", 2)
    monkeypatch.setattr(P, "INPUT_BLOCK_GROUP_ALTERNATIVES", ())
    assert groups() == (2,) != default
    monkeypatch.undo()
    assert groups() == default
    first = jc.compile(dfa, pc).backend.program()
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30)
    other = jc.compile(dfa, pc).backend.program()
    assert other.assembly.circuit is not first.assembly.circuit


def test_the_tables_do_not_keep_the_circuit_alive(build_pc):
    pc = build_pc("hmm", KINDS["hmm"], V)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    cc.backend.top_down_program()
    alive = weakref.ref(pc)
    del pc, cc
    gc.collect()
    assert alive() is None, "something kept with the circuit keeps it"


_MOVED = """
import sys, threading
sys.setrecursionlimit(1 << 20)
import torch
import pyjuice as juice, pyjuice.constraints as jc

def build(dev):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 6, num_latents = 64, num_emits = 6)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(dev)

def answers(cc, dev):
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, 6, (4, 6), generator = g).to(dev)
    missing = (torch.rand(4, 6, generator = g) < 0.6).to(dev)
    return cc.marginal(data, missing), cc.conditional(data, missing)

dfa = jc.DFA.contains([[1, 2]], 6)
pc = build("cuda:0")
cc = jc.compile(dfa, pc)
answers(cc, "cuda:0")
first = cc.backend.program()
cc2 = jc.compile(jc.DFA.contains([[2]], 6), pc)
assert cc2.backend.program().assembly.circuit is first.assembly.circuit
moved = cc.to("cuda:1")
got = answers(moved, "cuda:1")
prog = moved.backend.program()
assert prog.assembly.circuit is not first.assembly.circuit
assert prog.assembly.circuit.dev == torch.device("cuda:1") and prog.reg_first.device == torch.device("cuda:1")
want = answers(jc.compile(dfa, build("cuda:1")), "cuda:1")
for a, b in zip(got, want):
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
print("CHILD_OK")
"""


def _two_gpus():
    """Two GPUs of the suite's pool for a child of its own (its workers see one each), or None."""
    pool = [g for g in os.environ.get("PYJUICE_TEST_GPU_POOL", "").split(",") if g]
    return pool[:2] if len(pool) >= 2 else None


@pytest.mark.skipif(_two_gpus() is None, reason = "requires two GPUs in PYJUICE_TEST_GPU_POOL")
def test_the_tables_follow_the_circuit_to_another_device():
    """A circuit moved: its next backend's tables are built on the new device, and answer as a twin's there."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(_two_gpus())
    env.pop("PYJUICE_TEST_GPU_POOL", None)
    proc = subprocess.run([sys.executable, "-c", _MOVED], capture_output = True, text = True, timeout = 600, env = env)
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_the_batches_do_not_change_the_automatons_tables(seed, grouped, monkeypatch):
    """One boundary a batch, none kept, against all of them together."""
    g = torch.Generator().manual_seed(seed)
    K, C, n = 12, 5, 7
    delta = torch.randint(0, K, (K, C), generator = g)
    layout = build_layout(jc.DFA(C, torch.arange(C), delta, 0, list(range(0, K, 3))), n)
    together = TR.transition_tables(layout.transitions(torch.device("cuda")), grouped)
    monkeypatch.setattr(TR, "BOUNDARY_BATCH", 1)
    monkeypatch.setattr(TR, "BOUNDARY_ENTRIES_KEPT", -1)
    tr = layout.transitions(torch.device("cuda"))
    del tr.__dict__["_boundary_groups"]                               # (kept with the transitions: built again)
    assert len(TR.boundary_groups(tr).batches) == len(TR.boundary_groups(tr).firsts) > 1
    apart = TR.transition_tables(tr, grouped)
    for k in together.__dataclass_fields__:
        a, b = getattr(together, k), getattr(apart, k)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, k


def test_boundaries_alike_are_found_exactly_whatever_their_hash(monkeypatch):
    """The hash only proposes: boundaries that merely hash alike are told apart."""
    layout = build_layout(jc.DFA.contains([[1, 2]], 3), 12)
    tr = layout.transitions(torch.device("cuda"))
    want = TR.BoundaryGroups(tr)
    monkeypatch.setattr(TR, "_boundary_weights", lambda W, dev: torch.zeros(W, dtype = torch.int64, device = dev))
    got = TR.BoundaryGroups(tr)
    assert got.firsts == want.firsts and got.group == want.group and len(got.firsts) > 1


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("tile", [1, 2, 16])
def test_reachability_on_the_device_is_the_hosts(seed, tile):
    """Bitset sweeps on the device against the dense ones on the host: every interval's slots, positions, occupancy
    and pairs, whatever order they are asked in; the skip masks and interval tables read from them too."""
    g = torch.Generator().manual_seed(seed)
    K, C, n = 40, 5, 9
    dfa = jc.DFA(C, torch.arange(C), torch.randint(0, K, (K, C), generator = g), 0, list(range(0, K, 2)))
    host = Reachability(build_layout(dfa, n), tile = tile)
    dfa_d = jc.DFA(C, torch.arange(C), dfa.delta, 0, dfa.accept)
    device = Reachability(build_layout(dfa_d, n, device = torch.device("cuda")), tile = tile)
    pairs = [(a, b) for a in range(n + 1) for b in range(a, n + 1)]
    random.Random(seed).shuffle(pairs)
    for a, b in pairs:
        assert device.slots(a, b) == host.slots(a, b), (a, b)
        assert device.keeps_every_pair(a, b) == (host.positions(a, b) is None), (a, b)
        p, q = device.positions(a, b), host.positions(a, b)
        assert (p is None) == (q is None) and (p is None or torch.equal(p, q)), (a, b)
        assert torch.equal(device.occupancy(a, b), host.occupancy(a, b)), (a, b)
        assert torch.equal(device.pairs(a, b), host.pairs(a, b)), (a, b)
    intervals = [(a, b) for a, b in pairs if b > a]
    specs = [(a, b, c) for a in range(n - 1) for b in range(a + 1, n) for c in range(b + 1, n + 1)]
    if tile <= 16 and 16 % tile == 0:
        got, want = skip_masks(device, specs, device = torch.device("cuda")), skip_masks(host, specs, device = torch.device("cuda"))
        assert (got.tiles_i, got.tiles_j, got.words) == (want.tiles_i, want.tiles_j, want.words)
        assert torch.equal(got.bits, want.bits)
    got, want = (interval_tables(r, intervals, torch.device("cuda")) for r in (device, host))
    assert torch.equal(got.of, want.of)
    for (S, at, ap), (S2, at2, ap2) in zip(got.info.tolist(), want.info.tolist()):
        assert S == S2 and (at < 0) == (at2 < 0)
        if at >= 0:
            assert torch.equal(got.table[at:ap + S], want.table[at2:ap2 + S2])


def test_alike_intervals_share_their_positions():
    """Past the shortest accepted string most boundaries are alike: an interval through them keeps one table, whatever
    its entry -- what makes an automaton of thousands of columns on a tree fit."""
    K, C, n = 60, 4, 24
    g = torch.Generator().manual_seed(0)
    dfa = jc.DFA(C, torch.arange(C), torch.randint(0, K, (K, C), generator = g), 0, list(range(K)))
    layout = build_layout(dfa, n, device = torch.device("cuda"))
    reach = layout.reachability
    intervals = [(a, a + 2) for a in range(1, n - 2)]
    tables = interval_tables(reach, intervals, torch.device("cuda"))
    separately = sum(reach.width[a] * reach.width[b] + reach.slots(a, b) for a, b in intervals if not reach.keeps_every_pair(a, b))
    assert separately > 0 and tables.table.numel() < separately / 2
