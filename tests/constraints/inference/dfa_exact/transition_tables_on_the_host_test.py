"""
The transition tables of a small automaton, built on the host (:mod:`pyjuice.constraints.inference.dfa_exact.kernels.transitions.host`),
against the device's: the same values, types and shapes, whatever the automaton, over random automata and sizes.
"""
import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.distributions import TokenClasses
from pyjuice.constraints.inference.dfa_exact.compiled.layout import Transitions, build_layout
from pyjuice.constraints.inference.dfa_exact.kernels import reachability as R
from pyjuice.constraints.inference.dfa_exact.kernels import transitions as TR
from pyjuice.constraints.inference.dfa_exact.kernels.transitions import host as H
from pyjuice.constraints.inference.dfa_exact.kernels.transitions import signatures as TS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same(a, b, what):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), what
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.to(b.device), b), (what, a.dtype, b.dtype, a.shape, b.shape)
    elif hasattr(a, "__dataclass_fields__"):
        for k in a.__dataclass_fields__:
            assert_same(getattr(a, k), getattr(b, k), f"{what}.{k}")
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{what}[{i}]")
    elif hasattr(a, "tolist"):
        assert a.tolist() == b.tolist(), what
    else:
        assert a == b, (what, a, b)


def random_automaton(seed):
    """Sizes from one state up, states that never leave, nothing or much accepted, classes of one token (enumerated) and of several, tokens
    of no class seen anywhere: what the tables read."""
    g = torch.Generator().manual_seed(seed)
    K, C = int(torch.randint(1, 48, (1,), generator = g)), int(torch.randint(1, 12, (1,), generator = g))
    delta = torch.randint(0, K, (K, C), generator = g)
    if seed % 3 == 1:                                          # states that never accept and never leave, entered often
        delta[K // 2:] = torch.arange(K // 2, K)[:, None]
        delta = torch.where(torch.rand(K, C, generator = g) < 0.5, torch.full((K, C), K - 1), delta)
    elif seed % 3 == 2:                                        # few successors a column, many classes to each
        delta = delta % max(K // 4, 1)
    sizes = 1 + (torch.randint(0, 2, (C,), generator = g) * torch.randint(1, 4, (C,), generator = g))
    token_class = torch.repeat_interleave(torch.arange(C), sizes)
    token_class = token_class[torch.randperm(token_class.numel(), generator = g)]
    accept = [] if seed % 11 == 5 else torch.randperm(K, generator = g)[:max(K // 3, 1)].tolist()
    return jc.DFA(token_class.numel(), token_class, delta, 0, accept)


def on_the_device(tr):
    return Transitions(tr.delta, tr.state_id, tr.col_of_state, tr.width)


@pytest.mark.parametrize("enumerate_max", [0, 1])
@pytest.mark.parametrize("seed", range(40))
def test_the_tables_of_a_random_automaton_are_the_devices(seed, enumerate_max):
    dfa = random_automaton(seed)
    for n in (1, 2, 5, 16, 33):
        layout = build_layout(dfa, n, device = DEV)
        tr, off = layout.transitions(DEV), None
        off = on_the_device(tr)
        assert isinstance(TR.boundary_groups(tr), H.HostTables) and isinstance(TR.boundary_groups(off), TR.BoundaryGroups)
        host_groups, device_groups = TR.boundary_groups(tr), TR.boundary_groups(off)
        assert host_groups.firsts == device_groups.firsts and host_groups.group == device_groups.group, (seed, n)
        assert host_groups.at.tolist() == device_groups.at.tolist()
        assert_same(host_groups.class_columns, device_groups.class_columns, "class columns")
        for k, end in device_groups.batches:
            for a, b in zip(host_groups.entries(k, end), device_groups.entries(k, end)):
                assert_same(a, b, "entries")
        classes = TokenClasses(layout.token_class, dfa.num_classes, enumerate_max, host = layout.token_class.numpy(), device = DEV)
        en, en_off = TR.enumerated_tokens(tr, classes), TR.enumerated_tokens(off, classes)
        assert_same(en, en_off, f"enumerated {seed} {n}")
        assert_same(TR.next_col_table(tr), TR.next_col_table(off), "next col")
        assert TR.transition_stats(tr) == TR.transition_stats(off), (seed, n, "before the tables")
        for grouped in (True, False):
            assert_same(TR.transition_tables(tr, grouped, en), TR.transition_tables(off, grouped, en_off), f"tables {seed} {n} {grouped}")
            assert_same(TR.transition_tables(tr, grouped), TR.transition_tables(off, grouped), f"tables {seed} {n} {grouped}, no enumeration")
        assert TR.transition_stats(tr) == TR.transition_stats(off), (seed, n, "after the tables")
        sweeps, sweeps_off = R.SweepTables(tr), R.SweepTables(off)
        assert sweeps.group == sweeps_off.group and sweeps.W == sweeps_off.W
        assert_same(sweeps.ptr, sweeps_off.ptr, "sweep ptr")
        assert sweeps.cols.numel() == sweeps_off.cols.numel() and torch.equal(sweeps.cols, sweeps_off.cols), (seed, n)


@pytest.mark.parametrize("enumerate_max", [0, 1])
@pytest.mark.parametrize("seed", range(12))
def test_token_classes_are_the_same_read_on_the_host(seed, enumerate_max):
    dfa = random_automaton(seed)
    host = TokenClasses(dfa.token_class, dfa.num_classes, enumerate_max, host = dfa.token_class.numpy(), device = DEV)
    device = TokenClasses(dfa.token_class.to(DEV), dfa.num_classes, enumerate_max)
    for name in ("token_class", "stored_of_class", "class_of_stored", "class_of_stored32", "stored_class", "enumerated_token", "enumerated_tokens"):
        assert_same(getattr(host, name), getattr(device, name), name)
    assert host.num_stored == device.num_stored and host.num_enumerated == device.num_enumerated


@pytest.mark.parametrize("limit", ["HOST_ENTRIES", "HOST_MOVES"])
def test_an_automaton_with_many_entries_stays_on_the_device(limit, monkeypatch):
    dfa = random_automaton(3)
    tr = build_layout(dfa, 16, device = DEV).transitions(DEV)
    monkeypatch.setattr(H, limit, 1)
    assert isinstance(TR.boundary_groups(tr), TR.BoundaryGroups)
    assert TR.transition_tables(tr, True) is not None


@pytest.mark.parametrize("big", [False, True])
def test_a_table_over_every_boundary_past_its_limit_is_the_same(big, monkeypatch):
    """From the distinct boundaries' table on the device, or from the entries where that is large too."""
    dfa = random_automaton(9)
    tr = build_layout(dfa, 33, device = DEV).transitions(DEV)
    want = TR.next_col_table(on_the_device(tr))
    monkeypatch.setattr(H, "HOST_TABLE_ENTRIES", 0 if big else tr.W * tr.C * len(set(TR.boundary_groups(tr).group)))
    assert_same(TR.next_col_table(tr), want, "next col")


CASES = [("hmm", 8), ("hmm_block_sparse", 8), ("pd", 8), ("pd_blockified", 8), ("hand", 5), ("hand_left", 5)]
AUTOMATA = {"contains": lambda V: jc.DFA.contains([[1, 2]], V), "either": lambda V: jc.DFA.contains([[1, 2]], V) | jc.DFA.contains([[2, 2]], V),
            "absent": lambda V: jc.Not(jc.DFA.contains([[2, 2]], V))}


@pytest.mark.parametrize("name", list(AUTOMATA))
@pytest.mark.parametrize("kind, n", CASES)
def test_a_programs_signature_layout_is_the_devices(kind, n, name, build_pc, monkeypatch):
    """A Program's tables and the layout of its masses by column signature, compiled both ways, on circuits of every shape."""
    V = 3
    pc = build_pc(kind, n, V)

    def program(entries):
        monkeypatch.setattr(H, "HOST_ENTRIES", entries)
        cc = jc.compile(AUTOMATA[name](V), pc)
        prog = cc.backend.program()
        masses = cc.backend.state.masses()
        return cc, prog, TS.signature_masses(prog, masses, prog.trans_size), isinstance(TR.boundary_groups(prog._tr), H.HostTables)

    cc, prog, sig, on_host = program(1 << 20)
    cc_off, prog_off, sig_off, off = program(0)
    assert on_host and not off
    assert (sig is None) == (sig_off is None)
    if sig is not None:
        assert_same(sig, sig_off, f"signature {kind} {name}")
    for field in ("succ", "enum", "next_col"):
        assert_same(getattr(prog, field), getattr(prog_off, field), f"{field} {kind} {name}")
    assert_same(prog.ungrouped_tables(), prog_off.ungrouped_tables(), f"ungrouped {kind} {name}")
    assert prog.transition_stats == prog_off.transition_stats


def enumerating_automaton(seed, V):
    """Few states, so that columns repeat their signatures; most classes of one token, so that they are enumerated."""
    g = torch.Generator().manual_seed(seed)
    K, C = int(torch.randint(2, 9, (1,), generator = g)), int(torch.randint(3, V, (1,), generator = g))
    delta = torch.randint(0, K, (K, C), generator = g)
    delta = torch.where(torch.rand(K, C, generator = g) < 0.3, torch.full((K, C), K - 1), delta)
    token_class = torch.cat([torch.arange(C), torch.randint(0, 2, (V - C,), generator = g)])     # (the first two classes: several tokens)
    return jc.DFA(V, token_class[torch.randperm(V, generator = g)], delta, 0, torch.randperm(K, generator = g)[:max(K // 2, 1)].tolist())


@pytest.mark.parametrize("kind, n", [("hmm", 24), ("hmm_untied", 12)])
def test_signatures_of_enumerated_tokens_and_of_positions_apart_are_the_devices(kind, n, build_pc, monkeypatch):
    """Columns told apart by their enumerated tokens alone, positions reading rows of their own (each a class of positions, with its own
    representatives): the same layout, column for column. What every saving would have refused is built here too."""
    from pyjuice.constraints.inference.dfa_exact import backend as BE
    V = 12
    pc = build_pc(kind, n, V)
    monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", 1)
    monkeypatch.setattr(BE, "ENUMERATE_MIN_FRACTION", 0.0)
    monkeypatch.setattr(TS, "SIGNATURE_MIN_SAVING", 0.0)
    enumerated, classes, compared = 0, 0, 0
    for seed in range(24):
        layouts = []
        for entries in (1 << 20, 0):
            monkeypatch.setattr(H, "HOST_ENTRIES", entries)
            cc = jc.compile(enumerating_automaton(seed, V), pc)
            if not cc.backend.layout.satisfiable:
                layouts = None
                break
            prog = cc.backend.program()
            layouts.append((prog, TS.signature_masses(prog, cc.backend.state.masses(), prog.trans_size), cc))
        if layouts is None:
            continue
        (prog, sig, _), (prog_off, sig_off, _) = layouts
        assert isinstance(TR.boundary_groups(prog._tr), H.HostTables) and not isinstance(TR.boundary_groups(prog_off._tr), H.HostTables)
        assert (sig is None) == (sig_off is None), seed
        if sig is None:                                        # (a position without steps, say: nothing to lay out)
            continue
        assert_same(sig, sig_off, f"signature {kind} {seed}")
        compared += 1
        enumerated += prog.enum.num > 0
        classes = max(classes, prog.assembly.position_classes.steps.size)
    assert compared >= 8 and enumerated >= 4, (compared, enumerated)   # (not vacuous)
    assert (classes > 1) == (kind == "hmm_untied"), classes


def test_two_groups_with_columns_alike_share_their_signatures(build_pc):
    """A long sequence: boundaries alike, one group, so no position repeats the work; the layout still covers every position."""
    V = 3
    pc = build_pc("hmm", 32, V)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    prog = cc.backend.program()
    sig = TS.signature_masses(prog, cc.backend.state.masses(), prog.trans_size)
    groups = TR.boundary_groups(prog._tr)
    assert len(groups.firsts) < prog.n
    assert sig is not None and sig.pairs.numel() == prog.n
