"""
The transition tables of a small automaton, built on the host (:mod:`pyjuice.constraints.inference.dfa_exact.kernels.transitions.host`),
against the device's: the same values, types and shapes, whatever the automaton, over random automata and sizes.
"""
import dataclasses

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
    return Transitions(tr.delta, tr.state_id, tr.col_of_state, tr.width, tr.order)


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


@pytest.mark.parametrize("seed", range(12))
def test_every_boundarys_columns_keep_the_order_of_the_states(seed):
    """What a state's pairs rest on: one order of the states, which the columns of every boundary keep."""
    dfa = random_automaton(seed)
    for n in (1, 5, 33):
        tr = build_layout(dfa, n, device = DEV).transitions(DEV)
        assert sorted(tr.order.tolist()) == list(range(tr.K))
        rank = torch.empty_like(tr.order)
        rank[tr.order] = torch.arange(tr.K, device = DEV)
        for t, w in enumerate(tr.width):
            states = tr.state_id[t, :w]
            assert (rank[states][1:] > rank[states][:-1]).all(), (seed, n, t)
            assert tr.col_of_state[t, states].tolist() == list(range(w)), (seed, n, t)


@pytest.mark.parametrize("seed", range(12))
def test_a_states_pairs_are_the_successors_of_its_moves(seed):
    """Against the moves, a state at a time: its distinct successors in the order of the columns, and every move's among them."""
    tr = on_the_device(build_layout(random_automaton(seed), 16, device = DEV).transitions(DEV))
    groups = TR.boundary_groups(tr)
    state, moves = groups.state.clone(), groups.succ.clone()
    count, start, succ, of_move = groups.pairs
    rank = {s: i for i, s in enumerate(tr.order.tolist())}
    for x in range(tr.K):
        here = state == x
        want = sorted(set(moves[here].tolist()), key = rank.__getitem__)
        assert succ[start[x]:start[x] + count[x]].tolist() == want, (seed, x)
        assert succ[start[x] + of_move[here].long()].tolist() == moves[here].tolist(), (seed, x)


@pytest.mark.parametrize("enumerate_max", [0, 1])
@pytest.mark.parametrize("seed", range(12))
def test_the_tables_a_group_of_boundaries_at_a_time_are_the_hosts(seed, enumerate_max, monkeypatch):
    """Every group its own batch, so that a sequence end has one of its own, and none kept: the same tables, and a batch's numbers are their group's."""
    monkeypatch.setattr(TR.tables, "BOUNDARY_BATCH", 1)
    monkeypatch.setattr(TR.tables, "BOUNDARY_ENTRIES_KEPT", -1)
    dfa = random_automaton(seed)
    for n in (2, 5, 16):
        layout = build_layout(dfa, n, device = DEV)
        tr = layout.transitions(DEV)
        off = on_the_device(tr)
        classes = TokenClasses(layout.token_class, dfa.num_classes, enumerate_max, host = layout.token_class.numpy(), device = DEV)
        en, en_off = TR.enumerated_tokens(tr, classes), TR.enumerated_tokens(off, classes)
        assert_same(en, en_off, f"enumerated {seed} {n}")
        for grouped in (True, False):
            assert_same(TR.transition_tables(tr, grouped, en), TR.transition_tables(off, grouped, en_off), f"tables {seed} {n} {grouped}")
        assert TR.transition_stats(tr) == TR.transition_stats(off), (seed, n)


def test_the_enumerated_tokens_and_the_grouped_tables_take_one_pass(monkeypatch):
    """Over the moves, the most of what a long layout costs. For the same tokens; for others, or a second time, a pass of its own, and the same tables."""
    dfa = jc.DFA.contains([[1, 2]], 5)
    layout = build_layout(dfa, 16, device = DEV)
    off = on_the_device(layout.transitions(DEV))
    classes = TokenClasses(layout.token_class, dfa.num_classes, 1, host = layout.token_class.numpy(), device = DEV)
    assert classes.num_enumerated > 0
    passes = []
    sweep = TR.tables.BoundaryGroups.sweep
    monkeypatch.setattr(TR.tables.BoundaryGroups, "sweep", lambda self, *a, **k: (passes.append(a[0]), sweep(self, *a, **k))[1])
    en = TR.enumerated_tokens(off, classes)
    assert passes == [True] and TR.boundary_groups(off).swept[0] is en
    first = TR.transition_tables(off, True, en)
    assert passes == [True] and TR.boundary_groups(off).swept is None and first.num_entries > 0
    assert_same(first, TR.transition_tables(off, True, en), "a second time")
    assert passes == [True, True]
    assert_same(TR.enumerated_tokens(off, classes), en, "again")
    other = TR.enumerated_tokens(off, classes)
    assert_same(first, TR.transition_tables(off, True, other), "the same tokens, another object")
    assert TR.boundary_groups(off).swept is None


def test_a_pass_without_the_lists_is_for_tables_without_them(monkeypatch):
    """What a program asks for: one pass, no lists, and the rest the same. Tables with the lists, after it: a pass of their own, for them."""
    dfa = jc.DFA.contains([[1, 2]], 5)
    layout = build_layout(dfa, 16, device = DEV)
    tr = layout.transitions(DEV)
    off = on_the_device(tr)
    classes = TokenClasses(layout.token_class, dfa.num_classes, 1, host = layout.token_class.numpy(), device = DEV)
    passes = []
    sweep = TR.tables.BoundaryGroups.sweep
    monkeypatch.setattr(TR.tables.BoundaryGroups, "sweep", lambda self, *a, **k: (passes.append(k.get("lists", True)), sweep(self, *a, **k))[1])
    en = TR.enumerated_tokens(off, classes, lists = False)
    without = TR.transition_tables(off, True, en, lists = False)
    assert passes == [False] and without.enum_e is None and without.enum_pair is None
    want = TR.transition_tables(tr, True, TR.enumerated_tokens(tr, classes))
    assert_same(want, dataclasses.replace(without, enum_e = want.enum_e, enum_pair = want.enum_pair), "without the lists")
    code = en.code.long()
    stats = TR.transition_stats(off)
    e, pair = TR.boundary_groups(off).enumerated_lists(code)
    assert_same(e, want.enum_e, "every column's")
    assert_same(pair, want.enum_pair, "every column's")
    assert TR.transition_stats(off) == stats                            # (a pass for the lists alone: nothing else it leaves changed)
    TR.enumerated_tokens(off, classes, lists = False)
    other = TR.enumerated_tokens(off, classes, lists = False)
    assert_same(want, TR.transition_tables(off, True, other), "with them, after a pass without")


@pytest.mark.parametrize("asked", ["no tokens", "ungrouped", "no tokens, ungrouped"])
def test_a_pass_left_for_the_grouped_tables_of_these_tokens_is_for_nothing_else(asked):
    """What the enumerated tokens' pass made, asked for anything else first: that thing, and the pass let go of."""
    dfa = jc.DFA.contains([[1, 2]], 5)
    layout = build_layout(dfa, 16, device = DEV)
    tr = layout.transitions(DEV)
    off = on_the_device(tr)
    classes = TokenClasses(layout.token_class, dfa.num_classes, 1, host = layout.token_class.numpy(), device = DEV)
    en, en_off = TR.enumerated_tokens(tr, classes), TR.enumerated_tokens(off, classes)
    grouped = "ungrouped" not in asked
    tokens = "no tokens" not in asked
    got = TR.transition_tables(off, grouped, en_off if tokens else None)
    assert got.grouped == grouped and TR.boundary_groups(off).swept is None
    assert_same(TR.transition_tables(tr, grouped, en if tokens else None), got, asked)


def test_a_programs_tables_read_the_moves_once_and_no_entries(monkeypatch):
    """The statistics and the classes' columns come with the pass: no second one, none of what a pass makes unneeded."""
    dfa = jc.DFA.contains([[1, 2]], 5)
    layout = build_layout(dfa, 16, device = DEV)
    off = on_the_device(layout.transitions(DEV))
    classes = TokenClasses(layout.token_class, dfa.num_classes, 1, host = layout.token_class.numpy(), device = DEV)
    def no_entries(self, k, end):
        raise AssertionError("a pass over the entries")
    monkeypatch.setattr(TR.tables.BoundaryGroups, "entries", no_entries)
    en = TR.enumerated_tokens(off, classes)
    assert TR.transition_tables(off, True, en).num_entries > 0
    assert TR.transition_stats(off)[2] >= 1 and TR.boundary_groups(off).class_columns.numel() > 0


@pytest.mark.parametrize("domain", [2, (1 << 15) - 1, 1 << 15, (1 << 15) + 1, 1 << 16, 1 << 31, (1 << 31) + 1, 1 << 40])
def test_a_sort_key_in_the_smallest_type_is_the_key(domain):
    """To the last value below the domain; a stable sort of it the sort of the key."""
    key = torch.tensor([domain - 1, 0, domain // 2, domain - 1, 0], device = DEV)
    small = TR.tables._sort_key(key, domain)
    assert small.tolist() == key.tolist()
    assert torch.equal(torch.argsort(small, stable = True), torch.argsort(key, stable = True))
    assert small.dtype == (torch.int16 if domain <= 1 << 15 else torch.int32 if domain <= 1 << 31 else torch.int64)


def test_more_entries_than_int32_pointers_reach_is_an_error_not_a_wrong_table(monkeypatch):
    off = on_the_device(build_layout(jc.DFA.contains([[1, 2]], 5), 16, device = DEV).transitions(DEV))
    monkeypatch.setattr(TR.tables, "TABLE_ENTRIES_MOST", 0)
    with pytest.raises(RuntimeError, match = "int32"):
        TR.transition_tables(off, True)


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


def whole(prog):
    """A program's grouped tables with every column's enumerated entries, which on the device come when asked for."""
    ptr, e, pair = prog.enum_lists()
    return dataclasses.replace(prog.succ, enum_ptr = ptr, enum_e = e, enum_pair = pair)


def as_on_the_host(sig):
    """A signature layout without its own lists, which only the device's has."""
    return dataclasses.replace(sig, rep_enum_ptr = None, rep_enum_e = None, rep_enum_pair = None)


def assert_lists_of_representatives(prog, sig):
    """A signature layout's lists on the device: of its representative columns those columns' whole, of the others none."""
    if not prog.enum.on:
        assert sig.rep_enum_ptr is None
        return
    ptr, e, pair = prog.enum_lists()
    reps = {(t, q) for t in range(prog.n) for q in sig.rep_col[sig.rep_ptr[t]:sig.rep_ptr[t + 1]].tolist()}
    own = sig.rep_enum_ptr
    assert own.shape == ptr.shape and own.dtype == ptr.dtype
    for t in range(prog.n):
        for q in range(prog.W):
            a, b = int(own[t, q]), int(own[t, q + 1])
            if (t, q) not in reps:
                assert a == b, (t, q)
                continue
            c, d = int(ptr[t, q]), int(ptr[t, q + 1])
            assert sig.rep_enum_e[a:b].tolist() == e[c:d].tolist(), (t, q)
            assert sig.rep_enum_pair[a:b].tolist() == pair[c:d].tolist(), (t, q)


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
        assert_same(sig, as_on_the_host(sig_off), f"signature {kind} {name}")
        assert_lists_of_representatives(prog_off, sig_off)
    assert_same(prog.succ, whole(prog_off), f"succ {kind} {name}")
    for field in ("enum", "next_col"):
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
        assert_same(sig, as_on_the_host(sig_off), f"signature {kind} {seed}")
        assert_lists_of_representatives(prog_off, sig_off)
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


def test_masses_by_signature_read_no_columns_lists_but_their_representatives(build_pc, monkeypatch):
    """A first query on the device: no pass for every column's enumerated entries. Asked for, they come once, and are the host's."""
    from pyjuice.constraints.inference.dfa_exact import backend as BE
    V, n = 12, 24
    pc = build_pc("hmm", n, V)
    monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", 1)
    monkeypatch.setattr(BE, "ENUMERATE_MIN_FRACTION", 0.0)
    monkeypatch.setattr(TS, "SIGNATURE_MIN_SAVING", 0.0)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    for seed in range(24):
        cc = jc.compile(enumerating_automaton(seed, V), pc)
        prog = cc.backend.program()
        if cc.backend.layout.satisfiable and prog.enum.num > 0:
            break
    assert prog.succ.enum_e is None and prog.succ.enum_ptr is not None
    lists = TR.tables.BoundaryGroups.enumerated_lists

    def not_read(self, code):
        raise AssertionError("every column's enumerated entries were read")
    with monkeypatch.context() as m:
        m.setattr(TR.tables.BoundaryGroups, "enumerated_lists", not_read)
        m.setattr(prog, "group_worth", True)
        sig = prog.signature_layout(cc.backend.state.masses())
        assert sig is not None and sig.rep_enum_ptr is not None
    calls = []
    monkeypatch.setattr(TR.tables.BoundaryGroups, "enumerated_lists", lambda self, code: (calls.append(1), lists(self, code))[1])
    first = prog.enum_lists()
    assert first[1] is prog.enum_lists()[1] and calls == [1]


def test_a_program_takes_one_pass_over_the_moves(build_pc, monkeypatch):
    """Its enumerated tokens' and its tables', without the lists: the most of what a long layout costs, once."""
    from pyjuice.constraints.inference.dfa_exact import backend as BE
    monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", 1)
    monkeypatch.setattr(BE, "ENUMERATE_MIN_FRACTION", 0.0)
    monkeypatch.setattr(H, "HOST_ENTRIES", 0)
    passes = []
    sweep = TR.tables.BoundaryGroups.sweep
    monkeypatch.setattr(TR.tables.BoundaryGroups, "sweep", lambda self, *a, **k: (passes.append(k.get("lists", True)), sweep(self, *a, **k))[1])
    prog = jc.compile(jc.DFA.contains([[1, 2]], 5), build_pc("hmm", 8, 5)).backend.program()
    assert prog.enum.on and passes == [False]


def test_the_ungrouped_tables_come_when_first_read(build_pc):
    """A whole pass over the moves, which a program that may never read them does not make; what it is to read is known before."""
    pc = build_pc("hmm", 8, 3)
    cc = jc.compile(jc.DFA.contains([[1, 2]], 3), pc)
    prog = cc.backend.program()
    assert not prog.trans_grouped and not cc.backend.grouped and prog._ungrouped is None
    assert prog.transition_buffer().numel() == 1 and prog._ungrouped is None             # (no masses to hold, for those tables)
    assert prog.trans.grouped is False and prog.trans is prog._ungrouped and not prog.trans_persistent
    grouped = prog.succ
    prog.trans_grouped = True
    assert prog.trans is grouped
