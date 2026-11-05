"""
The byte automata algebra (:mod:`pyjuice.constraints.language.backend.automata`): every operation against a
reference on EVERY byte string of up to 5 bytes over a small alphabet (``a``, ``b`` and the two bytes of ``é``, so
characters split and invalid UTF-8 among them) -- Python's ``re`` on the decoded text for the regular expressions and
what concatenation, union and repetition make of them, the operands' answers for intersection and difference -- for
fixed and random patterns; the canonical minimal form (equal tables iff equal languages); characters by code point
ranges at the UTF-8 length boundaries and around the surrogates; literals of any bytes; the edge cases (no text, the
empty text, no or zero repetitions); long chains staying linear.
"""
import itertools
import random
import re

import numpy as np
import pytest

from pyjuice.constraints.language.backend import automata as A   # noqa: E402

ALPHABET = [b"a", b"b", b"\xc3", b"\xa9"]
STRINGS = [b"".join(p) for n in range(6) for p in itertools.product(ALPHABET, repeat = n)]
PATTERNS = ["", "a", "ab|a", "(a|b)*", "a*b?", "é", ".é?", "[ab]{2,3}", "(ab|é)+", "a(b|é)*a|b", "[^a]", "(é|a){0,2}b*"]


def oracle(pattern, s):
    try:
        text = s.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return re.fullmatch(pattern, text, re.ASCII) is not None


def language(a):
    return np.array([a.accepts(s) for s in STRINGS])


def reference(pattern):
    return np.array([oracle(pattern, s) for s in STRINGS])


def random_pattern(rng, depth = 3):
    """A random pattern over a, b and é: literals, classes, groups, alternatives, repetitions."""
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(["a", "b", "é", "[ab]", ".", "[^b]"])
    kind = rng.choice(["cat", "alt", "star", "plus", "opt", "count"])
    x = random_pattern(rng, depth - 1)
    if kind == "cat":
        return f"(?:{x})(?:{random_pattern(rng, depth - 1)})"
    if kind == "alt":
        return f"(?:{x})|(?:{random_pattern(rng, depth - 1)})"
    if kind == "count":
        lo = rng.randint(0, 2)
        return f"(?:{x}){{{lo},{lo + rng.randint(0, 2)}}}"
    return f"(?:{x}){dict(star = '*', plus = '+', opt = '?')[kind]}"


def patterns():
    rng = random.Random(0)
    return PATTERNS + [random_pattern(rng) for _ in range(28)]


def test_a_pattern_accepts_what_re_matches():
    for p in patterns():
        assert (language(A.regex(p)) == reference(p)).all(), p


def test_concatenation_union_and_repetition_match_the_composed_patterns():
    ps = patterns()
    rng = random.Random(1)
    for p, q in [(rng.choice(ps), rng.choice(ps)) for _ in range(40)]:
        a, b = A.regex(p), A.regex(q)
        assert (language(A.concat(a, b)) == reference(f"(?:{p})(?:{q})")).all(), (p, q)
        assert (language(A.union(a, b)) == reference(f"(?:{p})|(?:{q})")).all(), (p, q)
        assert (language(A.star(a)) == reference(f"(?:{p})*")).all(), p
        assert (language(A.optional(a)) == reference(f"(?:{p})?")).all(), p
        lo = rng.randint(0, 2)
        for hi in (lo, lo + 1, lo + 2, None):
            want = reference(f"(?:{p}){{{lo},{'' if hi is None else hi}}}")
            assert (language(A.repeat(a, lo, hi)) == want).all(), (p, lo, hi)
    three = [A.regex(p) for p in ("a", "(b|é)*", "a?")]
    assert (language(A.concat(*three)) == reference("a(?:b|é)*a?")).all()
    assert (language(A.union(*three)) == reference("a|(?:b|é)*|a?")).all()


def test_intersection_and_difference_answer_as_their_operands_do():
    ps = patterns()
    rng = random.Random(2)
    for p, q in [(rng.choice(ps), rng.choice(ps)) for _ in range(40)]:
        a, b = A.regex(p), A.regex(q)
        la, lb = reference(p), reference(q)
        assert (language(A.intersect(a, b)) == (la & lb)).all(), (p, q)
        assert (language(A.difference(a, b)) == (la & ~lb)).all(), (p, q)
    a, b, c = (A.regex(p) for p in ("(a|b)*", ".*b.*", "[^é]*"))
    assert (language(A.intersect(a, b, c)) == (reference("(a|b)*") & reference(".*b.*") & reference("[^é]*"))).all()


def test_the_canonical_form_is_equal_exactly_for_equal_languages():
    ps = patterns()
    forms = {}
    for p in ps:
        m = A.regex(p)
        assert A.equivalent(m, A.minimize(m)) and A.equivalent(m, A.concat(m, A.epsilon()))
        assert np.array_equal(A.minimize(m).delta, m.delta)                     # results come minimal, canonical
        key = language(m).tobytes()
        if key in forms:                                            # another pattern of the same strings up to 5
            other = forms[key]
            if A.equivalent(m, other):                              # (bytes: may still differ on longer ones)
                assert np.array_equal(m.delta, other.delta) and m.initial == other.initial
        forms[key] = m
    rng = np.random.default_rng(4)                                  # the same automaton, its states renumbered
    for p in ps:
        m = A.regex(p)
        for _ in range(3):
            perm = rng.permutation(m.num_states)
            delta = np.empty_like(m.delta)
            delta[perm] = perm[m.delta]
            accept = np.empty_like(m.accept)
            accept[perm] = m.accept
            shuffled = A.minimize(A.ByteAutomaton(delta, int(perm[m.initial]), accept, int(perm[m.dead])))
            assert np.array_equal(shuffled.delta, m.delta) and shuffled.initial == m.initial, p
    a, b = A.regex("ab|é"), A.regex("a")
    assert np.array_equal(A.union(a, b).delta, A.union(b, a).delta)
    assert A.equivalent(A.union(a, b), A.regex("é|a|ab"))
    assert not A.equivalent(a, b)
    live = A.regex("(a|b)*é")
    assert live.dead == live.num_states - 1 and not live.accept[live.dead]
    assert (live.delta[live.dead] == live.dead).all()


def test_characters_by_code_point_ranges():
    ranges = [(0x41, 0x43), (0x7F, 0x80), (0x7FF, 0x800), (0xD7FF, 0xE000), (0xFFFF, 0x10000), (0x10FFFF, 0x10FFFF)]
    a = A.char_ranges(ranges)
    inside = [cp for lo, hi in ranges for cp in range(lo, hi + 1) if not 0xD800 <= cp <= 0xDFFF]
    for cp in inside:
        assert a.accepts(chr(cp).encode()), hex(cp)
    for cp in (0x40, 0x44, 0x7E, 0x81, 0x7FE, 0x801, 0xFFFE, 0x10001, 0x10FFFE):
        assert not a.accepts(chr(cp).encode()), hex(cp)
    for bad in (b"\xed\xa0\x80", b"\xc0\x80", b"\xc2", b"AB", b""):              # a surrogate, overlong, cut, two
        assert not a.accepts(bad), bad
    rng = random.Random(3)
    for _ in range(20):
        lo = rng.randint(0, 0x10FFFF)
        hi = min(0x10FFFF, lo + rng.randint(0, 5000))
        b = A.char_ranges([(lo, hi)])
        for cp in {lo, hi, (lo + hi) // 2, lo - 1, hi + 1}:
            if 0 <= cp <= 0x10FFFF and not 0xD800 <= cp <= 0xDFFF:
                assert b.accepts(chr(cp).encode()) == (lo <= cp <= hi), (lo, hi, cp)


def test_literals_bytes_and_edge_cases():
    for data in (b"", b"a", b"\xff\x00\xc3", "é".encode() * 3):
        a = A.literal(data)
        assert a.accepts(data) and not a.accepts(data + b"a") and not a.accepts(data[:-1] if data else b"x")
    assert A.byte_ranges([(0x30, 0x39), (0xF0, 0xF0)]).accepts(b"\xf0")
    assert not A.byte_ranges([(0x30, 0x39)]).accepts(b"a")
    assert A.is_empty(A.nothing()) and not A.is_empty(A.epsilon())
    assert A.is_empty(A.concat(A.literal(b"a"), A.nothing()))
    assert A.is_empty(A.union()) and A.equivalent(A.concat(), A.epsilon())
    assert A.equivalent(A.star(A.nothing()), A.epsilon()) and A.equivalent(A.star(A.epsilon()), A.epsilon())
    assert A.equivalent(A.repeat(A.literal(b"a"), 0, 0), A.epsilon())
    assert A.equivalent(A.repeat(A.nothing(), 0, 3), A.epsilon())
    assert A.is_empty(A.repeat(A.nothing(), 1, None))
    assert A.equivalent(A.difference(A.regex("a*"), A.regex("a*")), A.nothing())
    with pytest.raises(ValueError):
        A.repeat(A.literal(b"a"), 2, 1)
    with pytest.raises(ValueError):
        A.intersect()
    with pytest.raises(ValueError, match = "Unsupported regular expression"):
        A.regex("^a")


def random_tree(rng, depth = 3):
    """A random expression: ("pat", pattern) or (kind, children...) -- concatenation, union, repetition."""
    if depth == 0 or rng.random() < 0.25:
        return ("pat", rng.choice(PATTERNS))
    kind = rng.choice(["cat", "alt", "rep", "opt", "star"])
    if kind in ("cat", "alt"):
        return (kind,) + tuple(random_tree(rng, depth - 1) for _ in range(rng.randint(2, 3)))
    if kind == "rep":
        lo = rng.randint(0, 2)
        return ("rep", random_tree(rng, depth - 1), lo, rng.choice([lo, lo + 1, lo + 2, None]))
    return (kind, random_tree(rng, depth - 1))


def evaluate(tree, eager, shared):
    """The tree through the algebra: every operation forced at once (``eager``), or left to one evaluation; the
    first subtree of each node reused (``shared``) where the tree repeats it."""
    kind = tree[0]
    if kind == "pat":
        return A.regex(tree[1])
    done = (lambda x: A.force(x)) if eager else (lambda x: x)
    kids = [evaluate(t, eager, shared) for t in tree[1:] if isinstance(t, tuple)]
    if kind == "cat":
        out = A.concat(*kids, kids[0]) if shared else A.concat(*kids)
    elif kind == "alt":
        out = A.union(*kids, kids[0]) if shared else A.union(*kids)
    elif kind == "rep":
        out = A.repeat(kids[0], tree[2], tree[3])
    elif kind == "opt":
        out = A.optional(kids[0])
    else:
        out = A.star(kids[0])
    return done(out)


def test_an_expression_is_its_operations_carried_out_in_turn():
    """Concatenations, unions and repetitions nested, one expression evaluated at once: the same canonical automaton as
    every operation evaluated in turn -- subexpressions used in several places among them."""
    rng = random.Random(6)
    for _ in range(60):
        tree = random_tree(rng)
        for shared in (False, True):
            lazy, eager = A.force(evaluate(tree, False, shared)), evaluate(tree, True, shared)
            assert np.array_equal(lazy.table, eager.table) and np.array_equal(lazy.byte_class, eager.byte_class) \
                and lazy.initial == eager.initial and np.array_equal(lazy.accept, eager.accept), tree
    e = A.union(A.literal(b"a"), A.regex("b+"))
    composed = A.concat(e, A.literal(b"-"), A.star(e), A.repeat(e, 1, 2))
    assert (language(composed) == reference(r"(?:a|b+)-(?:a|b+)*(?:a|b+){1,2}")).all()


def test_a_long_chain_stays_linear():
    """Thousands of parts in one concatenation (an object's members): one state per byte, then the dead state."""
    parts = [A.literal(f'"k{i}":'.encode()) for i in range(3000)]
    a = A.concat(*parts)
    assert a.num_states == sum(len(f'"k{i}":') for i in range(3000)) + 2
    text = b"".join(f'"k{i}":'.encode() for i in range(3000))
    assert a.accepts(text) and not a.accepts(text[:-1])


def moore(table, accept):
    """The coarsest partition of the states, the textbook way: refined until nothing changes."""
    block = np.asarray(accept, dtype = np.int64)
    while True:
        rows = np.concatenate([block[:, None], block[table]], axis = 1)
        _, new = np.unique(rows, axis = 0, return_inverse = True)
        new = new.reshape(-1)
        if new.max() == block.max() and len(np.unique(new)) == len(np.unique(block)):
            return new
        block = new


def random_table(rng, K, C, density):
    """A complete table of ``K`` states over ``C`` classes, state ``K - 1`` a sink, a move there unless ``density`` says."""
    table = np.full((K, C), K - 1, dtype = np.int32)
    moves = rng.random((K, C)) < density
    table[moves] = rng.integers(0, K, moves.sum())
    table[K - 1] = K - 1
    accept = rng.random(K) < rng.choice([0.0, 0.1, 0.5, 1.0])
    accept[K - 1] = False
    return table, accept


def test_the_partition_over_moves_is_the_coarsest():
    """:func:`~pyjuice.constraints.language.backend.dfa.hopcroft` against refining until nothing changes: random tables
    sparse and dense, with sinks, without an accepting state, with nothing else."""
    from pyjuice.constraints.language.backend.dfa import hopcroft
    rng = np.random.default_rng(0)
    for case in range(400):
        K, C = int(rng.integers(1, 40)), int(rng.integers(1, 8))
        table, accept = random_table(rng, K, C, rng.choice([0.05, 0.2, 0.6, 1.0]))
        if case % 7 == 0:                                       # another state like the sink: no accepting one reached
            table[K // 2] = K // 2
            accept[K // 2] = False
        block, num = hopcroft(table.astype(np.int64), accept)
        reference = moore(table, accept)
        assert num == len(np.unique(reference)) == len(np.unique(block)), case
        assert block.min() >= 0 and block.max() < num, case
        pairs = np.unique(np.stack([block, reference]), axis = 1)
        assert pairs.shape[1] == num, case                      # (the same partition: one to one)


def test_the_minimal_automaton_over_moves_is_the_table_minimized():
    """:func:`~pyjuice.constraints.language.backend.automata._from_classes` on random tables: a state per block of the
    reachable states from which something is accepted, breadth-first by class, the one dead state last, and the same
    answers on every string."""
    rng = np.random.default_rng(1)
    for case in range(300):
        K, C = int(rng.integers(1, 30)), int(rng.integers(1, 6))
        table, accept = random_table(rng, K, C, rng.choice([0.1, 0.3, 0.8]))
        initial = int(rng.integers(0, K))
        byte_class = np.arange(256) % C
        byte_class[:C] = np.arange(C)
        a = A._from_classes(table, initial, accept, byte_class)
        assert a.minimal and a.initial == 0 and a.dead == a.num_states - 1, case
        assert not a.accept[a.dead] and (a.table[a.dead] == a.dead).all(), case
        again = A._from_classes(a.table[:, a.byte_class[np.arange(C)]], 0, a.accept, np.arange(256) % C * 0 + byte_class)
        assert np.array_equal(again.table, a.table) and np.array_equal(again.accept, a.accept), case
        seen = [0]                                              # numbered breadth-first, by class
        for q in seen:
            for t in a.table[q].tolist():
                if t != a.dead and t not in seen:
                    seen.append(t)
        assert seen == list(range(a.num_states - 1)) or a.num_states == 1, case
        for text in itertools.product(range(C), repeat = 4):
            q, r = initial, 0
            for c in text:
                q, r = int(table[q, c]), int(a.table[r, a.byte_class[c]])
                assert bool(accept[q]) == bool(a.accept[r]), case


# -------------------------------------------------------------------------------------------------
# The subset construction and the minimization, against plain ones
# -------------------------------------------------------------------------------------------------

def plain_minimal(table, accept, initial):
    """The canonical minimal automaton of a complete table, by the book: the reachable states, those from which something
    is accepted, Moore's refinement, then breadth-first numbers by class, the one dead state last: ``(table, accept)``."""
    K, C = len(table), len(table[0]) if table else 0
    reach, stack = {initial}, [initial]
    while stack:
        for t in table[stack.pop()]:
            if t not in reach:
                reach.add(t)
                stack.append(t)
    live = {q for q in reach if accept[q]}
    grew = True
    while grew:
        grew = False
        for q in reach - live:
            if any(t in live for t in table[q]):
                live.add(q)
                grew = True
    if initial not in live:
        return [[0] * C], [False]
    block = {q: int(accept[q]) for q in live}
    while True:
        seen = {}
        new = {q: seen.setdefault((block[q], tuple(block.get(t, -1) for t in table[q])), len(seen)) for q in sorted(live)}
        if len(seen) == len(set(block.values())):
            break
        block = new
    rep = {}
    for q in sorted(live):
        rep.setdefault(block[q], q)
    number = {block[initial]: 0}
    order = [block[initial]]
    for b in order:
        for t in table[rep[b]]:
            if t in live and block[t] not in number:
                number[block[t]] = len(order)
                order.append(block[t])
    m = len(order)
    out = [[number[block[t]] if t in live else m for t in table[rep[b]]] for b in order] + [[m] * C]
    return out, [bool(accept[rep[b]]) for b in order] + [False]


def plain_subsets(N, moves, eps, accept, start, C):
    """The subset construction, by the book, of ``moves`` (``(state, class, successor)``) and empty moves: a complete table
    over sets of states, the empty one among them: ``(table, accepting)``, set 0 the initial one."""
    def closed(states):
        out, stack = set(states), list(states)
        while stack:
            u = stack.pop()
            for a, b in eps:
                if a == u and b not in out:
                    out.add(b)
                    stack.append(b)
        return frozenset(out)
    first = closed([start])
    number, order, table = {first: 0}, [first], []
    for s in order:
        row = []
        for c in range(C):
            t = closed([d for u, k, d in moves if u in s and k == c])
            if t not in number:
                number[t] = len(order)
                order.append(t)
            row.append(number[t])
        table.append(row)
    return table, [any(accept[q] for q in s) for s in order]


def random_moves(rng, N, C, density, one_a_class):
    """Moves of a random automaton: as parts' states have them (one on a class at the most, classes ascending), or any."""
    moves = []
    for u in range(N):
        classes = rng.sample(range(C), rng.randint(0, C)) if rng.random() < density else []
        for c in (sorted(classes) if one_a_class else classes):
            moves.append((u, c, rng.randrange(N)))
            if not one_a_class and rng.random() < 0.3:
                moves.append((u, rng.randrange(C), rng.randrange(N)))
    return moves


@pytest.mark.parametrize("one_a_class", [True, False])
@pytest.mark.parametrize("eps_share", [0.0, 0.1, 0.5])
def test_the_subset_construction_is_the_plain_one(one_a_class, eps_share):
    """Random automata, states moving on their own and in sets, found ahead of the sweep and behind it, however numbered:
    the minimal automaton of the sets is the one of the plain construction, table for table."""
    rng = random.Random(int(eps_share * 10) + one_a_class)
    for case in range(300):
        N, C = rng.randint(1, 14), rng.randint(1, 4)
        moves = random_moves(rng, N, C, rng.choice([0.5, 0.9, 1.0]), one_a_class)
        eps = [(u, rng.randrange(N)) for u in range(N) if rng.random() < eps_share]
        eps += [(rng.randrange(N), rng.randrange(N)) for _ in range(rng.randint(0, 2) if eps_share else 0)]
        accept = [rng.random() < 0.3 for _ in range(N)]
        start = rng.randrange(N)
        by_state = sorted(range(len(moves)), key = lambda i: moves[i][0])              # (stably: a state's moves as they came)
        edge_ptr = np.zeros(N + 1, dtype = np.int64)
        for u, _, _ in moves:
            edge_ptr[u + 1] += 1
        edge_ptr = np.cumsum(edge_ptr)
        edge_class = np.array([moves[i][1] for i in by_state], dtype = np.int32)
        edge_dst = np.array([moves[i][2] for i in by_state], dtype = np.int32)
        eps_ptr = np.zeros(N + 1, dtype = np.int64)
        for u, _ in eps:
            eps_ptr[u + 1] += 1
        eps_ptr = np.cumsum(eps_ptr)
        eps_dst = np.array([v for _, v in sorted(eps, key = lambda p: p[0])], dtype = np.int32)
        ptr, cls, dst, acc, initial, *_ = A._subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps_dst, np.array(accept, dtype = np.bool_), start, C)
        assert ptr.size - 1 == acc.size and ptr[-1] == dst.size == cls.size
        got = A._from_moves(ptr, cls, dst, initial, acc, C, np.arange(256) % C)
        table, accepting = plain_subsets(N, moves, eps, accept, start, C)
        want, want_accept = plain_minimal(table, accepting, 0)
        table_got = got.table[:, got.byte_class[np.arange(C)]]
        assert table_got.tolist() == want and got.accept.tolist() == want_accept, (case, N, C, moves, eps, accept, start)


def test_a_state_found_behind_the_sweep_through_a_set_is_read():
    """Moves of a set of two states, alone on a class once they are one: the state they both lead to stands behind the
    sweep, and what it moves to is not a set on its own -- unless it is read, its row leads to a state without its empty moves."""
    # 5 -a-> 4 (empty: 3); 4 and 3 -a-> 1; 1 -b-> 2 (empty: 0); 0 accepts
    moves = [(1, 1, 2), (3, 0, 1), (4, 0, 1), (5, 0, 4)]
    eps = [(2, 0), (4, 3)]
    N, C, start = 6, 2, 5
    accept = [True, False, False, False, False, False]
    edge_ptr = np.cumsum([0] + [sum(m[0] == u for m in moves) for u in range(N)]).astype(np.int64)
    edge_class = np.array([m[1] for m in moves], dtype = np.int32)
    edge_dst = np.array([m[2] for m in moves], dtype = np.int32)
    eps_ptr = np.cumsum([0] + [sum(e[0] == u for e in eps) for u in range(N)]).astype(np.int64)
    eps_dst = np.array([e[1] for e in eps], dtype = np.int32)
    ptr, cls, dst, acc, initial, *_ = A._subsets(edge_ptr, edge_class, edge_dst, eps_ptr, eps_dst, np.array(accept, dtype = np.bool_), start, C)
    got = A._from_moves(ptr, cls, dst, initial, acc, C, np.arange(256) % C)
    want, want_accept = plain_minimal(*plain_subsets(N, moves, eps, accept, start, C), 0)
    assert got.table[:, got.byte_class[np.arange(C)]].tolist() == want and got.accept.tolist() == want_accept
    assert any(want_accept)


@pytest.mark.parametrize("dead_share", [0.0, 0.3, 0.8])
def test_the_minimal_table_is_the_plain_one(dead_share):
    """Random tables, states out of reach and states from which nothing is accepted among them: Hopcroft's algorithm from the
    distances to acceptance gives the partition Moore's does from accepting or not, in the same numbers."""
    rng = random.Random(int(dead_share * 10))
    for case in range(300):
        K, C = rng.randint(1, 30), rng.randint(1, 5)
        table = [[rng.randrange(K) if rng.random() >= dead_share else q for _ in range(C)] for q in range(K)]
        accept = [rng.random() < 0.25 for _ in range(K)]
        got = A._from_classes(np.array(table, dtype = np.int32), 0, np.array(accept, dtype = np.bool_), np.arange(256) % C)
        want, want_accept = plain_minimal(table, accept, 0)
        assert got.table[:, got.byte_class[np.arange(C)]].tolist() == want and got.accept.tolist() == want_accept, (case, table, accept)
