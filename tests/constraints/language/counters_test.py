"""
Bounded repetition done once (:mod:`pyjuice.constraints.language.backend.counters`): a schema's token automaton through a short byte
automaton, the copies of a chain put back in its moves, against the whole byte automaton's, entry for entry: strings and arrays with
bounds, mandatory copies and optional ones, nested, in unbounded repetitions, under tokens that span many copies. Where it cannot
be done, the long way, the same.
"""
import copy
import pickle
import random

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.compiled import layout
from pyjuice.constraints.language.backend import automata as A, counters, token_lift
from pyjuice.constraints.language.backend.json_schema import compile_schema


@pytest.fixture(autouse = True, params = ["behaviours", "walks"])
def few_copies_left_out(monkeypatch, request):
    """(the rows and classes by behaviours or by walks: the copies are put back in either's)"""
    monkeypatch.setattr(counters, "MIN_LEFT_OUT", 1)
    if request.param == "walks":
        monkeypatch.setattr(token_lift, "BEHAVIOUR_ENTRIES", 0)


def vocabulary_of(rng, longest = 12, count = 300, pieces = ()):
    """Every byte, and tokens of the bytes of schemas' texts, some of them whole pieces of those texts many times over."""
    alphabet = b'{}[],:" ab\\0123-'
    tokens = [bytes([b]) for b in range(256)]
    for _ in range(count):
        tokens.append(bytes(rng.choice(alphabet) for _ in range(rng.randint(2, longest))))
    for piece in pieces:
        for times in (1, 2, 3, 5):
            tokens.append(piece * times)
    return jc.Vocabulary(list(dict.fromkeys(tokens)) + [None], eos = [len(set(tokens))])


def same(short, long, where):
    assert short.num_states == long.num_states, (where, short.num_states, long.num_states)
    assert short.initial == long.initial and torch.equal(short.delta, long.delta), where
    assert torch.equal(short.accept, long.accept) and torch.equal(short.token_class, long.token_class), where


def check(schema, vocabulary, eos = True):
    """The short way and the long way, the same; whether anything was left out."""
    short = jc.JSONSchema(schema, vocabulary, eos_padding = eos)
    cut = []
    real = token_lift.to_tokens

    def spied(a, *rest):
        cut.append(bool(a.counted))
        return real(a, *rest)
    token_lift.to_tokens = spied
    try:
        got = short._short_way()
    finally:
        token_lift.to_tokens = real
    long = jc.JSONSchema(schema, vocabulary, eos_padding = eos)
    want = real(long._byte_automaton(), vocabulary, eos)
    same(short.automaton() if got is None else got, want, schema)       # (none: the long way, through `automaton`)
    return got is not None and bool(cut and cut[0])


STRING = {"type": "string", "maxLength": 70}
LONG_TOKENS = (b'aaaa', b'"a",', b'{},', b',{}', b'"},{"a":"', b'[],')


@pytest.mark.parametrize("eos", [True, False])
@pytest.mark.parametrize("schema, cut", [
    (STRING, True),
    ({"type": "string", "minLength": 60, "maxLength": 120}, True),
    ({"type": "string", "minLength": 60}, True),                                        # (mandatory copies, then a loop)
    ({"type": "string", "minLength": 60, "maxLength": 130}, True),                      # (both stretches)
    ({"type": "string", "minLength": 60, "maxLength": 60}, True),
    ({"type": "array", "items": STRING, "maxItems": 40}, True),                         # (nested)
    ({"type": "array", "items": STRING}, True),                                         # (in an unbounded loop)
    ({"type": "array", "items": {"type": "integer"}, "minItems": 60, "maxItems": 130}, False),  # (a copy's entry loops: the long way)
    ({"type": "array", "items": {"type": "boolean"}, "maxItems": 90}, True),
    ({"type": "array", "items": {"enum": ["a", "b", 1]}, "minItems": 40}, True),
    ({"type": "object", "properties": {"a": STRING, "b": STRING}, "required": ["a"], "additionalProperties": False}, True),
    ({"type": "object", "properties": {"a": STRING}, "additionalProperties": False}, True),
    ({"type": "array", "items": {"type": "object", "properties": {"a": STRING, "b": {"type": "string", "maxLength": 50}}, "additionalProperties": False},
      "maxItems": 30}, True),                                                           # (the corpus's largest, smaller)
    ({"type": "object", "additionalProperties": STRING}, True),                         # (in a loop of members)
    ({"anyOf": [STRING, {"type": "array", "items": STRING, "maxItems": 40}]}, True),
    ({"anyOf": [{"type": "string", "maxLength": 70}, {"type": "string", "maxLength": 65}]}, None),      # (chains in one set of states)
    ({"type": "string", "maxLength": 5}, False),                                        # (nothing to leave out)
])
def test_the_short_way_is_the_long_way(schema, cut, eos):
    rng = random.Random(0)
    vocabulary = vocabulary_of(rng, pieces = LONG_TOKENS)
    got = check(schema, vocabulary, eos)
    if cut is not None:
        assert got == cut


def random_schema(rng, depth = 0):
    kind = rng.choice(["string", "string", "array", "object", "integer", "enum", "boolean"] if depth < 2 else ["string", "integer", "enum"])
    if kind == "string":
        hi = rng.choice([3, 30, 45, 64, 90])
        lo = rng.choice([0, 0, 1, hi // 2, hi])
        return {"type": "string", "minLength": lo, "maxLength": hi}
    if kind == "array":
        hi = rng.choice([2, 20, 33, 48])
        lo = rng.choice([0, 0, 1, hi // 2, hi])
        out = {"type": "array", "items": random_schema(rng, depth + 1), "minItems": lo}
        if rng.random() < 0.8:
            out["maxItems"] = hi
        return out
    if kind == "object":
        names = rng.sample(["a", "b", "cc", "ddd"], rng.randint(1, 3))
        return {"type": "object", "properties": {n: random_schema(rng, depth + 1) for n in names}, "required": rng.sample(names, rng.randint(0, len(names))),
                "additionalProperties": False}
    if kind == "enum":
        return {"enum": rng.sample(["a", "bb", 1, 22, True, None], rng.randint(1, 4))}
    return {"type": kind}


def size_of(schema):
    """About how many states the whole automaton has: the bound the test stays under."""
    if schema.get("type") == "string":
        return 12 * schema["maxLength"] + 2
    if schema.get("type") == "array":
        return (schema.get("maxItems") or 2) * size_of(schema["items"]) + 4
    if schema.get("type") == "object":
        return sum(size_of(s) for s in schema["properties"].values()) + 8
    return 12


@pytest.mark.parametrize("seed", range(6))
def test_random_schemas_the_same_either_way(seed):
    rng = random.Random(seed)
    vocabulary = vocabulary_of(rng, longest = rng.choice([4, 12, 24]), pieces = LONG_TOKENS)
    cut, done = 0, 0
    while done < 12:
        schema = random_schema(rng)
        if size_of(schema) > 60000:
            continue
        done += 1
        cut += check(schema, vocabulary, bool(done % 2))
    assert cut >= 2


def test_tokens_spanning_many_copies_are_counted():
    """Tokens that pass several items whole: the short chain keeps as many as they span."""
    item = {"type": "object", "properties": {"a": {"type": "boolean"}}, "additionalProperties": False}
    schema = {"type": "array", "items": item, "maxItems": 200}
    rng = random.Random(1)
    few = jc.Vocabulary([bytes([b]) for b in range(256)] + [None], eos = [256])
    many = vocabulary_of(rng, count = 10, pieces = (b"{},", b",{}", b"{}, ", b"{},{},{},{},{},{},{},{},{},{},"))
    assert check(schema, few)
    assert check(schema, many)
    part = A.literal(b",{}")
    assert counters._span(part, counters.Reach(few, np.frombuffer(few._buffer, dtype = np.uint8))) == 1
    assert counters._span(part, counters.Reach(many, np.frombuffer(many._buffer, dtype = np.uint8))) >= 10


def test_what_a_short_automaton_cannot_tell_goes_the_long_way():
    """A product reads the whole table: the short automaton raises, and the schema goes the long way, the same automaton."""
    vocabulary = vocabulary_of(random.Random(2))
    hard = {"allOf": [{"type": "string", "maxLength": 70}, {"not": {"type": "string", "maxLength": 3}}]}
    short = jc.JSONSchema(hard, vocabulary)
    long = jc.JSONSchema(hard, vocabulary)
    want = token_lift.to_tokens(long._byte_automaton(), vocabulary, True)
    same(short.automaton(), want, hard)
    short = jc.JSONSchema(STRING, vocabulary)
    counting = counters.COUNTING.set(counters.Reach(vocabulary, token_lift._trie(vocabulary).data))
    try:
        a = compile_schema(short._prepared, short._whitespace)
    finally:
        counters.COUNTING.reset(counting)
    assert a.counted
    with pytest.raises(counters.NotCounted):
        a.table
    with pytest.raises(counters.NotCounted):
        a.delta
    with pytest.raises(counters.NotCounted):
        a.accepts(b'""')


def test_a_vocabulary_without_every_byte_goes_the_long_way():
    vocabulary = jc.Vocabulary([b"a", b"b", b'"', b"ab", None], eos = [4])
    schema = {"type": "string", "maxLength": 70}
    short = jc.JSONSchema(schema, vocabulary)
    assert short._short_way() is None
    long = jc.JSONSchema(schema, vocabulary)
    same(short.automaton(), token_lift.to_tokens(long._byte_automaton(), vocabulary, True), schema)


def test_the_byte_automaton_is_the_whole_one_after_the_short_way():
    vocabulary = vocabulary_of(random.Random(3))
    k = jc.JSONSchema(STRING, vocabulary)
    assert k.automaton().num_states > 100
    a = k._byte_automaton()
    assert not a.counted and a.accepts(b'"' + b"a" * 70 + b'"') and not a.accepts(b'"' + b"a" * 71 + b'"')


def texts_of(a, rng, count, most = 240):
    """Texts ``a`` accepts or goes through, at random: every move out of a state alike."""
    table, dead = a.table, a.dead
    byte_of = {}
    for b in range(256):
        byte_of.setdefault(int(a.byte_class[b]), []).append(b)
    texts = []
    for _ in range(count):
        q, text = a.initial, bytearray()
        while len(text) < most and not (a.accept[q] and rng.random() < 0.3):
            live = [c for c in range(table.shape[1]) if table[q, c] != dead]
            if not live:
                break
            c = rng.choice(live)
            text.append(rng.choice(byte_of[c]))
            q = int(table[q, c])
        texts.append(bytes(text))
    return texts


def vocabulary_of_pieces(a, rng, longest = 24, count = 400):
    """Every byte, and pieces of texts ``a`` goes through: tokens that close what they are in, open another and go on in it."""
    tokens = [bytes([b]) for b in range(256)]
    texts = [t for t in texts_of(a, rng, 60) if len(t) >= 2]
    for _ in range(count if texts else 0):
        t = rng.choice(texts)
        at = rng.randrange(len(t) - 1)
        tokens.append(t[at:at + rng.randint(2, longest)])
    tokens = list(dict.fromkeys(tokens))
    return jc.Vocabulary(tokens + [None], eos = [len(tokens)])


def bytes_of(schema):
    """The whole byte automaton of a schema, whatever the vocabulary."""
    return jc.JSONSchema(schema, jc.Vocabulary([bytes([b]) for b in range(256)] + [None], eos = [256]))._byte_automaton()


SHORT = {"type": "string", "maxLength": 45}
TWO = {"type": "object", "properties": {"a": SHORT, "b": {"type": "string", "maxLength": 50}}, "additionalProperties": False}


@pytest.mark.parametrize("schema", [
    {"type": "array", "items": SHORT},                                                  # (back into the chain, some copies in)
    {"type": "array", "items": SHORT, "maxItems": 30},                                  # (on into the next item's chain)
    {"type": "array", "items": SHORT, "minItems": 25, "maxItems": 60},
    TWO,                                                                                # (from one chain into another)
    {"type": "array", "items": TWO},
    {"type": "array", "items": TWO, "maxItems": 25},
    {"type": "object", "additionalProperties": SHORT},
    {"anyOf": [SHORT, {"type": "array", "items": SHORT}]},
    {"type": "array", "items": {"type": "array", "items": SHORT}},
    {"type": "array", "items": {"enum": [1, 12, 123]}, "maxItems": 60},                 # (a copy's accepting states lead on)
])
@pytest.mark.parametrize("seed", range(3))
def test_tokens_that_leave_a_chain_and_come_back(schema, seed):
    """A token that closes a string and goes on in the next: from every copy it leads to the same state, not as far on from each."""
    rng = random.Random(seed)
    vocabulary = vocabulary_of_pieces(bytes_of(schema), rng, longest = rng.choice([8, 24]))
    assert check(schema, vocabulary, bool(seed % 2))


def test_a_token_back_into_the_chain_it_left():
    """The very moves: from the template and from a copy past it, a token closing the string and writing three characters of the next."""
    schema = {"type": "array", "items": SHORT}
    tokens = [bytes([b]) for b in range(256)] + [b'","aaa', b'a","aa', b'aaaa']
    vocabulary = jc.Vocabulary(tokens + [None], eos = [len(tokens)])
    assert check(schema, vocabulary)


def random_expression(rng, depth = 0):
    """A plan of an expression: pieces of it wired the way a schema's are."""
    kind = rng.choice(["rep", "rep", "seq", "seq", "alt", "opt", "lit"] if depth < 3 else ["lit", "lit", "class"])
    if kind == "lit":
        return ("lit", rng.choice([b"a", b"b", b"ab", b"abc", b"ba", b","]))
    if kind == "class":
        return ("class", [(48, 57)])                                    # (none of the letters: what makes sets of states explode)
    if kind == "opt":
        return ("opt", random_expression(rng, depth + 1))
    if kind == "rep":
        hi = rng.choice([None, 2, 40, 60, 90])
        lo = rng.choice([0, 0, 1, 2, 60]) if hi is None else rng.choice([0, 0, 1, hi // 2, hi])
        return ("rep", random_expression(rng, depth + 1), lo, hi)
    return (kind, [random_expression(rng, depth + 1) for _ in range(rng.randint(2, 3))])


def built(plan):
    kind = plan[0]
    if kind == "lit":
        return A.literal(plan[1])
    if kind == "class":
        return A.byte_ranges(plan[1])
    if kind == "opt":
        return A.optional(built(plan[1]))
    if kind == "rep":
        return A.repeat(built(plan[1]), plan[2], plan[3])
    return (A.concat if kind == "seq" else A.union)(*[built(p) for p in plan[1]])


def capped(plan, most):
    """The plan with no bound above ``most``."""
    kind = plan[0]
    if kind == "rep":
        hi = None if plan[3] is None else min(plan[3], most)
        return ("rep", capped(plan[1], most), min(plan[2], most), hi)
    if kind == "opt":
        return ("opt", capped(plan[1], most))
    return (kind, [capped(p, most) for p in plan[1]]) if kind in ("seq", "alt") else plan


def stays_small(plan):
    """Whether its automaton grows with its bounds as the moves do, and not as sets of its states explode."""
    return all(A.force(built(capped(plan, most))).num_states <= 40 * most for most in (2, 3, 5, 10))


def plan_size(plan):
    kind = plan[0]
    if kind in ("lit", "class"):
        return 4
    if kind == "opt":
        return plan_size(plan[1]) + 2
    if kind == "rep":
        return plan_size(plan[1]) * max(plan[3] or plan[2], 1) + 2
    return sum(plan_size(p) for p in plan[1]) + 2


@pytest.mark.parametrize("seed", range(8))
def test_random_expressions_the_same_either_way(seed):
    """Past what a schema writes: chains that can end anywhere, that can be empty, one after another, in unions, in loops."""
    rng = random.Random(100 + seed)
    done = cut = 0
    while done < 30:
        plan = ("seq", [random_expression(rng), ("lit", b"$")])                 # (no closer: the last copy differs, and the long way, every time)
        if plan_size(plan) > 4000 or not stays_small(plan):
            continue
        long = A.force(built(plan))
        done += 1
        vocabulary = vocabulary_of_pieces(long, rng, longest = rng.choice([3, 6, 12]), count = 120)
        eos = bool(done % 2)
        want = token_lift.to_tokens(long, vocabulary, eos)
        counting = counters.COUNTING.set(counters.Reach(vocabulary, token_lift._trie(vocabulary).data))
        try:
            short = A.force(built(plan))
            cut_here = bool(short.counted)
            got = token_lift.to_tokens(short, vocabulary, eos)
        except counters.NotCounted:
            continue
        finally:
            counters.COUNTING.reset(counting)
        same(got, want, plan)
        cut += cut_here
    assert cut >= 1


def test_how_far_a_token_goes():
    a = A.literal(b"ab")
    few = jc.Vocabulary([bytes([b]) for b in range(256)] + [b"abababab", b"xabab", None], eos = [258])
    reach = counters.Reach(few, np.frombuffer(few._buffer, dtype = np.uint8))
    assert counters._span(a, reach) == 5                                    # (the first copy it leaves, from its middle, and four whole)
    assert counters._span(A.force(A.optional(A.literal(b"a"))), reach) == -1          # (a copy can be empty)
    assert counters.kept_copies(A.force(A.optional(A.literal(b"a"))), 0, 100) is None
    carries_on = A.force(A.union(A.literal(b"a"), A.literal(b"ab")))        # (an accepting state that leads on: only the length tells)
    assert counters._span(carries_on, reach) == (8 - 1) // 1 + 1
    counting = counters.COUNTING.set(reach)
    try:
        assert counters.kept_copies(A.force(A.optional(A.literal(b"a"))), 0, 100) is None
        lo, copies, wired = counters.kept_copies(a, 90, 100)                # (the copies before the text can end, cut; those after, too few)
        assert (lo, copies) == (5 + counters.KEPT_AROUND + 1, 5 + counters.KEPT_AROUND + 11)
        assert wired.regions == [(1, 7, 89 - 5 - counters.KEPT_AROUND)]
        assert counters.kept_copies(a, 3, 3) is None
    finally:
        counters.COUNTING.reset(counting)


def test_a_token_longer_than_a_kilobyte():
    a = A.literal(b"a")
    many = jc.Vocabulary([bytes([b]) for b in range(256)] + [b"a" * 3000, None], eos = [257])
    assert counters._span(a, counters.Reach(many, np.frombuffer(many._buffer, dtype = np.uint8))) == 3001


@pytest.mark.parametrize("first", [True, False])
def test_two_moves_to_one_state_from_the_template_and_not_from_a_copy(first):
    """From the template one token walks on two copies, and another closes the string, opens the next and writes as far: one state.
    From a copy further on they lead apart, and the order of the states has to see both, whichever class comes first."""
    schema = {"type": "array", "items": SHORT}
    pair = [b'","aaa', b"aa"] if first else [b"aa", b'","aaa']
    tokens = [bytes([b]) for b in range(256)]
    tokens = pair + [t for t in tokens if t != b"a"] + [b"a"]           # (these classes before those of the bytes)
    vocabulary = jc.Vocabulary(tokens + [None], eos = [len(tokens)])
    assert check(schema, vocabulary)


def test_two_moves_of_a_copy_in_a_copy_that_lead_apart_only_at_the_end():
    """A chain in a chain in a loop. From a string's copy put back, in an item past the outer template, one token goes on to the next
    item and another out of the array and in again as far: one state until the outer copies are put back, and two after. A long
    token gets to such a copy first of its item, and so it is that the order of the states has to see both from."""
    schema = {"type": "array", "items": {"type": "array", "items": {"type": "string", "maxLength": 70}, "maxItems": 60}}
    for empties in (3, 4, 5, 6):
        out_and_in = b'"],["' + b'","' * empties + b'aaa'
        tokens = [out_and_in, b'","aaa', b"a" * 20] + [bytes([b]) for b in range(256)]
        vocabulary = jc.Vocabulary(tokens + [None], eos = [len(tokens)])
        assert check(schema, vocabulary), empties


class Chain:
    """
    What :func:`counters._window` reads of a chain, as moves that can be spoiled one at a time: ``copies`` copies of an entry ``e(c)``,
    a state ``m(c)`` and a set ``x(c)`` of ``m(c)`` and a state not in the chain, then a state ``END`` and another not in the chain.
    """
    copies, size, t, G = 8, 2, 1, 3

    def __init__(self):
        c_ = range(self.copies)
        self.N = self.copies * self.size + 2
        self.END, self.OTHER = self.N - 2, self.N - 1
        self.sets = [[self.m(c), self.END] for c in c_]
        self.moves = {q: [] for q in range(self.N + self.copies)}
        for c in c_:
            on = self.e(c + 1) if c + 1 < self.copies else self.END
            self.moves[self.e(c)] = [(0, self.m(c)), (1, self.x(c)), (2, self.END)]
            self.moves[self.m(c)] = [(0, on)]
            self.moves[self.x(c)] = [(0, on), (1, self.END)]
        self.accepting = np.zeros(self.N + self.copies, dtype = np.bool_)
        self.accepting[self.END] = True
        self.read = np.ones(self.N, dtype = np.bool_)
        self.number = np.arange(self.N + self.copies, dtype = np.int32)

    def e(self, c):
        return 2 * c

    def m(self, c):
        return 2 * c + 1

    def x(self, c):
        return self.N + c

    def another_set(self, states):
        """A set more, not moving: its number."""
        self.sets.append(states)
        at = self.N + len(self.sets) - 1
        self.moves[at] = []
        self.accepting = np.append(self.accepting, False)
        self.number = np.append(self.number, at)
        return at

    def window(self):
        K = self.N + len(self.sets)
        ptr = np.zeros(K + 1, dtype = np.int64)
        for q in range(K):
            ptr[q + 1] = ptr[q] + len(self.moves[q])
        cls = np.asarray([c for q in range(K) for c, _ in self.moves[q]], dtype = np.int32)
        dst = np.asarray([d for q in range(K) for _, d in self.moves[q]], dtype = np.int32)
        pool = np.asarray([x for s in self.sets for x in s], dtype = np.int32)
        at = np.cumsum([0] + [len(s) for s in self.sets], dtype = np.int64)
        known = self.number[self.number >= 0]
        K_final = int(known.max()) + 1
        total = np.bincount(known, minlength = K_final + 1).astype(np.int64)
        return counters._window(self.N, 0, self.size, self.copies, self.t, self.G, ptr, cls, dst, self.accepting, self.read, pool, at, len(self.sets), self.number,
                                total, K_final)


def test_a_window_of_alike_copies():
    chain = Chain()
    table = chain.window()
    assert table.tolist() == [[chain.e(c), chain.m(c), chain.x(c)] for c in (1, 2, 3)]
    chain = Chain()
    for c in range(chain.copies):                                           # (two states one in every copy: one column)
        chain.number[chain.x(c)] = chain.number[chain.m(c)]
    assert chain.window().tolist() == [[chain.e(c), chain.m(c)] for c in (1, 2, 3)]
    chain = Chain()
    for c in range(chain.copies):                                           # (a set of no number in every copy: none)
        chain.number[chain.x(c)] = -1
    assert chain.window().tolist() == [[chain.e(c), chain.m(c)] for c in (1, 2, 3)]


def spoil_read(c):
    c.read[c.m(2)] = False


def spoil_accept(c):
    c.accepting[c.m(2)] = True


def spoil_row(c):
    c.moves[c.e(2)] = c.moves[c.e(2)][:-1]


def spoil_row_with_the_next_row_alike(c):
    """A row a move short, the next row's first move what it lacks."""
    for k in range(c.copies):
        c.moves[c.m(k)] = [(2, c.END)] + c.moves[c.m(k)]
    c.moves[c.e(2)] = c.moves[c.e(2)][:-1]


def spoil_class(c):
    c.moves[c.m(2)] = [(1, c.moves[c.m(2)][0][1])]


def spoil_target(c):
    c.moves[c.m(2)] = [(0, c.m(3))]


def spoil_too_far(c):
    for k in range(c.copies - 2):
        c.moves[c.m(k)] = [(0, c.e(k + 2))]


def spoil_backwards(c):
    for k in range(1, c.copies):
        c.moves[c.m(k)] = c.moves[c.m(k)] + [(3, c.e(k - 1))]


def spoil_outside(c):
    c.moves[c.e(2)] = c.moves[c.e(2)][:2] + [(2, c.OTHER)]


def spoil_into_the_chain(c):
    for k in range(c.copies):
        c.moves[c.m(k)] = c.moves[c.m(k)] + [(3, c.e(7))]


def spoil_entry_loops(c):
    for k in range(c.copies):
        c.moves[c.m(k)] = c.moves[c.m(k)] + [(3, c.e(k))]


def spoil_entry_not_alone(c):
    for k in range(c.copies):
        c.number[c.e(k)] = c.number[c.m(k)]


def spoil_entry_no_number(c):
    for k in range(c.copies):
        c.number[c.e(k)] = -1


def spoil_told_apart_later(c):
    c.number[c.m(2)] = c.number[c.e(2)]


def spoil_with_another_copy(c):
    c.number[c.x(2)] = c.number[c.x(1)]


def spoil_with_a_state_elsewhere(c):
    c.number[c.m(2)] = c.number[c.OTHER]


def spoil_with_the_last_copy(c):
    c.number[c.m(2)] = c.number[c.m(7)]


def spoil_one_later(c):
    """Two states told apart in the template and one further on: all counts hold."""
    c.number[c.x(2)] = c.number[c.m(2)]


def spoil_apart_later(c):
    """Two states one in the template and told apart further on."""
    for k in range(c.copies):
        if k != 2:
            c.number[c.x(k)] = c.number[c.m(k)]


def spoil_number_later(c):
    c.number[c.x(1)] = -1


def spoil_no_number_later(c):
    c.number[c.x(2)] = -1


def spoil_a_set_more_later(c):
    """A set in a copy past the template that the template has none like: it stands where no copy put back says."""
    c.another_set([c.e(3), c.OTHER])


def spoil_set(c):
    c.sets[2] = [c.m(2), c.OTHER]


def spoil_set_size(c):
    """The last set compared holds what the one before does, and one more."""
    c.sets[4] = [c.m(4), c.OTHER, c.END]


@pytest.mark.parametrize("spoil", [spoil_read, spoil_accept, spoil_row, spoil_row_with_the_next_row_alike, spoil_class, spoil_target, spoil_too_far, spoil_backwards,
                                   spoil_outside, spoil_into_the_chain, spoil_entry_loops, spoil_entry_not_alone, spoil_entry_no_number, spoil_told_apart_later,
                                   spoil_with_another_copy, spoil_with_a_state_elsewhere, spoil_with_the_last_copy, spoil_one_later, spoil_apart_later,
                                   spoil_number_later, spoil_no_number_later, spoil_a_set_more_later, spoil_set, spoil_set_size])
def test_a_window_not_alike(spoil):
    """Every one of the checks: a chain that does not hold as this one says is not left short."""
    chain = Chain()
    spoil(chain)
    assert chain.window().size == 0


def test_a_run_in_an_automaton_around_it():
    run = counters.Run(np.asarray([[0, 1], [2, 3]], dtype = np.int32), 3, [[], []])
    number = np.asarray([5, 6, 7, 8, 9, 10], dtype = np.int32)
    total = np.ones(11, dtype = np.int64)
    none = np.zeros(6, dtype = np.bool_)
    assert run.at(2, number, none, total).table.tolist() == [[7, 8], [9, 10]]
    gone = number.copy()
    gone[3] = -1
    with pytest.raises(counters.NotCounted):                                # (a state no set is)
        run.at(2, gone, none, total)
    held = none.copy()
    held[4] = True
    with pytest.raises(counters.NotCounted):                                # (a set holds a state past the template)
        run.at(2, number, held, total)
    held = none.copy()
    held[3] = True
    assert run.at(2, number, held, total).table.tolist() == [[7, 8], [9, 10]]           # (only of the template)
    many = total.copy()
    many[7] = 2
    with pytest.raises(counters.NotCounted):                                # (a state one with another, of the template)
        run.at(2, number, none, many)
    many = total.copy()
    many[10] = 2
    with pytest.raises(counters.NotCounted):                                # (past it)
        run.at(2, number, none, many)


def test_a_set_of_a_copy_past_the_template_and_something_else():
    """What runs along the first copies of a part with a chain of its own: states of those copies, in sets, are at the start of the
    chain, wherever the copies put back are: the long way."""
    digits, letters = [(48, 57)], [(97, 98)]
    any_ = [(97, 99), (48, 57)]

    def plan():
        along = A.optional(A.concat(A.byte_ranges(letters), A.literal(b"b"), A.byte_ranges(digits)))
        return A.concat(A.union(A.repeat(A.repeat(A.byte_ranges(any_), 60, 60), 0, None), along), A.literal(b"a"))
    long = A.force(plan())
    for seed in range(6):
        vocabulary = vocabulary_of_pieces(long, random.Random(seed), longest = 3, count = 120)
        want = token_lift.to_tokens(long, vocabulary, True)
        counting = counters.COUNTING.set(counters.Reach(vocabulary, token_lift._trie(vocabulary).data))
        try:
            with pytest.raises(counters.NotCounted):
                A.force(plan())
        finally:
            counters.COUNTING.reset(counting)
        assert want.num_states == long.num_states + 1 or want.num_states == long.num_states


def forward_of(rows, window, where):
    """What :func:`counters._forward_moves` says of ``rows`` (class, target) of every state, ``window`` of one run."""
    K = 1 + max(where)
    moves = np.asarray([(c << 32) | t for s in range(K) for c, t in rows.get(s, [])], dtype = np.int64)
    row_len = np.asarray([len(rows.get(s, [])) for s in range(K)], dtype = np.int64)
    row_at = np.cumsum(np.concatenate([[0], row_len]), dtype = np.int64)[:K]
    where_g = np.full(K + 1, -1, dtype = np.int32)
    where_l = np.zeros(K + 1, dtype = np.int32)
    where_r = np.full(K + 1, -1, dtype = np.int32)
    for s, (g, l) in where.items():
        where_g[s], where_l[s], where_r[s] = g, l, 0
    forward = np.zeros(moves.size + 1, dtype = np.bool_)
    good = counters._forward_moves(moves, row_at, row_len, np.asarray(window, dtype = np.int32), where_g, where_l, where_r, 0, forward)
    return good, forward[:len(rows.get(window[0][0], []))].tolist()


def test_on_through_the_chain_or_out_of_it():
    window = [[0], [1], [2], [3]]
    where = {s: (s, 0) for s in range(4)}
    where[9] = (-1, 0)
    def moves(*template, next_):
        return {0: list(template), 1: list(next_)}
    assert forward_of(moves((0, 1), next_ = [(0, 2)]), window, where) == (True, [True])         # (one copy on from each)
    assert forward_of(moves((0, 2), next_ = [(0, 2)]), window, where) == (True, [False])        # (the same state from both: it left)
    assert forward_of(moves((0, 9), next_ = [(0, 9)]), window, where) == (True, [False])        # (out of the window)
    assert forward_of(moves((0, 1), next_ = [(0, 3)]), window, where)[0] is False               # (neither)
    assert forward_of(moves((0, 3), next_ = [(0, 9)]), window, where)[0] is False               # (past the window: spans more than was thought)
    assert forward_of(moves((0, 1), next_ = [(1, 2)]), window, where)[0] is False               # (another class)
    assert forward_of(moves((0, 1), next_ = []), window, where)[0] is False                     # (fewer moves)
    rows = {0: [(0, 1), (1, 9)], 1: [(0, 2)], 2: [(1, 9)]}                                      # (fewer, the next row's first what it lacks)
    assert forward_of(rows, window, where)[0] is False
    wide = [[0, 10], [1, 11], [2, 12], [3, 13]]
    where_wide = {g + 10 * l: (g, l) for g in range(4) for l in range(2)}
    where_wide[19] = (-1, 0)
    assert forward_of({0: [(0, 1)], 1: [(0, 12)]}, wide, where_wide)[0] is False                # (a copy on, another state of it)
    assert forward_of({0: [(0, 1)], 1: [(0, 2)]}, wide, where_wide) == (True, [True])


def test_a_state_in_two_chains_is_not_put_back_twice():
    table = np.asarray([[0], [1], [2]], dtype = np.int32)
    other = np.asarray([[5], [1], [6]], dtype = np.int32)
    roots = [counters.Run(table, 2, [[], [], []]), counters.Run(other, 2, [[], [], []])]
    inner, outer = counters.states_put_back(roots)
    K_kept, K_full = 7 + inner, 7 + inner + outer
    row_len = np.zeros(K_kept + 2, dtype = np.int64)
    with pytest.raises(counters.NotCounted):
        counters.expanded_moves(roots, np.arange(K_full + 1, dtype = np.int64), 7, K_kept, K_full, np.zeros(8, dtype = np.int64), np.zeros(K_kept + 2, dtype = np.int64), row_len,
                                np.zeros(K_full + 2, dtype = bool), 1)


def test_moves_of_a_copy_past_the_template():
    """
    A copy past the template stands after those put back. A move on through the chain stays where it is, as far on as the template's;
    a move that left the chain, its class's from the template or not, comes in by the start of the chain: into those put back.
    """
    window = np.asarray([[0, 5], [1, 6], [2, 7], [3, 8]], dtype = np.int32)
    K = 12
    where_g, where_l, where_r = np.full(K + 2, -1, dtype = np.int32), np.zeros(K + 2, dtype = np.int32), np.full(K + 2, -1, dtype = np.int32)
    template_of = np.zeros(K + 2, dtype = np.int32)
    for g in range(4):
        where_g[window[g]], where_l[window[g]], where_r[window[g]] = g, [0, 1], 0
        template_of[window[g]] = window[0]
    first_new, u = np.asarray([20], dtype = np.int64), np.asarray([2], dtype = np.int64)
    template = [(0, 1), (1, 1), (2, 1), (3, 2), (5, 9)]                     # (class, target) from state 0: g 1, l 0 and g 2, l 0, out of the window
    forward_at = [True, True, False, True, False]
    past = [(0, 2), (1, 3), (2, 2), (3, 7), (4, 3), (5, 9)]                 # from state 1: what it is
    want = [2, 20 + 2 * 2, 20 + 1 * 2, 20 + 2 * 1 + 1, 20 + 2 * 2, 9]
    moves = np.asarray([(c << 32) | t for c, t in template + past], dtype = np.int64)
    row_at, row_len = np.zeros(K + 2, dtype = np.int64), np.zeros(K + 2, dtype = np.int64)
    row_at[1], row_len[0], row_len[1] = len(template), len(template), len(past)
    forward = np.zeros(moves.size, dtype = np.bool_)
    forward[:len(template)] = forward_at
    counters._led_on(moves, row_at, row_len, forward, template_of, where_g, where_l, where_r, first_new, u, True, 0, K)
    assert (moves[len(template):] & counters._LOW).tolist() == want
    assert (moves[:len(template)] & counters._LOW).tolist() == [t for _, t in template]         # (the template's own: not in this pass)
    counters._led_on(moves, row_at, row_len, forward, template_of, where_g, where_l, where_r, first_new, u, False, 0, K)
    assert (moves[:len(template)] & counters._LOW).tolist() == [20, 20, 20, 22, 9]              # (from the template all into those put back)
    assert (moves[len(template):] & counters._LOW).tolist() == want


def test_moves_neither_on_nor_out_are_not_put_back():
    """A token going further than the span counted: the copies are not made from its moves."""
    roots = [counters.Run(np.asarray([[0], [1], [2]], dtype = np.int32), 2, [[], [], []])]
    inner, outer = counters.states_put_back(roots)
    K_kept, K_full = 3 + inner, 3 + inner + outer
    moves = np.asarray([(0 << 32) | 1, (0 << 32) | 0, (0 << 32) | 2, 0], dtype = np.int64)
    row_at = np.zeros(K_kept + 2, dtype = np.int64)
    row_len = np.zeros(K_kept + 2, dtype = np.int64)
    row_at[:3], row_len[:3] = [0, 1, 2], 1
    with pytest.raises(counters.NotCounted):
        counters.expanded_moves(roots, np.arange(K_full + 1, dtype = np.int64), 3, K_kept, K_full, moves, row_at, row_len, np.zeros(K_full + 2, dtype = bool), 1)


def test_a_counted_automaton_on_a_vocabulary_without_every_byte():
    with_all = vocabulary_of(random.Random(4))
    counting = counters.COUNTING.set(counters.Reach(with_all, token_lift._trie(with_all).data))
    try:
        a = A.force(A.concat(A.repeat(A.literal(b"ab"), 0, 90), A.literal(b"$")))
    finally:
        counters.COUNTING.reset(counting)
    assert a.counted
    with pytest.raises(counters.NotCounted):
        token_lift.to_tokens(a, jc.Vocabulary([b"a", b"b", None], eos = [2]), True)


def test_chains_of_lengths_one_in_the_short_automaton():
    """Two chains that come out one when both are cut, whose copies a token tells apart only in the whole automaton: the long way."""
    rng = random.Random(5)
    one = {"type": "object", "properties": {"a": {"type": "string", "maxLength": 60}}, "required": ["a"], "additionalProperties": False}
    two = {"type": "object", "properties": {"b": {"type": "string", "maxLength": 90}}, "required": ["b"], "additionalProperties": False}
    assert not check({"anyOf": [one, two]}, vocabulary_of(rng))


def test_copies_put_back_that_accept():
    """A text can end in any copy, as it can after the closer it leaves out."""
    vocabulary = vocabulary_of_pieces(A.force(A.concat(A.repeat(A.literal(b"ab"), 0, 90), A.optional(A.literal(b"$")))), random.Random(7), longest = 6)
    want = token_lift.to_tokens(A.force(A.concat(A.repeat(A.literal(b"ab"), 0, 90), A.optional(A.literal(b"$")))), vocabulary, True)
    counting = counters.COUNTING.set(counters.Reach(vocabulary, token_lift._trie(vocabulary).data))
    try:
        short = A.force(A.concat(A.repeat(A.literal(b"ab"), 0, 90), A.optional(A.literal(b"$"))))
        assert short.counted
        same(token_lift.to_tokens(short, vocabulary, True), want, "copies that accept")
    finally:
        counters.COUNTING.reset(counting)


def test_the_same_chain_twice():
    """In one set of states: the long way, the same."""
    check({"anyOf": [dict(SHORT), {"type": "array", "items": dict(SHORT)}, dict(SHORT)]}, vocabulary_of(random.Random(6)))


def test_a_chain_no_closer_follows_goes_the_long_way():
    """Its last copy differs from those before it where nothing follows the chain."""
    vocabulary = vocabulary_of(random.Random(8))
    counting = counters.COUNTING.set(counters.Reach(vocabulary, token_lift._trie(vocabulary).data))
    try:
        with pytest.raises(counters.NotCounted):
            A.force(A.repeat(A.literal(b"ab"), 0, 90))
    finally:
        counters.COUNTING.reset(counting)


# ---------------------------------------------------------------------------------------------
# A large automaton's table: written when first read, or made where it is wanted
# ---------------------------------------------------------------------------------------------

TO_COME = [
    STRING,                                                                             # (a run of copies not written out)
    {"type": "string", "minLength": 60, "maxLength": 130},                              # (two)
    {"type": "array", "items": STRING, "maxItems": 40},                                 # (one in another: the inner written out)
    {"type": "array", "items": STRING},                                                 # (in a loop: moves out of the copies)
    {"type": "object", "properties": {"a": STRING, "b": STRING}, "required": ["a"], "additionalProperties": False},
    {"anyOf": [STRING, {"type": "array", "items": STRING, "maxItems": 40}]},
    {"type": "array", "items": {"type": "boolean"}, "maxItems": 90},
    {"type": "string", "maxLength": 5},                                                 # (no copies)
    {"type": "number"},                                                                 # (no state accepts the empty text alone: the padding's own)
]


def to_come(schema, vocabulary, eos, monkeypatch):
    """The automaton, its table left to come, and as it is without that."""
    with monkeypatch.context() as m:
        m.setattr(token_lift, "TABLE_TO_COME", 1 << 62)
        want = jc.JSONSchema(schema, vocabulary, eos_padding = eos).automaton()
    assert torch.is_tensor(want._table)
    with monkeypatch.context() as m:
        m.setattr(token_lift, "TABLE_TO_COME", 0)
        m.setattr(token_lift, "SPARSE_ENTRIES", 0)
        got = jc.JSONSchema(schema, vocabulary, eos_padding = eos).automaton()
    assert isinstance(got._table, token_lift.TableToCome)
    return got, want


def on_devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("eos", [True, False])
@pytest.mark.parametrize("threads", [0, 1 << 62])
@pytest.mark.parametrize("schema", TO_COME)
def test_a_table_written_when_first_read(schema, threads, eos, monkeypatch):
    """Nothing but a read of the table writes it, once, in threads or not."""
    monkeypatch.setattr(token_lift, "THREADS_ENTRIES", threads)
    got, want = to_come(schema, vocabulary_of(random.Random(0), pieces = LONG_TOKENS), eos, monkeypatch)
    table = got._table
    assert (got.num_states, got.num_classes, got.initial) == (want.num_states, want.num_classes, want.initial)
    assert torch.equal(got.accept, want.accept) and torch.equal(got.token_class, want.token_class)
    assert got._table is table and table.made is None
    same(got, want, schema)
    assert got._table is table.made and table.dense() is table.made


@pytest.mark.parametrize("eos", [True, False])
@pytest.mark.parametrize("at_a_time", [1, 1 << 24])
@pytest.mark.parametrize("device", on_devices())
@pytest.mark.parametrize("schema", TO_COME)
def test_a_table_made_on_a_device_is_the_hosts(schema, device, at_a_time, eos, monkeypatch):
    """Copies a few at a time, or one by one: entry for entry, and none of it on the host."""
    monkeypatch.setattr(token_lift, "TABLE_ON_DEVICE_MOVES", at_a_time)
    got, want = to_come(schema, vocabulary_of(random.Random(0), pieces = LONG_TOKENS), eos, monkeypatch)
    on = got._table.on_device(device)
    assert got._table.made is None and on.dtype == torch.int32 and on.device.type == device
    assert torch.equal(on.cpu().long(), want._delta), schema
    same(got, want, schema)                                             # (the host's, after it)
    assert torch.equal(got._table, on.cpu().long())


@pytest.mark.parametrize("eos", [True, False])
@pytest.mark.parametrize("at_a_time", [1, 1 << 24])
@pytest.mark.parametrize("device", on_devices())
@pytest.mark.parametrize("schema", TO_COME)
def test_the_pairs_of_a_table_to_come(schema, device, at_a_time, eos, monkeypatch):
    """Every pair of states a move of the table joins but into the dead state, and none other, and few twice: without the table."""
    monkeypatch.setattr(token_lift, "TABLE_ON_DEVICE_MOVES", at_a_time)
    got, want = to_come(schema, vocabulary_of(random.Random(0), pieces = LONG_TOKENS), eos, monkeypatch)
    state, succ = got._table.pairs_on(device)
    assert got._table.made is None and state.dtype == succ.dtype == torch.int64 and state.device.type == device
    table, K = want._delta, want.num_states
    dead = (table == torch.arange(K)[:, None]).all(dim = 1) & ~want._accept
    rows, cls = torch.nonzero(~dead[table], as_tuple = True)
    pairs = set(zip(rows.tolist(), table[rows, cls].tolist()))
    found = list(zip(state.tolist(), succ.tolist()))
    assert set(found) == pairs, schema
    assert len(found) <= 2 * len(pairs)


def test_a_copy_nothing_leads_to():
    """A state of no place has no row, on the host or on the device. By hand: three copies of a state, all leading to the first, which is all
    that is reached; what the others would write is not in the table's last row, the dead state's."""
    copies = counters.Copies(np.asarray([3, 6], dtype = np.int64), np.asarray([1], dtype = np.int64), np.asarray([3], dtype = np.int64), np.asarray([0], dtype = np.int64),
                             np.asarray([0, 1], dtype = np.int64), np.asarray([0], dtype = np.int32), np.asarray([-1], dtype = np.int32), np.asarray([3], dtype = np.int32),
                             np.asarray([0], dtype = np.int32), np.asarray([0, 0], dtype = np.int64), np.zeros(0, dtype = np.int32), np.asarray([0], dtype = np.int64),
                             np.asarray([0], dtype = np.int32))
    moves = np.asarray([(0 << 32) | 3, 0], dtype = np.int64)
    row_at = np.asarray([0, 1, 1], dtype = np.int64)
    row_len = np.asarray([1, 0, 0], dtype = np.int64)
    order = np.asarray([0, 3, 1], dtype = np.int32)
    first_moves = np.asarray([1, 0], dtype = np.int32)
    for device in on_devices():
        table = token_lift.TableToCome(moves, row_at, row_len, copies, order, first_moves, 1, 2, 6, 2)
        on = table.on_device(device)
        assert on.size() == (3, 2)
        assert on.cpu().tolist() == [[1, 2], [1, 2], [2, 2]] == table.dense().tolist()
        state, succ = table.pairs_on(device)
        assert sorted(zip(state.tolist(), succ.tolist())) == [(0, 1), (1, 1)]    # (the first, and the first copy's; not the others')


def test_tables_on_a_device_leave_the_hosts_to_come(monkeypatch):
    """Where the table goes: the device's made there, the host's written for the host; either way the one read after is the same."""
    vocabulary = vocabulary_of(random.Random(0), pieces = LONG_TOKENS)
    got, want = to_come(STRING, vocabulary, True, monkeypatch)
    table = got._table
    for device in on_devices()[1:]:
        on = got._tables_on(device)[1]
        assert got._table is table and table.made is None and torch.equal(on.cpu().long(), want._delta)
        assert got._tables_on(device)[1] is on
    assert torch.equal(got._tables_on("cpu")[1], want._delta) and got._table is table.made
    for device in on_devices()[1:]:
        got, want = to_come(STRING, vocabulary, True, monkeypatch)      # the host's first, then the device's: that one
        assert torch.equal(got.delta, want.delta) and torch.equal(got._tables_on(device)[1].cpu().long(), want._delta)


def test_a_table_written_goes_to_a_device_as_it_is(monkeypatch):
    """A layout keeps the table to come, and may write it, the automaton still holding it to come."""
    got, want = to_come(STRING, vocabulary_of(random.Random(0), pieces = LONG_TOKENS), True, monkeypatch)
    table = got._table
    host = table.dense()
    assert got._table is table
    for device in on_devices():
        assert torch.equal(table.on_device(device).cpu().long(), want._delta)
        assert torch.equal(got._tables_on(device)[1].cpu().long(), host)
        state, succ = table.pairs_on(device)
        assert table.written_on is None                                 # (the table's moves let go)
        assert torch.equal(host[state.cpu()].eq(succ.cpu()[:, None]).any(dim = 1), torch.ones(state.numel(), dtype = torch.bool))
    first, last = on_devices()[-1], on_devices()[0]                     # (the moves kept on one device, the pairs asked for on another)
    table.on_device(first)
    state, _ = table.pairs_on(last)
    assert state.device.type == last and state.numel() > 0


def test_a_token_past_every_copy():
    """A move on from any copy past the last, into the window, none staying among the copies (short chains so far leave out more copies than that). By hand: two copies
    of one state, a move three on; what would be a copy before the first is the padding's state, here with a place and a row of its own."""
    copies = counters.Copies(np.asarray([7, 9], dtype = np.int64), np.asarray([1], dtype = np.int64), np.asarray([2], dtype = np.int64), np.asarray([0], dtype = np.int64),
                             np.asarray([0, 1], dtype = np.int64), np.asarray([0], dtype = np.int32), np.asarray([3], dtype = np.int32), np.asarray([0], dtype = np.int32),
                             np.asarray([0], dtype = np.int32), np.asarray([0, 1], dtype = np.int64), np.asarray([3], dtype = np.int32), np.asarray([0], dtype = np.int64),
                             np.asarray([0, 1, 2, 3, 4], dtype = np.int32))
    moves = np.asarray([(0 << 32) | 7, (1 << 32) | 8, (1 << 32) | 6, 0], dtype = np.int64)
    row_at = np.asarray([0, 3, 3, 3, 3, 3, 2], dtype = np.int64)
    row_len = np.asarray([2, 0, 0, 0, 0, 0, 1], dtype = np.int64)
    order = np.asarray([0, 7, 8, 2, 3, 5, 6], dtype = np.int32)
    first_moves = np.asarray([1, 0], dtype = np.int32)
    want = [[1, 2], [3, 5], [4, 5], [5, 5], [5, 5], [5, 5], [5, 6]]
    for device in on_devices():
        table = token_lift.TableToCome(moves, row_at, row_len, copies, order, first_moves, 5, 6, 9, 2)
        assert table.on_device(device).cpu().tolist() == want == table.dense().tolist()
        state, succ = table.pairs_on(device)
        assert sorted(zip(state.tolist(), succ.tolist())) == [(0, 1), (0, 2), (1, 3), (2, 4), (6, 6)]


@pytest.mark.parametrize("copying", [lambda d: pickle.loads(pickle.dumps(d)), copy.deepcopy, copy.copy])
def test_a_table_to_come_is_kept_and_copied_with_its_automaton(copying, monkeypatch):
    got, want = to_come(STRING, vocabulary_of(random.Random(0), pieces = LONG_TOKENS), True, monkeypatch)
    other = copying(got)
    assert isinstance(other._table, token_lift.TableToCome)
    same(other, want, "a copy")
    for device in on_devices():
        assert torch.equal(copying(got)._tables_on(device)[1].cpu().long(), want._delta)
    assert got == want


def test_a_tree_circuit_on_a_table_to_come(monkeypatch):
    """Its layout and reachability read what they read of the table, however it is held; the same answers."""
    if not torch.cuda.is_available():
        pytest.skip("needs a device")
    import pyjuice.nodes.distributions as dists
    vocabulary = vocabulary_of(random.Random(0), pieces = LONG_TOKENS)
    schema = {"type": "array", "items": {"type": "string", "maxLength": 12}, "maxItems": 6}
    n = 8
    torch.manual_seed(0)
    ns = juice.structures.PD(data_shape = (n,), num_latents = 4, split_intervals = 1, structure_type = "sum_dominated", input_dist = dists.Categorical(num_cats = vocabulary.size))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to("cuda")
    monkeypatch.setattr(layout, "LAYOUT_HOST_ENTRIES", 0)
    monkeypatch.setattr(token_lift, "TABLE_TO_COME", 1 << 62)
    eager = jc.compile(jc.JSONSchema(schema, vocabulary), pc)
    dfa = eager.backend.automaton
    data = dfa.sample(torch.zeros(1, n, vocabulary.size, device = "cuda"), 4, generator = torch.Generator(device = "cuda").manual_seed(0))[0]
    missing = torch.zeros(4, n, dtype = torch.bool, device = "cuda")
    missing[:, n // 2:] = True
    want = eager.marginal(data, missing), eager.conditional(data, missing)
    monkeypatch.setattr(token_lift, "TABLE_TO_COME", 0)
    monkeypatch.setattr(token_lift, "SPARSE_ENTRIES", 0)
    late = jc.compile(jc.JSONSchema(schema, vocabulary), pc)
    assert isinstance(late.backend.automaton._table, token_lift.TableToCome)
    for a, b in zip((late.marginal(data, missing), late.conditional(data, missing)), want):
        assert torch.allclose(a, b, atol = 1e-4, rtol = 1e-4)


@pytest.mark.parametrize("eos", [True, False])
def test_a_compile_and_its_queries_leave_the_table_unwritten(eos, monkeypatch):
    """On a device, the layout and every query read the device's: no host table for the largest automata. The answers are the eager table's."""
    if not torch.cuda.is_available():
        pytest.skip("needs a device")
    vocabulary = vocabulary_of(random.Random(0), pieces = LONG_TOKENS)
    schema = {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 12}
    n = 6
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = n, num_latents = 8, num_emits = vocabulary.size)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to("cuda")
    monkeypatch.setattr(layout, "LAYOUT_HOST_ENTRIES", 0)
    monkeypatch.setattr(token_lift, "TABLE_TO_COME", 1 << 62)
    eager = jc.compile(jc.JSONSchema(schema, vocabulary, eos_padding = eos), pc)

    def answers(cc, data, missing):
        with cc.decoder() as decoder:
            session = decoder.marginal(data, missing), decoder.conditional(data, missing)
        return cc.marginal(data, missing), cc.conditional(data, missing), *session

    dfa = eager.backend.automaton
    g = torch.Generator(device = "cuda").manual_seed(0)
    data = dfa.sample(torch.zeros(1, n, vocabulary.size, device = "cuda"), 4, generator = g)[0]
    missing = torch.zeros(4, n, dtype = torch.bool, device = "cuda")
    missing[:, n // 2:] = True
    want = answers(eager, data, missing)
    monkeypatch.setattr(token_lift, "TABLE_TO_COME", 0)
    monkeypatch.setattr(token_lift, "SPARSE_ENTRIES", 0)

    def not_written(self):
        raise AssertionError("the host's table was written")
    real = token_lift.TableToCome.dense
    monkeypatch.setattr(token_lift.TableToCome, "dense", not_written)
    constraint = jc.JSONSchema(schema, vocabulary, eos_padding = eos)
    late = jc.compile(constraint, pc)
    table = late.backend.automaton._table
    assert isinstance(table, token_lift.TableToCome)
    for a, b in zip(answers(late, data, missing), want):
        assert torch.allclose(a, b, atol = 1e-4, rtol = 1e-4)
    automaton = late.backend.automaton                                  # (the device's table, as it is)
    weights = torch.randn(2, n, vocabulary.size, device = "cuda", generator = torch.Generator(device = "cuda").manual_seed(1))
    assert torch.allclose(automaton.wmc(weights), dfa.wmc(weights))
    drawn = automaton.sample(weights, 3, generator = torch.Generator(device = "cuda").manual_seed(2))
    assert torch.equal(drawn, dfa.sample(weights, 3, generator = torch.Generator(device = "cuda").manual_seed(2)))
    built = late.backend.layout
    assert (built.num_states, built.num_classes) == (dfa.num_states, dfa.num_classes)
    monkeypatch.setattr(token_lift.TableToCome, "dense", real)
    assert torch.equal(built.delta, dfa._delta)                         # (asked for: as the eager one, kept where the layout keeps it)
    assert built.delta is table.made and late.backend.automaton._table is table
    longer, other = layout.build_layout(automaton, n + 1, "cuda"), layout.build_layout(dfa, n + 1, "cuda")      # (its pairs from the table to come, written or not)
    assert torch.equal(longer.width, other.width) and torch.equal(longer.state_id, other.state_id) and torch.equal(longer.col_of_state, other.col_of_state)
