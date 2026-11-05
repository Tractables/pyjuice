"""
:class:`pyjuice.constraints.JSONSchema`: JSON instances of a schema in outlines-core's canonical form (properties in
the schema's order, no others, an optional space around separators). Per form -- properties with required and optional
members, references, anyOf, patterns, formats, prefixItems, enum and const, free-form values, maps -- valid instances
written compactly and with spaces are accepted in random tokenizations, invalid or non-canonical texts in none; the
automaton's exact draws are valid instances (jsonschema); what the converter would drop or get wrong is refused, with
where; annotations are ignored; identity; widening; outlines-core missing.
"""
import decimal
import glob
import itertools
import json
import os
import random
import re
import sys

import numpy as np
import pytest
import torch

import pyjuice.constraints as jc


#: every byte a token, a few longer ones, an id without text, an end of sequence
TOKENS = [bytes([b]) for b in range(256)] + [b'{"', b'":', b'","', b"null", b"true", b'"ab"', "é".encode(), None,
                                              b"<eos>"]
VOCAB = jc.Vocabulary(TOKENS, eos = [len(TOKENS) - 1])

PERSON = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 5}, "age": {"type": "integer"},
                                            "tags": {"type": "array", "items": {"enum": ["a", 'b"c']}, "maxItems": 2}},
          "required": ["name"], "description": "a person"}
POINTS = {"$defs": {"pt": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                           "required": ["x", "y"]}},
          "type": "object", "properties": {"a": {"$ref": "#/$defs/pt"}, "b": {"anyOf": [{"$ref": "#/$defs/pt"},
                                                                                        {"type": "null"}]}},
          "required": ["a", "b"]}
CASES = [  # (schema, valid instances, invalid texts)
    (PERSON, [{"name": "Ada"}, {"name": "", "age": -3}, {"name": "x", "tags": ["a", 'b"c']},
              {"name": "é", "age": 0, "tags": []}],
     ['{"age":3}', '{"name":"Ada","extra":1}', '{"age":3,"name":"Ada"}', '{"name":"toolong"}',
      '{"name":"x","tags":["z"]}', '{"name":"x","tags":["a","a","a"]}', '{"name":3}', '{"name":"Ada"']),
    (POINTS, [{"a": {"x": 1, "y": -2.5}, "b": None}, {"a": {"x": 0, "y": 1e+3}, "b": {"x": 2, "y": 3}}],
     ['{"a":{"x":1},"b":null}', '{"a":{"x":1,"y":2}}', '{"b":null,"a":{"x":1,"y":2}}']),
    ({"type": "string", "pattern": "^(ab|cd)$"}, ["ab", "cd"], ['"abcd"', '"ab|cd"', '"ab', 'ab"', '"a"']),
    ({"type": "string", "pattern": "ab|cd"}, ["ab", "cd"], ['"ab', 'cd"', '"abcd"']),        # (grouped by us)
    ({"type": "object", "properties": {"d": {"type": "string", "format": "date"},
                                       "u": {"type": "string", "format": "uuid"}}, "required": ["d", "u"]},
     [{"d": "2026-10-17", "u": "123e4567-e89b-12d3-a456-426614174000"}],
     ['{"d":"2026-13-01","u":"123e4567-e89b-12d3-a456-426614174000"}', '{"d":"2026-10-17","u":"x"}']),
    ({"prefixItems": [{"type": "integer"}, {"type": "string"}]}, [[1, "a"], [-7, ""]], ['[1]', '["a",1]', '[1,"a",2]']),
    ({"type": "integer", "enum": [1, 22]}, [1, 22], ["2", '"1"', "122"]),
    ({"const": {"k": [True, None]}}, [{"k": [True, None]}], ['{"k":[true]}', '{"k":[null,true]}']),
    ({"title": "anything"}, [1, "x", None, [1, {"a": True}], {"k": [1, 2]}], ['{"a":', "[1,]", "nul"]),
    ({"type": "object", "properties": {"a": {}, "b": {"default": None}}, "required": ["a", "b"]},   # free members
     [{"a": 1, "b": "x"}, {"a": [1], "b": {"k": None}}], ['"x"', "1", '{"a":1}', '{"a":1,"b":}']),
    ({"prefixItems": [{}, {"type": "integer"}]}, [["x", 1], [None, -2]], ['"x"', '["x"]', '[1,"x"]']),
    ({"type": "object", "additionalProperties": {"type": "boolean"}, "minProperties": 1},
     [{"a": True}, {"a": True, "b": False, "c": True}], ["{}", '{"a":1}', '{"a":true,}']),
]


def texts(instance):
    """The instance compactly and with a space after every separator."""
    return [json.dumps(instance, separators = (",", ":"), ensure_ascii = False),
            json.dumps(instance, ensure_ascii = False)]


def tokenization(data: bytes, rng):
    """A random sequence of tokens spelling ``data``."""
    pieces = [(v, VOCAB.token_bytes(v)) for v in range(VOCAB.size) if VOCAB.token_bytes(v)]
    out, i = [], 0
    while i < len(data):
        v, t = rng.choice([(v, t) for v, t in pieces if data.startswith(t, i)])     # (single bytes always fit)
        out.append(v)
        i += len(t)
    return out


@pytest.mark.parametrize("case", range(len(CASES)))
def test_instances(case):
    schema, good, bad = CASES[case]
    c = jc.JSONSchema(schema, VOCAB)
    dfa = c.automaton()
    rng = random.Random(case)
    eos = VOCAB.eos[0]
    for want, items in ((True, [t for g in good for t in texts(g)]), (False, bad)):
        for text in items:
            for _ in range(2):
                ids = tokenization(text.encode(), rng) + [eos]
                assert c.accepts(ids) == want and dfa.accepts(ids) == want, (schema, text)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_draws_are_valid(case):
    """The automaton's exact uniform draws parse, and are valid instances (jsonschema, formats not asserted)."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = CASES[case][0]
    dfa = jc.JSONSchema(schema, VOCAB).automaton()
    validator = jsonschema.Draft202012Validator(schema)
    g = torch.Generator().manual_seed(case)
    n = next(n for n in (16, 32, 64, 128) if bool(dfa.accept_within(n)[dfa.initial]))
    for row in dfa.sample(torch.zeros(1, 2 * n, VOCAB.size), 32, generator = g)[0].tolist():
        instance = json.loads(VOCAB.decode([t for t in row if t not in VOCAB.eos]).decode())
        assert next(validator.iter_errors(instance), None) is None, (schema, instance)


INT_A = {"type": "object", "properties": {"a": {"type": "integer"}}}
INT_B = {"type": "object", "properties": {"b": {"type": "integer"}}}
KIND = {"oneOf": [{"type": "object", "properties": {"kind": {"const": "a"}, "a": {"type": "integer"}}, "required": ["kind"]},
                  {"type": "object", "properties": {"kind": {"const": "b"}, "b": {"type": "integer"}}, "required": ["kind"]}]}
COMBINATORS = [  # (schema, texts written, texts not written -- invalid, or valid but outside the canonical form)
    ({"oneOf": [INT_A, INT_B]}, [], ["{}", '{"a":1}', '{"b":2}']),           # open objects: every instance is both's
    ({"oneOf": [dict(INT_A, additionalProperties = False), dict(INT_B, additionalProperties = False)]},
     ['{"a":1}', '{"b":-2}'], ["{}", '{"a":1,"b":2}', '{"a":"x"}']),
    (KIND, ['{"kind":"a"}', '{"kind":"a","a":1}', '{"kind":"b","b":2}'], ['{"kind":"c"}', '{"kind":"a","a":"x"}']),
    ({"oneOf": [{"type": "integer"}, {"type": "number"}]}, ["1.5", "-2.25"], ["1", "0", "1.0", "1e5"]),
    ({"oneOf": [{"properties": {"a": {"type": "integer"}}}, {"type": "string"}]}, ['{"a":1}', "{}"], ['"x"']),
    ({"not": {"type": "string"}}, ["1", "null", "{}", "[1]", '{"x":[true]}'], ['"a"']),
    ({"not": {"enum": [1, "a"]}}, ["2", '"b"', "-7"], ["1", '"a"']),
    ({"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "string"}}}]},
     ['{"a":1}', '{"a":1,"b":"x"}'], ['{"b":"x"}', '{"a":"x"}', '{"a":1,"b":2}']),
    ({"allOf": [{"type": "string", "maxLength": 2}, {"type": "string", "minLength": 1}]}, ['"a"', '"ab"'],
     ['""', '"abc"']),
    ({"allOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "array", "maxItems": 1}]}, ["[]", "[1]"],
     ["[1,2]", '["a"]']),
    ({"allOf": [{"type": "object", "properties": {"a": {}}, "additionalProperties": False}, INT_B]}, ['{"a":1}', "{}"],
     ['{"b":1}']),                                            # (b is no property of the closed one)
    ({"if": {"type": "integer"}, "then": {"enum": [1, 2]}}, ["1", "2", '"a"', "null", "1.5"], ["3"]),
    ({"if": {"type": "object", "properties": {"kind": {"const": "a"}}, "required": ["kind"]},
      "then": {"type": "object", "properties": {"kind": {}, "a": {"type": "integer"}}, "required": ["kind", "a"]},
      "else": {"type": "string"}}, ['{"kind":"a","a":1}', '"x"'], ['{"kind":"a"}', '{"kind":"b"}', "1"]),
    ({"allOf": [{"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]}, KIND]},
     ['{"kind":"a"}', '{"kind":"b","b":1}'], ['{"kind":"c"}', "{}"]),
    ({"anyOf": [{"oneOf": [{"enum": [1, 2]}, {"enum": [2, 3]}]}, {"const": "x"}]}, ["1", "3", '"x"'], ["2", "4"]),
    ({"not": INT_A}, ['"x"', "1"], ['{"b":1}', "{}", '{"a":1}']),          # (an open object holds any other name)
    ({"if": {"type": "object", "properties": {"a": {"const": 1}}}, "then": {"type": "object", "maxProperties": 0},
      "else": {"type": "object"}}, ["{}"], ['{"b":1}', '{"a":1}']),     # ({"b":1} meets the if: then, not else)
    ({"allOf": [{"type": "object"}, {"anyOf": [{"oneOf": [INT_A, INT_B]}]}]}, [], ['{"a":1}', '{"b":2}', "{}"]),
]


@pytest.mark.parametrize("case", range(len(COMBINATORS)))
def test_combinators(case):
    schema, good, bad = COMBINATORS[case]
    c = jc.JSONSchema(schema, VOCAB)
    for want, items in ((True, good), (False, bad)):
        for text in items:
            assert c.accepts(list(text.encode()) + [VOCAB.eos[0]]) == want, (schema, text)


def instances():
    """A small universe of JSON values: scalars, short arrays, objects of a few names in every order."""
    scalars = [None, True, False, 0, 1, -1, 2, 3, 1.5, -2.25, "", "a", "b", "x", "ab", "abc"]
    values = list(scalars) + [[], [1], ["a"], [1, 2], [1, "a"], [None], [[1]], [{}]]
    for k in ("a", "b", "c", "kind"):
        values += [{k: v} for v in (1, 2, "a", "b", "x", None, True)]
    values += [{}, {"a": 1, "b": 2}, {"a": 1, "b": "x"}, {"a": "x", "b": 2}, {"a": 1, "c": True}]
    values += [{"kind": kind, k: v} for kind in ("a", "b", "c") for k in ("a", "b") for v in (1, "x")]
    return values


def written(value):
    """Texts of ``value``: every order of an object's names; an integer also as ``n.0`` and ``ne0``; compact and
    spaced."""
    out = set()
    if isinstance(value, dict):
        for order in itertools.permutations(value):
            body = ",".join(f"{json.dumps(k)}:{json.dumps(value[k])}" for k in order)
            out |= {"{" + body + "}", "{ " + body.replace(",", ", ").replace(":", ": ") + " }"}
    else:
        out |= {json.dumps(value, separators = (",", ":")), json.dumps(value)}
        if isinstance(value, int) and not isinstance(value, bool):
            out |= {f"{value}.0", f"{value}e0"}
    return out


@pytest.mark.parametrize("case", range(len(COMBINATORS)))
def test_combinators_write_instances_only(case):
    """Every text of the universe the automaton accepts -- in any order of names, any way to write a number -- is an
    instance (jsonschema): what is subtracted holds every instance's text."""
    jsonschema = pytest.importorskip("jsonschema")
    from pyjuice.constraints.language.backend.json_schema import compile_schema
    from pyjuice.constraints.language.json_schema import _prepare
    schema = COMBINATORS[case][0]
    automaton, validator = compile_schema(_prepare(schema), r"[ ]?"), jsonschema.Draft202012Validator(schema)
    for value in instances():
        for text in written(value):
            if automaton.accepts(text.encode()):
                assert validator.is_valid(json.loads(text)), (schema, text)


@pytest.mark.parametrize("case", range(len(COMBINATORS)))
def test_combinators_draws_are_valid(case):
    jsonschema = pytest.importorskip("jsonschema")
    schema = COMBINATORS[case][0]
    dfa = jc.JSONSchema(schema, VOCAB).automaton()
    if not bool(dfa.live_states()[dfa.initial]):
        return                                                  # (no text written)
    validator = jsonschema.Draft202012Validator(schema)
    g = torch.Generator().manual_seed(case)
    n = next(n for n in (16, 32, 64, 128) if bool(dfa.accept_within(n)[dfa.initial]))
    for row in dfa.sample(torch.zeros(1, 2 * n, VOCAB.size), 32, generator = g)[0].tolist():
        text = VOCAB.decode([t for t in row if t not in VOCAB.eos]).decode()
        instance = json.loads(text, parse_float = decimal.Decimal)      # (numbers exactly: as JSON Schema reads them)
        assert validator.is_valid(instance), (schema, text)


def test_the_token_lift_of_a_minimal_automaton_over_every_byte_is_minimal(monkeypatch):
    """Every byte a token and the byte automaton minimal: the lift walks its states apart and needs no minimization
    -- the same DFA as the walk from the initial state minimized, with the padding (merged into the state accepting
    the empty text alone, where there is one) and without."""
    from pyjuice.constraints.language.backend import automata as A, token_lift
    from pyjuice.constraints.language.backend.json_schema import compile_schema
    from pyjuice.constraints.language.json_schema import _prepare
    automata = [compile_schema(_prepare(s), r"[ ]?") for s in (PERSON, POINTS, KIND, {"type": "string"})]
    automata += [A.regex(p) for p in ("a*", "(ab|é)+c?", "")] + [A.nothing()]
    numba = pytest.importorskip("numba")
    threads = numba.get_num_threads()
    for a in automata:
        assert a.minimal
        for eos in (True, False):
            with monkeypatch.context() as m:
                m.setattr(token_lift, "_byte_complete", lambda trie: False)
                slow = token_lift.to_tokens(a, VOCAB, eos)
            for n in sorted({1, 2, 3, threads}):                # (walkers fewer than states: each takes several)
                numba.set_num_threads(min(n, threads))
                try:
                    fast = token_lift.to_tokens(a, VOCAB, eos)
                    assert fast == slow and fast.num_states == slow.num_states, (a.num_states, eos, n)
                    for first, serial, in_threads in ((256, 1 << 12, 1 << 24), (1, 1 << 12, 1 << 24), (1, 0, 1 << 24), (2, 0, 0)):
                        with monkeypatch.context() as m:               # (no walk first: all by the checks)
                            m.setattr(token_lift, "SERIAL_STEPS", 0)
                            m.setattr(token_lift, "FIRST_CHECKS", first)
                            m.setattr(token_lift, "SERIAL_CHECKS", serial)
                            m.setattr(token_lift, "WALKS_IN_THREADS", in_threads)
                            walked = token_lift.to_tokens(a, VOCAB, eos)
                        assert walked == slow and walked.num_states == slow.num_states, (a.num_states, eos, n, first)
                finally:
                    numba.set_num_threads(threads)


def test_the_checks_on_a_quotient_of_the_trie_tell_tokens_apart_as_every_walk_does(monkeypatch):
    """Random minimal byte automata, random tokens over all 256 bytes: tokens are one class exactly if they lead every
    state to one state, however few states have walked and with threads or without -- every token walked from every state
    here -- and the DFA is the one every walk gives."""
    from pyjuice.constraints.language.backend import automata as A, token_lift
    rng = np.random.default_rng(0)
    for case in range(60):
        C = int(rng.integers(2, 8))
        byte_class = np.arange(256) % C
        K = int(rng.integers(2, 60))
        table = np.full((K, C), K - 1, dtype = np.int32)
        moves = rng.random((K, C)) < rng.choice([0.2, 0.5, 0.9])
        table[moves] = rng.integers(0, K, moves.sum())
        table[K - 1] = K - 1
        accept = rng.random(K) < 0.5
        accept[K - 1] = False
        a = A._from_classes(table, 0, accept, byte_class)
        tokens = [bytes([b]) for b in range(256)] + [bytes(rng.integers(0, 6, int(rng.integers(2, 7))).tolist()) for _ in range(300)]
        vocabulary = jc.Vocabulary(tokens)
        first, classes = {}, []
        for text in tokens:
            column = []
            for q in range(a.num_states):
                for b in text:
                    q = int(a.table[q, a.byte_class[b]])
                column.append(q)
            classes.append(first.setdefault(tuple(column), len(first)))
        with monkeypatch.context() as m:
            m.setattr(token_lift, "_byte_complete", lambda trie: False)
            slow = token_lift.to_tokens(a, vocabulary, False)
        for steps, first_checks, serial, in_threads in ((1 << 19, 256, 1 << 12, 1 << 24), (0, 1, 1 << 12, 1 << 24), (1, 3, 0, 1 << 24),
                                                        (0, 2, 0, 0), (40, 1, 1 << 12, 1)):
            with monkeypatch.context() as m:
                m.setattr(token_lift, "SERIAL_STEPS", steps)
                m.setattr(token_lift, "FIRST_CHECKS", first_checks)
                m.setattr(token_lift, "SERIAL_CHECKS", serial)
                m.setattr(token_lift, "WALKS_IN_THREADS", in_threads)
                dfa = token_lift.to_tokens(a, vocabulary, False)
            assert dfa == slow and dfa.num_states == slow.num_states, (case, steps, first_checks)
            if a.num_states > 1:                                # (else no state but the dead one: no walk)
                got = dfa.token_class.tolist()
                assert got == classes, (case, steps, first_checks)


def same_tables(a, b, where):
    assert a.num_states == b.num_states and a.initial == b.initial, where
    assert torch.equal(a.delta, b.delta) and torch.equal(a.accept, b.accept) and torch.equal(a.token_class, b.token_class), where


def test_the_table_of_a_token_automaton_from_its_moves_is_the_dense_one(monkeypatch):
    """Minimal already: the table from every state's moves, numbered as it is made, against the dense table of every state's
    every class numbered afterwards, entry for entry: the dead state wherever a row first has a gap, the padding merged or a
    state of its own, tokens of empty text, on one thread and in parts -- schemas, and random automata over all 256 bytes."""
    from pyjuice.constraints.language.backend import automata as A, token_lift
    from pyjuice.constraints.language.backend.json_schema import compile_schema
    from pyjuice.constraints.language.json_schema import _prepare
    numba = pytest.importorskip("numba")
    threads = numba.get_num_threads()
    rng = np.random.default_rng(0)
    cases = [(compile_schema(_prepare(s), r"[ ]?"), VOCAB) for s in (PERSON, POINTS, KIND, {"type": "string"})]
    cases += [(A.regex(p), VOCAB) for p in ("a*", "(ab|é)+c?", "", "(a|b|c)*")]
    for case in range(40):
        C = int(rng.integers(1, 8))
        K = int(rng.integers(2, 60))
        table = np.full((K, C), K - 1, dtype = np.int32)
        moves = rng.random((K, C)) < rng.choice([0.2, 0.5, 1.0])
        table[moves] = rng.integers(0, K, moves.sum())
        table[K - 1] = K - 1
        accept = rng.random(K) < 0.5
        accept[K - 1] = False
        tokens = [bytes([b]) for b in range(256)] + [bytes(rng.integers(0, 6, int(rng.integers(2, 7))).tolist()) for _ in range(200)]
        tokens += [b""] if case % 2 else []
        cases.append((A._from_classes(table, 0, accept, np.arange(256) % C), jc.Vocabulary(tokens + [None], eos = [len(tokens)])))
    made, real = [], token_lift._canonical_token_table
    monkeypatch.setattr(token_lift, "_canonical_token_table", lambda *args: made.append(1) or real(*args))
    from_moves = 0
    for a, vocabulary in cases:
        for eos in (True, False):
            with monkeypatch.context() as m:
                m.setattr(token_lift, "SPARSE_ENTRIES", 1 << 62)
                dense = token_lift.to_tokens(a, vocabulary, eos)
            assert not made
            for n, parts in ((threads, 0), (1, 0), (min(3, threads), 0), (threads, 1 << 62)):
                numba.set_num_threads(n)
                try:
                    with monkeypatch.context() as m:
                        m.setattr(token_lift, "SPARSE_ENTRIES", 0)
                        m.setattr(token_lift, "THREADS_ENTRIES", parts)
                        same_tables(token_lift.to_tokens(a, vocabulary, eos), dense, (a.num_states, eos, n, parts))
                finally:
                    numba.set_num_threads(threads)
                from_moves += bool(made)                        # (the others: an automaton accepting nothing, no walk)
                made.clear()
    assert from_moves > 0.9 * 8 * len(cases)


def test_walks_by_ranges_give_the_hashes_of_every_step():
    """Random automata and tokens: the walks that take the levels of a subtree at once, where every byte below a node leads
    a state to one successor, against a walk of every node from every state: the same hash of every trie node, to the
    bit, and the same boundary states -- the successor holding for a level or more and then no longer, included."""
    from pyjuice.constraints.language.backend import token_lift as T
    rng = np.random.default_rng(1)
    uses_ranges = 0
    for case in range(120):
        C = int(rng.integers(1, 7))
        K = int(rng.integers(2, 40))
        dead = K - 1
        table = np.full((K, C), dead, dtype = np.int32)
        kind = case % 4
        for q in range(K - 1):                                  # (strings, counters, mostly dead, anything)
            if kind == 0 or (kind == 3 and rng.random() < 0.5):
                table[q] = q
            elif kind == 1:
                table[q] = min(q + 1, K - 2)
            elif kind == 2 and rng.random() < 0.3:
                table[q] = dead
            for c in rng.integers(0, C, int(rng.integers(0, max(C // 2, 1) + 1))).tolist():
                table[q, c] = rng.integers(0, K)
        shortest = 3 if case % 8 == 1 else 1                    # (counters, and no token ends at the first levels: not a state of a boundary there)
        tokens = [bytes(rng.integers(0, C, int(rng.integers(shortest, 7))).tolist()) for _ in range(int(rng.integers(5, 120)))]
        trie = T._trie(jc.Vocabulary(tokens))
        byte_class = np.minimum(np.arange(256), C - 1)
        first_child, node_byte, end_ptr = trie.first_child, trie.node_byte, trie.end_ptr
        M, S = first_child.size - 1, K
        r = [rng.integers(1, 2 ** 63, size = S, dtype = np.uint64) | np.uint64(1) for _ in range(4)]
        want1, want2 = np.zeros(M + 2, dtype = np.uint64), np.zeros(M + 2, dtype = np.uint64)
        order, seen = [0], {0}
        head = 0
        while head < len(order):                                # every node from every state found
            s = order[head]
            head += 1
            stack = [(0, s)]
            while stack:
                u, q = stack.pop()
                for w in range(first_child[u], first_child[u + 1]):
                    q2 = int(table[q, byte_class[node_byte[w]]])
                    if q2 == dead:
                        continue
                    x = np.uint64(q2 + 1)
                    with np.errstate(over = "ignore"):
                        want1[w] += r[0][s] * x + r[1][s]
                        want2[w] += r[2][s] * x + r[3][s]
                    if end_ptr[w + 1] > end_ptr[w] and q2 not in seen:
                        seen.add(q2)
                        order.append(q2)
                    stack.append((w, q2))
        class_bytes = T._bytes_of_classes(byte_class, C)
        usual, same = np.full(S, -1, dtype = np.int32), np.zeros((S, 4), dtype = np.uint64)
        count = np.zeros(S, dtype = np.int32)
        h1, h2, states, walked = T._walk(table, dead, 0, byte_class, first_child, node_byte, trie.ends_before, *r, trie.below, class_bytes, usual, same, count, 1 << 62)
        assert np.array_equal(h1, want1) and np.array_equal(h2, want2), case
        assert sorted(states.tolist()) == sorted(order) and walked == states.size, case
        uses_ranges += bool((usual >= 0).any())
        some = np.array(order, dtype = np.int64)                # from these states again, both other ways of walking
        again1, again2 = np.zeros(M + 2, dtype = np.uint64), np.zeros(M + 2, dtype = np.uint64)
        assert T._walks(table, dead, some, byte_class, first_child, node_byte, *r, trie.below, class_bytes, usual, same, count, again1, again2, 1 << 62) == some.size
        assert np.array_equal(again1, want1) and np.array_equal(again2, want2), case
        T._usual_moves(table, class_bytes, dead, usual, same, count)
        par1, par2 = T._walks_parallel(table, dead, some, byte_class, first_child, node_byte, *r, trie.below, class_bytes, usual, same, min(3, some.size), M + 2)
        assert np.array_equal(par1, want1) and np.array_equal(par2, want2), case
    assert uses_ranges == 120


def every_token_walked(a, vocabulary, eos):
    """The minimal automaton of a walk of every token from every state of a byte automaton, byte by byte: nothing else of the module's."""
    V, S = vocabulary.size, a.num_states
    offsets, has_text = vocabulary._offsets, vocabulary._has_text
    data = np.frombuffer(vocabulary._buffer, dtype = np.uint8)
    table = np.full((S + 1, V), a.dead, dtype = np.int64)
    for t in np.nonzero(has_text)[0].tolist():
        q = np.arange(S)
        for x in data[offsets[t]:offsets[t + 1]].tolist():
            q = a.table[q, a.byte_class[x]]
        table[:S, t] = q
    accept = np.append(a.accept, False)
    if eos:
        for e in vocabulary.eos:
            table[np.nonzero(a.accept)[0], e] = S
            table[S, e] = S
        accept[S] = True
    return jc.DFA(V, torch.arange(V), torch.from_numpy(table), a.initial, torch.from_numpy(accept)).minimize()


@pytest.mark.parametrize("few", [0, 300])
def test_the_trie_of_the_classes_or_of_the_bytes_the_same_automaton(few, monkeypatch):
    """Whichever trie the walks go through: the automaton of a walk of every token from every state, on few classes and many, tokens no
    walk gets to, tokens of empty text, ids past the trie's, with every byte a token or not."""
    from pyjuice.constraints.language.backend import automata as A, token_lift as T
    monkeypatch.setattr(T, "FEW_CLASSES", few)
    rng = np.random.default_rng(2)
    for case in range(60):
        C = int(rng.integers(2, 12))
        byte_class = np.minimum(np.arange(256), C - 1)
        K = int(rng.integers(2, 40))
        table = np.full((K, C), K - 1, dtype = np.int32)
        moves = rng.random((K, C)) < rng.choice([0.2, 0.5, 0.9])
        table[moves] = rng.integers(0, K, moves.sum())
        table[K - 1] = K - 1
        accept = rng.random(K) < 0.5
        accept[K - 1] = False
        a = A._from_classes(table, 0, accept, byte_class)
        if case % 3 == 0:
            a = A.minimize(a)
        tokens = [bytes([b]) for b in range(256)] if case % 2 else []
        if case % 2:                                                        # (a node of many children, bytes that live on none of them)
            tokens += [bytes([first, b]) for first in range(C) for b in rng.choice(256, 128, replace = False).tolist()]
        tokens += [bytes(rng.integers(0, C + 3, int(rng.integers(1, 7))).tolist()) for _ in range(120)] + [b""]
        vocabulary = jc.Vocabulary(tokens + [None], eos = [len(tokens)], size = len(tokens) + 3)
        eos = bool(case % 5)
        assert T.to_tokens(a, vocabulary, eos) == every_token_walked(a, vocabulary, eos), case


def test_token_classes_from_the_hashes():
    """Tokens are one class exactly if both hashes of their places are the same, numbered by their smallest, a class of no hash told
    as such; more classes than the table of slots starts with, neighbours with one hash the same."""
    from pyjuice.constraints.language.backend import token_lift as T
    rng = np.random.default_rng(3)
    for distinct in (1, 3, 40, 3000):
        E = 6000
        end_nodes = np.sort(rng.choice(20000, E, replace = False)).astype(np.int32)
        h1, h2 = np.zeros(20002, dtype = np.uint64), np.zeros(20002, dtype = np.uint64)
        values = rng.integers(1, 2 ** 62, size = (distinct, 2), dtype = np.uint64)
        values[distinct // 2] = values[0, 0], values[0, 1] ^ np.uint64(1)         # (one hash the same, the other nearly: in one slot)
        pick = rng.integers(0, distinct, E)
        pick[:3] = [0, distinct // 2, 0]
        if distinct > 3:
            pick[3:40] = pick[3]                                        # (a run)
        h1[end_nodes], h2[end_nodes] = values[pick, 0], values[pick, 1]
        zero = rng.random(E) < 0.1
        h1[end_nodes[zero]], h2[end_nodes[zero]] = 0, 0
        V = 9000
        token_end = rng.integers(0, E, V).astype(np.int32)
        token_class, reps, hashed = T._token_classes(h1, h2, end_nodes, token_end)
        first, want = {}, []
        for t, e in enumerate(token_end.tolist()):
            key = (int(h1[end_nodes[e]]), int(h2[end_nodes[e]]))
            want.append(first.setdefault(key, len(first)))
        assert token_class.tolist() == want, distinct
        assert reps.tolist() == [want.index(c) for c in range(len(first))], distinct
        assert hashed.tolist() == [key != (0, 0) for key in first], distinct


def test_the_quotient_of_the_trie_from_the_hashes():
    """Nodes are one exactly if both hashes are the same (the root alone); edges one per (node, class), from every member's children,
    on classes in every word of the bits; more nodes than the table of slots starts with; nothing goes on from a node no walk reached."""
    from pyjuice.constraints.language.backend import token_lift as T
    rng = np.random.default_rng(4)
    for distinct, per in ((2, 3), (30, 3), (5000, 2)):
        tokens = {bytes(rng.integers(0, 256, int(rng.integers(1, 5))).tolist()) for _ in range(30000)}
        trie = T._trie(jc.Vocabulary(sorted(tokens)))
        first_child, node_byte = trie.first_child, trie.node_byte
        M = first_child.size - 1
        byte_class = (np.arange(256) * 7 % 200).astype(np.int64)
        values = rng.integers(1, 2 ** 62, size = (distinct, 2), dtype = np.uint64)
        values[distinct // 2] = values[0, 0], values[0, 1] ^ np.uint64(1)          # (one hash the same, the other nearly: in one slot)
        pick = rng.integers(-1, distinct, M + 2)
        pick[1:3] = 0, distinct // 2                            # (neighbours)
        h1, h2 = values[pick, 0], values[pick, 1]
        h1[pick < 0], h2[pick < 0] = 0, 0
        h1[0], h2[0] = 0, 0
        gptr, gclass, gnode, into, zero = T._quotient(h1, h2, first_child, node_byte, byte_class)
        group = {}
        of = [0] * M
        for u in range(1, M):
            of[u] = group.setdefault((int(h1[u]), int(h2[u])), len(group) + 1)
        assert gptr.size - 1 == len(group) + 1
        assert (zero == group[(0, 0)]) if (0, 0) in group else zero == -1
        want = {}
        for u in range(M):
            if of[u] != zero:
                for w in range(first_child[u], first_child[u + 1]):
                    want.setdefault((of[u], int(byte_class[node_byte[w]])), set()).add(of[w])
        got = {}
        for g in range(gptr.size - 1):
            for e in range(gptr[g], gptr[g + 1]):
                assert (g, int(gclass[e])) not in got
                got[g, int(gclass[e])] = int(gnode[e])
        assert set(got) == set(want), distinct
        assert all(got[key] in want[key] for key in got), distinct
        assert into.tolist() == np.bincount(np.asarray(list(got.values())), minlength = into.size).tolist()


def test_a_schema_is_copied_as_deepcopy_copies_it():
    """What is written twice stays one object, a cycle a cycle, anything not JSON copied; nothing of the copy is the schema's but what cannot be written to."""
    import copy
    import decimal
    from pyjuice.constraints.language.json_schema import _copied
    shared = {"type": "string", "enum": ["a", None, 1, 1.5, True]}
    other = decimal.Decimal("1.5")
    schema = {"a": shared, "b": [shared, {"c": shared}], "d": other, "e": (1, [2]), "f": [[]]}
    schema["g"] = schema
    got, want = _copied(schema, {}), copy.deepcopy(schema)
    assert got["a"] is got["b"][0] is got["b"][1]["c"] and got["g"] is got
    assert got["a"] is not shared and got["b"] is not schema["b"] and got["f"][0] is not schema["f"][0]
    assert got["e"] == want["e"] and got["e"][1] is not schema["e"][1]
    del got["g"], want["g"]
    assert got == want
    got["a"]["enum"].append(2)
    assert shared["enum"] == ["a", None, 1, 1.5, True]


def test_the_random_words():
    from pyjuice.constraints.language.backend import token_lift as T
    for S in (1, 100, T._KEPT_WORDS, T._KEPT_WORDS + 1):
        words = T._random_words(S)
        assert len(words) == 4 and all(w.size == S and w.dtype == np.uint64 for w in words)
        assert (words[0] & np.uint64(1)).all() and (words[2] & np.uint64(1)).all()      # (odd: they multiply, nothing lost)
        assert all(np.array_equal(a, b) for a, b in zip(words, T._random_words(S)))


@pytest.mark.parametrize("few", [0, 300])
def test_ids_a_vocabulary_grew_after_its_trie(few, monkeypatch):
    """The trie made for a vocabulary serves it with more ids, and those lead nowhere, on either trie."""
    from pyjuice.constraints.language.backend import automata as A, token_lift as T
    monkeypatch.setattr(T, "FEW_CLASSES", few)
    tokens = [bytes([b]) for b in range(256)] + [b"ab", b"abab", None]
    small = jc.Vocabulary(tokens, eos = [len(tokens) - 1])
    T._trie(small)
    large = small.with_size(small.size + 5)
    for a in (A.force(A.repeat(A.literal(b"ab"), 0, 3)), A.force(A.repeat(A.byte_ranges([(i, i) for i in range(0, 200, 8)]), 0, 3))):
        for eos in (False, True):
            got = T.to_tokens(a, large, eos)
            assert got == every_token_walked(a, large, eos), (a.byte_class.max(), eos)


def test_numbers_below_a_combinator_are_written_one_way():
    from pyjuice.constraints.language.backend.json_schema import canonical_number
    for x, text in [(1, "1"), (1.0, "1"), (-0.0, "0"), (0, "0"), (1.5, "1.5"), (1e-7, "0.0000001"),
                    (2.5e-7, "0.00000025"), (1e20, "100000000000000000000"), (-12.5, "-12.5"), (10 ** 30, "1" + "0" * 30)]:
        assert canonical_number(x) == text, (x, canonical_number(x))
    c = jc.JSONSchema({"oneOf": [{"const": 1.0}, {"type": "string"}]}, VOCAB)
    eos = [VOCAB.eos[0]]
    assert c.accepts(list(b"1") + eos) and not c.accepts(list(b"1.0") + eos)


RECURSIVE = {"$defs": {"n": {"type": "object", "properties": {"next": {"$ref": "#/$defs/n"}}}}, "$ref": "#/$defs/n"}


@pytest.mark.parametrize("schema, why", [
    ({"type": "integer", "minimum": 0}, "#: ['minimum'] for type integer"),
    ({"type": "number", "multipleOf": 2}, "multipleOf"),
    ({"type": "array", "items": {}, "uniqueItems": True}, "uniqueItems"),
    ({"type": "string", "not": {"const": "x"}}, "['type'] beside not"),
    ({"oneOf": [{"type": "string"}], "type": "string"}, "['type'] beside oneOf"),
    ({"oneOf": []}, "oneOf of no schema"),
    ({"if": {"type": "string"}, "then": {"maxLength": 3}}, "#/then: ['maxLength'] (no form"),
    ({"if": {}, "then": {}, "type": "object"}, "['type'] beside if/then/else"),
    ({"allOf": [{"type": "string"}, {"maxLength": 3}]}, "#/allOf/1: ['maxLength'] (no form"),
    ({"anyOf": [{"type": "string"}], "type": "string"}, "['type'] beside anyOf"),
    ({"type": "object", "properties": {"a": {}}, "minProperties": 1}, "['minProperties'] beside properties"),
    ({"type": "object", "properties": {"a": {}}, "required": ["b"]}, "required ['b'] not among the properties"),
    ({"type": "object", "properties": {"a": {"type": "integer", "maximum": 3}}}, "#/properties/a: ['maximum']"),
    (RECURSIVE, "a recursive $ref"),
    ({"$ref": "other.json#/x"}, "an external $ref"),
    ({"$ref": "#/$defs/missing"}, "does not resolve"),
    ({"$defs": {"a": {}}, "$ref": "#/$defs/a", "type": "object"}, "['type'] beside $ref"),
    ({"type": "string", "minLength": 1, "pattern": "a"}, "together"),
    ({"type": "string", "format": "email"}, "format 'email'"),
    ({"type": "string", "format": "time"}, "format 'time'"),
    ({"type": "string", "pattern": 'a|"'}, "can match a quote"),
    ({"type": "string", "pattern": "a\\\\b"}, "can match a quote, a backslash"),
    ({"type": "string", "pattern": "(?=a)a"}, "lookahead"),
    ({"type": "integer", "enum": ["a"]}, "outside type"),
    ({"type": "integer", "enum": [1, True]}, "outside type"),                  # (true is no integer)
    ({"properties": {"a": True}}, "a boolean schema"),
    ({"type": "object", "additionalProperties": False}, "additionalProperties false"),
    ({"type": "object", "additionalProperties": {"type": "boolean"}, "maxProperties": 2}, "maxProperties above 0"),
    ({"type": "object", "minProperties": 2}, "minProperties above 1"),
    ({"required": ["a"]}, "no form the converter reads"),
])
def test_refused(schema, why):
    with pytest.raises(ValueError, match = "Unsupported JSON Schema") as e:
        jc.JSONSchema(schema, VOCAB)
    assert why in str(e.value), str(e.value)


def test_annotations_and_identity():
    """Keywords without a meaning in validation are ignored (an object of them only: any value); a schema and its
    JSON text are the same constraint; whitespace and padding make others."""
    plain = jc.JSONSchema({"type": "string"}, VOCAB)
    assert jc.JSONSchema({"type": "string", "_description": "x", "id": "y", "nullable": True}, VOCAB) == plain
    assert jc.JSONSchema({"description": "free"}, VOCAB) == jc.JSONSchema({}, VOCAB)
    assert jc.JSONSchema(json.dumps(PERSON), VOCAB) == jc.JSONSchema(PERSON, VOCAB)
    assert jc.JSONSchema(PERSON, VOCAB).schema == PERSON
    string = r'"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])*"'                  # (the same texts, but a Regex)
    for other in (jc.JSONSchema({"type": "string"}, VOCAB, eos_padding = False), jc.Regex(string, VOCAB)):
        assert other != plain
    assert jc.Regex(string, VOCAB).automaton() == plain.automaton()
    spaced = jc.JSONSchema(PERSON, VOCAB, whitespace = r"[ \n]*")               # (objects: separators, whitespace)
    assert spaced != jc.JSONSchema(PERSON, VOCAB) and spaced.accepts(list(b'{\n  "name": "x"\n}') + [VOCAB.eos[0]])
    with pytest.raises(TypeError):
        jc.JSONSchema([1], VOCAB)


def test_widened():
    c = jc.JSONSchema(PERSON, VOCAB)
    wide = c._widened(VOCAB.size + 3)
    assert isinstance(wide, jc.JSONSchema) and wide.schema == PERSON and wide.vocab_size == VOCAB.size + 3
    assert wide == jc.JSONSchema(PERSON, VOCAB.with_size(VOCAB.size + 3))


def test_without_outlines_core(monkeypatch):
    monkeypatch.setitem(sys.modules, "outlines_core", None)
    c = jc.JSONSchema(PERSON, VOCAB)
    assert c.accepts(list(b'{"name":"x"}') + [VOCAB.eos[0]]) and not c.accepts(list(b'{"age":1}') + [VOCAB.eos[0]])


#: every form the checker lets through, one schema each (with CASES)
FORMS = [{"type": "string"}, {"type": "integer"}, {"type": "number", "format": "double"}, {"type": "boolean"},
         {"type": "null"}, {"type": ["string", "null"]}, {"type": "string", "minLength": 2, "maxLength": 4},
         {"type": "string", "minLength": 2}, {"type": "string", "maxLength": 3}, {"type": "string", "pattern": "a[bc]+"},
         {"type": "string", "format": "date-time"}, {"type": "array"}, {"type": "array", "minItems": 1},
         {"type": "array", "items": {"type": "null"}, "minItems": 1, "maxItems": 3},
         {"type": "array", "items": {"type": "null"}, "minItems": 2}, {"type": "array", "items": {}, "maxItems": 0},
         {"type": "object"}, {"type": "object", "minProperties": 1}, {"type": "object", "maxProperties": 0},
         {"type": "object", "additionalProperties": {"type": "integer"}},
         {"type": "object", "properties": {"a": {"type": "null"}, "b": {"type": "null"}, "c": {"type": "null"}},
          "required": ["b"]},
         {"type": "object", "properties": {"a": {"type": "null"}, "b": {"type": "null"}}},
         {"type": "object", "properties": {}}, {"enum": ["x", 'q"\\', 1, None, True, [1, "a"], {"k": 2}, 1.5, [], {}]},
         {"const": -2.5e-7}, {"prefixItems": [{"type": "null"}, {"type": "boolean"}], "items": {"type": "null"}},
         {"allOf": [{"type": "boolean"}]}, {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "integer"}}]}]


def outlines_core_regex(schema, whitespace):
    """outlines-core's regex of the (checked) schema."""
    oc = pytest.importorskip("outlines_core")
    from pyjuice.constraints.language.json_schema import _prepare
    return oc.json_schema.build_regex_from_schema(json.dumps(_prepare(schema)), whitespace)


def outlines_core_texts(schema, whitespace, pattern = None):
    """``(ours, outlines-core's)``: the compiled texts of the (checked) schema, and those of outlines-core's regex
    (``pattern``, when made already)."""
    from pyjuice.constraints.language.backend import automata as A
    from pyjuice.constraints.language.backend.json_schema import compile_schema
    from pyjuice.constraints.language.json_schema import _prepare
    ref = A.regex(outlines_core_regex(schema, whitespace) if pattern is None else pattern)
    return compile_schema(_prepare(schema), whitespace), ref


def same_as_outlines_core(schema, whitespace):
    """Whether the compiled texts are those of outlines-core's regex of the (checked) schema."""
    from pyjuice.constraints.language.backend import automata as A
    return A.equivalent(*outlines_core_texts(schema, whitespace))


def ref_depth(schema):
    """The most references followed one inside another from the root (none recursive: checked)."""
    memo = {}

    def depth(node):
        if isinstance(node, list):
            return max((depth(v) for v in node), default = 0)
        if not isinstance(node, dict):
            return 0
        if isinstance(node.get("$ref"), str) and node["$ref"].startswith("#"):
            ref = node["$ref"]
            if ref not in memo:
                target = schema
                for part in [p for p in ref[1:].split("/") if p]:
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                memo[ref] = 1 + depth(target)
            return memo[ref]
        return max((depth(v) for k, v in node.items() if k not in ("definitions", "$defs")), default = 0)

    return depth(schema)


@pytest.mark.slow                                       # (MEASURED: 22 s each, every form against outlines-core)
@pytest.mark.parametrize("whitespace", [r"[ ]?", r"[ \n]*"])
def test_the_texts_are_outlines_cores(whitespace):
    for schema in FORMS + [case[0] for case in CASES]:
        assert same_as_outlines_core(schema, whitespace), (schema, whitespace)


def test_property_names_are_json_strings():
    """outlines-core writes a property name as it is; here it is a JSON string (its quotes and backslashes escaped)."""
    c = jc.JSONSchema({"type": "object", "properties": {'a"b\\': {"type": "null"}}, "required": ['a"b\\']}, VOCAB)
    eos = [VOCAB.eos[0]]
    assert c.accepts(list(b'{"a\\"b\\\\":null}') + eos) and not c.accepts(list(b'{"a"b\\":null}') + eos)


def test_floats_are_written_as_outlines_core_writes_them():
    from pyjuice.constraints.language.backend.json_schema import float_text
    for x, text in [(1e16, "1e+16"), (1e15, "1000000000000000.0"), (1e21, "1e+21"), (1.2345678901234568e20,
                     "1.2345678901234568e+20"), (0.1, "0.1"), (1e-5, "0.00001"), (1e-6, "1e-6"), (1e-7, "1e-7"),
                    (1.5e300, "1.5e+300"), (-2e-10, "-2e-10"), (3.0, "3.0"), (100.0, "100.0"), (1e5, "100000.0"),
                    (12.5, "12.5"), (2.0 ** 53, "9007199254740992.0"), (5e-324, "5e-324"), (-0.0, "-0.0"),
                    (0.0, "0.0"), (-1.25e-5, "-0.0000125")]:
        assert float_text(x) == text, (x, float_text(x), text)
        assert float(text) == x


def jsonschemabench(max_chars):
    """The JSONSchemaBench test schemas in the Hugging Face cache of at most ``max_chars`` characters (skips if none)."""
    hub = pytest.importorskip("huggingface_hub")
    pq = pytest.importorskip("pyarrow.parquet")
    try:
        root = hub.snapshot_download("epfl-dlab/JSONSchemaBench", repo_type = "dataset", local_files_only = True)
    except Exception:                                       # (not in the cache)
        pytest.skip("JSONSchemaBench is not in the Hugging Face cache")
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "test-*.parquet"))):
        subset = os.path.basename(os.path.dirname(path))
        out += [(subset, k, text) for k, text in enumerate(pq.read_table(path).column("json_schema").to_pylist())
                if len(text) <= max_chars]
    return out


JSONSCHEMABENCH = ("Glaiveai2K", "Github_trivial", "Github_easy", "Github_medium", "Github_hard", "Kubernetes", "Snowplow",
                   "WashingtonPost", "JsonSchemaStore")


@pytest.mark.slow
@pytest.mark.parametrize("subset", JSONSCHEMABENCH)
def test_the_texts_are_outlines_cores_on_jsonschemabench(subset):
    """Every schema the checker lets through (up to 4000 characters) has outlines-core's texts -- but those whose
    property names hold a quote, a backslash or a control character, and those nesting more than 3 references: past
    them outlines-core drops a property, required or not (its texts then miss documents with it, or hold documents
    without a required one), or gives up. A schema with `oneOf` has some of them: outlines-core reads it as `anyOf`,
    and the texts more than one branch holds are no instances (`{"a": {}}` under `{"a": {"oneOf": [x, y]}}`, two
    closed objects without `required`). Left out: schemas with `allOf`, `not` or `if` -- outlines-core concatenates
    `allOf`'s branches (`{"v": 1}{"w": 2}`, no JSON) and refuses the others; the combinator tests above check those --
    and those whose regex repeats a piece more than 256 times (`maxLength: 32767`) or runs past 8k characters
    (outlines-core writes a free-form value's regex out in place, 18 of them 1 MB; a tenth of the corpus): its
    automaton takes seconds to minutes, over ten for the longest, where ours compiles in under a second."""
    from pyjuice.constraints.language.backend import automata as A
    from pyjuice.constraints.language.json_schema import _prepare
    equal = subsets = 0
    for part, k, text in jsonschemabench(4000):
        if part != subset:
            continue
        schema = json.loads(text)
        try:
            _prepare(schema)
        except ValueError:
            continue
        if any(any(c in name for c in '"\\') or any(ord(c) < 0x20 for c in name) for name in property_names(schema)):
            continue
        if ref_depth(schema) > 3 or combinators(schema) & {"allOf", "not", "if"}:
            continue
        pattern = outlines_core_regex(schema, r"[ ]?")
        if max_repetition(pattern) > 256 or len(pattern) > 8192:
            continue
        ours, theirs = outlines_core_texts(schema, r"[ ]?", pattern)
        if "oneOf" in combinators(schema):
            assert A.is_empty(A.difference(ours, theirs)), (subset, k)
            subsets += 1
        else:
            assert A.equivalent(ours, theirs), (subset, k)
            equal += 1
    assert equal + subsets > 0, subset


def combinators(node):
    """The combinators (`anyOf`, `oneOf`, `allOf`, `not`, `if`) anywhere in a schema."""
    out = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("anyOf", "oneOf", "allOf", "not", "if"):
                out.add(key)
            out |= combinators(value)
    elif isinstance(node, list):
        for value in node:
            out |= combinators(value)
    return out


def max_repetition(pattern):
    """The largest bound of a counted repetition (`{n}`, `{m,n}`) in a regex."""
    return max((int(x) for pair in re.findall(r"\{\d*,(\d+)\}|\{(\d+)\}", pattern) for x in pair if x), default = 0)


def test_references_are_followed_at_any_depth():
    """Five references one inside another: outlines-core drops the property past its third (required or not); here it
    is read."""
    defs = {f"d{i}": {"type": "object", "properties": {f"x{i}": {"$ref": f"#/$defs/d{i + 1}"}, "y": {"type": "null"}}}
            for i in range(4)}
    defs["d4"] = {"type": "null"}
    c = jc.JSONSchema({"$defs": defs, "$ref": "#/$defs/d0"}, VOCAB)
    assert c.accepts(list(b'{"x0":{"x1":{"x2":{"x3":null}}}}') + [VOCAB.eos[0]])
    assert not c.accepts(list(b'{"x0":{"x1":{"x2":{"x3":1}}}}') + [VOCAB.eos[0]])


def property_names(node):
    """Every key of every ``properties`` in a schema."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "properties" and isinstance(v, dict):
                yield from v
            yield from property_names(v)
    elif isinstance(node, list):
        for v in node:
            yield from property_names(v)
