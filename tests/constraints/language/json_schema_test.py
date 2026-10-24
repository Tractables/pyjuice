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

import pytest
import torch

import pyjuice.constraints as jc

pytest.importorskip("interegular")

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
    from pyjuice.constraints.language.backend import automata as A, byte_level
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
                m.setattr(byte_level, "_byte_complete", lambda trie: False)
                slow = byte_level.to_tokens(a, VOCAB, eos)
            for n in sorted({1, 2, 3, threads}):                # (walkers fewer than states: each takes several)
                numba.set_num_threads(min(n, threads))
                try:
                    fast = byte_level.to_tokens(a, VOCAB, eos)
                finally:
                    numba.set_num_threads(threads)
                assert fast == slow and fast.num_states == slow.num_states, (a.num_states, eos, n)


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
