"""
:class:`pyjuice.constraints.JSONSchema`: JSON instances of a schema in outlines-core's canonical form (properties in
the schema's order, no others, an optional space around separators). Per form -- properties with required and optional
members, references, anyOf, patterns, formats, prefixItems, enum and const, free-form values, maps -- valid instances
written compactly and with spaces are accepted in random tokenizations, invalid or non-canonical texts in none; the
automaton's exact draws are valid instances (jsonschema); what the converter would drop or get wrong is refused, with
where; annotations are ignored; identity; widening; outlines-core missing.
"""
import json
import random
import sys

import pytest
import torch

import pyjuice.constraints as jc

pytest.importorskip("interegular")
pytest.importorskip("outlines_core")

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


RECURSIVE = {"$defs": {"n": {"type": "object", "properties": {"next": {"$ref": "#/$defs/n"}}}}, "$ref": "#/$defs/n"}


@pytest.mark.parametrize("schema, why", [
    ({"type": "integer", "minimum": 0}, "#: ['minimum'] for type integer"),
    ({"type": "number", "multipleOf": 2}, "multipleOf"),
    ({"type": "array", "items": {}, "uniqueItems": True}, "uniqueItems"),
    ({"type": "string", "not": {"const": "x"}}, "['not'] for type string"),
    ({"oneOf": [{"type": "string"}]}, "#: oneOf"),
    ({"allOf": [{"type": "string"}, {"maxLength": 3}]}, "allOf of 2 schemas"),
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
    for other in (jc.JSONSchema({"type": "string"}, VOCAB, eos_padding = False), jc.Regex(plain.pattern, VOCAB)):
        assert other != plain
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
    with pytest.raises(ImportError, match = r"pyjuice\[constraints\]"):
        jc.JSONSchema({"type": "string"}, VOCAB)
