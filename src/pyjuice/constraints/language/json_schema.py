"""
JSON Schema over a token sequence's text.

:class:`JSONSchema` is the set of token sequences whose text is an instance of a schema written as JSON in a CANONICAL
form -- outlines-core's (``build_regex_from_schema``): properties in the schema's order and no others, ``whitespace``
around separators, strings without ``\\u`` escapes or characters U+007F-U+009F, numbers with a signed exponent,
free-form values (``{}``, objects and arrays whose members have no schema) nested at most two levels deep, a
``pattern`` matching a string's whole content. Every such text is valid under the schema, formats taken as annotations
(JSON Schema's default); not every valid instance has one. The automaton is compiled from the schema directly, through
an algebra of byte automata (:mod:`.backend.json_schema`, :mod:`.backend.automata`), each schema object once.

A schema object holds one form -- ``properties``, ``allOf``, ``anyOf``, ``oneOf``, ``not``, ``if``/``then``/``else``,
``prefixItems``, ``enum``, ``const``, ``$ref`` or ``type`` -- with the keywords that form reads, and annotations (any
keyword without a meaning in JSON Schema validation: titles, descriptions, unknown keywords). So every schema object
of a schema (those its references reach included) is checked first; refused, with where and why, are a validation
keyword its form does not read (numeric bounds, ``multipleOf``, ``uniqueItems``, ``contains``, ``patternProperties``,
``minProperties`` beside ``properties``, a combinator beside another keyword, ...), a recursive or external ``$ref``,
``required`` names outside ``properties``, bounds on the number of an object's members other than "not empty" and
"empty", a string's length together with its ``pattern`` or ``format``, a format other than ``date-time``, ``date``
and ``uuid``, and a ``pattern`` that can match a quote, a backslash or a control character (it would be read against
the JSON text of the string, escapes and all). A pattern's anchors are stripped and it is put in a group, a schema of
annotations only (``{}`` among them) is any value, and ``then``/``else`` without an ``if`` constrain nothing.
References are followed to any depth (outlines-core drops a property, required or not, past its third).

The combinators are read exactly, up to the canonical form: ``allOf`` intersects its schemas (objects merged member by
member, a name one closes left out), ``oneOf`` admits the values exactly one of its schemas admits, ``not`` those its
schema does not, ``if``/``then``/``else`` those ``then`` admits among those ``if`` does and those ``else`` admits among
the others. What a combinator subtracts holds EVERY text of the other schema's instances that it could write, so it
writes instances only: below one, ``properties`` without ``"type": "object"`` is met by every value not an object (and
``prefixItems`` without ``"type": "array"`` by every value not an array), an object closed by ``additionalProperties:
false`` admits no other name and an open one any. Below a combinator that subtracts (``oneOf``, ``not``, ``if``, and an
``allOf`` holding one), texts are written in one canonical form of their own: property names in the order they first
appear there, numbers without exponent or trailing zero (``0`` for zero), strings without ``\\/``, free-form objects
only with names no schema there declares. Numbers are exact decimals, as JSON Schema reads them: a non-integer of many
digits that a binary floating-point reading would round to an integer is still no integer.

Two constraints are equal when their languages are: the same canonical texts over the same vocabulary and padding.

Needs ``interegular`` (``pip install pyjuice[constraints]``): patterns and the ``whitespace`` pattern.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional, Union

from .base import Constraint
from .regex import _Text, _char_automaton, _checked, _interegular, _unsupported

#: the keywords with a meaning in JSON Schema validation (every draft); any other keyword is an annotation
VALIDATION = frozenset({
    "type", "enum", "const", "multipleOf", "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum", "maxLength",
    "minLength", "pattern", "format", "items", "prefixItems", "additionalItems", "maxItems", "minItems", "uniqueItems",
    "contains", "maxContains", "minContains", "unevaluatedItems", "maxProperties", "minProperties", "required",
    "properties", "patternProperties", "additionalProperties", "dependentRequired", "dependentSchemas",
    "dependencies", "propertyNames", "unevaluatedProperties", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
    "$ref", "$dynamicRef", "$recursiveRef"})
#: any value (what a schema of annotations only is read as)
ANY_VALUE = {"anyOf": [{"type": t} for t in ("boolean", "null", "number", "integer", "string", "array", "object")]}
#: the formats read (:data:`.backend.json_schema.FORMATS`); others are refused
FORMATS = frozenset({"date-time", "date", "uuid"})
#: the keywords a ``type`` form reads, by type (a format beside another type than string: an annotation it ignores)
TYPE_KEYWORDS = {"string": {"minLength", "maxLength", "pattern", "format"}, "number": {"format"},
                 "integer": {"format"}, "boolean": {"format"}, "null": {"format"},
                 "array": {"items", "minItems", "maxItems", "format"},
                 "object": {"additionalProperties", "minProperties", "maxProperties", "format"}}


class _Refused(Exception):
    pass


def _json_types(value) -> set:
    if value is None:
        return {"null"}
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int):
        return {"integer", "number"}
    if isinstance(value, float):
        return {"number"} | ({"integer"} if value.is_integer() else set())
    if isinstance(value, str):
        return {"string"}
    return {"array"} if isinstance(value, list) else {"object"}


#: what may not stand unescaped in a JSON string: control characters (and U+007F-U+009F, as the converter has it),
#: the quote, the backslash -- inclusive code point ranges
JSON_UNSAFE = ((0x00, 0x1F), (0x22, 0x22), (0x5C, 0x5C), (0x7F, 0x9F))


def _matches_any(fsm, ranges, chars) -> bool:
    """Whether some string the character automaton accepts holds a character of ``chars`` (code point ranges): a
    transition on such a character from a reachable state into one that still reaches acceptance."""
    reach, stack = {fsm.initial}, [fsm.initial]
    while stack:
        for nxt in fsm.map.get(stack.pop(), {}).values():
            if nxt not in reach:
                reach.add(nxt)
                stack.append(nxt)
    live = set(fsm.finals)
    while True:
        more = {q for q in reach if q not in live and any(n in live for n in fsm.map.get(q, {}).values())}
        if not more:
            break
        live |= more
    hits = lambda rs: any(a <= d and c <= b for a, b in rs for c, d in chars)
    return any(n in live and hits(ranges.get(sym, ()))
               for q in reach for sym, n in fsm.map.get(q, {}).items())


def _pattern(pattern: str, where: str) -> str:
    """A string's pattern as the converter is to get it: anchors stripped, in a group; refused when it can match a
    quote, a backslash or a control character, or uses what Regex does not read."""
    q = pattern[1:] if pattern.startswith("^") else pattern
    if q.endswith("$") and not q.endswith("\\$"):
        q = q[:-1]
    try:
        why = _unsupported(q)
    except Exception as e:                      # (not a pattern Python's `re` reads)
        why = f"{type(e).__name__}: {e}"
    if why is not None:
        raise _Refused(f"{where}: pattern {pattern!r}: {why}")
    interegular, _ = _interegular()
    try:
        fsm, ranges = _char_automaton(interegular.parse_pattern(q))
    except Exception as e:
        raise _Refused(f"{where}: pattern {pattern!r}: {type(e).__name__}: {e}") from None
    if _matches_any(fsm, ranges, JSON_UNSAFE):
        raise _Refused(f"{where}: pattern {pattern!r} can match a quote, a backslash or a control character")
    return f"(?:{q})"


def _prepare(schema: Dict[str, Any]) -> Dict[str, Any]:
    """The schema as the converter is to get it (a copy: patterns rewritten, objects of annotations only emptied), or
    :class:`ValueError` with where and why it is refused (see the module docstring)."""
    root = copy.deepcopy(schema)
    rewritten, sound = set(), set()

    def resolve(ref: str, where: str):
        if not ref.startswith("#"):
            raise _Refused(f"{where}: an external $ref ({ref})")
        node = root
        for part in [p for p in ref[1:].split("/") if p]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise _Refused(f"{where}: a $ref that does not resolve ({ref})")
            node = node[part]
        return node

    def walk(s, where: str, active: frozenset):
        if not isinstance(s, dict):
            raise _Refused(f"{where}: {'a boolean schema' if isinstance(s, bool) else 'not a schema object'}")
        keys = set(s) & VALIDATION
        if not keys:                            # annotations only: any value, as ANY_VALUE (see there)
            s.clear()
            s.update(copy.deepcopy(ANY_VALUE))
            return
        if "properties" in keys:
            extra = keys - {"properties", "required", "type", "additionalProperties"}
            if extra:
                raise _Refused(f"{where}: {sorted(extra)} beside properties")
            t = s.get("type", "object")
            if "object" not in (t if isinstance(t, list) else [t]):
                raise _Refused(f"{where}: properties of type {t}")
            missing = set(s.get("required", [])) - set(s["properties"])
            if missing:
                raise _Refused(f"{where}: required {sorted(missing)} not among the properties")
            for k, sub in s["properties"].items():
                walk(sub, f"{where}/properties/{k}", active)
            return
        for form in ("allOf", "anyOf", "oneOf"):
            if form in keys:
                if keys != {form}:
                    raise _Refused(f"{where}: {sorted(keys - {form})} beside {form}")
                if not isinstance(s[form], list) or not s[form]:
                    raise _Refused(f"{where}: {form} of no schema")
                for i, sub in enumerate(s[form]):
                    walk(sub, f"{where}/{form}/{i}", active)
                return
        if "not" in keys:
            if keys != {"not"}:
                raise _Refused(f"{where}: {sorted(keys - {'not'})} beside not")
            walk(s["not"], f"{where}/not", active)
            return
        if keys & {"if", "then", "else"}:
            if not keys <= {"if", "then", "else"}:
                raise _Refused(f"{where}: {sorted(keys - {'if', 'then', 'else'})} beside if/then/else")
            if "if" not in keys:                # (then and else without if: no constraint)
                s.clear()
                s.update(copy.deepcopy(ANY_VALUE))
                return
            for k in ("if", "then", "else"):
                if k in s:
                    walk(s[k], f"{where}/{k}", active)
            return
        if "prefixItems" in keys:
            extra = keys - {"prefixItems", "type", "items"}
            if extra:
                raise _Refused(f"{where}: {sorted(extra)} beside prefixItems")
            for i, sub in enumerate(s["prefixItems"]):
                walk(sub, f"{where}/prefixItems/{i}", active)
            return
        if "enum" in keys or "const" in keys:
            extra = keys - {"enum", "const", "type"}
            if extra or {"enum", "const"} <= keys:
                raise _Refused(f"{where}: {sorted(keys)} together")
            values = s["enum"] if "enum" in keys else [s["const"]]
            if "type" in s:
                types = set(s["type"] if isinstance(s["type"], list) else [s["type"]])
                if any(not (_json_types(v) & types) for v in values):
                    raise _Refused(f"{where}: an enum or const value outside type {s['type']}")
            return
        if "$ref" in keys:
            if keys != {"$ref"}:
                raise _Refused(f"{where}: {sorted(keys - {'$ref'})} beside $ref")
            ref = s["$ref"]
            if ref in active:
                raise _Refused(f"{where}: a recursive $ref ({ref})")
            if ref not in sound:
                walk(resolve(ref, where), ref, active | {ref})
                sound.add(ref)
            return
        if "type" in keys:
            types = s["type"] if isinstance(s["type"], list) else [s["type"]]
            if any(t not in TYPE_KEYWORDS for t in types):
                raise _Refused(f"{where}: type {s['type']}")
            extra = keys - {"type"} - set().union(*(TYPE_KEYWORDS[t] for t in types))
            if extra:
                raise _Refused(f"{where}: {sorted(extra)} for type {s['type']}")
            if "string" in types:
                string = keys & TYPE_KEYWORDS["string"]
                if bool(string & {"minLength", "maxLength"}) + ("pattern" in string) + ("format" in string) > 1:
                    raise _Refused(f"{where}: a string's {sorted(string)} together")
                if "format" in keys and s["format"] not in FORMATS:
                    raise _Refused(f"{where}: format {s['format']!r} (read: {', '.join(sorted(FORMATS))})")
            if "pattern" in keys and id(s) not in rewritten:
                s["pattern"] = _pattern(s["pattern"], where)
                rewritten.add(id(s))
            if "items" in keys:
                walk(s["items"], f"{where}/items", active)
            if s.get("minProperties", 0) > 1 or s.get("maxProperties", 0) > 0:
                raise _Refused(f"{where}: minProperties above 1 or maxProperties above 0 without properties (the "
                               f"converter reads only whether the object may be empty)")
            members = s.get("additionalProperties")
            if members is False:
                raise _Refused(f"{where}: additionalProperties false without properties")
            if isinstance(members, dict):
                walk(members, f"{where}/additionalProperties", active)
            return
        raise _Refused(f"{where}: {sorted(keys)} (no form the converter reads)")

    try:
        walk(root, "#", frozenset())
    except _Refused as e:
        raise ValueError(f"Unsupported JSON Schema: {e}.") from None
    return root


class JSONSchema(_Text):
    """
    Token sequences whose text is an instance of ``schema`` written as JSON in the canonical form (see the module
    docstring).

    :param schema: the schema, a dict or JSON text
    :type schema: Union[dict, str]

    :param vocabulary: the token texts: a :class:`~pyjuice.constraints.Vocabulary`, or a tokenizer

    :param whitespace: the regular expression of what may stand around ``{ } [ ] , :``
    :type whitespace: str

    :param eos_padding: the text may end early, then only end-of-sequence ids follow (default); False: every token
        spells the text
    :type eos_padding: bool
    """

    def __init__(self, schema: Union[Dict[str, Any], str], vocabulary, whitespace: str = r"[ ]?",
                 eos_padding: bool = True):
        if isinstance(schema, str):
            schema = json.loads(schema)
        if not isinstance(schema, dict):
            raise TypeError(f"`schema` must be a dict or JSON text of an object, got {type(schema).__name__}.")
        super().__init__(vocabulary, eos_padding)
        self._prepared = _prepare(schema)
        _checked(whitespace)
        self._schema = copy.deepcopy(schema)
        self._whitespace = whitespace
        self._bytes = None

    @property
    def schema(self) -> Dict[str, Any]:
        return copy.deepcopy(self._schema)

    @property
    def whitespace(self) -> str:
        return self._whitespace

    def _byte_automaton(self):
        """The canonical texts' minimal byte automaton, compiled on first use."""
        if self._bytes is None:
            from .backend.json_schema import compile_schema
            self._bytes = compile_schema(self._prepared, self._whitespace)
        return self._bytes

    def _text_matches(self, text: str) -> bool:
        return self._byte_automaton().accepts(text.encode("utf-8"))

    def _fingerprint_payload(self):
        a = self._byte_automaton()
        return (a.initial, a.accept.tobytes(), a.table.tobytes(), a.byte_class.tobytes(), self._vocabulary.fingerprint(),
                self._eos_padding)

    def info(self):
        info = Constraint.info(self)
        info.update(byte_states = self._byte_automaton().num_states, whitespace = self._whitespace,
                    eos_padding = self.eos_padding)
        return info
