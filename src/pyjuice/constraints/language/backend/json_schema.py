"""
JSON Schema to byte automata: the texts of a schema's instances in the CANONICAL form :class:`~pyjuice.constraints.
JSONSchema` reads (its module docstring), compiled through the automata algebra (:mod:`.automata`) -- no regular
expression of the whole schema is written. Each schema object is compiled once (a reference's target too) and its
minimal automaton reused wherever it is read.

The canonical form is outlines-core's (``build_regex_from_schema``), piece for piece: its strings, numbers and formats,
its whitespace around separators -- ``W`` below, the ``whitespace`` pattern: ``{W"name"W:Wvalue}`` with ``W,W`` between
members, ``[Witem,Witem]`` between an array's items (``W,W`` between a tuple's), literals of ``enum`` and ``const``
spaced the same way, an empty one ``[WW]`` or ``{WW}`` -- and free-form values nested at most :data:`FREE_DEPTH`
levels deep. A property name is written as a JSON string (outlines-core writes it as it is, quotes and backslashes
unescaped).

``allOf`` intersects its schemas: objects (and arrays) are merged member by member (a property's schemas, the union of
the required ones), anything else intersected as texts; an ``anyOf`` among them is distributed over. ``oneOf``,
``not`` and ``if``/``then``/``else`` SUBTRACT, which is sound only if what is subtracted holds every text of the other
schema's values: below such a combinator every schema object is compiled twice -- its texts ``A`` (those written: a
part of its instances', as everywhere) and ``B``, every text an ``A`` below the combinator may write that is an
instance -- and ``B`` is what is subtracted: ``oneOf`` is the union over its schemas of ``A_i`` minus the other ``B_j``,
``not X`` any value minus ``B(X)``, ``if I then T else E`` ``A(I and T)`` and ``A(E)`` minus ``B(I)``; ``B`` the same
with the roles swapped. To keep ``B`` exact where it can be, the texts below such a combinator are written in one
canonical form (:class:`_Canon`): property names in one order (first appearance there), every number and string one way
(no exponent, no trailing zero, ``0`` for zero; no ``\\/``), free-form objects only with names that no schema there
declares.
"""

from __future__ import annotations

import copy
import json
import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from . import automata as A
from .automata import _Joined
from .byte_level import ByteAutomaton

#: a string's character: any but a quote, a backslash, a control character or U+007F-U+009F, or an escape other than
#: ``\u``
STRING_INNER = r'(?:[^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])'
INTEGER = r"-?(?:0|[1-9][0-9]*)"
NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-][0-9]+)?"
FORMATS = {
    "date": r'"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])"',
    "date-time": r'"-?(?:[1-9][0-9]*)?[0-9]{4}-(?:1[0-2]|0[1-9])-(?:3[01]|0[1-9]|[12][0-9])T(?:2[0-3]|[01][0-9]):'
                 r'[0-5][0-9]:[0-5][0-9](?:\.[0-9]{3})?Z?"',
    "uuid": r'"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"',
}
#: how deep free-form values nest: an object or array without a schema for its members holds scalars and free-form
#: values one level less deep, down to scalars alone
FREE_DEPTH = 2

#: below a subtracting combinator: a string's character written one way (``/`` never escaped), a number one way (no
#: exponent, no trailing zero, ``0`` for zero)
CANON_STRING_INNER = r'(?:[^"\\\x00-\x1F\x7F-\x9F]|\\["\\bfnrt])'
CANON_INTEGER = r"0|-?[1-9][0-9]*"
CANON_NUMBER = r"0|-?[1-9][0-9]*|-?(?:0|[1-9][0-9]*)\.[0-9]*[1-9]"
#: a string's character in any text written below a subtracting combinator (a literal's ``\u`` escapes and characters
#: U+007F-U+009F among them): what ``B`` reads a string as
ANY_STRING_INNER = r'(?:[^"\\\x00-\x1F]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
#: the keywords a type reads (beside ``type``), to split a schema object of several types into one per type
TYPE_ONLY = {"object": {"properties", "required", "additionalProperties", "minProperties", "maxProperties"},
             "array": {"items", "minItems", "maxItems", "prefixItems"},
             "string": {"minLength", "maxLength", "pattern", "format"}}
#: any value, as :func:`~pyjuice.constraints.language.json_schema._prepare` writes a schema of annotations only
ANY_VALUE = {"anyOf": [{"type": t} for t in ("boolean", "null", "number", "integer", "string", "array", "object")]}

_LEAVES: Dict[str, ByteAutomaton] = {}


def _leaf(pattern: str) -> ByteAutomaton:
    """The automaton of a pattern every compilation reads (kept: they do not depend on the schema)."""
    hit = _LEAVES.get(pattern)
    if hit is None:
        hit = _LEAVES[pattern] = A.regex(pattern)
    return hit


def float_text(x: float) -> str:
    """A float as the canonical form writes it: the shortest digits that read back, positional for decimal
    exponents -5 to 15 (``.0`` if integral), else ``d.ddde+N`` / ``d.ddde-N``."""
    if x == 0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    sign, r = ("-" if x < 0 else ""), repr(abs(x))
    mant, _, exp = r.partition("e")
    whole, _, frac = mant.partition(".")
    digits = (whole + frac).lstrip("0")
    if whole.strip("0"):
        e10 = int(exp or 0) + len(whole.lstrip("0")) - 1
    else:
        e10 = int(exp or 0) - (len(frac) - len(frac.lstrip("0")) + 1)
    digits = digits.rstrip("0") or "0"
    if -5 <= e10 < 16:
        if e10 >= 0:
            return f"{sign}{digits[:e10 + 1].ljust(e10 + 1, '0')}.{digits[e10 + 1:] or '0'}"
        return f"{sign}0.{'0' * (-e10 - 1)}{digits}"
    mantissa = digits[0] + (f".{digits[1:]}" if len(digits) > 1 else "")
    return f"{sign}{mantissa}e{'+' if e10 > 0 else '-'}{abs(e10)}"


def canonical_number(x) -> str:
    """A number written one way per value (:data:`CANON_NUMBER`): positional, the shortest digits of a float that
    read back, no trailing zero, ``0`` for zero."""
    d = Decimal(x) if isinstance(x, int) else Decimal(repr(float(x)))
    if d == 0:
        return "0"
    text = format(d, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def compile_schema(schema: Dict[str, Any], whitespace: str) -> ByteAutomaton:
    """The minimal byte automaton of the canonical texts of ``schema``'s instances: ``schema`` as
    :func:`~pyjuice.constraints.language.json_schema._prepare` returns it (checked: one form per schema object, the
    keywords that form reads), ``whitespace`` the pattern of what may stand around separators."""
    return _Compiler(schema, whitespace).value(schema)


class _Canon:
    """
    The canonical form below a subtracting combinator (see the module docstring): every property name declared or
    written by a literal there (references followed), in order of first appearance -- objects write theirs in that
    order --, and the names of free-form objects' members: any other.
    """

    def __init__(self, compiler: "_Compiler", node: Dict[str, Any]):
        names: Dict[str, None] = {}
        seen = set()

        def literal_keys(v):
            if isinstance(v, dict):
                for k, x in v.items():
                    names.setdefault(k)
                    literal_keys(x)
            elif isinstance(v, list):
                for x in v:
                    literal_keys(x)

        def walk(s):
            if not isinstance(s, dict) or id(s) in seen:
                return
            seen.add(id(s))
            for k in s.get("properties", {}):
                names.setdefault(k)
            for sub in s.get("properties", {}).values():
                walk(sub)
            for key in ("items", "additionalProperties", "not", "if", "then", "else"):
                walk(s.get(key))
            for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
                for sub in s.get(key, ()):
                    walk(sub)
            for v in s.get("enum", ()):
                literal_keys(v)
            if "const" in s:
                literal_keys(s["const"])
            if isinstance(s.get("$ref"), str):
                walk(compiler.target(s["$ref"]))

        walk(node)
        self.root = node
        self.any_b = None
        self.names = list(names)
        self.rank = {k: i for i, k in enumerate(self.names)}
        self.free = None
        self.any = None
        self._other: Dict[str, ByteAutomaton] = {}

    def other(self, compiler: "_Compiler", mode: str) -> ByteAutomaton:
        """The names of free-form objects' members (``A``), or every text such a name may have (``B``): strings but
        those of the declared and written names."""
        if mode not in self._other:
            strings = compiler.canon_string if mode == "A" else compiler.any_string
            known = [compiler.lit(json.dumps(k, ensure_ascii = False)) for k in self.names]
            self._other[mode] = A.difference(strings, A.union(*known)) if known else strings
        return self._other[mode]


#: (:meth:`_Compiler._merge`) the schemas merged admit no value
_NOTHING = object()


class _Compiler:

    def __init__(self, root: Dict[str, Any], whitespace: str):
        self.root = root
        self.memo: Dict[Tuple, ByteAutomaton] = {}
        self.combos: Dict[Tuple, ByteAutomaton] = {}
        self.kept: List[Any] = []                   # objects made here: kept alive, so their ids stay theirs
        self.ws = _leaf(whitespace)
        self.string_inner, self.string = _leaf(STRING_INNER), _leaf(f'"{STRING_INNER}*"')
        self.integer, self.number = _leaf(INTEGER), _leaf(NUMBER)
        self.boolean, self.null = _leaf("true|false"), A.literal(b"null")
        self._free = None
        self._any = self.keep(copy.deepcopy(ANY_VALUE))

    def keep(self, x):
        self.kept.append(x)
        return x

    def __getattr__(self, name):                     # the canonical form's leaves, made on first use
        patterns = {"canon_inner": CANON_STRING_INNER, "canon_string": f'"{CANON_STRING_INNER}*"',
                    "canon_integer": CANON_INTEGER, "canon_number": CANON_NUMBER, "any_inner": ANY_STRING_INNER,
                    "any_string": f'"{ANY_STRING_INNER}*"'}
        if name in patterns:
            value = _leaf(patterns[name])
        elif name == "ws_any":
            value = A.star(self.ws)
        else:
            raise AttributeError(name)
        setattr(self, name, value)
        return value

    def target(self, ref: str) -> Dict[str, Any]:
        node = self.root
        for part in [p for p in ref[1:].split("/") if p]:
            node = node[part.replace("~1", "/").replace("~0", "~")]
        return node

    # ---------------------------------------------------------------------------------------------
    # Pieces
    # ---------------------------------------------------------------------------------------------

    @staticmethod
    def lit(text: str) -> ByteAutomaton:
        return A.literal(text.encode("utf-8"))

    def sep(self, mark: str, ws: Optional[ByteAutomaton] = None) -> ByteAutomaton:
        """``W<mark>W``."""
        ws = self.ws if ws is None else ws
        return A.concat(ws, self.lit(mark), ws)

    def object_of(self, value: ByteAutomaton, nonempty: bool, key: Optional[ByteAutomaton] = None) -> ByteAutomaton:
        """Objects of ``key`` names (default: any string) and ``value`` values (none: ``nonempty`` False)."""
        member = A.concat(self.string if key is None else key, self.sep(":"), value)
        body = A.concat(member, A.star(A.concat(self.sep(","), member)))
        return A.concat(self.lit("{"), self.ws, body if nonempty else A.optional(body), self.ws, self.lit("}"))

    def array_of(self, item: ByteAutomaton, lo: int, hi: Optional[int]) -> ByteAutomaton:
        """Arrays of ``lo`` to ``hi`` (None: any number of) ``item``s."""
        if hi == 0:
            return A.concat(self.lit("["), self.ws, self.lit("]"))
        rest = A.concat(self.lit(","), self.ws, item)
        body = A.concat(item, A.repeat(rest, max(lo - 1, 0), None if hi is None else hi - 1))
        return A.concat(self.lit("["), self.ws, body if lo >= 1 else A.optional(body), self.ws, self.lit("]"))

    def empty_object(self) -> ByteAutomaton:
        return A.concat(self.lit("{"), self.ws, self.lit("}"))

    def free(self):
        """``(object values, array items)`` of free-form values at :data:`FREE_DEPTH`."""
        if self._free is None:
            self._free = self._free_values(A.union(self.string, self.number, self.boolean, self.null),
                                           A.union(self.boolean, self.null, self.number, self.integer, self.string),
                                           None)
        return self._free

    def _free_values(self, values, items, key):
        v, i = A.force(values), A.force(items)                  # an object's and an array's, at depth 0
        for _ in range(FREE_DEPTH):                             # containers of the level below, then theirs
            obj, arr = A.force(self.object_of(v, False, key)), A.force(self.array_of(i, 0, None))
            v, i = A.force(A.union(values, obj, arr)), A.force(A.union(items, obj, arr))
        return v, i

    def free_canon(self, canon: _Canon):
        """:meth:`free` in the canonical form below a subtracting combinator (members only of other names)."""
        if canon.free is None:
            scalars = A.union(self.canon_string, self.canon_number, self.boolean, self.null)
            canon.free = self._free_values(scalars, scalars, canon.other(self, "A"))
        return canon.free

    def any_value(self, canon: _Canon) -> ByteAutomaton:
        """``B`` of any value: every JSON text -- strings and numbers as any text below the combinator writes them,
        any whitespace -- nested at most as deep as a text written below it may be (:meth:`depth`), and more: a
        bracket of either kind holds members (``"name": value``) or items, and either closes it, so that the automaton
        counts the nesting alone (exact brackets would cost 2 ** depth states). Structured still: a value's text is a
        scalar or a balanced bracket, never reaching past it, so objects and arrays of these stay deterministic."""
        if canon.any_b is None:
            ws = self.ws_any
            scalars = A.union(self.any_string, self.canon_number, self.boolean, self.null)
            opening, closing = A.byte_ranges([(0x5B, 0x5B), (0x7B, 0x7B)]), A.byte_ranges([(0x5D, 0x5D), (0x7D, 0x7D)])
            v = scalars
            for _ in range(self.depth(canon.root)):
                entry = A.concat(A.optional(A.concat(self.any_string, self.sep(":", ws))), v)
                inner = A.optional(A.concat(entry, A.star(A.concat(self.sep(",", ws), entry))))
                v = A.force(A.union(scalars, A.concat(opening, ws, inner, ws, closing)))
            canon.any_b = v
        return canon.any_b

    def depth(self, s: Dict[str, Any], memo: Optional[Dict[int, int]] = None) -> int:
        """The deepest nesting of brackets a text written for ``s`` may have (free-form values, and what a
        subtracting combinator writes of any value, :data:`FREE_DEPTH` + 1)."""
        memo = {} if memo is None else memo
        if id(s) in memo:
            return memo[id(s)]
        free = FREE_DEPTH + 1
        literal = lambda v: 1 + max([literal(x) for x in (v.values() if isinstance(v, dict) else v)], default = 0) \
            if isinstance(v, (list, dict)) else 0
        subs = [b for k in ("allOf", "anyOf", "oneOf", "prefixItems") for b in s.get(k, ())]
        subs += [s[k] for k in ("not", "if", "then", "else") if isinstance(s.get(k), dict)]
        if isinstance(s.get("$ref"), str):
            subs.append(self.target(s["$ref"]))
        d = max([self.depth(b, memo) for b in subs], default = 0)
        if s == ANY_VALUE or "not" in s or ("if" in s and "else" not in s):    # (these write any value)
            d = max(d, free)
        if "properties" in s:
            d = max(d, 1 + max([self.depth(v, memo) for v in s["properties"].values()], default = 0))
        if "prefixItems" in s:
            d = max(d, 1 + d)
        d = max([d] + [literal(v) for v in s.get("enum", ())] + ([literal(s["const"])] if "const" in s else []))
        types = s.get("type", [])
        types = types if isinstance(types, list) else [types]
        if "array" in types:
            d = max(d, 1 + self.depth(s["items"], memo) if "items" in s else free)
        if "object" in types and "properties" not in s:
            members = s.get("additionalProperties")
            d = max(d, 1 + self.depth(members, memo) if isinstance(members, dict) else free)
        memo[id(s)] = d
        return d

    def anything(self, mode: str, canon: Optional[_Canon]) -> ByteAutomaton:
        """Any value: free-form (``A``), or every text (``B``)."""
        if canon is None:
            return self.value(self._any)
        if mode == "B":
            return self.any_value(canon)
        if canon.any is None:
            v, i = self.free_canon(canon)
            canon.any = A.force(A.union(self.canon_string, self.canon_number, self.boolean, self.null,
                                        self.object_of(v, False, canon.other(self, "A")), self.array_of(i, 0, None)))
        return canon.any

    def literal_value(self, v: Any) -> ByteAutomaton:
        """An ``enum`` or ``const`` value, written in the canonical form."""
        if v is None or isinstance(v, bool):
            return self.lit("null" if v is None else ("true" if v else "false"))
        if isinstance(v, int):
            return self.lit(str(v))
        if isinstance(v, float):
            return self.lit(float_text(v))
        if isinstance(v, str):
            return self.lit(json.dumps(v, ensure_ascii = False))
        if isinstance(v, list):
            parts: List[ByteAutomaton] = []
            for k, x in enumerate(v):
                parts += ([self.sep(",")] if k else []) + [self.literal_value(x)]
            return A.concat(self.lit("["), self.ws, *parts, self.ws, self.lit("]"))
        if isinstance(v, dict):
            parts = []
            for k, (name, x) in enumerate(v.items()):
                parts += ([self.sep(",")] if k else []) + [self.lit(json.dumps(name, ensure_ascii = False)),
                                                            self.sep(":"), self.literal_value(x)]
            return A.concat(self.lit("{"), self.ws, *parts, self.ws, self.lit("}"))
        raise TypeError(f"Not a JSON value: {type(v).__name__}.")

    def canon_literal(self, v: Any, mode: str, canon: _Canon) -> ByteAutomaton:
        """:meth:`literal_value` below a subtracting combinator: numbers one way, names in the canonical order, and
        (``B``) any whitespace where some may stand."""
        ws = self.ws if mode == "A" else self.ws_any
        if v is None or isinstance(v, bool) or isinstance(v, str):
            return self.literal_value(v)
        if isinstance(v, (int, float)):
            return self.lit(canonical_number(v))
        parts: List[ByteAutomaton] = []
        if isinstance(v, list):
            for k, x in enumerate(v):
                parts += ([self.sep(",", ws)] if k else []) + [self.canon_literal(x, mode, canon)]
            return A.concat(self.lit("["), ws, *parts, ws, self.lit("]"))
        if isinstance(v, dict):
            for k, name in enumerate(sorted(v, key = canon.rank.__getitem__)):
                parts += ([self.sep(",", ws)] if k else []) + [self.lit(json.dumps(name, ensure_ascii = False)),
                                                                self.sep(":", ws),
                                                                self.canon_literal(v[name], mode, canon)]
            return A.concat(self.lit("{"), ws, *parts, ws, self.lit("}"))
        raise TypeError(f"Not a JSON value: {type(v).__name__}.")

    # ---------------------------------------------------------------------------------------------
    # Schema objects
    # ---------------------------------------------------------------------------------------------

    def value(self, s: Dict[str, Any], mode: str = "A", canon: Optional[_Canon] = None) -> ByteAutomaton:
        """``s``'s texts: ``A`` (written) or ``B`` (subtracted: below a subtracting combinator only), in the
        canonical form below the combinator ``canon`` is of (None: above every one)."""
        key = (id(s), mode, None if canon is None else id(canon))
        hit = self.memo.get(key)
        if hit is None:
            if canon is None and self._subtracts(s):                # a subtracting combinator: its own form
                hit = self.value(s, "A", self.keep(_Canon(self, s)))
            elif canon is None:
                hit = A.force(self._parity(s))
            else:
                hit = A.force(self._value(s, mode, canon))
            self.memo[key] = hit
        return hit

    def _subtracts(self, s: Dict[str, Any]) -> bool:
        """Whether ``s`` subtracts: a ``oneOf``, ``not`` or ``if``, or an ``allOf`` with one among its schemas
        (through ``allOf``, ``anyOf`` and references)."""
        if any(k in s for k in ("oneOf", "not", "if")):
            return True
        if "$ref" in s:
            return False
        branches = s.get("allOf", ())
        stack = list(branches)
        while stack:
            b = stack.pop()
            if "$ref" in b:
                b = self.target(b["$ref"])
            if any(k in b for k in ("oneOf", "not", "if")):
                return True
            stack += b.get("allOf", []) + b.get("anyOf", [])
        return False

    def _parity(self, s: Dict[str, Any]) -> ByteAutomaton:
        """``s``'s texts above every subtracting combinator: outlines-core's."""
        if "properties" in s:
            props, required = s["properties"], set(s.get("required", ()))
            if s.get("maxProperties") == 0:
                return A.nothing() if required else self.empty_object()
            return self._members([(k, self.value(v), k in required) for k, v in props.items()],
                                 s.get("minProperties", 0) >= 1)
        if "allOf" in s:
            return self._all([s], "A", None)
        if "anyOf" in s:
            return A.union(*[self.value(b) for b in s["anyOf"]])
        if "prefixItems" in s:
            parts: List[ByteAutomaton] = []
            for k, b in enumerate(s["prefixItems"]):
                parts += ([self.sep(",")] if k else []) + [self.value(b)]
            return A.concat(self.lit("["), self.ws, *parts, self.ws, self.lit("]"))
        if "enum" in s:
            return A.union(*[self.literal_value(v) for v in s["enum"]])
        if "const" in s:
            return self.literal_value(s["const"])
        if "$ref" in s:
            return self.value(self.target(s["$ref"]))
        types = s["type"] if isinstance(s["type"], list) else [s["type"]]
        return A.union(*[self._typed(t, s) for t in types])

    def _typed(self, t: str, s: Dict[str, Any]) -> ByteAutomaton:
        if t == "string":
            if "pattern" in s:
                return A.concat(self.lit('"'), A.regex(s["pattern"]), self.lit('"'))
            if s.get("format") in FORMATS:
                return _leaf(FORMATS[s["format"]])
            lo, hi = s.get("minLength", 0), s.get("maxLength")
            if lo == 0 and hi is None:
                return self.string
            return A.concat(self.lit('"'), A.repeat(self.string_inner, lo, hi), self.lit('"'))
        if t in ("integer", "number", "boolean", "null"):
            return getattr(self, t)
        if t == "array":
            item = self.value(s["items"]) if "items" in s else self.free()[1]
            return self.array_of(item, s.get("minItems", 0), s.get("maxItems"))
        # an object: a map of its additionalProperties' values, or free-form; empty or not
        members = s.get("additionalProperties")
        if s.get("maxProperties") == 0:
            return self.empty_object()
        if members is False:                                    # (an allOf's objects merged)
            return A.nothing() if s.get("minProperties", 0) >= 1 else self.empty_object()
        value = self.value(members) if isinstance(members, dict) else self.free()[0]
        return self.object_of(value, s.get("minProperties", 0) >= 1)

    def _members(self, items, nonempty: bool) -> ByteAutomaton:
        """``{W"a"W:Wvalue W,W ...W}``: ``items`` -- ``(name, value automaton, required)`` -- in order, the optional
        ones may be left out (not all if ``nonempty``). Two tracks of states between the members -- nothing written
        yet, or something (the next member then follows ``W,``) -- through one copy of every value."""
        j = _Joined()
        k = len(items)
        none, some = [j.fresh() for _ in range(k + 1)], [j.fresh() for _ in range(k + 1)]

        def enter(x, sources, targets):
            entry, accepting = j.wire(x)
            j.eps.extend((u, entry) for u in sources)
            j.eps.extend((q, t) for q in accepting for t in targets)
            return entry, accepting

        start, _ = enter(self.lit("{"), [], [none[0]])
        for i, (name, value, required) in enumerate(items):
            first = A.concat(self.ws, self.lit(json.dumps(name, ensure_ascii = False)), self.sep(":"))
            later = A.concat(self.ws, self.lit(","), first)
            v, done = j.wire(value)
            enter(first, [none[i]], [v])
            enter(later, [some[i]], [v])
            j.eps.extend((q, some[i + 1]) for q in done)
            if not required:
                j.eps += [(none[i], none[i + 1]), (some[i], some[i + 1])]
        empty = not nonempty and not any(r for _, _, r in items)
        _, j.accepting = enter(A.concat(self.ws, self.lit("}")), [some[k]] + ([none[k]] if empty else []), [])
        return j.build(start)

    # ---------------------------------------------------------------------------------------------
    # Below a subtracting combinator
    # ---------------------------------------------------------------------------------------------

    def _value(self, s: Dict[str, Any], mode: str, canon: _Canon) -> ByteAutomaton:
        if s == ANY_VALUE:
            return self.anything(mode, canon)
        if any(k in s for k in ("allOf", "anyOf", "oneOf", "not", "if")) or \
                (isinstance(s.get("type"), list) and len(s["type"]) > 1):
            return self._all([s], mode, canon)
        if "properties" in s or "prefixItems" in s:
            out = self._object(s, mode, canon) if "properties" in s else self._tuple(s, mode, canon)
            if mode == "B" and "type" not in s:                 # (other types' values are instances too)
                out = A.union(out, self._others("object" if "properties" in s else "array", canon))
            return out
        if "enum" in s:
            return A.union(*[self.canon_literal(v, mode, canon) for v in s["enum"]])
        if "const" in s:
            return self.canon_literal(s["const"], mode, canon)
        if "$ref" in s:
            return self.value(self.target(s["$ref"]), mode, canon)
        t = s["type"][0] if isinstance(s["type"], list) else s["type"]
        return self._canon_typed(t, s, mode, canon)

    def _canon_typed(self, t: str, s: Dict[str, Any], mode: str, canon: _Canon) -> ByteAutomaton:
        if t == "string":
            if "pattern" in s:                                  # (B: any string)
                return A.concat(self.lit('"'), A.regex(s["pattern"]), self.lit('"')) if mode == "A" \
                    else self.any_string
            if s.get("format") in FORMATS:
                return _leaf(FORMATS[s["format"]])
            lo, hi = s.get("minLength", 0), s.get("maxLength")
            if lo == 0 and hi is None:
                return self.canon_string if mode == "A" else self.any_string
            inner = self.canon_inner if mode == "A" else self.any_inner
            return A.concat(self.lit('"'), A.repeat(inner, lo, hi), self.lit('"'))
        if t == "integer":
            return self.canon_integer
        if t == "number":
            return self.canon_number
        if t in ("boolean", "null"):
            return getattr(self, t)
        lo, hi = s.get("minItems", 0), s.get("maxItems")
        if t == "array" and mode == "A":
            return self.array_of(self.value(s["items"], "A", canon) if "items" in s else self.free_canon(canon)[1],
                                 lo, hi)
        if t == "array":
            return self._array_b(self.value(s["items"], "B", canon) if "items" in s else self.any_value(canon), lo, hi)
        members = s.get("additionalProperties")
        min1, max0 = s.get("minProperties", 0) >= 1, s.get("maxProperties") == 0
        if mode == "B":
            return self._object_b({}, set(), members, min1, max0, canon)
        if max0 or members is False:
            return A.nothing() if min1 else self.empty_object()
        value = self.value(members, "A", canon) if isinstance(members, dict) else self.free_canon(canon)[0]
        return self.object_of(value, min1, canon.other(self, "A"))

    def _object(self, s: Dict[str, Any], mode: str, canon: _Canon) -> ByteAutomaton:
        """An object of ``properties``: ``A`` writes those alone, in the canonical order; ``B`` every name, those not
        among them as ``additionalProperties`` has it (its schema read as any value: unchecked beside properties)."""
        props, required = s["properties"], set(s.get("required", ()))
        min1, max0 = s.get("minProperties", 0) >= 1, s.get("maxProperties") == 0
        if mode == "B":
            members = s.get("additionalProperties")
            return self._object_b(props, required, None if isinstance(members, dict) else members, min1, max0, canon)
        if max0:
            return A.nothing() if required else self.empty_object()
        names = sorted(props, key = canon.rank.__getitem__)
        return self._members([(k, self.value(props[k], "A", canon), k in required) for k in names], min1)

    def _object_b(self, props, required, members, min1: bool, max0: bool, canon: _Canon) -> ByteAutomaton:
        """``B`` of an object: the declared and written names in the canonical order -- ``props``' with their values,
        the others as ``members`` (``additionalProperties``: False none, a schema its values, else any) -- then
        members of other names in any order (as ``members``); ``required`` among them; any whitespace."""
        ws = self.ws_any
        if max0:
            return A.nothing() if required else A.concat(self.lit("{"), ws, self.lit("}"))
        extra = None if members is False else \
            (self.value(members, "B", canon) if isinstance(members, dict) else self.any_value(canon))
        entries = [(k, self.value(props[k], "B", canon) if k in props else extra) for k in canon.names
                   if k in props or extra is not None]
        j = _Joined()
        n = len(entries)
        none, some, tail = [j.fresh() for _ in range(n + 1)], [j.fresh() for _ in range(n + 1)], j.fresh()

        def enter(x, sources, targets):
            entry, accepting = j.wire(x)
            j.eps.extend((u, entry) for u in sources)
            j.eps.extend((q, t) for q in accepting for t in targets)
            return entry, accepting

        start, _ = enter(self.lit("{"), [], [none[0]])
        for i, (name, value) in enumerate(entries):
            first = A.concat(ws, self.lit(json.dumps(name, ensure_ascii = False)), self.sep(":", ws))
            later = A.concat(ws, self.lit(","), first)
            v, done = j.wire(value)
            enter(first, [none[i]], [v])
            enter(later, [some[i]], [v])
            j.eps.extend((q, some[i + 1]) for q in done)
            if name not in required:
                j.eps += [(none[i], none[i + 1]), (some[i], some[i + 1])]
        if extra is not None:                                   # members of the other names, after them
            member = A.force(A.concat(canon.other(self, "B"), self.sep(":", ws), extra))
            enter(A.concat(ws, member), [none[n]], [tail])
            enter(A.concat(self.sep(",", ws), member), [some[n], tail], [tail])
        empty = not (min1 or required)
        _, j.accepting = enter(A.concat(ws, self.lit("}")), [some[n], tail] + ([none[n]] if empty else []), [])
        return j.build(start)

    def _others(self, kind: str, canon: _Canon) -> ByteAutomaton:
        """``B`` of every value but objects (``kind`` "object") or arrays ("array"): what a ``properties`` or a
        ``prefixItems`` without a ``type`` admits besides (JSON Schema's keywords of a type hold for other types)."""
        key = (kind, id(canon))
        if key not in self.combos:
            pieces = [self.any_string, self.canon_number, self.boolean, self.null]
            pieces.append(self._array_b(self.any_value(canon), 0, None) if kind == "object" else
                          self._object_b({}, set(), None, False, False, canon))
            self.combos[key] = A.force(A.union(*pieces))
        return self.combos[key]

    def _array_b(self, item: ByteAutomaton, lo: int, hi: Optional[int]) -> ByteAutomaton:
        """``B`` of an array of ``lo`` to ``hi`` ``item``s: any whitespace."""
        ws = self.ws_any
        if hi == 0:
            return A.concat(self.lit("["), ws, self.lit("]"))
        body = A.concat(item, A.repeat(A.concat(self.sep(",", ws), item), max(lo - 1, 0),
                                       None if hi is None else hi - 1))
        return A.concat(self.lit("["), ws, body if lo >= 1 else A.optional(body), ws, self.lit("]"))

    def _tuple(self, s: Dict[str, Any], mode: str, canon: _Canon) -> ByteAutomaton:
        """``prefixItems``: ``A`` those items, as outlines-core reads them; ``B`` any of their prefixes, then any items
        (``items`` beside them is unchecked: any value)."""
        ws = self.ws if mode == "A" else self.ws_any
        items = [self.value(b, mode, canon) for b in s["prefixItems"]]
        if mode == "A":
            parts: List[ByteAutomaton] = []
            for k, a in enumerate(items):
                parts += ([self.sep(",")] if k else []) + [a]
            return A.concat(self.lit("["), ws, *parts, ws, self.lit("]"))
        tail = A.star(A.concat(self.sep(",", ws), self.any_value(canon)))
        body = A.concat(items[-1] if items else self.any_value(canon), tail)
        for a in reversed(items[:-1]):
            body = A.concat(a, A.optional(A.concat(self.sep(",", ws), body)))
        return A.concat(self.lit("["), ws, A.optional(body), ws, self.lit("]"))

    # ---------------------------------------------------------------------------------------------
    # allOf, and the combinators below it
    # ---------------------------------------------------------------------------------------------

    def _parts(self, s: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The schemas ``s`` intersects: ``allOf``'s (flattened), references followed, any value left out."""
        if "allOf" in s:
            return [p for b in s["allOf"] for p in self._parts(b)]
        if "$ref" in s:
            return self._parts(self.target(s["$ref"]))
        return [] if s == ANY_VALUE else [s]

    def _all(self, parts, mode: str, canon: Optional[_Canon]) -> ByteAutomaton:
        """The texts of the values every one of ``parts`` admits (``A`` or ``B``)."""
        flat = [p for s in parts for p in self._parts(s)]
        key = (tuple(id(p) for p in flat), mode, None if canon is None else id(canon))
        hit = self.combos.get(key)
        if hit is None:
            hit = self.combos[key] = A.force(self._all_flat(flat, mode, canon))
        return hit

    def _all_flat(self, parts, mode: str, canon: Optional[_Canon]) -> ByteAutomaton:
        other = "B" if mode == "A" else "A"
        for i, p in enumerate(parts):
            rest = parts[:i] + parts[i + 1:]
            if "anyOf" in p:
                return A.union(*[self._all(rest + [b], mode, canon) for b in p["anyOf"]])
            if isinstance(p.get("type"), list) and len(p["type"]) > 1:
                return A.union(*[self._all(rest + [q], mode, canon) for q in self._split(p)])
            if "oneOf" in p:                                    # exactly one: each minus the others (B / A)
                out = []
                for j, b in enumerate(p["oneOf"]):
                    others = [self.value(c, other, canon) for k, c in enumerate(p["oneOf"]) if k != j]
                    mine = self._all(rest + [b], mode, canon)
                    out.append(A.difference(mine, A.union(*others)) if others else mine)
                return A.union(*out)
            if "not" in p:
                return A.difference(self._all(rest, mode, canon), self.value(p["not"], other, canon))
            if "if" in p:
                yes = self._all(rest + [p["if"]] + [p[k] for k in ("then",) if k in p], mode, canon)
                no = self._all(rest + [p[k] for k in ("else",) if k in p], mode, canon)
                return A.union(yes, A.difference(no, self.value(p["if"], other, canon)))
        if not parts:
            return self.anything(mode, canon)
        if len(parts) == 1:
            return self.value(parts[0], mode, canon)
        merged = self._merge(parts, mode)
        if merged is _NOTHING:
            return A.nothing()
        if merged is not None:
            return self.value(merged, mode, canon)
        return A.intersect(*[self.value(p, mode, canon) for p in parts])

    def _split(self, s: Dict[str, Any]) -> List[Dict[str, Any]]:
        """A schema object of several types as one per type (each with the keywords it reads)."""
        others = set().union(*TYPE_ONLY.values())
        return [self.keep(dict({k: v for k, v in s.items() if k not in others or k in TYPE_ONLY.get(t, ())},
                               type = t)) for t in s["type"]]

    def _merge(self, parts, mode: str):
        """``allOf`` of objects (or of arrays of items) as one: a property's schemas intersected, the required ones
        united, a name one object closes left out (no value if required), members of other names as every object
        admits them; arrays' items intersected and their bounds tightened. None for other schemas, ``_NOTHING`` when
        the objects admit none. A properties' ``additionalProperties`` schema is unchecked (outlines-core does not read
        it): ``A`` takes it as false, ``B`` as any value."""
        if all("properties" in p or p.get("type") == "object" for p in parts):
            names: Dict[str, None] = {}
            for p in parts:
                for k in p.get("properties", {}):
                    names.setdefault(k)
            required = set().union(*(p.get("required", ()) for p in parts))

            def members(p):
                m = p.get("additionalProperties")
                if "properties" in p and isinstance(m, dict):
                    return False if mode == "A" else None
                return m

            props = {}
            for k in names:
                subs, allowed = [], True
                for p in parts:
                    m = members(p)
                    if k in p.get("properties", {}):
                        subs.append(p["properties"][k])
                    elif m is False:
                        allowed = False
                    elif isinstance(m, dict):
                        subs.append(m)
                if not allowed:
                    if k in required:
                        return _NOTHING
                    continue
                props[k] = (subs[0] if len(subs) == 1 else self.keep({"allOf": subs})) if subs else \
                    self.keep(copy.deepcopy(ANY_VALUE))
            ms = [members(p) for p in parts]
            dicts = [m for m in ms if isinstance(m, dict)]
            merged: Dict[str, Any] = {}
            if any(p.get("type") == "object" for p in parts):   # (else other types' values are instances too)
                merged["type"] = "object"
            if any(m is False for m in ms):
                merged["additionalProperties"] = False
            elif dicts:
                merged["additionalProperties"] = dicts[0] if len(dicts) == 1 else self.keep({"allOf": dicts})
            if any("properties" in p for p in parts):
                merged["properties"], merged["required"] = props, [k for k in props if k in required]
            mins = [p["minProperties"] for p in parts if "minProperties" in p]
            maxs = [p["maxProperties"] for p in parts if "maxProperties" in p]
            if mins:
                merged["minProperties"] = max(mins)
            if maxs:
                merged["maxProperties"] = min(maxs)
            return self.keep(merged)
        if all(p.get("type") == "array" and "prefixItems" not in p for p in parts):
            items = [p["items"] for p in parts if "items" in p]
            merged = {"type": "array", "minItems": max(p.get("minItems", 0) for p in parts)}
            maxs = [p["maxItems"] for p in parts if "maxItems" in p]
            if maxs:
                merged["maxItems"] = min(maxs)
            if items:
                merged["items"] = items[0] if len(items) == 1 else self.keep({"allOf": items})
            if merged.get("maxItems", merged["minItems"]) < merged["minItems"]:
                return _NOTHING
            return self.keep(merged)
        return None
