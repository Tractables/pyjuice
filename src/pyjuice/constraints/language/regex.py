"""
Regular expressions over a token sequence's text.

:class:`Regex` is the set of token sequences whose TEXT -- the tokens' bytes concatenated (see
:class:`~pyjuice.constraints.Vocabulary`), valid UTF-8 -- matches a pattern in full, as Python's
``re.fullmatch(pattern, text, re.ASCII)`` does: ASCII character classes (``\\d`` is ``[0-9]``, ``\\w`` is
``[A-Za-z0-9_]``, ``\\s`` ASCII whitespace), ASCII case folding under ``(?i)``. A pattern may hold literals, ``.``,
character classes, alternatives, groups (``(?i)`` and ``(?s)`` flags included) and greedy or lazy repetition; anchors
(``^``, ``$``, ``\\A``, ``\\Z``) are never needed -- the whole text always matches -- and are refused, with ``\\b``,
lookarounds, backreferences, possessive repetition, atomic groups, ``(?m)``, ``(?x)``, flags turned off (``(?-i:...)``),
``(?i)`` with a character outside ASCII, and a ``]`` first in a character class (write ``\\]``). Every tokenization of
a matching text is in the language, tokens that split a character included.

The PC's n positions hold the text followed by end-of-sequence ids (``eos_padding``, the default: a text may end before
the last position, and then only end-of-sequence ids follow), or the text alone. The automaton -- interegular's
automaton over characters, its UTF-8 byte automaton, then the token automaton (:mod:`.byte_level`) -- is built on first
use. Needs the ``interegular`` package (``pip install pyjuice[constraints]``).
"""

from __future__ import annotations

import re
from typing import Optional

import torch

from .base import Constraint, Matcher, Tokens
from .byte_level import ByteAutomaton, code_point_ranges, complement_ranges, to_tokens, widened
from .dfa import DFA
from .vocabulary import Vocabulary


def _interegular():
    """The ``interegular`` module and the errors its parser and automaton construction raise."""
    try:
        import interegular
        from interegular.patterns import InvalidSyntax, Unsupported
    except ImportError as e:
        raise ImportError("`jc.Regex` needs the `interegular` package: pip install pyjuice[constraints] "
                          "(or pip install interegular).") from e
    return interegular, (Unsupported, InvalidSyntax)


def _unsupported(pattern: str) -> Optional[str]:
    """
    Why ``pattern`` is outside what Regex reads exactly, or None. Python's own parse of the pattern (the semantics
    promised) may hold only literals, ``.``, character classes of characters, ranges and categories, alternatives,
    groups and greedy or lazy repetition -- what interegular's automata read as ``re`` does; refused are anchors and
    ``\\b``, lookarounds (interegular gets some wrong: ``(?=a)[abc]*`` accepting "aA"), backreferences, possessive
    repetition and atomic groups (``a++a`` accepting "aa"), ``(?m)`` and ``(?x)``, a flag turned off (``(?-i:a)`` in
    a ``(?i:...)`` matching "A"), ``(?i)`` with a character outside ASCII (interegular folds those by Unicode, and "ß"
    upper-cases to two characters), and a ``]`` first in a character class (which interegular reads otherwise).
    """
    try:
        from re import _constants as C, _parser as sre_parse           # Python >= 3.11
    except ImportError:                                                  # pragma: no cover
        import sre_constants as C
        import sre_parse
    tree = sre_parse.parse(pattern, re.ASCII)
    flags = tree.state.flags
    ignorecase, non_ascii = bool(flags & re.IGNORECASE), False
    if flags & (re.MULTILINE | re.VERBOSE):
        return "the (?m) or (?x) flag"
    special = {getattr(C, name, None): why for name, why in (
        ("AT", "an anchor or \\b"), ("ASSERT", "a lookahead or lookbehind"), ("ASSERT_NOT", "a lookahead or lookbehind"),
        ("GROUPREF", "a backreference"), ("GROUPREF_EXISTS", "a conditional backreference"),
        ("POSSESSIVE_REPEAT", "a possessive repetition"), ("ATOMIC_GROUP", "an atomic group"))}
    stack = [tree]
    while stack:
        for op, av in stack.pop().data:
            if op in (C.LITERAL, C.NOT_LITERAL):
                non_ascii |= av > 0x7F
            elif op is C.IN:
                for item, value in av:
                    if item in (C.LITERAL, C.NOT_LITERAL):
                        non_ascii |= value > 0x7F
                    elif item is C.RANGE:
                        non_ascii |= value[1] > 0x7F
                    elif item not in (C.CATEGORY, C.NEGATE):
                        return f"{item} in a character class"
            elif op is C.BRANCH:
                stack += av[1]
            elif op is C.SUBPATTERN:
                if av[1] & (re.MULTILINE | re.VERBOSE):
                    return "the (?m) or (?x) flag"
                if av[2]:
                    return "a flag turned off, as in (?-i:...)"
                ignorecase |= bool(av[1] & re.IGNORECASE)
                stack.append(av[3])
            elif op in (C.MAX_REPEAT, C.MIN_REPEAT):
                stack.append(av[2])
            elif op is not C.ANY:
                return special.get(op, f"{op}")
    if ignorecase and non_ascii:
        return "(?i) with a character outside ASCII"
    i, inside = 0, False                                               # a ']' first in a class
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if not inside and ch == "[":
            inside, i = True, i + 1
            if pattern[i:i + 1] == "^":
                i += 1
            if pattern[i:i + 1] == "]":
                return "a ']' first in a character class (escape it: \\])"
            continue
        if inside and ch == "]":
            inside = False
        i += 1
    return None


def _char_automaton(parsed):
    """interegular's automaton over characters of a parsed pattern, reduced, and every symbol's characters as code
    point ranges (its "anything else" symbol: every character no other symbol names)."""
    from interegular.fsm import anything_else
    fsm = parsed.to_fsm().reduce()
    groups = fsm.alphabet.by_transition                             # symbol -> its characters (or "anything else")
    rest = complement_ranges(code_point_ranges(ch for chars in groups.values() for ch in chars
                                               if ch is not anything_else))
    ranges = {}
    for symbol, chars in groups.items():
        own = code_point_ranges(ch for ch in chars if ch is not anything_else)
        ranges[symbol] = sorted(own + rest) if any(ch is anything_else for ch in chars) else own
    return fsm, ranges


def _refused(pattern: str, err) -> ValueError:
    return ValueError(f"Unsupported regular expression {pattern!r}: {err}. Regex matches the whole text, so anchors "
                      f"(^ $ \\A \\Z) are never needed; \\b, lookarounds, backreferences, possessive repetition, atomic "
                      f"groups, (?m), (?x) and (?i) with characters outside ASCII are not supported.")




class Regex(Constraint):
    """
    Token sequences whose text matches ``pattern`` in full (see the module docstring).

    :param pattern: the regular expression (Python syntax)
    :type pattern: str

    :param vocabulary: the token texts: a :class:`~pyjuice.constraints.Vocabulary`, or a tokenizer
        (:meth:`Vocabulary.from_tokenizer <pyjuice.constraints.Vocabulary.from_tokenizer>` reads it)

    :param eos_padding: the text may end early, then only end-of-sequence ids follow (default); False: every token
        spells the text
    :type eos_padding: bool
    """

    def __init__(self, pattern: str, vocabulary, eos_padding: bool = True):
        if not isinstance(pattern, str):
            raise TypeError(f"`pattern` must be a str, got {type(pattern).__name__}.")
        if not isinstance(vocabulary, Vocabulary):
            vocabulary = Vocabulary.from_tokenizer(vocabulary)
        if eos_padding and not vocabulary.eos:
            raise ValueError("The vocabulary has no end-of-sequence ids, which `eos_padding` needs: give them to the "
                             "Vocabulary (`eos = ...`), or pass `eos_padding = False`.")
        super().__init__(vocabulary.size)
        interegular, _ = _interegular()
        try:
            compiled = re.compile(pattern, re.ASCII)
        except re.error as e:
            raise _refused(pattern, e) from None
        why = _unsupported(pattern)
        if why is not None:
            raise _refused(pattern, why)
        try:
            parsed = interegular.parse_pattern(pattern)
        except Exception as e:                  # interegular's own refusals, and its failures on what it cannot read
            raise _refused(pattern, f"{type(e).__name__}: {e}") from None
        self._pattern, self._parsed, self._re = pattern, parsed, compiled
        self._vocabulary = vocabulary
        self._eos_padding = bool(eos_padding)
        self._automaton: Optional[DFA] = None

    @property
    def pattern(self) -> str:
        return self._pattern

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    @property
    def eos_padding(self) -> bool:
        return self._eos_padding

    # ---------------------------------------------------------------------------------------------
    # Membership: Python's `re` on the decoded text, independent of the automaton
    # ---------------------------------------------------------------------------------------------

    def accepts(self, tokens: Tokens) -> bool:
        toks = self._check_tokens(tokens)
        if self._eos_padding:
            eos = set(self._vocabulary.eos)
            end = next((i for i, t in enumerate(toks) if t in eos), len(toks))
            if any(t not in eos for t in toks[end:]):
                return False
            toks = toks[:end]
        try:
            text = self._vocabulary.decode(toks).decode("utf-8")
        except ValueError:                      # an id with no text, or bytes that are not UTF-8
            return False
        return self._re.fullmatch(text) is not None

    # ---------------------------------------------------------------------------------------------
    # Capabilities, from the automaton
    # ---------------------------------------------------------------------------------------------

    def automaton(self) -> DFA:
        if self._automaton is None:
            self._automaton = to_tokens(self._byte_automaton(), self._vocabulary, self._eos_padding)
        return self._automaton

    def _byte_automaton(self) -> ByteAutomaton:
        _, errors = _interegular()
        try:
            fsm, ranges = _char_automaton(self._parsed)
        except errors as e:                     # (nothing the checks let through should get here)
            raise _refused(self._pattern, e) from None
        return ByteAutomaton.from_char_automaton(fsm.states, fsm.initial, fsm.finals, fsm.map, ranges)

    def matcher(self) -> Matcher:
        return self.automaton().matcher()

    def wmc(self, log_weights: torch.Tensor) -> torch.Tensor:
        return self.automaton().wmc(log_weights)

    def sample(self, log_weights: torch.Tensor, num_samples: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.automaton().sample(log_weights, num_samples, generator = generator)

    def _widened(self, vocab_size: int) -> "Regex":
        out = type(self).__new__(type(self))                # (a subclass's own fields come along)
        out.__dict__.update(self.__dict__)
        Constraint.__init__(out, vocab_size)
        out._vocabulary = self._vocabulary.with_size(vocab_size)
        out._automaton = None if self._automaton is None else widened(self._automaton, vocab_size)
        return out

    # ---------------------------------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------------------------------

    def _fingerprint_payload(self):
        return (self._pattern, self._vocabulary.fingerprint(), self._eos_padding)

    def info(self):
        info = super().info()
        info.update(pattern = self._pattern, eos_padding = self._eos_padding)
        return info
