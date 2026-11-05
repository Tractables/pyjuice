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
the last position, and then only end-of-sequence ids follow), or the text alone. The automaton -- the byte automaton of
Python's own parse of the pattern (:mod:`.backend.patterns`), then the token automaton (:mod:`.backend.byte_level`) -- is
built on first use.
"""

from __future__ import annotations

import re
import weakref
from typing import Optional

import torch

from .base import Constraint, Matcher, Tokens
from .backend.byte_level import ByteAutomaton
from .backend.token_lift import to_tokens, widened
from .backend.patterns import byte_automaton, parse
from .dfa import DFA
from .vocabulary import Vocabulary


def _unsupported(pattern: str, tree = None) -> Optional[str]:
    """
    Why ``pattern`` is outside what Regex reads exactly, or None. Python's own parse of the pattern (``tree``, if it is
    at hand) may hold only literals, ``.``, character classes of characters, ranges and categories, alternatives, groups
    and greedy or lazy repetition, what a regular language is; refused are anchors and ``\\b``, lookarounds,
    backreferences, possessive repetition and atomic groups, ``(?m)`` and ``(?x)``, a flag turned off, ``(?u:...)``
    (Unicode categories and folding, not this module's), ``(?i)`` with a character outside ASCII (what ``re`` folds under
    ``re.ASCII`` depends on it), and a ``]`` first in a character class.
    """
    try:
        from re import _constants as C                                  # Python >= 3.11
    except ImportError:                                                  # pragma: no cover
        import sre_constants as C
    tree = parse(pattern) if tree is None else tree
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
                if av[1] & re.UNICODE:
                    return "the (?u) flag"
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
    if "[]" not in pattern and "[^]" not in pattern:                    # (no ']' first in a class, then)
        return None
    i, inside = 0, False
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


def _refused(pattern: str, err) -> ValueError:
    return ValueError(f"Unsupported regular expression {pattern!r}: {err}. Regex matches the whole text, so anchors "
                      f"(^ $ \\A \\Z) are never needed; \\b, lookarounds, backreferences, possessive repetition, atomic "
                      f"groups, (?m), (?x) and (?i) with characters outside ASCII are not supported.")


def _checked(pattern: str):
    """Python's parse of ``pattern``, or the :class:`ValueError` saying why it is refused (see the module docstring)."""
    if not isinstance(pattern, str):
        raise TypeError(f"`pattern` must be a str, got {type(pattern).__name__}.")
    try:
        tree = parse(pattern)
    except (re.error, ValueError) as e:                 # (ValueError: flags Python's parser finds incompatible)
        raise _refused(pattern, e) from None
    why = _unsupported(pattern, tree)
    if why is not None:
        raise _refused(pattern, why)
    return tree


#: the vocabularies of the tokenizers given for one, by the tokenizer (reading one takes 0.1-0.2 s, and its byte trie goes
#: with its vocabulary): a constraint per example should not pay them
_TOKENIZERS = weakref.WeakKeyDictionary()


def _vocabulary_of(tokenizer) -> Vocabulary:
    """:meth:`Vocabulary.from_tokenizer` of ``tokenizer``, kept while it lives and says the same of its size and end of
    sequence (a tokenizer that cannot be referred to weakly is read every time)."""
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    size = backend.get_vocab_size(with_added_tokens = True) if hasattr(backend, "get_vocab_size") else None
    said = (size, getattr(tokenizer, "eos_token_id", None))
    try:
        kept = _TOKENIZERS.get(tokenizer)
    except TypeError:
        return Vocabulary.from_tokenizer(tokenizer)
    if kept is not None and kept[0] == said:
        return kept[1]
    vocabulary = Vocabulary.from_tokenizer(tokenizer)
    try:
        _TOKENIZERS[tokenizer] = (said, vocabulary)
    except TypeError:
        pass
    return vocabulary


class _Text(Constraint):
    """
    Token sequences whose TEXT -- the tokens' bytes concatenated, valid UTF-8 -- is in a set of texts, under the
    end-of-sequence padding convention (see the module docstring): what :class:`Regex` and
    :class:`~pyjuice.constraints.JSONSchema` share. A subclass gives the texts as a byte automaton
    (:meth:`_byte_automaton`) and its membership test on a decoded text (:meth:`_text_matches`).
    """

    def __init__(self, vocabulary, eos_padding: bool):
        if not isinstance(vocabulary, Vocabulary):
            vocabulary = _vocabulary_of(vocabulary)
        if eos_padding and not vocabulary.eos:
            raise ValueError("The vocabulary has no end-of-sequence ids, which `eos_padding` needs: give them to the "
                             "Vocabulary (`eos = ...`), or pass `eos_padding = False`.")
        super().__init__(vocabulary.size)
        self._vocabulary = vocabulary
        self._eos_padding = bool(eos_padding)
        self._automaton: Optional[DFA] = None

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    @property
    def eos_padding(self) -> bool:
        return self._eos_padding

    def _byte_automaton(self) -> ByteAutomaton:
        raise NotImplementedError

    def _text_matches(self, text: str) -> bool:
        raise NotImplementedError

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
        return self._text_matches(text)

    # ---------------------------------------------------------------------------------------------
    # Capabilities, from the automaton
    # ---------------------------------------------------------------------------------------------

    def automaton(self) -> DFA:
        if self._automaton is None:
            self._automaton = to_tokens(self._byte_automaton(), self._vocabulary, self._eos_padding)
        return self._automaton

    def matcher(self) -> Matcher:
        return self.automaton().matcher()

    def wmc(self, log_weights: torch.Tensor) -> torch.Tensor:
        return self.automaton().wmc(log_weights)

    def sample(self, log_weights: torch.Tensor, num_samples: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.automaton().sample(log_weights, num_samples, generator = generator)

    def _widened(self, vocab_size: int) -> "_Text":
        out = type(self).__new__(type(self))                # (a subclass's own fields come along)
        out.__dict__.update(self.__dict__)
        Constraint.__init__(out, vocab_size)
        out._vocabulary = self._vocabulary.with_size(vocab_size)
        out._automaton = None if self._automaton is None else widened(self._automaton, vocab_size)
        return out


class Regex(_Text):
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
        super().__init__(vocabulary, eos_padding)
        self._pattern, self._tree = pattern, _checked(pattern)

    @property
    def pattern(self) -> str:
        return self._pattern

    def _text_matches(self, text: str) -> bool:
        """Python's `re` on the decoded text, independent of the automaton."""
        return re.fullmatch(self._pattern, text, re.ASCII) is not None

    def _byte_automaton(self) -> ByteAutomaton:
        return byte_automaton(parse(self._pattern) if self._tree is None else self._tree)

    def __getstate__(self):                                     # (Python's parse does not pickle)
        return dict(self.__dict__, _tree = None)

    # ---------------------------------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------------------------------

    def _fingerprint_payload(self):
        return (self._pattern, self._vocabulary.fingerprint(), self._eos_padding)

    def info(self):
        info = super().info()
        info.update(pattern = self._pattern, eos_padding = self._eos_padding)
        return info
