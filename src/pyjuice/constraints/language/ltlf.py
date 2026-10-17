"""
Linear temporal logic over finite traces (LTLf) on token sequences.

:class:`LTLf` is the set of token sequences whose trace satisfies a formula. An ATOM is a set of token ids, true at the
positions whose token is in it. The trace is the whole sequence and is never empty: the empty sequence is never
accepted. Compiled against a PC, the trace is the PC's variables in index order.

The syntax and the semantics are Spot's, for LTLf (https://spot.lre.epita.fr/tl.pdf):

=================================  =====================================================================================
``a``, ``"any name"``              an atom: an identifier starting with a lowercase letter or ``_``, or a quoted name
``true``, ``false``, ``1``, ``0``  constants
``! f``                            not
``f & g``, ``f | g``               and, or
``f -> g``, ``f <-> g``            implies, if and only if
``X f``                            WEAK next: there is no next position, or ``f`` holds at it (true at the last one)
``X[!] f``                         STRONG next: there is a next position, and ``f`` holds at it
``F f``                            ``f`` holds at this position or a later one
``G f``                            ``f`` holds at this position and every later one
``f U g``                          ``g`` holds at this position or a later one, and ``f`` at every position before
``f R g``                          ``g`` holds up to and including the first position where ``f`` does, or everywhere
``f W g``                          ``f U g`` or ``G f``
``f M g``                          ``f R g`` and ``F f``
=================================  =====================================================================================

Many LTLf papers and tools write ``X`` for the strong next; here, as in Spot, ``X`` is the weak one. Precedence, loosest
first: ``->`` and ``<->`` (one level, right-associative), ``|``, ``&``, ``U R W M`` (right-associative: ``a U b U c``
is ``a U (b U c)``), then the prefix operators ``! X X[!] F G``, which may be written together with what follows
(``GFa`` is ``G(F(a))``). Refused with a message: Spot's synonyms (``&& || => <=> ~ [] <>``), ``xor``, a bare name
starting with an uppercase letter (Spot reads some of them as atoms, ``Ua``, ``FG``, and ``True`` as a constant: quote
the name), PSL, and the bounded operators ``X[n] F[n:m] G[n:m]`` (Spot expands them with weak nexts, so ``F[2:4] a``
holds on a one-token trace).

The automaton is built on first use (:mod:`.backend.ltlf`).
"""

from __future__ import annotations

import re
from collections import abc
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from .backend.ltlf import to_dfa
from .base import Constraint, Matcher, Tokens
from .dfa import DFA


class LTLf(Constraint):
    """
    The token sequences whose trace satisfies an LTLf formula (syntax and semantics: the module docstring).

    :param formula: the formula, in Spot's syntax
    :type formula: str

    :param atoms: the tokens of every atom the formula reads, as a collection of token ids or a bool mask of size
        [vocab_size]; entries it does not read are ignored
    :type atoms: Mapping[str, Union[Iterable[int], torch.Tensor]]

    :param vocab_size: size of the token vocabulary
    :type vocab_size: int

    Compiled against a PC whose inputs range over more values than ``vocab_size``, the constraint widens: the extra
    ids satisfy no atom.
    """

    def __init__(self, formula: str, atoms: Mapping[str, Union[Iterable[int], torch.Tensor]], vocab_size: int):
        super().__init__(vocab_size)
        if not isinstance(formula, str):
            raise TypeError(f"`formula` must be a string, got {type(formula).__name__}.")
        if not isinstance(atoms, abc.Mapping):
            raise TypeError(f"`atoms` must map atom names to token ids, got {type(atoms).__name__}.")
        tree = _Parser(formula).parse()
        names = _atoms_in(tree)
        missing = [a for a in names if a not in atoms]
        if missing:
            raise ValueError(f"The formula reads atoms without tokens: {missing}.")
        self._formula, self._tree = formula, tree
        self._masks = {a: _atom_mask(a, atoms[a], self.vocab_size) for a in names}
        self._automaton: Optional[DFA] = None

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def atoms(self) -> Dict[str, torch.Tensor]:
        """The tokens of every atom the formula reads: bool [vocab_size]."""
        return {a: m.clone() for a, m in self._masks.items()}

    # ---------------------------------------------------------------------------------------------
    # Membership and the automaton
    # ---------------------------------------------------------------------------------------------

    def accepts(self, tokens: Tokens) -> bool:
        toks = self._check_tokens(tokens)
        if not toks:
            return False
        index = torch.as_tensor(toks, dtype = torch.long)
        at = {a: m[index].tolist() for a, m in self._masks.items()}
        return _values(self._tree, at, len(toks))[0]

    def automaton(self) -> DFA:
        if self._automaton is None:
            self._automaton = to_dfa(self._tree, self._masks, self.vocab_size)
        return self._automaton

    def matcher(self) -> Matcher:
        return self.automaton().matcher()

    def wmc(self, log_weights: torch.Tensor) -> torch.Tensor:
        return self.automaton().wmc(log_weights)

    def sample(self, log_weights: torch.Tensor, num_samples: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.automaton().sample(log_weights, num_samples, generator = generator)

    def _widened(self, vocab_size: int) -> "LTLf":
        if vocab_size < self.vocab_size:
            raise ValueError(f"Cannot widen a constraint over {self.vocab_size} ids to {vocab_size}.")
        extra = torch.zeros(vocab_size - self.vocab_size, dtype = torch.bool)
        return LTLf(self._formula, {a: torch.cat([m, extra]) for a, m in self._masks.items()}, vocab_size)

    # ---------------------------------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------------------------------

    def _fingerprint_payload(self):
        return (self._tree, tuple((a, self._masks[a]) for a in sorted(self._masks)))

    def info(self):
        info = super().info()
        info.update(formula = self._formula, atoms = list(self._masks))
        return info


# -------------------------------------------------------------------------------------------------
# Atoms
# -------------------------------------------------------------------------------------------------

def _atom_mask(name: str, spec, vocab_size: int) -> torch.Tensor:
    """An atom's tokens as a bool mask [vocab_size], from a bool mask or a collection of token ids."""
    if isinstance(spec, (torch.Tensor, np.ndarray)) and spec.dtype in (torch.bool, np.bool_):
        mask = torch.as_tensor(spec).cpu()
        if mask.shape != (vocab_size,):
            raise ValueError(f"Atom {name!r}: a bool mask must have shape [{vocab_size}], got {tuple(mask.shape)}.")
        return mask.clone()
    if isinstance(spec, (torch.Tensor, np.ndarray)):
        ids = spec.reshape(-1).tolist()
    elif isinstance(spec, abc.Iterable) and not isinstance(spec, (str, bytes)):
        ids = list(spec)
    else:
        ids = None
    if ids is None or any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) for t in ids):
        raise TypeError(f"Atom {name!r}: expected a collection of token ids or a bool mask, got {spec!r}.")
    mask = torch.zeros(vocab_size, dtype = torch.bool)
    if ids:
        ids = torch.as_tensor([int(t) for t in ids], dtype = torch.long)
        if ids.min() < 0 or ids.max() >= vocab_size:
            raise ValueError(f"Atom {name!r}: token ids must lie in [0, {vocab_size}).")
        mask[ids] = True
    return mask


def _atoms_in(tree, out: Optional[List[str]] = None) -> List[str]:
    """The atoms ``tree`` reads, in order of first appearance."""
    out = [] if out is None else out
    if tree[0] == "ap":
        if tree[1] not in out:
            out.append(tree[1])
    else:
        for t in tree[1:]:
            _atoms_in(t, out)
    return out


# -------------------------------------------------------------------------------------------------
# Parsing: a formula -> nested tuples (op, *operands); ("ap", name) an atom, ("true",) / ("false",) the constants
# -------------------------------------------------------------------------------------------------

_SPACE = re.compile(r"\s*")
_LEXEME = re.compile(r'(?P<op><->|->|[!&|()])|(?P<quoted>"[^"]*")|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<number>[0-9]+)')
_STRONG = re.compile(r"\s*\[\s*!\s*\]")                                    # after an X: X[!]
_OPERATORS = "the operators are ! & | -> <-> X X[!] F G U R W M"


def _refused(formula: str, pos: int, why: str) -> ValueError:
    return ValueError(f"Cannot read the LTLf formula {formula!r} at position {pos}: {why}.")


def _lex(formula: str) -> List[Tuple[str, Any, int]]:
    """``(kind, value, position)`` lexemes: kind an operator or constant, or "atom" with the name as the value."""
    out, pos = [], 0
    while True:
        pos = _SPACE.match(formula, pos).end()
        if pos == len(formula):
            return out
        m = _LEXEME.match(formula, pos)
        if m is None:
            ch = formula[pos]
            if ch == "[" and out and out[-1][0] in ("X", "F", "G"):
                raise _refused(formula, pos, "bounded operators (X[n], F[n:m], G[n:m]) are not supported")
            raise _refused(formula, pos, f"unexpected {ch!r} ({_OPERATORS})")
        kind, text, end = m.lastgroup, m.group(), m.end()
        if kind == "op":
            out.append((text, None, pos))
        elif kind == "quoted":
            if len(text) == 2:
                raise _refused(formula, pos, "an empty atom name")
            out.append(("atom", text[1:-1], pos))
        elif kind == "number":
            if text not in ("0", "1"):
                raise _refused(formula, pos, f"{text!r} is not a constant (1 or 0)")
            out.append(("true" if text == "1" else "false", None, pos))
        elif text[0].islower() or text[0] == "_":
            if text == "xor":
                raise _refused(formula, pos, "xor is not supported (write !(f <-> g))")
            out.append((text, None, pos) if text in ("true", "false") else ("atom", text, pos))
        elif text in ("U", "R", "W", "M"):
            out.append((text, None, pos))
        else:                                                               # prefix operators, maybe then a name
            run = len(text) - len(text.lstrip("FGX"))
            rest = text[run:]
            if run == 0 or (rest and not (rest[0].islower() or rest[0] == "_")):
                raise _refused(formula, pos, f"{text!r} is neither operators nor a name; a name that does not start "
                                             f"with a lowercase letter or '_' must be double-quoted")
            out.extend((ch, None, pos + i) for i, ch in enumerate(text[:run]))
            if rest:
                out.append((rest, None, pos + run) if rest in ("true", "false") else ("atom", rest, pos + run))
            elif text[-1] == "X":
                strong = _STRONG.match(formula, end)
                if strong is not None:
                    out[-1] = ("X[!]", None, pos + run - 1)
                    end = strong.end()
        pos = end


class _Parser:
    """Recursive descent over Spot's precedence (loosest first): -> <-> (right), |, &, U R W M (right), prefix."""

    def __init__(self, formula: str):
        self._formula = formula
        self._lexemes = _lex(formula)
        self._i = 0

    def parse(self):
        if not self._lexemes:
            raise _refused(self._formula, 0, "the formula is empty")
        tree = self._implication()
        if self._i < len(self._lexemes):
            kind, value, pos = self._lexemes[self._i]
            raise _refused(self._formula, pos, f"unexpected {value if kind == 'atom' else kind!r} after a formula")
        return tree

    def _peek(self) -> Optional[str]:
        return self._lexemes[self._i][0] if self._i < len(self._lexemes) else None

    def _take(self):
        lexeme = self._lexemes[self._i]
        self._i += 1
        return lexeme

    def _implication(self):
        left = self._disjunction()
        if self._peek() in ("->", "<->"):
            op = self._take()[0]
            return (op, left, self._implication())
        return left

    def _disjunction(self):
        tree = self._conjunction()
        while self._peek() == "|":
            self._take()
            tree = ("|", tree, self._conjunction())
        return tree

    def _conjunction(self):
        tree = self._binary()
        while self._peek() == "&":
            self._take()
            tree = ("&", tree, self._binary())
        return tree

    def _binary(self):
        left = self._prefix()
        if self._peek() in ("U", "R", "W", "M"):
            op = self._take()[0]
            return (op, left, self._binary())
        return left

    def _prefix(self):
        if self._peek() in ("!", "X", "X[!]", "F", "G"):
            op = self._take()[0]
            return (op, self._prefix())
        return self._operand()

    def _operand(self):
        if self._i == len(self._lexemes):
            raise _refused(self._formula, len(self._formula), "the formula ends where an operand should be")
        kind, value, pos = self._take()
        if kind == "(":
            tree = self._implication()
            if self._peek() != ")":
                where = self._lexemes[self._i][2] if self._i < len(self._lexemes) else len(self._formula)
                raise _refused(self._formula, where, "expected ')'")
            self._take()
            return tree
        if kind in ("true", "false"):
            return (kind,)
        if kind == "atom":
            return ("ap", value)
        raise _refused(self._formula, pos, f"expected an operand, got {kind!r} ({_OPERATORS})")


# -------------------------------------------------------------------------------------------------
# The reference semantics: every subformula's value at every position, from the last one back
# -------------------------------------------------------------------------------------------------

def _values(tree, at: Mapping[str, List[bool]], n: int) -> List[bool]:
    """The value of ``tree`` at every position of a trace of ``n >= 1`` positions; ``at`` every atom's values."""
    op = tree[0]
    if op in ("true", "false"):
        return [op == "true"] * n
    if op == "ap":
        return at[tree[1]]
    if op == "!":
        return [not v for v in _values(tree[1], at, n)]
    if op in ("X", "X[!]"):                                                 # past the last position: X true, X[!] false
        return _values(tree[1], at, n)[1:] + [op == "X"]
    if op in ("F", "G"):
        f, out = _values(tree[1], at, n), [False] * n
        acc = op == "G"                                                     # the value past the last position
        for i in range(n - 1, -1, -1):
            acc = (f[i] or acc) if op == "F" else (f[i] and acc)
            out[i] = acc
        return out
    f, g = _values(tree[1], at, n), _values(tree[2], at, n)
    if op == "&":
        return [x and y for x, y in zip(f, g)]
    if op == "|":
        return [x or y for x, y in zip(f, g)]
    if op == "->":
        return [(not x) or y for x, y in zip(f, g)]
    if op == "<->":
        return [x == y for x, y in zip(f, g)]
    out = [False] * n
    acc = op in ("R", "W")                                                  # the value past the last position
    for i in range(n - 1, -1, -1):
        if op in ("U", "W"):                                                # g, or f and the same from i + 1
            acc = g[i] or (f[i] and acc)
        else:                                                               # R, M: g, and f or the same from i + 1
            acc = g[i] and (f[i] or acc)
        out[i] = acc
    return out
