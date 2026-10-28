"""
The automaton of an :class:`~pyjuice.constraints.language.ltlf.LTLf` constraint: a parsed formula (the tuples of
:mod:`pyjuice.constraints.language.ltlf`) -> the minimal DFA of the non-empty token sequences that satisfy it.

The Boolean structure of the formula (``! & | -> <->``, with ``G`` distributed over ``&`` and ``F`` over ``|``) is
built from the DFAs of its temporal parts by products and complements, each minimised. A temporal part becomes a DFA
by FORMULA PROGRESSION: a state is a formula in disjunctive normal form over elementary formulas (literals, temporal
formulas, and END: "the trace has ended"), up to absorption; the state after a letter is the formula the rest of the
trace must satisfy, and a state accepts when the empty rest satisfies it. Its letters are the valuations of the
part's own atoms that occur among the tokens. Last, the empty sequence is rejected (traces are non-empty).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..dfa import DFA, complement, product


def to_dfa(tree, masks: Mapping[str, torch.Tensor], vocab_size: int) -> DFA:
    """The minimal DFA of the non-empty token sequences that satisfy the parsed formula ``tree``; ``masks`` the
    tokens of every atom it reads, bool [vocab_size]."""
    return _non_empty(_composed(tree, masks, vocab_size))


def _composed(tree, masks, vocab_size: int) -> DFA:
    """The DFA of ``tree`` (the empty sequence accepted as progression reads it): the Boolean structure by products
    and complements of its parts' DFAs."""
    op = tree[0]
    if op in ("&", "|"):
        parts = [_composed(tree[1], masks, vocab_size), _composed(tree[2], masks, vocab_size)]
        return product(parts, "and" if op == "&" else "or")
    if op == "!":
        return complement(_composed(tree[1], masks, vocab_size))
    if op == "->":
        return product([complement(_composed(tree[1], masks, vocab_size)), _composed(tree[2], masks, vocab_size)],
                       "or")
    if op == "<->":
        a, b = _composed(tree[1], masks, vocab_size), _composed(tree[2], masks, vocab_size)
        return product([product([a, b], "and"), product([complement(a), complement(b)], "and")], "or")
    if op == "G" and tree[1][0] == "&":                                     # G(f & g) = G f & G g
        return _composed(("&", ("G", tree[1][1]), ("G", tree[1][2])), masks, vocab_size)
    if op == "F" and tree[1][0] == "|":                                     # F(f | g) = F f | F g
        return _composed(("|", ("F", tree[1][1]), ("F", tree[1][2])), masks, vocab_size)
    return _progressed(tree, masks, vocab_size)


def _progressed(tree, masks, vocab_size: int) -> DFA:
    """The DFA of ``tree`` (the empty sequence accepted as progression reads it) by formula progression over the
    letters of its atoms."""
    names = _atoms_in(tree)
    token_class, letters = _letters([masks[a] for a in names], vocab_size)
    progression = _Progression({a: i for i, a in enumerate(names)})
    start = _dnf(_nnf(tree))
    ids, order, rows = {start: 0}, [start], []
    head = 0
    while head < len(order):
        state = order[head]
        head += 1
        row = []
        for letter in letters:
            nxt = progression.formula(state, letter)
            j = ids.get(nxt)
            if j is None:
                j = ids[nxt] = len(order)
                order.append(nxt)
            row.append(j)
        rows.append(row)
    accept = torch.tensor([_holds_at_end(s) for s in order], dtype = torch.bool)
    return DFA(vocab_size, torch.from_numpy(token_class), torch.tensor(rows, dtype = torch.long), 0, accept).minimize()


def _non_empty(dfa: DFA) -> DFA:
    """The same language without the empty sequence: a fresh initial state that moves as the old one, rejecting."""
    delta = dfa.delta
    delta = torch.cat([delta, delta[dfa.initial:dfa.initial + 1]], dim = 0)
    accept = torch.cat([dfa.accept, torch.zeros(1, dtype = torch.bool)])
    return DFA(dfa.vocab_size, dfa.token_class, delta, delta.size(0) - 1, accept).minimize()


def _atoms_in(tree, out = None) -> List[str]:
    """The atoms ``tree`` reads, in order of first appearance."""
    out = [] if out is None else out
    if tree[0] == "ap":
        if tree[1] not in out:
            out.append(tree[1])
    else:
        for t in tree[1:]:
            _atoms_in(t, out)
    return out


def _letters(masks: Sequence[torch.Tensor], vocab_size: int) -> Tuple[np.ndarray, List[Tuple[bool, ...]]]:
    """The atom valuations that occur among the tokens: every token's (int64 [vocab_size]) and each one's truth
    values, in the atoms' order."""
    if not masks:
        return np.zeros(vocab_size, dtype = np.int64), [()]
    truth = torch.stack([m.cpu() for m in masks], dim = 1).numpy()
    rows, token_class = np.unique(truth, axis = 0, return_inverse = True)
    return token_class.reshape(-1).astype(np.int64), [tuple(bool(x) for x in r) for r in rows]


# -------------------------------------------------------------------------------------------------
# Negation normal form, DNF
# -------------------------------------------------------------------------------------------------

_DUAL = {"&": "|", "|": "&", "X": "X[!]", "X[!]": "X", "F": "G", "G": "F", "U": "R", "R": "U"}


def _nnf(tree, negated: bool = False):
    """``tree`` (negated if asked) with negations on atoms only, over ``true false ap !ap & | X X[!] F G U R``."""
    op = tree[0]
    if op in ("true", "false"):
        return (("false",) if op == "true" else ("true",)) if negated else tree
    if op == "ap":
        return ("!ap", tree[1]) if negated else tree
    if op == "!":
        return _nnf(tree[1], not negated)
    if op == "->":
        return _nnf(("|", ("!", tree[1]), tree[2]), negated)
    if op == "<->":
        return _nnf(("|", ("&", tree[1], tree[2]), ("&", ("!", tree[1]), ("!", tree[2]))), negated)
    if op == "W":                                                           # f W g = g R (f | g)
        return _nnf(("R", tree[2], ("|", tree[1], tree[2])), negated)
    if op == "M":                                                           # f M g = g U (f & g)
        return _nnf(("U", tree[2], ("&", tree[1], tree[2])), negated)
    return (_DUAL[op] if negated else op,) + tuple(_nnf(t, negated) for t in tree[1:])


# A formula in DNF: a frozenset of conjunctions, each a frozenset of elementary formulas (NNF tuples whose operator is
# not & or |, and the end markers).
_TRUE = frozenset([frozenset()])
_FALSE = frozenset()
_END, _NOT_END = ("END",), ("!END",)


def _clean(conjunctions):
    """Without contradictory conjunctions, and without every one that contains another (absorption)."""
    kept = []
    for c in sorted(set(conjunctions), key = len):
        if _END in c and _NOT_END in c:
            continue
        if any(e[0] == "ap" and ("!ap", e[1]) in c for e in c):
            continue
        if not any(k <= c for k in kept):
            kept.append(c)
    return frozenset(kept)


def _or(a, b):
    return _clean(list(a) + list(b))


def _and(a, b):
    return _clean([x | y for x in a for y in b])


def _lit(e):
    return frozenset([frozenset([e])])


def _dnf(tree):
    op = tree[0]
    if op == "true":
        return _TRUE
    if op == "false":
        return _FALSE
    if op == "&":
        return _and(_dnf(tree[1]), _dnf(tree[2]))
    if op == "|":
        return _or(_dnf(tree[1]), _dnf(tree[2]))
    return _lit(tree)


#: whether an elementary formula holds on the empty rest of a trace (past its last position)
_AT_END = {"ap": False, "!ap": False, "END": True, "!END": False, "X": True, "X[!]": False, "F": False, "G": True,
           "U": False, "R": True}


def _holds_at_end(dnf) -> bool:
    return any(all(_AT_END[e[0]] for e in c) for c in dnf)


class _Progression:
    """The formula the rest of the trace must satisfy after one letter (a tuple of the atoms' truth values)."""

    def __init__(self, index: Dict[str, int]):
        self._index = index
        self._memo = {}

    def formula(self, dnf, letter):
        out = _FALSE
        for conjunction in dnf:
            acc = _TRUE
            for e in conjunction:
                acc = _and(acc, self._elementary(e, letter))
                if not acc:
                    break
            out = _or(out, acc)
        return out

    def _elementary(self, e, letter):
        key = (e, letter)
        out = self._memo.get(key)
        if out is not None:
            return out
        op = e[0]
        if op in ("ap", "!ap"):
            out = _TRUE if letter[self._index[e[1]]] == (op == "ap") else _FALSE
        elif op == "END":                                                   # a letter: the trace goes on
            out = _FALSE
        elif op == "!END":
            out = _TRUE
        elif op == "X":                                                     # weak: no next position, or f there
            out = _or(_dnf(e[1]), _lit(_END))
        elif op == "X[!]":                                                  # strong: a next position, and f there
            out = _and(_dnf(e[1]), _lit(_NOT_END))
        elif op == "F":
            out = _or(self.formula(_dnf(e[1]), letter), _and(_lit(e), _lit(_NOT_END)))
        elif op == "G":
            out = _and(self.formula(_dnf(e[1]), letter), _or(_lit(e), _lit(_END)))
        elif op == "U":
            out = _or(self.formula(_dnf(e[2]), letter),
                      _and(self.formula(_dnf(e[1]), letter), _and(_lit(e), _lit(_NOT_END))))
        elif op == "R":
            out = _and(self.formula(_dnf(e[2]), letter),
                       _or(self.formula(_dnf(e[1]), letter), _or(_lit(e), _lit(_END))))
        else:
            raise AssertionError(f"Not an elementary formula: {e!r}.")
        self._memo[key] = out
        return out
