"""jc.LTLf: Spot's LTLf syntax and semantics on token sequences; its automaton against the independent evaluation of
`accepts`, the composed construction against progression alone, minimal sizes against Spot's."""

import itertools
import random

import numpy as np
import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.language.backend import ltlf as ltlf_backend


V = 4
# overlapping atoms, so that letters are not one per atom: token 0 is a, 1 is a and b, 2 is b, 3 is c
ATOMS = {"a": [0, 1], "b": [1, 2], "c": [3]}


def ltl(formula, atoms = ATOMS, vocab_size = V):
    return jc.LTLf(formula, atoms, vocab_size)


def sequences(max_len, vocab_size = V):
    for n in range(max_len + 1):
        yield from itertools.product(range(vocab_size), repeat = n)


def assert_automaton_agrees(c, max_len = 5):
    """The automaton accepts exactly what `accepts` does, on every sequence up to `max_len` tokens."""
    dense = c.automaton().to_dense().numpy()
    dfa = c.automaton()
    accept = dfa.accept.numpy()
    for n in range(max_len + 1):
        seqs = list(itertools.product(range(c.vocab_size), repeat = n))
        seqs = np.array(seqs, dtype = np.int64).reshape(len(seqs), n)
        state = np.full(seqs.shape[0], dfa.initial)
        for t in range(n):
            state = dense[state, seqs[:, t]]
        for s, ok in zip(seqs.tolist(), accept[state].tolist()):
            assert ok == c.accepts(s), (c.formula, s)


# -------------------------------------------------------------------------------------------------
# Syntax
# -------------------------------------------------------------------------------------------------

# (formula, its grouping as Spot 2.16 parses it)
SPOT_GROUPINGS = [
    ("a U b U c", "a U (b U c)"),
    ("a U b U c U d", "a U (b U (c U d))"),
    ("a R b U c", "a R (b U c)"),
    ("a W b M c", "a W (b M c)"),
    ("a -> b -> c", "a -> (b -> c)"),
    ("a <-> b <-> c", "a <-> (b <-> c)"),
    ("a -> b <-> c", "a -> (b <-> c)"),
    ("a <-> b -> c", "a <-> (b -> c)"),
    ("a & b | c", "(a & b) | c"),
    ("a | b -> c", "(a | b) -> c"),
    ("a -> b | c", "a -> (b | c)"),
    ("a <-> b | c", "a <-> (b | c)"),
    ("a U b & c", "(a U b) & c"),
    ("a & b U c", "a & (b U c)"),
    ("!a U b", "(!a) U b"),
    ("F a U b", "(F a) U b"),
    ("X a U b", "(X a) U b"),
    ("a U !b", "a U (!b)"),
    ("a U X b", "a U (X b)"),
    ("a U b -> c U d", "(a U b) -> (c U d)"),
    ("F a & G b | c", "((F a) & (G b)) | c"),
    ("G a -> F b", "(G a) -> (F b)"),
    ("! a -> b", "(!a) -> b"),
    ("GFa", "G(F(a))"),
    ("Foo", "F(oo)"),
    ("XXa", "X(X(a))"),
    ("F!a", "F(!a)"),
    ("X[!]X a", "X[!](X(a))"),
    ("X [!] a", "X[!](a)"),
    ("GX[!]a", "G(X[!](a))"),
    ("1 U a", "true U a"),
    ("0 R a", "false R a"),
    ('"Name" U a', "(\"Name\") U (a)"),
]
NAMED = {"a": [0], "b": [1], "c": [2], "d": [3], "oo": [1, 2], "Name": [3]}


@pytest.mark.parametrize("formula, grouped", SPOT_GROUPINGS)
def test_precedence_and_associativity_are_spots(formula, grouped):
    """Equal parses (a constraint's identity is its parse and its atoms' tokens)."""
    assert jc.LTLf(formula, NAMED, V) == jc.LTLf(grouped, NAMED, V)


def test_the_parse_is_the_formulas_tree():
    """The tree itself, for a few shapes (the groupings above compare two parses)."""
    tree = lambda f: jc.LTLf(f, NAMED, V)._tree
    assert tree("a U b U c") == ("U", ("ap", "a"), ("U", ("ap", "b"), ("ap", "c")))
    assert tree("a & b | !c") == ("|", ("&", ("ap", "a"), ("ap", "b")), ("!", ("ap", "c")))
    assert tree("GX[!]a -> true") == ("->", ("G", ("X[!]", ("ap", "a"))), ("true",))
    assert tree('X "Name"') == ("X", ("ap", "Name"))


@pytest.mark.parametrize("formula, message", [
    ("", "empty"),
    ("   ", "empty"),
    ("a b", "unexpected 'b' after a formula"),
    ("a U", "ends where an operand should be"),
    ("(a", "expected '\\)'"),
    ("a)", "unexpected '\\)' after a formula"),
    ("a & & b", "expected an operand, got '&'"),
    ("A", "must be double-quoted"),
    ("True", "must be double-quoted"),
    ("Ua", "must be double-quoted"),
    ("a U Rb", "must be double-quoted"),
    ("FG", "ends where an operand should be"),
    ("F[2:4] a", "bounded operators"),
    ("X[3] a", "bounded operators"),
    ("G[1:2] a", "bounded operators"),
    ("a xor b", "xor is not supported"),
    ("a ^ b", "unexpected '\\^'"),
    ("a && b", "expected an operand, got '&'"),
    ("a || b", "expected an operand, got '\\|'"),
    ("a => b", "unexpected '='"),
    ("a <=> b", "unexpected '<'"),
    ("~a", "unexpected '~'"),
    ("[]a", "unexpected '\\['"),
    ("<>a", "unexpected '<'"),
    ("{a;b}", "unexpected '{'"),
    ('""', "empty atom name"),
    ("2", "not a constant"),
    ("a.b", "unexpected '\\.'"),
])
def test_refused_syntax(formula, message):
    with pytest.raises(ValueError, match = message):
        jc.LTLf(formula, NAMED, V)


def test_atoms_are_checked():
    with pytest.raises(ValueError, match = r"without tokens: \['d'\]"):
        ltl("F a & G d")
    with pytest.raises(ValueError, match = r"must lie in \[0, 4\)"):
        ltl("F a", {"a": [4]})
    with pytest.raises(ValueError, match = r"must lie in \[0, 4\)"):
        ltl("F a", {"a": [-1]})
    with pytest.raises(ValueError, match = "shape \\[4\\]"):
        ltl("F a", {"a": torch.ones(5, dtype = torch.bool)})
    for spec in (3, "ab", [True, False], [0.0, 1.0], torch.tensor([0.5])):
        with pytest.raises(TypeError, match = "collection of token ids or a bool mask"):
            ltl("F a", {"a": spec})
    with pytest.raises(TypeError, match = "map atom names"):
        ltl("F a", [("a", [0])])
    with pytest.raises(TypeError, match = "must be a string"):
        ltl(("F", ("ap", "a")))
    # one token set, written as ids, a tensor of ids, a bool tensor or a bool array; entries not read are ignored
    ref = ltl("F a", {"a": [0, 1]})
    for spec in ((1, 0, 1), torch.tensor([0, 1]), np.array([1, 0]), torch.tensor([True, True, False, False]),
                 np.array([True, True, False, False])):
        assert ltl("F a", {"a": spec, "unused": "not even checked"}) == ref
    assert ltl("G !a", {"a": []}).accepts([0, 3])                       # an atom of no token never holds
    assert ref.atoms["a"].tolist() == [True, True, False, False]


# -------------------------------------------------------------------------------------------------
# Semantics
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("formula, accepted, rejected", [
    ("X a", [[2], [2, 0], [3, 1]], [[0, 2], [3, 3]]),                     # weak next: true at the last token
    ("X[!] a", [[2, 0], [3, 1, 3]], [[2], [0, 2]]),                       # strong next: false at the last token
    ("X false", [[0], [3]], [[0, 0]]),                                    # one token
    ("X[!] X[!] true", [[0, 0, 0], [3, 2, 1, 0]], [[0], [0, 0]]),         # at least three
    ("F c", [[3], [0, 1, 3]], [[0, 1, 2]]),
    ("G a", [[0], [0, 1, 1]], [[2], [0, 2], [0, 1, 3]]),
    ("a U c", [[3], [0, 1, 3], [0, 3, 2]], [[0, 1], [0, 2, 3], [2]]),
    ("a W c", [[3], [0, 1], [0, 3, 2]], [[0, 2, 3], [2]]),
    ("b R a", [[0, 0], [0, 1, 2], [1], [1, 3]], [[2], [0, 2], [0, 0, 3]]),   # a up to and including the first b
    ("b M a", [[0, 1, 2], [1]], [[0, 0], [2]]),                           # ... and a b at all
    ("a -> X[!] b", [[2], [0, 2], [1, 1]], [[0], [0, 3]]),
    ("a <-> b", [[1], [3]], [[0], [2]]),
    ("!a", [[2], [3, 0]], [[0], [1, 2]]),
    ("true", [[0], [3, 3]], []),
    ("false", [], [[0], [1, 2]]),
    ("G(a -> F c)", [[3], [0, 3], [2], [0, 2, 3]], [[0], [0, 3, 1]]),      # every a answered by a later c
    ("!c W a", [[2, 2], [0, 3], [2, 1, 3]], [[3], [2, 3, 0]]),            # no c before the first a
])
def test_accepts_is_ltlf_on_finite_traces(formula, accepted, rejected):
    """Hand-checked traces; the empty sequence is never accepted (traces are non-empty)."""
    c = ltl(formula)
    for s in accepted:
        assert c.accepts(s), s
    for s in rejected + [[]]:
        assert not c.accepts(s), s


PATTERNS = [
    "a", "!a", "true", "false", "X a", "X[!] a", "F a", "G a", "a U b", "a R b", "a W b", "a M b", "a -> b",
    "a <-> b", "a & c", "a | c",
    "G(a -> F c)",                                      # response
    "!c W a",                                           # precedence
    "F a & F c",                                        # existence of both
    "G !c",                                             # absence
    "G(a -> X[!] b)", "G(a -> X b)",                    # chain response, strong and weak
    "!(F a & F c)",                                     # not both
    "F(a & X[!] F(b & X[!] F c))",                      # in this order
    "G(a -> X G !a)",                                   # at most once
    "G((a -> F b) & (c -> F a))",                       # G over &
    "F(G a | G c)",                                     # F over |
    "F(a & X[!] c) | F(c & X[!] a)",
    "(a U b) U c", "a U (b U c)",
    "G(a <-> X[!] b)", "!(a U b) -> G c", "a M (b W !c)", "X[!] X[!] true",
]


def random_formula(rng, depth):
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(["a", "b", "c"] * 4 + ["true", "false"])
    op = rng.choice(["!", "X", "X[!]", "F", "G", "&", "|", "->", "<->", "U", "R", "W", "M"])
    if op in ("!", "X", "X[!]", "F", "G"):
        return f"{op}({random_formula(rng, depth - 1)})"
    return f"({random_formula(rng, depth - 1)}) {op} ({random_formula(rng, depth - 1)})"


RANDOM = [random_formula(random.Random(seed), 5) for seed in range(150)]


@pytest.mark.parametrize("formula", PATTERNS + RANDOM, ids = PATTERNS + [f"random{i}" for i in range(len(RANDOM))])
def test_the_automaton_accepts_what_accepts_does(formula):
    assert_automaton_agrees(ltl(formula))


@pytest.mark.parametrize("formula", [
    "G((a -> F b) & (c -> F a))", "F(G a | G c)", "a <-> X[!] b", "!(F a & F c)", "(a -> F c) & G(b -> X !b)",
    "F a | G(b U c) | !X[!] c", "G(a & X b) -> F(c | X[!] a)", "G(F a & F(b & X[!] c) & !(c U a))",
])
def test_composition_gives_the_automaton_of_progression_alone(formula, monkeypatch):
    composed = ltl(formula).automaton()
    monkeypatch.setattr(ltlf_backend, "_composed", ltlf_backend._progressed)
    assert ltl(formula).automaton() == composed


def keyword_family(name, k):
    """Formulas over k keyword atoms w0..w{k-1}, token i the i-th keyword and token k none of them."""
    w = [f"w{i}" for i in range(k)]
    if name == "all-of":
        formula = " & ".join(f"F {a}" for a in w)
    elif name == "in-order":
        formula = f"F {w[-1]}"
        for a in reversed(w[:-1]):
            formula = f"F({a} & X[!]({formula}))"
    elif name == "response":
        formula = " & ".join(f"G({w[2 * i]} -> X[!] F {w[2 * i + 1]})" for i in range(k // 2))
    elif name == "none-before":
        formula = " & ".join(f"((!{w[2 * i + 1]}) U {w[2 * i]})" for i in range(k // 2))
    else:                                               # at-most-twice
        formula = " & ".join(f"G({a} -> X G({a} -> X G !{a}))" for a in w)
    return jc.LTLf(formula, {a: [i] for i, a in enumerate(w)}, k + 1)


@pytest.mark.parametrize("name, k, states", [
    ("all-of", 8, 256), ("in-order", 8, 9), ("response", 8, 17), ("none-before", 8, 17), ("at-most-twice", 6, 731),
    ("at-most-twice", 10, 59051),
])
def test_minimal_sizes_are_spots(name, k, states):
    """Spot 2.16's minimal DFAs of these formulas have these many states (non-empty traces)."""
    assert keyword_family(name, k).automaton().num_states == states


# -------------------------------------------------------------------------------------------------
# A constraint like the others
# -------------------------------------------------------------------------------------------------

def test_widening_gives_the_new_ids_no_atom():
    c = ltl("G !a & F b")
    wide = c._widened(V + 2)
    padded = {a: torch.cat([m, torch.zeros(2, dtype = torch.bool)]) for a, m in c.atoms.items()}
    assert wide.vocab_size == V + 2 and wide == jc.LTLf("G !a & F b", padded, V + 2)
    assert wide.accepts([4, 2, 5]) and not ltl("G(a | b | c)")._widened(V + 2).accepts([0, 4])
    assert_automaton_agrees(wide, max_len = 4)
    with pytest.raises(ValueError, match = "Cannot widen"):
        c._widened(V - 1)


def test_identity_is_the_parse_and_the_tokens():
    assert ltl("F a & G b") == ltl("(F(a)) & (G b)")
    assert ltl("F a & G b") != ltl("G b & F a")                         # structural, not semantic
    assert ltl("F a") != ltl("F a", {"a": [0]})
    assert ltl("F a") != ltl("F a", ATOMS, V + 1)
    assert hash(ltl("F a")) == hash(ltl("F(a)"))


def test_composes_with_other_constraints():
    c = ltl("G(a -> X[!] b)") & ~jc.DFA.contains([[3, 3]], vocab_size = V)
    assert_automaton_agrees(c)
    assert_automaton_agrees(ltl("F c").concat(ltl("G !c")))


def test_capabilities_wmc_and_info():
    c = ltl("G(a -> X[!] b) & F c")
    assert {"accepts", "automaton", "matcher", "wmc", "sample", "relax"} <= c.capabilities()
    lw = torch.randn(2, 3, V, generator = torch.Generator().manual_seed(0))
    brute = torch.stack([torch.logsumexp(torch.stack([lw[b, torch.arange(3), torch.tensor(s)].sum()
                                                      for s in sequences(3) if len(s) == 3 and c.accepts(s)]), 0)
                         for b in range(2)])
    assert torch.allclose(c.wmc(lw), brute, atol = 1e-5)
    m = c.matcher()
    assert m.advance(0)                                                 # an a: a b next, and a c by the end
    assert m.allowed_next(2).tolist() == [False, False, True, False]    # in two tokens: only [0, 2, 3]
    info = c.info()
    assert info["kind"] == "LTLf" and info["formula"] == "G(a -> X[!] b) & F c" and info["atoms"] == ["a", "b", "c"]
    assert "formula=G(a -> X[!] b) & F c" in repr(c)
