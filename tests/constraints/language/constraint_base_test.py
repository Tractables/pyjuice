import itertools

import pytest
import torch

import pyjuice as juice
from pyjuice.constraints import Constraint, And, Or, Not, Concat


class _Pred(Constraint):
    """Test-only leaf: membership decided by a Python predicate, identified by `name`."""

    def __init__(self, vocab_size, name, pred):
        super().__init__(vocab_size)
        self._name, self._pred = name, pred

    def accepts(self, tokens):
        return bool(self._pred(self._check_tokens(tokens)))

    def _fingerprint_payload(self):
        return (self._name,)


class _PredWithWmc(_Pred):
    """Test-only leaf for "contains token 1" that also implements `wmc` (by brute force) and provides
    its automaton."""

    def wmc(self, log_weights):
        B, n, V_ = log_weights.shape
        out = torch.full((B,), float("-inf"), dtype = log_weights.dtype)
        for x in itertools.product(range(V_), repeat = n):
            if self.accepts(x):
                out = torch.logaddexp(out, log_weights[:, torch.arange(n), list(x)].sum(dim = 1))
        return out

    def automaton(self):
        from pyjuice.constraints import DFA
        return DFA.contains([[1]], self.vocab_size)


V = 3
contains_1 = _Pred(V, "contains_1", lambda t: 1 in t)
even_len = _Pred(V, "even_len", lambda t: len(t) % 2 == 0)
starts_2 = _Pred(V, "starts_2", lambda t: len(t) > 0 and t[0] == 2)


def all_strings(max_len):
    for L in range(max_len + 1):
        yield from itertools.product(range(V), repeat = L)


def concat_ref(a, b, t):
    return any(a.accepts(t[:k]) and b.accepts(t[k:]) for k in range(len(t) + 1))


def test_composition_semantics_match_definitions():
    cases = {
        "and": (contains_1 & even_len, lambda t: contains_1.accepts(t) and even_len.accepts(t)),
        "or": (contains_1 | starts_2, lambda t: contains_1.accepts(t) or starts_2.accepts(t)),
        "not": (~contains_1, lambda t: not contains_1.accepts(t)),
        "concat": (starts_2.concat(contains_1), lambda t: concat_ref(starts_2, contains_1, t)),
        "concat3": (starts_2.concat(contains_1).concat(even_len),
                    lambda t: any(starts_2.accepts(t[:i]) and contains_1.accepts(t[i:j]) and even_len.accepts(t[j:])
                                  for i in range(len(t) + 1) for j in range(i, len(t) + 1))),
        "nested": (~(contains_1 & starts_2) | even_len,
                   lambda t: (not (contains_1.accepts(t) and starts_2.accepts(t))) or even_len.accepts(t)),
    }
    for name, (c, ref) in cases.items():
        for t in all_strings(5):
            assert c.accepts(t) == ref(t), (name, t)
        # tensors are accepted as token sequences too
        assert c.accepts(torch.tensor([2, 1, 0, 1])) == ref((2, 1, 0, 1)), name


def test_flattening_and_fingerprint_equality():
    a, b, c = contains_1, even_len, starts_2
    assert len(((a & b) & c).children) == 3 and len((a & (b & c)).children) == 3
    assert (a & b) & c == a & (b & c)                       # associative
    assert a & b == b & a and a | b == b | a                 # commutative
    assert a.concat(b) != b.concat(a)                        # concatenation is not
    assert a & b != a | b                                    # different operators
    assert ~~a == a
    assert len({a & b, b & a, a.concat(b)}) == 2             # usable as dict / set keys
    # the fingerprint is a structural hash: equal for equal structure built separately
    a2 = _Pred(V, "contains_1", lambda t: 1 in t)
    assert a2 == a and a2.fingerprint() == a.fingerprint()
    assert _Pred(V + 1, "contains_1", lambda t: 1 in t) != a  # vocabulary size is part of identity


def test_capabilities_are_reported_from_overrides():
    assert contains_1.capabilities() == {"accepts"}
    rich = _PredWithWmc(V, "rich", lambda t: 1 in t)
    # overriding `automaton` also grants `relax` (an exact automaton is a valid over-approximation)
    assert rich.capabilities() == {"accepts", "wmc", "automaton", "relax"}
    assert rich.relax() is rich
    assert juice.constraints.OPTIONAL_CAPABILITIES == ("matcher", "wmc", "sample", "automaton", "relax")


def test_every_claimed_capability_works_and_the_rest_raise():
    rich = _PredWithWmc(V, "rich", lambda t: 1 in t)
    calls = {
        "matcher": lambda c: c.matcher(),
        "wmc": lambda c: c.wmc(torch.zeros(2, 2, V)),
        "sample": lambda c: c.sample(torch.zeros(2, 2, V), 1),
        "automaton": lambda c: c.automaton(),
        "relax": lambda c: c.relax(),
    }
    for c in [contains_1, rich, contains_1 & rich, ~rich, rich.concat(contains_1)]:
        caps = c.capabilities()
        assert "accepts" in caps
        for name, call in calls.items():
            if name in caps:
                call(c)
            else:
                with pytest.raises(NotImplementedError, match = f"does not support `{name}`"):
                    call(c)
    # wmc semantics: log of the number of length-2 strings over {0,1,2} that contain a 1 (= 5)
    assert torch.allclose(rich.wmc(torch.zeros(1, 2, V)), torch.log(torch.tensor([5.0])))


def test_composites_get_capabilities_from_their_children():
    rich = _PredWithWmc(V, "rich", lambda t: 1 in t)
    every = {"accepts", "matcher", "wmc", "sample", "automaton", "relax"}
    # every child has an automaton -> the node builds its own and has every capability
    for c in [rich & rich, ~rich, rich.concat(rich), (rich | ~rich) & rich]:
        assert c.capabilities() == every, c
    # a child without one -> membership only
    for c in [rich | contains_1, ~contains_1, rich.concat(contains_1)]:
        assert c.capabilities() == {"accepts"}, c
    assert "capabilities" in (rich & contains_1).info()
    # the automaton is built once and cached
    c = rich & ~rich
    assert c.automaton() is c.automaton()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match = "different vocabularies"):
        contains_1 & _Pred(V + 1, "x", lambda t: True)
    with pytest.raises(ValueError, match = "outside the vocabulary"):
        contains_1.accepts([0, V])
    with pytest.raises(ValueError, match = "1-D"):
        contains_1.accepts(torch.zeros(2, 2, dtype = torch.long))
    with pytest.raises(TypeError):
        And(contains_1, "not a constraint")
    with pytest.raises(ValueError, match = "at least two"):
        Or(contains_1)
    with pytest.raises(ValueError):
        _Pred(0, "empty", lambda t: True)


def test_exposed_as_juice_constraints():
    assert juice.constraints.Constraint is Constraint
