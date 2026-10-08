"""
The two constraints of Ctrl-G's tutorial, built with pyjuice.constraints over a small synthetic
vocabulary (no tokenizer download):

  1. one of {" riding a bike", " ride bikes", " rides a bike", " biking", " bikes"} appears, AND one of
     {" park", " beach"} appears, AND the text has exactly 10 words;
  2. one of {" girl", " boy", " girls", " boys", " children"} appears and is FOLLOWED (later) by one of
     {" dogs", " cats", " dog", " cat"}, AND the text has 7 to 12 words.

The equivalence with Ctrl-G's own DFAs on the real GPT-2 tokenizer is checked in the
constraints-in-pyjuice repository (baselines/ctrlg/test_pyjuice_dfa_matches_ctrlg.py).
"""
import itertools
import random

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.language.text import DEFAULT_SEPARATORS, word_token_kinds_from_strings

VOCAB = ["<|endoftext|>", " a", " boy", " girl", " girls", " boys", " children", " is", " are", " riding",
         " ride", " rides", " bike", " bikes", " biking", " in", " the", " park", " beach", " dog", " dogs",
         " cat", " cats", " with", " her", " his", " walking", ".", ",", " .", "s", "ing", "park", " on",
         " day", " sunny"]
TOK = {s: i for i, s in enumerate(VOCAB)}
V = len(VOCAB)
EOS = TOK["<|endoftext|>"]


def enc(text_tokens):
    return [TOK[t] for t in text_tokens]


def kinds():
    return word_token_kinds_from_strings(VOCAB, special_ids = [EOS])


def example1():
    bike = jc.DFA.contains([enc([" riding", " a", " bike"]), enc([" ride", " bikes"]),
                         enc([" rides", " a", " bike"]), enc([" biking"]), enc([" bikes"])], V)
    place = jc.DFA.contains([enc([" park"]), enc([" beach"])], V)
    return bike & place & jc.DFA.word_count(10, 10, kinds())


def example2():
    who = jc.DFA.contains([enc([w]) for w in [" girl", " boy", " girls", " boys", " children"]], V)
    pet = jc.DFA.contains([enc([w]) for w in [" dogs", " cats", " dog", " cat"]], V)
    return who.concat(pet) & jc.DFA.word_count(7, 12, kinds())


# Independent references, written from the definitions (not from the automata).

def has_any(tokens, patterns):
    return any(tuple(tokens[i:i + len(p)]) == tuple(p) for p in patterns for i in range(len(tokens) - len(p) + 1))


def first_end(tokens, patterns):
    """End index of the earliest-ending occurrence of any pattern, or None."""
    ends = [i + len(p) for p in patterns for i in range(len(tokens) - len(p) + 1)
            if tuple(tokens[i:i + len(p)]) == tuple(p)]
    return min(ends) if ends else None


def word_count_ref(tokens):
    """Ctrl-G's word count, by scanning token texts: a word starts at a token with a letter or digit
    that begins with a separator or follows one (the text is treated as preceded by a separator)."""
    count, after_sep = 0, True
    for t in tokens:
        text = VOCAB[t]
        if t == EOS or text == "":
            continue
        word_char = any(ch.isalnum() for ch in text)
        starts_sep = text[0] in DEFAULT_SEPARATORS
        if word_char:
            count += starts_sep or after_sep
            after_sep = False
        elif starts_sep:
            after_sep = True
    return count


BIKES = [enc(p) for p in ([" riding", " a", " bike"], [" ride", " bikes"], [" rides", " a", " bike"],
                          [" biking"], [" bikes"])]
PLACES = [enc([" park"]), enc([" beach"])]
WHO = [enc([w]) for w in [" girl", " boy", " girls", " boys", " children"]]
PETS = [enc([w]) for w in [" dogs", " cats", " dog", " cat"]]


def ref1(t):
    return has_any(t, BIKES) and has_any(t, PLACES) and word_count_ref(t) == 10


def ref2(t):
    e = first_end(t, WHO)
    return e is not None and has_any(t[e:], PETS) and 7 <= word_count_ref(t) <= 12


def random_strings(num, seed):
    rng = random.Random(seed)
    hot = [TOK[w] for w in [" riding", " a", " bike", " ride", " bikes", " biking", " park", " beach", " boy",
                            " girl", " dog", " cats", ".", " .", "s", "park", ","]]
    for _ in range(num):
        L = rng.randint(0, 16)
        yield [rng.choice(hot) if rng.random() < 0.6 else rng.randrange(V) for _ in range(L)]


SENTENCES = {
    # (tokens, accepted by example 1, accepted by example 2)
    "boy rides a bike in the park (10 words)":
        ([" boy", " is", " riding", " a", " bike", " in", " the", " park", " on", " day"], True, False),
    "11 words":
        ([" a", " boy", " is", " riding", " a", " bike", " in", " the", " park", " on", " day"], False, False),
    "9 words":
        ([" a", " boy", " is", " riding", " a", " bike", " in", " the", " park", "."], False, False),
    "no place":
        ([" a", " boy", " is", " riding", " a", " bike", " with", " his", " dog", " on", " day"], False, True),
    "girl then dog":
        ([" a", " girl", " walking", " her", " dog", " in", " the", " park", "."], False, True),
    "dog then girl (wrong order)":
        ([" a", " dog", " walking", " with", " a", " girl", " in", " the", " park"], False, False),
    "glued suffix is not a new word":
        ([" boy", " is", " riding", " a", " bike", " in", " the", " park", "s", " on", " day"], True, False),
    "a leading glued token IS a word (the text starts after a separator)":
        (["s", " boy", " is", " riding", " a", " bike", " in", " the", " park", " on", " day"], False, False),
    "punctuation starts a new word":
        ([" a", " boy", " .", "park", " bikes", " in", " the", " beach", " on", " a", " day"], True, False),
}


def test_hand_written_sentences():
    c1, c2 = example1(), example2()
    for name, (words, want1, want2) in SENTENCES.items():
        t = enc(words)
        assert ref1(t) == want1 and ref2(t) == want2, f"reference disagrees with the label: {name}"
        assert c1.accepts(t) == want1, name
        assert c2.accepts(t) == want2, name


@pytest.mark.parametrize("which", [1, 2])
def test_constraint_and_its_automaton_match_the_definition(which):
    c, ref = (example1(), ref1) if which == 1 else (example2(), ref2)
    assert isinstance(c, jc.And) and (which == 1 or isinstance(c.children[0], jc.Concat) or
                                   any(isinstance(ch, jc.Concat) for ch in c.children))
    dfa = c.automaton()
    pos = 0
    for t in random_strings(4000, seed = which):
        want = ref(t)
        assert c.accepts(t) == want, t          # composite membership (children's own `accepts`)
        assert dfa.accepts(t) == want, t        # the built product / concatenation automaton
        pos += want
    assert pos > 0                              # the random strings do reach acceptance


def test_automaton_sizes():
    # Ctrl-G's minimised DFAs have 100 and 66 states over the GPT-2 vocabulary; with this smaller
    # vocabulary the keyword parts are the same, so the canonical minimal automata are comparable.
    d1, d2 = example1().automaton(), example2().automaton()
    assert d1.minimize() == d1 and d2.minimize() == d2
    print(f"example 1: {d1.num_states} states, {d1.num_classes} classes; "
          f"example 2: {d2.num_states} states, {d2.num_classes} classes")


@pytest.mark.parametrize("which", [1, 2])
def test_wmc_and_sample_through_the_composite(which):
    """wmc / sample of the composite under factorised weights vs brute force, on a sub-vocabulary small
    enough to enumerate (all other tokens get weight 0)."""
    c, ref_fn = (example1(), ref1) if which == 1 else (example2(), ref2)
    # example 1 needs >= 10 tokens; "s" at the very start counts as a word, elsewhere it does not
    words, n = ([" bikes", " park", "s"], 11) if which == 1 else ([" girl", " dog", " the", "s", "."], 7)
    sub = enc(words)
    torch.manual_seed(which)
    lw = torch.full((2, n, V), float("-inf"), dtype = torch.float64)
    lw[:, :, sub] = torch.randn(2, n, len(sub), dtype = torch.float64)
    xs_all = torch.tensor(list(itertools.product(sub, repeat = n)))                 # [N, n]
    ok = torch.tensor([ref_fn(x) for x in xs_all.tolist()])
    scores = lw[:, torch.arange(n)[None, :], xs_all].sum(dim = 2)                     # [2, N]
    ref = torch.logsumexp(scores[:, ok], dim = 1)
    assert ok.any() and not ok.all()
    assert torch.allclose(c.wmc(lw), ref, atol = 1e-9)
    xs = c.sample(lw, 200, generator = torch.Generator().manual_seed(0))
    for b in range(2):
        for x in xs[b].tolist():
            assert set(x) <= set(sub) and ref_fn(x)


def test_matcher_through_the_composite():
    c = example2()
    m = c.matcher()
    for w in [" a", " girl", " walking", " with", " her"]:
        assert m.advance(TOK[w])
    assert not m.is_accepting()
    # 5 words so far: 2 more tokens can only finish if one is a pet word that is also a word
    allowed = m.allowed_next(remaining = 2)
    assert allowed[TOK[" dog"]] and allowed[TOK[" cats"]] and allowed[TOK[" the"]]
    assert not allowed[TOK["s"]] and not allowed[TOK["."]]
    assert m.advance(TOK[" dog"]) and m.advance(TOK[" in"]) and m.is_accepting()
