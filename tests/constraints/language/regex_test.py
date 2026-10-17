"""
:class:`pyjuice.constraints.Regex`: the token sequences whose text matches a pattern in full, as
``re.fullmatch(pattern, text, re.ASCII)``. The automaton against the reference -- `accepts`, Python's `re` on the
decoded text -- on EVERY sequence of up to 4 tokens of a small vocabulary (characters split across tokens, an empty and
a textless token, two end-of-sequence ids), for patterns an anchored leftmost-first DFA gets wrong (shared prefixes,
optional suffixes, lazy repetition) among others; on tokenizers made in-test (tests/constraints/conftest.py): matching
texts in random tokenizations, near misses, exact draws from the automaton; the padding convention; refused syntax;
identity; composition; a tokenizer in place of a vocabulary; the missing optional dependency.
"""
import random
import sys

import pytest
import torch

import pyjuice.constraints as jc

pytest.importorskip("interegular")

#: a small vocabulary: é (c3 a9) whole and split, an empty token (19), a token with no text (20), two ends (21, 22)
TOKENS = [b"a", b"b", b"ab", b"A", b"c", b"ca", b"t", b"cat", b"e", b"0", b"10", b".", b"\n", b"\xc3", b"\xa9",
          "é".encode(), b"x", b"cate", b"\xc3\xa9t", b"", None, b"<eos>", b"<pad>"]
EOS = (21, 22)
VOCAB = jc.Vocabulary(TOKENS, eos = EOS)
PATTERNS = [r"(cat|cate)", r"[0-9]+(\.[0-9]+)?", r"a+?b?", r".é.?", r"(?i)ab", r"(a|ab)(c|bcd)*|x{2,3}", r"[^a\n]*",
            r""]


def sequences(V, n):
    """Every sequence of ``n`` tokens of ``0..V-1``, [V^n, n]."""
    grids = torch.meshgrid(*[torch.arange(V)] * n, indexing = "ij")
    return torch.stack([g.reshape(-1) for g in grids], dim = 1)


def run(dfa, seqs):
    """The automaton on every row of ``seqs`` at once: bool [rows]."""
    tc, delta, accept = dfa.token_class, dfa.delta, dfa.accept
    q = torch.full((seqs.size(0),), dfa.initial, dtype = torch.long)
    for i in range(seqs.size(1)):
        q = delta[q, tc[seqs[:, i]]]
    return accept[q]


@pytest.mark.parametrize("eos_padding", [True, False])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_short_sequence(pattern, eos_padding):
    """The automaton accepts exactly the sequences of 1 to 4 tokens whose text `re.fullmatch` accepts (with the
    padding convention, or without)."""
    c = jc.Regex(pattern, VOCAB, eos_padding = eos_padding)
    dfa = c.automaton()
    for n in range(1, 5):
        seqs = sequences(VOCAB.size, n)
        want = torch.tensor([c.accepts(s) for s in seqs.tolist()])
        got = run(dfa, seqs)
        assert torch.equal(got, want), (pattern, n, seqs[got != want][:5].tolist())


def random_pattern(rng, depth = 3):
    """A random pattern of the supported constructs over the small vocabulary's characters."""
    atoms = ["a", "b", "c", "e", "t", "x", "0", "é", r"\.", ".", "[ab]", "[^a\n]", "[a-c0]", r"\d", r"\w", "[éx]",
             "[^é]", "A"]

    def node(d):
        r = rng.random()
        if d == 0 or r < 0.35:
            return rng.choice(atoms)
        if r < 0.55:
            return node(d - 1) + node(d - 1)
        if r < 0.7:
            return f"({node(d - 1)}|{node(d - 1)})"
        if r < 0.9:
            return f"(?:{node(d - 1)})" + rng.choice(["*", "+", "?", "{1,2}", "{2}", "*?", "+?", "??", "{0,2}?"])
        return rng.choice(["(?i:", "(?s:", "(?is:"]) + node(d - 1) + ")"
    return node(depth)


def test_random_patterns():
    """Random patterns of the supported constructs: the automaton accepts exactly the sequences of up to 3 tokens
    `re.fullmatch` accepts (those refused -- (?i) with é -- skipped, and few)."""
    rng = random.Random(0)
    seqs = [sequences(VOCAB.size, n) for n in range(1, 4)]
    checked = 0
    for _ in range(40):
        pattern = random_pattern(rng)
        try:
            c = jc.Regex(pattern, VOCAB)
        except ValueError:
            continue
        dfa = c.automaton()
        for s in seqs:
            want = torch.tensor([c.accepts(row) for row in s.tolist()])
            assert torch.equal(run(dfa, s), want), (pattern, s[run(dfa, s) != want][:5].tolist())
        checked += 1
    assert checked >= 30, checked


def random_tokenization(vocab, data: bytes, rng):
    """A random sequence of tokens spelling ``data`` (each step a token that is a prefix of the rest and leaves a rest
    that can still be spelled)."""
    texts = [(v, vocab.token_bytes(v)) for v in range(vocab.size) if vocab.token_bytes(v)]
    ok = [False] * len(data) + [True]
    for i in range(len(data) - 1, -1, -1):
        ok[i] = any(data.startswith(t, i) and ok[i + len(t)] for _, t in texts)
    assert ok[0]
    out, i = [], 0
    while i < len(data):
        v, t = rng.choice([(v, t) for v, t in texts if data.startswith(t, i) and ok[i + len(t)]])
        out.append(v)
        i += len(t)
    return out


CASES = [(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", ["2026-10-17", "1999-01-01"], ["2026-10-1", "2026/10/17"]),
         (r"(the|a) (quick|lazy) [a-z]+", ["the quick fox", "a lazy dog"], ["the quick", "the slow fox"]),
         (r"[^\"\\]*", ["naïve café — “quotes” …", "日本語"], ['say "hi"']),
         (r".{2,10}", ["日本語のテキスト", "emoji 🎉🚀"], ["x", "tabs\tand\nnewlines"]),
         (r"emoji .+ done", ["emoji 🎉🚀 done"], ["emoji done"])]


@pytest.mark.parametrize("kind", ["byte_level", "sentencepiece"])
def test_tokenizers(kind, request):
    """Matching texts are accepted in random tokenizations (padded with ends), near misses in none; the automaton's
    exact uniform draws all match."""
    tok = request.getfixturevalue(kind)()
    end = tok.token_to_id("<|endoftext|>" if kind == "byte_level" else "</s>")
    vocab = jc.Vocabulary.from_tokenizer(tok, eos = [end])
    rng = random.Random(0)
    g = torch.Generator().manual_seed(0)
    for pattern, good, bad in CASES:
        c = jc.Regex(pattern, vocab)
        dfa = c.automaton()
        for texts, want in ((good, True), (bad, False)):
            for text in texts:
                for _ in range(3):
                    ids = random_tokenization(vocab, text.encode(), rng) + [end] * 2
                    assert c.accepts(ids) == want and dfa.accepts(ids) == want, (pattern, text, ids)
        for row in dfa.sample(torch.zeros(1, 24, vocab.size), 16, generator = g)[0].tolist():
            assert c.accepts(row), (pattern, row)


def test_padding():
    """Text, then ends only, mixed ends included; nothing but ends after the first; no padding: no end at all; a
    vocabulary without ends needs `eos_padding = False`."""
    c, exact = jc.Regex("ab", VOCAB), jc.Regex("ab", VOCAB, eos_padding = False)
    for seq, padded, plain in [([0, 1], True, True), ([2, 21, 22, 21], True, False), ([0, 1, 21, 21], True, False),
                               ([0, 21, 1, 21], False, False), ([21, 0, 1], False, False), ([2, 19, 21], True, False),
                               ([2, 20], False, False), ([21, 22], False, False)]:
        assert c.accepts(seq) == c.automaton().accepts(seq) == padded, seq
        assert exact.accepts(seq) == exact.automaton().accepts(seq) == plain, seq
    no_ends = jc.Vocabulary([b"a", b"b"])
    with pytest.raises(ValueError, match = "end-of-sequence"):
        jc.Regex("ab", no_ends)
    assert jc.Regex("ab", no_ends, eos_padding = False).automaton().accepts([0, 1])


@pytest.mark.parametrize("pattern", [r"^a", r"a$", r"\bcat", r"(a)\1", r"(?m)a", r"a(?m:b)", r"(?x)a b", r"\p{L}",
                                     r"(a", r"a{2,1}", r"(?=a)[abc]*", r"a(?=b)", r"(x(?!a))*", r"b|(?<=a)b", "(?i)é",
                                     r"(?i:\xdf)", "(?i)[a-z]|ä", r"a++a", r"(?>a|ab)c", r"[]a]", r"[^]a]",
                                     r"(?#note)a", r"(?i:(?-i:a)b)"])
def test_refused(pattern):
    """Anchors, \\b, backreferences, (?m), (?x), \\p{} and malformed patterns; lookarounds anywhere in the pattern
    (interegular's automata for some are wrong: (?=a)[abc]* accepting "aA"); (?i) with a character outside ASCII
    (interegular folds those by Unicode, and ß upper-cases to two characters); possessive repetition (interegular's
    a++a accepts "aa"), atomic groups; a ']' first in a class (which interegular reads otherwise); what interegular
    fails on (a comment)."""
    with pytest.raises(ValueError, match = "Unsupported regular expression"):
        jc.Regex(pattern, VOCAB)
    with pytest.raises(TypeError):
        jc.Regex(b"ab", VOCAB)


def test_identity_and_capabilities():
    """Equal pattern, vocabulary and padding: equal constraints; any of them different: different; the automaton is
    built once; every capability."""
    c = jc.Regex("a+b", VOCAB)
    assert c == jc.Regex("a+b", VOCAB) and c.fingerprint() == jc.Regex("a+b", VOCAB).fingerprint()
    for other in (jc.Regex("a*b", VOCAB), jc.Regex("a+b", VOCAB, eos_padding = False),
                  jc.Regex("a+b", VOCAB.with_size(VOCAB.size + 1))):
        assert other != c
    assert c.automaton() is c.automaton()
    assert c.capabilities() == {"accepts", "matcher", "wmc", "sample", "automaton", "relax"}
    assert c.info()["pattern"] == "a+b" and c.vocab_size == VOCAB.size


@pytest.mark.parametrize("built", [False, True])
def test_widened(built):
    """Over a larger vocabulary (what `jc.compile` does for a PC with more values): the same Regex over the vocabulary
    with the extra ids, which are never allowed -- from scratch, or from the automaton already built; composites
    widen; a DFA does not."""
    vocab = jc.Vocabulary([b"a", b"b", b"ab", None], eos = [3])
    c = jc.Regex("a+b?", vocab)
    if built:
        c.automaton()
    wide = c._widened(5)
    assert wide.vocab_size == 5 and wide == jc.Regex("a+b?", vocab.with_size(5))
    m = wide.automaton().matcher()
    for t in (None, 0, 0, 1):
        if t is not None:
            assert m.advance(t)
        assert not bool(m.allowed_next()[4])
    assert (c & ~jc.Regex("ab", vocab))._widened(5).vocab_size == 5
    assert jc.DFA.anything(4)._widened(5) is None and (c & jc.DFA.anything(4))._widened(5) is None


def test_composition():
    """With a token-level DFA: the product accepts what both accept (every sequence of up to 3 tokens)."""
    c = jc.Regex("[a-z]+", VOCAB) & ~jc.DFA.contains([[7]], VOCAB.size)          # never the token "cat"
    dfa = c.automaton()
    for n in range(1, 4):
        seqs = sequences(VOCAB.size, n)
        assert torch.equal(run(dfa, seqs), torch.tensor([c.accepts(s) for s in seqs.tolist()]))


def test_a_tokenizer_for_a_vocabulary(byte_level):
    tok = byte_level()
    transformers = pytest.importorskip("transformers")
    wrapped = transformers.PreTrainedTokenizerFast(tokenizer_object = tok, eos_token = "<|endoftext|>")
    assert jc.Regex("[a-z]+", wrapped) == jc.Regex("[a-z]+", jc.Vocabulary.from_tokenizer(wrapped))


def test_without_interegular(monkeypatch):
    monkeypatch.setitem(sys.modules, "interegular", None)
    with pytest.raises(ImportError, match = r"pyjuice\[constraints\]"):
        jc.Regex("ab", VOCAB)
