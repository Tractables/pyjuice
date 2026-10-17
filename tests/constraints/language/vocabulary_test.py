"""
:class:`pyjuice.constraints.Vocabulary`: the bytes every token id spells. Built from byte strings (sizes, end-of-sequence
ids, decoding, fingerprints), and read from Hugging Face tokenizers of every supported kind, made in-test with the
``tokenizers`` library (no download; tests/constraints/conftest.py) -- byte-level BPE, SentencePiece BPE with byte
fallback, SentencePiece unigram: the tokens of every text spell it back byte for byte; special and unknown tokens have
no text; WordPiece, suffix BPE and tokenizers without a decoder are refused.
"""
import pytest
import torch

import pyjuice.constraints as jc

TEXTS = ["the quick brown fox", " leading space", "naïve café — “quotes” …", "日本語のテキスト", "emoji 🎉🚀 done",
         "tabs\tand\nnewlines", '{"k": [1, 2.5, null]}']


def test_built_from_bytes():
    """Texts per id, None for none, extra ids past the list without text, and the end-of-sequence ids textless
    whatever bytes are given for them."""
    vocab = jc.Vocabulary([b"a", b"bc", None, b"", b"\xff", b"<eos>"], eos = [5, 5], size = 8)
    assert vocab.size == len(vocab) == 8 and vocab.eos == (5,)
    assert [vocab.token_bytes(v) for v in range(8)] == [b"a", b"bc", None, b"", b"\xff", None, None, None]
    assert vocab.has_text.tolist() == [True, True, False, True, True, False, False, False]
    with pytest.raises(ValueError):
        vocab.token_bytes(8)
    with pytest.raises(ValueError):
        jc.Vocabulary([b"a", b"b"], size = 1)
    with pytest.raises(ValueError):
        jc.Vocabulary([b"a"], eos = [1])
    with pytest.raises(ValueError):
        jc.Vocabulary([])
    with pytest.raises(TypeError):
        jc.Vocabulary(["a"])


def test_decode():
    """A sequence's text is its tokens' bytes concatenated; an id without text (end of sequence included) never
    occurs inside text."""
    vocab = jc.Vocabulary([b"a", b"bc", None, b"", b"\xff"], eos = [2], size = 6)
    assert vocab.decode([0, 1, 3, 4, 0]) == b"abc\xffa"
    assert vocab.decode(torch.tensor([1, 1])) == b"bcbc"
    assert vocab.decode([]) == b""
    for bad in ([0, 2], [5], [6], [-1]):
        with pytest.raises(ValueError):
            vocab.decode(bad)


def test_fingerprint():
    """Equal contents, equal fingerprints (and ==, hash); one byte, the end-of-sequence ids, the size, or text vs no
    text differ."""
    make = lambda tb = (b"a", b"bc", None, None), eos = (2,), size = None: jc.Vocabulary(list(tb), eos = eos, size = size)
    base = make()
    assert make() == base and make().fingerprint() == base.fingerprint() and len({base, make()}) == 1
    for other in (make(tb = (b"a", b"bd", None, None)), make(eos = ()), make(eos = (3,)), make(size = 5),
                  make(tb = (b"a", b"bc", None, b""))):
        assert other != base and other.fingerprint() != base.fingerprint()


@pytest.mark.parametrize("wrapped", [False, True])
def test_byte_level_tokenizer(byte_level, wrapped):
    """Byte-level BPE, raw and as a `transformers` fast tokenizer: every byte a token of its own, every text spelled
    back exactly; the end-of-text token has no text and is the end of sequence the wrapper declares."""
    tok = byte_level()
    eos = tok.token_to_id("<|endoftext|>")
    source = tok
    if wrapped:
        transformers = pytest.importorskip("transformers")
        source = transformers.PreTrainedTokenizerFast(tokenizer_object = tok, eos_token = "<|endoftext|>")
    vocab = jc.Vocabulary.from_tokenizer(source)
    assert vocab.size == tok.get_vocab_size(with_added_tokens = True)
    assert vocab.eos == ((eos,) if wrapped else ())
    assert vocab.token_bytes(eos) is None
    assert {vocab.token_bytes(v) for v in range(vocab.size)} >= {bytes([b]) for b in range(256)}
    for text in TEXTS:
        assert vocab.decode(tok.encode(text).ids) == text.encode()
    if wrapped:
        assert vocab == jc.Vocabulary.from_tokenizer(tok, eos = [eos])


def test_sentencepiece_byte_fallback(sentencepiece):
    """SentencePiece BPE with byte fallback: the marker is a space, <0xNN> the byte NN, characters outside the
    pieces spelled through the byte tokens; <unk> <s> </s> have no text."""
    tok = sentencepiece()
    vocab = jc.Vocabulary.from_tokenizer(tok)
    assert all(vocab.token_bytes(v) is None for v in (0, 1, 2))
    assert all(vocab.token_bytes(3 + b) == bytes([b]) for b in range(256))
    assert vocab.token_bytes(tok.token_to_id("▁the")) == b" the"
    for text in TEXTS:                                                  # (the space marker prepended to the text)
        assert vocab.decode(tok.encode(text).ids) == (" " + text).encode()


def test_sentencepiece_unigram(unigram):
    """SentencePiece unigram without byte fallback: the marker is a space; a character outside the pieces is the
    unknown token, which has no text."""
    tok = unigram()
    vocab = jc.Vocabulary.from_tokenizer(tok)
    assert vocab.token_bytes(0) is None and vocab.token_bytes(1) is None
    assert vocab.token_bytes(tok.token_to_id("▁fox")) == b" fox"
    assert vocab.decode(tok.encode("the fox jumps").ids) == b" the fox jumps"
    ids = tok.encode("the café").ids
    assert 0 in ids
    with pytest.raises(ValueError):
        vocab.decode(ids)


def test_added_tokens(byte_level):
    """An added token spells its content; a special one has no text."""
    tok = byte_level()
    tok.add_tokens(["<tool_call>"])
    tok.add_special_tokens(["<|eot_id|>"])
    vocab = jc.Vocabulary.from_tokenizer(tok)
    assert vocab.token_bytes(tok.token_to_id("<tool_call>")) == b"<tool_call>"
    assert vocab.token_bytes(tok.token_to_id("<|eot_id|>")) is None
    assert vocab.decode(tok.encode("a <tool_call> b").ids) == b"a <tool_call> b"


def test_size_and_eos_override(byte_level):
    """A model's embedding beyond its tokenizer: ids past the tokenizer's have no text; the end-of-sequence ids given
    replace the tokenizer's (and lose their text); a size below the tokenizer's is refused; `with_size` widens."""
    tok = byte_level()
    n = tok.get_vocab_size(with_added_tokens = True)
    vocab = jc.Vocabulary.from_tokenizer(tok, size = n + 5, eos = [3, n + 1])
    assert vocab.size == n + 5 and vocab.eos == (3, n + 1)
    assert not vocab.has_text[n:].any() and vocab.token_bytes(3) is None
    with pytest.raises(ValueError):
        jc.Vocabulary.from_tokenizer(tok, size = n - 1)
    plain = jc.Vocabulary.from_tokenizer(tok, eos = [3, n + 1], size = n + 2)
    assert plain.with_size(n + 5) == vocab and plain.with_size(n + 2) is plain
    with pytest.raises(ValueError):
        plain.with_size(n + 1)


def test_unsupported_tokenizers_are_refused(tk):
    """WordPiece (spaces that depend on the neighbouring tokens), suffix BPE, no decoder: not read; neither is
    anything that is not a Hugging Face fast tokenizer."""
    wordpiece = tk.Tokenizer(tk.models.WordPiece({"[UNK]": 0, "the": 1, "fox": 2, "##es": 3}, unk_token = "[UNK]"))
    wordpiece.decoder = tk.decoders.WordPiece()
    suffix = tk.Tokenizer(tk.models.BPE({"a</w>": 0, "b": 1}, []))
    suffix.decoder = tk.decoders.BPEDecoder(suffix = "</w>")
    plain = tk.Tokenizer(tk.models.WordLevel({"a": 0, "[UNK]": 1}, unk_token = "[UNK]"))
    for tok in (wordpiece, suffix, plain):
        with pytest.raises(NotImplementedError):
            jc.Vocabulary.from_tokenizer(tok)
    with pytest.raises(TypeError):
        jc.Vocabulary.from_tokenizer(object())
