"""
:class:`pyjuice.constraints.Vocabulary`: the bytes every token id spells. Built from byte strings (sizes, end-of-sequence
ids, decoding, fingerprints), and read from Hugging Face tokenizers of every supported kind, made in-test with the
``tokenizers`` library (no download) -- byte-level BPE, SentencePiece BPE with byte fallback, SentencePiece unigram:
the tokens of every text spell it back byte for byte; special and unknown tokens have no text; WordPiece, suffix BPE
and tokenizers without a decoder are refused.
"""
import pytest
import torch

import pyjuice.constraints as jc

TEXTS = ["the quick brown fox", " leading space", "naïve café — “quotes” …", "日本語のテキスト", "emoji 🎉🚀 done",
         "tabs\tand\nnewlines", '{"k": [1, 2.5, null]}']
CORPUS = TEXTS * 10 + ["the lazy dog jumps over the fox"] * 10


@pytest.fixture(name = "tk")
def tk_fixture():
    return pytest.importorskip("tokenizers")


def byte_level(tk):
    """GPT-2's layout: byte-level BPE over all 256 bytes, an end-of-text special token."""
    tok = tk.Tokenizer(tk.models.BPE())
    tok.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space = False)
    tok.decoder = tk.decoders.ByteLevel()
    trainer = tk.trainers.BpeTrainer(vocab_size = 400, special_tokens = ["<|endoftext|>"], show_progress = False,
                                     initial_alphabet = tk.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    return tok


def sentencepiece_bpe(tk):
    """Llama 2's layout: <unk> <s> </s>, the 256 byte tokens, then pieces; a space marker prepended to the text."""
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"<0x{b:02X}>": 3 + b for b in range(256)}}
    for p in ["▁", "t", "h", "e", "▁t", "he", "▁the", "f", "o", "x", "▁f", "ox", "▁fox"]:
        vocab.setdefault(p, len(vocab))
    merges = [("▁", "t"), ("h", "e"), ("▁t", "he"), ("▁", "f"), ("o", "x"), ("▁f", "ox")]
    tok = tk.Tokenizer(tk.models.BPE(vocab = vocab, merges = merges, unk_token = "<unk>", byte_fallback = True,
                                     fuse_unk = True))
    tok.normalizer = tk.normalizers.Sequence([tk.normalizers.Prepend("▁"), tk.normalizers.Replace(" ", "▁")])
    tok.decoder = tk.decoders.Sequence([tk.decoders.Replace("▁", " "), tk.decoders.ByteFallback(),
                                        tk.decoders.Fuse(), tk.decoders.Strip(" ", 1, 0)])
    tok.add_special_tokens(["<unk>", "<s>", "</s>"])
    return tok


def unigram(tk):
    """T5's layout: a SentencePiece unigram without byte fallback."""
    pieces = [("<unk>", 0.0), ("</s>", 0.0), ("▁", -2.0), ("▁the", -1.0), ("▁fox", -1.0)] + \
             [(c, -3.0) for c in "abcdefghijklmnopqrstuvwxyz"]
    tok = tk.Tokenizer(tk.models.Unigram(pieces, unk_id = 0))
    tok.pre_tokenizer = tk.pre_tokenizers.Metaspace()
    tok.decoder = tk.decoders.Metaspace()
    tok.add_special_tokens(["<unk>", "</s>"])
    return tok


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
def test_byte_level_tokenizer(tk, wrapped):
    """Byte-level BPE, raw and as a `transformers` fast tokenizer: every byte a token of its own, every text spelled
    back exactly; the end-of-text token has no text and is the end of sequence the wrapper declares."""
    tok = byte_level(tk)
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


def test_sentencepiece_byte_fallback(tk):
    """SentencePiece BPE with byte fallback: the marker is a space, <0xNN> the byte NN, characters outside the
    pieces spelled through the byte tokens; <unk> <s> </s> have no text."""
    tok = sentencepiece_bpe(tk)
    vocab = jc.Vocabulary.from_tokenizer(tok)
    assert all(vocab.token_bytes(v) is None for v in (0, 1, 2))
    assert all(vocab.token_bytes(3 + b) == bytes([b]) for b in range(256))
    assert vocab.token_bytes(tok.token_to_id("▁the")) == b" the"
    for text in TEXTS:                                                  # (the space marker prepended to the text)
        assert vocab.decode(tok.encode(text).ids) == (" " + text).encode()


def test_sentencepiece_unigram(tk):
    """SentencePiece unigram without byte fallback: the marker is a space; a character outside the pieces is the
    unknown token, which has no text."""
    tok = unigram(tk)
    vocab = jc.Vocabulary.from_tokenizer(tok)
    assert vocab.token_bytes(0) is None and vocab.token_bytes(1) is None
    assert vocab.token_bytes(tok.token_to_id("▁fox")) == b" fox"
    assert vocab.decode(tok.encode("the fox jumps").ids) == b" the fox jumps"
    ids = tok.encode("the café").ids
    assert 0 in ids
    with pytest.raises(ValueError):
        vocab.decode(ids)


def test_added_tokens(tk):
    """An added token spells its content; a special one has no text."""
    tok = byte_level(tk)
    tok.add_tokens(["<tool_call>"])
    tok.add_special_tokens(["<|eot_id|>"])
    vocab = jc.Vocabulary.from_tokenizer(tok)
    assert vocab.token_bytes(tok.token_to_id("<tool_call>")) == b"<tool_call>"
    assert vocab.token_bytes(tok.token_to_id("<|eot_id|>")) is None
    assert vocab.decode(tok.encode("a <tool_call> b").ids) == b"a <tool_call> b"


def test_size_and_eos_override(tk):
    """A model's embedding beyond its tokenizer: ids past the tokenizer's have no text; the end-of-sequence ids given
    replace the tokenizer's (and lose their text); a size below the tokenizer's is refused."""
    tok = byte_level(tk)
    n = tok.get_vocab_size(with_added_tokens = True)
    vocab = jc.Vocabulary.from_tokenizer(tok, size = n + 5, eos = [3, n + 1])
    assert vocab.size == n + 5 and vocab.eos == (3, n + 1)
    assert not vocab.has_text[n:].any() and vocab.token_bytes(3) is None
    with pytest.raises(ValueError):
        jc.Vocabulary.from_tokenizer(tok, size = n - 1)


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
