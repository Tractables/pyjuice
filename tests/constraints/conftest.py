"""Fixtures shared by every test of :mod:`pyjuice.constraints`."""
import pytest
import torch


@pytest.fixture(autouse = True)
def nan_fresh_storage(monkeypatch):
    """The dfa_exact backend's new storage filled with NaN (it is not filled in use, see ``state.FRESH_FILL``): a
    query reading a slot no kernel wrote gives NaN, and fails the test."""
    from pyjuice.constraints.backends.dfa_exact import state
    monkeypatch.setattr(state, "FRESH_FILL", float("nan"))


def word_kinds(strings, separators, special_ids = ()) -> torch.Tensor:
    """
    Every token's kind for :func:`word_count` from its text as it appears mid-sequence (``strings[v]``):
    ``2 * (starts with a separator) + (contains a letter or digit)``, 0 for the ``special_ids`` and tokens of no
    text -- Ctrl-G's ``WordCountBuilder`` classification.
    """
    separators = set(separators)
    special = set(special_ids)
    kinds = torch.zeros(len(strings), dtype = torch.long)
    for v, text in enumerate(strings):
        if v in special or len(text) == 0:
            continue
        has_word_char = any(ch.isalpha() or ch.isdigit() for ch in text)
        kinds[v] = 2 * (text[0] in separators) + has_word_char
    return kinds


def word_count(lo: int, hi: int, token_kind):
    """
    A :class:`~pyjuice.constraints.DFA` of the sequences of ``lo`` to ``hi`` words (inclusive), word boundaries read
    off every token's kind (:func:`word_kinds`)::

        token_kind[v] = 2 * (v starts with a separator) + (v contains a letter or digit)

    A word starts at a token with a letter or digit that begins with a separator or follows one; the sequence is
    taken as preceded by a separator -- Ctrl-G's ``WordCountBuilder``.

    :param token_kind: int tensor of size [vocab_size] with values in {0, 1, 2, 3}
    """
    from pyjuice.constraints import DFA
    token_kind = torch.as_tensor(token_kind, dtype = torch.long)
    if not 0 <= lo <= hi:
        raise ValueError(f"Need 0 <= lo <= hi, got lo={lo}, hi={hi}.")
    if token_kind.dim() != 1 or token_kind.min() < 0 or token_kind.max() > 3:
        raise ValueError("`token_kind` must be a 1-D tensor with values in {0, 1, 2, 3}.")
    # state 2k + s: k words so far, s = 1 if the previous token ended at a separator; 2(hi+1) = too many
    sink = 2 * (hi + 1)
    word = lambda k: 2 * (k + 1) if k + 1 <= hi else sink
    delta = []
    for k in range(hi + 1):
        # kinds: 0 = no sep / no word char, 1 = no sep / word char, 2 = sep / no word char, 3 = sep / word char
        delta.append([2 * k, 2 * k, 2 * k + 1, word(k)])            # s = 0: continue unless a separator
        delta.append([2 * k + 1, word(k), 2 * k + 1, word(k)])      # s = 1: any word char starts a word
    delta.append([sink] * 4)
    accept = [2 * k + s for k in range(lo, hi + 1) for s in (0, 1)]
    return DFA(token_kind.numel(), token_kind, delta, 1, accept).minimize()


#: what :func:`byte_level_tokenizer` is trained on: several scripts, emoji, whitespace
TOKENIZER_CORPUS = ["the quick brown fox", " leading space", "naïve café — “quotes” …", "日本語のテキスト",
                    "emoji 🎉🚀 done", "tabs\tand\nnewlines", '{"k": [1, 2.5, null]}'] * 10 + \
                   ["the lazy dog jumps over the fox"] * 10


def byte_level_tokenizer(tk):
    """A ``tokenizers.Tokenizer`` in GPT-2's layout (``tk``: the ``tokenizers`` module): byte-level BPE over all 256
    bytes, an end-of-text special token (id 0)."""
    tok = tk.Tokenizer(tk.models.BPE())
    tok.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space = False)
    tok.decoder = tk.decoders.ByteLevel()
    trainer = tk.trainers.BpeTrainer(vocab_size = 400, special_tokens = ["<|endoftext|>"], show_progress = False,
                                     initial_alphabet = tk.pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(TOKENIZER_CORPUS, trainer)
    return tok


def sentencepiece_tokenizer(tk):
    """Llama 2's layout: <unk> <s> </s>, the 256 byte tokens (byte fallback), then pieces; a space marker prepended to
    the text."""
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


def unigram_tokenizer(tk):
    """T5's layout: a SentencePiece unigram without byte fallback (<unk> 0, </s> 1)."""
    pieces = [("<unk>", 0.0), ("</s>", 0.0), ("▁", -2.0), ("▁the", -1.0), ("▁fox", -1.0)] + \
             [(c, -3.0) for c in "abcdefghijklmnopqrstuvwxyz"]
    tok = tk.Tokenizer(tk.models.Unigram(pieces, unk_id = 0))
    tok.pre_tokenizer = tk.pre_tokenizers.Metaspace()
    tok.decoder = tk.decoders.Metaspace()
    tok.add_special_tokens(["<unk>", "</s>"])
    return tok


@pytest.fixture(name = "tk")
def tk_fixture():
    """The ``tokenizers`` module (the test is skipped without it)."""
    return pytest.importorskip("tokenizers")


@pytest.fixture(name = "byte_level")
def byte_level_fixture(tk):
    """A builder of fresh :func:`byte_level_tokenizer` s."""
    return lambda: byte_level_tokenizer(tk)


@pytest.fixture(name = "sentencepiece")
def sentencepiece_fixture(tk):
    """A builder of fresh :func:`sentencepiece_tokenizer` s."""
    return lambda: sentencepiece_tokenizer(tk)


@pytest.fixture(name = "unigram")
def unigram_fixture(tk):
    """A builder of fresh :func:`unigram_tokenizer` s."""
    return lambda: unigram_tokenizer(tk)


@pytest.fixture(name = "word_kinds")
def word_kinds_fixture():
    """:func:`word_kinds`, for tests."""
    return word_kinds


@pytest.fixture(name = "word_count")
def word_count_fixture():
    """:func:`word_count`, for tests."""
    return word_count
