"""
Autoregressive decoding from an HMM under a JSON Schema, and under a regular expression.

Both constrain the TEXT the tokens spell. A :class:`~pyjuice.constraints.Vocabulary` gives every token id its bytes --
here a small one made in-test (every printable ASCII character, a few longer pieces, an end-of-sequence id); with a
real tokenizer, ``jc.Vocabulary.from_tokenizer(tokenizer)``. The text may end before the last position: the rest are
end-of-sequence ids. Each token is sampled from p(x_t | x_<t, C), so every sequence drawn spells a valid text.
"""
import json
import re

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc

pytest.importorskip("interegular")                     # (Regex and JSONSchema read patterns with it)

#: every printable ASCII character, a few longer pieces, and an end-of-sequence id (it spells no text)
TOKENS = [bytes([b]) for b in range(32, 127)] + [b'{"', b'":', b'","', b'"}', b"name", b"20", None]
EOS = len(TOKENS) - 1
VOCAB = jc.Vocabulary(TOKENS, eos = [EOS])


def decode(c, n, B = 16):
    """``B`` sequences of ``n`` tokens, each token sampled from p(x_t | x_<t, C) of an HMM over the vocabulary."""
    torch.manual_seed(0)
    pc = juice.compile(juice.structures.HMM(seq_length = n, num_latents = 64, num_emits = VOCAB.size)).to("cuda")
    cc = jc.compile(c, pc)
    x = torch.zeros(B, n, dtype = torch.long, device = "cuda")
    with cc.decoder() as decoder:
        for t in range(n):
            logp = decoder.conditional(x[:, :t], targets = [t])     # log p(x_t = v | x_<t, C): [B, V]
            x[:, t] = torch.distributions.Categorical(logits = logp).sample()
    return x.tolist()


def text(seq):
    """The text a sequence spells: its tokens before the first end-of-sequence id."""
    end = seq.index(EOS) if EOS in seq else len(seq)
    return VOCAB.decode(seq[:end]).decode("utf-8")


def test_json_schema_decoding():
    schema = {"type": "object",
              "properties": {"name": {"type": "string", "maxLength": 8},
                             "age": {"type": "integer"},
                             "role": {"enum": ["admin", "user"]}},
              "required": ["name", "role"]}
    c = jc.JSONSchema(schema, VOCAB)                   # (an optional space around the separators)
    for seq in decode(c, n = 32):
        value = json.loads(text(seq))                  # e.g. {"name": "Kq", "age": 30, "role": "user"}
        assert list(value) in (["name", "role"], ["name", "age", "role"])      # the schema's order
        assert isinstance(value["name"], str) and len(value["name"]) <= 8
        assert isinstance(value.get("age", 0), int) and value["role"] in ("admin", "user")
        assert c.accepts(seq)


def test_regex_decoding():
    date = r"(19|20)\d\d-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"         # YYYY-MM-DD
    c = jc.Regex(date, VOCAB)
    for seq in decode(c, n = 12):
        assert re.fullmatch(date, text(seq))           # e.g. 2031-07-29
        assert c.accepts(seq)
