"""
Autoregressive decoding from an HMM under a constraint.

Each token is sampled from p(x_t | x_<t, C): the HMM's next-token distribution given the tokens so far and the
constraint C. Token by token, this draws whole sequences from p(x | C), so every sequence satisfies C. The decoder
session reuses work between the steps.
"""
import torch

import pyjuice as juice
import pyjuice.constraints as jc


def test_hmm_autoregressive_decoding():
    torch.manual_seed(0)
    n, V, B = 12, 32, 16                       # sequence length, vocabulary size, number of sequences

    pc = juice.compile(juice.structures.HMM(seq_length = n, num_latents = 64, num_emits = V)).to("cuda")

    # the constraint: tokens 3 and 7 appear in a row, and token 5 never appears
    c = jc.DFA.contains([[3, 7]], vocab_size = V) & ~jc.DFA.contains([[5]], vocab_size = V)
    cc = jc.compile(c, pc)

    x = torch.zeros(B, n, dtype = torch.long, device = "cuda")
    with cc.decoder() as decoder:
        for t in range(n):
            logp = decoder.conditional(x[:, :t], targets = [t])     # log p(x_t = v | x_<t, C): [B, V]
            x[:, t] = torch.distributions.Categorical(logits = logp).sample()    # (greedy: logp.argmax(dim = 1))

    assert all(c.accepts(seq) for seq in x.tolist())
