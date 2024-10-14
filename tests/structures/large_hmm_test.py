import pyjuice as juice
import pyjuice.nodes.distributions as dists
import torch
import time

import pytest


@pytest.mark.slow
def test_large_hmm():

    device = torch.device("cuda:0")

    seq_length = 16
    vocab_size = 1024
    num_latents = 65536

    root_ns = juice.structures.HMM(
        seq_length = seq_length,
        num_latents = num_latents,
        num_emits = vocab_size,
        homogeneous = True
    )

    pc = juice.compile(root_ns)
    pc.print_statistics()

    pc.to(device)

    data = torch.randint(0, vocab_size, (64, seq_length)).to(device)

    lls = pc(data, propagation_alg = "LL")
    pc.backward(data, flows_memory = 1.0, allow_modify_flows = False,
                propagation_alg = "LL", logspace_flows = True)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    test_large_hmm()