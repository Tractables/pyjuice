import pyjuice as juice
import torch

import pytest


def test_param_normalization():
    
    device = torch.device("cuda:0")

    root_ns = juice.structures.HMM(
        seq_length = 16,
        num_latents = 512,
        num_emits = 2378
    )
    root_ns.init_parameters(perturbation = 32.0)

    params = root_ns.chs[0].chs[1].get_source_ns().get_params(as_matrix = True)
    mask = torch.rand(512) < 0.8
    params[:,mask] = 0.0

    root_ns.chs[0].chs[1].set_params(params)

    pc = juice.compile(root_ns)
    pc.to(device)

    assert not torch.any(torch.isnan(pc.params))


if __name__ == "__main__":
    test_param_normalization()
