import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer, ProdLayer, SumLayer
from pyjuice.model.backend import eval_partition_fn

import pytest


def test_zero_param_preserving_col():

    device = torch.device("cuda:0")

    root_ns = juice.structures.HMM(
        seq_length = 32,
        num_latents = 128,
        num_emits = 256
    )

    root_ns.init_parameters()
    alpha = root_ns.chs[0].chs[1].get_source_ns()._params.clone().reshape(128, 128)
    alpha[:,0] = 0.0
    root_ns.chs[0].chs[1].set_params(alpha)

    pc = juice.compile(root_ns)
    pc.to(device)

    x = torch.randint(0, 256, [64, 32], device = device)

    lls = pc(x)
    pc.backward(x, logspace_flows = True, allow_modify_flows = False)

    pc.mini_batch_em(step_size = 1.0, keep_zero_params = True)

    pc.update_parameters()

    alpha_new = root_ns.chs[0].chs[1].get_source_ns()._params.clone().reshape(128, 128)

    assert torch.all((alpha > 1e-12) | (alpha_new < 1e-12))

    pc.mini_batch_em(step_size = 0.1, step_size_rescaling = True, keep_zero_params = True)

    pc.update_parameters()

    alpha_new = root_ns.chs[0].chs[1].get_source_ns()._params.clone().reshape(128, 128)

    assert torch.all((alpha > 1e-12) | (alpha_new < 1e-12))


def test_zero_param_preserving_rand():

    device = torch.device("cuda:0")

    root_ns = juice.structures.HMM(
        seq_length = 32,
        num_latents = 128,
        num_emits = 256
    )

    root_ns.init_parameters()
    alpha = root_ns.chs[0].chs[1].get_source_ns()._params.clone().reshape(128, 128)
    while True:
        mask = torch.rand(128, 128) < 0.1
        if torch.all(mask.long().sum(dim = 1) > 0):
            break
    alpha[mask] = 0.0
    root_ns.chs[0].chs[1].set_params(alpha)

    pc = juice.compile(root_ns)
    pc.to(device)

    x = torch.randint(0, 256, [64, 32], device = device)

    lls = pc(x)
    pc.backward(x, logspace_flows = True, allow_modify_flows = False)

    pc.mini_batch_em(step_size = 1.0, keep_zero_params = True)

    pc.update_parameters()

    alpha_new = root_ns.chs[0].chs[1].get_source_ns()._params.clone().reshape(128, 128)

    assert torch.all((alpha > 1e-12) | (alpha_new < 1e-12))

    pc.mini_batch_em(step_size = 0.1, step_size_rescaling = True, keep_zero_params = True)

    pc.update_parameters()

    alpha_new = root_ns.chs[0].chs[1].get_source_ns()._params.clone().reshape(128, 128)

    assert torch.all((alpha > 1e-12) | (alpha_new < 1e-12))


if __name__ == "__main__":
    test_zero_param_preserving_col()
    test_zero_param_preserving_rand()
