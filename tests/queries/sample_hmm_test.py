import pyjuice as juice
import torch
import torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_sample_hmm_correctness():
    
    device = torch.device("cuda:0")

    seq_length = 4
    num_latents = 256
    num_emits = 5

    ns = juice.structures.HMM(
        seq_length = seq_length, 
        num_latents = num_latents, 
        num_emits = num_emits, 
        homogeneous = True
    )
    ns.init_parameters(perturbation = 16.0)

    pc = juice.compile(ns)
    pc.to(device)

    samples = juice.queries.sample(pc, num_samples = 16)

    assert ((samples >= 0) & (samples < 100)).all()


if __name__ == "__main__":
    test_sample_hmm_correctness()
