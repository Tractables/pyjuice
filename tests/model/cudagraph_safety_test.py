"""
Recorded CUDA graphs of the inner layers must not outlive a change their key cannot see.
"""

import pytest
import torch

import pyjuice as juice


def test_partial_evaluation_drops_recorded_graphs():
    """A graph recorded under partial evaluation computes only part of the circuit, and nothing in its key
    says so. The next plain forward after `disable_partial_evaluation()` used to replay it: log-likelihoods
    off by up to 38 nats."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 16, num_latents = 64, num_emits = 10, homogeneous = False)
    pc = juice.compile(ns, verbose = False).to(device)
    data = torch.randint(0, 10, (32, 16), device = device)
    full = pc(data, apply_cudagraph = False).clone()

    pc.enable_partial_evaluation(scopes = list(range(8)), forward = True)
    pc(data, record_cudagraph = True)
    pc.disable_partial_evaluation()
    pc.node_mars.fill_(0.0)                                  # whatever a replay leaves out shows up
    assert torch.equal(pc(data), full), "a forward replayed the graph recorded under partial evaluation"

    pc(data, record_cudagraph = True)
    pc.enable_partial_evaluation(scopes = list(range(8)), forward = True)
    assert len(pc._recorded_cuda_graphs) == 0, "graphs of the whole circuit survived into partial evaluation"
