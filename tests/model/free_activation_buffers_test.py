"""
`TensorCircuit.free_activation_buffers` releases the buffers a pass fills (and their storage and CUDA
graphs), keeps the parameter flows, and leaves the circuit working exactly as before.
"""
import pytest
import torch

import pyjuice as juice

ACTIVATIONS = ("node_mars", "element_mars", "node_mars_tempered", "node_flows", "element_flows")


def hmm(**compile_kwargs):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 8, num_latents = 256, num_emits = 16)
    return juice.compile(ns, verbose = False, **compile_kwargs).to(torch.device("cuda:0"))


def test_free_activation_buffers_releases_memory_and_keeps_param_flows():
    pc = hmm()
    x = torch.randint(0, 16, (512, 8), device = pc.device)
    pc.init_param_flows(flows_memory = 0.0)
    lls = pc(x).detach().clone()       # detached: the output's autograd graph would keep node_mars alive
    pc.backward(x, allow_modify_flows = False)
    param_flows = pc.param_flows.clone()

    held = sum(t.numel() * t.element_size() for name, t in pc._buffer_storage.items() if name in ACTIVATIONS)
    before = torch.cuda.memory_allocated(pc.device)
    pc.free_activation_buffers()
    assert before - torch.cuda.memory_allocated(pc.device) >= held                  # the storage is gone
    assert not any(hasattr(pc, name) for name in ACTIVATIONS)
    assert not any(name in pc._buffer_storage for name in ACTIVATIONS)
    assert torch.equal(pc.param_flows, param_flows)                                  # EM statistics kept

    assert torch.equal(pc(x), lls)                                                   # allocates again
    pc.backward(x, allow_modify_flows = False)
    assert torch.allclose(pc.param_flows, 2 * param_flows, rtol = 1e-5)             # flows still accumulate


def test_free_activation_buffers_drops_recorded_cuda_graphs():
    pc = hmm(cuda_graphs = True)
    x = torch.randint(0, 16, (4, 8), device = pc.device)
    lls = pc(x).clone()
    pc(x)                                                                            # records the inner layers
    assert len(pc._recorded_cuda_graphs) > 0
    pc.free_activation_buffers()
    assert len(pc._recorded_cuda_graphs) == 0
    assert torch.equal(pc(x), lls)


def test_free_activation_buffers_on_a_fresh_circuit_is_a_no_op():
    pc = hmm()
    pc.free_activation_buffers()
    x = torch.randint(0, 16, (4, 8), device = pc.device)
    assert torch.isfinite(pc(x)).all()
