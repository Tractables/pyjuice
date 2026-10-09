"""
`TensorCircuit.precision` (and `juice.compile(..., precision = ...)`): it is validated, settable after compiling,
reaches every sum layer in the forward and the backward pass, keys recorded CUDA graphs, and replaces the removed
`force_use_bf16` / `force_use_fp32` flags (set: refused; unset: ignored).
"""
import pytest
import torch

import pyjuice as juice
from pyjuice.layer import SumLayer

DEV = torch.device("cuda:0")


def hmm(latents = 32, **compile_kwargs):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 4, num_latents = latents, num_emits = 8)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, **compile_kwargs).to(DEV)


def test_compile_and_set():
    assert hmm().precision == "auto"
    pc = hmm(precision = "fp32")
    assert pc.precision == "fp32"
    pc.precision = "tf32"
    assert pc.precision == "tf32"
    with pytest.raises(ValueError, match = "precision"):
        pc.precision = "bf16"
    with pytest.raises(ValueError, match = "precision"):
        hmm(precision = "half")


@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
def test_it_reaches_every_sum_layer(precision, monkeypatch):
    pc = hmm(precision = precision)
    seen = []
    fw, bw = SumLayer.forward, SumLayer.backward
    monkeypatch.setattr(SumLayer, "forward", lambda self, *a, **k: (seen.append(("fw", k.get("precision"))), fw(self, *a, **k))[1])
    monkeypatch.setattr(SumLayer, "backward", lambda self, *a, **k: (seen.append(("bw", k.get("precision"))), bw(self, *a, **k))[1])
    x = torch.randint(0, 8, (32, 4), device = DEV)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    num_sum_layers = sum(len(lg.layers) for lg in pc.inner_layer_groups if lg.is_sum())
    assert seen.count(("fw", precision)) == num_sum_layers and seen.count(("bw", precision)) == num_sum_layers
    assert len(seen) == 2 * num_sum_layers


def test_removed_flags():
    pc = hmm()
    x = torch.randint(0, 8, (32, 4), device = DEV)
    want = pc(x)
    assert torch.equal(pc(x, force_use_fp32 = False, force_use_bf16 = False), want)     # their old default: ignored
    pc.backward(x, flows_memory = 0.0, force_use_fp32 = False)
    for name in ("force_use_fp32", "force_use_bf16"):
        with pytest.raises(TypeError, match = f"{name}.*precision"):
            pc(x, **{name: True})
    with pytest.raises(TypeError, match = "force_use_fp32.*precision"):
        pc.backward(x, flows_memory = 0.0, force_use_fp32 = True)


def test_recorded_graphs_are_keyed_on_it():
    x = torch.randint(0, 8, (64, 4), device = DEV)
    eager = {}
    for precision in ("auto", "fp32"):
        pc = hmm(latents = 64, precision = precision)
        eager[precision] = pc(x)
    assert not torch.equal(eager["auto"], eager["fp32"])                               # bf16 vs fp32-level products
    pc = hmm(latents = 64, cuda_graphs = True)
    for _ in range(2):                                                                  # record, then replay
        assert torch.equal(pc(x), eager["auto"])
    pc.precision = "fp32"
    for _ in range(2):
        assert torch.equal(pc(x), eager["fp32"])
