"""
The class weights on CUDA (kernels/c/class_weights.cu, ``cw_class_cuda``: the entry tables, then the rows staged in
shared memory; dispatched from ``score.run_class_weights`` for ``input @ block`` steps with one exit column and one
step a destination, on compact values under evidence pruning): sessions with the kernels forced on, at every CTA shape
(1, 2 or 4 destinations, 128 or 256 threads), against the float64 reference (strong_check's bars) and, bit for bit,
against the same sessions on the Triton kernel (the walk-length rule off: every launch it applies to); what it declines
(dense storage). Skipped where the extension cannot build.
"""
import importlib.util
import os

import pytest
import torch

import pyjuice.constraints as jc
import pyjuice.constraints.inference.dfa_exact.kernels.score as score
from pyjuice.constraints.inference.dfa_exact.kernels import c as dfa_c

_spec = importlib.util.spec_from_file_location("strong_check_test", os.path.join(os.path.dirname(__file__), "strong_check_test.py"))
strong = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strong)
Float64Reference, check_against_reference, run_session, schedule, small_case = (
    strong.Float64Reference, strong.check_against_reference, strong.run_session, strong.schedule, strong.small_case)

needs_kernel = pytest.mark.skipif(not torch.cuda.is_available() or dfa_c.get_class_weights_module() is None,
                                  reason = "needs the CUDA class weights (nvcc, sm_80+)")


def _calls(monkeypatch):
    """Every call of the extension's kernel, recorded."""
    mod = dfa_c.get_class_weights_module()
    real, calls = mod.cw_class_cuda, []

    def record(*a):
        calls.append(a)
        return real(*a)

    monkeypatch.setattr(mod, "cw_class_cuda", record)
    return calls


@needs_kernel
@pytest.mark.parametrize("shape", [(4, 256), (2, 256), (1, 128), (4, 128)], ids = lambda s: f"tn{s[0]}x{s[1]}")
@pytest.mark.parametrize("mode", ["diffusion", "l2r"])
@pytest.mark.parametrize("kind", ["hmm", "codpula"])
def test_sessions_match_the_reference_and_the_triton_kernel_to_the_bit(kind, mode, shape, monkeypatch):
    pc, dfa, data, inputs = small_case(kind)
    cc = jc.compile(dfa, pc)
    masks = schedule(mode, data.size(0), data.size(1))
    calls = _calls(monkeypatch)
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA", True)
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA_WALK", 0)                  # (every launch: the small cases' walks are short)
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA_DESTS", shape[0])
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA_THREADS", shape[1])
    check_against_reference(cc, Float64Reference(pc, dfa, data.size(1)), data, masks, inputs, "compact", "fp32")
    cuda = run_session(cc, data, masks, inputs, "compact", "fp32")
    assert calls, "the CUDA kernel served no launch"
    n = len(calls)
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA", False)
    triton = run_session(cc, data, masks, inputs, "compact", "fp32")
    assert len(calls) == n, "the Triton kernel forced, the CUDA kernel still ran"
    assert cuda.keys() == triton.keys()
    for i in cuda:
        for u, v in zip(cuda[i], triton[i]):
            assert torch.equal(u, v), f"call {i}: the CUDA and Triton class weights differ"


@needs_kernel
def test_it_declines_dense_storage(monkeypatch):
    pc, dfa, data, inputs = small_case("hmm")
    cc = jc.compile(dfa, pc)
    calls = _calls(monkeypatch)
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA", True)
    monkeypatch.setattr(score, "CLASS_WEIGHT_CUDA_WALK", 0)
    run_session(cc, data, schedule("l2r", data.size(0), data.size(1)), inputs, "dense", "fp32")
    assert not calls
