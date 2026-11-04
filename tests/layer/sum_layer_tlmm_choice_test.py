"""
The CUDA (CuTe) tlmm forward fast path's {Triton, CUDA tile config} choice is measured once per launch SHAPE and
shared process-wide (`autotune.choose`), not per layer: its arms agree only to ~1e-6, so two structurally identical
models whose layers each measured their own could take different arms and disagree in the last bits.
"""
import pytest
import torch

import pyjuice as juice
from pyjuice.layer import sum_layer
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as cuda_kernels

DEV = torch.device("cuda:0")


def hmm():
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 4, num_latents = 64, num_emits = 8)     # 64 edges: the tlmm regime
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_identical_models_share_the_tlmm_choice(monkeypatch):
    """
    REGRESSION. With a measurement that names a different arm every time it is asked -- the timing noise that
    tips a near-tie either way, made deterministic -- the second of two identical models must reuse the first's
    choice instead of measuring its own layers again, and so compute bit-identical outputs.
    """
    if not (sum_layer.FORWARD_SUM_CUDA and cuda_kernels.is_available()):
        pytest.skip("the CUDA tlmm fast path is not available here")
    monkeypatch.setattr(autotune, "_CACHE", dict())                  # (nothing measured before this test)
    asked = []

    def flipping(candidates, *args, **kwargs):
        asked.append([choice for choice, _ in candidates])
        return candidates[0][0] if len(asked) % 2 else candidates[-1][0]

    monkeypatch.setattr(autotune, "best_of", flipping)
    x = torch.randint(0, 8, (64, 4), device = DEV, generator = torch.Generator(device = DEV).manual_seed(0))

    def gates():                                                      # the tlmm gate's measurements so far
        return sum(c[0] == ("triton", -1) and any(k[0] == "cuda" for k in c[1:]) for c in asked)

    first = hmm()(x)
    measured = gates()
    second = hmm()(x)
    assert measured > 0, "the tlmm gate was never measured (check its conditions)"
    assert gates() == measured, "the second, identical model measured its layers again"
    assert torch.equal(first, second)
