"""
Constrained queries on a circuit whose sum parameters are STORED in bf16 (`TensorCircuit.param_dtype`), against the same
circuit storing the same -- bf16 -- values in fp32: the queries' kernels then compute on identical numbers (their
weights an fp32 copy kept with the weight stacks, see `kernels.sum.query_weights`), so the answers agree to fp32
rounding. Conditionals and marginals, the fused and the dense sum paths, dense and compact storage, plain and
BlockScale-gated circuits, one-off queries and decoder sessions inside `fast_inference` (the copy kept by the scope).
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
from pyjuice.constraints.inference.dfa_exact.queries.backend import forward as F

V = 3
KINDS = {"hmm": 6, "pd": 8, "hand": 5}
DEV = torch.device("cuda:0")
BF16 = torch.bfloat16
CONSTRAINT = lambda: jc.DFA.contains([[1, 2]], V) & jc.Not(jc.DFA.contains([[2, 2]], V))
GATES = dict(child_gate = lambda ns: next(d for d in (2, 1) if ns.ch_block_size % d == 0),
             node_gate = lambda ns: ns.block_size)


@pytest.fixture(params = ["fused", "dense"])
def sum_path(request, monkeypatch):
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", lambda ncols, live: live < ncols)
    return request.param


def _pair(build_pc, kind, gated = False):
    """(fp32 circuit holding the bf16 values, bf16 circuit), from one initialization."""
    kw = dict(perturbation = 4.0, gates = GATES) if gated else {}
    pc16 = build_pc(kind, KINDS[kind], V, **kw).set_param_dtype(BF16)
    pc32 = build_pc(kind, KINDS[kind], V, **kw).set_param_dtype(BF16).set_param_dtype(torch.float32)
    assert pc16.params.dtype == BF16 and torch.equal(pc32.params, pc16.params.float())
    return pc32, pc16


def _evidence(n, B, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0] = True
    return data.to(DEV), missing.to(DEV)


def _gates(pc, B, seed = 0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for ns in pc.external_params_nodes:
        owner = ns.external_params.storage_owner(ns)
        if owner not in out:
            out[owner] = (torch.randn(ns.external_params.tensor_shapes(owner, B)[0], generator = g) * 2.0).to(DEV)
    return out


def _close(got, want):
    assert got.shape == want.shape and not torch.isnan(got).any()
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert torch.allclose(got[fin], want[fin], atol = 1e-5, rtol = 1e-5), float((got[fin] - want[fin]).abs().max())


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_constrained_queries_on_bf16_parameters(kind, storage, sum_path, build_pc):
    pc32, pc16 = _pair(build_pc, kind)
    data, missing = _evidence(KINDS[kind], 6)
    out = {}
    for tag, pc in (("fp32", pc32), ("bf16", pc16)):
        cc = jc.compile(CONSTRAINT(), pc)
        out[tag] = (cc.marginal(data, missing, storage = storage, precision = "fp32"),
                    cc.conditional(data, missing, storage = storage, precision = "fp32"))
    for got, want in zip(out["bf16"], out["fp32"]):
        _close(got, want)
    assert pc16.params.dtype == BF16                                  # (the queries' copy left the circuit alone)


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_a_decoder_session_inside_fast_inference(kind, build_pc):
    pc32, pc16 = _pair(build_pc, kind)
    data, missing = _evidence(KINDS[kind], 4, seed = 1)
    out = {}
    for tag, pc in (("fp32", pc32), ("bf16", pc16)):
        cc = jc.compile(CONSTRAINT(), pc)
        with juice.fast_inference():
            with cc.decoder(storage = "compact") as decoder:
                out[tag] = [decoder.marginal(data, missing) for _ in range(2)]        # (the second reuses the copy)
    for got, want in zip(out["bf16"], out["fp32"]):
        _close(got, want)


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_blockscale_gated_queries_on_bf16_parameters(kind, sum_path, build_pc):
    pc32, pc16 = _pair(build_pc, kind, gated = True)
    data, missing = _evidence(KINDS[kind], 3, seed = 2)
    out = {}
    for tag, pc in (("fp32", pc32), ("bf16", pc16)):
        cc = jc.compile(CONSTRAINT(), pc)
        gates = _gates(pc, 3, seed = 2)                  # (keyed by the circuit's own nodes: the same values for both)
        with cc.decoder(sum_external_params = gates, storage = "dense") as decoder:
            out[tag] = decoder.marginal(data, missing)
    _close(out["bf16"], out["fp32"])
