"""
A regular sum layer's tied parameter flows deferred to one contraction (pyjuice/layer/deferred_par.py, the kernels of
layer/kernels/c/par_deferred_fp16.cu, reached from `SumLayer._backward_block_sparse_par_flows`): the flows a backward
leaves against float64 recomputed from every deferred step's inputs -- homogeneous HMMs of one and two blocks, fp32 and
bf16 parameters, 39 steps (two flow blocks: two groups), a batch past 512 (two slots a step), and more steps than the
slots hold (flushes during the pass); the deferral an accumulation (two backwards with `flows_memory = 1` give twice
one); what keeps the per-step kernels (a layer callback, an untied HMM, a small group, fp32-level dots); the deferred
flows against the per-step kernels'; a CUDA graph replaying it to the bit; and a backward that raised leaving nothing.
The gated layers' copy of these: tests/model/external_params/blockscale_par_deferred_test.py. Skipped where the CUDA
kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
from pyjuice.layer import deferred_par as dpar

DEV = torch.device("cuda:0")
NEG = -float("inf")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
V = 32


def _module():
    if not torch.cuda.is_available():
        return None
    from pyjuice.layer.kernels.c import get_par_deferred_module
    try:
        return get_par_deferred_module()
    except Exception:
        return None


needs_kernels = pytest.mark.skipif(_module() is None,
                                   reason = "needs the deferred parameter-flow kernels (nvcc + CUTLASS)")


@pytest.fixture(autouse = True)
def _defer_from_1m_flows(monkeypatch):
    """The mechanism from 1M flows a group, the shapes here (the default threshold is a speed choice, not a limit)."""
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)


def _instance(L, nb, T, dt, B, tied = True, seed = 0, **compile_kw):
    """A homogeneous HMM (`tied`: one transition and one emission node, the others their copies) of T - 1 transitions
    in `nb` blocks, and a batch of its inputs."""
    torch.manual_seed(seed)
    ns = juice.structures.HMM(seq_length = T, num_latents = L, num_emits = V, homogeneous = tied, block_size = L // nb)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = DTYPES[dt], **compile_kw).to(DEV)
    x = torch.randint(0, V, (B, T), device = DEV)
    return pc, x


def _record(monkeypatch):
    """Every partition `deferred_par.try_defer` takes from now on: its tables and the rows of the call's tensors it
    reads (the real deferral still runs)."""
    steps = []
    real = dpar.try_defer

    def try_defer(mod, layer, ctx, *rest):
        took = real(mod, layer, ctx, *rest)
        if took:
            blk = ctx["block_size"]
            rows = ctx["nids"].long().view(-1, 1) + torch.arange(blk, device = DEV).view(1, -1)
            steps.append(dict(nf = ctx["node_flows"][rows].clone(), nm = ctx["node_mars"][rows].clone(),
                              em = ctx["element_mars"][ctx["cids"].long()].clone(), pids = ctx["pids"].long(),
                              pfids = ctx["pfids"].long(), blk = blk))
        return took

    monkeypatch.setattr(dpar, "try_defer", try_defer)
    return steps


def _reference(steps, pc):
    """float64 over the deferred steps and their samples of exp(lf + emars) times theta, where each step's flows go in
    `pc.param_flows` (`ref`), and which entries that is (`touched`)."""
    ref = torch.zeros(pc.param_flows.numel(), dtype = torch.float64, device = DEV)
    touched = torch.zeros(pc.param_flows.numel(), dtype = torch.bool, device = DEV)
    params = pc.params.double()
    for s in steps:
        nf, nm, em = s["nf"].double(), s["nm"].double(), s["em"].double()
        lf = torch.where(nm == NEG, torch.full_like(nm, NEG), nf - nm)
        M = lf.max(1, keepdim = True).values
        M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
        ar = torch.arange(s["blk"], device = DEV).view(1, 1, -1)
        flows = torch.einsum("knb,keb->ken", torch.exp(lf - M), torch.exp(em + M)) * params[s["pids"][:, :, None] + ar]
        at = (s["pfids"][:, :, None] + ar).reshape(-1)
        ref.index_put_((at,), flows.reshape(-1), accumulate = True)
        touched[at] = True
    return ref, touched


def _check(pc, ref, touched, what, scale = 1.0, tol = 1e-3):
    got, ref = pc.param_flows.double()[touched], ref[touched] * scale
    d = float((got - ref).abs().max() / ref.abs().max())
    big = ref > 1e-6 * ref.max()
    rel = float(((got - ref).abs() / ref.abs().clamp(min = 1e-300))[big].max())
    assert d < tol and rel < 3 * tol, f"{what}: flows off by {d:.2e} of the max, {rel:.2e} relative"


def _slots_mb(nb, blk, ne, W, S):
    """The slot budget (MB) that holds S slots of W columns (fp16 values, a float scale a row)."""
    return S * nb * (blk + ne) * (W * 2 + 4) / 2 ** 20


# (name, L, node blocks, T, params dtype, B, slot budget MB or None, deferred steps)
CASES = [
    ("hmm1024_T8", 1024, 1, 8, "bf16", 128, None, 7),
    ("hmm1024_fp32_B96", 1024, 1, 8, "fp32", 96, None, 7),
    ("hmm2048x2_T6", 2048, 2, 6, "bf16", 64, None, 5),
    ("hmm1024_T40_two_groups", 1024, 1, 40, "bf16", 64, None, 39),              # 39 copies: flow blocks of 32 and 7
    ("hmm1024_T9_two_slots", 1024, 1, 9, "fp32", 64, _slots_mb(1, 1024, 1024, 64, 2), 8),       # S = 2
    ("hmm1024_B608_two_slots_a_step", 1024, 1, 6, "bf16", 608, None, 5),       # W = 320, 2 slots a step
    ("hmm1024_B608_fp32_flushed", 1024, 1, 7, "fp32", 608, _slots_mb(1, 1024, 1024, 320, 5), 6),
]


@needs_kernels
@pytest.mark.parametrize("name,L,nb,T,dt,B,mb,n", CASES, ids = [c[0] for c in CASES])
def test_the_deferred_flows_match_float64(name, L, nb, T, dt, B, mb, n, monkeypatch):
    if mb is not None:
        monkeypatch.setattr(dpar, "PAR_DEFER_MB", mb)
    pc, x = _instance(L, nb, T, dt, B)
    steps = _record(monkeypatch)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert len(steps) == n, f"{name}: {len(steps)} steps deferred"
    _check(pc, *_reference(steps, pc), name)


@needs_kernels
def test_it_accumulates_like_the_per_step_kernels(monkeypatch):
    """Flows left in the buffer (`flows_memory = 1`) are kept: two backwards give twice one."""
    pc, x = _instance(1024, 1, 6, "bf16", 64)
    steps = _record(monkeypatch)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert len(steps) == 5
    ref, touched = _reference(steps, pc)
    _check(pc, ref, touched, "one backward")
    pc(x)
    pc.backward(x, flows_memory = 1.0)
    torch.cuda.synchronize()
    _check(pc, ref, touched, "two backwards", scale = 2.0)


@needs_kernels
@pytest.mark.parametrize("label", ["a layer callback", "untied", "below MIN_FLOWS", "fp32-level dots"])
def test_what_keeps_the_per_step_kernels(label, monkeypatch):
    """No deferral where a callback sees the flows between layers, where nothing is tied, where the group is too small
    for it (2 x 256 x 512 = 262K flows < 1M), or where the dots are fp32-level (`precision = "fp32"`: not the regime of
    the fp16 operands); the flows are the per-step kernels', against float64."""
    if label == "below MIN_FLOWS":
        pc, x = _instance(512, 2, 6, "bf16", 64)
    else:
        pc, x = _instance(1024, 1, 6, "bf16", 64, tied = label != "untied",
                          **({"precision": "fp32"} if label == "fp32-level dots" else {}))
    steps = _record(monkeypatch)
    if label == "a layer callback":                                     # (after a backward that deferred: the flag
        pc(x)                                                           # is the call's, not a plan's recorded one)
        pc.backward(x, flows_memory = 0.0)
        assert len(steps) == 5
        del steps[:]
        seen = []
        pc(x)
        pc.backward(x, flows_memory = 0.0, sum_layer_post_backward_callback = lambda layer, **kw: seen.append(layer))
        assert seen
    else:
        pc(x)
        pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert steps == [], f"{label}: {len(steps)} steps deferred"


@needs_kernels
def test_the_deferred_flows_equal_the_per_step_ones_within_their_rounding(monkeypatch):
    pc, x = _instance(1024, 1, 8, "bf16", 128)
    steps = _record(monkeypatch)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    assert len(steps) == 7
    deferred = pc.param_flows.clone()
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 62)                     # no group reaches them: the per-step kernels
    monkeypatch.setattr(dpar, "MIN_FLOW_SAMPLES", 1 << 62)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    assert len(steps) == 7
    fin = torch.isfinite(deferred)
    assert torch.equal(fin, torch.isfinite(pc.param_flows))
    d = float((deferred[fin] - pc.param_flows[fin]).abs().max() / pc.param_flows[fin].abs().max())
    assert d < 3e-3, f"the deferred and per-step flows differ by {d:.2e} of the max"


@needs_kernels
def test_a_cuda_graph_replays_it_to_the_bit(monkeypatch):
    """The group's flows (captures, the flush: no atomics on their values) from a replayed graph, as an eager pass's."""
    pc, x = _instance(1024, 1, 6, "bf16", 64)
    steps = _record(monkeypatch)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    assert len(steps) == 5
    _, touched = _reference(steps, pc)
    eager = pc.param_flows[touched].clone()
    for _ in range(3):                                                  # record, then replay
        pc(x)
        pc.backward(x, flows_memory = 0.0, record_cudagraph = True)
    torch.cuda.synchronize()
    assert torch.equal(pc.param_flows[touched].view(torch.int32), eager.view(torch.int32))


@needs_kernels
def test_a_backward_that_raised_leaves_nothing(monkeypatch):
    """Steps captured by a backward that then failed must not reach the next one's flows."""
    pc, x = _instance(1024, 1, 6, "bf16", 64)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    clean = pc.param_flows.clone()
    mod = _module()
    real_flush, calls = mod.flush, []

    def failing_flush(*a):
        calls.append(1)
        raise RuntimeError("injected")

    monkeypatch.setattr(mod, "flush", failing_flush)
    pc(x)
    with pytest.raises(RuntimeError, match = "injected"):
        pc.backward(x, flows_memory = 0.0)
    assert calls
    monkeypatch.setattr(mod, "flush", real_flush)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    fin = torch.isfinite(clean)
    d = float((pc.param_flows[fin] - clean[fin]).abs().max() / clean[fin].abs().max())
    assert torch.equal(fin, torch.isfinite(pc.param_flows)) and d < 1e-5, f"the next backward's flows off by {d:.2e}"
