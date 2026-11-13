"""
A regular sum layer's tied parameter flows deferred to one contraction (pyjuice/layer/deferred_par.py, the kernels of
layer/kernels/c/par_deferred_fp16.cu, reached from `SumLayer._backward_block_sparse_par_flows`): the flows a backward
leaves against float64 recomputed from every deferred step's inputs -- homogeneous HMMs of one and two blocks, fp32 and
bf16 parameters, 39 steps (two flow blocks: two groups), a batch past 512 (two slots a step), and more steps than the
slots hold (flushes during the pass); the deferral an accumulation (two backwards with `flows_memory = 1` give twice
one); what keeps the per-step kernels (a layer callback, an untied HMM, a small group, fp32-level dots); the deferred
flows against the per-step kernels'; a CUDA graph replaying it to the bit; and a backward that raised leaving nothing --
to the next backward, nor alive after its circuit.
The gated layers' copy of these: tests/model/parameterizations/blockscale_par_deferred_test.py. Skipped where the CUDA
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


def _flush_modules():
    """The modules a group's flush may take (`_Group._pick_cfg`): par_deferred_fp16.cu's and its producer-warp copy's."""
    from pyjuice.layer.kernels.c import get_par_deferred_flush_ws_module
    return [m for m in (_module(), get_par_deferred_flush_ws_module()) if m is not None]


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
    # the deferral's own shapes, not the CUDA per-step kernel's (its `batch % 32`, `num_edges % 128`)
    ("hmm1024_B16", 1024, 1, 8, "bf16", 16, None, 7),
    ("hmm1024_fp32_B40", 1024, 1, 8, "fp32", 40, None, 7),
    ("hmm1088x17_edges_in_64s", 1088, 17, 6, "bf16", 64, None, 5),
    # node blocks of 32: one operand group of them (`deferred_par.operand_group`), and a batch the chained capture does
    # not take (62: not in 4s) -- each node block its own operand, the layout before groups
    ("hmm2048_blocks_of_32", 2048, 64, 6, "bf16", 64, None, 5),
    ("hmm2048_blocks_of_32_B16_fp32", 2048, 64, 6, "fp32", 16, None, 5),
    ("hmm1024_B62_per_block_operands", 1024, 16, 6, "bf16", 62, None, 5),
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
def test_the_slots_take_the_lesser_of_their_budget_and_a_share_of_the_device(monkeypatch):
    """A group's slots fit `PAR_DEFER_MB`, and at most `PAR_DEFER_SHARE` of the device's memory (a smaller GPU's budget
    its share): the lesser sets how many it holds -- here a slot of one node block of 64 nodes and 64 edges at 32
    columns, 8704 bytes; the budgets half a slot past the counts."""
    pf = torch.zeros(1 << 14, device = DEV)
    pfids = (torch.arange(64, device = DEV) * 64).view(1, 64)
    per_slot = (64 + 64) * (32 * 2 + 4)
    total = torch.cuda.get_device_properties(DEV).total_memory
    assert dpar.slot_budget(DEV) == min(4096 * 2 ** 20, int(total / 16))         # (the defaults)
    monkeypatch.setattr(dpar, "PAR_DEFER_MB", 5.5 * per_slot / 2 ** 20)
    monkeypatch.setattr(dpar, "PAR_DEFER_SHARE", 1.0)
    g = dpar._Group(pfids, pfids.clone(), 1, 64, 64, 32, pf)
    assert g.W == 32 and g.S == 5 and g.A.size(2) == 5 * 32                     # the MB the lesser
    monkeypatch.setattr(dpar, "PAR_DEFER_MB", 4096)
    monkeypatch.setattr(dpar, "PAR_DEFER_SHARE", 3.5 * per_slot / total)
    g = dpar._Group(pfids, pfids.clone(), 1, 64, 64, 32, pf)
    assert g.S == 3 and g.Bop.size(2) == 3 * 32                                 # the share the lesser


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
@pytest.mark.parametrize("label", ["a layer callback", "untied", "below MIN_FLOWS", "fp32-level dots",
                                   "blocks of 32 ungrouped", "blocks of 32 of other children"])
def test_what_keeps_the_per_step_kernels(label, monkeypatch):
    """No deferral where a callback sees the flows between layers, where nothing is tied, where the group is too small
    for it (2 x 256 x 512 = 262K flows < 1M), where the dots are fp32-level (`precision = "fp32"`: not the regime of
    the fp16 operands), or where node blocks of 32 form no operand group -- a batch the chained capture does not take,
    node blocks of other children (`_block_sparse_chain`): each its own operand at 32 rows, measured slower deferred
    (`deferred_par.may_defer`)."""
    if label == "below MIN_FLOWS":
        pc, x = _instance(512, 2, 6, "bf16", 64)
    elif label == "blocks of 32 ungrouped":
        pc, x = _instance(1024, 32, 6, "bf16", 62)
    elif label == "blocks of 32 of other children":
        torch.manual_seed(0)
        root = _block_sparse_chain(2048, 32, 6, 16)                # 64 node blocks x 32 nodes x 512 edges: 1M flows
        root.init_parameters(perturbation = 2.0)
        pc = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV)
        x = torch.randint(0, V, (64, 6), device = DEV)
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
@pytest.mark.parametrize("L,nb,B,G", [(2048, 64, 64, 64), (1024, 16, 16, 16), (1024, 1, 128, 1), (1024, 16, 62, 1)],
                         ids = ["blocks_of_32", "blocks_of_64", "one_node_block", "batch_not_in_4s"])
def test_the_operand_groups_follow_the_structure(L, nb, B, G, monkeypatch):
    """An HMM's node blocks have the same children and no gates: one operand group of all of them, where the chained
    capture takes the step's slots (a batch in 4s) and the group has more than one node block; else each node block its
    own operand. `GROUP_OPERANDS` off: each its own -- and node blocks of 32 not deferred (`deferred_par.may_defer`)."""
    pc, x = _instance(L, nb, 6, "bf16", B)
    steps = _record(monkeypatch)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    groups = list(dpar._groups_of(pc.param_flows, create = False).values())
    assert len(steps) == 5 and groups and all(g.G == G for g in groups), [g.G for g in groups]
    assert all(g.Bop.size(0) == nb // G and g.A.size(0) == nb for g in groups)
    monkeypatch.setattr(dpar, "GROUP_OPERANDS", False)
    pc2, _ = _instance(L, nb, 6, "bf16", B)
    del steps[:]
    pc2(x)
    pc2.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    groups = list(dpar._groups_of(pc2.param_flows, create = False).values())
    if L // nb == 32:
        assert steps == [] and not groups
    else:
        assert len(steps) == 5 and groups and all(g.G == 1 for g in groups)
    # the two layouts' flows (or the per-step kernels'), each the float64 one's to the operands' rounding: each other's
    d = float((pc.param_flows - pc2.param_flows).abs().max() / pc2.param_flows.abs().max())
    assert d < 2e-3, f"grouped and per-block flows differ by {d:.2e} of the max"


def _block_sparse_chain(L, blk, T, fan):
    """A tied chain whose sum layer joins each node block of `blk` nodes to `fan` consecutive product blocks (its own
    run of the L // blk below, the runs not all the same: node blocks of other children)."""
    from pyjuice.nodes import inputs, multiply, summate
    nb = L // blk
    starts = torch.clamp(torch.arange(nb), max = nb - fan)
    edges = torch.stack([torch.arange(nb).repeat_interleave(fan),
                         (starts[:, None] + torch.arange(fan)[None, :]).reshape(-1)])
    with juice.set_block_size(blk):
        ns = inputs(0, num_node_blocks = nb, dist = juice.distributions.Categorical(num_cats = V))
        src = esrc = None
        for t in range(1, T):
            emit = esrc.duplicate(t, tie_params = True) if esrc is not None else None
            if emit is None:
                emit = esrc = inputs(t, num_node_blocks = nb, dist = juice.distributions.Categorical(num_cats = V))
            prod = multiply(ns, emit)
            ns = src = summate(prod, num_node_blocks = nb, edge_ids = edges) if src is None else \
                src.duplicate(prod, tie_params = True)
        return summate(multiply(ns), num_node_blocks = 1, block_size = 1)


@needs_kernels
@pytest.mark.parametrize("L,nb,B", [(1024, 1, 16), (1024, 1, 40), (1088, 17, 64)])
def test_where_only_the_deferrals_shapes_hold_the_per_step_kernel_is_the_triton_one(L, nb, B, monkeypatch):
    """At a shape the deferral takes and the CUDA per-step kernel does not (a batch not a multiple of 32, edges in 64s
    not 128s), a layer that does not defer -- an untied HMM's -- runs the Triton kernel (the CUDA one, which refuses
    the shape, not even tried): its flows those of a circuit
    whose CUDA path is off from the start, within the per-step kernels' rounding (each circuit's layers measure their
    own kernel arms, so the two may compute in different ones: MEASURED 7e-4 of the max apart in a run beside other
    tests' workers, 5e-7 alone)."""
    from pyjuice.layer import sum_layer as SL
    pc, x = _instance(L, nb, 6, "bf16", B, tied = False)
    steps = _record(monkeypatch)
    tried, real = [], SL.cuda_kernels.par_backward_sum
    monkeypatch.setattr(SL.cuda_kernels, "par_backward_sum", lambda *a, **k: tried.append(1) or real(*a, **k))
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert steps == [] and tried == []
    got = pc.param_flows.clone()
    monkeypatch.setattr(SL, "BACKWARD_PAR_FLOW_CUDA", False)
    ref, _ = _instance(L, nb, 6, "bf16", B, tied = False)
    ref(x)
    ref.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert torch.isfinite(got).all()
    d = float((got - ref.param_flows).abs().max() / ref.param_flows.abs().max())
    assert d < 3e-3, f"the flows differ from the Triton kernels' by {d:.2e} of the max"


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
    calls = []

    def failing_flush(*a):
        calls.append(1)
        raise RuntimeError("injected")

    mods = _flush_modules()
    real_flushes = [m.flush for m in mods]
    for m in mods:
        monkeypatch.setattr(m, "flush", failing_flush)
    pc(x)
    with pytest.raises(RuntimeError, match = "injected"):
        pc.backward(x, flows_memory = 0.0)
    assert calls
    for m, real in zip(mods, real_flushes):
        monkeypatch.setattr(m, "flush", real)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    fin = torch.isfinite(clean)
    d = float((pc.param_flows[fin] - clean[fin]).abs().max() / clean[fin].abs().max())
    assert torch.equal(fin, torch.isfinite(pc.param_flows)) and d < 1e-5, f"the next backward's flows off by {d:.2e}"


@needs_kernels
def test_a_backward_that_raised_keeps_nothing_alive(monkeypatch):
    """A backward that raised after a step was captured, before the flush: its scope drops the flush it held -- a
    group keeps its last scope, through which the circuit's parameter flows would outlive the circuit."""
    import gc
    import weakref
    pc, x = _instance(1024, 1, 6, "bf16", 64)
    pc(x)
    pc.backward(x, flows_memory = 0.0)
    real, calls = dpar.try_defer, []

    def try_defer(*a, **k):
        calls.append(1)
        if len(calls) == 2:                                   # (the first step captured, its group's flush pending)
            raise RuntimeError("injected")
        return real(*a, **k)

    monkeypatch.setattr(dpar, "try_defer", try_defer)
    pc(x)
    with pytest.raises(RuntimeError, match = "injected"):
        pc.backward(x, flows_memory = 0.0)
    monkeypatch.setattr(dpar, "try_defer", real)
    assert len(calls) == 2
    flows = weakref.ref(pc.param_flows)
    del pc, x
    gc.collect()
    assert flows() is None, "the raised backward's scope keeps the circuit's parameter flows alive"
