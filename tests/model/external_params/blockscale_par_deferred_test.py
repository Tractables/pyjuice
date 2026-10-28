"""
A tied group's gated parameter flows deferred to one contraction (pyjuice/layer/deferred_par.py, layer/kernels/c/
par_deferred_fp16.cu): the flows a backward leaves against float64 recomputed from every captured step's
inputs -- tied gated HMMs of one and two blocks, fp32 and bf16 parameters, a batch past the slots' 32 columns, a
batch past 512 (two slots a step), -inf log-gates and log-gates spread wide (the rows' scales apart from slot to
slot), 39 steps, and more steps than the slots hold (flushes during the pass), with the L2's bulk adds and with the
threads' read-add-write; the deferral an accumulation (two backwards with `flows_memory = 1` give twice one); a layer
callback, an untied or a small layer keeping the per-step kernels; a CUDA graph replaying it to the bit; and a
backward that raised part-way leaving nothing to the next. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
from pyjuice.layer import deferred_par as dpar

DEV = torch.device("cuda:0")
NEG = -float("inf")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CATS = 16


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


def _hmm(L, nb, gcb, T, tied = True):
    """A gated HMM of T - 1 transitions (tied: one transition and one emission node, the others their copies)."""
    with juice.set_block_size(L // nb):
        ns = inputs(0, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS))
        gated, src, esrc = [], None, None
        for t in range(1, T):
            if tied and esrc is not None:
                emit = esrc.duplicate(t, tie_params = True)
            else:
                emit = esrc = inputs(t, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS))
            if tied and src is not None:
                ns = src.duplicate(multiply(ns, emit), tie_params = True)
            else:
                ns = src = summate(multiply(ns, emit), num_node_blocks = nb,
                                   external_params = BlockScaleSumParams(ch_block_size = gcb))
            gated.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    return root, gated, T


def _instance(L, nb, gcb, T, dt, B, neg_gates = False, tied = True, seed = 0, gate_scale = 2.0, **compile_kw):
    torch.manual_seed(seed)
    root, gated, V = _hmm(L, nb, gcb, T, tied)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = DTYPES[dt], **compile_kw).to(DEV)
    torch.manual_seed(seed + 1)
    phi = {}
    for ns in gated:
        p = torch.randn(ns.external_params.tensor_shapes(ns, B)[0], device = DEV) * gate_scale
        if neg_gates:
            p[torch.rand_like(p) < 0.2] = NEG
        phi[ns] = p
    x = torch.randint(0, CATS, (B, V), device = DEV)
    return pc, phi, x


def _record(monkeypatch):
    """Every `capture` (its inputs, cloned) and `flush` the backward runs, the real ones still run."""
    mod = _module()
    caps, flushes = [], []
    real_cap, real_flush = mod.capture, mod.flush

    def capture(A, Bop, fA, fB, M, nf, nm, em, ext, nids, cids, gate, batch, slot, node_cbs, gate_cbs, ext_base,
                gated):
        assert gated, "a gated layer's capture reads its gates"
        caps.append(dict(A = A, nf = nf.clone(), nm = nm.clone(), em = em.clone(), ext = ext.clone(), nids = nids,
                         cids = cids, gate = gate, batch = batch, node_cbs = node_cbs, gate_cbs = gate_cbs,
                         ext_base = ext_base, blk = A.size(1), ne = Bop.size(1), slot = slot))
        return real_cap(A, Bop, fA, fB, M, nf, nm, em, ext, nids, cids, gate, batch, slot, node_cbs, gate_cbs,
                        ext_base, gated)

    def flush(pf, A, Bop, fA, fB, pfids, pids, params, slots, cfg, bulk):
        flushes.append(dict(A = A, pfids = pfids, pids = pids, params = params, slots = slots, bulk = bulk))
        return real_flush(pf, A, Bop, fA, fB, pfids, pids, params, slots, cfg, bulk)

    monkeypatch.setattr(mod, "capture", capture)
    monkeypatch.setattr(mod, "flush", flush)
    return caps, flushes


def _reference(steps, pfids, pids, params):
    """float64: sum over the steps and samples of exp(lf + emars + log phi), times theta, per (block, edge, node)."""
    ref = 0
    for c in steps:
        B, blk, ne = c["batch"], c["blk"], c["ne"]
        nf, nm, em, ext = (c[k].double() for k in ("nf", "nm", "em", "ext"))
        ext = ext.view(-1, B)
        nbk = c["nids"].numel()
        rows = c["nids"].view(-1, 1) + torch.arange(blk, device = DEV).view(1, -1)
        lf = torch.where(nm[rows] == NEG, torch.full_like(nm[rows], NEG), nf[rows] - nm[rows])
        e = torch.arange(ne, device = DEV)
        gcol = e // c["node_cbs"]
        gb = c["gate"][:, gcol.clamp(max = c["gate"].size(1) - 1)]
        gb = torch.where(gcol[None, :] < c["gate"].size(1), gb, torch.full_like(gb, -1))
        grow = gb + c["ext_base"] + (e % c["node_cbs"]) // c["gate_cbs"]
        lp = torch.where((gb >= 0)[:, :, None], ext[grow.clamp(min = 0)],
                         torch.full((nbk, ne, B), NEG, device = DEV, dtype = torch.float64))
        ce = em[c["cids"]] + lp
        M = lf.max(1, keepdim = True).values
        M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
        ref = ref + torch.einsum("knb,keb->ken", torch.exp(lf - M), torch.exp(ce + M))
    blk = steps[0]["blk"]
    return ref * params.double()[pids[:, :, None] + torch.arange(blk, device = DEV).view(1, 1, -1)]


def _block(pf, pfids, blk):
    return pf.double()[pfids[:, :, None] + torch.arange(blk, device = DEV).view(1, 1, -1)]


def _check(got, ref, what, tol = 1e-3):
    d = float((got - ref).abs().max() / ref.abs().max())
    big = ref > 1e-6 * ref.amax(1, keepdim = True)
    rel = float(((got - ref).abs() / ref.abs().clamp(min = 1e-300))[big].max())
    assert d < tol and rel < 3 * tol, f"{what}: flows off by {d:.2e} of the max, {rel:.2e} relative"


def _slots_mb(nb, blk, ne, W, S):
    """The slot budget (MB) that holds S slots of W columns (fp16 values, a float scale a row)."""
    return S * nb * (blk + ne) * (W * 2 + 4) / 2 ** 20


# (name, L, node blocks, gate width, T, params dtype, B, -inf log-gates, log-gate scale, slot budget MB or None)
CASES = [
    ("hmm1024_T8", 1024, 1, 64, 8, "bf16", 128, False, 2.0, None),
    ("hmm1024_g8_fp32_B100", 1024, 1, 8, 8, "fp32", 100, True, 2.0, None),
    ("hmm2048x2_T6", 2048, 2, 64, 6, "bf16", 64, False, 2.0, None),
    ("hmm1024_T40", 1024, 1, 16, 40, "bf16", 64, False, 2.0, None),                 # 39 copies, one flow block
    ("hmm1024_T9_two_slots", 1024, 1, 64, 9, "fp32", 64, True, 2.0, _slots_mb(1, 1024, 1024, 64, 2)),  # S = 2
    ("hmm1024_B600_two_slots_a_step", 1024, 1, 64, 6, "bf16", 600, False, 2.0, None),   # W = 320, 2 slots a step
    ("hmm1024_B600_fp32_flushed", 1024, 1, 64, 7, "fp32", 600, True, 2.0, _slots_mb(1, 1024, 1024, 320, 5)),
    ("hmm1024_wide_gates", 1024, 1, 64, 8, "bf16", 128, False, 12.0, None),          # log-gates ~ N(0, 144)
]


@needs_kernels
@pytest.mark.parametrize("name,L,nb,gcb,T,dt,B,neg,scale,mb", CASES, ids = [c[0] for c in CASES])
def test_the_deferred_flows_match_float64(name, L, nb, gcb, T, dt, B, neg, scale, mb, monkeypatch):
    if mb is not None:
        monkeypatch.setattr(dpar, "PAR_DEFER_MB", mb)
    pc, phi, x = _instance(L, nb, gcb, T, dt, B, neg_gates = neg, gate_scale = scale)
    caps, flushes = _record(monkeypatch)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert len(caps) == T - 1, f"{name}: {len(caps)} of {T - 1} steps deferred"
    groups = {}
    for c in caps:
        groups.setdefault(c["A"].data_ptr(), []).append(c)
    if mb is not None:
        assert len(flushes) > len(groups), f"{name}: no flush before the end of the pass"
    for ptr, steps in groups.items():
        f = next(f for f in flushes if f["A"].data_ptr() == ptr)
        ref = _reference(steps, f["pfids"], f["pids"], f["params"])
        _check(_block(pc.param_flows, f["pfids"], steps[0]["blk"]), ref, f"{name}: a group of {len(steps)} steps")
    # (one group: an HMM's copies sit one a layer, and `max_tied_ns_per_parflow_block` counts the copies in a layer)
    assert len(groups) == 1


EPILOGUES = pytest.mark.parametrize("bulk", [True, False], ids = ["bulk_adds", "threads_adds"])


@needs_kernels
@EPILOGUES
@pytest.mark.parametrize("dt", ["bf16", "fp32"])
def test_both_epilogues_match_float64(bulk, dt, monkeypatch):
    """The flush's adds as the L2's bulk reductions and as the threads' read-add-write (where an edge's run of flows is
    not 16-byte aligned) -- the same flows; and the default is the bulk adds where the runs are aligned."""
    if not bulk:
        monkeypatch.setattr(dpar, "_bulk_ok", lambda pfids, pf: False)
    pc, phi, x = _instance(2048, 2, 64, 6, dt, 96, neg_gates = True)
    caps, flushes = _record(monkeypatch)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    f = flushes[-1]
    assert f["bulk"] == (bulk and bool((f["pfids"] % 4 == 0).all())), f"the flush took bulk = {f['bulk']}"
    assert bulk == f["bulk"], "these circuits' runs of flows are aligned: the default is the bulk adds"
    _check(_block(pc.param_flows, f["pfids"], caps[0]["blk"]), _reference(caps, f["pfids"], f["pids"], f["params"]),
           f"bulk = {bulk}")


@needs_kernels
@EPILOGUES
def test_it_accumulates_like_the_per_step_kernels(bulk, monkeypatch):
    """Two backwards into `flows_memory = 1` give twice one -- the flush adds to what the buffer holds (either
    epilogue)."""
    if not bulk:
        monkeypatch.setattr(dpar, "_bulk_ok", lambda pfids, pf: False)
    pc, phi, x = _instance(1024, 1, 64, 6, "bf16", 128)
    caps, flushes = _record(monkeypatch)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    one = pc.param_flows.clone()
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 1.0)
    torch.cuda.synchronize()
    assert len(caps) == 2 * 5 and flushes[-1]["bulk"] == bulk
    pfids, blk = flushes[0]["pfids"], caps[0]["blk"]
    _check(_block(pc.param_flows, pfids, blk), 2 * _block(one, pfids, blk), "two backwards", tol = 1e-5)


@needs_kernels
@EPILOGUES
def test_an_edge_without_a_gate_adds_nothing(bulk):
    """The kernels on their own, two steps: an edge block whose gate table entry is -1 (a ragged partition's padding)
    adds nothing -- whatever its element row and the gate rows hold (here finite) -- and the others match float64,
    added to the flows already there."""
    mod = _module()
    torch.manual_seed(0)
    nb, blk, ne, B, cbs, E = 2, 64, 256, 40, 64, 300
    W, S = 64, 2
    nids = torch.tensor([0, blk], device = DEV)
    cids = torch.randint(1, E, (nb, ne), device = DEV)
    gate = torch.tensor([[0, 1, 2, 3], [4, -1, 5, -1]], device = DEV)        # (node block, gate column of cbs edges)
    cids[1, cbs:2 * cbs] = 0                                                 # (padding: the dummy element)
    cids[1, 3 * cbs:] = 0
    pfids = (torch.arange(nb * ne, device = DEV).view(nb, ne) * blk)
    pids = pfids.clone()
    params = torch.rand(nb * ne * blk, device = DEV, dtype = torch.bfloat16) + 0.01
    pf = torch.rand(nb * ne * blk, device = DEV) * 3
    pf0 = pf.clone()
    A = torch.zeros(nb, blk, S * W, device = DEV, dtype = torch.half)
    Bop = torch.zeros(nb, ne, S * W, device = DEV, dtype = torch.half)
    fA, fB = torch.ones(nb, S, blk, device = DEV), torch.ones(nb, S, ne, device = DEV)
    M = torch.empty(nb, W, device = DEV, dtype = torch.int32)
    steps = []
    for slot in range(S):
        nf, nm = torch.randn(nb * blk, B, device = DEV) * 3, torch.randn(nb * blk, B, device = DEV) * 3
        nm[torch.rand_like(nm) < 0.05] = NEG
        em = torch.randn(E, B, device = DEV) * 2
        ext = torch.randn(1 + 6, B, device = DEV) * 2                       # (ext_base 1: row 0 a finite stray)
        mod.capture(A, Bop, fA, fB, M, nf, nm, em, ext, nids, cids, gate, B, slot, cbs, cbs, 1, True)
        steps.append(dict(nf = nf, nm = nm, em = em, ext = ext, nids = nids, cids = cids, gate = gate, batch = B,
                          node_cbs = cbs, gate_cbs = cbs, ext_base = 1, blk = blk, ne = ne))
    mod.flush(pf, A, Bop, fA, fB, pfids, pids, params, S, mod.flush_configs().index([64, 64, 32]), bulk)
    torch.cuda.synchronize()
    got = _block(pf, pfids, blk) - _block(pf0, pfids, blk)
    ungated = (gate < 0).repeat_interleave(cbs, dim = 1)                    # [nb, ne]
    assert torch.equal(_block(pf, pfids, blk)[ungated], _block(pf0, pfids, blk)[ungated]), \
        "an ungated edge's flows changed"
    _check(got, _reference(steps, pfids, pids, params), f"bulk = {bulk}")


@needs_kernels
def test_each_capture_takes_its_own_column_maxima():
    """The kernels on their own: each capture takes its own column maxima -- two steps whose node flows sit 120 nats
    apart, each step's products near 1; a maximum left over from the first would scale the second's node operands
    below the floats' range."""
    mod = _module()
    torch.manual_seed(1)
    nb, blk, ne, B, E = 2, 64, 128, 64, 200
    W, S = 64, 2
    nids, cids = torch.tensor([0, blk], device = DEV), torch.randint(1, E, (nb, ne), device = DEV)
    gate = torch.tensor([[0, 1], [0, 1]], device = DEV)
    pfids = torch.arange(nb * ne, device = DEV).view(nb, ne) * blk
    pids = pfids.clone()
    params = torch.rand(nb * ne * blk, device = DEV, dtype = torch.bfloat16) + 0.01
    pf = torch.zeros(nb * ne * blk, device = DEV)
    A = torch.zeros(nb, blk, S * W, device = DEV, dtype = torch.half)
    Bop = torch.zeros(nb, ne, S * W, device = DEV, dtype = torch.half)
    fA, fB = torch.ones(nb, S, blk, device = DEV), torch.ones(nb, S, ne, device = DEV)
    M = torch.empty(nb, W, device = DEV, dtype = torch.int32)
    steps = []
    for slot, shift in enumerate((60.0, -60.0)):                              # lf ~ +60, then ~ -60
        nm = torch.randn(nb * blk, B, device = DEV) - 5.0
        nf = nm + shift + torch.randn(nb * blk, B, device = DEV)
        em = torch.randn(E, B, device = DEV) - shift                           # (the products near 1)
        ext = torch.randn(3, B, device = DEV)
        mod.capture(A, Bop, fA, fB, M, nf, nm, em, ext, nids, cids, gate, B, slot, 64, 64, 1, True)
        steps.append(dict(nf = nf, nm = nm, em = em, ext = ext, nids = nids, cids = cids, gate = gate, batch = B,
                          node_cbs = 64, gate_cbs = 64, ext_base = 1, blk = blk, ne = ne))
    mod.flush(pf, A, Bop, fA, fB, pfids, pids, params, S, mod.flush_configs().index([64, 64, 32]), True)
    _check(_block(pf, pfids, blk), _reference(steps, pfids, pids, params), "two captures 120 nats apart")


@needs_kernels
@pytest.mark.parametrize("B,defers", [(256, True), (64, False)])
def test_a_group_below_MIN_FLOWS_defers_on_its_flows_times_samples(B, defers, monkeypatch):
    """A group under `MIN_FLOWS` defers where its flows x samples reach `MIN_FLOW_SAMPLES` (HMM-1024: 1M flows x 256),
    and not below both lines (x 64)."""
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 23)
    monkeypatch.setattr(dpar, "MIN_FLOW_SAMPLES", 1 << 28)
    pc, phi, x = _instance(1024, 1, 64, 6, "bf16", B)
    caps, flushes = _record(monkeypatch)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert len(caps) == (5 if defers else 0), f"B {B}: {len(caps)} steps deferred"
    if defers:
        _check(_block(pc.param_flows, flushes[-1]["pfids"], caps[0]["blk"]),
               _reference(caps, flushes[-1]["pfids"], flushes[-1]["pids"], flushes[-1]["params"]), f"B {B}")


@needs_kernels
@pytest.mark.parametrize("label", ["a layer callback", "untied", "below MIN_FLOWS"])
def test_what_keeps_the_per_step_kernels(label, monkeypatch):
    if label == "below MIN_FLOWS":
        pc, phi, x = _instance(512, 2, 16, 6, "bf16", 64)               # 2 x 256 x 512 = 262K flows < 1M
    else:
        pc, phi, x = _instance(1024, 1, 64, 6, "bf16", 64, tied = label != "untied")
    caps, _ = _record(monkeypatch)
    pc(x, sum_external_params = phi)
    if label == "a layer callback":
        seen = []
        pc.backward(x, flows_memory = 0.0, sum_layer_post_backward_callback = lambda layer, **kw: seen.append(layer))
        assert seen
    else:
        pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert caps == [], f"{label}: {len(caps)} steps deferred"


@needs_kernels
def test_the_deferred_flows_equal_the_per_step_ones_within_their_rounding(monkeypatch):
    pc, phi, x = _instance(1024, 1, 64, 8, "bf16", 128)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    deferred = pc.param_flows.clone()
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 62)                     # no group reaches them: the per-step kernels
    monkeypatch.setattr(dpar, "MIN_FLOW_SAMPLES", 1 << 62)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    fin = torch.isfinite(deferred)
    assert torch.equal(fin, torch.isfinite(pc.param_flows))
    d = float((deferred[fin] - pc.param_flows[fin]).abs().max() / pc.param_flows[fin].abs().max())
    assert d < 3e-3, f"the deferred and per-step flows differ by {d:.2e} of the max"


@needs_kernels
def test_a_cuda_graph_replays_it_to_the_bit(monkeypatch):
    """The group's flows (captures, the flush: no atomics on their values) from a replayed graph, as an eager pass's."""
    pc, phi, x = _instance(1024, 1, 64, 6, "bf16", 64)
    caps, flushes = _record(monkeypatch)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    assert caps and flushes
    pfids, blk = flushes[0]["pfids"], caps[0]["blk"]
    eager = _block(pc.param_flows, pfids, blk).float()
    for _ in range(3):                                                  # record, then replay
        pc(x, sum_external_params = phi)
        pc.backward(x, flows_memory = 0.0, record_cudagraph = True)
    torch.cuda.synchronize()
    assert torch.equal(_block(pc.param_flows, pfids, blk).float().view(torch.int32), eager.view(torch.int32))


@needs_kernels
def test_a_backward_that_raised_leaves_nothing(monkeypatch):
    """Steps captured by a backward that then failed must not reach the next one's flows."""
    pc, phi, x = _instance(1024, 1, 64, 6, "bf16", 64)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    clean = pc.param_flows.clone()
    mod = _module()
    real_flush, calls = mod.flush, []

    def failing_flush(*a):
        calls.append(1)
        raise RuntimeError("injected")

    monkeypatch.setattr(mod, "flush", failing_flush)
    pc(x, sum_external_params = phi)
    with pytest.raises(RuntimeError, match = "injected"):
        pc.backward(x, flows_memory = 0.0)
    assert calls
    monkeypatch.setattr(mod, "flush", real_flush)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    fin = torch.isfinite(clean)
    d = float((pc.param_flows[fin] - clean[fin]).abs().max() / clean[fin].abs().max())
    assert torch.equal(fin, torch.isfinite(pc.param_flows)) and d < 1e-5, f"the next backward's flows off by {d:.2e}"
