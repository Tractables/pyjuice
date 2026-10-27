"""
Launch plans (`pyjuice.utils.launch_plans`) and the planned backwards of sum layers, eager.

The module: a plan is kept only with exactly the launchers expected and none failed, never during a CUDA graph capture
(nor looked up then), never with plans off; a raising call leaves no recording open; recordings are per thread and
tagged (a launcher goes to the innermost recording of its tag only; `nested`); a bounded store; `clear`. The layers: a
gated layer's planned backward -- the launchers an earlier call with this signature recorded, the parameterization's
hooks and the standard backward between them replayed whole (`ExternalParamsSumLayer.backward`) -- gives the full
path's element and node flows to the bit and its parameter flows and gate gradients within their atomics' rounding,
launching the same GPU kernels, over gated HMMs (bf16 and fp32 parameters, tied gates, a ragged layer, two gated nodes
in one layer, batches below 16), with and without gate gradients, deferred parameter flows (a tied group deferring),
batch sizes alternating, the call signatures interleaved, a new forward state, a plain forward in between (a plain sum
layer's own plan), and `.to()` dropping the plans; the denominator's torch reference is not planned.
"""
import os
import sys
import threading

import pytest
import torch

import pyjuice as juice
import pyjuice.utils.launch_plans as lp
from pyjuice.nodes import BlockScaleSumParams
from pyjuice.nodes.external_params.blockscale import deferred_par as dpar
from pyjuice.utils.launch_plans import LaunchPlans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_params"))
from blockscale_ele_backward_prepass_test import _hmm, _topology, _two, PAD, CATS  # noqa: E402
from blockscale_ele_backward_stream_test import _tied_hmm                          # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
DEV = torch.device("cuda:0")


class _Owner:
    pass


# --------------------------------------------------------------------------------- the module

@cuda_only
def test_a_plan_needs_exactly_the_expected_launchers():
    o = _Owner()
    plans = LaunchPlans.of(o)
    assert LaunchPlans.of(o) is plans
    for n, expected, failed, kept in [(2, 2, False, True), (1, 2, False, False), (3, 2, False, False),
                                      (2, 2, True, False), (0, 0, False, True)]:
        plans.clear()
        with LaunchPlans.record() as rec:
            for i in range(n):
                LaunchPlans.add(lambda *a, i = i: i)
            if failed:
                LaunchPlans.fail()
        plans.store("k", rec, expected)
        assert (plans.lookup("k") is not None) == kept, (n, expected, failed)
    assert plans.lookup(None) is None


@cuda_only
def test_nothing_is_added_outside_a_recording_and_a_raise_closes_it():
    LaunchPlans.add(lambda: None)                       # (no recording open: a no-op)
    LaunchPlans.fail()
    with pytest.raises(RuntimeError):
        with LaunchPlans.record():
            raise RuntimeError("boom")
    assert not getattr(LaunchPlans._local, "open", [])


@cuda_only
def test_plans_off_and_captures_take_the_full_path(monkeypatch):
    o = _Owner()
    plans = LaunchPlans.of(o)
    with LaunchPlans.record() as rec:
        LaunchPlans.add(lambda: None)
    plans.store("k", rec, 1)
    assert plans.lookup("k") is not None
    monkeypatch.setattr(lp, "ENABLED", False)
    assert plans.lookup("k") is None
    with LaunchPlans.record() as rec2:
        LaunchPlans.add(lambda: None)
    plans.store("k2", rec2, 1)
    assert "k2" not in plans.plans
    monkeypatch.setattr(lp, "ENABLED", True)
    x = torch.zeros(4, device = DEV)
    g = torch.cuda.CUDAGraph()
    seen = {}
    with torch.cuda.graph(g):
        x.add_(1.0)
        seen["lookup"] = plans.lookup("k")
        with LaunchPlans.record() as rec3:
            LaunchPlans.add(lambda: None)
        plans.store("k3", rec3, 1)
    assert seen["lookup"] is None and "k3" not in plans.plans


@cuda_only
def test_recordings_are_per_thread():
    with LaunchPlans.record() as rec:
        t = threading.Thread(target = lambda: LaunchPlans.add(lambda: "other thread"))
        t.start()
        t.join()
        LaunchPlans.add(lambda: "this thread")
    assert [f() for f in rec.launchers] == ["this thread"]


@cuda_only
def test_the_store_is_bounded():
    o = _Owner()
    plans = LaunchPlans.of(o)
    for i in range(LaunchPlans.MAX_PLANS + 5):
        with LaunchPlans.record() as rec:
            LaunchPlans.add(lambda: None)
        plans.store(i, rec, 1)
    assert len(plans.plans) <= LaunchPlans.MAX_PLANS


@cuda_only
def test_a_launcher_goes_to_the_innermost_recording_of_its_tag():
    assert not LaunchPlans.nested() and not LaunchPlans.recording("outer")
    with LaunchPlans.record("outer") as outer:
        assert LaunchPlans.nested() and LaunchPlans.recording("outer") and not LaunchPlans.recording()
        LaunchPlans.add(lambda: "outer 1", "outer")
        LaunchPlans.add(lambda: "untagged")                 # (no untagged recording open: dropped)
        with LaunchPlans.record() as inner:
            assert LaunchPlans.recording("outer") and LaunchPlans.recording()
            LaunchPlans.add(lambda: "outer 2", "outer")     # (to the enclosing one of its tag)
            LaunchPlans.add(lambda: "inner")
            with LaunchPlans.record("outer") as outer2:
                LaunchPlans.add(lambda: "outer2", "outer")  # (the innermost of the tag only)
        LaunchPlans.add(lambda: "outer 3", "outer")
    assert [f() for f in outer.launchers] == ["outer 1", "outer 2", "outer 3"]
    assert [f() for f in inner.launchers] == ["inner"]
    assert [f() for f in outer2.launchers] == ["outer2"]
    assert not LaunchPlans.nested()


@cuda_only
def test_an_incomplete_inner_recording_fails_the_ones_around_it():
    o = _Owner()
    plans = LaunchPlans.of(o)
    for inner_count, outer_kept in [(1, True), (0, False), (2, False)]:
        plans.clear()
        with LaunchPlans.record("outer") as outer:
            LaunchPlans.add(lambda: None, "outer")
            with LaunchPlans.record() as inner:
                for _ in range(inner_count):
                    LaunchPlans.add(lambda: None)
            plans.store("inner", inner, 1)
        plans.store("outer", outer, 1)
        assert (plans.lookup("inner") is not None) == (inner_count == 1)
        assert (plans.lookup("outer") is not None) == outer_kept, inner_count
    with LaunchPlans.record("a") as a:                      # `fail` marks every recording open
        with LaunchPlans.record("b") as b:
            LaunchPlans.fail()
    assert a.failed and b.failed


# --------------------------------------------------------------------------------- the sum layer's backward

def _instance(builder, dt, seed = 0):
    torch.manual_seed(seed)
    root, gated, V = builder()
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = dt).to(DEV)
    return pc, gated, V


def _inputs(gated, V, B, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    phi = {ns: (torch.randn(ns.external_params.tensor_shapes(ns, B)[0], generator = g) * 2.0).to(DEV)
           for ns in gated}
    x = torch.randint(0, CATS, (B, V), generator = g).to(DEV)
    return phi, x


def _backward_kwargs(phi, grad, defer):
    """The backward's kwargs: the gate gradients into new buffers, or none computed (`compute_external_grads`, on by
    default)."""
    if grad:
        kw = dict(sum_external_params_grad = {ns: torch.zeros_like(p) for ns, p in phi.items()})
    else:
        kw = dict(compute_external_grads = False)
    if defer:
        kw["defer_param_flows"] = True
    return kw


def _step(pc, phi, x, grad, defer, param_flows = True, **backward_kw):
    """A gated forward and backward; copies of the flows, parameter and denominator flows (None without), gate
    gradients and the node values it leaves (the `log Z` shift undone)."""
    pc(x, sum_external_params = phi)
    kw = _backward_kwargs(phi, grad, defer)
    pc.backward(x, flows_memory = 0.0, compute_param_flows = param_flows, **kw, **backward_kw)
    torch.cuda.synchronize()
    return (pc.node_flows.clone(), pc.element_flows.clone(), pc.param_flows.clone() if param_flows else None,
            {ns: g.clone() for ns, g in kw["sum_external_params_grad"].items()} if grad else {},
            pc.denom_param_flows.clone() if param_flows else None, pc.node_mars.clone())


def _sum_layers(pc):
    return [layer for group in pc.inner_layer_groups if group.is_sum() for layer in group.layers]


def _gated_layers(pc):
    return [layer for layer in _sum_layers(pc) if hasattr(layer, "external_node_infos")]


def _clear_plans(pc):
    for layer in _sum_layers(pc):
        LaunchPlans.of(layer).clear()


def _layer_keys(layer):
    """The keys of a gated layer's plans of its whole backward (`ExternalParamsSumLayer._backward_plan_key`)."""
    return [k for k in LaunchPlans.of(layer).plans if k[0] == "external_backward"]


def _count_full_paths(monkeypatch):
    """`[n]`: how many gated backwards take the full path (`pre_backward_layer`) from now on."""
    n = [0]
    real = BlockScaleSumParams.pre_backward_layer

    def counted(self, *args, **kwargs):
        n[0] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(BlockScaleSumParams, "pre_backward_layer", counted)
    return n


def _count_deferred(monkeypatch):
    """`[n]`: how many partitions' parameter flows `deferred_par.try_defer` takes from now on."""
    n = [0]
    real = dpar.try_defer

    def counted(*args):
        took = real(*args)
        n[0] += int(took)
        return took

    monkeypatch.setattr(dpar, "try_defer", counted)
    return n


def _close(ref, got, what):
    """`got` is `ref` within atomics' rounding: the same finite pattern, 1e-5 of the largest apart."""
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(got)), f"{what}' finite pattern"
    scale = float(ref[fin].abs().max()) if bool(fin.any()) else 0.0
    assert scale > 0, f"{what}: nothing to compare"
    d = float((got[fin] - ref[fin]).abs().max()) / scale
    assert d < 1e-5, f"{what} differ by {d:.1e} (relative)"


def _same(ref, got, what):
    nf_r, ef_r, pf_r, g_r, w_r, nm_r = ref
    nf, ef, pf, g, w, nm = got
    bits = lambda t: t.view(torch.int32)
    assert torch.equal(bits(nf), bits(nf_r)), f"{what}: node flows differ"
    assert torch.equal(bits(ef), bits(ef_r)), f"{what}: element flows differ"
    assert torch.equal(bits(nm), bits(nm_r)), f"{what}: the node values left differ"
    if pf_r is not None:                                   # (None: a backward without parameter flows)
        _close(pf_r, pf, f"{what}: parameter flows")
        _close(w_r, w, f"{what}: denominator flows")
    for ns in g_r:
        nan = torch.isnan(g_r[ns])
        assert torch.equal(nan, torch.isnan(g[ns])), f"{what}: the gradient's NaN pattern"
        s = float(g_r[ns][~nan].abs().max())
        assert s > 0, f"{what}: no gradient to compare"
        d = float((g[ns] - g_r[ns])[~nan].abs().max()) / s
        assert d < 1e-5, f"{what}: the gate gradient differs by {d:.1e} (relative)"


def _kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted(e.name for e in prof.events() if e.device_type.name == "CUDA")


CASES = [
    ("hmm512x2_g32_bf16", lambda: _hmm(512, 2, 32), "bf16", (128,)),
    ("hmm1024_g16_fp32", lambda: _hmm(1024, 1, 16), "fp32", (64,)),
    ("tied_hmm1024x4_g16", lambda: _tied_hmm(1024, 4, 16), "bf16", (128,)),
    ("pad", lambda: _topology(PAD, 128, 128, 16), "bf16", (128,)),
    ("two_nodes", lambda: _two(8, bs = 128), "bf16", (64,)),
    ("alternating_batches", lambda: _hmm(512, 2, 32), "bf16", (64, 128, 64, 128)),
    ("small_batches", lambda: _hmm(512, 2, 32), "bf16", (8, 12)),         # the small-batch forks
]


@cuda_only
@pytest.mark.parametrize("grad,defer", [(True, False), (False, False), (True, True)],
                         ids = ["gradient", "no_gradient", "deferred"])
@pytest.mark.parametrize("name,builder,dt,batches", CASES, ids = [c[0] for c in CASES])
def test_a_planned_backward_gives_the_full_paths_results(name, builder, dt, batches, grad, defer, monkeypatch):
    if defer:                                              # (the tied case's groups of 1M flows defer)
        monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    deferred = _count_deferred(monkeypatch)
    pc, gated, V = _instance(builder, {"bf16": torch.bfloat16, "fp32": torch.float32}[dt])
    inputs = [_inputs(gated, V, B, seed = i) for i, B in enumerate(batches)]

    monkeypatch.setattr(lp, "ENABLED", False)
    for phi, x in inputs:                                  # (warm: every choice tuned and cached)
        _step(pc, phi, x, grad, defer)
    refs = [_step(pc, phi, x, grad, defer) for phi, x in inputs]
    assert all(not LaunchPlans.of(l).plans for l in _sum_layers(pc)), "plans stored while off"
    if defer and name.startswith("tied"):
        assert deferred[0] > 0, "the tied group did not defer"

    monkeypatch.setattr(lp, "ENABLED", True)
    full = _count_full_paths(monkeypatch)
    for rnd in range(3):                                   # round 0 records, the later ones replay
        for (phi, x), ref, B in zip(inputs, refs, batches):
            _same(ref, _step(pc, phi, x, grad, defer), f"{name} B {B} round {rnd}")
        if rnd == 0:
            recorded, deferred[0] = full[0], 0
    assert recorded > 0 and full[0] == recorded, "a gated backward took the full path after its plan was recorded"
    if defer and name.startswith("tied"):
        assert deferred[0] > 0, "a replayed backward did not defer the tied group"
    assert all(_layer_keys(l) for l in _gated_layers(pc)), "the gated layers' backwards were not planned"

    phi, x = inputs[0]
    kw = _backward_kwargs(phi, grad, defer)

    def profiled():                                        # (after a call at the same batch: no buffer resized)
        _step(pc, phi, x, grad, defer)
        pc(x, sum_external_params = phi)
        return _kernels(lambda: pc.backward(x, flows_memory = 0.0, **kw))

    with_plans = profiled()
    monkeypatch.setattr(lp, "ENABLED", False)
    without = profiled()
    assert with_plans == without, "a planned backward launched other kernels than the full path"


@cuda_only
def test_a_plain_forward_in_between_is_not_planned_and_does_not_disturb():
    """A forward without `sum_external_params` runs the gated layers as plain sum layers: their backward has no hooks,
    so another key -- the unconverted plain paths, never planned -- and the gated plan is still right after."""
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    phi, x = _inputs(gated, V, 128, seed = 0)
    lp_enabled = lp.ENABLED
    try:
        lp.ENABLED = False
        _step(pc, phi, x, True, False)
        ref = _step(pc, phi, x, True, False)
        pc(x)
        pc.backward(x, flows_memory = 0.0)
        torch.cuda.synchronize()
        plain = (pc.node_flows.clone(), pc.element_flows.clone())
        lp.ENABLED = True
        _clear_plans(pc)
        _same(ref, _step(pc, phi, x, True, False), "gated, recording")
        for _ in range(2):
            pc(x)
            pc.backward(x, flows_memory = 0.0)
            torch.cuda.synchronize()
            assert torch.equal(pc.node_flows, plain[0]) and torch.equal(pc.element_flows, plain[1])
            _same(ref, _step(pc, phi, x, True, False), "gated after a plain backward")
    finally:
        lp.ENABLED = lp_enabled


@cuda_only
def test_a_layers_move_drops_its_plans():
    """What `pc.to()` does to each inner layer when the device changes (`SumLayer.to`): the plans go with the
    device caches. (`pc.to()` onto the device it is on moves nothing and keeps them: they are still right.)"""
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    phi, x = _inputs(gated, V, 128, seed = 0)
    for _ in range(2):
        _step(pc, phi, x, True, False)
    layers = _gated_layers(pc)
    assert layers and all(LaunchPlans.of(l).plans for l in layers)
    for layer in layers:
        layer.to(DEV)
    assert not any(LaunchPlans.of(l).plans for l in layers)
    _step(pc, phi, x, True, False)
    assert all(LaunchPlans.of(l).plans for l in layers)


def _two_gpus():
    """Two GPU ids for a real move, from the pool the conftest published (a worker sees ONE GPU: see
    tests/queries/sample/sample_scope_plan_test.py), or None."""
    pool = [g for g in os.environ.get("PYJUICE_TEST_GPU_POOL", "").split(",") if g]
    if len(pool) < 2:
        return None
    idle = pool[int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")):]
    return idle[:2] if len(idle) >= 2 else pool[-2:]


_MOVE_CHILD = r"""
import os, sys, torch
sys.path.insert(0, os.environ["LP_TESTS"])
import pyjuice as juice
from pyjuice.utils.launch_plans import LaunchPlans
from blockscale_ele_backward_prepass_test import _hmm, CATS

def build(dev):
    torch.manual_seed(0)
    root, gated, V = _hmm(512, 2, 32)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(dev), gated, V

def step(pc, gated, V, dev):
    g = torch.Generator(device = "cpu").manual_seed(1)
    phi = {ns: (torch.randn(ns.external_params.tensor_shapes(ns, 128)[0], generator = g) * 2.0).to(dev)
           for ns in gated}
    x = torch.randint(0, CATS, (128, V), generator = g).to(dev)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0)
    torch.cuda.synchronize(dev)
    return pc.node_flows.cpu(), pc.element_flows.cpu()

d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
pc, gated, V = build(d0)
for _ in range(3):
    step(pc, gated, V, d0)
layers = [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers if hasattr(l, "external_node_infos")]
assert layers and all(LaunchPlans.of(l).plans for l in layers), "no plan on cuda:0"
pc.to(d1)
assert not any(LaunchPlans.of(l).plans for l in layers), "plans survived the move"
moved = [step(pc, gated, V, d1) for _ in range(3)][-1]
assert all(LaunchPlans.of(l).plans for l in layers), "no plan on cuda:1"
ref_pc, ref_gated, _ = build(d1)
ref = [step(ref_pc, ref_gated, V, d1) for _ in range(3)][-1]
assert torch.equal(moved[0], ref[0]) and torch.equal(moved[1], ref[1]), "the moved circuit's flows differ"
print("CHILD_OK")
"""


@cuda_only
@pytest.mark.skipif(_two_gpus() is None, reason = "needs two GPUs in PYJUICE_TEST_GPU_POOL")
def test_a_real_device_move_replans_on_the_new_device():
    """In a CHILD with two GPUs visible: plans made on cuda:0 are dropped by `pc.to("cuda:1")`, made again there,
    and the moved circuit's flows equal those of one built on cuda:1."""
    import subprocess
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(_two_gpus())
    env.pop("PYJUICE_TEST_GPU_POOL", None)
    env["LP_TESTS"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_params")
    proc = subprocess.run([sys.executable, "-c", _MOVE_CHILD], capture_output = True, text = True, timeout = 900,
                          env = env)
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, (
        f"exit {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}")


@cuda_only
@pytest.mark.parametrize("B", [128, 8])
def test_a_plan_writes_the_buffers_of_its_call(B):
    """The circuit's flow and marginal buffers swapped for new tensors between calls at one batch size: a planned
    backward writes the ones it is given, not the ones its plan was recorded with (a launcher's tensors are its
    arguments) -- at a batch below 16 too (the small-batch forks)."""
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    phi, x = _inputs(gated, V, B, seed = 0)
    lp_enabled = lp.ENABLED
    try:
        lp.ENABLED = False
        _step(pc, phi, x, True, False)
        ref = _step(pc, phi, x, True, False)
        lp.ENABLED = True
        _clear_plans(pc)
        for _ in range(2):
            _step(pc, phi, x, True, False)
        assert all(LaunchPlans.of(l).plans for l in _gated_layers(pc))
        for name in ("node_flows", "element_flows", "node_mars", "element_mars"):
            old = getattr(pc, name)
            setattr(pc, name, torch.full_like(old, float("nan")))
            assert getattr(pc, name).data_ptr() != old.data_ptr()
        _same(ref, _step(pc, phi, x, True, False), "new buffers")
    finally:
        lp.ENABLED = lp_enabled


@cuda_only
def test_a_replay_reads_the_forward_state_of_its_call(monkeypatch):
    """The gated layers' forward plans dropped before a forward with other gates: the forward builds a new state --
    new `log Z` buffers holding other values -- and a replayed backward, recorded against the old one, shifts and
    unshifts by the new."""
    import pyjuice.layer.kernels.c as ck
    monkeypatch.setattr(ck, "autotune", lambda cands, *a, **k: cands[0][0])   # (a rebuilt plan picks the same)
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    (phi_a, x), (phi_b, _) = _inputs(gated, V, 128, seed = 0), _inputs(gated, V, 128, seed = 1)
    monkeypatch.setattr(lp, "ENABLED", False)
    _step(pc, phi_b, x, True, False)
    ref = _step(pc, phi_b, x, True, False)
    monkeypatch.setattr(lp, "ENABLED", True)
    for _ in range(2):
        _step(pc, phi_a, x, True, False)                   # recorded, then replayed
    layers = _gated_layers(pc)
    assert all(_layer_keys(l) for l in layers)
    old = [l._bs_bw_state for l in layers]
    for l in layers:
        l.__dict__["_bs_fw_plans"].clear()
    full = _count_full_paths(monkeypatch)
    got = _step(pc, phi_b, x, True, False)
    assert full[0] == 0, "the backward was not replayed"
    assert all(l._bs_bw_state is not s for l, s in zip(layers, old)), "the forward kept its state"
    _same(ref, got, "a new forward state")


@cuda_only
def test_each_call_signature_gets_its_own_plan(monkeypatch):
    """With and without the gate gradient, with and without parameter flows (a query's backward: no denominator flow,
    no deferral), and with a layer callback (no deferral), interleaved on one circuit whose tied group defers: each its
    own plan, each the full path's results."""
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    deferred = _count_deferred(monkeypatch)
    pc, gated, V = _instance(lambda: _tied_hmm(1024, 4, 16), torch.bfloat16)
    phi, x = _inputs(gated, V, 64, seed = 0)

    def run(kind):
        grad = kind in ("gradient", "query_gradient", "callback")
        kw = dict(sum_layer_post_backward_callback = lambda layer, **k: None) if kind == "callback" else {}
        return _step(pc, phi, x, grad, False, param_flows = not kind.startswith("query"), **kw)

    kinds = ("gradient", "plain", "query", "query_gradient", "callback")
    monkeypatch.setattr(lp, "ENABLED", False)
    refs = {k: [run(k) for _ in range(2)][-1] for k in kinds}
    monkeypatch.setattr(lp, "ENABLED", True)
    for rnd in range(2):
        for k in ("gradient", "query", "callback", "plain", "query_gradient", "query", "gradient", "callback",
                  "plain", "query_gradient"):
            deferred[0] = 0
            _same(refs[k], run(k), f"{k} round {rnd}")
            assert (deferred[0] > 0) == (k in ("gradient", "plain")), f"{k} round {rnd}: deferred {deferred[0]}"
    for layer in _gated_layers(pc):
        assert len(_layer_keys(layer)) == len(kinds), "not one plan a call signature"


@cuda_only
def test_the_denominators_torch_reference_is_not_planned(monkeypatch):
    """`PYJUICE_BLOCKSCALE_DENOM_REF=1` runs the reference beside the kernels: the gated backwards take the full path
    every call (their standard part still plans itself) and give the same results as with plans off."""
    monkeypatch.setenv("PYJUICE_BLOCKSCALE_DENOM_REF", "1")
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    phi, x = _inputs(gated, V, 128, seed = 0)
    monkeypatch.setattr(lp, "ENABLED", False)
    _step(pc, phi, x, True, False)
    ref = _step(pc, phi, x, True, False)
    denom_ref = pc.denom_param_flows.clone()
    monkeypatch.setattr(lp, "ENABLED", True)
    for _ in range(3):
        _same(ref, _step(pc, phi, x, True, False), "the reference")
        assert torch.equal(pc.denom_param_flows, denom_ref)
    assert not any(_layer_keys(l) for l in _gated_layers(pc))


@cuda_only
def test_a_plan_reads_the_staging_buffers_of_its_call(monkeypatch):
    """A batch big enough to move the circuit's buffers to new storage between a plan's recording and its replay, and
    other gates at the replay: the replayed backward reads the staging buffer and writes the gradient buffer of its
    call, not those it was recorded with (which, held, would still hold the recorded call's values)."""
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    (phi_a, x), (phi_b, _) = _inputs(gated, V, 64, seed = 0), _inputs(gated, V, 64, seed = 1)
    phi_big, x_big = _inputs(gated, V, 1024, seed = 2)
    monkeypatch.setattr(lp, "ENABLED", False)
    for phi_, x_ in ((phi_b, x), (phi_big, x_big)):       # (warm: every choice tuned and cached)
        _step(pc, phi_, x_, True, False)
    ref = _step(pc, phi_b, x, True, False)
    monkeypatch.setattr(lp, "ENABLED", True)
    for _ in range(2):
        _step(pc, phi_a, x, True, False)                   # recorded, then replayed
    old = (pc.external_params.data_ptr(), pc.external_params_grad.data_ptr())
    _step(pc, phi_big, x_big, True, False)                 # the buffers' storage grows ...
    full = _count_full_paths(monkeypatch)
    got = _step(pc, phi_b, x, True, False)                 # ... and batch 64 is cut from a new one
    assert full[0] == 0, "the backward was not replayed"
    new = (pc.external_params.data_ptr(), pc.external_params_grad.data_ptr())
    assert new[0] != old[0] and new[1] != old[1], "the staging buffers did not move"
    _same(ref, got, "staging buffers moved")


@cuda_only
def test_an_unplanned_call_inside_a_recording_fails_it(monkeypatch):
    """A recording around a call that records nothing of its own -- a sum layer in a regime it does not plan (flows
    modified in place), a gated layer whose parameterization does not plan it (the denominator's torch reference) --
    cannot capture what that call launches, so the call fails it; a planned gated layer leaves it as it was."""
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    phi, x = _inputs(gated, V, 64, seed = 0)

    pc(x)                                                  # (plain: the gated layers as plain sum layers)
    with LaunchPlans.record("outer") as outer:
        pc.backward(x, flows_memory = 0.0, allow_modify_flows = True, logspace_flows = False)
    assert outer.failed, "an unplanned sum layer's backward left the recording around it replayable"

    def around_gated_layers(denom_ref):
        if denom_ref:
            monkeypatch.setenv("PYJUICE_BLOCKSCALE_DENOM_REF", "1")
        else:
            monkeypatch.delenv("PYJUICE_BLOCKSCALE_DENOM_REF", raising = False)
        recs, open_ = [], []

        def pre(layer, **kw):                              # (a recording around each gated layer's backward)
            if hasattr(layer, "external_node_infos"):
                cm = LaunchPlans.record("outer")
                recs.append(cm.__enter__())
                open_.append(cm)

        def post(layer, **kw):
            while open_:
                open_.pop().__exit__(None, None, None)

        for _ in range(2):                                 # (the second call replays the plan where there is one)
            pc(x, sum_external_params = phi)
            pc.backward(x, flows_memory = 0.0, sum_layer_pre_backward_callback = pre,
                        sum_layer_post_backward_callback = post)
        return recs

    recs = around_gated_layers(True)
    assert recs and all(r.failed for r in recs), "an unplanned gated layer left the recording around it replayable"
    recs = around_gated_layers(False)
    assert recs and not any(r.failed for r in recs), "a planned gated layer failed the recording around it"
