"""
Launch plans (`pyjuice.utils.launch_plans`) and the sum layer's planned backward, eager.

The module: a plan is kept only with exactly the launchers expected and none failed, never during a CUDA graph capture
(nor looked up then), never with plans off; a raising call leaves no recording open; recordings are per thread; a
bounded store; `clear`. The sum layer: a planned backward -- the launchers an earlier call with this signature
recorded -- gives the full path's element and node flows to the bit and its parameter flows and gate gradients within
their atomics' rounding, launching the same GPU kernels, over gated HMMs (bf16 and fp32 parameters, tied gates, a
ragged layer, two gated nodes in one layer), with and without gate gradients, deferred parameter flows, batch sizes
alternating, a plain forward in between (no hooks: not planned), and `.to()` dropping the plans.
"""
import os
import sys
import threading

import pytest
import torch

import pyjuice as juice
import pyjuice.utils.launch_plans as lp
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


def _step(pc, phi, x, grad, defer):
    """A gated forward and backward; copies of the flows, parameter flows and gate gradients."""
    grads = {ns: torch.zeros_like(p) for ns, p in phi.items()} if grad else None
    pc(x, sum_external_params = phi)
    kw = dict(sum_external_params_grad = grads) if grad else {}
    if defer:
        kw["defer_param_flows"] = True
    pc.backward(x, flows_memory = 0.0, **kw)
    torch.cuda.synchronize()
    return (pc.node_flows.clone(), pc.element_flows.clone(), pc.param_flows.clone(),
            {ns: g.clone() for ns, g in grads.items()} if grad else {})


def _sum_layers(pc):
    return [layer for group in pc.inner_layer_groups if group.is_sum() for layer in group.layers]


def _gated_layers(pc):
    return [layer for layer in _sum_layers(pc) if hasattr(layer, "external_node_infos")]


def _clear_plans(pc):
    for layer in _sum_layers(pc):
        LaunchPlans.of(layer).clear()


def _same(ref, got, what):
    nf_r, ef_r, pf_r, g_r = ref
    nf, ef, pf, g = got
    bits = lambda t: t.view(torch.int32)
    assert torch.equal(bits(nf), bits(nf_r)), f"{what}: node flows differ"
    assert torch.equal(bits(ef), bits(ef_r)), f"{what}: element flows differ"
    fin = torch.isfinite(pf_r)
    assert torch.equal(fin, torch.isfinite(pf)), f"{what}: parameter flows' finite pattern"
    scale = float(pf_r[fin].abs().max()) if bool(fin.any()) else 0.0
    assert scale > 0, f"{what}: no parameter flow to compare"
    d = float((pf[fin] - pf_r[fin]).abs().max()) / scale
    assert d < 1e-5, f"{what}: parameter flows differ by {d:.1e} (relative)"
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
]


@cuda_only
@pytest.mark.parametrize("grad,defer", [(True, False), (False, False), (True, True)],
                         ids = ["gradient", "no_gradient", "deferred"])
@pytest.mark.parametrize("name,builder,dt,batches", CASES, ids = [c[0] for c in CASES])
def test_a_planned_backward_gives_the_full_paths_results(name, builder, dt, batches, grad, defer, monkeypatch):
    pc, gated, V = _instance(builder, {"bf16": torch.bfloat16, "fp32": torch.float32}[dt])
    inputs = [_inputs(gated, V, B, seed = i) for i, B in enumerate(batches)]

    monkeypatch.setattr(lp, "ENABLED", False)
    for phi, x in inputs:                                  # (warm: every choice tuned and cached)
        _step(pc, phi, x, grad, defer)
    refs = [_step(pc, phi, x, grad, defer) for phi, x in inputs]
    assert all(not LaunchPlans.of(l).plans for l in _sum_layers(pc)), "plans stored while off"

    monkeypatch.setattr(lp, "ENABLED", True)
    for rnd in range(3):                                   # round 0 records, the later ones replay
        for (phi, x), ref, B in zip(inputs, refs, batches):
            _same(ref, _step(pc, phi, x, grad, defer), f"{name} B {B} round {rnd}")
    planned = [l for l in _gated_layers(pc) if LaunchPlans.of(l).plans]
    assert planned and len(planned) == len(_gated_layers(pc)), "the gated layers' backwards were not planned"

    phi, x = inputs[0]
    kw = dict(sum_external_params_grad = {ns: torch.zeros_like(p) for ns, p in phi.items()}) if grad else {}
    if defer:
        kw["defer_param_flows"] = True

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
def test_a_plan_writes_the_buffers_of_its_call():
    """The circuit's flow and marginal buffers swapped for new tensors between calls at one batch size: a planned
    backward writes the ones it is given, not the ones its plan was recorded with (a launcher's tensors are its
    arguments)."""
    pc, gated, V = _instance(lambda: _hmm(512, 2, 32), torch.bfloat16)
    phi, x = _inputs(gated, V, 128, seed = 0)
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
