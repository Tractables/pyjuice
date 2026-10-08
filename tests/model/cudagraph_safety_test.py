"""
The guarantees behind replaying a recorded CUDA graph of the inner layers (`TensorCircuit._run_inner_pass`):

* a graph is replayed only under exactly what it was recorded with -- the buffers, every option the inner
  layers are called with, and, for a backward, the state the forward left on the layers;
* per-call evidence that only the input layers read is left out of the key, so it does not stop a
  replay -- sound because no inner layer reads it (checked over their source);
* nothing a graph baked a pointer to is freed while the graph lives;
* recording is bounded: in the number of graphs kept, and in how often one pass shape is recorded;
* graphs are OFF unless asked for (`record_cudagraph = True`, or `juice.compile(..., cuda_graphs = True)`),
  and whatever cannot be captured -- a host sync, a caller's own capture, a parameterization that does
  not declare itself graph-safe -- runs eagerly instead of failing.
"""

import ast
import gc
import pathlib
import weakref

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate


def _hmm(device, seed = 0, **compile_kwargs):
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(seq_length = 8, num_latents = 32, homogeneous = True,
                                         input_dist = dists.Categorical(num_cats = 10))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, **compile_kwargs).to(device)


@pytest.fixture
def replays(monkeypatch):
    """How many CUDA graphs were replayed."""
    count = [0]
    replay = torch.cuda.CUDAGraph.replay

    def counting(self):
        count[0] += 1
        return replay(self)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counting)
    return count


def test_every_option_the_inner_layers_receive_is_part_of_the_key(replays):
    """The key is built from the very dict the inner layers are called with, so even an option this test
    makes up -- which no layer reads -- separates graphs: nothing can reach a recorded pass unkeyed."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)

    pc(data, record_cudagraph = True, some_option = 1)
    assert replays[0] == 0, "the recording call replayed its own graph (it runs the pass eagerly)"
    pc(data, some_option = 2)
    assert replays[0] == 0, "a graph recorded under `some_option = 1` was replayed under 2"
    pc(data, some_option = 1)
    assert replays[0] == 1


def test_per_call_evidence_does_not_stop_replays(replays):
    """Soft evidence is a fresh tensor on every call, and only the input layers read it
    (`SoftEvidenceCategorical.call_kwargs`), so it is not in the key: one graph
    per pass, replayed on every later call, and every call -- recording or replaying -- matches eager."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    B, V, K, C = 6, 5, 4, 12
    ns = juice.structures.GeneralizedHMM(seq_length = V, num_latents = 16, homogeneous = True,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = C))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(device)

    def step(seed, graphs):
        g = torch.Generator(device = device).manual_seed(seed)
        data = torch.randint(0, C, [B, V], device = device, generator = g)
        ids = torch.stack([torch.randperm(C, device = device, generator = g)[:K] for _ in range(B * V)])
        ids = ids.view(B, V, K)
        ids[:, :, -1] = torch.where((ids == data[..., None]).any(-1), ids[:, :, -1], data)
        ids = ids.sort(dim = 2)[0].contiguous()
        lp = torch.log_softmax(torch.randn(B, V, K, device = device, generator = g), dim = 2).contiguous()
        grad = torch.zeros_like(lp)
        pc.init_param_flows(flows_memory = 0.0)
        lls = pc(data, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids,
                 record_cudagraph = graphs, apply_cudagraph = graphs).clone()
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, categorical_evidence_logp = lp,
                    soft_evidence_cat_ids = ids, categorical_evidence_logp_grad = grad,
                    record_cudagraph = graphs, apply_cudagraph = graphs)
        return lls, grad, pc.param_flows.clone()

    for seed in range(4):
        ref = step(seed, graphs = False)
        got = step(seed, graphs = True)
        for name, a, b in zip(("log-likelihoods", "evidence gradient", "parameter flows"), got, ref):
            assert torch.allclose(a, b, rtol = 1e-5, atol = 1e-6), f"step {seed}: {name} differ from eager"

    assert len(pc._recorded_cuda_graphs) == 2, f"{len(pc._recorded_cuda_graphs)} graphs for one batch size"
    assert replays[0] == 2 * 3, f"{replays[0]} replays over the three steps after the recording one"


def test_the_number_of_graphs_is_bounded():
    device = torch.device("cuda:0")
    pc = _hmm(device)
    pc.max_cuda_graphs = 4
    batches = [12, 3, 5, 7, 9, 11, 3, 12, 7, 5]               # six sizes through room for four
    data = {B: torch.randint(0, 10, [B, 8], device = device) for B in set(batches)}
    reference = {B: pc(data[B], apply_cudagraph = False).clone() for B in set(batches)}

    for B in batches:
        lls = pc(data[B], record_cudagraph = True)
        assert torch.allclose(lls, reference[B], atol = 1e-5), f"batch {B} differs from eager"
        assert len(pc._recorded_cuda_graphs) <= 4


def test_an_option_new_on_every_call_stops_recording():
    """A per-call tensor no input distribution declares is keyed on, so every call records -- a capture
    each, replaying nothing. After `max_cuda_graphs_per_layout` recordings for one buffer layout the pass
    stops recording, and says which option kept changing."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)
    limit = pc.max_cuda_graphs_per_layout
    fresh = [torch.zeros(3, device = device) for _ in range(limit + 3)]      # all alive: distinct addresses
    reference = pc(data, apply_cudagraph = False).clone()

    with pytest.warns(RuntimeWarning, match = "per_call_tensor"):
        for t in fresh:
            lls = pc(data, record_cudagraph = True, per_call_tensor = t)
            assert torch.allclose(lls, reference, atol = 1e-5)
    assert len(pc._recorded_cuda_graphs) == limit


# ---- layers that hand state from the forward to the backward (`Layer.cuda_graph_state`) ----

def _gated_pc(device, kind):
    """One external-parameter sum layer over a product of two inputs: `kind` is "blockscale" or "lowrank"."""
    from pyjuice.nodes import BlockScaleSumParams, LowRankSumParams
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 6)) for v in range(2)]
        ext = BlockScaleSumParams(ch_block_size = 16) if kind == "blockscale" else LowRankSumParams(rank = 4)
        ns = summate(multiply(*ni), num_node_blocks = 2, external_params = ext)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(device), ns


def _gated_inputs(ns, batch, device, seed):
    g = torch.Generator(device = device).manual_seed(seed)
    data = torch.randint(0, 6, [batch, 2], device = device, generator = g)
    shapes = ns.external_params.tensor_shapes(ns, batch)
    tensors = tuple(torch.randn(s, device = device, generator = g) * 0.5 - 0.5 for s in shapes)
    return data, {ns: tensors if len(tensors) > 1 else tensors[0]}


@pytest.mark.parametrize("kind", ["blockscale", "lowrank"])
def test_a_replayed_forward_hands_its_own_state_to_the_backward(kind):
    """The gated forward leaves its `log Z` on the layer for the backward. A replay runs no Python, so after
    forwards at batch 32 and 16 a replayed batch-32 forward left batch 16's state for the next backward;
    the circuit now restores what the recording left."""
    if kind == "lowrank":
        from pyjuice.nodes.external_params.kernels.c import is_available
        if not is_available():
            pytest.skip("the low-rank backward needs the CUDA extension")
    device = torch.device("cuda:0")
    pc, ns = _gated_pc(device, kind)
    (xa, ea), (xb, eb) = _gated_inputs(ns, 32, device, 1), _gated_inputs(ns, 16, device, 2)

    def flows(x, e, record, apply):
        pc(x, sum_external_params = e, record_cudagraph = record, apply_cudagraph = apply)
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(x, allow_modify_flows = False, record_cudagraph = record, apply_cudagraph = apply)
        return pc.param_flows.clone()

    ref = flows(xa, ea, False, False)
    flows(xa, ea, True, True)                                # record forward and backward at 32 ...
    flows(xb, eb, True, True)                                # ... and at 16

    # Replayed forward at 32, then an EAGER backward: it reads whatever state the layer holds
    pc(xa, sum_external_params = ea)
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(xa, allow_modify_flows = False, apply_cudagraph = False)
    assert torch.allclose(pc.param_flows, ref, rtol = 1e-5, atol = 1e-6), "eager backward after a replayed forward"

    # Replayed forward at 32 after an eager forward at 16, then a REPLAYED backward
    pc(xb, sum_external_params = eb, apply_cudagraph = False)
    assert torch.allclose(flows(xa, ea, False, True), ref, rtol = 1e-5, atol = 1e-6), "replayed pair"


def test_a_graph_keeps_what_it_baked_in_alive(monkeypatch):
    """BlockScale keeps a plan per batch size, each with its own `log Z` buffers, and evicts the oldest. A
    graph recorded with a plan bakes those buffers in; once evicted, a replay would write into memory since
    handed to something else. The graph holds a reference to everything the inner layers owned when it was
    captured, so the buffer stays alive exactly as long as the graph does."""
    import pyjuice.nodes.external_params.block_scale as block_scale
    monkeypatch.setattr(block_scale, "_FW_PLAN_CACHE", 1)
    device = torch.device("cuda:0")
    pc, ns = _gated_pc(device, "blockscale")
    (xa, ea), (xb, eb) = _gated_inputs(ns, 32, device, 1), _gated_inputs(ns, 16, device, 2)
    reference = pc(xa, sum_external_params = ea, apply_cudagraph = False).clone()
    layer = next(layer for group in pc.inner_layer_groups for layer in group if hasattr(layer, "_bs_bw_state"))

    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v)

    # The whole plan: its launch arguments as well as the state it hands the backward (which the graph
    # also keeps, to restore it after a replay)
    pc(xa, sum_external_params = ea, record_cudagraph = True)
    baked = [weakref.ref(t) for t in tensors(layer._bs_fw_plans[32])]
    assert baked, "the plan holds no tensors -- the test needs one to watch"
    pc(xb, sum_external_params = eb, apply_cudagraph = False)    # evicts the batch-32 plan
    gc.collect()
    assert all(r() is not None for r in baked), "a tensor the graph baked in was freed while it lives"

    assert torch.allclose(pc(xa, sum_external_params = ea), reference, atol = 1e-5)

    # The plan's own buffers die with the graph (the state also names tables the layer keeps anyway). If
    # none did, something else held them all along and the check above proved nothing.
    pc._drop_cuda_graphs()
    layer._bs_bw_state = None
    gc.collect()
    assert any(r() is None for r in baked), "nothing was freed with the graph: the test watched nothing it held"


# ---- static guard ----

def test_inner_layers_never_read_an_input_only_kwarg():
    """The graph key leaves out `Distribution.call_kwargs` (and `InputLayer.call_kwargs`). That is only
    sound if no inner layer reads one: none may appear in their source as a string or as the name of a
    parameter (which `**kwargs` would bind)."""
    from pyjuice.layer.input_layer import InputLayer
    from pyjuice.nodes.distributions.distributions import Distribution

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    names = set(InputLayer.call_kwargs)
    for cls in subclasses(Distribution):
        names.update(cls.call_kwargs)
    assert "categorical_evidence_logp" in names

    src = pathlib.Path(juice.__file__).parent
    files = [src / "layer" / f for f in ("sum_layer.py", "prod_layer.py", "external_sum_layer.py",
                                         "layer.py", "layer_group.py")]
    files += sorted((src / "layer" / "kernels").rglob("*.py")) + sorted((src / "nodes" / "external_params").rglob("*.py"))

    found = []
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Constant) and node.value in names:
                found.append((f.name, node.lineno, node.value))
            elif isinstance(node, ast.arg) and node.arg in names:
                found.append((f.name, node.lineno, node.arg))
    assert not found, f"inner-layer code reads input-only call kwargs: {found}"


# ---- the `cuda_graphs` compile option, and falling back to eager ----

def test_graphs_are_off_by_default(replays):
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)
    assert pc.cuda_graphs is False
    for _ in range(3):
        pc(data)
        pc.backward(data, logspace_flows = True)
    assert len(pc._recorded_cuda_graphs) == 0 and replays[0] == 0


def test_the_compile_option_records_a_pass_seen_twice(replays):
    """`cuda_graphs = True` records a pass the SECOND time it runs with the same buffers and options, so a
    one-off shape never pays a capture; every call matches eager, and `record_cudagraph = False` wins."""
    device = torch.device("cuda:0")
    pc, ref = _hmm(device, cuda_graphs = True), _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)

    def step(circuit):
        circuit.init_param_flows(flows_memory = 0.0)
        lls = circuit(data).clone()
        circuit.backward(data, allow_modify_flows = False, logspace_flows = True)
        return lls, circuit.param_flows.clone()

    expected = step(ref)
    counts = []
    for _ in range(3):
        lls, pflows = step(pc)
        assert torch.allclose(lls, expected[0], atol = 1e-5)
        assert torch.allclose(pflows, expected[1], rtol = 1e-4, atol = 1e-6)
        counts.append((len(pc._recorded_cuda_graphs), replays[0]))
    assert counts == [(0, 0), (2, 0), (2, 2)], f"(graphs, replays) after each step: {counts}"

    other = torch.randint(0, 10, [5, 8], device = device)
    for _ in range(3):
        pc(other, record_cudagraph = False)
    assert len(pc._recorded_cuda_graphs) == 2, "`record_cudagraph = False` recorded"


def test_a_pass_that_cannot_be_captured_runs_eagerly(monkeypatch):
    """A host sync inside the inner layers cannot be captured. Recording then warns once, keeps the call's
    (eager) result, does not try again on later calls, and leaves CUDA usable."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)
    expected = pc(data).clone()

    from pyjuice.layer import ProdLayer
    forward = ProdLayer.forward

    def syncing(self, *args, **kwargs):
        torch.cuda.synchronize()                       # not allowed while the stream is captured
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(ProdLayer, "forward", syncing)
    begins = [0]
    begin = torch.cuda.CUDAGraph.capture_begin

    def counting(self, *args, **kwargs):
        begins[0] += 1
        return begin(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", counting)

    with pytest.warns(RuntimeWarning, match = "could not capture"):
        lls = pc(data, record_cudagraph = True)
    assert torch.allclose(lls, expected, atol = 1e-5)
    for _ in range(3):
        assert torch.allclose(pc(data, record_cudagraph = True), expected, atol = 1e-5)
    assert begins[0] == 1, f"{begins[0]} captures attempted; a failed one must not be retried"
    assert len(pc._recorded_cuda_graphs) == 0


def test_inside_a_callers_capture_the_inner_layers_run_eagerly():
    """A caller capturing their own graph around `pc(x)` gets the inner layers as part of THEIR graph; a
    nested capture used to fail ("capturing stream has unjoined work")."""
    device = torch.device("cuda:0")
    pc = _hmm(device, cuda_graphs = True)
    x = torch.randint(0, 10, [6, 8], device = device)
    for _ in range(3):
        pc(x)                                          # compiled, tuned, and our own graph recorded

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            lls = pc(x)
    torch.cuda.current_stream().wait_stream(stream)

    x.copy_(torch.randint(0, 10, [6, 8], device = device))
    graph.replay()
    assert torch.allclose(lls, pc(x, apply_cudagraph = False), atol = 1e-5)


def test_another_threads_cuda_calls_do_not_break_a_capture():
    """Recording captures in "thread_local" mode. In the default "global" mode another thread's CUDA call
    -- a DataLoader pinning host memory -- failed both that thread and the capture."""
    import threading, time
    device = torch.device("cuda:0")
    pc = _hmm(device)
    stop, errors, kept = [False], [], []

    def pin():
        # Kept, so each is a real `cudaHostAlloc` rather than a block from PyTorch's pinned cache; capped
        # (48 MB at most), and paced, as a pinning thread would be. MEASURED with exactly this thread: in
        # "global" mode it failed after ~300 allocations, in "thread_local" it ran all 3000.
        while not stop[0] and len(kept) < 3000:
            try:
                kept.append(torch.empty(4096, pin_memory = True))
                time.sleep(0.0002)
            except Exception as e:
                errors.append(e)
                return

    thread = threading.Thread(target = pin, daemon = True)
    thread.start()
    try:
        for batch in range(2, 14):
            data = torch.randint(0, 10, [batch, 8], device = device)
            pc(data, record_cudagraph = True)
            pc.backward(data, logspace_flows = True, record_cudagraph = True)
    finally:
        stop[0] = True
        thread.join()
    assert not errors, f"the pinning thread failed: {errors[0]!r}"
    assert len(kept) > 0


def test_a_parameterization_must_declare_itself_graph_safe():
    """A replay runs none of an external parameterization's Python, so one that does not set
    `cuda_graph_safe = True` keeps its circuit eager, with a warning when graphs were asked for."""
    from pyjuice.nodes import BlockScaleSumParams

    class Undeclared(BlockScaleSumParams):
        cuda_graph_safe = False

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 6)) for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = 2, external_params = Undeclared(ch_block_size = 16))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(device)
    data, ext = _gated_inputs(ns, 16, device, 1)

    expected = pc(data, sum_external_params = ext).clone()
    with pytest.warns(RuntimeWarning, match = "cuda_graph_safe"):
        for _ in range(2):
            lls = pc(data, sum_external_params = ext, record_cudagraph = True)
            assert torch.allclose(lls, expected, atol = 1e-5)
    assert len(pc._recorded_cuda_graphs) == 0


def test_a_node_axis_gate_is_captured():
    """BlockScale with several gates per node block reads a per-row stride table, whose cache key used to
    read two values back from the device on every call -- a host sync, so these passes could never be
    captured. A recorded forward and backward must replay to what eager computes."""
    from pyjuice.nodes import BlockScaleSumParams
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    with juice.set_block_size(256):
        ni = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = 6)) for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(block_size = 128, ch_block_size = 8))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(device)
    data, ext = _gated_inputs(ns, 32, device, 1)

    def step(graphs):
        pc.init_param_flows(flows_memory = 0.0)
        lls = pc(data, sum_external_params = ext, record_cudagraph = graphs, apply_cudagraph = graphs).clone()
        pc.backward(data, allow_modify_flows = False, compute_external_grads = False,
                    record_cudagraph = graphs, apply_cudagraph = graphs)
        return lls, pc.param_flows.clone()

    expected = step(False)
    step(True)                                         # records
    assert len(pc._recorded_cuda_graphs) == 2, "the gated passes were not captured"
    pc.node_mars.fill_(0.0)
    lls, pflows = step(True)                           # replays
    assert torch.allclose(lls, expected[0], atol = 1e-5)
    assert torch.allclose(pflows, expected[1], rtol = 1e-4, atol = 1e-6)
