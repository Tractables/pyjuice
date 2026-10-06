import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def _hmm(device, seed = 0):
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(seq_length = 8, num_latents = 32, homogeneous = True,
                                         input_dist = dists.Categorical(num_cats = 10))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns).to(device)


def _step(pc, data, graphs):
    pc.init_param_flows(flows_memory = 0.0)
    lls = pc(data, record_cudagraph = graphs, apply_cudagraph = graphs).clone()
    pc.backward(data, allow_modify_flows = False, logspace_flows = True,
                record_cudagraph = graphs, apply_cudagraph = graphs)
    return lls, pc.param_flows.clone()


def test_alternating_batch_sizes_keep_one_graph_per_size():
    device = torch.device("cuda:0")
    pc = _hmm(device)
    batches = [12, 5, 9, 5, 12, 9, 5, 9, 12, 5]           # the largest first: no storage grows after it
    data = {B: torch.randint(0, 10, [B, 8], device = device) for B in set(batches)}

    reference = {B: _step(pc, data[B], graphs = False) for B in set(batches)}

    pc._drop_cuda_graphs()
    addresses = set()
    for B in batches:
        lls, pflows = _step(pc, data[B], graphs = True)
        addresses.add((pc.node_mars.data_ptr(), pc.element_mars.data_ptr(),
                       pc.node_flows.data_ptr(), pc.element_flows.data_ptr()))
        ref_lls, ref_pflows = reference[B]
        assert torch.allclose(lls, ref_lls, atol = 1e-5), f"graphed LL differs from eager at batch {B}"
        assert torch.allclose(pflows, ref_pflows, rtol = 1e-4, atol = 1e-6), \
            f"graphed param flows differ from eager at batch {B}"

    assert len(addresses) == 1, "a buffer moved although no batch exceeded the first"
    assert len(pc._recorded_cuda_graphs) == 2 * len(set(batches)), \
        f"{len(pc._recorded_cuda_graphs)} graphs for {len(set(batches))} batch sizes (forward + backward each)"


def test_a_larger_batch_moves_the_buffers_and_drops_stale_graphs():
    """Past the largest batch so far the storage must grow, which moves every view of it: the graphs
    recorded against the old storage are dropped (each holds a memory pool), and results stay right."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    small, large = torch.randint(0, 10, [4, 8], device = device), torch.randint(0, 10, [20, 8], device = device)

    ref_small = _step(pc, small, graphs = False)
    ref_large = _step(pc, large, graphs = False)

    pc2 = _hmm(device)
    _step(pc2, small, graphs = True)
    assert len(pc2._recorded_cuda_graphs) == 2
    before = pc2.node_mars.data_ptr()

    lls, pflows = _step(pc2, large, graphs = True)
    assert pc2.node_mars.data_ptr() != before
    # at most the forward and backward graphs of the new storage (a freed block may legitimately come
    # back as another buffer's storage, so this counts graphs rather than matching addresses)
    assert len(pc2._recorded_cuda_graphs) <= 2, "graphs recorded against the outgrown storage were kept"
    assert torch.allclose(lls, ref_large[0], atol = 1e-5)
    assert torch.allclose(pflows, ref_large[1], rtol = 1e-4, atol = 1e-6)

    lls, pflows = _step(pc2, small, graphs = True)
    assert torch.allclose(lls, ref_small[0], atol = 1e-5)
    assert torch.allclose(pflows, ref_small[1], rtol = 1e-4, atol = 1e-6)


# ---- a recorded graph must compute exactly what the eager pass would ----

@pytest.mark.parametrize("logspace_flows", [False, True])
def test_the_step_that_records_a_graph_matches_eager(logspace_flows):
    """Recording runs the backward three times to warm up, for real. Those runs used to accumulate into
    the parameter flows and to reset `node_flows` to 0.0 -- an all-ones flow in log space -- so the step
    that recorded a graph came out with ~4x its parameter flows, and far more in log space."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)

    out = {}
    for graphs in (False, True):
        pc._recorded_cuda_graphs.clear()
        pc.init_param_flows(flows_memory = 0.0)
        pc(data, record_cudagraph = graphs, apply_cudagraph = graphs)
        pc.backward(data, allow_modify_flows = False, logspace_flows = logspace_flows,
                    record_cudagraph = graphs, apply_cudagraph = graphs)
        out[graphs] = pc.param_flows.clone()
    assert torch.allclose(out[True], out[False], rtol = 1e-4, atol = 1e-6), \
        f"recording step: max abs diff {float((out[True] - out[False]).abs().max()):.3e}"


def test_a_recorded_graph_is_not_replayed_under_other_options():
    """A recorded graph is replayed whenever its signature matches (`apply_cudagraph` is on by default),
    so every option that changes the captured work has to be in it. Forward: an MPE pass after a recorded
    LL pass used to replay LL. Backward: `compute_param_flows = False` (what the queries use) after a
    recorded training backward used to replay a graph that writes parameter flows."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)

    mpe_eager = pc(data, propagation_alg = "MPE", apply_cudagraph = False).clone()
    pc(data, record_cudagraph = True)                                     # records the LL forward
    assert torch.allclose(pc(data, propagation_alg = "MPE"), mpe_eager, atol = 1e-5), \
        "an MPE forward replayed the recorded LL graph"

    pc.init_param_flows(flows_memory = 0.0)
    pc(data)
    pc.backward(data, logspace_flows = True, record_cudagraph = True)      # records the training backward
    before = pc.param_flows.clone()
    pc.backward(data, logspace_flows = True, compute_param_flows = False)
    assert torch.equal(pc.param_flows, before), "a backward without parameter flows wrote them"


def test_backward_callbacks_run_on_every_call():
    """Callbacks run inside the captured region and could do anything, so a backward given them runs
    eagerly -- it must neither record a graph nor replay one."""
    device = torch.device("cuda:0")
    pc = _hmm(device)
    data = torch.randint(0, 10, [6, 8], device = device)
    calls = []

    pc(data)
    pc.backward(data, logspace_flows = True, record_cudagraph = True)      # a graph for these buffers exists
    for _ in range(2):
        pc(data)
        pc.backward(data, logspace_flows = True, record_cudagraph = True,
                    sum_layer_pre_backward_callback = lambda layer, **kw: calls.append(layer))
    num_sum_layers = sum(1 for g in pc.inner_layer_groups if g.is_sum() for _ in g)
    assert len(calls) == 2 * num_sum_layers, f"{len(calls)} callback calls, expected {2 * num_sum_layers}"
