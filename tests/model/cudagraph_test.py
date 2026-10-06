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
