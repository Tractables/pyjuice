"""
Minimal end-to-end example: an HMM whose transitions take per-sample gates, fed as ONE tensor.

This is the shortest complete path through the feature, and is meant to be read as much as run:

    1. build an HMM, giving every transition a `BlockScaleSumParams`;
    2. group the transitions under a name, so they share one tensor;
    3. one forward with the gates, one backward.

Run it directly (`python blockscale_hmm_example_test.py`) or under pytest.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams


NUM_STATES = 128        # hidden states; also the block size, so the HMM is one node block wide
NUM_EMITS = 4           # observation alphabet
SEQ_LEN = 6             # timesteps
GATE_CH_BLOCK = 8       # a gate spans 8 children -> NUM_STATES / 8 = 16 gates per transition
BATCH = 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_gated_hmm_forward_and_backward():
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # ---- 1. the HMM. Every transition is a sum layer with a per-sample gate on its parameters.
    with juice.set_block_size(NUM_STATES):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))

        transitions = []
        for t in range(1, SEQ_LEN):
            emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))
            ns = summate(
                multiply(ns, emit), num_node_blocks = 1,
                external_params = BlockScaleSumParams(ch_block_size = GATE_CH_BLOCK),
            )
            transitions.append(ns)

        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root).to(device)

    # ---- 2. one name for all the transitions, so they take a single concatenated tensor.
    # Without this, `sum_external_params` would need one entry per timestep.
    pc.register_external_params_group("transitions", transitions)

    # Each transition's own gate tensor is [BATCH, 1, 16]; the group's is the concatenation along
    # dim 1 (the default), so [BATCH, SEQ_LEN - 1, 16] -- one gate per (sample, timestep, gate block).
    # In a real model this comes straight out of whatever head produces it; here it is random.
    num_gates = NUM_STATES // GATE_CH_BLOCK
    log_phi = torch.randn([BATCH, SEQ_LEN - 1, num_gates], device = device)

    data = torch.randint(0, NUM_EMITS, [BATCH, SEQ_LEN], device = device)

    # ---- 3. forward and backward.
    lls = pc(data, sum_external_params = {"transitions": log_phi})
    pc.backward(data, flows_memory = 0.0)

    assert lls.shape == (BATCH, 1)
    assert torch.isfinite(lls).all()
    assert float(pc.param_flows.abs().sum()) > 0.0, "the backward wrote no parameter flows"

    # The gates matter: a different tensor gives a different answer. (All-zero log-gates would be
    # phi = 1, i.e. the plain HMM, so use something else to see an actual effect.)
    other = pc(data, sum_external_params = {"transitions": log_phi * 2.0})
    assert not torch.allclose(lls, other)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_training_a_router_that_produces_the_gates():
    """The end-to-end loop the gate exists for: a head emits the gates, and its own weights learn.

    `pc.backward` computes `d LL / d log phi` by default (`compute_external_grads = True`), and
    `get_external_params_grad` hands it back laid out exactly like the tensor that was supplied -- so
    the gates go into the circuit detached, and the gradient comes back out into ordinary autograd."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    with juice.set_block_size(NUM_STATES):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))
        transitions = []
        for t in range(1, SEQ_LEN):
            emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))
            ns = summate(multiply(ns, emit), num_node_blocks = 1,
                         external_params = BlockScaleSumParams(ch_block_size = GATE_CH_BLOCK))
            transitions.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root).to(device)
    pc.register_external_params_group("transitions", transitions)

    num_gates = NUM_STATES // GATE_CH_BLOCK
    features = torch.randn([BATCH, 16], device = device)
    router = torch.nn.Linear(16, (SEQ_LEN - 1) * num_gates).to(device)
    opt = torch.optim.Adam(router.parameters(), lr = 0.05)

    data = torch.randint(0, NUM_EMITS, [BATCH, SEQ_LEN], device = device)

    before = None
    for step in range(15):
        log_phi = router(features).view(BATCH, SEQ_LEN - 1, num_gates)

        lls = pc(data, sum_external_params = {"transitions": log_phi.detach()})
        pc.backward(data, flows_memory = 0.0)

        # d LL / d log phi, in the layout the gates were supplied in
        g = pc.get_external_params_grad("transitions")[0]
        assert g.shape == log_phi.shape

        opt.zero_grad(set_to_none = True)
        torch.autograd.backward([log_phi], [-g])      # ascend the likelihood
        opt.step()

        if before is None:
            before = float(lls.mean())

    after = float(pc(data, sum_external_params = {
        "transitions": router(features).view(BATCH, SEQ_LEN - 1, num_gates).detach()}).mean())

    assert after > before, f"training the router did not improve the likelihood: {before} -> {after}"


if __name__ == "__main__":
    test_gated_hmm_forward_and_backward()
    test_training_a_router_that_produces_the_gates()
    print("ok")
