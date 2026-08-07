"""
A worked example: sampling from an HMM whose transitions carry per-sample external parameters.

This file is meant to be READ. It is the shortest complete path through the feature -- build a gated
circuit, make a gate tensor, draw unconditionally, draw conditionally on evidence -- with the shape
rules spelled out where they are easy to get wrong. The thorough tests live in `queries/sample/`;
nothing here tries to be exhaustive.

**What a `BlockScaleSumParams` node does.** An ordinary sum node weighs its edges by one parameter
vector shared across the batch. A gated one multiplies each edge by a per-sample factor:

    theta'[b, n, c]  proportional to  phi[b, g(n, c)] * theta[n, c]

so every sample in a batch can steer the same circuit differently -- which is the point when the
gates are produced per sample by another model. The normalizer cancels out of an ancestral draw, so
sampling needs nothing from a forward pass unless you are conditioning on evidence.

**The gate tensor's shape** is `[batch_size, num_node_gates, num_ch_gates]`, where one gate covers a
BLOCK of parameters rather than a single edge -- that is what makes it cheap. The two counts are
therefore the node and child counts divided by the gate's block sizes; ask the node for them rather
than deriving them by hand (see `_gate_shape` below), since `ch_block_size = None` means "the node's
own", not 1.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

SEQ_LEN = 4         # timesteps, one observed variable each
NUM_EMITS = 5       # observation alphabet
BLOCK_SIZE = 4      # hidden states per block
NUM_BLOCKS = 2      # -> 8 hidden states, and a 2 x 2 gate grid per transition
NUM_SAMPLES = 4096


def _build_gated_hmm():
    """A chain of `SEQ_LEN` steps whose transitions each take their own external parameters."""
    with juice.set_block_size(BLOCK_SIZE):
        emissions = [inputs(0, num_node_blocks = NUM_BLOCKS,
                            dist = dists.Categorical(num_cats = NUM_EMITS))]
        transitions, chain = [], emissions[0]

        for t in range(1, SEQ_LEN):
            emit = inputs(t, num_node_blocks = NUM_BLOCKS,
                          dist = dists.Categorical(num_cats = NUM_EMITS))
            emissions.append(emit)

            # `external_params` is all that distinguishes this from a plain `summate`
            chain = summate(multiply(chain, emit), num_node_blocks = NUM_BLOCKS,
                            external_params = BlockScaleSumParams(ch_block_size = BLOCK_SIZE))
            transitions.append(chain)

        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 4.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0")), transitions


def _gate_shape(ns):
    """The `[num_node_gates, num_ch_gates]` this node expects, asked of the node itself."""
    node_gate_size, ch_gate_size = ns.external_params.gate_sizes(ns)
    return ns.num_nodes // node_gate_size, ns.num_ch_nodes // ch_gate_size


def _gates_favouring(transitions, block, batch_size, device, strength = 6.0):
    """One gate tensor per transition, each pushing the draw towards child-gate column `block`.

    Gates are LOG factors, so 0.0 is "no effect" and a positive entry favours that block.
    """
    gates = {}
    for ns in transitions:
        num_node_gates, num_ch_gates = _gate_shape(ns)
        g = torch.zeros([batch_size, num_node_gates, num_ch_gates], device = device)
        g[:, :, block] = strength
        gates[ns] = g
    return gates


@cuda_only
def test_unconditional_sampling_from_a_gated_hmm():
    """Draw `NUM_SAMPLES` sequences, with the gates supplied exactly as they would be to `pc(...)`."""
    torch.manual_seed(0)
    pc, transitions = _build_gated_hmm()

    gates = _gates_favouring(transitions, block = 0, batch_size = NUM_SAMPLES, device = pc.device)
    samples = juice.queries.sample(pc, num_samples = NUM_SAMPLES, sum_external_params = gates)

    assert samples.shape == (NUM_SAMPLES, SEQ_LEN)
    assert bool(((samples >= 0) & (samples < NUM_EMITS)).all())


@cuda_only
def test_the_gate_actually_steers_the_draw():
    """
    The gate is not decoration: pointing it at a different block of hidden states changes what comes
    out. Compared as per-variable means, in units of the standard error, so the check does not depend
    on any particular emission being sharp.
    """
    torch.manual_seed(0)
    pc, transitions = _build_gated_hmm()

    def draw(block):
        torch.manual_seed(1)                    # same randomness, so only the gate differs
        gates = _gates_favouring(transitions, block, NUM_SAMPLES, pc.device)
        return juice.queries.sample(pc, num_samples = NUM_SAMPLES,
                                    sum_external_params = gates).float()

    first, second = draw(0), draw(1)
    se = ((first.var(dim = 0) + second.var(dim = 0)) / NUM_SAMPLES).sqrt().clamp(min = 1e-9)
    z = float(((first.mean(dim = 0) - second.mean(dim = 0)) / se).abs().max())

    assert z > 5.0, f"the two gates gave the same distribution: max |z| = {z:.2f}"


@cuda_only
def test_conditional_sampling_from_a_gated_hmm():
    """
    Conditioning takes two calls: a forward pass over the evidence, then the draw.

    The gates go to the FORWARD pass, not to the draw. A conditional draw runs against the state
    that pass left behind, which was built under those gates, so it takes them from there.

    Worth knowing: gates handed to `sample(conditional = True)` are accepted and then IGNORED, the
    staged ones winning. MEASURED -- passing a different gate here moves the answer by 0.64 sigma,
    i.e. not at all, while a forward pass genuinely run with that gate differs by 60. So supply them
    once, to `pc(...)`, as below.
    """
    torch.manual_seed(0)
    pc, transitions = _build_gated_hmm()

    # observe the even timesteps, leave the odd ones missing
    evidence = torch.randint(0, NUM_EMITS, [NUM_SAMPLES, SEQ_LEN], device = pc.device)
    missing = torch.zeros([SEQ_LEN], dtype = torch.bool, device = pc.device)
    missing[1::2] = True

    gates = _gates_favouring(transitions, block = 0, batch_size = NUM_SAMPLES, device = pc.device)
    pc(evidence, missing_mask = missing, sum_external_params = gates)

    samples = juice.queries.sample(pc, conditional = True)

    # one sample per row of the evidence -- `num_samples` is not passed, it comes from that batch
    assert samples.shape == (NUM_SAMPLES, SEQ_LEN)
    assert bool(((samples >= 0) & (samples < NUM_EMITS)).all())
