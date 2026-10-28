"""`TensorCircuit.backward(compute_param_flows = False)` with a `missing_mask`: it ran the input layers' missing-flows
pass, which accumulates into `param_flows` -- not allocated then -- and failed compiling its kernel against None. The
flows of marginalized positions are parameter flows only, so without them there is nothing to do: the node flows are
those of a backward with parameter flows, to the bit."""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists

DEV = torch.device("cuda:0")


def _circuits():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (256, 12))
    yield "hmm", juice.structures.HMM(seq_length = 8, num_latents = 16, num_emits = 5), 8
    yield "hclt", juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Categorical(num_cats = 5)), 12
    yield "gaussian", juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Gaussian(mu = 0.0, sigma = 1.0)), 12


@pytest.mark.parametrize("mask", ["per_sample", "per_variable"])
def test_a_backward_without_parameter_flows_takes_a_missing_mask(mask):
    for name, ns, V in _circuits():
        ns.init_parameters(perturbation = 2.0)
        pc = juice.compile(ns, verbose = False).to(DEV)
        B = 4
        x = (torch.randint(0, 5, (B, V)) if name != "gaussian" else torch.randn(B, V)).to(DEV)
        miss = torch.rand(B, V, device = DEV) < 0.5 if mask == "per_sample" else torch.rand(V, device = DEV) < 0.5
        pc(x, missing_mask = miss)
        pc.backward(x, missing_mask = miss, compute_param_flows = False)         # raised before the fix (no
        got = pc.node_flows.clone()                                              # `param_flows` allocated yet)
        pc(x, missing_mask = miss)
        pc.backward(x, missing_mask = miss, compute_param_flows = True)
        assert torch.equal(got, pc.node_flows), name
