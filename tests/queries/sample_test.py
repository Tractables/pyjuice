import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def simple_sample_test():

    device = torch.device("cuda:0")

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 3))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 4))

    ms = multiply(ni0, ni1)
    ns = summate(ms, num_nodes = 1)

    ni0.set_params(torch.tensor([[0.99, 0.005, 0.005], [0.02, 0.49, 0.49]]).reshape(-1))
    ni1.set_params(torch.tensor([[0.0, 0.5, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]]).reshape(-1))

    ns.set_params(torch.tensor([0.5, 0.5]))

    pc = TensorCircuit(ns)
    pc.to(device)

    data = torch.zeros([1000, 2], dtype = torch.long, device = device)
    data[500:,0] = 1
    missing_mask = torch.tensor([False, True], device = device)

    samples = juice.queries.sample(pc, data, missing_mask)
    
    assert (samples[:500,1] == 1).sum() > 200
    assert (samples[:500,1] == 2).sum() > 200
    assert (samples[500:,1] == 0).sum() > 475


def hmm_sample_test():

    device = torch.device("cuda:0")

    # Construct HMM

    num_vars = 32
    num_latents = 3
    num_cats = num_latents * 2

    # Emission probabilities
    beta = torch.zeros([num_latents, num_cats])
    for i in range(num_latents):
        beta[i,i*2:(i+1)*2] = 0.5

    # Transition probabilities
    alpha = torch.zeros([num_latents, num_latents])
    alpha[:,:] = 0.2 / (num_latents - 1)
    alpha.fill_diagonal_(0.8)

    # Init probabilities
    gamma = torch.zeros([num_latents])
    gamma[:] = 1.0 / num_latents

    ns_input = juice.inputs(num_vars - 1, num_latents, dists.Categorical(num_cats = num_cats), params = beta.reshape(-1))
    ns_sum = None

    curr_zs = ns_input
    for var in range(num_vars - 2, -1, -1):
        curr_xs = ns_input.duplicate(var, tie_params = True)

        if ns_sum is None:
            ns = juice.summate(curr_zs, num_nodes = num_latents, params = alpha.reshape(-1))
            ns_sum = ns
        else:
            ns = ns_sum.duplicate(curr_zs, tie_params = True)

        curr_zs = juice.multiply(curr_xs, ns)

    ns = juice.summate(curr_zs, num_nodes = 1, params = gamma)

    pc = TensorCircuit(ns)
    pc.to(device)

    ## Unconditional sample ##

    data = torch.zeros([400, num_vars], dtype = torch.long, device = device)
    missing_mask = torch.ones([num_vars], dtype = torch.bool, device = device)

    samples = juice.queries.sample(pc, data, missing_mask)

    vals = samples.reshape(-1)
    for i in range(num_latents):
        crit = (vals == 2 * i) | (vals == 2 * i + 1)
        joint_count = (crit[:-1] & crit[1:]).sum()
        single_count = crit.sum()

        assert single_count > 0.9 * 400 * num_vars / num_latents
        assert joint_count > 0.7 * single_count

    ## Conditional sample ##

    data = torch.zeros([400, num_vars], dtype = torch.long, device = device)
    missing_mask = torch.ones([num_vars], dtype = torch.bool, device = device)
    data[:,4] = 2
    missing_mask[4] = False

    samples = juice.queries.sample(pc, data, missing_mask)

    assert torch.all(samples[:,4] == 2)
    assert ((samples[:,3] == 2) | (samples[:,3] == 3)).sum() > 0.7 * 400
    assert ((samples[:,5] == 2) | (samples[:,5] == 3)).sum() > 0.7 * 400

    assert ((samples[:,2] == 2) | (samples[:,2] == 3)).sum() > 0.55 * 400
    assert ((samples[:,6] == 2) | (samples[:,6] == 3)).sum() > 0.55 * 400

    data = torch.zeros([400, num_vars], dtype = torch.long, device = device)
    missing_mask = torch.ones([num_vars], dtype = torch.bool, device = device)
    data[:,4] = 2
    data[:,6] = 2
    missing_mask[4] = False
    missing_mask[6] = False

    samples = juice.queries.sample(pc, data, missing_mask)

    assert ((samples[:,5] == 2) | (samples[:,5] == 3)).sum() > 0.9 * 400


if __name__ == "__main__":
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

    simple_sample_test()
    hmm_sample_test()