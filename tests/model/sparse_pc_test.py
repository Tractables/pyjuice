import pyjuice as juice
import torch
import numpy as np
import time

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def sparse_pc_test():
    num_latents = 4096
    num_cats = 200
    num_vars = 8

    sparsity = 0.05

    counts = torch.rand([num_latents, num_latents])
    counts[:2,:] = 0.0 # manually create nchs imbalanceness
    indices = torch.where(counts < sparsity)
    latent_edge_ids = torch.stack(indices, dim = 0)

    ns_input = juice.inputs(num_vars - 1, num_latents, dists.Categorical(num_cats = num_cats))
    ns_sum = None

    curr_zs = ns_input
    for var in range(num_vars - 2, -1, -1):
        curr_xs = ns_input.duplicate(var, tie_params = True)

        if ns_sum is None:
            ns = juice.summate(curr_zs, num_nodes = num_latents, edge_ids = latent_edge_ids)
            ns_sum = ns
        else:
            ns = ns_sum.duplicate(curr_zs, tie_params = True)

        curr_zs = juice.multiply(curr_xs, ns)

    ns = juice.summate(curr_zs, num_nodes = 1)

    t0 = time.time()
    pc = juice.TensorCircuit(ns, layer_sparsity_tol = 0.8)
    t1 = time.time()
    
    # This takes ~17s on a RTX 4090 GPU
    assert t1 - t0 < 40


if __name__ == "__main__":
    sparse_pc_test()