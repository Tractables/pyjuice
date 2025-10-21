import pyjuice as juice
import torch
import numpy as np
from pyjuice.nodes.distributions import Categorical, External
from pyjuice.utils.util import max_cdf_power_of_2

import pytest


def hmm(seq_length: int, num_latents: int, num_emits: int, use_obs_soft_evi: bool, use_latent_soft_evi: bool, homogeneous: bool = True):
    
    block_size = min(max_cdf_power_of_2(num_latents), 1024)
    num_node_blocks = num_latents // block_size

    if use_obs_soft_evi and use_latent_soft_evi:
        obs_se_sid = seq_length
        latent_se_sid = seq_length * 2
    elif use_obs_soft_evi:
        obs_se_sid = seq_length
    elif use_latent_soft_evi:
        latent_se_sid = seq_length

    def apply_soft_evi_nodes(ns, v):
        if use_obs_soft_evi or use_latent_soft_evi:
            prod_chs = [ns]

            prod_edge_ids = [torch.arange(0, num_node_blocks)]

            if use_obs_soft_evi:
                prod_chs.append(
                    juice.inputs(obs_se_sid + v, num_node_blocks = 1, block_size = block_size, dist = External())
                )
                prod_edge_ids.append(torch.zeros([num_node_blocks], dtype = torch.long))

            if use_latent_soft_evi:
                prod_chs.append(
                    juice.inputs(latent_se_sid + v, num_node_blocks = num_node_blocks, block_size = block_size, dist = External())
                )
                prod_edge_ids.append(torch.arange(0, num_node_blocks))

            ns = juice.multiply(*prod_chs, edge_ids = torch.stack(prod_edge_ids, dim = 1))
        
        return ns
    
    # Create HMM
    ns_input = juice.inputs(
        seq_length - 1, num_node_blocks = num_node_blocks,
        block_size = block_size,
        dist = Categorical(num_cats = num_emits)
    )
    
    ns_sum = None
    curr_zs = apply_soft_evi_nodes(ns_input, seq_length - 1)
    for var in range(seq_length - 2, -1, -1):
        curr_xs = ns_input.duplicate(var, tie_params = homogeneous)
        curr_xs = apply_soft_evi_nodes(curr_xs, var)
        
        if ns_sum is None:
            ns = juice.summate(curr_zs, num_node_blocks = num_node_blocks, block_size = block_size)
            ns_sum = ns
        else:
            ns = ns_sum.duplicate(curr_zs, tie_params = homogeneous)

        if curr_xs.is_prod():
            curr_zs = juice.multiply(ns, *curr_xs.chs)
        else:
            curr_zs = juice.multiply(curr_xs, ns)
        
    ns = juice.summate(curr_zs, num_node_blocks = 1, block_size = 1)

    return ns


def test_soft_evi_hmm_type1():

    seq_length = 32
    num_latents = 512
    vocab_size = 50257

    device = torch.device("cuda:0")
    
    # Create an HMM with virtual evidence applied to observed variables
    root_ns = hmm(
        seq_length = seq_length,
        num_latents = num_latents,
        num_emits = vocab_size,
        use_obs_soft_evi = True,
        use_latent_soft_evi = False,
        homogeneous = True
    )

    pc = juice.compile(root_ns)
    pc.to(device)

    print(f"# variables: {pc.num_vars}") # In this case the number of variables should be 2 * seq_length

    data = torch.randint(0, vocab_size, [16, seq_length * 2]).to(device)
    external_soft_evi = torch.rand([16, seq_length])[:,:,None].repeat(1, 1, root_ns.chs[0].block_size).to(device)

    external_soft_evi_grad = torch.zeros_like(external_soft_evi)

    # Forward and backward pass
    lls = pc(data, external_soft_evi = external_soft_evi)
    pc.backward(data, external_soft_evi_grad = external_soft_evi_grad)

    # Runtests
    for ns in root_ns:
        if ns.is_input() and isinstance(ns.dist, External):
            v = ns.scope.to_list()[0]
            sid, eid = ns._output_ind_range

            assert torch.all(torch.abs(external_soft_evi[:,v-seq_length,0][None,:] - pc.node_mars[sid:eid,:]) < 1e-5)
            assert torch.all(torch.abs(external_soft_evi_grad[:,v-seq_length,:].permute(1, 0) - pc.node_flows[sid:eid,:]) < 1e-5)

    external_soft_evi_grad = external_soft_evi_grad.exp()


def test_soft_evi_hmm_type2():

    seq_length = 32
    num_latents = 512
    vocab_size = 50257

    device = torch.device("cuda:0")
    
    # Create an HMM with virtual evidence applied to observed variables
    root_ns = hmm(
        seq_length = seq_length,
        num_latents = num_latents,
        num_emits = vocab_size,
        use_obs_soft_evi = True,
        use_latent_soft_evi = True,
        homogeneous = True
    )

    pc = juice.compile(root_ns)
    pc.to(device)

    print(f"# variables: {pc.num_vars}") # In this case the number of variables should be 2 * seq_length

    data = torch.randint(0, vocab_size, [16, seq_length * 3]).to(device)
    external_soft_evi = torch.zeros([16, seq_length * 2, num_latents]).to(device)

    external_soft_evi[:,:seq_length,:root_ns.chs[0].block_size] = torch.rand([16, seq_length])[:,:,None].to(device)
    external_soft_evi[:,seq_length:,:] = torch.rand([16, seq_length, num_latents]).to(device)

    external_soft_evi_grad = torch.zeros_like(external_soft_evi)

    # Forward and backward pass
    lls = pc(data, external_soft_evi = external_soft_evi)
    pc.backward(data, external_soft_evi_grad = external_soft_evi_grad)

    # Runtests
    for ns in root_ns:
        if ns.is_input() and isinstance(ns.dist, External):
            v = ns.scope.to_list()[0]
            sid, eid = ns._output_ind_range

            if v < 2 * seq_length:
                assert torch.all(torch.abs(external_soft_evi[:,v-seq_length,0][None,:] - pc.node_mars[sid:eid,:]) < 1e-5)
                assert torch.all(torch.abs(external_soft_evi_grad[:,v-seq_length,:].permute(1, 0) - pc.node_flows[sid:eid,:]) < 1e-5)
            elif v >= 2 * seq_length:
                assert torch.all(torch.abs(external_soft_evi[:,v-seq_length,:].permute(1, 0) - pc.node_mars[sid:eid,:]) < 1e-5)
                assert torch.all(torch.abs(external_soft_evi_grad[:,v-seq_length,:].permute(1, 0) - pc.node_mars[sid:eid,:]) < 1e-5)


if __name__ == "__main__":
    test_soft_evi_hmm_type1()
    test_soft_evi_hmm_type2()