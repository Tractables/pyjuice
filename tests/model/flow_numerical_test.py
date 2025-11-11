import torch

import pyjuice as juice

import pytest


def test_hmm_flow_stability():

    device = torch.device("cuda:0")

    seq_length = 16
    num_latents = 1024
    vocab_size = 1023
    batch_size = 32

    for num_latents in [256, 512, 1024, 2048]:

        root_ns = juice.structures.HMM(
            seq_length = seq_length,
            num_latents = num_latents,
            num_emits = vocab_size
        )
        root_ns.init_parameters(perturbation = 32.0)

        pc = juice.compile(root_ns)
        pc.to(device)

        data = torch.randint(0, vocab_size, [batch_size, seq_length], device = device)

        lls = pc(data)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True)

        pc.update_parameters()
        pc.update_param_flows()

        alpha = root_ns.chs[0].chs[1].get_source_ns().get_params()
        block_size = alpha.size(1)
        num_node_blocks = num_latents // block_size
        alpha = alpha.reshape(num_node_blocks, num_node_blocks, block_size, block_size).permute(
            0, 2, 1, 3
        ).reshape(num_latents, num_latents).to(device)

        alpha_flow = root_ns.chs[0].chs[1].get_source_ns().get_param_flows().reshape(
            num_node_blocks, num_node_blocks, block_size, block_size).permute(
            0, 2, 1, 3
        ).reshape(num_latents, num_latents).to(device)

        gamma = root_ns.get_params().reshape(num_latents).to(device)

        # Root node
        pc_nflows = pc.get_node_flows(root_ns)
        assert torch.all(pc_nflows.abs() < 1e-4)

        # First product layer
        pc_cflows = pc.get_node_flows(root_ns.chs[0])
        pc_cflows_chk = pc.get_node_flows(root_ns.chs[0].chs[0])
        assert torch.all(pc_cflows.logsumexp(dim = 0) < 1e-4)
        assert torch.all((pc_cflows - pc_cflows_chk).abs() < 1e-4)

        nmars = pc.get_node_mars(root_ns)
        cmars = pc.get_node_mars(root_ns.chs[0])
        cflows = gamma[:,None] * (cmars - nmars).exp()
        assert torch.all((pc_cflows - cflows.log()).abs() < 1e-4)

        # The following product layers
        nflows = cflows.log()
        ns = root_ns.chs[0].chs[1]
        layer_id = len(pc.inner_layer_groups) - 3
        for _ in range(seq_length - 2):
            cs = ns.chs[0]

            pc_cflows = pc.get_node_flows(cs)
            pc_cflows_chk = pc.get_node_flows(cs.chs[0])
            assert torch.all((pc_cflows - pc_cflows_chk).abs() < 1e-4)

            nmars = pc.get_node_mars(ns)
            cmars = pc.get_node_mars(cs)

            xmatr = alpha[:,:,None] * (cmars[None,:,:] - nmars[:,None,:]).exp()
            cflows = (nflows[:,None,:] + xmatr.log()).logsumexp(dim = 0)

            # Test node flows
            assert torch.all(((pc_cflows.exp() - cflows.exp()).abs() / cflows.exp()) < 1e-2)
            assert torch.all((pc_cflows.exp() - cflows.exp()).abs().mean() < 1e-4)
            
            pc.zero_param_flows()

            prod_layer = pc.inner_layer_groups[layer_id-1][0]
            prod_layer.forward(pc.node_mars, pc.element_mars, _for_backward = True)
            layer = pc.inner_layer_groups[layer_id][0]
            layer.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars,
                        pc.params, pc.param_flows, logspace_flows = True)

            pc.update_param_flows()

            local_alpha_flow = ns.get_source_ns().get_param_flows().reshape(
                num_node_blocks, num_node_blocks, block_size, block_size).permute(
                0, 2, 1, 3
            ).reshape(num_latents, num_latents).to(device)

            xmatr = alpha[:,:,None] * (cmars[None,:,:] - nmars[:,None,:]).exp()
            pflows = (nflows[:,None,:] + xmatr.log()).logsumexp(dim = 2)

            # Test parameter flows
            torch.all((local_alpha_flow - pflows.exp()).abs() / batch_size < 4e-5)
            torch.all(((local_alpha_flow.sum(dim = 1) - batch_size) / batch_size).abs() < 1e-6)

            ns = ns.chs[0].chs[1]
            nflows = cflows
            layer_id -= 2


if __name__ == "__main__":
    test_hmm_flow_stability()
