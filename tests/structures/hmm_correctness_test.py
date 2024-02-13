import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def hmm_forward_backward_test():

    device = torch.device("cuda:0")

    seq_length = 16
    vocab_size = 1023
    batch_size = 32

    num_node_blocks = 4 # 4096 // 32 # 4
    block_size = 1024 # 32 # 1024
    num_latents = block_size * num_node_blocks

    with juice.set_block_size(block_size = block_size):
        ns_input = juice.inputs(seq_length - 1, num_node_blocks = num_node_blocks,
                                dist = dists.Categorical(num_cats = vocab_size))
        
        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = True)
            
            if ns_sum is None:
                ns = juice.summate(
                    curr_zs, num_node_blocks = num_node_blocks)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)

            curr_zs = juice.multiply(curr_xs, ns)
            
        root_ns = juice.summate(curr_zs, num_node_blocks = 1, block_size = 1)
    
    root_ns.init_parameters()

    pc = juice.TensorCircuit(root_ns)
    pc.to(device)

    data = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    data_cpu = data.cpu()

    ## Forward tests ##

    lls = pc(data)

    ns2mars = dict()

    node_mars = pc.node_mars.detach().cpu()

    with torch.no_grad():
        for ns in root_ns:
            if ns.is_input():
                v = ns.scope.to_list()[0]
                params = ns.get_source_ns()._params.reshape(num_latents, vocab_size)

                mars = params[:,data_cpu[:,v]].log()

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(mars - node_mars[sid:eid,:]) < 1e-4)

                ns2mars[ns] = mars

            elif ns.is_prod():
                mars = torch.zeros([num_latents, batch_size])
                for cs in ns.chs:
                    mars += ns2mars[cs]

                ns2mars[ns] = mars

            elif ns.is_sum() and ns != root_ns:
                emars = torch.cat([ns2mars[cs] for cs in ns.chs], dim = 0)
                params = ns.get_source_ns()._params.reshape(num_node_blocks, num_node_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3)
                params = params.reshape(num_latents, num_latents * ns.num_chs)

                emars_max = torch.max(emars, dim = 0).values[None,:]
                emars = (emars - emars_max).exp()

                nmars = torch.matmul(params, emars)
                nmars = nmars.log() + emars_max

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(nmars - node_mars[sid:eid,:]) < 4e-3)

                ns2mars[ns] = nmars

            else:
                assert ns == root_ns

                emars = torch.cat([ns2mars[cs] for cs in ns.chs], dim = 0)
                params = ns._params.reshape(1, num_node_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
                params = params.reshape(1, num_latents * ns.num_chs)

                emars_max = torch.max(emars, dim = 0).values[None,:]
                emars = (emars - emars_max).exp()

                nmars = torch.matmul(params, emars)
                nmars = nmars.log() + emars_max

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(nmars - node_mars[sid:eid,:]) < 4e-3)

    ## Backward tests ##

    pc.backward(data.permute(1, 0), allow_modify_flows = False)

    pc.update_param_flows()

    node_mars = pc.node_mars.cpu()
    node_flows = pc.node_flows.cpu()

    ns2flows = dict()
    ns2flows[root_ns] = torch.ones([1, batch_size])

    cum_pflows = 0.0

    with torch.no_grad():
        for ns in root_ns(reverse = True):
            if ns == root_ns:

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(node_flows[sid:eid,:] - 1.0) < 1e-4)

                nflows = ns2flows[ns]
                nmars = node_mars[sid:eid,:]

                for i, cs in enumerate(ns.chs):
                    params = ns._params.reshape(1, num_node_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
                    params = params[:,:,i*num_node_blocks:(i+1)*num_node_blocks,:].reshape(1, num_latents)

                    param_flows = ns._param_flows.reshape(1, num_node_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
                    param_flows = param_flows[:,:,i*num_node_blocks:(i+1)*num_node_blocks,:].reshape(1, num_latents)

                    if cs.is_prod():
                        emars = torch.zeros([num_latents, batch_size])
                        for cns in cs.chs:
                            sid, eid = cns._output_ind_range
                            emars += node_mars[sid:eid,:]
                    else:
                        raise ValueError()

                    eflows = nflows * params.permute(1, 0) * (emars - nmars).exp()
                    pflows = eflows.sum(dim = 1)

                    assert torch.all(torch.abs(pflows - param_flows[0,:]) < 6e-3)

                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size])
                    ns2flows[cs] += eflows

            elif ns.is_prod():
                nflows = ns2flows[ns]
                for cs in ns.chs:
                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size])
                    ns2flows[cs] += nflows

            elif ns.is_sum():

                nflows = ns2flows[ns]

                sid, eid = ns._output_ind_range

                assert torch.all(torch.abs(nflows - node_flows[sid:eid,:]) < 1e-5)

                nmars = node_mars[sid:eid,:]

                for i, cs in enumerate(ns.chs):
                    params = ns.get_source_ns()._params.reshape(num_node_blocks, num_node_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3)
                    params = params[:,:,i*num_node_blocks:(i+1)*num_node_blocks,:].reshape(num_latents, num_latents)

                    if cs.is_prod():
                        emars = torch.zeros([num_latents, batch_size])
                        for cns in cs.chs:
                            sid, eid = cns._output_ind_range
                            emars += node_mars[sid:eid,:]
                    else:
                        raise ValueError()

                    log_n_fdm = nflows.log() - nmars
                    log_n_fdm_max = log_n_fdm.max(dim = 0).values
                    n_fdm_sub = (log_n_fdm - log_n_fdm_max[None,:]).exp()

                    eflows = torch.matmul(params.permute(1, 0), n_fdm_sub) * (emars + log_n_fdm_max[None,:]).exp()

                    scaled_emars = (emars + log_n_fdm_max[None,:]).exp()
                    pflows = torch.matmul(n_fdm_sub, scaled_emars.permute(1, 0)) * params

                    cum_pflows = cum_pflows + pflows

                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size])
                    ns2flows[cs] += eflows

    pflows = ns_sum._param_flows.reshape(num_node_blocks, num_node_blocks, block_size, block_size).permute(
        0, 2, 1, 3).flatten(2, 3).flatten(0, 1)
    assert torch.all(torch.abs(pflows - cum_pflows) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(23289)
    hmm_forward_backward_test()
