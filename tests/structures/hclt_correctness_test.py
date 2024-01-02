import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def hclt_forward_test():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)[:5000,:]

    num_features = train_data.size(1)
    num_latents = 128

    root_ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = num_latents, 
        chunk_size = 32
    )
    root_ns.init_parameters()

    pc = juice.TensorCircuit(root_ns)

    pc.to(device)

    group_size = root_ns.chs[0].group_size
    num_groups = num_latents // group_size

    batch_data = train_data[:512,:].contiguous().to(device)
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    lls = pc(batch_data)

    node_mars = pc.node_mars.cpu()

    ns2mars = dict()

    with torch.no_grad():
        for ns in root_ns:
            if ns.is_input():
                v = ns.scope.to_list()[0]
                params = ns._params.reshape(num_latents, 256)

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
                params = ns._params.reshape(num_groups, num_groups * ns.num_chs, group_size, group_size).permute(0, 2, 1, 3)
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
                params = ns._params.reshape(1, num_groups * ns.num_chs, 1, group_size).permute(0, 2, 1, 3)
                params = params.reshape(1, num_latents * ns.num_chs)

                emars_max = torch.max(emars, dim = 0).values[None,:]
                emars = (emars - emars_max).exp()

                nmars = torch.matmul(params, emars)
                nmars = nmars.log() + emars_max

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(nmars - node_mars[sid:eid,:]) < 4e-3)


def hclt_backward_test():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)[:5000,:]

    num_features = train_data.size(1)
    num_latents = 128

    root_ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = num_latents, 
        chunk_size = 32
    )
    root_ns.init_parameters()

    pc = juice.TensorCircuit(root_ns)

    pc.to(device)

    group_size = root_ns.chs[0].group_size
    num_groups = num_latents // group_size

    batch_data = train_data[:512,:].contiguous().to(device)
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    lls = pc(batch_data)
    pc.backward(batch_data.permute(1, 0), allow_modify_flows = False)

    pc.update_param_flows()

    node_mars = pc.node_mars.cpu()
    node_flows = pc.node_flows.cpu()

    ns2flows = dict()
    ns2flows[root_ns] = torch.ones([1, batch_size])

    with torch.no_grad():
        for ns in root_ns(reverse = True):
            if ns == root_ns:

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(node_flows[sid:eid,:] - 1.0) < 1e-4)

                nflows = ns2flows[ns]
                nmars = node_mars[sid:eid,:]

                for i, cs in enumerate(ns.chs):
                    params = ns._params.reshape(1, num_groups * ns.num_chs, 1, group_size).permute(0, 2, 1, 3)
                    params = params[:,:,i*num_groups:(i+1)*num_groups,:].reshape(1, num_latents)

                    param_flows = ns._param_flows.reshape(1, num_groups * ns.num_chs, 1, group_size).permute(0, 2, 1, 3)
                    param_flows = param_flows[:,:,i*num_groups:(i+1)*num_groups,:].reshape(1, num_latents)

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

                assert (torch.abs(nflows - node_flows[sid:eid,:]) > 1e-3).float().mean() < 0.02

                nmars = node_mars[sid:eid,:]

                for i, cs in enumerate(ns.chs):
                    params = ns._params.reshape(num_groups, num_groups * ns.num_chs, group_size, group_size).permute(0, 2, 1, 3)
                    params = params[:,:,i*num_groups:(i+1)*num_groups,:].reshape(num_latents, num_latents)

                    param_flows = ns._param_flows.reshape(num_groups, num_groups * ns.num_chs, group_size, group_size).permute(0, 2, 1, 3)
                    param_flows = param_flows[:,:,i*num_groups:(i+1)*num_groups,:].reshape(num_latents, num_latents)

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

                    assert torch.all(torch.abs(pflows - param_flows) < 0.5)

                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size])
                    ns2flows[cs] += eflows


def hclt_em_test():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)[:5000,:]

    num_features = train_data.size(1)
    num_latents = 128

    root_ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = num_latents, 
        chunk_size = 32
    )
    root_ns.init_parameters()

    pc = juice.TensorCircuit(root_ns)

    pc.to(device)

    group_size = root_ns.chs[0].group_size
    num_groups = num_latents // group_size

    batch_data = train_data[:512,:].contiguous().to(device)
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    lls = pc(batch_data)
    pc.backward(batch_data.permute(1, 0), allow_modify_flows = False)

    ns2old_params = dict()
    for ns in root_ns:
        if ns.is_sum() and ns.has_params():
            ns2old_params[ns] = ns._params.clone()

    pseudocount = 0.01
    step_size = 0.24

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    pc.update_parameters()
    pc.update_param_flows()

    for ns in root_ns:
        if ns.is_sum() and ns != root_ns:
            old_params = ns2old_params[ns].reshape(num_groups, num_groups * ns.num_chs, group_size, group_size).permute(0, 2, 1, 3)
            old_params = old_params.reshape(num_latents, num_latents * ns.num_chs)

            ref_params = ns._params.reshape(num_groups, num_groups * ns.num_chs, group_size, group_size).permute(0, 2, 1, 3)
            ref_params = ref_params.reshape(num_latents, num_latents * ns.num_chs)

            par_flows = ns._param_flows.reshape(num_groups, num_groups * ns.num_chs, group_size, group_size).permute(0, 2, 1, 3)
            par_flows = par_flows.reshape(num_latents, num_latents * ns.num_chs)

            new_params = (par_flows + pseudocount / par_flows.size(1)) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)

            updated_params = (1.0 - step_size) * old_params + step_size * new_params

            assert torch.all(torch.abs(ref_params - updated_params) < 1e-4)

        elif ns == root_ns:
            old_params = ns2old_params[ns].reshape(1, num_groups * ns.num_chs, 1, group_size).permute(0, 2, 1, 3)
            old_params = old_params.reshape(1, num_latents * ns.num_chs)

            ref_params = ns._params.reshape(1, num_groups * ns.num_chs, 1, group_size).permute(0, 2, 1, 3)
            ref_params = ref_params.reshape(1, num_latents * ns.num_chs)

            par_flows = ns._param_flows.reshape(1, num_groups * ns.num_chs, 1, group_size).permute(0, 2, 1, 3)
            par_flows = par_flows.reshape(1, num_latents * ns.num_chs)

            new_params = (par_flows + pseudocount / par_flows.size(1)) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)

            updated_params = (1.0 - step_size) * old_params + step_size * new_params

            assert torch.all(torch.abs(ref_params - updated_params) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(320942)
    hclt_forward_test()
    hclt_backward_test()
    hclt_em_test()
