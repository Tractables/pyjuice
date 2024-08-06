import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice
import pyjuice.distributions as dists


def logsubexp(x, y):
    """
    Compute log(exp(x) - exp(y)) in a numerically stable way.
    """
    x, y = torch.maximum(x, y), torch.minimum(x, y)

    # Compute the maximum value between x and y element-wise
    max_val = torch.max(x, y)
    
    # Compute the result using logsumexp trick
    result = max_val + torch.log(torch.exp(x - max_val) - torch.exp(y - max_val))
    
    return result


def test_logspace_hclt_backward():

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

    block_size = root_ns.chs[0].block_size
    num_blocks = num_latents // block_size

    batch_data = train_data[:512,:].contiguous().to(device)
    data_cpu = batch_data.long()
    batch_size = batch_data.size(0)

    pc.init_param_flows(flows_memory = 0.0)

    lls = pc(batch_data)
    pc.backward(batch_data, allow_modify_flows = False, logspace_flows = True)

    pc.update_param_flows()

    node_mars = pc.node_mars
    node_flows = pc.node_flows

    temp_node_mars = pc.node_mars.clone()
    temp_node_flows = pc.node_flows.clone()
    temp_element_mars = pc.element_mars.clone()
    temp_element_flows = pc.element_flows.clone()
    temp_params = pc.params
    temp_param_flows = pc.param_flows.clone()

    ns2flows = dict()
    ns2flows[root_ns] = torch.ones([1, batch_size], device = device)

    ch2par = dict()
    for ns in root_ns:
        for cs in ns.chs:
            if cs not in ch2par:
                ch2par[cs] = set()
            ch2par[cs].add(ns)

    visited = set()

    with torch.no_grad():
        for ns in root_ns(reverse = True):
            visited.add(ns)
            if ns == root_ns:

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(node_flows[sid:eid,:] - 0.0) < 1e-4)

                nflows = ns2flows[ns]
                nmars = node_mars[sid:eid,:]

                for i, cs in enumerate(ns.chs):
                    params = ns._params.reshape(1, num_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3).to(device)
                    params = params[:,:,i*num_blocks:(i+1)*num_blocks,:].reshape(1, num_latents)

                    param_flows = ns._param_flows.reshape(1, num_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3).to(device)
                    param_flows = param_flows[:,:,i*num_blocks:(i+1)*num_blocks,:].reshape(1, num_latents)

                    if cs.is_prod():
                        emars = torch.zeros([num_latents, batch_size], device = device)
                        for cns in cs.chs:
                            sid, eid = cns._output_ind_range
                            emars += node_mars[sid:eid,:]
                    else:
                        raise ValueError()

                    eflows = nflows.log() + params.log().permute(1, 0) + emars - nmars
                    pflows = eflows.exp().sum(dim = 1)

                    assert torch.all(torch.abs(pflows - param_flows[0,:]) < 1e-4 * batch_size)

                    ns2flows[cs] = eflows

            elif ns.is_prod():
                nflows = ns2flows[ns]

                for cs in ns.chs:
                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size], device = device) - float("inf")
                    ns2flows[cs] = torch.logaddexp(ns2flows[cs], nflows)

            elif ns.is_sum():

                for par_cs in ch2par[ns]:
                    assert par_cs in visited

                nflows = ns2flows[ns]

                sid, eid = ns._output_ind_range

                assert torch.all(logsubexp(nflows, node_flows[sid:eid,:]).exp() < 1e-3)
                assert (logsubexp(nflows, node_flows[sid:eid,:]).exp() > 1e-5).float().mean() < 0.2

                nflows = node_flows[sid:eid,:]

                nmars = node_mars[sid:eid,:]

                ch_eflows = []

                for i, cs in enumerate(ns.chs):
                    params = ns._params.reshape(num_blocks, num_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3).to(device)
                    params = params[:,:,i*num_blocks:(i+1)*num_blocks,:].reshape(num_latents, num_latents)

                    param_flows = ns._param_flows.reshape(num_blocks, num_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3).to(device)
                    param_flows = param_flows[:,:,i*num_blocks:(i+1)*num_blocks,:].reshape(num_latents, num_latents)

                    if cs.is_prod():
                        emars = torch.zeros([num_latents, batch_size], device = device)
                        for cns in cs.chs:
                            sid, eid = cns._output_ind_range
                            emars += node_mars[sid:eid,:]
                    else:
                        raise ValueError()

                    eflows = (nflows[None,:,:] + params.permute(1, 0).log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).logsumexp(dim = 1)
                    pflows = (nflows[None,:,:] + params.permute(1, 0).log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).logsumexp(dim = 2).permute(1, 0).exp()

                    ch_eflows.append(eflows)

                    assert torch.all(torch.abs(pflows - param_flows) < 1e-4 * batch_size)

                    ns2flows[cs] = eflows


if __name__ == "__main__":
    torch.set_num_threads(4)
    test_logspace_hclt_backward()
