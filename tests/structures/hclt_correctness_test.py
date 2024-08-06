import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def test_hclt_forward():

    torch.manual_seed(238900)

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
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    lls = pc(batch_data, force_use_fp32 = True)

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
                params = ns._params.reshape(num_blocks, num_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3)
                params = params.reshape(num_latents, num_latents * ns.num_chs)

                emars_max = torch.max(emars, dim = 0).values[None,:]
                emars = (emars - emars_max).exp()

                nmars = torch.matmul(params, emars)
                nmars = nmars.log() + emars_max

                sid, eid = ns._output_ind_range
                
                assert torch.all(torch.abs(nmars - node_mars[sid:eid,:]) < 1e-3)

                ns2mars[ns] = node_mars[sid:eid,:]

            else:
                assert ns == root_ns

                emars = torch.cat([ns2mars[cs] for cs in ns.chs], dim = 0)
                params = ns._params.reshape(1, num_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
                params = params.reshape(1, num_latents * ns.num_chs)

                emars_max = torch.max(emars, dim = 0).values[None,:]
                emars = (emars - emars_max).exp()

                nmars = torch.matmul(params, emars)
                nmars = nmars.log() + emars_max

                sid, eid = ns._output_ind_range
                assert torch.all(torch.abs(nmars - node_mars[sid:eid,:]) < 4e-3)


def test_hclt_single_layer_backward():

    torch.manual_seed(84738)

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
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    pc.init_param_flows(flows_memory = 0.0)

    lls = pc(batch_data)
    pc.backward(batch_data, allow_modify_flows = False)

    pc.update_param_flows()

    for layer_id in range(1, len(pc.inner_layer_groups) - 2, 2):

        node_mars = pc.node_mars.clone()
        node_flows = pc.node_flows.clone()
        element_mars = pc.element_mars.clone()
        element_flows = pc.element_flows.clone()
        params = pc.params.clone()
        param_flows = pc.param_flows.clone().zero_()

        my_layer = pc.inner_layer_groups[layer_id][0]
        previous_layer = pc.inner_layer_groups[layer_id-1][0]

        previous_layer.forward(node_mars, element_mars, _for_backward = True)

        my_layer.backward(node_flows, element_flows, node_mars, element_mars, params, 
                          param_flows = param_flows, allow_modify_flows = False, propagation_alg = "LL")

        chids = my_layer.partitioned_chids[0]
        parids = my_layer.partitioned_parids[0]
        parpids = my_layer.partitioned_parpids[0]

        nids = my_layer.partitioned_nids[0]
        cids = my_layer.partitioned_cids[0]
        pids = my_layer.partitioned_pids[0]
        pfids = my_layer.partitioned_pfids[0]

        for i in range(chids.size(0)):
            eflows = torch.zeros([block_size, batch_size], dtype = torch.float32, device = device)

            for j in range(parids.size(1)):
                nflows = node_flows[parids[i,j]:parids[i,j]+block_size,:] # [num_par_nodes, batch_size]
                nmars = node_mars[parids[i,j]:parids[i,j]+block_size,:] # [num_par_nodes, batch_size]
                emars = element_mars[chids[i]:chids[i]+block_size,:] # [num_ch_nodes, batch_size]
                epars = params[parpids[i,j]:parpids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]
                fpars = param_flows[pfids[i,j]:pfids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]

                curr_eflows = (nflows[None,:,:] * (epars.log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 1)
                eflows += curr_eflows

            assert torch.all(torch.abs(eflows - element_flows[chids[i]:chids[i]+block_size,:]) < 1e-3)

        for i in range(nids.size(0)):
            for j in range(0, cids.size(1), block_size):
                nflows = node_flows[nids[i]:nids[i]+block_size,:] # [num_par_nodes, batch_size]
                nmars = node_mars[nids[i]:nids[i]+block_size,:] # [num_par_nodes, batch_size]
                emars = element_mars[cids[i,j]:cids[i,j]+block_size,:] # [num_ch_nodes, batch_size]
                epars = params[pids[i,j]:pids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]
                fpars = param_flows[pfids[i,j]:pfids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]

                pflows = (nflows[None,:,:] * (epars.log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 2)

                assert torch.all(torch.abs(fpars - pflows) < 3e-4 * batch_size)


def test_hclt_single_layer_backward_general_em():

    torch.manual_seed(62328)

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
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    alpha = 2.0

    pc.init_param_flows(flows_memory = 0.0)

    lls = pc(batch_data, propagation_alg = "GeneralLL", alpha = alpha)
    pc.backward(batch_data, allow_modify_flows = False,
                propagation_alg = "GeneralLL", alpha = alpha)

    pc.update_param_flows()

    for layer_id in range(1, len(pc.inner_layer_groups) - 2, 2):

        node_mars = pc.node_mars.clone()
        node_flows = pc.node_flows.clone()
        element_mars = pc.element_mars.clone()
        element_flows = pc.element_flows.clone()
        params = pc.params.clone()
        param_flows = pc.param_flows.clone().zero_()

        my_layer = pc.inner_layer_groups[layer_id][0]
        previous_layer = pc.inner_layer_groups[layer_id-1][0]

        previous_layer.forward(node_mars, element_mars, _for_backward = True)

        my_layer.backward(node_flows, element_flows, node_mars, element_mars, params, 
                          param_flows = param_flows, allow_modify_flows = False, 
                          propagation_alg = "GeneralLL", alpha = alpha)

        chids = my_layer.partitioned_chids[0]
        parids = my_layer.partitioned_parids[0]
        parpids = my_layer.partitioned_parpids[0]

        nids = my_layer.partitioned_nids[0]
        cids = my_layer.partitioned_cids[0]
        pids = my_layer.partitioned_pids[0]
        pfids = my_layer.partitioned_pfids[0]

        for i in range(chids.size(0)):
            eflows = torch.zeros([block_size, batch_size], dtype = torch.float32, device = device)

            for j in range(parids.size(1)):
                nflows = node_flows[parids[i,j]:parids[i,j]+block_size,:] # [num_par_nodes, batch_size]
                nmars = node_mars[parids[i,j]:parids[i,j]+block_size,:] # [num_par_nodes, batch_size]
                emars = element_mars[chids[i]:chids[i]+block_size,:] # [num_ch_nodes, batch_size]
                epars = params[parpids[i,j]:parpids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]
                fpars = param_flows[pfids[i,j]:pfids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]

                curr_eflows = (nflows[None,:,:] * ((epars.log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]) * alpha).exp()).sum(dim = 1)
                eflows += curr_eflows

            assert torch.all(torch.abs(eflows - element_flows[chids[i]:chids[i]+block_size,:]) < 1e-3)

        for i in range(nids.size(0)):
            for j in range(0, cids.size(1), block_size):
                nflows = node_flows[nids[i]:nids[i]+block_size,:] # [num_par_nodes, batch_size]
                nmars = node_mars[nids[i]:nids[i]+block_size,:] # [num_par_nodes, batch_size]
                emars = element_mars[cids[i,j]:cids[i,j]+block_size,:] # [num_ch_nodes, batch_size]
                epars = params[pids[i,j]:pids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]
                fpars = param_flows[pfids[i,j]:pfids[i,j]+block_size**2].reshape(block_size, block_size) # [num_ch_nodes, num_par_nodes]

                pflows = (nflows[None,:,:] * (epars.log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 2)

                assert torch.all(torch.abs(fpars - pflows) < 3e-4 * batch_size)


def test_hclt_backward():

    torch.manual_seed(3467)

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
    pc.backward(batch_data, allow_modify_flows = False)

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

    gt_ch_flows = dict()

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
                assert torch.all(torch.abs(node_flows[sid:eid,:] - 1.0) < 1e-4)

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

                    eflows = nflows * params.permute(1, 0) * (emars - nmars).exp()
                    pflows = eflows.sum(dim = 1)

                    assert torch.all(torch.abs(pflows - param_flows[0,:]) < 6e-3)

                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size], device = device)
                    ns2flows[cs] += eflows

                    gt_ch_flows[cs] = eflows.detach().clone()

            elif ns.is_prod():
                nflows = ns2flows[ns]
                gt_flows = gt_ch_flows[ns]

                assert torch.all(torch.abs(gt_flows - nflows) < 1e-4)

                for cs in ns.chs:
                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size], device = device)
                    ns2flows[cs] += nflows

            elif ns.is_sum():

                for par_cs in ch2par[ns]:
                    assert par_cs in visited

                nflows = ns2flows[ns]

                sid, eid = ns._output_ind_range

                # if len(ns.scope) > 2:
                #     if not torch.all(torch.abs(nflows - node_flows[sid:eid,:]) < 1e-3):
                #         import pdb; pdb.set_trace()
                #     assert torch.all(torch.abs(nflows - node_flows[sid:eid,:]) < 1e-3)

                # ns2flows[ns] = node_flows[sid:eid,:]
                # print(">>>>>>", torch.abs(nflows - node_flows[sid:eid,:]).max())

                nflows = node_flows[sid:eid,:]

                nmars = node_mars[sid:eid,:]

                ch_eflows = []
                ch_pflows = []

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

                    eflows = (nflows[None,:,:] * (params.permute(1, 0).log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 1)
                    pflows = (nflows[None,:,:] * (params.permute(1, 0).log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 2).permute(1, 0)

                    log_n_fdm = nflows.log() - nmars
                    log_n_fdm_max = log_n_fdm.max(dim = 0).values
                    n_fdm_sub = (log_n_fdm - log_n_fdm_max[None,:]).exp()

                    eflows_prim = torch.matmul(params.permute(1, 0), n_fdm_sub) * (emars + log_n_fdm_max[None,:]).exp()

                    scaled_emars = (emars + log_n_fdm_max[None,:]).exp()
                    pflows_prim = torch.matmul(n_fdm_sub, scaled_emars.permute(1, 0)) * params

                    # From `pc`
                    pc_eflows = (node_flows[sid:eid,:][None,:,:] * (params.permute(1, 0).log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 1)
                    pc_pflows = (node_flows[sid:eid,:][None,:,:] * (params.permute(1, 0).log()[:,:,None] + emars[:,None,:] - nmars[None,:,:]).exp()).sum(dim = 2).permute(1, 0)

                    # log_n_fdm = node_flows[sid:eid,:].log() - nmars
                    # log_n_fdm_max = log_n_fdm.max(dim = 0).values
                    # n_fdm_sub = (log_n_fdm - log_n_fdm_max[None,:]).exp()

                    # pc_eflows_prim = torch.matmul(params.permute(1, 0), n_fdm_sub) * (emars + log_n_fdm_max[None,:]).exp()

                    # print(torch.abs(eflows - eflows_prim).max())
                    # print(torch.abs(pflows - pflows_prim).max())

                    ch_eflows.append(eflows)
                    ch_pflows.append(pflows)

                    assert torch.all(torch.abs(pflows - param_flows) < 1e-3 * batch_size)

                    if cs not in ns2flows:
                        ns2flows[cs] = torch.zeros([num_latents, batch_size], device = device)
                    ns2flows[cs] += eflows

                ## Run the actual layer ##

                curr_layer_id = -1
                curr_layer = None
                for layer_id in range(1, len(pc.inner_layer_groups), 2):
                    layer = pc.inner_layer_groups[layer_id][0]
                    if ns in layer.nodes:
                        curr_layer_id = layer_id
                        curr_layer = layer

                assert curr_layer is not None

                nsid, neid = ns._output_ind_range

                temp_node_flows[nsid:neid,:] = nflows
                temp_param_flows[:] = 0.0

                pc.inner_layer_groups[curr_layer_id - 1].forward(temp_node_mars, temp_element_mars, _for_backward = True)

                curr_layer.backward(temp_node_flows, temp_element_flows, temp_node_mars, temp_element_mars, temp_params, 
                                    param_flows = temp_param_flows, allow_modify_flows = False, propagation_alg = "LL")

                pfsid, pfeid = ns._param_flow_range

                for i, cs in enumerate(ns.chs):
                    eflows = ch_eflows[i]
                    pflows = ch_pflows[i]

                    csid, ceid = cs._output_ind_range

                    # print("value", torch.abs(eflows - temp_element_flows[csid:ceid,:]).max())

                    assert torch.all(torch.abs(eflows - temp_element_flows[csid:ceid,:]) < 1e-3)
                    assert torch.all(torch.abs(temp_param_flows[pfsid:pfeid].reshape(num_latents, num_latents) - pflows.permute(1, 0)) < batch_size * 1e-4)

                    assert cs not in gt_ch_flows
                    gt_ch_flows[cs] = eflows.detach().clone()


def test_hclt_em():

    torch.manual_seed(76767)

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
    data_cpu = batch_data.cpu().long()
    batch_size = batch_data.size(0)

    lls = pc(batch_data)
    pc.backward(batch_data, allow_modify_flows = False)

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
            old_params = ns2old_params[ns].reshape(num_blocks, num_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3)
            old_params = old_params.reshape(num_latents, num_latents * ns.num_chs)

            ref_params = ns._params.reshape(num_blocks, num_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3)
            ref_params = ref_params.reshape(num_latents, num_latents * ns.num_chs)

            par_flows = ns._param_flows.reshape(num_blocks, num_blocks * ns.num_chs, block_size, block_size).permute(0, 2, 1, 3)
            par_flows = par_flows.reshape(num_latents, num_latents * ns.num_chs)

            new_params = (par_flows + pseudocount / par_flows.size(1)) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)

            updated_params = (1.0 - step_size) * old_params + step_size * new_params

            assert torch.all(torch.abs(ref_params - updated_params) < 1e-4)

        elif ns == root_ns:
            old_params = ns2old_params[ns].reshape(1, num_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
            old_params = old_params.reshape(1, num_latents * ns.num_chs)

            ref_params = ns._params.reshape(1, num_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
            ref_params = ref_params.reshape(1, num_latents * ns.num_chs)

            par_flows = ns._param_flows.reshape(1, num_blocks * ns.num_chs, 1, block_size).permute(0, 2, 1, 3)
            par_flows = par_flows.reshape(1, num_latents * ns.num_chs)

            new_params = (par_flows + pseudocount / par_flows.size(1)) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)

            updated_params = (1.0 - step_size) * old_params + step_size * new_params

            assert torch.all(torch.abs(ref_params - updated_params) < 1e-4)


if __name__ == "__main__":
    test_hclt_forward()
    test_hclt_single_layer_backward()
    test_hclt_backward()
    test_hclt_em()
    test_hclt_single_layer_backward_general_em()
