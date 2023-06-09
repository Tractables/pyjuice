import pyjuice as juice
import torch
import pyjuice.nodes.distributions as dists
from pyjuice.functional import tie_param_flows
from pyjuice import inputs, multiply, summate


def tie_function_test():

    device = torch.device("cuda:0")

    N = 20

    param_flows = torch.rand([1000]).cuda()
    tied_param_ids = torch.arange(20).cuda()
    tied_param_group_ids = torch.arange(10).unsqueeze(1).repeat(1, 2).reshape(-1).cuda()

    tied_flows = param_flows[:20].reshape(10, 2).sum(dim = 1)

    tie_param_flows(
        param_flows = param_flows, 
        num_tied_params = N, 
        tied_param_ids = tied_param_ids, 
        tied_param_group_ids = tied_param_group_ids
    )

    assert torch.max(torch.abs(tied_flows.unsqueeze(1).repeat(1, 2).reshape(-1) - param_flows[:20])) < 1e-6


def tie_sum_nodes_test():

    device = torch.device("cuda:0")

    num_nodes = 2
    
    i0 = juice.inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i1 = juice.inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i2 = juice.inputs(2, num_nodes, dists.Categorical(num_cats = 5))
    i3 = juice.inputs(3, num_nodes, dists.Categorical(num_cats = 5))

    m1 = juice.multiply(i0, i1)
    n1 = juice.summate(m1, num_nodes = num_nodes)

    m2 = juice.multiply(i2, i3)
    n2 = n1.duplicate(m2, tie_params = True)

    m = juice.multiply(n1, n2)
    n = juice.summate(m, num_nodes = 1)

    pc = juice.TensorCircuit(n)
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    f11 = (torch.exp(pc.node_mars[1,:] + pc.node_mars[3,:] + torch.log(pc.params[1]) - pc.node_mars[9,:]) * pc.node_flows[9,:]).sum()
    f12 = (torch.exp(pc.node_mars[2,:] + pc.node_mars[4,:] + torch.log(pc.params[2]) - pc.node_mars[9,:]) * pc.node_flows[9,:]).sum()
    f13 = (torch.exp(pc.node_mars[1,:] + pc.node_mars[3,:] + torch.log(pc.params[3]) - pc.node_mars[10,:]) * pc.node_flows[10,:]).sum()
    f14 = (torch.exp(pc.node_mars[2,:] + pc.node_mars[4,:] + torch.log(pc.params[4]) - pc.node_mars[10,:]) * pc.node_flows[10,:]).sum()
    
    f21 = (torch.exp(pc.node_mars[5,:] + pc.node_mars[7,:] + torch.log(pc.params[1]) - pc.node_mars[11,:]) * pc.node_flows[11,:]).sum()
    f22 = (torch.exp(pc.node_mars[6,:] + pc.node_mars[8,:] + torch.log(pc.params[2]) - pc.node_mars[11,:]) * pc.node_flows[11,:]).sum()
    f23 = (torch.exp(pc.node_mars[5,:] + pc.node_mars[7,:] + torch.log(pc.params[3]) - pc.node_mars[12,:]) * pc.node_flows[12,:]).sum()
    f24 = (torch.exp(pc.node_mars[6,:] + pc.node_mars[8,:] + torch.log(pc.params[4]) - pc.node_mars[12,:]) * pc.node_flows[12,:]).sum()

    assert torch.abs(f11 + f21 - pc.param_flows[1]) < 1e-4
    assert torch.abs(f12 + f22 - pc.param_flows[2]) < 1e-4
    assert torch.abs(f13 + f23 - pc.param_flows[3]) < 1e-4
    assert torch.abs(f14 + f24 - pc.param_flows[4]) < 1e-4


def tie_input_nodes_test():
    
    device = torch.device("cuda:0")

    num_nodes = 2
    
    i0 = juice.inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i1 = i0.duplicate(1, tie_params = True)

    m = juice.multiply(i0, i1)
    n = juice.summate(m, num_nodes = 1)

    n.init_parameters()

    assert i1.is_tied()
    assert i1.get_source_ns() == i0

    pc = juice.TensorCircuit(n)

    assert torch.all(pc.input_layers[0].vids == torch.tensor([0,0,1,1]))
    assert torch.all(pc.input_layers[0].psids == torch.tensor([0,5,0,5]))
    assert torch.all((pc.input_layers[0].params - i0._params).abs() < 1e-6)

    pc.to(device)

    data = torch.randint(0, 5, [16, 2]).to(device)

    lls = pc(data)

    pc.backward(data)

    dids = data.clone().cpu()
    m1p = i0._params[dids[:,0]] * i0._params[dids[:,1]]
    m2p = i0._params[dids[:,0]+5] * i0._params[dids[:,1]+5]
    log_np = torch.log(m1p * n._params[0] + m2p * n._params[1])

    assert torch.all((log_np - lls.cpu()).abs() < 1e-6)

    m1f = m1p * n._params[0] / (m1p * n._params[0] + m2p * n._params[1])
    m2f = m2p * n._params[1] / (m1p * n._params[0] + m2p * n._params[1])

    assert torch.all((m1f - pc.node_flows[3,:].cpu()).abs() < 1e-6)
    assert torch.all((m2f - pc.node_flows[4,:].cpu()).abs() < 1e-6)

    for i in range(5):
        assert (pc.input_layers[0].param_flows[i] - m1f[dids[:,0] == i].sum() - m1f[dids[:,1] == i].sum()).abs() < 1e-3
        assert (pc.input_layers[0].param_flows[i+5] - m2f[dids[:,0] == i].sum() - m2f[dids[:,1] == i].sum()).abs() < 1e-3


def tie_sparse_nodes_test():
    
    device = torch.device("cuda:0")

    num_nodes = 2
    
    i0 = juice.inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i1 = juice.inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i2 = juice.inputs(2, num_nodes, dists.Categorical(num_cats = 5))
    i3 = juice.inputs(3, num_nodes, dists.Categorical(num_cats = 5))

    m00 = multiply(i0, i1)
    m01 = multiply(i0, i1, edge_ids = torch.tensor([[1,0]], dtype = torch.long))
    n0 = summate(m00, m01, edge_ids = torch.tensor([[0,0,0,1,1],[0,1,2,1,2]], dtype = torch.long))

    m10 = multiply(i2, i3)
    m11 = multiply(i2, i3, edge_ids = torch.tensor([[1,0]], dtype = torch.long))
    n1 = n0.duplicate(m10, m11, tie_params = True)

    m = multiply(n0, n1)
    n = summate(m, num_nodes = 1)

    n.init_parameters()

    pc = juice.TensorCircuit(n)

    pc.to(device)

    data = torch.randint(0, 5, [1, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## Unit tests for compilation result ##

    assert torch.all(pc.inner_layers[1].grouped_nids[0].cpu() == torch.tensor([10, 12], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_nids[1].cpu() == torch.tensor([9, 11], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_cids[0].cpu() == torch.tensor([[2,3],[5,6]], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_cids[1].cpu() == torch.tensor([[1,2,3,0],[4,5,6,0]], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_pids[0].cpu() == torch.tensor([[4,5],[4,5]], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_pids[1].cpu() == torch.tensor([[1,2,3,0],[1,2,3,0]], dtype = torch.long))

    assert torch.all(pc.inner_layers[1].grouped_chids[0].cpu() == torch.tensor([1,4], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_chids[1].cpu() == torch.tensor([2,3,5,6], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_parids[0].cpu() == torch.tensor([[9],[11]], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_parids[1].cpu() == torch.tensor([[9,10],[9,10],[11,12],[11,12]], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_parpids[0].cpu() == torch.tensor([[1],[1]], dtype = torch.long))
    assert torch.all(pc.inner_layers[1].grouped_parpids[1].cpu() == torch.tensor([[2,4],[3,5],[2,4],[3,5]], dtype = torch.long))

    ## Unit tests for parameter flows ##

    p9 = torch.exp(pc.node_mars[9,0])
    p10 = torch.exp(pc.node_mars[10,0])
    p11 = torch.exp(pc.node_mars[11,0])
    p12 = torch.exp(pc.node_mars[12,0])

    f9 = pc.node_flows[9,0]
    f10 = pc.node_flows[10,0]
    f11 = pc.node_flows[11,0]
    f12 = pc.node_flows[12,0]

    pm1 = torch.exp(pc.node_mars[1,0] + pc.node_mars[3,0])
    pm2 = torch.exp(pc.node_mars[2,0] + pc.node_mars[4,0])
    pm3 = torch.exp(pc.node_mars[2,0] + pc.node_mars[3,0])
    pm4 = torch.exp(pc.node_mars[5,0] + pc.node_mars[7,0])
    pm5 = torch.exp(pc.node_mars[6,0] + pc.node_mars[8,0])
    pm6 = torch.exp(pc.node_mars[6,0] + pc.node_mars[7,0])

    assert torch.abs(f9 * pm1 * pc.params[1] / p9 + f11 * pm4 * pc.params[1] / p11 - pc.param_flows[1]) < 1e-4
    assert torch.abs(f9 * pm2 * pc.params[2] / p9 + f11 * pm5 * pc.params[2] / p11 - pc.param_flows[2]) < 1e-4
    assert torch.abs(f9 * pm3 * pc.params[3] / p9 + f11 * pm6 * pc.params[3] / p11 - pc.param_flows[3]) < 1e-4
    assert torch.abs(f10 * pm2 * pc.params[4] / p10 + f12 * pm5 * pc.params[4] / p12 - pc.param_flows[4]) < 1e-4
    assert torch.abs(f10 * pm3 * pc.params[5] / p10 + f12 * pm6 * pc.params[5] / p12 - pc.param_flows[5]) < 1e-4


if __name__ == "__main__":
    tie_function_test()
    tie_sum_nodes_test()
    tie_input_nodes_test()
    tie_sparse_nodes_test()