import pyjuice as juice
import torch
import pyjuice.nodes.distributions as dists
from pyjuice.functional import tie_param_flows


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

    aaa = pc.param_flows[1:5].clone()
    bbb = pc.param_flows[5:9].clone()

    pc._tie_param_flows(pc.param_flows)

    ccc = pc.param_flows[1:5].clone()
    ddd = pc.param_flows[5:9].clone()

    assert torch.abs(aaa + bbb - ccc).max() < 1e-6
    assert torch.abs(aaa + bbb - ddd).max() < 1e-6


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


if __name__ == "__main__":
    tie_function_test()
    tie_sum_nodes_test()
    tie_input_nodes_test()