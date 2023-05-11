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
    n2 = n1.duplicate(m2)

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
    i1 = i0.duplicate(1)

    m = juice.multiply(i0, i1)
    n = juice.summate(m, num_nodes = 1)

    pc = juice.TensorCircuit(n)
    pc.to(device)

    data = torch.randint(0, 5, [16, 2]).to(device)

    lls = pc(data)

    pc.backward(data)

    aaa = pc.input_layers[0].param_flows[:10].clone()
    bbb = pc.input_layers[0].param_flows[10:].clone()

    pc.input_layers[0]._tie_param_flows(pc.input_layers[0].param_flows)

    ccc = pc.input_layers[0].param_flows[:10]
    ddd = pc.input_layers[0].param_flows[10:]

    assert torch.abs(aaa + bbb - ccc).max() < 1e-6
    assert torch.abs(aaa + bbb - ddd).max() < 1e-6


if __name__ == "__main__":
    tie_function_test()
    tie_sum_nodes_test()
    tie_input_nodes_test()