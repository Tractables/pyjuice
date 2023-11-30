import pyjuice as juice
import torch
import numpy as np
import math

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def categorical_nodes_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = inputs(2, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = inputs(3, num_nodes = 2, dist = dists.Categorical(num_cats = 2))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## Input node forward tests ##

    for i in range(16):
        assert torch.abs(pc.node_mars[1,i] - torch.log(pc.input_layers[0].params[data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[2,i] - torch.log(pc.input_layers[0].params[2+data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[3,i] - torch.log(pc.input_layers[0].params[4+data[i,1]])) < 1e-4
        assert torch.abs(pc.node_mars[4,i] - torch.log(pc.input_layers[0].params[6+data[i,1]])) < 1e-4
        assert torch.abs(pc.node_mars[5,i] - torch.log(pc.input_layers[0].params[8+data[i,2]])) < 1e-4
        assert torch.abs(pc.node_mars[6,i] - torch.log(pc.input_layers[0].params[10+data[i,2]])) < 1e-4
        assert torch.abs(pc.node_mars[7,i] - torch.log(pc.input_layers[0].params[12+data[i,3]])) < 1e-4
        assert torch.abs(pc.node_mars[8,i] - torch.log(pc.input_layers[0].params[14+data[i,3]])) < 1e-4

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([16], device = pc.node_flows.device)

    for i in range(16):
        gt_param_flows[data[i,0]] += pc.node_flows[1,i]
        gt_param_flows[2+data[i,0]] += pc.node_flows[2,i]
        gt_param_flows[4+data[i,1]] += pc.node_flows[3,i]
        gt_param_flows[6+data[i,1]] += pc.node_flows[4,i]
        gt_param_flows[8+data[i,2]] += pc.node_flows[5,i]
        gt_param_flows[10+data[i,2]] += pc.node_flows[6,i]
        gt_param_flows[12+data[i,3]] += pc.node_flows[7,i]
        gt_param_flows[14+data[i,3]] += pc.node_flows[8,i]

    assert torch.all(torch.abs(gt_param_flows - pc.input_layers[0].param_flows) < 1e-4)

    ## EM tests ##

    original_params = pc.input_layers[0].params.clone()

    step_size = 0.3
    pseudocount = 0.1

    par_flows = pc.input_layers[0].param_flows.clone().reshape(8, 2)
    new_params = (1.0 - step_size) * original_params + step_size * ((par_flows + pseudocount / 2) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)).reshape(-1)

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(new_params - pc.input_layers[0].params) < 1e-4)


def bernoulli_nodes_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Bernoulli())
    ni1 = inputs(1, num_nodes = 2, dist = dists.Bernoulli())
    ni2 = inputs(2, num_nodes = 2, dist = dists.Bernoulli())
    ni3 = inputs(3, num_nodes = 2, dist = dists.Bernoulli())

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## Input node forward tests ##

    for i in range(16):
        assert torch.abs(pc.node_mars[1,i].exp() - (pc.input_layers[0].params[0] if data[i,0] == 1 else (1.0 - pc.input_layers[0].params[0]))) < 1e-4
        assert torch.abs(pc.node_mars[2,i].exp() - (pc.input_layers[0].params[1] if data[i,0] == 1 else (1.0 - pc.input_layers[0].params[1]))) < 1e-4
        assert torch.abs(pc.node_mars[3,i].exp() - (pc.input_layers[0].params[2] if data[i,1] == 1 else (1.0 - pc.input_layers[0].params[2]))) < 1e-4
        assert torch.abs(pc.node_mars[4,i].exp() - (pc.input_layers[0].params[3] if data[i,1] == 1 else (1.0 - pc.input_layers[0].params[3]))) < 1e-4
        assert torch.abs(pc.node_mars[5,i].exp() - (pc.input_layers[0].params[4] if data[i,2] == 1 else (1.0 - pc.input_layers[0].params[4]))) < 1e-4
        assert torch.abs(pc.node_mars[6,i].exp() - (pc.input_layers[0].params[5] if data[i,2] == 1 else (1.0 - pc.input_layers[0].params[5]))) < 1e-4
        assert torch.abs(pc.node_mars[7,i].exp() - (pc.input_layers[0].params[6] if data[i,3] == 1 else (1.0 - pc.input_layers[0].params[6]))) < 1e-4
        assert torch.abs(pc.node_mars[8,i].exp() - (pc.input_layers[0].params[7] if data[i,3] == 1 else (1.0 - pc.input_layers[0].params[7]))) < 1e-4

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([16], device = pc.node_flows.device)

    offsets = 1 - data
    for i in range(16):
        gt_param_flows[offsets[i,0]] += pc.node_flows[1,i]
        gt_param_flows[2+offsets[i,0]] += pc.node_flows[2,i]
        gt_param_flows[4+offsets[i,1]] += pc.node_flows[3,i]
        gt_param_flows[6+offsets[i,1]] += pc.node_flows[4,i]
        gt_param_flows[8+offsets[i,2]] += pc.node_flows[5,i]
        gt_param_flows[10+offsets[i,2]] += pc.node_flows[6,i]
        gt_param_flows[12+offsets[i,3]] += pc.node_flows[7,i]
        gt_param_flows[14+offsets[i,3]] += pc.node_flows[8,i]

    assert torch.all(torch.abs(gt_param_flows - pc.input_layers[0].param_flows) < 1e-4)

    ## EM tests ##

    original_params = pc.input_layers[0].params.clone()

    step_size = 0.3
    pseudocount = 0.1

    par_flows = pc.input_layers[0].param_flows.clone().reshape(8, 2)
    new_params = (1.0 - step_size) * original_params + step_size * ((par_flows + pseudocount / 2) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount))[:,0]

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(new_params - pc.input_layers[0].params) < 1e-4)


def gaussian_nodes_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Gaussian(mu = 0.0, sigma = 1.0))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Gaussian(mu = 0.0, sigma = 1.0))
    ni2 = inputs(2, num_nodes = 2, dist = dists.Gaussian(mu = 0.0, sigma = 1.0))
    ni3 = inputs(3, num_nodes = 2, dist = dists.Gaussian(mu = 0.0, sigma = 1.0))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randn([16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## Input node forward tests ##

    for j in range(8):
        gt_probs = torch.distributions.normal.Normal(pc.input_layers[0].params[2*j], pc.input_layers[0].params[2*j+1]).log_prob(data[:,j//2])
        assert torch.all(torch.abs(gt_probs - pc.node_mars[j+1,:]) < 1e-4)

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([24], device = pc.node_flows.device)

    for j in range(8):
        gt_param_flows[3*j] = (data[:,j//2] * pc.node_flows[j+1,:]).sum()
        gt_param_flows[3*j+1] = ((data[:,j//2] ** 2) * pc.node_flows[j+1,:]).sum()
        gt_param_flows[3*j+2] = (pc.node_flows[j+1,:]).sum()

    assert torch.all(torch.abs(gt_param_flows - pc.input_layers[0].param_flows) < 1e-2)

    ## EM tests ##

    mu = pc.input_layers[0].params.reshape(8, 2)[:,0].clone()
    sigma = pc.input_layers[0].params.reshape(8, 2)[:,1].clone()
    ori_theta1 = mu
    ori_theta2 = sigma * sigma + mu * mu

    step_size = 0.3
    pseudocount = 0.1
    min_sigma = 0.01

    stat1 = pc.input_layers[0].param_flows.reshape(8, 3)[:,0]
    stat2 = pc.input_layers[0].param_flows.reshape(8, 3)[:,1]
    stat3 = pc.input_layers[0].param_flows.reshape(8, 3)[:,2]

    new_theta1 = stat1 / (stat3 + 1e-10)
    new_theta2 = stat2 / (stat3 + 1e-10)

    # Get the updated natural parameters
    updated_theta1 = (1.0 - step_size) * ori_theta1 + step_size * new_theta1
    updated_theta2 = (1.0 - step_size) * ori_theta2 + step_size * new_theta2

    # Reconstruct `mu` and `sigma` from the natural parameters
    updated_mu = updated_theta1
    updated_sigma2 = updated_theta2 - updated_mu * updated_mu
    updated_sigma = torch.where(updated_sigma2 < min_sigma * min_sigma, min_sigma, torch.sqrt(updated_sigma2))

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(updated_mu - pc.input_layers[0].params.reshape(8, 2)[:,0]) < 1e-4)
    assert torch.all(torch.abs(updated_sigma.clamp(min = 0.01) - pc.input_layers[0].params.reshape(8, 2)[:,1]) < 1e-4)


def discrete_logistic_nodes_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))
    ni1 = inputs(1, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))
    ni2 = inputs(2, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))
    ni3 = inputs(3, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.arange(0, 5)[:,None].repeat(1, 4).to(device)

    lls = pc(data)

    pc.backward(data)

    ## Input node forward tests ##

    range_low = -1.0
    range_high = 1.0

    num_cats = 5

    for j in range(8):
        interval = (range_high - range_low) / num_cats
        vlow = data[:,j//2] * interval + range_low
        vhigh = vlow + interval

        mu = pc.input_layers[0].params[2*j]
        s = pc.input_layers[0].params[2*j+1]

        cdfhigh = torch.where(data[:,j//2] == num_cats - 1, 1.0, 1.0 / (1.0 + torch.exp((mu - vhigh) / s)))
        cdflow = torch.where(data[:,j//2] == 0, 0.0, 1.0 / (1.0 + torch.exp((mu - vlow) / s)))

        log_probs = torch.log(cdfhigh - cdflow)

        assert torch.all(torch.abs(pc.node_mars[j+1,:] - log_probs) < 1e-4)

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([24], device = pc.node_flows.device)

    for j in range(8):
        interval = (range_high - range_low) / num_cats
        vmid = data[:,j//2] * interval + range_low + 0.5 * interval # (vlow + vhigh) / 2

        gt_param_flows[3*j] = (vmid * pc.node_flows[j+1,:]).sum()
        gt_param_flows[3*j+1] = ((vmid ** 2) * pc.node_flows[j+1,:]).sum()
        gt_param_flows[3*j+2] = (pc.node_flows[j+1,:]).sum()

    assert torch.all(torch.abs(gt_param_flows - pc.input_layers[0].param_flows) < 1e-4)

    ## EM tests ##

    mu = pc.input_layers[0].params.reshape(8, 2)[:,0].clone()
    std = pc.input_layers[0].params.reshape(8, 2)[:,1].clone() * math.pi / math.sqrt(3.0)
    ori_theta1 = mu
    ori_theta2 = std * std + mu * mu

    step_size = 0.3
    pseudocount = 0.1
    min_std = 0.01

    stat1 = pc.input_layers[0].param_flows.reshape(8, 3)[:,0]
    stat2 = pc.input_layers[0].param_flows.reshape(8, 3)[:,1]
    stat3 = pc.input_layers[0].param_flows.reshape(8, 3)[:,2]

    new_theta1 = stat1 / (stat3 + 1e-10)
    new_theta2 = stat2 / (stat3 + 1e-10)

    # Get the updated natural parameters
    updated_theta1 = (1.0 - step_size) * ori_theta1 + step_size * new_theta1
    updated_theta2 = (1.0 - step_size) * ori_theta2 + step_size * new_theta2

    # Reconstruct `mu` and `std` from the expectation parameters (moment matching)
    updated_mu = updated_theta1
    updated_std2 = updated_theta2 - updated_mu * updated_mu
    updated_std = torch.where(updated_std2 < min_std * min_std, min_std, torch.sqrt(updated_std2))
    updated_s = updated_std * math.sqrt(3.0) / math.pi

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(updated_mu - pc.input_layers[0].params.reshape(8, 2)[:,0]) < 1e-4)
    assert torch.all(torch.abs(updated_s - pc.input_layers[0].params.reshape(8, 2)[:,1]) < 1e-4)


def discrete_logistic_nodes_behavior_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))
    ni1 = inputs(1, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))
    ni2 = inputs(2, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))
    ni3 = inputs(3, num_nodes = 2, dist = dists.DiscreteLogistic(val_range = [-1.0, 1.0], num_cats = 5))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.tensor([
        [0,0,2,2],
        [1,0,3,3],
        [0,1,3,2],
        [3,3,4,4],
        [2,2,4,4],
        [2,3,4,4]
    ]).to(device)

    for _ in range(40):
        lls = pc(data)

        pc.backward(data)

        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.01)

    assert lls.mean() > -3.5

    pc.update_parameters()

    assert (ni0._params[0] > 0.05 and ni0._params[2] < -0.6) or (ni0._params[2] > 0.05 and ni0._params[0] < -0.6)
    assert (ni1._params[0] > 0.05 and ni1._params[2] < -0.6) or (ni1._params[2] > 0.05 and ni1._params[0] < -0.6)
    assert (ni2._params[0] > 0.78 and ni2._params[2] < 0.4) or (ni2._params[2] > 0.78 and ni2._params[0] < 0.4)
    assert (ni3._params[0] > 0.78 and ni3._params[2] < 0.4) or (ni3._params[2] > 0.78 and ni3._params[0] < 0.4)


if __name__ == "__main__":
    categorical_nodes_test()
    bernoulli_nodes_test()
    gaussian_nodes_test()
    discrete_logistic_nodes_test()
    discrete_logistic_nodes_behavior_test()
