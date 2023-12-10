import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer, ProdLayer, SumLayer

import pytest


def prod_layer_compilation_test():
    
    for group_size in [1, 8, 16]:
    
        with juice.set_group_size(group_size):

            ni0 = inputs(0, num_node_groups = 3, dist = dists.Categorical(num_cats = 2))
            ni1 = inputs(1, num_node_groups = 7, dist = dists.Categorical(num_cats = 2))
            ni2 = inputs(2, num_node_groups = 6, dist = dists.Categorical(num_cats = 2))
            ni3 = inputs(3, num_node_groups = 12, dist = dists.Categorical(num_cats = 2))
            ni4 = inputs(4, num_node_groups = 4, dist = dists.Categorical(num_cats = 2))

            np0 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 1, 2, 2, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6]]).permute(1, 0))
            np1 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 5, 4, 1, 2, 3, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]).permute(1, 0))
            np2 = multiply(ni1, ni2, edge_ids = torch.tensor([[2, 3, 1, 4, 0, 6, 5], [0, 0, 1, 2, 3, 4, 5]]).permute(1, 0))
            np3 = multiply(ni1, ni3, ni4, edge_ids = torch.tensor([[3, 6, 5, 1, 0, 5, 3, 4, 2, 2, 3, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 3, 1, 2, 0, 2, 3, 0, 1, 2, 1, 3]]).permute(1, 0))
            np4 = multiply(ni3, edge_ids = torch.tensor([[0, 1, 2, 3]]).permute(1, 0))
            np5 = multiply(ni0, ni1, ni2, ni4, edge_ids = torch.tensor([[0, 1, 2, 2, 1, 2, 0], [0, 1, 2, 3, 4, 5, 6], [0, 1, 1, 2, 3, 4, 5], [1, 3, 2, 0, 1, 2, 2]]).permute(1, 0))

        input_layer = InputLayer([ni0, ni1, ni2, ni3, ni4], cum_nodes = group_size)

        prod_layer_cpu = ProdLayer([np0, np1, np2, np3, np4, np5], layer_sparsity_tol = 0.1, disable_gpu_compilation = True)
        prod_layer_gpu = ProdLayer([np0, np1, np2, np3, np4, np5], layer_sparsity_tol = 0.1, force_gpu_compilation = True)

        for i in range(3):
            assert torch.all(prod_layer_cpu.partitioned_nids[i] == prod_layer_gpu.partitioned_nids[i])
            assert torch.all(prod_layer_cpu.partitioned_cids[i] == prod_layer_gpu.partitioned_cids[i])

        for i in range(2):
            assert torch.all(prod_layer_cpu.partitioned_u_cids[i] == prod_layer_gpu.partitioned_u_cids[i])
            assert torch.all(prod_layer_cpu.partitioned_parids[i] == prod_layer_gpu.partitioned_parids[i])


def sum_layer_compilation_test():

    for group_size in [8, 16]:
    
        with juice.set_group_size(group_size):

            ni0 = inputs(0, num_node_groups = 3, dist = dists.Categorical(num_cats = 2))
            ni1 = inputs(1, num_node_groups = 7, dist = dists.Categorical(num_cats = 2))
            ni2 = inputs(2, num_node_groups = 6, dist = dists.Categorical(num_cats = 2))
            ni3 = inputs(3, num_node_groups = 12, dist = dists.Categorical(num_cats = 2))
            ni4 = inputs(4, num_node_groups = 4, dist = dists.Categorical(num_cats = 2))

            np0 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 1, 2, 2, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6]]).permute(1, 0))
            np1 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 5, 4, 1, 2, 3, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]).permute(1, 0))
            np2 = multiply(ni1, ni2, edge_ids = torch.tensor([[2, 3, 1, 4, 0, 6, 5], [0, 0, 1, 2, 3, 4, 5]]).permute(1, 0))
            np3 = multiply(ni1, ni3, ni4, edge_ids = torch.tensor([[3, 6, 5, 1, 0, 5, 3, 4, 2, 2, 3, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 3, 1, 2, 0, 2, 3, 0, 1, 2, 1, 3]]).permute(1, 0))
            np4 = multiply(ni3, edge_ids = torch.tensor([[0, 1, 2, 3]]).permute(1, 0))
            np5 = multiply(ni0, ni1, ni2, ni4, edge_ids = torch.tensor([[0, 1, 2, 2, 1, 2, 0], [0, 1, 2, 3, 4, 5, 6], [0, 1, 1, 2, 3, 4, 5], [1, 3, 2, 0, 1, 2, 2]]).permute(1, 0))
            np6 = multiply(ni0, ni1, edge_ids = torch.tensor([[2, 2, 1, 0, 1, 2, 0], [0, 5, 6, 3, 4, 2, 1]]).permute(1, 0))

            ns0 = summate(np0, edge_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 2, 4, 2, 1, 5, 6, 2, 1]]))
            ns1 = summate(np0, np6, edge_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5], [0, 2, 4, 2, 1, 5, 6, 2, 1, 10, 3, 8, 9]]))

        input_layer = InputLayer([ni0, ni1, ni2, ni3, ni4], cum_nodes = group_size)
        prod_layer = ProdLayer([np0, np1, np2, np3, np4, np5, np6], layer_sparsity_tol = 0.1, force_gpu_compilation = True)

        sum_layer_cpu = SumLayer([ns0, ns1], global_nid_start = input_layer.num_nodes + group_size, 
                                 param_ends = [1], layer_sparsity_tol = 0.1, disable_gpu_compilation = True)
        sum_layer_gpu = SumLayer([ns0, ns1], global_nid_start = input_layer.num_nodes + group_size, 
                                 param_ends = [1], layer_sparsity_tol = 0.1, force_gpu_compilation = True)

        for i in range(3):
            assert torch.all(sum_layer_cpu.partitioned_nids[i] == sum_layer_gpu.partitioned_nids[i])
            assert torch.all(sum_layer_cpu.partitioned_cids[i] == sum_layer_gpu.partitioned_cids[i])
            assert torch.all(sum_layer_cpu.partitioned_pids[i] == sum_layer_gpu.partitioned_pids[i])

        import pdb; pdb.set_trace()
        


if __name__ == "__main__":
    # prod_layer_compilation_test()
    sum_layer_compilation_test()
