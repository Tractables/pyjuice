from __future__ import annotations

import torch
import networkx as nx
from typing import Type
from pyjuice.graph import *
from pyjuice.layer import *
from pyjuice.model import TensorCircuit
from pyjuice.nodes import multiply, summate, inputs, set_block_size, CircuitNodes
from pyjuice.nodes.distributions import Distribution, Categorical
from pyjuice.utils.util import max_cdf_power_of_2
from .compilation import BayesianTreeToHiddenRegionGraph


def HMM(length: int, num_latents: int, 
        input_layer_type: Type[Distribution] = Categorical, 
        input_layer_params: dict = {"num_cats": 256}):

    # The Graph is just a path x_1 => x_2 ..... => x_n
    N = length
    T = nx.Graph()
    for v in range(N):
        T.add_node(v)
        if v > 0:
            T.add_edge(v-1, v)

    root = 0
    root_r = BayesianTreeToHiddenRegionGraph(T, root, num_latents, input_layer_type, input_layer_params)
    
    return root_r


def HMM(seq_length: int, num_latents: int, num_emits: int, homogeneous: bool = True):
    """
    Constructs Hidden Markov Models.

    :param seq_length: sequence length
    :type seq_length: int

    :param num_latents: size of the latent space
    :type num_latents: int

    :param num_emits: size of the emission space
    :type num_emits: int

    :param homogeneous: whether to define a homogeneous (or inhomogeneous) HMM
    :type homogeneous: bool
    """
    
    group_size = min(max_cdf_power_of_2(num_latents), 1024)
    num_node_groups = num_latents // group_size
    
    with juice.set_group_size(group_size = group_size):

        ns_input = inputs(
            seq_length - 1, num_node_groups = num_node_groups,
            dist = Categorical(num_cats = num_emits)
        )
        
        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = homogeneous)
            
            if ns_sum is None:
                ns = summate(curr_zs, num_node_groups = num_node_groups)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = homogeneous)

            curr_zs = multiply(curr_xs, ns)
            
        ns = summate(curr_zs, num_node_groups = 1, group_size = 1)
    
    return ns
