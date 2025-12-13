from __future__ import annotations

import torch
import networkx as nx
from typing import Type, Optional
from pyjuice.graph import *
from pyjuice.layer import *
from pyjuice.model import TensorCircuit
from pyjuice.nodes import multiply, summate, inputs, set_block_size, CircuitNodes
from pyjuice.nodes.distributions import Distribution, Categorical
from pyjuice.utils.util import max_cdf_power_of_2
from .compilation import BayesianTreeToHiddenRegionGraph


def HMM(seq_length: int, num_latents: int, num_emits: int, homogeneous: bool = True, block_size: Optional[int] = None,
        alpha: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None, gamma: Optional[torch.Tensor] = None,
        sum_edge_ids_constructor: Optional[Callable] = None):
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

    :param block_size: block size of the PC
    :type block_size: Optional[int]

    :param alpha: optional transition parameters of size `[num_latents, num_latents]`
    :type alpha: Optional[torch.Tensor]

    :param beta: optional emission parameters of size `[num_latents, num_emits]`
    :type beta: Optional[torch.Tensor]

    :param gamma: optional init parameters of size `[num_latents]`
    :type gamma: Optional[torch.Tensor]

    :param sum_edge_ids_constructor: optional helper functions to create special edge patterns (e.g., block-sparse)
    :type sum_edge_ids_constructor: Callable
    """
    
    if block_size is None:
        block_size = min(max_cdf_power_of_2(num_latents), 1024)
    num_node_blocks = num_latents // block_size
    
    with set_block_size(block_size = block_size):

        ns_input = inputs(
            seq_length - 1, num_node_blocks = num_node_blocks,
            dist = Categorical(num_cats = num_emits)
        )

        if beta is not None:
            assert beta.size(0) == num_latents and beta.size(1) == num_emits
            ns_input.set_params(beta)
        
        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = homogeneous)
            
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks = num_node_blocks, sum_edge_ids_constructor = sum_edge_ids_constructor)
                if alpha is not None:
                    assert alpha.size(0) == num_latents and alpha.size(1) == num_latents
                    ns.set_params(alpha)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = homogeneous)

            curr_zs = multiply(curr_xs, ns)
            
        ns = summate(curr_zs, num_node_blocks = 1, block_size = 1, sum_edge_ids_constructor = sum_edge_ids_constructor)

        if gamma is not None:
            assert gamma.dim() == 1 and gamma.size(0) == num_latents
            ns.set_params(gamma.unsqueeze(0))
    
    return ns


def GeneralizedHMM(seq_length: int, num_latents: int, homogeneous: bool = True, block_size: Optional[int] = None,
                   alpha: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None, gamma: Optional[torch.Tensor] = None,
                   input_dist: Optional[Distribution] = None, input_node_type: Type[Distribution] = Categorical, 
                   input_node_params: dict = {}, sum_edge_ids_constructor: Optional[Callable] = None):
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

    :param block_size: block size of the PC
    :type block_size: Optional[int]

    :param alpha: optional transition parameters of size `[num_latents, num_latents]`
    :type alpha: Optional[torch.Tensor]

    :param beta: optional emission parameters of size `[num_latents, num_emits]`
    :type beta: Optional[torch.Tensor]

    :param gamma: optional init parameters of size `[num_latents]`
    :type gamma: Optional[torch.Tensor]

    :param sum_edge_ids_constructor: optional helper functions to create special edge patterns (e.g., block-sparse)
    :type sum_edge_ids_constructor: Callable
    """

    if input_dist is not None:
        input_node_type, input_node_params = input_dist._get_constructor()

    def construct_dist():
        return input_node_type(**input_node_params)
    
    if block_size is None:
        block_size = min(max_cdf_power_of_2(num_latents), 1024)
    num_node_blocks = num_latents // block_size
    
    with set_block_size(block_size = block_size):

        ns_input = inputs(
            seq_length - 1, num_node_blocks = num_node_blocks,
            dist = construct_dist()
        )

        if beta is not None:
            ns_input.set_params(beta)
        
        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = homogeneous)
            
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks = num_node_blocks, sum_edge_ids_constructor = sum_edge_ids_constructor)
                if alpha is not None:
                    assert alpha.size(0) == num_latents and alpha.size(1) == num_latents
                    ns.set_params(alpha)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = homogeneous)

            curr_zs = multiply(curr_xs, ns)
            
        ns = summate(curr_zs, num_node_blocks = 1, block_size = 1, sum_edge_ids_constructor = sum_edge_ids_constructor)

        if gamma is not None:
            assert gamma.dim() == 1 and gamma.size(0) == num_latents
            ns.set_params(gamma.unsqueeze(0))
    
    return ns
