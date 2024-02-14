from __future__ import annotations

import torch
import numpy as np
import random
import networkx as nx
from typing import Type, Optional

from pyjuice.nodes.distributions import *
from pyjuice.nodes import multiply, summate, inputs, set_block_size, CircuitNodes
from pyjuice.utils.util import max_cdf_power_of_2
from pyjuice.utils import BitSet


def RAT_SPN(num_vars: int, num_latents: int, depth: int, num_repetitions: int, num_pieces: int = 2,
            input_dists: Optional[Distribution] = None,
            input_layer_type: Type[Distribution] = Categorical, 
            input_layer_params: dict = {"num_cats": 256},
            block_size: Optional[int] = None):
    """
    Generate Random and Tensorized SPNs (https://proceedings.mlr.press/v115/peharz20a/peharz20a.pdf)

    :param num_vars: number of variables
    :type num_vars: int

    :param num_latents: size of the latent space
    :type num_latents: int

    :param depth: splitting depth of variable scopes
    :type depth: int

    :param num_repetitions: number of random splits
    :type num_repetitions: int

    :param num_pieces: the number of sub-scopes splitted from any scope
    :type num_pieces: int

    :param input_dist: input distribution
    :type input_dist: Distribution

    :param block_size: block size
    :type block_size: int
    """

    # Specify block size
    if block_size is None:
        block_size = min(256, max_cdf_power_of_2(num_latents))

    if input_dist is not None:
        input_node_type, input_node_params = input_dist._get_constructor()

    assert num_latents % block_size == 0, f"`num_latents` ({num_latents}) not divisible by `block_size` ({block_size})."
    num_node_blocks = num_latents // block_size

    with set_block_size(block_size):

        # Input nodes
        input_ns = []
        for v in range(num_vars):
            ns = inputs(v, num_node_blocks = num_node_blocks, dist = input_layer_type(**input_layer_params))
            input_ns.append(ns)

        # Top-down partition
        def partition_ns(scope, curr_depth = 0):
            if curr_depth >= depth or len(scope) < num_pieces:
                chs = [input_ns[v] for v in scope]
                np = multiply(*chs)
                ns = summate(np, num_node_blocks = num_node_blocks)

                return ns

            scope_list = scope.to_list()
            random.shuffle(scope_list)

            quotient, remainder = len(scope) // num_pieces, len(scope) % num_pieces
            list_lengths = [quotient + (i < remainder) for i in range(num_pieces)]

            ch_scopes = []
            sid = 0
            for i in range(num_pieces):
                eid = sid + list_lengths[i]
                ch_scopes.append(BitSet.from_array(scope_list[sid:eid]))
                sid = eid

            chs = [partition_ns(scope, curr_depth + 1) for scope in ch_scopes]
            np = multiply(*chs)
            ns = summate(np, num_node_blocks = num_node_blocks)

            return ns

        root_nps = []
        for rep_id in range(num_repetitions):
            root_ns = partition_ns(BitSet.from_array([v for v in range(num_vars)]))
            root_nps.append(root_ns.chs[0])

        root_ns = summate(*root_nps, num_node_blocks = 1, block_size = 1)

    return root_ns
    