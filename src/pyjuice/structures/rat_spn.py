from __future__ import annotations

import torch
import numpy as np
import random
import networkx as nx
from typing import Type, Optional

from pyjuice.nodes.distributions import *
from pyjuice.nodes import multiply, summate, inputs, set_group_size, CircuitNodes
from pyjuice.utils.util import max_cdf_power_of_2
from pyjuice.utils import BitSet


def RAT_SPN(num_vars: int, num_latents: int, depth: int, num_repetitions: int, num_pieces: int = 2,
            group_size: Optional[int] = None,
            input_layer_type: Type[Distribution] = Categorical, 
            input_layer_params: dict = {"num_cats": 256}):

    # Specify group size
    if group_size is None:
        group_size = min(256, max_cdf_power_of_2(num_latents))

    assert num_latents % group_size == 0, f"`num_latents` ({num_latents}) not divisible by `group_size` ({group_size})."
    num_node_groups = num_latents // group_size

    with set_group_size(group_size):

        # Input nodes
        input_ns = []
        for v in range(num_vars):
            ns = inputs(v, num_node_groups = num_node_groups, dist = input_layer_type(**input_layer_params))
            input_ns.append(ns)

        # Top-down partition
        def partition_ns(scope, curr_depth = 0):
            if curr_depth >= depth or len(scope) < num_pieces:
                chs = [input_ns[v] for v in scope]
                np = multiply(*chs)
                ns = summate(np, num_node_groups = num_node_groups)

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
            ns = summate(np, num_node_groups = num_node_groups)

            return ns

        root_nps = []
        for rep_id in range(num_repetitions):
            root_ns = partition_ns(BitSet.from_array([v for v in range(num_vars)]))
            root_nps.append(root_ns.chs[0])

        root_ns = summate(*root_nps, num_node_groups = 1, group_size = 1)

    return root_ns
    