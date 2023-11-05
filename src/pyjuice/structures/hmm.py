from __future__ import annotations

import torch
import networkx as nx
from typing import Type
from pyjuice.graph import *
from pyjuice.layer import *
from pyjuice.model import TensorCircuit
from pyjuice.nodes.distributions import Distribution, Categorical
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
