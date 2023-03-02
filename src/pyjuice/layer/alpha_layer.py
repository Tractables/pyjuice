from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List
from juice2.graph.region_graph import RegionGraph, InputRegionNode
from .layer import Layer


class AlphaLayer(Layer):
    def __init__(self, layer_id, region_nodes: List[RegionGraph], num_nodes: int) -> None:
        Layer.__init__(self, layer_id)

        self.region_nodes = region_nodes
        self.num_nodes = num_nodes
        self.num_ch_regions = len(self.region_nodes)

        self.alphas = None


        self.param_flows = None
        self.device = torch.device("cpu")

    def init_param_flows(self):
        if self.param_flows is None:
            self.param_flows = torch.zeros([self.param_flows_size], device = self.device)
        else:
            assert self.param_flows.size(0) == self.param_flows_size
            self.param_flows[:] = 0.0

        return None 