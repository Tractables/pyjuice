from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List, Dict

from pyjuice.graph.region_graph import RegionGraph, InputRegionNode
from .layer import Layer

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


class InputLayer(Layer):
    def __init__(self, layer_id, region_nodes: List[RegionGraph], num_nodes: int) -> None:
        Layer.__init__(self, layer_id)

        for rnode in region_nodes:
            assert isinstance(rnode, InputRegionNode), "InputLayer must respect to InputRegionNode."

        self.region_nodes = region_nodes
        self.num_nodes = num_nodes
        self.num_ch_regions = len(self.region_nodes)

        self.param_flows = None

        self.device = torch.device("cpu")

    def init_param_flows(self, flows_memory: float = 0.0):
        if self.param_flows is None:
            self.param_flows = torch.zeros([self.param_flows_size], device = self.device)
        else:
            assert self.param_flows.size(0) == self.param_flows_size
            self.param_flows[:] *= flows_memory

        return None
