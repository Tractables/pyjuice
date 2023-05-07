from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Optional
from copy import deepcopy
from pyjuice.utils import BitSet
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode


class CircuitNodes():
    def __init__(self, num_nodes: int, region_node: RegionGraph):
        self.num_nodes = num_nodes
        self.region_node = region_node

        self.chs = []
        
        self._output_ind_range = None
        self._param_ids = None

    @property
    def scope(self):
        return self.region_node.scope

    def issum(self):
        return isinstance(self.region_node, InnerRegionNode)

    def isprod(self):
        return isinstance(self.region_node, PartitionNode)

    def isinput(self):
        return isinstance(self.region_node, InputRegionNode)

    @property
    def num_child_regions(self):
        return len(self.chs)