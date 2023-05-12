from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Optional
from copy import deepcopy
from pyjuice.utils import BitSet
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode


class CircuitNodes():

    # A list of function that will be called at the end of `__init__`
    # This should only be changed by context managers, so please do not 
    # add anything here.
    INIT_CALLBACKS = []

    def __init__(self, num_nodes: int, region_node: RegionGraph, source_node: Optional[CircuitNodes] = None, **kwargs):
        self.num_nodes = num_nodes
        self.region_node = region_node

        self.chs = []
        
        self._output_ind_range = None
        self._param_ids = None
        self._params = None

        # Source nodes it points to (for parameter tying)
        if source_node is not None:
            while source_node._source_node is not None:
                source_node = source_node._source_node
        self._source_node = source_node

        self._tied_param_group_ids = None

    def _run_init_callbacks(self, **kwargs):
        for func in self.INIT_CALLBACKS:
            func(self, **kwargs)

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

    def duplicate(self, *args, **kwargs):
        raise ValueError(f"{type(self)} does not support `duplicate`.")