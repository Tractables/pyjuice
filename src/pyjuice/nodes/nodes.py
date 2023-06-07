from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Optional
from copy import deepcopy
from pyjuice.utils import BitSet
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode


def node_iterator(root_ns: CircuitNodes):
    visited = set()
    node_list = list()

    def dfs(ns: CircuitNodes):
        if ns in visited:
            return

        visited.add(ns)

        # Recursively traverse children
        if ns.issum() or ns.isprod():
            for cs in ns.chs:
                dfs(cs)

        node_list.append(ns)

    dfs(root_ns)

    for ns in node_list:
        yield ns


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

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, visited: set = set(), 
                        is_root = True, **kwargs):
        if recursive:
            if self in visited:
                return None
            
            visited.add(self)

            for cs in self.chs:
                cs.init_parameters(
                    perturbation = perturbation, 
                    recursive = recursive, 
                    visited = visited, 
                    is_root = False,
                    **kwargs
                )
        else:
            visited.add(self)

        # Process all tied nodes
        for ns in visited:
            if ns._source_node is not None:
                # Do not store parameters explicitly for tied nodes
                # We can always retrieve them from the source nodes when required
                ns._params = None

    def is_tied(self):
        return self._source_node is not None

    def get_source_ns(self):
        return self._source_node

    def has_params(self):
        if not self.is_tied():
            return hasattr(self, "_params") and self._params is not None
        else:
            source_ns = self.get_source_ns()
            return hasattr(source_ns, "_params") and source_ns._params is not None

    def __iter__(self):
        return node_iterator(self)