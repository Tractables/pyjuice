from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Optional, Callable
from copy import deepcopy
from collections import deque

from pyjuice.utils import BitSet
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode


def node_iterator(root_ns: CircuitNodes, reverse: bool = False):
    def dfs(ns: CircuitNodes, fn: Callable, visited: set = set()):
        if ns in visited:
            return

        visited.add(ns)

        # Recursively traverse children
        if ns.is_sum() or ns.is_prod():
            for cs in ns.chs:
                dfs(cs, fn = fn, visited = visited)

        fn(ns)

    if not reverse:
        visited = set()
        node_list = list()

        def record_fn(ns):
            node_list.append(ns)

        dfs(root_ns, record_fn)

        for ns in node_list:
            yield ns
    
    else:
        parcount = dict()
        node_list = list()

        def inc_parcount(ns):
            for cs in ns.chs:
                if cs not in parcount:
                    parcount[cs] = 0
                parcount[cs] += 1

        dfs(root_ns, inc_parcount)

        queue = deque()
        queue.append(root_ns)
        while len(queue) > 0:
            ns = queue.popleft()
            node_list.append(ns)
            for cs in ns.chs:
                parcount[cs] -= 1
                if parcount[cs] == 0:
                    queue.append(cs)

        assert len(parcount) + 1 == len(node_list)

        for ns in node_list:
            yield ns


class CircuitNodes():

    # A list of function that will be called at the end of `__init__`
    # This should only be changed by context managers, so please do not 
    # add anything here.
    INIT_CALLBACKS = []

    # Default `block_size`. Used by the context managers.
    DEFAULT_BLOCK_SIZE = 1

    def __init__(self, num_node_blocks: int, region_node: RegionGraph, block_size: int = 0, source_node: Optional[CircuitNodes] = None, **kwargs):

        if block_size == 0:
            block_size = self.DEFAULT_BLOCK_SIZE

        assert num_node_blocks > 0
        assert block_size > 0 and (block_size & (block_size - 1)) == 0, f"`block_size` must be a power of 2, but got `block_size={block_size}`."
        
        self.num_node_blocks = num_node_blocks
        self.block_size = block_size
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

        self._tied_param_block_ids = None

        self._reverse_iter = False

    def _run_init_callbacks(self, **kwargs):
        for func in self.INIT_CALLBACKS:
            func(self, **kwargs)

    @property
    def scope(self):
        return self.region_node.scope

    def is_sum(self):
        return isinstance(self.region_node, InnerRegionNode)

    def is_prod(self):
        return isinstance(self.region_node, PartitionNode)

    def is_input(self):
        return isinstance(self.region_node, InputRegionNode)

    @property
    def num_chs(self):
        return len(self.chs)

    @property
    def num_nodes(self):
        """
        Number of PC nodes within the current node.
        """
        return self.num_node_blocks * self.block_size

    @property
    def num_edges(self):
        raise NotImplementedError()

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
        return self._source_node if self.is_tied() else self

    def set_source_ns(self, source_ns: CircuitNodes):
        assert type(source_ns) == type(self), f"Node type of the source ns ({type(source_ns)}) does not match that of self ({type(self)})."
        assert len(source_ns.chs) == len(self.chs), "Number of children does not match."
        assert not hasattr(self, "_params") or self._params is None, "The current node should not have parameters to avoid confusion."
        assert source_ns.num_node_blocks == self.num_node_blocks, "`num_node_blocks` does not match."
        assert source_ns.block_size == self.block_size,  "`block_size` does not match."

        self._source_node = source_ns

    def has_params(self):
        if self.is_input():
            return self._param_initialized

        if not self.is_tied():
            return hasattr(self, "_params") and self._params is not None
        else:
            source_ns = self.get_source_ns()
            return hasattr(source_ns, "_params") and source_ns._params is not None

    def _clear_tensor_circuit_hooks(self, recursive: bool = True):

        def clear_hooks(ns):
            if hasattr(ns, "_param_range"):
                ns._param_range = None
            if hasattr(ns, "_param_ids"):
                ns._param_ids = None
            if hasattr(ns, "_inverse_param_ids"):
                ns._inverse_param_ids = None
            if hasattr(ns, "_param_flow_range"):
                ns._param_flow_range = None
            if hasattr(ns, "_output_ind_range"):
                ns._output_ind_range = None

        if recursive:
            for ns in self:
                clear_hooks(ns)
        else:
            clear_hooks(self)

    def __iter__(self):
        return node_iterator(self, self._reverse_iter)

    def __call__(self, reverse: bool = False):
        self._reverse_iter = reverse

        return self

    def provided(self, var_name):
        return hasattr(self, var_name) and getattr(self, var_name) is not None
