from __future__ import annotations

import torch
import numpy as np
from typing import Union, Sequence
from copy import deepcopy

from pyjuice.utils.context_manager import _DecoratorContextManager
from pyjuice.utils import BitSet
from .nodes import CircuitNodes
from .input_nodes import InputNodes
from .prod_nodes import ProdNodes
from .sum_nodes import SumNodes
from .distributions import Distribution

Tensor = Union[np.ndarray,torch.Tensor]
ProdNodesChs = Union[SumNodes,InputNodes]
SumNodesChs = Union[ProdNodes,InputNodes]


def inputs(var: Union[int,Sequence[int]], num_node_groups: int = 0, dist: Distribution = Distribution(), 
           params: Optional[Tensor] = None, num_nodes: int = 0, group_size: int = 0, **kwargs):

    if num_nodes > 0:
        assert num_node_groups == 0, "Only one of `num_nodes` and `num_node_groups` can be set at the same time."
        if group_size == 0:
            group_size = CircuitNodes.DEFAULT_GROUP_SIZE

        assert num_nodes % group_size == 0

        num_node_groups = num_nodes // group_size

    return InputNodes(
        num_node_groups = num_node_groups,
        scope = [var] if isinstance(var, int) else var,
        dist = dist,
        params = params,
        group_size = group_size,
        **kwargs
    )


def multiply(nodes1: ProdNodesChs, *args, edge_ids: Optional[Tensor] = None, **kwargs):

    assert isinstance(nodes1, SumNodes) or isinstance(nodes1, InputNodes), "Children of product nodes must be input or sum nodes." 

    chs = [nodes1]
    num_node_groups = nodes1.num_node_groups
    group_size = nodes1.group_size
    scope = deepcopy(nodes1.scope)

    for nodes in args:
        assert nodes.is_input() or nodes.is_sum(), f"Children of product nodes must be input or sum nodes, but found input of type {type(nodes)}."
        if edge_ids is None:
            assert nodes.num_node_groups == num_node_groups, f"Input nodes should have the same `num_node_groups`, but got {nodes.num_node_groups} and {num_node_groups}."
        assert nodes.group_size == group_size, "Input nodes should have the same `num_node_groups`."
        assert len(nodes.scope & scope) == 0, "Children of a `ProdNodes` should have disjoint scopes."
        chs.append(nodes)
        scope |= nodes.scope

    if edge_ids is not None:
        assert edge_ids.shape[0] == num_node_groups or edge_ids.shape[0] == num_node_groups * group_size

    return ProdNodes(num_node_groups, chs, edge_ids, group_size = group_size, **kwargs)


def summate(nodes1: SumNodesChs, *args, num_nodes: int = 0, num_node_groups: int = 0, edge_ids: Optional[Tensor] = None, group_size: int = 0, **kwargs):

    if num_nodes > 0:
        assert num_node_groups == 0, "Only one of `num_nodes` and `num_node_groups` can be set at the same time."
        if group_size == 0:
            group_size = CircuitNodes.DEFAULT_GROUP_SIZE

        assert num_nodes % group_size == 0

        num_node_groups = num_nodes // group_size

    assert isinstance(nodes1, ProdNodes) or isinstance(nodes1, InputNodes), f"Children of sum nodes must be input or product nodes, but found input of type {type(nodes1)}." 

    if edge_ids is not None and num_node_groups == 0:
        num_node_groups = edge_ids[0,:].max().item() + 1

    assert num_node_groups > 0, "Number of node groups should be greater than 0."

    chs = [nodes1]
    scope = deepcopy(nodes1.scope)
    for nodes in args:
        assert isinstance(nodes, ProdNodes) or isinstance(nodes, InputNodes), f"Children of sum nodes must be input or product nodes, but found input of type {type(nodes)}."
        assert nodes.scope == scope, "Children of a `SumNodes` should have the same scope."
        chs.append(nodes)

    return SumNodes(num_node_groups, chs, edge_ids, group_size = group_size, **kwargs)


class set_group_size(_DecoratorContextManager):
    def __init__(self, group_size: int = 1):

        self.group_size = group_size
        
        self.original_group_size = None

    def __enter__(self) -> None:
        self.original_group_size = CircuitNodes.DEFAULT_GROUP_SIZE
        CircuitNodes.DEFAULT_GROUP_SIZE = self.group_size

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        CircuitNodes.DEFAULT_GROUP_SIZE = self.original_group_size