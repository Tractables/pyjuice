from __future__ import annotations

import torch
import numpy as np
from typing import Union, Sequence
from copy import deepcopy

from pyjuice.utils import BitSet
from .nodes import CircuitNodes
from .input_nodes import InputNodes
from .prod_nodes import ProdNodes
from .sum_nodes import SumNodes
from .distributions import Distribution

Tensor = Union[np.ndarray,torch.Tensor]
ProdNodesChs = Union[SumNodes,InputNodes]
SumNodesChs = Union[ProdNodes,InputNodes]


def inputs(var: Union[int,Sequence[int]], num_nodes: int, dist: Distribution):
    return InputNodes(
        num_nodes = num_nodes,
        scope = [var] if isinstance(var, int) else var,
        dist = dist
    )


def multiply(nodes1: ProdNodesChs, nodes2: ProdNodesChs, *args, 
             edge_ids: Optional[Tensor] = None):

    chs = [nodes1, nodes2]
    num_nodes = nodes1.num_nodes
    scope = deepcopy(nodes1.scope)
    if edge_ids is None:
        assert nodes2.num_nodes == num_nodes, "Input nodes should have the same `num_nodes`."
    assert len(nodes2.scope & scope) == 0, "Children of a `ProdNodes` should have disjoint scopes."
    scope |= nodes2.scope
    for nodes in args:
        if edge_ids is None:
            assert nodes.num_nodes == num_nodes, "Input nodes should have the same `num_nodes`."
        assert len(nodes.scope & scope) == 0, "Children of a `ProdNodes` should have disjoint scopes."
        chs.append(nodes)
        scope |= nodes.scope

    if edge_ids is not None:
        num_nodes = edge_ids.shape[0]

    return ProdNodes(num_nodes, chs, edge_ids)


def summate(nodes1: SumNodesChs, *args, num_nodes: int = 0, edge_ids: Optional[Tensor] = None):

    if edge_ids is not None and num_nodes == 0:
        num_nodes = edge_ids[0,:].max().item() + 1

    assert num_nodes > 0, "Number of nodes should be greater than 0."

    chs = [nodes1]
    scope = deepcopy(nodes1.scope)
    for nodes in args:
        assert nodes.scope == scope, "Children of a `SumNodes` should have the same scope."
        chs.append(nodes)

    return SumNodes(num_nodes, chs, edge_ids)