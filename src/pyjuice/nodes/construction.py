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


def inputs(var: Union[int,Sequence[int]], num_node_blocks: int = 0, dist: Distribution = Distribution(), 
           params: Optional[Tensor] = None, num_nodes: int = 0, block_size: int = 0, **kwargs) -> InputNodes:
    """
    Construct a vector of input nodes defined on the same variable and have the same input distribution.

    :param var: the variable ID (or a list of variable IDs if the parameter `dist` is a multivariate distribution)
    :type var: Union[int,Sequence[int]]

    :param num_node_blocks: number of node blocks each of size `block_size`. I.e., the total number of nodes is `num_node_blocks` * `block_size`
    :type num_node_blocks: int

    :param dist: the input distribution from `pyjuice.nodes.distributions`
    :type dist: Distribution

    :param params: an optional tensor that defines the parameters of the nodes
    :type params: Optional[Tensor]

    :param num_nodes: an alternative of `num_node_blocks`; if set, it should be divicible by `block_size`
    :type num_nodes: int

    :param block_size: block size of the nodes; it does not change the semantics of the nodes, but will affect the speed of the compiled PC
    :type block_size: int

    :returns: an `InputNodes` object (a subclass of `CircuitNodes`)
    """

    assert block_size == 0 or block_size & (block_size - 1) == 0, "`block_size` must be a power of 2."

    if num_nodes > 0:
        assert num_node_blocks == 0, "Only one of `num_nodes` and `num_node_blocks` can be set at the same time."
        if block_size == 0:
            block_size = CircuitNodes.DEFAULT_BLOCK_SIZE

        assert num_nodes % block_size == 0

        num_node_blocks = num_nodes // block_size

    return InputNodes(
        num_node_blocks = num_node_blocks,
        scope = [var] if isinstance(var, int) else var,
        dist = dist,
        params = params,
        block_size = block_size,
        **kwargs
    )


def multiply(nodes1: ProdNodesChs, *args, edge_ids: Optional[Tensor] = None, sparse_edges: bool = False, **kwargs) -> ProdNodes:
    """
    Construct a vector of product nodes given a list of children PCs defined on disjoint sets of variables.

    :note: It requires all children to have the same block size.

    :note: If all children have the same `num_node_blocks`, if `edge_ids` is not specified, `multiply` outputs a vector of `num_node_blocks` * `block_size` nodes, where the ith (product) node is connected to the ith node in every child.

    :note: By default every edge specified by `edge_ids` denotes a block of `block_size` edges, unless `sparse_edge` is set to `True`, where the block size is assumed to be 1.

    :param nodes1: the first child node
    :type nodes1: Union[SumNodes,InputNodes]

    :param args: the remaining child nodes
    :type args: Union[SumNodes,InputNodes]

    :param edge_ids: a matrix of size [# product node blocks, # children] - the ith product node block is connected to the `edge_ids[i,j]`th node block in the jth child
    :type edge_ids: Optional[Tensor]

    :param sparse_edges: if set to `True`, the size of `edge_ids` becomes [# product nodes, # children] (i.e., we assume the block size of every child is 1)
    :type sparse_edges: bool

    :returns: an `ProdNodes` object (a subclass of `CircuitNodes`)
    """

    assert isinstance(nodes1, SumNodes) or isinstance(nodes1, InputNodes), "Children of product nodes must be input or sum nodes." 

    chs = [nodes1]
    num_node_blocks = nodes1.num_node_blocks
    block_size = nodes1.block_size
    scope = deepcopy(nodes1.scope)

    for nodes in args:
        assert nodes.is_input() or nodes.is_sum(), f"Children of product nodes must be input or sum nodes, but found input of type {type(nodes)}."
        if edge_ids is None:
            assert nodes.num_node_blocks == num_node_blocks, f"Input nodes should have the same `num_node_blocks`, but got {nodes.num_node_blocks} and {num_node_blocks}."
        assert nodes.block_size == block_size, "Input nodes should have the same `num_node_blocks`."
        assert len(nodes.scope & scope) == 0, "Children of a `ProdNodes` should have disjoint scopes."
        chs.append(nodes)
        scope |= nodes.scope

    if edge_ids is not None:
        if sparse_edges:
            assert edge_ids.shape[0] % block_size == 0
            num_node_blocks = edge_ids.shape[0] // block_size
        else:
            num_node_blocks = edge_ids.shape[0]
        assert edge_ids.shape[0] == num_node_blocks or edge_ids.shape[0] == num_node_blocks * block_size

    return ProdNodes(num_node_blocks, chs, edge_ids, block_size = block_size, **kwargs)


def summate(nodes1: SumNodesChs, *args, num_node_blocks: int = 0, num_nodes: int = 0,
            edge_ids: Optional[Tensor] = None, block_size: int = 0, **kwargs) -> SumNodes:
    """
    Construct a vector of sum nodes given a list of children PCs defined on the same sets of variables.

    :note: It requires all children to have the same block size.

    :note: If `edge_ids` is not set, `summate` defines a set of sum nodes fully connected to the input nodes.

    :param nodes1: the first child node
    :type nodes1: Union[ProdNodes,InputNodes]

    :param args: the remaining child nodes
    :type args: Union[ProdNodes,InputNodes]

    :param num_node_blocks: number of node blocks each of size `block_size`. I.e., the total number of nodes is `num_node_blocks` * `block_size`
    :type num_node_blocks: int

    :param num_nodes: an alternative of `num_node_blocks`; if set, it should be divicible by `block_size`
    :type num_nodes: int

    :param edge_ids: a matrix of size [2, # edges] - every size-2 column vector [i,j] defines a set of edges that fully connect the ith sum node block and the jth child node block
    :type edge_ids: Optional[Tensor]

    :param block_size: block size of the nodes; it does not change the semantics of the nodes, but will affect the speed of the compiled PC
    :type block_size: int

    :returns: an `SumNodes` object (a subclass of `CircuitNodes`)
    """

    assert block_size == 0 or block_size & (block_size - 1) == 0, "`block_size` must be a power of 2."

    if num_nodes > 0:
        assert num_node_blocks == 0, "Only one of `num_nodes` and `num_node_blocks` can be set at the same time."
        if block_size == 0:
            block_size = CircuitNodes.DEFAULT_BLOCK_SIZE

        assert num_nodes % block_size == 0

        num_node_blocks = num_nodes // block_size

    assert isinstance(nodes1, ProdNodes) or isinstance(nodes1, InputNodes), f"Children of sum nodes must be input or product nodes, but found input of type {type(nodes1)}." 

    if edge_ids is not None and num_node_blocks == 0:
        num_node_blocks = edge_ids[0,:].max().item() + 1

    assert num_node_blocks > 0, "Number of node blocks should be greater than 0."

    chs = [nodes1]
    scope = deepcopy(nodes1.scope)
    for nodes in args:
        assert isinstance(nodes, ProdNodes) or isinstance(nodes, InputNodes), f"Children of sum nodes must be input or product nodes, but found input of type {type(nodes)}."
        assert nodes.scope == scope, "Children of a `SumNodes` should have the same scope."
        chs.append(nodes)

    return SumNodes(num_node_blocks, chs, edge_ids, block_size = block_size, **kwargs)


class set_block_size(_DecoratorContextManager):
    """
    Context-manager that sets the block size of PC nodes constructed by `inputs`, `multiply`, and `summate`.

    :param block_size: the target block size
    :type block_size: int

    Example::
        >>> with pyjuice.set_block_size(16):
        ...     nis = pyjuice.inputs(var = 0, num_node_blocks = 4, dist = Categorical(num_cats = 20))
        >>> print(nis.num_nodes) # Should be 16 * 4
        64
    """

    def __init__(self, block_size: int = 1):

        assert block_size & (block_size - 1) == 0, "`block_size` must be a power of 2."

        self.block_size = block_size
        
        self.original_block_size = None

    def __enter__(self) -> None:
        self.original_block_size = CircuitNodes.DEFAULT_BLOCK_SIZE
        CircuitNodes.DEFAULT_BLOCK_SIZE = self.block_size

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        CircuitNodes.DEFAULT_BLOCK_SIZE = self.original_block_size