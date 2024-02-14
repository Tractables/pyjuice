from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type, Optional, Dict
from copy import deepcopy

from pyjuice.graph import InputRegionNode
from .distributions import Distribution
from .nodes import CircuitNodes


class InputNodes(CircuitNodes):
    """
    A class representing vectors of input nodes. It is created by `pyjuice.inputs`.

    :param num_node_blocks: number of node blocks
    :type num_node_blocks: int

    :param scope: variable scope (set of variables)
    :type scope: Union[Sequence,BitSet]

    :param dist: input distribution
    :type dist: Distribution

    :param params: parameters of the vector of nodes
    :type params: Optional[Tensor]

    :param block_size: block size
    :type block_size: int
    """
    def __init__(self, num_node_blocks: int, scope: Union[Sequence,BitSet], dist: Distribution, 
                 params: Optional[torch.Tensor] = None, block_size: int = 0, 
                 _no_set_meta_params: bool = False, **kwargs) -> None:

        rg_node = InputRegionNode(scope)
        super(InputNodes, self).__init__(num_node_blocks, rg_node, block_size = block_size, **kwargs)

        self.chs = [] # InputNodes has no children

        self.dist = dist

        # Init parameters and meta-parameters
        if not _no_set_meta_params and self.dist.need_meta_parameters:
            self.set_meta_params(**kwargs)
        if params is not None:
            self.set_params(params)

        # Callbacks
        self._run_init_callbacks(**kwargs)

        # Parameter initialization flag
        self._param_initialized = False

    @property
    def num_edges(self):
        return 0

    def duplicate(self, scope: Optional[Union[int,Sequence,BitSet]] = None, tie_params: bool = False) -> InputNodes:
        """
        Create a duplication of the current node with the same specification (i.e., number of nodes, block size, distribution).

        :param scope: variable scope of the duplication
        :type scope: Optional[Union[int,Sequence,BitSet]]

        :param tie_params: whether to tie the parameters of the current node and the duplicated node
        :type tie_params: bool

        :returns: a duplicated `InputNodes`
        """
        if scope is None:
            scope = self.scope
        else:
            if isinstance(scope, int):
                scope = [scope]

            assert len(scope) == len(self.scope)

        dist = deepcopy(self.dist)

        ns = InputNodes(self.num_node_blocks, scope = scope, dist = dist, block_size = self.block_size, source_node = self if tie_params else None)

        if hasattr(self, "_params") and self._params is not None and not tie_params:
            ns._params = self._params.clone()

        return ns

    def get_params(self) -> torch.Tensor:
        """
        Get the input node parameters.
        """
        if not self.provided("_params"):
            return None
        else:
            return self._params

    def set_params(self, params: Union[torch.Tensor,Dict], normalize: bool = True):
        """
        Set the input node parameters.

        :param params: parameters to be set
        :type params: Union[torch.Tensor,Dict]

        :param normalize: whether to normalize the parameters
        :type normalize: bool
        """
        assert params.numel() == self.num_nodes * self.dist.num_parameters()

        params = params.reshape(-1)
        if normalize:
            params = self.dist.normalize_parameters(params)

        self._param_initialized = True
        self._params = params

    def set_meta_params(self, **kwargs):
        """
        Set the meta-parameters such as the mask of input nodes with the `MaskedCategorical` distribution.
        """
        params = self.dist.set_meta_parameters(self.num_nodes, **kwargs)

        self._param_initialized = False
        self._params = params

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, 
                        is_root: bool = True, ret_params: bool = False, **kwargs) -> None:
        """
        Randomly initialize node parameters.

        :param perturbation: "amount of perturbation" added to the parameters (should be greater than 0)
        :type perturbation: float

        :param recursive: whether to recursively apply the function to child nodes
        :type recursive: bool
        """
        if not self.is_tied() and not self.has_params():
            self._params = self.dist.init_parameters(
                num_nodes = self.num_nodes,
                perturbation = perturbation,
                params = self.get_params(),
                **kwargs
            )

            if ret_params:
                return self._params

        elif self.is_tied() and ret_params:
            return self.get_source_ns().init_parameters(
                perturbation = perturbation,
                recursive = False,
                is_root = True,
                ret_params = True,
                **kwargs
            )

    def __repr__(self):
        return f"InputNodes(num_node_blocks={self.num_node_blocks}, block_size={self.block_size}, dist={type(self.dist)})"
