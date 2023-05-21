from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type
from copy import deepcopy

from pyjuice.graph import InputRegionNode
from .distributions import Distribution
from .nodes import CircuitNodes


class InputNodes(CircuitNodes):
    def __init__(self, num_nodes: int, scope: Union[Sequence,BitSet], dist: Distribution, **kwargs) -> None:

        rg_node = InputRegionNode(scope)
        super(InputNodes, self).__init__(num_nodes, rg_node, **kwargs)

        self.chs = [] # InputNodes has no children

        self.dist = dist

        # Callbacks
        self._run_init_callbacks(**kwargs)

    def duplicate(self, scope: Optional[Union[int,Sequence,BitSet]] = None, tie_params: bool = True):
        if scope is None:
            scope = self.scope
        else:
            if isinstance(scope, int):
                scope = [scope]

            assert len(scope) == len(self.scope)

        dist = deepcopy(self.dist)

        ns = InputNodes(self.num_nodes, scope = scope, dist = dist, source_node = self if tie_params else None)

        if hasattr(self, "_params") and self._params is not None:
            ns._params = self._params.clone()

        return ns

    def get_params(self):
        if self._params is None:
            return None
        else:
            return self.dist.raw2processed_params(self._params)

    def set_params(self, params):
        params = self.dist.processed2raw_params(params)
        self._params = params

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, **kwargs):
        self._params = self.dist.init_parameters(
            num_nodes = self.num_nodes,
            perturbation = perturbation,
            **kwargs
        )