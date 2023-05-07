from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type
from copy import deepcopy

from pyjuice.graph import InputRegionNode
from .distributions import Distribution
from .nodes import CircuitNodes


class InputNodes(CircuitNodes):
    def __init__(self, num_nodes: int, scope: Union[Sequence,BitSet], dist: Distribution) -> None:

        rg_node = InputRegionNode(scope)
        super(InputNodes, self).__init__(num_nodes, rg_node)

        self.chs = [] # InputNodes has no children

        self.dist = dist

