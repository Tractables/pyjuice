
from __future__ import annotations

import numpy as np
import torch
from typing import List, Union, Type
from copy import deepcopy
from pyjuice.utils import BitSet


class RegionGraph():
    def __init__(self, scope: BitSet, children: List[RegionGraph]) -> None:
        self.scope = scope
        self.children = children
        self.num_chs = len(children)

        self.parents = []
        for region_node in self.children:
            region_node.parents.append(self)

    
class PartitionNode(RegionGraph):
    def __init__(self, children: List[Union[InnerRegionNode,InputRegionNode]]) -> None:

        assert len(children) > 0, "PartitionNode receives no child."

        scope = BitSet()
        for n in children:
            assert len(scope & n.scope) == 0, "Children of a PartitionNode have overlapping scopes."
            scope |= n.scope

        super().__init__(scope, children)

    def __hash__(self):
        ch_scopes = tuple(hash(c.scope) for c in self.children)
        return hash(("PartitionNode", hash(self.scope), ch_scopes))


class InnerRegionNode(RegionGraph):
    def __init__(self, children: List[Union[InputRegionNode,PartitionNode]]) -> None:

        assert len(children) > 0, "InnerRegionNode receives no child."

        scope = deepcopy(children[0].scope)
        for n in children[1:]:
            assert scope == n.scope, "Children of an InnerRegionNode must have the same scope."

        super().__init__(scope, children)

    def __hash__(self):
        return hash(("InnerRegionNode", hash(self.scope)))


class InputRegionNode(RegionGraph):
    def __init__(self, scope: Union[List,BitSet]) -> None:
        if isinstance(scope, List):
            scope = BitSet.from_array(scope)

        super().__init__(scope, [])

    def __hash__(self):
        return hash(("InputRegionNode", hash(self.scope)))
