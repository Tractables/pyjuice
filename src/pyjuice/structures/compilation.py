from __future__ import annotations

import torch
import networkx as nx
from typing import Type, Optional

from pyjuice.nodes import multiply, summate, inputs, set_group_size
from pyjuice.nodes.distributions import Distribution
from pyjuice.utils.util import max_cdf_power_of_2


def BayesianTreeToHiddenRegionGraph(tree: nx.Graph, 
                                    root,
                                    num_latents: int,  
                                    InputDist: Type[Distribution], 
                                    dist_params: dict,
                                    num_root_ns: int = 1,
                                    group_size: Optional[int] = None) -> RegionGraph:
    """
    Given a Tree Bayesian Network tree T1 (i.e. at most one parents), 
    
        1. Bayesian Network T1 becomes `tree` rooted at `root`
        2. Construct Bayesian bayesian network T2:
            - T2 = copy of T1
            - In T2, convert each variable x_i to a categoriacal latent variable z_i with `num_latents` categories
            - In T2, Adds back x_i nodes, with one edge x_i -> z_i 

        3. Returns a RegionGraph of a probabilistics circuit that is equivalent to T2

    For example, if T1 = "x1 -> x2" , then T2  becomes  x1    x2 
                                                        ^     ^
                                                        |     |
                                                        z1 -> z2   
    """

    # Specify group size
    if group_size is None:
        group_size = min(64, max_cdf_power_of_2(num_latents))

    num_node_groups = num_latents // group_size

    # Root the tree at `root`
    clt = nx.bfs_tree(tree, root)
    def children(n: int):
        return [c for c in clt.successors(n)]
    
    # Assert at most one parent
    for n in clt.nodes:
        assert len(list(clt.predecessors(n))) <= 1

    # Compile the region graph for the circuit equivalent to T2
    node_seq = list(nx.dfs_postorder_nodes(tree, root))
    var2rnode = dict()
    with set_group_size(group_size):
        for v in node_seq:
            chs = children(v)

            if len(chs) == 0:
                # Input Region
                r = inputs(v, num_node_groups = num_node_groups, dist = InputDist(**dist_params))
                var2rnode[v] = r
            else:
                # Inner Region
                
                # children(z_v)
                ch_regions = [var2rnode[c] for c in chs]

                # Add x_v to children(z_v)
                leaf_r = inputs(v, num_node_groups = num_node_groups, dist = InputDist(**dist_params))
                ch_regions.append(leaf_r)

                rp = multiply(*ch_regions)

                if v == root:
                    if group_size == 1:
                        r = summate(rp, num_node_groups = num_root_ns, group_size = 1)
                    else:
                        r = summate(rp, num_node_groups = num_root_ns // group_size, group_size = group_size)
                else:
                    r = summate(rp, num_node_groups = num_node_groups)

                var2rnode[v] = r

    root_r = var2rnode[root]
    return root_r