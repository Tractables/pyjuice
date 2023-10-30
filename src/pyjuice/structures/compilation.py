from __future__ import annotations

import torch
import networkx as nx
from typing import Type

from pyjuice.nodes import multiply, summate, inputs
from pyjuice.nodes.distributions import Distribution


def BayesianTreeToHiddenRegionGraph(tree: nx.Graph, 
                                    root,
                                    num_latents: int,  
                                    InputDist: Type[Distribution], 
                                    dist_params: dict,
                                    num_root_ns: int = 1) -> RegionGraph:
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
    for v in node_seq:
        chs = children(v)

        if len(chs) == 0:             
            # Input Region
            r = inputs(v, num_nodes = num_latents, dist = InputDist(**dist_params))
            var2rnode[v] = r
        else:
            # Inner Region
            
            # children(z_v)
            ch_regions = [var2rnode[c] for c in chs]

            # Add x_v to children(z_v)
            leaf_r = inputs(v, num_nodes = num_latents, dist = InputDist(**dist_params))
            ch_regions.append(leaf_r)

            rp = multiply(*ch_regions)

            if v == root:
                r = summate(rp, num_nodes = num_root_ns)
            else:
                r = summate(rp, num_nodes = num_latents)

            var2rnode[v] = r

    root_r = var2rnode[root]
    return root_r