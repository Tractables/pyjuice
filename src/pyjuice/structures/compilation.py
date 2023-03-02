from __future__ import annotations

import torch
import networkx as nx
from typing import Type
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode
from pyjuice.layer import InputLayer 

def BayesianTreeToHiddenRegionGraph(tree: nx.Graph, 
                                    root,
                                    num_latents: int,  
                                    input_layer_type: Type[InputLayer], 
                                    input_layer_params: dict) -> RegionGraph:
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
            r = InputRegionNode(scope = [v], num_nodes = num_latents, node_type = input_layer_type, **input_layer_params)
            var2rnode[v] = r
        else:
            # Inner Region
            
            # children(z_v)
            ch_regions = [var2rnode[c] for c in chs]

            # Add x_v to children(z_v)
            leaf_r = InputRegionNode(scope = [v], num_nodes = num_latents, node_type = input_layer_type, **input_layer_params)
            ch_regions.append(leaf_r)

            edge_ids = torch.arange(0, num_latents).view(-1, 1).repeat(1, len(ch_regions)) 
            rp = PartitionNode(ch_regions, num_nodes = num_latents, edge_ids = edge_ids)

            if v == root:
                par_ids = torch.zeros([num_latents], dtype = torch.int64)
                chs_ids = torch.arange(0, num_latents)
                r = InnerRegionNode([rp], num_nodes = 1, edge_ids = torch.stack((par_ids, chs_ids), dim = 0))
            else:
                par_ids = torch.arange(0, num_latents).view(-1, 1).repeat(1, num_latents).reshape(-1)
                chs_ids = torch.arange(0, num_latents).view(1, -1).repeat(num_latents, 1).reshape(-1)
                r = InnerRegionNode([rp], num_nodes = num_latents, edge_ids = torch.stack((par_ids, chs_ids), dim = 0))

            var2rnode[v] = r

    root_r = var2rnode[root]
    return root_r