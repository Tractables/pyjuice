from __future__ import annotations

import pyjuice as juice
import torch
import networkx as nx
import math
from typing import Optional, Union, List


def is_power_of_2(n: int):
    """
    Returns True if n is a power of 2.
    """
    if n <= 0:
        return False
    
    return (n & (n - 1)) == 0


def max_power_of_2_factor(n):
    if n == 0:
        return 0
    if n % 2 != 0:
        return 1

    power_of_2 = 1
    while n % 2 == 0:
        power_of_2 *= 2
        n //= 2  # Use integer division

    return power_of_2


class BayesianNetwork:
    """
    A class for defining Bayesian Network structures and parameters.
    """
    def __init__(self, variables: Optional[List[str]] = None, variable_num_states: Optional[List[int]] = None):
        """
        Construct Bayesian Networks.

        :param variables: An optional list of strings containing variable names
        :type variables: Optional[List[str]]

        :param variable_num_states: An optional list of integers containing the number of states in each variable
        :type variable_num_states: Optional[List[int]]
        """
        self.graph = nx.DiGraph()
        
        # Mappings to support both str and int indexing
        self.name_to_id = {}
        self.id_to_name = {}
        self._next_id = 0

        if variables is not None and variable_num_states is not None:
            assert len(variables) == len(variable_num_states), f"`len(variables)` and `len(variable_num_states)` should be the same."
        
        # Pre-populate index if variables are provided
        if variables is not None:
            for i, var_name in enumerate(variables):
                if variable_num_states is not None:
                    self.add_node(var_name, variable_num_states[i])
                else:
                    self._register_node(var_name)

    def add_node(self, node_ref: Union[str,int], num_states: int):
        """
        Add a node to the BN using either its string name or integer ID.
        
        :param node_ref: Name or ID of the node
        :type node_ref: Union[str,int]

        :param num_states: Number of states
        :type num_states: int
        """
        node_id = self._register_node(node_ref)

        assert node_id not in self.graph, f"Node {node_ref} has already been added."
        
        # Store attributes in the networkx graph
        self.graph.add_node(
            node_id,
            name = self.id_to_name[node_id], 
            num_states = num_states
        )

    def add_edge(self, u_ref: Union[str,int], v_ref: Union[str,int]):
        """
        Add a directed edge from u to v.
        
        :param u_ref: ID or name of the source variable
        :type u_ref: Union[str,int]

        :param v_ref: ID or name of the target/sink variable
        :type v_ref: Union[str,int]
        """
        u_id = self._register_node(u_ref)
        v_id = self._register_node(v_ref)
        
        # Ensure nodes exist in the graph before adding an edge
        assert u_id in self.graph and v_id in self.graph, "At least one node is not added to the graph, use `bn.add_node`."
            
        self.graph.add_edge(u_id, v_id)

    def add_edges_from(self, edges: List[Tuple[Union[str,int],Union[str,int]]]):
        """
        Add multiple directed edges from a list of tuples.

        :param edges: A list of tuples indicating the edges.
        :type edges: List[Tuple[Union[str,int],Union[str,int]]]
        """
        for u_ref, v_ref in edges:
            self.add_edge(u_ref, v_ref)

    def compile(self, observed_variable_list: Optional[List[Union[str,int]]] = None, max_treewidth: int = 10):
        """
        Compile the encoded Bayesian Network into a Probabilistic Circuit represented by `juice.nodes.CircuitNodes`.

        :param observed_variable_list: An optional list of variables (by ID or name) specifying the list of observed variables. The variable indices of the compiled PC will follow this order.
        :type observed_variable_list: Optional[List[Union[str,int]]]

        :param max_treewidth: The maximum treewidth for which the construction will be attempted.
        :type max_treewidth: int
        """
        if observed_variable_list is None:
            observed_variable_list = [i for i in range(self._next_id)]

        observed_variable_list = [
            self._get_node_id(node_ref) for node_ref in observed_variable_list
        ]
        nid_to_pcid = {
            node_id: i for i, node_id in enumerate(observed_variable_list)
        }

        # Need to trace the cutset.. Not that straightforward though

        # # Map from every node ID to its "current" PC
        # # This is initialized to representing the input nodes (if is observed)
        # nid_to_pc = dict()
        # for nid in range(self._next_id):
        #     if nid in nid_to_pcid:
        #         v = nid_to_pcid[nid]
        #         num_states = self.graph.nodes[nid]['num_states']

        #         nid_to_pc[nid] = juice.inputs(
        #             v, 
        #             num_node_blocks = num_states,
        #             block_size = 1,
        #             dist = juice.distributions.Indicator(num_states = num_states)
        #         )
        #     else:
        #         nid_to_pc[nid] = None

        # # Main loop: traverse the nodes in reverse topological order (children before parents)
        # reverse_topo_order = list(reversed(list(nx.topological_sort(self.graph))))
        # for node_id in reverse_topo_order:
        #     parent_node_ids = list(self.graph.predecessors(node_id))
        #     if len(parent_node_ids) == 0:
        #         continue # Skip if this is a "source" node (no parents)

        #     if len(parent_node_ids) > max_treewidth:
        #         return None # In this case, treewidth is too high, and we should give up

        #     # The number of required parent states
        #     par_num_states = [self.graph.nodes[pid]['num_states'] for pid in parent_node_ids]
        #     num_par_states = math.prod(par_num_states)

        #     # Step 1: preprocess the current variable
        #     ns = nid_to_pc[node_id]
        #     if ns is None:
        #         continue # Latent leaf nodes will be discarded as it doesn't affect the structure

        #     sum_ns = juice.summate(ns, num_node_blocks = num_par_states, block_size = 1)

        #     # Step 2: preprocess the parent PCs
        #     par_ns = [nid_to_pc[par_id] for par_id in parent_node_ids]
            
        #     grid_vecs = [torch.arange(num_states) for num_states in par_num_states]
        #     grid_tensors = torch.meshgrid(*grid_vecs, indexing = "ij")
            
        #     observed_par_ns = []
        #     observed_grid_tensors = []
        #     for ms, grid_tensor in zip(par_ns, grid_tensors):
        #         if ms is not None:
        #             observed_par_ns.append(ms)
        #             observed_grid_tensors.append(grid_tensor.reshape(-1))

        #     # Step 3: assemble them
        #     observed_grid_tensors.append(torch.arange(num_par_states))
        #     edge_ids = torch.stack(observed_grid_tensors, dim = 1)

        #     all_ns = observed_par_ns + [sum_ns]

        #     pns = juice.multiply(*all_ns, edge_ids = edge_ids)



    def _register_node(self, node_ref: Union[str,int]):
        """
        Internal helper to register a node and return its integer ID.
        """
        if isinstance(node_ref, int):
            if node_ref not in self.id_to_name:
                # User passed a new int ID; generate a default name
                self.id_to_name[node_ref] = f"Var_{node_ref}"
                self.name_to_id[f"Var_{node_ref}"] = node_ref
                # Keep _next_id above the highest manual int provided
                self._next_id = max(self._next_id, node_ref + 1)
            return node_ref
            
        elif isinstance(node_ref, str):
            if node_ref not in self.name_to_id:
                # User passed a new string; assign the next available integer ID
                var_id = self._next_id
                self.name_to_id[node_ref] = var_id
                self.id_to_name[var_id] = node_ref
                self._next_id += 1
            return self.name_to_id[node_ref]
            
        else:
            raise TypeError("Node reference must be an int or str.")

    def _get_node_id(self, node_ref: Union[str,int]):
        """
        Internal helper to get the node ID.
        """
        if isinstance(node_ref, int):
            assert node_ref in self.id_to_name, "Node not registered."
            return node_ref

        elif isinstance(node_ref, str):
            assert node_ref in self.name_to_id, "Node not registered."
            return self.name_to_id[node_ref]

        else:
            raise TypeError("Node reference must be an int or str.")