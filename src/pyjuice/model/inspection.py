"""
Reading :class:`~pyjuice.model.tensorcircuit.TensorCircuit`'s state per node group (:class:`_InspectionMixin`, one of
its bases): the values and flows of the last passes, the parameters and their flows, and the PC's statistics.
"""

from __future__ import annotations

from pyjuice.nodes import CircuitNodes


class _InspectionMixin:
    """Reading the state per node group, for :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (one of its bases)."""

    def print_statistics(self):
        """
        Print the statistics of the PC.
        """
        print(f"> Number of nodes: {self.num_nodes}")
        print(f"> Number of edges: {self.num_edges}")
        print(f"> Number of sum parameters: {self.num_sum_params}")

        num_input_parameters = 0
        for layer in self.input_layer_group:
            try:
                num_input_parameters += layer.params.numel()
            except AttributeError:
                pass
        
        print(f"> Number of input parameters: {num_input_parameters}")

    def get_node_mars(self, ns: CircuitNodes):
        """
        Retrieve the node values of `ns` from the previous forward pass.

        :params ns: the target nodes
        :type ns: CircuitNodes
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "node_mars") and self.node_mars is not None
        assert hasattr(self, "element_mars") and self.element_mars is not None

        nsid, neid = ns._output_ind_range

        if ns.is_sum() or ns.is_input():
            return self.node_mars[nsid:neid,:].detach()
        else:
            assert ns.is_prod()

            target_layer = None
            for layer_group in self.inner_layer_groups:
                for layer in layer_group:
                    if layer.is_prod() and ns in layer.nodes:
                        target_layer = layer
                        break

                if target_layer is not None:
                    break

            # Rerun the corresponding product layer to get the node values
            layer(self.node_mars, self.element_mars)

            return self.element_mars[nsid:neid,:].detach()

    def get_node_flows(self, ns: CircuitNodes, **kwargs):
        """
        Retrieve the node flows of `ns` from the previous backward pass.

        :params ns: the target nodes
        :type ns: CircuitNodes
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "node_flows") and self.node_flows is not None
        assert hasattr(self, "element_flows") and self.element_flows is not None

        nsid, neid = ns._output_ind_range

        if ns.is_sum() or ns.is_input():
            return self.node_flows[nsid:neid,:].detach()
        else:
            assert ns.is_prod()

            layer_id = None
            for idx, layer_group in enumerate(self.inner_layer_groups):
                for layer in layer_group:
                    if layer.is_prod() and ns in layer.nodes:
                        layer_id = idx
                        break

                if layer_id is not None:
                    break

            # Rerun the corresponding product layer to get the node values
            self.inner_layer_groups[layer_id].forward(self.node_mars, self.element_mars)
            self.inner_layer_groups[layer_id+1].backward(
                self.node_flows, self.element_flows, self.node_mars, self.element_mars, self.params, 
                param_flows = None, allow_modify_flows = self._run_params["allow_modify_flows"], 
                propagation_alg = self._run_params["propagation_alg"], 
                logspace_flows = self._run_params["logspace_flows"], 
                negate_pflows = self._run_params["negate_pflows"],
                force_use_fp32 = self._run_params["force_use_fp32"], **kwargs
            )

            return self.element_flows[nsid:neid,:].detach()

    def get_node_params(self, ns: CircuitNodes, clone: bool = True, **kwargs):
        """
        Retrieve the node parameters of `ns`.

        :params ns: the target nodes
        :type ns: CircuitNodes

        :params clone: whether to clone the parameters
        :type clone: bool
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "params") and self.params is not None

        if not ns.is_sum() or ns.is_tied():
            return None

        psid, peid = ns._param_range
        if clone:
            ns_params = self.params[psid:peid].detach().clone()
        else:
            ns_params = self.params[psid:peid]

        local_parids = (ns._param_ids - psid) // (ns.block_size * ns.ch_block_size)
        num_parblocks = local_parids.size(0)
        ns_params = ns_params.reshape(num_parblocks, ns.ch_block_size, ns.block_size)

        return ns_params[local_parids,:,:].permute(0, 2, 1)

    def get_node_param_flows(self, ns: CircuitNodes, clone: bool = True, **kwargs):
        """
        Retrieve the node parameter flows of `ns`.

        :params ns: the target nodes
        :type ns: CircuitNodes

        :params clone: whether to clone the parameter flows
        :type clone: bool
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "param_flows") and self.param_flows is not None

        if not ns.is_sum() or ns.is_tied():
            return None

        pfsid, pfeid = ns._param_flow_range
        if clone:
            ns_param_flows = self.param_flows[pfsid:pfeid].detach().clone()
        else:
            ns_param_flows = self.param_flows[pfsid:pfeid]

        local_parfids = (ns._param_flow_ids - pfsid) // (ns.block_size * ns.ch_block_size)
        num_parfblocks = local_parfids.size(0)
        ns_param_flows = ns_param_flows.reshape(num_parfblocks, ns.ch_block_size, ns.block_size)
        return ns_param_flows[local_parfids,:,:].permute(0, 2, 1)
