from __future__ import annotations

import torch
import torch.nn as nn
import time
import triton
import triton.language as tl
from tqdm import tqdm
from functools import partial
from typing import Optional, Sequence, Callable, Union

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, foreach
from pyjuice.layer import Layer, InputLayer, ProdLayer, SumLayer
from pyjuice.utils.grad_fns import ReverseGrad, PseudoHookFunc
from pyjuice.utils import BitSet

from .backend import compile_cum_par_flows_fn, compute_cum_par_flows, cum_par_flows_to_device, \
                     compile_par_update_fn, em_par_update, par_update_to_device, \
                     normalize_parameters


def _pc_model_backward_hook(grad, pc, **kwargs):
    grad = grad.permute(1, 0)
    pc.backward(
        ll_weights = grad / grad.sum() * grad.size(1),
        compute_param_flows = pc._optim_hyperparams["compute_param_flows"], 
        flows_memory = pc._optim_hyperparams["flows_memory"],
        **kwargs
    )

    pc._backward_buffer.clear()

    return None


def _pc_inputs_hook(grad, pc, i):

    if pc._inputs_grad[i] is not None:
        if grad is not None:
            grad = grad + pc._inputs_grad[i]
        else:
            grad = pc._inputs_grad[i]

    if pc._inputs[i] is not None:
        pc._inputs[i] = None
    
    if pc._inputs_grad[i] is not None:
        pc._inputs_grad[i] = None
    
    return grad


class TensorCircuit(nn.Module):
    def __init__(self, root_ns: CircuitNodes, layer_sparsity_tol: float = 0.5, 
                 max_num_partitions: Optional[int] = None, disable_gpu_compilation: bool = False, 
                 force_gpu_compilation: bool = False,
                 max_tied_ns_per_parflow_group: int = 8,
                 verbose: bool = True) -> None:
        """
        Create a tensorized circuit for the circuit rooted at `root_ns`.

        Parameters:
        `root_ns`:                       root nodes of the circuit
        `layer_sparsity_tol`:            the minimum allowed sparsity of compiled layers; ranges from 0.0 to 1.0; smaller means more strict
        `max_num_partitions`:            how many groups do we want to split a layer into
        `disable_gpu_compilation`:       disable GPU compilation of the layers
        `force_gpu_compilation`:         always use GPU when compiling the layers
        `max_tied_ns_per_parflow_group`: when there are tied nodes, specify at most how many nodes share a parameter flow accumulation buffer 
        """

        super(TensorCircuit, self).__init__()

        self.root_ns = root_ns
        self.device = torch.device("cpu")

        self._init_pass_tensors()
        self._init_layers(
            layer_sparsity_tol = layer_sparsity_tol, 
            max_num_partitions = max_num_partitions, 
            disable_gpu_compilation = disable_gpu_compilation, 
            force_gpu_compilation = force_gpu_compilation,
            max_tied_ns_per_parflow_group = max_tied_ns_per_parflow_group,
            verbose = verbose
        )
        
        # Hyperparameters for backward pass
        self._optim_hyperparams = {
            "compute_param_flows": True,
            "flows_memory": 0.0
        }

    def _init_pass_tensors(self):
        self.node_mars = None
        self.element_mars = None
        self.node_flows = None
        self.element_flows = None
        self.param_flows = None
        
    def forward(self, inputs: torch.Tensor, 
                params: Optional[torch.Tensor] = None, 
                input_params: Optional[Dict[str,torch.Tensor]] = None,
                input_layer_fn: Optional[Union[str,Callable]] = None,
                cache: Optional[dict] = None,
                return_cache: bool = False,
                **kwargs):
        """
        Forward the circuit.

        Parameters:
        `inputs`:         [B, num_vars]
        `params`:         None or [B, num_params]
        `input_params`:   A dictionary of input parameters
        `input_layer_fn`: Custom forward function for input layers;
                          if it is a string, then try to call 
                          the corresponding member function of `input_layer`
        `kwargs`:         Additional arguments for input layers
        """
        
        B = inputs.size(0)
        inputs = inputs.permute(1, 0)
        
        ## Initialize buffers for forward pass ##

        if not isinstance(self.node_mars, torch.Tensor) or self.node_mars.size(0) != self.num_nodes or \
                self.node_mars.size(1) != B or self.node_mars.device != self.device:
            self.node_mars = torch.empty([self.num_nodes, B], device = self.device)

        if not isinstance(self.element_mars, torch.Tensor) or self.element_mars.size(0) != self.num_elements or \
                self.element_mars.size(1) != B or self.element_mars.device != self.device:
            self.element_mars = torch.empty([self.num_elements, B], device = self.device)

        # Load cached node marginals
        if cache is not None and "node_mars" in cache:
            assert cache["node_mars"].dim() == 2 and cache["node_mars"].size(0) == self.node_mars.size(0) and \
                cache["node_mars"].size(1) == self.node_mars.size(1)
            self.node_mars[:,:] = cache["node_mars"]

        self.node_mars[0,:] = 0.0
        self.element_mars[0,:] = -torch.inf

        ## Preprocess parameters ##

        if params is None:
            params = self.params
        else:
            if params.dim() == 2:
                if params.size(1) == self.num_sum_params:
                    params = params.permute(1, 0)
                else:
                    assert params.size(0) == self.num_sum_params, "Size of `params` does not match the number of sum parameters."

            self._inputs[1] = ReverseGrad.apply(params)

            # normalize
            params = flat_softmax_fw(logits = params, node_ids = self.node_ids, inplace = False)
            params[0] = 1.0
            self._backward_buffer["normalized_params"] = params

        if input_params is not None:
            grad_hook_idx = 2
            self._backward_buffer["external_input_layers"] = set()

        ## Run forward pass ##

        with torch.no_grad():
            # Compute forward pass for all input layers
            for idx, layer in enumerate(self.input_layers):
                if input_params is not None and f"input_{idx}" in input_params:
                    layer_params = input_params[f"input_{idx}"]

                    self._backward_buffer["external_input_layers"].add(idx)
                    grad_hook_idx = layer._hook_params(grad_hook_idx, self._inputs, layer_params)
                else:
                    layer_params = None

                if input_layer_fn is None:
                    layer(inputs, self.node_mars, params = layer_params, **kwargs)
                elif isinstance(input_layer_fn, str):
                    assert hasattr(layer, input_layer_fn), f"Custom input function `{input_layer_fn}` not found for layer type {type(layer)}."
                    getattr(layer, input_layer_fn)(inputs, self.node_mars, params = layer_params, **kwargs)
                else:
                    assert isinstance(input_layer_fn, Callable), f"Custom input function should be either a `str` or a `Callable`. " + \
                                                           f"Found {type(input_layer_fn)} instead."
                    input_layer_fn(layer, inputs, self.node_mars, params = layer_params, **kwargs)

            for layer in self.inner_layers:
                if isinstance(layer, ProdLayer):
                    # Prod layer
                    layer(self.node_mars, self.element_mars)

                elif isinstance(layer, SumLayer):
                    # Sum layer
                    layer(self.node_mars, self.element_mars, params)

                else:
                    raise ValueError(f"Unknown layer type {type(layer)}.")
                
        lls = self.node_mars[self._root_node_range[0]:self._root_node_range[1],:]
        lls = lls.permute(1, 0)

        ## Create/Update cache if needed ##

        if return_cache:
            if cache is None:
                cache = dict()

            with torch.no_grad():
                cache["node_mars"] = self.node_mars.clone()

        ## Add gradient hook for backward pass ##

        if torch.is_grad_enabled():
            lls.requires_grad = True
            lls.register_hook(partial(_pc_model_backward_hook, pc = self, **kwargs))

            self._inputs[0] = ReverseGrad.apply(inputs) # Record inputs for backward

            tensors = []
            for i in range(len(self._inputs)):
                if self._inputs[i] is not None and self._inputs[i].requires_grad:
                    self._inputs[i].register_hook(partial(_pc_inputs_hook, pc = self, i = i))
                    tensors.append(self._inputs[i])
            tensors.append(lls)

            if return_cache:
                return PseudoHookFunc.apply(*tensors).clone(), cache
            else:
                return PseudoHookFunc.apply(*tensors).clone()

        if return_cache:
            return lls.clone(), cache
        else:
            return lls.clone()

    def backward(self, inputs: Optional[torch.Tensor] = None, 
                 ll_weights: Optional[torch.Tensor] = None,
                 compute_param_flows: bool = True, 
                 flows_memory: float = 0.0,
                 input_layer_fn: Optional[Union[str,Callable]] = None,
                 cache: Optional[dict] = None,
                 return_cache: bool = False,
                 **kwargs):
        """
        Compute circuit flows.

        Parameters:
        `inputs`:         None or [B, num_vars]
        `ll_weights`:     None or [B] or [B, num_roots]
        `input_layer_fn`: Custom forward function for input layers;
                          if it is a string, then try to call 
                          the corresponding member function of `input_layer`
        `kwargs`:         Additional arguments for input layers
        """

        assert self.node_mars is not None and self.element_mars is not None, "Should run forward path first."

        B = self.node_mars.size(1)

        ## Initialize buffers for backward pass ##

        if not isinstance(self.node_flows, torch.Tensor) or self.node_flows.size(0) != self.num_nodes or \
                self.node_flows.size(1) != B or self.node_flows.device != self.device:
            self.node_flows = torch.zeros([self.num_nodes, B], device = self.device)
        
        if not isinstance(self.element_flows, torch.Tensor) or self.element_flows.size(0) != self.num_elements or \
                self.element_flows.size(1) != B or self.element_flows.device != self.device:
            self.element_flows = torch.zeros([self.num_elements, B], device = self.device)

        # Clear node flows
        self.node_flows[:,:] = 0.0

        # Set root node flows
        if ll_weights is None:
            self.node_flows[self._root_node_range[0]:self._root_node_range[1],:] = 1.0
        else:
            if ll_weights.dim() == 1:
                ll_weights = ll_weights.unsqueeze(1)

            assert ll_weights.size(0) == self.num_root_nodes

            self.node_flows[self._root_node_range[0]:self._root_node_range[1],:] = ll_weights

        # Load cached node flows
        if cache is not None and "node_flows" in cache:
            assert cache["node_flows"].dim() == 2 and cache["node_flows"].size(0) == self.node_flows.size(0) and \
                cache["node_flows"].size(1) == self.node_flows.size(1)

            if "replace_root_flows" in cache and cache["replace_root_flows"]:
                if hasattr(self, "_pv_node_flows_mask") and getattr(self, "_pv_node_flows_mask") is not None:
                    self.node_flows[self._pv_node_flows_mask,:] = 0.0

                self.node_flows[:self._root_node_range[0],:] = cache["node_flows"][:self._root_node_range[0],:].to(self.device)
                self.node_flows[self._root_node_range[1]:,:] = cache["node_flows"][self._root_node_range[1]:,:].to(self.device)
            else:
                self.node_flows[:,:] = cache["node_flows"]

                if hasattr(self, "_pv_node_flows_mask") and getattr(self, "_pv_node_flows_mask") is not None:
                    self.node_flows[self._pv_node_flows_mask,:] = 0.0
                    self.node_flows[self._root_node_range[0]:self._root_node_range[1],:] = cache["node_flows"][self._root_node_range[0]:self._root_node_range[1],:]

        ## Retrieve parameters and initialize parameter flows ##
        if self._inputs[1] is not None:
            params = self._backward_buffer["normalized_params"]
        else:
            params = self.params

        if compute_param_flows:
            self.init_param_flows(flows_memory = flows_memory)

        ## Run backward pass ##

        with torch.no_grad():
            for layer_id in range(len(self.inner_layers) - 1, -1, -1):
                layer = self.inner_layers[layer_id]

                if isinstance(layer, ProdLayer):
                    # Prod layer
                    layer.backward(self.node_flows, self.element_flows)

                elif isinstance(layer, SumLayer):
                    # Sum layer

                    # First recompute the previous product layer
                    self.inner_layers[layer_id-1].forward(self.node_mars, self.element_mars, _for_backward = True)

                    # Backward sum layer
                    layer.backward(self.node_flows, self.element_flows, self.node_mars, self.element_mars, params, 
                                   param_flows = self.param_flows if compute_param_flows else None)

                else:
                    raise ValueError(f"Unknown layer type {type(layer)}.")

            if inputs is None:
                inputs = self._inputs[0]
            else:
                inputs = inputs.permute(1, 0)

            # Compute backward pass for all input layers
            grad_hook_idx = 2
            for idx, layer in enumerate(self.input_layers):
                if input_layer_fn is None:
                    layer.backward(inputs, self.node_flows, self.node_mars, **kwargs)
                elif isinstance(input_layer_fn, str):
                    assert hasattr(layer, input_layer_fn), f"Custom input function `{input_layer_fn}` not found for layer type {type(layer)}."
                    getattr(layer, input_layer_fn)(inputs, self.node_flows, self.node_mars, **kwargs)
                else:
                    assert isinstance(input_layer_fn, Callable), f"Custom input function should be either a `str` or a `Callable`. " + \
                                                                    f"Found {type(input_layer_fn)} instead."
                    input_layer_fn(layer, inputs, self.node_flows, self.node_mars, **kwargs)

                if "external_input_layers" in self._backward_buffer and idx in self._backward_buffer["external_input_layers"]:
                    grad_hook_idx = layer._hook_param_grads(grad_hook_idx, self._inputs, self._inputs_grad)

            if self._inputs[1] is not None:
                B = self._inputs[0].size(0)

                # Below computes the parameter gradients derived from flows
                # grads = self.param_flows / params / B
                # grads[0] = 0.0
                # self._inputs_grad[1] = flat_softmax_bp(grads, params, self.node_ids, log_param_grad = False, inplace = False)

                # However, using the gradients directly generally leads to slow convergence
                # Instead, we use a scaled version of the gradient, as shown below
                flows = self.param_flows
                self._normalize_parameters(flows, pseudocount = self._pseudocount)
                flows[0] = 1.0
                grads = 0.5 * (torch.log(flows) - torch.log(params))
                self._inputs_grad[1] = flat_softmax_bp(grads, params, self.node_ids, log_param_grad = True, inplace = False)

                self._used_external_sum_params = True
            else:
                self._used_external_sum_params = False

        if return_cache:
            if cache is None:
                cache = dict()

            with torch.no_grad():
                cache["node_flows"] = self.node_flows.clone()

            return cache

        else:
            return None

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        for layer in self.input_layers:
            layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount)
        
        # Only apply parameter update if external parameters are not used in the previous forward/backward pass
        if not self._used_external_sum_params:
            # Normalize and update parameters
            with torch.no_grad():
                flows = self.param_flows
                if flows is None:
                    return None
                self._normalize_parameters(flows, pseudocount = pseudocount)
                self.params.data = (1.0 - step_size) * self.params.data + step_size * flows
                self.params[0] = 1.0

    def cumulate_flows(self, inputs: torch.Tensor, params: Optional[torch.Tensor] = None):
        with torch.no_grad():
            self.forward(inputs, params)
            self.backward(inputs = inputs, compute_param_flows = True, flows_memory = 1.0)

    def init_param_flows(self, flows_memory: float = 0.0):
        batch_size = self._inputs[1].size(1) if self._inputs[1] is not None and self._inputs[1].dim() == 2 else 1
        if self.param_flows is None or self.param_flows.size(0) != self.params.size(0) \
                or (self.param_flows.dim() == 1 and batch_size > 1) \
                or (self.param_flows.dim() == 2 and batch_size != self.param_flows.size(1)):
            if batch_size == 1:
                shape = [self.params.size(0)]
            else:
                shape = [self.params.size(0), batch_size]
            self.param_flows = torch.zeros(shape, device = self.device)
        else:
            assert self.param_flows.size(0) == self.params.size(0)
            self.param_flows[:] *= flows_memory

        # For input layers
        for layer in self.input_layers:
            layer.init_param_flows(flows_memory = flows_memory)

        return None

    def to(self, device):
        super(TensorCircuit, self).to(device)

        for layer in self.input_layers:
            layer.device = device

        self.device = device

        # For parameter flow accumulation
        self.parflow_fusing_kwargs = cum_par_flows_to_device(self.parflow_fusing_kwargs, device)
        
        # For parameter update
        self.par_update_kwargs = par_update_to_device(self.par_update_kwargs, device)

        return self

    def get_param_specs(self):
        param_specs = dict()
        param_specs["inner"] = torch.Size([self.num_sum_params])
        for i, layer in enumerate(self.input_layers):
            param_specs[f"input_{i}"] = layer.get_param_specs()

        return param_specs

    def update_parameters(self, clone_params: bool = True, update_flows: bool = False):
        """
        Copy parameters from this `TensorCircuit` to the original `CircuitNodes`
        """
        params = self.params.detach().cpu()
        if update_flows:
            param_flows = self.param_flows.detach().cpu()

        for ns in self.root_nodes:
            if ns.is_sum() and not ns.is_tied():
                psid, peid = ns._param_range
                if clone_params:
                    ns._params = params[ns._param_ids].clone()
                else:
                    ns._params = params[ns._param_ids]
                
                if update_flows:
                    if clone_params:
                        ns._flows = param_flows[ns._param_ids].clone()
                    else:
                        ns._flows = param_flows[ns._param_ids]

        for layer in self.input_layers:
            layer.update_parameters()

        return None

    def print_statistics(self):
        """
        Print the statistics of the PC.
        """
        print(f"> Number of nodes: {self.num_nodes}")
        print(f"> Number of edges: {self.num_edges}")
        print(f"> Number of sum parameters: {self.num_sum_params}")

    def copy_param_flows(self, clone_param_flows: bool = True, target_name: str = "_scores"):
        param_flows = self.param_flows.detach().cpu()

        for ns in self.root_nodes:
            if clone_params:
                setattr(ns, target_name, param_flows[ns._param_ids].clone())
            else:
                setattr(ns, target_name, param_flows[ns._param_ids])

        return None

    def enable_partial_evaluation(self, scopes: Union[Sequence[BitSet],Sequence[int]], 
                                  forward: bool = False, backward: bool = False):
        
        # Create scope2nid cache
        self._create_scope2nid_cache()

        if isinstance(scopes[0], int):
            scopes = [BitSet.from_array([var]) for var in scopes]

        fw_scopes = scopes if forward else None
        bk_scopes = scopes if backward else None

        # Input layers
        for layer in self.input_layers:
            layer.enable_partial_evaluation(fw_scopes = fw_scopes, bk_scopes = bk_scopes)

        # Inner layers
        for layer in self.inner_layers:
            layer.enable_partial_evaluation(fw_scopes = fw_scopes, bk_scopes = bk_scopes)

        if backward:
            scopes = set(scopes)
            _pv_node_flows_mask = torch.zeros([self.num_nodes], dtype = torch.bool)
            for ns in self.root_nodes:
                if (ns.is_sum() or ns.is_input()) and ns.scope in scopes:
                    sid, eid = ns._output_ind_range
                    _pv_node_flows_mask[sid:eid] = True
            self._pv_node_flows_mask = _pv_node_flows_mask.to(self.device)

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        # Input layers
        for layer in self.input_layers:
            layer.disable_partial_evaluation(forward = forward, backward = backward)

        # Inner layers
        for layer in self.inner_layers:
            layer.disable_partial_evaluation(forward = forward, backward = backward)

        self._pv_node_flows_mask = None

    def _init_layers(self, layer_sparsity_tol: Optional[float] = None, max_num_partitions: Optional[int] = None,
                     disable_gpu_compilation: bool = False, force_gpu_compilation: bool = False, 
                     max_tied_ns_per_parflow_group: int = 8, verbose: bool = True):

        if hasattr(self, "input_layers") or hasattr(self, "inner_layers"):
            raise ValueError("Attempting to initialize a TensorCircuit for the second time. " + \
                "Please instead create a new TensorCircuit instance by calling `pc = TensorCircuit(root_ns)`.")

        # Clear hooks/pointers used by previous `TensorCircuit`s
        self.root_ns._clear_tensor_circuit_hooks()

        # Create layers
        depth2nodes, num_layers, max_node_group_size, max_ele_group_size = self._create_node_layers()

        self.input_layers = []
        self.inner_layers = []

        self.num_dummy_nodes = max_ele_group_size
        self.num_dummy_eles = max_node_group_size

        # Nodes include `max_ele_group_size` dummy nodes and all input/sum nodes in the PC
        num_nodes = max_ele_group_size

        # Total number of edges
        num_edges = 0

        # Elements include `max_node_group_size` dummy elements and all product nodes in the PC
        num_elements = max_node_group_size

        # Number of parameters
        num_parameters = max_node_group_size

        # Number of parameter flows
        num_param_flows = 0

        # Stores distributed parameter flows
        node2tiednodes = dict()

        if verbose:
            print(f"Compiling {num_layers} layers...")

        layer_id = 0
        for depth in tqdm(range(num_layers), disable = not verbose):
            if depth == 0:
                # Input layer
                signature2nodes = self._categorize_input_nodes(depth2nodes[0]["input"])
                input_layer_id = 0
                for signature, nodes in signature2nodes.items():
                    input_layer = InputLayer(
                        nodes = nodes, cum_nodes = num_nodes,
                        max_tied_ns_per_parflow_group = max_tied_ns_per_parflow_group
                    )

                    self.input_layers.append(input_layer)
                    self.add_module(f"input_layer_{input_layer_id}", input_layer)
                    
                    input_layer_id += 1
                    num_nodes += input_layer.num_nodes
            else:
                assert len(depth2nodes[depth]["prod"]) > 0 and len(depth2nodes[depth]["sum"]) > 0, \
                    "Depth {}: (# prod nodes: {}, # sum nodes: {})".format(depth, len(depth2nodes[depth]["prod"]), len(depth2nodes[depth]["sum"]))

                # Product layer(s)
                gsize2prod_nodes = dict()
                for ns in depth2nodes[depth]["prod"]:
                    gsize = ns.group_size
                    if gsize not in gsize2prod_nodes:
                        gsize2prod_nodes[gsize] = []
                    gsize2prod_nodes[gsize].append(ns)
                
                layer_num_elements = max_node_group_size
                for gsize, nodes in gsize2prod_nodes.items():
                    prod_layer = ProdLayer(
                        nodes = nodes, 
                        global_nid_start = layer_num_elements,
                        layer_sparsity_tol = layer_sparsity_tol,
                        max_num_partitions = max_num_partitions,
                        disable_gpu_compilation = disable_gpu_compilation,
                        force_gpu_compilation = force_gpu_compilation
                    )

                    layer_num_elements += prod_layer.num_nodes
                    num_edges += prod_layer.num_edges

                    self.add_module(f"prod_layer_{layer_id}_{gsize}", prod_layer)
                    self.inner_layers.append(prod_layer)

                if layer_num_elements > num_elements:
                    num_elements = layer_num_elements

                # Sum layer(s)
                gsize2sum_nodes = dict()
                for ns in depth2nodes[depth]["sum"]:
                    gsize = ns.group_size
                    if gsize not in gsize2sum_nodes:
                        gsize2sum_nodes[gsize] = []
                    gsize2sum_nodes[gsize].append(ns)
                
                for gsize, nodes in gsize2sum_nodes.items():
                    sum_layer = SumLayer(
                        nodes = nodes,
                        global_nid_start = num_nodes, 
                        global_pid_start = num_parameters,
                        global_pfid_start = num_param_flows,
                        node2tiednodes = node2tiednodes,
                        layer_sparsity_tol = layer_sparsity_tol,
                        max_num_partitions = max_num_partitions,
                        max_tied_ns_per_parflow_group = max_tied_ns_per_parflow_group,
                        disable_gpu_compilation = disable_gpu_compilation,
                        force_gpu_compilation = force_gpu_compilation
                    )

                    num_nodes += sum_layer.num_nodes
                    num_edges += sum_layer.num_edges
                    num_parameters += sum_layer.num_parameters

                    self.add_module(f"sum_layer_{layer_id}_{gsize}", sum_layer)
                    self.inner_layers.append(sum_layer)

                layer_id += 1

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_elements = num_elements
        self.num_sum_params = num_parameters
        self.num_param_flows = num_param_flows

        # For parameter flow accumulation
        self.parflow_fusing_kwargs = compile_cum_par_flows_fn(node2tiednodes, MAX_NGROUPS = 2048, BLOCK_SIZE = 2048)
        
        # For parameter update
        self.par_update_kwargs = compile_par_update_fn(self.root_ns, BLOCK_SIZE = 32)

        # Register root nodes
        self.num_root_nodes = self.root_ns.num_nodes
        self._root_node_range = (self.num_nodes - self.num_root_nodes, self.num_nodes)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self, perturbation: float = 4.0, pseudocount: float = 0.0):
        params = torch.exp(torch.rand([self.num_sum_params]) * -perturbation)
        params[:self.num_dummy_eles] = 0.0

        # Copy initial parameters if provided
        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied() and ns.has_params():
                sidx, eidx = ns._param_range
                ns_params = ns._params[ns._inverse_param_ids,:,:].permute(0, 2, 1).reshape(-1)
                params[sidx:eidx] = ns_params.to(params.device)

        self._normalize_parameters(params, pseudocount = pseudocount)
        self.params = nn.Parameter(params)

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

        # Initialize parameters for input layers
        for idx, layer in enumerate(self.input_layers):
            layer._init_parameters(perturbation)

    def _normalize_parameters(self, params, pseudocount: float = 0.0):
        if params is not None:
            normalize_parameters(params, self.par_update_kwargs, pseudocount)

    def _create_node_layers(self):
        depth2nodes = dict()
        nodes2depth = dict()

        num_layers = 1
        max_node_group_size = 0
        max_ele_group_size = 0

        def dfs(ns: CircuitNodes):

            nonlocal num_layers
            nonlocal max_node_group_size
            nonlocal max_ele_group_size

            if ns in nodes2depth:
                return
            if ns.is_input():
                nodes2depth[ns] = 0
                if 0 not in depth2nodes:
                    depth2nodes[0] = {"input": []}
                depth2nodes[0]["input"].append(ns)
            else:
                for cs in ns.chs:
                    dfs(cs)

                depth = max(map(lambda ms: nodes2depth[ms], ns.chs)) + (1 if ns.is_prod() else 0)
                num_layers = max(depth + 1, num_layers)
                nodes2depth[ns] = depth

                if depth not in depth2nodes:
                    depth2nodes[depth] = {"sum": [], "prod": []} # lists for sum and product nodes
                
                if ns.is_sum():
                    depth2nodes[depth]["sum"].append(ns)
                    if ns.group_size > max_node_group_size:
                        max_node_group_size = ns.group_size
                elif ns.is_prod():
                    if ns.group_size > max_ele_group_size:
                        max_ele_group_size = ns.group_size
                else:
                    raise NotImplementedError(f"Unsupported node type {type(n)}.")

        dfs(self.root_ns)

        pns2layer = dict()
        for layer in range(1, len(depth2nodes)):
            for ns in depth2nodes[layer]["sum"]:
                for cs in ns.chs:
                    if cs.is_prod():
                        if id(cs) in pns2layer:
                            assert pns2layer[id(cs)] == layer, "Disallowed circumstance: a product node requested by sum nodes at different layers."
                        else:
                            depth2nodes[layer]["prod"].append(cs)
                            pns2layer[id(cs)] = layer

        return depth2nodes, num_layers, max_node_group_size, max_ele_group_size

    def _categorize_input_nodes(self, nodes: Sequence[InputNodes]):
        signature2nodes = dict()
        for ns in nodes:
            signature = ns.dist.get_signature()
            if signature not in signature2nodes:
                signature2nodes[signature] = []
            signature2nodes[signature].append(ns)

        return signature2nodes

    def _create_scope2nid_cache(self):
        # Input layers
        for idx, layer in enumerate(self.input_layers):
            layer._prepare_scope2nids()

        # Inner layers
        prod_scope_eleids = None
        for layer in self.inner_layers:
            if isinstance(layer, ProdLayer):
                prod_scope_eleids = layer._prepare_scope2nids()
            else:
                assert isinstance(layer, SumLayer)

                layer._prepare_scope2nids(prod_scope_eleids)
