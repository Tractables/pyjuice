from __future__ import annotations

import torch
import torch.nn as nn
import time
import triton
import triton.language as tl
from functools import partial
from typing import Optional, Sequence

from pyjuice.graph import RegionGraph, InputRegionNode, InnerRegionNode, PartitionNode, truncate_npartition
from pyjuice.layer import Layer, InputLayer, ProdLayer, SumLayer
from pyjuice.functional import normalize_parameters, flat_softmax_fw, flat_softmax_bp
from pyjuice.utils.grad_fns import ReverseGrad, PseudoHookFunc

def _pc_model_backward_hook(grad, pc):
    pc.backward(
        ll_weights = grad / grad.sum() * grad.size(0),
        compute_param_flows = pc._optim_hyperparams["compute_param_flows"], 
        flows_memory = pc._optim_hyperparams["flows_memory"]
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

class ProbCircuit(nn.Module):
    def __init__(self, region_graph: RegionGraph, max_npartitions: Optional[int] = None, max_num_groups: int = 1) -> None:
        super().__init__()

        self.region_graph = self._convert_region_graph(region_graph, max_npartitions)
        self.device = torch.device("cpu")
        
        # Experimental, wheter to do calculations NOT in logdomain. Does extra bookkeeping to avoid logsumexp.
        self.skip_logsumexp = False

        self._init_pass_tensors_()
        self._init_layers(max_num_groups = max_num_groups)
        self._init_ad_tensors()

    def _init_pass_tensors_(self):
        self.node_mars = None
        self.element_mars = None
        self.node_flows = None
        self.element_flows = None
        self.param_flows = None
        self.sum_region_mars = None
        self.alphas = None

    def _init_ad_tensors(self):
        self._inputs = [None, None]
        self._inputs_grad = [None, None]
        self._backward_buffer = dict()

        self._optim_hyperparams = {
            "compute_param_flows": True,
            "flows_memory": 0.0
        }
        self._used_external_sum_params = False
        

    def forward(self, inputs: torch.Tensor, 
                        params: Optional[torch.Tensor] = None, 
                        input_params: Optional[Dict[str,torch.Tensor]] = None,
                        missing_mask: Optional[torch.Tensor] = None,
                        alphas: Optional[torch.Tensor]=None,
                        ):
        if missing_mask is not None:
            assert inputs.size() == missing_mask.size(), f"inputs.size {inputs.size()} != mask.size {missing_mask.size()}" 
        
        if alphas is not None:
            assert inputs.size() == alphas.size(), f"inputs.size() {inputs.size()} != alphas.size() {alphas.size()}" 

        B = inputs.size(0)
        inputs = inputs.permute(1, 0)
        if alphas is not None:
            alphas = alphas.permute(1, 0)
        if missing_mask is not None:
            missing_mask = missing_mask.permute(1, 0)
        
        self.node_mars = torch.empty([self.num_nodes, B], device = self.device)
        self.element_mars = torch.empty([self.num_elements, B], device = self.device)

        if self.skip_logsumexp:
            self.sum_region_mars = torch.zeros([self.num_sum_regions, B], device = inputs.device)
            self.node_mars[0,:] = 1.0
            self.element_mars[0,:] = 0.0
        else:
            self.node_mars[0,:] = 0.0
            self.element_mars[0,:] = -torch.inf

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

        with torch.no_grad():
            for idx, layer in enumerate(self.input_layers):
                if input_params is not None and f"input_{idx}" in input_params:
                    layer_params = input_params[f"input_{idx}"]

                    self._backward_buffer["external_input_layers"].add(idx)
                    grad_hook_idx = layer._hook_params(grad_hook_idx, self._inputs, layer_params)
                else:
                    layer_params = None

                layer(inputs, self.node_mars, params = layer_params, missing_mask=missing_mask, skip_logsumexp = self.skip_logsumexp, alphas=alphas)

            for ltype, layer in self.inner_layers:
                if ltype == "prod":
                    layer(self.node_mars, self.element_mars, skip_logsumexp = self.skip_logsumexp)
                elif ltype == "sum":
                    layer(self.node_mars, self.element_mars, params, sum_region_mars=self.sum_region_mars, skip_logsumexp = self.skip_logsumexp)
                else:
                    raise ValueError(f"Unknown layer type {ltype}.")
                
        if self.skip_logsumexp:
            lls = self.node_mars[-1,:] + self.sum_region_mars.log().sum(dim = 0)
        else:
            lls = self.node_mars[-1,:]

        if torch.is_grad_enabled():
            lls.requires_grad = True
            lls.register_hook(partial(_pc_model_backward_hook, pc = self))

            self._inputs[0] = ReverseGrad.apply(inputs) # Record inputs for backward

            tensors = []
            for i in range(len(self._inputs)):
                if self._inputs[i] is not None and self._inputs[i].requires_grad:
                    self._inputs[i].register_hook(partial(_pc_inputs_hook, pc = self, i = i))
                    tensors.append(self._inputs[i])
            tensors.append(lls)

            return PseudoHookFunc.apply(*tensors)
        
        return lls

    def backward(self, inputs: Optional[torch.Tensor] = None, ll_weights: Optional[torch.Tensor] = None,
                 compute_param_flows: bool = True, flows_memory: float = 0.0):
        assert self.node_mars is not None and self.element_mars is not None, "Should run forward path first."

        self.node_flows = torch.zeros([self.num_nodes, self.node_mars.size(1)], device = self.device)
        self.element_flows = torch.zeros([self.num_elements, self.node_mars.size(1)], device = self.device)

        if ll_weights is None:
            self.node_flows[-1,:] = 1.0
        else:
            self.node_flows[-1,:] = ll_weights.squeeze()

        if self._inputs[1] is not None:
            params = self._backward_buffer["normalized_params"]
        else:
            params = self.params

        if compute_param_flows:
            self.init_param_flows(flows_memory = flows_memory)

        with torch.no_grad():
            for layer_id in range(len(self.inner_layers) - 1, -1, -1):
                ltype, layer = self.inner_layers[layer_id]

                if ltype == "prod":
                    layer.backward(self.node_flows, self.element_flows, skip_logsumexp=self.skip_logsumexp)

                elif ltype == "sum":
                    self.inner_layers[layer_id-1][1].forward(self.node_mars, self.element_mars, skip_logsumexp = self.skip_logsumexp)

                    layer.backward(self.node_flows, self.element_flows, self.node_mars, self.element_mars, params, 
                                   param_flows = self.param_flows if compute_param_flows else None, 
                                   skip_logsumexp = self.skip_logsumexp, 
                                   sum_region_mars = self.sum_region_mars)

                else:
                    raise ValueError(f"Unknown layer type {ltype}.")

            if compute_param_flows:
                if inputs is None:
                    inputs = self._inputs[0]
                else:
                    inputs = inputs.permute(1, 0)

                grad_hook_idx = 2
                for idx, layer in enumerate(self.input_layers):
                    layer.backward(inputs, self.node_flows, self.node_mars)

                    if "external_input_layers" in self._backward_buffer and idx in self._backward_buffer["external_input_layers"]:
                        grad_hook_idx = layer._hook_param_grads(grad_hook_idx, self._inputs, self._inputs_grad)

                    layer._hook_input_grads(self._inputs, self._inputs_grad)

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

        return None

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        for layer in self.input_layers:
            layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount)
        
        # Only apply parameter update if external parameters are not used in the previous forward/backward pass
        if not self._used_external_sum_params:
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

        for layer in self.input_layers:
            layer.init_param_flows(flows_memory = flows_memory)

        return None
    
    @staticmethod
    def load(filename):
        pc = torch.load(filename, map_location='cpu')
        pc._init_pass_tensors_()
        pc._init_ad_tensors()

        for layer in pc.input_layers:
            layer.param_flows = None
    
        return pc
    

    def to(self, device):
        super(ProbCircuit, self).to(device)

        for layer in self.input_layers:
            layer.device = device

        self.device = device

    def get_param_specs(self):
        param_specs = dict()
        param_specs["inner"] = torch.Size([self.num_sum_params])
        for i, layer in enumerate(self.input_layers):
            param_specs[f"input_{i}"] = layer.get_param_specs()

        return param_specs

    def _convert_region_graph(self, region_graph: RegionGraph, max_npartitions: Optional[int] = None):
        if max_npartitions is not None:
            region_graph = truncate_npartition(region_graph, max_npartitions)

        return region_graph

    def _init_layers(self, init_input_params: Optional[Sequence[torch.Tensor]] = None, init_inner_params: Optional[torch.Tensor] = None,
                     max_num_groups: int = 1):
        depth2rnodes, num_layers = self._create_region_node_layers()

        if hasattr(self, "input_layers") or hasattr(self, "inner_layers"):
            raise ValueError("Attempting to initialize a ProbCircuit for the second time. " + \
                "Please instead create a new ProbCircuit instance by `ProbCircuit(region_graph)`.")

        self.input_layers = []
        self.inner_layers = []

        # Nodes include one dummy node and all input/sum nodes in the PC
        num_nodes = 1

        # Elements include one dummy element and all product nodes in the PC
        num_elements = 1

        # Number of parameters for sum nodes in the PC, plus one dummy parameter
        param_ends = [1]

        num_sum_regions = 0 # tally of sum regions so far

        layer_id = 0
        for depth in range(num_layers):
            if depth == 0:
                # Input layer
                type2rnodes = self._categorize_input_region_nodes(depth2rnodes[0]["input"])
                for NodeType, rnodes in type2rnodes.items():
                    input_layer = NodeType(layer_id, region_nodes = rnodes, cum_nodes = num_nodes)

                    num_nodes += input_layer.num_nodes
                    self.input_layers.append(input_layer)
                    self.add_module(f"input_layer_{layer_id}", input_layer)
                    layer_id += 1
            else:
                assert len(depth2rnodes[depth]["prod"]) > 0 and len(depth2rnodes[depth]["sum"]) > 0, \
                    "Depth {}: ({}, {})".format(depth, len(depth2rnodes[depth]["prod"]), len(depth2rnodes[depth]["sum"]))

                # Product layer
                prod_layer = ProdLayer(layer_id, depth2rnodes[depth]["prod"], max_num_groups = max_num_groups)

                if prod_layer.num_nodes + 1 > num_elements:
                    num_elements = prod_layer.num_nodes + 1
                self.add_module(f"prod_layer_{layer_id}", prod_layer)
                self.inner_layers.append(("prod", prod_layer))
                layer_id += 1

                # Sum layer
                sum_layer = SumLayer(layer_id, depth2rnodes[depth]["sum"],
                                     cum_nodes = num_nodes, 
                                     param_ends = param_ends, 
                                     ch_prod_layer_size = prod_layer.num_nodes + 1,
                                     sum_region_start_id = num_sum_regions,
                                     max_num_groups = max_num_groups)

                num_nodes += sum_layer.num_nodes
                num_sum_regions += len(depth2rnodes[depth]["sum"])
                self.add_module(f"sum_layer_{layer_id}", sum_layer)
                self.inner_layers.append(("sum", sum_layer))
                layer_id += 1

        self.num_nodes = num_nodes
        self.num_elements = num_elements
        self.num_sum_params = param_ends[-1]
        self.param_ends = param_ends
        self.num_sum_regions = num_sum_regions

        # For parameter normalization
        # Node that input nodes are implicitly omitted as they have no child
        node_ids = torch.empty([self.num_sum_params], dtype = torch.long)
        node_nchs = torch.empty([len(self.param_ends)], dtype = torch.long)
        node_ids[:self.param_ends[0]] = 0
        node_nchs[0] = self.param_ends[0]
        for i in range(1, len(self.param_ends)):
            node_ids[self.param_ends[i-1]:self.param_ends[i]] = i
            node_nchs[i] = self.param_ends[i] - self.param_ends[i-1]
        
        self.register_buffer("node_ids", node_ids)
        self.register_buffer("node_nchs", node_nchs)

        self._init_params()

    def _init_params(self, perturbation: float = 4.0):
        params = torch.exp(torch.rand([self.num_sum_params]) * -perturbation)

        # Copy initial parameters if provided
        for layer_type, layer in self.inner_layers:
            if layer_type == "sum":
                for rnode in layer.region_nodes:
                    if hasattr(rnode, "_params"):
                        sidx, eidx = rnode._param_range
                        params[sidx:eidx] = rnode._params[rnode._inverse_param_ids].to(params.device)

        self._normalize_parameters(params)
        self.params = nn.Parameter(params)

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

        for idx, layer in enumerate(self.input_layers):
            layer._init_params(perturbation)

    def _normalize_parameters(self, params, pseudocount: float = 0.0):
        if params is not None:
            normalize_parameters(params, self.node_ids, self.node_nchs, pseudocount)

    @staticmethod
    @triton.jit
    def _cum_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, tot_num_params, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tot_num_params

        n_offsets = tl.load(node_ids_ptr + offsets, mask = mask, other = 0)

        params = tl.load(params_ptr + offsets, mask = mask, other = 0)

        tl.atomic_add(cum_params_ptr + n_offsets, params, mask = mask)

    @staticmethod
    @triton.jit
    def _norm_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, node_nchs_ptr, tot_num_params, 
                            pseudocount, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tot_num_params

        n_offsets = tl.load(node_ids_ptr + offsets, mask = mask, other = 0)

        params = tl.load(params_ptr + offsets, mask = mask, other = 0)
        cum_params = tl.load(cum_params_ptr + n_offsets, mask = mask, other = 1)
        nchs = tl.load(node_nchs_ptr + n_offsets, mask = mask, other = 1)

        normed_params = (params + pseudocount / nchs) / (cum_params + pseudocount)
        tl.store(params_ptr + offsets, normed_params, mask = mask)

    def _extract_params_to_rnodes(self):
        """
        Extract all inner and input parameters from this ProbCircuit to individual region nodes.
        """
        assert self.device.type == "cpu", "Please move the ProbCircuit to CPU before extracting its parameters."

        # Inner region nodes
        for layer_type, layer in self.inner_layers:
            if layer_type == "sum":
                for rnode in layer.region_nodes:
                    sidx, eidx = rnode._param_range
                    rnode._params = self.params.data[sidx:eidx].detach().cpu().clone()

        # Input region nodes
        for layer in self.input_layers:
            layer._extract_params_to_rnodes()

    def _create_region_node_layers(self):
        depth2rnodes = dict()
        rnode2depth = dict()

        num_layers = [1]

        def dfs(n: RegionGraph):
            if n in rnode2depth:
                return
            if isinstance(n, InputRegionNode):
                rnode2depth[n] = 0
                if 0 not in depth2rnodes:
                    depth2rnodes[0] = {"input": []}
                depth2rnodes[0]["input"].append(n)
            else:
                for c in n.children:
                    dfs(c)

                depth = max(map(lambda m: rnode2depth[m], n.children)) + (1 if isinstance(n, PartitionNode) else 0)
                num_layers[0] = max(depth + 1, num_layers[0])
                rnode2depth[n] = depth

                if depth not in depth2rnodes:
                    depth2rnodes[depth] = {"sum": [], "prod": []} # lists for sum and product regions
                
                if isinstance(n, PartitionNode):
                    depth2rnodes[depth]["prod"].append(n)
                elif isinstance(n, InnerRegionNode):
                    depth2rnodes[depth]["sum"].append(n)
                else:
                    raise NotImplementedError(f"Unsupported region node type {type(n)}.")

        dfs(self.region_graph)

        return depth2rnodes, num_layers[0]

    def _categorize_input_region_nodes(self, rnodes):
        type2rnodes = dict()
        for r in rnodes:
            if r.node_type not in type2rnodes:
                type2rnodes[r.node_type] = []
            type2rnodes[r.node_type].append(r)

        return type2rnodes
