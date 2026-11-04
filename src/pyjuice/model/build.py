"""
How :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (:class:`_BuildMixin`, one of its bases) compiles a PC's
nodes into layers -- the input, product and sum layer groups, with the layout of their parameters and flows -- and
fills its parameters from the nodes'.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Sequence, Union

from pyjuice.nodes import CircuitNodes, InputNodes, ExternalParamsSumNodes, summate, multiply
from pyjuice.layer import InputLayer, ProdLayer, SumLayer, ExternalParamsSumLayer, LayerGroup
from pyjuice.layer.fused_children import attach_element_children
from pyjuice.utils import BitSet

from .backend import compile_cum_par_flows_fn, compile_par_update_fn, normalize_parameters


class _BuildMixin:
    """Compiling the nodes into layers, for :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (one of its bases)."""

    def _init_layers(self, layer_sparsity_tol: Optional[float] = None, max_num_partitions: Optional[int] = None,
                     disable_gpu_compilation: bool = False, force_gpu_compilation: bool = False, 
                     max_tied_ns_per_parflow_block: int = 32, verbose: bool = True, device: Optional[Union[str,torch.device]] = None):

        if hasattr(self, "input_layer_group") or hasattr(self, "inner_layer_groups"):
            raise ValueError("Attempting to initialize a TensorCircuit for the second time. " + \
                "Please instead create a new TensorCircuit instance by calling `pc = TensorCircuit(root_ns)`.")

        # Clear hooks/pointers used by previous `TensorCircuit`s
        self.root_ns._clear_tensor_circuit_hooks()

        # TOtal number of variables
        pc_num_vars = len(self.root_ns.scope)

        # Create layers
        depth2nodes, num_layers, max_node_block_size, max_ele_block_size = self._create_node_layers()

        self.input_layer_group = None
        self.inner_layer_groups = []

        self.num_dummy_nodes = max_ele_block_size
        self.num_dummy_eles = max_node_block_size
        self.num_dummy_params = max_node_block_size * max_ele_block_size

        # Nodes include `max_ele_block_size` dummy nodes and all input/sum nodes in the PC
        num_nodes = self.num_dummy_nodes

        # Total number of edges
        num_edges = 0

        # Elements include `max_node_block_size` dummy elements and all product nodes in the PC
        num_elements = self.num_dummy_eles

        # Number of parameters
        num_parameters = self.num_dummy_params

        # Number of parameter flows
        num_param_flows = 0

        # Stores distributed parameter flows
        node2tiednodes = dict()

        # Every `ns` in this PC that takes external parameters, mapped to the layer that compiled it.
        # Insertion-ordered, and the order is what fixes each node's slot in the staging buffer, so it
        # must stay deterministic. Also used to reject `sum_external_params` entries keyed by a node
        # the PC does not have -- which would otherwise be a silent no-op.
        self.external_params_nodes = dict()

        # (batch size) -> (total numel, {ns: [(offset, shape), ...]}) for the staging buffer
        self._external_params_layouts = dict()

        # Views staged by the most recent forward pass; the backward reuses them, and writes the
        # matching per-sample gradients into views of the same shape
        self._staged_external_params = None
        self._staged_external_params_grad = None
        self._external_grad_dsts = None

        # name -> (nodes, dim): several `ns` addressed by ONE concatenated tensor. See
        # `register_external_params_group`.
        self.external_params_groups = dict()

        # (buffer name, batch size) -> (buffer, {ns: views}); see `_external_params_views`
        self._external_params_views_cache = dict()

        if verbose:
            print(f"Compiling {num_layers} TensorCircuit layers...")

        # Select device to use
        if device is None:
            device = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            device = device.index

        with torch.cuda.device(f"cuda:{device}"):
            layer_id = 0
            for depth in tqdm(range(num_layers), disable = not verbose):
                if depth == 0:
                    # Input layer
                    signature2nodes = self._categorize_input_nodes(depth2nodes[0]["input"])
                    input_layer_id = 0
                    input_layers = []
                    for signature, nodes in signature2nodes.items():
                        input_layer = InputLayer(
                            nodes = nodes, cum_nodes = num_nodes,
                            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
                            pc_num_vars = pc_num_vars
                        )

                        # Special treatment for input layers with the `External` distribution
                        if input_layer.dist.requires_external_inputs():
                            scope = BitSet()
                            for ns in input_layer.nodes:
                                scope |= ns.scope
                            vars = torch.sort(torch.tensor(scope.to_list())).values
                            var_idmapping = torch.zeros([pc_num_vars], dtype = torch.long)
                            var_idmapping[vars] = torch.arange(0, vars.size(0))
                            input_layer.register_buffer("var_idmapping", var_idmapping)

                        input_layers.append(input_layer)
                        
                        input_layer_id += 1
                        num_nodes += input_layer.num_nodes

                    self.input_layer_group = LayerGroup(input_layers)

                else:
                    assert len(depth2nodes[depth]["prod"]) > 0 and len(depth2nodes[depth]["sum"]) > 0, \
                        "Depth {}: (# prod nodes: {}, # sum nodes: {})".format(depth, len(depth2nodes[depth]["prod"]), len(depth2nodes[depth]["sum"]))

                    # Product layer(s)
                    gsize2prod_nodes = dict()
                    for ns in depth2nodes[depth]["prod"]:
                        gsize = ns.block_size
                        if gsize not in gsize2prod_nodes:
                            gsize2prod_nodes[gsize] = []
                        gsize2prod_nodes[gsize].append(ns)
                    
                    layer_num_elements = max_node_block_size
                    prod_layers = []
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

                        prod_layers.append(prod_layer)
                    
                    prod_layer_group = LayerGroup(prod_layers)
                    # Every element row's child node rows, for the sum layers' fused passes (a buffer of the group)
                    attach_element_children(prod_layer_group, max_node_block_size)
                    self.inner_layer_groups.append(prod_layer_group)
                    self.add_module(f"prod_layer_{layer_id}", prod_layer_group)

                    if layer_num_elements > num_elements:
                        num_elements = layer_num_elements

                    # Sum layer(s)
                    # Nodes are grouped by (block size, external-parameter signature): one layer
                    # compiles one set of kernels, so nodes whose effective parameters are formed
                    # differently -- or not modified at all -- must not share a layer. This keeps the
                    # standard sum-layer kernels free of any per-call branch on the parameterization.
                    gsize2sum_nodes = dict()
                    for ns in depth2nodes[depth]["sum"]:
                        ext_signature = ns.get_external_signature() if isinstance(ns, ExternalParamsSumNodes) else None
                        gsize = (ns.block_size, ext_signature)
                        if gsize not in gsize2sum_nodes:
                            gsize2sum_nodes[gsize] = []
                        gsize2sum_nodes[gsize].append(ns)

                    sum_layers = []
                    for (gsize, ext_signature), nodes in gsize2sum_nodes.items():
                        layer_class = SumLayer if ext_signature is None else ExternalParamsSumLayer

                        sum_layer = layer_class(
                            nodes = nodes,
                            global_nid_start = num_nodes, 
                            global_pid_start = num_parameters,
                            global_pfid_start = num_param_flows,
                            node2tiednodes = node2tiednodes,
                            layer_sparsity_tol = layer_sparsity_tol,
                            max_num_partitions = max_num_partitions,
                            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
                            disable_gpu_compilation = disable_gpu_compilation,
                            force_gpu_compilation = force_gpu_compilation
                        )

                        num_nodes += sum_layer.num_nodes
                        num_edges += sum_layer.num_edges
                        num_parameters += sum_layer.num_parameters
                        num_param_flows += sum_layer.num_param_flows

                        if ext_signature is not None:
                            for ns in sum_layer.nodes:
                                self.external_params_nodes[ns] = sum_layer

                        sum_layers.append(sum_layer)

                    sum_layer_group = LayerGroup(sum_layers)
                    self.inner_layer_groups.append(sum_layer_group)
                    self.add_module(f"sum_layer_{layer_id}", sum_layer_group)

                    layer_id += 1

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_elements = num_elements
        self.num_sum_params = num_parameters
        self.num_param_flows = num_param_flows

        # Does any layer need the denominator flow `F-`? Resolved once here, so `init_param_flows`
        # allocates `denom_param_flows` only when it is actually used -- an ordinary PC never does and
        # keeps `denom_param_flows = None`.
        #
        # Requesting layers are laid out end to end in ONE flat buffer, each getting the slices its
        # parameterization asked for (`assign_denom_flow_offsets`). Sizing happens HERE, at compile,
        # not at first forward, because `init_param_flows` may run before any forward pass.
        #
        # `_denom_sources_of_ns` is the denominator's answer to `compute_cum_par_flows`: tied copies
        # share one `_param_range`, so every copy's flow has to reach the M-step of the node that owns
        # those parameters. The numerator is fused in the `pfid` space it shares; the denominator, whose
        # layout is the parameterization's own, is combined while `compute_em_correction` reconstructs
        # `F-`, which is why the copies are collected rather than summed here.
        denom_layers = [layer for group in self.inner_layer_groups for layer in group.layers
                        if layer.requests_denom_param_flows]
        self._requests_denom_param_flows = len(denom_layers) > 0
        self.num_denom_flows = 0
        self._denom_sources_of_ns = dict()
        for layer in denom_layers:
            self.num_denom_flows = layer.assign_denom_flow_offsets(self.num_denom_flows)
            for ns in layer.nodes:
                target = ns.get_source_ns() if ns.is_tied() else ns
                self._denom_sources_of_ns.setdefault(target, []).append((layer, ns))

        # For parameter flow accumulation
        self.parflow_fusing_kwargs = compile_cum_par_flows_fn(node2tiednodes, MAX_NBLOCKS = 2048, BLOCK_SIZE = 2048)
        
        # For parameter update
        self.par_update_kwargs = compile_par_update_fn(self.root_ns, BLOCK_SIZE = 32)

        # Register root nodes
        self.num_root_nodes = self.root_ns.num_nodes
        self._root_node_range = (self.num_nodes - self.num_root_nodes, self.num_nodes)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self, perturbation: float = 4.0, pseudocount: float = 0.0):
        for ns in self.root_ns:
            if not ns.is_tied() and (ns.is_sum() or ns.is_input()) and not ns.has_params():
                ns.init_parameters(perturbation = perturbation, recursive = False)

        params = torch.exp(torch.rand([self.num_sum_params]) * -perturbation)
        params[:self.num_dummy_params] = 0.0

        # Copy initial parameters if provided
        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied() and ns.has_params():
                ns.gather_parameters(params)

        self._normalize_parameters(params, pseudocount = pseudocount)
        self.params = nn.Parameter(params.to(self.param_dtype))         # initialized and normalized in fp32, stored as asked

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

        # Initialize parameters for input layers
        for idx, layer in enumerate(self.input_layer_group):
            layer._init_parameters(perturbation)

    def _normalize_parameters(self, params, pseudocount: float = 0.0):
        if params is not None:
            normalize_parameters(params, self.par_update_kwargs, pseudocount)

    def _create_node_layers(self):
        depth2nodes = dict()
        nodes2depth = dict()

        num_layers = 1
        max_node_block_size = 0
        max_ele_block_size = 0

        def dfs(ns: CircuitNodes):

            nonlocal num_layers
            nonlocal max_node_block_size
            nonlocal max_ele_block_size

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
                    for idx, cs in enumerate(ns.chs):
                        cs_depth = nodes2depth[cs]
                        if cs_depth < depth:
                            # TODO: Make the block size be 1
                            pass_sum_ns = summate(
                                cs, num_node_blocks = cs.num_node_blocks, block_size = cs.block_size,
                                edge_ids = torch.arange(0, cs.num_node_blocks)[None,:].repeat(2, 1),
                                params = torch.eye(cs.block_size)[None,:,:].repeat(cs.num_node_blocks, 1, 1)
                            )
                            pass_prod_ns = multiply(pass_sum_ns)
                            ns.chs[idx] = pass_prod_ns

                            depth2nodes[cs_depth]["sum"].append(pass_sum_ns)

                            nodes2depth[pass_sum_ns] = cs_depth
                            nodes2depth[pass_prod_ns] = depth

                    depth2nodes[depth]["sum"].append(ns)

                    if ns.block_size > max_node_block_size:
                        max_node_block_size = ns.block_size
                elif ns.is_prod():
                    if ns.block_size > max_ele_block_size:
                        max_ele_block_size = ns.block_size
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

        return depth2nodes, num_layers, max_node_block_size, max_ele_block_size

    def _categorize_input_nodes(self, nodes: Sequence[InputNodes]):
        signature2nodes = dict()
        for ns in nodes:
            signature = ns.dist.get_signature()
            if signature not in signature2nodes:
                signature2nodes[signature] = []
            signature2nodes[signature].append(ns)

        return signature2nodes
