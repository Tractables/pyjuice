from __future__ import annotations

import torch
from typing import Union, Callable, Optional

from pyjuice.nodes import CircuitNodes
from pyjuice.layer import ProdLayer, SumLayer
from pyjuice.model import TensorCircuit


# @torch.compile(mode = "reduce-overhead", fullgraph = False)
def _em_mask_generation(layer: SumLayer, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                            node_mask: torch.Tensor):
    for group_id in range(layer.num_fw_groups):
        nids = layer.grouped_nids[group_id]
        cids = layer.grouped_cids[group_id]
        pids = layer.grouped_pids[group_id]

        N, C = cids.size()
        B = node_mars.size(1)

        ch_mars = element_mars[cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        unnorm_probs = (ch_mars - maxval).exp() * params[pids] # [num_nodes, num_chs, B]
        node_mask[nids] = torch.argmax(unnorm_probs, dim = 1) # [num_nodes, B]

    return None


# @torch.compile(mode = "reduce-overhead", fullgraph = False)
def _em_backward_pass(layer: SumLayer, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                        element_mars: torch.Tensor, params: torch.Tensor, param_flows: torch.Tensor, node_mask: torch.Tensor):
    for group_id in range(layer.num_bk_groups):
        chids = layer.grouped_chids[group_id]
        parids = layer.grouped_parids[group_id]

        element_flows[chids] = (node_flows[parids] * (node_mask[parids] == chids[:,None,None])).sum(dim = 1)

    return None


def _em_pflow(layer: SumLayer, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
              element_mars: torch.Tensor, params: torch.Tensor, param_flows: torch.Tensor, node_mask: torch.Tensor):
    for group_id in range(layer.num_fw_groups):
        nids = layer.grouped_nids[group_id]
        cids = layer.grouped_cids[group_id]
        pids = layer.grouped_pids[group_id]

        N, C = cids.size()
        B = node_mars.size(1)

        ch_mars = element_mars[cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        unnorm_probs = (ch_mars - maxval).exp() * params[pids] # [num_nodes, num_chs, B]
        curr_node_mask = torch.argmax(unnorm_probs, dim = 1) # [num_nodes, B]

        r = torch.arange(0, N, device = nids.device)
        for b in range(B):
            curr_pids = pids[r,curr_node_mask[:,b]]
            param_flows[curr_pids] += node_flows[nids,b]

    return None


def _hard_em_sum_layer(layer: SumLayer, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                      node_mars: torch.Tensor, element_mars: torch.Tensor, 
                      params: torch.Tensor, param_flows: torch.Tensor, node_mask: torch.Tensor) -> None:
    """
    Compute sampling flow.

    Parameters:
    `node_flows`:    [num_nodes, B]
    `element_flows`: [max_num_els, B]
    `node_mars`:     [num_nodes, B]
    `element_mars`:  [max_num_els, B]
    `params`:        [num_params] or [num_params, B]
    `node_mask`:     [num_nodes, B]
    """
    if params.dim() == 1:
        params = params.unsqueeze(1)

    _em_mask_generation(layer, node_mars, element_mars, params, node_mask)
    _em_backward_pass(layer, node_flows, element_flows, node_mars, element_mars, params, param_flows, node_mask)
    _em_pflow(layer, node_flows, element_flows, node_mars, element_mars, params, param_flows, node_mask)

    return None


def hard_em(pc: TensorCircuit, inputs: torch.Tensor, flows_memory = 0.0):

    ## Run forward pass ##

    lls = pc.forward(inputs)

    pc.init_param_flows(flows_memory = flows_memory)

    ## Run backward pass ##

    samples = inputs.clone().permute(1, 0).contiguous()
    
    # Initialize buffers
    node_flows = torch.zeros([pc.num_nodes, pc.node_mars.size(1)], device = pc.device, dtype = torch.float32)
    element_flows = torch.zeros([pc.num_elements, pc.node_mars.size(1)], device = pc.device, dtype = torch.float32)
    node_mask = torch.zeros([pc.num_nodes, pc.node_mars.size(1)], device = pc.device, dtype = torch.long)
    node_flows[-1,:] = 1.0

    with torch.no_grad():
        for layer_id in range(len(pc.inner_layers) - 1, -1, -1):
            layer = pc.inner_layers[layer_id]

            if isinstance(layer, ProdLayer):
                # Nothing special needed, same as backward
                layer.backward(node_flows, element_flows)

            elif isinstance(layer, SumLayer):
                # Recompute `element_mars` for previous prod layer
                pc.inner_layers[layer_id-1].forward(pc.node_mars, pc.element_mars)
                _hard_em_sum_layer(layer, node_flows, element_flows, pc.node_mars, pc.element_mars, pc.params, pc.param_flows, node_mask)

            else:
                raise ValueError(f"Unknown layer type {ltype}.")
            
        for idx, layer in enumerate(pc.input_layers):
            layer.backward(inputs, node_flows, pc.node_mars)

    return lls