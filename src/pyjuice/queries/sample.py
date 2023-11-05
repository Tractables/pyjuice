from __future__ import annotations

import torch
from typing import Union, Callable, Optional

from pyjuice.nodes import CircuitNodes
from pyjuice.layer import ProdLayer, SumLayer
from pyjuice.model import TensorCircuit


## Kernels for sample ##


@torch.compile(mode = "reduce-overhead", fullgraph = False)
def _sample_mask_generation(layer: SumLayer, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
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
        dist = torch.distributions.Categorical(probs = unnorm_probs.permute(0, 2, 1))
        node_mask[nids] = cids.unsqueeze(2).expand(N, C, B).gather(1, dist.sample().unsqueeze(1)).squeeze(1) # [num_nodes, B]

    return None


@torch.compile(mode = "reduce-overhead", fullgraph = False)
def _sample_backward_pass(layer: SumLayer, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                          element_mars: torch.Tensor, params: torch.Tensor, node_mask: torch.Tensor):
    for group_id in range(layer.num_bk_groups):
        chids = layer.grouped_chids[group_id]
        parids = layer.grouped_parids[group_id]

        element_flows[chids] = (node_flows[parids] * (node_mask[parids] == chids[:,None,None])).sum(dim = 1)

    return None


def _sample_sum_layer(layer: SumLayer, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                      node_mars: torch.Tensor, element_mars: torch.Tensor, 
                      params: torch.Tensor, node_mask: torch.Tensor) -> None:
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

    _sample_mask_generation(layer, node_mars, element_mars, params, node_mask)
    _sample_backward_pass(layer, node_flows, element_flows, node_mars, element_mars, params, node_mask)

    return None


## Main API ##


def sample(pc: TensorCircuit, inputs: torch.Tensor, missing_mask: Optional[torch.Tensor] = None):
    """
    conditional samples from pr(x_missing | x_not missing) for each input in the batch

    Arguments:
        - inputs:             tensor       [num_vars, batch_size]
        - missing_mask:       tensor[bool] [num_vars, batch_size] or [num_vars] or None
        
    Outputs:
        - samples: tensor [num_vars, batch_size]:
                replaces the missing values in inputs sampled by pr(x_miss | x_not miss)
    """
    if missing_mask is not None and missing_mask.dim() == 2:
        missing_mask = missing_mask.permute(1, 0)

    ## Run forward pass ##

    pc.forward(inputs, missing_mask = missing_mask)

    ## Run backward pass ##

    samples = inputs.clone().permute(1, 0).contiguous()
    
    # Initialize buffers
    node_flows = torch.zeros([pc.num_nodes, pc.node_mars.size(1)], device = pc.device, dtype = torch.bool)
    element_flows = torch.zeros([pc.num_elements, pc.node_mars.size(1)], device = pc.device, dtype = torch.bool)
    node_mask = torch.zeros([pc.num_nodes, pc.node_mars.size(1)], device = pc.device, dtype = torch.long)
    node_flows[-1,:] = True

    with torch.no_grad():
        for layer_id in range(len(pc.inner_layers) - 1, -1, -1):
            layer = pc.inner_layers[layer_id]

            if isinstance(layer, ProdLayer):
                # Nothing special needed, same as backward
                layer.backward(node_flows, element_flows)

            elif isinstance(layer, SumLayer):
                # Recompute `element_mars` for previous prod layer
                pc.inner_layers[layer_id-1].forward(pc.node_mars, pc.element_mars)
                _sample_sum_layer(layer, node_flows, element_flows, pc.node_mars, pc.element_mars, pc.params, node_mask)

            else:
                raise ValueError(f"Unknown layer type {ltype}.")
            
        for idx, layer in enumerate(pc.input_layers):
            layer.sample(samples, node_flows, missing_mask)

        if missing_mask.dim() == 1:
            samples[~missing_mask,:] = inputs.permute(1, 0)[~missing_mask,:]
        else:
            samples[~missing_mask] = inputs.permute(1, 0)[~missing_mask]

    return samples.permute(1, 0)