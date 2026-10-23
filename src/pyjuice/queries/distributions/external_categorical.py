from __future__ import annotations

import torch
from typing import Optional


def forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
            missing_mask: Optional[torch.Tensor] = None, **kwargs):

    inputs = inputs.permute(1, 0).contiguous()

    layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask, **kwargs)


def backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
             params: Optional[torch.Tensor] = None, **kwargs):

    assert "target_vars" not in kwargs or kwargs["target_vars"] is None

    # num_vars = layer.vids.max().item() + 1
    num_cats = int(layer.metadata[layer.s_mids].max().item())
    batch_size = node_flows.size(1)

    num_vars = layer.var_idmapping.size(0)

    cat_probs = torch.zeros([batch_size, num_vars, num_cats], dtype = torch.float32, device = node_flows.device)
    kwargs["external_categorical_logps_grad"] = cat_probs

    kwargs["no_param_update"] = True
    kwargs["extern_product_categorical_mode"] = "normalizing_constant"

    layer.backward(inputs, node_flows, node_mars, **kwargs)

    cat_probs /= (cat_probs.sum(dim = 2, keepdim = True) + 1e-12)

    return cat_probs
