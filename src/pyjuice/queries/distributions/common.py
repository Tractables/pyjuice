from __future__ import annotations

import torch


def target_nodes(layer, target_vars, num_vars, device):
    """
    The input nodes a conditional query reads, and where each variable's result goes.

    Returns ``node_ids`` (the layer-local ids of the nodes whose variable is a target, in node order),
    ``rev_vars_mapping`` (variable id -> output row) and the number of target variables. Without
    ``target_vars`` every node and every variable is a target. The kernels always go through
    ``node_ids``, so a node's parameters, metadata and flow are all read at its own id.
    """
    if target_vars is None:
        target_vars = list(range(num_vars))
    rev_vars_mapping = torch.zeros([num_vars], dtype = torch.long)
    for i, var in enumerate(target_vars):
        rev_vars_mapping[var] = i
    is_target = torch.isin(layer.vids[:, 0], torch.as_tensor(list(target_vars), dtype = torch.long, device = layer.vids.device))
    node_ids = torch.nonzero(is_target).flatten().to(device)
    return node_ids, rev_vars_mapping.to(device), len(target_vars)
