"""
The children of every element row of a circuit, per product layer group, for the sum layers' fused passes
(`SumLayer.forward` / `.backward(..., compute_children = ...)`): a sum kernel that reads an element's value as the sum
of its product's child node values -- what the product layers' forward writes into ``element_mars`` -- instead of
launching those forwards first.

Element rows are shared by the layer groups: every product layer group's rows start at ``pc.num_dummy_eles`` (the
dummy block, -inf, that the sum tables' padding points into at row 0), so row ``r`` is a different product in every
group. So there is one table per product layer group -- ``table [n_k, K]`` for a group of ``n_k`` element rows
``[first_row, first_row + n_k)`` -- built when the circuit is compiled (:func:`attach_element_children`, a buffer of
the group's `LayerGroup`, moved with it) and read with :func:`element_children`; the sum layers of group ``k`` take
the table of group ``k - 1``. Children past a product's are the dummy node (row 0 of ``node_mars``, value 0: it adds
nothing, as the product kernels' own padding does); ``K`` is the group's largest child count, a power of two.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def element_children(pc) -> List[Optional[Tuple[torch.Tensor, int]]]:
    """Per inner layer group, ``(table, first_row)`` for a product group -- ``table[r - first_row, j]`` the ``j``-th
    child node row of element row ``r``, int32 on the group's device -- and None for a sum group; attached when the
    circuit was compiled, or now for a circuit compiled before the tables existed."""
    out = []
    for lg in pc.inner_layer_groups:
        if not lg.is_prod():
            out.append(None)
            continue
        if getattr(lg, "element_children", None) is None:
            attach_element_children(lg, pc.num_dummy_eles)
        out.append((lg.element_children, lg.element_children_first_row))
    return out


def form_elements(element_mars: torch.Tensor, node_mars: torch.Tensor, children: Tuple[torch.Tensor, int]) -> None:
    """Write a product layer group's element values into ``element_mars`` from ``children`` (its `(table, first_row)`,
    `element_children`): each element the sum of its two child node values, one fp32 add -- what the product layer's
    forward writes for products of two children, the same bits. What a layer handed the product group below (its
    `stream_children`) does where it cannot form them in its own kernels: the product layer's forward was not run."""
    table, first = children
    if table.size(1) != 2:
        raise ValueError("`stream_children`: products of two children only.")
    element_mars[first:first + table.size(0)] = node_mars[table[:, 0].long()] + node_mars[table[:, 1].long()]


def scatter_flows(node_flows: torch.Tensor, element_flows: torch.Tensor, children: Tuple[torch.Tensor, int],
                  rows: torch.Tensor) -> None:
    """Write the element flows of ``rows`` (int64 element rows) as their two children's node flows (``children``: the
    product group's `(table, first_row)`): what the product layer's backward writes for children of one parent element
    whose log-space flows are -inf before -- logaddexp(-inf, f) = f, the same bits. What a sum layer handed the product
    group below (`stream_children`) does where it computed its element flows without the fused kernels."""
    table, first = children
    ch = table[rows - first].long()
    flows = element_flows[rows]
    node_flows[ch[:, 0]] = flows
    node_flows[ch[:, 1]] = flows


def attach_element_children(group, first_row: int) -> None:
    """Build a product layer group's table (rows ``[first_row, ...)`` of its layers) and keep it as a buffer of the
    group, ``group.element_children``, with ``group.element_children_first_row``. A block-sparse product layer's node
    block ``b`` holds rows ``nids[b] + i`` with children ``cids[b, j] + i``, ``i`` within the block; a layer of single
    nodes the same with blocks of one."""
    layers = list(group)
    device = layers[0].partitioned_nids[0].device
    n = max(layer._layer_nid_range[1] for layer in layers) - first_row
    K = max(int(c.size(1)) for layer in layers for c in layer.partitioned_cids)
    # int32, as the layers' own node and element ids: the kernels index the table and `node_mars` with 32-bit
    # arithmetic (MEASURED: 64-bit addresses cost the fused forward 48 spilled registers, 7-20% of its time)
    if n * K >= 2 ** 31:
        raise ValueError(f"The children table of a product layer group ({n} x {K}) exceeds 32-bit indexing.")
    table = torch.zeros([n, K], dtype = torch.int32, device = device)
    for layer in layers:
        bs = layer.block_size
        within = torch.arange(bs, device = device, dtype = torch.int64)
        for nids, cids in zip(layer.partitioned_nids, layer.partitioned_cids):
            nids, cids = nids.to(torch.int64), cids.to(torch.int64)
            rows = (nids[:, None] + within[None, :]).reshape(-1) - first_row                     # [blocks * bs]
            children = torch.where(cids[:, None, :] > 0, cids[:, None, :] + within[None, :, None], 0)
            table[rows, :cids.size(1)] = children.reshape(-1, cids.size(1)).to(torch.int32)     # the rest stays 0
    group.register_buffer("element_children", table, persistent = False)
    group.element_children_first_row = first_row
