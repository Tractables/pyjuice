from __future__ import annotations

import torch
import pickle
from functools import partial
from typing import Sequence

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, inputs, multiply, summate


def serialize_nodes(root_ns: CircuitNodes):
    nodes_list = list()
    ns2id = dict()
    for ns in root_ns:
        if ns.is_input():
            ntype = "Input"
        elif ns.is_prod():
            ntype = "Product"
        else:
            assert ns.is_sum()
            ntype = "Sum"

        ns_info = {
            "type": ntype, 
            "num_node_blocks": ns.num_node_blocks,
            "block_size": ns.block_size,
            "chs": tuple(ns2id[cs] for cs in ns.chs)
        }

        if ns.is_prod() or ns.is_sum():
            ns_info["edge_ids"] = ns.edge_ids.detach().cpu().numpy().copy()

        if hasattr(ns, "_params") and ns._params is not None:
            ns_info["params"] = ns._params.detach().cpu().numpy().copy()

        if hasattr(ns, "_zero_param_mask") and ns._zero_param_mask is not None:
            ns_info["zero_param_mask"] = ns._zero_param_mask.detach().cpu().numpy().copy()

        if ns.is_input():
            ns_info["scope"] = ns.scope.to_list()
            ns_info["dist"] = pickle.dumps(ns.dist)

        ns2id[ns] = len(nodes_list)
        nodes_list.append(ns_info)

    for ns in root_ns:
        # Tied nodes
        if hasattr(ns, "_source_node") and ns._source_node is not None:
            nodes_list[ns2id[ns]]["source_node"] = ns2id[ns._source_node]

    return nodes_list


def deserialize_nodes(nodes_list: Sequence):
    id2ns = dict()
    for ns_id, ns_info in enumerate(nodes_list):
        num_node_blocks = ns_info["num_node_blocks"]
        block_size = ns_info["block_size"]
        chids = ns_info["chs"]

        if ns_info["type"] == "Input":
            scope = ns_info["scope"]
            dist = pickle.loads(ns_info["dist"])

            ns = inputs(scope, num_node_blocks, dist, block_size = block_size, 
                        _no_set_meta_params = dist.need_meta_parameters)

            if "params" in ns_info:
                if dist.need_meta_parameters:
                    ns._params = torch.from_numpy(ns_info["params"])
                else:
                    ns.set_params(torch.from_numpy(ns_info["params"]), normalize = False)

        elif ns_info["type"] == "Product":
            chs = [id2ns[cid] for cid in chids]
            edge_ids = torch.from_numpy(ns_info["edge_ids"])

            if edge_ids.size(0) == chs[0].num_node_blocks:
                sparse_edges = False
            else:
                sparse_edges = True

            ns = multiply(*chs, edge_ids = edge_ids, sparse_edges = sparse_edges)

        else:
            assert ns_info["type"] == "Sum"

            chs = [id2ns[cid] for cid in chids]
            edge_ids = ns_info["edge_ids"]
            if "params" in ns_info:
                params = torch.from_numpy(ns_info["params"])
            else:
                params = None

            ns = summate(*chs, edge_ids = edge_ids, params = params, block_size = block_size)

            if "zero_param_mask" in ns_info:
                zero_param_mask = torch.from_numpy(ns_info["zero_param_mask"])
                ns.set_zero_param_mask(zero_param_mask)

        id2ns[ns_id] = ns

    for ns_id, ns_info in enumerate(nodes_list):
        if "source_node" in ns_info:
            ns = id2ns[ns_id]
            ns._source_node = id2ns[ns_info["source_node"]]

    return id2ns[len(nodes_list) - 1]
