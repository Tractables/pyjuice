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
        if ns.isinput():
            ntype = "Input"
        elif ns.isprod():
            ntype = "Product"
        else:
            assert ns.issum()
            ntype = "Sum"

        ns_info = {
            "type": ntype, 
            "num_nodes": ns.num_nodes,
            "chs": tuple(ns2id[cs] for cs in ns.chs)
        }

        if ns.isprod() or ns.issum():
            ns_info["edge_ids"] = ns.edge_ids.detach().cpu().numpy().copy()

        if hasattr(ns, "_params") and ns._params is not None:
            ns_info["params"] = ns._params.detach().cpu().numpy().copy()

        if ns.isinput():
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
        num_nodes = ns_info["num_nodes"]
        chids = ns_info["chs"]

        if ns_info["type"] == "Input":
            scope = ns_info["scope"]
            dist = pickle.loads(ns_info["dist"])

            ns = inputs(scope, num_nodes, dist)

            if "params" in ns_info:
                ns._params = torch.from_numpy(ns_info["params"])

        elif ns_info["type"] == "Product":
            chs = [id2ns[cid] for cid in chids]
            edge_ids = ns_info["edge_ids"]

            ns = multiply(*chs, edge_ids = edge_ids)

        else:
            assert ns_info["type"] == "Sum"

            chs = [id2ns[cid] for cid in chids]
            edge_ids = ns_info["edge_ids"]
            if "params" in ns_info:
                params = torch.from_numpy(ns_info["params"])
            else:
                params = None

            ns = summate(*chs, edge_ids = edge_ids, params = params)            

        id2ns[ns_id] = ns

    for ns_id, ns_info in enumerate(nodes_list):
        if "source_node" in ns_info:
            ns = id2ns[ns_id]
            ns._source_node = id2ns[ns_info["source_node"]]

    return id2ns[len(nodes_list) - 1]
