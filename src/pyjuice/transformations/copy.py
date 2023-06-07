from __future__ import annotations

from copy import deepcopy

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes


def deepcopy(root_nodes: CircuitNodes):
    old2new = dict()
    tied_ns_pairs = []

    def dfs(ns: CircuitNodes):
        if ns in cache:
            return

        # Recursively traverse children
        if ns.issum() or ns.isprod():
            for cs in ns.chs:
                dfs(cs)

        new_chs = [cache[cs] for cs in ns.chs]

        if ns.is_tied():
            tied_ns_pairs.append((ns, ns.get_source_ns()))

        if ns.issum():
            new_ns = SumNodes(
                ns.num_nodes,
                new_chs,
                ns.edge_ids.clone()
            )
            params = ns.get_params()
            if params is not None:
                new_ns.set_params(params.clone())
        elif ns.isprod():
            new_ns = ProdNodes(
                ns.num_nodes,
                new_chs,
                ns.edge_ids.clone()
            )
        else:
            assert ns.isinput()
            new_ns = InputNodes(
                ns.num_nodes,
                deepcopy(ns.scope),
                deepcopy(ns.dist)
            )
            params = ns.get_params()
            if params is not None:
                new_ns.set_params(params.clone())

        old2new[ns] = new_ns

    dfs(root_nodes)

    for ns, source_ns in tied_ns_pairs:
        new_ns = old2new[ns]
        new_source_ns = old2new[source_ns]

        new_ns._source_node = new_source_ns

    return old2new[root_nodes]