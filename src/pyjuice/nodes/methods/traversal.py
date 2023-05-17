from __future__ import annotations

from typing import Callable, Optional, Dict

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes


def foreach(func: Callable, root_nodes: CircuitNodes):

    visited = set()

    def dfs(ns: CircuitNodes):
        if ns in visited:
            return

        visited.add(ns)

        # Recursively traverse children
        if ns.issum() or ns.isprod():
            for cs in ns.chs:
                dfs(cs)

        func(ns)

    dfs(root_nodes)

    return None


def foldup_aggregate(func: Callable, root_nodes: CircuitNodes, cache: Optional[Dict] = None):

    if cache is None:
        cache = dict()

    def dfs(ns: CircuitNodes):
        if ns in cache:
            return

        # Recursively traverse children
        if ns.issum() or ns.isprod():
            for cs in ns.chs:
                dfs(cs)

        ch_outputs = [cache[cs] for cs in ns.chs]
        ns_output = func(ns, ch_outputs)

        cache[ns] = ns_output

    dfs(root_nodes)

    return cached_outputs[root_nodes]