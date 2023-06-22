from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Dict

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes


def foreach(func: Callable, root_nodes: CircuitNodes):

    visited = set()

    def dfs(ns: CircuitNodes):
        if ns in visited:
            return

        visited.add(ns)

        # Recursively traverse children
        if ns.is_sum() or ns.is_prod():
            for cs in ns.chs:
                dfs(cs)

        func(ns)

    dfs(root_nodes)

    return None


def foldup_aggregate(func: Callable, root_nodes: CircuitNodes, cache: Optional[Dict] = None, **kwargs):

    if cache is None:
        cache = dict()

    if len(kwargs) > 0:
        func = partial(func, **kwargs)

    def dfs(ns: CircuitNodes):
        if ns in cache:
            return

        # Recursively traverse children
        if ns.is_sum() or ns.is_prod():
            for cs in ns.chs:
                dfs(cs)

        ch_outputs = [cache[cs] for cs in ns.chs]
        ns_output = func(ns, ch_outputs)

        cache[ns] = ns_output

    dfs(root_nodes)

    return cache[root_nodes]