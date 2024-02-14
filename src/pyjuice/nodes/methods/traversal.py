from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Dict

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes


def foreach(func: Callable, root_ns: CircuitNodes):
    """
    Traverse all nodes of a PC and can a specific function.

    :note: An alternative is to use the iterator directly via the `for` statement. See below for an example.

    :param func: the function to be called
    :type func: Callable

    :param root_ns: the root PC node
    :type root_ns: CircuitNodes

    Example::
        >>> for ns in root_ns: # Traverse the PC bottom-up
        ...     [do something...]
        >>> for ns in root_ns(reverse = True): # Traverse the PC top-down
        ...     [do something...]
    """

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

    dfs(root_ns)

    return None


def foldup_aggregate(func: Callable, root_ns: CircuitNodes, cache: Optional[Dict] = None, **kwargs):
    """
    Traverse all nodes of a PC bottom-up and aggregate a per-node object.

    :param func: the function to compute the per-node object
    :type func: Callable

    :param root_ns: the root PC node
    :type root_ns: CircuitNodes

    :param cache: an optional dictionary to store the per-node object for all nodes
    :type cache: Dict
    """

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

    dfs(root_ns)

    return cache[root_ns]