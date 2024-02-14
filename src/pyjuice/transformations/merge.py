from __future__ import annotations

import torch
from functools import partial
from typing import Callable, Optional, Dict

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode


def merge_sum_nodes(ns1: SumNodes, ns2: SumNodes, *args) -> SumNodes:
    all_ns = [ns1, ns2, *args]
    for idx, ns in enumerate(all_ns):
        assert ns1.scope == ns.scope, "Sum nodes to be merged should have the same scope."
        assert ns1.block_size == ns.block_size, "To-be-merged sum nodes must have the same block size."
        if not isinstance(ns, SumNodes):
            edge_ids = torch.arange(0, ns.num_node_blocks).unsqueeze(0).repeat(2, 1)
            block_size = ns.block_size
            params = torch.eye(ns.block_size).unsqueeze(0).repeat(ns.num_node_blocks, 1, 1)
            new_ns = SumNodes(ns.num_node_blocks, [ns], edge_ids, params = params, block_size = block_size)
            all_ns[idx] = new_ns

    sum_edge_ids = []
    sum_chs = []
    cs2start_id = dict()
    ns_start_id = 0
    global_cs_start_id = 0
    ch_block_size = None
    for ns in all_ns:
        ns_end_id = ns_start_id + ns.num_node_blocks
        curr_cs_sid = 0
        edge_ids = ns.edge_ids.clone()
        for cs in ns.chs:
            if ch_block_size is None:
                ch_block_size = cs.block_size
            else:
                assert ch_block_size == cs.block_size, "Children must have the same block size."

            curr_cs_eid = curr_cs_sid + cs.num_node_blocks
            if cs in cs2start_id:
                cs_start_id = cs2start_id[cs]
            else:
                cs_start_id = global_cs_start_id

            filter = (ns.edge_ids[1,:] >= curr_cs_sid) & (ns.edge_ids[1,:] < curr_cs_eid)
            edge_ids[1,filter] += cs_start_id - curr_cs_sid

            curr_cs_sid = curr_cs_eid
            if cs not in cs2start_id:
                cs2start_id[cs] = global_cs_start_id
                global_cs_start_id += cs.num_node_blocks
                sum_chs.append(cs)

        edge_ids[0,:] += ns_start_id
        sum_edge_ids.append(edge_ids)
        
        ns_start_id = ns_end_id

    num_node_blocks = ns_start_id
    edge_ids = torch.cat(sum_edge_ids, dim = 1)
    if all([hasattr(ns, "_params") and ns._params is not None for ns in all_ns]):
        params = torch.cat([ns._params for ns in all_ns], dim = 0)
    else:
        params = None
    
    return SumNodes(num_node_blocks, sum_chs, edge_ids, params = params, block_size = ns1.block_size)


def merge_prod_nodes(ns1: ProdNodes, ns2: ProdNodes, *args) -> ProdNodes:
    all_ns = [ns1, ns2, *args]
    ch_scopes = [cs.scope for cs in ns1.chs]
    num_scopes = len(ch_scopes)
    for ns in all_ns:
        assert isinstance(ns, ProdNodes), "Inputs should all be ProdNodes."
        assert ns1.scope == ns.scope, "Product nodes to be merged should have the same scope."
        assert ns1.block_size == ns.block_size, "To-be-merged product nodes must have the same block size."
        for cs, scope in zip(ns.chs, ch_scopes):
            assert cs.scope == scope

    cs2start_id = dict()
    sum_chs = [[] for _ in range(num_scopes)]
    global_start_ids = [0 for _ in range(num_scopes)]
    ch_block_size = None
    for ns in all_ns:
        for scope_id in range(num_scopes):
            cs = ns.chs[scope_id]

            if ch_block_size is None:
                ch_block_size = cs.block_size
            else:
                assert ch_block_size == cs.block_size, "Children must have the same block size."

            if cs not in cs2start_id:
                cs2start_id[cs] = global_start_ids[scope_id]
                global_start_ids[scope_id] += cs.num_node_blocks
                sum_chs[scope_id].append(cs)

    new_sum_chs = []
    for scope_id in range(num_scopes):
        sum_ns = sum_chs[scope_id]
        if len(sum_ns) == 1:
            new_sum_chs.append(sum_ns[0])
        else:
            new_sum_chs.append(merge_sum_nodes(*sum_ns))

    prod_edge_ids = []
    use_sparse_mode = any([ns.is_sparse() for ns in all_ns])
    for ns in all_ns:
        edge_ids = ns.edge_ids.clone()
        if use_sparse_mode and ns.is_block_sparse():
            edge_ids = (edge_ids[:,None,:].repeat(1, ns.block_size, 1) * ns.block_size + torch.arange(0, ns.block_size)[None,:,None]).flatten(0, 1)
        for scope_id in range(num_scopes):
            cs = ns.chs[scope_id]
            edge_ids[:,scope_id] += cs2start_id[cs]

        prod_edge_ids.append(edge_ids)

    edge_ids = torch.cat(prod_edge_ids, dim = 0)
    num_node_blocks = edge_ids.size(0)

    return ProdNodes(num_node_blocks, new_sum_chs, edge_ids, block_size = ns1.block_size)


def merge_by_region_node(root_ns: CircuitNodes) -> CircuitNodes:

    rg2nodes = dict()
    rgs_list = list()
    rgs_set = set()
    for ns in root_ns:
        rg = ns.region_node
        rg_hash = hash(rg)
        scope = ns.scope
        if rg_hash in rgs_set:
            rg2nodes[rg_hash].append(ns)
        else:
            rg2nodes[rg_hash] = [ns]

            rgs_list.append(rg)
            rgs_set.add(rg_hash)

    ns_old2new = dict()
    for rg in rgs_list:
        rg_hash = hash(rg)
        if isinstance(rg, InputRegionNode):
            for ns in rg2nodes[rg_hash]:
                ns_old2new[ns] = (ns, (0, ns.num_node_blocks))
        elif isinstance(rg, PartitionNode):
            prod_ns = []
            for ns in rg2nodes[rg_hash]:
                chs = []
                edge_ids = ns.edge_ids.clone()
                for scope_id, cs in enumerate(ns.chs):
                    new_cs, (sid, eid) = ns_old2new[cs]
                    edge_ids[:,scope_id] += sid
                    chs.append(new_cs)

                prod_ns.append(ProdNodes(ns.num_node_blocks, chs, edge_ids, block_size = ns.block_size))

            if len(prod_ns) == 1:
                new_ns = prod_ns[0]
            else:
                new_ns = merge_prod_nodes(*prod_ns)
            sid = 0
            for ns in rg2nodes[rg_hash]:
                nid = sid + ns.num_node_blocks
                ns_old2new[ns] = (new_ns, (sid, nid))
                sid = nid

        elif isinstance(rg, InnerRegionNode):
            sum_ns = []
            for ns in rg2nodes[rg_hash]:
                chs = []
                ch2sid = dict()
                edge_ids = ns.edge_ids.clone()
                global_sid = 0
                origin_sid = 0
                for scope_id, cs in enumerate(ns.chs):
                    origin_eid = origin_sid + cs.num_node_blocks
                    new_cs, (offset_sid, offset_eid) = ns_old2new[cs]
                    if new_cs in ch2sid:
                        sid = ch2sid[new_cs]
                    else:
                        sid = global_sid

                    filter = (ns.edge_ids[1,:] >= origin_sid) & (ns.edge_ids[1,:] < origin_eid)
                    edge_ids[1,filter] += sid + offset_sid - origin_sid

                    if new_cs not in ch2sid:
                        chs.append(new_cs)
                        ch2sid[new_cs] = global_sid
                        global_sid += new_cs.num_node_blocks

                    origin_sid = origin_eid

                if hasattr(ns, "_params") and ns._params is not None:
                    params = ns._params
                else:
                    params = None
                sum_ns.append(SumNodes(ns.num_node_blocks, chs, edge_ids, params = params, block_size = ns.block_size))

            if len(sum_ns) == 1:
                new_ns = sum_ns[0]
            else:
                new_ns = merge_sum_nodes(*sum_ns)
            sid = 0
            for ns in rg2nodes[rg_hash]:
                nid = sid + ns.num_node_blocks
                ns_old2new[ns] = (new_ns, (sid, nid))
                sid = nid

    return ns_old2new[root_ns][0]


def merge(ns1: CircuitNodes, *args) -> CircuitNodes:
    """
    Merge nodes with identical region node together.

    :param ns1: the first PC node
    :type ns1: CircuitNodes

    :param args: the remaining PC nodes
    :type args: CircuitNodes

    Example::
        >>> i00 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
        >>> i01 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
        >>> i10 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
        >>> i11 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
 
        >>> m00 = multiply(i00, i10)
        >>> m01 = multiply(i01, i11)

        >>> n0 = summate(m00, num_node_blocks = num_node_blocks)
        >>> n1 = summate(m01, num_node_blocks = num_node_blocks)

        >>> n_new = pyjuice.merge(n0, n1)
    """
    if ns1.is_sum() and len(args) > 0 and args[0].is_sum():
        return merge_sum_nodes(ns1, args[0], *args[1:])
    elif ns1.is_prod() and len(args) > 0 and args[0].is_prod():
        return merge_prod_nodes(ns1, args[0], *args[1:])
    elif len(args) == 0:
        return merge_by_region_node(ns1)
    else:
        raise ValueError()