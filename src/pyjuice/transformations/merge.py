from __future__ import annotations

import torch
from functools import partial
from typing import Callable, Optional, Dict, Sequence

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from pyjuice.graph import RegionGraph, PartitionNode, InnerRegionNode, InputRegionNode


def _resolved_params(ns: CircuitNodes) -> Optional[torch.Tensor]:
    """
    The parameters of `ns`, looked up through its source node when `ns` is tied.

    A tied node deliberately stores none of its own, so reading `ns._params` directly yields `None`
    and a node rebuilt from it would come out unparameterized -- and then be randomly initialized at
    compile time, silently changing the PC. A tied node shares the source's parameter block
    edge-for-edge, so the source's tensor is the one to carry over.
    """
    source_ns = ns.get_source_ns()

    return getattr(source_ns, "_params", None)


def _merged_reference_ns(all_ns: Sequence[CircuitNodes]) -> CircuitNodes:
    """
    Pick the node whose concrete type and subclass configuration the merged node inherits.

    Merging is only well defined if the inputs agree on both, since the merged node has to be one
    node of one type; disagreement is reported rather than silently resolved to the base class.
    """
    ref_ns = all_ns[0]
    ref_kwargs = ref_ns._construction_kwargs()
    for ns in all_ns[1:]:
        assert type(ns) == type(ref_ns) and ns._construction_kwargs() == ref_kwargs, \
            f"Cannot merge nodes with different types or configurations: {type(ref_ns).__name__}" \
            f"{ref_kwargs} vs {type(ns).__name__}{ns._construction_kwargs()}."

    return ref_ns


def merge_sum_nodes(ns1: SumNodes, ns2: SumNodes, *args) -> SumNodes:
    all_ns = [ns1, ns2, *args]
    for idx, ns in enumerate(all_ns):
        assert ns1.scope == ns.scope, "Sum nodes to be merged should have the same scope."
        assert ns1.block_size == ns.block_size, "To-be-merged sum nodes must have the same block size."
        if not isinstance(ns, SumNodes):
            edge_ids = torch.arange(0, ns.num_node_blocks).unsqueeze(0).repeat(2, 1)
            block_size = ns.block_size
            params = torch.eye(ns.block_size).unsqueeze(0).repeat(ns.num_node_blocks, 1, 1)
            # A fresh passthrough node over a non-sum input: there is no source sum node to inherit a
            # type or configuration from, so this is a plain `SumNodes` by construction. If the other
            # to-be-merged nodes are of a different sum-node type, `_merged_reference_ns` reports it.
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

    # A merged node is one node, so it cannot stay tied to the several sources its inputs had; the
    # parameters are carried over instead, which keeps the merged PC equivalent (it just no longer
    # shares them with the sources).
    all_params = [_resolved_params(ns) for ns in all_ns]
    if all([params is not None for params in all_params]):
        params = torch.cat(all_params, dim = 0)
    else:
        params = None


    return _merged_reference_ns(all_ns).rebuild(num_node_blocks, sum_chs, edge_ids, params = params,
                                                block_size = ns1.block_size)


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

    return _merged_reference_ns(all_ns).rebuild(num_node_blocks, new_sum_chs, edge_ids, block_size = ns1.block_size)


def merge_by_region_node(root_ns: CircuitNodes) -> CircuitNodes:

    # Longest-path depth of every node, counting from the input nodes. `root_ns` iterates in
    # post-order, so every node's children are already assigned when it is reached.
    ns2depth = dict()
    for ns in root_ns:
        ns2depth[ns] = 0 if ns.is_input() else max(ns2depth[cs] for cs in ns.chs) + 1

    # Group the nodes that will be merged into one. `RegionGraph.__hash__` is derived purely from
    # scopes, so it does NOT separate nodes that sit over the same scope at different depths -- as
    # happens whenever a PC stacks layers over one scope. Grouping on the hash alone would then put a
    # node and one of its own descendants in the same group, which is both an invalid merge (they are
    # different layers, and may even differ in block size) and unorderable: the groups would depend on
    # each other cyclically, so no processing order exists. Keying on (region, depth) as well keeps
    # such nodes apart. It is a refinement, so PCs whose regions each live at a single depth -- the
    # layered PCs this function is normally applied to -- group exactly as before.
    rg2nodes = dict()
    rgs_list = list()
    for ns in root_ns:
        rg = ns.region_node
        rg_key = (hash(rg), ns2depth[ns])
        if rg_key in rg2nodes:
            rg2nodes[rg_key].append(ns)
        else:
            rg2nodes[rg_key] = [ns]

            rgs_list.append((rg_key, rg))

    # Process the groups shallowest-first, which is a topological order of the groups: every node's
    # children have a strictly smaller depth, so their group is guaranteed to be mapped already. The
    # sort is stable, so groups of equal depth keep the order they were discovered in.
    rgs_list.sort(key = lambda rg_key_and_rg: rg_key_and_rg[0][1])

    ns_old2new = dict()
    for rg_key, rg in rgs_list:
        if isinstance(rg, InputRegionNode):
            for ns in rg2nodes[rg_key]:
                ns_old2new[ns] = (ns, (0, ns.num_node_blocks))
        elif isinstance(rg, PartitionNode):
            prod_ns = []
            for ns in rg2nodes[rg_key]:
                chs = []
                edge_ids = ns.edge_ids.clone()
                for scope_id, cs in enumerate(ns.chs):
                    new_cs, (sid, eid) = ns_old2new[cs]
                    edge_ids[:,scope_id] += sid
                    chs.append(new_cs)

                prod_ns.append(ns.rebuild(ns.num_node_blocks, chs, edge_ids, block_size = ns.block_size))

            if len(prod_ns) == 1:
                new_ns = prod_ns[0]
            else:
                new_ns = merge_prod_nodes(*prod_ns)
            sid = 0
            for ns in rg2nodes[rg_key]:
                nid = sid + ns.num_node_blocks
                ns_old2new[ns] = (new_ns, (sid, nid))
                sid = nid

        elif isinstance(rg, InnerRegionNode):
            sum_ns = []
            for ns in rg2nodes[rg_key]:
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

                sum_ns.append(ns.rebuild(ns.num_node_blocks, chs, edge_ids, params = _resolved_params(ns),
                                         block_size = ns.block_size))

            if len(sum_ns) == 1:
                new_ns = sum_ns[0]
            else:
                new_ns = merge_sum_nodes(*sum_ns)
            sid = 0
            for ns in rg2nodes[rg_key]:
                nid = sid + ns.num_node_blocks
                ns_old2new[ns] = (new_ns, (sid, nid))
                sid = nid

    _restore_tying(rgs_list, rg2nodes, ns_old2new)

    return ns_old2new[root_ns][0]


def _restore_tying(rgs_list, rg2nodes, ns_old2new) -> None:
    """
    Re-establish parameter tying on the merged nodes.

    Each group collapses into a single node, so a tie only survives when the whole group is tied and
    its sources were themselves rebuilt into a single node that lines up with it block-for-block.
    Where that holds the merged node is tied and drops its own parameters; where it does not, it
    keeps the parameters resolved from its sources, so the PC stays equivalent but no longer shares
    them. Either way the parameters are never lost.

    Input nodes are reused as-is rather than rebuilt, so their tying is already intact and is skipped.
    """

    for rg_key, rg in rgs_list:
        if isinstance(rg, InputRegionNode):
            continue

        nodes = rg2nodes[rg_key]
        if not all([ns.is_tied() for ns in nodes]):
            continue

        source_nss = [ns.get_source_ns() for ns in nodes]
        if any([source_ns not in ns_old2new for source_ns in source_nss]):
            # Tied to something outside this PC, which the merge cannot speak about
            continue

        new_ns = ns_old2new[nodes[0]][0]
        new_source_nss = {ns_old2new[source_ns][0] for source_ns in source_nss}
        if len(new_source_nss) != 1:
            # The group's sources ended up in different merged nodes
            continue

        new_source_ns = next(iter(new_source_nss))

        # The tie is only meaningful if the two merged nodes agree block-for-block: same type and
        # configuration, same shape, same edges, and every member sitting at the same offset within
        # its merged node as its source does within the merged source node.
        if new_source_ns is new_ns or type(new_source_ns) is not type(new_ns) or \
                new_source_ns._construction_kwargs() != new_ns._construction_kwargs() or \
                new_source_ns.num_node_blocks != new_ns.num_node_blocks or \
                new_source_ns.block_size != new_ns.block_size or \
                new_source_ns.num_chs != new_ns.num_chs or \
                not torch.equal(new_source_ns.edge_ids, new_ns.edge_ids) or \
                any([ns_old2new[ns][1] != ns_old2new[source_ns][1] for ns, source_ns in zip(nodes, source_nss)]):
            continue

        # A tied node holds no parameters of its own; they live on the source
        new_ns._params = None
        new_ns.set_source_ns(new_source_ns.get_source_ns())


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