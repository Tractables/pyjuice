from __future__ import annotations

import torch
from copy import deepcopy as pydeepcopy
from typing import Optional, Dict, Sequence

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, foldup_aggregate
from pyjuice.utils import BitSet
from pyjuice.utils.util import max_cdf_power_of_2


def group(root_ns: CircuitNodes, sparsity_tolerance: float = 0.25, max_target_group_size: int = 32):

    ## Do an initial pass to compute the maximum group size of every `ns` ##

    ns2group_size = dict()
    for ns in root_ns:
        if ns.is_input():
            ns2group_size[ns] = min(max_cdf_power_of_2(ns.num_nodes), max_target_group_size)

        elif ns.is_prod():
            ns2group_size[ns] = min(max_cdf_power_of_2(ns.num_nodes), max_target_group_size)

        else:
            assert ns.is_sum()

            old_group_size = ns.group_size
            old_ch_group_size = ns.ch_group_size
            edge_ids = ns.edge_ids

            old_ns_num_ngroups = ns.num_node_groups
            old_cs_num_ngroups = sum([cs.num_node_groups for cs in ns.chs])

            flag = False
            plausible_combinations = list()

            group_size = min(max_cdf_power_of_2(ns.num_nodes), max_target_group_size)
            while group_size >= old_group_size:
                group_mul_size = group_size // old_group_size

                ns_num_ngroups = old_ns_num_ngroups // group_mul_size
                
                ch_group_size = ns2group_size[ns.chs[0]]
                while ch_group_size >= old_ch_group_size:
                    ch_group_mul_size = ch_group_size // old_ch_group_size

                    cs_num_ngroups = old_cs_num_ngroups // group_mul_size

                    n_edge_ids = edge_ids[0,:] // group_mul_size
                    c_edge_ids = edge_ids[1,:] // ch_group_mul_size
                    _, counts = torch.unique(n_edge_ids * cs_num_ngroups + c_edge_ids, return_counts = True)

                    if counts.float().mean() >= (1.0 - sparsity_tolerance) * group_mul_size * ch_group_mul_size:
                        plausible_combinations.append((group_size, ch_group_size))

                    ch_group_size = ch_group_size // 2

                group_size = group_size // 2

            # Find the best group size combination
            best_group_size = 0
            best_ch_group_size = 0
            for group_size, ch_group_size in plausible_combinations:
                if group_size >= 16 and ch_group_size >= 16:
                    best_group_size = group_size
                    best_ch_group_size = ch_group_size
                    break

            if best_group_size == 0:
                best_val = 0
                best_frac = 0
                for group_size, ch_group_size in plausible_combinations:
                    cond1 = group_size * ch_group_size > best_val
                    cond2 = (group_size * ch_group_size > best_val) and \
                        (max(group_size, ch_group_size) // min(group_size, ch_group_size) < best_frac)
                    if cond1 or cond2:
                        best_group_size = group_size
                        best_ch_group_size = ch_group_size
                        best_val = group_size * ch_group_size
                        best_frac = max(group_size, ch_group_size) // min(group_size, ch_group_size)

            ns2group_size[ns] = best_group_size
            for cs in ns.chs:
                ns2group_size[cs] = best_ch_group_size

    ## Do a second pass to finalize the group sizes ##

    for ns in root_ns:
        if ns.is_prod():
            group_size = ns2group_size[ns]
            for cs in ns.chs:
                group_size = min(group_size, ns2group_size[cs])

            ns2group_size[ns] = group_size
            for cs in ns.chs:
                ns2group_size[cs] = group_size

    ## Apply the new group sizes ##

    def update_ns(ns: CircuitNodes, ns_chs: Sequence[CircuitNodes]):
        new_group_size = ns2group_size[ns]
        group_mul_size = new_group_size // ns.group_size

        new_num_ngroups = ns.num_node_groups // group_mul_size

        assert new_num_ngroups * new_group_size == ns.num_node_groups * ns.group_size

        if ns.is_input():
            new_ns = InputNodes(
                num_node_groups = new_num_ngroups,
                scope = pydeepcopy(ns.scope),
                dist = pydeepcopy(ns.dist),
                group_size = new_group_size
            )

            if not ns.is_tied():
                params = ns.get_params()
                if params is not None:
                    new_ns.set_params(params.clone(), normalize = False)

        elif ns.is_prod():
            edge_ids = ns.edge_ids.clone()
            edge_ids = edge_ids.reshape(new_num_ngroups, group_mul_size, ns.num_chs)
            if torch.all(edge_ids[:,1:,:] - edge_ids[:,:-1,:]) == 1:
                # Block-sparse mode
                edge_ids = edge_ids[:,0,:].contiguous()
                mode = "block_sparse"
            else:
                # Sparse mode
                edge_ids = (edge_ids.reshape(ns.num_node_groups, ns.num_chs)[:,None,:] * ns.group_size + \
                    torch.arange(0, ns.group_size)[None,:,None]).flatten(0, 1)
                mode = "sparse"
            
            new_ns = ProdNodes(
                num_node_groups = new_num_ngroups,
                chs = ns_chs,
                edge_ids = edge_ids,
                group_size = new_group_size
            )

            if mode == "block_sparse":
                assert new_ns.is_block_sparse()
            elif mode == "sparse":
                assert new_ns.is_sparse()

        else:
            assert ns.is_sum()

            old_num_ngroups = ns.num_node_groups
            old_num_cgroups = sum([cs.num_node_groups for cs in ns.chs])

            new_ch_group_size = ns2group_size[ns.chs[0]]
            ch_group_mul_size = new_ch_group_size // ns.chs[0].group_size

            new_num_cgroups = old_num_cgroups // ch_group_mul_size

            edge_ids = ns.edge_ids.clone()
            grid_edge_ids = torch.zeros([old_num_ngroups, old_num_cgroups], dtype = torch.bool)
            grid_edge_ids[edge_ids[0,:],edge_ids[1,:]] = True

            grid_edge_ids = grid_edge_ids.reshape(new_num_ngroups, group_mul_size, new_num_cgroups, ch_group_mul_size)
            new_edge_ids = torch.nonzero(grid_edge_ids.any(dim = 3).any(dim = 1), as_tuple = False).permute(1, 0)

            new_ns = SumNodes(
                num_node_groups = new_num_ngroups,
                chs = ns_chs,
                edge_ids = new_edge_ids,
                group_size = new_group_size
            )
            
            if not ns.is_tied():
                # Collect selected blocks
                grid_edge_ids = grid_edge_ids.permute(0, 2, 1, 3).flatten(0, 1)
                block_ids = new_edge_ids[0,:] * new_num_cgroups + new_edge_ids[1,:]
                param_indicator = grid_edge_ids[block_ids,:,:]
                if not torch.all(param_indicator):
                    param_indicator = param_indicator[:,:,None,:,None].repeat(1, 1, ns.group_size, 1, ns.chs[0].group_size)
                    param_indicator = param_indicator.flatten(3, 4).flatten(1, 2)
                    zero_param_mask = ~param_indicator

                    new_ns.set_zero_param_mask(zero_param_mask)

                params = ns.get_params()
                if params is not None:
                    # TODO: add a GPU implementation
                    new_params = torch.zeros([new_edge_ids.size(1), new_group_size, new_ch_group_size])
                    for par_group_id in range(new_edge_ids.size(1)):
                        nsid = new_edge_ids[0,par_group_id] * group_mul_size
                        neid = nsid + group_mul_size
                        csid = new_edge_ids[1,par_group_id] * ch_group_mul_size
                        ceid = csid + ch_group_mul_size

                        blk_ids = torch.where((edge_ids[0,:] >= nsid) & (edge_ids[0,:] < neid) & (edge_ids[1,:] >= csid) & (edge_ids[1,:] < ceid))[0]
                        for blk_id in blk_ids:
                            nid0, nid1 = (edge_ids[0,blk_id] - nsid) * ns.group_size, (edge_ids[0,blk_id] - nsid + 1) * ns.group_size
                            cid0, cid1 = (edge_ids[1,blk_id] - csid) * ns.chs[0].group_size, (edge_ids[1,blk_id] - csid + 1) * ns.chs[0].group_size
                            new_params[par_group_id,nid0:nid1,cid0:cid1] = params[blk_id,:,:]

                    new_ns.set_params(new_params, normalize = False)

        return new_ns

    old2new = dict()
    new_root_ns = foldup_aggregate(update_ns, root_ns, cache = old2new)

    # Re-link tied nodes to their source
    for ns in root_ns:
        if ns.is_tied():
            new_source_ns = old2new[ns.get_source_ns()]
            new_ns = old2new[ns]
            new_ns.set_source_ns(new_source_ns)

    return new_root_ns
