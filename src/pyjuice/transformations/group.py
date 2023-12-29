from __future__ import annotations

from typing import Optional, Dict, Sequence

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
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
            old_cs_group_size = ns.cs_group_size
            edge_ids = ns.edge_ids

            old_ns_num_ngroups = ns.num_node_groups
            old_cs_num_ngroups = sum([cs.num_node_groups for cs in ns.chs])

            flag = False
            plausible_combinations = list()

            group_size = min(max_cdf_power_of_2(ns.num_nodes), max_target_group_size)
            while group_size > old_group_size:
                group_mul_size = group_size // old_group_size

                ns_num_ngroups = old_ns_num_ngroups // group_mul_size
                
                cs_group_size = ns2group_size[ns.chs[0]]
                while cs_group_size > old_cs_group_size:
                    cs_group_mul_size = cs_group_size // old_cs_group_size

                    cs_num_ngroups = old_cs_num_ngroups // group_mul_size

                    n_edge_ids = edge_ids[0,:] // group_mul_size
                    c_edge_ids = edge_ids[1,:] // cs_group_mul_size
                    _, counts = torch.unique(n_edge_ids * cs_num_ngroups + c_edge_ids, return_counts = True)

                    if torch.all(counts >= (1.0 - sparsity_tolerance) * group_mul_size * cs_group_mul_size):
                        plausible_combinations.append((group_size, cs_group_size))

                    cs_group_size = cs_group_size // 2

                group_size = group_size // 2

            # Find the best group size combination
            best_group_size = 0
            best_cs_group_size = 0
            for group_size, cs_group_size in plausible_combinations:
                if group_size >= 16 and cs_group_size >= 16:
                    best_group_size = group_size
                    best_cs_group_size = cs_group_size
                    break

            if best_group_size == 0:
                best_val = 0
                best_frac = 0
                for group_size, cs_group_size in plausible_combinations:
                    cond1 = group_size * cs_group_size > best_val
                    cond2 = (group_size * cs_group_size > best_val) and \
                        (max(group_size, cs_group_size) // min(group_size, cs_group_size) < best_frac)
                    if cond1 or cond2:
                        best_group_size = group_size
                        best_cs_group_size = cs_group_size
                        best_val = group_size * cs_group_size
                        best_frac = max(group_size, cs_group_size) // min(group_size, cs_group_size)

            ns2group_size[ns] = best_group_size
            for cs in ns.chs:
                ns2group_size[cs] = best_cs_group_size

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
        if ns.isinput():
            pass

        elif ns.isprod():
            pass

        else:
            assert ns.issum()

    return foldup_aggregate(update_ns, root_ns)
