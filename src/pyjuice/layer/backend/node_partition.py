import torch
import numpy as np
import numba

from typing import Union


@numba.jit(forceobj = True)
def _partition_nodes_dp_simple(nodes_n_edges: np.ndarray, dp: np.ndarray, backtrace: np.ndarray, max_num_groups: int):
    num_nodes = nodes_n_edges.shape[0]

    # Init
    dp[:,1] = nodes_n_edges * np.arange(1, num_nodes + 1)

    # Main DP
    for n_group in range(2, max_num_groups + 1):
        dp[0,n_group] = nodes_n_edges[0]
        backtrace[0,n_group] = 0
        for i in range(1, num_nodes):
            min_overhead = 2 ** 31 - 1
            best_idx = -1
            for j in range(i):
                curr_overhead = dp[j,n_group-1] + nodes_n_edges[i] * (i - j)
                if curr_overhead < min_overhead:
                    min_overhead = curr_overhead
                    best_idx = j

            dp[i,n_group] = min_overhead
            backtrace[i,n_group] = best_idx

    # Choose number of groups
    n_group = np.min(np.where(dp[-1,1:] <= dp[-1,-1] * 1.05)[0]) + 1

    # Backtrace
    partitions = np.zeros([n_group], dtype = np.int64)
    i = num_nodes - 1
    for n in range(n_group, 0, -1):
        partitions[n-1] = i
        i = backtrace[i,n_group]

    return np.unique(nodes_n_edges[partitions])


def partition_nodes_by_n_edges(node_n_edges: Union[np.ndarray, torch.Tensor], max_num_groups: int, algorithm: str = "dp_simple"):

    assert max_num_groups >= 1, "Should provide at least 1 group."

    if isinstance(node_n_edges, torch.Tensor):
        node_n_edges = node_n_edges.detach().cpu().numpy()

    if max_num_groups == 1:
        partitions = np.zeros([1], dtype = np.int64)
        partitions[0] = np.max(node_n_edges)
        return partitions

    # Sort in non-descending order
    node_n_edges = np.sort(node_n_edges)

    if algorithm == "dp_simple":
        dp = np.zeros([node_n_edges.shape[0], max_num_groups + 1], dtype = np.int64)
        backtrace = np.zeros([node_n_edges.shape[0], max_num_groups + 1], dtype = np.int64)
        return _partition_nodes_dp_simple(node_n_edges, dp, backtrace, max_num_groups)
    else:
        raise ValueError(f"Unknown algorithm {algorithm} for `partition_nodes_by_n_edges`.")