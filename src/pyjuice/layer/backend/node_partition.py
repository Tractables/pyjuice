import torch
import numpy as np
from numba import njit
import math

from typing import Union, Optional


@njit()
def _partition_nodes_dp_simple_compiled(nodes_n_edges, dp, backtrace, max_num_groups, target_overhead):
    num_nodes = nodes_n_edges.shape[0]

    # Init
    for i in range(num_nodes):
        dp[i,1] = nodes_n_edges[i] * (i + 1)

    # Main DP
    target_n_group = max_num_groups
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

        if dp[-1,n_group] < target_overhead:
            target_n_group = n_group
            break

    overhead = dp[-1,target_n_group]

    return overhead, target_n_group

@njit
def _backtrace_fn(partitions, backtrace, target_n_group, num_nodes):
    i = num_nodes - 1
    for n in range(target_n_group, 0, -1):
        partitions[n-1] = i
        i = backtrace[i,target_n_group]


def _partition_nodes_dp_simple(nodes_n_edges: np.ndarray, dp: np.ndarray, backtrace: np.ndarray, max_num_groups: int, 
                               target_overhead: Optional[int]):

    overhead, target_n_group = _partition_nodes_dp_simple_compiled(
        np.ascontiguousarray(nodes_n_edges), 
        np.ascontiguousarray(dp), 
        np.ascontiguousarray(backtrace),
        max_num_groups,
        target_overhead = 0 if target_overhead is None else target_overhead
    )

    # Backtrace
    partitions = np.zeros([target_n_group], dtype = np.int64)
    num_nodes = nodes_n_edges.shape[0]
    _backtrace_fn(partitions, backtrace, target_n_group, num_nodes)

    return np.unique(nodes_n_edges[partitions]), overhead


def partition_nodes_by_n_edges(node_n_edges: Union[np.ndarray, torch.Tensor], 
                               max_num_groups: Optional[int] = None, 
                               sparsity_tolerance: Optional[float] = None,
                               algorithm: str = "dp_simple"):

    if sparsity_tolerance is not None and sparsity_tolerance < 1e-6:
        sparsity_tolerance = None
        max_num_groups = 1

    if sparsity_tolerance is not None:
        assert sparsity_tolerance > 1e-6 and sparsity_tolerance <= 1.0
        if max_num_groups is None:
            max_num_groups = max(min(int(math.ceil(node_n_edges.shape[0] * sparsity_tolerance)), 16), 1)
    elif max_num_groups is None:
        max_num_groups = 1
    else:
        assert max_num_groups >= 1, "Should provide at least 1 group."

    if isinstance(node_n_edges, torch.Tensor):
        node_n_edges = node_n_edges.detach().cpu().numpy()

    total_num_edges = node_n_edges.sum()
    target_overhead = None if sparsity_tolerance is None else int(math.ceil(total_num_edges / sparsity_tolerance))

    if max_num_groups == 1:
        partitions = np.zeros([1], dtype = np.int64)
        partitions[0] = np.max(node_n_edges)
        return torch.from_numpy(partitions)

    # Sort in non-descending order
    node_n_edges = np.sort(node_n_edges)

    if algorithm == "dp_simple":
        dp = np.zeros([node_n_edges.shape[0], max_num_groups + 1], dtype = np.int64)
        backtrace = np.zeros([node_n_edges.shape[0], max_num_groups + 1], dtype = np.int64)
        group_sizes, overhead = _partition_nodes_dp_simple(node_n_edges, dp, backtrace, max_num_groups, target_overhead)
    else:
        raise ValueError(f"Unknown algorithm {algorithm} for `partition_nodes_by_n_edges`.")

    if isinstance(group_sizes, np.ndarray):
        group_sizes = torch.from_numpy(group_sizes)

    return torch.sort(group_sizes).values