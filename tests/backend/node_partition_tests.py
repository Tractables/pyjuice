import pyjuice as juice
import torch
import numpy as np

from pyjuice.layer.backend.node_partition import partition_nodes_by_n_edges

import pytest


def node_partition_tests():
    node_n_edges = torch.tensor([1, 1, 2, 2, 10, 10, 13])
    partitions = partition_nodes_by_n_edges(node_n_edges, max_num_groups = 2, algorithm = "dp_simple")

    assert partitions[0] == 2 and partitions[1] == 13

    node_n_edges = torch.tensor([2, 3, 4, 1, 4, 8, 9, 13, 15, 11])
    partitions = partition_nodes_by_n_edges(node_n_edges, max_num_groups = 2, algorithm = "dp_simple")

    assert partitions[0] == 4 and partitions[1] == 15

    partitions = partition_nodes_by_n_edges(node_n_edges, max_num_groups = 3, algorithm = "dp_simple")

    assert partitions[0] == 4 and partitions[1] == 9 and partitions[2] == 15


if __name__ == "__main__":
    node_partition_tests()