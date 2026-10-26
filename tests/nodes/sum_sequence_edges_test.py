"""Per-child sum edges use local block IDs without mutating caller tensors."""

import pytest
import torch

from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes.distributions import Categorical


def _children(block_counts, block_size = 1):
    return [multiply(inputs(0, num_node_blocks = count, block_size = block_size,
                            dist = Categorical(num_cats = 2)))
            for count in block_counts]


@pytest.mark.parametrize("sequence_type", [list, tuple], ids = ["list", "tuple"])
@pytest.mark.parametrize("block_size", [1, 4])
def test_per_child_edges_match_explicit_global_ids(sequence_type, block_size):
    children = _children((2, 3), block_size)
    local = [torch.tensor([[1, 0, 1], [1, 0, 1]]),
             torch.tensor([[0, 1, 0], [2, 0, 1]])]
    before = [edges.clone() for edges in local]
    # Parent IDs, caller order and the repeated first-child edge are retained.
    expected = torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 4, 2, 3]])
    reference = summate(*children, num_node_blocks = 2, block_size = block_size,
                        edge_ids = expected)

    for _ in range(2):
        actual = summate(*children, num_node_blocks = 2, block_size = block_size,
                         edge_ids = sequence_type(local))
        assert torch.equal(actual.edge_ids, reference.edge_ids)
        assert torch.equal(actual.edge_ids, expected)
        assert all(torch.equal(edges, original) for edges, original in zip(local, before))


def test_shared_edge_tensor_can_be_reused_across_children_and_constructions():
    children = _children((2, 3))
    shared = torch.tensor([[0, 1], [1, 0]])
    before = shared.clone()
    expected = torch.tensor([[0, 1, 0, 1], [1, 0, 3, 2]])

    for _ in range(2):
        actual = summate(*children, num_node_blocks = 2, block_size = 1,
                         edge_ids = [shared, shared])
        assert torch.equal(actual.edge_ids, expected)
        assert torch.equal(shared, before)


def test_child_with_no_edges_still_contributes_its_block_offset():
    children = _children((2, 3))
    empty = torch.empty((2, 0), dtype = torch.long)
    second = torch.tensor([[0, 1, 0], [2, 1, 0]])
    before = second.clone()

    actual = summate(*children, num_node_blocks = 2, block_size = 1,
                     edge_ids = [empty, second])

    assert torch.equal(actual.edge_ids, torch.tensor([[0, 1, 0], [4, 3, 2]]))
    assert empty.shape == (2, 0)
    assert torch.equal(second, before)


@pytest.mark.parametrize("bad_id", [-1, 3])
def test_local_child_indices_are_checked_before_global_offset(bad_id):
    children = _children((2, 3, 4))
    local = [torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0], [bad_id]]),
             torch.empty((2, 0), dtype = torch.long)]
    before = [edges.clone() for edges in local]
    error = "underflow" if bad_id < 0 else "overflow"

    with pytest.raises(AssertionError, match = "Edge index " + error):
        summate(*children, num_node_blocks = 2, block_size = 1, edge_ids = local)
    assert all(torch.equal(edges, original) for edges, original in zip(local, before))

    # A valid local ID still permits the trailing child to have no edges.
    valid = [local[0], torch.tensor([[0], [0]]), local[2]]
    result = summate(*children, num_node_blocks = 2, block_size = 1, edge_ids = valid)
    assert result.edge_ids.shape == (2, 3)
    assert all(torch.equal(edges, original) for edges, original in zip(local, before))


def test_noncontiguous_edge_views_leave_their_backing_tensors_unchanged():
    children = _children((2, 3))
    backing = [torch.tensor([[1, 99, 0, 99, 1, 99], [1, 99, 0, 99, 1, 99]]),
               torch.tensor([[0, 99, 1, 99, 0, 99], [2, 99, 0, 99, 1, 99]])]
    before = [tensor.clone() for tensor in backing]
    local = [tensor[:, ::2] for tensor in backing]
    assert all(not edges.is_contiguous() for edges in local)

    actual = summate(*children, num_node_blocks = 2, block_size = 1, edge_ids = local)

    assert torch.equal(actual.edge_ids,
                       torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 4, 2, 3]]))
    assert all(torch.equal(tensor, original) for tensor, original in zip(backing, before))
