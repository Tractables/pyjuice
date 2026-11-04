"""
Per-child sum edges (`edge_ids` a sequence of tensors, one a child) use block IDs local to each child, without mutating
the caller's tensors. An ID left in the wrong child's range is no error but a circuit with other values: MEASURED before
this fix, log p(x = 0) -0.4155 for a mixture whose value is -0.6349.
"""

import pytest
import torch

import pyjuice as juice
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


@pytest.mark.parametrize("empty_child,expected", [(0, [[0, 1, 0], [4, 3, 2]]), (1, [[0, 1, 0], [2, 1, 0]])],
                         ids = ["first", "last"])
def test_child_with_no_edges_still_contributes_its_block_offset(empty_child, expected):
    children = _children((2, 3) if empty_child == 0 else (3, 2))
    empty = torch.empty((2, 0), dtype = torch.long)
    other = torch.tensor([[0, 1, 0], [2, 1, 0]])
    before = other.clone()

    actual = summate(*children, num_node_blocks = 2, block_size = 1,
                     edge_ids = [empty, other] if empty_child == 0 else [other, empty])

    assert torch.equal(actual.edge_ids, torch.tensor(expected))
    assert empty.shape == (2, 0)
    assert torch.equal(other, before)


@pytest.mark.parametrize("child,bad_id,error", [(0, 2, "overflow"), (1, -1, "underflow"), (1, 3, "overflow")])
def test_local_child_indices_are_checked_before_global_offset(child, bad_id, error):
    """Each of these becomes a valid global ID once shifted, in another child's range: global checks cannot see it."""
    children = _children((2, 3, 4))
    local = [torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0], [0]]), torch.empty((2, 0), dtype = torch.long)]
    local[child][1, 0] = bad_id
    before = [edges.clone() for edges in local]

    with pytest.raises(AssertionError, match = "Edge index " + error):
        summate(*children, num_node_blocks = 2, block_size = 1, edge_ids = local)
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


def test_set_edges_takes_per_child_edges():
    ns = summate(*_children((2, 3)), num_node_blocks = 2, block_size = 1)
    local = [torch.tensor([[0], [1]]), torch.tensor([[1], [2]])]
    before = [edges.clone() for edges in local]

    ns.set_edges(local)

    assert torch.equal(ns.edge_ids, torch.tensor([[0, 1], [1, 4]]))
    assert all(torch.equal(edges, original) for edges, original in zip(local, before))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_per_child_edges_give_the_mixture_they_describe():
    """What a user sees: the compiled circuit's values, against the same circuit with global IDs and against float64.
    Every child block has its own distribution, so an edge to any other block changes the value."""
    dev = torch.device("cuda:0")
    first = torch.tensor([[.9, .1], [.7, .3]])
    second = torch.tensor([[.2, .8], [.4, .6], [.6, .4]])
    weights = torch.tensor([.5, .3, .2])

    def values(edge_ids):
        with juice.set_block_size(1):
            a = multiply(inputs(0, num_node_blocks = 2, dist = Categorical(num_cats = 2), params = first.clone()))
            b = multiply(inputs(0, num_node_blocks = 3, dist = Categorical(num_cats = 2), params = second.clone()))
            root = summate(a, b, num_node_blocks = 1, edge_ids = edge_ids)
            root.set_params(weights.clone())
        pc = juice.compile(root, verbose = False).to(dev)
        return pc(torch.tensor([[0], [1]], device = dev)).flatten().double().cpu()

    # The first child's block 1, the second child's blocks 0 and 2
    per_child = values([torch.tensor([[0], [1]]), torch.tensor([[0, 0], [0, 2]])])
    with_global_ids = values(torch.tensor([[0, 0, 0], [1, 2, 4]]))
    w = weights.double()
    expected = torch.log(w[0] * first[1].double() + w[1] * second[0].double() + w[2] * second[2].double())

    assert torch.allclose(with_global_ids, expected, atol = 1e-5, rtol = 0)
    assert torch.allclose(per_child, expected, atol = 1e-5, rtol = 0), (per_child, expected)
