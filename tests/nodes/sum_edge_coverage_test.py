"""Every sum node has an edge: the check counts the parents, after the bounds checks, on any tensor it is given."""

import pytest
import torch

import pyjuice as juice
from pyjuice.nodes import SumNodes
from pyjuice.nodes.distributions import Categorical


def _child():
    return juice.multiply(juice.inputs(0, num_node_blocks = 3, block_size = 1, dist = Categorical(num_cats = 2)))


def _sum(child, edges, parents = 2):
    return SumNodes(parents, [child], edge_ids = edges, block_size = 1)


def _forbid_coverage(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the coverage check ran before its guard")
    monkeypatch.setattr(torch, "unique", forbidden)
    monkeypatch.setattr(torch, "bincount", forbidden)


@pytest.mark.parametrize("parents,edges", [
    (1, [[], []]),
    (3, [[0, 2, 2], [0, 1, 2]]),
    (3, [[0, 0, 0], [0, 1, 2]]),
    (2, [[0, 0], [0, 1]]),
])
def test_a_node_without_an_edge_is_an_error_and_leaves_the_edges_alone(parents, edges):
    caller = torch.tensor(edges, dtype = torch.long)
    before = caller.clone()
    with pytest.raises(AssertionError, match = "Some node has no edge"):
        _sum(_child(), caller, parents)
    assert torch.equal(caller, before)


@pytest.mark.parametrize("strided", [False, True])
def test_duplicate_edges_stay_as_given_without_a_copy(strided):
    literal = [[1, 0, 1, 0], [2, 0, 1, 2]]
    edges = torch.tensor(literal, dtype = torch.long)
    if strided:
        storage = torch.full((2, 8), -99, dtype = torch.long)
        storage[:, ::2].copy_(edges)
        edges = storage[:, ::2]
        assert not edges.is_contiguous()
    node = _sum(_child(), edges)
    assert node.edge_ids is edges and node.edge_ids.tolist() == literal
    assert node.num_edges == 4, "every duplicate parent occurrence keeps its own edge"


def test_fewer_edges_than_parents_never_allocates_a_parent_sized_count(monkeypatch):
    _forbid_coverage(monkeypatch)
    with pytest.raises(AssertionError, match = "Some node has no edge"):
        _sum(_child(), torch.tensor([[0], [2]], dtype = torch.long), parents = 10 ** 12)


@pytest.mark.parametrize("edges,message", [
    ([[0, 1], [0, 1], [-1, 99]], r"edge_ids.size\(0\) == 2"),
    ([[-1, 2], [0, 4]], "Edge index underflow"),
    ([[0, 1], [-1, 4]], "Edge index underflow"),
    ([[0, 2], [0, 1]], "Edge index overflow"),
    ([[0, 1], [0, 3]], "Edge index overflow"),
])
def test_shape_and_bounds_errors_come_first(monkeypatch, edges, message):
    caller = torch.tensor(edges, dtype = torch.long)
    _forbid_coverage(monkeypatch)
    with pytest.raises(AssertionError, match = message):
        _sum(_child(), caller)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "needs a GPU"))])
@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.uint8, torch.float32])
@pytest.mark.parametrize("parents", [2, 2.0], ids = ["int", "integral_float"])
def test_the_check_takes_any_tensor_and_count_it_was_given(device, dtype, parents):
    covered = torch.tensor([[1, 0, 1], [2, 0, 1]], dtype = dtype, device = device)
    assert _sum(_child(), covered, parents).edge_ids is covered
    with pytest.raises(AssertionError, match = "Some node has no edge"):
        _sum(_child(), torch.tensor([[1, 1, 1], [2, 0, 1]], dtype = dtype, device = device), parents)
