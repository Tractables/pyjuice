"""Sum edge coverage keeps literal edges, legacy dispatch and validation order.

CPU cases inspect constructor metadata; the device fallback case requires CUDA.
"""

import pytest
import torch

import pyjuice as juice
from pyjuice.nodes import SumNodes
from pyjuice.nodes.distributions import Categorical


class MarkedSum(SumNodes):
    pass


class EdgeTensor(torch.Tensor):
    pass


def _child():
    return juice.multiply(juice.inputs(0, num_node_blocks=3, block_size=1,
                                       dist=Categorical(num_cats=2)))


def _sum(child, edges, parents=2, cls=SumNodes, **kwargs):
    return cls(parents, [child], edge_ids=edges, block_size=1, **kwargs)


def _forbid_coverage(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("coverage operator ran before its admission boundary")
    monkeypatch.setattr(torch, "unique", forbidden)
    monkeypatch.setattr(torch, "bincount", forbidden)


@pytest.mark.parametrize("strided", [False, True])
def test_exact_cpu_long_sum_counts_parent_coverage_without_reordering_or_copying(monkeypatch, strided):
    child = _child()
    literal = [[1, 0, 1, 0], [2, 0, 1, 2]]
    edges = torch.tensor(literal, dtype=torch.long)
    if strided:
        storage = torch.full((2, 8), -99, dtype=torch.long)
        storage[:, ::2].copy_(edges)
        edges = storage[:, ::2]
        assert not edges.is_contiguous()
    before, version = edges.clone(), edges._version
    original_count, counted = torch.bincount, []
    def counted_parents(values, *args, **kwargs):
        counted.append(values.tolist())
        return original_count(values, *args, **kwargs)
    def forbidden_unique(*args, **kwargs):
        raise AssertionError("exact CPU-long SumNodes used legacy unique")
    monkeypatch.setattr(torch, "bincount", counted_parents)
    monkeypatch.setattr(torch, "unique", forbidden_unique)
    node = _sum(child, edges)
    assert counted == [[1, 0, 1, 0]]
    assert node.edge_ids is edges and node.edge_ids.tolist() == literal
    assert edges._version == version and torch.equal(edges, before)
    assert node.num_edges == 4, "every duplicate parent occurrence keeps its own edge"
    if strided:
        assert storage[:, 1::2].tolist() == [[-99] * 4, [-99] * 4]


@pytest.mark.parametrize("parents,edges", [
    (1, [[], []]),
    (3, [[0, 2, 2], [0, 1, 2]]),
    (3, [[0, 0, 0], [0, 1, 2]]),
    (2, [[0, 0], [0, 1]]),
])
def test_missing_parent_and_empty_inputs_keep_coverage_error_and_original_edges(parents, edges):
    child, caller = _child(), torch.tensor(edges, dtype=torch.long)
    before, version = caller.clone(), caller._version
    with pytest.raises(AssertionError, match="Some node has no edge"):
        _sum(child, caller, parents)
    assert torch.equal(caller, before) and caller._version == version


def test_fewer_edges_than_parents_never_allocates_a_parent_sized_count(monkeypatch):
    child = _child()
    _forbid_coverage(monkeypatch)
    # Construction itself holds only scalar metadata; the supplied edge tensor
    # is tiny. A coverage implementation must reject before a huge bincount.
    with pytest.raises(AssertionError, match="Some node has no edge"):
        _sum(child, torch.tensor([[0], [2]], dtype=torch.long), parents=10 ** 12)


def test_one_edge_one_parent_needs_no_unique_or_count(monkeypatch):
    child, edges = _child(), torch.tensor([[0], [2]], dtype=torch.long)
    _forbid_coverage(monkeypatch)
    node = _sum(child, edges, parents=1)
    assert node.edge_ids is edges and node.edge_ids.tolist() == [[0], [2]]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32,
                                   torch.float32, torch.float64, torch.bool])
def test_non_long_dtypes_keep_legacy_unique_and_caller_ownership(monkeypatch, dtype):
    child = _child()
    edges = torch.tensor([[1, 0, 1, 0], [2, 0, 1, 2]], dtype=dtype)
    before = edges.clone()
    original_unique, seen = torch.unique, []
    def legacy_unique(values, *args, **kwargs):
        seen.append(values.dtype)
        return original_unique(values, *args, **kwargs)
    def forbidden_count(*args, **kwargs):
        raise AssertionError("a legacy dtype entered the CPU-long coverage route")
    monkeypatch.setattr(torch, "unique", legacy_unique)
    monkeypatch.setattr(torch, "bincount", forbidden_count)
    node = _sum(child, edges)
    assert seen == [dtype]
    assert node.edge_ids is edges and node.edge_ids.dtype == dtype
    assert torch.equal(edges, before)


@pytest.mark.parametrize("kind", ["tensor_subclass", "sum_subclass"])
def test_subclasses_keep_the_original_unique_dispatch(monkeypatch, kind):
    child = _child()
    edges = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    cls = MarkedSum if kind == "sum_subclass" else SumNodes
    if kind == "tensor_subclass":
        edges = edges.as_subclass(EdgeTensor)
    original_unique, seen = torch.unique, []
    def legacy_unique(values, *args, **kwargs):
        seen.append(type(values))
        return original_unique(values, *args, **kwargs)
    def forbidden_count(*args, **kwargs):
        raise AssertionError("a subclass entered the exact-type coverage route")
    monkeypatch.setattr(torch, "unique", legacy_unique)
    monkeypatch.setattr(torch, "bincount", forbidden_count)
    node = _sum(child, edges, cls=cls)
    assert len(seen) == 1 and type(node) is cls and node.edge_ids is edges
    if kind == "tensor_subclass":
        assert seen == [EdgeTensor]


def test_integral_float_parent_count_keeps_legacy_constructor_admission(monkeypatch):
    # Constructor admission is preserved; this is not a compilation claim.
    child = _child()
    edges = torch.tensor([[1, 0, 1], [2, 0, 1]], dtype=torch.long)
    before, version = edges.clone(), edges._version
    original_unique, seen = torch.unique, []
    def legacy_unique(values, *args, **kwargs):
        seen.append(values.tolist())
        return original_unique(values, *args, **kwargs)
    def forbidden_count(*args, **kwargs):
        raise AssertionError("a non-int parent count entered the exact CPU route")
    monkeypatch.setattr(torch, "unique", legacy_unique)
    monkeypatch.setattr(torch, "bincount", forbidden_count)
    node = _sum(child, edges, parents=2.0)
    assert seen == [[1, 0, 1]]
    assert type(node.num_node_blocks) is float and node.num_node_blocks == 2.0
    assert node.edge_ids is edges and edges._version == version
    assert torch.equal(edges, before)


def test_higher_dimensional_edges_keep_accepted_legacy_unique_semantics(monkeypatch):
    child = juice.multiply(juice.inputs(0, num_node_blocks=1, block_size=1,
                                       dist=Categorical(num_cats=2)))
    edges = torch.tensor([[[0]], [[0]]], dtype=torch.long)
    before, version = edges.clone(), edges._version
    original_unique, seen = torch.unique, []
    def legacy_unique(values, *args, **kwargs):
        seen.append((tuple(values.shape), values.tolist()))
        return original_unique(values, *args, **kwargs)
    def forbidden_count(*args, **kwargs):
        raise AssertionError("a higher-dimensional edge tensor entered the 1D count route")
    monkeypatch.setattr(torch, "unique", legacy_unique)
    monkeypatch.setattr(torch, "bincount", forbidden_count)
    node = _sum(child, edges, parents=1)
    assert seen == [((1, 1), [[0]])]
    assert node.edge_ids is edges and tuple(node.edge_ids.shape) == (2, 1, 1)
    assert torch.equal(edges, before) and edges._version == version


@pytest.mark.parametrize("edges,message", [
    ([[0, 1], [0, 1], [-1, 99]], r"edge_ids.size\(0\) == 2"),
    ([[-1, 2], [0, 4]], "Edge index underflow"),
    ([[0, 1], [-1, 4]], "Edge index underflow"),
    ([[0, 2], [0, 1]], "Edge index overflow"),
    ([[0, 1], [0, 3]], "Edge index overflow"),
])
def test_shape_and_bounds_errors_precede_every_coverage_route(monkeypatch, edges, message):
    child, caller = _child(), torch.tensor(edges, dtype=torch.long)
    before = caller.clone()
    _forbid_coverage(monkeypatch)
    with pytest.raises(AssertionError, match=message):
        _sum(child, caller)
    assert torch.equal(caller, before)


def test_invalid_tensor_comparison_and_rank_keep_precoverage_errors(monkeypatch):
    child = _child()
    complex_edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.complex64)
    _forbid_coverage(monkeypatch)
    with pytest.raises(RuntimeError):
        _sum(child, complex_edges)
    with pytest.raises(IndexError):
        _sum(child, torch.tensor([0, 1], dtype=torch.long))


def test_source_edges_remain_borrowed_and_callbacks_run_after_validation(monkeypatch):
    child = _child()
    source = _sum(child, torch.tensor([[0, 1], [0, 1]], dtype=torch.long))
    caller = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    seen = []
    def callback(node, **kwargs):
        seen.append((node.edge_ids is caller, node.edge_ids.tolist(), node.get_source_ns() is source,
                     kwargs["marker"]))
        node.edge_ids[1, 0] = 0
    monkeypatch.setattr(SumNodes, "INIT_CALLBACKS", [callback])
    tied = _sum(child, caller, source_node=source, marker="validated")
    assert seen == [(True, [[0, 1], [1, 2]], True, "validated")]
    assert tied.edge_ids is caller and caller.tolist() == [[0, 1], [0, 2]]
    assert source.edge_ids.tolist() == [[0, 1], [0, 1]]
    before = list(seen)
    with pytest.raises(AssertionError, match="Some node has no edge"):
        _sum(child, torch.tensor([[0, 0], [0, 1]], dtype=torch.long), marker="invalid")
    assert seen == before, "an invalid edge table must not reach construction callbacks"


def test_set_edges_preserves_new_caller_identity_and_duplicate_occurrences():
    child = _child()
    original = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    node = _sum(child, original)
    replacement = torch.tensor([[1, 0, 1], [2, 0, 2]], dtype=torch.long)
    version = replacement._version
    node.set_edges(replacement)
    assert node.edge_ids is replacement and replacement._version == version
    assert node.edge_ids.tolist() == [[1, 0, 1], [2, 0, 2]]
    assert original.tolist() == [[0, 1], [0, 1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_long_edges_keep_legacy_unique_route(monkeypatch):
    child = _child()
    edges = torch.tensor([[0, 1], [2, 0]], dtype=torch.long, device="cuda:0")
    original_unique, seen = torch.unique, []
    def legacy_unique(values, *args, **kwargs):
        seen.append(values.device.type)
        return original_unique(values, *args, **kwargs)
    def forbidden_count(*args, **kwargs):
        raise AssertionError("CUDA edges entered a CPU-only coverage route")
    monkeypatch.setattr(torch, "unique", legacy_unique)
    monkeypatch.setattr(torch, "bincount", forbidden_count)
    node = _sum(child, edges)
    assert seen == ["cuda"] and node.edge_ids is edges
    assert node.edge_ids.cpu().tolist() == [[0, 1], [2, 0]]
