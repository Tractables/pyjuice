"""Bounded integer metadata matches a scalar oracle and ordinary categorical PCs."""

from collections import Counter
from itertools import product
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

import pyjuice.layer.backend.node_partition as partition


def scalar_weighted(values, counts, maximum, target, *, legacy_backtrace=True):
    """Literal integer recurrence, independent of vectorized production tables."""
    values, counts = list(map(int, values)), list(map(int, counts))
    cumulative = [sum(counts[:index + 1]) for index in range(len(counts))]
    dp = [[0] * (maximum + 1) for _ in values]
    backtrace = [[0] * (maximum + 1) for _ in values]
    for index, value in enumerate(values):
        dp[index][1] = value * cumulative[index]
    threshold = 0 if target is None else target
    chosen = maximum
    if dp[-1][1] <= threshold:
        chosen = 1
    else:
        for group in range(2, maximum + 1):
            dp[0][group] = values[0] * cumulative[0]
            for end in range(1, len(values)):
                best_cost, best_index = 2 ** 31 - 1, -1
                for begin in range(end):
                    cost = dp[begin][group - 1] + values[end] * (cumulative[end] - cumulative[begin])
                    if cost < best_cost:
                        best_cost, best_index = cost, begin
                dp[end][group], backtrace[end][group] = best_cost, best_index
            if dp[-1][group] <= threshold:
                chosen = group
                break
    positions, current = [], len(values) - 1
    for group in range(chosen, 0, -1):
        positions.append(current)
        current = backtrace[current][chosen if legacy_backtrace else group]
    return sorted({values[index] for index in positions})


def scalar_entry(degrees, max_num_partitions=None, sparsity_tolerance=None, algorithm="dp_with_coalesce"):
    """Independent public-entry oracle for positive integral degree values."""
    raw = degrees.detach().cpu().numpy() if isinstance(degrees, torch.Tensor) else np.asarray(degrees)
    original_dtype = raw.dtype
    values = [int(value) for value in raw]
    if sparsity_tolerance is not None and sparsity_tolerance < 1e-6:
        sparsity_tolerance, max_num_partitions = None, 1
    if sparsity_tolerance is not None:
        assert 1e-6 < sparsity_tolerance <= 1.
        if max_num_partitions is None:
            max_num_partitions = max(min(len(set(values)), 16), 1)
    elif max_num_partitions is None:
        max_num_partitions = 1
    else:
        assert max_num_partitions >= 1
    target = None if sparsity_tolerance is None else math.ceil(sum(values) * (1. + sparsity_tolerance))
    if max_num_partitions == 1:
        return torch.tensor([max(values)], dtype=torch.int64)
    frequencies = Counter(value for value in values if value != 0)
    unique = sorted(frequencies)
    assert unique and unique[0] > 0
    assert algorithm == "dp_with_coalesce"
    tolerance = (unique[-1] - unique[0]) // 1000
    coalesced, counts, start = [], [], 0
    while start < len(unique):
        end = start + 1
        while end < len(unique) and unique[end] - unique[start] <= tolerance:
            end += 1
        coalesced.append(unique[end - 1])
        counts.append(sum(frequencies[value] for value in unique[start:end]))
        start = end
    result = scalar_weighted(coalesced, counts, max_num_partitions, target)
    return torch.from_numpy(np.array(result, dtype=original_dtype))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("seed", [3, 47, 91])
def test_random_histograms_match_independent_recurrence_without_metadata_jit(dtype, seed):
    rng = np.random.default_rng(seed)
    unique = np.sort(rng.choice(np.arange(1, 120), size=11, replace=False)).astype(dtype)
    values = np.repeat(unique, rng.integers(1, 13, size=len(unique)))
    rng.shuffle(values)
    original = values.copy()
    with patch.object(partition, "_coalesce_fn", side_effect=AssertionError("coalesce JIT")), \
            patch.object(partition, "_weighted_partition_nodes_dp_simple_compiled", side_effect=AssertionError("DP JIT")), \
            patch.object(partition, "_backtrace_fn", side_effect=AssertionError("backtrace JIT")):
        for maximum, tolerance in product((2, 3, 8, 16), (None, .03, .25, 1.)):
            supplied = torch.from_numpy(values)
            actual = partition.partition_nodes_by_n_edges(supplied,
                max_num_partitions=maximum, sparsity_tolerance=tolerance)
            expected = scalar_entry(supplied, maximum, tolerance)
            assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("maximum", [None, 2, 3, 16])
def test_single_cpu_histogram_replaces_large_unique_sort_and_coalesce_inputs(maximum):
    values = torch.tensor([1, 3, 7, 31, 99] * 4096, dtype=torch.int64)[::2]
    # Non-contiguous input is still a read-only NumPy view for histogramming.
    expected = scalar_entry(values, maximum, .2)
    original_unique, unique_sizes = np.unique, []

    def record_small_unique(values, *args, **kwargs):
        unique_sizes.append(np.asarray(values).size)
        return original_unique(values, *args, **kwargs)

    with patch.object(torch, "unique", side_effect=AssertionError("large Torch unique")), \
            patch.object(np, "unique", side_effect=record_small_unique), \
            patch.object(np, "sort", side_effect=AssertionError("full degree sort")), \
            patch.object(partition, "_coalesce", side_effect=AssertionError("duplicate histogram")):
        actual = partition.partition_nodes_by_n_edges(values, max_num_partitions=maximum, sparsity_tolerance=.2)
    assert torch.equal(actual, expected)
    assert all(size <= 16 for size in unique_sizes)


def test_auto_coalescing_keeps_group_endpoints_and_counts():
    values = torch.tensor([1, 2, 3, 4, 7, 1001, 1002, 1003, 4096, 4096], dtype=torch.int32)
    for maximum, tolerance in product((2, 3, 16), (None, .01, .2, 1.)):
        actual = partition.partition_nodes_by_n_edges(values,
            max_num_partitions=maximum, sparsity_tolerance=tolerance)
        expected = scalar_entry(values, maximum, tolerance)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)


def test_existing_fixed_column_backtrace_quirk_remains_explicit():
    values = torch.tensor([1, 2, 3, 4, 5])
    actual = partition.partition_nodes_by_n_edges(values, max_num_partitions=3)
    assert actual.tolist() == scalar_weighted([1, 2, 3, 4, 5], [1] * 5, 3, None) == [2, 3, 5]
    assert scalar_weighted([1, 2, 3, 4, 5], [1] * 5, 3, None, legacy_backtrace=False) == [1, 3, 5]


def test_compact_histogram_agrees_with_unchanged_compiled_dp_on_ties_and_target_boundaries():
    for values, counts in (([1, 2, 3, 4, 5], [1] * 5), ([1, 3, 7, 31], [7, 2, 1, 3]),
                           ([2, 3, 5, 9, 17], [1, 4, 2, 8, 1])):
        values, counts = np.array(values, dtype=np.int64), np.array(counts, dtype=np.int64)
        minimum = sum(int(value) * int(count) for value, count in zip(values, counts))
        single = int(values[-1]) * int(counts.sum())
        for maximum, target in product((2, 3, 4, 16), (None, minimum - 1, minimum, single - 1, single, single + 1)):
            dp = np.zeros((len(values), maximum + 1), dtype=np.int64)
            backtrace = np.zeros_like(dp)
            _, groups = partition._weighted_partition_nodes_dp_simple_compiled(
                values, np.cumsum(counts), dp, backtrace, maximum, 0 if target is None else target)
            positions = np.zeros(groups, dtype=np.int64)
            partition._backtrace_fn(positions, backtrace, groups, len(values))
            expected = np.unique(values[positions])
            actual = partition._partition_compact_integer_histogram(values, counts, maximum, target)
            np.testing.assert_array_equal(actual, expected)
            assert actual.tolist() == scalar_weighted(values, counts, maximum, target)


@pytest.mark.parametrize("case", ["zeros", "float", "numpy", "large_degree", "many_bins", "many_partitions", "large_cost"])
def test_outside_compact_domain_keeps_existing_dispatch(case):
    values, maximum = torch.tensor([1, 3, 7, 31], dtype=torch.int64), 4
    if case == "zeros":
        values = torch.tensor([0, 1, 3, 7, 31], dtype=torch.int64)
    elif case == "float":
        values = values.float()
    elif case == "numpy":
        values = values.numpy()
    elif case == "large_degree":
        values = torch.tensor([1, 3, 7, 4097], dtype=torch.int64)
    elif case == "many_bins":
        values = torch.arange(1, 66, dtype=torch.int64)
    elif case == "many_partitions":
        maximum = 17
    else:
        values = torch.full((524288,), 4096, dtype=torch.int64)
        values[:2] = torch.tensor([1, 2])
    expected = scalar_entry(values, maximum, .3)
    original = partition._coalesce
    with patch.object(partition, "_partition_compact_integer_histogram", side_effect=AssertionError("changed fallback")), \
            patch.object(partition, "_coalesce", wraps=original) as coalesce:
        actual = partition.partition_nodes_by_n_edges(values, max_num_partitions=maximum, sparsity_tolerance=.3)
    coalesce.assert_called_once()
    assert actual.dtype == expected.dtype and torch.equal(actual, expected)


def test_unknown_algorithm_and_one_partition_behavior_remain():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        partition.partition_nodes_by_n_edges(torch.tensor([1, 3, 7]), max_num_partitions=3, algorithm="unknown")
    for algorithm in ("unknown", "dp_simple", "dp_with_coalesce"):
        actual = partition.partition_nodes_by_n_edges(torch.tensor([1, 3, 7]),
            sparsity_tolerance=0., algorithm=algorithm)
        assert actual.dtype == torch.int64 and actual.tolist() == [7]


def ordinary_categorical_mixture():
    """Two observed categorical variables, with irregular product-parent degrees."""
    import pyjuice as juice
    from pyjuice.nodes.distributions import Categorical

    left = np.array([[1., 2., 3.], [5., 1., 2.], [2., 7., 1.], [3., 2., 9.]], dtype=np.float32)
    right = np.array([[1. + index, 2. + index % 3, 3. + index % 5] for index in range(8)], dtype=np.float32)
    left /= left.sum(1, keepdims=True)
    right /= right.sum(1, keepdims=True)
    left_ids = np.repeat(np.arange(4), [1, 3, 7, 31])
    right_ids = np.arange(len(left_ids)) % 8
    weights = np.arange(1, len(left_ids) + 1, dtype=np.float32)
    weights /= weights.sum()
    with juice.set_block_size(1):
        a = juice.inputs(0, num_nodes=4, dist=Categorical(3), params=torch.tensor(left))
        b = juice.inputs(1, num_nodes=8, dist=Categorical(3), params=torch.tensor(right))
        terms = juice.multiply(a, b, edge_ids=torch.tensor(np.stack((left_ids, right_ids), axis=1)))
        root = juice.summate(terms, num_nodes=1, edge_ids=torch.stack((
            torch.zeros(len(left_ids), dtype=torch.long), torch.arange(len(left_ids)))))
        root.set_params(torch.tensor(weights))
    return root, a, b, (left.astype(np.float64), right.astype(np.float64),
                        weights.astype(np.float64), left_ids, right_ids)


def compiled_tables(pc):
    result = []
    for group in pc.inner_layer_groups:
        for layer in group:
            for name in ("partitioned_nids", "partitioned_cids", "partitioned_pids", "partitioned_pfids",
                         "partitioned_u_cids", "partitioned_parids", "partitioned_chids", "partitioned_parpids"):
                if hasattr(layer, name):
                    result.append((name, tuple(value.detach().cpu().clone() for value in getattr(layer, name))))
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ordinary native CUDA inference")
@pytest.mark.parametrize("logspace", [False, True])
def test_non_coffee_irregular_mixture_compiler_tables_masses_flows_and_sampling(logspace):
    import pyjuice as juice

    root, left, right, factors = ordinary_categorical_mixture()
    baseline_root, _, _, _ = ordinary_categorical_mixture()
    original_partition, seen = partition.partition_nodes_by_n_edges, []

    def record_histogram(values, *args, **kwargs):
        seen.append(sorted(set(values.tolist())))
        return original_partition(values, *args, **kwargs)

    with patch("pyjuice.layer.prod_layer.partition_nodes_by_n_edges", side_effect=record_histogram):
        pc = juice.compile(root, disable_gpu_compilation=True, verbose=False)
    assert any(len(values) >= 3 and {1, 3, 7, 31} <= set(values) for values in seen)
    with patch("pyjuice.layer.prod_layer.partition_nodes_by_n_edges", side_effect=scalar_entry), \
            patch("pyjuice.layer.sum_layer.partition_nodes_by_n_edges", side_effect=scalar_entry):
        baseline = juice.compile(baseline_root, disable_gpu_compilation=True, verbose=False)
    current_tables, baseline_tables = compiled_tables(pc), compiled_tables(baseline)
    assert len(current_tables) == len(baseline_tables)
    for (name, current), (expected_name, expected) in zip(current_tables, baseline_tables):
        assert name == expected_name and len(current) == len(expected)
        assert all(torch.equal(a, b) for a, b in zip(current, expected))
    assert torch.equal(pc.params, baseline.params)

    observations = np.array(list(product(range(3), repeat=2)))
    a, b, weights, a_ids, b_ids = factors
    masses = np.stack([weights * a[a_ids, x] * b[b_ids, y] for x, y in observations])
    totals, responsibilities = masses.sum(1), masses / masses.sum(1, keepdims=True)
    expected_left, expected_right = np.zeros((9, 4)), np.zeros((9, 8))
    for edge, (a_id, b_id) in enumerate(zip(a_ids, b_ids)):
        expected_left[:, a_id] += responsibilities[:, edge]
        expected_right[:, b_id] += responsibilities[:, edge]

    pc.to("cuda:0")
    data = torch.tensor(observations, device="cuda:0")
    np.testing.assert_allclose(pc(data).flatten().detach().cpu(), np.log(totals), atol=3e-6, rtol=0)
    pc.backward(data, logspace_flows=logspace, flows_memory=0.)
    for leaf, expected in ((left, expected_left), (right, expected_right)):
        flow = pc.get_node_flows(leaf).T
        np.testing.assert_allclose((flow.exp() if logspace else flow).cpu(), expected, atol=4e-6, rtol=0)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(19)
        sampled = juice.queries.sample(pc, num_samples=8192, conditional=False).cpu().numpy()
    observed = np.bincount(sampled[:, 0] * 3 + sampled[:, 1], minlength=9) / 8192
    np.testing.assert_allclose(observed, totals / totals.sum(), atol=.025, rtol=0)
