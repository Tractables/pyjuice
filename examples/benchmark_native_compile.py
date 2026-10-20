"""Measure ordinary PyJuice construction and a complete first categorical query.

Run each version in a fresh process on the same GPU, for example:
  CUDA_VISIBLE_DEVICES=0 python benchmark_native_compile.py \
    --no-gpu-compilation --output baseline-process-01.json
  CUDA_VISIBLE_DEVICES=0 python benchmark_native_compile.py \
    --no-gpu-compilation --steady-queries 30 --output candidate-process-01.json

The fixture is a literal mixture over two three-category variables. It uses
ordinary Categorical inputs and ordinary sums/products, without COFFEE objects.
"""

import argparse
import hashlib
from importlib import metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time


def timed(torch, device, operation):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def construct(juice, torch, np):
    left = np.array([[1., 2., 3.], [5., 1., 2.], [2., 7., 1.], [3., 2., 9.]], dtype=np.float32)
    right = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 2., 6.],
                      [5., 3., 7.], [6., 4., 3.], [7., 2., 4.], [8., 3., 5.]], dtype=np.float32)
    left /= left.sum(1, keepdims=True)
    right /= right.sum(1, keepdims=True)
    left_ids = np.repeat(np.arange(4), [1, 3, 7, 31])
    right_ids = np.arange(len(left_ids)) % 8
    weights = np.arange(1, len(left_ids) + 1, dtype=np.float32)
    weights /= weights.sum()
    with juice.set_block_size(1):
        a = juice.inputs(0, num_nodes=4, dist=juice.nodes.distributions.Categorical(3), params=torch.tensor(left))
        b = juice.inputs(1, num_nodes=8, dist=juice.nodes.distributions.Categorical(3), params=torch.tensor(right))
        terms = juice.multiply(a, b, edge_ids=torch.tensor(np.stack((left_ids, right_ids), axis=1)))
        root = juice.summate(terms, num_nodes=1, edge_ids=torch.stack((
            torch.zeros(len(left_ids), dtype=torch.long), torch.arange(len(left_ids)))))
        root.set_params(torch.tensor(weights))
    factors = {"left_probabilities": left, "right_probabilities": right, "mixture_weights": weights,
               "left_node_ids": left_ids, "right_node_ids": right_ids}
    return root, a, b, factors


def complete_query(pc, left, right, torch, device):
    # Input materialization and all requested device-to-host outputs belong to
    # every query, including the measured first one.
    data = torch.tensor(list(product(range(3), repeat=2)), dtype=torch.long, device=device)
    log_partition = pc(data).flatten().detach().cpu()
    pc.backward(data, compute_param_flows=False, logspace_flows=False, flows_memory=0.)
    left_posterior = pc.get_node_flows(left).T.detach().cpu()
    right_posterior = pc.get_node_flows(right).T.detach().cpu()
    return {"log_partition": log_partition, "left_posterior": left_posterior, "right_posterior": right_posterior}


def validate(outputs, factors, np):
    # Sum original, declared FP32 factor values with independent FP64 scalar
    # arithmetic. No circuit messages, compiled edges or parameters enter here.
    left, right = factors["left_probabilities"], factors["right_probabilities"]
    weights, left_ids, right_ids = (factors[name] for name in ("mixture_weights", "left_node_ids", "right_node_ids"))
    expected_z, expected_left, expected_right = [], [], []
    for x, y in product(range(3), repeat=2):
        terms = [float(weight) * float(left[a, x]) * float(right[b, y])
                 for weight, a, b in zip(weights, left_ids, right_ids)]
        total = math.fsum(terms)
        expected_z.append(math.log(total))
        expected_left.append([math.fsum(term for term, a in zip(terms, left_ids) if a == node) / total
                              for node in range(len(left))])
        expected_right.append([math.fsum(term for term, b in zip(terms, right_ids) if b == node) / total
                               for node in range(len(right))])
    targets = {"log_partition": np.asarray(expected_z), "left_posterior": np.asarray(expected_left),
               "right_posterior": np.asarray(expected_right)}
    errors, finite = {}, True
    for name, expected in targets.items():
        actual = outputs[name].numpy()
        valid = actual.shape == expected.shape and bool(np.isfinite(actual).all())
        finite = finite and valid
        errors[name] = float(np.max(np.abs(actual - expected))) if valid else None
    tolerances = {"log_partition": 3e-6, "left_posterior": 4e-6, "right_posterior": 4e-6}
    passed = finite and all(errors[name] <= tolerance for name, tolerance in tolerances.items())
    return {"passed": bool(passed), "finite_outputs_with_expected_shapes": finite,
            "max_absolute_errors": errors, "absolute_tolerances": tolerances,
            "oracle": "FP64 literal mixture sums and conditional leaf responsibilities",
            "independent_oracle_log_partition": expected_z}


def memory(torch, device):
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def numeric_binding(logical_device):
    order = os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    binding = {"logical_device": logical_device,
               "cuda_device_order": order if order in ("FASTEST_FIRST", "PCI_BUS_ID") else "unknown"}
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is None:
        return {**binding, "status": "unknown", "reason": "numeric_visibility_selector_unset"}
    entries = [value.strip() for value in selector.split(",")]
    if not entries or any(not value or any(char not in "0123456789" for char in value) for value in entries):
        return {**binding, "status": "unknown", "reason": "visibility_selector_is_not_an_all_numeric_list"}
    indices = [int(value) for value in entries]
    if len(set(indices)) != len(indices) or not 0 <= logical_device < len(indices) or binding["cuda_device_order"] == "unknown":
        return {**binding, "status": "unknown", "reason": "ambiguous_numeric_selector_or_device_order"}
    return {**binding, "status": "numeric_selector", "visible_indices": indices,
            "selected_index": indices[logical_device]}


def source_fingerprints(juice):
    package = Path(juice.__file__).resolve().parent
    files = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if (path.is_file() and path.suffix in (".py", ".cu", ".cuh")
                and all(not part.startswith(".") for part in relative.parts)):
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pyjuice_tree_sha256": tree, "pyjuice_files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    compiler = parser.add_mutually_exclusive_group(required=True)
    compiler.add_argument("--gpu-compilation", dest="gpu_compilation", action="store_true")
    compiler.add_argument("--no-gpu-compilation", dest="gpu_compilation", action="store_false")
    parser.add_argument("--device", type=int, default=0, help="logical CUDA device index")
    parser.add_argument("--steady-queries", type=int, default=0, help="optional identical-query diagnostic repetitions")
    parser.add_argument("--warmup-queries", type=int, default=10, help="warmups only when steady queries are requested")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.device, args.steady_queries, args.warmup_queries) < 0:
        parser.error("device and query counts must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)

    common_start = time.perf_counter()
    start = time.perf_counter()
    import torch
    torch_import = time.perf_counter() - start
    start = time.perf_counter()
    import numpy as np
    numpy_import = time.perf_counter() - start
    start = time.perf_counter()
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("the selected CUDA device is unavailable")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.synchronize(device)
    cuda_initialization = time.perf_counter() - start
    common_total = time.perf_counter() - common_start
    common_memory = memory(torch, device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        primary_start = time.perf_counter()
        def import_backend():
            import pyjuice
            return pyjuice

        juice, backend_import = timed(torch, device, import_backend)
        (root, left, right, factors), construction = timed(torch, device, lambda: construct(juice, torch, np))
        pc, compilation = timed(torch, device, lambda: juice.compile(
            root, disable_gpu_compilation=not args.gpu_compilation, verbose=False))
        pc, device_copy = timed(torch, device, lambda: pc.to(device))
        query = lambda: complete_query(pc, left, right, torch, device)
        first, first_seconds = timed(torch, device, query)
        primary = time.perf_counter() - primary_start
        first_memory = memory(torch, device)
        validation_start = time.perf_counter()
        validation = validate(first, factors, np)
        validation_seconds = time.perf_counter() - validation_start
        steady_seconds = []
        if validation["passed"] and args.steady_queries:
            for _ in range(args.warmup_queries):
                query()
            torch.cuda.synchronize(device)
            for _ in range(args.steady_queries):
                _, seconds = timed(torch, device, query)
                steady_seconds.append(seconds)

    # Metadata and source reads deliberately follow all measured operations.
    versions = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "cuda": torch.version.cuda}
    for name in ("pyjuice", "triton", "numba"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    properties = torch.cuda.get_device_properties(device)
    versions.update({"gpu": properties.name, "compute_capability": [properties.major, properties.minor],
                     "device_total_memory_bytes": properties.total_memory,
                     "physical_device_binding": numeric_binding(args.device)})
    literal_factors = {name: value.tolist() for name, value in factors.items()}
    fixture_sha256 = hashlib.sha256(json.dumps(literal_factors, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    stage_sum = backend_import + construction + compilation + device_copy + first_seconds
    record = {"schema": "pyjuice.ordinary-categorical-first-request-benchmark.v1",
        "status": "passed" if validation["passed"] else "failed_numerical_validation",
        "fixture": {"name": "ordinary_categorical_mixture_2x3_irregular_degrees_v1", "sha256": fixture_sha256,
                    "variables": 2, "categories_per_variable": 3, "left_product_parent_degrees": [1, 3, 7, 31],
                    "product_terms": 42, "literal_factors": literal_factors},
        "configuration": {"allow_gpu_compilation": args.gpu_compilation, "block_size": 1,
                          "torch_grad_mode": "no_grad", "actual_primary_query_count": 1,
                          "observations_per_query": 9, "compute_param_flows": False},
        "timings": {"common_torch_import_seconds": torch_import, "common_numpy_import_seconds": numpy_import,
                    "common_cuda_initialization_seconds": cuda_initialization,
                    "common_initialization_seconds": common_total,
                    "backend_import_seconds": backend_import, "node_construction_seconds": construction,
                    "compile_seconds": compilation, "device_copy_seconds": device_copy,
                    "first_complete_query_seconds": first_seconds, "primary_first_request_seconds": primary,
                    "primary_stage_sum_seconds": stage_sum, "primary_orchestration_seconds": primary - stage_sum,
                    "common_plus_primary_first_request_seconds": common_total + primary,
                    "independent_validation_seconds": validation_seconds},
        "steady_diagnostic": {"warmup_queries": args.warmup_queries if steady_seconds else 0,
            "queries": len(steady_seconds), "seconds": steady_seconds,
            "median_seconds": statistics.median(steady_seconds) if steady_seconds else None},
        "memory": {"common_initialization": common_memory, "first_request": first_memory,
                   "lifetime_including_steady": memory(torch, device)},
        "first_query_outputs": {name: value.tolist() for name, value in first.items()},
        "validation": validation, "versions": versions,
        "environment": {
            "thread_environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cache_directories_configured": {name: bool(os.environ.get(name)) for name in (
                "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR", "TORCH_EXTENSIONS_DIR")},
            "cache_policy": "Externally managed directories are not cleared. Fresh processes do not imply cold disk caches; set flags do not prove identical locations.",
        },
        "source_fingerprints": source_fingerprints(juice),
        "boundary": "Fresh process after separately reported common runtime initialization. Primary includes backend import, node construction with parameter normalization, compilation, device copy and one complete native forward/backward query with CPU outputs. Optional steady repetitions are diagnostic and do not amortize the first request.",
        "comparison": "Compare source versions under the same compiler flag and GPU. The FP64 oracle checks correctness and is not a performance baseline."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "primary_first_request_seconds": primary,
        "first_complete_query_seconds": first_seconds, "validation": validation}, allow_nan=False), flush=True)
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
