"""Compare complete CPU Sum-node construction in separate fresh processes.

Run this script against each installed revision with CUDA_VISIBLE_DEVICES=-1.
Imports, common child fixtures and metadata checks are outside constructor clocks.
No TensorCircuit, compiled layer, parameter initialization or inference is run.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import random
import resource
import sys
import time
import traceback


SPEC = {
    'single_edge': {'objects': 1024, 'parent_blocks': 1, 'child_blocks': 32,
                    'block_size': 64, 'child_block': 31, 'edge_blocks': 1},
    'large_token': {'objects': 1, 'parent_blocks': 2048, 'child_blocks': 131072,
                    'block_size': 1, 'degree': 64, 'edge_blocks': 131072},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('Linux with CUDA_VISIBLE_DEVICES=-1 is required for this CPU benchmark')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('choose a fresh output path')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('output parent must already exist')
    if any(n == 'torch' or n.startswith(('torch.', 'numpy', 'pyjuice')) for n in sys.modules):
        raise RuntimeError('a fresh process without numerical imports is required')
    result = dict(schema='pyjuice.sum-constructor-CPU.v1', status='running',
        driver_sha256=sha(__file__), shapes=SPEC, cases=[],
        boundary='Two fixed source-Sum construction cases, no TensorCircuit/InputLayer '
                 'construction, compilation, CUDA initialization, FW/BW or sampling. '
                 'Fixtures, input hashing and validation are outside construction clocks. '
                 'No stable, general or full-workload performance claim.')
    try:
        started = time.perf_counter()
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        common_seconds = time.perf_counter() - started
        assert not torch.cuda.is_initialized()
        started = time.perf_counter()
        import pyjuice
        from pyjuice.nodes import InputNodes, ProdNodes, SumNodes
        from pyjuice.nodes.distributions import Categorical
        engine_seconds = time.perf_counter() - started
        assert not torch.cuda.is_initialized()

        def rng_snapshot():
            return random.getstate(), torch.get_rng_state().clone(), np.random.get_state()

        def same_rng(before):
            now = rng_snapshot()
            return (before[0] == now[0] and torch.equal(before[1], now[1])
                    and before[2][0] == now[2][0] and np.array_equal(before[2][1], now[2][1])
                    and before[2][2:] == now[2][2:])

        def tensor_sha(tensor):
            return hashlib.sha256(tensor.detach().numpy().tobytes(order='C')).hexdigest()

        # These are genuine source nodes, not an InputLayer or a compiled PC.
        # Each timed call receives its own already-owned edge tensor, mirroring
        # the frontend's independent arguments without timing fixture generation.
        setup_rng = rng_snapshot()
        started = time.perf_counter()
        fixtures = {}
        for name, spec in SPEC.items():
            size, count = spec['block_size'], spec['child_blocks']
            leaf = InputNodes(count, [0], Categorical(num_cats=2), block_size=size)
            child = ProdNodes(count, [leaf], block_size=size)
            if name == 'single_edge':
                edges = [torch.tensor([[0], [spec['child_block']]], dtype=torch.long)
                         for _ in range(spec['objects'])]
            else:
                edges = [torch.stack((torch.arange(spec['parent_blocks']).repeat_interleave(spec['degree']),
                                      torch.arange(spec['edge_blocks'])))]
            fixtures[name] = child, edges
        setup_seconds = time.perf_counter() - started
        assert same_rng(setup_rng) and not torch.cuda.is_initialized()
        identities = {}
        for name, (_, edges) in fixtures.items():
            hashes = [tensor_sha(value) for value in edges]
            assert len(set(hashes)) == 1
            identities[name] = dict(edge_sha256=hashes[0], edge_shape=list(edges[0].shape),
                edge_dtype=str(edges[0].dtype), edge_argument_storages=len({v.data_ptr() for v in edges}))

        for name, spec in SPEC.items():
            child, edges = fixtures[name]
            before_rng = rng_snapshot()
            started = time.perf_counter()
            retained = [SumNodes(spec['parent_blocks'], [child], edge_ids=value,
                                 block_size=spec['block_size']) for value in edges]
            elapsed = time.perf_counter() - started
            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            # All proofs and serialization follow both clock and RSS point.
            assert len(retained) == spec['objects'] and same_rng(before_rng)
            assert not torch.cuda.is_initialized()
            alias_count = 0
            for node, argument in zip(retained, edges):
                assert type(node) is SumNodes and node.chs == [child]
                assert node.block_size == spec['block_size'] and node.num_node_blocks == spec['parent_blocks']
                assert node.num_ch_node_blocks == spec['child_blocks']
                assert torch.equal(node.edge_ids, argument) and tensor_sha(argument) == identities[name]['edge_sha256']
                assert node.get_params() is None and not node.has_params() and not node.is_tied()
                assert all(vars(node).get(key) is None for key in
                           ('_param_range', '_param_ids', '_param_flow_range', '_output_ind_range'))
                assert node.scope.to_list() == [0]
                alias_count += int(node.edge_ids is argument and node.edge_ids.data_ptr() == argument.data_ptr())
            assert alias_count == spec['objects']
            result['cases'].append(dict(name=name, construction_seconds=elapsed,
                calls=len(retained), retained_objects_at_check=len(retained), input_identity=identities[name],
                edge_metadata_unchanged=True, caller_edge_aliases=alias_count,
                parameters_uninitialized=True, rng_unchanged=True, cuda_initialized=False,
                maxrss_bytes=rss,
                maxrss_scope='Cumulative child high-water mark at this construction end; includes '
                    'imports, both fixtures, and earlier case/validation. Not a per-case or PC increment.'))
            del retained, node

        result.update(status='passed_two_CPU_sum_construction_cases', imports=dict(
            common_torch_numpy_thread_setup_seconds=common_seconds, pyjuice_source_import_seconds=engine_seconds),
            fixture_setup_seconds=setup_seconds, versions={name:md.version(name) for name in ('torch','numpy','pyjuice')},
            python=sys.version, threads=dict(intra=torch.get_num_threads(), inter=torch.get_num_interop_threads()),
            source=dict(sum_nodes_module=SumNodes.__module__,
                sum_nodes_sha256=sha(inspect.getfile(SumNodes))), cuda_initialized=torch.cuda.is_initialized())
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        with args.output.open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
