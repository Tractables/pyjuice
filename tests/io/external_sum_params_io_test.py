import pyjuice as juice
import torch
import tempfile

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, SumNodes, ExternalParamsSumNodes, LowRankSumParams

import pytest


def test_external_sum_params_io():

    with juice.set_block_size(4):

        i0 = inputs(0, 2, dists.Categorical(num_cats = 5))
        i1 = inputs(1, 2, dists.Categorical(num_cats = 5))

        m = multiply(i0, i1)
        ns = summate(m, num_node_blocks = 2, external_params = LowRankSumParams(rank = 5))

        m2 = multiply(ns)
        root_ns = summate(m2, num_node_blocks = 1, block_size = 1)

    root_ns.init_parameters(perturbation = 2.0)

    temp_file = tempfile.NamedTemporaryFile(suffix = '.jpc')
    juice.save(temp_file.name, root_ns)

    new_root_ns = juice.load(temp_file.name)

    new_nss = [n for n in new_root_ns if isinstance(n, ExternalParamsSumNodes)]

    assert len(new_nss) == 1

    new_ns = new_nss[0]

    # The descriptor is restored, with its configuration -- the rank is part of the layer signature,
    # so losing it would silently recompile the PC differently
    assert isinstance(new_ns.external_params, LowRankSumParams)
    assert new_ns.external_params.rank == 5
    assert new_ns.get_external_signature() == ns.get_external_signature()

    assert new_ns.num_node_blocks == ns.num_node_blocks and new_ns.block_size == ns.block_size
    assert torch.all(new_ns.edge_ids == ns.edge_ids)
    assert torch.all(torch.abs(new_ns._params - ns._params) < 1e-6)


def test_plain_sum_nodes_io_unaffected():
    """
    Checkpoints written before external parameterizations existed carry no descriptor, and must keep
    loading as plain sum nodes.
    """

    with juice.set_block_size(4):

        i0 = inputs(0, 2, dists.Categorical(num_cats = 5))
        i1 = inputs(1, 2, dists.Categorical(num_cats = 5))

        root_ns = summate(multiply(i0, i1), num_node_blocks = 1)

    root_ns.init_parameters(perturbation = 2.0)

    temp_file = tempfile.NamedTemporaryFile(suffix = '.jpc')
    juice.save(temp_file.name, root_ns)

    new_root_ns = juice.load(temp_file.name)

    assert type(new_root_ns) is SumNodes
    assert not any(isinstance(n, ExternalParamsSumNodes) for n in new_root_ns)


if __name__ == "__main__":
    test_external_sum_params_io()
    test_plain_sum_nodes_io_unaffected()
