import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, SumNodes, ExternalParamsSumNodes, LowRankSumParams

import pytest


def test_lowrank_sum_params():
    """
    The descriptor is pure configuration: a rank, plus the signature that layer grouping keys on.
    """

    ext_params = LowRankSumParams(rank = 5)

    assert ext_params.rank == 5
    assert ext_params.get_signature() == "LowRank_r5"

    # The rank is baked into the signature, so nodes of different rank never share a compiled layer
    assert LowRankSumParams(rank = 8).get_signature() != ext_params.get_signature()
    assert LowRankSumParams(rank = 5) == ext_params
    assert LowRankSumParams(rank = 8) != ext_params

    with pytest.raises(AssertionError):
        LowRankSumParams(rank = 0)


def test_external_sum_nodes():

    for block_size in [1, 2, 4, 8]:

        with juice.set_block_size(block_size):

            i0 = inputs(0, 3, dists.Categorical(num_cats = 5))
            m0 = multiply(i0)

            ns = summate(m0, num_node_blocks = 2, external_params = LowRankSumParams(rank = 5))
            plain_ns = summate(m0, num_node_blocks = 2)

        # A subclass of `SumNodes`, so everything that treats it as a sum node keeps working
        assert isinstance(ns, ExternalParamsSumNodes) and isinstance(ns, SumNodes)
        assert ns.is_sum() and ns.num_node_blocks == 2 and ns.block_size == block_size
        assert ns.get_external_signature() == "LowRank_r5"

        # `summate` without a descriptor is untouched
        assert type(plain_ns) is SumNodes

        # The low-rank factors are indexed per edge block, matching the block-sparse parameter layout
        batch_size, rank = 3, 5
        shape_U, shape_V = ns.external_params.tensor_shapes(ns, batch_size = batch_size)

        assert shape_U == (batch_size, ns.edge_ids.size(1), ns.ch_block_size, rank)
        assert shape_V == (batch_size, ns.edge_ids.size(1), ns.block_size, rank)

        # The descriptor is what makes the node an `ExternalParamsSumNodes`, so it cannot be omitted
        with pytest.raises(AssertionError):
            ExternalParamsSumNodes(2, [m0], block_size = block_size)


def test_external_sum_nodes_duplication():
    """
    Duplication has to preserve the node type and its descriptor -- otherwise the per-timestep copies
    of a tied transition would silently become plain sum nodes.
    """

    for block_size in [1, 4]:

        with juice.set_block_size(block_size):

            i0 = inputs(0, 3, dists.Categorical(num_cats = 5))
            m0 = multiply(i0)
            m1 = multiply(inputs(0, 3, dists.Categorical(num_cats = 5)))

            ns = summate(m0, num_node_blocks = 2, external_params = LowRankSumParams(rank = 5))
            plain_ns = summate(m0, num_node_blocks = 2)

        tied_ns = ns.duplicate(m1, tie_params = True)

        assert type(tied_ns) is ExternalParamsSumNodes
        assert tied_ns.external_params is ns.external_params
        assert tied_ns.is_tied() and tied_ns.get_source_ns() is ns

        untied_ns = ns.duplicate(m1, tie_params = False)

        assert type(untied_ns) is ExternalParamsSumNodes and not untied_ns.is_tied()
        assert untied_ns.external_params is ns.external_params

        # Plain sum nodes still duplicate into plain sum nodes
        assert type(plain_ns.duplicate(m1, tie_params = True)) is SumNodes

        # Tying across the two types would give a layer no consistent parameterization
        with pytest.raises(AssertionError):
            plain_ns.duplicate(m1, tie_params = False).set_source_ns(ns)


if __name__ == "__main__":
    test_lowrank_sum_params()
    test_external_sum_nodes()
    test_external_sum_nodes_duplication()
