import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, SumNodes, ParameterizedSumNodes
from pyjuice.parameterizations import LowRank

import pytest


def test_lowrank_sum_params():
    """
    The descriptor is pure configuration: a rank and a form, plus the signature that layer grouping keys on.
    """

    ext_params = LowRank(rank = 5)

    assert ext_params.rank == 5 and ext_params.form == "exp_sum"
    assert ext_params.get_signature() == "LowRank_exp_sum_r5"

    # ... and so are the form, the tying of the tensors and whether the denominator flow is kept
    assert LowRank(rank = 5, form = "sum_exp").get_signature() == "LowRank_sum_exp_r5"
    assert LowRank(rank = 5, form = "sum_exp") != ext_params
    assert LowRank(rank = 5, tie_external = True) != ext_params
    assert LowRank(rank = 5, apply_z_correction = False) != ext_params

    # The rank is baked into the signature, so nodes of different rank never share a compiled layer
    assert LowRank(rank = 8).get_signature() != ext_params.get_signature()
    assert LowRank(rank = 5) == ext_params
    assert LowRank(rank = 8) != ext_params

    with pytest.raises(AssertionError):
        LowRank(rank = 0)


def test_parameterized_sum_nodes():

    for block_size in [1, 2, 4, 8]:

        with juice.set_block_size(block_size):

            i0 = inputs(0, 3, dists.Categorical(num_cats = 5))
            m0 = multiply(i0)

            ns = summate(m0, num_node_blocks = 2, parameterization = LowRank(rank = 5))
            plain_ns = summate(m0, num_node_blocks = 2)

        # A subclass of `SumNodes`, so everything that treats it as a sum node keeps working
        assert isinstance(ns, ParameterizedSumNodes) and isinstance(ns, SumNodes)
        assert ns.is_sum() and ns.num_node_blocks == 2 and ns.block_size == block_size
        assert ns.get_parameterization_signature() == "LowRank_exp_sum_r5"

        # `summate` without a descriptor is untouched
        assert type(plain_ns) is SumNodes

        # The low-rank factors are indexed by node and by child, whatever blocks the node vector is stored in
        batch_size, rank = 3, 5
        shape_A, shape_B = ns.parameterization.tensor_shapes(ns, batch_size = batch_size)

        assert shape_A == (batch_size, ns.num_nodes, rank)
        assert shape_B == (batch_size, rank, ns.num_ch_nodes)

        # The descriptor is what makes the node a `ParameterizedSumNodes`, so it cannot be omitted
        with pytest.raises(AssertionError):
            ParameterizedSumNodes(2, [m0], block_size = block_size)


def test_parameterized_sum_nodes_duplication():
    """
    Duplication has to preserve the node type and its descriptor -- otherwise the per-timestep copies
    of a tied transition would silently become plain sum nodes.
    """

    for block_size in [1, 4]:

        with juice.set_block_size(block_size):

            i0 = inputs(0, 3, dists.Categorical(num_cats = 5))
            m0 = multiply(i0)
            m1 = multiply(inputs(0, 3, dists.Categorical(num_cats = 5)))

            ns = summate(m0, num_node_blocks = 2, parameterization = LowRank(rank = 5))
            plain_ns = summate(m0, num_node_blocks = 2)

        tied_ns = ns.duplicate(m1, tie_params = True)

        assert type(tied_ns) is ParameterizedSumNodes
        assert tied_ns.parameterization is ns.parameterization
        assert tied_ns.is_tied() and tied_ns.get_source_ns() is ns

        untied_ns = ns.duplicate(m1, tie_params = False)

        assert type(untied_ns) is ParameterizedSumNodes and not untied_ns.is_tied()
        assert untied_ns.parameterization is ns.parameterization

        # Plain sum nodes still duplicate into plain sum nodes
        assert type(plain_ns.duplicate(m1, tie_params = True)) is SumNodes

        # Tying across the two types would give a layer no consistent parameterization
        with pytest.raises(AssertionError):
            plain_ns.duplicate(m1, tie_params = False).set_source_ns(ns)


def test_the_former_keyword_is_refused():
    """`summate` hands the keywords it does not know to the nodes' init callbacks, which ignore them: `external_params =`
    would leave a plain sum node, and nothing would say so until a forward was given tensors for it."""
    with juice.set_block_size(4):
        m = multiply(inputs(0, 2, dists.Categorical(num_cats = 5)), inputs(1, 2, dists.Categorical(num_cats = 5)))
        with pytest.raises(TypeError, match = "parameterization"):
            summate(m, num_node_blocks = 2, external_params = LowRank(rank = 5))


if __name__ == "__main__":
    test_lowrank_sum_params()
    test_parameterized_sum_nodes()
    test_parameterized_sum_nodes_duplication()
