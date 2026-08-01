import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, SumNodes, ExternalParamsSumNodes, LowRankSumParams
from pyjuice.transformations import deepcopy, prune_by_score
from pyjuice.transformations.merge import merge_sum_nodes, merge_by_region_node

import pytest


def _build_pc(block_size = 4, rank = 5):
    """
    A small PC whose first sum node carries an external parameterization.
    """

    with juice.set_block_size(block_size):

        i0 = inputs(0, 2, dists.Categorical(num_cats = 5))
        i1 = inputs(1, 2, dists.Categorical(num_cats = 5))
        i2 = inputs(2, 2, dists.Categorical(num_cats = 5))
        i3 = inputs(3, 2, dists.Categorical(num_cats = 5))

        m1 = multiply(i0, i1)
        n1 = summate(m1, num_node_blocks = 2, external_params = LowRankSumParams(rank = rank))

        m2 = multiply(i2, i3)
        n2 = summate(m2, num_node_blocks = 2)

        m = multiply(n1, n2)
        n = summate(m, num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    return n, n1


def _external_nodes(root_ns):
    return [ns for ns in root_ns if isinstance(ns, ExternalParamsSumNodes)]


def test_external_params_preserved_by_transformations():
    """
    Every transformation that rebuilds a sum node must carry the node type and its descriptor over.
    Dropping either one silently downgrades the node to a plain sum node -- the external tensors
    would then be ignored instead of erroring.
    """

    ## deepcopy ##

    root_ns, ns = _build_pc()
    new_nss = _external_nodes(deepcopy(root_ns))

    assert len(new_nss) == 1 and new_nss[0] is not ns
    assert new_nss[0].external_params.rank == 5

    ## deepcopy with parameter tying ##

    root_ns, ns = _build_pc()
    new_nss = _external_nodes(deepcopy(root_ns, tie_params = True))

    assert len(new_nss) == 1 and new_nss[0].external_params.rank == 5

    ## deepcopy that re-blocks: the tensor layout must follow the new blocks ##

    root_ns, ns = _build_pc(block_size = 4)
    new_nss = _external_nodes(deepcopy(root_ns, max_block_size = 2))

    assert len(new_nss) == 1 and new_nss[0].block_size == 2
    assert new_nss[0].external_params.tensor_shapes(new_nss[0], batch_size = 3)[1] == \
        (3, new_nss[0].edge_ids.size(1), 2, 5)

    ## blockify / unblockify ##

    root_ns, ns = _build_pc(block_size = 2)
    blocked_ns = juice.blockify(root_ns, max_target_block_size = 4, use_cuda = False)
    new_nss = _external_nodes(blocked_ns)

    assert len(new_nss) == 1 and new_nss[0].external_params.rank == 5

    new_nss = _external_nodes(juice.unblockify(blocked_ns, block_size = 1))

    assert len(new_nss) == 1 and new_nss[0].external_params.rank == 5

    ## prune: the edge blocks change, so the expected tensor shape must change with them ##

    root_ns, ns = _build_pc()
    for cs in root_ns:
        if cs.is_sum() and not cs.is_tied():
            cs._scores = torch.rand([cs.edge_ids.size(1)])
    ns._scores = torch.linspace(0.0, 1.0, ns.edge_ids.size(1))

    new_nss = _external_nodes(prune_by_score(root_ns, score_threshold = 0.5))

    assert len(new_nss) == 1 and new_nss[0].external_params.rank == 5
    assert new_nss[0].edge_ids.size(1) < ns.edge_ids.size(1)
    assert new_nss[0].external_params.tensor_shapes(new_nss[0], batch_size = 3)[0][1] == \
        new_nss[0].edge_ids.size(1)

    ## merge ##

    root_ns, ns = _build_pc()
    new_nss = _external_nodes(merge_by_region_node(root_ns))

    assert len(new_nss) == 1 and new_nss[0].external_params.rank == 5


def test_merge_rejects_mismatched_external_params():
    """
    A merged node is one node of one type, so merging is only defined when the inputs agree on the
    parameterization. Disagreement must be reported instead of resolved to the base class.
    """

    with juice.set_block_size(4):

        i0 = inputs(0, 3, dists.Categorical(num_cats = 5))
        m0 = multiply(i0)

        ns = summate(m0, num_node_blocks = 2, external_params = LowRankSumParams(rank = 5))
        plain_ns = summate(m0, num_node_blocks = 2)
        other_rank_ns = summate(m0, num_node_blocks = 2, external_params = LowRankSumParams(rank = 8))

    with pytest.raises(AssertionError):
        merge_sum_nodes(ns, plain_ns)

    with pytest.raises(AssertionError):
        merge_sum_nodes(ns, other_rank_ns)

    merged_ns = merge_sum_nodes(ns, ns.duplicate(m0, tie_params = False))

    assert type(merged_ns) is ExternalParamsSumNodes
    assert merged_ns.external_params.rank == 5
    assert merged_ns.num_node_blocks == 2 * ns.num_node_blocks


def test_plain_nodes_unaffected():
    """
    The rebuild hook is generic, so plain PCs must come out of every transformation exactly as before.
    """

    with juice.set_block_size(4):

        i0 = inputs(0, 2, dists.Categorical(num_cats = 5))
        i1 = inputs(1, 2, dists.Categorical(num_cats = 5))

        m = multiply(i0, i1)
        n = summate(m, num_node_blocks = 1)

    n.init_parameters(perturbation = 2.0)

    for new_ns in (deepcopy(n), merge_by_region_node(n), juice.blockify(n, use_cuda = False)):
        assert type(new_ns) is SumNodes
        assert len(_external_nodes(new_ns)) == 0


if __name__ == "__main__":
    test_external_params_preserved_by_transformations()
    test_merge_rejects_mismatched_external_params()
    test_plain_nodes_unaffected()
