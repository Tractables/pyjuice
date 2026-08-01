import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.transformations.merge import merge_sum_nodes, merge_prod_nodes, merge_by_region_node

import pytest


def test_sum_nodes_merge():
    num_node_blocks = 2

    for block_size in [1, 2, 4, 8]:
        
        with juice.set_block_size(block_size):

            i00 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
            i01 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
            i10 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
            i11 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
            
            m00 = multiply(i00, i10)
            m01 = multiply(i01, i11)

            n0 = summate(m00, num_node_blocks = num_node_blocks)
            n1 = summate(m01, num_node_blocks = num_node_blocks)
            n2 = summate(m00, num_node_blocks = num_node_blocks)

            n_new = merge_sum_nodes(n0, n1)
            assert (n_new.edge_ids == torch.Tensor([[0,0,1,1,2,2,3,3],[0,1,0,1,2,3,2,3]])).all()
            assert len(n_new.chs) == 2
            assert n_new.chs[0] == m00
            assert n_new.chs[1] == m01

            n_new = merge_sum_nodes(n0, n2)
            assert (n_new.edge_ids == torch.Tensor([[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1]])).all()
            assert len(n_new.chs) == 1
            assert n_new.chs[0] == m00


def test_prod_nodes_merge():
    num_node_blocks = 2

    for block_size in [1, 2, 4, 8]:
        
        with juice.set_block_size(block_size):

            i00 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
            i01 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
            i10 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
            i11 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))

            m00 = multiply(i00, i10)
            m01 = multiply(i01, i11)
            m02 = multiply(i00, i10)

            m_new = merge_prod_nodes(m00, m01)
            assert (m_new.edge_ids == torch.Tensor([[0,0],[1,1],[2,2],[3,3]])).all()
            assert m_new.chs[0].chs[0].chs[0] == i00
            assert m_new.chs[0].chs[1].chs[0] == i01
            assert m_new.chs[1].chs[0].chs[0] == i10
            assert m_new.chs[1].chs[1].chs[0] == i11

            m_new = merge_prod_nodes(m00, m02)
            assert (m_new.edge_ids == torch.Tensor([[0,0],[1,1],[0,0],[1,1]])).all()
            assert m_new.chs[0] == i00
            assert m_new.chs[1] == i10


def test_merge_by_region_node():
    num_node_blocks = 2

    for block_size in [1, 2, 4, 8]:
        
        with juice.set_block_size(block_size):

            i00 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
            i01 = inputs(0, num_node_blocks, dists.Categorical(num_cats = 5))
            i10 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
            i11 = inputs(1, num_node_blocks, dists.Categorical(num_cats = 5))
            i20 = inputs(2, num_node_blocks, dists.Categorical(num_cats = 5))
            i30 = inputs(3, num_node_blocks, dists.Categorical(num_cats = 5))

            m00 = multiply(i00, i10)
            m01 = multiply(i01, i11)
            m02 = multiply(i00, i10)
            m10 = multiply(i20, i30)

            n00 = summate(m00, num_node_blocks = num_node_blocks)
            n01 = summate(m01, m02, num_node_blocks = num_node_blocks)
            n10 = summate(m10, num_node_blocks = num_node_blocks)

            m20 = multiply(n00, n10)
            m21 = multiply(n01, n10)

            n = summate(m20, m21, num_node_blocks = 1, block_size = 1)

            new_n = merge_by_region_node(n)
            
            assert (new_n.edge_ids == torch.Tensor([[0,0,0,0],[0,1,2,3]])).all()
            assert len(new_n.chs) == 1
            assert (new_n.chs[0].edge_ids == torch.Tensor([[0,0],[1,1],[2,0],[3,1]])).all()
            assert len(new_n.chs[0].chs) == 2
            assert (new_n.chs[0].chs[0].edge_ids == torch.Tensor([[0,0,1,1,2,2,2,2,3,3,3,3],[0,1,0,1,2,3,4,5,2,3,4,5]])).all()
            assert len(new_n.chs[0].chs[0].chs) == 1
            assert (new_n.chs[0].chs[1].edge_ids == torch.Tensor([[0,0,1,1],[0,1,0,1]])).all()
            assert len(new_n.chs[0].chs[1].chs) == 1
            assert (new_n.chs[0].chs[0].chs[0].edge_ids == torch.Tensor([[0,0],[1,1],[2,2],[3,3],[0,0],[1,1]])).all()
            assert len(new_n.chs[0].chs[0].chs[0].chs) == 2
            assert (new_n.chs[0].chs[1].chs[0].edge_ids == torch.Tensor([[0,0],[1,1]])).all()
            assert len(new_n.chs[0].chs[1].chs[0].chs) == 2
            assert new_n.chs[0].chs[0].chs[0].chs[0].chs[0].chs[0] == i00
            assert new_n.chs[0].chs[0].chs[0].chs[0].chs[1].chs[0] == i01
            assert new_n.chs[0].chs[0].chs[0].chs[1].chs[0].chs[0] == i10
            assert new_n.chs[0].chs[0].chs[0].chs[1].chs[1].chs[0] == i11
            assert new_n.chs[0].chs[1].chs[0].chs[0] == i20
            assert new_n.chs[0].chs[1].chs[0].chs[1] == i30


def test_merge_by_region_node_repeated_scope():
    """
    Region nodes are hashed by scope alone, so a PC that stacks layers over one scope has the same
    region appearing at several depths. Nodes at different depths must not be grouped together: a
    node would be grouped with its own descendants, which is an invalid merge and leaves the groups
    mutually dependent, so no processing order exists.
    """

    for block_size in [1, 4]:
        for depth in [1, 2, 4]:

            with juice.set_block_size(block_size):

                i0 = inputs(0, 2, dists.Categorical(num_cats = 5))

                ns = i0
                for _ in range(depth):
                    ns = summate(multiply(ns), num_node_blocks = 2)

                n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

            n.init_parameters(perturbation = 2.0)

            new_n = merge_by_region_node(n)

            # Nothing to merge here -- every layer is its own group, so the PC comes back intact
            assert len(list(new_n)) == len(list(n))
            assert new_n.num_nodes == n.num_nodes and new_n.block_size == n.block_size

            old_ns, new_ns = n, new_n
            for _ in range(depth):
                old_ns, new_ns = old_ns.chs[0].chs[0], new_ns.chs[0].chs[0]

                assert type(new_ns) is type(old_ns)
                assert new_ns.num_nodes == old_ns.num_nodes and new_ns.block_size == old_ns.block_size
                assert torch.all(new_ns.edge_ids == old_ns.edge_ids)
                assert torch.all(torch.abs(new_ns._params - old_ns._params) < 1e-6)

    # And the merge is semantics-preserving
    with juice.set_block_size(4):
        i0 = inputs(0, 2, dists.Categorical(num_cats = 5))
        ns = summate(multiply(summate(multiply(i0), num_node_blocks = 2)), num_node_blocks = 1, block_size = 1)

    ns.init_parameters(perturbation = 2.0)

    data = torch.randint(0, 5, [16, 1]).to(torch.device("cuda:0"))

    lls = juice.compile(ns).to(torch.device("cuda:0"))(data)
    merged_lls = juice.compile(merge_by_region_node(ns)).to(torch.device("cuda:0"))(data)

    assert torch.all(torch.abs(lls - merged_lls) < 1e-4)


def test_merge_by_region_node_parameter_tying():
    """
    A tied node stores no parameters of its own, so rebuilding it has to look them up through its
    source -- otherwise the merged node comes out unparameterized and gets randomly initialized at
    compile time, silently changing the PC. Where the tie can still be expressed after merging it
    should also be re-established, so the merged PC keeps sharing those parameters during training.
    """

    device = torch.device("cuda:0")

    ## A homogeneous HMM: every transition but the first is tied to the first ##

    torch.manual_seed(42)

    root_ns = juice.structures.HMM(seq_length = 5, num_latents = 4, num_emits = 5, homogeneous = True)
    root_ns.init_parameters(perturbation = 2.0)

    new_ns = merge_by_region_node(root_ns)

    assert sum([ns.is_tied() for ns in new_ns]) == sum([ns.is_tied() for ns in root_ns])

    for ns in new_ns:
        if ns.is_sum() and ns.is_tied():
            assert ns._params is None      # the parameters live on the source
            assert ns.has_params()         # ... and are reachable through it

    # All the tied transitions still share one source, so EM keeps updating them together
    tied_sources = set([id(ns.get_source_ns()) for ns in new_ns if ns.is_sum() and ns.is_tied()])
    assert len(tied_sources) == 1

    data = torch.randint(0, 5, [16, len(root_ns.scope)]).to(device)

    lls = juice.compile(root_ns).to(device)(data)
    merged_lls = juice.compile(new_ns).to(device)(data)

    assert torch.all(torch.abs(lls - merged_lls) < 1e-4)

    ## Merging tied nodes into one: the tie cannot survive, but the parameters must ##

    with juice.set_block_size(4):

        i0 = inputs(0, 2, dists.Categorical(num_cats = 5))
        i1 = inputs(1, 2, dists.Categorical(num_cats = 5))

        m = multiply(i0, i1)

        source_ns = summate(m, num_node_blocks = 2)
        source_ns.init_parameters(perturbation = 2.0)

        tied_ns = source_ns.duplicate(m, tie_params = True)

    merged_ns = merge_sum_nodes(source_ns, tied_ns)

    num_edges = source_ns.edge_ids.size(1)

    assert not merged_ns.is_tied()
    assert merged_ns._params is not None
    assert torch.all(torch.abs(merged_ns._params[:num_edges] - source_ns._params) < 1e-6)
    assert torch.all(torch.abs(merged_ns._params[num_edges:] - source_ns._params) < 1e-6)


if __name__ == "__main__":
    test_sum_nodes_merge()
    test_prod_nodes_merge()
    test_merge_by_region_node()
    test_merge_by_region_node_repeated_scope()
    test_merge_by_region_node_parameter_tying()