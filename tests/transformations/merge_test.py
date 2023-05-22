import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.transformations.merge import merge_sum_nodes, merge_prod_nodes, merge_by_region_node

import pytest


def sum_nodes_merge_test():
    num_nodes = 2

    i00 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i01 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i10 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i11 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    
    m00 = multiply(i00, i10)
    m01 = multiply(i01, i11)

    n0 = summate(m00, num_nodes = num_nodes)
    n1 = summate(m01, num_nodes = num_nodes)
    n2 = summate(m00, num_nodes = num_nodes)

    n_new = merge_sum_nodes(n0, n1)
    assert (n_new.edge_ids == torch.Tensor([[0,0,1,1,2,2,3,3],[0,1,0,1,2,3,2,3]])).all()
    assert len(n_new.chs) == 2
    assert n_new.chs[0] == m00
    assert n_new.chs[1] == m01

    n_new = merge_sum_nodes(n0, n2)
    assert (n_new.edge_ids == torch.Tensor([[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1]])).all()
    assert len(n_new.chs) == 1
    assert n_new.chs[0] == m00


def prod_nodes_merge_test():
    num_nodes = 2

    i00 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i01 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i10 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i11 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))

    m00 = multiply(i00, i10)
    m01 = multiply(i01, i11)
    m02 = multiply(i00, i10)

    m_new = merge_prod_nodes(m00, m01)
    assert (m_new.edge_ids == torch.Tensor([[0,0],[1,1],[2,2],[3,3]])).all()
    assert m_new.chs[0].chs[0] == i00
    assert m_new.chs[0].chs[1] == i01
    assert m_new.chs[1].chs[0] == i10
    assert m_new.chs[1].chs[1] == i11

    m_new = merge_prod_nodes(m00, m02)
    assert (m_new.edge_ids == torch.Tensor([[0,0],[1,1],[0,0],[1,1]])).all()
    assert m_new.chs[0] == i00
    assert m_new.chs[1] == i10


def merge_by_region_node_test():
    num_nodes = 2

    i00 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i01 = inputs(0, num_nodes, dists.Categorical(num_cats = 5))
    i10 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i11 = inputs(1, num_nodes, dists.Categorical(num_cats = 5))
    i20 = inputs(2, num_nodes, dists.Categorical(num_cats = 5))
    i30 = inputs(3, num_nodes, dists.Categorical(num_cats = 5))

    m00 = multiply(i00, i10)
    m01 = multiply(i01, i11)
    m02 = multiply(i00, i10)
    m10 = multiply(i20, i30)

    n00 = summate(m00, num_nodes = num_nodes)
    n01 = summate(m01, m02, num_nodes = num_nodes)
    n10 = summate(m10, num_nodes = num_nodes)

    m20 = multiply(n00, n10)
    m21 = multiply(n01, n10)

    n = summate(m20, m21, num_nodes = 1)

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
    assert new_n.chs[0].chs[0].chs[0].chs[0].chs[0] == i00
    assert new_n.chs[0].chs[0].chs[0].chs[0].chs[1] == i01
    assert new_n.chs[0].chs[0].chs[0].chs[1].chs[0] == i10
    assert new_n.chs[0].chs[0].chs[0].chs[1].chs[1] == i11
    assert new_n.chs[0].chs[1].chs[0].chs[0] == i20
    assert new_n.chs[0].chs[1].chs[0].chs[1] == i30


if __name__ == "__main__":
    sum_nodes_merge_test()
    prod_nodes_merge_test()
    merge_by_region_node_test()