"""
Bookkeeping for the ancestral sampler's frontier.

The sampler carries two dense `[scopes, num_samples]` int64 buffers -- `node_samples` and
`element_samples` -- holding the node / element ids currently selected for each sample, with `-1`
marking an empty slot. Nothing here touches the GPU beyond the compaction: these are the routines
that decide *where* in those buffers a newly sampled id belongs, so that the kernels only ever have
to store to a precomputed address.
"""

from __future__ import annotations

import torch
from numba import njit


@njit
def assign_cids_ind_target(ind_target, element_pointers, ind_b, num_samples):
    """
    Allocate one slot of `element_samples` per selected sum node.

    `element_pointers[b]` is how many slots of column `b` are already used; each selected node takes
    the next one, so `ind_target[i]` is the flat index its sampled child will be stored at.
    """
    for i in range(ind_target.shape[0]):
        bid = ind_b[i]
        ind_t = element_pointers[bid]
        ind_target[i] = ind_t * num_samples + bid
        element_pointers[bid] = ind_t + 1


@njit
def assign_nids_ind_target(ind_target, ind_target_sid, node_pointers, ind_b, num_samples):
    """
    Allocate slots of `node_samples` for the children of every selected product node.

    A product node contributes as many children as it has, so the slots are variable-length runs:
    `ind_target_sid[n]` is where node `n`'s run starts within `ind_target`, and `node_pointers[b]`
    is the cursor into column `b`.
    """
    nid = 0
    for i in range(ind_target.shape[0]):
        if nid < ind_target_sid.shape[0] - 1 and i >= ind_target_sid[nid+1]:
            nid += 1
        bid = ind_b[nid]
        ind_t = node_pointers[bid]
        ind_target[i] = ind_t * num_samples + bid
        node_pointers[bid] = ind_t + 1


def push_non_neg_ones_to_front(matrix):
    """
    Compact every column of `matrix` in place, moving its non-`-1` entries to the front, and return
    how many each column holds.

    Run between layers so the next layer's slot allocation starts from a dense prefix.
    """

    result = torch.full_like(matrix, -1)

    s_mask = (matrix != -1)
    d_mask = torch.sum(s_mask, dim = 0, keepdims = True) > torch.arange(matrix.size(0), device = matrix.device)[:,None]

    result[d_mask] = matrix[s_mask]
    matrix[:] = result[:]

    return s_mask.long().sum(dim = 0)
