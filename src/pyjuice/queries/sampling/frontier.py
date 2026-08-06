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


def push_non_neg_ones_to_front(matrix, dst = None, buffer = None):
    """
    Compact every column of `matrix` in place, moving its non-`-1` entries to the front while keeping
    their relative order, and return how many each column holds.

    Run between layers so the next layer's slot allocation starts from a dense prefix.

    The compaction is a scatter to `cumsum - 1`: an entry's destination row is how many kept entries
    precede it in its column. The boolean-mask form this replaces --

        result[d_mask] = matrix[s_mask]

    -- is shorter, but each boolean index goes through `nonzero`, whose output size is data-dependent,
    so it blocks the host until the device catches up. MEASURED on a gated 8-step HMM at batch 512:
    the whole sampling pass went from 3.80 ms to 3.44 ms, i.e. this one routine was ~9% of it.

    :note: moving the sampler's OTHER host-side bookkeeping (the per-column slot cursors in
           `assign_cids_ind_target` / `assign_nids_ind_target`) onto the device the same way was tried
           and is a REGRESSION -- 3.44 ms -> 3.85 ms for the sum-layer cursor alone, and 4.21 ms with
           the product-layer one as well. The top-down pass is bound by the NUMBER of launches, not by
           its synchronizations: those cursors need several extra kernels each (a scatter, a scan, a
           `repeat_interleave`) to replace one small copy and a numba loop that costs microseconds.
           This routine wins because it removes work without adding launches. Do not "finish the job"
           without re-measuring.

    :param dst: a destination-row map from a previous, identical call. Supplying it skips deriving
                one -- which is four of the seven operations here -- and is what the plan cache does
                on a structured-decomposable circuit, where the map is the same on every call.
    :param buffer: scratch of shape `[rows + 1, cols]` to scatter through, so the cached path does
                   not allocate one per layer per call.

    :returns: `(counts, dst)`. `counts` is the number of kept entries per column, or `None` when
              `dst` was supplied -- the caller then already has it cached too.
    """
    num_rows = matrix.size(0)

    counts = None
    if dst is None:
        kept = matrix != -1
        dst = kept.to(torch.long).cumsum(dim = 0) - 1

        # Dropped entries are scattered to a scratch row that is then discarded, which keeps the
        # scatter unconditional -- there is no way to mask a `scatter_`, and branching would cost more
        dst = torch.where(kept, dst, torch.full_like(dst, num_rows))
        counts = kept.sum(dim = 0)

    if buffer is None:
        buffer = matrix.new_full((num_rows + 1, matrix.size(1)), -1)
    else:
        buffer.fill_(-1)

    buffer.scatter_(0, dst, matrix)
    matrix.copy_(buffer[:num_rows])

    return counts, dst
