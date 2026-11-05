"""
The dense forward sum: a node block's weights in place as a matrix, its children shifted by their column maximum and exponentiated once,
a matrix product and the log taken back (see :mod:`pyjuice.constraints.inference.dfa_exact.kernels.sum`).
"""

from typing import Optional
import torch

from pyjuice.utils.util import host_cdiv
from .dense_common import _EW_TILE, _bf16_products, _log_add_kernel, _matmul_precision, _products, _shifted_exp_kernel
from .dense_live import _dense_sum_live


#: Dense node blocks smaller than this run the fused kernel: one cuBLAS call per node block only pays for itself
#: on large blocks (a partition of many small blocks is one fused launch instead).
DENSE_MIN_BLOCK = 64


# -------------------------------------------------------------------------------------------------
# Dense node blocks
# -------------------------------------------------------------------------------------------------

def dense_groups(nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor, block_size: int, elem_first: int):
    """
    Split a partition's node blocks into dense ones, grouped by their run of children, and the rest.

    :returns: ``(groups, rest)``: ``groups`` maps the first child row to ``[(node block index, nid, pid0)]`` for
        node blocks whose children are ``E`` contiguous element rows and whose weights are ``params[pid0 + e * BS
        + i]``; ``rest`` is a long tensor of the other node blocks' indices
    """
    E = cids.size(1)
    e = torch.arange(E, device = cids.device)
    dense = (cids == cids[:, :1] + e).all(dim = 1) & (pids == pids[:, :1] + e * block_size).all(dim = 1) & \
        (cids[:, 0] >= elem_first)
    if block_size < DENSE_MIN_BLOCK:
        dense[:] = False
    groups = {}
    for k, ok, nid, c0, p0 in zip(range(cids.size(0)), dense.tolist(), nids.tolist(), cids[:, 0].tolist(),
                                  pids[:, 0].tolist()):
        if ok:
            groups.setdefault(c0, []).append((k, nid, p0))
    rest = torch.nonzero(~dense).flatten()
    return groups, rest


def dense_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, groups: dict, E: int,
              block_size: int, region: tuple, elem_first: int, elem_width: int, ncols: int, precision: str,
              live = None, stacked: dict = None, bf16: bool = False, copies: Optional[dict] = None,
              keep: bool = False):
    """
    Dense node blocks of one sum region (see :func:`dense_groups`), into ``node_mars``: per run of children,
    ``log(W_k @ exp(X - m)) + m`` with ``m`` the column maximum of the children ``X``.

    :param region: ``(offset, width, first)`` of the node blocks' sum region (Python ints)
    :param live: None (every column), or the region's interval's ``(perm, rank, num_live)`` (see
        :meth:`~.live.LaneOrder.segment`): the live columns are gathered and multiplied, the others set to -inf
    :param stacked: first child row -> the :func:`stacked_weights` of that run's node blocks, which occupy
        consecutive rows in the order listed: one product for the run instead of one per block
    :param bf16: the live columns' products on bf16 operands, summed and returned in fp32 (a ``precision = "bf16"``
        query's, ``precision`` then "tf32"): the exponentials, and the weights' copies kept in ``copies`` (the
        query's :meth:`~..state.State.weight_stacks`), rounded to bf16 -- as at ``precision`` where ``torch.mm``
        cannot (:func:`_bf16_products`), and for every column (no ``live``)
    :param keep: with ``live``, the columns not live left as they are (their values held), not set to -inf
    """
    offset, width, first = region
    BS = block_size
    stacked = stacked or {}
    elems = element_mars[:element_mars.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_mars[offset:offset + (node_mars.numel() - offset) // width * width].view(-1, width)
    TILE_R, TILE_C = _EW_TILE
    num_ctiles = host_cdiv(ncols, TILE_C)
    if live is not None:
        _dense_sum_live(nodes, elems, params, groups, E, BS, first, elem_first, ncols, precision, live, stacked,
                        copies if bf16 and _bf16_products(nodes.device) else None, keep)
        return
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            x = elems[c0 - elem_first:c0 - elem_first + E, :ncols]
            m = x.amax(dim = 0)
            ex = torch.empty(E, ncols, dtype = x.dtype, device = x.device)
            _shifted_exp_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](x, ex, m, E, ncols, elem_width, num_ctiles,
                                                                     TILE_R = TILE_R, TILE_C = TILE_C)
            for w, nid in _products(blocks, params, E, BS, stacked.get(c0)):
                rows = w.size(0)
                out = nodes[nid - first:nid - first + rows, :ncols]
                torch.mm(w, ex, out = out)
                _log_add_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](out, m, rows, ncols, width, num_ctiles,
                                                                        TILE_R = TILE_R, TILE_C = TILE_C)
