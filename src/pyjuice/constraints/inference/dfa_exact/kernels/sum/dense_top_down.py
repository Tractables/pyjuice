"""
The dense sum's top-down pass, for a batch at once or a sample at a time; the rows no launch writes.
"""

from typing import Optional
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv
from ...external_params.block_scale import sample_weights
from .dense_common import (_EW_TILE, _bf16_products, _bf16_weights, _live_shifted_exp, _log_accumulate_kernel,
                           _log_add_kernel, _log_scatter_accumulate_kernel, _log_scatter_kernel, _matmul_precision,
                           _shifted_exp_kernel)
from .dense_live import _TRANSPOSED_PRODUCTS, _bf16_live_step


def per_sample_dense_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor,
                                  dense: dict, region: tuple, elem_first: int, elem_width: int, ncols: int,
                                  precision: str, gated: dict, spans: list, ident: torch.Tensor, bf16: bool = False):
    """
    The top-down values of a sum part under external parameters that is one dense run (``dense``: a fused group's, see
    :func:`~..compiled.top_down_program._fused_tables`) into its children's rows of ``element_flows`` (no earlier launch
    wrote them), on compact rows (``ncols`` lanes; ``ident`` their identity order): ``O_elem = log(W_b^T @ exp(O_node -
    m)) + m`` on every sample's lanes (``spans``: :func:`sample_spans`), ``W_b`` the sample's weights with its gates and
    ``1 / Z_b`` in them (:func:`~..external_params.block_scale.sample_weights`, one matrix at a time) -- the dense
    path's live-column products (transposed; bf16 operands at a bf16 query where ``torch.mm`` can, see
    :func:`dense_sum_top_down`) by sample, which the fused kernels compute with every parent block's gate instead.

    :param region: ``(offset, width, first)`` of the run's sum region in ``node_flows``
    :param gated: the part's gates (:func:`~..queries.backend.forward.part_gates`)
    """
    run, (nids, _, pids, _, _, _, gate) = dense["run"], dense["part"]
    c0, E, nid0, R = run["c0"], run["E"], run["nid0"], run["rows"]
    offset, width, first = region
    elems = element_flows[:element_flows.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_flows[offset:offset + (node_flows.numel() - offset) // width * width].view(-1, width)
    bf = bf16 and _bf16_products(nodes.device)
    ex_t, m = _live_shifted_exp(nodes[nid0 - first:nid0 - first + R], ident, ncols,                 # [lanes, rows]
                                torch.bfloat16 if bf else None)
    prod = torch.empty(ncols, E, dtype = torch.float32, device = nodes.device)                      # [lanes, E]
    w = torch.empty(R, E, dtype = torch.bfloat16 if bf else torch.float32, device = nodes.device)
    kw = dict(out_dtype = torch.float32) if bf else {}
    with _matmul_precision(precision):
        for b, o, n in spans:
            sample_weights(w, params, pids, nids, gate, gated, b, R // nids.numel())
            torch.mm(ex_t[o:o + n], w, out = prod[o:o + n], **kw)
    TILE_R, TILE_C = _EW_TILE
    num_ctiles = host_cdiv(ncols, TILE_C)
    _log_scatter_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](elems[c0 - elem_first:c0 - elem_first + E], prod, m,
                                                              ident, E, ncols, elem_width, ncols, num_ctiles,
                                                              TILE_R = TILE_R, TILE_C = TILE_C, TRANS = True)


def dense_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor, blocks, E: int,
                      block_size: int, region: tuple, c0: int, elem_first: int, elem_width: int, ncols: int,
                      precision: str, add: bool, live = None, stacked = None, piece = None, bf16: bool = False,
                      copies: Optional[dict] = None):
    """
    The top-down values of the ``E`` children ``c0 ..`` of a run of dense node blocks of one sum region:
    ``O_elem = log(W^T @ exp(O_node - m)) + m`` with ``m`` the column maximum of the node blocks' top-down values,
    written to ``element_flows``, or log-added to it (``add``); ``W^T`` is the parameter buffer's own ``[E, BS]``
    view per block (``stacked``: the blocks' :func:`stacked_weights`, transposed for free).

    :param blocks: ``[(k, nid, p0)]`` as in :func:`dense_sum`'s groups (consecutive rows when ``stacked``)
    :param region: ``(offset, width, first)`` of the node blocks' sum region
    :param live: None, or the interval's ``(perm, rank, num_live)`` (see :meth:`~.live.LaneOrder.segment`)
    :param piece: None, or ``(lo, hi)``: only the children ``c0 + lo .. c0 + hi``
    :param bf16, copies: the live columns' products on bf16 operands, as in :func:`dense_sum`
    """
    offset, width, first = region
    BS = block_size
    lo, hi = piece if piece is not None else (0, E)
    bf16 = bf16 and live is not None and _bf16_products(node_flows.device)
    elems = element_flows[:element_flows.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_flows[offset:offset + (node_flows.numel() - offset) // width * width].view(-1, width)
    out = elems[c0 + lo - elem_first:c0 + hi - elem_first]
    if bf16:                                                     # (the bf16 copies of the whole run's weights)
        products = [(w.t()[lo:hi], nid, w.size(0)) for w, nid in _bf16_weights(copies, blocks, params, E, BS, stacked)]
    E = hi - lo
    TILE_R, TILE_C = _EW_TILE
    if not bf16:
        products = [(stacked.t()[lo:hi], blocks[0][1], stacked.size(0))] if stacked is not None else \
            [(params[p0 + lo * BS:p0 + hi * BS].view(E, BS), nid, BS) for _, nid, p0 in blocks]
    with _matmul_precision(precision):
        for wt, nid, rows in products:
            x = nodes[nid - first:nid - first + rows, :ncols]
            if live is None:
                m = x.amax(dim = 0)
                ex = torch.empty(rows, ncols, dtype = x.dtype, device = x.device)
                num_ctiles = host_cdiv(ncols, TILE_C)
                _shifted_exp_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](x, ex, m, rows, ncols, width, num_ctiles,
                                                                            TILE_R = TILE_R, TILE_C = TILE_C)
                if add:
                    prod = torch.mm(wt, ex)
                    _log_accumulate_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](
                        out, prod, m, E, ncols, elem_width, num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C)
                else:
                    torch.mm(wt, ex, out = out[:, :ncols])
                    _log_add_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](out, m, E, ncols, elem_width, num_ctiles,
                                                                         TILE_R = TILE_R, TILE_C = TILE_C)
            elif bf16:
                perm, rank, num_live = live
                _bf16_live_step(x, perm, rank, num_live, ncols, [(wt, out, elem_width, 2 if add else 0)])
            else:
                perm, rank, num_live = live
                trans = precision in _TRANSPOSED_PRODUCTS
                ex_t, m = _live_shifted_exp(x, perm, num_live)                                    # [live, rows]
                prod = torch.mm(ex_t, wt.t()) if trans else torch.mm(wt, ex_t.t())             # [live, E] / [E, live]
                num_ctiles = host_cdiv(ncols, TILE_C)
                kernel = _log_scatter_accumulate_kernel if add else _log_scatter_kernel
                kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](out, prod, m, rank, E, ncols, elem_width, num_live,
                                                            num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C, TRANS = trans)
            add = True                                                        # the run's later blocks add to the first


@triton_jit
def _fill_dead_rows_kernel(node_flows, rows, reg_offset, reg_width, reg_first, num_tiles, TILE: tl.constexpr):
    """Sum rows of ``node_flows`` set to -inf over their whole width: one program per (``(row, region)`` of ``rows``,
    tile of ``TILE`` columns)."""
    k = tl.program_id(0) // num_tiles
    row = tl.load(rows + k * 2)
    reg = tl.load(rows + k * 2 + 1)
    width = tl.load(reg_width + reg)
    base = node_flows + tl.load(reg_offset + reg) + (row - tl.load(reg_first + reg)).to(tl.int64) * width
    c = (tl.program_id(0) % num_tiles) * TILE + tl.arange(0, TILE)
    tl.store(base + c, tl.full([TILE], float("-inf"), dtype = tl.float32), mask = c < width)


def fill_dead_rows(node_flows: torch.Tensor, rows: torch.Tensor, regions: dict, max_width: int):
    """
    The top-down value -inf on the sum rows no chain of edges from the root reaches (no launch writes them).

    :param rows: int32 [D, 2], ``(row, region)`` per dead row
    :param regions: ``offset``, ``width`` and ``first`` of every sum region (device tensors)
    :param max_width: the widest of their regions' rows
    """
    TILE = 1024
    num_tiles = host_cdiv(max_width, TILE)
    _fill_dead_rows_kernel[(rows.size(0) * num_tiles,)](node_flows, rows, regions["offset"], regions["width"],
                                                        regions["first"], num_tiles, TILE = TILE)
