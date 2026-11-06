"""
The dense sum under evidence pruning: the live columns alone, gathered, multiplied and scattered back, by a chain of kernels or one
kernel fused (``c/dense_live_mma.cu``), the tuner's choice.
"""

import torch

from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .dense_common import (_EW_TILE, _bf16_weights, _live_shifted_exp, _log_scatter_accumulate_kernel,
                           _log_scatter_kernel, _matmul_precision, _products)


#: The precisions whose dense live-column products come out transposed, ``[live, rows] = ex_t @ W^T``: those on tensor
#: cores, where cuBLAS picks faster kernels for that than for ``W @ ex_t^T`` [rows, live]. Over every product of an HMM
#: ex108 B8 diffusion session (M = K = 4096, 1 to 83k live columns; toast): TF32 1806 -> 1469 ms, faster at every width,
#: the weights stacked or not; fp32 (SIMT kernels) 4423 -> 4480 ms, so its products stay [rows, live].
_TRANSPOSED_PRODUCTS = ("tf32",)


def _bf16_live_chain(x, perm, rank, num_live, ncols, products):
    """The live-column products of :func:`_bf16_live_step` by the chain of kernels: the live columns' bf16 exponentials
    once (:func:`_live_shifted_exp`), then per product ``torch.mm`` on bf16 operands into fp32, logged and scattered."""
    TILE_R, TILE_C = _EW_TILE
    num_ctiles = host_cdiv(ncols, TILE_C)
    ex_t, m = _live_shifted_exp(x, perm, num_live, torch.bfloat16)                                     # [live, K]
    for w, out, ld, mode in products:
        rows = w.size(0)
        prod = torch.mm(ex_t, w.t(), out_dtype = torch.float32)                                       # [live, rows]
        if mode == 2:
            _log_scatter_accumulate_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](
                out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C, TRANS = True)
        else:
            _log_scatter_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](
                out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C, TRANS = True,
                KEEP = mode == 1)


def _dense_live_operand(w: torch.Tensor):
    """``(k_major, ld)`` by which the fused dense live product reads the bf16 weights ``w`` [rows, K] -- K-major (rows
    of ``ld``) or MN-major (columns of ``ld``) --, or None where it cannot (its 16-byte copies: an aligned base, ``ld`` a
    multiple of 8)."""
    if w.dtype != torch.bfloat16 or w.dim() != 2 or w.data_ptr() % 16:
        return None
    if w.stride(1) == 1 and w.stride(0) % 8 == 0:
        return True, w.stride(0)
    if w.stride(0) == 1 and w.stride(1) % 8 == 0:
        return False, w.stride(1)
    return None


def _bf16_live_step(x, perm, rank, num_live, ncols, products):
    """
    The live-column products of a bf16 query's dense step: per ``(w, out, ld, mode)`` of ``products``, the bf16 weights
    ``w`` [rows, K] over the rows of ``x`` [K, *] (fp32; its live lanes the columns ``perm[:num_live]``) into ``out``'s
    first ``ncols`` columns (fp32, rows of ``ld``; column ``c`` the lane ``rank[c]``, -1 dead): ``log(w @ exp(x[:,
    perm] - m)) + m`` written (``mode`` 0, the dead columns -inf; 1, the dead ones kept) or log-added (2, the dead ones
    kept) -- by the chain (:func:`_bf16_live_chain`) or, below 64 live lanes, by the fused tensor-core kernel
    (``kernels/c/dense_live_mma.cu``: the gather, the exponentials, the product, the log and the scatter in two
    launches; the chain's operand precision, its own summation order), whichever measured faster at the step's shape:
    `pyjuice.layer.kernels.autotune.choose`, once per shape, the chain the reference -- both into scratch outputs, as a
    log-add is not idempotent.
    """
    from ..c import dense_live_arms, get_dense_live_module
    K = x.size(0)
    arms = dense_live_arms(products[0][0].size(0), K, num_live) if products else []
    ops = [_dense_live_operand(w) for w, _, _, _ in products] if arms else []
    if not arms or any(op is None for op in ops) or perm.dtype != torch.int32 or rank.dtype != torch.int32 \
            or x.dtype != torch.float32 or x.stride(1) != 1 or any(w.size(0) != products[0][0].size(0) for w, *_ in products):
        _bf16_live_chain(x, perm, rank, num_live, ncols, products)
        return
    from pyjuice.layer.kernels import autotune
    mod = get_dense_live_module()

    def fused(code, outs):
        for (w, _, _, mode), (k_major, a_ld), (out, ld) in zip(products, ops, outs):
            mod.dense_live_mma(out, ld, w, a_ld, k_major, x, x.stride(0), perm, rank, w.size(0), K, num_live, ncols, mode,
                               code)

    rows = products[0][0].size(0)
    # (the arms stand for the lanes' column tile and the splits; the layouts and modes, per product; whether the lanes
    # are every column -- a compact row's, in order -- or a pruned subset: the fused kernel gathers x's live columns per
    # row tile, cheap for runs of lanes, dear for lanes scattered over the columns)
    key = ("dfa_dense_live", rows, K, tuple(arms), host_next_power_of_2(ncols), num_live == ncols,
           tuple((op[0], mode) for op, (_, _, _, mode) in zip(ops, products)))
    choice = autotune.cached(key)
    if choice is None:
        scratch = [(torch.empty(rows, ncols, device = x.device), ncols) for _ in products]
        chain_scratch = [(w, out, ld, mode) for (w, _, _, mode), (out, ld) in zip(products, scratch)]
        choice = autotune.choose(key, [("chain", lambda: _bf16_live_chain(x, perm, rank, num_live, ncols, chain_scratch))]
                                 + [(code, (lambda code = code: fused(code, scratch))) for code in arms])
    if choice == "chain":
        _bf16_live_chain(x, perm, rank, num_live, ncols, products)
    else:
        fused(choice, [(out, ld) for _, out, ld, _ in products])


def _dense_sum_live(nodes, elems, params, groups, E, BS, first, elem_first, ncols, precision, live, stacked,
                    copies = None, keep = False):
    """:func:`dense_sum` on the live columns (``copies``: the bf16 weights' dict, products on bf16 operands; ``keep``:
    the others left as they are)."""
    perm, rank, num_live = live
    width = nodes.size(1)
    TILE_R, TILE_C = _EW_TILE
    bf16 = copies is not None
    trans = precision in _TRANSPOSED_PRODUCTS
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            x = elems[c0 - elem_first:c0 - elem_first + E]
            if bf16:
                _bf16_live_step(x, perm, rank, num_live, ncols,
                                [(w, nodes[nid - first:nid - first + w.size(0)], width, 1 if keep else 0)
                                 for w, nid in _bf16_weights(copies, blocks, params, E, BS, stacked.get(c0))])
                continue
            ex_t, m = _live_shifted_exp(x, perm, num_live)                                       # [live, E]
            for w, nid in _products(blocks, params, E, BS, stacked.get(c0)):
                rows = w.size(0)
                prod = torch.mm(ex_t, w.t()) if trans else torch.mm(w, ex_t.t())      # [live, rows] / [rows, live]
                num_ctiles = host_cdiv(ncols, TILE_C)
                _log_scatter_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](
                    nodes[nid - first:nid - first + rows], prod, m, rank, rows, ncols, width, num_live, num_ctiles,
                    TILE_R = TILE_R, TILE_C = TILE_C, TRANS = trans, KEEP = keep)
