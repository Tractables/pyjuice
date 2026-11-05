"""
What the dense sums of both directions share: a block shifted by its column maximum and exponentiated, its log added back and
scattered, the weights a query reads and their products.
"""

import contextlib
from typing import Optional

import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

from ..common import _logaddexp_tile


@contextlib.contextmanager
def _matmul_precision(precision: str):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


#: The dense path's elementwise passes: (rows, columns) per program.
_EW_TILE = (32, 128)


@triton_jit
def _shifted_exp_kernel(x, ex, m, rows, ncols, x_ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``ex = exp(x - m)`` (``ex`` contiguous [rows, ncols]), ``m`` the column maxima of ``x``; a column of -inf
    is shifted by 0 and stays all 0."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    shift = tl.where(shift == float("-inf"), 0.0, shift)
    r = r.to(tl.int64)
    v = tl.load(x + r[:, None] * x_ld + c[None, :], mask = mask, other = float("-inf"))
    tl.store(ex + r[:, None] * ncols + c[None, :], tl.exp(v - shift[None, :]), mask = mask)


@triton_jit
def _live_shifted_exp_kernel(x, perm, ex_t, m, rows, x_ld, TILE_R: tl.constexpr):
    """One program per live column ``k`` (``perm[k]`` of ``x``, [rows, *] with row stride ``x_ld``): its maximum
    ``m[k]`` and ``ex_t[k, :] = exp(x[:, perm[k]] - m[k])`` -- the live columns gathered, shifted and exponentiated in
    one pass, stored transposed (contiguous per column) for the product to read as ``ex_t.t()``."""
    k = tl.program_id(0)
    col = tl.load(perm + k).to(tl.int64)
    big = tl.full([TILE_R], float("-inf"), dtype = tl.float32)
    for r0 in range(0, rows, TILE_R):
        r = r0 + tl.arange(0, TILE_R)
        big = tl.maximum(big, tl.load(x + r.to(tl.int64) * x_ld + col, mask = r < rows, other = float("-inf")))
    top = tl.max(big, axis = 0)
    tl.store(m + k, top)
    shift = tl.where(top == float("-inf"), 0.0, top)
    for r0 in range(0, rows, TILE_R):
        r = r0 + tl.arange(0, TILE_R)
        v = tl.load(x + r.to(tl.int64) * x_ld + col, mask = r < rows, other = float("-inf"))
        tl.store(ex_t + k.to(tl.int64) * rows + r, tl.exp(v - shift), mask = r < rows)


@triton_jit
def _live_shifted_exp_tiles_kernel(x, perm, ex_t, m, rows, x_ld, num_live, TILE_R: tl.constexpr,
                                   TILE_C: tl.constexpr):
    """:func:`_live_shifted_exp_kernel` for ``TILE_C`` live columns a program, ``TILE_R`` rows of them read at a time
    -- along the rows, where a program per column reads one float per memory sector at the row stride. The same
    values to the bit: a maximum is exact in any order, and every exponential is the same."""
    k = tl.program_id(0) * TILE_C + tl.arange(0, TILE_C)
    km = k < num_live
    col = tl.load(perm + k, mask = km, other = 0).to(tl.int64)
    big = tl.full([TILE_R, TILE_C], float("-inf"), dtype = tl.float32)
    for r0 in range(0, rows, TILE_R):
        r = r0 + tl.arange(0, TILE_R)
        mask = (r < rows)[:, None] & km[None, :]
        big = tl.maximum(big, tl.load(x + r.to(tl.int64)[:, None] * x_ld + col[None, :], mask = mask,
                                      other = float("-inf")))
    top = tl.max(big, axis = 0)
    tl.store(m + k, top, mask = km)
    shift = tl.where(top == float("-inf"), 0.0, top)
    for r0 in range(0, rows, TILE_R):
        r = r0 + tl.arange(0, TILE_R)
        mask = (r < rows)[:, None] & km[None, :]
        v = tl.load(x + r.to(tl.int64)[:, None] * x_ld + col[None, :], mask = mask, other = float("-inf"))
        tl.store(ex_t + k.to(tl.int64)[None, :] * rows + r[:, None], tl.exp(v - shift[None, :]), mask = mask)


#: The live exp pass's column tiles (:func:`_live_shifted_exp_tiles_kernel`): ``(fewest live columns, TILE_R,
#: TILE_C)``, the last entry a pass reaches; a narrower pass takes a program per column, whose many programs win there.
#: Measured over every pass of an HMM ex108 B8 diffusion session (4096 rows, 1 to 83k live columns; toast): below 256
#: columns 4.5 ms a session per column, 7.4 in tiles of 8; 256 to 4096 columns 46 -> 11 ms; above, 740 -> 248 ms.
_LIVE_EXP_TILES = ((256, 512, 16), (4096, 512, 32))


def _live_shifted_exp(x: torch.Tensor, perm: torch.Tensor, num_live: int, dtype: Optional[torch.dtype] = None):
    """``(ex_t, m)`` of :func:`_live_shifted_exp_kernel` (or :func:`_live_shifted_exp_tiles_kernel`, the same bits):
    [num_live, rows] (``dtype``, default ``x``'s: bf16 rounds the exponentials to nearest) and [num_live] for ``x``
    [rows, *]."""
    rows = x.size(0)
    ex_t = torch.empty(num_live, rows, dtype = dtype or x.dtype, device = x.device)
    m = torch.empty(num_live, dtype = x.dtype, device = x.device)
    tile = None
    for least, TILE_R, TILE_C in _LIVE_EXP_TILES:
        if num_live >= least:
            tile = (TILE_R, TILE_C)
    if tile is not None:
        _live_shifted_exp_tiles_kernel[(host_cdiv(num_live, tile[1]),)](x, perm, ex_t, m, rows, x.stride(0), num_live,
                                                                       TILE_R = tile[0], TILE_C = tile[1],
                                                                       num_warps = 4)
    elif num_live > 0:
        _live_shifted_exp_kernel[(num_live,)](x, perm, ex_t, m, rows, x.stride(0), TILE_R = 1024, num_warps = 4)
    return ex_t, m


@triton_jit
def _log_add_kernel(out, m, rows, ncols, ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``out = log(out) + m`` in place, ``m`` the column maxima :func:`_shifted_exp_kernel` shifted by (a column of
    -inf has a product of 0, and log 0 - inf = -inf: no guard needed)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    ptr = out + r.to(tl.int64)[:, None] * ld + c[None, :]
    tl.store(ptr, tl.log(tl.load(ptr, mask = mask, other = 1.0)) + shift[None, :], mask = mask)


@triton_jit
def _log_scatter_kernel(out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R: tl.constexpr,
                        TILE_C: tl.constexpr, TRANS: tl.constexpr = False, KEEP: tl.constexpr = False):
    """``out[:, c] = log(prod[:, rank[c]]) + m[rank[c]]`` for every live column ``c`` of ``out``, -inf for the others
    (``prod`` [rows, num_live], the products of the live columns; ``TRANS``: [num_live, rows], read transposed) --
    ``KEEP``: the others left as they are."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    k = tl.load(rank + c, mask = cmask, other = -1)
    alive = k >= 0
    shift = tl.load(m + k, mask = cmask & alive, other = 0.0)
    r = r.to(tl.int64)
    if TRANS:
        at = k.to(tl.int64)[None, :] * rows + r[:, None]
    else:
        at = r[:, None] * num_live + k[None, :]
    v = tl.load(prod + at, mask = mask & alive[None, :], other = 1.0)
    if KEEP:
        mask = mask & alive[None, :]
    tl.store(out + r[:, None] * ld + c[None, :], tl.where(alive[None, :], tl.log(v) + shift[None, :], float("-inf")),
             mask = mask)


def query_weights(params: torch.Tensor, stacks: dict) -> torch.Tensor:
    """The sum layers' weights as the query's kernels take them: ``params`` themselves when stored in fp32; stored
    below fp32 (`TensorCircuit.param_dtype`), an fp32 copy kept in ``stacks`` (`~.state.State.weight_stacks`: a
    `pyjuice.fast_inference` scope's, else the query's own) -- the validity of the weight stacks and fragment-order
    copies kept there, which assume the parameters fixed for as long -- made once."""
    if params.dtype == torch.float32:
        return params
    hit = stacks.get("fp32 params")
    if hit is None or hit.shape != params.shape or hit.device != params.device:
        hit = stacks["fp32 params"] = params.detach().to(torch.float32)
    return hit


def stacked_weights(params: torch.Tensor, p0s, E: int, block_size: int) -> torch.Tensor:
    """``[K * block_size, E]``: the weights of K dense node blocks over the same ``E`` children (node block ``k``'s
    ``[E, block_size]`` matrix at ``params[p0s[k]]``), transposed and stacked in the order of ``p0s`` -- a copy, as
    the blocks' matrices are not one strided view of the parameter buffer."""
    return torch.cat([params[p0:p0 + E * block_size].view(E, block_size).t() for p0 in p0s], dim = 0)


def _products(blocks, params, E, BS, stacked):
    """``(weights [rows, E], first node row)`` of every matrix product of a run of node blocks: one with their
    ``stacked`` weights (the blocks then occupy consecutive rows), else one per block, read in place."""
    if stacked is not None:
        return [(stacked, blocks[0][1])]
    return [(params[p0:p0 + E * BS].view(E, BS).t(), nid) for _, nid, p0 in blocks]


_BF16_PRODUCTS = {}


def _bf16_products(dev: torch.device) -> bool:
    """Whether ``torch.mm`` multiplies bf16 matrices into fp32 ones on ``dev`` (its ``out_dtype``: recent PyTorch, a
    GPU with bf16 tensor cores) -- else a bf16 query's dense products run as at "tf32"."""
    key = (dev.type, dev.index)
    hit = _BF16_PRODUCTS.get(key)
    if hit is None:
        a = torch.ones(16, 16, dtype = torch.bfloat16, device = dev)
        try:
            hit = torch.mm(a, a, out_dtype = torch.float32).dtype == torch.float32
        except (TypeError, RuntimeError):
            hit = False
        _BF16_PRODUCTS[key] = hit
    return hit


def _bf16_copy(copies: dict, key, make) -> torch.Tensor:
    """``copies[key]``, made from ``make()`` rounded to bf16 (to nearest) on first use."""
    hit = copies.get(key)
    if hit is None:
        hit = copies[key] = make().to(torch.bfloat16)
    return hit


def _bf16_weights(copies: dict, blocks, params, E, BS, stacked):
    """:func:`_products` of a run from bf16 copies of its weights kept in ``copies``: of the ``stacked`` weights, or of
    every block's ``[E, BS]`` matrix (transposed for free)."""
    if stacked is not None:
        key = ("bf16 stacked", E, BS) + tuple(p0 for _, _, p0 in blocks)
        return [(_bf16_copy(copies, key, lambda: stacked), blocks[0][1])]
    return [(_bf16_copy(copies, ("bf16", p0, E, BS), lambda p0 = p0: params[p0:p0 + E * BS].view(E, BS)).t(), nid)
            for _, nid, p0 in blocks]


#: The top-down pass of a sum part under external parameters that is one dense run (``block_scale.dense_run``)
#: multiplies its compact rows sample by sample (:func:`per_sample_dense_sum_top_down`) from this many lanes a row per
#: sample (every sample's gates the same: lanes), on tensor cores ("tf32": tf32 and bf16 queries); else fused. Replayed
#: over every fused launch of a Codpula ex108 B8 diffusion session (4096 nodes in blocks of 32, a gate per pair of
#: blocks; toast): narrower than 128 lanes a sample the fused kernel won every launch, from 256 the products by sample
#: won 99% (Codpula ex108 unpruned B8 diffusion decode 7160 -> 6551 ms). The forward stays fused: by sample, its bf16
#: products left the conditionals of that session's improbable tokens 10x the fused kernels' error (1.6e-2 vs 1.6e-3
#: against fp32; the top-down pass's, 1.6e-3).
PER_SAMPLE_MIN_LANES = 256


def sample_spans(compact, iv: int, ncols: int, shared: bool) -> Optional[list]:
    """The ``(sample, first lane, lanes)`` by which a compact row of interval ``iv`` (``ncols`` lanes) multiplies
    sample by sample -- one span of sample 0 over every lane when the gates are every sample's (``shared``) --, or None
    when its rows are too narrow for it (:data:`PER_SAMPLE_MIN_LANES`)."""
    if ncols < PER_SAMPLE_MIN_LANES * (1 if shared else compact.B):
        return None
    if shared:
        return [(0, 0, ncols)]
    off = compact.sample_offsets(iv)
    return [(b, int(off[b]), int(off[b + 1] - off[b])) for b in range(compact.B) if off[b + 1] > off[b]]


@triton_jit
def _log_accumulate_kernel(out, prod, m, rows, ncols, ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``out = logaddexp(out, log(prod) + m)`` (``prod`` contiguous [rows, ncols])."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    r = r.to(tl.int64)
    v = tl.log(tl.load(prod + r[:, None] * ncols + c[None, :], mask = mask, other = 0.0)) + shift[None, :]
    ptr = out + r[:, None] * ld + c[None, :]
    tl.store(ptr, _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), v), mask = mask)


@triton_jit
def _log_scatter_accumulate_kernel(out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R: tl.constexpr,
                                   TILE_C: tl.constexpr, TRANS: tl.constexpr = False):
    """``out[:, c] = logaddexp(out[:, c], log(prod[:, rank[c]]) + m[rank[c]])`` for every live column ``c``; dead
    columns untouched (they hold -inf from the first writer). ``TRANS``: ``prod`` [num_live, rows], read transposed."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    k = tl.load(rank + c, mask = cmask, other = -1)
    alive = k >= 0
    mask = (r < rows)[:, None] & (cmask & alive)[None, :]
    shift = tl.load(m + k, mask = cmask & alive, other = 0.0)
    r = r.to(tl.int64)
    if TRANS:
        at = k.to(tl.int64)[None, :] * rows + r[:, None]
    else:
        at = r[:, None] * num_live + k[None, :]
    v = tl.log(tl.load(prod + at, mask = mask, other = 0.0)) + shift[None, :]
    ptr = out + r[:, None] * ld + c[None, :]
    tl.store(ptr, _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), v), mask = mask)
