"""
The lifted product layer. A product chains its children's blocks in scope order: a log-space product over the
boundary columns that adjacent children share. There is one table-driven implementation for every circuit
shape. :class:`~pyjuice.constraints.backends.lifted.forward.Program` folds each product into binary steps,
right to left. Each step is one of three contractions, chosen by what its two operands are, never by the
circuit's shape:

* ``input @ block`` (:func:`_input_block_kernel`): an input at position ``t``, then a block from ``t + 1``.
  Class ``c`` of the token moves column ``i`` at ``t`` to ``next_col[t, i, c]``, so this is a gather over
  the classes. The block may be the identity, which materializes the input's own block (a product with a
  single input child, or an input next to another input);
* ``block @ input`` (:func:`_block_input_kernel`): the mirror case, a gather over the predecessors of each
  column at ``t + 1`` (:func:`predecessor_tables`);
* ``block @ block`` (:func:`_block_block_kernel`): a per-sample log-space matrix product over the shared
  boundary, skipping the (output tile, shared chunk) pairs the automaton cannot join (:class:`SkipMasks`).

Blocks are either sum nodes' rows in ``node_mars`` or intermediate results in a scratch buffer. Outputs go
to the product layer's ``element_mars`` rows, or to scratch when another step reads them. A row holds one
block per sample, sample-major: sample ``b``'s block in columns ``[b * slots, (b + 1) * slots)``, slot ``i *
X + j`` for entry column ``i`` and exit column ``j``. An input child is read only as the log-probability of
its observed token (pyjuice's own input layers wrote it) or as the log-mass of a token class when the token
is missing (:mod:`.inputs`).
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ..plan import Reachability, _tile_any

#: operand kinds: a sum node's block in ``node_mars``, an intermediate block in scratch, the identity
SUM, TEMP, IDENTITY = 0, 1, 2

#: CUDA's limit on the programs along a grid's first axis, and along each of the others
MAX_GRID = (2 ** 31 - 1, 65535)


def grid_chunks(count: int, axis: int):
    """``(first, size)`` of every launch that ``count`` programs along grid ``axis`` take: one launch, unless the
    count passes CUDA's limit there (each kernel adds ``first`` to its program id on that axis)."""
    limit = MAX_GRID[axis]
    return [(first, min(limit, count - first)) for first in range(0, count, limit)]

#: fields of a step, per contraction (int32 rows of the step tables the Program builds)
INPUT_BLOCK_FIELDS = ("out", "u", "t", "r_row", "r_reg", "t2")             # input u at t, then block over [t+1, t2)
BLOCK_INPUT_FIELDS = ("out", "l_row", "l_reg", "t0", "u", "t")             # block over [t0, t), then input u at t
BLOCK_BLOCK_FIELDS = ("out", "l_row", "l_reg", "r_row", "r_reg", "t0", "t1", "t2")
COPY_FIELDS = ("out", "l_row", "l_reg", "t0", "t2")


@triton.jit
def _block_ptr(row, reg, node_mars, temp, reg_offset, reg_width, reg_first, temp_width, KIND: tl.constexpr):
    """Start of a block operand: a sum node's row in its region of ``node_mars``, or a scratch row."""
    if KIND == 0:
        base = tl.load(reg_offset + reg) + (row - tl.load(reg_first + reg)) * tl.load(reg_width + reg)
        return node_mars + base
    else:
        return temp + row.to(tl.int64) * temp_width


@triton.jit
def _out_ptr(row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP: tl.constexpr):
    if OUT_TEMP:
        return temp + row.to(tl.int64) * temp_width
    else:
        return element_mars + (row - out_first).to(tl.int64) * out_width


@triton.jit
def _exit_width(width, t2, n):
    """Columns at the exit boundary ``t2``: a sequence end is one column (summed over the accepting states)."""
    return tl.where(t2 == n, 1, tl.load(width + tl.minimum(t2, n)))


@triton.jit
def _lse_step(m, acc, v):
    """One step of an online log-sum-exp (``m``: running max, ``acc``: running sum scaled by ``exp(-m)``)."""
    m_new = tl.maximum(m, v)
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc * scale + tl.exp(v - shift)


@triton.jit
def _lse_value(m, acc):
    return tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)


@triton.jit
def _input_block_kernel(element_mars, node_mars, temp, input_mars, class_mars, obs_class, next_col, width, steps,
                        reg_offset, reg_width, reg_first, out_first, out_width, temp_width,
                        B, n, W, C, input_start, pid0,
                        R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, TILE: tl.constexpr):
    """One program per step and tile of its whole output, flattened (sample, entry column, exit column): with the
    samples' blocks contiguous, a step's output is one run of columns, and the lanes stay full whatever the
    widths."""
    s = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 6 + 0)
    u = tl.load(steps + s * 6 + 1).to(tl.int64)
    t = tl.load(steps + s * 6 + 2)
    t2 = tl.load(steps + s * 6 + 5)
    E = tl.load(width + t)
    X = _exit_width(width, t2, n)
    EX = E * X
    if tile * TILE < B * EX:
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < B * EX
        b = (idx // EX).to(tl.int64)
        pos = idx % EX
        i = pos // X
        j = pos % X
        nc = (t * W + i) * C
        end = t + 1 == n
        if R_KIND != 2:
            M = tl.load(width + t + 1)                                       # the block's entry columns
            rp = _block_ptr(tl.load(steps + s * 6 + 3), tl.load(steps + s * 6 + 4), node_mars, temp, reg_offset,
                            reg_width, reg_first, temp_width, R_KIND) + b * M * X
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)            # observed token's class, -1 if missing

        # an observed token: its own log-probability, then the block from the column its class leads to
        obs = m & (oc >= 0)
        q = tl.load(next_col + nc + oc, mask = obs, other = -1)
        if R_KIND == 2:
            rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
        else:
            rv = tl.load(rp + q * X + j, mask = obs & (q >= 0), other = float("-inf"))
        val = tl.load(input_mars + u * B + b, mask = obs, other = float("-inf")) + rv

        # a missing token: log-sum over the classes of class mass + block (skipped when every lane is observed)
        miss = m & (oc < 0)
        if tl.max(tl.where(miss, 1, 0), axis = 0) > 0:
            mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
            acc = tl.zeros([TILE], dtype = tl.float32)
            for c in range(C):
                q = tl.load(next_col + nc + c, mask = miss, other = -1)
                if R_KIND == 2:
                    rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
                else:
                    rv = tl.load(rp + q * X + j, mask = miss & (q >= 0), other = float("-inf"))
                mx, acc = _lse_step(mx, acc, tl.load(class_mars + (u - input_start) * C + c) + rv)
            val = tl.where(oc >= 0, val, _lse_value(mx, acc))
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + idx, val, mask = m)


@triton.jit
def _block_input_kernel(element_mars, node_mars, temp, input_mars, class_mars, obs_class, width,
                        pred_ptr, pred_q, pred_c, steps, reg_offset, reg_width, reg_first, out_first, out_width,
                        temp_width, B, n, W, C, input_start, X_MAX, pid0,
                        L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, TILE: tl.constexpr):
    """One program per (step, sample, exit column) and tile of entry columns: a log-sum over the exit column's
    predecessors (the same list for every lane)."""
    pid = pid0 + tl.program_id(0)
    s = pid // (B * X_MAX)
    r = pid % (B * X_MAX)
    b = (r // X_MAX).to(tl.int64)
    j = r % X_MAX                                                            # exit column (at boundary t + 1)
    tile = tl.program_id(1)
    out_row = tl.load(steps + s * 6 + 0)
    t0 = tl.load(steps + s * 6 + 3)
    u = tl.load(steps + s * 6 + 4).to(tl.int64)
    t = tl.load(steps + s * 6 + 5)
    E = tl.load(width + t0)
    M = tl.load(width + t)
    X = _exit_width(width, t + 1, n)
    if (j < X) & (tile * TILE < E):
        i = tile * TILE + tl.arange(0, TILE)
        m = i < E
        oc = tl.load(obs_class + b * n + t)
        own = tl.load(input_mars + u * B + b)
        lp = _block_ptr(tl.load(steps + s * 6 + 1), tl.load(steps + s * 6 + 2), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, L_KIND) + b * E * M
        p0 = tl.load(pred_ptr + t * (W + 1) + j)
        p1 = tl.load(pred_ptr + t * (W + 1) + j + 1)
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for p in range(p0, p1):                                              # (column at t, class) -> j
            q = tl.load(pred_q + p)
            c = tl.load(pred_c + p)
            mass = tl.where(oc >= 0, tl.where(oc == c, own, float("-inf")),
                            tl.load(class_mars + (u - input_start) * C + c))
            lv = tl.load(lp + i * M + q, mask = m, other = float("-inf"))
            mx, acc = _lse_step(mx, acc, lv + mass)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + b * E * X
        tl.store(op + i * X + j, _lse_value(mx, acc), mask = m)


@triton.jit
def _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I: tl.constexpr, SUB_J: tl.constexpr):
    """Whether shared chunk ``c`` can contribute to output tile ``(ti, tj)``: the OR of its ``SUB_I x SUB_J``
    mask tiles' bits (``mb``: the step's triple's masks)."""
    need = 0
    for si in tl.static_range(SUB_I):
        for sj in tl.static_range(SUB_J):
            bi = ti * SUB_I + si
            bj = tj * SUB_J + sj
            w = tl.load(mb + (bi * TILES_J + bj) * WORDS + c // 32, mask = (bi < TILES_I) & (bj < TILES_J), other = 0)
            need = need | ((w >> (c % 32)) & 1)
    return need != 0


@triton.jit
def _block_block_tile(lp, rp, mb, i, j, ti, tj, E, M, X, TILES_I, TILES_J, WORDS,
                      TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                      SKIP: tl.constexpr, PRECISION: tl.constexpr):
    """``log(sum_q exp(L[i, q] + R[q, j]))`` for one output tile: every row's and column's maximum over the
    shared chunks the tile needs, then the product of the shifted exponentials over those chunks."""
    im, jm = i < E, j < X
    runl = tl.full([TM, TK], float("-inf"), dtype = tl.float32)
    runr = tl.full([TK, TN], float("-inf"), dtype = tl.float32)
    for c in range(0, tl.cdiv(M, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if need:
            q = c * TK + tl.arange(0, TK)
            qm = q < M
            # an elementwise running maximum, reduced once after the loop (a reduction inside the loop crashes
            # Triton 3.7's TritonGPUOptimizeThreadLocality pass)
            runl = tl.maximum(runl, tl.load(lp + (i[:, None] * M + q[None, :]), mask = im[:, None] & qm[None, :],
                                            other = float("-inf")))
            runr = tl.maximum(runr, tl.load(rp + (q[:, None] * X + j[None, :]), mask = qm[:, None] & jm[None, :],
                                            other = float("-inf")))
    sL = tl.max(runl, axis = 1)
    sR = tl.max(runr, axis = 0)
    sL = tl.where(sL == float("-inf"), 0.0, sL)
    sR = tl.where(sR == float("-inf"), 0.0, sR)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for c in range(0, tl.cdiv(M, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if need:
            q = c * TK + tl.arange(0, TK)
            qm = q < M
            lt = tl.load(lp + (i[:, None] * M + q[None, :]), mask = im[:, None] & qm[None, :], other = float("-inf"))
            rt = tl.load(rp + (q[:, None] * X + j[None, :]), mask = qm[:, None] & jm[None, :], other = float("-inf"))
            acc += tl.dot(tl.exp(lt - sL[:, None]), tl.exp(rt - sR[None, :]), input_precision = PRECISION)
    return tl.log(acc) + sL[:, None] + sR[None, :]


@triton.jit
def _block_block_kernel(element_mars, node_mars, temp, width, steps, triples, skip_bits, reg_offset, reg_width,
                        reg_first, out_first, out_width, temp_width, B, n, TILES_I, TILES_J, WORDS, pid0,
                        L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr,
                        SUB_J: tl.constexpr, SKIP: tl.constexpr, PRECISION: tl.constexpr):
    """One program per (step, sample), every output tile in turn: the sample's two blocks are contiguous and stay
    in L1 across the tiles."""
    pid = pid0 + tl.program_id(0)
    s = pid // B
    b = (pid % B).to(tl.int64)
    out_row = tl.load(steps + s * 8 + 0)
    t0 = tl.load(steps + s * 8 + 5)
    t1 = tl.load(steps + s * 8 + 6)
    t2 = tl.load(steps + s * 8 + 7)
    E = tl.load(width + t0)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    lp = _block_ptr(tl.load(steps + s * 8 + 1), tl.load(steps + s * 8 + 2), node_mars, temp, reg_offset,
                    reg_width, reg_first, temp_width, L_KIND) + b * E * M
    rp = _block_ptr(tl.load(steps + s * 8 + 3), tl.load(steps + s * 8 + 4), node_mars, temp, reg_offset,
                    reg_width, reg_first, temp_width, R_KIND) + b * M * X
    op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + b * E * X
    mb = skip_bits + tl.load(triples + s).to(tl.int64) * TILES_I * TILES_J * WORDS
    NJ = tl.cdiv(X, TN)
    for t in range(0, tl.cdiv(E, TM) * NJ):
        ti = t // NJ
        tj = t % NJ
        i = ti * TM + tl.arange(0, TM)
        j = tj * TN + tl.arange(0, TN)
        out = _block_block_tile(lp, rp, mb, i, j, ti, tj, E, M, X, TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I,
                                SUB_J, SKIP, PRECISION)
        tl.store(op + (i[:, None] * X + j[None, :]), out, mask = (i < E)[:, None] & (j < X)[None, :])


@triton.jit
def _copy_kernel(element_mars, node_mars, temp, width, steps, reg_offset, reg_width, reg_first, out_first,
                 out_width, temp_width, B, n, pid0, L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                 TILE: tl.constexpr):
    s = tl.program_id(0)
    pid = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 5 + 0)
    t0 = tl.load(steps + s * 5 + 3)
    t2 = tl.load(steps + s * 5 + 4)
    ncols = tl.load(width + t0) * _exit_width(width, t2, n) * B
    if pid * TILE < ncols:
        cols = pid * TILE + tl.arange(0, TILE)
        m = cols < ncols
        lp = _block_ptr(tl.load(steps + s * 5 + 1), tl.load(steps + s * 5 + 2), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, L_KIND)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + cols, tl.load(lp + cols, mask = m), mask = m)


def predecessor_tables(next_col: torch.Tensor, width, n: int):
    """
    For every boundary ``t`` and column ``j`` at ``t + 1`` (one column if ``t + 1 == n``), the (column at ``t``,
    token class) pairs that lead to ``j``, as a CSR: ``pred_ptr [n, W + 1]`` (global offsets; empty past the
    last column) into ``pred_q`` and ``pred_c``.
    """
    W, C = next_col.size(1), next_col.size(2)
    ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    qs, cs = [], []
    total = 0
    nc = next_col.cpu().long()
    for t in range(n):
        wi = int(width[t])
        succ = nc[t, :wi]                                                    # [wi, C]
        q, c = torch.nonzero(succ >= 0, as_tuple = True)
        j = torch.zeros_like(q) if t + 1 == n else succ[q, c]
        order = torch.argsort(j * (wi * C) + q * C + c)                      # by successor, then column, class
        q, c, j = q[order], c[order], j[order]
        counts = torch.bincount(j, minlength = W)[:W]
        ptr[t, 1:] = total + torch.cumsum(counts, 0)
        ptr[t, 0] = total
        qs.append(q); cs.append(c)
        total += q.numel()
    dev = next_col.device
    cat = lambda xs: torch.cat(xs).to(dev, torch.int32) if total > 0 else torch.zeros(1, dtype = torch.int32, device = dev)
    return ptr.to(dev, torch.int32).contiguous(), cat(qs), cat(cs)


@dataclass(frozen = True)
class SkipMasks:
    """
    Which chunks of the shared boundary every output tile of a ``block @ block`` step needs. For triple ``s``
    (boundaries ``(t0, t1, t2)``), word ``(s * tiles_i + ti) * tiles_j * words + tj * words + c // 32`` of
    ``bits`` has bit ``c % 32`` set when some column of entry tile ``ti`` (at ``t0``) reaches some column of chunk
    ``c`` (at ``t1``) that reaches some column of exit tile ``tj`` (at ``t2``). Every other (tile, chunk) pair is
    structurally ``-inf`` on one side and adds exactly zero. Tiles are ``tm x tn`` columns, chunks ``tk``.

    :ivar bits: int32 ``[triples * tiles_i * tiles_j * words]`` (one word if there are no triples)
    """

    bits: torch.Tensor
    tiles_i: int
    tiles_j: int
    words: int
    tm: int
    tn: int
    tk: int


def skip_masks(reach: Reachability, triples: Sequence[Tuple[int, int, int]], tm: int = 16, tn: int = 16,
               tk: int = 16, device = None) -> SkipMasks:
    """
    The :class:`SkipMasks` of ``triples`` (``(t0, t1, t2)`` boundaries, in the order their index refers to), from
    the occupancy ``reach`` keeps; ``tm``, ``tn`` and ``tk`` are multiples of its tile.
    """
    T = reach.tile
    if tm % T or tn % T or tk % T:
        raise ValueError(f"Tiles ({tm}, {tn}, {tk}) must be multiples of the occupancy tile {T}.")
    if len(triples) == 0:
        return SkipMasks(torch.zeros(1, dtype = torch.int32, device = device), 0, 0, 1, tm, tn, tk)
    lefts = sorted({(a, b) for a, b, _ in triples})
    rights = sorted({(b, c) for _, b, c in triples})
    L = [_tile_any(reach.occupancy(a, b), tm // T, tk // T) for a, b in lefts]     # [entry tiles, chunks]
    R = [_tile_any(reach.occupancy(b, c), tk // T, tn // T) for b, c in rights]    # [chunks, exit tiles]
    I = max(m.size(0) for m in L)
    J = max(m.size(1) for m in R)
    words = max(1, math.ceil(max(m.size(1) for m in L) / 32))
    NC = 32 * words

    def stack(ms, shape):
        out = torch.zeros(len(ms), *shape, dtype = torch.bool)
        for k, m in enumerate(ms):
            out[k, :m.size(0), :m.size(1)] = m
        return out

    Ls, Rs = stack(L, (I, NC)), stack(R, (NC, J))
    lpos = {iv: k for k, iv in enumerate(lefts)}
    rpos = {iv: k for k, iv in enumerate(rights)}
    li = torch.tensor([lpos[a, b] for a, b, _ in triples], dtype = torch.long)
    ri = torch.tensor([rpos[b, c] for _, b, c in triples], dtype = torch.long)
    shifts = torch.arange(32, dtype = torch.int64)
    chunk = max(1, (1 << 26) // (I * J * NC))
    parts = []
    for g0 in range(0, len(triples), chunk):
        need = Ls[li[g0:g0 + chunk], :, None, :] & Rs[ri[g0:g0 + chunk]].transpose(1, 2)[:, None, :, :]
        w = (need.view(need.size(0), I, J, words, 32).to(torch.int64) << shifts).sum(dim = -1)
        parts.append(torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32).flatten())
    return SkipMasks(torch.cat(parts).to(device), I, J, words, tm, tn, tk)


#: :func:`_block_block_kernel`'s output tile (entry x exit columns), shared chunk and warps; the tile is a multiple
#: of :class:`SkipMasks`'s, and the chunk equal to it
BLOCK_BLOCK_TILES = dict(TM = 16, TN = 32, TK = 16, warps = 4)
#: the products' dot precision: "ieee" (exact fp32, CUDA cores) or "tf32x3" (three TF32 tensor-core passes)
PRODUCT_PRECISION = "ieee"
#: skip the (output tile, shared chunk) pairs the automaton cannot join; False computes every pair
SKIP_EMPTY_TILES = True


def _tile(cols: int) -> int:
    return min(256, max(16, triton.next_power_of_2(cols)))


def run_products(stages, bufs, prog, regions, element_mars, node_mars, input_mars, class_mars, obs_class,
                 elem_first: int, elem_width: int, B: int):
    """
    Every step of one product layer, stage by stage (a stage reads only earlier stages' scratch rows). Products
    are fp32-level whatever the query's precision, which applies to the sum layers: ``block @ block`` dots in
    :data:`PRODUCT_PRECISION` (exact fp32 by default), the other contractions exact fp32 log-sum-exps.

    :param stages: the layer's stages from :class:`~pyjuice.constraints.backends.lifted.forward.Program`: lists
        of ``(form, kinds, steps, E_max, X_max)`` launches (``block @ block`` steps: ``(steps, triple index)``,
        the latter indexing ``Program.skip``)
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows
    """
    temp, temp_width = bufs
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  out_first = elem_first, out_width = elem_width, temp_width = temp_width)
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    for launches in stages:
        for form, kinds, steps, E_max, X_max in launches:
            S = steps[0].size(0) if isinstance(steps, tuple) else steps.size(0)
            if form == "input_block":
                r_kind, out_temp = kinds
                tile = min(512, max(128, triton.next_power_of_2(B * E_max * X_max) // 8))
                for first, size in grid_chunks(triton.cdiv(B * E_max * X_max, tile), 1):
                    _input_block_kernel[(S, size)](
                        element_mars, node_mars, temp, input_mars, class_mars, obs_class, prog.next_col,
                        prog.width_t, steps, B = B, n = n, W = W, C = C, input_start = prog.input_start, pid0 = first,
                        R_KIND = r_kind, OUT_TEMP = out_temp, TILE = tile, num_warps = 4, **common)
            elif form == "block_input":
                l_kind, out_temp = kinds
                tile = _tile(E_max)
                for first, size in grid_chunks(S * B * X_max, 0):
                    _block_input_kernel[(size, triton.cdiv(E_max, tile))](
                        element_mars, node_mars, temp, input_mars, class_mars, obs_class, prog.width_t,
                        prog.pred_ptr, prog.pred_q, prog.pred_c, steps, B = B, n = n, W = W, C = C,
                        input_start = prog.input_start, X_MAX = X_max, pid0 = first, L_KIND = l_kind,
                        OUT_TEMP = out_temp, TILE = tile, num_warps = 4, **common)
            elif form == "block_block":
                l_kind, r_kind, out_temp = kinds
                steps, triples = steps
                cfg, skip = BLOCK_BLOCK_TILES, prog.skip
                assert cfg["TK"] == skip.tk and cfg["TM"] % skip.tm == 0 and cfg["TN"] % skip.tn == 0
                for first, size in grid_chunks(S * B, 0):
                    _block_block_kernel[(size,)](
                        element_mars, node_mars, temp, prog.width_t, steps, triples, skip.bits, B = B, n = n,
                        TILES_I = skip.tiles_i, TILES_J = skip.tiles_j, WORDS = skip.words, pid0 = first,
                        L_KIND = l_kind, R_KIND = r_kind, OUT_TEMP = out_temp, TM = cfg["TM"], TN = cfg["TN"],
                        TK = cfg["TK"], SUB_I = cfg["TM"] // skip.tm, SUB_J = cfg["TN"] // skip.tn,
                        SKIP = SKIP_EMPTY_TILES, PRECISION = PRODUCT_PRECISION, num_warps = cfg["warps"],
                        num_stages = 1, **common)
            else:                                                            # "copy"
                l_kind, out_temp = kinds
                tile = _tile(E_max * X_max * B)
                for first, size in grid_chunks(triton.cdiv(E_max * X_max * B, tile), 1):
                    _copy_kernel[(S, size)](
                        element_mars, node_mars, temp, prog.width_t, steps, B = B, n = n, pid0 = first,
                        L_KIND = l_kind, OUT_TEMP = out_temp, TILE = tile, num_warps = 4, **common)
