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
  boundary, on tensor cores.

Blocks are either sum nodes' rows in ``node_mars`` or intermediate results in a scratch buffer. Outputs go
to the product layer's ``element_mars`` rows, or to scratch when another step reads them. An input child is
read only as the log-probability of its observed token (pyjuice's own input layers wrote it) or as the
log-mass of a token class when the token is missing (:mod:`.inputs`).
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

#: fields of a step, per contraction (int32 rows of the step tables the Program builds)
INPUT_BLOCK_FIELDS = ("out", "u", "t", "r_row", "r_reg", "t2")             # input u at t, then block over [t+1, t2)
BLOCK_INPUT_FIELDS = ("out", "l_row", "l_reg", "t0", "u", "t")             # block over [t0, t), then input u at t
BLOCK_BLOCK_FIELDS = ("out", "l_row", "l_reg", "r_row", "r_reg", "t0", "t1", "t2", "l_op", "r_op")
OPERAND_FIELDS = ("row", "reg", "t_first", "t_last")                      # a block over [t_first, t_last)
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
                        B, n, W, C, input_start,
                        R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, TILE: tl.constexpr):
    s = tl.program_id(0)
    j = tl.program_id(1)                                                     # exit column
    pid = tl.program_id(2)                                                   # tile of (entry column, sample)
    out_row = tl.load(steps + s * 6 + 0)
    u = tl.load(steps + s * 6 + 1).to(tl.int64)
    t = tl.load(steps + s * 6 + 2)
    t2 = tl.load(steps + s * 6 + 5)
    E = tl.load(width + t)
    X = _exit_width(width, t2, n)
    if (j < X) & (pid * TILE < E * B):
        idx = pid * TILE + tl.arange(0, TILE)
        m = idx < E * B
        i = idx // B
        b = idx % B
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)            # observed token's class, -1 if missing
        nc = (t * W + i) * C
        end = t + 1 == n
        if R_KIND != 2:
            rp = _block_ptr(tl.load(steps + s * 6 + 3), tl.load(steps + s * 6 + 4), node_mars, temp, reg_offset,
                            reg_width, reg_first, temp_width, R_KIND)

        # an observed token: its own log-probability, then the block from the column its class leads to
        obs = m & (oc >= 0)
        q = tl.load(next_col + nc + oc, mask = obs, other = -1)
        if R_KIND == 2:
            rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
        else:
            rv = tl.load(rp + (q * X + j).to(tl.int64) * B + b, mask = obs & (q >= 0), other = float("-inf"))
        v_obs = tl.load(input_mars + u * B + b, mask = obs, other = float("-inf")) + rv

        # a missing token: log-sum over the classes of class mass + block
        miss = m & (oc < 0)
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for c in range(C):
            q = tl.load(next_col + nc + c, mask = miss, other = -1)
            if R_KIND == 2:
                rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
            else:
                rv = tl.load(rp + (q * X + j).to(tl.int64) * B + b, mask = miss & (q >= 0), other = float("-inf"))
            mx, acc = _lse_step(mx, acc, tl.load(class_mars + (u - input_start) * C + c) + rv)
        v_miss = _lse_value(mx, acc)

        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + (i * X + j).to(tl.int64) * B + b, tl.where(oc >= 0, v_obs, v_miss), mask = m)


@triton.jit
def _block_input_kernel(element_mars, node_mars, temp, input_mars, class_mars, obs_class, width,
                        pred_ptr, pred_q, pred_c, steps, reg_offset, reg_width, reg_first, out_first, out_width,
                        temp_width, B, n, W, C, input_start,
                        L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, TILE: tl.constexpr):
    s = tl.program_id(0)
    j = tl.program_id(1)                                                     # exit column (at boundary t + 1)
    pid = tl.program_id(2)
    out_row = tl.load(steps + s * 6 + 0)
    t0 = tl.load(steps + s * 6 + 3)
    u = tl.load(steps + s * 6 + 4).to(tl.int64)
    t = tl.load(steps + s * 6 + 5)
    E = tl.load(width + t0)
    M = tl.load(width + t)
    X = _exit_width(width, t + 1, n)
    if (j < X) & (pid * TILE < E * B):
        idx = pid * TILE + tl.arange(0, TILE)
        m = idx < E * B
        i = idx // B
        b = idx % B
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)
        lp = _block_ptr(tl.load(steps + s * 6 + 1), tl.load(steps + s * 6 + 2), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, L_KIND)
        own = tl.load(input_mars + u * B + b, mask = m & (oc >= 0), other = float("-inf"))
        p0 = tl.load(pred_ptr + t * (W + 1) + j)
        p1 = tl.load(pred_ptr + t * (W + 1) + j + 1)
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for p in range(p0, p1):                                              # (column at t, class) -> j
            q = tl.load(pred_q + p)
            c = tl.load(pred_c + p)
            mass = tl.where(oc >= 0, tl.where(oc == c, own, float("-inf")),
                            tl.load(class_mars + (u - input_start) * C + c))
            lv = tl.load(lp + (i * M + q).to(tl.int64) * B + b, mask = m, other = float("-inf"))
            mx, acc = _lse_step(mx, acc, lv + mass)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + (i * X + j).to(tl.int64) * B + b, _lse_value(mx, acc), mask = m)


@triton.jit
def _row_max_kernel(node_mars, temp, ops, width, reg_offset, reg_width, reg_first, temp_width, mx, B, n, EMAX,
                    KIND: tl.constexpr, TILE: tl.constexpr, TQ: tl.constexpr):
    """``mx[o, i, b] = max_q block_o(i, q, b)``: every left operand's row maxima over the boundary it shares."""
    o = tl.program_id(0)
    pid = tl.program_id(1)
    t0 = tl.load(ops + o * 4 + 2)
    t1 = tl.load(ops + o * 4 + 3)
    E = tl.load(width + t0)
    M = tl.load(width + t1)
    if pid * TILE < E * B:
        idx = pid * TILE + tl.arange(0, TILE)
        m = idx < E * B
        i = idx // B
        b = idx % B
        bp = _block_ptr(tl.load(ops + o * 4), tl.load(ops + o * 4 + 1), node_mars, temp, reg_offset, reg_width,
                        reg_first, temp_width, KIND)
        # an elementwise running max over tiles of the shared boundary, reduced once after the loop (a reduction
        # inside the loop crashes Triton 3.7's TritonGPUOptimizeThreadLocality pass)
        run = tl.full([TILE, TQ], float("-inf"), dtype = tl.float32)
        for q0 in range(0, M, TQ):
            q = q0 + tl.arange(0, TQ)
            run = tl.maximum(run, tl.load(bp + ((i[:, None] * M + q[None, :]).to(tl.int64) * B + b[:, None]),
                                          mask = m[:, None] & (q < M)[None, :], other = float("-inf")))
        acc = tl.max(run, axis = 1)
        tl.store(mx + (o.to(tl.int64) * EMAX + i) * B + b, acc, mask = m)


@triton.jit
def _col_max_kernel(node_mars, temp, ops, width, reg_offset, reg_width, reg_first, temp_width, mx, B, n, XMAX,
                    KIND: tl.constexpr, TILE: tl.constexpr, TQ: tl.constexpr):
    """``mx[o, j, b] = max_q block_o(q, j, b)``: every right operand's column maxima over the boundary it shares."""
    o = tl.program_id(0)
    pid = tl.program_id(1)
    t1 = tl.load(ops + o * 4 + 2)
    t2 = tl.load(ops + o * 4 + 3)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    if pid * TILE < X * B:
        idx = pid * TILE + tl.arange(0, TILE)
        m = idx < X * B
        j = idx // B
        b = idx % B
        bp = _block_ptr(tl.load(ops + o * 4), tl.load(ops + o * 4 + 1), node_mars, temp, reg_offset, reg_width,
                        reg_first, temp_width, KIND)
        # an elementwise running max over tiles of the shared boundary, reduced once after the loop (a reduction
        # inside the loop crashes Triton 3.7's TritonGPUOptimizeThreadLocality pass)
        run = tl.full([TILE, TQ], float("-inf"), dtype = tl.float32)
        for q0 in range(0, M, TQ):
            q = q0 + tl.arange(0, TQ)
            run = tl.maximum(run, tl.load(bp + ((q[None, :] * X + j[:, None]).to(tl.int64) * B + b[:, None]),
                                          mask = m[:, None] & (q < M)[None, :], other = float("-inf")))
        acc = tl.max(run, axis = 1)
        tl.store(mx + (o.to(tl.int64) * XMAX + j) * B + b, acc, mask = m)


@triton.jit
def _block_block_kernel(element_mars, node_mars, temp, width, steps, reg_offset, reg_width, reg_first,
                        out_first, out_width, temp_width, mL, mR, EMAX, XMAX, B, n, NB, NI, NJ,
                        L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        TB: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                        PRECISION: tl.constexpr):
    # one flat grid, a step's tiles consecutive (they read the same blocks, which then stay in L2)
    pid = tl.program_id(0)
    per_step = NB * NI * NJ
    s = pid // per_step
    r = pid % per_step
    pid_b = r // (NI * NJ)
    ti = (r // NJ) % NI
    tj = r % NJ
    out_row = tl.load(steps + s * 10 + 0)
    t0 = tl.load(steps + s * 10 + 5)
    t1 = tl.load(steps + s * 10 + 6)
    t2 = tl.load(steps + s * 10 + 7)
    E = tl.load(width + t0)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    if (ti * TM < E) & (tj * TN < X):
        lp = _block_ptr(tl.load(steps + s * 10 + 1), tl.load(steps + s * 10 + 2), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, L_KIND)
        rp = _block_ptr(tl.load(steps + s * 10 + 3), tl.load(steps + s * 10 + 4), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, R_KIND)
        l_op = tl.load(steps + s * 10 + 8).to(tl.int64)
        r_op = tl.load(steps + s * 10 + 9).to(tl.int64)
        b = pid_b * TB + tl.arange(0, TB)
        i = ti * TM + tl.arange(0, TM)
        j = tj * TN + tl.arange(0, TN)
        bm, im, jm = b < B, i < E, j < X
        # each left row's and right column's maximum over the shared boundary, from the pre-pass
        sL = tl.load(mL + (l_op * EMAX + i[None, :]) * B + b[:, None], mask = bm[:, None] & im[None, :],
                     other = float("-inf"))
        sR = tl.load(mR + (r_op * XMAX + j[None, :]) * B + b[:, None], mask = bm[:, None] & jm[None, :],
                     other = float("-inf"))
        sL = tl.where(sL == float("-inf"), 0.0, sL)
        sR = tl.where(sR == float("-inf"), 0.0, sR)
        acc = tl.zeros([TB, TM, TN], dtype = tl.float32)
        for q0 in range(0, M, TK):
            q = q0 + tl.arange(0, TK)
            qm = q < M
            lt = tl.load(lp + ((i[None, :, None] * M + q[None, None, :]).to(tl.int64) * B + b[:, None, None]),
                         mask = bm[:, None, None] & im[None, :, None] & qm[None, None, :], other = float("-inf"))
            rt = tl.load(rp + ((q[None, :, None] * X + j[None, None, :]).to(tl.int64) * B + b[:, None, None]),
                         mask = bm[:, None, None] & qm[None, :, None] & jm[None, None, :], other = float("-inf"))
            acc += tl.dot(tl.exp(lt - sL[:, :, None]), tl.exp(rt - sR[:, None, :]), input_precision = PRECISION)
        out = tl.log(acc) + sL[:, :, None] + sR[:, None, :]
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + ((i[None, :, None] * X + j[None, None, :]).to(tl.int64) * B + b[:, None, None]), out,
                 mask = bm[:, None, None] & im[None, :, None] & jm[None, None, :])


@triton.jit
def _copy_kernel(element_mars, node_mars, temp, width, steps, reg_offset, reg_width, reg_first, out_first,
                 out_width, temp_width, B, n, L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, TILE: tl.constexpr):
    s = tl.program_id(0)
    pid = tl.program_id(1)
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


#: tiles of :func:`_block_block_kernel`: samples, entry and exit columns, shared columns per step; launch knobs
BLOCK_BLOCK_CONFIG = dict(TB = 4, TM = 32, TN = 32, TK = 16, warps = 4, stages = 2)


def _fit(width: int, largest: int) -> int:
    """A dot tile for ``width`` columns: the power of two in [16, largest] that pads them least."""
    best = None
    for t in (16, 32, 64, 128):
        if t > largest:
            break
        pad = triton.cdiv(width, t) * t
        if best is None or pad < best[0]:
            best = (pad, t)
    return best[1]


def _tile(cols: int) -> int:
    return min(256, max(16, triton.next_power_of_2(cols)))


def run_products(stages, bufs, prog, regions, element_mars, node_mars, input_mars, class_mars, obs_class,
                 elem_first: int, elem_width: int, B: int):
    """
    Every step of one product layer, stage by stage (a stage reads only earlier stages' scratch rows). Products
    are fp32-level whatever the query's precision, which applies to the sum layers: ``block @ block`` takes
    three TF32 passes (``tf32x3``), and the other contractions are exact fp32 log-sum-exps.

    :param stages: the layer's stages from :class:`~pyjuice.constraints.backends.lifted.forward.Program`: lists
        of ``(form, kinds, steps, E_max, X_max)`` launches (``block @ block`` steps: ``(steps, left operands, right
        operands, triple index)``, the last indexing ``Program.skip``)
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
                tile = _tile(E_max * B)
                _input_block_kernel[(S, X_max, triton.cdiv(E_max * B, tile))](
                    element_mars, node_mars, temp, input_mars, class_mars, obs_class, prog.next_col, prog.width_t,
                    steps, B = B, n = n, W = W, C = C, input_start = prog.input_start, R_KIND = r_kind,
                    OUT_TEMP = out_temp, TILE = tile, num_warps = 4, **common)
            elif form == "block_input":
                l_kind, out_temp = kinds
                tile = _tile(E_max * B)
                _block_input_kernel[(S, X_max, triton.cdiv(E_max * B, tile))](
                    element_mars, node_mars, temp, input_mars, class_mars, obs_class, prog.width_t,
                    prog.pred_ptr, prog.pred_q, prog.pred_c, steps, B = B, n = n, W = W, C = C,
                    input_start = prog.input_start, L_KIND = l_kind, OUT_TEMP = out_temp, TILE = tile, num_warps = 4,
                    **common)
            elif form == "block_block":
                l_kind, r_kind, out_temp = kinds
                steps, left_ops, right_ops = steps[:3]
                S = steps.size(0)
                maxima = []
                for ops, kernel, kind, wmax in ((left_ops, _row_max_kernel, l_kind, E_max),
                                                (right_ops, _col_max_kernel, r_kind, X_max)):
                    mx = torch.empty(ops.size(0) * wmax * B, device = node_mars.device)
                    tile = min(128, _tile(wmax * B))
                    kernel[(ops.size(0), triton.cdiv(wmax * B, tile))](
                        node_mars, temp, ops, prog.width_t, regions["offset"], regions["width"], regions["first"],
                        temp_width, mx, B, n, wmax, KIND = kind, TILE = tile, TQ = 16, num_warps = 4)
                    maxima.append(mx)
                cfg = BLOCK_BLOCK_CONFIG
                TB = min(cfg["TB"], triton.next_power_of_2(B))
                TM, TN = _fit(E_max, cfg["TM"]), _fit(X_max, cfg["TN"])
                NB, NI, NJ = triton.cdiv(B, TB), triton.cdiv(E_max, TM), triton.cdiv(X_max, TN)
                _block_block_kernel[(S * NB * NI * NJ,)](
                    element_mars, node_mars, temp, prog.width_t, steps, mL = maxima[0], mR = maxima[1], EMAX = E_max,
                    XMAX = X_max, B = B, n = n, NB = NB, NI = NI, NJ = NJ, L_KIND = l_kind, R_KIND = r_kind,
                    OUT_TEMP = out_temp, TB = TB, TM = TM, TN = TN, TK = cfg["TK"], PRECISION = "tf32x3",
                    num_warps = cfg["warps"], num_stages = cfg["stages"], **common)
            else:                                                            # "copy"
                l_kind, out_temp = kinds
                tile = _tile(E_max * X_max * B)
                _copy_kernel[(S, triton.cdiv(E_max * X_max * B, tile))](
                    element_mars, node_mars, temp, prog.width_t, steps, B = B, n = n, L_KIND = l_kind,
                    OUT_TEMP = out_temp, TILE = tile, num_warps = 4, **common)
