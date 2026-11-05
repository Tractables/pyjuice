"""
A class's weight to its tokens' scores: the kernels of a next-token distribution, a group of positions at a time (:func:`score_group`).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from pyjuice.constraints.distributions import find_token
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ..common import _lse_step, _tf32_operand
from ..sum.tile_rules import _num_sms


# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

@triton.jit
def _column_rows(nodes, mrow, mstride, b, ks, km, STRIDE: tl.constexpr):
    """The class mass rows of a column's input nodes ``ks`` (its position's ``nodes``, its sample ``b``)."""
    u = tl.load(nodes + ks, mask = km, other = 0)
    r = tl.load(mrow + u, mask = km, other = 0).to(tl.int64)
    if STRIDE:
        r += b * tl.load(mstride + u, mask = km, other = 0).to(tl.int64)
    return r


@triton_jit
def _score_weights_kernel(w, row, rep, class_mars, rows, nodes, positions, mrow, mstride, lognorm, shift, zc, part,
                          class_of_stored, C, K, B, KS, Cs, TC: tl.constexpr, TK: tl.constexpr,
                          PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr, SHARES: tl.constexpr, FINISH: tl.constexpr,
                          SPLIT: tl.constexpr):
    """One program per (column ``n``, tile of ``TC`` STORED classes, share ``s`` of the rows ``[s KS, (s + 1) KS)``) of a
    row group's log class weights ``w`` [live rows, K], (column, class)'s row ``row[n, c]`` (-1: dead, its weights
    -inf; see :class:`ClassWeightRows`) -- an enumerated class's weights are left as they are: its token's
    dot takes them in log (:func:`_log_dot`). For stored column ``x`` (class ``class_of_stored[x]``): ``shift[n, c]``,
    the maximum over the rows of the class's weights (``PER_COLUMN``, the class masses of each column's own, see
    :class:`~..state.ClassMasses`: less the row's ``lognorm`` -- a token's own weight ``q(v) = p(v) w(v) / Z`` is
    ``p(v)``'s times ``w(v) / Z``), the weights shifted by it and exponentiated in place, and ``zc[n, x] = log sum_k
    exp(w[n, c, k] + mass_k(c))``, the class's part of ``p(e, C)``, from the mass table. Column ``n`` is sample ``n %
    B`` of position ``positions[n // B]``, whose input nodes are that row of ``nodes`` ([positions of the PC, K]):
    node ``u`` reads row ``mrow[u] + b mstride[u]`` (``STRIDE``; else ``mrow[u]``) of the masses; without per-column
    masses every column reads ``rows``. One share (``SHARES`` 1): all of it. Several: the shares' maxima, running
    maxima and sums into ``part`` ([N, class tiles, SHARES, 3, TC]), then (``FINISH``, a second launch) every share
    combines them in order and writes its rows. A program none of whose classes is live for its column (most of them:
    a column has a few live stored classes, an observed one none) skips its walks over the rows -- every load of
    them would be masked off -- and writes what they would have left: -inf shifts and parts. ``SPLIT`` (classes
    sharing rows, see :class:`ClassWeightRows`): every class reads its row's weights in the first launch (the shifts,
    the parts, the shares' parts), and the second (``FINISH``) exponentiates every row once, by the class that owns it
    (``rep[n, c] == c``, its representative), from the shift the first wrote."""
    n = tl.program_id(0).to(tl.int64)
    ct = tl.program_id(1)
    sh_ = tl.program_id(2)
    xs = ct * TC + tl.arange(0, TC)
    cm = xs < Cs
    cs = tl.load(class_of_stored + xs, mask = cm, other = 0).to(tl.int64)
    rr = tl.load(row + n * C + cs, mask = cm, other = -1)                    # the classes' rows (-1: dead, -inf)
    lv = cm & (rr >= 0)
    any_live = tl.max(lv.to(tl.int32), axis = 0) > 0
    base = w + tl.maximum(rr, 0).to(tl.int64) * K
    t = tl.load(positions + n // B)
    b = n % B
    k_lo = sh_ * KS
    k_hi = tl.where(any_live, tl.minimum(K, k_lo + KS), k_lo)               # (no live class: no rows to walk)
    if not FINISH:
        big = tl.full([TC], float("-inf"), dtype = tl.float32)
        mz = tl.full([TC], float("-inf"), dtype = tl.float32)
        az = tl.zeros([TC], dtype = tl.float32)
        for k0 in range(k_lo, k_hi, TK):
            ks = k0 + tl.arange(0, TK)
            km = ks < k_hi
            if PER_COLUMN:
                r = _column_rows(nodes + t * K, mrow, mstride, b, ks, km, STRIDE)
                lz = tl.load(lognorm + r, mask = km, other = 0.0)
            else:
                r = tl.load(rows + ks, mask = km, other = 0).to(tl.int64)
                lz = tl.zeros([TK], dtype = tl.float32)
            mk = lv[:, None] & km[None, :]
            v = tl.load(base[:, None] + ks[None, :], mask = mk, other = float("-inf"))                  # [TC, TK]
            x = v + tl.load(class_mars + r[None, :] * Cs + xs[:, None], mask = mk, other = float("-inf"))
            mr = tl.max(x, axis = 1)                     # (selects, not maxima: Triton's thread-locality pass asserts
            m_new = tl.where(mr > mz, mr, mz)            # on a reduction accumulated by max whose result has several
            sh = tl.where(m_new == float("-inf"), 0.0, m_new)                             # uses after the loop)
            az = az * tl.exp(mz - sh) + tl.sum(tl.exp(x - sh[:, None]), axis = 1)
            mz = m_new
            br = tl.max(v - lz[None, :], axis = 1)
            big = tl.where(br > big, br, big)
        if SHARES > 1:
            out = part + (((n * tl.num_programs(1) + ct) * SHARES + sh_) * 3) * TC + tl.arange(0, TC)
            tl.store(out, big)
            tl.store(out + TC, mz)
            tl.store(out + 2 * TC, az)
            return
    elif SHARES > 1:
        big = tl.full([TC], float("-inf"), dtype = tl.float32)
        mz = tl.full([TC], float("-inf"), dtype = tl.float32)
        for j in range(SHARES):                                             # (in order: the same in every share)
            src = part + (((n * tl.num_programs(1) + ct) * SHARES + j) * 3) * TC + tl.arange(0, TC)
            big = tl.maximum(big, tl.load(src))
            mz = tl.maximum(mz, tl.load(src + TC))
        zs0 = tl.where(mz == float("-inf"), 0.0, mz)
        az = tl.zeros([TC], dtype = tl.float32)
        for j in range(SHARES):
            src = part + (((n * tl.num_programs(1) + ct) * SHARES + j) * 3) * TC + tl.arange(0, TC)
            az += tl.load(src + 2 * TC) * tl.exp(tl.load(src + TC) - zs0)
    else:                                                                   # (SPLIT: the shifts the first launch wrote)
        big = tl.load(shift + n * C + cs, mask = cm, other = float("-inf"))
    if not FINISH or SHARES > 1:
        if sh_ == 0:
            tl.store(shift + n * C + cs, big, mask = cm)
            tl.store(zc + n * Cs + xs, tl.log(az) + tl.where(mz == float("-inf"), 0.0, mz), mask = cm)
    if SPLIT and not FINISH:
        return
    own = lv
    if SPLIT:                                                               # (a row exponentiated once, by its owner)
        own = lv & (tl.load(rep + n * C + cs, mask = lv, other = -1) == cs)
    s = tl.where(big == float("-inf"), 0.0, big)
    for k0 in range(k_lo, k_hi, TK):
        ks = k0 + tl.arange(0, TK)
        km = ks < k_hi
        if PER_COLUMN:
            r = _column_rows(nodes + t * K, mrow, mstride, b, ks, km, STRIDE)
            lz = tl.load(lognorm + r, mask = km, other = 0.0)
        else:
            lz = tl.zeros([TK], dtype = tl.float32)
        mk = own[:, None] & km[None, :]
        ptr = base[:, None] + ks[None, :]
        v = tl.load(ptr, mask = mk, other = float("-inf")) - lz[None, :]
        tl.store(ptr, tl.exp(v - s[:, None]), mask = mk)


#: :func:`_score_weights_kernel`: the fewest rows of a share (the rows are split only when the columns x class tiles
#: are fewer than twice the SMs)
SCORE_SHARE_ROWS = 256


@triton.jit
def _plain_dot(trow, wrow, ok, K, BH: tl.constexpr, EVEN_K: tl.constexpr):
    """``sum_k T[v, k] a[k]`` of ``BV`` lanes, over the lanes' rows of the table (``trow`` [BV, 1] pointers) and
    their exponentiated weights (``wrow``), in fp32; ``ok`` the lanes on; ``EVEN_K``: ``BH`` divides ``K``."""
    acc = tl.zeros([trow.shape[0]], dtype = tl.float32)
    for k0 in range(0, K, BH):
        ks = k0 + tl.arange(0, BH)
        if EVEN_K:
            mk = ok[:, None]
        else:
            mk = ok[:, None] & (ks < K)[None, :]
        tv = tl.load(trow + ks[None, :], mask = mk, other = 0.0).to(tl.float32)                     # [BV, BH]
        a = tl.load(wrow + ks[None, :], mask = mk, other = 0.0)                                     # [BV, BH]
        acc += tl.sum(tv * a, axis = 1)
    return acc


@triton.jit
def _log_dot(trow, wrow, nodes_t, mrow, mstride, lognorm, b, ok, enum, K, BH: tl.constexpr, EVEN_K: tl.constexpr,
             ENUM: tl.constexpr, PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr):
    """``(dot, shift)`` of ``BV`` lanes: ``sum_k T[v, k] a[k]`` over the lanes' rows of the table (``trow`` [BV, 1]
    pointers) and weights (``wrow``) -- the weights as :func:`_score_weights_kernel` left them, exponentiated, for a
    stored class's lane (:func:`_plain_dot`); ``enum`` (``ENUM``: some class is enumerated), an enumerated class's:
    in log, so the lane first takes their maximum (less the rows' ``lognorm`` under per-column masses,
    ``PER_COLUMN``; the rows of the column's position's input nodes ``nodes_t``, see :func:`_column_rows`) as its
    shift, then exponentiates them under it as the dot goes -- a program without such a lane takes the plain dot.
    The dot in fp32; ``ok`` the lanes on; ``EVEN_K``: ``BH`` divides ``K``."""
    m = tl.full([trow.shape[0]], float("-inf"), dtype = tl.float32)
    if ENUM:
        if tl.max(tl.where(enum, 1, 0), axis = 0) > 0:
            for k0 in range(0, K, BH):
                ks = k0 + tl.arange(0, BH)
                km = ks < K
                a = tl.load(wrow + ks[None, :], mask = enum[:, None] & km[None, :], other = float("-inf"))
                if PER_COLUMN:
                    a = a - tl.load(lognorm + _column_rows(nodes_t, mrow, mstride, b, ks, km, STRIDE), mask = km,
                                    other = 0.0)[None, :]
                mr = tl.max(a, axis = 1)
                m = tl.where(mr > m, mr, m)                      # (a select: see _score_weights_kernel)
            s = tl.where(m == float("-inf"), 0.0, m)
            acc = tl.zeros([trow.shape[0]], dtype = tl.float32)
            for k0 in range(0, K, BH):
                ks = k0 + tl.arange(0, BH)
                if EVEN_K:
                    mk = ok[:, None]
                else:
                    mk = ok[:, None] & (ks < K)[None, :]
                tv = tl.load(trow + ks[None, :], mask = mk, other = 0.0).to(tl.float32)             # [BV, BH]
                a = tl.load(wrow + ks[None, :], mask = mk, other = 0.0)                             # [BV, BH]
                if PER_COLUMN:
                    lz = tl.load(lognorm + _column_rows(nodes_t, mrow, mstride, b, ks, ks < K, STRIDE),
                                 mask = ks < K, other = 0.0)
                    e = tl.exp(a - lz[None, :] - s[:, None])
                else:
                    e = tl.exp(a - s[:, None])
                acc += tl.sum(tv * tl.where(enum[:, None] & mk, e, a), axis = 1)   # (a row past K adds its 0:
                                                       # exp(0 - s) is inf for a lane's shift below -88, 0 x inf NaN)
        else:
            acc = _plain_dot(trow, wrow, ok, K, BH, EVEN_K)
    else:
        acc = _plain_dot(trow, wrow, ok, K, BH, EVEN_K)
    return acc, m


@triton_jit
def _score_tokens_kernel(table, w, row, enum_tokens, token_class, code, tok_ptr, enum_logw, positions, nodes, mrow,
                         mstride, lognorm, out_row, out, uz, N, K, V, C, B, E, logw_stride, BV: tl.constexpr,
                         BH: tl.constexpr, EVEN_K: tl.constexpr, LOGW: tl.constexpr, WRITE: tl.constexpr,
                         PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of the ``E`` enumerated classes' tokens): ``u_v = log sum_k T[v, k]
    exp(w[row[n, c(v)], k])`` (:func:`_log_dot`) -- ``LOGW``: plus the token's log-weight for the column's sample,
    ``enum_logw[b, tok_ptr[t] + e]``, ``e`` its number among the live enumerated tokens of the column's position ``t``
    (from its class's code); a token weighing nothing, or of a class dead at ``t`` (or for the column, its row -1),
    -inf, its dot skipped -- the
    token's UNNORMALIZED log score, into ``out[out_row[n], v]`` (``WRITE``; ``out`` [M, V]), and ``uz[n, tile] = log
    sum_v exp(u_v)`` over the tile: the tokens' part of ``p(e, C)`` (see :func:`_logz_kernel`). Columns of no row
    (``out_row[n]`` -1): -inf, nothing written."""
    n = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    es = tile * BV + tl.arange(0, BV)
    em = es < E
    r = tl.load(out_row + n)
    t = tl.load(positions + n // B)
    b = n % B
    tk = tl.load(enum_tokens + es, mask = em, other = 0).to(tl.int64)
    c = tl.load(token_class + tk, mask = em, other = 0).to(tl.int64)
    x = tl.load(code + t * C + c, mask = em, other = 0)
    t0 = tl.load(tok_ptr + t)
    e = tl.maximum(-1 - x, 0)
    live = em & (r >= 0) & (x < 0) & (e < tl.load(tok_ptr + t + 1) - t0)
    rw = tl.load(row + n * C + c, mask = live, other = -1)                    # its class's row (-1: dead, -inf)
    live = live & (rw >= 0)
    if LOGW:
        lw = tl.load(enum_logw + b * logw_stride + t0 + e, mask = live, other = float("-inf"))
    else:
        lw = tl.where(live, 0.0, float("-inf"))
    ok = lw > float("-inf")
    KD = tl.where(tl.max(ok.to(tl.int32), axis = 0) > 0, K, 0)              # (no lane on: no dot, every lane -inf)
    acc, m = _log_dot(table + tk[:, None] * K, w + tl.maximum(rw, 0).to(tl.int64)[:, None] * K, nodes + t * K, mrow,
                      mstride, lognorm, b, ok, ok, KD, BH, EVEN_K, True, PER_COLUMN, STRIDE)
    u = tl.where(ok, tl.log(acc) + m + lw, float("-inf"))
    top = tl.max(u, axis = 0)
    s = tl.where(top == float("-inf"), 0.0, top)
    tl.store(uz + n * tl.num_programs(1) + tile, tl.log(tl.sum(tl.exp(u - s), axis = 0)) + s)
    if WRITE:
        tl.store(out + tl.maximum(r, 0).to(tl.int64) * V + tk, u, mask = em & (r >= 0))


@triton_jit
def _logz_kernel(zc, uz, logz, Cs, UT, TC: tl.constexpr):
    """One program per column ``n``: ``logz[n] = log (sum_x exp(zc[n, x]) + sum_j exp(uz[n, j]))``, ``log p(e, C)`` --
    the ``Cs`` stored classes' parts and the enumerated tokens' (``UT`` tiles of them, see :func:`_score_tokens_kernel`)."""
    n = tl.program_id(0).to(tl.int64)
    m = tl.full([TC], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TC], dtype = tl.float32)
    for c0 in range(0, Cs, TC):
        cs = c0 + tl.arange(0, TC)
        m, acc = _lse_step(m, acc, tl.load(zc + n * Cs + cs, mask = cs < Cs, other = float("-inf")))
    for j0 in range(0, UT, TC):
        js = j0 + tl.arange(0, TC)
        m, acc = _lse_step(m, acc, tl.load(uz + n * UT + js, mask = js < UT, other = float("-inf")))
    top = tl.max(m, axis = 0)
    s = tl.where(top == float("-inf"), 0.0, top)
    tl.store(logz + n, tl.log(tl.sum(acc * tl.exp(m - s), axis = 0)) + s)


@triton_jit
def _normalize_kernel(out, out_row, logz, idx, off, L, V, KC, BV: tl.constexpr, MODE: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of its row's ``L`` scored entries): less the column's ``logz``
    (-inf where ``p(e, C)`` is 0). The entries: ``MODE`` 0, ``out[r, j]`` (``out`` [M, KC], the row's candidates); 1,
    ``out[r, idx[off[n] + j]]`` (``out`` [M, V], the run of tokens its weights list, none for a column whose offset is
    -1); 2, ``out[r, idx[j]]`` (the enumerated classes' tokens)."""
    n = tl.program_id(0).to(tl.int64)
    js = tl.program_id(1) * BV + tl.arange(0, BV)
    r = tl.load(out_row + n)
    jm = (js < L) & (r >= 0)
    rr = tl.maximum(r, 0).to(tl.int64)
    if MODE == 0:
        ptr = out + rr * KC + js
    elif MODE == 1:
        wo = tl.load(off + n)
        jm = jm & (wo >= 0)
        ptr = out + rr * V + tl.load(idx + tl.maximum(wo, 0) + js, mask = jm, other = 0).to(tl.int64)
    else:
        ptr = out + rr * V + tl.load(idx + js, mask = jm, other = 0).to(tl.int64)
    lz = tl.load(logz + n)
    val = tl.load(ptr, mask = jm, other = float("-inf"))
    tl.store(ptr, tl.where(lz == float("-inf"), float("-inf"), val - lz), mask = jm)


@triton_jit
def _score_kernel(table, w, row, shift, logz, toks, tile_cls, out_row, out, logw, logw_off, N, K, V, C, KW,
                  BB: tl.constexpr, BV: tl.constexpr, BH: tl.constexpr, EVEN_K: tl.constexpr,
                  PRECISION: tl.constexpr, WEIGHTS: tl.constexpr):
    """One program per (``BB`` columns, tile of ``BV`` tokens of one stored class): ``log sum_k T[v, k] w[row[n, c],
    k] + shift[n, c] - logz[n]`` (a dead row, -1: -inf) from the token-major table ``T`` [V, K] (fp32 or bf16) and
    the shifted, exponentiated weights (:func:`_score_weights_kernel`), into row ``out_row[n]`` of ``out`` [M, V] (-1:
    not requested); -inf where ``p(e, C)`` is 0. ``EVEN_K``: ``BH`` divides ``K``. ``WEIGHTS``: plus the column's
    tokens' log-weights, ``logw[logw_off[n] + v]`` for ``v < KW`` (-inf beyond; none for a column whose offset is
    -1)."""
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    tk = tl.load(toks + pid_v * BV + tl.arange(0, BV))                        # -1: padding
    ok = tk >= 0
    c = tl.load(tile_cls + pid_v)
    ns = pid_n * BB + tl.arange(0, BB)
    nm = ns < N
    trow = table + tk.to(tl.int64)[None, :] * K
    rw = tl.load(row + ns.to(tl.int64) * C + c, mask = nm, other = -1)       # the columns' rows of the class
    nl = nm & (rw >= 0)                                                       # (-1: dead, weights -inf)
    wrow = w + tl.maximum(rw, 0).to(tl.int64)[:, None] * K
    acc = tl.zeros([BB, BV], dtype = tl.float32)
    KD = tl.where(tl.max(nl.to(tl.int32), axis = 0) > 0, K, 0)             # (the class dead for every column: no dot)
    for k0 in range(0, KD, BH):
        ks = k0 + tl.arange(0, BH)
        if EVEN_K:
            bt = tl.load(trow + ks[:, None], mask = ok[None, :], other = 0.0)                           # [BH, BV]
            a = tl.load(wrow + ks[None, :], mask = nl[:, None], other = 0.0)                           # [BB, BH]
        else:
            km = ks < K
            bt = tl.load(trow + ks[:, None], mask = km[:, None] & ok[None, :], other = 0.0)
            a = tl.load(wrow + ks[None, :], mask = nl[:, None] & km[None, :], other = 0.0)
        acc += tl.dot(_tf32_operand(a, PRECISION), _tf32_operand(bt, PRECISION).to(tl.float32),   # (bf16: exact)
                      input_precision = PRECISION)
    sc = tl.load(shift + ns.to(tl.int64) * C + c, mask = nm, other = float("-inf"))
    lz = tl.load(logz + ns, mask = nm, other = float("-inf"))
    val = tl.where((lz == float("-inf"))[:, None], float("-inf"), tl.log(acc) + (sc - lz)[:, None])
    if WEIGHTS:
        wo = tl.load(logw_off + ns, mask = nm, other = -1)
        lw = tl.load(logw + tl.maximum(wo, 0)[:, None] + tk[None, :],
                     mask = (wo >= 0)[:, None] & (ok & (tk < KW))[None, :], other = float("-inf"))
        val = tl.where((wo >= 0)[:, None], val + lw, val)
    r = tl.load(out_row + ns, mask = nm, other = -1)
    tl.store(out + r.to(tl.int64)[:, None] * V + tk[None, :], val, mask = (r >= 0)[:, None] & ok[None, :])


@triton_jit
def _score_candidates_kernel(table, w, row, shift, token_class, code, positions, nodes, mrow, mstride, lognorm, cand,
                             out_row, out, uz, logw, logw_off, ids, N, K, V, C, B, KC, KW, BV: tl.constexpr,
                             BH: tl.constexpr, EVEN_K: tl.constexpr, WEIGHTS: tl.constexpr, STEPS: tl.constexpr,
                             UZ: tl.constexpr, ENUM: tl.constexpr, PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of its row's candidates): ``log sum_k T[v, k] exp(w[row[n,
    c(v)], k])`` (a dead row, -1: -inf), the UNNORMALIZED log score (less ``logz`` afterwards,
    :func:`_normalize_kernel`), for every candidate ``v = cand[r, j]`` of row ``r = out_row[n]`` (-1: not requested),
    into ``out[r, j]`` (``out`` [M, KC]); -inf where ``v`` is outside the vocabulary (padding). The weights as
    :func:`_score_weights_kernel` left them -- ``ENUM``: an enumerated class's (by its code at the column's position,
    ``code[t, c]`` < 0) in log, see :func:`_log_dot` --, the dot in fp32 over the candidates' rows of the table
    alone. ``EVEN_K``: ``BH`` divides ``K``. ``WEIGHTS``:
    plus the column's tokens' log-weights (none for a column whose offset ``logw_off[n]`` is -1) -- 1: as
    :func:`_score_kernel`; 2: of the tokens its ascending run of ``KW`` in ``ids`` lists, ``logw`` aligned with it
    (found in ``STEPS`` halvings; the others -inf); 3: the candidates are that run's tokens themselves, each scored
    into ``out[r, v]`` (``out`` [M, V]; the rest of the row is not written). ``UZ``: ``uz[n, tile] = log sum exp`` of
    the tile's enumerated candidates' scores: the tokens' part of ``p(e, C)`` (see :func:`_logz_kernel`) -- when the
    candidates are every token with a weight (3)."""
    n = tl.program_id(0).to(tl.int64)
    js = tl.program_id(1) * BV + tl.arange(0, BV)
    r = tl.load(out_row + n)
    rr = tl.where(r >= 0, r, 0).to(tl.int64)
    t = tl.load(positions + n // B)
    b = n % B
    if WEIGHTS != 0:
        wo = tl.load(logw_off + n)
        wb = tl.maximum(wo, 0)
    if WEIGHTS == 3:
        jm = (js < KW) & (r >= 0) & (wo >= 0)
        tk = tl.load(ids + wb + js, mask = jm, other = -1)
    else:
        jm = (js < KC) & (r >= 0)
        tk = tl.load(cand + rr * KC + js, mask = jm, other = -1)
    ok = jm & (tk >= 0) & (tk < V)
    tks = tl.where(ok, tk, 0).to(tl.int64)
    c = tl.load(token_class + tks, mask = ok, other = 0).to(tl.int64)
    rw = tl.load(row + n * C + c, mask = ok, other = -1)                      # its class's row (-1: dead, -inf)
    on = ok & (rw >= 0)
    if ENUM:
        enum = on & (tl.load(code + t * C + c, mask = on, other = 0) < 0)
    else:
        enum = on & (c < 0)
    KD = tl.where(tl.max(on.to(tl.int32), axis = 0) > 0, K, 0)              # (no lane on: no dot, every lane -inf)
    acc, m = _log_dot(table + tks[:, None] * K, w + tl.maximum(rw, 0).to(tl.int64)[:, None] * K, nodes + t * K, mrow,
                      mstride, lognorm, b, on, enum, KD, BH, EVEN_K, ENUM, PER_COLUMN, STRIDE)
    sc = tl.where(enum, m, tl.load(shift + n * C + c, mask = on & ~enum, other = float("-inf")))
    val = tl.where(on, tl.log(acc) + sc, float("-inf"))
    if WEIGHTS == 1:
        lw = tl.load(logw + wb + tks, mask = ok & (wo >= 0) & (tks < KW), other = float("-inf"))
        val = tl.where(wo >= 0, val + lw, val)
    elif WEIGHTS == 2:
        k = find_token(ids, wb + tl.zeros_like(tks), KW, tks, ok & (wo >= 0), STEPS)
        lw = tl.load(logw + wb + k, mask = k >= 0, other = float("-inf"))
        val = tl.where(wo >= 0, val + lw, val)
    elif WEIGHTS == 3:
        val = val + tl.load(logw + wb + js, mask = ok, other = float("-inf"))
    if UZ:
        u = tl.where(enum, val, float("-inf"))
        top = tl.max(u, axis = 0)
        s = tl.where(top == float("-inf"), 0.0, top)
        tl.store(uz + n * tl.num_programs(1) + tl.program_id(1), tl.log(tl.sum(tl.exp(u - s), axis = 0)) + s)
    if WEIGHTS == 3:
        tl.store(out + rr * V + tks, val, mask = ok)
    else:
        tl.store(out + rr * KC + js, val, mask = jm)


#: :func:`_score_tokens_kernel`'s ``(BV, BH, warps)`` -- tokens per program, the dot's chunk of rows -- for at most
#: :data:`SCORE_CANDIDATE_PAIRS` (column, token) pairs, and for more: from a sweep of the cached decode step on Ctrl-G's
#: HMM4096 (4096 rows) at 1/16/128 columns x 50/1024 candidates, the first within 2% of the best everywhere but
#: 128 x 1024, where the second takes the step from 3.32 to 2.87 ms (scoring every token: 2.92)
SCORE_TOKEN_TILES = ((16, 256, 4), (64, 64, 2))
#: ... and :func:`_score_candidates_kernel`'s, from its launches' captured inputs (1024 candidates a column, 4096 rows):
#: Codpula-4096 x JSON hard_0 diffusion, B8 / B32 / B64 (1024 / 4096 / 8192 columns) 1396 -> 942 / 1906 -> 1212 /
#: 9843 -> 7729 us from (64, 64, 2), HMM-4096 x GPT-2 JSON with candidates (B16, 2048 columns) 1146 -> 755 -- 242
#: registers a thread at 64 tokens, 12% occupancy --; l2r B8 / B32 (8 / 32 columns) 47 -> 43 / 101 -> 94 from
#: (16, 256, 4). The same chunks of rows: every candidate's score the same bits; log Z's sums over the enumerated
#: candidates run in other tiles: the conditionals within fp32 rounding (3.8e-6 in log at most there)
SCORE_CANDIDATE_TILES = ((16, 256, 8), (16, 64, 2))
SCORE_CANDIDATE_PAIRS = 1 << 16
#: :func:`_score_weights_kernel`'s warps (its tiles, and so its arithmetic, unchanged by them). Its two launches on
#: their captured inputs, Codpula-4096 x JSON hard_0: l2r B8 37 -> 29 us, diffusion B8 / B32 / B64 1185 -> 484 /
#: 3180 -> 1491 / 5848 -> 2964 us from 4 (184 registers a thread: 16% occupancy)
SCORE_WEIGHT_WARPS = 2
#: :func:`_score_kernel`'s tiles ``(BB, BV, BH, warps, stages)`` by table dtype and dot precision, for at most 16, 64 and
#: more columns: from a sweep of the kernel on Ctrl-G's HMM4096 (V = 50257, 4096 rows) and first example's 10 classes at
#: 1/16, 64 and 128 columns, ``BH`` (the dots' reduction chunk) fixed per dtype. (16, 64, 32, 4, 2) -- the fastest
#: before the epilogue read the shifts -- lost 16% with it, to code generation: no other tile changed.
SCORE_TILES = {("fp32", "tf32x3"): ((16, 128, 32, 4, 2), (32, 64, 32, 8, 3), (32, 128, 32, 4, 3)),
               ("fp32", "tf32"): ((16, 64, 32, 4, 2), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3)),
               ("bf16", "tf32"): ((16, 64, 64, 4, 2), (32, 64, 64, 4, 2), (32, 128, 64, 8, 3))}


def score_group(out: torch.Tensor, table: torch.Tensor, w: torch.Tensor, row: torch.Tensor, rows: torch.Tensor,
                class_mars: torch.Tensor, out_row: torch.Tensor, classes, precision: str, positions: torch.Tensor,
                enum: dict, candidates: Optional[torch.Tensor] = None, sample: Optional[dict] = None,
                rep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Score one row group: ``out[out_row[n], v] = log p(x_t = v | e, C)`` for every column ``n`` (a target position's
    sample) that ``out_row`` asks for, from its log class weights -- ``w`` [live rows, K] (the stored classes'
    overwritten), column ``n``'s class ``c``'s at row ``row[n, c]`` (int32 [N, C]; -1: dead, its weights -inf, see
    :class:`ClassWeightRows`) --, the group's token-major table [V, K] and its rows of the class masses (``rows``,
    int64 [K] on the device).

    ``p(e, C) = sum_c sum_k A[k, c] mass_k(c)``: the stored classes' parts from the class masses
    (:func:`_score_weights_kernel`), an enumerated class's from its token's unnormalized score -- the dot the scoring
    computes anyway (:func:`_score_tokens_kernel`; under top-k weights, of the tokens with a weight alone, scored by
    :func:`_score_candidates_kernel`) -- so no second pass over the tokens, and no mass per (row, enumerated class).
    An enumerated class's weights are read once, by its token's dot, in log (:func:`_log_dot`): the shift-and-exp
    pass is over the stored classes alone.

    :param classes: the constraint's :class:`~pyjuice.constraints.distributions.TokenClasses`
    :param precision: the dot's: "tf32x3" (fp32-level) or "tf32"
    :param positions: int64 [N / columns of a position] on the device: the columns' positions (the samples of a
        position its consecutive columns)
    :param enum: how the class masses are read by class code (:func:`~.transitions.enumerated_args`)
    :param candidates: None (every token), or int32 [M, k] on the device: row ``r`` scores ``candidates[r]`` alone,
        into ``out[r]`` ([M, k]), by :func:`_score_candidates_kernel` (in fp32 whatever ``precision``)
    :param sample: None, or the columns' own class masses under per-sample inputs (see
        :class:`~..state.ClassMasses`): a dict with ``nodes`` (int64 [positions of the PC, K] on the device: every
        position's input nodes), ``row`` and ``stride`` (int32 per node; ``stride`` None
        for the same rows for every sample: node ``u`` of sample ``b`` reads row ``row[u] + b stride[u]`` of
        ``class_mars``), ``lognorm`` (every row's ``log Z``) and the tokens' weights, ``logw`` (flat), ``offset`` (int64
        [N], every column's, -1 for none), ``width`` (the tokens with a weight) and ``ids`` (None: the first ``width``
        tokens; else int32, flat as ``logw``, the tokens of its ascending runs -- every column then has an offset, and
        without ``candidates`` only those tokens are written: the rest of ``out``'s rows must be -inf already)
    :param rep: None, or int32 [N, C]: every (column, class)'s representative, when classes share rows (see
        :class:`ClassWeightRows`): every row is then exponentiated once, by its representative
    :returns: ``log p(e, C)`` of every column, [N]
    """
    (N, C), K = row.shape, w.size(1)
    V = table.size(0)
    B = N // positions.numel()
    dev = w.device
    Cs, E = classes.num_stored, classes.num_enumerated                        # (one token each)
    shift = torch.empty(N, C, dtype = torch.float32, device = dev)
    zc = torch.empty(N, max(Cs, 1), dtype = torch.float32, device = dev)      # the stored classes' parts of p(e, C)
    logz = torch.empty(N, dtype = torch.float32, device = dev)
    per_column = sample is not None
    ids = sample["ids"] if per_column else None
    weights = dict(logw = sample["logw"] if per_column else w, logw_off = sample["offset"] if per_column else rows,
                   KW = sample["width"] if per_column else 0)
    stride = per_column and sample["stride"] is not None
    columns = dict(nodes = sample["nodes"] if per_column else rows, mrow = sample["row"] if per_column else rows,
                   mstride = sample["stride"] if stride else rows, lognorm = sample["lognorm"] if per_column else w)
    modes = dict(PER_COLUMN = per_column, STRIDE = stride)

    # the stored classes: the shifts, the weights exponentiated, their parts of p(e, C)
    if Cs > 0:
        TC, TK = min(64, max(16, host_next_power_of_2(Cs))), 64
        ctiles = host_cdiv(Cs, TC)
        shares = max(1, min(host_cdiv(2 * _num_sms(dev), N * ctiles), host_cdiv(K, SCORE_SHARE_ROWS)))
        KS = host_cdiv(host_cdiv(K, shares), TK) * TK
        shares = host_cdiv(K, KS)
        part = torch.empty(N * ctiles * shares * 3 * TC if shares > 1 else 1, dtype = torch.float32, device = dev)
        args = (w, row, row if rep is None else rep, class_mars, rows, columns["nodes"], positions, columns["mrow"],
                columns["mstride"], columns["lognorm"], shift, zc, part, classes.class_of_stored32, C, K, B, KS, Cs)
        consts = dict(TC = TC, TK = TK, SHARES = shares, SPLIT = rep is not None, num_warps = SCORE_WEIGHT_WARPS, **modes)
        _score_weights_kernel[(N, ctiles, shares)](*args, FINISH = False, **consts)
        if shares > 1 or rep is not None:
            _score_weights_kernel[(N, ctiles, shares)](*args, FINISH = True, **consts)

    # the scores: the candidates', or the enumerated tokens' (unnormalized), with the enumerated tokens' parts of p(e, C)
    run = ids is not None and candidates is None                           # the weights' run of tokens, every one
    by_tokens = E > 0 and not run                                           # with a weight: their parts come with
    KC = (candidates.size(1) if candidates is not None else weights["KW"]) if candidates is not None or ids is not None \
        else 0
    tokens_tile = SCORE_TOKEN_TILES[N * E > SCORE_CANDIDATE_PAIRS]
    cand_tile = SCORE_CANDIDATE_TILES[N * KC > SCORE_CANDIDATE_PAIRS]
    UT = host_cdiv(E, tokens_tile[0]) if by_tokens else host_cdiv(KC, cand_tile[0]) if run and E > 0 else 0
    uz = torch.empty(N, max(UT, 1), dtype = torch.float32, device = dev)
    if candidates is not None or ids is not None:                           # top-k weights: their tokens alone
        mode = (2 if ids is not None else int(per_column)) if candidates is not None else 3
        BV, BH, warps = cand_tile
        _score_candidates_kernel[(N, host_cdiv(KC, BV))](
            table, w, row, shift, classes.token_class, enum["code"], positions, columns["nodes"], columns["mrow"],
            columns["mstride"], columns["lognorm"], candidates if candidates is not None else rows, out_row, out, uz,
            ids = ids if ids is not None else rows, N = N, K = K, V = V, C = C, B = B, KC = KC, BV = BV, BH = BH,
            EVEN_K = K % BH == 0, WEIGHTS = mode, STEPS = max(1, int(weights["KW"]).bit_length()),
            UZ = run and E > 0, ENUM = E > 0, num_warps = warps, **weights, **modes)
    if by_tokens:
        BV, BH, warps = tokens_tile
        _score_tokens_kernel[(N, UT)](table, w, row, classes.enumerated_tokens, classes.token_class, enum["code"],
                                      enum["tok_ptr"], enum["enum_logw"], positions, columns["nodes"], columns["mrow"],
                                      columns["mstride"], columns["lognorm"], out_row, out, uz, N = N, K = K, V = V,
                                      C = C, B = B, E = E, logw_stride = enum["logw_stride"], BV = BV, BH = BH,
                                      EVEN_K = K % BH == 0, LOGW = enum["LOGW"] and per_column,
                                      WRITE = candidates is None and ids is None, num_warps = warps, **modes)
    _logz_kernel[(N,)](zc, uz, logz, Cs, UT, TC = min(1024, max(16, host_next_power_of_2(max(Cs, UT, 1)))),
                       num_warps = 1)
    if candidates is not None or ids is not None:
        BV = cand_tile[0]
        _normalize_kernel[(N, host_cdiv(KC, BV))](out, out_row, logz, ids if run else rows, weights["logw_off"], KC,
                                                  V, KC, BV = BV, MODE = 1 if run else 0, num_warps = 1)
        return logz
    if E > 0:                                                               # (the enumerated tokens: written already)
        BV = tokens_tile[0]
        _normalize_kernel[(N, UT)](out, out_row, logz, classes.enumerated_tokens, rows, E, V, V, BV = BV, MODE = 2,
                                   num_warps = 1)
    if Cs > 0:                                                              # the stored classes' tokens, in tiles
        dtype = "bf16" if table.dtype == torch.bfloat16 else "fp32"
        prec = "tf32" if dtype == "bf16" else precision
        BB, BV, BH, warps, stages = SCORE_TILES[(dtype, prec)][0 if N <= 16 else 1 if N <= 64 else 2]
        BV = min(BV, max(16, host_next_power_of_2(-(-V // Cs))))            # many small classes: narrower tiles
        toks, tile_cls = classes.class_tiles(BV, stored = True)
        _score_kernel[(host_cdiv(N, BB), tile_cls.numel())](table, w, row, shift, logz, toks, tile_cls, out_row, out,
                                                            N = N, K = K, V = V, C = C, BB = BB, BV = BV, BH = BH,
                                                            EVEN_K = K % BH == 0, PRECISION = prec,
                                                            WEIGHTS = per_column, num_warps = warps,
                                                            num_stages = stages, **weights)
    return logz
