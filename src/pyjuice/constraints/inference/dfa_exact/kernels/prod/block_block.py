"""
The ``block @ block`` kernels, sparse and compact, and the copy of a block.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit

from ..common import _block_ptr, _child, _exit_width, _interval, _lse_step, _slot, _tf32_operand
from .launches import _lse_value, _out_count, _out_ptr


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
def _tile_load(ptr, iv_table, positions, r, c, ncols, rm, cm):
    """The tile of a block at rows ``r`` and columns ``c``, -inf where not joined."""
    mask = rm[:, None] & cm[None, :]
    sl = _slot(iv_table, positions, r[:, None], c[None, :], ncols, mask, False)
    return tl.load(ptr + sl, mask = mask & (sl >= 0), other = float("-inf"))


@triton.jit
def _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, live_q, i, j, ti, tj, E, M, X, TILES_I, TILES_J, WORDS,
                      TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                      SKIP: tl.constexpr, PRUNE: tl.constexpr, PRECISION: tl.constexpr):
    """``log(sum_q exp(L[i, q] + R[q, j]))`` for one output tile, in one pass over the shared chunks it needs: the
    product of the operands' exponentials, shifted by every row's and column's running maximum, the sum rescaled
    whenever a maximum grows (``lpos`` and ``rpos``: the operands' position tables; pruned, ``live_q``: the shared
    boundary's liveness, a chunk without a live column skipped). The shifts can lie farther above an entry's own terms
    than fp32's exponent reaches -- blocks of a wide dynamic range, or an operand kept for every sample (``prior_mars``),
    whose dead columns hold finite values the sample's evidence rules out -- and the entry's terms then flush: the
    returned sums tell (see :data:`EXACT_BELOW`). Below that, fp32 keeps its precision at any scale."""
    im, jm = i < E, j < X
    mL = tl.full([TM], float("-inf"), dtype = tl.float32)
    mR = tl.full([TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for c in range(0, tl.cdiv(M, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            qs = c * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_q + qs, mask = qs < M, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            q = c * TK + tl.arange(0, TK)
            qm = q < M
            lt = _tile_load(lp, iv_table, lpos, i, q, M, im, qm)
            rt = _tile_load(rp, iv_table, rpos, q, j, X, qm, jm)
            nL = tl.maximum(mL, tl.max(lt, axis = 1))
            nR = tl.maximum(mR, tl.max(rt, axis = 0))
            sL = tl.where(nL == float("-inf"), 0.0, nL)
            sR = tl.where(nR == float("-inf"), 0.0, nR)
            fL = tl.where(nL == float("-inf"), 1.0, tl.exp(mL - sL))
            fR = tl.where(nR == float("-inf"), 1.0, tl.exp(mR - sR))
            acc = acc * fL[:, None] * fR[None, :] + tl.dot(_tf32_operand(tl.exp(lt - sL[:, None]), PRECISION),
                                                            _tf32_operand(tl.exp(rt - sR[None, :]), PRECISION),
                                                            input_precision = PRECISION)
            mL, mR = nL, nR
    sL = tl.where(mL == float("-inf"), 0.0, mL)
    sR = tl.where(mR == float("-inf"), 0.0, mR)
    return tl.log(acc) + sL[:, None] + sR[None, :], acc


#: An output entry of :func:`_block_block_tile` whose sum of shifted exponentials is below this is computed again by
#: :func:`_block_block_tile_exact`. Every term the shifts flush out of fp32's range is below 2^-126 (so is a
#: subnormal term's error), at most 2^-112 together over a 16k-column boundary: above 2^-90 they are at most 2^-22
#: of the sum, fp32 rounding's order; below, they need not be.
EXACT_BELOW = 2.0 ** -90


@triton.jit
def _block_block_tile_exact(lp, rp, iv_table, lpos, rpos, mb, live_q, i, j, ti, tj, E, M, X, TILES_I, TILES_J, WORDS,
                            TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr,
                            SUB_J: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr):
    """:func:`_block_block_tile`'s output by an online log-sum-exp per entry over the shared columns, one at a time:
    exact whatever the blocks' dynamic range, at the cost of an exponential per (entry, column) instead of the
    tensor-core product. For the tiles whose entries the shifts flush (see :data:`EXACT_BELOW`): the entries of a
    block over a wide dynamic range that lie far below their row's and column's maxima -- those of a sample pinned by
    its evidence to columns of little mass, in a block kept for every sample."""
    im, jm = i < E, j < X
    m = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for c in range(0, tl.cdiv(M, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            qs = c * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_q + qs, mask = qs < M, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            for k in range(0, TK):
                q = c * TK + k
                ok = q < M
                if PRUNE:
                    ok = ok & (tl.load(live_q + q, mask = q < M, other = 0) != 0)
                if ok:
                    sl = _slot(iv_table, lpos, i, q, M, im, False)
                    lv = tl.load(lp + sl, mask = im & (sl >= 0), other = float("-inf"))
                    sr = _slot(iv_table, rpos, q, j, X, jm, False)
                    rv = tl.load(rp + sr, mask = jm & (sr >= 0), other = float("-inf"))
                    m, acc = _lse_step(m, acc, lv[:, None] + rv[None, :])
    return _lse_value(m, acc)


@triton.jit
def _block_block_program(element_mars, node_mars, temp, width, iv_of, iv_info, live, steps, sel, triples, skip_bits,
                         reg_offset, reg_width, reg_first, prior_mars, prior_offset, prior_width, reg_prior, out_first,
                         out_width, temp_width, B, n, W, TILES_I, TILES_J, WORDS, SPLIT, pid,
                         L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, INDEXED: tl.constexpr):
    """What program ``pid`` of a ``block @ block`` launch works on: its tile group ``g``, step and sample, the
    boundaries, widths and position tables of its three intervals, its operands' and output's blocks, its step's skip
    masks and its sample's liveness row."""
    g = pid % SPLIT
    s = _child(sel, pid // SPLIT // B, INDEXED)
    b = (pid // SPLIT % B).to(tl.int64)
    out_row = tl.load(steps + s * 8 + 0)
    t0 = tl.load(steps + s * 8 + 5)
    t1 = tl.load(steps + s * 8 + 6)
    t2 = tl.load(steps + s * 8 + 7)
    E = tl.load(width + t0)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    LS, lpos, _ = _interval(iv_of, iv_info, t0, t1, n)
    RS, rpos, _ = _interval(iv_of, iv_info, t1, t2, n)
    OS, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    lp, l_own = _block_ptr(tl.load(steps + s * 8 + 1), tl.load(steps + s * 8 + 2), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, L_KIND)
    rp, r_own = _block_ptr(tl.load(steps + s * 8 + 3), tl.load(steps + s * 8 + 4), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, R_KIND)
    lp = lp + tl.where(l_own, b * LS, 0)
    rp = rp + tl.where(r_own, b * RS, 0)
    op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + b * OS
    mb = skip_bits + tl.load(triples + s).to(tl.int64) * TILES_I * TILES_J * WORDS
    lrow = live + b * (n + 1) * W                                            # the sample's liveness (if pruned)
    return g, t0, t1, t2, E, M, X, lpos, rpos, opos, lp, rp, op, mb, lrow


@triton_jit
def _block_block_kernel(element_mars, node_mars, temp, width, iv_of, iv_info, iv_table, live, steps, sel, triples,
                        skip_bits, reg_offset, reg_width, reg_first, prior_mars, prior_offset, prior_width, reg_prior,
                        out_first, out_width, temp_width, B, n, W, TILES_I, TILES_J, WORDS, SPLIT, GROUP, exact_below,
                        pid0, L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr,
                        SUB_J: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr, EXACT: tl.constexpr,
                        PRECISION: tl.constexpr, INDEXED: tl.constexpr):
    """One program per (step, sample, group of ``GROUP`` consecutive output tiles), its tiles in turn: the sample's
    two blocks are contiguous and stay in L1 across them. A (step, sample) has ``SPLIT`` groups: 1 unless the
    launch has too few (step, sample) pairs to fill the GPU. ``EXACT``: the program records, entry by entry, whether
    the shifts flushed any of its tiles' entries (see :data:`EXACT_BELOW`), and when they did -- rarely -- goes over
    its tiles again to find those (the shifted sums again, a reduction per tile) and compute them exactly
    (:func:`_block_block_tile_exact`). An elementwise record and one reduction per program in the common case: a
    pruned tile does little work, and a reduction per tile cost a fifth of it; a second launch for the flagged
    programs cost more still in launches."""
    g, t0, t1, t2, E, M, X, lpos, rpos, opos, lp, rp, op, mb, lrow = _block_block_program(
        element_mars, node_mars, temp, width, iv_of, iv_info, live, steps, sel, triples, skip_bits, reg_offset,
        reg_width, reg_first, prior_mars, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n,
        W, TILES_I, TILES_J, WORDS, SPLIT, pid0 + tl.program_id(0), L_KIND, R_KIND, OUT_TEMP, INDEXED)
    NJ = tl.cdiv(X, TN)
    if EXACT:
        bad = tl.zeros([TM, TN], dtype = tl.int32)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(E, TM) * NJ)):
        ti = t // NJ
        tj = t % NJ
        i = ti * TM + tl.arange(0, TM)
        j = tj * TN + tl.arange(0, TN)
        if PRUNE:                                                            # a tile without a live slot: -inf
            lr = tl.load(lrow + t0 * W + i, mask = i < E, other = 0) != 0
            lc = tl.load(lrow + t2 * W + j, mask = j < X, other = 0) != 0
            out = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
            if EXACT:
                acc = tl.full([TM, TN], 1.0, dtype = tl.float32)
            if (tl.max(lr.to(tl.int32), axis = 0) > 0) & (tl.max(lc.to(tl.int32), axis = 0) > 0):
                out, acc = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M, X,
                                             TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
            out = tl.where(lr[:, None] & lc[None, :], out, float("-inf"))
        else:
            out, acc = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow, i, j, ti, tj, E, M, X, TILES_I,
                                         TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
        om = (i < E)[:, None] & (j < X)[None, :]
        sl = _slot(iv_table, opos, i[:, None], j[None, :], X, om, False)
        if EXACT:
            valid = om & (sl >= 0)
            if PRUNE:
                valid = valid & lr[:, None] & lc[None, :]
            bad = bad | (valid & (acc < exact_below)).to(tl.int32)
        tl.store(op + sl, out, mask = om & (sl >= 0))
    if EXACT:
        if tl.max(bad) > 0:
            for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(E, TM) * NJ)):
                ti = t // NJ
                tj = t % NJ
                i = ti * TM + tl.arange(0, TM)
                j = tj * TN + tl.arange(0, TN)
                om = (i < E)[:, None] & (j < X)[None, :]
                sl = _slot(iv_table, opos, i[:, None], j[None, :], X, om, False)
                valid = om & (sl >= 0)
                if PRUNE:
                    lr = tl.load(lrow + t0 * W + i, mask = i < E, other = 0) != 0
                    lc = tl.load(lrow + t2 * W + j, mask = j < X, other = 0) != 0
                    valid = valid & lr[:, None] & lc[None, :]
                _, acc = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M, X,
                                           TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
                if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                    out = _block_block_tile_exact(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M,
                                                  X, TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE)
                    tl.store(op + sl, tl.where(valid, out, float("-inf")), mask = om & (sl >= 0))


@triton.jit
def _rect_tile(p, base, stride, by_slot, positions, ncols, iv_table, x, y, xp, yp, xm, ym, T: tl.constexpr):
    """The ``[len(x), len(y)]`` tile of a block operand ``S``'s values ``S[x, y]`` (``T``: ``S[y, x]``, the operand
    read transposed), ``x`` and ``y`` columns of the automaton and ``xp`` and ``yp`` their places among the sample's
    live ones (``xm``, ``ym``: the real ones): compact (:mod:`.compact`), at ``p + base + (row's place) * stride +
    (column's place)`` -- ``base`` the sample's first lane in the row, ``stride`` its live columns at the block's
    exit --; ``by_slot``, at the pair's slot from ``p`` (a block kept once, or a sample's dense one), -inf where the
    automaton does not join them."""
    m = xm[:, None] & ym[None, :]
    if T:
        ri, ci, pr, pc = y[None, :], x[:, None], yp[None, :], xp[:, None]
    else:
        ri, ci, pr, pc = x[:, None], y[None, :], xp[:, None], yp[None, :]
    if by_slot:
        sl = _slot(iv_table, positions, ri, ci, ncols, m, False).to(tl.int64)
    else:
        sl = base + pr.to(tl.int64) * stride + pc
    return tl.load(p + sl, mask = m & (sl >= 0), other = float("-inf"))


@triton.jit
def _rect_product(ap, a0, a_stride, a_slot, a_pos, a_ncols, bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table, rid,
                  pr, rm, cid, pc, cm, columns, kstart, ck, A_T: tl.constexpr, B_T: tl.constexpr, TM: tl.constexpr,
                  TN: tl.constexpr, TK: tl.constexpr, PRECISION: tl.constexpr):
    """``log(sum_k exp(A[r, k] + B[k, c]))`` for one output tile -- rows ``rid`` (places ``pr``), columns ``cid``
    (places ``pc``) -- over the sample's ``ck`` live columns at the shared boundary (``columns[kstart]`` on), ``A``
    and ``B`` read by :func:`_rect_tile` (``A_T``, ``B_T``: stored transposed), shifted by every row's and column's
    running maximum as :func:`_block_block_tile` does. Returns the tile and its shifted sums (see :data:`EXACT_BELOW`)."""
    mA = tl.full([TM], float("-inf"), dtype = tl.float32)
    mB = tl.full([TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for k0 in range(0, ck, TK):
        pk = k0 + tl.arange(0, TK)
        km = pk < ck
        kid = tl.load(columns + kstart + pk, mask = km, other = 0)
        at = _rect_tile(ap, a0, a_stride, a_slot, a_pos, a_ncols, iv_table, rid, kid, pr, pk, rm, km, A_T)
        bt = _rect_tile(bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table, kid, cid, pk, pc, km, cm, B_T)
        nA = tl.maximum(mA, tl.max(at, axis = 1))
        nB = tl.maximum(mB, tl.max(bt, axis = 0))
        sA = tl.where(nA == float("-inf"), 0.0, nA)
        sB = tl.where(nB == float("-inf"), 0.0, nB)
        fA = tl.where(nA == float("-inf"), 1.0, tl.exp(mA - sA))
        fB = tl.where(nB == float("-inf"), 1.0, tl.exp(mB - sB))
        acc = acc * fA[:, None] * fB[None, :] + tl.dot(_tf32_operand(tl.exp(at - sA[:, None]), PRECISION),
                                                        _tf32_operand(tl.exp(bt - sB[None, :]), PRECISION),
                                                        input_precision = PRECISION)
        mA, mB = nA, nB
    sA = tl.where(mA == float("-inf"), 0.0, mA)
    sB = tl.where(mB == float("-inf"), 0.0, mB)
    return tl.log(acc) + sA[:, None] + sB[None, :], acc


@triton.jit
def _rect_product_exact(ap, a0, a_stride, a_slot, a_pos, a_ncols, bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table,
                        rid, pr, rm, cid, pc, cm, columns, kstart, ck, A_T: tl.constexpr, B_T: tl.constexpr,
                        TM: tl.constexpr, TN: tl.constexpr):
    """:func:`_rect_product` by an online log-sum-exp per entry, one shared column at a time: exact whatever the
    operands' dynamic range."""
    m = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    one = tl.zeros([1], dtype = tl.int32)
    for k in range(0, ck):
        kid = one + tl.load(columns + kstart + k)
        at = _rect_tile(ap, a0, a_stride, a_slot, a_pos, a_ncols, iv_table, rid, kid, pr, one + k, rm, one == 0, A_T)
        bt = _rect_tile(bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table, kid, cid, one + k, pc, one == 0, cm, B_T)
        m, acc = _lse_step(m, acc, at + bt)                                  # [TM, 1] + [1, TN]
    return _lse_value(m, acc)


def compact_tiles(rows: int, cols: int):
    """``(TM, TN, TK, warps)`` of a compact ``block @ block`` launch whose samples have at most ``rows`` and ``cols``
    live columns at its outputs' ends (see :func:`_block_block_compact_kernel`)."""
    return (32, 32, 16, 4) if max(rows, cols) >= 32 else (16, 16, 16, 2)


@triton_jit
def _block_block_compact_kernel(element_mars, node_mars, temp, width, iv_of, iv_info, iv_table, coff, ccount, cstart,
                                ccolumns, steps, sel, reg_offset, reg_width, reg_first, prior_mars, prior_offset,
                                prior_width, reg_prior, out_first, out_width, temp_width, B, n, SPLIT, GROUP,
                                exact_below, pid0, L_KIND: tl.constexpr, R_KIND: tl.constexpr,
                                OUT_TEMP: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                                PRECISION: tl.constexpr, INDEXED: tl.constexpr):
    """:func:`_block_block_kernel` on compact values (:mod:`.compact`): one program per (step, sample, group of
    ``GROUP`` tiles of the sample's rectangle of live columns at the output's ends), each tile a log-space matrix
    product over the sample's live columns at the shared boundary (:func:`_rect_product`), the operands' values by
    their places -- by slot, kept once --, -inf at the pairs the automaton does not join. The entries the shifts flush
    are recorded entry by entry and their tiles computed again exactly, as there."""
    pid = pid0 + tl.program_id(0)
    g = pid % SPLIT
    s = _child(sel, pid // SPLIT // B, INDEXED)
    b = (pid // SPLIT % B).to(tl.int64)
    t0 = tl.load(steps + s * 8 + 5)
    t1 = tl.load(steps + s * 8 + 6)
    t2 = tl.load(steps + s * 8 + 7)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    _, lpos, _ = _interval(iv_of, iv_info, t0, t1, n)
    _, rpos, _ = _interval(iv_of, iv_info, t1, t2, n)
    _, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    lp, l_own = _block_ptr(tl.load(steps + s * 8 + 1), tl.load(steps + s * 8 + 2), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, L_KIND)
    rp, r_own = _block_ptr(tl.load(steps + s * 8 + 3), tl.load(steps + s * 8 + 4), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, R_KIND)
    row = b * (n + 1)
    c0 = tl.load(ccount + row + t0)                                          # the sample's live columns at t0, t1, t2
    c1 = tl.load(ccount + row + t1)
    c2 = tl.load(ccount + row + t2)
    s0 = tl.load(cstart + row + t0)
    s1 = tl.load(cstart + row + t1)
    s2 = tl.load(cstart + row + t2)
    l0 = tl.load(coff + tl.load(iv_of + t0 * (n + 1) + t1).to(tl.int64) * (B + 1) + b)
    r0 = tl.load(coff + tl.load(iv_of + t1 * (n + 1) + t2).to(tl.int64) * (B + 1) + b)
    op = _out_ptr(tl.load(steps + s * 8 + 0), element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + \
        tl.load(coff + tl.load(iv_of + t0 * (n + 1) + t2).to(tl.int64) * (B + 1) + b)
    NJ = tl.cdiv(c2, TN)
    bad = tl.zeros([TM, TN], dtype = tl.int32)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(c0, TM) * NJ)):
        pr = (t // NJ) * TM + tl.arange(0, TM)
        pc = (t % NJ) * TN + tl.arange(0, TN)
        rm, cm = pr < c0, pc < c2
        rid = tl.load(ccolumns + s0 + pr, mask = rm, other = 0)
        cid = tl.load(ccolumns + s2 + pc, mask = cm, other = 0)
        om = rm[:, None] & cm[None, :]
        valid = om & (_slot(iv_table, opos, rid[:, None], cid[None, :], X, om, False) >= 0)
        out, acc = _rect_product(lp, l0, c1, ~l_own, lpos, M, rp, r0, c2, ~r_own, rpos, X, iv_table, rid, pr, rm, cid,
                                 pc, cm, ccolumns, s1, c1, False, False, TM, TN, TK, PRECISION)
        bad = bad | (valid & (acc < exact_below)).to(tl.int32)
        tl.store(op + pr[:, None].to(tl.int64) * c2 + pc[None, :], tl.where(valid, out, float("-inf")), mask = om)
    if tl.max(bad) > 0:                                                      # an entry flushed (rare): its tiles
        for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(c0, TM) * NJ)):   # again, exactly where so
            pr = (t // NJ) * TM + tl.arange(0, TM)
            pc = (t % NJ) * TN + tl.arange(0, TN)
            rm, cm = pr < c0, pc < c2
            rid = tl.load(ccolumns + s0 + pr, mask = rm, other = 0)
            cid = tl.load(ccolumns + s2 + pc, mask = cm, other = 0)
            om = rm[:, None] & cm[None, :]
            valid = om & (_slot(iv_table, opos, rid[:, None], cid[None, :], X, om, False) >= 0)
            fast, acc = _rect_product(lp, l0, c1, ~l_own, lpos, M, rp, r0, c2, ~r_own, rpos, X, iv_table, rid, pr, rm,
                                      cid, pc, cm, ccolumns, s1, c1, False, False, TM, TN, TK, PRECISION)
            if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                out = _rect_product_exact(lp, l0, c1, ~l_own, lpos, M, rp, r0, c2, ~r_own, rpos, X, iv_table, rid, pr,
                                          rm, cid, pc, cm, ccolumns, s1, c1, False, False, TM, TN)
                tl.store(op + pr[:, None].to(tl.int64) * c2 + pc[None, :], tl.where(valid, out, float("-inf")),
                         mask = om)


@triton_jit
def _copy_kernel(element_mars, node_mars, temp, iv_of, iv_info, coff, steps, sel, reg_offset, reg_width, reg_first,
                 prior_mars, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n, pid0,
                 L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, COMPACT: tl.constexpr, INDEXED: tl.constexpr,
                 TILE: tl.constexpr):
    """One program per step and tile of its output's lanes (``COMPACT``: its interval's, see :mod:`.compact`; the
    block's are the same)."""
    s = _child(sel, tl.program_id(0), INDEXED)
    pid = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 5 + 0)
    t0 = tl.load(steps + s * 5 + 3)
    t2 = tl.load(steps + s * 5 + 4)
    S, _, _ = _interval(iv_of, iv_info, t0, t2, n)
    _, ncols = _out_count(iv_of, coff, t0, t2, B, S, n, COMPACT)
    if pid * TILE < ncols:
        cols = pid * TILE + tl.arange(0, TILE)
        m = cols < ncols
        # the block's scope is the output's: in prior_mars only when this is a batch of one (sample stride moot)
        lp, l_own = _block_ptr(tl.load(steps + s * 5 + 1), tl.load(steps + s * 5 + 2), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               L_KIND)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + cols, tl.load(lp + cols, mask = m), mask = m)
