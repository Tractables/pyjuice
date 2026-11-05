"""
The top-down ``block @ block`` kernels, sparse and compact.
"""

from __future__ import annotations

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit

from ..common import _block_ptr, _child, _exit_width, _interval, _logaddexp_tile, _lse_step, _slot, _tf32_operand
from ..prod.block_block import _chunk_needed, _rect_product, _rect_product_exact, _tile_load
from ..prod.launches import _lse_value
from .launches import _flows_ptr, _once, _out_flows_ptr, _store_flows


@triton.jit
def _tile_load_t(ptr, iv_table, positions, rows, cols, ncols, rm, cm):
    """The ``[len(rows), len(cols)]`` tile whose entry ``(r, c)`` is ``S[c, r]`` of a block ``S`` with ``ncols``
    columns: the transposed operand, read without a transposition in registers."""
    mask = rm[:, None] & cm[None, :]
    sl = _slot(iv_table, positions, cols[None, :], rows[:, None], ncols, mask, False)
    return tl.load(ptr + sl, mask = mask & (sl >= 0), other = float("-inf"))


@triton.jit
def _top_down_tile(ap, bp, iv_table, apos, bpos, mb, live_k, r, c, tr, tc, Rn, K, Cn, TILES_I, TILES_J, WORDS,
                  TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                  TRANS_A: tl.constexpr, TRANS_B: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr,
                  PRECISION: tl.constexpr):
    """``log(sum_k exp(A[r, k] + B[k, c]))`` for one output tile, as :func:`~.kernels.prod.block_block._block_block_tile`: ``A`` is
    ``[Rn, K]`` (stored ``[K, Rn]`` when ``TRANS_A``), ``B`` is ``[K, Cn]`` (stored ``[Cn, K]`` when ``TRANS_B``);
    ``live_k``: the shared boundary's liveness (pruned). Returns the tile and its shifted sums."""
    rm, cm = r < Rn, c < Cn
    mA = tl.full([TM], float("-inf"), dtype = tl.float32)
    mB = tl.full([TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for ch in range(0, tl.cdiv(K, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, tr, tc, ch, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            ks = ch * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_k + ks, mask = ks < K, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            k = ch * TK + tl.arange(0, TK)
            km = k < K
            if TRANS_A:
                at = _tile_load_t(ap, iv_table, apos, r, k, Rn, rm, km)
            else:
                at = _tile_load(ap, iv_table, apos, r, k, K, rm, km)
            if TRANS_B:
                bt = _tile_load_t(bp, iv_table, bpos, k, c, K, km, cm)
            else:
                bt = _tile_load(bp, iv_table, bpos, k, c, Cn, km, cm)
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
def _top_down_tile_exact(ap, bp, iv_table, apos, bpos, mb, live_k, r, c, tr, tc, Rn, K, Cn, TILES_I, TILES_J, WORDS,
                        TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                        TRANS_A: tl.constexpr, TRANS_B: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr):
    """:func:`_top_down_tile` by an online log-sum-exp per entry over the shared columns one at a time (see
    :func:`~.kernels.prod.block_block._block_block_tile_exact`)."""
    rm, cm = r < Rn, c < Cn
    m = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for ch in range(0, tl.cdiv(K, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, tr, tc, ch, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            ks = ch * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_k + ks, mask = ks < K, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            for kk in range(0, TK):
                k = ch * TK + kk
                ok = k < K
                if PRUNE:
                    ok = ok & (tl.load(live_k + k, mask = k < K, other = 0) != 0)
                if ok:
                    if TRANS_A:
                        sa = _slot(iv_table, apos, k, r, Rn, rm, False)
                    else:
                        sa = _slot(iv_table, apos, r, k, K, rm, False)
                    av = tl.load(ap + sa, mask = rm & (sa >= 0), other = float("-inf"))
                    if TRANS_B:
                        sb = _slot(iv_table, bpos, c, k, K, cm, False)
                    else:
                        sb = _slot(iv_table, bpos, k, c, Cn, cm, False)
                    bv = tl.load(bp + sb, mask = cm & (sb >= 0), other = float("-inf"))
                    m, acc = _lse_step(m, acc, av[:, None] + bv[None, :])
    return _lse_value(m, acc)


@triton.jit
def _parent_tile(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of, iv_info, iv_table,
                 once_iv, lrow, skip_bits, reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior,
                 out_first, out_width, temp_width, tflow_width, b, n, W, r, cc, tr, tc, Rn, Cn, valid, exact_below,
                 TILES_I, TILES_J, WORDS,
                 O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, SIDE_L: tl.constexpr, TM: tl.constexpr,
                 TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr, SKIP: tl.constexpr,
                 PRUNE: tl.constexpr, PRECISION: tl.constexpr, RECHECK: tl.constexpr):
    """Parent ``p``'s contribution to a child's output tile on the parent's own intervals -- the child's interval is
    every parent's, the other operand's and the product's are not (a sum shared by products reaching different
    boundaries) -- as ``(out, shifted sums, fast)``: ``RECHECK``, ``out`` is computed again exactly where the shifts
    flush a valid entry and ``fast`` is the shifted product's value; else both are the shifted product's."""
    t0 = tl.load(parents + p * 9 + 5)
    t1 = tl.load(parents + p * 9 + 6)
    t2 = tl.load(parents + p * 9 + 7)
    OS, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    op = _out_flows_ptr(tl.load(parents + p * 9 + 0), element_flows, temp_flows, out_first, out_width, tflow_width,
                        OUT_TEMP) + tl.where(_once(once_iv, iv_of, t0, t2, n), 0, b * OS)   # kept once: stride 0
    mb = skip_bits + tl.load(parents + p * 9 + 8).to(tl.int64) * TILES_I * TILES_J * WORDS
    if SIDE_L:                                                               # O_out @ R^T, shared: the exit columns
        K = _exit_width(width, t2, n)
        XS, xpos, _ = _interval(iv_of, iv_info, t1, t2, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 3), tl.load(parents + p * 9 + 4), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        xp = xp + tl.where(x_own, b * XS, 0)
        out, acc = _top_down_tile(op, xp, iv_table, opos, xpos, mb, lrow + t2 * W, r, cc, tr, tc, Rn, K, Cn, TILES_I,
                                  TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, False, True, SKIP, PRUNE, PRECISION)
        fast = out
        if RECHECK:
            if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                out = _top_down_tile_exact(op, xp, iv_table, opos, xpos, mb, lrow + t2 * W, r, cc, tr, tc, Rn, K, Cn,
                                           TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, False, True, SKIP, PRUNE)
    else:                                                                    # L^T @ O_out, shared: the entry columns
        K = tl.load(width + t0)
        XS, xpos, _ = _interval(iv_of, iv_info, t0, t1, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 1), tl.load(parents + p * 9 + 2), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        xp = xp + tl.where(x_own, b * XS, 0)
        out, acc = _top_down_tile(xp, op, iv_table, xpos, opos, mb, lrow + t0 * W, r, cc, tr, tc, Rn, K, Cn, TILES_I,
                                  TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, True, False, SKIP, PRUNE, PRECISION)
        fast = out
        if RECHECK:
            if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                out = _top_down_tile_exact(xp, op, iv_table, xpos, opos, mb, lrow + t0 * W, r, cc, tr, tc, Rn, K, Cn,
                                           TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, True, False, SKIP, PRUNE)
    return out, acc, fast


@triton_jit
def _block_block_top_down_kernel(element_flows, temp_flows, node_flows, node_mars, prior_mars, temp, width, iv_of,
                                iv_info, iv_table, once_iv, live, children, children_idx, ptr, parents, skip_bits,
                                reg_offset, reg_width,
                                reg_first, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width,
                                tflow_width, B, n, W, TILES_I, TILES_J, WORDS, SPLIT, GROUP, exact_below, pid0,
                                C_KIND: tl.constexpr,
                                O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, SIDE_L: tl.constexpr, TM: tl.constexpr,
                                TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                                SKIP: tl.constexpr, PRUNE: tl.constexpr, EXACT: tl.constexpr, PRECISION: tl.constexpr,
                                INDEXED: tl.constexpr):
    """One program per (child, sample, group of ``GROUP`` output tiles): the child's top-down tile is the log-sum over
    its parents of ``O_out @ R^T`` (``SIDE_L``: the child is the left operand ``L`` over ``[t0, t1)``, the other
    operand ``R`` over ``[t1, t2)``, the shared columns the exit ``j``) or ``L^T @ O_out`` (the child is ``R`` over
    ``[t1, t2)``, the shared columns the entry ``i``), each parent on its own triple (:func:`_parent_tile`). ``O_KIND``:
    the other operand's kind (a sum's block in ``node_mars`` or ``prior_mars``, or a fold's scratch row). ``EXACT``: as
    :func:`~.kernels.prod.block_block._block_block_kernel`, per (parent, tile)."""
    pid = pid0 + tl.program_id(0)
    g = pid % SPLIT
    c = _child(children_idx, pid // SPLIT // B, INDEXED)
    b = (pid // SPLIT % B).to(tl.int64)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c * 4 + 2)
    p0 = tl.load(ptr + c)
    p1 = tl.load(ptr + c + 1)
    if SIDE_L:                                                               # output (i, q): the child [t0, t1)
        ra = tl.load(parents + p0 * 9 + 5)
        rb = tl.load(parents + p0 * 9 + 6)
        Rn = tl.load(width + ra)
        Cn = tl.load(width + rb)
    else:                                                                    # output (q, j): the child [t1, t2)
        ra = tl.load(parents + p0 * 9 + 6)
        rb = tl.load(parents + p0 * 9 + 7)
        Rn = tl.load(width + ra)
        Cn = _exit_width(width, rb, n)
    CS, cpos, _ = _interval(iv_of, iv_info, ra, rb, n)
    lrow = live + b * (n + 1) * W
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, tflow_width, C_KIND) + b * CS
    NJ = tl.cdiv(Cn, TN)
    if EXACT:
        bad = tl.zeros([TM, TN], dtype = tl.int32)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(Rn, TM) * NJ)):
        tr = t // NJ
        tc = t % NJ
        r = tr * TM + tl.arange(0, TM)
        cc = tc * TN + tl.arange(0, TN)
        om = (r < Rn)[:, None] & (cc < Cn)[None, :]
        sl = _slot(iv_table, cpos, r[:, None], cc[None, :], Cn, om, False)
        valid = om & (sl >= 0)
        if PRUNE:
            lr = tl.load(lrow + ra * W + r, mask = r < Rn, other = 0) != 0
            lc = tl.load(lrow + rb * W + cc, mask = cc < Cn, other = 0) != 0
            valid = valid & lr[:, None] & lc[None, :]
        tot = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
        for p in range(p0, p1):
            out, acc, fast = _parent_tile(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width,
                                          iv_of, iv_info, iv_table, once_iv, lrow, skip_bits, reg_offset, reg_width,
                                          reg_first, prior_offset, prior_width, reg_prior, out_first, out_width,
                                          temp_width, tflow_width, b, n, W, r, cc, tr, tc, Rn, Cn, valid, exact_below,
                                          TILES_I, TILES_J, WORDS, O_KIND, OUT_TEMP, SIDE_L, TM, TN, TK, SUB_I, SUB_J,
                                          SKIP, PRUNE, PRECISION, False)
            if EXACT:
                bad = bad | (valid & (acc < exact_below)).to(tl.int32)
            tot = _logaddexp_tile(tot, out)
        _store_flows(cp, sl, tl.where(valid, tot, float("-inf")), om & (sl >= 0), add)
    if EXACT:
        # an entry flushed (rare): the program's tiles again, exactly where so. A child written here is written again;
        # one an earlier launch wrote holds the earlier value log-added to the shifted products' sum, which the exact
        # sum replaces in it (the shifted one is computed again, bit for bit)
        if tl.max(bad) > 0:
            for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(Rn, TM) * NJ)):
                tr = t // NJ
                tc = t % NJ
                r = tr * TM + tl.arange(0, TM)
                cc = tc * TN + tl.arange(0, TN)
                om = (r < Rn)[:, None] & (cc < Cn)[None, :]
                sl = _slot(iv_table, cpos, r[:, None], cc[None, :], Cn, om, False)
                valid = om & (sl >= 0)
                if PRUNE:
                    lr = tl.load(lrow + ra * W + r, mask = r < Rn, other = 0) != 0
                    lc = tl.load(lrow + rb * W + cc, mask = cc < Cn, other = 0) != 0
                    valid = valid & lr[:, None] & lc[None, :]
                tot = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
                tot_fast = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
                for p in range(p0, p1):
                    out, acc, fast = _parent_tile(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp,
                                                  width, iv_of, iv_info, iv_table, once_iv, lrow, skip_bits,
                                                  reg_offset, reg_width, reg_first, prior_offset, prior_width,
                                                  reg_prior, out_first, out_width, temp_width, tflow_width, b, n, W, r,
                                                  cc, tr, tc, Rn, Cn, valid,
                                                  exact_below, TILES_I, TILES_J, WORDS, O_KIND, OUT_TEMP, SIDE_L, TM,
                                                  TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION, True)
                    tot = _logaddexp_tile(tot, out)
                    tot_fast = _logaddexp_tile(tot_fast, fast)
                tot = tl.where(valid, tot, float("-inf"))
                tot_fast = tl.where(valid, tot_fast, float("-inf"))
                m = om & (sl >= 0)
                if add:                       # stored = log(earlier + exp(tot_fast)): swap tot_fast for tot in it
                    old = tl.load(cp + sl, mask = m, other = float("-inf"))
                    s = tl.where(old == float("-inf"), 0.0, old)
                    v = tl.maximum(1.0 - tl.exp(tot_fast - s) + tl.exp(tot - s), tl.exp(tot - s))
                    tot = tl.where(old == float("-inf"), tot, tl.log(v) + s)
                tl.store(cp + sl, tot, mask = m)


@triton.jit
def _bb_compact_parent(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of, iv_info,
                       iv_table, once_iv, coff, ccount, cstart, ccolumns, mars_offset, mars_width, reg_first,
                       prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, tflow_width, b, B, n,
                       rid, cid, pr, pc, rm, cm, valid, exact_below, O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                       SIDE_L: tl.constexpr, MARS: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr,
                       TK: tl.constexpr, PRECISION: tl.constexpr):
    """Parent ``p``'s contribution to a compact child's output tile (rows ``rid``, columns ``cid``; ``pr``, ``pc``:
    their places among the sample's live columns), a log-space matrix product over the sample's live columns at the
    shared boundary -- the exit ``t2`` (``SIDE_L``: ``O_out @ R^T``) or the entry ``t0`` (``L^T @ O_out``) -- of
    :func:`~.kernels.prod.block_block._rect_product`, computed again exactly (one shared column at a time) where its shifts flush a
    ``valid`` entry. The parent's values are compact, or kept once (by slot); the other operand's inside values
    compact too (``MARS``), or the sample's dense block, or kept once (by slot)."""
    t0 = tl.load(parents + p * 9 + 5)
    t1 = tl.load(parents + p * 9 + 6)
    t2 = tl.load(parents + p * 9 + 7)
    _, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    once = _once(once_iv, iv_of, t0, t2, n)
    op = _out_flows_ptr(tl.load(parents + p * 9 + 0), element_flows, temp_flows, out_first, out_width, tflow_width,
                        OUT_TEMP)
    o0 = tl.load(coff + tl.load(iv_of + t0 * (n + 1) + t2).to(tl.int64) * (B + 1) + b)
    ostride = tl.load(ccount + b * (n + 1) + t2)                             # the parent's exit columns live
    X_o = _exit_width(width, t2, n)
    if SIDE_L:                                                               # the other operand R over [t1, t2)
        ks = t2
        X_x = X_o
        XS, xpos, _ = _interval(iv_of, iv_info, t1, t2, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 3), tl.load(parents + p * 9 + 4), node_mars, prior_mars, temp,
                               mars_offset, mars_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        kx = tl.load(iv_of + t1 * (n + 1) + t2)
        xstride = ostride                                                    # its exit columns live
    else:                                                                    # ... or L over [t0, t1)
        ks = t0
        X_x = tl.load(width + t1)
        XS, xpos, _ = _interval(iv_of, iv_info, t0, t1, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 1), tl.load(parents + p * 9 + 2), node_mars, prior_mars, temp,
                               mars_offset, mars_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        kx = tl.load(iv_of + t0 * (n + 1) + t1)
        xstride = tl.load(ccount + b * (n + 1) + t1)
    if MARS:
        x0 = tl.load(coff + kx.to(tl.int64) * (B + 1) + b)
        x_slot = ~x_own
    else:
        xp = xp + tl.where(x_own, b * XS, 0)
        x0 = 0
        x_slot = True
    krow = b * (n + 1) + ks
    ck = tl.load(ccount + krow)
    k0p = tl.load(cstart + krow)
    if SIDE_L:                                       # A[r, k] = O_out[r, k], B[k, c] = R[c, k]
        out, acc = _rect_product(op, o0, ostride, once, opos, X_o, xp, x0, xstride, x_slot, xpos, X_x, iv_table, rid,
                                 pr, rm, cid, pc, cm, ccolumns, k0p, ck, False, True, TM, TN, TK, PRECISION)
        if tl.min(tl.where(valid, acc, 1.0)) < exact_below:                  # flushed: one shared column at a time
            out = _rect_product_exact(op, o0, ostride, once, opos, X_o, xp, x0, xstride, x_slot, xpos, X_x, iv_table,
                                      rid, pr, rm, cid, pc, cm, ccolumns, k0p, ck, False, True, TM, TN)
    else:                                            # A[r, k] = L[k, r], B[k, c] = O_out[k, c]
        out, acc = _rect_product(xp, x0, xstride, x_slot, xpos, X_x, op, o0, ostride, once, opos, X_o, iv_table, rid,
                                 pr, rm, cid, pc, cm, ccolumns, k0p, ck, True, False, TM, TN, TK, PRECISION)
        if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
            out = _rect_product_exact(xp, x0, xstride, x_slot, xpos, X_x, op, o0, ostride, once, opos, X_o, iv_table,
                                      rid, pr, rm, cid, pc, cm, ccolumns, k0p, ck, True, False, TM, TN)
    return out


@triton_jit
def _block_block_top_down_compact_kernel(element_flows, temp_flows, node_flows, node_mars, prior_mars, temp, width,
                                        iv_of, iv_info, iv_table, once_iv, coff, ccount, cstart, ccolumns, children,
                                        children_idx, ptr, parents, reg_offset, reg_width, reg_first, mars_offset,
                                        mars_width, prior_offset, prior_width, reg_prior, out_first, out_width,
                                        temp_width, tflow_width, B, n, SPLIT, GROUP, exact_below, pid0,
                                        C_KIND: tl.constexpr, O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                                        SIDE_L: tl.constexpr, MARS: tl.constexpr, TM: tl.constexpr,
                                        TN: tl.constexpr, TK: tl.constexpr, PRECISION: tl.constexpr,
                                        INDEXED: tl.constexpr):
    """:func:`_block_block_top_down_kernel` on compact values (:mod:`.compact`): one program per (child, sample, group
    of ``GROUP`` tiles of the sample's rectangle), each tile the log-sum over the child's parents of a log-space matrix
    product over the sample's live columns at the shared boundary (:func:`_bb_compact_parent`), -inf at the pairs the
    automaton does not join; the inside operands (``mars_offset``, ``mars_width``; ``temp_width``) compact too
    (``MARS``) or dense."""
    pid = pid0 + tl.program_id(0)
    g = pid % SPLIT
    c = _child(children_idx, pid // SPLIT // B, INDEXED)
    b = (pid // SPLIT % B).to(tl.int64)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c * 4 + 2)
    p0 = tl.load(ptr + c)
    p1 = tl.load(ptr + c + 1)
    if SIDE_L:                                                               # output (i, q): the child [t0, t1)
        ra = tl.load(parents + p0 * 9 + 5)
        rb = tl.load(parents + p0 * 9 + 6)
    else:                                                                    # output (q, j): the child [t1, t2)
        ra = tl.load(parents + p0 * 9 + 6)
        rb = tl.load(parents + p0 * 9 + 7)
    Cn = _exit_width(width, rb, n)
    _, cpos, _ = _interval(iv_of, iv_info, ra, rb, n)
    kc = tl.load(iv_of + ra * (n + 1) + rb)
    cr = tl.load(ccount + b * (n + 1) + ra)                                  # the sample's rectangle
    cc_ = tl.load(ccount + b * (n + 1) + rb)
    r0 = tl.load(cstart + b * (n + 1) + ra)
    c0 = tl.load(cstart + b * (n + 1) + rb)
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, tflow_width, C_KIND) + \
        tl.load(coff + kc.to(tl.int64) * (B + 1) + b)
    NJ = tl.cdiv(cc_, TN)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(cr, TM) * NJ)):
        pr = (t // NJ) * TM + tl.arange(0, TM)
        pc = (t % NJ) * TN + tl.arange(0, TN)
        rm, cm = pr < cr, pc < cc_
        rid = tl.load(ccolumns + r0 + pr, mask = rm, other = 0)
        cid = tl.load(ccolumns + c0 + pc, mask = cm, other = 0)
        om = rm[:, None] & cm[None, :]
        valid = om & (_slot(iv_table, cpos, rid[:, None], cid[None, :], Cn, om, False) >= 0)
        tot = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
        for p in range(p0, p1):
            out = _bb_compact_parent(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of,
                                     iv_info, iv_table, once_iv, coff, ccount, cstart, ccolumns, mars_offset,
                                     mars_width, reg_first, prior_offset, prior_width, reg_prior, out_first, out_width,
                                     temp_width, tflow_width, b, B, n, rid, cid, pr, pc, rm, cm, valid, exact_below,
                                     O_KIND, OUT_TEMP, SIDE_L, MARS, TM, TN, TK, PRECISION)
            tot = _logaddexp_tile(tot, out)
        _store_flows(cp, pr[:, None].to(tl.int64) * cc_ + pc[None, :], tl.where(valid, tot, float("-inf")), om, add)
