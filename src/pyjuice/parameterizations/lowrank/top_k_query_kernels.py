"""
The conditional's read-out of `LowRank(top_k)` soft-evidence leaves (`top_k_queries.conditional`) by one Triton kernel,
from the kernels' forward state (`top_k_leaf_fused`): a program a (target, sample), the nodes walked a chunk at a time.

In log2 units, as the state holds them -- tab[row(c), i] = log2 theta[i, ids(c)], lz_i = log2 Z_i, lb_i = log2 of the
bulk's tilted sum (the candidates past the head, K = Kl: the rest slot's), tau_i(s) the tilt at slot s, lp2 = lp log2 e:

  * a missing token a sum reads (its counts row: the slots' posterior a node, then its flow f):
        P[c] = sum_i post[i, c] f_i                                              c < K (a slot of its own)
        P[c] = sum_i post[i, K] f_i 2^(tab[row(c), i] + lp2(c) + tau_i(K) - lb_i)    c >= K (the rest's share)
  * elsewhere (the node flows F):
        P[c] = sum_i 2^(tab[row(c), i] + lp2(c) + tau_i(min(c, K)) + log2 F_i - lz_i)

every exponent <= 0 (a probability times a flow); P normalized over the candidates and added at their tokens into the
dense output (a token twice among them: its probabilities add).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _tk_conditional_kernel(out_ptr, s_ob, s_or, nf_ptr, lz_ptr, lb_ptr, tauk_ptr, tauf_ptr, tab_ptr, crow_ptr, ids_ptr,
                           lp_ptr, emap_ptr, cnt_ptr, wrow_ptr, tg_t_ptr, tg_r_ptr, tg_cnt_ptr, tg_tf_ptr, nv_tab_ptr,
                           nv_lvid_ptr, nv_row0_ptr, B, Bx, M, C, K, S, nrows, nvars, ev_mapped, x_mapped, CS,
                           BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr, DENSE: tl.constexpr):
    pid = tl.program_id(0)
    g = pid // B                                                       # (the target)
    b = pid % B
    t = tl.load(tg_t_ptr + g)
    r = tl.load(tg_r_ptr + g)
    has_cnt = tl.load(tg_cnt_ptr + g)
    tf = tl.load(tg_tf_ptr + g)
    tb = tl.load(nv_tab_ptr + t).to(tl.int64)
    lvid = tl.load(nv_lvid_ptr + t).to(tl.int64)
    row0 = tl.load(nv_row0_ptr + t).to(tl.int64)
    xr = b
    if x_mapped:
        xr = tl.load(emap_ptr + b)                                     # (the row of tensors the sample reads)
    er = b
    if ev_mapped:
        er = xr                                                        # (the evidence of the rows' batch)
    wr = -1
    if has_cnt != 0:
        wr = tl.load(wrow_ptr + t * B + b)

    offs_c = tl.arange(0, BLOCK_C)
    cm = offs_c < C
    ev_off = (er.to(tl.int64) * nvars + lvid) * C
    tok = tl.load(ids_ptr + ev_off + offs_c, mask = cm, other = 0)
    lp2 = tl.load(lp_ptr + ev_off + offs_c, mask = cm, other = -float("inf")) * 1.4426950408889634
    crw = tl.load(crow_ptr + tok, mask = cm, other = 0).to(tl.int64)
    trow = tab_ptr + (tb * nrows + crw) * M                            # [BLOCK_C] each candidate's table row
    sl = tl.minimum(offs_c, S - 1)
    tmb = (t.to(tl.int64) * B + b) * M                                 # (lz, lb at [t, b, :])
    acc = tl.zeros([BLOCK_C], dtype = tl.float32)

    if wr >= 0:
        crow_base = cnt_ptr + wr.to(tl.int64) * M * CS
        tkb = tauk_ptr + (t.to(tl.int64) * Bx + xr) * M
        rest = cm & (offs_c >= K)
        own = cm & (offs_c < K)
        for m0 in range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            f = tl.load(crow_base + offs_m * CS + K + 1)               # [BLOCK_M] the node's flow
            pK = tl.load(crow_base + offs_m * CS + K)                  # its rest slot's posterior
            lbm = tl.load(lb_ptr + tmb + offs_m)
            tkm = tl.load(tkb + offs_m)
            post = tl.load(crow_base + offs_m[None, :] * CS + offs_c[:, None], mask = own[:, None], other = 0.0)
            acc += tl.sum(post * f[None, :], axis = 1)
            w = tl.where(lbm > -float("inf"), pK * f, 0.0)
            th = tl.load(trow[:, None] + offs_m[None, :], mask = rest[:, None], other = -float("inf"))
            x = th + lp2[:, None] + (tkm - lbm)[None, :]
            x = tl.where(rest[:, None] & (w[None, :] > 0.0), x, -float("inf"))
            acc += tl.sum(tl.exp2(x) * w[None, :], axis = 1)
    else:
        tfb = tauf_ptr + ((tf.to(tl.int64) * Bx + xr) * S + sl.to(tl.int64)) * M      # [BLOCK_C] each slot's row
        for m0 in range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            lf2 = tl.load(nf_ptr + (row0 + offs_m) * B + b) * 1.4426950408889634
            lzm = tl.load(lz_ptr + tmb + offs_m)
            th = tl.load(trow[:, None] + offs_m[None, :], mask = cm[:, None], other = -float("inf"))
            tau = tl.load(tfb[:, None] + offs_m[None, :], mask = cm[:, None], other = 0.0)
            x = th + lp2[:, None] + tau + (lf2 - lzm)[None, :]
            acc += tl.sum(tl.exp2(x), axis = 1)

    acc = tl.where(cm, acc, 0.0)
    tot = tl.sum(acc, axis = 0)
    p = acc / tl.maximum(tot, 1e-30)
    if DENSE:                                                          # (at the tokens: one twice, its shares add)
        tl.atomic_add(out_ptr + b.to(tl.int64) * s_ob + r.to(tl.int64) * s_or + tok, p, mask = cm)
    else:                                                              # (at the candidates' places)
        tl.store(out_ptr + b.to(tl.int64) * s_ob + r.to(tl.int64) * s_or + offs_c, p, mask = cm)


def conditional(st: dict, plan: dict, node_flows: torch.Tensor, staged, targets, num_rows: int, num_cats: int,
                over_candidates: bool = False):
    """The dense conditional `[B, num_rows, num_cats]` float32 of the targets `targets` -- `(node vector index, its
    output row)` -- or (`over_candidates`) `[B, num_rows, C]` over each target's candidates, from the leaf layer's
    forward state `st`, its plan, the backward's log-space `node_flows`; `staged` the tilts' tensors (made into their
    full tilt where a flows pair needs it). None where the state cannot serve (the sums' top_k not the leaves')."""
    from .top_k_leaf_fused import _tilts
    B, K, S, M = st["B"], st["K"], st["S"], st["lz"].size(2)
    if K != S - 1 or not targets:
        return None
    dev = node_flows.device
    nodes = plan["nodes"]
    written = st.get("written") or set()
    counts = st["counts"]
    has_cnt = [counts is not None and t in written for t, _ in targets]
    # (the targets with a flows pair -- a sample not among the counts' rows -- need the tilt at every slot)
    need = [t for (t, _), hc in zip(targets, has_cnt) if not hc or st["offs"][t + 1] - st["offs"][t] < B]
    tf_of = {}
    tau = st["tau"]
    Bx = (st["tauK"] if st["tauK"] is not None else tau).size(1)
    if tau is not None:
        tauf, tf_of = tau, {t: t for t in need}
    elif need:
        tauf = torch.empty(len(need), Bx, S, M, dtype = torch.float32, device = dev)
        _tilts([staged[nodes[t]["ns"]] for t in need], tauf)
        tf_of = {t: k for k, t in enumerate(need)}
    else:
        tauf = torch.empty(0, dtype = torch.float32, device = dev)
    tauk = st["tauK"] if st["tauK"] is not None else tau[:, :, S - 1, :].contiguous()
    i32 = lambda xs: torch.tensor(xs, dtype = torch.int32, device = dev)
    tg = i32([[t, r, int(hc), tf_of.get(t, -1)] for (t, r), hc in zip(targets, has_cnt)]).t().contiguous()
    ids, lp, emap = st["ids"], st["lp"], st["emap"]
    C = ids.size(2)
    out = torch.zeros(B, num_rows, C if over_candidates else num_cats, dtype = torch.float32, device = dev)
    nf = node_flows if node_flows.dtype == torch.float32 and node_flows.is_contiguous() else \
        node_flows.float().contiguous()
    cnt = counts if counts is not None else torch.empty(0, dtype = torch.float32, device = dev)
    _tk_conditional_kernel[(len(targets) * B,)](
        out, out.stride(0), out.stride(1), nf, st["lz"], st["lb"], tauk, tauf, st["tab"], st["crow"], ids, lp,
        emap, cnt, st["wrow"], tg[0], tg[1], tg[2], tg[3], plan["nv_tab"], plan["nv_lvid"], plan["nv_row0"],
        B, Bx, M, C, K, S, st["tab"].size(1), ids.size(1), int(ids.size(0) != B), int(emap.numel() > 0),
        cnt.size(2) if cnt.dim() == 3 else 72,
        BLOCK_C = triton.next_power_of_2(C), BLOCK_M = 32, DENSE = not over_candidates, num_warps = 4)
    return out


# ---------------------------------------------------------------------------------------------- sampling

@triton.jit
def _gumbel(seed, offs):
    u = tl.maximum(tl.rand(seed, offs), 1e-30)
    return -tl.log(-tl.log(u))


@triton.jit
def _leaf_log_q(i, xr, er, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, nvars, lvid, C, u_ptr, su0, su1, su2, d_ptr, sd0,
                sd1, sd2, re, Kl, BLOCK_C: tl.constexpr, BLOCK_RE: tl.constexpr):
    """(tokens [BLOCK_C], log P(c | i) [BLOCK_C], -inf past C) of node i of a leaf for one sample: log theta[i, ids(c)]
    + lp(c) + U[i] . D[:, min(c, Kl)], normalized over the candidates; the tilt one [re, C] tile (one round of loads,
    not a rank at a time: a level's latency)."""
    offs_c = tl.arange(0, BLOCK_C)
    cm = offs_c < C
    ev = (er * nvars + lvid) * C
    tok = tl.load(ids_ptr + ev + offs_c, mask = cm, other = 0).to(tl.int64)
    lpc = tl.load(lp_ptr + ev + offs_c, mask = cm, other = -float("inf"))
    spid = tl.load(lspid_ptr + i)
    th = tl.load(lpar_ptr + spid + tok, mask = cm, other = 0.0)
    slc = tl.minimum(offs_c, Kl)
    offs_r = tl.arange(0, BLOCK_RE)
    rm = offs_r < re
    u = tl.load(u_ptr + xr * su0 + i * su1 + offs_r * su2, mask = rm, other = 0.0)
    d = tl.load(d_ptr + xr * sd0 + offs_r[:, None] * sd1 + slc[None, :] * sd2, mask = rm[:, None] & cm[None, :],
                other = 0.0)
    tau = tl.sum(u[:, None] * d, axis = 0)
    logit = tl.where(cm & (th > 0.0), tl.log(th) + lpc + tau, -float("inf"))
    mx = tl.max(logit, axis = 0)
    mx = tl.where(mx > -float("inf"), mx, 0.0)
    lz = tl.log(tl.sum(tl.exp(logit - mx), axis = 0)) + mx
    return tok, logit - lz


@triton.jit
def _draw_node(i, xr, er, eid, ok, cj, pl, seed, rbase, lth_ptr, N, a_ptr, sa0, sa1, sa2, b_ptr, sb0, sb1, sb2, g_ptr,
               sg0, sg1, sg2, r, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, nvars, lvid, C, u_ptr, su0, su1, su2, d_ptr,
               sd0, sd1, sd2, re, Kl, K, COND: tl.constexpr, HAS_PLACE: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_C: tl.constexpr, RB: tl.constexpr, SB: tl.constexpr, BLOCK_RE: tl.constexpr):
    """`(j, c)`: one sample's draw at node i of a `LowRank(top_k)` sum node vector -- its slot s and child j together,
    in proportion to P(s | i) theta(j | i, s) (times exp(c_j), the children's values `cj`, where COND; an observed
    token's slot fixed, `pl` >= -1), then the leaf's candidate c within that slot (an observed token itself). `eid`,
    `ok` [BLOCK_N] the children's element ids and which are; BLOCK_C covers the candidates and the slots (K + 1)."""
    # the leaf's candidates and the slots' log-weights (the sum's K slots one by one, the rest summed)
    tok, lq = _leaf_log_q(i, xr, er, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, nvars, lvid, C, u_ptr, su0, su1, su2,
                          d_ptr, sd0, sd1, sd2, re, Kl, BLOCK_C, BLOCK_RE)
    offs_c = tl.arange(0, BLOCK_C)
    restm = (offs_c >= K) & (offs_c < C)
    lr = tl.where(restm, lq, -float("inf"))
    mr = tl.max(lr, axis = 0)
    mr = tl.where(mr > -float("inf"), mr, 0.0)
    lrest = tl.log(tl.sum(tl.exp(lr - mr), axis = 0)) + mr
    lw = tl.where(offs_c < K, lq, tl.where(offs_c == K, lrest, -float("inf")))
    if HAS_PLACE:
        sobs = tl.minimum(tl.maximum(pl, 0), K)
        lw = tl.where(pl != -2, tl.where(offs_c == sobs, 0.0, -float("inf")), lw)

    # the children's log-weights theta[i, j] exp(A_i . B_j): the ranks RB at a time
    offs_n = tl.arange(0, BLOCK_N)
    base = tl.load(lth_ptr + i * N + offs_n, mask = ok, other = -float("inf"))
    acc = tl.zeros([BLOCK_N], dtype = tl.float32)
    offs_k = tl.arange(0, RB)
    ab = a_ptr + xr * sa0 + i * sa1
    bb = b_ptr + xr * sb0 + offs_n[None, :] * sb2
    for k0 in range(0, r, RB):
        km = (k0 + offs_k) < r
        ak = tl.load(ab + (k0 + offs_k) * sa2, mask = km, other = 0.0)
        bk = tl.load(bb + (k0 + offs_k)[:, None] * sb1, mask = km[:, None] & ok[None, :], other = 0.0)
        acc += tl.sum(ak[:, None] * bk, axis = 0)
    base = tl.where(ok, base + acc, -float("inf"))
    gb = g_ptr + xr * sg0 + offs_n * sg2
    if COND:
        # each slot's log sum_j theta(j | i, s) exp(c_j): its normalizer Z_s = sum_j E_sj w_j and its values' U_s =
        # sum_j E_sj w_j v_j, from the shifted exponentials (w = exp(base - max), v = exp(c - max), E = exp(G - a
        # slot's max): every shift cancels in U / Z or is the same for every slot), SB slots at a time
        mb = tl.max(base, axis = 0)
        w = tl.exp(base - tl.where(mb > -float("inf"), mb, 0.0))
        mc = tl.max(tl.where(ok, cj, -float("inf")), axis = 0)
        wv = w * tl.exp(cj - tl.where(mc > -float("inf"), mc, 0.0))
        offs_sb = tl.arange(0, SB)
        for s0 in range(0, K + 1, SB):
            sm = (s0 + offs_sb) < K + 1
            g = tl.load(g_ptr + xr * sg0 + (s0 + offs_sb)[:, None] * sg1 + offs_n[None, :] * sg2,
                        mask = sm[:, None] & ok[None, :], other = -float("inf"))
            mg = tl.max(g, axis = 1)
            e = tl.exp(g - tl.where(mg > -float("inf"), mg, 0.0)[:, None])
            Z = tl.sum(e * w[None, :], axis = 1)
            U = tl.sum(e * wv[None, :], axis = 1)
            sc = tl.where((Z > 0.0) & (U > 0.0), tl.log(U) - tl.log(Z), -float("inf"))
            hit = offs_c[None, :] == (s0 + offs_sb)[:, None]
            lw = lw + tl.sum(tl.where(hit, sc[:, None], 0.0), axis = 0)
    s_star = tl.argmax(lw + _gumbel(seed, rbase + offs_c), axis = 0)
    z = base + tl.load(gb + s_star * sg1, mask = ok, other = 0.0)
    if COND:
        z = z + cj
    j_star = tl.argmax(tl.where(ok, z, -float("inf")) + _gumbel(seed, rbase + BLOCK_C + offs_n), axis = 0)
    c_rest = tl.argmax(lr + _gumbel(seed, rbase + BLOCK_C + BLOCK_N + offs_c), axis = 0)
    c = tl.where(s_star < K, s_star, c_rest)
    if HAS_PLACE:
        c = tl.where(pl != -2, tl.maximum(pl, 0), c)
    return j_star, c


@triton.jit
def _tk_sample_sum_kernel(ns_ptr, es_ptr, rows_ptr, erows_ptr, B, nid0, M, lth_ptr, N,
                          a_ptr, sa0, sa1, sa2, b_ptr, sb0, sb1, sb2, g_ptr, sg0, sg1, sg2, r, emap_ptr, x_mapped,
                          eid_ptr, em_ptr, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, ev_mapped, nvars, lvid, C,
                          u_ptr, su0, su1, su2, d_ptr, sd0, sd1, sd2, re, Kl, place_ptr, cand_ptr, seed_ptr, K,
                          COND: tl.constexpr, HAS_PLACE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
                          RB: tl.constexpr, SB: tl.constexpr, BLOCK_RE: tl.constexpr):
    """A `LowRank(top_k)` sum node vector's draw (`_draw_node`): a program a (frontier row, sample) whose node is one
    of the vector's (`nid0` .. `nid0 + M`), the children's values (COND) from `element_mars` [elements, B]: the child's
    element id into `element_samples`, the candidate's place into `cand` [B]."""
    pid_r = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    row = tl.load(rows_ptr + pid_r).to(tl.int64)
    erow = tl.load(erows_ptr + pid_r).to(tl.int64)
    nid = tl.load(ns_ptr + row * B + b)
    live = (nid >= nid0) & (nid < nid0 + M)
    i = tl.where(live, nid - nid0, 0).to(tl.int64)
    xr = b
    if x_mapped:
        xr = tl.load(emap_ptr + b).to(tl.int64)
    er = b
    if ev_mapped:
        er = xr
    seed = tl.load(seed_ptr)
    rbase = ((row * B + b) * (2 * BLOCK_C + BLOCK_N)).to(tl.int32)   # (each draw its own stream of uniforms)
    offs_n = tl.arange(0, BLOCK_N)
    eid = tl.load(eid_ptr + offs_n, mask = offs_n < N, other = -1)
    ok = (offs_n < N) & (eid >= 0)
    cj = tl.zeros([BLOCK_N], dtype = tl.float32)
    if COND:
        cj = tl.load(em_ptr + eid * B + b, mask = ok, other = -float("inf"))
    pl = -2
    if HAS_PLACE:
        pl = tl.load(place_ptr + b).to(tl.int32)
    j_star, c = _draw_node(i, xr, er, eid, ok, cj, pl, seed, rbase, lth_ptr, N, a_ptr, sa0, sa1, sa2, b_ptr, sb0, sb1,
                           sb2, g_ptr, sg0, sg1, sg2, r, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, nvars, lvid, C, u_ptr,
                           su0, su1, su2, d_ptr, sd0, sd1, sd2, re, Kl, K, COND, HAS_PLACE, BLOCK_N, BLOCK_C, RB, SB,
                           BLOCK_RE)
    tl.store(es_ptr + erow * B + b, tl.load(eid_ptr + j_star), mask = live)
    tl.store(cand_ptr + b, c.to(tl.int32), mask = live)


#: the walk's per-level tables: `lvi` [L, WALK_NI] int64 -- row, nid0, M, N, r, re, Kl, lvid, K, then the strides of
#: A, B, G, U, D (three each) --; `lvp` [L, WALK_NP] int64 addresses -- A, B, G, U, D, log theta [M, N], the leaf's
#: parameters, its first parameter ids [M], the children's element ids [N], their children's node ids and rows
#: [N, E_P], the leaf's places [B] (conditioning) and its candidates' record [B]
WALK_NI, WALK_NP = 24, 13


@triton.jit
def _tk_walk_kernel(lvi_ptr, lvp_ptr, L, ns_ptr, nm_ptr, B, ids_ptr, lp_ptr, ev_mapped, nvars, C, emap_ptr, x_mapped,
                    seed_ptr, NI: tl.constexpr, NP: tl.constexpr, COND: tl.constexpr, HAS_PLACE: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, RB: tl.constexpr, SB: tl.constexpr,
                    BLOCK_RE: tl.constexpr, E_P: tl.constexpr):
    """A run of `LowRank(top_k)` levels -- a sum layer of one node vector, the product layer below it -- walked top-down
    by ONE program a sample: each level's draw (`_draw_node`; the children's values, COND, their own children's
    `node_mars` summed: no element values made), the candidate into the leaf's record, the level's frontier row
    cleared and the drawn product's children written into theirs (`node_samples` [rows, B], read by the next level --
    other threads: a barrier each handover)."""
    b = tl.program_id(0).to(tl.int64)
    xr = b
    if x_mapped:
        xr = tl.load(emap_ptr + b).to(tl.int64)
    er = b
    if ev_mapped:
        er = xr
    seed = tl.load(seed_ptr)
    offs_n = tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, E_P)
    for l in range(L):
        qi = lvi_ptr + l * NI
        qp = lvp_ptr + l * NP
        row = tl.load(qi)
        nid0 = tl.load(qi + 1)
        M = tl.load(qi + 2)
        N = tl.load(qi + 3)
        r = tl.load(qi + 4)
        re = tl.load(qi + 5)
        Kl = tl.load(qi + 6)
        lvid = tl.load(qi + 7)
        K = tl.load(qi + 8)
        a_ptr = tl.load(qp).to(tl.pointer_type(tl.float32))
        b_ptr = tl.load(qp + 1).to(tl.pointer_type(tl.float32))
        g_ptr = tl.load(qp + 2).to(tl.pointer_type(tl.float32))
        u_ptr = tl.load(qp + 3).to(tl.pointer_type(tl.float32))
        d_ptr = tl.load(qp + 4).to(tl.pointer_type(tl.float32))
        lth_ptr = tl.load(qp + 5).to(tl.pointer_type(tl.float32))
        lpar_ptr = tl.load(qp + 6).to(tl.pointer_type(tl.float32))
        lspid_ptr = tl.load(qp + 7).to(tl.pointer_type(tl.int64))
        eid_ptr = tl.load(qp + 8).to(tl.pointer_type(tl.int64))
        cn_ptr = tl.load(qp + 9).to(tl.pointer_type(tl.int64))
        cr_ptr = tl.load(qp + 10).to(tl.pointer_type(tl.int64))
        cand_ptr = tl.load(qp + 12).to(tl.pointer_type(tl.int32))
        nid = tl.load(ns_ptr + row * B + b)
        live = (nid >= nid0) & (nid < nid0 + M)
        i = tl.where(live, nid - nid0, 0).to(tl.int64)
        eid = tl.load(eid_ptr + offs_n, mask = offs_n < N, other = -1)
        ok = (offs_n < N) & (eid >= 0)
        cj = tl.zeros([BLOCK_N], dtype = tl.float32)
        if COND:
            cn = tl.load(cn_ptr + offs_n[:, None] * E_P + offs_e[None, :], mask = ok[:, None], other = 0)
            v = tl.load(nm_ptr + cn * B + b, mask = ok[:, None] & (cn > 0), other = 0.0)
            cj = tl.where(ok, tl.sum(v, axis = 1), -float("inf"))
        pl = -2
        if HAS_PLACE:
            pl = tl.load(tl.load(qp + 11).to(tl.pointer_type(tl.int64)) + b).to(tl.int32)
        rbase = ((row * B + b) * (2 * BLOCK_C + BLOCK_N)).to(tl.int32)
        j_star, c = _draw_node(i, xr, er, eid, ok, cj, pl, seed, rbase, lth_ptr, N, a_ptr, tl.load(qi + 9),
                               tl.load(qi + 10), tl.load(qi + 11), b_ptr, tl.load(qi + 12), tl.load(qi + 13),
                               tl.load(qi + 14), g_ptr, tl.load(qi + 15), tl.load(qi + 16), tl.load(qi + 17), r,
                               lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, nvars, lvid, C, u_ptr, tl.load(qi + 18),
                               tl.load(qi + 19), tl.load(qi + 20), d_ptr, tl.load(qi + 21), tl.load(qi + 22),
                               tl.load(qi + 23), re, Kl, K, COND, HAS_PLACE, BLOCK_N, BLOCK_C, RB, SB, BLOCK_RE)
        tl.store(cand_ptr + b, c.to(tl.int32), mask = live)
        # the drawn product's children into their rows: the level's row cleared first (a child may land on it)
        tl.store(ns_ptr + row * B + b, -1)
        tl.debug_barrier()
        cnj = tl.load(cn_ptr + j_star * E_P + offs_e)
        crj = tl.load(cr_ptr + j_star * E_P + offs_e)
        tl.store(ns_ptr + crj * B + b, cnj, mask = live & (cnj > 0) & (crj >= 0))
        tl.debug_barrier()


@triton.jit
def _tk_sample_leaf_kernel(samples_ptr, v, nf_ptr, row0, M, B, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, ev_mapped, nvars,
                           lvid, C, u_ptr, su0, su1, su2, d_ptr, sd0, sd1, sd2, re, Kl, emap_ptr, x_mapped, place_ptr,
                           seed, HAS_PLACE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
                           BLOCK_RE: tl.constexpr):
    """A leaf node vector no sum reads: a program a sample whose node of it is selected (its flow 1) -- its candidate
    drawn as the node's P(c | i) (an observed token, conditioning, itself), its token into `samples` [num_vars, B]."""
    b = tl.program_id(0).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    fl = tl.load(nf_ptr + (row0 + offs_m) * B + b, mask = offs_m < M, other = 0.0)
    live = tl.max(fl, axis = 0) > 0.5
    i = tl.argmax(fl, axis = 0).to(tl.int64)
    xr = b
    if x_mapped:
        xr = tl.load(emap_ptr + b).to(tl.int64)
    er = b
    if ev_mapped:
        er = xr
    tok, lq = _leaf_log_q(i, xr, er, lpar_ptr, lspid_ptr, ids_ptr, lp_ptr, nvars, lvid, C, u_ptr, su0, su1, su2,
                          d_ptr, sd0, sd1, sd2, re, Kl, BLOCK_C, BLOCK_RE)
    offs_c = tl.arange(0, BLOCK_C)
    c = tl.argmax(lq + _gumbel(seed, (b * BLOCK_C).to(tl.int32) + offs_c), axis = 0)
    if HAS_PLACE:
        pl = tl.load(place_ptr + b).to(tl.int32)
        c = tl.where(pl >= 0, pl, c)
    t_c = tl.sum(tl.where(offs_c == c, tok, 0), axis = 0)
    tl.store(samples_ptr + v * B + b, t_c, mask = live)
