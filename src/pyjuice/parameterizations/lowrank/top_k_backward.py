"""
`LowRank(top_k)` sum layers' BACKWARD by the CUDA kernels of `kernels/c/lowrank_top_k.cu` where they serve
(`top_k_fused.TOP_K_FUSED`, the module built, the fused forward's shapes, at most 4096 samples): from the STASH of
the forward's sums (pi the slots' posterior a node, pi / U, pi / Z, W's shift -- kept by the forward where a backward
may follow, else made here by its kernel) and the node flows: COLS over the forward's work items (a CTA 128 children:
the element flows, dB, dG, and a masked item's Gl stored), dA's GEMM of the stored Gl, an observed slot's dA pass --
their notes at the top of the .cu file.
The gradients go straight into their destinations (each sample's into the row of tensors it read, times
its sign, where `external_params_rows` folds them; else into its own); the EM statistics (F+, F- over the samples, a
tile of the parameters a CTA) by a CUDA kernel for the observed samples, the Triton one below for the masked.
Else `top_k._sum_backward_terms`, the reference.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                              # pragma: no cover
    triton = None

BM, BN = 64, 64                                                  # the EM kernel's tile: nodes, children
SH = 72                                                          # the stash's fp16 alpha / beta rows (halves; 64 slots, a tail)
MAX_SAMPLES = 4096                                               # (the work items' scans)


if triton is not None:

    E16_UNIT = tl.constexpr(2.0 ** -15)                          # E16's unit: the prep's E16 is E times 2^15

    @triton.jit
    def _bw_em_kernel(AH, BH, L2T, E16, EV, ALPHA, BETA, ALPHA_T, BETA_T, INVD, HROW, FLOW, MROWS, NM, LWIDX, EMAP, FP, FM,
                      M, N, KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      SH: tl.constexpr, ADD: tl.constexpr):
        """a CTA a 64 x 64 tile of the parameters: over the masked samples MROWS[:NM] (NM on the device), F+ = sum_b
        Q, F- = sum_b T ([M, N]: q, t summed over the slots), stored -- the observed samples' then added by
        `top_k_bw_em_observed` (CUDA). W from the CUDA passes' operands -- AH = fp16(log2 e A) [Bx, M, KP], BH [Bx, KP,
        N] fp16, l2t [M, N] fp16 -- and the stash's shift HROW [B, M]; a masked sample's alpha, beta [Rm, M, SH] fp16
        rows at its place LWIDX (its node flows not in them: FLOW [B, M], `top_k_flows`', times W): the slots times a
        node's power of 2 d by fp16 dots against E16 [Bx, 65, N] (E times 2^15, the prep's: the same 10-bit operands
        as TF32's, fp32 sums; its 2^15 out of the sums at the end), the rest slot times d from the rows' tail (ALPHA_T,
        BETA_T: their float32 view), 1 / d (INVD [Rm, M]) into W. ADD: added into FP, FM (a pass's accumulator:
        `em_accumulator`) -- nothing at all where no masked sample is taken."""
        if ADD:
            if tl.load(NM) == 0:
                return
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        s = tl.arange(0, 64)
        k = tl.arange(0, K0)
        l2t = tl.load(L2T + rows[:, None] * N + cols[None, :]).to(tl.float32)
        Fp = tl.zeros([BM, BN], tl.float32)
        Fm = tl.zeros([BM, BN], tl.float32)
        for j in range(0, tl.load(NM)):
            b64 = tl.load(MROWS + j).to(tl.int64)
            eb = tl.load(EMAP + b64).to(tl.int64)
            lg = tl.dot(tl.load(AH + (eb * M + rows[:, None]) * KP + k[None, :]),
                        tl.load(BH + (eb * KP + k[:, None]) * N + cols[None, :]))
            if K1 > 0:
                k1 = K0 + tl.arange(0, K1)
                lg += tl.dot(tl.load(AH + (eb * M + rows[:, None]) * KP + k1[None, :]),
                             tl.load(BH + (eb * KP + k1[:, None]) * N + cols[None, :]))
            lrow = tl.load(LWIDX + b64).to(tl.int64) * M + rows
            W = tl.exp2(lg + l2t - tl.load(HROW + b64 * M + rows)[:, None]) * \
                (tl.load(FLOW + b64 * M + rows) * tl.load(INVD + lrow))[:, None]
            ev = tl.load(EV + b64 * N + cols)
            base = lrow * SH
            Es = tl.load(E16 + (eb * 65 + s[:, None]) * N + cols[None, :])
            er = tl.load(E16 + (eb * 65 + 64) * N + cols).to(tl.float32)
            X = tl.dot(tl.load(ALPHA + base[:, None] + s[None, :]), Es) + \
                tl.load(ALPHA_T + lrow * (SH // 2) + 32)[:, None] * er[None, :]
            Y = tl.dot(tl.load(BETA + base[:, None] + s[None, :]), Es) + \
                tl.load(BETA_T + lrow * (SH // 2) + 32)[:, None] * er[None, :]
            Fp += W * X * ev[None, :]
            Fm += W * Y
        Fp = Fp * E16_UNIT
        Fm = Fm * E16_UNIT
        out = rows[:, None] * N + cols[None, :]
        if ADD:
            Fp += tl.load(FP + out)
            Fm += tl.load(FM + out)
        tl.store(FP + out, Fp)
        tl.store(FM + out, Fm)


def serves(desc, M: int, N: int, r: int, B: int, device, kwargs: dict, want_flows: bool) -> bool:
    """Whether the kernels take a node vector's backward (`B` samples): its element flows, gradients and (`want_flows`:
    by Triton) EM statistics."""
    from . import top_k_fused
    return (top_k_fused.TOP_K_FUSED and device.type == "cuda" and desc.top_k == 64 and M % 128 == 0 and N % 128 == 0
            and r <= 128 and B <= MAX_SAMPLES and kwargs.get("precision", "auto") != "fp32"
            and (triton is not None or not want_flows) and top_k_fused.top_k_module() is not None)


def scatter_element_flows(EF, cols, kids, element_flows) -> None:
    """`element_flows[kids, :] = logaddexp(element_flows[kids, :], log EF[:, cols]^T)` (`top_k.sum_backward`'s) by one
    kernel: the children's element flows from `terms`' EF [B, N] (fp32 flows, a sample a column)."""
    from . import top_k_fused as tkf
    tkf.top_k_module().top_k_scatter_ef(EF, cols, kids, element_flows)


def scatter_statistics(geo, Fp, Fm, params, param_flows, pf0: int, denom, d0) -> bool:
    """The EM statistics onto the node vector's slots by one kernel (`LowRank._scatter_add`'s): F+ into the parameter
    flows from `pf0` on (`Fp` None: none), F- / theta into the denominators from `d0` on (`denom` None: none) -- where
    its parameters lie block-major (`geo["grid"]`) and all is fp32; False else (the torch passes then)."""
    from . import top_k_fused as tkf
    grid = geo["grid"]
    if grid is None or params.dtype != torch.float32 or param_flows.dtype != torch.float32 or \
            (denom is not None and denom.dtype != torch.float32):
        return False
    off, _, Nb, cbs, bs = grid
    none = tkf._empty(params.device, torch.float32)
    tkf.top_k_module().top_k_scatter_em(none if Fp is None else Fp, none if denom is None else Fm, params,
                                        none if Fp is None else param_flows, none if denom is None else denom,
                                        geo["p0"], pf0 + off, 0 if denom is None else d0 + off, Nb, cbs, bs)
    return True


def em_accumulator(geo, params, param_flows, pf0: int, denom, d0, dev):
    """`(F+, F-)` [M, N] zeros to add a node vector's EM statistics into in the backward's deferred scope
    (`pyjuice.utils.deferred`), scattered at its end -- or None (no scope, or the scatter does not serve): each its own
    then, scattered at once. F+ is SHARED by the node vectors whose statistics go into the same flow block (tied copies:
    compilation gives up to `max_tied_ns_per_parflow_block` of them one `_param_flow_range`), scattered once into it. F-
    is the node vector's OWN, scattered (as F- / theta) into its own denominators: a copy's F- row is its collapsed model
    (`LowRank.collapsed_top_down`, the mini-batch EM's top-down step, reads each copy's slice) -- not interchangeable
    with the tie group's sum, though `materialize_denom_flows` only ever reads the sum."""
    from pyjuice.utils.deferred import current_scope
    scope = current_scope()
    grid = geo["grid"]
    if scope is None or grid is None or params.dtype != torch.float32 or param_flows.dtype != torch.float32 or \
            (denom is not None and denom.dtype != torch.float32):
        return None
    acc = scope.__dict__.setdefault("_top_k_em", {})
    key = (id(param_flows), param_flows.data_ptr(), pf0, geo["p0"], tuple(grid), geo["M"], geo["N"])
    got = acc.get(key)
    if got is None:
        FP = torch.zeros(geo["M"], geo["N"], dtype = torch.float32, device = dev)
        got = acc[key] = (FP, torch.empty_like(FP))           # (F+, and the F- the kernels write where none is kept)
        scope.defer(lambda: scatter_statistics(geo, FP, None, params, param_flows, pf0, None, None))
    if denom is None:
        return got
    FM = torch.zeros_like(got[0])
    scope.defer(lambda: scatter_statistics(geo, None, FM, params, param_flows, pf0, denom, d0))
    return got[0], FM


def _fold_operands(gviews, fold, B: int, dev):
    """`(drow, sgn)` int32, float32 [B]: the row each sample's gradient goes into and its sign -- the fold's
    (`StagedExternalParams.fold`), made once a backward, else a sample's own row, sign 1"""
    if fold is None:
        return torch.arange(B, dtype = torch.int32, device = dev), torch.ones(B, device = dev)
    made = gviews.__dict__.get("_top_k_fold")
    if made is None or made[0] is not fold:
        made = gviews.__dict__["_top_k_fold"] = (fold, fold[0].to(torch.int32).contiguous(), fold[1].float().contiguous())
    return made[1], made[2]


def _dest_ok(g: torch.Tensor) -> bool:
    """`g` [batch, rows, n] as the kernels add into it: float32, its rows contiguous and adjacent, 16-byte aligned with a
    batch stride of a multiple of 4 floats"""
    return (g.dtype == torch.float32 and g.is_cuda and g.dim() == 3 and g.stride(2) == 1 and g.stride(1) == g.size(2)
            and g.stride(0) % 4 == 0 and g.data_ptr() % 16 == 0)


def terms(A, Bm, G, tab, element_mars, geo, slot, rows, logw_rows, nf, emap, grads, gviews, fresh: bool,
          want_flows: bool = False, pmask = None, counts = None, kept = None, em_out = None):
    """`top_k._sum_backward_terms`' element flows `[B, N]`, the posterior counts `a` of the masked samples `rows`
    `[b, M, 65]` (None where they went into `counts`) and the EM statistics `(F+, F-)` `[M, N]` (None unless
    `want_flows`; of the samples `pmask` [B] holds, if given) by the kernels; the gradients of
    `A [Bx, M, r]`, `Bm [Bx, r, N]`, `G [Bx, 65, N]` (a row of tensors each) ADDED into the destinations `grads` (None:
    none) -- where they fold (`StagedExternalParams.fold`) each sample's times its sign into the row it read, else
    into its own. `emap [B]` the row a sample reads (None: its own); `tab` the parameters, `theta [M, N]` (0 where none),
    and their tables made so far in the pass (`top_k_fused.theta_tables`: the ones made here added); the
    children's values `element_mars [geo["eid"], :]`; `nf [M, B]` the node flows' log-space rows (the circuit's
    buffer); the observed samples' slots `slot [B]`, the masked ones' weights `logw_rows`; `counts` None, or the buffer
    `[b, M, 72]` the posterior counts go into, as it holds them (their leaf's: `top_k_leaf_fused.counts_buffer`);
    `kept` None, or the forward's operands and stash (`top_k_fused.kept_of`: its items made with the samples of a row
    sharing them)."""
    from . import top_k_fused as tkf
    mod = tkf.top_k_module()
    Bx, M, r = A.shape
    N = Bm.size(2)
    B = nf.size(1)
    dev = A.device
    fold = getattr(gviews, "fold", None) if grads is not None else None
    # (the samples reading a row share one item -- its tile -- where their gradients merge: a fold, or none asked)
    pair = emap is None or fold is not None or grads is None
    A_ = tkf._rows(A)
    ops, stash = kept if kept is not None else (None, None)
    if ops is None or not pair:
        # (its own items; the forward's stash still serves: a sample's terms, whatever the items that made them)
        ops = tkf.prep(Bm, G, element_mars, geo["eid"], None if geo["full"] else geo["has"], tab["theta"], slot, rows,
                       logw_rows, emap, pair = pair, lth = tab.get("LTh"))
        tab.setdefault("LTh", ops[4])
    BP, E16, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags = ops
    none_h = tkf._empty(dev, torch.float16)
    AH, L2T, LTc = mod.top_k_bw_prep(A_, tab["theta"], BP.size(1), tab.get("L2T", none_h),
                                     tab.get("LTc", tkf._empty(dev, torch.int32)))
    tab.setdefault("L2T", L2T)
    tab.setdefault("LTc", LTc)
    if stash is None:
        stash = tkf.forward(A_, ops, None, 0, True, counts)
    alpha16, beta16, invd, acnt, ao, bo, hrow = stash
    # the node flows: into the stash's ao, bo, and beside the masked samples' posterior where that lies in their leaf's
    # buffer; fl [B, M] the flows themselves (a masked sample's terms' factor in the passes below), fmx [B] their largest
    Rm = rows.numel()
    own = counts is not None and Rm > 0 and acnt.data_ptr() == counts.data_ptr()
    none32 = tkf._empty(dev, torch.float32)
    # (the gradients' destinations: handed over unzeroed with this node vector their first writer, the kernels initialize
    # them -- the masked passes' first writer of a tile stores it, the rows no masked sample writes are zeroed
    # (`top_k_flows`' init, `top_k_bw_grads`' `dflag`) -- instead of a zeroing pass (MEASURED, the best Codpula head:
    # 0.87 ms of fills a step); else zeroed here, or added into)
    stores, dflag, nonei = False, None, tkf._empty(dev, torch.int32)
    if grads is not None:
        drow, sgn = _fold_operands(gviews, fold, B, dev)
        dests = [g if _dest_ok(g) else torch.zeros(g.shape, dtype = torch.float32, device = dev) for g in grads]
        first = fresh and gviews.claim(grads)
        stores = first and all(d is g for g, d in zip(grads, dests))
        if first and not stores:
            torch._foreach_zero_(list(grads))
        if stores:
            Bd, M_, N_ = grads[0].size(0), grads[0].size(1), grads[1].size(2)
            dflag = torch.empty(Bd * (M_ // 128 + N_ // 128), dtype = torch.int32, device = dev)
    # (the node flows into the stash; EF zeroed by the same launch, the destinations' init too where the kernels store)
    EF = torch.empty(B, N, dtype = torch.float32, device = dev)
    fl, fmx = mod.top_k_flows(nf, ao, bo, rows.long().contiguous(), acnt if own else none32, EF,
                              mitems if stores else nonei, drow if stores else nonei, dflag if stores else nonei,
                              *(dests if stores else (none32, none32, none32)))
    if grads is not None:
        mod.top_k_bw_grads(A_, AH, BP, E16, ev, LTh, LTc, mitems, oitems, noitems, slot32, alpha16, beta16, invd, ao,
                           bo, hrow, fl, fmx, drow, sgn, EF, *dests, dflag if stores else nonei)
        for g, d in zip(grads, dests):
            if d is not g:
                g.add_(d.to(g.dtype))
    else:
        none = torch.empty(0, dtype = torch.float32, device = dev)
        mod.top_k_bw_grads(A_, AH, BP, E16, ev, LTh, LTc, mitems, oitems, noitems, slot32, alpha16, beta16, invd, ao,
                           bo, hrow, fl, fmx, none, none, EF, none, none, none, nonei)
    flows = None
    if want_flows:
        KP = BP.size(1)
        K0 = min(KP, 64)
        # (the samples the statistics take, in the kernels' terms: the observed ones' slots, -1 the others; the masked
        # ones' places; the rows read; the masked and the observed ones taken, in order, -1 padded)
        none = tkf._empty(dev, torch.long)
        SLOT, LWIDX, EMAP, MROWS, NM, OLIST = mod.top_k_em_index(
            slot32, rows.long().contiguous(), none if emap is None else emap.long().contiguous(),
            tkf._empty(dev, torch.bool) if pmask is None else pmask.contiguous())
        # the masked samples' (Triton: stored, or zeros where none), then the observed ones' added (CUDA: a tile of
        # the parameters a CTA walking the samples)
        # (or added into `em_out`, a pass's accumulator of the node vectors sharing a flow block: `em_accumulator`)
        add = em_out is not None
        if add:
            FP, FM = em_out
        if Rm > 0:
            if not add:
                FP = torch.empty(M, N, dtype = torch.float32, device = dev)
                FM = torch.empty_like(FP)
            _bw_em_kernel[(M // BM, N // BN)](AH, BP, L2T, E16, ev, alpha16, beta16, alpha16.view(torch.float32),
                                              beta16.view(torch.float32), invd, hrow, fl, MROWS, NM, LWIDX, EMAP, FP,
                                              FM, M, N, KP = KP, K0 = K0, K1 = KP - K0, BM = BM, BN = BN, SH = SH,
                                              ADD = add, num_warps = 4, num_stages = 1)
        elif not add:
            FP = torch.zeros(M, N, dtype = torch.float32, device = dev)
            FM = torch.zeros_like(FP)
        if B > Rm:
            mod.top_k_bw_em_observed(AH, BP, E16, ev, LTh, ao, bo, hrow, SLOT,
                                     tkf._empty(dev, torch.int32) if emap is None else EMAP, OLIST, FP, FM)
        flows = (FP, FM)
    # (the counts: in the leaf's rows as it holds them -- the slots' posterior, the flow beside it --, else their product)
    a_rows = None if own else acnt[:Rm, :, :65] * fl.index_select(0, rows)[:, :, None]
    return EF, a_rows, flows
