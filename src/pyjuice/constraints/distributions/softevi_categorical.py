"""
SoftEvidenceCategorical input nodes (:class:`pyjuice.nodes.distributions.SoftEvidenceCategorical`) under a
constraint. Their parameters are a Categorical's (rows of a ``[rows, num_cats]`` table, node ``i`` at row ``s_pids[i]
// num_cats``), so without evidence they are Categorical nodes: the functions of :mod:`.categorical` serve them.

Dense soft evidence, ``categorical_evidence_logp`` [batch, the layer's variables, K] (log-weights: token ``v < K``
weighs ``w(v) = exp(E[b, i, v])`` at the layer's ``i``-th variable, the others nothing), makes a node's emission at
variable ``i`` of sample ``b`` the product of experts the library defines, ``q(v) = p(v) w(v) / Z`` with ``Z = sum_v
p(v) w(v)``: an observed token's value is ``log q(d)`` (the library's forward), and a marginalized one's class masses
are ``log sum_{v in c} q(v)`` -- they sum to 1 over the classes, as the library's 0 for a missing token. They depend on
the sample, the variable and the node's row: :func:`sample_inputs` computes them once for a decoder's session, a row
per (variable, row) pair some node reads, in one pass over the parameter table.

Top-k soft evidence, ``soft_evidence_cat_ids`` besides (int [batch, the layer's variables, K], distinct per row): token
``ids[b, i, k]`` weighs ``exp(E[b, i, k])``, the others nothing -- an observed token that is not a candidate is
impossible (-inf), as in the library. The pass then reads only the candidates' probabilities, from the library's
transposed table where a :func:`pyjuice.fast_inference` scope keeps one (as the library's top-k forward does).

A value mask, ``soft_evidence_value_mask`` (bool [batch, the layer's variables]), makes the library's unnormalized
model: where True, an observed token's value is ``log p(d) w(d)`` (no ``Z``); where False, the token is ignored and the
value is ``log Z`` -- the sum over every token of ``p(v) w(v)``, so under a constraint the position is free
(:func:`free_positions`: marginalized, its tokens weighed by ``p w``), with class masses ``log sum_{v in c} p(v) w(v)``.
A missing token's masses stay ``q``'s, wherever the mask is True (a position cannot be missing and False: the library's
missing token contributes 1 whatever the mask, a False one ``Z``).
"""

import weakref

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from . import SampleInputs, find_token
from .categorical import class_mass_rows, class_masses, num_values, token_probs

__all__ = ["KEYWORDS", "num_values", "class_mass_rows", "class_masses", "token_probs", "check_inputs",
           "free_positions", "sample_rows", "sample_inputs", "input_values"]

#: the keywords of ``pc(x, ...)`` the distribution reads
KEYWORDS = ("categorical_evidence_logp", "soft_evidence_cat_ids", "soft_evidence_value_mask")

#: :func:`sample_inputs`' tiles: rows of the parameter table, columns ((variable, sample) pairs) and tokens (one class
#: each, in class order) per product
MASS_TILES = dict(TU = 32, TN = 32, TV = 64)
#: :func:`sample_inputs` splits the tokens over programs until there are at least this many (a constant, not the
#: device's SM count, so that the summation order and so the results do not depend on the GPU) ...
MASS_PROGRAMS = 1024
#: ... and keeps every split's partial sums only while they take at most this many floats (beyond, fewer splits)
MASS_PARTIAL_BUDGET = 1 << 26
#: the top-k pass's tiles (:func:`_topk_kernel`): rows of the parameter table and candidates per product, and the most
#: classes per program (more: the candidates sorted by tile of classes, a tile at a time)
TOPK_TILES = dict(TU = 32, TK = 64, TC = 64)
#: Whether the top-k pass sums every stored class's candidates on their own (:func:`_topk_class_kernel`: the candidates
#: sorted by class, a class's run a sum) rather than as a one-hot product over a tile of classes (:func:`_topk_kernel`),
#: and its tiles: rows of the parameter table, candidates a step, classes a fill. MEASURED (Codpula-4096 x JSON hard_0
#: B8, LLaDA vocabulary: 75 stored classes, a column's 1024 candidates in ~16 of them, 744 in the largest): the one-hot
#: product multiplies every candidate's probabilities by the whole tile of 64 classes -- 18 G instructions, 203
#: registers, 46.8 ms under ncu --, where a class's sum takes one multiply-add a candidate and row.
TOPK_BY_CLASS = True
TOPK_CLASS_TILES = dict(TU = 32, TK = 64, TF = 32)


def check_inputs(layer, inputs: dict):
    """The batch size of the layer's soft evidence in ``inputs`` (see :data:`KEYWORDS`), or None if it has none."""
    ev, ids = inputs.get("categorical_evidence_logp"), inputs.get("soft_evidence_cat_ids")
    vm = inputs.get("soft_evidence_value_mask")
    if ev is None:
        for key, value in (("soft_evidence_cat_ids", ids), ("soft_evidence_value_mask", vm)):
            if value is not None:
                raise ValueError(f"`{key}` needs `categorical_evidence_logp`: the tokens' log-weights.")
        return None
    num_vars = _num_vars(layer)
    if not torch.is_tensor(ev) or ev.dim() != 3 or not ev.is_floating_point():
        raise ValueError(f"`categorical_evidence_logp` must be a float tensor [batch, {num_vars}, K], got "
                         f"{type(ev).__name__ if not torch.is_tensor(ev) else list(ev.shape)}.")
    if ev.device != layer.params.device:
        raise ValueError(f"`categorical_evidence_logp` is on {ev.device}, the circuit on {layer.params.device}.")
    if ev.size(1) < num_vars or ev.size(2) > layer.dist.num_cats or ev.size(0) == 0:
        raise ValueError(f"`categorical_evidence_logp` must be [batch, {num_vars}, K] with K <= "
                         f"{layer.dist.num_cats} (the SoftEvidenceCategorical layer's variables and values), got "
                         f"{list(ev.shape)}.")
    if ids is not None:
        if not torch.is_tensor(ids) or ids.shape != ev.shape or ids.is_floating_point() or ids.dtype == torch.bool:
            raise ValueError(f"`soft_evidence_cat_ids` must be an integer tensor of `categorical_evidence_logp`'s shape "
                             f"{list(ev.shape)}, got {type(ids).__name__ if not torch.is_tensor(ids) else ids.dtype} "
                             f"{list(ids.shape) if torch.is_tensor(ids) else ''}.")
        if ids.device != ev.device:
            raise ValueError(f"`soft_evidence_cat_ids` is on {ids.device}, `categorical_evidence_logp` on {ev.device}.")
        s = torch.sort(ids[:, :num_vars], dim = 2).values                  # the rows the layer reads
        if bool(((s[..., 1:] == s[..., :-1]).any(dim = 2) | (s[..., 0] < 0) |
                 (s[..., -1] >= layer.dist.num_cats)).any()):
            raise ValueError(f"`soft_evidence_cat_ids` must list distinct tokens in [0, {layer.dist.num_cats}) for every "
                             f"sample and variable (as `torch.topk` gives them).")
    if vm is None:
        return int(ev.size(0))
    if not torch.is_tensor(vm) or vm.dtype != torch.bool or vm.dim() != 2 or vm.size(1) < num_vars or \
            vm.size(0) not in (1, ev.size(0)) and ev.size(0) != 1:
        raise ValueError(f"`soft_evidence_value_mask` must be a bool tensor [batch, {num_vars}], its batch the "
                         f"evidence's ({ev.size(0)}) or either's 1, got "
                         f"{type(vm).__name__ if not torch.is_tensor(vm) else f'{vm.dtype} {list(vm.shape)}'}.")
    if vm.device != ev.device:
        raise ValueError(f"`soft_evidence_value_mask` is on {vm.device}, `categorical_evidence_logp` on {ev.device}.")
    return max(int(ev.size(0)), int(vm.size(0)))


def free_positions(layer, inputs: dict, n: int):
    """bool [batch, n]: the PC's positions the layer's value mask leaves free (False there: the token is ignored and
    marginalized under the evidence's weights, see the module docstring); None without a value mask."""
    vm = inputs.get("soft_evidence_value_mask")
    if vm is None:
        return None
    vids = torch.unique(layer.vids.view(-1)).to(vm.device)
    free = torch.zeros(vm.size(0), n, dtype = torch.bool, device = vm.device)
    free[:, vids] = ~vm[:, layer.var_idmapping.to(vm.device)[vids]]
    return free


def sample_rows(layer, inputs: dict) -> int:
    """The rows per sample of :func:`sample_inputs`' masses: the (variable, parameter row) pairs the nodes read."""
    return _structure(layer, _num_vars(layer))["num_pairs"]


def sample_inputs(layer, classes, inputs: dict, out: torch.Tensor) -> SampleInputs:
    """
    The class masses the layer's soft evidence gives its nodes (see the module docstring), into ``out`` [batch,
    :func:`sample_rows`, num_classes]: a row per (variable, parameter row) pair some node reads, per sample, ``log
    sum_{v in c} p(v) w(v) - log Z``, with ``log Z`` per row and sample (what an observed token's value subtracts,
    :func:`input_values`) and the evidence as the tokens' weights (a node's variable its row; top-k: the candidates
    sorted, see :class:`~pyjuice.constraints.distributions.SampleInputs`). One pass, fp32: every row of the parameter
    table read once for all the variables it serves (tied rows: every variable) -- dense evidence: every token
    class's tokens against the evidence of those variables (:func:`_dense_masses`); top-k: every variable's
    candidates alone (:func:`_topk_masses`). Under a value mask, rows of a sample and variable it marks False keep
    the masses unnormalized (``log Z`` 0: a free position's tokens weigh ``p w``).
    """
    Vl = _num_vars(layer)                                                   # the rows of the evidence the layer reads
    ids, vm = inputs.get("soft_evidence_cat_ids"), inputs.get("soft_evidence_value_mask")
    Be = max(inputs["categorical_evidence_logp"].size(0), 1 if vm is None else vm.size(0))   # (a mask per sample
    ev = inputs["categorical_evidence_logp"][:, :Vl].float().expand(Be, -1, -1).contiguous()  # over shared evidence)
    ids = None if ids is None else ids[:, :Vl].expand(Be, -1, -1)
    st = _structure(layer, Vl)
    # the evidence's shift per (sample, variable): the max weight scales to 1 (a row of -inf weighs nothing anyway)
    shift = ev.amax(dim = 2)
    shift = torch.where(torch.isfinite(shift), shift, 0.0).contiguous()
    logz = torch.empty(Be, st["num_pairs"], dtype = torch.float32, device = ev.device)   # log Z of every row: over
    if ids is None:                                                         # every token, the enumerated classes' too
        _dense_masses(layer, classes, ev, shift, st, out, logz)
    else:                                                                   # ascending: what the lookups search
        ids, order = torch.sort(ids.to(torch.int32), dim = 2)
        ev = torch.gather(ev, 2, order)
        _topk_masses(layer, classes, inputs, ev, ids, shift, st, out, logz)
    logz = torch.where(torch.isfinite(logz), logz, 0.0)
    if vm is not None:                                                      # False: unnormalized (a free position)
        logz = torch.where(vm[:, :Vl].expand(Be, -1)[:, st["pair_lv"].long()], logz, 0.0)
    logz = logz.contiguous()
    out -= logz[:, :, None]                                                 # q's classes: they sum to one
    return SampleInputs(masses = out, rows = st["node_rows"], logz = logz, token_logw = ev,
                        token_rows = _lvid32(layer).long(), batch = Be, variables = st["variables"],
                        token_ids = None if ids is None else ids.contiguous(), pair_rows = st["pair_rows"],
                        data = None if vm is None else dict(value_mask = True))


def _dense_masses(layer, classes, ev, shift, st, out, logz):
    """:func:`sample_inputs`' masses under dense evidence, before ``log Z``: every row's tokens in class order against
    every variable's evidence -- split over the tokens for enough programs, each split's sums kept apart while they
    fit :data:`MASS_PARTIAL_BUDGET`, else in one split, written straight. The stored classes' masses into ``out``, and
    every row's ``log Z`` (over every token, the enumerated classes' too) into ``logz`` [Be, R_s]."""
    Be, Vl, K = ev.shape
    C = classes.num_stored
    dev = ev.device
    cols, tiles = _columns(st, Be, dev)
    tc, TV = MASS_TILES, MASS_TILES["TV"]
    tokens, tile_rank, class_rank, rank_stored, stored_rank = _class_runs(classes, TV)
    T = tile_rank.numel()
    R_s = st["num_pairs"]
    present = int(class_rank.max()) + 1
    splits = max(1, min(host_cdiv(MASS_PROGRAMS, tiles.size(0)), T, MASS_PARTIAL_BUDGET // max(1, Be * R_s) - present))
    split_len = host_cdiv(T, splits)
    splits = host_cdiv(T, split_len)
    table = layer.params.view(-1, layer.dist.num_cats)
    direct = splits == 1
    if direct:                                                          # every class's sum in one piece: its log
        out.fill_(-float("inf"))                                        # (a class without a token: -inf)
        part = out
    else:                                                               # (zeros: the segments between two classes'
        part = torch.zeros(present + splits, Be, R_s, dtype = torch.float32, device = dev)   # pieces are never written)
    _partial_kernel[(tiles.size(0) * splits,)](
        table, ev, shift, tokens, tile_rank, rank_stored, st["rows"], cols, tiles, st["pair_index"], part, logz,
        layer.dist.num_cats, K, Vl, Be, R_s, C, splits, split_len, T, DIRECT = direct, TU = tc["TU"], TN = tc["TN"],
        TV = TV, num_warps = 4)
    if not direct:
        seg_lo, seg_hi = _segments(tile_rank, class_rank, split_len, dev)
        TS, TC = 32, max(16, min(64, host_next_power_of_2(max(C, 1))))
        if C > 0:
            _combine_kernel[(host_cdiv(R_s, TS) * host_cdiv(C, TC), Be)](
                part, out, shift, st["pair_lv"], stored_rank, seg_lo, seg_hi, Vl, Be, R_s, C, TS = TS, TC = TC)
        m = shift.view(Be, Vl)[:, st["pair_lv"].long()]                 # log Z: every segment's sum, unscaled
        logz.copy_(torch.log(part.sum(dim = 0)) - 44.3614195558365 + m)


def _topk_class_masses(layer, classes, inputs, ev, ids, shift, st, out, logz):
    """:func:`_topk_masses` by :func:`_topk_class_kernel`: the candidates sorted by (stored class, token), the
    enumerated classes' first; ``bounds`` [Be * Vl, C + 2] every column's runs."""
    Be, Vl, K = ev.shape
    C, num_cats = classes.num_stored, layer.dist.num_cats
    dev = ev.device
    cols, tiles = _columns(st, Be, dev)
    sc = classes.stored_class[ids.long()].long() + 1                        # 0: an enumerated class
    order = torch.argsort(sc * num_cats + ids, dim = 2)
    ids, ev, sc = (torch.gather(a, 2, order).contiguous() for a in (ids, ev, sc))
    at = torch.arange(C + 2, device = dev).expand(Be * Vl, C + 2).contiguous()
    bounds = torch.searchsorted(sc.view(Be * Vl, K), at).to(torch.int32).contiguous()
    table = layer.params.view(-1, num_cats)
    table_t = _transposed_table(layer, inputs)
    cfg = TOPK_CLASS_TILES
    for first in range(0, tiles.size(0), 65535):                            # (a grid's second axis: at most 65535)
        _topk_class_kernel[(MASS_TILES["TN"], min(65535, tiles.size(0) - first))](
            table, table if table_t is None else table_t, table.size(0), ev, ids, shift, bounds, st["rows"], cols, tiles,
            st["pair_index"], out, logz, num_cats, K, Vl, st["num_pairs"], C, first, TRANSPOSED = table_t is not None,
            TU = cfg["TU"], TK = cfg["TK"], TF = cfg["TF"], num_warps = 4)


@triton_jit
def _topk_class_kernel(table, table_t, num_rows, ev, ids, shift, bounds, rows, cols, tiles, pair_index, out, logz,
                       num_cats, K, Vl, R_s, C, tile0, TRANSPOSED: tl.constexpr, TU: tl.constexpr, TK: tl.constexpr,
                       TF: tl.constexpr):
    """:func:`_topk_kernel`'s masses by class: one program per (column of a tile, tile), the column the fast axis (a
    tile's rows' probabilities read by its columns' programs together, while in the L2): the row block's every class
    -inf, then every stored class present among the column's candidates (``bounds``: its run in the class-sorted
    candidates) the sum ``sum_k p(v_k) w_k`` over its run alone -- one multiply-add a candidate and row --, scaled by
    2^64, its log (the shift added back); every row's ``log Z`` the sum over every candidate (the enumerated classes'
    first, ``bounds[col, 0] .. bounds[col, 1]``)."""
    j = tl.program_id(0)
    tile = tile0 + tl.program_id(1)
    r0, nr = tl.load(tiles + tile * 4), tl.load(tiles + tile * 4 + 1)
    c0, nc = tl.load(tiles + tile * 4 + 2), tl.load(tiles + tile * 4 + 3)
    if j < nc:
        lv = tl.load(cols + (c0 + j) * 2).to(tl.int64)
        b = tl.load(cols + (c0 + j) * 2 + 1).to(tl.int64)
        col = b * Vl + lv
        m = tl.load(shift + col)
        ru = tl.arange(0, TU)
        rm = ru < nr
        r = tl.load(rows + r0 + ru, mask = rm, other = 0).to(tl.int64)
        s = tl.load(pair_index + r * Vl + lv, mask = rm, other = -1)
        keep = rm & (s >= 0)
        dst = out + (b * R_s + s) * C
        for f0 in range(0, C, TF):                                          # every class: none
            f = f0 + tl.arange(0, TF)
            tl.store(dst[:, None] + f[None, :], float("-inf"), mask = keep[:, None] & (f < C)[None, :])
        tl.debug_barrier()                                                  # (the present ones written over)
        bb = bounds + col * (C + 2)
        total = tl.zeros([TU], dtype = tl.float32)
        for c in range(-1, C):                                              # the enumerated classes (log Z), then
            k0 = tl.load(bb + c + 1)                                        # every stored class's run
            k1 = tl.load(bb + c + 2)
            if k1 > k0:
                acc = tl.zeros([TU], dtype = tl.float32)
                for k in range(k0, k1, TK):
                    kk = k + tl.arange(0, TK)
                    km = kk < k1
                    v = tl.load(ids + col * K + kk, mask = km, other = 0).to(tl.int64)
                    w = tl.exp(tl.load(ev + col * K + kk, mask = km, other = float("-inf")) - m)
                    if TRANSPOSED:
                        p = tl.load(table_t + v[:, None] * num_rows + r[None, :], mask = km[:, None] & rm[None, :],
                                    other = 0.0)
                    else:
                        p = tl.load(table + r[None, :] * num_cats + v[:, None], mask = km[:, None] & rm[None, :],
                                    other = 0.0)
                    acc += tl.sum(w[:, None] * (p * 18446744073709551616.0), axis = 0)          # x 2^64
                total += acc
                if c >= 0:
                    tl.store(dst + c, tl.log(acc) - 44.3614195558365 + m, mask = keep)
        tl.store(logz + b * R_s + s, tl.log(total) - 44.3614195558365 + m, mask = keep)


def _topk_masses(layer, classes, inputs, ev, ids, shift, st, out, logz):
    """:func:`sample_inputs`' masses under top-k evidence (``ids`` the candidates, ``ev`` their log-weights), before
    ``log Z``: every row's probabilities of every variable's candidates, a program per (tile of rows, variable and
    sample) -- the classes ``TC`` at a time, the candidates sorted by tile of classes where there are more (those of
    enumerated classes first: they count in ``log Z`` alone). The stored classes' masses into ``out``, every row's
    ``log Z`` into ``logz`` [Be, R_s]. (:data:`TOPK_BY_CLASS`: :func:`_topk_class_masses`.)"""
    if TOPK_BY_CLASS:
        return _topk_class_masses(layer, classes, inputs, ev, ids, shift, st, out, logz)
    Be, Vl, K = ev.shape
    C, num_cats = classes.num_stored, layer.dist.num_cats
    dev = ev.device
    cols, tiles = _columns(st, Be, dev)
    TU, TK = TOPK_TILES["TU"], TOPK_TILES["TK"]
    TC = max(16, min(TOPK_TILES["TC"], host_next_power_of_2(max(C, 1))))
    NCT = host_cdiv(C, TC)
    bounds = shift                                                          # (unread with one tile of classes)
    if NCT > 1:
        sc = classes.stored_class[ids.long()].long()
        ctile = torch.where(sc >= 0, sc // TC, -1)
        order = torch.argsort(ctile * num_cats + ids, dim = 2)
        ids, ev, ctile = (torch.gather(a, 2, order).contiguous() for a in (ids, ev, ctile))
        at = torch.arange(NCT + 1, device = dev).expand(Be * Vl, NCT + 1).contiguous()
        bounds = torch.searchsorted(ctile.view(Be * Vl, K), at).to(torch.int32).contiguous()
    table = layer.params.view(-1, num_cats)
    table_t = _transposed_table(layer, inputs)
    _topk_kernel[(tiles.size(0), MASS_TILES["TN"])](
        table, table if table_t is None else table_t, table.size(0), ev, ids, shift, classes.stored_class, bounds,
        st["rows"], cols, tiles, st["pair_index"], out, logz, num_cats, K, Vl, st["num_pairs"], C, max(NCT, 1),
        TRANSPOSED = table_t is not None, BY_TILE = NCT > 1, TU = TU, TK = TK, TC = TC, num_warps = 4)


def _transposed_table(layer, inputs):
    """The layer's parameter table transposed ([num_cats, rows]), where the library keeps one for its top-k forward:
    inside a :func:`pyjuice.fast_inference` scope that allows parameter copies (built on first use there); else
    None."""
    ensure = getattr(layer, "_ensure_fast_inference_params", None)
    table = ensure(inputs) if ensure is not None else None
    if table is None or table.dim() != 2 or table.size(0) != layer.dist.num_cats or \
            table.numel() != layer.params.numel():
        return None
    return table


def input_values(layer, x: torch.Tensor, missing: torch.Tensor, out: torch.Tensor, sample = None):
    """``log q(d)`` of every observed token ``d`` at every node of the layer, into its rows of ``out`` (0 where the
    token is missing): ``log p(d)``, and under soft evidence (``sample``, :func:`sample_inputs`') ``+ log w(d) - log
    Z`` (top-k: -inf for a token that is not a candidate; under a value mask, True wherever a token is observed, no
    ``Z``). The weight of the token a sample observes at a variable is looked up once (:func:`_token_weights_kernel`),
    for all the nodes over the variable."""
    B, n = x.shape
    first, end = layer._output_ind_range
    nodes = end - first
    num_cats = layer.dist.num_cats
    table = layer.params.view(-1, num_cats)
    prow = _rows32(layer)
    miss = missing.view(torch.uint8)
    has_ev = sample is not None
    normalize = has_ev and not (sample.data or {}).get("value_mask", False)
    if has_ev:
        ev = sample.token_logw
        Be, Vl, K = sample.batch, ev.size(1), ev.size(2)
        topk = sample.token_ids is not None
        w = torch.empty(B * Vl, device = x.device)                            # every (sample, variable)'s log w(d)
        _token_weights_kernel[(host_cdiv(B, 16), host_cdiv(Vl, 32))](
            w, x, miss, sample.variables, ev, sample.token_ids if topk else ev, B, n, Vl, K, PER_SAMPLE = Be > 1,
            TOPK = topk, STEPS = max(1, K.bit_length()), TB = 16, TV = 32)
        logz, lvid, srow, R_s = sample.logz, _lvid32(layer), _rows_of(sample), sample.masses.size(1)
    else:
        w = logz = lvid = srow = prow
        Be = Vl = R_s = 1
    TN, TB = 32, max(16, min(64, host_next_power_of_2(B)))
    _input_values_kernel[(host_cdiv(nodes, TN), host_cdiv(B, TB))](
        out, table, x, miss, layer.vids, prow, lvid, srow, w, logz, first, nodes, B, n, num_cats, Vl, R_s,
        HAS_EV = has_ev, PER_SAMPLE = Be > 1, NORMALIZE = normalize, TN = TN, TB = TB)


# -------------------------------------------------------------------------------------------------
# What the layer's structure fixes (kept per layer)
# -------------------------------------------------------------------------------------------------

_STRUCTURES = weakref.WeakKeyDictionary()                  # layer -> {Vl: structure}


def _rows_of(sample) -> torch.Tensor:
    """``sample.rows`` as int32 (converted once per session's inputs)."""
    rows = sample.__dict__.get("_rows32")
    if rows is None:
        rows = sample.__dict__["_rows32"] = sample.rows.to(torch.int32).contiguous()
    return rows


def _rows32(layer) -> torch.Tensor:
    return _cached(layer, "prow", lambda: class_mass_rows(layer).to(torch.int32).contiguous())


def _lvid32(layer) -> torch.Tensor:
    return _cached(layer, "lvid", lambda: layer.var_idmapping[layer.vids.view(-1).long()].to(torch.int32).contiguous())


def _num_vars(layer) -> int:
    """The layer's variables (its nodes' variable ids map to ``0 ..`` that, the rows of the evidence it reads)."""
    return _cached(layer, "num_vars", lambda: int(torch.unique(layer.vids.view(-1)).numel()))


def _cached(layer, key, build):
    entry = _STRUCTURES.setdefault(layer, {})
    if key not in entry:
        entry[key] = build()
    return entry[key]


def _structure(layer, Vl: int) -> dict:
    """
    The (variable, parameter row) pairs the layer's nodes read, and how :func:`sample_inputs` covers them: every
    node's pair (``node_rows``), every pair's variable (``pair_lv``), the pair of a (row, variable) (``pair_index``,
    -1 for none), and the parameter rows grouped by the set of variables they serve (``groups``: (rows, variables)
    host arrays), every row once.
    """
    def build():
        pr = class_mass_rows(layer).long()
        lv = layer.var_idmapping[layer.vids.view(-1).long()].long()
        R = int(pr.max()) + 1
        uniq, inverse = torch.unique(pr * Vl + lv, return_inverse = True)            # by row, then variable
        pair_index = torch.full((R * Vl,), -1, dtype = torch.int32, device = pr.device)
        pair_index[uniq] = torch.arange(uniq.numel(), dtype = torch.int32, device = pr.device)
        u = uniq.cpu().numpy()
        rows, lvs = u // Vl, u % Vl
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        groups = {}
        for s, e in zip(starts, np.r_[starts[1:], len(u)]):                         # rows by their variables
            groups.setdefault(lvs[s:e].tobytes(), (lvs[s:e], []))[1].append(rows[s])
        groups = [(np.asarray(r, dtype = np.int64), v) for v, r in groups.values()]
        return dict(num_pairs = int(uniq.numel()), node_rows = inverse, pair_index = pair_index,
                    pair_lv = (uniq % Vl).to(torch.int32).contiguous(), pair_rows = uniq // Vl, groups = groups,
                    rows = torch.as_tensor(np.concatenate([g for g, _ in groups]), dtype = torch.int32,
                                           device = pr.device),
                    variables = torch.unique(layer.vids.view(-1).long()))
    return _cached(layer, ("structure", Vl), build)


def _columns(st: dict, Be: int, dev):
    """The (variable, sample) columns of every group and the tiles of the pass: ``cols`` int32 [columns, 2], ``tiles``
    int32 [tiles, 4] (first row in the structure's rows list, rows, first column, columns)."""
    TU, TN = MASS_TILES["TU"], MASS_TILES["TN"]
    cols, tiles, r0, c0 = [], [], 0, 0
    for grows, glvs in st["groups"]:
        cols.append(np.stack([np.repeat(glvs, Be), np.tile(np.arange(Be), len(glvs))], axis = 1))
        nc = len(glvs) * Be
        tiles += [(r0 + i, min(TU, len(grows) - i), c0 + j, min(TN, nc - j)) for i in range(0, len(grows), TU)
                  for j in range(0, nc, TN)]
        r0, c0 = r0 + len(grows), c0 + nc
    i32 = lambda a: torch.as_tensor(np.asarray(a, dtype = np.int64), dtype = torch.int32, device = dev).contiguous()
    return i32(np.concatenate(cols)), i32(tiles)


def _class_runs(classes, TV: int):
    """The tokens in class order (every class's, the enumerated ones' too: ``log Z`` sums them all), every class's
    run padded to whole tiles of ``TV`` (:meth:`TokenClasses.class_tiles`), every tile's class's rank among the
    classes with a token, every class's rank (-1 for none), every rank's stored column (-1: an enumerated class) and
    every stored column's rank (-1: a stored class without a token); kept with the classes."""
    key = ("softevi_runs", TV)
    found = classes._class_tiles.get(key)
    if found is None:
        tokens, tile_class = classes.class_tiles(TV)
        tc = tile_class.long()
        present = torch.zeros(classes.num_classes, dtype = torch.bool, device = tc.device)
        present[tc] = True
        rank = torch.cumsum(present.long(), 0) - 1
        class_rank = torch.where(present, rank, -1).to(torch.int32).contiguous()
        rank_class = torch.nonzero(present).view(-1)
        i32 = lambda t: t.to(torch.int32).contiguous()
        found = classes._class_tiles[key] = (tokens, i32(rank[tc]), class_rank,
                                             i32(classes.stored_of_class[rank_class]),
                                             i32(class_rank[classes.class_of_stored]))
    return found


def _segments(tile_rank: torch.Tensor, class_rank: torch.Tensor, split_len: int, dev):
    """Every class rank's first and last partial sum: split ``j``'s piece of the class of rank ``r`` is segment ``r +
    j`` (distinct, as classes do not interleave), so a class's pieces are the segments ``lo .. hi``."""
    tr = tile_rank.long()
    k = torch.arange(tr.numel(), device = tr.device)
    num = int(class_rank.max()) + 1
    lo = torch.full((num,), tr.numel(), dtype = torch.long, device = tr.device).scatter_reduce(
        0, tr, k, reduce = "amin")
    hi = torch.zeros(num, dtype = torch.long, device = tr.device).scatter_reduce(0, tr, k, reduce = "amax")
    r = torch.arange(num, device = tr.device)
    return (r + lo // split_len).to(torch.int32).contiguous(), (r + hi // split_len).to(torch.int32).contiguous()


# -------------------------------------------------------------------------------------------------
# Kernels
# -------------------------------------------------------------------------------------------------

@triton_jit
def _partial_kernel(table, ev, shift, tokens, tile_rank, rank_stored, rows, cols, tiles, pair_index, part, logz,
                    num_cats, K, Vl, Be, R_s, C, num_splits, split_len, T, DIRECT: tl.constexpr, TU: tl.constexpr,
                    TN: tl.constexpr, TV: tl.constexpr):
    """One program per (tile, split of the token tiles), splits fastest so that a tile's rows stay in L2: per class,
    the sums ``sum_v p(v) w(v)`` of the tile's rows against its columns' evidence over the split's tokens of the class
    -- linear, the probabilities scaled by 2^64 so that tiny products do not flush --, into ``part`` [segments, Be,
    R_s] at the pairs the nodes read; ``DIRECT`` (one split): each stored class's log mass, into ``part`` [Be, R_s, C]
    (an enumerated class's sum counts in ``log Z`` alone), and every row's ``log Z`` into ``logz`` [Be, R_s]."""
    pid = tl.program_id(0)
    sp = pid % num_splits
    tile = pid // num_splits
    r0, nr = tl.load(tiles + tile * 4), tl.load(tiles + tile * 4 + 1)
    c0, nc = tl.load(tiles + tile * 4 + 2), tl.load(tiles + tile * 4 + 3)
    ru = tl.arange(0, TU)
    rm = ru < nr
    r = tl.load(rows + r0 + ru, mask = rm, other = 0).to(tl.int64)
    cu = tl.arange(0, TN)
    cm = cu < nc
    lv = tl.load(cols + (c0 + cu) * 2, mask = cm, other = 0).to(tl.int64)
    b = tl.load(cols + (c0 + cu) * 2 + 1, mask = cm, other = 0).to(tl.int64)
    m = tl.load(shift + b * Vl + lv, mask = cm, other = 0.0)
    s = tl.load(pair_index + r[:, None] * Vl + lv[None, :], mask = rm[:, None] & cm[None, :], other = -1)
    keep = rm[:, None] & cm[None, :] & (s >= 0)
    if DIRECT:                                                              # the pair's class 0
        dst = part + (b[None, :] * R_s + s) * C
    else:                                                                   # the pair's sum, segment 0
        dst = part + b[None, :] * R_s + s
    evp = ev + b * (Vl * K) + lv * K
    k0 = sp * split_len
    k1 = tl.minimum(k0 + split_len, T)
    cur = tl.load(tile_rank + k0)
    acc = tl.zeros([TU, TN], dtype = tl.float32)
    total = tl.zeros([TU, TN], dtype = tl.float32)
    v_off = tl.arange(0, TV)
    for k in range(k0, k1):
        rk = tl.load(tile_rank + k)
        if rk != cur:                                                       # a class's piece done
            _flush(dst, acc, cur, sp, Be * R_s, m, keep, rank_stored, DIRECT)
            total += acc
            acc = tl.zeros([TU, TN], dtype = tl.float32)
            cur = rk
        v = tl.load(tokens + k * TV + v_off)
        vm = (v >= 0) & (v < K)
        p = tl.load(table + r[:, None] * num_cats + v[None, :], mask = rm[:, None] & vm[None, :], other = 0.0)
        w = tl.exp(tl.load(evp[None, :] + v[:, None], mask = vm[:, None] & cm[None, :], other = float("-inf")) -
                   m[None, :])
        acc += tl.dot(p * 18446744073709551616.0, w, input_precision = "ieee")       # x 2^64
    _flush(dst, acc, cur, sp, Be * R_s, m, keep, rank_stored, DIRECT)
    if DIRECT:                                                              # (one split: every token summed here)
        total += acc
        tl.store(logz + b[None, :] * R_s + s, tl.log(total) - 44.3614195558365 + m[None, :], mask = keep)


@triton.jit
def _flush(dst, acc, cur, sp, seg_stride, m, keep, rank_stored, DIRECT: tl.constexpr):
    """A class's piece of :func:`_partial_kernel` (``cur``: the class's rank among those with a token, see
    :func:`_class_runs`): its sums, at segment ``sp + cur`` -- or ``DIRECT``, its log mass (unscaled, the evidence's
    shift ``m`` added back) at the class's stored column (none for an enumerated class: nothing stored)."""
    if DIRECT:
        col = tl.load(rank_stored + cur)
        tl.store(dst + col, tl.log(acc) - 44.3614195558365 + m[None, :], mask = keep & (col >= 0))
    else:
        tl.store(dst + (sp + cur) * seg_stride, acc, mask = keep)


@triton_jit
def _combine_kernel(part, masses, shift, pair_lv, stored_rank, seg_lo, seg_hi, Vl, Be, R_s, C, TS: tl.constexpr,
                    TC: tl.constexpr):
    """One program per (tile of pairs, tile of stored columns) and sample: every stored class's pieces summed, then
    its log mass (unscaled, the evidence's shift added back; -inf for a class without a token)."""
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    num_ctiles = tl.cdiv(C, TC)
    s = (pid // num_ctiles) * TS + tl.arange(0, TS)
    c = (pid % num_ctiles) * TC + tl.arange(0, TC)
    sm, cm = s < R_s, c < C
    rk = tl.load(stored_rank + c, mask = cm, other = -1)
    has = cm & (rk >= 0)
    lo = tl.load(seg_lo + rk, mask = has, other = 0)
    hi = tl.load(seg_hi + rk, mask = has, other = -1)
    acc = tl.zeros([TS, TC], dtype = tl.float32)
    for g in range(0, tl.max(hi - lo + 1, axis = 0)):
        acc += tl.load(part + ((lo + g)[None, :] * Be + b) * R_s + s[:, None],
                       mask = sm[:, None] & (has & (lo + g <= hi))[None, :], other = 0.0)
    m = tl.load(shift + b * Vl + tl.load(pair_lv + s, mask = sm, other = 0), mask = sm, other = 0.0)
    tl.store(masses + (b * R_s + s[:, None]) * C + c[None, :], tl.log(acc) - 44.3614195558365 + m[:, None],
             mask = sm[:, None] & cm[None, :])                              # 64 ln 2


@triton_jit
def _topk_kernel(table, table_t, num_rows, ev, ids, shift, stored_class, bounds, rows, cols, tiles, pair_index, out,
                 logz, num_cats, K, Vl, R_s, C, NCT, TRANSPOSED: tl.constexpr, BY_TILE: tl.constexpr,
                 TU: tl.constexpr, TK: tl.constexpr, TC: tl.constexpr):
    """One program per (tile, column of the tile: a variable and sample): per stored class, the sums ``sum_k p(v_k)
    w_k`` of the tile's rows over the column's candidates ``v_k`` of the class -- the probabilities read at the
    candidates alone (``TRANSPOSED``: from the token-major table ``table_t``, a candidate's rows contiguous), scaled
    by 2^64 so that tiny products do not flush, dotted with the weights spread over ``TC`` classes -- and their logs
    (the evidence's shift added back; -inf for a class without a candidate), into ``out`` [Be, R_s, C] at the pairs
    the nodes read; and every row's ``log Z``, the sum over every candidate (those of enumerated classes, column -1,
    count there alone), into ``logz`` [Be, R_s]. ``BY_TILE``: ``NCT`` tiles of classes, the candidates sorted by tile
    (``bounds``: where every tile starts, per column; the enumerated classes' candidates before the first), else one."""
    tile = tl.program_id(0)
    j = tl.program_id(1)
    r0, nr = tl.load(tiles + tile * 4), tl.load(tiles + tile * 4 + 1)
    c0, nc = tl.load(tiles + tile * 4 + 2), tl.load(tiles + tile * 4 + 3)
    on = j < nc
    jc = tl.minimum(j, nc - 1)
    lv = tl.load(cols + (c0 + jc) * 2).to(tl.int64)
    b = tl.load(cols + (c0 + jc) * 2 + 1).to(tl.int64)
    col = b * Vl + lv
    m = tl.load(shift + col)
    ru = tl.arange(0, TU)
    rm = ru < nr
    r = tl.load(rows + r0 + ru, mask = rm, other = 0).to(tl.int64)
    s = tl.load(pair_index + r * Vl + lv, mask = rm, other = -1)
    keep = rm & (s >= 0) & on
    dst = out + (b * R_s + s) * C
    cls = tl.arange(0, TC)
    total = tl.zeros([TU], dtype = tl.float32)
    if BY_TILE:                                                             # the enumerated classes' candidates
        k1 = tl.where(on, tl.load(bounds + col * (NCT + 1)), 0)
        for k in range(0, k1, TK):
            kk = k + tl.arange(0, TK)
            km = kk < k1
            v = tl.load(ids + col * K + kk, mask = km, other = 0).to(tl.int64)
            w = tl.exp(tl.load(ev + col * K + kk, mask = km, other = float("-inf")) - m)
            if TRANSPOSED:
                p = tl.load(table_t + v[:, None] * num_rows + r[None, :], mask = km[:, None] & rm[None, :], other = 0.0)
            else:
                p = tl.load(table + r[None, :] * num_cats + v[:, None], mask = km[:, None] & rm[None, :], other = 0.0)
            total += tl.sum(w[:, None] * (p * 18446744073709551616.0), axis = 0)
    for t in range(0, NCT):
        if BY_TILE:
            k0 = tl.load(bounds + col * (NCT + 1) + t)
            k1 = tl.load(bounds + col * (NCT + 1) + t + 1)
        else:
            k0 = 0
            k1 = K
        k1 = tl.where(on, k1, k0)
        acc = tl.zeros([TC, TU], dtype = tl.float32)
        for k in range(k0, k1, TK):
            kk = k + tl.arange(0, TK)
            km = kk < k1
            v = tl.load(ids + col * K + kk, mask = km, other = 0).to(tl.int64)
            w = tl.exp(tl.load(ev + col * K + kk, mask = km, other = float("-inf")) - m)
            c = tl.load(stored_class + v, mask = km, other = -1) - t * TC
            if TRANSPOSED:
                p = tl.load(table_t + v[:, None] * num_rows + r[None, :], mask = km[:, None] & rm[None, :], other = 0.0)
            else:
                p = tl.load(table + r[None, :] * num_cats + v[:, None], mask = km[:, None] & rm[None, :], other = 0.0)
            ps = p * 18446744073709551616.0                                                       # x 2^64
            wc = tl.where(cls[:, None] == c[None, :], w[None, :], 0.0)                            # [TC, TK]
            acc += tl.dot(wc, ps, input_precision = "ieee")
            total += tl.sum(w[:, None] * ps, axis = 0)
        cc = t * TC + cls
        tl.store(dst[None, :] + cc[:, None], tl.log(acc) - 44.3614195558365 + m,
                 mask = keep[None, :] & (cc < C)[:, None])
    tl.store(logz + b * R_s + s, tl.log(total) - 44.3614195558365 + m, mask = keep)


@triton_jit
def _token_weights_kernel(w, x, missing, variables, ev, ids, B, n, Vl, K, PER_SAMPLE: tl.constexpr,
                          TOPK: tl.constexpr, STEPS: tl.constexpr, TB: tl.constexpr, TV: tl.constexpr):
    """One program per (tile of samples, tile of the evidence's variables): ``w[b, v] = log w(d)`` of the token ``d``
    sample ``b`` observes at variable ``variables[v]``, its weight in the evidence's row ``v`` of the sample (of sample 0
    unless ``PER_SAMPLE``; ``TOPK``: the candidate's, found among ``ids``' sorted ones, -inf for none); -inf where the
    token is missing."""
    b = tl.program_id(0) * TB + tl.arange(0, TB)
    v = tl.program_id(1) * TV + tl.arange(0, TV)
    bm, vm = b < B, v < Vl
    mask = bm[:, None] & vm[None, :]
    var = tl.load(variables + v, mask = vm, other = 0).to(tl.int64)
    at = b[:, None].to(tl.int64) * n + var[None, :]
    d = tl.load(x + at, mask = mask, other = 0)
    obs = mask & (tl.load(missing + at, mask = mask, other = 1) == 0)
    be = b.to(tl.int64) if PER_SAMPLE else tl.zeros_like(b).to(tl.int64)
    row = (be[:, None] * Vl + v[None, :]) * K
    if TOPK:
        k = find_token(ids, row, K, d, obs, STEPS)
        val = tl.load(ev + row + k, mask = k >= 0, other = float("-inf"))
    else:
        val = tl.load(ev + row + d, mask = obs & (d < K), other = float("-inf"))
    tl.store(w + b[:, None].to(tl.int64) * Vl + v[None, :], val, mask = mask)


@triton_jit
def _input_values_kernel(out, table, x, missing, vids, prow, lvid, srow, w, logz, first, nodes, B, n, num_cats, Vl,
                         R_s, HAS_EV: tl.constexpr, PER_SAMPLE: tl.constexpr, NORMALIZE: tl.constexpr,
                         TN: tl.constexpr, TB: tl.constexpr):
    """One program per (tile of the layer's nodes, tile of samples): every observed token's ``log p(d)`` (``HAS_EV``:
    ``+ log w(d)``, the sample's at the node's variable from ``w`` (:func:`_token_weights_kernel`), ``- log Z`` if
    ``NORMALIZE``), 0 where it is missing."""
    i = tl.program_id(0) * TN + tl.arange(0, TN)
    b = tl.program_id(1) * TB + tl.arange(0, TB)
    im, bm = i < nodes, b < B
    mask = im[:, None] & bm[None, :]
    var = tl.load(vids + i, mask = im, other = 0).to(tl.int64)
    at = b[None, :].to(tl.int64) * n + var[:, None]
    d = tl.load(x + at, mask = mask, other = 0)
    miss = tl.load(missing + at, mask = mask, other = 1) != 0
    obs = mask & ~miss
    r = tl.load(prow + i, mask = im, other = 0).to(tl.int64)
    val = tl.log(tl.load(table + r[:, None] * num_cats + d, mask = obs, other = 1.0))
    if HAS_EV:
        lv = tl.load(lvid + i, mask = im, other = 0).to(tl.int64)
        wd = tl.load(w + b[None, :].to(tl.int64) * Vl + lv[:, None], mask = obs, other = float("-inf"))
        if NORMALIZE:
            be = b.to(tl.int64) if PER_SAMPLE else tl.zeros_like(b).to(tl.int64)
            z = tl.load(logz + be[None, :] * R_s + tl.load(srow + i, mask = im, other = 0)[:, None], mask = obs,
                        other = 0.0)
            val = tl.where(z == float("-inf"), float("-inf"), val + wd - z)  # no token weighs: an impossible sample
        else:
            val = val + wd
    tl.store(out + (first + i[:, None]).to(tl.int64) * B + b[None, :], tl.where(miss, 0.0, val), mask = mask)
