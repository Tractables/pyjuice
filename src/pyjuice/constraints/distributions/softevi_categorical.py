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
"""

import weakref

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from . import SampleInputs
from .categorical import class_mass_rows, class_masses, num_values, token_probs

__all__ = ["KEYWORDS", "num_values", "class_mass_rows", "class_masses", "token_probs", "check_inputs",
           "sample_rows", "sample_inputs", "input_values"]

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


def check_inputs(layer, inputs: dict):
    """The batch size of the layer's soft evidence in ``inputs`` (see :data:`KEYWORDS`), or None if it has none."""
    for key in ("soft_evidence_cat_ids", "soft_evidence_value_mask"):
        if inputs.get(key) is not None:
            raise NotImplementedError(f"`{key}` is not supported under a constraint yet: only dense soft evidence "
                                      f"(`categorical_evidence_logp` alone) is.")
    ev = inputs.get("categorical_evidence_logp")
    if ev is None:
        return None
    num_vars = int(torch.unique(layer.vids.view(-1)).numel())
    if not torch.is_tensor(ev) or ev.dim() != 3 or not ev.is_floating_point():
        raise ValueError(f"`categorical_evidence_logp` must be a float tensor [batch, {num_vars}, K], got "
                         f"{type(ev).__name__ if not torch.is_tensor(ev) else list(ev.shape)}.")
    if ev.device != layer.params.device:
        raise ValueError(f"`categorical_evidence_logp` is on {ev.device}, the circuit on {layer.params.device}.")
    if ev.size(1) < num_vars or ev.size(2) > layer.dist.num_cats or ev.size(0) == 0:
        raise ValueError(f"`categorical_evidence_logp` must be [batch, {num_vars}, K] with K <= "
                         f"{layer.dist.num_cats} (the SoftEvidenceCategorical layer's variables and values), got "
                         f"{list(ev.shape)}.")
    return int(ev.size(0))


def sample_rows(layer, inputs: dict) -> int:
    """The rows per sample of :func:`sample_inputs`' masses: the (variable, parameter row) pairs the nodes read."""
    return _structure(layer, inputs["categorical_evidence_logp"].size(1))["num_pairs"]


def sample_inputs(layer, classes, inputs: dict, out: torch.Tensor) -> SampleInputs:
    """
    The class masses the layer's soft evidence gives its nodes (see the module docstring), into ``out`` [batch,
    :func:`sample_rows`, num_classes]: a row per (variable, parameter row) pair some node reads, per sample, ``log
    sum_{v in c} p(v) w(v) - log Z``, with ``log Z`` per row and sample in ``data`` (what an observed token's value
    subtracts, :func:`input_values`). One pass over the parameter table: every row is read once for all the variables
    it serves (tied rows: every variable), every token class's tokens against the evidence of those variables, fp32
    -- split over the tokens for enough programs, each split's sums kept apart while they fit
    :data:`MASS_PARTIAL_BUDGET`, else in one split, written straight.
    """
    ev = inputs["categorical_evidence_logp"].float().contiguous()
    Be, Vl, K = ev.shape
    st = _structure(layer, Vl)
    C = classes.num_classes
    dev = ev.device
    # the evidence's shift per (sample, variable): the max weight scales to 1 (a row of -inf weighs nothing anyway)
    shift = ev.amax(dim = 2)
    shift = torch.where(torch.isfinite(shift), shift, 0.0).contiguous()
    cols, tiles = _columns(st, Be, dev)
    tc, TV = MASS_TILES, MASS_TILES["TV"]
    tokens, tile_rank, class_rank, rank_class = _class_runs(classes, TV)
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
    else:
        part = torch.empty(present + splits, Be, R_s, dtype = torch.float32, device = dev)
    _partial_kernel[(tiles.size(0) * splits,)](
        table, ev, shift, tokens, tile_rank, rank_class, st["rows"], cols, tiles, st["pair_index"], part,
        layer.dist.num_cats, K, Vl, Be, R_s, C, splits, split_len, T, DIRECT = direct, TU = tc["TU"], TN = tc["TN"],
        TV = TV, num_warps = 4)
    if not direct:
        seg_lo, seg_hi = _segments(tile_rank, class_rank, split_len, dev)
        TS, TC = 32, max(16, min(64, host_next_power_of_2(C)))
        _combine_kernel[(host_cdiv(R_s, TS) * host_cdiv(C, TC), Be)](
            part, out, shift, st["pair_lv"], class_rank, seg_lo, seg_hi, Vl, Be, R_s, C, TS = TS, TC = TC)
        del part
    logz = torch.logsumexp(out, dim = 2)
    out -= torch.where(torch.isfinite(logz), logz, 0.0)[:, :, None]         # q's classes: they sum to one
    return SampleInputs(masses = out, rows = st["node_rows"], batch = Be, variables = st["variables"],
                        data = dict(ev = ev, logz = logz.contiguous()))


def input_values(layer, x: torch.Tensor, missing: torch.Tensor, out: torch.Tensor, sample = None):
    """``log q(d)`` of every observed token ``d`` at every node of the layer, into its rows of ``out`` (0 where the
    token is missing): ``log p(d)``, and under soft evidence (``sample``, :func:`sample_inputs`') ``+ log w(d) - log
    Z``."""
    B, n = x.shape
    first, end = layer._output_ind_range
    nodes = end - first
    num_cats = layer.dist.num_cats
    table = layer.params.view(-1, num_cats)
    prow = _rows32(layer)
    has_ev = sample is not None
    if has_ev:
        ev, logz = sample.data["ev"], sample.data["logz"]
        lvid, srow, Be, Vl, K, R_s = _lvid32(layer), sample.rows.to(torch.int32), sample.batch, ev.size(1), \
            ev.size(2), sample.masses.size(1)
    else:
        ev = logz = lvid = srow = prow
        Be = Vl = K = R_s = 1
    TN, TB = 32, max(16, min(64, host_next_power_of_2(B)))
    _input_values_kernel[(host_cdiv(nodes, TN), host_cdiv(B, TB))](
        out, table, x, missing.view(torch.uint8), layer.vids, prow, lvid, srow, ev, logz, first, nodes, B, n,
        num_cats, Vl, K, R_s,
        HAS_EV = has_ev, PER_SAMPLE = Be > 1, TN = TN, TB = TB)


# -------------------------------------------------------------------------------------------------
# What the layer's structure fixes (kept per layer)
# -------------------------------------------------------------------------------------------------

_STRUCTURES = weakref.WeakKeyDictionary()                  # layer -> {Vl: structure}


def _rows32(layer) -> torch.Tensor:
    return _cached(layer, "prow", lambda: class_mass_rows(layer).to(torch.int32).contiguous())


def _lvid32(layer) -> torch.Tensor:
    return _cached(layer, "lvid", lambda: layer.var_idmapping[layer.vids.view(-1).long()].to(torch.int32).contiguous())


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
                    pair_lv = (uniq % Vl).to(torch.int32).contiguous(), groups = groups,
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
    """The tokens in class order, every class's run padded to whole tiles of ``TV`` (:meth:`TokenClasses.class_tiles`),
    every tile's class's rank among the classes with a token, every class's rank (-1 for none) and every rank's class;
    kept with the classes."""
    key = ("softevi_runs", TV)
    found = classes._class_tiles.get(key)
    if found is None:
        tokens, tile_class = classes.class_tiles(TV)
        tc = tile_class.long()
        present = torch.zeros(classes.num_classes, dtype = torch.bool, device = tc.device)
        present[tc] = True
        rank = torch.cumsum(present.long(), 0) - 1
        class_rank = torch.where(present, rank, -1).to(torch.int32).contiguous()
        rank_class = torch.nonzero(present).view(-1).to(torch.int32).contiguous()
        found = classes._class_tiles[key] = (tokens, rank[tc].to(torch.int32).contiguous(), class_rank, rank_class)
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
def _partial_kernel(table, ev, shift, tokens, tile_rank, rank_class, rows, cols, tiles, pair_index, part, num_cats, K,
                    Vl, Be, R_s, C, num_splits, split_len, T, DIRECT: tl.constexpr, TU: tl.constexpr, TN: tl.constexpr,
                    TV: tl.constexpr):
    """One program per (tile, split of the token tiles), splits fastest so that a tile's rows stay in L2: per class,
    the sums ``sum_v p(v) w(v)`` of the tile's rows against its columns' evidence over the split's tokens of the class
    -- linear, the probabilities scaled by 2^64 so that tiny products do not flush --, into ``part`` [segments, Be,
    R_s] at the pairs the nodes read; ``DIRECT`` (one split): each class's log mass, into ``part`` [Be, R_s, C]."""
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
    v_off = tl.arange(0, TV)
    for k in range(k0, k1):
        rk = tl.load(tile_rank + k)
        if rk != cur:                                                       # a class's piece done
            _flush(dst, acc, cur, sp, Be * R_s, m, keep, rank_class, DIRECT)
            acc = tl.zeros([TU, TN], dtype = tl.float32)
            cur = rk
        v = tl.load(tokens + k * TV + v_off)
        vm = (v >= 0) & (v < K)
        p = tl.load(table + r[:, None] * num_cats + v[None, :], mask = rm[:, None] & vm[None, :], other = 0.0)
        w = tl.exp(tl.load(evp[None, :] + v[:, None], mask = vm[:, None] & cm[None, :], other = float("-inf")) -
                   m[None, :])
        acc += tl.dot(p * 18446744073709551616.0, w, input_precision = "ieee")       # x 2^64
    _flush(dst, acc, cur, sp, Be * R_s, m, keep, rank_class, DIRECT)


@triton.jit
def _flush(dst, acc, cur, sp, seg_stride, m, keep, rank_class, DIRECT: tl.constexpr):
    """A class's piece of :func:`_partial_kernel` (``cur``: the class's rank among those with a token, see
    :func:`_class_runs`): its sums, at segment ``sp + cur`` -- or ``DIRECT``, its log mass (unscaled, the evidence's
    shift ``m`` added back) at the class itself."""
    if DIRECT:
        tl.store(dst + tl.load(rank_class + cur), tl.log(acc) - 44.3614195558365 + m[None, :], mask = keep)
    else:
        tl.store(dst + (sp + cur) * seg_stride, acc, mask = keep)


@triton_jit
def _combine_kernel(part, masses, shift, pair_lv, class_rank, seg_lo, seg_hi, Vl, Be, R_s, C, TS: tl.constexpr,
                    TC: tl.constexpr):
    """One program per (tile of pairs, tile of classes) and sample: every class's pieces summed, then its log mass
    (unscaled, the evidence's shift added back; -inf for a class without a token)."""
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    num_ctiles = tl.cdiv(C, TC)
    s = (pid // num_ctiles) * TS + tl.arange(0, TS)
    c = (pid % num_ctiles) * TC + tl.arange(0, TC)
    sm, cm = s < R_s, c < C
    rk = tl.load(class_rank + c, mask = cm, other = -1)
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
def _input_values_kernel(out, table, x, missing, vids, prow, lvid, srow, ev, logz, first, nodes, B, n, num_cats, Vl,
                         K, R_s, HAS_EV: tl.constexpr, PER_SAMPLE: tl.constexpr, TN: tl.constexpr, TB: tl.constexpr):
    """One program per (tile of the layer's nodes, tile of samples): every observed token's ``log p(d)`` (``HAS_EV``:
    ``+ log w(d) - log Z``), 0 where it is missing."""
    i = tl.program_id(0) * TN + tl.arange(0, TN)
    b = tl.program_id(1) * TB + tl.arange(0, TB)
    im, bm = i < nodes, b < B
    mask = im[:, None] & bm[None, :]
    var = tl.load(vids + i, mask = im, other = 0).to(tl.int64)
    at = b[None, :].to(tl.int64) * n + var[:, None]
    d = tl.load(x + at, mask = mask, other = 0)
    miss = tl.load(missing + at, mask = mask, other = 1) != 0
    r = tl.load(prow + i, mask = im, other = 0).to(tl.int64)
    val = tl.log(tl.load(table + r[:, None] * num_cats + d, mask = mask & ~miss, other = 1.0))
    if HAS_EV:
        be = b.to(tl.int64) if PER_SAMPLE else tl.zeros_like(b).to(tl.int64)
        lv = tl.load(lvid + i, mask = im, other = 0).to(tl.int64)
        w = tl.load(ev + (be[None, :] * Vl + lv[:, None]) * K + d, mask = mask & ~miss & (d < K),
                    other = float("-inf"))
        z = tl.load(logz + be[None, :] * R_s + tl.load(srow + i, mask = im, other = 0)[:, None], mask = mask,
                    other = 0.0)
        val = tl.where(z == float("-inf"), float("-inf"), val + w - z)     # no token weighs: an impossible sample
    tl.store(out + (first + i[:, None]).to(tl.int64) * B + b[None, :], tl.where(miss, 0.0, val), mask = mask)
