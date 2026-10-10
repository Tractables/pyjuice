"""
Categorical input nodes: a class's mass is the sum of the probabilities of its tokens (pyjuice keeps Categorical
parameters normalized, in probability space). The class-mass table has a row per row of the layer's ``[rows,
num_cats]`` parameter table, summed once however many tied nodes read it: node ``i`` reads row ``s_pids[i] //
num_cats``.

Two passes over the table, chosen by the number of classes (a fixed rule, so results never depend on timing):

* up to :data:`NATURAL_ORDER_MAX_CLASSES`: the tokens in their natural order, coalesced, every class at once (a
  0/1 class tile built in registers, the products on tensor cores). ``rows x num_cats x classes`` multiply-adds:
  1.08-1.14x a single read of the table on a 4096 x 50257 table at 10-64 classes;
* above: the tokens in class order (:meth:`TokenClasses.by_class`), each chunk of them against a chunk-local
  class tile, then each class's chunks summed. ``rows x num_cats x CLASS_CHUNK`` multiply-adds whatever the
  number of classes: 1.7-1.9x a read of the table at 512-4096 classes (the natural order: 6-54x), 7x when every
  token is its own class.

Both are fp32-accurate: the class tile is exact in TF32, so two TF32 products (of the probabilities' high and low
halves) carry their full precision, and the probabilities are scaled by 2^64 first, because tensor cores flush
subnormal inputs to zero (a class whose tokens are all subnormal would read -inf).
"""

import torch
import triton
import triton.language as tl

#: up to this many classes, the natural-order pass; above, the class-order pass
NATURAL_ORDER_MAX_CLASSES = 64
#: sorted positions per chunk of the class-order pass
CLASS_CHUNK = 32
#: the natural-order pass splits the tokens over programs until there are at least this many (a constant, not the
#: device's SM count, so that the summation order and so the results do not depend on the GPU)
NATURAL_ORDER_PROGRAMS = 1024


def num_values(dist) -> int:
    return dist.num_cats


def class_mass_rows(layer) -> torch.Tensor:
    """The row every node reads: every node of the layer ranges over the same ``num_cats`` values (the compiler
    refuses any other vocabulary), so its parameters are one ``[rows, num_cats]`` table, node ``i``'s at row
    ``s_pids[i] // num_cats``."""
    return (layer.s_pids // layer.dist.num_cats).long()


def class_masses(layer, classes, out = None) -> torch.Tensor:
    """``[rows, num_classes]``: ``log sum_{v: token_class[v] == c} p(v)`` for every row of the parameter table,
    into ``out`` when given (contiguous)."""
    num_cats, C = layer.dist.num_cats, classes.num_classes
    table = layer.params.view(-1, num_cats)
    if out is None:
        out = torch.empty(table.size(0), C, dtype = torch.float32, device = table.device)
    if C <= NATURAL_ORDER_MAX_CLASSES:
        _natural_order(table, classes.token_class, C, out)
    else:
        _class_order(table, classes.by_class(CLASS_CHUNK), C, out)
    return out


# -------------------------------------------------------------------------------------------------
# Kernels
# -------------------------------------------------------------------------------------------------

@triton.jit
def _exact_dot(p, onehot):
    """``p @ onehot`` for a 0/1 ``onehot``, fp32-accurate in two TF32 products: ``p`` = a TF32-exact high half
    (its mantissa cut to 10 bits) + the rest."""
    hi = (p.to(tl.int32, bitcast = True) & -8192).to(tl.float32, bitcast = True)
    return tl.dot(hi, onehot, input_precision = "tf32") + tl.dot(p - hi, onehot, input_precision = "tf32")


@triton.jit
def _natural_order_kernel(table, token_class, out, R, V, C, num_splits, split_len,
                          TU: tl.constexpr, TV: tl.constexpr, TC: tl.constexpr, PARTIAL: tl.constexpr):
    """One program per (tile of rows, split of the tokens): the rows' sums over every class (``PARTIAL``: the
    split's sums, scaled by 2^64, into ``out`` [num_splits, R, C]; else the log masses, into ``out`` [R, C])."""
    pid = tl.program_id(0)
    sp = pid % num_splits
    rows = (pid // num_splits) * TU + tl.arange(0, TU)
    rmask = rows < R
    cls = tl.arange(0, TC)
    base = rows.to(tl.int64) * V
    acc = tl.zeros([TU, TC], dtype = tl.float32)
    v_end = tl.minimum(sp * split_len + split_len, V)
    for v0 in range(sp * split_len, v_end, TV):
        v = v0 + tl.arange(0, TV)
        vmask = v < v_end
        p = tl.load(table + base[:, None] + v[None, :], mask = rmask[:, None] & vmask[None, :], other = 0.0)
        c = tl.load(token_class + v, mask = vmask, other = -1)
        acc += _exact_dot(p * 18446744073709551616.0, tl.where(c[:, None] == cls[None, :], 1.0, 0.0))     # 2^64
    mask = rmask[:, None] & (cls < C)[None, :]
    if PARTIAL:
        tl.store(out + (sp * R + rows.to(tl.int64))[:, None] * C + cls[None, :], acc, mask = mask)
    else:
        tl.store(out + rows.to(tl.int64)[:, None] * C + cls[None, :], tl.log(acc) - 44.3614195558365,   # 64 ln 2
                 mask = mask)


@triton.jit
def _natural_order_combine(part, out, R, C, num_splits, TU: tl.constexpr, TC: tl.constexpr):
    rows = tl.program_id(0) * TU + tl.arange(0, TU)
    cls = tl.arange(0, TC)
    mask = (rows < R)[:, None] & (cls < C)[None, :]
    acc = tl.zeros([TU, TC], dtype = tl.float32)
    for sp in range(num_splits):
        acc += tl.load(part + (sp * R + rows.to(tl.int64))[:, None] * C + cls[None, :], mask = mask, other = 0.0)
    tl.store(out + rows.to(tl.int64)[:, None] * C + cls[None, :], tl.log(acc) - 44.3614195558365, mask = mask)


def _natural_order(table, token_class, C, out, TU = 32, TV = 64):
    R, V = table.shape
    TC = max(16, triton.next_power_of_2(C))
    row_tiles = triton.cdiv(R, TU)
    splits = max(1, min(triton.cdiv(NATURAL_ORDER_PROGRAMS, row_tiles), triton.cdiv(V, TV * 8)))
    split_len = triton.cdiv(triton.cdiv(V, splits), TV) * TV
    splits = triton.cdiv(V, split_len)
    if splits == 1:
        _natural_order_kernel[(row_tiles,)](table, token_class, out, R, V, C, 1, split_len, TU = TU, TV = TV, TC = TC,
                                            PARTIAL = False, num_warps = 4, num_stages = 2)
        return
    part = torch.empty(splits, R, C, dtype = torch.float32, device = table.device)
    _natural_order_kernel[(row_tiles * splits,)](table, token_class, part, R, V, C, splits, split_len, TU = TU,
                                                 TV = TV, TC = TC, PARTIAL = True, num_warps = 4, num_stages = 2)
    _natural_order_combine[(triton.cdiv(R, 16),)](part, out, R, C, splits, TU = 16, TC = TC)


@triton.jit
def _class_order_kernel(table, order, rank, part, R, V, num_chunks, num_segments, TU: tl.constexpr,
                        CHUNK: tl.constexpr):
    """One program per (tile of rows, chunk of sorted positions), chunks fastest so that a row tile stays in L2:
    the rows' sums over each class's piece of the chunk, scaled by 2^64, into ``part`` [R, num_segments]."""
    pid = tl.program_id(0)
    j = pid % num_chunks
    rows = (pid // num_chunks) * TU + tl.arange(0, TU)
    rmask = rows < R
    k = j * CHUNK + tl.arange(0, CHUNK)
    kmask = k < V
    tok = tl.load(order + k, mask = kmask, other = 0)
    r0 = tl.load(rank + j * CHUNK)
    local = tl.where(kmask, tl.load(rank + k, mask = kmask, other = 0) - r0, -1)      # class within the chunk
    p = tl.load(table + rows.to(tl.int64)[:, None] * V + tok[None, :], mask = rmask[:, None] & kmask[None, :],
                other = 0.0)
    idx = tl.arange(0, CHUNK)
    acc = _exact_dot(p * 18446744073709551616.0, tl.where(local[:, None] == idx[None, :], 1.0, 0.0))
    tl.store(part + rows.to(tl.int64)[:, None] * num_segments + (r0 + j + idx)[None, :], acc,
             mask = rmask[:, None] & (idx <= tl.max(local, axis = 0))[None, :])


@triton.jit
def _class_order_combine(part, class_rank, seg_lo, seg_hi, out, R, C, num_segments, TU: tl.constexpr,
                         TC: tl.constexpr):
    """One program per (tile of rows, tile of classes): each class's segments summed, then its log mass."""
    pid = tl.program_id(0)
    num_ctiles = tl.cdiv(C, TC)
    rows = (pid // num_ctiles) * TU + tl.arange(0, TU)
    cls = (pid % num_ctiles) * TC + tl.arange(0, TC)
    rmask, cmask = rows < R, cls < C
    r = tl.load(class_rank + cls, mask = cmask, other = -1)
    has = cmask & (r >= 0)
    lo = tl.load(seg_lo + r, mask = has, other = 0)
    hi = tl.load(seg_hi + r, mask = has, other = -1)
    acc = tl.zeros([TU, TC], dtype = tl.float32)
    for s in range(0, tl.max(hi - lo + 1, axis = 0)):
        acc += tl.load(part + rows.to(tl.int64)[:, None] * num_segments + (lo + s)[None, :],
                       mask = rmask[:, None] & (has & (lo + s <= hi))[None, :], other = 0.0)
    tl.store(out + rows.to(tl.int64)[:, None] * C + cls[None, :], tl.log(acc) - 44.3614195558365,
             mask = rmask[:, None] & cmask[None, :])


def _class_order(table, by_class, C, out, TU = 32):
    R, V = table.shape
    part = torch.empty(R, by_class.num_segments, dtype = torch.float32, device = table.device)
    _class_order_kernel[(triton.cdiv(R, TU) * by_class.num_chunks,)](
        table, by_class.order, by_class.rank, part, R, V, by_class.num_chunks, by_class.num_segments, TU = TU,
        CHUNK = by_class.chunk, num_warps = 4)
    _class_order_combine[(triton.cdiv(R, 16) * triton.cdiv(C, 64),)](
        part, by_class.class_rank, by_class.seg_lo, by_class.seg_hi, out, R, C, by_class.num_segments, TU = 16,
        TC = 64)
