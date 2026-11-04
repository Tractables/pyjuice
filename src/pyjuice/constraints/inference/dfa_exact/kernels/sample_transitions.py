"""
Every sample's own transition masses (:func:`sample_transition_masses`): for every input step of a query under
per-sample class masses, every entry of the sample's successor lists (:class:`~.transitions.SampleMasses`), the
log-sum of the row's masses over the classes the sample allows that lead along the entry.

MEASURED (ncu, Codpula-4096 x JSON hard_0 B8, l2r call 0: 854 M masses, 2.56 G class terms): the kernel this replaces
(until 2026-10-30, in :mod:`.transitions`) took 48.8 ms
with 31 G instructions on 255 registers (17% occupancy) -- a [128 rows, 32 entries] tile a program on 128 threads,
every term computing the log of its token's probability and both a stored class's and an enumerated class's mass,
each step a two-exp log-sum. Here a program takes ``TR`` of the tile's rows and ``TQ`` entries a step (registers for
more programs), the steps are :func:`~.prod._lse_step_one_exp` (the two-exp step's floats with one exp), and the
tokens' log-probabilities come from a table of them built once a call (:func:`_log_table_kernel`: the same ``tl.log``
of the same floats). Every class's code at its boundary comes resolved (``SampleMasses.xcode`` / ``wsel``, from
:func:`~.transitions.sample_classes`): one load before a class's mass, not three. And a step's entries are taken in
the order of their numbers of classes (:func:`entry_order`) -- an entry's classes are 3 on average and up to 31 (JSON
hard_0), so in entry order a step looped to the most of its ``TQ`` entries' with a fifth of its lanes busy. Every
entry's classes are still summed one after another in their order, so every mass is the old kernel's own (the
decoder's conditionals the same bits). Same launch: 39.5 (the old kernel) -> 12.7 (lighter tiles) -> 7.6 ms (codes
resolved, entries by class count); the floor of its bytes (3.42 GB of masses written, 1.26 GB of class masses
read) ~3.2 ms.
"""

from __future__ import annotations

import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

from .common import grid_chunks
from .prod import _lse_step_one_exp
from .transitions import SampleMasses, enum_logw_stride

#: :func:`_sample_transition_kernel`'s tiles: the rows of a job tile a program takes (dividing the job's tile), entries
#: a step, warps, and the class loop's unrolling (MEASURED on the launch above, ncu: bound by its loads' latency --
#: long-scoreboard stalls 9.7 of ~15 cycles an instruction, 72 registers, 58% occupancy; unrolled twice, two classes'
#: loads in flight: 6.99 -> 6.33 ms, 3 or 4 times 6.7 / 7.1, Triton's pipelining 25-28; tiles at 2: 16 x 16 8.5, 32 x
#: 32 6.8, 8 x 32 7.0, 8 x 64 6.9, 16 x 64 8.8)
SAMPLE_TRANSITION_TILES = dict(TR = 16, TQ = 32, warps = 4, unroll = 2)


@triton_jit
def _log_table_kernel(src, dst, numel, BLOCK: tl.constexpr):
    """``dst = log(src)`` elementwise."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < numel
    tl.store(dst + i, tl.log(tl.load(src + i, mask = m, other = 1.0)), mask = m)


@triton_jit
def _sample_transition_kernel(class_mars, mass_row, mass_stride, trans, rows, tiles, sptr, sbase, scount, cls_ptr,
                              xcode, ew, perm, elogp, prow, lognorm, width, W, n, Cs, num_enum, input_start, E, pid0,
                              SPLITS, TR: tl.constexpr, TQ: tl.constexpr, SAMPLE: tl.constexpr, ENUM: tl.constexpr,
                              LOGW: tl.constexpr, UNROLL: tl.constexpr):
    """
    One program per (rows ``split * TR ..`` of a job tile of rows at one position ``t``, sample ``b``; program ``tile *
    SPLITS + split``): the sample's entries at boundary ``t`` (``sptr`` [Be, n, W + 1], its lists' pointers), ``TQ`` at
    a time in the order ``perm`` (by their numbers of classes, so that a step's entries loop alike) -- each the
    log-sum, in the classes' order, over the classes the sample allows that lead along it (``cls_ptr``, their codes
    ``xcode``: see :class:`~.transitions.SampleMasses`) of the row's mass of the class: a stored class's from the row of
    the class masses (``SAMPLE``: the sample's row), an enumerated class's (``ENUM``, code ``-1 - j``) its token's
    log-probability under the row's parameter row (``elogp``, column ``j``), plus its weight ``ew`` less the row's
    ``log Z`` (``LOGW``) --, stored at ``trans[sbase[b, t] + rank * scount[b, t] + e]`` for the entry's place ``e`` in
    the sample's list at ``t``.
    """
    pid = pid0 + tl.program_id(0)
    tile = pid // SPLITS
    split = pid % SPLITS
    b = tl.program_id(1).to(tl.int64)
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = split * TR + tl.arange(0, TR)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    rk = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)   # the step's rank at t
    row = tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    if SAMPLE:
        row = row + b * tl.load(mass_stride + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    masses = class_mars + row * Cs
    if ENUM:
        pr = tl.load(prow + row, mask = rmask, other = 0).to(tl.int64)
        if LOGW:
            lz = tl.load(lognorm + row, mask = rmask, other = 0.0)
    bt = b * n + t
    e0 = tl.load(sptr + bt * (W + 1))                                        # the sample's entries at t
    e1 = tl.load(sptr + bt * (W + 1) + tl.load(width + t))
    out = trans + tl.load(sbase + bt) + rk[:, None] * tl.load(scount + bt).to(tl.int64)
    jj = tl.arange(0, TQ)
    if split * TR < count:
        for q0 in range(e0, e1, TQ):
            em = q0 + jj < e1
            q = tl.load(perm + b * E + q0 + jj, mask = em, other = 0)        # the step's entries
            f0 = tl.load(cls_ptr + b * (E + 1) + q, mask = em, other = 0)    # ... their allowed classes
            f1 = tl.load(cls_ptr + b * (E + 1) + q + 1, mask = em, other = 0)
            mx = tl.full([TR, TQ], float("-inf"), dtype = tl.float32)
            acc = tl.zeros([TR, TQ], dtype = tl.float32)
            for k in tl.range(0, tl.max(f1 - f0, axis = 0), loop_unroll_factor = UNROLL):
                km = em & (f0 + k < f1)
                x = tl.load(xcode + f0 + k, mask = km, other = 0)
                miss = rmask[:, None] & km[None, :]
                if ENUM:
                    enum = (x < 0)[None, :]
                    stored = tl.load(masses[:, None] + tl.maximum(x, 0)[None, :], mask = miss & ~enum,
                                     other = float("-inf"))
                    val = tl.load(elogp + pr[:, None] * num_enum + tl.maximum(-1 - x, 0).to(tl.int64)[None, :],
                                  mask = miss & enum, other = float("-inf"))
                    if LOGW:
                        val = val + tl.load(ew + f0 + k, mask = km & (x < 0), other = 0.0)[None, :] - lz[:, None]
                    mass = tl.where(enum, val, stored)
                else:
                    mass = tl.load(masses[:, None] + x[None, :], mask = miss, other = float("-inf"))
                mx, acc = _lse_step_one_exp(mx, acc, mass)
            val = tl.log(acc) + mx
            tl.store(out + (q - e0)[None, :], val, mask = rmask[:, None] & em[None, :])


def entry_order(layout, prog) -> torch.Tensor:
    """int32 [Be, E]: every sample's entries at every boundary by their numbers of classes, the most first (ties in
    entry order) -- what :func:`_sample_transition_kernel` takes ``TQ`` at a time."""
    Be, n, W = layout.Be, prog.n, prog.W
    E = layout.rank.size(1)
    cp = layout.cls_ptr.long()
    ncls = cp[:, 1:] - cp[:, :-1]                                            # [Be, E] (0 past the sample's)
    starts = layout.ptr.view(Be, n, W + 1)[:, :, 0].long().contiguous()      # every boundary's first entry
    j = torch.arange(E, device = cp.device).expand(Be, E).contiguous()
    t = torch.searchsorted(starts, j, right = True) - 1                      # every entry's boundary
    num = layout.ptr.view(Be, n, W + 1)[:, n - 1:, :].long().gather(
        2, prog.width_t[n - 1:n].long().view(1, 1, 1).expand(Be, 1, 1))[:, 0]   # [Be, 1]: the sample's entries
    t = torch.where(j < num, t, n)                                           # (past them: last)
    order = torch.argsort(t * (1 << 32) - ncls, dim = 1, stable = True)
    return order.to(torch.int32).contiguous()


def sample_transition_masses(trans, job, masses, prog, layout: SampleMasses):
    """Write every sample's transition masses over its successor entries (``layout``, a
    :class:`~.transitions.SampleMasses`), for the steps of ``job`` (from :func:`~.transitions.transition_job`), from the
    class masses ``masses``: see :func:`_sample_transition_kernel`."""
    rows, tiles, TS = job
    cfg = SAMPLE_TRANSITION_TILES
    enum = prog.enum.num > 0
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    TR = min(cfg["TR"], TS)
    splits = host_cdiv(TS, TR)
    if enum:                                                    # the tokens' log-probabilities, once a call
        elogp = torch.empty_like(eprobs)
        BLOCK = 1024
        _log_table_kernel[(host_cdiv(eprobs.numel(), BLOCK),)](eprobs, elogp, eprobs.numel(), BLOCK = BLOCK)
    else:
        elogp = masses.table
    ew = masses.table
    if logw:                                                    # every class's weight (its sample's, its token's)
        items = torch.arange(layout.xcode.numel(), device = trans.device, dtype = torch.int32)
        b = torch.searchsorted(layout.cls_ptr[:, 0].contiguous(), items, right = True).long() - 1    # its sample
        at = b * enum_logw_stride(masses) + layout.wsel.long().clamp(min = 0)
        ew = torch.where(layout.wsel >= 0, masses.enum_logw.view(-1)[at], 0.0).contiguous()
    perm = entry_order(layout, prog)
    for first, size in grid_chunks(tiles.size(0) * splits, 0):
        _sample_transition_kernel[(size, layout.Be)](
            masses.table, masses.row, masses.stride if masses.batch > 1 else masses.row, trans, rows, tiles,
            layout.ptr, layout.base, layout.count, layout.cls_ptr, layout.xcode, ew, perm, elogp, masses.prow,
            masses.lognorm if logw else masses.table, prog.width_t, prog.W, prog.n, masses.table.size(1),
            eprobs.size(1), prog.input_start, layout.rank.size(1), first, splits,
            TR = TR, TQ = cfg["TQ"], SAMPLE = masses.batch > 1, ENUM = enum, LOGW = logw, UNROLL = cfg["unroll"],
            num_warps = cfg["warps"])
