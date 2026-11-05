"""
The top-down ``input @ block`` kernel in CUDA (``c/walk.cu``): when it is taken, the boundaries it walks and what it stages.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch

from ..common import NODE_TILE, pack_aligned
from ..transitions import enumerated_args
from .launches import TopDownLaunch


#: ``input @ block`` launches on compact values run the CUDA kernels (``c/walk.cu``) when those take them
#: (:func:`_input_block_cuda`); False: the Triton kernel always
INPUT_BLOCK_CUDA = True
#: the most children of a node tile a CTA of the CUDA kernel takes (a node tile's at most), and the most dynamic
#: shared memory (a sample's lanes of its children's parents, their class masses): past it, CTAs of fewer -- and past
#: it for one child, CTAs of one child with as much as the device takes. A choice, not the device's limit: the kernels
#: raise that themselves for whatever a launch takes (``c/walk.cu``, ``kernel_allowed``)
INPUT_BLOCK_CUDA_NODES = NODE_TILE
INPUT_BLOCK_CUDA_SMEM = 45 * 1024
#: (lane, 4 children) items of a sample from which its CTA has 256 threads, not 128; up to which a CTA takes 8 samples
#: instead, a warp each, their columns not ordered by their terms (most of a CTA's threads would idle: left to right,
#: B 64, 61 -> 32 us a call)
INPUT_BLOCK_CUDA_WIDE = 512
INPUT_BLOCK_CUDA_WARP_ITEMS = 64
#: the most int32 of work space a launch takes (every sample's copy of its boundaries' predecessor lists): past it, the
#: Triton kernel
INPUT_BLOCK_CUDA_SPACE = 1 << 26
#: the codes of a unit's class masses (a sample's, at a CTA's children) the CUDA kernel's CTAs stage -- its terms' alone,
#: by rank --: a unit reading more (or a kept-once parent's every entry) is summed by the deferred kernel, every code
#: staged (`c/walk.cu`). Codpula-4096 x JSON hard_0 B64 diffusion (1224 codes, a unit's terms 43 at most): every code
#: staged took 39.5 KB a CTA, 2 CTAs an SM, the launches of a call 91-104 ms; 32 / 64 / 128 / 256 / 512 codes 26-29 /
#: 26-29 / 33-36 / 33-37 / 45-51 ms (8 samples a CTA, a warp each, up to 128)
INPUT_BLOCK_CUDA_CODES = 64
#: the deferred kernel's CTAs (0: as many as the device holds at once, at most one a deferred unit)
INPUT_BLOCK_CUDA_DEFER_CTAS = 0


def _walk_boundaries(prog) -> None:
    """Into ``host["walk"]`` of every ``input @ block`` launch of ``prog``'s :class:`~..compiled.circuit.Assembly`, what
    :func:`_input_block_boundaries` says whatever the automaton (a launch's dict, the same for its program of any
    constraint), in one copy to the device: copies wait for the stream."""
    template = prog.assembly.top_down
    kids = [c for kind, _, layers in template.steps if kind == "prod" for _, stages, _, _ in layers for out, _, _, _ in stages
            for c in out if c.form == "input_block"]
    n = prog.n
    ts = [np.unique(c.host["parents"][:, 2]).astype(np.int64) for c in kids]
    arrays = []
    for x in ts:
        t2i = np.full(n, -1, dtype = np.int32)
        t2i[x] = np.arange(x.size, dtype = np.int32)
        arrays += [x.astype(np.int32), t2i]
    buf, starts = pack_aligned(arrays, np.int32)
    tables, starts = torch.from_numpy(buf).to(prog.dev), starts.tolist()
    for k, (c, x) in enumerate(zip(kids, ts)):
        ends = prog.iv_ends_np[np.asarray(c.host["child_ivs"], dtype = np.int64)]
        c.host["walk"].update(at = k, ts = x, tlist = tables[starts[2 * k]:starts[2 * k] + x.size],
                              t2i = tables[starts[2 * k + 1]:starts[2 * k + 1] + n], ta = ends[:, 0] - 1, te = ends[:, 1])
    sizes = np.fromiter((x.size for x in ts), dtype = np.int64, count = len(ts))
    runs = np.zeros(len(ts) + 1, dtype = np.int64)                     # (every walk's entries, 16 bytes in and one more)
    np.cumsum(-(-(sizes + 1) // 2) * 2, out = runs[1:])
    flat = np.concatenate(ts) if ts else np.zeros(0, dtype = np.int64)
    run = np.repeat(np.arange(len(ts)), sizes)
    within = np.arange(flat.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    template.shared["walk_entries"] = (flat, run, runs[run] + within + 1, np.cumsum(sizes) - sizes, runs, sizes.tolist())


def _input_block_boundaries(la: TopDownLaunch, prog) -> tuple:
    """An ``input @ block`` launch's boundaries for the CUDA kernels: ``host["walk"]`` (:func:`_walk_boundaries`, whatever
    the automaton: ``ts``, its parents' ``t``, ascending, ``tlist`` int32 on the device, ``t2i``, int32 [n] on the device:
    every one's place among them, and every child's interval as ``(t, t2)``, ``ta`` and ``te``), and what the automaton
    says: ``ecum`` (int64 [T + 1]) and ``ecum_t`` (the same on the device), the predecessor entries at the boundaries
    before -- a program's launches' together, in one copy."""
    walk = la.host["walk"]
    if not walk:
        _walk_boundaries(prog)
    kept = getattr(prog, "walk_entries", None)
    if kept is None:
        n, W = prog.n, prog.W
        pp = prog.ungrouped_tables().pred_ptr.view(n, W + 1).cpu().numpy().astype(np.int64)   # (the CUDA walk runs
        counts = pp[:, W] - pp[:, 0]                                   # ungrouped: every boundary's predecessors)
        flat, run, places, firsts, runs, sizes = prog.assembly.top_down.shared["walk_entries"]
        entries = counts[flat]
        total = np.cumsum(entries)                                     # (every walk's sums: all of them, less those before)
        buf = np.zeros(int(runs[-1]), dtype = np.int64)
        buf[places] = total - (total[firsts[run]] - entries[firsts[run]])
        kept = prog.walk_entries = (buf, torch.from_numpy(buf).to(prog.dev), runs.tolist(), sizes)
    buf, tables, starts, sizes = kept
    k = walk["at"]
    first, last = starts[k], starts[k] + sizes[k] + 1
    return walk, buf[first:last], tables[first:last]


def _work_space(prog, size: int, counters: int):
    """``(scratch, counters)``: int32 scratch of at least ``size`` on ``prog``'s device, and ``counters`` int32 zeros
    (each launch leaves them zero), both kept for the next launch (launches on one stream use them one after the
    other)."""
    ws, done = getattr(prog, "walk_space", None), getattr(prog, "walk_done", None)
    if ws is None or ws.numel() < size:
        ws = prog.walk_space = torch.empty(max(size, 1 << 16), dtype = torch.int32, device = prog.dev)
    if done is None or done.numel() < counters:
        done = prog.walk_done = torch.zeros(max(counters, 1 << 10), dtype = torch.int32, device = prog.dev)
    return ws, done


def _defer_space(prog, size: int) -> torch.Tensor:
    """The CUDA walk's list of deferred units: int32 of at least ``size`` on ``prog``'s device, zeros where it counts
    them (each launch leaves them zero), kept for the next launch."""
    d = getattr(prog, "walk_defer", None)
    if d is None or d.numel() < size:
        d = prog.walk_defer = torch.zeros(max(size, 1 << 12), dtype = torch.int32, device = prog.dev)
    return d


def _ibtd_smem_max(mod) -> int:
    """The most dynamic shared memory of a CTA of the CUDA kernel of one child on the current device (kept per
    device)."""
    dev = torch.cuda.current_device()
    kept = _IBTD_SMEM_MAX.get(dev)
    if kept is None:
        kept = _IBTD_SMEM_MAX[dev] = int(mod.ibtd_smem_max())
    return kept


_IBTD_SMEM_MAX = {}


def _input_block_cuda(la: TopDownLaunch, tiles: torch.Tensor, height: int, prog, compact, element_flows, temp_flows,
                      node_flows, node_mars, masses, obs_class, once_iv, common: dict, B: int,
                      allowed: Optional[torch.Tensor] = None) -> Optional[dict]:
    """
    Run an ``input @ block`` launch on compact values with the CUDA kernels (``c/walk.cu``: every sample's live terms
    per column, its columns by their terms, then a CTA per (node tile's children, sample) summing them from shared
    memory), if they take it -- ungrouped transitions, a term's place and class in 32 bits, its CTAs' shared memory
    within :data:`INPUT_BLOCK_CUDA_SMEM` (or a CTA of one child within the device's), its work space within
    :data:`INPUT_BLOCK_CUDA_SPACE` --, and return the CTAs' shape (``tn`` children, ``threads``, ``warps``: a warp per
    sample, ``smem``: bytes), else None. The values are the Triton kernel's, bit for bit. ``allowed``: None, or the
    classes a session's per-sample inputs leave each sample (:attr:`~.live.Liveness.allowed`): where a token is missing,
    the terms of a class ruled out (-inf, nothing to add) are not summed.
    """
    if not INPUT_BLOCK_CUDA or prog.trans_top_down.grouped or not obs_class.is_contiguous():
        return None
    from ..c import get_module
    mod = get_module()
    n, W, C = prog.n, prog.W, prog.C
    ea = enumerated_args(prog, masses)
    ptr = prog.enum.ptr_host
    mw = int(ea["Cs"]) + (int((ptr[1:] - ptr[:-1]).max()) if prog.enum.num > 0 else 0)   # a child's staged masses:
    xb = max(1, (mw - 1).bit_length())                      # the stored columns, a boundary's most enumerated tokens
    if mod is None or xb >= 32 or W >= 1 << (32 - xb) or B > 65535:
        return None
    st, ecum, ecum_t = _input_block_boundaries(la, prog)
    mc = compact.max_count
    cq = int(mc[st["ts"] + 1].max())                                         # a sample's most live columns at t + 1
    lp = max(int((mc[st["ta"]] * mc[st["te"]]).max()), 1)                   # ... lanes of a parent's row
    team = lambda tn: (4 * tn * (lp + mw + 1) + 15) // 16 * 16             # a sample's shared memory (bytes:
    ms = min(mw, INPUT_BLOCK_CUDA_CODES)                                    # c/walk.cu's `ibtd_smem`), and its
    staged = lambda tn: (4 * tn * (lp + ms + 1) + 15) // 16 * 16           # CTAs' (`ibtd_smem_compact`)
    tn = min(height, INPUT_BLOCK_CUDA_NODES)
    while tn > 1 and team(tn) > INPUT_BLOCK_CUDA_SMEM:
        tn //= 2
    if team(tn) > INPUT_BLOCK_CUDA_SMEM and team(tn) > _ibtd_smem_max(mod):  # (one child: the device's most)
        return None
    items = cq * int(mc[st["te"]].max()) * max(tn // 4, 1)                  # a sample's most (lane, 4 children)
    warps = items <= INPUT_BLOCK_CUDA_WARP_ITEMS and 8 * staged(tn) <= INPUT_BLOCK_CUDA_SMEM
    threads = 256 if warps or items >= INPUT_BLOCK_CUDA_WIDE else 128
    T, nterms = st["ts"].size, B * int(ecum[-1])
    space = nterms + 2 * T * B * W + T * B * (mw + 1)                       # (terms, cnt, perm, codes, ncodes)
    if space > INPUT_BLOCK_CUDA_SPACE or 4 * (2 * ((mw + 31) // 32) + 1) > 48 * 1024:
        return None
    ws, done = _work_space(prog, space, T * B)
    defer = _defer_space(prog, 2 + 2 * (tiles.numel() // NODE_TILE) * (height // tn) * B)
    tt, ivs = prog.trans_top_down, prog.intervals
    c_kind, out_temp, (c_full, out_full) = la.kinds
    sample = masses.batch > 1
    classes = allowed is not None and tt.pred_class is not None
    a_stride = n * C if classes and allowed.size(0) > 1 else 0               # (one set: every sample's)
    mod.ibtd_cuda(element_flows, temp_flows, node_flows, node_mars, obs_class, masses.table, masses.row,
                  masses.stride if sample else masses.row, tt.pred_ptr, tt.pred_col, tt.pred_x, prog.width_t, ivs.of,
                  ivs.info, ivs.table, once_iv, compact.off, compact.count, compact.start, compact.columns,
                  compact.rank, la.children, tiles, la.ptr, la.parents, common["reg_offset"], common["reg_width"],
                  common["reg_first"], st["tlist"], st["t2i"], ecum_t, ws[:nterms],
                  ws[nterms:nterms + T * B * W], ws[nterms + T * B * W:nterms + 2 * T * B * W], done,
                  ws[nterms + 2 * T * B * W:space - T * B], ws[space - T * B:space],
                  common["out_first"], common["out_width"], common["temp_width"], B, n, W, C, prog.input_start,
                  height // tn, tn, threads, warps, not warps, lp, cq, sample, c_kind, out_temp, c_full, out_full,
                  ea["code"], ea["tok_ptr"], ea["eidx"], ea["eprobs"], ea["prow"], ea["lognorm"], ea["enum_logw"],
                  ea["Cs"], mw, ea["num_enum"], ea["logw_stride"], ea["LOGW"],
                  tt.pred_class if classes else tt.pred_x, allowed if classes else obs_class, a_stride, classes,
                  ms, defer, INPUT_BLOCK_CUDA_DEFER_CTAS)
    return dict(tn = tn, threads = threads, warps = warps, smem = staged(tn) * (8 if warps else 1), codes = ms)
