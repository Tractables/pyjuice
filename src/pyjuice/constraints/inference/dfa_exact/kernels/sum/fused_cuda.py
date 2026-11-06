"""
The CUDA forms of the fused sum, both directions (``c/fused_sum.cu``): when they are taken, what they read, their launches.
"""

import hashlib
from typing import Optional

import torch

from pyjuice.utils.util import host_cdiv

from . import tile_rules
from .tile_rules import _num_sms, pre_buffer, split_buffer


#: A fused forward launch the pre-pass serves runs on the CUDA kernel (``c/fused_sum.cu``: node blocks of 32 over the
#: pre-pass's chunks of 32 children, every chunk's sums on tensor cores times its factors) when it has at least this
#: many tiles of 4 node blocks x its columns (:data:`_CUDA_TNS`).
CUDA_MIN_TILES = 0
#: The CUDA kernel's tiles of columns: a launch's narrowest that covers its columns in as few tiles as 128 would. The
#: narrow ones serve a session's pruned launches (Codpula hard_0 B8: 8 live columns a launch, one a sample). A tile of
#: 16 takes 47 KB of shared memory and 64 registers a thread, so two CTAs share an SM and the split (:func:`_cuda_split`)
#: puts a tile's chunks in twice the shares: the chunk loop is bound by its instructions' issue, not by its copies, and
#: an SM running one CTA's chain is half idle (a launch of 33 MB of weights 37.7 -> 20.7 us).
_CUDA_TNS = (16, 32, 64, 80, 96, 112, 128)
#: Bytes of the shares' sums the merge reads in a chunk's time (L2 at ~4.7 TB/s, a chunk's products ~1.6 us): the
#: merge's cost in :func:`_cuda_split`.
_CUDA_MERGE_BYTES = 7.5e6
#: A CUDA launch splits only while its shares' sums take at most this many bytes of the split buffer.
_CUDA_SPLIT_BYTES = 1 << 28
_CUDA_TABLES = {}
_INT32_DUMMIES = {}
_CTAS_PER_SM = {}


def _cuda_tables(pids: torch.Tensor, cells_of: Optional[torch.Tensor]):
    """``(wbase, cells, nch, affine)`` of a fused part for the CUDA kernel -- every (node block, chunk of 32 edges)'s
    first weight, the cell of its external parameters' log-scales (``cells_of``: every edge's, None without), over the
    ``nch`` chunks before the tables' padding (a zero suffix: weight 0 of the -inf dummy child, nothing to add), and
    whether every node block's weights are one run (chunk ``c`` at its first + 1024 ``c``) -- or None if the weights
    are not laid out as the kernel reads them (a chunk's 32 x 32 contiguous, child-major, 16 bytes aligned). Kept per
    part (its tables are the program's, held alive here with the answer)."""
    hit = _CUDA_TABLES.get(id(pids))
    if hit is None or hit[0] is not pids or hit[1] is not cells_of:
        nb = pids.size(0)
        real = int((pids != 0).any(dim = 0).sum())
        nch = real // 32
        p = pids[:, :nch * 32].reshape(nb, nch, 32).long()
        wbase = p[:, :, 0].contiguous()
        ok = real % 32 == 0 and nch > 0 and bool((wbase % 4 == 0).all()) and \
            bool(((p - wbase[:, :, None]) == 32 * torch.arange(32, device = p.device)).all())
        cells, affine = None, False
        if ok:
            affine = bool((wbase == wbase[:, :1] + 1024 * torch.arange(nch, device = p.device)).all())
            if cells_of is not None:
                cells = cells_of[:, :nch * 32].reshape(nb, nch, 32).max(dim = 2).values.to(torch.int32).contiguous()
        hit = _CUDA_TABLES[id(pids)] = (pids, cells_of, (wbase, cells, nch, affine) if ok else None)
    return hit[2]


def _int32_dummy(dev) -> torch.Tensor:
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _INT32_DUMMIES:
        _INT32_DUMMIES[key] = torch.zeros(1, dtype = torch.int32, device = dev)
    return _INT32_DUMMIES[key]


def _no_fragments(params: torch.Tensor) -> torch.Tensor:
    """An empty tensor: no fragment-ordered weights (the CUDA kernel reads ``params`` as they are)."""
    return params[:0]


def _ctas_per_sm(mod, dev, td: bool, ext: bool, x3: bool, fa: bool, bf: bool, tn: int, affine: bool) -> int:
    """How many CTAs of the CUDA kernel (``td``: top-down; ``fa``: fragment-ordered TF32 weights; ``bf``: bf16
    operands) an SM of ``dev`` holds at once (at a share's most chunks)."""
    key = (dev.index if dev.index is not None else torch.cuda.current_device(), td, ext, x3, fa, bf, tn, affine)
    if key not in _CTAS_PER_SM:
        with torch.cuda.device(key[0]):
            _CTAS_PER_SM[key] = max(1, int(mod.ctas_per_sm(td, ext, x3, fa, bf, tn, 256 if ext else 128, affine)))
    return _CTAS_PER_SM[key]


def _cuda_split(tiles: int, nch: int, ctas: int, tn: int, cap: int) -> Optional[int]:
    """Into how many shares a CUDA launch of ``tiles`` tiles of ``nch`` chunks splits every tile's chunks (shares of at
    most ``cap``; None if none fits): the split whose waves of ``ctas`` CTAs take the fewest chunks of products, plus
    the merge's reads (:data:`_CUDA_MERGE_BYTES`); within 5% of the fewest, the fewest shares."""
    lo = host_cdiv(nch, cap)
    hi = max(lo, min(tile_rules.SPLIT_MAX, nch // tile_rules.SPLIT_MIN_SHARE))
    share_bytes = 4 * (4 * 32 * tn + 4 * tn)
    costs = []
    for split in range(lo, hi + 1):
        share = host_cdiv(nch, split)
        split = host_cdiv(nch, share)                       # (every share some chunks)
        if split > 1 and tiles * split * share_bytes > _CUDA_SPLIT_BYTES:
            break
        merge = tiles * split * share_bytes / _CUDA_MERGE_BYTES if split > 1 else 0.0
        costs.append((host_cdiv(tiles * split, ctas) * share + merge, split))
    if not costs or costs[0][1] > tile_rules.SPLIT_MAX:
        return None
    least = min(c for c, _ in costs)
    return min(sp for c, sp in costs if c <= 1.05 * least)


def _cuda_shape(num_blocks: int, max_cols: int) -> Optional[tuple]:
    """``(tn, tiles)`` of a CUDA launch of ``num_blocks`` blocks over ``max_cols`` columns (:data:`_CUDA_TNS`), or
    None if it is not the CUDA kernel's (:data:`CUDA_MIN_TILES`)."""
    col_tiles = host_cdiv(max_cols, 128)
    per = host_cdiv(max_cols, col_tiles)
    tn = next(t for t in _CUDA_TNS if t >= per)
    col_tiles = host_cdiv(max_cols, tn)
    tiles = host_cdiv(num_blocks, 4) * col_tiles
    return None if tiles < CUDA_MIN_TILES or col_tiles > 65535 else (tn, tiles)


def _cuda_shares(mod, dev, td: bool, ext: bool, precision: str, ops: Optional[str], tn: int, tiles: int, nch: int,
                 affine: bool) -> Optional[int]:
    """:func:`_cuda_split` of a CUDA launch (``td``: top-down; ``ops``: :func:`_cuda_operands`) of ``tiles`` tiles of
    ``tn`` columns over ``nch`` chunks."""
    cap = min(256 if ext else nch, nch if affine else 128)
    ctas = _ctas_per_sm(mod, dev, td, ext, precision == "tf32x3", ops == "tf32", ops == "bf16", tn, affine) * \
        _num_sms(dev)
    key = (tiles, nch, ctas, tn, cap, tile_rules.SPLIT_MAX, tile_rules.SPLIT_MIN_SHARE, _CUDA_SPLIT_BYTES, _CUDA_MERGE_BYTES)
    if key not in _CUDA_SPLITS:                         # (by the launch's shape: a few per program)
        if len(_CUDA_SPLITS) >= 4096:
            _CUDA_SPLITS.clear()
        _CUDA_SPLITS[key] = _cuda_split(tiles, nch, ctas, tn, cap)
    return _CUDA_SPLITS[key]


def _cuda_operands(precision: str, frags: Optional[dict], bf16: bool) -> Optional[str]:
    """The operands of a CUDA launch's products besides the pre-pass's e: "bf16" (bf16 weights and e, a launch of a
    ``bf16`` query: TF32 ``precision`` and its copies in ``frags``), "tf32" (the TF32 weights' fragment-ordered copy,
    in ``frags``), or None (the weights as they are)."""
    if frags is None or precision != "tf32":
        return None
    return "bf16" if bf16 else "tf32"


def _cuda_plan(params, pids, num_blocks, gated, g, num_edges, max_cols, block_size, TILE_E, precision, aliased,
               dev, frags: Optional[dict] = None, bf16: bool = False) -> Optional[tuple]:
    """``(module, tables, tn, split, wfrag)`` if a fused forward launch the pre-pass serves is the CUDA kernel's (see
    :data:`CUDA_MIN_TILES`), else None. ``g``: the launch's :func:`gate_launch` arguments; ``wfrag``: the weights in
    the kernel's fragment order (:func:`_cuda_operands`, from ``frags``: :func:`_cuda_fragments`; bf16: the products
    on bf16 operands), or None."""
    if block_size != 32 or TILE_E != 32 or num_edges % 32 or aliased is not None or \
            precision not in ("tf32", "tf32x3") or \
            (gated is not None and (gated["node_gate"] < block_size or not gated["uniform"].get(32, False))):
        return None                                      # (external parameters: a cell per node block and chunk)
    shape = _cuda_shape(num_blocks, max_cols)
    if shape is None:
        return None
    tn, tiles = shape
    from ..c import get_module
    mod = get_module()
    tables = _cuda_tables(pids, g[0] if gated is not None else None) if mod is not None else None
    if tables is None:
        return None
    ops = _cuda_operands(precision, frags, bf16)
    split = _cuda_shares(mod, dev, False, gated is not None, precision, ops, tn, tiles, tables[2], tables[3])
    if split is None:
        return None
    return mod, tables, tn, split, None if ops is None else _cuda_fragments(mod, frags, params, tables[0], False,
                                                                           ops == "bf16")


_TILE_KEYS = {}


def _tiles_key(wbase: torch.Tensor) -> str:
    """A digest of the tiles a part's launches read (``wbase``, the program's: kept alive here with it), the same
    for parts that read the same tiles (an HMM's tied transitions). Read off the device once per part."""
    hit = _TILE_KEYS.get(id(wbase))
    if hit is None or hit[0] is not wbase:
        hit = _TILE_KEYS[id(wbase)] = (wbase, hashlib.sha1(wbase.cpu().numpy().tobytes()).hexdigest())
    return hit[1]


def _cuda_fragments(mod, frags: dict, params: torch.Tensor, wbase: torch.Tensor, td: bool,
                    bf16: bool = False) -> torch.Tensor:
    """The weights of a CUDA launch (its every 32 x 32 tile's first ``wbase``) in the kernel's A-fragment order
    (``c/fused_sum.cu``, ``fragment_weights``): rounded to TF32, or (``bf16``) a bf16 copy in mma m16n8k16's order --
    a copy of ``params`` per direction (``td``), every tile in its own place, kept in ``frags``: the stacked weights
    of a :func:`pyjuice.fast_inference` scope that allows parameter copies, or a bf16 query's own
    (:meth:`~..state.State.weight_stacks`). A part's tiles are copied by the first launch that reads them (parts
    reading the same tiles share the copy), the parameters not changing inside the scope or query. At TF32 the
    fragments are the products' own, so the sums do not change."""
    key = ("cuda_fragments", td, bf16)
    if key not in frags:
        frags[key] = (torch.empty(params.numel(), dtype = torch.bfloat16 if bf16 else params.dtype,
                                  device = params.device), set())
    frag, done = frags[key]
    tiles = _tiles_key(wbase)
    if tiles not in done:
        mod.fragment_weights(params, frag, wbase, td)
        done.add(tiles)
    return frag


def _cuda_split_buffer(dev, num_blocks: int, max_cols: int, tn: int, nch: int, split: int, fallback: torch.Tensor):
    """The split buffer of a CUDA launch (``fallback`` unsplit: never read)."""
    if split == 1:
        return fallback
    share = host_cdiv(nch, split)
    per_tile = host_cdiv(4 * tn + host_cdiv(nch, share) * (4 * 32 * tn + 4 * tn), 4) * 4
    return split_buffer(dev, host_cdiv(num_blocks, 4) * host_cdiv(max_cols, tn), per_tile)


def _cuda_forward(plan, node_mars, element_mars, params, nids, cids, nb_reg, sel, num_blocks, regions, lanes, gated,
                  g, pre_args, elem_first, elem_width, batch_size, max_cols, precision):
    """The fused forward launch ``plan`` (:func:`_cuda_plan`) on the CUDA kernel: one call -- the pre-pass over the
    children's chunks (``cids`` row 0 of ``element_mars``, into ``pre_args``: its buffer, ``pre_cm``, ``pre_ld``), the
    sums, and a split launch's merge by the last share in."""
    mod, (wbase, cells, nch, affine), tn, split, wfrag = plan
    dev = node_mars.device
    sbuf = _cuda_split_buffer(dev, num_blocks, max_cols, tn, nch, split, node_mars)
    pre, pre_cm, pre_ld = pre_args
    _, _, _, _, sstride, lz, lzstride, coff, reg_ivk, Bq, steps = g
    ext = gated is not None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else (nids,) * 3
    dummy = _int32_dummy(dev)
    mod.fused_sum_cuda(False, node_mars, node_mars, params, _no_fragments(params) if wfrag is None else wfrag, wbase,
                       1024, pre, pre_ld, pre_cm, nids, nb_reg, nb_reg, nids if sel is None else sel, regions["offset"],
                       regions["width"], regions["first"], regions["slots"], perm, reg_perm, reg_live, elem_first,
                       elem_width, gated["minor"] if ext else node_mars,
                       cells if ext else dummy, gated["mstride"] if ext else 0,
                       lz if ext else node_mars, lzstride if ext else 0, coff, reg_ivk, Bq, steps, batch_size,
                       num_blocks, nch, max_cols, ext, ext and sstride == 0, ext and gated["lanes"] is not None,
                       lanes is not None, sel is not None, precision == "tf32x3", affine, tn, sbuf, split,
                       element_mars, cids, dummy, dummy, dummy, 1, 1,
                       2 if wfrag is not None and wfrag.dtype == torch.bfloat16 else 1, nch * 32)


_CUDA_TD_TABLES = {}
_CUDA_SPLITS = {}


def _cuda_td_tables(tables: dict, gt: Optional[dict], cbs: int):
    """The CUDA kernel's tables of a fused top-down part whose child blocks (of ``cbs`` children) it reads in slices
    of 32 children -- its row blocks: every slice's first row, region and region of columns, its first weight for
    every parent entry and (``gt``: the part's gate tables) the cell of its log-scales -- as ``(rows, rreg, colreg,
    wbase, cells, npar, affine, wstride, slices)``, or None unless every child block reads the same parents (node
    blocks and regions, in the same order: one region of columns) at 16-byte aligned weights and none is log-added to
    values written before (``cb_add``: with blocks of 32 a child's parents are one launch's).
    ``affine``: every slice's weights one run (entry ``j``'s at its first + ``wstride`` ``j``). Kept per part (its
    tables are the program's, held alive here with the answer)."""
    pid = tables["par_pid"]
    hit = _CUDA_TD_TABLES.get(id(pid))
    if hit is None or hit[0] is not pid or hit[1] is not gt:
        ptr = tables["par_ptr"].long()
        n_cb, sl = ptr.numel() - 1, cbs // 32
        counts = ptr[1:] - ptr[:-1]
        npar = int(counts[0]) if n_cb > 0 else 0
        out = None
        if npar > 0 and bool((counts == npar).all()) and not bool(tables["cb_add"].any()):
            dev = ptr.device
            at = ptr[:-1, None] + torch.arange(npar, device = dev)[None, :]                     # [child blocks, j]
            nid, reg, ivreg = tables["par_nid"].long()[at], tables["par_reg"].long()[at], tables["cb_ivreg"].long()
            ar = torch.arange(sl, device = dev)
            wbase = (pid.long()[at][:, None, :] + 1024 * ar[None, :, None]).reshape(n_cb * sl, npar).contiguous()
            if bool((nid == nid[:1]).all()) and bool((reg == reg[:1]).all()) and bool((ivreg == ivreg[0]).all()) and \
                    bool((wbase % 4 == 0).all()):
                d = wbase[:, 1:] - wbase[:, :-1]
                wstride = int(d[0, 0]) if npar > 1 else 0
                affine = npar == 1 or bool((d == wstride).all())
                cells = None
                if gt is not None:                       # (the slice's child gate)
                    lc = gt["lc"].long()[at][:, None, :] + 32 * ar[None, :, None]
                    cells = (gt["row"].long()[at][:, None, :] + lc // gt["cbs"].long()[at][:, None, :])
                    cells = cells.reshape(n_cb * sl, npar).to(torch.int32).contiguous()
                each = lambda t: t.long().repeat_interleave(sl).contiguous()
                rows = (tables["cb_row"].long()[:, None] + 32 * ar[None, :]).reshape(-1).contiguous()
                out = (rows, each(tables["cb_reg"]), each(tables["cb_ivreg"]), wbase, cells, npar, affine, wstride,
                       sl)
        hit = _CUDA_TD_TABLES[id(pid)] = (pid, gt, out)
    return hit[2]


def _cuda_td_plan(tables, gt, idx, block_size, cbs, max_cols, TILE_I, precision, dev, params,
                  frags: Optional[dict] = None, bf16: bool = False) -> Optional[tuple]:
    """``(module, tables, tn, split, wfrag)`` if a fused top-down launch is the CUDA kernel's: child blocks of 32
    children or a multiple (read in slices of 32, :func:`_cuda_td_tables`) under the same parent node blocks of 32,
    none log-added (with external parameters, every slice in one child gate and a parent block in one node gate), see
    :data:`CUDA_MIN_TILES` -- else None. ``wfrag``: the weights ``params`` in the kernel's fragment order
    (:func:`_cuda_operands`, from ``frags``: :func:`_cuda_fragments`), or None."""
    if block_size != 32 or cbs % 32 or TILE_I != 32 or precision not in ("tf32", "tf32x3") or \
            (gt is not None and (gt["node_gate"] < block_size or not gt["uniform"].get(32, False))):
        return None
    shape = _cuda_shape((tables["cb_row"].numel() if idx is None else idx.numel()) * (cbs // 32), max_cols)
    if shape is None:
        return None
    tn, tiles = shape
    from ..c import get_module
    mod = get_module()
    td = _cuda_td_tables(tables, gt, cbs) if mod is not None else None
    if td is None:
        return None
    ops = _cuda_operands(precision, frags, bf16)
    split = _cuda_shares(mod, dev, True, gt is not None, precision, ops, tn, tiles, td[5], td[6])
    if split is None:
        return None
    return mod, td, tn, split, None if ops is None else _cuda_fragments(mod, frags, params, td[3], True,
                                                                       ops == "bf16")


def _cuda_top_down(plan, element_flows, node_flows, params, tables, regions, elem_first, elem_width, batch_size,
                   max_cols, precision, liveness, idx, gated, gl):
    """The fused top-down launch ``plan`` (:func:`_cuda_td_plan`) on the CUDA kernel: the parents' chunks once, by the
    pre-pass, then the products. ``gl``: the launch's :func:`gate_launch` arguments past the edges'."""
    mod, (rows, rreg, colreg, wbase, cells, npar, affine, wstride, sl), tn, split, wfrag = plan
    dev = element_flows.device
    sel = None
    if idx is not None:                                  # (the index list's child blocks' slices)
        sel = (idx.long()[:, None] * sl + torch.arange(sl, device = dev)[None, :]).reshape(-1)
    num_blocks = rows.numel() if sel is None else sel.numel()
    lanes = liveness.lanes if liveness is not None else None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else (rows,) * 3
    pre_ld = host_cdiv(max_cols, 8) * 8                                 # (groups of 8 columns, :func:`_frag_offsets`)
    pre = pre_buffer(dev, npar * (32 + 1) * pre_ld)
    ext = gated is not None
    sbuf = _cuda_split_buffer(dev, num_blocks, max_cols, tn, npar, split, element_flows)
    dummy = _int32_dummy(dev)
    mod.fused_sum_cuda(True, element_flows, node_flows, params, _no_fragments(params) if wfrag is None else wfrag,
                       wbase, wstride, pre, pre_ld, npar * 32 * pre_ld, rows, rreg, colreg,
                       rows if sel is None else sel, regions["offset"], regions["width"], regions["first"],
                       regions["slots"], perm, reg_perm, reg_live, elem_first, elem_width,
                       gated["minor"] if ext else element_flows, cells if ext else dummy,
                       gated["mstride"] if ext else 0,
                       gl[2] if ext else element_flows, gl[3] if ext else 0, gl[4], gl[5], gl[6], gl[7], batch_size,
                       num_blocks, npar, max_cols, ext, ext and gl[1] == 0, ext and gated["lanes"] is not None,
                       liveness is not None, sel is not None, precision == "tf32x3", affine, tn, sbuf, split,
                       node_flows, dummy, tables["par_ptr"], tables["par_nid"], tables["par_reg"], sl, 2,
                       2 if wfrag is not None and wfrag.dtype == torch.bfloat16 else 1, 0)
