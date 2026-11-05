"""
Which (output tile, shared chunk) pairs a ``block @ block`` step can skip, from the blocks' occupancy.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import torch

from ...compiled.reachability import Reachability, _tile_any


@dataclass(frozen = True)
class SkipMasks:
    """
    Which chunks of the shared boundary every output tile of a ``block @ block`` step needs. For triple ``s``
    (boundaries ``(t0, t1, t2)``), word ``(s * tiles_i + ti) * tiles_j * words + tj * words + c // 32`` of
    ``bits`` has bit ``c % 32`` set when some column of entry tile ``ti`` (at ``t0``) reaches some column of chunk
    ``c`` (at ``t1``) that reaches some column of exit tile ``tj`` (at ``t2``). Every other (tile, chunk) pair is
    structurally ``-inf`` on one side and adds exactly zero. Tiles are ``tm x tn`` columns, chunks ``tk``.

    :ivar bits: int32 ``[triples * tiles_i * tiles_j * words]`` (one word if there are no triples)
    """

    bits: torch.Tensor
    tiles_i: int
    tiles_j: int
    words: int
    tm: int
    tn: int
    tk: int


def skip_masks(reach: Reachability, triples: Sequence[Tuple[int, int, int]], tm: int = 16, tn: int = 16,
               tk: int = 16, device = None) -> SkipMasks:
    """
    The :class:`SkipMasks` of ``triples`` (``(t0, t1, t2)`` boundaries, in the order their index refers to), from
    the occupancy ``reach`` keeps; ``tm``, ``tn`` and ``tk`` are multiples of its tile.
    """
    return skip_masks_for(reach, [((a, b, False), (b, c, False)) for a, b, c in triples], tm, tn, tk, device)


def skip_masks_for(reach: Reachability, specs, tm: int = 16, tn: int = 16, tk: int = 16, device = None) -> SkipMasks:
    """
    The :class:`SkipMasks` of log-space products ``A @ B`` whose operands are blocks or transposed blocks: spec ``k``
    is ``((a, b, transposed), (c, d, transposed))``, ``A`` the block over ``[a, b)`` (rows: its entry columns, shared:
    its exit columns; transposed, the other way round) and ``B`` the block over ``[c, d)`` (shared: entry, output
    columns: exit; or the other way round). A forward step is ``((t0, t1, False), (t1, t2, False))``; the top-down
    pass reads one operand transposed.
    """
    T = reach.tile
    if tm % T or tn % T or tk % T:
        raise ValueError(f"Tiles ({tm}, {tn}, {tk}) must be multiples of the occupancy tile {T}.")
    if len(specs) == 0:
        return SkipMasks(torch.zeros(1, dtype = torch.int32, device = device), 0, 0, 1, tm, tn, tk)
    table = np.asarray(specs, dtype = np.int64).reshape(len(specs), 6)       # (a, b, transposed) twice (or as such, at hand)
    whole = tm == T and tn == T and tk == T                                  # (no tiling: the occupancy as it is)

    def sides_of(side):
        """The distinct operands of a side of the specs, ``(a, b, transposed)`` int64, and every spec's among them."""
        key = (side[:, 0] * (reach.n + 1) + side[:, 1]) * 2 + side[:, 2]
        distinct, inverse = np.unique(key, return_inverse = True)
        return distinct // 2 // (reach.n + 1), distinct // 2 % (reach.n + 1), distinct % 2, inverse

    def packed(a, b, transposed, rows, chunks_along_rows):
        """int32 [operands, rows, words]: the shared chunks' flags of every operand, a word per 32 chunks (bit by chunk);
        ``chunks_along_rows``: its rows' side is the shared one in its output's order. Its occupancy is read from its
        stack, all of them in a device call a stack; an interval whose every pair is joined is all set."""
        out = torch.zeros(a.size, rows, NC, dtype = torch.bool, device = where)
        stack, place = reach.occupancy_places(a, b)
        for s in np.unique(stack[stack >= 0]).tolist():
            for flip in (0, 1):
                own = np.flatnonzero((stack == s) & (transposed == flip))
                if own.size == 0:
                    continue
                part = reach._stacks[s][0][torch.from_numpy(place[own]).to(where).long()]
                if bool(flip) != chunks_along_rows:
                    part = part.transpose(1, 2)
                part = part[:, :rows, :NC]                               # (what is cut: no column reaches it)
                out[torch.from_numpy(own).to(where), :part.size(1), :part.size(2)] = part
        for k in np.flatnonzero(stack < 0).tolist():                     # (not swept together: every pair, or alone)
            m = reach.device_occupancy(int(a[k]), int(b[k]))
            m = m.t() if bool(transposed[k]) != chunks_along_rows else m
            out[k, :m.size(0), :m.size(1)] = m
        w = (out.view(a.size, rows, words, 32).to(torch.int64) << shifts).sum(dim = -1)
        return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)

    la, lb, lt, li = sides_of(table[:, :3])
    ra, rb, rt, ri = sides_of(table[:, 3:])
    where = reach.device_occupancy(int(la[0]), int(lb[0])).device
    shifts = torch.arange(32, dtype = torch.int64, device = where)
    if whole:
        widths = np.asarray(reach.width, dtype = np.int64)
        tiles = lambda v: (v + T - 1) // T
        exits = lambda b: np.where(b == reach.n, 1, widths[np.minimum(b, reach.n)])
        lrows, lchunks = np.where(lt, tiles(exits(lb)), tiles(widths[la])), np.where(lt, tiles(widths[la]), tiles(exits(lb)))
        rrows, rchunks = np.where(rt, tiles(widths[ra]), tiles(exits(rb))), np.where(rt, tiles(exits(rb)), tiles(widths[ra]))
        I, J = int(lrows.max()), int(rrows.max())
        words = max(1, math.ceil(max(int(lchunks.max()), int(rchunks.max())) / 32))
        NC = 32 * words
        Lw, Rw = packed(la, lb, lt, I, False), packed(ra, rb, 1 - rt, J, False)
    else:
        def tiled(a, b, transposed, tr, tc):
            return [_tile_any(reach.device_occupancy(int(x), int(y)).t() if t else reach.device_occupancy(int(x), int(y)), tr, tc)
                    for x, y, t in zip(a, b, transposed)]

        L, R = tiled(la, lb, lt, tm // T, tk // T), tiled(ra, rb, rt, tk // T, tn // T)   # [output row tiles, chunks], [chunks, ...]
        I, J = max(m.size(0) for m in L), max(m.size(1) for m in R)
        words = max(1, math.ceil(max(max(m.size(1) for m in L), max(m.size(0) for m in R)) / 32))
        NC = 32 * words

        def stacked(ms, rows):
            out = torch.zeros(len(ms), rows, NC, dtype = torch.bool, device = where)
            for k, m in enumerate(ms):
                out[k, :m.size(0), :m.size(1)] = m
            w = (out.view(len(ms), rows, words, 32).to(torch.int64) << shifts).sum(dim = -1)
            return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)

        Lw, Rw = stacked(L, I), stacked([m.t() for m in R], J)
    li, ri = torch.from_numpy(li).to(where), torch.from_numpy(ri).to(where)
    chunk = max(1, (1 << 26) // max(I * J * words, 1))
    parts = []
    for g0 in range(0, len(specs), chunk):
        parts.append((Lw[li[g0:g0 + chunk], :, None, :] & Rw[ri[g0:g0 + chunk], None, :, :]).flatten())
    return SkipMasks(torch.cat(parts).to(device), I, J, words, tm, tn, tk)
