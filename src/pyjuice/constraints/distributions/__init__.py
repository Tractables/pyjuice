"""
Input distributions under a constraint. A constrained backend needs two things from the distribution of an input
node, besides what its input layer already computes (the log-probability of an observed token):

* how many values it ranges over -- the constraint's tokens must be exactly those;
* for a token that is marginalized, the mass of every token class of the constraint's automaton:
  ``log sum_{v in c} p_n(v)``.

Both are computed here, one module per distribution, named as in :mod:`pyjuice.nodes.distributions`. Class masses
are kept once per set of parameters, however many tied nodes share it: the rows of a layer's class-mass table. Each
module provides

* ``num_values(dist) -> int``: the values (tokens) the distribution ranges over;
* ``class_mass_rows(layer) -> torch.Tensor``: ``[number of nodes of layer]`` int64, the row of the layer's
  class-mass table every node reads, numbered from 0 (the table has ``max + 1`` rows);
* ``class_masses(layer, classes, out = None) -> torch.Tensor``: the table, ``[rows, num_classes]``, the log-mass of
  every class for every row, the same for every sample (``classes``: the constraint's :class:`TokenClasses`);
  written into ``out`` when given;
* ``token_probs(layer, rows, out) -> torch.Tensor``: the probability of every value under the table's rows ``rows``
  (sorted, host int64), token-major: ``out[v, k] = p(v)`` of row ``rows[k]``, ``out`` [values, len(rows)] in fp32 or
  bf16 -- what the conditionals score (:mod:`~pyjuice.constraints.backends.dfa_exact.conditional`).

Only the distributions in :data:`SUPPORTED` can be compiled under a constraint, matched by exact type: a subclass
may compute different probabilities, so it is refused until it has its own module.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from pyjuice.nodes.distributions import Categorical


class TokenClasses:
    """
    The token classes of a constraint's automaton, as the input distributions read them. The ``dfa_exact`` backend
    owns one (``cc.backend.token_classes``), built with it.

    :ivar token_class: [V] int32, the class of every token, on the circuit's device
    :ivar num_classes: the number of classes
    """

    def __init__(self, token_class: torch.Tensor, num_classes: int):
        self.token_class = token_class.to(torch.int32).contiguous()
        self.num_classes = int(num_classes)
        self._by_class = {}
        self._class_tiles = {}

    def to(self, device) -> "TokenClasses":
        """The same classes on ``device`` (a new object: nothing derived is carried over)."""
        return TokenClasses(self.token_class.to(device), self.num_classes)

    def by_class(self, chunk: int) -> "ByClass":
        """The tokens in class order, cut into chunks of ``chunk`` positions (built on first use, then kept)."""
        found = self._by_class.get(chunk)
        if found is None:
            found = self._by_class[chunk] = ByClass.build(self.token_class, self.num_classes, chunk)
        return found

    def class_tiles(self, width: int):
        """``(tokens, tile_class)``: the tokens in class order (by id within a class), every class's run padded with -1
        to whole tiles of ``width`` (int32 [tiles * width]), and every tile's class (int32 [tiles]) -- one class per
        tile, for a kernel that reads a class's weights once per tile (built on first use, then kept)."""
        found = self._class_tiles.get(width)
        if found is None:
            tc = self.token_class.cpu().long()
            order = torch.argsort(tc, stable = True)
            counts = torch.bincount(tc, minlength = self.num_classes)
            tiles = (counts + width - 1) // width                               # per class
            first_tile = torch.cumsum(tiles, 0) - tiles
            first_token = torch.cumsum(counts, 0) - counts
            cls = tc[order]
            tokens = torch.full((int(tiles.sum()) * width,), -1, dtype = torch.long)
            tokens[first_tile[cls] * width + torch.arange(tc.numel()) - first_token[cls]] = order
            tile_class = torch.repeat_interleave(torch.arange(self.num_classes), tiles)
            dev = self.token_class.device
            found = self._class_tiles[width] = (tokens.to(dev, torch.int32), tile_class.to(dev, torch.int32))
        return found


@dataclass(frozen = True)
class ByClass:
    """
    The tokens sorted by class (stably, so by id within a class), for a kernel that reduces class by class over
    chunks of ``chunk`` sorted positions. Classes are numbered by ``rank`` among the classes that have a token. A
    chunk's piece of a class is a segment; the segment of rank ``r`` in chunk ``j`` has index ``r + j`` (distinct,
    since classes do not interleave), so a class's segments are the contiguous indices
    ``seg_lo[r]..seg_hi[r]``.

    :ivar order: [V] int32, the token at every sorted position
    :ivar rank: [V] int32, the rank of the class at every sorted position
    :ivar class_rank: [num_classes] int32, the rank of every class (-1 for a class without a token)
    :ivar seg_lo, seg_hi: [classes with a token] int32, a class's first and last segment index
    :ivar num_chunks: chunks of sorted positions
    :ivar num_segments: an upper bound on the segment indices
    """

    order: torch.Tensor
    rank: torch.Tensor
    class_rank: torch.Tensor
    seg_lo: torch.Tensor
    seg_hi: torch.Tensor
    chunk: int
    num_chunks: int
    num_segments: int

    @staticmethod
    def build(token_class: torch.Tensor, num_classes: int, chunk: int) -> "ByClass":
        tc = token_class.long()
        order = torch.argsort(tc, stable = True)
        counts = torch.bincount(tc, minlength = num_classes)
        present = counts > 0
        rank_of = torch.cumsum(present.long(), 0) - 1
        ends = torch.cumsum(counts[present], 0)                     # end of every class's run, in rank order
        starts = ends - counts[present]
        ranks = torch.arange(ends.numel(), device = tc.device)
        num_chunks = -(-tc.numel() // chunk)
        i32 = lambda t: t.to(torch.int32).contiguous()
        return ByClass(order = i32(order), rank = i32(rank_of[tc[order]]), class_rank = i32(torch.where(present, rank_of, -1)),
                       seg_lo = i32(ranks + starts // chunk), seg_hi = i32(ranks + (ends - 1) // chunk), chunk = chunk,
                       num_chunks = num_chunks, num_segments = ends.numel() + num_chunks)


from . import categorical                                                      # noqa: E402 (uses TokenClasses)

#: the module of every supported input distribution, by exact type
SUPPORTED = {Categorical: categorical}


def lookup(dist) -> Optional[object]:
    """The module of ``dist``'s exact type in :data:`SUPPORTED`, or None."""
    return SUPPORTED.get(type(dist))
