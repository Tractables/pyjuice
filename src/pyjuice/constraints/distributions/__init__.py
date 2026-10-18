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

A distribution that also reads per-sample inputs at query time -- keyword arguments of ``pc(x, ...)``, such as soft
evidence; a decoder takes them for its whole session (see :meth:`~pyjuice.constraints.ConstrainedCircuit.decoder`) --
provides besides

* ``KEYWORDS``: the keywords it reads;
* ``check_inputs(layer, inputs) -> Optional[int]``: checks the layer's inputs (``inputs``: keyword -> value, those of
  ``KEYWORDS`` given) and returns their batch size, or None if the layer reads none of them;
* optionally ``free_positions(layer, inputs, n) -> Optional[torch.Tensor]``: bool [batch, n], the PC's positions the
  inputs leave free -- marginalized under them whatever a query's data holds there (None: none);
* ``sample_rows(layer, inputs) -> int`` and ``sample_inputs(layer, classes, inputs, out) -> SampleInputs``: the class
  masses the inputs give its nodes, per sample, in ``sample_rows`` rows per sample written into ``out`` ([batch,
  rows, num_classes]);
* ``input_values(layer, x, missing, out, sample = None)``: the log-probability of every observed token, into the
  layer's rows of ``out`` (``[input nodes, B]``, as a :class:`TensorCircuit`'s ``node_mars``; ``x`` [B, n] token ids,
  ``missing`` [B, n] bool), under the inputs (``sample``, :func:`sample_inputs`') or none -- written by the backend
  instead of the layer's own forward, which would need the inputs every query.

Only the distributions in :data:`SUPPORTED` can be compiled under a constraint, matched by exact type: a subclass
may compute different probabilities, so it is refused until it has its own module.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.nodes.distributions import Categorical, SoftEvidenceCategorical


@dataclass
class SampleInputs:
    """
    What an input layer's nodes take from a query's per-sample inputs (see the module docstring). The inputs reweight
    every token: at node ``i`` of sample ``b`` the token distribution is ``q(v) = p_i(v) w(v) / Z``, ``w`` the inputs'
    weights there. Its class masses are in a table of their own per sample: node ``i`` reads row ``rows[i]`` of sample
    ``b``'s (sample 0's for every sample when ``batch`` is 1), a token class's mass ``log sum_{v in c} q(v)`` at every
    row, and ``log Z`` per row besides; the weights ``log w`` are in ``token_logw``, node ``i`` at row
    ``token_rows[i]`` -- of the first ``K`` tokens, or of the tokens ``token_ids`` lists (:func:`find_token` finds
    one). Where the layer's other values (an observed token's log-probability, ``input_values``) need more, ``data``.

    :ivar masses: [batch, rows, num_classes] fp32, log
    :ivar rows: int64 [nodes of the layer], every node's row
    :ivar logz: [batch, rows] fp32, every row's ``log Z``: its masses are ``log sum_{v in c} p(v) w(v) - log Z`` (0
        where nothing weighs, the masses -inf -- or where they are unnormalized: a token there weighs ``p(v) w(v)``)
    :ivar token_logw: [batch, weight rows, K] fp32, ``log w`` of the first ``K`` tokens, or of ``token_ids``' (the
        others weigh nothing)
    :ivar token_rows: int64 [nodes of the layer], every node's row of ``token_logw``
    :ivar batch: the inputs' batch size
    :ivar variables: int64, the PC's variables whose input nodes read them
    :ivar token_ids: None, or int32 [batch, weight rows, K], ascending in every row (distinct): the tokens
        ``token_logw`` weighs
    :ivar pair_rows: int64 [rows], the row of the layer's parameter table every row of ``masses`` is of
    :ivar data: what the distribution's module keeps besides (its own)
    """

    masses: torch.Tensor
    rows: torch.Tensor
    logz: torch.Tensor
    token_logw: torch.Tensor
    token_rows: torch.Tensor
    batch: int
    variables: torch.Tensor
    token_ids: Optional[torch.Tensor] = None
    pair_rows: Optional[torch.Tensor] = None
    data: Any = None


@triton.jit
def find_token(ids, base, K, v, mask, STEPS: tl.constexpr):
    """Lane-wise, where token ``v`` is in the ascending run ``ids[base : base + K]`` (``SampleInputs.token_ids``' rows):
    its index in the run, or -1 if it is not there (or the lane is masked). A binary search, ``STEPS`` (at least ``K``'s
    bit length) halvings."""
    lo = tl.zeros_like(base)
    left = tl.zeros_like(base) + K                                          # candidates left from lo on
    for _ in tl.static_range(STEPS):
        half = left // 2
        live = left > 0
        right = live & (tl.load(ids + base + lo + half, mask = mask & live, other = 0) < v)
        lo = tl.where(right, lo + half + 1, lo)
        left = tl.where(right, left - half - 1, half)
    hit = mask & (lo < K)
    found = tl.load(ids + base + lo, mask = hit, other = -1) == v
    return tl.where(hit & found, lo, -1)


class TokenClasses:
    """
    The token classes of a constraint's automaton, as the input distributions read them. The ``dfa_exact`` backend
    owns one (``cc.backend.token_classes``), built with it.

    Classes of at most ``enumerate_max`` tokens are ENUMERATED: they have no column in any class-mass table -- wherever
    the mass of such a class is needed, its token's own probability is looked up instead (the backend's kernels read
    ``stored_class`` and write ``num_stored`` columns). The other classes, a class without a token among them, are
    STORED, numbered by ``stored_of_class`` in class order. ``enumerate_max`` 0: every class stored, as plain tables.

    :ivar token_class: [V] int32, the class of every token, on the circuit's device
    :ivar num_classes: the number of classes
    :ivar enumerate_max: classes of up to this many tokens are enumerated (0: none; 1 is the most supported)
    :ivar stored_of_class: [num_classes] int32, the column of every class in the class-mass tables, -1 if enumerated
    :ivar num_stored: the stored classes (the tables' columns)
    :ivar class_of_stored: [num_stored] int64, the class of every column (``class_of_stored32``: the same, int32)
    :ivar stored_class: [V] int32, ``stored_of_class`` of every token's class (-1: the token's class is enumerated)
    :ivar enumerated_token: [num_classes] int32, the token of every enumerated class, -1 for a stored one
    :ivar enumerated_tokens: [num_enumerated] int32, the enumerated classes' tokens in class order
    :ivar num_enumerated: the enumerated classes
    """

    def __init__(self, token_class: torch.Tensor, num_classes: int, enumerate_max: int = 0):
        self.token_class = token_class.to(torch.int32).contiguous()
        self.num_classes = int(num_classes)
        self.enumerate_max = int(enumerate_max)
        if self.enumerate_max > 1:
            raise NotImplementedError("Only classes of one token can be enumerated (`enumerate_max` is 0 or 1).")
        dev = self.token_class.device
        sizes = torch.bincount(self.token_class.long(), minlength = self.num_classes)
        enumerated = (sizes > 0) & (sizes <= self.enumerate_max)
        self.stored_of_class = torch.where(enumerated, -1, torch.cumsum((~enumerated).long(), 0) - 1).to(torch.int32)
        self.num_stored = int((~enumerated).sum())
        self.num_enumerated = self.num_classes - self.num_stored
        self.class_of_stored = torch.nonzero(~enumerated).view(-1)
        self.class_of_stored32 = self.class_of_stored.to(torch.int32).contiguous()   # (for the kernels)
        self.stored_class = self.stored_of_class[self.token_class.long()].contiguous()
        self.enumerated_token = torch.full((self.num_classes,), -1, dtype = torch.int32, device = dev)
        single = torch.nonzero(self.stored_class < 0).view(-1)                 # (one token per enumerated class)
        self.enumerated_token[self.token_class[single].long()] = single.to(torch.int32)
        self.enumerated_tokens = self.enumerated_token[self.enumerated_token >= 0].contiguous()   # in class order
        self._by_class = {}
        self._host = None
        self._class_tiles = {}

    def to(self, device) -> "TokenClasses":
        """The same classes on ``device`` (a new object: nothing derived is carried over)."""
        return TokenClasses(self.token_class.to(device), self.num_classes, self.enumerate_max)

    def host(self) -> np.ndarray:
        """``token_class`` on the host (int32), copied on first use."""
        if self._host is None:
            self._host = self.token_class.cpu().numpy()
        return self._host

    def by_class(self, chunk: int) -> "ByClass":
        """The tokens of the stored classes in class order, cut into chunks of ``chunk`` positions (built on first
        use, then kept)."""
        found = self._by_class.get(chunk)
        if found is None:
            found = self._by_class[chunk] = ByClass.build(self.stored_class, self.num_stored, chunk)
        return found

    def class_tiles(self, width: int, stored: bool = False):
        """``(tokens, tile_class)``: the tokens in class order (by id within a class), every class's run padded with -1
        to whole tiles of ``width`` (int32 [tiles * width]), and every tile's class (int32 [tiles]) -- one class per
        tile, for a kernel that reads a class's weights once per tile (built on first use, then kept). ``stored``: the
        stored classes' tokens alone (scoring: an enumerated class's token is scored on its own, no tile of one)."""
        found = self._class_tiles.get((width, stored))
        if found is None:
            tc = self.token_class.cpu().long()
            order = torch.argsort(tc, stable = True)
            if stored:
                order = order[self.stored_class.cpu()[order] >= 0]
            counts = torch.bincount(tc[order], minlength = self.num_classes)
            tiles = (counts + width - 1) // width                               # per class
            first_tile = torch.cumsum(tiles, 0) - tiles
            first_token = torch.cumsum(counts, 0) - counts
            cls = tc[order]
            tokens = torch.full((int(tiles.sum()) * width,), -1, dtype = torch.long)
            tokens[first_tile[cls] * width + torch.arange(order.numel()) - first_token[cls]] = order
            tile_class = torch.repeat_interleave(torch.arange(self.num_classes), tiles)
            dev = self.token_class.device
            found = self._class_tiles[(width, stored)] = (tokens.to(dev, torch.int32), tile_class.to(dev, torch.int32))
        return found


@dataclass(frozen = True)
class ByClass:
    """
    The tokens sorted by class (stably, so by id within a class), for a kernel that reduces class by class over
    chunks of ``chunk`` sorted positions. Classes are numbered by ``rank`` among the classes that have a token. A
    chunk's piece of a class is a segment; the segment of rank ``r`` in chunk ``j`` has index ``r + j`` (distinct,
    since classes do not interleave), so a class's segments are the contiguous indices
    ``seg_lo[r]..seg_hi[r]``. Tokens whose class is -1 (an enumerated class's) are left out.

    :ivar order: [sorted positions] int32, the token at every sorted position
    :ivar rank: [sorted positions] int32, the rank of the class at every sorted position
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

    @property
    def num_sorted(self) -> int:
        return self.order.numel()

    @staticmethod
    def build(token_class: torch.Tensor, num_classes: int, chunk: int) -> "ByClass":
        tc = token_class.long()
        order = torch.argsort(tc, stable = True)
        order = order[tc[order] >= 0]                               # (the tokens of enumerated classes left out)
        counts = torch.bincount(tc[order], minlength = num_classes)
        present = counts > 0
        rank_of = torch.cumsum(present.long(), 0) - 1
        ends = torch.cumsum(counts[present], 0)                     # end of every class's run, in rank order
        starts = ends - counts[present]
        ranks = torch.arange(ends.numel(), device = tc.device)
        num_chunks = max(1, -(-order.numel() // chunk))
        i32 = lambda t: t.to(torch.int32).contiguous()
        return ByClass(order = i32(order), rank = i32(rank_of[tc[order]]), class_rank = i32(torch.where(present, rank_of, -1)),
                       seg_lo = i32(ranks + starts // chunk), seg_hi = i32(ranks + (ends - 1) // chunk), chunk = chunk,
                       num_chunks = num_chunks, num_segments = ends.numel() + num_chunks)


from . import categorical, softevi_categorical                                 # noqa: E402 (uses TokenClasses)

#: the module of every supported input distribution, by exact type
SUPPORTED = {Categorical: categorical, SoftEvidenceCategorical: softevi_categorical}


def lookup(dist) -> Optional[object]:
    """The module of ``dist``'s exact type in :data:`SUPPORTED`, or None."""
    return SUPPORTED.get(type(dist))
