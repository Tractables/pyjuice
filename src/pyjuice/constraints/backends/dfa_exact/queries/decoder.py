"""
The dfa_exact backend's decoding session (:class:`DFAExactDecoder`): the circuit's queries, each reusing the
per-sample values its previous one left in the buffers wherever the evidence did not change them.

A sum region's inside values depend only on the evidence on its scope, and its top-down (log outside) values only on
the evidence outside it. Every row of a call continues a row of the previous call
(:func:`~pyjuice.constraints.decoder.match_rows`): it observes all that row observed, maybe more. So where no row
newly observes a position of a region's scope, the region's inside values are the ones the previous call left, row by
row. Where no row newly observes a position outside the scope, its top-down values are the previous call's too. The
call holds the ones it needs (``held``, see :mod:`.plans`) and computes none of them. They stay in place when every row
continues the row at its own index. Otherwise they are first moved into the call's rows
(:func:`~..kernels.blocks.move_blocks`: only the lanes live in the previous call are read), and the others are given
up.

They are exact under the call's evidence pruning (:mod:`..kernels.live`). Every slot holds its true value if live and
at most its true value if dead. More evidence leaves fewer slots live, never more, so that stays true: a slot live
before and dead now holds its true value, as the slots of a block kept once do. Every kernel reads such a slot only as
a term of a product that is -inf anyway.

Nothing is reused after a call where a row continues no row of the previous one (a new prompt, a position observed and
then missing again). Nothing is reused when anything else used the buffers in between (:attr:`~..state.State.epoch`:
another query, inside the session or not), or when the precision or the storage changed -- also under ``storage =
"auto"`` (the default), where every call picks the storage of its own values (see :func:`~.conditional.conditional`):
one that picks the other storage than the previous call's computes the values again.

Compact (``storage = "compact"``, :mod:`..kernels.compact`), a call's values are laid out by its own liveness, which
only the call knows: the values held, inside and top-down, are re-gathered into its rows by the call itself
(:func:`~..kernels.compact.regather`, ``carry``), in place or not -- every row's values at the pairs of columns still
live, which are among the previous ones. A call keeps rows for the values it needs only, so the others are given up.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ....decoder import Decoder
from ..kernels.blocks import move_blocks, move_table
from ..kernels.common import pack_aligned
from ..kernels.compact import CompactLayout
from ..kernels.live import Liveness
from .conditional import conditional
from .marginal import marginal


def _empty(numel: int, device) -> torch.Tensor:
    """New storage for moved values (see :meth:`~..state.State.replace_storage`): not filled."""
    return torch.empty(numel, device = device)


@dataclass
class _Kept:
    """What the buffers hold after a decoder's call: per sum region, whether ``node_mars`` holds its per-sample inside
    values for the call's rows, and whether ``node_flows`` holds its top-down values. They are valid while the state
    is the same (a weak reference: a backend moved to another device frees its old state), its buffers were not used
    since (``epoch``), and the next call has the same ``precision`` and ``storage`` (the one the call used: "dense"
    or "compact"; a call asking "auto" uses them if it picks it too). ``liveness``: the call's liveness
    (None: every lane), whose live lanes hold their true values in every region valid here; ``compact``: the layout of
    the values when compact (None: dense)."""
    state: weakref.ref
    epoch: int
    precision: str
    storage: str
    B: int
    inside: np.ndarray
    top_down: np.ndarray
    liveness: Optional[Liveness]
    compact: Optional[CompactLayout]


class DFAExactDecoder(Decoder):
    """
    A :class:`~pyjuice.constraints.decoder.Decoder` over a constrained circuit with the dfa_exact backend
    (:meth:`~..backend.DFAExactBackend.decoder`): a query reuses what the previous one computed where the evidence did
    not change it (see the module docstring). It keeps its own copy of the last call's evidence.
    """

    remembers = True

    def __init__(self, cc, **options):
        super().__init__(cc, **options)
        self._kept = None                           # what the buffers hold for the next call (`_Kept`)
        self._regions = None                        # see `_region_tables`
        self._layouts = {}                          # batch size -> (sum offsets, sum widths) as int64 arrays

    def __exit__(self, *exc) -> bool:
        self._kept, self._layouts = None, {}
        return super().__exit__(*exc)

    def _marginal(self, data: torch.Tensor, missing: torch.Tensor, precision: str = "fp32",
                  storage: str = "auto") -> torch.Tensor:
        be = self.cc.backend
        held, keep, compact, carry = self._reuse(data, missing, None, precision, storage, marginal = True)
        report = {}
        out = marginal(be, data, missing, precision, held = None if held is None else held[0], report = report,
                       storage = storage, carry = carry)
        self._after(data.size(0), precision, storage, keep, report, compact)
        return out

    def _conditional(self, data: torch.Tensor, missing: torch.Tensor, targets: Optional[torch.Tensor],
                     return_log_marginal: bool, candidates: Optional[torch.Tensor], precision: str = "fp32",
                     storage: str = "auto"):
        be = self.cc.backend
        held, keep, compact, carry = self._reuse(data, missing, targets, precision, storage, marginal = False)
        report = {}
        out = conditional(be, data, missing, targets, precision, return_log_marginal = return_log_marginal,
                          held = None if held is None else dict(inside = held[0], top_down = held[1]),
                          candidates = candidates, report = report, storage = storage, carry = carry)
        self._after(data.size(0), precision, storage, keep, report, compact)
        return out

    # ---------------------------------------------------------------------------------------------

    def _reuse(self, data: torch.Tensor, missing: torch.Tensor, targets: Optional[torch.Tensor], precision: str,
               storage: str, marginal: bool):
        """
        ``(held, keep, compact, carry)`` for a call: ``held`` None (nothing reused) or ``(inside, top_down)``, the
        values the call holds (bool per sum region, see :mod:`.plans`), in place or moved there; ``keep`` None or
        ``(inside, top_down)``, those the buffers still hold for the call's rows besides what it computes; compact, the
        layout of the values the buffers hold, while it is the call's rows' (``compact``: None if not), and what the
        call needs to re-gather those it holds into its own (``carry``, see :func:`~.conditional.conditional`).
        """
        kept, self._kept = self._kept, None         # given up until the call returns (if it raises: for good)
        be = self.cc.backend
        state, B = be.state, data.size(0)
        if kept is None or kept.state() is not state or kept.epoch != state.epoch or kept.precision != precision or \
                storage not in ("auto", kept.storage):                  # (auto: the call says if it uses them)
            return None, None, None, None
        if not marginal:
            asked = (missing if targets is None else targets).cpu().numpy()  # every missing position by default
            asked = asked.any(axis = 0) if asked.ndim == 2 else asked
            if asked.shape != (be.n,):
                return None, None, None, None                           # the query says what is wrong
        parents, new = self._match_rows(data, missing)                  # on the host (see `remembers`)
        if (parents < 0).any():
            return None, None, None, None

        _, cover, _, _ = self._region_tables(be.program())
        on_new = cover[new.any(axis = 0)]                               # [newly observed positions, regions]
        inside = kept.inside & ~on_new.any(axis = 0)                    # nothing new on the scope
        top_down = kept.top_down & on_new.all(axis = 0)                 # nothing new outside it
        if marginal:                                                    # every inside value; no top-down one
            held = (inside, np.zeros_like(top_down))
        else:                                                           # those of a scope missing a target; meeting one
            on_asked = cover[asked]
            held = (inside & ~on_asked.all(axis = 0), top_down & on_asked.any(axis = 0))
        identity = B == kept.B and (parents == np.arange(B)).all()
        if kept.compact is not None:                                    # compact: the call re-gathers what it holds
            carry = None
            if held[0].any() or held[1].any():
                carry = dict(layout = kept.compact, inside = held[0], regions = held[1],
                             parents = torch.from_numpy(parents.astype(np.int64)).to(be.device))
            if identity:
                return held, (inside, top_down), kept.compact, carry
            return held, held, None, carry
        if identity:
            return held, (inside, top_down), None, None                 # in place: all of them stay
        self._move(state, kept.B, B, parents, *held, kept.liveness)
        return held, held, None, None

    def _move(self, state, B_old: int, B: int, parents: np.ndarray, inside: np.ndarray, top_down: np.ndarray,
              liveness: Optional[Liveness]):
        """The ``inside`` and ``top_down`` values of the previous call's rows (``B_old`` of them) moved into this
        call's: row ``b`` gets row ``parents[b]``'s, in new storage laid out for a batch of ``B``, which replaces the
        old (in one pass: the old and the new layout of a storage overlap); only the lanes live under the previous
        call's ``liveness``, -inf at the others."""
        regs = np.concatenate([np.nonzero(inside)[0], np.nonzero(top_down)[0]])
        if regs.size == 0:
            return
        be = self.cc.backend
        _, _, rows, slots = self._region_tables(be.program())
        rows, slots = rows[regs], slots[regs]
        which = np.repeat(np.array([0, 1]), [int(inside.sum()), int(top_down.sum())])
        old_offset, old_width = self._layout(state, B_old)
        new_offset, new_width = self._layout(state, B)
        prog = be.program()
        bounds = np.asarray(prog.sum_ivs, dtype = np.int64).reshape(-1, 2)[prog.reg_iv_np[regs]]
        table = move_table(old_offset[regs], old_width[regs], new_offset[regs], new_width[regs], slots, rows, which,
                           bounds)
        tables, starts = pack_aligned([table.ravel(), parents], np.int64)
        tables = torch.from_numpy(tables).to(be.device)
        size = state.batch_layout(B)["node_size"]
        srcs, dsts = [], []
        for name, any_ in (("node_mars", inside.any()), ("node_flows", top_down.any())):
            if any_:
                dsts.append(_empty(size, be.device))
                srcs.append(state.replace_storage(name, dsts[-1]))
        move_blocks(srcs, dsts, tables[:starts[1]], B, int(rows.max()), int(slots.max()),
                    parents = tables[starts[1]:starts[1] + B], live = None if liveness is None else liveness.live,
                    prog = prog)

    def _after(self, B: int, precision: str, storage: str, keep, report: dict, compact: Optional[CompactLayout]):
        """What the buffers hold after a call: what it kept, and what it computed per sample, in the storage it used
        (``storage`` "auto": the one the call reports, or for a call that returned early the one its held values are
        in, compact if ``compact``). Compact, the values are laid out by the call's own layout if it made one (only
        those it has rows for, held or computed; a marginal's has none for top-down values), else by ``compact``, the
        one they were in before it (None: given up). A call that computed again the values held in the other storage
        keeps none of them."""
        be = self.cc.backend
        R = len(be.sum_regions)
        storage = report.get("storage", storage)
        if storage == "auto":
            storage = "compact" if compact is not None else "dense"
        if storage == "dense":                                          # (the values held before were compact)
            compact = None
        if report.get("dropped"):
            keep = None
        inside, top_down = (np.zeros(R, dtype = bool),) * 2 if keep is None else keep
        inside, top_down = inside.copy(), top_down.copy()
        plan = report.get("plan")
        if plan is not None:
            inside |= plan.sample[be.program().reg_iv_np]
            if plan.top_down is not None:
                top_down |= plan.top_down.sample
        if report.get("inputs_only"):                                   # node_mars's storage given up: none kept
            inside[:] = False
        if report.get("compact") is not None:                           # the call laid out its values
            compact = report["compact"]
            inside &= compact.mars["regions"]                           # (those with rows there)
            top_down &= plan.top_down.per_sample if plan.top_down is not None else False
        elif storage == "compact" and compact is None:
            inside[:] = False
            top_down[:] = False
        self._kept = _Kept(weakref.ref(be.state), be.state.epoch, precision, storage, B, inside, top_down,
                           report.get("liveness"), compact)

    def _region_tables(self, prog):
        """``(prog, cover, rows, slots)``: whether every sum region's scope holds every position (bool [n, R]), and
        every region's rows and slots per sample (int64 [R]); read once per Program."""
        if self._regions is None or self._regions[0] is not prog:
            regions = np.asarray(self.cc.backend.sum_regions, dtype = np.int64).reshape(-1, 3)
            bounds = np.asarray([prog.sum_ivs[k] for k in prog.reg_iv_host], dtype = np.int64).reshape(-1, 2)
            at = np.arange(self.cc.backend.n)[:, None]
            cover = (bounds[None, :, 0] <= at) & (at < bounds[None, :, 1])
            self._regions = (prog, cover, regions[:, 1] - regions[:, 0], regions[:, 2])
        return self._regions

    def _layout(self, state, B: int):
        """Every sum region's offset and row width (floats) in the buffers of a batch of ``B`` (int64 arrays)."""
        if B not in self._layouts:
            lay = state.batch_layout(B)
            self._layouts[B] = (np.asarray(lay["sum_offsets"], dtype = np.int64),
                                np.asarray(lay["sum_widths"], dtype = np.int64))
        return self._layouts[B]
