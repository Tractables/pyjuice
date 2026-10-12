"""
Decoding sessions over a constrained circuit (:meth:`~pyjuice.constraints.ConstrainedCircuit.decoder`).

A :class:`Decoder` answers the circuit's queries -- ``marginal``, ``conditional``, ``sample``, with the same arguments
and the same results -- inside a :func:`pyjuice.fast_inference` scope it opens and closes::

    with cc.decoder() as decoder:
        for t in range(n):
            logp = decoder.conditional(data[:, :t], targets = [t])

Every call computes its whole answer; nothing happens between calls. A backend's decoder may make a call cheaper by
reusing what its previous call computed where the evidence did not change -- :func:`match_rows` says, for every row
of a call, which row of the previous call it continues and what it newly observes. This class, every backend's by
default, reuses nothing.
"""

from contextlib import ExitStack
from typing import Optional

import torch

from pyjuice.utils.fast_inference import fast_inference

from .evidence import candidate_tensor, evidence, target_mask

#: the options of the :func:`pyjuice.fast_inference` scope a decoder opens; every other option is a query default
FAST_INFERENCE_OPTIONS = ("allow_param_copy", "allow_bf16_params", "allow_cudagraph")

#: :func:`match_rows` compares at most this many (row, previous row, position) triples at a time
MATCH_ELEMENTS = 1 << 24


def match_rows(data: torch.Tensor, missing: torch.Tensor, previous_data: torch.Tensor,
               previous_missing: torch.Tensor):
    """
    For every row of a call's evidence (``data`` and ``missing``, [B, n]), the row of the previous call's evidence
    ([B', n]) it continues -- one whose every observed position it still observes, with the same token -- and the
    positions it newly observes.

    :returns: ``(parents, newly_observed)``: int64 [B], the first such previous row (-1: none -- a new row, or one
        that no longer observes a position it did), and bool [B, n] (all False on a row without a parent), on
        ``data``'s device
    """
    dev = data.device
    previous_data, previous_missing = previous_data.to(dev), previous_missing.to(dev)
    B, n = data.shape
    Bp = previous_data.size(0)
    parents = torch.full((B,), -1, dtype = torch.long, device = dev)
    chunk = max(1, MATCH_ELEMENTS // max(1, Bp * n))
    for b0 in range(0, B, chunk):
        d, m = data[b0:b0 + chunk], missing[b0:b0 + chunk]
        same = (d[:, None, :] == previous_data[None, :, :]) & ~m[:, None, :]          # observed, the same token
        extends = (same | previous_missing[None, :, :]).all(dim = 2)                    # [rows, B']
        first = extends.to(torch.int32).argmax(dim = 1)                                # the first, ties included
        parents[b0:b0 + chunk] = torch.where(extends.any(dim = 1), first, -1)
    has = parents >= 0
    newly_observed = ~missing & previous_missing[parents.clamp(min = 0)] & has[:, None]
    return parents, newly_observed


class Decoder:
    """
    A decoding session over a :class:`~pyjuice.constraints.ConstrainedCircuit`, made by
    :meth:`ConstrainedCircuit.decoder` and used as a context manager: its queries are the circuit's (same arguments,
    same results), inside the :func:`pyjuice.fast_inference` scope it opens. Options given when it is made are
    defaults for every call, which a call's own keyword overrides. One decoder is open on a circuit at a time.

    A backend's decoder reuses work between calls by overriding :meth:`_marginal` and :meth:`_conditional` (and setting
    :attr:`remembers`, to keep the last call's evidence for :meth:`_match_rows`); this one calls the backend's queries.
    """

    #: whether the decoder keeps its own copy of the last call's evidence (for :meth:`_match_rows`)
    remembers = False

    def __init__(self, cc, **options):
        self.cc = cc
        self._fast_inference = {k: options.pop(k) for k in FAST_INFERENCE_OPTIONS if k in options}
        self._defaults = options
        self._scope = None                          # the open fast_inference scope
        self._used = False                          # a decoder is opened once
        self._previous = None                       # (data, missing) of the last call, when it remembers

    def __enter__(self) -> "Decoder":
        if self._used:
            raise RuntimeError("A decoder is opened once; make another with `cc.decoder()`.")
        if self.cc._decoder is not None:
            raise RuntimeError("This constrained circuit already has an open decoder; close it first.")
        self._used = True
        scope = ExitStack()
        scope.enter_context(fast_inference(**self._fast_inference))
        self._scope = scope
        self.cc._decoder = self
        return self

    def __exit__(self, *exc) -> bool:
        try:
            self._scope.close()
        finally:
            self._scope, self._previous = None, None
            self.cc._decoder = None
        return False

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, *,
                 mask_id: Optional[int] = None, **options) -> torch.Tensor:
        """``log p(C, e)`` for every sample, as :meth:`ConstrainedCircuit.marginal` (``mask_id``: None for the
        decoder's default, -1 unless it was made with another)."""
        options = self._options(options)
        data, missing = evidence(data, missing_mask, self._mask_id(mask_id), self.cc.n)
        out = self._marginal(data, missing, **options)
        self._remember(data, missing)
        return out

    def conditional(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                    targets: Optional[torch.Tensor] = None, return_log_marginal: bool = False, *,
                    mask_id: Optional[int] = None, candidates: Optional[torch.Tensor] = None, **options):
        """``log p(x_t = v | e, C)`` for every target, as :meth:`ConstrainedCircuit.conditional` (``mask_id``: None
        for the decoder's default, -1 unless it was made with another)."""
        options = self._options(options)
        data, missing = evidence(data, missing_mask, self._mask_id(mask_id), self.cc.n)
        B, n = data.size(0), self.cc.n
        out = self._conditional(data, missing, target_mask(targets, B, n), return_log_marginal,
                                candidate_tensor(candidates, B, n), **options)
        self._remember(data, missing)
        return out

    def sample(self, *args, **kwargs):
        """Samples under the constraint, as :meth:`ConstrainedCircuit.sample`."""
        return self.cc.backend.sample(*args, **self._options(kwargs))

    # ---------------------------------------------------------------------------------------------
    # What a backend's decoder overrides
    # ---------------------------------------------------------------------------------------------

    def _marginal(self, data: torch.Tensor, missing: torch.Tensor, **options) -> torch.Tensor:
        """The marginal of the canonical evidence (see :mod:`pyjuice.constraints.evidence`)."""
        return self.cc.backend.marginal(data, missing, **options)

    def _conditional(self, data: torch.Tensor, missing: torch.Tensor, targets: Optional[torch.Tensor],
                     return_log_marginal: bool, candidates: Optional[torch.Tensor], **options):
        """The conditionals of the canonical evidence, targets and candidates (see
        :mod:`pyjuice.constraints.evidence`)."""
        return self.cc.backend.conditional(data, missing, targets, return_log_marginal = return_log_marginal,
                                           candidates = candidates, **options)

    def _match_rows(self, data: torch.Tensor, missing: torch.Tensor):
        """:func:`match_rows` against the last call's evidence; None before the first call."""
        if self._previous is None:
            return None
        return match_rows(data, missing, *self._previous)

    # ---------------------------------------------------------------------------------------------

    def _options(self, options: dict) -> dict:
        if self._scope is None:
            raise RuntimeError("Use the decoder inside its `with` block: `with cc.decoder() as decoder: ...`.")
        self.cc._check_pc_unchanged()
        return {**{k: v for k, v in self._defaults.items() if k != "mask_id"}, **options}

    def _mask_id(self, mask_id: Optional[int]) -> int:
        return self._defaults.get("mask_id", -1) if mask_id is None else mask_id

    def _remember(self, data: torch.Tensor, missing: torch.Tensor):
        if self.remembers:
            self._previous = (data.clone(), missing.clone())                # never the caller's tensors
