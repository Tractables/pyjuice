"""
Decoding sessions over a constrained circuit (:meth:`~pyjuice.constraints.ConstrainedCircuit.decoder`).

A :class:`Decoder` answers the circuit's queries -- ``marginal``, ``conditional``, ``sample``, with the same arguments
and the same results (at an approximate precision, within its error) -- inside a :func:`pyjuice.fast_inference` scope
it opens and closes::

    with cc.decoder() as decoder:
        for t in range(n):
            logp = decoder.conditional(data[:, :t], targets = [t])

Every call computes its whole answer; nothing happens between calls. A backend's decoder may make a call cheaper by
reusing what its previous call computed where the evidence did not change -- :func:`match_rows` says, for every row
of a call, which row of the previous call it continues and what it newly observes. This class, every backend's by
default, reuses nothing. :meth:`Decoder.reset` forgets what a session kept: the next call starts afresh.
"""

from contextlib import ExitStack
from typing import Optional

import numpy as np
import torch
from numba import njit

from pyjuice.utils.fast_inference import fast_inference

from .evidence import candidate_tensor, evidence, target_mask

#: the options of the :func:`pyjuice.fast_inference` scope a decoder opens; every other option is a query default
FAST_INFERENCE_OPTIONS = ("allow_param_copy", "allow_bf16_params", "allow_cudagraph")
#: the per-sample inputs a decoder takes for its whole session, as ``pc(x, ...)`` takes them (keyword arguments of
#: the PC's layers: sum layers' external parameters, input layers' soft evidence)
EXTERNAL_INPUTS = ("sum_external_params", "categorical_evidence_logp", "soft_evidence_cat_ids",
                   "soft_evidence_value_mask")

def match_rows(data: np.ndarray, missing: np.ndarray, previous_data: np.ndarray, previous_missing: np.ndarray):
    """
    For every row of a call's evidence (``data`` and ``missing``, [B, n], on the host), the row of the previous call's
    evidence ([B', n]) it continues -- one whose every observed position it still observes, with the same token -- and
    the positions it newly observes. A row continuing the previous row at its own index (left to right, any order, a
    sample's draws) costs ``O(n)``; any other, up to ``O(B' n)``.

    :returns: ``(parents, newly_observed)``: int64 [B], the previous row at the row's own index if it continues that
        one, else the first such previous row (-1: none -- a new row, or one that no longer observes a position it
        did), and bool [B, n] (all False on a row without a parent)
    """
    return _match_rows(np.ascontiguousarray(data, dtype = np.int64), np.ascontiguousarray(missing, dtype = bool),
                       np.ascontiguousarray(previous_data, dtype = np.int64),
                       np.ascontiguousarray(previous_missing, dtype = bool))


@njit(cache = True)
def _continues(data, missing, previous_data, previous_missing, b, p):
    for j in range(data.shape[1]):
        if not previous_missing[p, j] and (missing[b, j] or data[b, j] != previous_data[p, j]):
            return False
    return True


@njit(cache = True)
def _match_rows(data, missing, previous_data, previous_missing):
    B, n = data.shape
    Bp = previous_data.shape[0]
    parents = np.full(B, -1, dtype = np.int64)
    newly_observed = np.zeros((B, n), dtype = np.bool_)
    for b in range(B):
        if b < Bp and _continues(data, missing, previous_data, previous_missing, b, b):
            parents[b] = b
        else:
            for p in range(Bp):
                if _continues(data, missing, previous_data, previous_missing, b, p):
                    parents[b] = p
                    break
        p = parents[b]
        if p >= 0:
            for j in range(n):
                newly_observed[b, j] = previous_missing[p, j] and not missing[b, j]
    return parents, newly_observed


class Decoder:
    """
    A decoding session over a :class:`~pyjuice.constraints.ConstrainedCircuit`, made by
    :meth:`ConstrainedCircuit.decoder` and used as a context manager: its queries are the circuit's (same arguments,
    same results -- at an approximate precision, within its error), inside the :func:`pyjuice.fast_inference` scope it
    opens. Options given when it is made are defaults for every call, which a call's own keyword overrides -- but the
    per-sample inputs (:data:`EXTERNAL_INPUTS`), which hold for the whole session: checked against the PC when the
    decoder is made, and given to every call as ``external`` (the backend's
    :meth:`~.inference.base.Backend.external_inputs`). One decoder is open on a circuit at a time.

    A backend's decoder reuses work between calls by overriding :meth:`_marginal` and :meth:`_conditional` (and setting
    :attr:`remembers`, to keep the last call's evidence for :meth:`_match_rows`); this one calls the backend's queries.
    """

    #: whether the decoder keeps its own copy of the last call's evidence (for :meth:`_match_rows`); a call's
    #: evidence then reaches its hooks on the host, where rows are matched, copied there once
    remembers = False

    def __init__(self, cc, **options):
        self.cc = cc
        self._fast_inference = {k: options.pop(k) for k in FAST_INFERENCE_OPTIONS if k in options}
        inputs = {k: options.pop(k) for k in EXTERNAL_INPUTS if options.get(k) is not None}
        self._external = cc.backend.external_inputs(inputs) if inputs else None
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
        self._scope = self._open_scope()
        self.cc._decoder = self
        return self

    def __exit__(self, *exc) -> bool:
        try:
            self._scope.close()
        finally:
            self._scope, self._previous = None, None
            self.cc._decoder = None
        return False

    def reset(self) -> None:
        """
        Forget what the session kept between calls: the next call computes its whole answer, as the first call of a
        new decoder with the same options and per-sample inputs would. The values a backend's decoder reuses and the
        last call's evidence are dropped, and the session's :func:`pyjuice.fast_inference` scope is closed and opened
        again: what it kept -- parameter copies, input-independent values, token tables -- is computed again, from
        the parameters as they are now (a session inside another ``fast_inference`` scope: that scope keeps them
        until it exits). The decoder stays open.
        """
        if self._scope is None:
            raise RuntimeError("Use the decoder inside its `with` block: `with cc.decoder() as decoder: ...`.")
        self._forget()
        self._previous = None
        try:
            self._scope.close()
        finally:
            self._scope = self._open_scope()

    def _open_scope(self) -> ExitStack:
        scope = ExitStack()
        scope.enter_context(fast_inference(**self._fast_inference))
        return scope

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, *,
                 mask_id: Optional[int] = None, **options) -> torch.Tensor:
        """``log p(C, e)`` for every sample, as :meth:`ConstrainedCircuit.marginal` (``mask_id``: None for the
        decoder's default, -1 unless it was made with another)."""
        options = self._options(options)
        data, missing = self._evidence(data, missing_mask, mask_id)
        out = self._marginal(data, missing, **options)
        self._remember(data, missing)
        return out

    def conditional(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                    targets: Optional[torch.Tensor] = None, return_log_marginal: bool = False, *,
                    mask_id: Optional[int] = None, candidates: Optional[torch.Tensor] = None, **options):
        """``log p(x_t = v | e, C)`` for every target, as :meth:`ConstrainedCircuit.conditional` (``mask_id``: None
        for the decoder's default, -1 unless it was made with another)."""
        options = self._options(options)
        data, missing = self._evidence(data, missing_mask, mask_id)
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

    def _forget(self):
        """Drop what the decoder kept for its next call (:meth:`reset`): this one keeps nothing."""

    def _match_rows(self, data: torch.Tensor, missing: torch.Tensor):
        """:func:`match_rows` of a call's evidence (on the host) against the last call's; None before the first
        call."""
        if self._previous is None:
            return None
        return match_rows(data.numpy(), missing.numpy(), *self._previous)

    # ---------------------------------------------------------------------------------------------

    def _options(self, options: dict) -> dict:
        if self._scope is None:
            raise RuntimeError("Use the decoder inside its `with` block: `with cc.decoder() as decoder: ...`.")
        self.cc._check_pc_unchanged()
        bound = [k for k in options if k in EXTERNAL_INPUTS]
        if bound:
            raise TypeError(f"{', '.join(f'`{k}`' for k in bound)}: per-sample inputs hold for a decoder's whole "
                            f"session; give them to `cc.decoder(...)`.")
        options = {**{k: v for k, v in self._defaults.items() if k != "mask_id"}, **options}
        if self._external is not None:
            options["external"] = self._external
        return options

    def _evidence(self, data, missing_mask, mask_id: Optional[int]):
        """The canonical evidence (:func:`~pyjuice.constraints.evidence.evidence`); on the host if the decoder
        remembers it (copied there as given). The positions the session's per-sample inputs leave free (``free``, see
        :meth:`~.inference.base.Backend.external_inputs`) are missing too: marginalized, whatever ``data`` holds."""
        mask_id = self._defaults.get("mask_id", -1) if mask_id is None else mask_id
        if self.remembers:
            data = torch.as_tensor(data).cpu()
            missing_mask = None if missing_mask is None else torch.as_tensor(missing_mask).cpu()
        data, missing = evidence(data, missing_mask, mask_id, self.cc.n)
        free = getattr(self._external, "free", None)
        if free is None or free.size(0) not in (1, missing.size(0)):         # (a wrong batch: the query says so)
            return data, missing
        host = getattr(self._external, "free_host", None)
        free = torch.from_numpy(host) if host is not None and missing.device.type == "cpu" else free.to(missing.device)
        if (missing & free).any():
            raise ValueError("Positions marked missing are left free by the decoder's per-sample inputs too (a soft "
                             "evidence's `soft_evidence_value_mask` False there): a missing token contributes 1 "
                             "whatever its weights, as in `pc(x, missing_mask = ...)`, a free one the sum of its "
                             "weights -- mark each position one way.")
        return data, missing | free

    def _remember(self, data: torch.Tensor, missing: torch.Tensor):
        if self.remembers:
            self._previous = (data.numpy().copy(), missing.numpy().copy())  # never the caller's tensors
