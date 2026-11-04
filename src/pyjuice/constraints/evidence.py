"""
A constrained query's inputs, from the forms users give them to the one form every backend reads:

* the evidence -- ``data`` [B, n] token ids, or a prefix [B, L] with ``L <= n`` (the positions after it missing),
  where a token equal to ``mask_id`` is missing, and an optional ``missing_mask`` ([n] or [B, n]) marking more --
  becomes ``(data [B, n], missing [B, n] bool)`` on ``data``'s device (:func:`evidence`);
* the targets -- a bool mask ([n] or [B, n]) or positions ([k] or [B, k], -1 padding) -- become a bool mask
  (:func:`target_mask`);
* the candidates -- token ids for every target ([k]), every target of a sample ([B, k]) or every target position of a
  sample ([B, n, k]) -- become [B, n, k], a view when they are shared (:func:`candidate_tensor`).
"""

import contextlib
from typing import Optional

import numpy as np
import torch
from numba import njit


def current_device(device):
    """A query's context on ``device``: Triton launches kernels on the CURRENT device, whatever device their tensors
    are on, and :meth:`TensorCircuit.to` does not change it -- a kernel launched there faults on the tensors of another
    GPU ("an illegal memory access", at a later launch, as the launches are asynchronous)."""
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def evidence(data, missing_mask, mask_id: int, n: int):
    """
    ``(data, missing)``: ``data`` [B, n] (a prefix padded with ``mask_id``) and ``missing`` [B, n] bool, True where
    ``data == mask_id`` or ``missing_mask`` says so, both on ``data``'s device.

    :param data: [B, n] or [B, L] (``L <= n``) token ids
    :param missing_mask: None, [n] or [B, n]
    :param mask_id: the token id that marks a missing position (never a real token)
    :param n: the number of positions
    """
    data = torch.as_tensor(data)
    if data.dim() != 2 or data.size(1) > n:
        raise ValueError(f"`data` must be [batch, {n}] or a prefix [batch, L] with L <= {n}, got {list(data.shape)}.")
    B, L = data.shape
    if L < n:
        data = torch.cat([data, data.new_full((B, n - L), mask_id)], dim = 1)
    missing = data == mask_id
    if missing_mask is not None:
        m = torch.as_tensor(missing_mask, device = data.device)
        if m.dim() == 1:
            m = m[None, :]
        if m.dim() != 2 or m.size(1) != n or m.size(0) not in (1, B):
            raise ValueError(f"`missing_mask` must be [{n}] or [{B}, {n}], got {list(m.shape)}.")
        missing = missing | m.bool()
    return data, missing


def target_mask(targets, B: int, n: int) -> Optional[torch.Tensor]:
    """
    None (every missing position), or a bool mask: ``targets`` itself when it is one ([n] or [B, n]), else the mask
    [B, n] of the positions it lists ([k] for every sample or [B, k] per sample, integers in 0..n-1, -1 padding),
    built on the host.
    """
    if targets is None:
        return None
    if isinstance(targets, torch.Tensor):
        if targets.dtype == torch.bool:
            return targets
        pos = targets.cpu().numpy()
    else:
        pos = np.asarray(targets)                                              # numpy: a few us, not tens
        if pos.dtype == bool:
            return torch.from_numpy(pos)
    if pos.dtype.kind not in "iu":
        raise TypeError(f"`targets` must be a bool mask or integer positions, got {pos.dtype}.")
    if pos.ndim not in (1, 2) or (pos.ndim == 2 and pos.shape[0] != B):
        raise ValueError(f"`targets` given as positions must be [k] or [{B}, k], got {list(pos.shape)}.")
    mask, ok = _positions_mask(np.ascontiguousarray(pos[None] if pos.ndim == 1 else pos, dtype = np.int64), B, n)
    if not ok:
        raise ValueError(f"`targets` given as positions must be in 0..{n - 1} (-1 pads).")
    return torch.from_numpy(mask)


@njit(cache = True)
def _positions_mask(pos, B, n):
    """``(mask, ok)``: bool [B, n], True at the positions ``pos`` lists for each sample (int64 [1, k]: every sample's,
    or [B, k]; -1 pads), and whether they are all in 0..n-1 (but -1)."""
    mask = np.zeros((B, n), dtype = np.bool_)
    for b in range(B):
        r = 0 if pos.shape[0] == 1 else b
        for j in range(pos.shape[1]):
            p = pos[r, j]
            if p < -1 or p >= n:
                return mask, False
            if p >= 0:
                mask[b, p] = True
    return mask, True


def candidate_tensor(candidates, B: int, n: int) -> Optional[torch.Tensor]:
    """
    None (every token), or the candidates as [B, n, k] integer token ids: ``[k]`` (every target) and ``[B, k]``
    (every target of sample ``b``) expanded as views, ``[B, n, k]`` (per sample and position) as given. Ids outside
    the vocabulary (-1) pad.
    """
    if candidates is None:
        return None
    c = torch.as_tensor(candidates)
    if c.dtype == torch.bool or c.is_floating_point() or c.is_complex():
        raise TypeError(f"`candidates` must be integer token ids, got {c.dtype}.")
    if c.dim() == 1:
        return c[None, None, :].expand(B, n, -1)
    if c.dim() == 2 and c.size(0) == B:
        return c[:, None, :].expand(B, n, -1)
    if c.dim() == 3 and tuple(c.shape[:2]) == (B, n):
        return c
    raise ValueError(f"`candidates` must be [k], [{B}, k] or [{B}, {n}, k], got {list(c.shape)}.")
