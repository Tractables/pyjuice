"""
The glue of a decoder session's conditional fused into a few launches: what were chains of small torch ops, one kernel
launch each. The session's host issues every launch, and at small batch the host is what a call waits on (MEASURED,
HCLT-1024 left to right at B 16: 1.5 ms a call on the host against 1.0 ms of GPU work; the two chains below were ~35
launches and ~340 us of it).

* :func:`normalize_target_flows`: the targets' flows divided by their total per (target, sample), in log space -- a
  segmented log-sum-exp over every target's rows, and the rows shifted by it in place -- one launch for the ~16 of the
  torch form in `Decoder._normalize_target_flows`.
* :func:`categorical_conditional`: the conditional of one target of a Categorical input layer whose nodes read
  consecutive rows of the emission table (the left-to-right case), computed as the matmul form of
  `queries.distributions.categorical.backward` computes it -- the nodes' weights shifted by their largest per sample and
  transposed (one launch), the product with the table (cuBLAS, as there), normalized per sample (one launch) -- for the
  ~13 launches of that form. None wherever that form does not apply as it is, and the caller takes the general path.

Both compute what the torch forms compute, in the same order of operations per element; only the reductions' order
differs (the torch forms' `index_add_` adds atomically, in no fixed order), so results agree to the last bits of fp32.
Used by the decoder session alone.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

# libdevice's exp / log -- the accurate `expf` / `logf` torch's elementwise kernels compute, rather than Triton's
# approximate `tl.exp` -- so that the weights below are the torch form's to the bit
if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math


@triton_jit
def _segment_lse_kernel(flows_ptr, rows_ptr, logp_ptr, row_lo, rows_per_target, batch_size,
                        IS_RUN: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_B: tl.constexpr):
    """Program (target t, batch tile): the rows of t -- ``row_lo + [t * rows_per_target, (t + 1) * rows_per_target)``
    when ``IS_RUN``, else those entries of ``rows_ptr`` -- of ``flows`` ([rows, batch_size], log space): per sample, their
    largest value ``m`` (0 where not finite), ``log_p = m + log(sum exp(row - m))`` (0 where the sum is 0), every row
    shifted by ``-log_p`` in place and ``log_p`` stored at ``logp[t, b]``."""
    t = tl.program_id(0)
    offs_b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    first = t * rows_per_target

    m = tl.full((BLOCK_B,), -float("inf"), tl.float32)
    for r0 in range(0, rows_per_target, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rows_per_target
        if IS_RUN:
            rows = (row_lo + first + offs_r).to(tl.int64)
        else:
            rows = tl.load(rows_ptr + first + offs_r, mask = mask_r, other = 0).to(tl.int64)
        mask = mask_r[:, None] & mask_b[None, :]
        v = tl.load(flows_ptr + rows[:, None] * batch_size + offs_b[None, :], mask = mask, other = -float("inf"))
        m = tl.maximum(m, tl.max(v, axis = 0))
    m = tl.where((m == m) & (tl.abs(m) < float("inf")), m, 0.0)          # torch.isfinite

    tot = tl.zeros((BLOCK_B,), tl.float32)
    for r0 in range(0, rows_per_target, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rows_per_target
        if IS_RUN:
            rows = (row_lo + first + offs_r).to(tl.int64)
        else:
            rows = tl.load(rows_ptr + first + offs_r, mask = mask_r, other = 0).to(tl.int64)
        mask = mask_r[:, None] & mask_b[None, :]
        v = tl.load(flows_ptr + rows[:, None] * batch_size + offs_b[None, :], mask = mask, other = -float("inf"))
        tot += tl.sum(tl.where(mask, tlmath.exp(v - m[None, :]), 0.0), axis = 0)
    log_p = tl.where(tot > 0, m + tlmath.log(tot), 0.0)

    for r0 in range(0, rows_per_target, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rows_per_target
        if IS_RUN:
            rows = (row_lo + first + offs_r).to(tl.int64)
        else:
            rows = tl.load(rows_ptr + first + offs_r, mask = mask_r, other = 0).to(tl.int64)
        mask = mask_r[:, None] & mask_b[None, :]
        ptrs = flows_ptr + rows[:, None] * batch_size + offs_b[None, :]
        v = tl.load(ptrs, mask = mask, other = 0.0)
        tl.store(ptrs, v - log_p[None, :], mask = mask)

    tl.store(logp_ptr + t * batch_size + offs_b, log_p, mask = mask_b)


def _tiles(rows_per_target: int, batch_size: int):
    BLOCK_B = min(triton.next_power_of_2(batch_size), 64)
    BLOCK_R = max(min(triton.next_power_of_2(rows_per_target), 4096 // BLOCK_B), 1)
    return BLOCK_R, BLOCK_B


def normalize_target_flows(flows: torch.Tensor, rows, num_targets: int) -> torch.Tensor:
    """In place, ``flows[rows] -= log_p[target of the row]`` with ``log_p[t, b] = log sum_{rows of t} exp(flows[row, b])``
    (its usual guards: a non-finite largest value shifts by 0, a total of 0 gives ``log_p = 0``); returns ``log_p[0]``,
    [B]. ``rows``: a slice (every target that many consecutive rows of it, in order) or an int64 tensor (``num_targets``
    equal parts, in order)."""
    batch_size = flows.size(1)
    if isinstance(rows, slice):
        row_lo, total, rows_t, is_run = rows.start, rows.stop - rows.start, flows, True
    else:
        row_lo, total, rows_t, is_run = 0, rows.numel(), rows, False
    rows_per_target = total // num_targets
    logp = torch.empty([num_targets, batch_size], dtype = flows.dtype, device = flows.device)
    BLOCK_R, BLOCK_B = _tiles(rows_per_target, batch_size)
    grid = (num_targets, host_cdiv(batch_size, BLOCK_B))
    _segment_lse_kernel[grid](flows, rows_t, logp, row_lo, rows_per_target, batch_size,
                              IS_RUN = is_run, BLOCK_R = BLOCK_R, BLOCK_B = BLOCK_B)
    return logp[0]


@triton_jit
def _shifted_weights_t_kernel(flows_ptr, out_ptr, row_lo, num_rows, batch_size,
                              BLOCK_R: tl.constexpr, BLOCK_B: tl.constexpr):
    """``out[b, j] = exp(flows[row_lo + j, b] - shift[b])`` ([batch_size, num_rows]), ``shift[b]`` the largest of the rows
    at ``b`` (0 where not finite): `queries.distributions.common.target_weights` of one target, transposed."""
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size

    m = tl.full((BLOCK_B,), -float("inf"), tl.float32)
    for r0 in range(0, num_rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask = (offs_r < num_rows)[:, None] & mask_b[None, :]
        rows = (row_lo + offs_r).to(tl.int64)
        v = tl.load(flows_ptr + rows[:, None] * batch_size + offs_b[None, :], mask = mask, other = -float("inf"))
        m = tl.maximum(m, tl.max(v, axis = 0))
    m = tl.where((m == m) & (tl.abs(m) < float("inf")), m, 0.0)

    for r0 in range(0, num_rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask = (offs_r < num_rows)[:, None] & mask_b[None, :]
        rows = (row_lo + offs_r).to(tl.int64)
        v = tl.load(flows_ptr + rows[:, None] * batch_size + offs_b[None, :], mask = mask, other = -float("inf"))
        w = tlmath.exp(v - m[None, :])
        tl.store(out_ptr + offs_b[None, :].to(tl.int64) * num_rows + offs_r[:, None], w, mask = mask)


@triton_jit
def _normalize_rows_kernel(x_ptr, num_cols, BLOCK_C: tl.constexpr):
    """Row ``b`` of ``x`` ([rows, num_cols]) divided in place by its sum plus ``1e-12``."""
    base = tl.program_id(0).to(tl.int64) * num_cols
    s = tl.zeros((BLOCK_C,), tl.float32)
    for c0 in range(0, num_cols, BLOCK_C):
        offs = c0 + tl.arange(0, BLOCK_C)
        s += tl.load(x_ptr + base + offs, mask = offs < num_cols, other = 0.0)
    denom = tl.sum(s, axis = 0) + 1e-12
    for c0 in range(0, num_cols, BLOCK_C):
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < num_cols
        tl.store(x_ptr + base + offs, tl.load(x_ptr + base + offs, mask = mask, other = 0.0) / denom, mask = mask)


class CategoricalTargets:
    """What :func:`categorical_conditional` needs of an input layer per target variable, structure only (kept per
    session): ``(first node row in node_flows, number of nodes, first row of the emission table)``, or None where the
    fused form does not apply as such."""

    def __init__(self, layer):
        from ..distributions import MODULES, categorical
        self.layer = layer
        self.applies = MODULES.get(layer.dist_signature) is categorical
        self.cache = dict()

    def get(self, var: int):
        hit = self.cache.get(var, False)
        if hit is not False:
            return hit
        self.cache[var] = hit = self._find(var)
        return hit

    def _find(self, var: int):
        from ..distributions import categorical
        from ..distributions.common import layer_facts
        if not self.applies:
            return None
        layer = self.layer
        num_cats, uniform = categorical._facts(layer)
        f = layer_facts(layer)
        # The matmul form of `categorical.backward` for one target: every node with the same number of categories
        if not (uniform and num_cats >= min(categorical.MATMUL_MIN_CATS, categorical.MATMUL_MIN_CATS_ONE_TARGET)):
            return None
        if not (0 <= var < f.num_vars):
            return None
        a, b = int(f.var_ptr[var]), int(f.var_ptr[var + 1])
        run = f.var_nodes[a:b]
        n = b - a
        if n == 0 or int(run[-1] - run[0]) != n - 1:                       # the variable's nodes, consecutive
            return None
        s_pids = f.s_pids[run]
        if bool((s_pids % num_cats != 0).any()):                           # the table: `params` viewed as rows
            return None
        rows = s_pids // num_cats
        if int(rows[-1] - rows[0]) != n - 1 or (n > 1 and not bool((np.diff(rows) == 1).all())):
            return None                                                    # its rows ascending and consecutive
        return (layer._output_ind_range[0] + int(run[0]), n, int(rows[0]), num_cats)


def categorical_conditional(targets: CategoricalTargets, node_flows: torch.Tensor, target_vars) -> Optional[torch.Tensor]:
    """``[B, 1, num_cats]``: the conditional of the one target ``target_vars`` of ``targets.layer`` from the log-space
    flows ``node_flows``, as `queries.distributions.categorical.backward` computes it in its matmul form; None where
    that form does not apply as it is (several targets; nodes or table rows not consecutive; another distribution)."""
    if target_vars is None or len(target_vars) != 1:
        return None
    found = targets.get(int(target_vars[0]))
    if found is None:
        return None
    row_lo, n, trow, num_cats = found
    params = targets.layer.params
    if params.dtype != torch.float32 or params.numel() % num_cats != 0 or node_flows.dtype != torch.float32:
        return None
    batch_size = node_flows.size(1)
    weights_t = torch.empty([batch_size, n], dtype = torch.float32, device = node_flows.device)
    BLOCK_B = min(triton.next_power_of_2(batch_size), 32)
    BLOCK_R = max(min(triton.next_power_of_2(n), 4096 // BLOCK_B), 1)
    _shifted_weights_t_kernel[(host_cdiv(batch_size, BLOCK_B),)](node_flows, weights_t, row_lo, n, batch_size,
                                                                  BLOCK_R = BLOCK_R, BLOCK_B = BLOCK_B)
    table = params.view(-1, num_cats)[trow:trow + n]
    sums = torch.matmul(weights_t, table)                                  # [B, num_cats]
    BLOCK_C = min(triton.next_power_of_2(num_cats), 2048)
    _normalize_rows_kernel[(batch_size,)](sums, num_cats, BLOCK_C = BLOCK_C)
    return sums.view(batch_size, 1, num_cats)
