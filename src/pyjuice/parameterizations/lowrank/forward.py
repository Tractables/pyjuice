"""
The forward of :class:`~pyjuice.parameterizations.low_rank.LowRank` (:class:`_ForwardMixin`, one of its bases): the
forward: the circuit's fusion of the product layer below, the layer's pass -- the virtual batch (`.virtual`) where it
serves, the torch passes of both forms else (`exp_sum` in chunks of samples).
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple

from . import virtual as lowrank_virtual

#: The most (sample, node, child) triples of one node vector the `exp_sum` passes hold at a time: a larger batch is
#: taken in chunks of samples
LOWRANK_CHUNK_CELLS = 1 << 25


class _ForwardMixin:
    """The forward of :class:`~pyjuice.parameterizations.low_rank.LowRank`, a base of it."""

    # ------------------------------------------------------------------ forward

    def stream_fusion_plan(self, layer, fast_key: tuple, children) -> Optional[tuple]:
        """What `ParameterizedSumLayer._stream_fusion` asks for the circuit's decision to leave the product group
        below (``children``) to this layer's forward: taken where the forward goes through the mixture's virtual batch,
        whose `expand` then reads each product's value from its two children."""
        return (children,) if lowrank_virtual.fuses(self, layer, children) else None

    def stream_fusion_bk(self, layer, batch_size: int, dtype, op: bool, param_flows: bool) -> bool:
        """What `ParameterizedSumLayer._stream_fusion_bk` asks, for the circuit's decision and at every backward
        handed the product group below (`_bk_stream_children`: its forward and backward not run): taken where this
        layer's last forward went through the virtual batch (a backward at another batch size than that forward's is
        refused whatever this says), whose backward then reads each product's value from its two children and writes
        its flow as theirs."""
        return isinstance(layer.__dict__.get("_lr_state"), tuple) and lowrank_virtual.fuses_bk()

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        raise NotImplementedError("`LowRank` is served by `forward_layer`.")

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        The node values `log sum_j theta_b[i, j] c_b[j]`, written over the layer's rows of `node_mars`; every node
        vector's `log Z_b` (and `sum_exp`'s shifts) kept on the layer for the backward.
        """
        self._require_full_supply(layer, ns_tensors)

        children = kwargs.pop("stream_children", None)
        if lowrank_virtual.serves(self, layer, node_mars, element_mars):
            return lowrank_virtual.forward(self, layer, ns_tensors, node_mars, element_mars, params, kwargs,
                                           stream_children = children)
        if children is not None:                                # (handed the product layer: its values, as it forms them)
            from pyjuice.layer.fused_children import form_elements
            form_elements(element_mars, node_mars, children)

        state = layer.__dict__["_lr_state"] = dict()
        for ns_info, tensors in ns_tensors:
            A, Bm = self._as_given(tensors)
            geo = self._geometry(layer, ns_info, node_mars.device)
            theta = self._dense_params(geo, params)
            lc = self._child_values(geo, element_mars)

            if self.form == "exp_sum":
                ln, lz = self._forward_exp_sum(A, Bm, theta.log(), lc)
            else:
                ln, lz = self._forward_sum_exp(A, Bm, theta.double(), lc)

            node_mars[ns_info.nid_start:ns_info.nid_end, :] = ln.t()
            state[id(ns_info.ns)] = lz

        return None

    @staticmethod
    def _chunks(batch: int, cells: int):
        step = max(1, LOWRANK_CHUNK_CELLS // max(cells, 1))
        return [(s, min(s + step, batch)) for s in range(0, batch, step)]

    def _forward_exp_sum(self, A, Bm, ltheta, lc) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(log n, log Z)`, both `[B, M]`: `log w = log theta + A @ B` per sample, `Z = sum_j w`, `n = sum_j w c / Z`."""
        ln, lz = [], []
        for s, e in self._chunks(A.size(0), ltheta.numel()):
            lw = torch.bmm(A[s:e], Bm[s:e])                                  # [b, M, N]
            lw += ltheta[None, :, :]
            lz.append(torch.logsumexp(lw, dim = 2))
            lw += lc[s:e, None, :]
            ls = torch.logsumexp(lw, dim = 2)
            ln.append(torch.where(ls > -float("inf"), ls - lz[-1], ls))     # (no mass: 0, not 0 / 0)
        return torch.cat(ln, dim = 0), torch.cat(lz, dim = 0)

    @staticmethod
    def _mixture_terms(Bm, add: Optional[torch.Tensor], theta64):
        """
        For `sum_exp`: `x[b, r, j] = exp(B[b, r, j] (+ add[b, j]) - shift[b, r])`, float64, the shift each (sample,
        rank)'s largest exponent; and `y[b, r, i] = sum_j theta[i, j] x[b, r, j]`, `[B, k, M]`.
        """
        x = Bm if add is None else Bm + add[:, None, :]
        shift = x.amax(dim = 2, keepdim = True)
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        x = (x - shift).double().exp()
        return x, shift, torch.matmul(x, theta64.t())

    def _forward_sum_exp(self, A, Bm, theta64, lc) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(log n, log Z)`: `S[b, i] = sum_r e^A[b, i, r] * sum_j theta[i, j] e^B[b, r, j] c[b, j]`, `Z` the same
        without `c`, `n = S / Z`."""
        At = A.transpose(1, 2)                                               # [B, k, M]
        _, sc, yc = self._mixture_terms(Bm, lc, theta64)
        _, sb, yb = self._mixture_terms(Bm, None, theta64)
        ls = torch.logsumexp(At.double() + yc.log() + sc, dim = 1)           # [B, M]
        lz = torch.logsumexp(At.double() + yb.log() + sb, dim = 1)
        return torch.where(ls > -float("inf"), ls - lz, ls).float(), lz.float()    # (no mass: 0, not 0 / 0)
