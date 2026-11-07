"""
The backward of :class:`~pyjuice.parameterizations.low_rank.LowRank` (:class:`_BackwardMixin`, one of its bases): the
backward: the layer's pass -- the virtual batch where its forward went through it, the torch passes else -- with the
parameter flows, the denominator statistic and the factors' gradients.
"""

from __future__ import annotations

import torch

from . import virtual as lowrank_virtual


class _BackwardMixin:
    """The backward of :class:`~pyjuice.parameterizations.low_rank.LowRank`, a base of it."""

    # ------------------------------------------------------------------ backward

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars, element_mars, params,
                           **kwargs) -> None:
        if not kwargs.get("logspace_flows", False):
            raise NotImplementedError("`LowRank` requires `logspace_flows = True` (the default).")
        self._require_full_supply(layer, ns_tensors)
        state = layer.__dict__.get("_lr_state")
        if isinstance(state, tuple):                            # (the forward went through the virtual batch)
            fresh = state[1] == node_mars.size(1)
        else:
            fresh = state is not None and all(id(i.ns) in state and state[id(i.ns)].size(0) == node_mars.size(1)
                                              for i, _ in ns_tensors)
        if not fresh:
            raise RuntimeError("`LowRank`: the backward needs the forward pass of the same batch first.")

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars,
                            element_mars, params, param_flows = None, denom_param_flows = None, **kwargs) -> None:
        """
        Everything the backward owes for the layer, per node vector, with `f_b[i]` the node flow:

        .. code-block:: text

            q_b[i, j]   = f_b[i] * theta_b[i, j] * c_b[j] / n_b[i]           # the flow along the edge
            element flow[j, b] = sum_i q_b[i, j]          F+[i, j] += sum_b q_b[i, j]
            W[i, j]    += sum_b f_b[i] * g_b[i, j] / Z_b[i]                  # with `apply_z_correction`: F- / theta
            dLL/dg-log  = q_b - f_b * theta_b                                # chained into dLL/dA, dLL/dB

        The element flows of a child several node vectors of the layer share add up.
        """
        dev = node_flows.device
        state = layer._lr_state
        if isinstance(state, tuple):
            # (with the product group below, where the layer handed it over: `stream_fusion_bk` said so in this call)
            return lowrank_virtual.backward(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                                            node_mars, element_mars, params, param_flows, denom_param_flows, kwargs,
                                            stream_children = layer.__dict__.get("_bk_stream_children"))
        geos = [self._geometry(layer, ns_info, dev) for ns_info, _ in ns_tensors]
        for geo in geos:
            element_flows[geo["kids"], :] = -float("inf")

        want_denom = param_flows is not None and denom_param_flows is not None and self.apply_z_correction

        for (ns_info, tensors), grads, geo in zip(ns_tensors, ns_grad_tensors, geos):
            A, Bm = self._as_given(tensors)
            theta = self._dense_params(geo, params)
            lc = self._child_values(geo, element_mars)
            rows = slice(ns_info.nid_start, ns_info.nid_end)
            lf = node_flows[rows, :].t()                                     # [B, M], log
            ln = node_mars[rows, :].t()
            lz = state[id(ns_info.ns)]

            fn = self._backward_exp_sum if self.form == "exp_sum" else self._backward_sum_exp
            lef, Fp, Fm, dA, dB = fn(A, Bm, theta, lc, lf, ln, lz, param_flows is not None, want_denom,
                                     grads is not None)

            kids = geo["kids"]
            element_flows[kids, :] = torch.logaddexp(element_flows[kids, :],
                                                     lef.t()[geo["cols"], :].to(element_flows.dtype))

            if param_flows is not None:
                self._scatter_add(geo, param_flows, ns_info.ns._param_flow_range[0], Fp)
            if want_denom:
                off = self._denom_offsets(layer)[id(ns_info.ns)][0]
                self._scatter_add(geo, denom_param_flows, off, Fm)
            if grads is not None:                                   # (the buffers as stored: batch innermost)
                grads[0].add_(dA.permute(1, 2, 0))
                grads[1].add_(dB.permute(1, 2, 0))

        return None

    @staticmethod
    def _log_ratio(a, b):
        """`a - b` where `a` is finite, -inf elsewhere (a node without flow, whatever its value)."""
        return torch.where(torch.isfinite(a), a - b, torch.full_like(a, -float("inf")))

    def _backward_exp_sum(self, A, Bm, theta, lc, lf, ln, lz, want_flows, want_denom, want_grads):
        ltheta = theta.log()
        M, N = ltheta.shape
        lu = self._log_ratio(lf, ln + lz)                                    # log f / (n Z)
        lv = self._log_ratio(lf, lz)                                         # log f / Z
        lef = []
        Fp = torch.zeros(M, N, dtype = torch.float32, device = theta.device) if want_flows else None
        Fm = torch.zeros(M, N, dtype = torch.float32, device = theta.device) if want_denom else None
        dA = torch.empty_like(A) if want_grads else None
        dB = torch.empty_like(Bm) if want_grads else None
        for s, e in self._chunks(A.size(0), ltheta.numel()):
            lw = torch.bmm(A[s:e], Bm[s:e])
            lw += ltheta[None, :, :]                                         # log theta g
            lq = lw + lu[s:e, :, None]
            lq += lc[s:e, None, :]                                           # log q
            lef.append(torch.logsumexp(lq, dim = 1))                         # [b, N]
            q = lq.exp_()
            if want_flows:
                Fp += q.sum(dim = 0)
            if want_denom or want_grads:
                t = (lw + lv[s:e, :, None]).exp_()                           # f theta_b
                if want_denom:
                    Fm += t.sum(dim = 0)
                if want_grads:
                    q -= t                                                   # dLL / d log g
                    dA[s:e] = torch.bmm(q, Bm[s:e].transpose(1, 2))
                    dB[s:e] = torch.bmm(A[s:e].transpose(1, 2), q)
        if want_denom:                                                       # sum_b f theta_b -> W, free of theta
            Fm = torch.where(theta > 0, Fm / theta, torch.zeros_like(Fm))
        return torch.cat(lef, dim = 0), Fp, Fm, dA, dB

    def _backward_sum_exp(self, A, Bm, theta, lc, lf, ln, lz, want_flows, want_denom, want_grads):
        theta64 = theta.double()
        At = A.transpose(1, 2).double()                                      # [B, k, M]
        xc, sc, yc = self._mixture_terms(Bm, lc, theta64)                    # e^B c, its shift, sum_j theta e^B c
        lu = self._log_ratio(lf, ln + lz).double()
        # p[b, r, i] = f e^A e^shift / S: what node i's flow puts on rank r, per unit of `y`
        p = (lu[:, None, :] + At + sc).exp()
        pt = torch.matmul(p, theta64)                                        # [B, k, N]
        lef = (xc * pt).sum(dim = 1).log().float()                           # [B, N]

        Fp = (theta64 * torch.einsum("bri,brj->ij", p, xc)).float() if want_flows else None
        Fm, dA, dB = None, None, None
        if want_denom or want_grads:
            xb, sb, yb = self._mixture_terms(Bm, None, theta64)
            lv = self._log_ratio(lf, lz).double()
            pz = (lv[:, None, :] + At + sb).exp()                            # f e^A e^shift / Z
            if want_denom:                                                   # W: `F-` without its `theta`
                Fm = torch.einsum("bri,brj->ij", pz, xb).float()
            if want_grads:
                dA = (p * yc - pz * yb).transpose(1, 2).float()
                dB = (xc * pt - xb * torch.matmul(pz, theta64)).float()
        return lef, Fp, Fm, dA, dB
