"""
The backward of :class:`~pyjuice.parameterizations.low_rank.LowRank` (:class:`_BackwardMixin`, one of its bases): the
backward: the layer's pass -- the virtual batch where its forward went through it, the torch passes else -- with the
parameter flows, the denominator statistic and the factors' gradients.
"""

from __future__ import annotations

import torch

from . import virtual as lowrank_virtual
from . import forward as lowrank_forward
from .forward import exp_sum_bf16, exp_sum_fused, exp_sum_module


_EMPTY = {}


def _empty(device) -> torch.Tensor:
    """An empty fp32 tensor on `device` (the kernels' "not wanted"), made once a device."""
    e = _EMPTY.get(device)
    if e is None:
        e = _EMPTY[device] = torch.empty(0, dtype = torch.float32, device = device)
    return e


class _BackwardMixin:
    """The backward of :class:`~pyjuice.parameterizations.low_rank.LowRank`, a base of it."""

    # ------------------------------------------------------------------ backward

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars, element_mars, params,
                           **kwargs) -> None:
        if not kwargs.get("logspace_flows", False):
            raise NotImplementedError("`LowRank` requires `logspace_flows = True` (the default).")
        self._require_full_supply(layer, ns_tensors)
        if self.top_k is not None:                              # (the per-candidate form: its reference passes)
            if layer.__dict__.get("_lr_topk_batch") != node_mars.size(1):
                raise RuntimeError("`LowRank`: the backward needs the forward pass of the same batch first.")
            return None
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
        if self.top_k is not None:                              # (the per-candidate form: its reference passes)
            from . import top_k as lowrank_top_k
            return lowrank_top_k.sum_backward(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                                              node_mars, element_mars, params, param_flows, denom_param_flows, kwargs)
        state = layer._lr_state
        if isinstance(state, tuple):
            # (with the product group below, where the layer handed it over: `stream_fusion_bk` said so in this call)
            return lowrank_virtual.backward(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                                            node_mars, element_mars, params, param_flows, denom_param_flows, kwargs,
                                            stream_children = layer.__dict__.get("_bk_stream_children"))
        gviews = kwargs.get("sum_external_params_grad")
        fresh = getattr(gviews, "fresh", False)                 # (destinations not zeroed: each one's first writer here
                                                                #  initializes it -- `init` below)
        geos = [self._geometry(layer, ns_info, dev) for ns_info, _ in ns_tensors]
        # (the product group below handed over: its children's values read and their flows written by the fp16
        #  kernels, else formed / scattered here)
        sc = layer.__dict__.get("_bk_stream_children")
        if sc is not None and not all(self._h_bk_ok(layer, ns_info, grads, node_flows, element_flows)
                                      for (ns_info, _), grads in zip(ns_tensors, ns_grad_tensors)):
            from pyjuice.layer.fused_children import form_elements, scatter_flows
            form_elements(element_mars, node_mars, sc)
            self._post_backward(layer, ns_tensors, ns_grad_tensors, geos, None, node_flows, element_flows, node_mars,
                                element_mars, params, param_flows, denom_param_flows, gviews, fresh, kwargs)
            scatter_flows(node_flows, element_flows, sc, layer._element_rows())
            return None
        return self._post_backward(layer, ns_tensors, ns_grad_tensors, geos, sc, node_flows, element_flows, node_mars,
                                   element_mars, params, param_flows, denom_param_flows, gviews, fresh, kwargs)

    def _post_backward(self, layer, ns_tensors, ns_grad_tensors, geos, sc, node_flows, element_flows, node_mars,
                       element_mars, params, param_flows, denom_param_flows, gviews, fresh, kwargs) -> None:
        """`post_backward_layer`'s node vectors; `sc`: the product group below handed to the fp16 kernels (every node
        vector theirs: `_h_bk_ok`), whose element flows they write as its children's node flows."""
        dev = node_flows.device
        state = layer._lr_state
        own = sc is not None or self._kids_own(layer, dev)    # (each node vector's element flows its pass's alone:
        if not own:                                             #  written by it, no -inf fill first -- else filled
            for geo in geos:                                    #  here, then log-added. A fill, not an index_put of
                element_flows.index_fill_(0, geo["kids"], -float("inf"))    # a host scalar: its copy to the GPU
                                                                #  synchronized every layer)

        want_denom = param_flows is not None and denom_param_flows is not None and self.apply_z_correction

        for (ns_info, tensors), grads, geo in zip(ns_tensors, ns_grad_tensors, geos):
            denom = denom_param_flows if want_denom else None
            init = fresh and gviews.claim(grads)                # (the node vector its destination's first writer)
            if self._exp_sum_backward_fused(layer, ns_info, tensors, grads, geo, node_flows, element_flows, node_mars,
                                            element_mars, params, param_flows, denom, exp_sum_bf16(kwargs), init, own,
                                            sc):
                continue
            if own:
                element_flows.index_fill_(0, geo["kids"], -float("inf"))
            A, Bm = (t.float() for t in self._as_given(tensors))  # (fp16 factors as float32; views as they lie)
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
            if grads is not None:                                   # (the buffers in the storage's layout)
                gA, gB = self._to_stored(dA, dB)
                (grads[0].copy_ if init else grads[0].add_)(gA)
                (grads[1].copy_ if init else grads[1].add_)(gB)

        return None

    def _h_bk_ok(self, layer, ns_info, grads, node_flows, element_flows) -> bool:
        """Whether a node vector's backward goes through the fp16 kernels (`_exp_sum_backward_fused`): its forward took
        them, the flow buffers fp32, the gradient buffers fp32 with each sample's block contiguous."""
        if id(ns_info.ns) not in layer.__dict__.get("_lr_h", {}):
            return False
        if node_flows.dtype != torch.float32 or element_flows.dtype != torch.float32:
            return False
        return grads is None or all(g.dtype == torch.float32 and lowrank_forward.samples_contiguous(g) for g in grads)

    def _h_stream_bk(self, layer, batch_size: int) -> bool:
        """`stream_fusion_bk` for the fp16 exp_sum kernels: this layer's last forward went through them for every node
        vector (`_lr_h`), at this batch, no child reached by two of them -- whose backward then reads each product's
        value from its two children and writes its flow as theirs (`_post_backward`'s `sc`)."""
        state, rounded = layer.__dict__.get("_lr_state"), layer.__dict__.get("_lr_h")
        if not (lowrank_forward.EXP_SUM_H_STREAM and isinstance(state, dict) and isinstance(rounded, dict)):
            return False
        infos = layer.external_node_infos
        if not infos or not all(id(i.ns) in rounded and id(i.ns) in state and state[id(i.ns)].size(0) == batch_size
                                for i in infos):
            return False
        return self._kids_own(layer, state[id(infos[0].ns)].device)

    def _exp_sum_backward_fused(self, layer, ns_info, tensors, grads, geo, node_flows, element_flows, node_mars,
                                element_mars, params, param_flows, denom, bf16, init = False, own = False,
                                sc = None) -> bool:
        """A node vector's backward through the "exp_sum" form's fused kernels, where they serve (`exp_sum_fused`, the
        flow buffers fp32, the gradient buffers fp32 with each sample's block contiguous): its element flows
        log-added into `element_flows` (set to -inf by the caller), F+, W (with `denom`) and the gradients added into
        their buffers -- what the torch pass computes, without the `[B, M, N]` grids -- by the kernels its forward took
        (the fp16-rounded factors' where `forward_layer` noted it). `init`: the gradient buffers hold anything, written
        rather than added into (the fp16 kernels zero a chunk's slice ahead of their adds, in the L2; the others' are
        zeroed first). `own`: no other node vector of the layer reaches its children, whose element flows are then
        written (the fp16 kernels' epilogue) or filled with -inf first (the others) -- else the caller filled them.
        False: nothing done."""
        rounded = layer.__dict__.get("_lr_h", {})
        if id(ns_info.ns) not in rounded and not exp_sum_fused(self, geo, tensors, params, node_mars, element_mars):
            return False                                        # (the fp16 path's forward checked these: the same)
        if node_flows.dtype != torch.float32 or element_flows.dtype != torch.float32:
            return False
        if grads is not None and not all(g.dtype == torch.float32 and lowrank_forward.samples_contiguous(g)
                                         for g in grads):
            return False
        empty = _empty(node_flows.device)
        off = geo["grid"][0]
        _, _, Nb, cbs, bs = geo["grid"]
        lz = layer._lr_state[id(ns_info.ns)]                    # [B, M] (the fused forward's: a view of [M, B])
        pf = param_flows if param_flows is not None else empty
        pf0 = ns_info.ns._param_flow_range[0] + off if param_flows is not None else 0
        dn = denom if denom is not None else empty
        dn0 = self._denom_offsets(layer)[id(ns_info.ns)][0] + off if denom is not None else 0
        if id(ns_info.ns) in rounded:                           # (the forward's kernels: the fp16-rounded factors')
            copy = rounded[id(ns_info.ns)]                      # (the forward's fp16 copy of the factors, read once:
            rounded[id(ns_info.ns)] = []                        #  freed with this backward, another one converts)
            lowrank_forward.exp_sum_h_module().exp_sum_h_backward(
                tensors[0], tensors[1], params, geo["p0"], Nb, cbs, bs, element_mars, geo["rows"], node_flows,
                node_mars, ns_info.nid_start, lz.t().contiguous(), element_flows, pf, pf0, dn, dn0,
                grads[0] if grads is not None else empty, grads[1] if grads is not None else empty, geo["M"],
                geo["N"], self.rank, copy, init, own, lowrank_forward.EXP_SUM_H_PDL,
                *((sc[0], sc[1]) if sc is not None else (None, 0)), lowrank_forward.EXP_SUM_H_ONE)
        else:                                                   # (the bf16 / TF32 kernels: the factors and the
            if init and grads is not None:                      #  gradients where they lie, the latter zeroed first)
                torch._foreach_zero_(list(grads))
            if own:
                element_flows.index_fill_(0, geo["kids"], -float("inf"))
            exp_sum_module().exp_sum_backward(
                tensors[0], tensors[1], params, geo["p0"], Nb, cbs, bs, element_mars, geo["rows"], node_flows,
                node_mars, ns_info.nid_start, lz.t().contiguous(), element_flows, pf, pf0, dn, dn0,
                grads[0] if grads is not None else empty, grads[1] if grads is not None else empty, geo["M"],
                geo["N"], self.rank, lowrank_forward.EXP_SUM_ZSPLIT, bf16)
        return True

    def _kids_own(self, layer, device) -> bool:
        """Whether no child of the layer is reached by two of its node vectors (each node vector's element flows its
        own pass's alone). Asked once a layer."""
        ok = layer.__dict__.get("_lr_kids_own")
        if ok is None:
            rows = torch.cat([self._geometry(layer, info, device)["kids"] for info in layer.external_node_infos])
            ok = layer.__dict__["_lr_kids_own"] = bool(torch.unique(rows).numel() == rows.numel())
        return ok

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
