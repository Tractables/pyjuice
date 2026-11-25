"""
The denominator flow of :class:`~pyjuice.parameterizations.low_rank.LowRank` (:class:`_DenominatorsMixin`, one of its
bases): the denominator flow `F-`: its slices of the circuit's buffer, the top-down term, `F-` as the M-step reads it.
"""

from __future__ import annotations

import torch


class _DenominatorsMixin:
    """The denominator flow of :class:`~pyjuice.parameterizations.low_rank.LowRank`, a base of it."""

    # ------------------------------------------------------------------ denominators

    def _denom_offsets(self, layer):
        """Per node vector of the layer: where its `W` (`F- / theta`, the parameter flows' layout) and its top-down
        term (one float a node) start in the circuit's denominator buffer."""
        cached = layer.__dict__.get("_lr_denom_offsets")
        if cached is None:
            base = layer.denom_flow_slices[0][0]
            cached = dict()
            for ns in layer.nodes:
                size = ns._param_flow_range[1] - ns._param_flow_range[0]
                cached[id(ns)] = (base, base + size)
                base += size + ns.num_nodes
            layer.__dict__["_lr_denom_offsets"] = cached
        return cached

    def denom_flow_sizes(self, layer) -> list:
        """All of a layer's in its first forward partition's slice: per node vector, `W = F- / theta` as the parameter
        flows lie, then one float a node for the top-down term (:func:`accumulate_denom_top_down`)."""
        total = sum(ns._param_flow_range[1] - ns._param_flow_range[0] + ns.num_nodes for ns in layer.nodes)
        return [total] + [0] * (layer.num_fw_partitions - 1)

    def accumulate_denom_top_down(self, layer, node_flows, denom_param_flows, scale) -> None:
        """The top-down term `scale * P_td[i] * theta[i, j]` of `F+` belongs in `F-` too: `scale * P_td[i]` is kept
        per node -- its part of `W`, the same for every child -- and multiplied by the parameters when `F-` is read
        (:func:`materialize_denom_flows`)."""
        offsets = self._denom_offsets(layer)
        for ns in layer.nodes:
            nid_start, nid_end = ns._output_ind_range
            off = offsets[id(ns)][1]
            denom_param_flows[off:off + ns.num_nodes] += node_flows[nid_start:nid_end].reshape(-1) * scale

    @staticmethod
    def _dense_flows(geo, buf, base: int) -> torch.Tensor:
        """`[M, N]` float64: a range of a flow buffer (from `base` on, laid as the node vector's parameters) on its grid."""
        M, N, grid = geo["M"], geo["N"], geo["grid"]
        if grid is not None:
            off, Mb, Nb, cbs, bs = grid
            return buf[base + off:base + off + M * N].view(Mb, Nb, cbs, bs).permute(0, 3, 1, 2).reshape(M, N).double()
        out = torch.zeros(M * N, dtype = torch.float64, device = buf.device)
        out[geo["flat"]] = buf[base + geo["local"]].double()
        return out.view(M, N)

    def collapsed_top_down(self, layer, node_flows, element_flows, params, param_flows, denom_param_flows, scale) -> bool:
        """See the base class. `F- = theta * W`, `W` this layer's slice: the term is `kappa[n] * theta * W` into `F+`
        and `W *= 1 + kappa[n]`, `kappa = scale * P_td / f`; a node without flow takes `scale * P_td` in its top-down
        slot and `scale * P_td * theta` in `F+`, as :func:`accumulate_denom_top_down` and the circuit's step do."""
        if not self.apply_z_correction:
            return False
        # The virtual batch's passes keep ONE `W` for a tied group (all its copies', in the first copy's slice): a
        # copy's own `F-` row, which its collapsed row is, is not kept apart there -- the circuit's own step then
        if type(layer.__dict__.get("_lr_state")) is tuple:
            return False
        offsets = self._denom_offsets(layer)
        for ns in layer.nodes:
            geo = self._geometry(layer, self._info(layer, ns), params.device)
            s, e = ns._output_ind_range
            w0, td = offsets[id(ns)]
            P = node_flows[s:e].reshape(-1).double()
            theta = self._dense_params(geo, params, dtype = torch.float64)
            W = self._dense_flows(geo, denom_param_flows, w0)
            Fm = theta * W
            f = Fm.sum(dim = 1)
            live = f > 0
            safe = torch.where(live, f, torch.ones_like(f))
            rows = torch.where(live[:, None], Fm / safe[:, None], theta)
            down = torch.mv(rows.t(), P)                                              # [N] the children's share
            if not geo["full"]:
                down = down * geo["has"]
            element_flows.index_add_(0, geo["eid"], down.to(element_flows.dtype))
            # The term's row: half the collapsed row, half the shared one (all of it where no flow came). The
            # collapsed half steps the entries the rows use by the node's step; the shared half keeps an entry the
            # rows give almost no probability from a step of `1 / probability` on one observation
            kappa = torch.where(live, 0.5 * scale * P / safe, torch.zeros_like(f))
            rest = torch.where(live, 0.5 * scale * P, scale * P)
            self._scatter_add(geo, param_flows, ns._param_flow_range[0],
                              (kappa[:, None] * W + rest[:, None]).to(param_flows.dtype), params)
            self._scatter_add(geo, denom_param_flows, w0, (kappa[:, None] * W).to(denom_param_flows.dtype))
            denom_param_flows[td:td + ns.num_nodes] += rest.to(denom_param_flows.dtype)
        return True

    def input_top_down(self, layer, node_flows, scale) -> bool:
        """See the base class: a soft-evidence leaf's phases `[F+ | F-]` a node; its collapsed distribution is its `F-`
        row normalized (the mean of the rows' own `P_b`, over their candidates): `F+ += kappa * F-`, `F- *= 1 + kappa`.
        A node without flow: its shared row in both phases, as the layer's own step."""
        if not getattr(layer.dist, "_dual_flow_backward", False):
            return False
        l0, l1 = layer._output_ind_range
        V = layer.dist.num_cats
        spf, inv = torch.unique(layer.s_pfids[:l1 - l0].long(), return_inverse = True)
        P = torch.zeros(spf.numel(), dtype = torch.float64, device = node_flows.device)
        P.index_add_(0, inv, node_flows[l0:l1].reshape(-1).double())
        first = torch.full((spf.numel(),), l1 - l0, dtype = torch.long, device = spf.device)
        first.scatter_reduce_(0, inv, torch.arange(l1 - l0, device = spf.device), reduce = "amin")
        sp = layer.s_pids[:l1 - l0].long()[first]
        pf, ar = layer.param_flows, torch.arange(V, device = spf.device)
        for a in range(0, spf.numel(), 64):                                           # (rows of V: by chunks)
            idx = spf[a:a + 64, None] + ar[None, :]
            Fm = pf[idx + V].double()
            f = Fm.sum(dim = 1)
            live = f > 0
            kappa = torch.where(live, 0.5 * scale * P[a:a + 64] / torch.where(live, f, torch.ones_like(f)), torch.zeros_like(f))
            rest = torch.where(live, 0.5 * scale * P[a:a + 64], scale * P[a:a + 64])[:, None] * \
                layer.params[sp[a:a + 64, None] + ar[None, :]].double()
            add = (kappa[:, None] * Fm + rest).to(pf.dtype)
            pf[idx] += add
            pf[idx + V] += add
        return True

    def materialize_denom_flows(self, ns, params, denom_param_flows, denom_sources, out = None):
        """`F-` of `ns`'s parameters, as `param_flows[pfs:pfe]` lies: the parameters (as they are NOW) times the sum,
        over `ns` and its tied copies, of the `W` each accumulated and of its top-down term."""
        size = ns._param_flow_range[1] - ns._param_flow_range[0]
        total = torch.zeros(size, dtype = denom_param_flows.dtype, device = denom_param_flows.device) if out is None \
            else out.zero_()
        top = torch.zeros(ns.num_nodes, dtype = denom_param_flows.dtype, device = denom_param_flows.device)
        geo = None
        for layer, member in denom_sources:
            off, td = self._denom_offsets(layer)[id(member)]
            total += denom_param_flows[off:off + size]
            top += denom_param_flows[td:td + ns.num_nodes]
            geo = self._geometry(layer, self._info(layer, member), denom_param_flows.device)
        if geo is not None:
            grid = geo["grid"]
            if grid is None:
                total.index_add_(0, geo["local"], top[geo["row"]])
            else:                                                   # (each parameter its node's top-down term)
                off, Mb, Nb, cbs, bs = grid
                total[off:off + geo["M"] * geo["N"]].view(Mb, Nb, cbs, bs).add_(top.view(Mb, 1, 1, bs))
            ps, pe = (ns.get_source_ns() if ns.is_tied() else ns)._param_range
            total.mul_(params[ps:pe].to(total.dtype))
        return total
