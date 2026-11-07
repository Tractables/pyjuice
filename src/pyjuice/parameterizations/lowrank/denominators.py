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
