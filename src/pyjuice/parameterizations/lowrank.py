from __future__ import annotations

import torch
from typing import Optional, Tuple

from .base import SumParameterization
from . import lowrank_virtual


#: The forms of :class:`LowRank`: how the two factors make a parameter's per-sample scale
LOWRANK_FORMS = ("exp_sum", "sum_exp")
#: The most (sample, node, child) triples of one node vector the `exp_sum` passes hold at a time: a larger batch is
#: taken in chunks of samples
LOWRANK_CHUNK_CELLS = 1 << 25


class LowRank(SumParameterization):
    """
    A per-sample **multiplicative low-rank scale** on the sum node's parameters, supplied externally at call time.

    A node vector of `M` nodes over `N` children takes two tensors per sample `b`, `A_b` of `[M, k]` and `B_b` of
    `[k, N]`; they make a scale `g_b[i, j]` of every (node, child) pair, and the effective parameters are

    .. code-block:: text

        theta_b[i, j] = theta_shared[i, j] * g_b[i, j] / Z_b[i]
        Z_b[i]        = sum_j theta_shared[i, j] * g_b[i, j]

    normalized over the children of node `i`, as :class:`~.block_scale.BlockScale`' are. `form` says how
    the factors make the scale:

    .. code-block:: text

        "exp_sum":  g_b[i, j] = exp( sum_r A_b[i, r] * B_b[r, j] )          # the log-scale is the rank-k product
        "sum_exp":  g_b[i, j] = sum_r exp( A_b[i, r] ) * exp( B_b[r, j] )   # a mixture of k rank-one scales

    `"exp_sum"` gives every pair an exponent of its own, so it costs one `exp` per (sample, node, child) and the
    samples share no matmul; `"sum_exp"` is `k` matmuls with the shared parameters. Both are defined on the pairs the
    node vector's `edge_ids` connect; a pair without parameters has none here either.

    **Tensor layout.** Two `float32` tensors per node vector, in this order:

    * `A`: `[B, ns.num_nodes, k]`
    * `B`: `[B, k, ns.num_ch_nodes]`

    whatever blocks the node vector is stored in. A registered group (`TensorCircuit.register_external_params_group`)
    concatenates both along axis 1: `A` over the members' nodes, `B` over their ranks.

    **Training.** The backward writes the element and parameter flows itself, and `d LL / d A`, `d LL / d B` into
    the gradient buffers. With `apply_z_correction` it also accumulates the denominator flow's statistic, and the
    M-step is the conditional `theta <- normalize(theta * F+ / F-)`
    (:func:`~.base.SumParameterization.compute_em_correction`): `Z_b` depends on `theta`, so
    `normalize(F+)` alone is not EM here, as for BlockScale. What is kept is

    .. code-block:: text

        W[i, j] = sum_b f_b[i] * g_b[i, j] / Z_b[i]            F-[i, j] = theta[i, j] * W[i, j]

    in the parameter flows' layout (it has no smaller form here), FREE OF THE PARAMETERS as BlockScale's gate-space
    statistic is, and multiplied by them when `F-` is read. That is what an optimizer averaging the flows over steps
    needs (Anemone's momentum): the update is then `normalize(avg F+ / avg W)`, the M-step of the averaged
    statistics. Kept as `F-` itself, the average would carry the parameters of past steps against the current ones
    in `theta * F+ / F-` -- a parameter that grew is pushed further -- and the training is another: MEASURED, a
    low-rank scale built to equal a BlockScale gate, Anemone with momentum 0.9, parameters against BlockScale's
    after 16 steps 3e-2 apart that way, and the same to rounding this way and without momentum either way.

    :note: the passes here are plain torch -- dense per node vector, `exp_sum` over `[B, M, N]` in chunks of samples,
           `sum_exp` by float64 matmuls (a sample's child values spread over hundreds of nats, more than float32
           holds beside one shift a sample). Correct on every structure; not yet kernels.

    :param rank: the rank `k`
    :type rank: int

    :param form: `"exp_sum"` or `"sum_exp"` (above)
    :type form: str

    :param apply_z_correction: accumulate the denominator flow and take the conditional M-step (default); `False`
                               leaves the plain `normalize(F+)` and allocates nothing
    :type apply_z_correction: bool

    :param tie_external: share one pair of tensors across every copy of a tied node, instead of one per copy
    :type tie_external: bool
    """

    #: The node values are not the shared ones corrected: the scale differs for every pair, so they are made here
    replaces_shared_forward = True
    #: ... and so are both flows
    replaces_shared_backward = True
    computes_external_grads = True
    #: A forward may be handed the product layer below, its own forward not run (`stream_fusion_plan`): the mixture's
    #: virtual batch reads the products' values from their children, every other pass forms them first
    fuses_stream_children = True

    def __init__(self, rank: int, form: str = "exp_sum", apply_z_correction: bool = True, tie_external: bool = False):
        super(LowRank, self).__init__()

        assert isinstance(rank, int) and rank >= 1, f"`rank` must be a positive integer, got {rank}."
        assert form in LOWRANK_FORMS, f"`form` must be one of {LOWRANK_FORMS}, got {form!r}."

        self.rank = rank
        self.form = form
        self.apply_z_correction = bool(apply_z_correction)
        self.tie_external = bool(tie_external)

    # ------------------------------------------------------------------ identity

    def get_signature(self) -> str:
        return f"LowRank_{self.form}_r{self.rank}" + ("" if self.apply_z_correction else "_noz") + \
            ("_tied" if self.tie_external else "")

    def _get_constructor(self):
        return LowRank, {"rank": self.rank, "form": self.form, "apply_z_correction": self.apply_z_correction,
                         "tie_external": self.tie_external}

    def __reduce__(self):
        return (self.__class__, (self.rank, self.form, self.apply_z_correction, self.tie_external))

    def storage_owner(self, ns):
        """With `tie_external`, every copy of a tied node reads one shared pair of tensors -- the source's."""
        if self.tie_external and ns.is_tied():
            return ns.get_source_ns()

        return ns

    @property
    def requests_denom_param_flows(self) -> bool:
        return self.apply_z_correction

    # ------------------------------------------------------------------ layout

    def tensor_shapes(self, ns, batch_size: int):
        """What the caller hands over: `A [B, M, K]`, `B [B, K, N]`."""
        return ((batch_size, ns.num_nodes, self.rank), (batch_size, self.rank, ns.num_ch_nodes))

    def storage_shapes(self, ns, batch_size: int):
        """Stored with the batch innermost -- `A [M, K, B]`, `B [K, N, B]` --, as the circuit's own buffers are: the
        kernels' threads run over the samples, and read consecutive addresses from every tensor alike. The staging
        copy is the transpose (`storage_perm`), the gradients come back the same way."""
        return ((ns.num_nodes, self.rank, batch_size), (self.rank, ns.num_ch_nodes, batch_size))

    def storage_perm(self):
        return (1, 2, 0)

    def storage_offsets(self, ns):
        """Where an edge block's entries begin in each slot, in per-batch units (the layer's tables; the passes here
        address the slots by their own geometry): `A` at the node block's first node, `B` at the child block's first
        child."""
        nb, cb = ns.edge_ids[0].long(), ns.edge_ids[1].long()
        return (nb * (ns.block_size * self.rank), cb * ns.ch_block_size)

    @staticmethod
    def _as_given(tensors):
        """The staged `(A [M, K, B], B [K, N, B])` as the caller's `(A [B, M, K], B [B, K, N])`: views."""
        return tensors[0].permute(2, 0, 1), tensors[1].permute(2, 0, 1)

    def _geometry(self, layer, ns_info, device):
        """
        What ties a node vector's dense `[M, N]` grid to the circuit's buffers, built once per (layer, node vector):

        * `grid` -- `(off, Mb, Nb, cbs, bs)` where the node vector's parameters lie BLOCK-MAJOR in one run: every
          cell's edge block in its place, node block by node block, each in child order (however the edge blocks
          are listed), the run from `off` on in its source's range (`p0` in `params`) -- the grid is then a strided
          view of the run (`_dense_params`, `_scatter_add`), and the tables below are not made; else None, and
        * `flat`, `row`, `pid`, `pfid`, `local` -- for every stored parameter, its cell of the grid (`i * N + j`),
          its node `i`, its index in `params` and in `param_flows`, and its place in the node vector's own range of
          either. An edge block `e` holds `[ch_block_size, block_size]`, child then node, from `_param_ids[e]` on;
          a tied copy reads its source's parameters, and its flows lie in its own range as the parameters do in
          theirs;
        * `eid`, `has` -- every child's element id, and whether an edge reaches it at all; `cols`, `kids` -- the
          children an edge reaches (their columns of the grid, their element ids), `full` -- whether that is all;
          `rows` -- every child's element id, -1 where no edge reaches it (the kernels' form).
        """
        cache = layer.__dict__.setdefault("_lr_geometry", {})
        hit = cache.get(id(ns_info.ns))
        if hit is not None and hit["eid"].device == device:
            return hit

        ns = ns_info.ns
        src = ns.get_source_ns() if ns.is_tied() else ns
        bs, cbs = ns.block_size, ns.ch_block_size
        M, N = ns.num_nodes, ns.num_ch_nodes
        nb = ns.edge_ids[0].to(device = device, dtype = torch.long)
        cb = ns.edge_ids[1].to(device = device, dtype = torch.long)
        m = torch.arange(bs, device = device)
        c = torch.arange(cbs, device = device)

        pids0 = src._param_ids.to(device = device, dtype = torch.long)
        E, Mb, Nb = nb.numel(), M // bs, N // cbs
        grid = None
        if E == Mb * Nb and E > 0:
            # (each edge block at its cell's place from the run's first id: the ids are distinct, so then the cells
            # are, and every cell is reached)
            base = int(pids0.min())
            if bool(torch.equal(pids0, base + (nb * Nb + cb) * (cbs * bs))):
                grid = (base - src._param_range[0], Mb, Nb, cbs, bs)

        eid = torch.full((N,), -1, dtype = torch.long, device = device)
        ch_eids = ns_info.ch_eids.to(device = device, dtype = torch.long)
        eid[(cb[:, None] * cbs + c[None, :]).reshape(-1)] = (ch_eids[:, None] + c[None, :]).reshape(-1)

        has = eid >= 0
        reached = torch.nonzero(has).reshape(-1)
        geo = dict(M = M, N = N, grid = grid, p0 = None if grid is None else src._param_range[0] + grid[0],
                   eid = eid.clamp(min = 0), has = has, cols = reached, kids = eid[reached],
                   full = reached.numel() == N, rows = eid.contiguous())
        if grid is None:
            rows = (nb[:, None, None] * bs + m[None, None, :]).expand(-1, cbs, -1)   # [E, cbs, bs]
            cols = (cb[:, None, None] * cbs + c[None, :, None]).expand(-1, -1, bs)
            within = (c[:, None] * bs + m[None, :]).reshape(-1)                 # [cbs * bs]
            pid = (pids0[:, None] + within[None, :]).reshape(-1)
            local = pid - src._param_range[0]
            geo.update(flat = (rows * N + cols).reshape(-1), row = rows.reshape(-1), pid = pid, local = local,
                       pfid = ns._param_flow_range[0] + local)
        cache[id(ns)] = geo
        return geo

    @staticmethod
    def _info(layer, ns):
        for info in layer.external_node_infos:
            if info.ns is ns:
                return info
        raise KeyError(f"{ns} is not a node vector of this layer.")

    @staticmethod
    def _dense_params(geo, params) -> torch.Tensor:
        """The node vector's shared parameters on its `[M, N]` grid, float32, 0 where it has none: one strided copy of
        their run where they lie block-major (`grid`), a gather and a scatter else."""
        M, N, grid = geo["M"], geo["N"], geo["grid"]
        if grid is not None:
            _, Mb, Nb, cbs, bs = grid
            theta = torch.empty(Mb, bs, Nb, cbs, dtype = torch.float32, device = params.device)
            theta.copy_(params[geo["p0"]:geo["p0"] + M * N].view(Mb, Nb, cbs, bs).permute(0, 3, 1, 2))
            return theta.view(M, N)
        theta = torch.zeros(M * N, dtype = torch.float32, device = params.device)
        theta[geo["flat"]] = params[geo["pid"]].float()
        return theta.view(M, N)

    @staticmethod
    def _scatter_add(geo, dst: torch.Tensor, base: int, values: torch.Tensor, times: Optional[torch.Tensor] = None):
        """`dst[base + local] += (values * times)[flat]`: a grid's values (`[M, N]`, times another's) onto the slots of
        the node vector's stored parameters in a range of `dst` from `base` on (its flows', its denominators'): one
        strided add where they lie block-major (`grid`), a gather and an `index_add_` else."""
        grid = geo["grid"]
        if grid is None:
            v = values if times is None else values * times
            dst.index_add_(0, base + geo["local"], v.reshape(-1)[geo["flat"]].to(dst.dtype))
            return
        off, Mb, Nb, cbs, bs = grid
        out = dst[base + off:base + off + geo["M"] * geo["N"]].view(Mb, Nb, cbs, bs)
        perm = lambda t: t.view(Mb, bs, Nb, cbs).permute(0, 2, 3, 1)
        if times is None:
            out.add_(perm(values))
        else:
            out.addcmul_(perm(values), perm(times))

    @staticmethod
    def _child_values(geo, element_mars) -> torch.Tensor:
        """`[B, N]`: the children's log-values, -inf for a child no edge reaches."""
        lc = element_mars[geo["eid"]].t()
        if not geo["full"]:
            lc = lc.masked_fill(~geo["has"][None, :], -float("inf"))
        return lc

    def _require_full_supply(self, layer, ns_tensors) -> None:
        """The node values are made here, so a layer's node vectors are supplied all or none."""
        if len(ns_tensors) != len(layer.external_node_infos):
            raise NotImplementedError(
                f"`LowRank`: {len(layer.external_node_infos) - len(ns_tensors)} of "
                f"{len(layer.external_node_infos)} node vectors sharing this sum layer were given no external "
                f"parameters. Supply all of them (a registered group is the convenient way), or none."
            )

    def compile(self, layer) -> None:
        """What the virtual batch's twin of the layer is made from (`lowrank_virtual._twin`), noted while the layer is
        as compilation left it: which of its attributes are caches -- every empty container and every None then --
        and which are its compiled tables and settings."""
        lowrank_virtual.note_compiled(layer)

    def drop_device_caches(self, layer) -> None:
        super().drop_device_caches(layer)
        for name in ("_lr_geometry", "_lr_state", "_lr_denom_offsets", "_lrv_state", "_lrv_disjoint", "_lrv_link",
                     "_lrv_want", "_lrv_wasted", "_lrv_twin"):
            layer.__dict__.pop(name, None)

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
            ln.append(torch.logsumexp(lw, dim = 2) - lz[-1])
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
        return (ls - lz).float(), lz.float()

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
