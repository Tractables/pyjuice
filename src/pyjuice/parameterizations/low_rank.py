from __future__ import annotations

import torch
from typing import Optional

from .sum_parameterization import SumParameterization
from .lowrank import virtual as lowrank_virtual
from .lowrank.backward import _BackwardMixin
from .lowrank.denominators import _DenominatorsMixin
from .lowrank.forward import _ForwardMixin


#: The forms of :class:`LowRank`: how the two factors make a parameter's per-sample scale
LOWRANK_FORMS = ("exp_sum", "sum_exp")


class LowRank(_ForwardMixin, _BackwardMixin, _DenominatorsMixin, SumParameterization):
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
    @property
    def reads_param_operands(self) -> bool:
        """The virtual batch's plain kernels take the circuit's bf16 copy of tied fp32 parameters
        (`TensorCircuit._param_operands`), as a plain layer's do: its forward `op_params`, its backward the layer's
        per-call copy (`_bk_op_params`), handed to the layer's twin (`lowrank.virtual`) -- the mixture form's, the
        `exp_sum` form's torch passes reading `params`."""
        return self.form == "sum_exp"

    def param_operand_columns(self, batch_size: int) -> int:
        """The mixture form's plain kernels run on `rank` columns a sample (the virtual batch)."""
        return batch_size * self.rank if self.form == "sum_exp" else batch_size

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
        """What the virtual batch's twin of the layer is made from (`lowrank.virtual._twin`), noted while the layer is
        as compilation left it: which of its attributes are caches -- every empty container and every None then --
        and which are its compiled tables and settings."""
        lowrank_virtual.note_compiled(layer)

    def drop_device_caches(self, layer) -> None:
        super().drop_device_caches(layer)
        for name in ("_lr_geometry", "_lr_state", "_lr_denom_offsets", "_lrv_state", "_lrv_disjoint", "_lrv_link",
                     "_lrv_want", "_lrv_wasted", "_lrv_twin"):
            layer.__dict__.pop(name, None)
