"""
The denominator flow of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams` under
`apply_z_correction` (:class:`_DenominatorsMixin`, one of its bases): the gate-space accumulator `W` -- its layout,
the backward that fills it, the Anemone top-down term -- and the exact M-step's `F-` rebuilt from it, or read by the
shared dual-EM kernels in its gate-space form.
"""

from __future__ import annotations

import os
import weakref

import torch

# `F-` is STORED and accumulated in GATE space. `theta[n,c]` has no batch index, so it leaves the batch
# sum entirely:
#
#     F-[n,c] = sum_b f_b[n] * theta_b[n,c] = theta[n,c] * W[n, g(c)],
#     W[n,g]  = sum_b u[n,b] * phi[g,b],       u[n,b] = exp(node_flows[n,b] - log Z[n,b])
#
# so `W` -- `gate_cbs` times smaller than a per-edge `F-` -- is a SUFFICIENT accumulator, and `F-` is
# reconstructed once per EM step in `compute_em_correction`. Two things follow:
#
#   * MEMORY. A param-flow-shaped `F-` was 13.6% of peak training memory on a 1024-state gated HMM
#     (60.0 MB of 440.8, `gate_cbs = 64`) and 18.5% on a 2048-state one (240.0 MB of 1298.5,
#     `gate_cbs = 128`). In gate space those become 0.94 MB and 1.88 MB.
#   * The contraction also runs `gate_cbs` times less often than the fused per-edge form did -- MEASURED
#     as a 16x re-read of `node_flows` / `log Z` at block_size 128 / 512 edges, 32 MB of a 38 MB kernel.
#
# `g(c) = c // gate_cbs` is exact because the gate's `ch_block_size` divides the node's (asserted in
# `validate_ns`) and the compiled edge count is a whole number of child blocks (asserted in
# `denom_gate_geometry`), which is what makes the gate axis a clean quotient of the edge axis.

# The conditional M-step as two Triton kernels instead of ~38 eager ATen ops per `ns` (MEASURED
# dispatch-bound, growing with the NODE count rather than the parameter count). `=0` restores the torch
# reference, which is what the fused path is validated against -- keep it working, it is the oracle.
_EM_FUSED = os.environ.get("PYJUICE_BLOCKSCALE_EM_FUSED", "1") != "0"


# Every launcher in this file sizes its grid against the SM count, so it lives in one place. The query
# itself is cheap (~1.3 us -- PyTorch caches it), but it is asked once per `ns` per partition per call,
# and a single accessor is one fewer thing to get inconsistent.
_SM_COUNT = {}


def _sm_count(device):
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    n = _SM_COUNT.get(idx)
    if n is None:
        n = _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return n


class _DenominatorsMixin:
    """The denominator flow of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams`, a base of it."""

    # Everything the exact M-step under a live gate needs: the layout of the gate-space
    # accumulator `W`, the backward that fills it, and the conditional update that consumes it.
    # Inert unless `apply_z_correction` is set.

    @property
    def requests_denom_param_flows(self) -> bool:
        # The gate makes `Z = sum_c phi * theta` depend on `theta`, so `normalize(F+)` is not exact EM
        # under a live gate (see `__init__`). The exact M-step needs the denominator flow `F-`; request
        # it only when `apply_z_correction` is on (the default), so a model built without it pays nothing.
        return self.apply_z_correction

    def denom_gate_geometry(self, layer):
        """
        `(rows, block_size, n_gates)` of the gate-space accumulator `W`, per forward partition.

        Read off the COMPILED tables only, so it is available before the first forward pass -- which is
        what lets the PC size `denom_param_flows` at compile time. `n_gates` counts the compiled edge
        slots, padding included, exactly as the kernels index them.

        Cached on the layer: structural, and the backward asks for it on every call.
        """
        cached = getattr(layer, "_bs_denom_geom", None)
        if cached is not None:
            return cached

        gate_cbs = self.gate_sizes(layer.nodes[0])[1]
        out = []
        for pid in range(layer.num_fw_partitions):
            rows = layer.partitioned_nids[pid].size(0)
            num_edges = layer.partitioned_cids[pid].size(1)
            assert num_edges % gate_cbs == 0, \
                f"the compiled edge count ({num_edges}) of partition {pid} must be a whole number of " \
                f"gate blocks ({gate_cbs})."
            out.append((rows, layer.block_size, num_edges // gate_cbs))

        layer._bs_denom_geom = out
        return out

    def denom_flow_sizes(self, layer) -> list:
        """One `W[rows * block_size, n_gates]` per forward partition -- see the module note on why the
        gate-space contraction is a sufficient (and `gate_cbs` times smaller) accumulator for `F-`."""
        return [rows * bs * n_gates for rows, bs, n_gates in self.denom_gate_geometry(layer)]

    def _denom_slice(self, layer, denom_param_flows, pid):
        """This partition's `W`, as a `[rows * block_size, n_gates]` view into the PC's flat buffer."""
        off, size = layer.denom_flow_slices[pid]
        rows, bs, n_gates = self.denom_gate_geometry(layer)[pid]
        return denom_param_flows[off : off + size].view(rows * bs, n_gates)

    def _accumulate_denom_torch(self, layer, node_flows, params, external_params, denom_param_flows):
        """
        The denominator flow `F-` for `apply_z_correction`:

        .. code-block:: text

            denom_param_flows[pfid(n,c)] += sum_b f_b[n] * theta_b[n,c],
            theta_b[n,c] = phi_b[g(c)] * theta[n,c] / Z_b[n]

        so each summand is `exp(node_flows[n,b] + log phi_b[g] + log theta[n,c] - log Z_b[n])`, which is
        `f_b[n] * theta_b[n,c] <= f_b[n]` -- the log-space form is stabilized by its per-`(n,c)` max over
        the batch. `f_b[n]` is the node flow, `Z_b[n]` the forward's cached `log Z`; neither needs
        `element_mars`. The M-step then forms `theta * F+ / F-` (see `requests_denom_param_flows`).

        Like the kernel, it accumulates the GATE-SPACE form `W[n,g] = sum_b u[n,b] * phi[g,b]`, from
        which `compute_em_correction` rebuilds `F-[n,c] = theta[n,c] * W[n, g(c)]` -- so `theta` does not
        appear here at all.

        REFERENCE implementation (torch), which `_bs_post_backward_kernel`'s denominator is validated
        against (`PYJUICE_BLOCKSCALE_DENOM_REF=1` runs it instead). It materializes a
        `[rows, block_size, gates, batch]` tile per partition, so the kernel replaces it on the hot path.

        Only one node gate per block is served (`gate_counts(ns)[0] == 1`), which `pre_backward_layer`
        has already enforced when `apply_z_correction` is on.
        """
        state = layer._bs_bw_state
        block_size, batch = state["block_size"], state["batch_size"]
        node_cbs, gate_cbs, ext_base = state["node_cbs"], state["gate_cbs"], state["ext_base"]

        dev = params.device
        ar_m = torch.arange(block_size, device = dev, dtype = torch.long)
        ar_b = torch.arange(batch, device = dev, dtype = torch.long)

        for pid, (nids, log_z, rows) in enumerate(state["shift_args"]):
            gate = layer.ext_slots[0][pid].to(torch.long)                  # [rows, max_n_eblks], -1 padded
            E = layer.partitioned_cids[pid].size(1)

            # log f and log Z, laid out [rows, block_size, batch]
            gid = nids.to(torch.long)[:, None] + ar_m[None, :]             # global node ids
            nf = node_flows[gid.reshape(-1)].reshape(rows, block_size, batch)
            lz = log_z.reshape(rows, block_size, batch)

            # log phi of each edge, per (row, edge, batch); -inf on a padded edge block
            e = torch.arange(E, device = dev, dtype = torch.long)
            # BOUNDED by the gate table's width, exactly as `_bs_post_backward_kernel` bounds its own
            # gate-table load. `E` is the compiled `num_edges`, padded up to a power of two, while the
            # table is only `ext_max_n_eblks` columns wide -- so whenever the widest row's edge-block
            # count is not a power of two the column runs PAST the row. Unlike the kernel (whose load
            # mask makes it read the sentinel) this is a torch gather, so it raised a device-side
            # assert instead: MEASURED on `num_node_blocks = 3` (4 columns against a 3-wide table).
            # The kernel was already right here; only this reference was not.
            gcol = e // node_cbs
            in_table = gcol < gate.size(1)
            gbase = gate[:, gcol.clamp(max = gate.size(1) - 1)]            # [rows, E], per-batch base
            ghas = (gbase >= 0) & in_table[None, :]
            grow = (gbase + ext_base + ((e % node_cbs) // gate_cbs)[None, :]).clamp(min = 0)
            phi = external_params[grow[:, :, None] * batch + ar_b[None, None, :]]   # [rows, E, batch]
            phi = torch.where(ghas[:, :, None], phi, torch.full_like(phi, -float("inf")))

            # Contract the edge axis down to GATES. Every edge of a gate block carries the same `phi`,
            # so one representative edge per gate -- the first, `g * gate_cbs` -- gives the whole column.
            # `gate_cbs` divides `node_cbs`, so a gate block never straddles two gate-table columns and
            # the representative's `ghas` speaks for the gate.
            n_gates = E // gate_cbs
            phi_g = phi[:, torch.arange(n_gates, device = dev, dtype = torch.long) * gate_cbs, :]

            # log u[n,b] + log phi[g,b] per (row, m, gate, batch); stabilized over the batch axis
            expo = nf[:, :, None, :] + phi_g[:, None, :, :] - lz[:, :, None, :]
            mx = expo.amax(dim = -1, keepdim = True)
            mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
            w = (expo - mx).exp().sum(dim = -1) * mx.squeeze(-1).exp()     # [rows, block_size, n_gates]

            self._denom_slice(layer, denom_param_flows, pid).view(rows, block_size, n_gates).add_(w)

    def accumulate_denom_top_down(self, layer, node_flows, denom_param_flows, scale) -> None:
        """
        Add the Anemone top-down term to this layer's `W`.

        `F+` gets `scale * P_td[n] * theta[n,c]`. That is `theta[n,c]` times a factor with no child
        index, and `F- = theta[n,c] * W[n, g(c)]` has exactly the same shape, so the matching term in
        gate space is

            W[n, g] += scale * P_td[n]                for EVERY gate g of n,

        a rank-one update. No gate table, no `pfids`, and nothing per-edge: the reconstruction supplies
        `theta` later. (The identity needs `F+`'s term to be `theta` times a node-only factor, which is
        what `pc_is_normalized` buys -- an unnormalized PC's term carries `exp(cmars - nmars)` per child
        and is not rank one; the caller asserts it.)

        `scale` arrives as a one-element DEVICE tensor, not a float, so a captured CUDA graph replays
        with whatever step size the next call writes into it.
        """
        cache = layer.__dict__.setdefault("_bs_denom_td_gids", {})
        ar_m = None
        for pid in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[pid]
            gid = cache.get(pid)
            if gid is None or gid.device != node_flows.device:
                if ar_m is None:
                    ar_m = torch.arange(layer.block_size, device = nids.device, dtype = torch.long)
                gid = (nids.to(torch.long)[:, None] + ar_m[None, :]).reshape(-1)
                cache[pid] = gid = gid.to(node_flows.device)

            W = self._denom_slice(layer, denom_param_flows, pid)
            W += (node_flows[gid] * scale).unsqueeze(1)

    def materialize_denom_flows(self, ns, params, denom_param_flows, denom_sources, out = None):
        """
        Rebuild `F-` for `ns`'s parameter-flow range from the gate-space accumulator:
        `F-[n,c] = theta[n,c] * W[n, g(c)]`, laid out exactly like `param_flows[pfs:pfe]`.

        The mini-batch lives entirely in `W`, so this is a pure parameter-sized map with no batch axis
        -- which is why the PC can afford to keep only `W` (`gate_cbs` times smaller) between backward
        calls and pay for `F-` once per EM step, over ONE node's range at a time rather than the whole
        circuit's. Every `(layer, member_ns)` in `denom_sources` is scattered into the same buffer with
        its own `pfid` base, which sums a tie group's flow the way `compute_cum_par_flows` does for the
        numerator.

        Also the accessor for anything that wants `F-` itself -- tests and diagnostics -- since the
        stored buffer is no longer `F-`.

        :param out: where to write, zeroed first. Omitted, a scratch is reused across calls, so the
            returned tensor is only valid until the next call -- copy it if you need to keep it (the
            M-step does not: it consumes `F-` into a fresh parameter tensor). Pass `out` to opt out.
        """
        import triton
        from ..kernels.blockscale_backward import _bs_triton_denom_scatter_kernel

        pfs, pfe = ns._param_flow_range
        size = pfe - pfs
        if out is None:
            # Reused across EM steps and across nodes of the same size: at a wide model this is a
            # parameter-sized buffer, and allocating one per node per step is churn for nothing. Safe to
            # share because the caller consumes `F-` into a fresh tensor before the next node's turn.
            cache = self.__dict__.setdefault("_bs_fminus_scratch", {})
            key = (size, str(params.device))
            out = cache.get(key)
            if out is None:
                out = cache[key] = torch.empty(size, device = params.device, dtype = torch.float32)
            out.zero_()
        else:
            assert out.numel() == size
            out.zero_()

        # One scatter per DISTINCT `(layer, pfid base)`, not per member. A scatter with base `p` picks up
        # every row of the layer whose `pfid` lands in `[p, p + size)` -- which is all the tie-group
        # members that SHARE that base, since compilation reuses one param-flow range once a group
        # outgrows `max_tied_ns_per_parflow_block`. Iterating members instead would scatter those rows
        # once per member and multiply their flow.
        seen = set()
        for layer, member in denom_sources:
            m_pfs = member._param_flow_range[0]
            key = (id(layer), m_pfs)
            if key in seen:
                continue
            seen.add(key)
            block_size = layer.block_size
            geom = self.denom_gate_geometry(layer)
            gate_cbs = self.gate_sizes(layer.nodes[0])[1]
            target = 4 * _sm_count(params.device)

            for pid in range(layer.num_fw_partitions):
                rows, _, n_gates = geom[pid]
                cids = layer.partitioned_cids[pid]
                pfids = layer.partitioned_pfids[pid]
                num_edges = cids.size(1)
                W = self._denom_slice(layer, denom_param_flows, pid)

                # A pure elementwise map over the parameter array, so it partitions perfectly and the
                # only thing that matters is filling the device. Both tiles shrink until the grid is big
                # enough; 16 is the floor so the `theta` runs stay long enough to be worth a memory
                # transaction (consecutive nodes are contiguous in `mparams`, consecutive edges are not).
                S_TILE_K = max(1, min(triton.next_power_of_2(num_edges), 64))
                S_TILE_M = max(1, min(block_size, max(1, 4096 // S_TILE_K)))
                s_floor = 16 if block_size >= 16 else 1
                while S_TILE_M > s_floor and \
                        triton.cdiv(num_edges, S_TILE_K) * rows * (block_size // S_TILE_M) < target:
                    S_TILE_M //= 2
                while S_TILE_K > 16 and \
                        triton.cdiv(num_edges, S_TILE_K) * rows * (block_size // S_TILE_M) < target:
                    S_TILE_K //= 2
                s_m_tiles = max(1, block_size // S_TILE_M)

                # Padded edges (`cids == 0`) are masked in-kernel; an atomic is needed only when REAL
                # slots still collide, which parameter tying inside one layer does.
                pf_atomic = self._par_write_flags(layer, cids, pfids)[1]

                _bs_triton_denom_scatter_kernel[(triton.cdiv(num_edges, S_TILE_K), rows * s_m_tiles)](
                    W = W, mparams = params, out = out,
                    cids = cids, pids = layer.partitioned_pids[pid], pfids = pfids,
                    num_edges = num_edges, n_gates = n_gates,
                    TILE_SIZE_K = S_TILE_K, TILE_SIZE_M = S_TILE_M, BLOCK_SIZE_M = block_size,
                    GATE_CBS = gate_cbs, pf_base = m_pfs, out_size = size,
                    PF_ATOMIC = 1 if pf_atomic else 0, num_stages = 1)

        return out

    def denom_kernel_spec(self, layer, denom_param_flows, pid):
        """
        `F-[n,c] = theta[n,c] * W[n, g(c)]` with `g(c) = c // gate_cbs` -- the gate-space row-group
        form, so the shared dual-EM kernels gather `W` and never materialize `F-`.

        `W` is `gate_cbs` times smaller than a param-flow mirror, which is the whole reason it is what
        gets carried between backward calls; see the module note.
        """
        if not _EM_FUSED:
            return None                       # `PYJUICE_BLOCKSCALE_EM_FUSED=0` -> the generic path

        from ..kernels.dual_em import DENOM_ROWGROUP

        _, _, n_gates = self.denom_gate_geometry(layer)[pid]
        gate_cbs = self.gate_sizes(layer.nodes[0])[1]
        return (DENOM_ROWGROUP, self._denom_slice(layer, denom_param_flows, pid),
                n_gates, gate_cbs, 0)

    def _sigma(self, layer, params, pid, block_size, gate_cbs):
        """
        `sigma[node, flat gate] = sum_{c in that gate} theta[node, c]`, cached until `params` changes.

        Recomputing it every backward would be a full read of theta for a quantity that only moves when
        the parameters do -- i.e. once per EM step, not once per batch. `Tensor._version` bumps on the
        in-place parameter update, which is exactly the invalidation signal. One entry per partition (a single entry
        for the layer had its partitions evict each other: a full gather of theta every backward), each holding which
        tensor it was summed from (a weak reference: a `params` tensor assigned anew may start at an equal version);
        and one for a tied group's layers, which read the same rows (`_sigma_slot`).

        Summed by `blockscale_sigma.cu` where it compiles (fp32 or bf16 parameters on the GPU): one read of theta.
        The torch gather below is the fallback -- an int64 index tensor the size of the partition's parameters and
        their fp32 copy: MEASURED 316 us and ~300 MiB transient at 4 node blocks of 1024 x 4096 edges, against 12.5 us.
        """
        cache = layer.__dict__.get("_bs_sigma_cache")
        if cache is None:
            cache = layer.__dict__["_bs_sigma_cache"] = {}
        key = (params._version, block_size, gate_cbs)
        hit = cache.get(pid)
        if hit is not None and hit[0] == key and hit[1]() is params:
            return hit[2], hit[3]
        # A tied group's layers read the same parameter rows: one sum for all of them (`_sigma_slot`)
        slot = self._sigma_slot(layer, pid, block_size, gate_cbs)
        if slot is not None:
            hit = slot[1]
            if hit is not None and hit[0] == key and hit[1]() is params:
                cache[pid] = hit
                return hit[2], hit[3]

        pids = layer.partitioned_pids[pid].to(torch.int64)
        rows, num_edges = pids.shape
        n_gates = num_edges // gate_cbs
        from ..kernels.c import get_sigma_module
        mod = (get_sigma_module() if params.is_cuda and params.dtype in (torch.float32, torch.bfloat16)
               and num_edges % gate_cbs == 0 else None)
        if mod is not None:
            sigma = torch.empty(rows * block_size, n_gates, dtype = torch.float32, device = params.device)
            mod.blockscale_sigma(params, pids.contiguous(), sigma, block_size, gate_cbs)
        else:
            ar = torch.arange(block_size, device = params.device, dtype = torch.int64)
            # (summed in fp32 whatever the parameters are stored in, as the forward sums them -- MEASURED: summed in
            # bf16, the gate gradient came out 0.4-0.5% off; `.to` is the identity at fp32)
            th = params[pids[:, :, None] + ar[None, None, :]].to(torch.float32)   # [rows, edges, block_size]
            sg = th.view(rows, n_gates, gate_cbs, block_size).sum(dim = 2)  # [rows, gates, block_size]
            sigma = sg.permute(0, 2, 1).reshape(rows * block_size, n_gates).contiguous()

        cache[pid] = (key, weakref.ref(params), sigma, n_gates)
        if slot is not None:
            slot[1] = cache[pid]
        return sigma, n_gates

    def _sigma_slot(self, layer, pid, block_size, gate_cbs):
        """
        The slot `[parameter ids, sigma entry]` that the layers of a tied group share for partition `pid`, or None.

        Tied layers -- an HMM's steps -- are compiled onto the same parameter rows, so their `sigma` is the same sum;
        cached per layer it was recomputed by each: MEASURED (a tied gated HMM-4096 x4, 7 layers, B 128) 7 x 18.8 us
        a step, the kernel at its read floor each time. A group is the layers whose nodes have the same source nodes
        (compilation's tie), and its slot is shared only after a member's `partitioned_pids[pid]` is found equal to the
        first member's -- once (one device sync; not during a graph capture, where the layer keeps its own).
        """
        memo = layer.__dict__.get("_bs_sigma_slot")
        if memo is None:
            memo = layer.__dict__["_bs_sigma_slot"] = {}
        slot = memo.get(pid)
        if slot is not None:
            return slot or None
        if torch.cuda.is_current_stream_capturing():
            return None
        pids = layer.partitioned_pids[pid]
        gkey = (tuple(ns.get_source_ns() for ns in layer.nodes), pid, block_size, gate_cbs, tuple(pids.shape),
                pids.device)
        groups = self.__dict__.setdefault("_bs_sigma_groups", {})
        slot = groups.get(gkey)
        if slot is None:
            slot = groups[gkey] = [pids, None]
        elif slot[0] is not pids and not torch.equal(slot[0], pids):
            slot = False
        memo[pid] = slot
        return slot or None
