"""
Sampling under :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams` (:class:`_SampleMixin`, one of
its bases): every selected node's child drawn in proportion to its GATED parameters, against the structural frontier
layout or a list of selected pairs.
"""

from __future__ import annotations

import random

from .common import _buffer_kwarg

class _SampleMixin:
    """Sampling under :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams`, one of its bases."""

    def sample_layer(self, layer, ns_tensors, node_mars, element_mars, params, node_samples,
                     element_samples, rows, erows, seed_ptr, conditional: bool = False,
                     **kwargs) -> None:
        """
        The gated draw against the structural frontier layout.

        Same distribution and same launch arguments as :func:`sample_layer_pairs` -- the two kernels
        share `_gate_weights`, so they cannot disagree about what an edge weighs -- addressed by
        (frontier row, tile of samples) instead of by a list of selected pairs.
        """
        from ..kernels.blockscale_sample import bs_scoped_sum_layer

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            return None

        if getattr(layer, "ext_slots", None) is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs the compiled edge-block tables, which require a "
                "batch-innermost storage layout."
            )

        self._require_full_supply(layer, ns_tensors)

        ns0 = ns_tensors[0][0].ns
        num_samples = node_samples.size(1)
        block_size = layer.block_size
        gate_bs, gate_cbs = self.gate_sizes(ns0)
        ext_base = self._ext_base(ns_tensors, external_params, num_samples)

        nstride_cache = layer.__dict__.setdefault("_bs_sample_nstride", {})

        for partition_id in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[partition_id]

            key = (partition_id, nids.data_ptr())
            nstride = nstride_cache.get(key)
            if nstride is None:
                nstride = nstride_cache[key] = self._gate_nstride(layer, nids)

            bs_scoped_sum_layer(
                nids, layer.partitioned_cids[partition_id], layer.partitioned_pids[partition_id],
                element_mars, params, external_params,
                layer.ext_slots[0][partition_id], nstride,
                node_samples, element_samples, rows, erows, seed_ptr,
                block_size = block_size, node_cbs = ns0.ch_block_size, gate_cbs = gate_cbs,
                gate_bs = gate_bs, ext_base = ext_base, conditional = conditional,
            )

        return None

    def sample_layer_pairs(self, layer, ns_tensors, node_mars, element_mars, params, node_samples,
                     element_samples, ind_target, ind_n, ind_b, conditional: bool = False,
                     rnd = None, rnd_offset = 0, **kwargs) -> None:
        """
        Draw one child per selected node, in proportion to the GATED parameters.

        The gate reweights the gate-level mixture weights per sample, so the child of `n` is drawn
        from

        .. code-block:: text

            P(c | n, b)  propto  phi[b, g(n,c)] * theta_shared[n, c]                 (unconditional)
            P(c | n, b)  propto  phi[b, g(n,c)] * theta_shared[n, c] * exp(element_mars[c, b])

        `Z_b[n]` divides both and so cancels: the draw needs no normalizer from the forward pass, and
        in particular none of the cached `log Z` -- which is why this path shares no state with
        :func:`forward_layer` and works for an unconditional draw, where no forward pass has run at
        all.

        The weights are still normalized, just locally: `phi` is a log gate with unbounded values (a
        router's logits), so `sum_c` has to be accumulated with a running max exactly as the forward
        does, and the inverse-CDF walk then runs against that sum.

        One portable Triton kernel serves this, with no CUDA forks, no plan cache and no autotuner --
        unlike the forward, where those buy a real fraction of a training step. Sampling is bounded by
        the driver, not by this: every sum layer of the top-down pass already costs a `torch.where`, a
        device-to-host sync, a numba loop over the frontier and a host-to-device copy, which together
        run to hundreds of microseconds against a kernel measured in single digits. Forks would buy
        nothing measurable and would cost the property that makes this reviewable -- that it is the
        shared-parameter sampler plus a gate.
        """
        from ..kernels.blockscale_sample import bs_sample_sum_layer

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            return None

        if getattr(layer, "ext_slots", None) is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs the compiled edge-block tables, which require a "
                "batch-innermost storage layout."
            )

        self._require_full_supply(layer, ns_tensors)

        ns0 = ns_tensors[0][0].ns
        num_samples = node_samples.size(1)
        block_size = layer.block_size
        gate_bs, gate_cbs = self.gate_sizes(ns0)
        node_cbs = ns0.ch_block_size

        # RELATIVE to the staging buffer, so it depends on the layout and the batch and not on any
        # address -- the same quantity the forward resolves, re-derived here because sampling shares
        # no state with it.
        ext_base = self._ext_base(ns_tensors, external_params, num_samples)

        # `_gate_nstride` is cached, but building its key costs two DEVICE SYNCS (`int(nids[0])`,
        # `int(nids[-1])`) on every call including a hit. The forward pays that once per plan; the
        # sampler would pay it per partition per layer per draw, in a loop whose cost is already
        # dominated by host round-trips. Keyed on the partition and the buffer's address, which
        # answers the same question -- compiled `nids` never change in place, and a device move
        # hands back a different pointer.
        nstride_cache = layer.__dict__.setdefault("_bs_sample_nstride", {})

        for partition_id in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[partition_id]
            cids = layer.partitioned_cids[partition_id]
            pids = layer.partitioned_pids[partition_id]

            key = (partition_id, nids.data_ptr())
            nstride = nstride_cache.get(key)
            if nstride is None:
                nstride = nstride_cache[key] = self._gate_nstride(layer, nids)

            # A fresh seed per partition, as the shared-parameter launcher draws one per launch. The
            # partitions serve disjoint sets of lanes -- a node lives in exactly one -- so they never
            # have to agree.
            bs_sample_sum_layer(
                nids, cids, pids, element_mars, params, external_params,
                layer.ext_slots[0][partition_id], nstride,
                node_samples, element_samples, ind_target, ind_n, ind_b,
                0 if rnd is not None else random.randint(0, 2**31), rnd, rnd_offset,
                block_size = block_size, node_cbs = node_cbs, gate_cbs = gate_cbs,
                gate_bs = gate_bs, ext_base = ext_base, conditional = conditional,
            )

        return None
