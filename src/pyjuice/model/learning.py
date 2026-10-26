"""
Learning in :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (:class:`_LearningMixin`, one of its bases): the
parameter flows -- initialized, accumulated over calls, all-reduced across a process group -- the EM step that
consumes them (the conditional update for the parameterizations that keep a denominator flow), and the parameters
copied back to the nodes.
"""

from __future__ import annotations

import torch
from typing import Optional

from .backend import compute_cum_par_flows, em_par_update, eval_top_down_probs
from .backend.params_master import em_par_update_master, sync_master
from .backend import em_blocks

from .common import device_grad_controller


class _LearningMixin:
    """Parameter flows and EM, for :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (one of its bases)."""

    def _denom_correction_nss(self):
        """The sum nodes whose EM update is the conditional `theta <- normalize(theta * F+ / F-)`, each
        with every `(layer, member_ns)` holding denominator flow for its parameters.

        These are the OWNERS of the parameters -- a tied copy is folded into its source, which is the
        node whose `_param_range` the M-step writes. Structural, built at compile."""
        return getattr(self, "_denom_sources_of_ns", {})

    def _par_update_kwargs_excluding(self, ranges):
        """`par_update_kwargs` with the blocks of `ranges` (parameter ranges) removed, cached.

        `compile_par_update_fn` emits its blocks PER `ns` -- each entry's `par_start_ids` is that node's
        `_param_range[0]` plus an offset, and tied nodes are skipped -- so no block straddles a range
        boundary and selecting on `par_start_ids` is exact rather than approximate.

        Used so the standard M-step does not compute an update for parameters the conditional one is
        about to overwrite. That work was previously done and discarded: on a gated HMM the corrected
        ranges are ~94% of the sum parameters, so almost the whole pass was wasted.

        Returns the unfiltered kwargs when `ranges` is empty, which is every circuit without the
        correction -- their behaviour is untouched.
        """
        if not ranges:
            return self.par_update_kwargs

        key = tuple(sorted(ranges))
        cache = self.__dict__.setdefault("_par_update_excl_cache", {})
        kwargs = cache.get(key)
        if kwargs is None:
            par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, \
                cum_pflows, metadata = self.par_update_kwargs
            keep = torch.ones(par_start_ids.size(0), dtype = torch.bool,
                              device = par_start_ids.device)
            for ps, pe in key:
                keep &= ~((par_start_ids >= ps) & (par_start_ids < pe))
            kwargs = cache[key] = [par_start_ids[keep].contiguous(),
                                   pflow_start_ids[keep].contiguous(),
                                   blk_sizes[keep].contiguous(),
                                   blk_intervals[keep].contiguous(),
                                   global_nids[keep].contiguous(),
                                   nchs[keep].contiguous(), cum_pflows, metadata]
        return kwargs

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0, keep_zero_params: bool = False,
                      step_size_rescaling: bool = False, use_cudagraph: bool = False,
                      _apply_denom_correction: bool = True):
        """
        Perform an EM parameter update step using the accumulated parameter flows.

        :param step_size: Step size - updated_params <- (1-step_size) * params + step_size * new_params
        :type step_size: float

        :param pseudocount: a pseudo count added to the parameter flows
        :type pseudocount: float

        :param keep_zero_params: if set to `True`, do not add pseudocounts to zero parameters
        :type keep_zero_params: bool

        :param step_size_rescaling: whether to rescale the step size by flows
        :type step_size_rescaling: bool

        :param _apply_denom_correction: internal. `False` runs the PLAIN M-step even on a circuit whose
            parameterization requests the denominator flow (``apply_z_correction``). For callers that use
            this method as a RENORMALIZER rather than as an EM step -- the gradient optimizers project
            back onto the normalized manifold with ``param_flows`` holding the partition flow and the
            denominator deliberately zeroed -- the conditional update is not merely inappropriate, it
            divides by `pseudocount * theta` and returns inf at ``pseudocount = 0``. Not for user code;
            an EM caller always wants the default.
        :type _apply_denom_correction: bool
        """
        assert not step_size_rescaling or self._cum_flow > 0.0, "Please perform a backward pass before calling `mini_batch_em`."
        assert 0.0 < step_size <= 1.0, "`step_size` should be between 0 and 1."

        # `apply_z_correction` + `step_size_rescaling` (Anemone) now agree: `init_param_flows` scales
        # BOTH flows by `step_size / cum_flow`, and `eval_top_down_probs` adds the same
        # `(1 - step_size) * P_td[n] * theta[n,c]` term to the denominator as it does to `param_flows`
        # (see the note there for why the identical term is the right one) -- through
        # `accumulate_denom_top_down`, since the denominator has its own layout. The conditional M-step
        # below therefore sees a numerator and denominator built the same way.
        with device_grad_controller(device = self.device, no_grad = True):

            # Apply step size rescaling according to the mini-batch EM objective derivation
            if step_size_rescaling:
                self.init_param_flows(flows_memory = step_size / self._cum_flow)

                # Below fp32, the `(1 - step_size) * P_td * theta` term from the fp32 master copy: built from the stored
                # rounding, the step-size-1 update below would reset the copy to it every step, and an update smaller
                # than half a bf16 ulp would never land (MEASURED: near convergence every parameter fp32 moved by > 2
                # ulps in 40 steps stayed where it was)
                td_params = self.params if self.params.dtype == torch.float32 else self._params_master(sync = True)
                eval_top_down_probs(self, update_pflow = True, scale = (1.0 - step_size), use_cudagraph = use_cudagraph,
                                    params = td_params)

                self._cum_flow = 0.0 # Zero out the cumulative flow value
                step_size = 1.0 # We have applied the step size within the parameter flows

            # Update input layers
            for layer in self.input_layer_group:
                layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount, keep_zero_params = keep_zero_params)

            # Accumulate parameter flows of tied nodes
            compute_cum_par_flows(self.param_flows, self.parflow_fusing_kwargs)

            # Conditional dual-flow M-step (`apply_z_correction`): each requesting `ns` computes its own
            # parameters from the pre-update `theta`, F+ and F-, and the standard M-step below runs only
            # on the ranges it still OWNS. All gated on `denom_param_flows` -- an ordinary PC has none,
            # `corrections` is empty, and nothing here changes its behaviour.
            #
            # No `compute_cum_par_flows` on the denominator: it is not laid out in `pfid` space, so the
            # numerator's tie fusing does not apply to it. The tie group is passed to the descriptor
            # instead (`denom_sources`), which sums the copies' flow as it reconstructs `F-`.
            denom_param_flows = self.denom_param_flows if _apply_denom_correction else None
            corrections = ([] if denom_param_flows is None
                           else list(self._denom_correction_nss().items()))

            # Normalize and update parameters. The skipped ranges' results were previously computed here
            # and then overwritten -- ~94% of the sum parameters on a gated HMM.
            exclude = [ns._param_range for ns, _ in corrections]
            # Below fp32: the update on the fp32 master copy, stored in the same pass
            params = self.params if self.params.dtype == torch.float32 else self._params_master(sync = False)
            master = None if params is self.params else params
            if em_blocks.ENABLED:
                # By node blocks, coalesced (`backend.em_blocks`: 6.6x the per-block kernels on HCLT-1024 at fp32)
                em_blocks.em_update(self, self.param_flows, step_size, pseudocount, keep_zero_params,
                                    exclude = exclude, master = master)
            else:
                kwargs = self._par_update_kwargs_excluding(exclude)
                if kwargs[0].size(0) > 0:        # every block may belong to a corrected range
                    if master is None:
                        em_par_update(params, self.param_flows, kwargs,
                                      step_size = step_size, pseudocount = pseudocount,
                                      keep_zero_params = keep_zero_params)
                    else:
                        em_par_update_master(params, self.params.data, self.param_flows, kwargs,
                                             step_size = step_size, pseudocount = pseudocount,
                                             keep_zero_params = keep_zero_params)

            # The conditional update, writing `params` in place -- which is what removes BOTH the
            # discarded pass above and a separate write-back. It reads the PRE-update `theta` of its own
            # range, which the call above no longer touches, so the ordering is safe either way round.
            for ns, sources in corrections:
                ps, pe = ns._param_range
                if params is not self.params:    # the range's master values in line with `params`, before reading them
                    sync_master(params[ps:pe], self.params.data[ps:pe])
                ns.external_params.compute_em_correction(
                    ns, params, self.param_flows, denom_param_flows,
                    step_size, pseudocount, keep_zero_params, denom_sources = sources,
                    out = params[ps:pe])
                if params is not self.params:
                    self.params.data[ps:pe].copy_(params[ps:pe])

            # The updates write `params` through Triton kernels, which do not move the tensor's version: signal it as an
            # in-place update would, for the caches keyed on it (BlockScale's per-gate parameter sums -- MEASURED before:
            # the gate gradient after an EM step used the sums from before it, 1.3e-2 off gradients of ~2.5e-2)
            torch.autograd.graph.increment_version(self.params)

    # ---------------------------------------------------------------------------------------------
    # The fp32 master copy of parameters stored below fp32 (`param_dtype`; see `backend.params_master`)
    # ---------------------------------------------------------------------------------------------

    def _params_for_update(self) -> torch.Tensor:
        """The tensor a parameter update writes: `params` themselves when stored in fp32; otherwise their fp32 master
        copy -- made by the first update (an exact upcast), and brought in line with `params` wherever those were written
        since it was last stored. After writing it, call `_store_updated_params` -- and nothing that calls this in
        between, which would take the not-yet-stored values for outside writes and undo them."""
        if self.params.dtype == torch.float32:
            return self.params
        return self._params_master(sync = True)

    def _params_master(self, sync: bool) -> torch.Tensor:
        """The fp32 master copy of `params` (stored below fp32), made now if there is none (an exact upcast); with
        ``sync``, brought in line with `params` where those were written since it was last stored (a pass over all of
        them: an update that reads and writes every parameter does it itself instead, `em_par_update_master`)."""
        master = getattr(self, "params_master", None)
        if master is None or master.shape != self.params.shape or master.device != self.params.device:
            master = self.params.detach().to(torch.float32)
            if "params_master" in self._buffers:
                self.params_master = master
            else:                                # a circuit made before the buffer existed
                self.register_buffer("params_master", master, persistent = False)
        elif sync:
            sync_master(master, self.params.data)
        return master

    def _store_updated_params(self) -> None:
        """Store the master copy's values into `params` (rounded to their dtype), after an update wrote it."""
        master = getattr(self, "params_master", None)
        if master is not None and self.params.dtype != torch.float32:
            self.params.data.copy_(master)
            # What the in-place update of `params` signals at fp32 (the gradient optimizers' `clamp_`), which a write
            # through `.data` does not: caches keyed on the version (BlockScale's parameter sums) must see it
            torch.autograd.graph.increment_version(self.params)

    def _params_in_fp32(self) -> torch.Tensor:
        """The parameters in fp32: `params`, or below fp32 the master copy (in line with `params`) where there is one
        and an upcast of `params` otherwise. Not a copy at fp32."""
        if self.params.dtype == torch.float32:
            return self.params.data
        master = getattr(self, "params_master", None)
        if master is not None and master.shape == self.params.shape and master.device == self.params.device:
            sync_master(master, self.params.data)
            return master
        return self.params.detach().to(torch.float32)

    def drop_params_master(self) -> None:
        """Free the fp32 master copy of parameters stored below fp32 (made again by the next update, from `params`)."""
        if "params_master" in self._buffers:
            self.params_master = None

    def cumulate_flows(self, inputs: torch.Tensor, params: Optional[torch.Tensor] = None):
        with torch.no_grad():
            self.forward(inputs, params)
            self.backward(inputs = inputs, compute_param_flows = True, flows_memory = 1.0)

    def init_param_flows(self, flows_memory: float = 1.0, batch_size: Optional[int] = None):
        """
        Initialize parameter flows.

        :param flows_memory: the number that the current parameter flows (if any) will be multiplied by; equivalent to zeroling the flows if set to 0
        :type flows_memory: float
        """

        assert 0.0 <= flows_memory <= 1.0, f"`flows_memory` should be in [0.0, 1.0]"

        if batch_size is None:
            pflow_shape = (self.num_param_flows,)
        else:
            pflow_shape = (self.num_param_flows, batch_size)
            
        self._init_buffer(name = "param_flows", shape = pflow_shape)

        # At 0 a write pass (`zero_`), not a read-scale-write one: the same zeros at half the traffic (MEASURED, HCLT-1024's
        # sum and input-layer flows: 2.4 ms each scaled), and a NaN or inf left in the buffer is cleared rather than kept
        if flows_memory == 0.0:
            self.param_flows.zero_()
        elif flows_memory != 1.0:
            self.param_flows[:] *= flows_memory

        # The denominator flow buffer rides the identical reset/scale cadence, so `zero_param_flows`
        # (which routes through here) zeros it too, and Anemone's `flows_memory = step_size / cum_flow`
        # scales numerator and denominator by the same factor. Allocated only on request -- an ordinary
        # PC keeps it `None` and pays nothing.
        #
        # Its size is `num_denom_flows`, fixed at compile by the requesting parameterizations, NOT
        # `pflow_shape`: the denominator is not a `param_flows` mirror. `BlockScaleSumParams` keeps the
        # gate-space contraction `W[node, gate]` instead, which is `gate_cbs` times smaller (MEASURED
        # 64x at a 1024-state gated HMM, 128x at 2048) and reconstructs `F-` at M-step time. Scaling
        # still commutes with the reconstruction because `F- = theta * W` is linear in `W`.
        if self._requests_denom_param_flows:
            self._init_buffer(name = "denom_param_flows", shape = (self.num_denom_flows,))
            if flows_memory == 0.0:
                self.denom_param_flows.zero_()
            elif flows_memory != 1.0:
                self.denom_param_flows[:] *= flows_memory

        # For input layers
        for layer in self.input_layer_group:
            layer.init_param_flows(flows_memory = flows_memory)

        return None

    def sync_param_flows(self, dtype: Optional[torch.dtype] = None, op = None, group = None,
                         sync_cum_flow: bool = True):
        """
        All-reduce the parameter flows across a ``torch.distributed`` process group, for DDP EM
        training. Sums ``self.param_flows`` (sum layers) and every input layer's ``param_flows`` so
        each rank ends up with the total flow over the global batch. Call after ``backward(...)`` and
        before the EM update (``mini_batch_em`` / ``full_batch_em``); it replaces the hand-rolled
        ``dist.all_reduce(pc.param_flows); for l in pc.input_layer_group: dist.all_reduce(l.param_flows)``.

        :param dtype: if given (e.g. ``torch.bfloat16``), reduce in this lower precision to halve the
            communication on bandwidth-bound interconnects (validated ~2x faster; the param-flow
            rounding is benign, ~1e-4 ΔLL). Each buffer is cast, reduced, and copied back in place;
            stored ``param_flows`` stay float32.
        :param op:    reduction op (default ``ReduceOp.SUM`` — flows are additive across data shards).
        :param group: process group (default: the default group).
        :param sync_cum_flow: also reduce the EM normalizer ``_cum_flow`` with the same op (default
            True), so the post-sync EM update is correct under load imbalance without a manual
            ``_cum_flow *= world_size``. It's a 1-element collective and its readback is the sync the
            EM update would force anyway -> no measurable overhead vs the param-flow reduce.

        No-op if ``torch.distributed`` is unavailable / uninitialized / ``world_size == 1``.
        """
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return None
        if op is None:
            op = dist.ReduceOp.SUM

        tensors = []
        if getattr(self, "param_flows", None) is not None:
            tensors.append(self.param_flows)
        # The denominator flow is additive across data shards exactly like the numerator, so it must be
        # reduced too -- otherwise a DDP M-step would divide by a partial `F-`. It stays a single
        # collective even though its layout is the parameterization's own: the PC keeps every requesting
        # layer's slices in ONE flat tensor, and a gate-space accumulator sums over samples like any
        # other flow. (`BlockScaleSumParams` also makes this the CHEAPER reduce of the two, since `W` is
        # `gate_cbs` times smaller than the param-flow mirror it replaced.)
        if getattr(self, "denom_param_flows", None) is not None:
            tensors.append(self.denom_param_flows)
        for layer in self.input_layer_group:
            pf = getattr(layer, "param_flows", None)
            if pf is not None:
                tensors.append(pf)

        for pf in tensors:
            if dtype is not None and pf.dtype != dtype:
                buf = pf.to(dtype)
                dist.all_reduce(buf, op = op, group = group)
                pf.copy_(buf)
            else:
                dist.all_reduce(pf, op = op, group = group)

        # Reduce the scalar EM normalizer in the SAME op. Must be called by ALL ranks unconditionally
        # (collectives can't be value-gated, or they deadlock); _cum_flow always exists (init 0.0).
        if sync_cum_flow:
            ct = torch.tensor([float(self._cum_flow)], dtype = torch.float64, device = self.device)
            dist.all_reduce(ct, op = op, group = group)
            self._cum_flow = ct.item()

        return None

    def zero_param_flows(self):
        """
        Zero out parameter flows.
        """
        self.init_param_flows(flows_memory = 0.0)

    def update_parameters(self, clone: bool = True):
        """
        Copy parameters from this `TensorCircuit` to the original `CircuitNodes`.

        :param clone: whether to deepcopy parameters
        :type clone: bool
        """
        params = self._params_in_fp32().detach().cpu()      # the nodes keep fp32 parameters, whatever the circuit stores

        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied():
                ns.update_parameters(params, clone = clone)

        for layer in self.input_layer_group:
            layer.update_parameters()

        return None

    def update_param_flows(self, clone: bool = True, origin_ns_only: bool = True):
        """
        Copy parameter flows from this `TensorCircuit` to the original `CircuitNodes`.

        :param clone: whether to deepcopy parameters
        :type clone: bool
        """
        param_flows = self.param_flows.detach().cpu()

        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied():
                ns.update_param_flows(param_flows, clone = clone, origin_ns_only = origin_ns_only)
