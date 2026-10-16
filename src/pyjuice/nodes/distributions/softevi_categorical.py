from __future__ import annotations

import torch
import triton.language as tl
import os
from typing import Optional, Any

from .distributions import Distribution

from .softevi.backward import _condition_apply_bk_params_kernel, _condition_apply_bk_softevi_kernel, \
                              _condition_bk_dense_denom, _condition_bk_dense_prologue, _condition_grad_cast, \
                              _condition_sample_kernel, _DenseDenomDispatch, _GradCastEpilogue, \
                              _prep_args_apply_bk_params_kernel, _prep_args_apply_bk_softevi_kernel, \
                              _prep_args_bk_dense_denom, _prep_args_bk_dense_prologue, _prep_args_grad_cast, \
                              _prep_args_sample_kernel
from .softevi.candidates import _sorted_soft_evidence, sort_soft_evidence_candidates
from .softevi.common import tlmath
from .softevi.dense_index import _fw_use_dense
from .softevi.forward import _condition_apply_fw_gemm, _condition_apply_fw_kernel, \
                             _condition_apply_fw_w_value_mask_kernel, _condition_fw_cuda_kernel, _fw_cuda_structural, \
                             _FwCudaDispatch, _FwDenseGemmDispatch, _prep_args_apply_fw_gemm, \
                             _prep_args_apply_fw_kernel, _prep_args_apply_fw_w_value_mask_kernel, \
                             _prep_args_fw_cuda_kernel
from .softevi.kernels import _KernelsMixin


class SoftEvidenceCategorical(_KernelsMixin, Distribution):
    """
    A class representing a Categorical distribution that allows external soft evidence.

    :note: with top-k soft evidence (i.e. when `soft_evidence_cat_ids` is supplied) the forward is bound
           by the `params` gather addressed through `cat_ids`, so the candidate ORDER dominates its cost:
           4.3x on the CoDD/latent training config, 1.67x on the CoDD decode config, for a ~0.05 ms sort.
           The forward now sorts by category id itself (`sort_soft_evidence`, on by default) -- see
           :func:`sort_soft_evidence_candidates` for why this is semantically a no-op.

           The BACKWARD is deliberately left in the caller's order, because
           `categorical_evidence_logp_grad` is indexed by candidate slot and belongs to the caller. A
           training loop that wants the backward sorted too should sort once at top-k time with
           :func:`sort_soft_evidence_candidates` and pass the sorted arrays through both passes; that
           also lets the two passes share one `_build_dense_index` build.

    :param num_cats: number of categories
    :type num_cats: int
    :param sort_soft_evidence: sort the candidate axis by category id inside the forward. On by default;
                               override per call with `sort_soft_evidence = False` in the forward's
                               kwargs, or process-wide with `PYJUICE_SOFTEVI_SORT=0`.
    :type sort_soft_evidence: bool
    """

    call_kwargs = ("categorical_evidence_logp", "soft_evidence_cat_ids", "soft_evidence_value_mask",
                   "categorical_evidence_logp_grad")   # see `Distribution.call_kwargs`

    def __init__(self, num_cats: int, _dual_flow_backward: bool = True, sort_soft_evidence: bool = True):
        super(SoftEvidenceCategorical, self).__init__()

        self.num_cats = num_cats
        # ON BY DEFAULT (1.67x on the CoDD decode forward, ~1.2x on training at K>=256). A
        # differential audit against the pre-sort baseline confirmed the sort is ALGEBRAICALLY exact
        # whenever the candidate ids in a row are UNIQUE -- this argument's documented precondition,
        # which `torch.topk` satisfies. It is not bit-identical at scale: permuting the candidate axis
        # permutes the log-sum-exp accumulation order, worth ~1e-5 on the input layer and up to 6.1e-5
        # on the root LL at K = 1024 (small circuits do come out bit-identical, so do not calibrate
        # this on a toy).
        # With a DUPLICATED id the sorted and unsorted answers differ (measured 2.44 nats), because
        # the kernels' slot search sums over matches; neither answer is right, the duplicate has
        # already destroyed the information. That path is now at least memory-safe and deterministic
        # -- see the clamp on `match_ids` in `fw_kernel` -- where before it read out of bounds.
        # See `preprocess_fw_kwargs` and `sort_soft_evidence_candidates`.
        # Not part of `get_metadata`, so it is not persisted.
        self.sort_soft_evidence = sort_soft_evidence

        self.post_fw_fns = [
            # Dense (full-vocab) forward as ONE GEMM. First in the list and mutually exclusive with
            # both Triton dense paths, which negate `_fw_gemm_applicable`. See the block comment above
            # that function for why the kernel's tiling cannot get this reuse.
            (_FwDenseGemmDispatch(), _condition_apply_fw_gemm, _prep_args_apply_fw_gemm),
            # CUDA top-k forward (index-driven, or the gather form); mutually exclusive with the Triton
            # kernel below, which remains the fallback for value-mask / no-cat-ids / partial-eval cases.
            (_FwCudaDispatch(), _condition_fw_cuda_kernel, _prep_args_fw_cuda_kernel),
            (self.fw_kernel, _condition_apply_fw_kernel, _prep_args_apply_fw_kernel),
            # Opt-in generation forward; the two conditions are mutually exclusive on
            # `soft_evidence_value_mask`, so the default path is untouched.
            (self.fw_w_value_mask_kernel, _condition_apply_fw_w_value_mask_kernel, _prep_args_apply_fw_w_value_mask_kernel)
        ]

        # Kept as an attribute: `_tune_dense_choice` launches it directly when timing dense vs scattered.
        self._dense_denom_dispatch = _DenseDenomDispatch(self.bk_dense_denom_kernel)

        self.post_bp_fns = [
            (self.bk_params_kernel, _condition_apply_bk_params_kernel, _prep_args_apply_bk_params_kernel),
            (self.bk_softevi_kernel, _condition_apply_bk_softevi_kernel, _prep_args_apply_bk_softevi_kernel),
            # Dense top-k denominator path (dedicated kernels; mutually exclusive with the scattered
            # `bk_softevi_kernel` above, which stays the fallback for every other case). Order matters:
            # the prologue writes the ratio scratch that the denominator kernel reads.
            (self.bk_dense_prologue_kernel, _condition_bk_dense_prologue, _prep_args_bk_dense_prologue),
            (self._dense_denom_dispatch, _condition_bk_dense_denom, _prep_args_bk_dense_denom),
            # Last: only fires when the caller's evidence gradient is not float32 (see `_grad_scratch`).
            (_GradCastEpilogue(), _condition_grad_cast, _prep_args_grad_cast)
        ]

        self.sampling_fns = [
            (self.sample_kernel, _condition_sample_kernel, _prep_args_sample_kernel)
        ]

        self._dual_flow_backward = _dual_flow_backward

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "ExternSoftEvidenceCategorical"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [self.num_cats]

    def normalize_parameters(self, params: torch.Tensor):
        params = params.reshape(-1, self.num_cats)
        params /= params.sum(dim = 1, keepdim = True)

        return params.reshape(-1)

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return self.num_cats

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        return self.num_cats * 2 if self._dual_flow_backward else self.num_cats

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == num_nodes * self.num_parameters()
            return params

        params = torch.exp(torch.rand([num_nodes, self.num_cats]) * -perturbation)
        params /= params.sum(dim = 1, keepdim = True)

        return params.reshape(-1)

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.long

    def get_em_fn(self):
        if self._dual_flow_backward:
            if self.num_cats <= 256:
                return self.small_ncats_dual_em_fn
            else:
                self.em_block_size = 8
                return self.large_ncats_dual_em_fn
        else:
            if self.num_cats <= 256:
                return self.small_ncats_em_fn
            else:
                self.em_block_size = 8
                return self.large_ncats_em_fn

    def get_flow_mask_fn(self):
        if self._dual_flow_backward:
            return self.bk_dual_flow_mask_fn
        else:
            return self.bk_flow_mask_fn

    def set_custom_kernel_kwargs(self, kwargs):
        kwargs["dual_flow_backward"] = self._dual_flow_backward

    def build_fast_inference_params(self, layer, kwargs = None):
        """
        The emission table transposed to `[num_cats, num_rows]`, for the forward's candidate gather.

        The kernels address an emission entry as `params[s_pids[n] + cat]`, so with the natural
        `[rows, cats]` layout two consecutive NODES -- the innermost axis of the forward's tile --
        are `num_cats` floats apart (505 KB on the CoDD circuit). Every useful float then costs its
        own 32-byte sector, and MEASURED, the kernel sits at ~50 GB/s of useful bandwidth no matter
        how its tiles are shaped. Transposed, those same two nodes are 1 float apart.

        Returns `None` unless the layout assumption actually holds, rather than assuming it: every
        node's row must start at a multiple of `num_cats`, and the rows must tile the table exactly.
        Both are true for the circuits this is for, and a layer where they are not simply keeps the
        ordinary path.
        """
        # Only the TOP-K gather reads the transposed table (see `_fw_transposed_args`), so a
        # full-vocabulary forward would allocate a duplicate of the emission table and never touch it.
        if kwargs is not None and kwargs.get("soft_evidence_cat_ids", None) is None:
            return None

        params = layer.params
        num_cats = self.num_cats

        # EVERY node in the layer must use this `num_cats`. `get_signature` does not include it, so a
        # layer happily holds `SoftEvidenceCategorical(4)` and `SoftEvidenceCategorical(8)` together --
        # and then a table built from one of them is the wrong shape for the other's rows, which reads
        # past the end of the table rather than failing. The ordinary path handles that layout, so the
        # only correct thing here is to decline and let it run.
        for ns in getattr(layer, "nodes", ()):
            if getattr(ns.dist, "num_cats", num_cats) != num_cats:
                return None

        if params.numel() % num_cats != 0:
            return None
        num_rows = params.numel() // num_cats

        s_pids = layer.s_pids
        if s_pids.numel() == 0 or bool((s_pids % num_cats != 0).any()):
            return None
        rows = s_pids // num_cats
        if int(rows.max()) >= num_rows:
            return None

        return params.detach().view(num_rows, num_cats).t().contiguous()

    def preprocess_fw_kwargs(self, layer, kwargs):
        """
        Sort top-k soft evidence by category id, which the forward is bound by -- see
        :func:`sort_soft_evidence_candidates` for why, and for the guarantee that it is a no-op.

        FORWARD ONLY, and deliberately so. `node_mars` is a log-sum over the candidate axis, so the
        forward cannot observe the order; the backward's `categorical_evidence_logp_grad` is indexed BY
        candidate slot and belongs to the caller, so reordering it there would hand back a gradient
        permuted out of the caller's own layout. A caller that wants both passes sorted should sort at
        top-k time and pass the sorted arrays throughout, as the class docstring says.

        Disable per call with `sort_soft_evidence = False`, per circuit with the constructor argument,
        or process-wide with `PYJUICE_SOFTEVI_SORT=0`.
        """
        if kwargs.get("soft_evidence_cat_ids", None) is None:
            return
        if not kwargs.get("sort_soft_evidence", getattr(self, "sort_soft_evidence", True)):
            return
        if os.environ.get("PYJUICE_SOFTEVI_SORT", "1") == "0":
            return
        # The dense forward reads the candidates through the inverted index, which is order-independent,
        # so sorting buys it nothing -- and costs it the index: the index is cached by tensor IDENTITY,
        # so a forward on the sorted copies and a backward on the caller's arrays each built their own.
        # MEASURED on the CoDD PC at batch 12: a second build per step, 1.9 ms of host time out of 8.9.
        # Leaving the caller's arrays in place lets the backward find the forward's build.
        if _fw_cuda_structural(layer, kwargs) and _fw_use_dense(layer, kwargs):
            return

        logp, cat_ids = _sorted_soft_evidence(layer, kwargs)
        kwargs["categorical_evidence_logp"] = logp
        kwargs["soft_evidence_cat_ids"] = cat_ids

    @staticmethod
    def bk_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                        s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        if TILE_SIZE_K > 1:
            num_iters = tlmath.ceil(max_num_cats / TILE_SIZE_K).to(tl.int64)

            cat_ids = tl.arange(0, TILE_SIZE_K)

            for i in range(num_iters):
                cat_mask = mask[:,None] & missing_mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

                p_offsets = s_pids[:,None] + cat_ids[None,:]
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                pf_offsets = s_pfids[:,None] + cat_ids[None,:]
                if are_source_nodes:
                    # Single writer per (row, category) -- see `bk_dual_flow_mask_fn`
                    pf_ptr = param_flows_ptr + pf_offsets
                    tl.store(pf_ptr, tl.load(pf_ptr, mask = cat_mask, other = 0.0) + flows[:,None] * param, mask = cat_mask)
                else:
                    tl.atomic_add(param_flows_ptr + pf_offsets, flows[:,None] * param, mask = cat_mask)

                cat_ids += TILE_SIZE_K
        else:
            for cat_id in range(max_num_cats):
                cat_mask = mask & missing_mask & (cat_id < num_cats)

                p_offsets = s_pids + cat_id
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                pf_offsets = s_pfids + cat_id
                tl.atomic_add(param_flows_ptr + pf_offsets, flows * param, mask = cat_mask)

    @staticmethod
    def bk_dual_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr,
                             s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        if TILE_SIZE_K > 1:
            num_iters = tlmath.ceil(max_num_cats / TILE_SIZE_K).to(tl.int64)

            cat_ids = tl.arange(0, TILE_SIZE_K)

            for i in range(num_iters):
                cat_mask = mask[:,None] & missing_mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

                p_offsets = s_pids[:,None] + cat_ids[None,:]
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                # Anchor = (scale·)TD·β, already carrying `scale=(1-step_size)` via `flows`.
                # Add the SAME anchor to both phases so ratio -> 1 at s=0 (no update).
                anchor = flows[:,None] * param

                pf_offsets = s_pfids[:,None] + cat_ids[None,:]
                if are_source_nodes:
                    # One writer per (row, category): the caller walks SOURCE nodes only, having already
                    # folded the tied nodes' flow onto them, so no two programs touch the same slot.
                    # Triton does not lower `atomic_add` to a bare reduction, so it costs a round trip to
                    # the SM anyway -- a plain read-modify-write is the same traffic without the atomic.
                    fp_ptr = param_flows_ptr + pf_offsets
                    fm_ptr = param_flows_ptr + pf_offsets + num_cats[:,None]
                    tl.store(fp_ptr, tl.load(fp_ptr, mask = cat_mask, other = 0.0) + anchor, mask = cat_mask)  # F⁺
                    tl.store(fm_ptr, tl.load(fm_ptr, mask = cat_mask, other = 0.0) + anchor, mask = cat_mask)  # F⁻
                else:
                    tl.atomic_add(param_flows_ptr + pf_offsets,                    anchor, mask = cat_mask)  # F⁺
                    tl.atomic_add(param_flows_ptr + pf_offsets + num_cats[:,None], anchor, mask = cat_mask)  # F⁻

                cat_ids += TILE_SIZE_K
        else:
            for cat_id in range(max_num_cats):
                cat_mask = mask & missing_mask & (cat_id < num_cats)

                p_offsets = s_pids + cat_id
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                anchor = flows * param

                pf_offsets = s_pfids + cat_id
                tl.atomic_add(param_flows_ptr + pf_offsets,            anchor, mask = cat_mask)  # F⁺
                tl.atomic_add(param_flows_ptr + pf_offsets + num_cats, anchor, mask = cat_mask)  # F⁻

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        pass

    @staticmethod
    def small_ncats_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                          step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        numerate_pseudocount = pseudocount / num_cats
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            if keep_zero_params:
                param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
                cum_flow += tl.where(param < 1e-12, 0.0, flow + numerate_pseudocount)
            else:
                cum_flow += flow

        # Parameter update
        cum_flow += pseudocount
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            if keep_zero_params:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / (cum_flow - pseudocount)
                new_param = tl.where(param < 1e-12, 0.0, new_param)
            else:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow

            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    @staticmethod
    def large_ncats_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        numerate_pseudocount = pseudocount / num_cats
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            flow = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            if keep_zero_params:
                param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
                cum_flow += tl.sum(tl.where(param < 1e-12, 0.0, flow + numerate_pseudocount[:,None]))
            else:
                cum_flow += tl.sum(flow, axis = 1)

            cat_ids += 128

        # Parameter update
        cum_flow += pseudocount
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            if keep_zero_params:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount[:,None]) / (cum_flow[:,None] - pseudocount)
                new_param = tl.where(param < 1e-12, 0.0, new_param)
            else:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount[:,None]) / cum_flow[:,None]
            tl.store(params_ptr + s_pids[:,None] + cat_ids[None,:], new_param, mask = cat_mask)

            cat_ids += 128

    @staticmethod
    def small_ncats_dual_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                               step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        numerate_pseudocount = pseudocount / num_cats
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids + num_cats + cat_id, mask = cat_mask, other = 0)

            # MAP M-step (Dirichlet prior): denominator is F- + pseudocount*beta, NOT F- + pseudocount.
            # This is the multiplicative form of beta = (F+ + pc/K)/(lambda + G-); the pc*beta term
            # floors never-observed categories (F+=0 => beta settles at (pc/K)/(G-+pc), independent
            # of beta) so they can't underflow, and reduces exactly to (F+ + pc/K)/(sum F+ + pc) when
            # p_theta is uniform (F- = beta*Gamma).
            flow = param * (flow_num + numerate_pseudocount) / (flow_denom + pseudocount * param)
            # Padding lanes (cat_id >= num_cats) load param/flows as 0 => 0/0 = NaN; zero them out.
            flow = tl.where(cat_mask, flow, 0.0)

            if keep_zero_params:
                cum_flow += tl.where(param < 1e-12, 0.0, flow)
            else:
                cum_flow += flow

        cum_flow = (1.0 - step_size) + step_size * cum_flow

        # Parameter update
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids + num_cats + cat_id, mask = cat_mask, other = 0)

            new_param = param * ((1.0 - step_size) + step_size * (flow_num + numerate_pseudocount) / (flow_denom + pseudocount * param)) / cum_flow

            # Numerical guard: the MAP denominator self-floors only when F- is computed fresh from
            # the current beta. Under flow momentum, a stale (larger) F- is divided against a fast-
            # collapsing beta, so dying categories underflow float32 -> NaN. Clamp far below any real
            # token probability (~1e-30, ~8 orders above float32 underflow); LL-neutral.
            new_param = tl.maximum(new_param, 1e-30)

            if keep_zero_params:
                new_param = tl.where(param < 1e-12, 0.0, new_param)

            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    @staticmethod
    def large_ncats_dual_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                               step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        numerate_pseudocount = pseudocount / num_cats
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids[:,None] + num_cats[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            # MAP M-step (Dirichlet prior): denominator F- + pseudocount*beta (see small_ncats variant).
            flow = param * (flow_num + numerate_pseudocount[:,None]) / (flow_denom + pseudocount * param)
            # Padding lanes (cat_id >= num_cats) load param/flows as 0 => 0/0 = NaN; zero them out.
            flow = tl.where(cat_mask, flow, 0.0)

            if keep_zero_params:
                cum_flow += tl.sum(tl.where(param < 1e-12, 0.0, flow), axis = 1)
            else:
                cum_flow += tl.sum(flow, axis = 1)

            cat_ids += 128

        cum_flow = (1.0 - step_size) + step_size * cum_flow

        # Parameter update
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids[:,None] + num_cats[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            new_param = param * ((1.0 - step_size) + step_size * (flow_num + numerate_pseudocount[:,None]) / (flow_denom + pseudocount * param)) / cum_flow[:,None]

            # Numerical guard against momentum-induced underflow (see small_ncats variant).
            new_param = tl.maximum(new_param, 1e-30)

            if keep_zero_params:
                new_param = tl.where(param < 1e-12, 0.0, new_param)

            tl.store(params_ptr + s_pids[:,None] + cat_ids[None,:], new_param, mask = cat_mask)

            cat_ids += 128

    def _need_2nd_kernel_dim(self):
        # The mask / missing-flow path walks all `num_cats` categories per node, which for this
        # distribution is a full vocabulary. Take the [BLOCK_SIZE, TILE_SIZE_K] tiled branch of
        # `bk_(dual_)flow_mask_fn` rather than the scalar `for cat_id in range(num_cats)` one, as
        # `Categorical` does. (This distribution has no `fw_partition_fn`, so the flag's other reader
        # never fires for it.)
        return True

    def _get_constructor(self):
        return SoftEvidenceCategorical, {"num_cats": self.num_cats, "_dual_flow_backward": self._dual_flow_backward}

    def __reduce__(self):
        # DELIBERATELY 2 ARGS. Adding `sort_soft_evidence` here made every .jpc written by this tree
        # unloadable by any older pyjuice (`TypeError: __init__() takes from 2 to 3 positional
        # arguments but 4 were given`) -- including a checkpoint merely round-tripped through it --
        # and froze the knob into the file, contradicting the comment on `__init__`. A performance
        # flag does not belong in the serialised form.
        return (self.__class__, (self.num_cats, self._dual_flow_backward))
