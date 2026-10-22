"""
The Triton kernels of :class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`
(:class:`_KernelsMixin`, one of its bases): the forward with and without a value mask, the parameter and soft-evidence
backward, the dense backward's prologue and expected-category flow, and the sampler.
"""

from __future__ import annotations

import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit

from .common import _missing_slot_mask, tlmath


class _KernelsMixin:
    """The Triton kernels of
    :class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`, one of its bases."""

    @staticmethod
    @triton_jit
    def fw_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, layer_num_nodes,
                  batch_size, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset, 
                  TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, use_tensor_core: tl.constexpr,
                  categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr, var_idmapping_ptr, num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr,
                  params_t_ptr, row_idx_ptr, num_rows, TRANSPOSED: tl.constexpr):

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # Global variable ID
        lvid = tl.load(var_idmapping_ptr + vid) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Same nodes, addressed as emission ROWS -- what the transposed table is indexed by.
        if TRANSPOSED:
            rows = tl.load(row_idx_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + \
            lvid * num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # Compute logZ
        logZ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_N], dtype = tl.float32) - float("inf")

        if has_ext_ids:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids # [BLOCK_SIZE_N]

            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Load the internal parameters
                # :note: unlike the backward's expected-category flow phase, the [B, K, N] tile (node
                #        axis innermost) measured FASTER here -- the forward only gathers, it does not
                #        also scatter, and the logsumexp then reduces over the innermost axis. Flipping
                #        this to [B, N, K] cost 43% on the sorted-candidate forward (3.3 -> 4.7 ms).
                if TRANSPOSED:
                    # Candidate-major table: consecutive nodes are ADJACENT rather than `num_cats`
                    # floats apart, so this tile is contiguous instead of one sector per element.
                    in_catpars_ptr = params_t_ptr + catids[:,:,None] * num_rows + rows[None,None,:]
                else:
                    in_catpars_ptr = inpars_ptr[None,None,:] + catids[:,:,None] # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]
                inpars = tl.load(in_catpars_ptr, mask = (mask_b[:,None,None] & mask_c[None,:,None] & mask_n[None,None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                params = expars[:,:,None] + tl.log(inpars)
                params_max = tl.max(params, axis = 1)
                cum_params = tl.log(tl.sum(tl.exp(params - params_max[:,None,:]), axis = 1)) + params_max # [BLOCK_SIZE_B, BLOCK_SIZE_N]

                # Compute logaddexp(logZ, cum_params)
                maxval = tl.maximum(logZ, cum_params)
                minval = tl.minimum(logZ, cum_params)
                diff = minval - maxval

                logZ = tl.where(logZ == -float("inf"),
                    cum_params,
                    maxval + tlmath.log1p(tl.exp(diff))
                )

        else:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + \
                tl.arange(0, TILE_SIZE_K)[:,None] + \
                s_pids[None,:] # [TILE_SIZE_K, BLOCK_SIZE_N]

            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the internal parameters
                inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[:,None] & mask_n[None,:]), other = 0.0) # [TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                expars_max = tl.max(expars, axis = 1)[:,None]
                expars_sub = tl.exp(expars - expars_max)

                if use_tensor_core:
                    params = tl.dot(expars_sub, inpars).log() + expars_max
                else:
                    # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                    params = tl.sum(tl.trans(expars_sub)[:,:,None] * inpars[:,None,:], axis = 0).log() + expars_max

                # Compute logaddexp(logZ, params)
                maxval = tl.maximum(logZ, params)
                minval = tl.minimum(logZ, params)
                diff = minval - maxval

                logZ = tl.where(logZ == -float("inf"),
                    params,
                    maxval + tlmath.log1p(tl.exp(diff))
                )

        # Compute unnormalized logprobs
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0).log() # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if has_ext_ids:
            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expar_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats # [BLOCK_SIZE_B]

            log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Find matching ids (mask out padding categories so they can't spuriously match `data == 0`)
                is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
                # CLAMPED. This is a SUM over matching slots, so it is the intended slot only when the
                # tile holds `data` at most once -- the documented precondition. Two matches at slots
                # i and j yield i + j, which for i + j >= TILE_SIZE_K addresses PAST the tile: an
                # out-of-bounds read (confirmed under compute-sanitizer), and the value it lands on
                # varies with allocator layout, so the log-likelihood is nondeterministic run to run.
                # The clamp cannot restore the right answer for a duplicated id -- nothing can, the
                # information is already gone -- but it keeps the read in bounds and the result
                # deterministic. Branchless on purpose: a `tl.where`/early-out here costs real
                # throughput on the dominant no-duplicate path (cf. the LowRank padded-slot clamp).
                match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1) # [BLOCK_SIZE_B]
                match_ids = tl.minimum(match_ids, TILE_SIZE_K - 1)
                has_match = (tl.sum(is_match, axis = 1) > 0) # [BLOCK_SIZE_B]

                # Load parameters if found
                expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0) # [BLOCK_SIZE_B]
                log_ex_p = tl.where(has_match, expar, log_ex_p)

        else:
            ex_p_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask_b, other = 0.0) # [BLOCK_SIZE_B]

        # Final output logprob
        log_p = log_in_p + log_ex_p[:,None] - logZ

        # Store results
        node_offsets = offsets_n + node_offset
        tl.store(node_mars_ptr + node_offsets[None,:] * batch_size + offsets_b[:,None], log_p, mask = (mask_b[:,None] & mask_n[None,:]))

    @staticmethod
    @triton_jit
    def fw_w_value_mask_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, layer_num_nodes,
                               batch_size, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset, 
                               TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, use_tensor_core: tl.constexpr,
                               categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr, soft_evidence_value_mask_ptr, var_idmapping_ptr,
                               num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr,
                               params_t_ptr, row_idx_ptr, num_rows, TRANSPOSED: tl.constexpr):
        """Forward pass with a per-variable value mask (used for generation/conditional queries).

        A mirror of `fw_kernel` -- everything up to and including `logZ`, `log_in_p` and `log_ex_p` is
        identical; only the final output differs. Per variable, `soft_evidence_value_mask` selects

            observed (`True`)  -> log beta_z(d) + log p_theta(d)      [unnormalized conditional, no -logZ]
            masked   (`False`) -> logZ_z = log sum_c beta_z(c) p_theta(c)   [evidence-weighted marginal]

        which is exactly what `ExternProductCategorical` computes in mode "unnormalized_ll" with a
        value mask. `fw_kernel` is deliberately left untouched so the training/LL forward keeps its
        code path, kernel and performance unchanged.
        """

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # Global variable ID
        lvid = tl.load(var_idmapping_ptr + vid) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Same nodes, addressed as emission ROWS -- what the transposed table is indexed by.
        if TRANSPOSED:
            rows = tl.load(row_idx_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + \
            lvid * num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # Which half of this program's work is actually needed.
        #
        # `log_p` below keeps `logZ` at masked positions and `log_in_p + log_ex_p` at observed ones and
        # DISCARDS the other -- yet both were computed unconditionally, and `logZ` is the full
        # K-candidate sum, by far the most expensive thing in this kernel. A block whose value mask is
        # uniform can skip the unused half outright. That is ALWAYS the case at batch 1, where `vid` is
        # a single variable and `value_mask` a single scalar -- i.e. the whole decode path -- and a
        # block being decoded runs from fully masked to fully observed, so on average about half of
        # this kernel's work was being thrown away. Where a block genuinely mixes the two (batch > 1)
        # both flags are true and the behaviour is exactly as before.
        value_mask = tl.load(soft_evidence_value_mask_ptr + offsets_b * ext_num_vars + lvid, mask = mask_b, other = False) # [BLOCK_SIZE_B]
        mask_bi = mask_b.to(tl.int32)
        any_masked = tl.sum(tl.where(value_mask, 0, 1) * mask_bi) > 0
        any_observed = tl.sum(tl.where(value_mask, 1, 0) * mask_bi) > 0

        # Compute logZ
        logZ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_N], dtype = tl.float32) - float("inf")

        if has_ext_ids:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids # [BLOCK_SIZE_N]

            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            if any_masked:
                for i in range(K_NUM_TILES):
                    mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                    # Load the category IDs from `soft_evidence_cat_ids`
                    catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                    # Load the internal parameters
                    # :note: unlike the backward's expected-category flow phase, the [B, K, N] tile (node
                    #        axis innermost) measured FASTER here -- the forward only gathers, it does not
                    #        also scatter, and the logsumexp then reduces over the innermost axis. Flipping
                    #        this to [B, N, K] cost 43% on the sorted-candidate forward (3.3 -> 4.7 ms).
                    if TRANSPOSED:
                        # See the note in `fw_kernel`: candidate-major makes this tile contiguous.
                        in_catpars_ptr = params_t_ptr + catids[:,:,None] * num_rows + rows[None,None,:]
                    else:
                        in_catpars_ptr = inpars_ptr[None,None,:] + catids[:,:,None] # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]
                    inpars = tl.load(in_catpars_ptr, mask = (mask_b[:,None,None] & mask_c[None,:,None] & mask_n[None,None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]

                    # Load the external parameters
                    expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                    params = expars[:,:,None] + tl.log(inpars)
                    params_max = tl.max(params, axis = 1)
                    cum_params = tl.log(tl.sum(tl.exp(params - params_max[:,None,:]), axis = 1)) + params_max # [BLOCK_SIZE_B, BLOCK_SIZE_N]

                    # Compute logaddexp(logZ, cum_params)
                    maxval = tl.maximum(logZ, cum_params)
                    minval = tl.minimum(logZ, cum_params)
                    diff = minval - maxval

                    logZ = tl.where(logZ == -float("inf"),
                        cum_params,
                        maxval + tlmath.log1p(tl.exp(diff))
                    )

        else:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + \
                tl.arange(0, TILE_SIZE_K)[:,None] + \
                s_pids[None,:] # [TILE_SIZE_K, BLOCK_SIZE_N]

            if any_masked:
                for i in range(K_NUM_TILES):
                    mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                    # Load the internal parameters
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[:,None] & mask_n[None,:]), other = 0.0) # [TILE_SIZE_K, BLOCK_SIZE_N]

                    # Load the external parameters
                    expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                    expars_max = tl.max(expars, axis = 1)[:,None]
                    expars_sub = tl.exp(expars - expars_max)

                    if use_tensor_core:
                        params = tl.dot(expars_sub, inpars).log() + expars_max
                    else:
                        # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                        params = tl.sum(tl.trans(expars_sub)[:,:,None] * inpars[:,None,:], axis = 0).log() + expars_max

                    # Compute logaddexp(logZ, params)
                    maxval = tl.maximum(logZ, params)
                    minval = tl.minimum(logZ, params)
                    diff = minval - maxval

                    logZ = tl.where(logZ == -float("inf"),
                        params,
                        maxval + tlmath.log1p(tl.exp(diff))
                    )

        # Compute unnormalized logprobs
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0).log() # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if has_ext_ids:
            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expar_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats # [BLOCK_SIZE_B]

            log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
            if any_observed:
                for i in range(K_NUM_TILES):
                    mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                    # Load the category IDs from `soft_evidence_cat_ids`
                    catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                    # Find matching ids (mask out padding categories so they can't spuriously match `data == 0`)
                    is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
                    # CLAMPED. This is a SUM over matching slots, so it is the intended slot only when the
                    # tile holds `data` at most once -- the documented precondition. Two matches at slots
                    # i and j yield i + j, which for i + j >= TILE_SIZE_K addresses PAST the tile: an
                    # out-of-bounds read (confirmed under compute-sanitizer), and the value it lands on
                    # varies with allocator layout, so the log-likelihood is nondeterministic run to run.
                    # The clamp cannot restore the right answer for a duplicated id -- nothing can, the
                    # information is already gone -- but it keeps the read in bounds and the result
                    # deterministic. Branchless on purpose: a `tl.where`/early-out here costs real
                    # throughput on the dominant no-duplicate path (cf. the LowRank padded-slot clamp).
                    match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1) # [BLOCK_SIZE_B]
                    match_ids = tl.minimum(match_ids, TILE_SIZE_K - 1)
                    has_match = (tl.sum(is_match, axis = 1) > 0) # [BLOCK_SIZE_B]

                    # Load parameters if found
                    expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0) # [BLOCK_SIZE_B]
                    log_ex_p = tl.where(has_match, expar, log_ex_p)

        else:
            ex_p_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask_b, other = 0.0) # [BLOCK_SIZE_B]


        # Final output logprob: unnormalized conditional where observed, logZ where masked
        log_p = tl.where(value_mask[:,None], log_in_p + log_ex_p[:,None], logZ)

        # Store results
        node_offsets = offsets_n + node_offset
        tl.store(node_mars_ptr + node_offsets[None,:] * batch_size + offsets_b[:,None], log_p, mask = (mask_b[:,None] & mask_n[None,:]))

    @staticmethod
    @triton_jit
    def bk_params_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr, nids_ptr,
                         layer_num_nodes, batch_size, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                         node_offset, logspace_flows: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                         categorical_evidence_logp_ptr, var_idmapping_ptr, num_cats: tl.constexpr, ext_num_vars: tl.constexpr):

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes
        mask_nb = mask_n[:,None] & mask_b[None,:]

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        data = tl.load(data_ptr + vids[:,None] * batch_size + offsets_b[None,:], mask = mask_nb, other = 0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        # Load node flows
        node_offsets = offsets_n + node_offset
        nflows = tl.load(node_flows_ptr + node_offsets[:,None] * batch_size + offsets_b[None,:], mask = mask_nb, other = 0.0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        if logspace_flows:
            nflows = nflows.exp()

        # Cumulate parameter flows
        tl.atomic_add(param_flows_ptr + s_pfids[:,None] + data, nflows, mask = mask_nb)

    @staticmethod
    @triton_jit
    def bk_softevi_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr, nids_ptr,
                          layer_num_nodes, batch_size, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                          node_offset, logspace_flows: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                          TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, use_tensor_core: tl.constexpr,
                          categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr, categorical_evidence_logp_grad_ptr, var_idmapping_ptr,
                          num_cats: tl.constexpr, tot_num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr, update_pflows: tl.constexpr, update_extflows: tl.constexpr,
                          missing_mask_ptr = None, missing_mask_mode: tl.constexpr = 0, missing_mask_num_vars: tl.constexpr = 0):

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # Global variable ID
        lvid = tl.load(var_idmapping_ptr + vid) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + \
            lvid * num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # A marginalized leaf never had its soft evidence applied (the forward overwrote `node_mars` with
        # 0), so `logZ` below is meaningless there and none of this kernel's statistics are valid.
        # `bk_(dual_)flow_mask_fn` writes what those positions should get instead. Every accumulation here
        # is gated on `mask_b`, so narrowing it covers all of them -- except the batch-REDUCED scatter in
        # the no-`ext_ids` branch, which the guard on `nflow_sub_logz_p_max` below handles.
        if missing_mask_mode > 0:
            miss_b = _missing_slot_mask(missing_mask_ptr, vid, offsets_b, batch_size,
                                        missing_mask_num_vars, missing_mask_mode)
            mask_b = mask_b & (miss_b == False)

        # Load nmars
        nmars_ptr = node_mars_ptr + \
            (offsets_n + node_offset)[None,:] * batch_size + \
            offsets_b[:,None] # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        nmars = tl.load(nmars_ptr, mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Load nflows
        nflows_ptr = node_flows_ptr + \
            (offsets_n + node_offset)[None,:] * batch_size + \
            offsets_b[:,None] # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        nflows = tl.load(nflows_ptr, mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if logspace_flows:
            nflows = nflows.exp()
            # `other = 0.0` is a zero flow in linear space but log(1) in log space, so an out-of-range or
            # masked lane would come back as a flow of ONE. Force it to zero either way: `logZ` is -inf on
            # those lanes, and `log(1) - -inf = +inf` (rather than `-inf - -inf = NaN`) is just as
            # poisonous to the batch-reduced scatter below.
            nflows = tl.where(mask_b[:,None] & mask_n[None,:], nflows, 0.0)

        # Compute unnormalized logprobs & backprop the "nominator" parts of the gradients
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        # Update the numerater part of `pflow` if required
        if update_pflows:
            # Get start parameter flow indices
            s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

            tl.atomic_add(param_flows_ptr + s_pfids[None,:] + data[:,None], nflows, mask = (mask_b[:,None] & mask_n[None,:]))

        if has_ext_ids:
            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expar_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats # [BLOCK_SIZE_B]

            # Ptrs pointing to external parameter gradients
            if update_extflows:
                expar_grad_ptr = categorical_evidence_logp_grad_ptr + \
                    offsets_b * (ext_num_vars * num_cats) + \
                    lvid * num_cats # [BLOCK_SIZE_B]

            log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Find matching ids (mask out padding categories so they can't spuriously match `data == 0`)
                is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
                # CLAMPED. This is a SUM over matching slots, so it is the intended slot only when the
                # tile holds `data` at most once -- the documented precondition. Two matches at slots
                # i and j yield i + j, which for i + j >= TILE_SIZE_K addresses PAST the tile: an
                # out-of-bounds read (confirmed under compute-sanitizer), and the value it lands on
                # varies with allocator layout, so the log-likelihood is nondeterministic run to run.
                # The clamp cannot restore the right answer for a duplicated id -- nothing can, the
                # information is already gone -- but it keeps the read in bounds and the result
                # deterministic. Branchless on purpose: a `tl.where`/early-out here costs real
                # throughput on the dominant no-duplicate path (cf. the LowRank padded-slot clamp).
                match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1) # [BLOCK_SIZE_B]
                match_ids = tl.minimum(match_ids, TILE_SIZE_K - 1)
                has_match = (tl.sum(is_match, axis = 1) > 0) # [BLOCK_SIZE_B]

                # Load parameters if found
                expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0) # [BLOCK_SIZE_B]
                log_ex_p = tl.where(has_match, expar, log_ex_p)

                # Accumulate gradients for `extflow`
                if update_extflows:
                    tl.atomic_add(expar_grad_ptr + i * TILE_SIZE_K + match_ids, tl.sum(nflows, axis = 1), mask = (mask_b & has_match)) # [BLOCK_SIZE_B]

        else:
            ex_p_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask_b, other = 0.0) # [BLOCK_SIZE_B]

            # Accumulate gradients
            if update_extflows:
                unnorm_ll_grad_ptr = categorical_evidence_logp_grad_ptr + \
                    offsets_b * (ext_num_vars * num_cats) + \
                    lvid * num_cats + \
                    data
                tl.atomic_add(unnorm_ll_grad_ptr, tl.sum(nflows, axis = 1), mask = mask_b)

        # Retrieve logZ
        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0).log() # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        logZ = log_in_p + log_ex_p[:,None] - nmars # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        # An out-of-range or masked lane loads `params` as 0, so its `logZ` is -inf and every
        # `nflows.log() - logZ` below would be NaN -- which the batch-reduced scatter in the
        # no-`ext_ids` branch then spreads over lanes that ARE valid. `nflows` is exactly 0 on those
        # lanes, so any finite `logZ` leaves them at -inf, i.e. contributing nothing.
        logZ = tl.where(mask_b[:,None] & mask_n[None,:], logZ, 0.0)

        # Loop-invariant, so hoisted out of the K loop below (which otherwise recomputes it once per
        # tile, per phase)
        log_nflow_sub_logz = nflows.log() - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Ptrs pointing to external parameter gradients
        if update_extflows:
            expars_grad_ptr = categorical_evidence_logp_grad_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # Backprop the "denominator" parts of the gradients
        if has_ext_ids:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids # [BLOCK_SIZE_N]

            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            # The working tile is laid out [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K] -- i.e. the CANDIDATE
            # axis is innermost -- rather than [B, K, N]. Both the `params` gather and the `param_flows`
            # scatter address `<row base>(n) + catids(b,k)`, so with N innermost each lane of a warp lands
            # in a different node row (~0.5-1 MB apart) and every access is its own sector; with K
            # innermost a warp stays inside ONE row. Measured on the CoDD config: this phase 25.4 -> 16.6
            # ms (and this is what makes sorted candidate ids pay off -- see the class docstring).
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Load the internal parameters
                in_catpars_ptr = inpars_ptr[None,:,None] + catids[:,None,:] # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]
                inpars = tl.load(in_catpars_ptr, mask = (mask_b[:,None,None] & mask_n[None,:,None] & mask_c[None,None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Shared by both phases below
                log_inpars = inpars.log() # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]

                if update_pflows:
                    vp_grads = log_nflow_sub_logz[:,:,None] + log_inpars + expars[:,None,:] # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]

                    tl.atomic_add(param_flows_ptr + tot_num_cats + s_pfids[None,:,None] + catids[:,None,:], tl.exp(vp_grads), mask = (mask_b[:,None,None] & mask_n[None,:,None]))

                if update_extflows:
                    ve_grads = log_nflow_sub_logz[:,:,None] + log_inpars # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]
                    ve_grads_max = tl.max(ve_grads, axis = 1)
                    ve_grads_sub = tl.exp(ve_grads - ve_grads_max[:,None,:])
                    cum_ve_grads = tl.sum(ve_grads_sub, axis = 1).log() + ve_grads_max # [BLOCK_SIZE_B, TILE_SIZE_K]

                    expars_grad = cum_ve_grads + expars

                    tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, -tl.exp(expars_grad), mask = (mask_b[:,None] & mask_c[None,:]))

        else:
            # Ptrs pointing to internal parameters
            if use_tensor_core:
                inpars_ptr = params_ptr + \
                    tl.arange(0, TILE_SIZE_K)[None,:] + \
                    s_pids[:,None] # [BLOCK_SIZE_N, TILE_SIZE_K]
            else:
                inpars_ptr = params_ptr + \
                    tl.arange(0, TILE_SIZE_K)[:,None] + \
                    s_pids[None,:] # [TILE_SIZE_K, BLOCK_SIZE_N]

            if update_pflows:
                nflow_sub_logz_p = tl.trans(nflows.log() - logZ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]
                nflow_sub_logz_p_max = tl.max(nflow_sub_logz_p, axis = 1)[:,None]
                # This max runs over the BATCH axis, and the scatter it feeds is batch-reduced -- so
                # unlike everywhere else in this kernel, `mask_b` cannot keep a bad lane out of the
                # result. When every batch element of the tile contributes zero flow (all masked, or a
                # node with no flow at all) the max is -inf and `-inf - -inf` would put a NaN into
                # `tl.dot`, which then spreads across the whole [N, K] tile. Pin it to a finite value:
                # every lane is then exp(-inf) = 0, the reduction is 0, and the scatter adds 0.
                nflow_sub_logz_p_max = tl.where(nflow_sub_logz_p_max == float("-inf"), 0.0, nflow_sub_logz_p_max)
                nflow_sub_logz_p_sub = tl.exp(nflow_sub_logz_p - nflow_sub_logz_p_max)

            if update_extflows:
                nflow_sub_logz = nflows.log() - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]
                nflow_sub_logz_max = tl.max(nflow_sub_logz, axis = 1)[:,None]
                nflow_sub_logz_sub = tl.exp(nflow_sub_logz - nflow_sub_logz_max)

            for i in range(K_NUM_TILES):
                offsets_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K))
                mask_c = (offsets_c < num_cats) # [TILE_SIZE_K]

                # Load the internal parameters
                if use_tensor_core:
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[None,:] & mask_n[:,None]), other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
                else:
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[:,None] & mask_n[None,:]), other = 0.0) # [TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                if update_pflows:
                    if use_tensor_core:
                        expars_max = tl.max(expars, axis = 0)[None,:]
                        expars_sub = tl.exp(expars - expars_max)
                        pars_grad = tl.dot(nflow_sub_logz_p_sub, expars_sub) + inpars.log() + nflow_sub_logz_p_max + expars_max
                    else:
                        expars_max = tl.max(expars, axis = 0)
                        expars_sub = tl.exp(tl.trans(expars) - expars_max[:,None]) # [TILE_SIZE_K, BLOCK_SIZE_B]
                        pars_grad = tl.sum(nflow_sub_logz_p_sub[:,None,:] * expars_sub[None,:,:], axis = 2).log() + tl.trans(inpars).log() + nflow_sub_logz_p_max + expars_max[None,:]

                    tl.atomic_add(param_flows_ptr + tot_num_cats + s_pfids[:,None] + offsets_c[None,:], tl.exp(pars_grad), mask = (mask_n[:,None] & mask_c[None,:]))

                if update_extflows:
                    if use_tensor_core:
                        expars_grad = tl.dot(nflow_sub_logz_sub, inpars).log() + nflow_sub_logz_max + expars
                    else:
                        expars_grad = tl.sum(nflow_sub_logz_sub[:,None,:] * inpars[None,:,:], axis = 2).log() + nflow_sub_logz_max + expars

                    tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, -tl.exp(expars_grad), mask = (mask_b[:,None] & mask_c[None,:]))

    @staticmethod
    @triton_jit
    def bk_dense_prologue_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                                 metadata_ptr, s_mids_ptr, nids_ptr, layer_num_nodes, batch_size,
                                 num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                                 node_offset, logspace_flows: tl.constexpr,
                                 BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                 categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr,
                                 categorical_evidence_logp_grad_ptr, var_idmapping_ptr, ratio_ptr,
                                 num_cats: tl.constexpr, ext_num_vars: tl.constexpr, num_latents: tl.constexpr,
                                 update_extflows: tl.constexpr, missing_mask_ptr,
                                 missing_mask_mode: tl.constexpr, missing_mask_num_vars: tl.constexpr,
                                 update_pflows: tl.constexpr = True):
        """First half of the dense top-k backward: everything except the expected-category flow phase.

        Accumulates the observed-category flow, adds the observed-token term of the external evidence
        gradient, and stores `ratio[slot, latent] = node_flow / Z` for the second kernel to consume.
        Identical to `bk_softevi_kernel` up to and including the local normalizer `logZ`; it just stores
        that ratio instead of running the scattered expected-flow loop."""

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes
        mask_bn = mask_b[:,None] & mask_n[None,:]

        offset_n = pid_n * BLOCK_SIZE_N

        vid = tl.load(vids_ptr + offset_n)
        lvid = tl.load(var_idmapping_ptr + vid)

        # A marginalized leaf contributes nothing through the evidence: its soft evidence was never
        # applied (the forward overwrote `node_mars` with 0), so `logZ` below is meaningless there. Drop
        # those batch elements from every accumulation, and store a zero ratio so the expected-flow
        # kernel skips them too. `bk_dual_flow_mask_fn` writes the statistic they should get instead.
        if missing_mask_mode > 0:
            miss_b = _missing_slot_mask(missing_mask_ptr, vid, offsets_b, batch_size,
                                        missing_mask_num_vars, missing_mask_mode)
            mask_b_ev = mask_b & (miss_b == False)
            mask_bn_ev = mask_b_ev[:,None] & mask_n[None,:]
        else:
            mask_b_ev = mask_b
            mask_bn_ev = mask_bn

        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0)
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0)

        # `node_mars` / `node_flows` are laid out [node, batch], so load them with the BATCH axis
        # innermost and transpose in registers. Loading them directly as [batch, node] -- which is the
        # orientation the rest of this kernel wants -- puts a stride of `batch_size` floats between
        # adjacent lanes, so each lane pulls its own 32-byte sector for 4 useful bytes. That alone was
        # 0.50 of the prologue's 0.63 ms.
        mask_nb = mask_n[:,None] & mask_b[None,:]
        nmars = tl.trans(tl.load(
            node_mars_ptr + (offsets_n + node_offset)[:,None] * batch_size + offsets_b[None,:],
            mask = mask_nb, other = 0.0))
        nflows = tl.trans(tl.load(
            node_flows_ptr + (offsets_n + node_offset)[:,None] * batch_size + offsets_b[None,:],
            mask = mask_nb, other = 0.0))

        # Keep the log form for the ratio and the linear form for the observed-category flow, rather
        # than the exp -> log round trip the scattered kernel does.
        if logspace_flows:
            log_nflows = nflows
            nflows = nflows.exp()
        else:
            log_nflows = nflows.log()

        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0)

        # Observed-category flow (phase 0)
        if update_pflows:
            tl.atomic_add(param_flows_ptr + s_pfids[None,:] + data[:,None], nflows, mask = mask_bn_ev)

        catids_ptr = soft_evidence_cat_ids_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + lvid * num_cats + tl.arange(0, TILE_SIZE_K)[None,:]
        expar_ptr = categorical_evidence_logp_ptr + offsets_b * (ext_num_vars * num_cats) + lvid * num_cats
        if update_extflows:
            expar_grad_ptr = categorical_evidence_logp_grad_ptr + offsets_b * (ext_num_vars * num_cats) + lvid * num_cats

        # Locate the observed token among the candidates, and accumulate the numerator half of the
        # external gradient onto its slot
        log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
        for i in range(K_NUM_TILES):
            mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats)
            catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0)

            is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64)
            # CLAMPED. This is a SUM over matching slots, so it is the intended slot only when the
            # tile holds `data` at most once -- the documented precondition. Two matches at slots
            # i and j yield i + j, which for i + j >= TILE_SIZE_K addresses PAST the tile: an
            # out-of-bounds read (confirmed under compute-sanitizer), and the value it lands on
            # varies with allocator layout, so the log-likelihood is nondeterministic run to run.
            # The clamp cannot restore the right answer for a duplicated id -- nothing can, the
            # information is already gone -- but it keeps the read in bounds and the result
            # deterministic. Branchless on purpose: a `tl.where`/early-out here costs real
            # throughput on the dominant no-duplicate path (cf. the LowRank padded-slot clamp).
            match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1)
            match_ids = tl.minimum(match_ids, TILE_SIZE_K - 1)
            has_match = (tl.sum(is_match, axis = 1) > 0)

            expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0)
            log_ex_p = tl.where(has_match, expar, log_ex_p)

            if update_extflows:
                tl.atomic_add(expar_grad_ptr + i * TILE_SIZE_K + match_ids, tl.sum(nflows, axis = 1),
                              mask = (mask_b_ev & has_match))

        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = mask_bn, other = 0.0).log()
        logZ = log_in_p + log_ex_p[:,None] - nmars

        # ratio[slot, latent], slot = lvid * batch_size + b; `nids` is the latent index, so the store is
        # contiguous along the innermost axis
        ratio = tl.exp(log_nflows - logZ)
        if missing_mask_mode > 0:
            # Store, do not skip: `ratio_ptr` is scratch that is reused across steps, so a skipped slot
            # would keep whatever the previous step left there.
            ratio = tl.where(mask_bn_ev, ratio, 0.0)
        tl.store(ratio_ptr + (lvid * batch_size + offsets_b)[:,None] * num_latents + nids[None,:],
                 ratio, mask = mask_bn)

    @staticmethod
    @triton_jit
    def bk_dense_denom_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                              metadata_ptr, s_mids_ptr, nids_ptr, layer_num_nodes, batch_size,
                              num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                              node_offset, logspace_flows: tl.constexpr,
                              uniq_ptr, ref_slot_ptr, ref_pt_ptr, ref_goff_ptr, ref_cnt_ptr, num_uniq_ptr,
                              pf_base_ptr, p_base_ptr, ratio_ptr, categorical_evidence_logp_grad_ptr,
                              num_latents: tl.constexpr, tot_num_cats: tl.constexpr, pf_row_stride: tl.constexpr,
                              MAX_REFS: tl.constexpr, UNIQ_STRIDE: tl.constexpr,
                              BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr,
                              update_extflows: tl.constexpr, update_pflows: tl.constexpr = True,
                              PF_ATOMIC: tl.constexpr = False):
        """Second half of the dense top-k backward: the expected-category flow phase (phase 1).

        Walks (latent x category) so that every (param-flow row, category) has a single owner -- unless a
        long list was cut into pieces, `PF_ATOMIC` -- and needs
        no atomic:
            phase1[row, c] = beta[l, c] * sum_j ratio[slot_j, l] * p_theta_j
        over the soft-evidence slots j whose candidate set contains category c. The same `beta` read
        also gives the expected-value term of the external evidence gradient, so it is folded in here:
            evidence_grad[slot_j, k_j] -= p_theta_j * sum_l ratio[slot_j, l] * beta[l, c]
        """
        pid_l = tl.program_id(axis = 0)
        pid_c = tl.program_id(axis = 1)
        pid_g = tl.program_id(axis = 2)

        num_uniq = tl.load(num_uniq_ptr + pid_g)

        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_l = offs_l < num_latents
        mask_c = offs_c < num_uniq
        m = mask_l[:,None] & mask_c[None,:]

        cats = tl.load(uniq_ptr + pid_g * UNIQ_STRIDE + offs_c, mask = mask_c, other = 0).to(tl.int64)

        # beta[l, cat]: `uniq` is sorted, so this is a near-contiguous run inside each params row
        p_base = tl.load(p_base_ptr + pid_g * num_latents + offs_l, mask = mask_l, other = 0)
        beta = tl.load(params_ptr + p_base[:,None] + cats[None,:], mask = m, other = 0.0) # [BLOCK_L, BLOCK_C]

        ref_base = pid_g * (MAX_REFS * UNIQ_STRIDE) + offs_c   # + j * UNIQ_STRIDE below

        # Reference lists are padded to MAX_REFS but average ~2 entries, and the evidence-gradient term
        # below costs real work on every iteration (it scaled +1.1 ms from MAX_REFS=1 to 16). So bound the
        # loop by the largest list actually present in THIS tile rather than by the global maximum.
        cnt = tl.load(ref_cnt_ptr + pid_g * UNIQ_STRIDE + offs_c, mask = mask_c, other = 0)
        num_refs = tl.max(cnt)

        acc = tl.zeros([BLOCK_L, BLOCK_C], dtype = tl.float32)
        for j in range(num_refs):
            # `j < cnt`, not just `mask_c`: past a category's own list the tables are UNINITIALIZED (see
            # `_build_dense_index`), so a shorter list in this tile must read nothing there. A masked lane
            # gets slot 0 (a valid row) and p = 0, so it contributes nothing to `acc` or the gradient.
            mask_j = mask_c & (j < cnt)
            s = tl.load(ref_slot_ptr + ref_base + j * UNIQ_STRIDE, mask = mask_j, other = 0).to(tl.int64)
            p = tl.load(ref_pt_ptr + ref_base + j * UNIQ_STRIDE, mask = mask_j, other = 0.0)

            # ratio[slot, latent] out of a ~1 MB (L2-resident) scratch. Load it [C, L] so the LATENT axis
            # is innermost and each (cat, ref) reads a contiguous run, then transpose in registers --
            # loading it [L, C] puts `slot` innermost, making every lane its own sector (3x slower).
            r = tl.load(ratio_ptr + s[:,None] * num_latents + offs_l[None,:],
                        mask = mask_c[:,None] & mask_l[None,:], other = 0.0) # [BLOCK_C, BLOCK_L]
            r_t = tl.trans(r)                                                # [BLOCK_L, BLOCK_C]

            if update_pflows:
                acc += r_t * p[None,:]

            if update_extflows:
                # expected-value term of the external evidence gradient, from the beta already in registers
                part = tl.sum(r_t * beta, axis = 0)                          # [BLOCK_C]
                goff = tl.load(ref_goff_ptr + ref_base + j * UNIQ_STRIDE, mask = mask_j, other = 0)
                tl.atomic_add(categorical_evidence_logp_grad_ptr + goff, -p * part,
                              mask = mask_j & (p != 0.0))

        if update_pflows:
            pf_base = tl.load(pf_base_ptr + pid_g * num_latents + offs_l, mask = mask_l, other = 0)
            optr = param_flows_ptr + pf_base[:,None] + tot_num_cats + cats[None,:]
            if PF_ATOMIC:
                # A long list was cut into pieces (`_DENSE_LIST_SPLIT`), so a category can have several
                # owners -- in other programs, and in other lanes of this one.
                tl.atomic_add(optr, beta * acc, mask = m)
            else:
                # One owner per (row, category), so this is a plain read-modify-write.
                # :note: `tl.atomic_add` here is 1.5x SLOWER (2.09 -> 3.22 ms) even though the equivalent
                #        `atomicAdd` in the CUDA kernel is 1.2x FASTER -- Triton does not lower it to a bare
                #        RED (reduction) instruction. That asymmetry is the reason the CUDA path exists.
                tl.store(optr, tl.load(optr, mask = m, other = 0.0) + beta * acc, mask = m)

    @staticmethod
    @triton_jit
    def sample_kernel(samples_ptr, params_ptr, nflow_xids_ptr, nflow_yids_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr,
                      num_activ_nodes, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, batch_size: tl.constexpr, seed, 
                      categorical_evidence_logp_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr,
                      TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_S: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_S

        offsets = block_start + tl.arange(0, BLOCK_S) # [BLOCK_S]
        mask = offsets < num_activ_nodes

        # Raw batch and (local) node id
        local_offsets = tl.load(nflow_xids_ptr + offsets, mask = mask, other = 0)
        batch_offsets = tl.load(nflow_yids_ptr + offsets, mask = mask, other = 0)

        # Load variable ids from `vids_ptr`
        vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask, other = 0)

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64) # [BLOCK_SIZE]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)

        # Ptrs pointing to internal parameters
        inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_S, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp_ptr + \
            batch_offsets[:,None] * (ext_num_vars * max_num_cats) + \
            lvids[:,None] * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:]  # [BLOCK_S, TILE_SIZE_K]

        # Compute logZ
        logZ = tl.zeros([BLOCK_S], dtype = tl.float32) - float("inf")
        for i in range(K_NUM_TILES):
            cat_mask = mask[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None])

            inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)
            expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)

            addlpar = inpar.log() + expar
            addlpar_max = tl.max(addlpar, axis = 1)
            lpar = (addlpar - addlpar_max[:,None]).exp().sum(axis = 1).log() + addlpar_max

            # Compute log-add-exp(logZ, lpar)
            maxval = tl.maximum(logZ, lpar)
            minval = tl.minimum(logZ, lpar)
            diff = minval - maxval

            logZ = tl.where(logZ == -float("inf"),
                lpar,
                maxval + tlmath.log1p(tl.exp(diff))
            )

        # Generate random number
        rnd_val = tl.rand(seed, offsets)

        # Draw samples
        sampled_ids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
        for i in range(K_NUM_TILES):
            cat_mask = mask[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None])

            inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)
            expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)

            probs = tl.exp(tl.log(inpar) + expar - logZ[:,None]) # [BLOCK_S, TILE_SIZE_K]
            cum_probs = tl.cumsum(probs, axis = 1) # [BLOCK_S, TILE_SIZE_K]

            local_catids = tl.sum((rnd_val[:,None] >= cum_probs).to(tl.int64), axis = 1) # [BLOCK_S]

            is_overflow = (local_catids == TILE_SIZE_K)
            rnd_val = tl.where(is_overflow, rnd_val - tl.sum(probs, axis = 1), rnd_val)
            sampled_ids = tl.where(is_overflow | (sampled_ids > -1), sampled_ids, local_catids + i * TILE_SIZE_K)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_ids, mask = mask)
