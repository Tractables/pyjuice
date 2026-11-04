"""
The dense top-k backward's prologue of :class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`
given each slot's observed candidate (`_DensePrologueDispatch`).

`kernels.bk_dense_prologue_kernel` finds where the observed token sits among a slot's candidates inside every program:
a program is a few nodes by a tile of the batch, so each slot's K candidates are re-scanned by every node tile of its
variable -- 128 times a slot at 1024 latents --, most of the prologue's 1.5-1.6 ms on CoDD's PC (64 blocks of 32 at
4096 latents, 256 blocks at 1024). Here the backward finds it once a slot (`_observed_slots`, a pass over the candidates
in torch) and `_bk_dense_prologue_obs_kernel` reads it; the rest -- the observed-category flow, the evidence gradient's
observed term, the ratio rows -- as the original computes them, in wider node tiles (no candidate tile beside them).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import max_power_of_2_factor

from .common import _missing_slot_mask


@triton_jit
def _bk_dense_prologue_obs_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr,
                                  s_pids_ptr, s_pfids_ptr, nids_ptr, layer_num_nodes, batch_size, node_offset,
                                  logspace_flows: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                  categorical_evidence_logp_ptr, categorical_evidence_logp_grad_ptr, var_idmapping_ptr,
                                  ratio_ptr, kstar_ptr, num_cats: tl.constexpr, ext_num_vars: tl.constexpr,
                                  num_latents: tl.constexpr, update_extflows: tl.constexpr, missing_mask_ptr,
                                  missing_mask_mode: tl.constexpr, missing_mask_num_vars: tl.constexpr,
                                  update_pflows: tl.constexpr):
    """`kernels.bk_dense_prologue_kernel` with the observed candidate read (`kstar_ptr`: [batch, ext_num_vars], -1 where
    the token is not a candidate) instead of searched."""
    pid_b = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)

    offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_b = offsets_b < batch_size
    mask_n = offsets_n < layer_num_nodes
    mask_bn = mask_b[:,None] & mask_n[None,:]

    vid = tl.load(vids_ptr + pid_n * BLOCK_SIZE_N)
    lvid = tl.load(var_idmapping_ptr + vid)

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

    mask_nb = mask_n[:,None] & mask_b[None,:]
    nmars = tl.trans(tl.load(node_mars_ptr + (offsets_n + node_offset)[:,None] * batch_size + offsets_b[None,:],
                             mask = mask_nb, other = 0.0))
    nflows = tl.trans(tl.load(node_flows_ptr + (offsets_n + node_offset)[:,None] * batch_size + offsets_b[None,:],
                              mask = mask_nb, other = 0.0))
    if logspace_flows:
        log_nflows = nflows
        nflows = nflows.exp()
    else:
        log_nflows = nflows.log()

    data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0)

    # Observed-category flow (phase 0)
    if update_pflows:
        tl.atomic_add(param_flows_ptr + s_pfids[None,:] + data[:,None], nflows, mask = mask_bn_ev)

    # The observed token's candidate, and the numerator half of the external gradient onto it
    kstar = tl.load(kstar_ptr + offsets_b * ext_num_vars + lvid, mask = mask_b, other = -1)
    has_match = kstar >= 0
    expar_ptr = categorical_evidence_logp_ptr + offsets_b * (ext_num_vars * num_cats) + lvid * num_cats
    log_ex_p = tl.where(has_match, tl.load(expar_ptr + kstar, mask = mask_b & has_match, other = 0.0), -float("inf"))
    if update_extflows:
        expar_grad_ptr = categorical_evidence_logp_grad_ptr + offsets_b * (ext_num_vars * num_cats) + lvid * num_cats
        tl.atomic_add(expar_grad_ptr + kstar, tl.sum(nflows, axis = 1), mask = mask_b_ev & has_match)

    log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = mask_bn, other = 0.0).log()
    logZ = log_in_p + log_ex_p[:,None] - nmars

    ratio = tl.exp(log_nflows - logZ)
    if missing_mask_mode > 0:
        ratio = tl.where(mask_bn_ev, ratio, 0.0)
    tl.store(ratio_ptr + (lvid * batch_size + offsets_b)[:,None] * num_latents + nids[None,:], ratio, mask = mask_bn)


@triton_jit
def _observed_slots_kernel(cat_ids_ptr, data_ptr, gv_ptr, var_idmapping_ptr, kstar_ptr, batch_size, ext_num_vars,
                           K: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr):
    """kstar[b, lvid] = the first k with cat_ids[b, lvid, k] == data[v, b], -1 if none, for each of the layer's
    variables v (`gv_ptr`, lvid its evidence row): a program a variable by BLOCK_B samples, BLOCK_K candidates a step."""
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)

    vid = tl.load(gv_ptr + pid_v)
    lvid = tl.load(var_idmapping_ptr + vid)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    tok = tl.load(data_ptr + vid * batch_size + offs_b, mask = mask_b, other = 0)

    first = tl.full([BLOCK_B], K, dtype = tl.int32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        ids = tl.load(cat_ids_ptr + (offs_b[:,None].to(tl.int64) * ext_num_vars + lvid) * K + offs_k[None,:],
                      mask = mask_b[:,None] & mask_k[None,:], other = 0)
        hit = (ids == tok[:,None]) & mask_k[None,:]
        first = tl.minimum(first, tl.min(tl.where(hit, offs_k[None,:], K), axis = 1))

    tl.store(kstar_ptr + offs_b * ext_num_vars + lvid, tl.where(first < K, first, -1), mask = mask_b)


def _observed_slots(layer, cat_ids, data):
    """Each slot's observed candidate: kstar[b, v] = the first k with cat_ids[b, v, k] == the token, -1 if none
    (`data`: [pc variables, batch], as the kernels read it) -- written for the layer's variables alone, the entries the
    prologue reads. One launch (`_observed_slots_kernel`, 4 samples a program: within ~0.5 us of the best of 2 / 4 / 8 /
    16 at batches 1-256, MEASURED); the torch form (a compare, its `argmax` and `any`, the gathers around them) was ~10
    launches: 26 against 3.1 us on CoDD's PC at 29 blocks of 32 tokens, 48 against 11.7 at 256."""
    B, V, K = cat_ids.shape
    gv = layer.layer_var_list
    kstar = torch.empty([B, V], dtype = torch.int32, device = cat_ids.device)
    bb = min(4, triton.next_power_of_2(B))
    _observed_slots_kernel[(gv.numel(), triton.cdiv(B, bb))](
        cat_ids_ptr = cat_ids, data_ptr = data, gv_ptr = gv, var_idmapping_ptr = layer.var_idmapping, kstar_ptr = kstar,
        batch_size = B, ext_num_vars = V, K = K, BLOCK_B = bb, BLOCK_K = min(1024, triton.next_power_of_2(K)))
    return kstar


class _DensePrologueDispatch:
    """The dense backward's prologue (`_bk_dense_prologue_obs_kernel`, each slot's observed candidate found once by
    `_observed_slots`), its tiles its own. Mimics `kernel[grid](**kwargs)` as the other dispatches do."""

    def __getitem__(self, grid):
        def launch(**kw):
            layer = kw.pop("_layer")
            B = kw["batch_size"]
            nn = kw["layer_num_nodes"]
            # (a tile of nodes one variable's, as the forward's: at most the node blocks' power-of-two factor). Small
            # tiles, one warp: CoDD's PC (1024 latents, real top-k 1024 evidence), batches 1-256, 8 x 32 is within ~10%
            # of the best tile at each, 15-74% below the tiles of before (up to 64 samples by 4096 / that many nodes, 4
            # warps: 40.1 -> 29.3 us at 29 blocks, 321 -> 274 at 256; MEASURED, device time) -- the observed-category
            # flow's atomics, a sector of `param_flows` each, are the floor
            bb = min(8, triton.next_power_of_2(B))
            bn = min(max_power_of_2_factor(layer.n_block_size), 32)
            kstar = _observed_slots(layer, kw["soft_evidence_cat_ids_ptr"], kw["data_ptr"])
            _bk_dense_prologue_obs_kernel[(triton.cdiv(B, bb), triton.cdiv(nn, bn))](
                params_ptr = kw["params_ptr"], param_flows_ptr = kw["param_flows_ptr"],
                node_flows_ptr = kw["node_flows_ptr"], node_mars_ptr = kw["node_mars_ptr"], data_ptr = kw["data_ptr"],
                vids_ptr = kw["vids_ptr"], s_pids_ptr = kw["s_pids_ptr"], s_pfids_ptr = kw["s_pfids_ptr"],
                nids_ptr = kw["nids_ptr"], layer_num_nodes = nn, batch_size = B, node_offset = kw["node_offset"],
                logspace_flows = kw["logspace_flows"], BLOCK_SIZE_B = bb, BLOCK_SIZE_N = bn,
                categorical_evidence_logp_ptr = kw["categorical_evidence_logp_ptr"],
                categorical_evidence_logp_grad_ptr = kw["categorical_evidence_logp_grad_ptr"],
                var_idmapping_ptr = kw["var_idmapping_ptr"], ratio_ptr = kw["ratio_ptr"], kstar_ptr = kstar,
                num_cats = kw["num_cats"], ext_num_vars = kw["ext_num_vars"], num_latents = kw["num_latents"],
                update_extflows = kw["update_extflows"], missing_mask_ptr = kw["missing_mask_ptr"],
                missing_mask_mode = kw["missing_mask_mode"], missing_mask_num_vars = kw["missing_mask_num_vars"],
                update_pflows = kw["update_pflows"], num_warps = 1)
        return launch
