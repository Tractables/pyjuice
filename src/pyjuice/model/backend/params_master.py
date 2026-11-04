"""
The fp32 master copy of the sum layers' parameters, for a circuit that STORES them below fp32
(`TensorCircuit.param_dtype`) while it trains.

A bf16 value moves only when an update exceeds half its ulp (2^-9 relative): MEASURED on an HCLT, a mini-batch EM step
of size 0.1 left 53% of the parameters unchanged, 0.01 99.9%, an SGD step 100% -- the sum layers stop learning. So the
updates (`em_par_update`, `sgd_par_update`, the external-parameter EM corrections, the gradient optimizers' clamp and
DDP's averaging) run on an fp32 copy, the exact fp32 updates they are at fp32 storage, and the stored parameters are
refreshed from it (`TensorCircuit._store_updated_params`); the forward and backward kernels keep reading the bf16
values. The copy is made by the first update and costs 4 bytes per parameter until dropped
(`TensorCircuit.drop_params_master`); a circuit that never updates never has one.

A write to the stored parameters that does not go through the copy (`pc.params.data[...] = x`, which no version counter
detects) leaves them no longer the copy's rounding: `sync_master` takes those values into the copy before the next
update, so the copy never silently overwrites them.

The kernels that make the parameter flows read the stored values too, so every flow has its parameter's ROUNDING as its
last factor (a flow is `theta * c`): EM on the flows as they are is the update of the rounded circuit, and with the same
rounding error in every step the fixed point it converges to moves along the update's slow directions (MEASURED,
HCLT-128 on MNIST, 350 epochs: -0.48 and -0.55 nats a sample against fp32 training, two seeds; an fp32 emulation that
rounds the parameters only for the passes reproduces it). So the updates take each flow to the master copy's value
first, `flows_to_master`: times `master / stored`. The rest of the flow (`c`) still comes from the rounded circuit;
its errors are not the parameter's own and do not repeat (the emulation with the factor: +0.05..+0.25 against its fp32
arm through 250 epochs, without it -0.26..-0.45 in the same runs).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from pyjuice.utils.util import device_scalars


@triton.jit
def _sync_master_kernel(master_ptr, params_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    m = tl.load(master_ptr + offs, mask = mask, other = 0.0)
    p = tl.load(params_ptr + offs, mask = mask, other = 0.0)
    # Written since the copy was last stored: the stored value is no longer the copy's rounding (round to nearest
    # even, as torch's `copy_` that stores it)
    stale = p != m.to(params_ptr.dtype.element_ty)
    tl.store(master_ptr + offs, p.to(tl.float32), mask = mask & stale)


def sync_master(master: torch.Tensor, params: torch.Tensor, BLOCK: int = 1024) -> None:
    """Take into ``master`` (fp32) every stored parameter of ``params`` that is not ``master``'s rounding."""
    n = params.numel()
    if n == 0:
        return
    _sync_master_kernel[(triton.cdiv(n, BLOCK),)](master, params, n, BLOCK = BLOCK)


@triton.jit
def flows_to_master(flows, stored, old):
    """Parameter flows made with the STORED values (``stored``, any dtype) taken to the values an update starts from
    (``old``, fp32: the master copy's, or the stored value where that was written from outside -- a factor of 1 there):
    times ``old / stored`` (see the module docstring). 1 where they are equal, and where the stored value is 0 (a zero
    parameter's flows are 0)."""
    s = stored.to(tl.float32)
    return flows * tl.where((s == 0.0) | (s == old), 1.0, old / s)


@triton.jit
def _cum_pflow_master_kernel(cum_pflows, master, stored, param_flows, nchs, par_start_ids, pflow_start_ids, blk_sizes,
                             blk_intervals, global_nids, constexprs, keep_zero_params: tl.constexpr,
                             BLOCK_ID: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # `par_update.cum_pflow_kernel`'s totals of the flows taken to the master copy's values (`flows_to_master`), the old
    # value read through the external-write check as the update reads it
    pid = tl.program_id(axis = 0)

    pseudocount = tl.load(constexprs + 1)
    num_blocks = tl.load(constexprs + 2).to(tl.int64)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    pflow_start = tl.load(pflow_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_pflow = pflow_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_pflow = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    pflows = tl.load(param_flows + offs_pflow, mask = mask_pflow, other = 0)

    offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    old_master = tl.load(master + offs_par, mask = mask_pflow, other = 0)
    old_stored = tl.load(stored + offs_par, mask = mask_pflow, other = 0)
    old_param = tl.where(old_stored != old_master.to(stored.dtype.element_ty), old_stored.to(tl.float32), old_master)
    pflows = flows_to_master(pflows, old_stored, old_param)

    if keep_zero_params:
        nch = tl.load(nchs + offs_m, mask = mask_m, other = 1)
        pflows += (pseudocount / nch[:,None])
        nflows = tl.sum(tl.where(old_param == 0.0, 0.0, pflows), axis = 1)
    else:
        nflows = tl.sum(pflows, axis = 1)

    tl.atomic_add(cum_pflows + global_nid, nflows, mask = mask_m)


@triton.jit
def _em_par_update_master_kernel(master, stored, param_flows, cum_pflows, nchs, par_start_ids, pflow_start_ids,
                                 blk_sizes, blk_intervals, global_nids, constexprs, keep_zero_params: tl.constexpr,
                                 BLOCK_ID: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # `par_update.em_par_update_kernel`'s update on the master copy, the external-write check and the store fused in:
    # the old value is the master's, or the stored one where that is no longer the master's rounding; the new value is
    # written to both -- one pass over the parameters instead of three (MEASURED on HCLT-1024's 472 M parameters: the
    # separate check and store added 3.6 ms to a 29.6 ms update). The flows taken to the master copy's values
    # (`flows_to_master`), as their totals were
    pid = tl.program_id(axis = 0)

    step_size = tl.load(constexprs)
    pseudocount = tl.load(constexprs + 1)
    num_blocks = tl.load(constexprs + 2).to(tl.int64)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    pflow_start = tl.load(pflow_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_pflow = pflow_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_pflow = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    pflows = tl.load(param_flows + offs_pflow, mask = mask_pflow, other = 0)

    offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    old_master = tl.load(master + offs_par, mask = mask_pflow, other = 0)
    old_stored = tl.load(stored + offs_par, mask = mask_pflow, other = 0)
    old_param = tl.where(old_stored != old_master.to(stored.dtype.element_ty), old_stored.to(tl.float32), old_master)
    pflows = flows_to_master(pflows, old_stored, old_param)

    nflows = tl.load(cum_pflows + global_nid, mask = mask_m, other = 1)
    nch = tl.load(nchs + offs_m, mask = mask_m, other = 1)

    if keep_zero_params:
        new_param = (pflows + pseudocount / nch[:,None]) / nflows[:,None]
    else:
        new_param = (pflows + pseudocount / nch[:,None]) / (nflows[:,None] + pseudocount)

    if keep_zero_params:
        updated_params = tl.where(old_param == 0.0, 0.0, (1.0 - step_size) * old_param + step_size * new_param)
    else:
        updated_params = (1.0 - step_size) * old_param + step_size * new_param

    tl.store(master + offs_par, updated_params, mask = mask_pflow)
    tl.store(stored + offs_par, updated_params.to(stored.dtype.element_ty), mask = mask_pflow)


def em_par_update_master(master: torch.Tensor, stored: torch.Tensor, param_flows: torch.Tensor, par_update_kwargs,
                         step_size: float, pseudocount: float = 0.0, keep_zero_params: bool = True) -> None:
    """`par_update.em_par_update` on parameters stored below fp32 (``stored``) through their fp32 master copy
    (``master``): the same update on the flows taken to the master copy's values (`flows_to_master`), the copy brought
    in line with outside writes and the result stored in one pass."""
    par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = par_update_kwargs
    tot_num_nodes = metadata["tot_num_nodes"]
    BLOCK_SIZE = metadata["BLOCK_SIZE"]
    if cum_pflows is None:
        cum_pflows = torch.zeros([tot_num_nodes], dtype = torch.float32, device = master.device)
    else:
        cum_pflows[:] = 0.0
    num_blocks = par_start_ids.size(0)
    BLOCK_ID = 2048 // BLOCK_SIZE
    grid = (triton.cdiv(num_blocks, BLOCK_ID),)
    constexprs = device_scalars([step_size, pseudocount, num_blocks], master.device)
    _cum_pflow_master_kernel[grid](
        cum_pflows, master, stored, param_flows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals,
        global_nids, constexprs, keep_zero_params, BLOCK_ID, BLOCK_SIZE
    )
    _em_par_update_master_kernel[grid](
        master, stored, param_flows, cum_pflows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals,
        global_nids, constexprs, keep_zero_params, BLOCK_ID, BLOCK_SIZE
    )
