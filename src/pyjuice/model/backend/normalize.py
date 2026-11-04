from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from numba import njit

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import device_scalars


@triton_jit
def entry_sums_kernel(entry_sums, mparams, par_start_ids, blk_sizes, blk_intervals, constexprs,
                      BLOCK_ID: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Every table entry's sum, each stored at its own place: no atomic adds, whose order would decide the rounding of a
    # node's total where the node takes several entries (~1e-7, run to run)
    pid = tl.program_id(axis = 0)

    # Retrieve the constants
    num_blocks = tl.load(constexprs + 1).to(tl.int64)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)

    offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_par = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    pars = tl.load(mparams + offs_par, mask = mask_par, other = 0).to(tl.float32)

    tl.store(entry_sums + offs_m, tl.sum(pars, axis = 1), mask = mask_m)


@triton_jit
def par_update_kernel(mparams, cum_pflows, nchs, par_start_ids, blk_sizes, blk_intervals,
                      global_nids, constexprs, BLOCK_ID: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)

    # Retrieve the constants
    pseudocount = tl.load(constexprs)
    num_blocks = tl.load(constexprs + 1).to(tl.int64)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_par = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    pars = tl.load(mparams + offs_par, mask = mask_par, other = 0).to(tl.float32)

    sum_pars = tl.load(cum_pflows + global_nid, mask = mask_m, other = 1)
    nch = tl.load(nchs + global_nid, mask = mask_m, other = 1)

    norm_param = (pars + pseudocount / nch[:,None]) / (sum_pars[:,None] + pseudocount)

    tl.store(mparams + offs_par, norm_param.to(mparams.dtype.element_ty), mask = mask_par)


@njit(cache = True)
def cum_par_numba_kernel(cum_pflows, params, par_start_ids, blk_sizes, blk_intervals, global_nids):
    for i in range(par_start_ids.shape[0]):
        par_start_id = par_start_ids[i]
        blk_size = blk_sizes[i]
        blk_interval = blk_intervals[i]
        global_nid = global_nids[i]

        cum_par = 0.0
        for j in range(blk_size):
            cum_par += params[par_start_id+j*blk_interval]

        cum_pflows[global_nid] += cum_par


@njit(cache = True)
def par_update_numba_kernel(params, cum_pflows, nchs, par_start_ids, blk_sizes, blk_intervals, global_nids, pseudocount):
    for i in range(par_start_ids.shape[0]):
        par_start = par_start_ids[i]
        blk_size = blk_sizes[i]
        blk_interval = blk_intervals[i]
        global_nid = global_nids[i]

        cum_par = cum_pflows[global_nid]
        nch = nchs[global_nid]

        for j in range(blk_size):
            par = params[par_start+j*blk_interval]
            norm_par = (par + pseudocount / nch) / (cum_par + pseudocount)
            params[par_start+j*blk_interval] = norm_par


def normalize_parameters(params, par_update_kwargs, pseudocount: float = 0.0):

    par_start_ids, _, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = par_update_kwargs

    tot_num_nodes = metadata["tot_num_nodes"]
    BLOCK_SIZE = metadata["BLOCK_SIZE"]

    if cum_pflows is None:
        cum_pflows = torch.zeros([tot_num_nodes], dtype = torch.float32, device = params.device)
    else:
        cum_pflows[:] = 0.0

    use_cuda = params.is_cuda

    if use_cuda:

        num_blocks = par_start_ids.size(0)
        BLOCK_ID = 2048 // BLOCK_SIZE

        constexprs = device_scalars([pseudocount, num_blocks], params.device)

        grid = (triton.cdiv(num_blocks, BLOCK_ID),)

        entry_sums = torch.empty([num_blocks], dtype = torch.float32, device = params.device)
        entry_sums_kernel[grid](
            entry_sums, params, par_start_ids, blk_sizes, blk_intervals, constexprs, BLOCK_ID, BLOCK_SIZE
        )

        # The nodes' totals in the table's order, on the CPU (`index_add_` on the GPU is not ordered): the same bits on
        # every run, and on every rank
        cum_pflows.copy_(torch.zeros([tot_num_nodes], dtype = torch.float32).index_add_(
            0, global_nids.cpu(), entry_sums.cpu()))

        par_update_kernel[grid](
            params, cum_pflows, nchs, par_start_ids, blk_sizes, blk_intervals,
            global_nids, constexprs, BLOCK_ID, BLOCK_SIZE
        )

    else:

        cum_pflows = cum_pflows.numpy()
        params = params.numpy()
        par_start_ids = par_start_ids.numpy()
        blk_sizes = blk_sizes.numpy()
        blk_intervals = blk_intervals.numpy()
        global_nids = global_nids.numpy()
        nchs = nchs.numpy()

        cum_par_numba_kernel(
            cum_pflows, params, par_start_ids, blk_sizes, 
            blk_intervals, global_nids
        )

        par_update_numba_kernel(
            params, cum_pflows, nchs, par_start_ids, blk_sizes, 
            blk_intervals, global_nids, pseudocount
        )
