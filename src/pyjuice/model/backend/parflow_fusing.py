from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import FastJITFunction


def compile_cum_par_flows_fn(node2tiednodes, MAX_NBLOCKS = 2048, BLOCK_SIZE = 2048):

    nblock2kernel_specs = dict()
    for source_ns, item in node2tiednodes.items():
        if len(item[0]) > 1: # If the length is 1, then everything is already accumulated in the source node's parflow
            num_par_flows = source_ns._param_flow_range[1] - source_ns._param_flow_range[0]
            pfid_start = source_ns._param_flow_range[0]
            ch_nodes = item[0]

            assert len(ch_nodes) <= MAX_NBLOCKS, f"We only support fusing at most {MAX_NBLOCKS} blocks for parameter flow accumulation. " \
                                                  "Consider setting a greater `max_tied_ns_per_parflow_block` when compiling sum layers."

            nblock = triton.next_power_of_2(len(ch_nodes))

            ch_pfids = []
            for ch_ns in ch_nodes:
                ch_pfids.append(ch_ns._param_flow_range[0])

            if nblock not in nblock2kernel_specs:
                nblock2kernel_specs[nblock] = []

            nblock2kernel_specs[nblock].append([pfid_start, num_par_flows, ch_pfids])

    kernels_args = []
    for nblock, kernel_specs in nblock2kernel_specs.items():

        BLOCK_G = nblock
        BLOCK_M = BLOCK_SIZE // BLOCK_G

        target_pfids = []
        block_sizes = []
        child_pfids = []
        for kernel_spec in kernel_specs:
            pfid_start, num_par_flows, ch_pfids = kernel_spec
            for blk_start in range(0, num_par_flows, BLOCK_M):
                blk_end = min(blk_start + BLOCK_M, num_par_flows)
                blk_size = blk_end - blk_start

                ch_pfid = [chid_start + blk_start for chid_start in ch_pfids]
                ch_pfid.extend([0] * (BLOCK_G - len(ch_pfid)))

                target_pfids.append(pfid_start + blk_start)
                block_sizes.append(blk_size)
                child_pfids.append(ch_pfid)

        target_pfids = torch.tensor(target_pfids).contiguous()
        block_sizes = torch.tensor(block_sizes).contiguous()
        child_pfids = torch.tensor(child_pfids).contiguous()

        kernels_args.append([target_pfids, block_sizes, child_pfids, BLOCK_G, BLOCK_M])

    return kernels_args


def cum_par_flows_to_device(kernels_args, device):
    for i in range(len(kernels_args)):
        target_pfids, block_sizes, ch_pfids, BLOCK_G, BLOCK_M = kernels_args[i]

        kernels_args[i] = [
            target_pfids.to(device),
            block_sizes.to(device),
            ch_pfids.to(device),
            BLOCK_G,
            BLOCK_M
        ]

    return kernels_args


# @triton.jit
@FastJITFunction
def cum_par_flows_kernel(param_flows, target_pfids, block_sizes, ch_pfids, BLOCK_G: tl.constexpr, BLOCK_M: tl.constexpr):

    pid = tl.program_id(axis = 0)

    offs_g = tl.arange(0, BLOCK_G) + pid * BLOCK_G
    offs_chblk = tl.load(ch_pfids + offs_g)
    mask_chblk = offs_chblk >= 0

    block_size = tl.load(block_sizes + pid)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < block_size

    offs_chs = offs_chblk[:,None] + tl.arange(0, BLOCK_M)[None,:]
    ch_pflows = tl.load(param_flows + offs_chs, mask = mask_chblk[:,None] & mask_m[None,:], other = 0)
    
    tar_pflows = tl.sum(ch_pflows, axis = 0)

    tar_pfid = tl.load(target_pfids + pid)
    tl.store(param_flows + tar_pfid + offs_m, tar_pflows, mask = mask_m)


def compute_cum_par_flows(param_flows, kernels_args):

    for kernel_args in kernels_args:

        target_pfids, block_sizes, ch_pfids, BLOCK_G, BLOCK_M = kernel_args

        grid = (target_pfids.size(0),)

        cum_par_flows_kernel[grid](param_flows, target_pfids, block_sizes, ch_pfids, BLOCK_G, BLOCK_M)

    return None
            