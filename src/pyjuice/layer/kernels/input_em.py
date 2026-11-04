"""
The mini-batch EM update of a layer of Categorical input nodes in a kernel of its own.

The layer's template (`InputLayer._em_kernel_template` with `Categorical.small_ncats_em_fn`) gives each thread one
node and walks its categories one at a time, 1024 nodes a program. Each category's loads wait on the previous
iteration's, so a thread takes ~2 x num_cats memory latencies however few nodes there are: MEASURED (RTX PRO 6000)
263 us for the 8192 source nodes of a 64-category HMM's emissions (8 programs on 188 SMs), 257 us for 2048 -- against
~6 MB of traffic, ~4 us at the DRAM's rate. And a thread per node reads addresses `num_cats` apart from its
neighbours': from 8 categories on, a load instruction touches a 32-byte sector per lane for one float of it (an
earlier kernel of this file kept the template's order that way: 8192 nodes of 64 categories 21 us, 262144 of 256
1950 us against a 550 us floor).

Here a program takes `BLOCK_N` nodes by `CHUNK` categories as one tile, the lanes along the categories: a node's
flows and parameters are read contiguously, and its sum over the categories is the tile's row sum (`tl.sum`),
accumulated chunk by chunk. That reassociates the template's sum -- its terms are flows, all >= 0, so the result is
the same to the rounding of an fp32 sum of `num_cats` terms (~1e-7 relative; the tests take both against a float64
reference) -- and everything after the sum is the template's expression. MEASURED: see `BLOCK_N` / `CHUNK`.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit

#: Nodes a program takes, and the categories a chunk spans (a lane a category: 32-wide loads of a node's row)
BLOCK_N = 8
CHUNK = 32
NUM_WARPS = 4


@triton_jit
def categorical_em_kernel(params_ptr, param_flows_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr,
                          source_nids_ptr, constexprs_ptr, layer_num_source_nodes, keep_zero_params: tl.constexpr,
                          CHUNK: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < layer_num_source_nodes

    step_size = tl.load(constexprs_ptr)
    pseudocount = tl.load(constexprs_ptr + 1)

    local_offsets = tl.load(source_nids_ptr + offsets, mask = mask, other = 0)
    s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
    s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)
    s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
    num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)
    max_num_cats = tl.max(num_cats, axis = 0)

    cats = tl.arange(0, CHUNK)
    pf_base = s_pfids[:, None] + cats[None, :]                         # [BLOCK_N, CHUNK] (chunk 0)
    p_base = s_pids[:, None] + cats[None, :]
    numerate_pseudocount = pseudocount / num_cats                      # [BLOCK_N]

    # Cumulative flows: a node's row read CHUNK at a time, summed over the chunk
    cum_flow = tl.zeros([BLOCK_N], dtype = tl.float32)
    for cat_start in range(0, max_num_cats, CHUNK):
        cat_mask = mask[:, None] & ((cat_start + cats)[None, :] < num_cats[:, None])
        flow = tl.load(param_flows_ptr + pf_base + cat_start, mask = cat_mask, other = 0.0)
        if keep_zero_params:
            param = tl.load(params_ptr + p_base + cat_start, mask = cat_mask, other = 0.0)
            # (a lane past the node's categories loads param 0: its term is 0 here as well)
            cum_flow += tl.sum(tl.where(param < 1e-12, 0.0, flow + numerate_pseudocount[:, None]), axis = 1)
        else:
            cum_flow += tl.sum(flow, axis = 1)

    # Parameter update (the template's expressions)
    cum_flow += pseudocount
    for cat_start in range(0, max_num_cats, CHUNK):
        cat_mask = mask[:, None] & ((cat_start + cats)[None, :] < num_cats[:, None])
        param = tl.load(params_ptr + p_base + cat_start, mask = cat_mask, other = 0.0)
        flow = tl.load(param_flows_ptr + pf_base + cat_start, mask = cat_mask, other = 0.0)
        if keep_zero_params:
            new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount[:, None]) / \
                (cum_flow - pseudocount)[:, None]
            new_param = tl.where(param < 1e-12, 0.0, new_param)
        else:
            new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount[:, None]) / \
                cum_flow[:, None]
        tl.store(params_ptr + p_base + cat_start, new_param, mask = cat_mask)


def categorical_em(layer, constexprs, keep_zero_params: bool) -> bool:
    """The update of `layer`'s source nodes, `constexprs` = `[step_size, pseudocount]` on the device; False (nothing
    launched) without source nodes."""
    n = layer.source_nids.size(0)
    if n == 0:
        return False
    categorical_em_kernel[(triton.cdiv(n, BLOCK_N),)](
        params_ptr = layer.params, param_flows_ptr = layer.param_flows, s_pids_ptr = layer.s_pids,
        s_pfids_ptr = layer.s_pfids, metadata_ptr = layer.metadata, s_mids_ptr = layer.s_mids,
        source_nids_ptr = layer.source_nids, constexprs_ptr = constexprs, layer_num_source_nodes = n,
        keep_zero_params = keep_zero_params, CHUNK = CHUNK, BLOCK_N = BLOCK_N, num_warps = NUM_WARPS)
    return True
