"""
The mini-batch EM update of a layer of Categorical input nodes in a kernel of its own.

The layer's template (`InputLayer._em_kernel_template` with `Categorical.small_ncats_em_fn`) gives each thread one
node and walks its categories one at a time, 1024 nodes a program. Each category's loads wait on the previous
iteration's, so a thread takes ~2 x num_cats memory latencies however few nodes there are: MEASURED (RTX PRO 6000)
263 us for the 8192 source nodes of a 64-category HMM's emissions (8 programs on 188 SMs), 257 us for 2048 -- against
~6 MB of traffic, ~4 us at the DRAM's rate.

Here the arithmetic is the template's, node for node and in the same order -- each node's flows summed category by
category, the same expressions after -- so the parameters come out the same bits. The categories are walked
`CHUNK` at a time, unrolled, so a thread has that many independent loads in flight (unrolling all of them took a
minute to compile at 200 categories), and a program takes 32 nodes, so a layer of a few thousand nodes still fills
the device. MEASURED (512 to 196608 nodes, 2 to 256 categories, uniform and mixed, both `keep_zero_params`): 8192
nodes of 64 categories 270 -> 21 us, of 256 1057 -> 50; 196608 of 64 622 -> 310; at 2 categories both ~10-13 us.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit

# (MEASURED over the shapes above: 32 nodes on one warp and chunks of 32 the fastest or within 10% everywhere; 64 nodes
# on two warps spilled at 256 categories with `keep_zero_params`, 4x slower)
BLOCK_SIZE = 32
NUM_WARPS = 1
CHUNK = 32


@triton_jit
def categorical_em_kernel(params_ptr, param_flows_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr,
                          source_nids_ptr, constexprs_ptr, layer_num_source_nodes, keep_zero_params: tl.constexpr,
                          CHUNK: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_source_nodes

    step_size = tl.load(constexprs_ptr)
    pseudocount = tl.load(constexprs_ptr + 1)

    local_offsets = tl.load(source_nids_ptr + offsets, mask = mask, other = 0)
    s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
    s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)
    s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
    num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)
    max_num_cats = tl.max(num_cats, axis = 0)

    # Cumulative flows (as `small_ncats_em_fn`: category by category)
    numerate_pseudocount = pseudocount / num_cats
    cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
    for cat_start in range(0, max_num_cats, CHUNK):
        for j in tl.static_range(CHUNK):
            cat_id = cat_start + j
            cat_mask = mask & (cat_id < num_cats)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            if keep_zero_params:
                param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
                cum_flow += tl.where(param < 1e-12, 0.0, flow + numerate_pseudocount)
            else:
                cum_flow += flow

    # Parameter update
    cum_flow += pseudocount
    for cat_start in range(0, max_num_cats, CHUNK):
        for j in tl.static_range(CHUNK):
            cat_id = cat_start + j
            cat_mask = mask & (cat_id < num_cats)
            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            if keep_zero_params:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / \
                    (cum_flow - pseudocount)
                new_param = tl.where(param < 1e-12, 0.0, new_param)
            else:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow
            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)


def categorical_em(layer, constexprs, keep_zero_params: bool) -> bool:
    """The update of `layer`'s source nodes, `constexprs` = `[step_size, pseudocount]` on the device; False (nothing
    launched) without source nodes."""
    n = layer.source_nids.size(0)
    if n == 0:
        return False
    categorical_em_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        params_ptr = layer.params, param_flows_ptr = layer.param_flows, s_pids_ptr = layer.s_pids,
        s_pfids_ptr = layer.s_pfids, metadata_ptr = layer.metadata, s_mids_ptr = layer.s_mids,
        source_nids_ptr = layer.source_nids, constexprs_ptr = constexprs, layer_num_source_nodes = n,
        keep_zero_params = keep_zero_params, CHUNK = CHUNK, BLOCK_SIZE = BLOCK_SIZE, num_warps = NUM_WARPS)
    return True
