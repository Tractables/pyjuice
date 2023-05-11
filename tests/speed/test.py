import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _forward_backward_kernel(fff_ptr, node_vals_ptr, element_vals_ptr, nids_ptr, cids_ptr, 
                                tot_n_nodes: tl.constexpr, tot_n_eles: tl.constexpr,
                                n_nodes: tl.constexpr, n_edges: tl.constexpr, 
                                batch_size: tl.constexpr, n_nodes_per_block_m: tl.constexpr,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # We use BLOCK_M to index over edges, and BLOCK_N to index over batches
    pid0 = tl.program_id(axis = 0)
    pid1 = tl.program_id(axis = 1)
    ne_start = pid0 * BLOCK_M
    b_start = pid1 * BLOCK_N

    # Id of edges processed by the current block
    ne_offsets = ne_start + tl.arange(0, BLOCK_M)
    # Batch ids processed by the current block
    b_offsets = b_start + tl.arange(0, BLOCK_N)
    b_mask = b_offsets < batch_size

    # Get node ids from `nids`
    n_start = ne_start // n_edges
    nid_offsets = n_start + tl.arange(0, n_nodes_per_block_m)
    nid_mask = nid_offsets < n_nodes
    n_ids = tl.load(nids_ptr + nid_offsets, mask = nid_mask, other = 0)

    # Get edge ids from `cids`
    cid_offsets = tl.reshape(ne_offsets, (n_edges, n_nodes_per_block_m))
    cid_mask = (cid_offsets < n_nodes * n_edges) & \
        tl.broadcast_to(nid_mask[None,:], (n_edges, n_nodes_per_block_m))
    ch_ids = tl.load(cids_ptr + cid_offsets, mask = cid_mask, other = 0)

    # Use `ch_ids` to retrieve the corresponding element mars
    ele_offsets = tl.broadcast_to(ch_ids[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) * batch_size + \
        tl.broadcast_to(b_offsets[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
    ele_mask = (ele_offsets < n_edges * batch_size) & \
        tl.broadcast_to(nid_mask[None,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) & \
        tl.broadcast_to(b_mask[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
    ch_logps = tl.load(element_vals_ptr + ele_offsets, mask = ele_mask, other = 0)

    # Take the sum of the child mars
    n_logps = tl.sum(ch_logps, axis = 1)

    # Read out the target indices for `node_vals`
    nmar_offsets = tl.broadcast_to(n_ids[None,:], (BLOCK_N, n_nodes_per_block_m)) * batch_size + \
        tl.broadcast_to(b_offsets[:,None], (BLOCK_N, n_nodes_per_block_m))
    nmar_mask = (nmar_offsets < n_nodes * batch_size) & \
        tl.broadcast_to(nid_mask[None,:], (BLOCK_N, n_nodes_per_block_m)) & \
        tl.broadcast_to(b_mask[:,None], (BLOCK_N, n_nodes_per_block_m))

    # sss = tl.reshape(tl.arange(0, n_edges * n_nodes_per_block_m), (n_edges, n_nodes_per_block_m))
    # sss = tl.reshape(tl.arange(0, BLOCK_N * n_nodes_per_block_m), (BLOCK_N, n_nodes_per_block_m))
    # sss = tl.reshape(tl.arange(0, BLOCK_N * n_edges * n_nodes_per_block_m), (BLOCK_N, n_edges, n_nodes_per_block_m))
    # tl.store(fff_ptr + sss, tl.broadcast_to(n_ids[None,:], (BLOCK_N, n_nodes_per_block_m)))
    
    tl.store(node_vals_ptr + nmar_offsets, n_logps, mask = nmar_mask)

def _forward_backward_triton(fff, node_vals: torch.Tensor, element_vals: torch.Tensor, 
                                nids: torch.Tensor, cids: torch.Tensor, 
                                MAX_BLOCK_M = 256, MAX_BLOCK_N = 64):
    """
    This function is equivalent to running:
    ``` node_vals[nids] = element_vals[cids].sum(dim = 1) ```
    
    Parameters:
    `node_vals`:    Tensor[N, B]
    `element_vals`: Tensor[M, B]
    `nids`:         Tensor[n]
    `cids`:         Tensor[n, c]
    """
    tot_n_nodes = node_vals.size(0)
    tot_n_eles = element_vals.size(0)
    n_nodes = nids.size(0)
    n_edges = cids.size(1)
    batch_size = node_vals.size(1)

    assert n_edges <= MAX_BLOCK_M, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
    assert n_edges & (n_edges - 1) == 0, "`n_edges` must be power of 2."

    BLOCK_M = MAX_BLOCK_M
    BLOCK_N = triton.next_power_of_2(min(MAX_BLOCK_N, batch_size))

    grid = (triton.cdiv(n_nodes * n_edges, BLOCK_M), triton.cdiv(batch_size, BLOCK_N), 1)

    print(BLOCK_M, BLOCK_N)
    print(n_edges)
    _forward_backward_kernel[grid](
        fff_ptr = fff,
        node_vals_ptr = node_vals, 
        element_vals_ptr = element_vals, 
        nids_ptr = nids, 
        cids_ptr = cids, 
        tot_n_nodes = tot_n_nodes,
        tot_n_eles = tot_n_eles,
        n_nodes = n_nodes, 
        n_edges = n_edges, 
        batch_size = batch_size, 
        n_nodes_per_block_m = BLOCK_M // n_edges,
        BLOCK_M = BLOCK_M, 
        BLOCK_N = BLOCK_N
    )

    return None


if __name__ == "__main__":

    # torch.manual_seed(10)

    device = torch.device("cuda:0")

    data = np.load("debug.npz")
    node_mars = torch.from_numpy(data["node_mars"]).to(device)
    element_mars = torch.from_numpy(data["element_mars"]).to(device)
    nids = torch.from_numpy(data["nids"]).to(device)
    cids = torch.from_numpy(data["cids"]).to(device)

    fff = torch.zeros([1000000]).to(device)

    _forward_backward_triton(fff, node_mars, element_mars, nids, cids)

    # print(element_mars[cids].sum(dim = 1))
    # print(fff[:256*4].reshape(64,4,4)[:4,:,:3])
    # print(fff[:64*4].reshape(64,4)[:4,:3])

    import pdb; pdb.set_trace()

    # print(torch.abs(element_mars[cids].sum(dim = 1) - node_mars[nids]).max())

    # print(element_mars[cids].sum(dim = 1))
    # print(node_mars[nids])