import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def _bk_triton_block_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                        chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                        local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                        BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_K: tl.constexpr, OP_MODE: tl.constexpr):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node group id from `pid_m`
    elegroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

    # Get the real node group id in the case of partial evaluation
    if partial_eval == 1:
        elegroup_id = tl.load(local_ids + elegroup_id)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // GROUP_SIZE_K
    offs_edge_nid = (offs_edge % GROUP_SIZE_K)
    par_start = tl.load(parpids_start + elegroup_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = params + \
        offs_ele[:,None] * GROUP_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars`
    edge_start = tl.load(parids_start + elegroup_id * ptr_inc_step + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    nflows_ptr = node_flows + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Initialize pointers to `element_mars`
    off_eleids = tl.load(chids + elegroup_id)
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(element_mars + offs_elemfs, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        nmars_max = tl.max(nmars, axis = 0) # [BLOCK_B]
        nflows_div_mars = nflows * tl.exp(nmars_max[None,:] - nmars)

        eflows = tl.sum(epars[:,:,None] * tl.exp(emars[:,None,:] - nmars[None,:,:]) * nflows[None,:,:], axis = 1)
        
        # if OP_MODE == 0:
        #     # Built-in matmul kernel of triton + float16
        #     epars = epars.to(tl.float16)
        #     nflows_div_mars = nflows_div_mars.to(tl.float16)
        #     eflows = tl.dot(epars, nflows_div_mars).to(tl.float32)
        # if OP_MODE == 1:
        #     # Built-in matmul kernel of triton + float32
        #     eflows = tl.dot(epars, nflows_div_mars)
        # if OP_MODE == 2:
        #     # Simulated matmul kernel + float16
        #     epars = epars.to(tl.float16)
        #     nflows_div_mars = nflows_div_mars.to(tl.float16)
        #     eflows = tl.sum(epars[:,:,None] * nflows_div_mars[None,:,:], axis = 1).to(tl.float32)
        # if OP_MODE == 3:
        #     # Simulated matmul kernel + float32
        #     eflows = tl.sum(epars[:,:,None] * nflows_div_mars[None,:,:], axis = 1)

        acc += eflows

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[:,None] * batch_size
        nflows_ptr += parids_inc[:,None] * batch_size
        parids_inc += ptr_inc_step

    # Write back
    off_eleids = tl.load(chids + elegroup_id)
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


def main():

    device = torch.device("cuda:0")

    data = np.load("temp.npz")

    node_flows = torch.from_numpy(data["node_flows"]).to(device)
    element_flows = torch.from_numpy(data["element_flow"]).to(device)
    node_mars = torch.from_numpy(data["node_mars"]).to(device)
    element_mars = torch.from_numpy(data["element_mars"]).to(device)
    params = torch.from_numpy(data["params"]).to(device)
    chids = torch.from_numpy(data["chids"]).to(device)
    parids = torch.from_numpy(data["parids"]).to(device)
    parids_start = torch.from_numpy(data["parids_start"]).to(device)
    parids_increment = torch.from_numpy(data["parids_increment"]).to(device)
    parpids = torch.from_numpy(data["parpids"]).to(device)
    parpids_start = torch.from_numpy(data["parpids_start"]).to(device)
    parpids_increment = torch.from_numpy(data["parpids_increment"]).to(device)
    batch_size = int(data["batch_size"])
    ptr_inc_step = int(data["ptr_inc_step"])
    BLOCK_B = int(data["BLOCK_B"])
    TILE_SIZE_M = int(data["TILE_SIZE_M"])
    TILE_SIZE_K = int(data["TILE_SIZE_K"])
    K_NUM_TILES = int(data["K_NUM_TILES"])
    GROUP_SIZE_M = int(data["GROUP_SIZE_M"])
    GROUP_SIZE_K = int(data["GROUP_SIZE_K"])
    OP_MODE = int(data["OP_MODE"])
    layer_n_nodes = int(data["layer_n_nodes"])

    grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

    import pdb; pdb.set_trace()

    ori_chids = chids
    ori_parids = parids
    ori_parpids = parpids

    num_ngroups = chids.size(0)
    num_egroups = parids.size(1)
    parids = (parids[:,:,None].repeat(1, 1, GROUP_SIZE_M) + torch.arange(0, GROUP_SIZE_M, device = parids.device)).reshape(num_ngroups, num_egroups * GROUP_SIZE_M)
    parpids = (parpids[:,:,None] + torch.arange(0, GROUP_SIZE_M, device = parids.device)).reshape(
        num_ngroups, num_egroups * GROUP_SIZE_M)

    chids = (chids[:,None].repeat(1, GROUP_SIZE_K) + torch.arange(0, GROUP_SIZE_K, device = chids.device)).reshape(num_ngroups * GROUP_SIZE_K)
    parids = parids[:,None,:].repeat(1, GROUP_SIZE_K, 1).reshape(num_ngroups * GROUP_SIZE_K, num_egroups * GROUP_SIZE_M)
    parpids = (parpids[:,None,:].repeat(1, GROUP_SIZE_K, 1) + torch.arange(0, GROUP_SIZE_K * GROUP_SIZE_M, GROUP_SIZE_M, device = parpids.device)[None,:,None]).reshape(
        num_ngroups * GROUP_SIZE_K, num_egroups * GROUP_SIZE_M
    )
    
    element_flows[chids] = (node_flows[parids] * params[parpids].unsqueeze(-1) * (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

    import pdb; pdb.set_trace()

    _bk_triton_block_sparse_ele_kernel[grid](
        node_flows = node_flows, 
        element_flows = element_flows, 
        node_mars = node_mars, 
        element_mars = element_mars, 
        params = params, 
        chids = ori_chids, 
        parids_start = parids_start,
        parids_increment = parids_increment,
        parpids_start = parpids_start,
        parpids_increment = parpids_increment, 
        local_ids = None, 
        batch_size = batch_size, 
        partial_eval = 0,
        ptr_inc_step = ptr_inc_step,
        BLOCK_B = BLOCK_B, 
        TILE_SIZE_K = TILE_SIZE_K, 
        K_NUM_TILES = K_NUM_TILES,
        TILE_SIZE_M = TILE_SIZE_M, 
        GROUP_SIZE_M = GROUP_SIZE_M,
        GROUP_SIZE_K = GROUP_SIZE_K,
        OP_MODE = OP_MODE
    )

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()