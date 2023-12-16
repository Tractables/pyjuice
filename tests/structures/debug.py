import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def ref_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                        chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                        local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                        BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_K: tl.constexpr):

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
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)
    log_max = tl.zeros([BLOCK_B], dtype = tl.float32) - float("inf")

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        # log_n_fdm = tl.log(nflows) - nmars
        # log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
        # n_fdm_sub = tl.exp(log_n_fdm - log_n_fdm_max[None,:])

        # partial_flows = tl.dot(epars, n_fdm_sub)

        # acc = tl.where(log_max[None,:] > log_n_fdm_max[None,:], 
        #                acc + tl.exp(log_n_fdm_max - log_max)[None,:] * partial_flows,
        #                partial_flows + tl.exp(log_max - log_n_fdm_max)[None,:] * acc)
        # log_max = tl.maximum(log_max, log_n_fdm_max)

        eflows = tl.sum(epars[:,:,None] * tl.exp(emars[:,None,:] - nmars[None,:,:]) * nflows[None,:,:], axis = 1)
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

    # # Initialize pointers to `element_mars`
    # off_eleids = tl.load(chids + elegroup_id)
    # emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    # emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

    # eflows = acc * tl.exp(emars + log_max[None,:])

    # Write back
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    tl.store(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


@triton.jit
def my_kernel(aaa, bbb, ccc, ddd, eee, node_flows, element_flows, node_mars, element_mars, params, 
                                        chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                        local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                        BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                        GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_K: tl.constexpr):

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

    epars = tl.load(epars_ptr)
    offs1 = pid_m * (TILE_SIZE_M * TILE_SIZE_K) + tl.arange(0, TILE_SIZE_M)[:,None] * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:]
    tl.store(aaa + offs1, epars)
    tl.store(bbb + offs1, offs_ele[:,None] * GROUP_SIZE_K + (par_start + offs_edge_nid)[None,:])

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

    # Batch increment pointers
    parids_inc_ptr = parids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")
    # acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
    # for k in range(0, 1):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        log_n_fdm = tl.log(nflows) - nmars
        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
        n_fdm_sub = tl.where(log_n_fdm_max[None,:] != -float("inf"), 
            tl.exp(log_n_fdm - log_n_fdm_max[None,:]), 0.0)

        offs2 = pid_m * (K_NUM_TILES * TILE_SIZE_K * batch_size) + k * (TILE_SIZE_K * batch_size) + tl.arange(0, TILE_SIZE_K)[:,None] * batch_size + offs_batch[None,:]
        tl.store(ccc + offs2, log_n_fdm, mask = mask_batch[None,:])
        tl.store(ddd + offs2, n_fdm_sub, mask = mask_batch[None,:])

        partial_flows = tl.dot(epars, n_fdm_sub)
        # partial_flows = tl.sum(epars[:,:,None] * n_fdm_sub[None,:,:], axis = 1)

        offs3 = pid_m * (K_NUM_TILES * TILE_SIZE_K * batch_size) + k * (TILE_SIZE_K * batch_size) + tl.arange(0, TILE_SIZE_M)[:,None] * batch_size + offs_batch[None,:]
        tl.store(eee + offs3, partial_flows, mask = mask_batch[None,:])

        acc = tl.where(log_n_fdm_max[None,:] > acc,
            tl.log(partial_flows + tl.exp(acc - log_n_fdm_max[None,:])) + log_n_fdm_max[None,:],
            tl.log(tl.exp(log_n_fdm_max[None,:] - acc) * partial_flows + 1.0) + acc
        )
        # acc += partial_flows

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[:,None] * batch_size
        nflows_ptr += parids_inc[:,None] * batch_size
        parids_inc += ptr_inc_step

    # Initialize pointers to `element_mars`
    off_eleids = tl.load(chids + elegroup_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

    eflows = tl.exp(acc + emars)
    # eflows = acc

    # Write back
    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    tl.store(element_flows + offs_elemfs, eflows, mask = mask_batch[None,:])


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

    # node_flows = torch.rand(node_flows.size(), device = device)

    grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))
    # grid = (1, triton.cdiv(layer_n_nodes, TILE_SIZE_M))

    # ref_kernel[grid](
    #     node_flows = node_flows, 
    #     element_flows = element_flows, 
    #     node_mars = node_mars, 
    #     element_mars = element_mars, 
    #     params = params, 
    #     chids = chids, 
    #     parids_start = parids_start,
    #     parids_increment = parids_increment,
    #     parpids_start = parpids_start,
    #     parpids_increment = parpids_increment, 
    #     local_ids = None, 
    #     batch_size = batch_size, 
    #     partial_eval = 0,
    #     ptr_inc_step = ptr_inc_step,
    #     BLOCK_B = BLOCK_B, 
    #     TILE_SIZE_K = TILE_SIZE_K, 
    #     K_NUM_TILES = K_NUM_TILES,
    #     TILE_SIZE_M = TILE_SIZE_M, 
    #     GROUP_SIZE_M = GROUP_SIZE_M,
    #     GROUP_SIZE_K = GROUP_SIZE_K
    # )

    aaa = ref_kernel[grid]
    aaa(
        node_flows = node_flows, 
        element_flows = element_flows, 
        node_mars = node_mars, 
        element_mars = element_mars, 
        params = params, 
        chids = chids, 
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
        GROUP_SIZE_K = GROUP_SIZE_K
    )

    torch.cuda.synchronize()

    element_flows_ref = element_flows.clone()

    aaa = torch.zeros([grid[1], TILE_SIZE_M, TILE_SIZE_K]).cuda()
    bbb = torch.zeros([grid[1], TILE_SIZE_M, TILE_SIZE_K], dtype = torch.long).cuda()
    ccc = torch.zeros([grid[1], K_NUM_TILES, TILE_SIZE_K, batch_size]).cuda()
    ddd = torch.zeros([grid[1], K_NUM_TILES, TILE_SIZE_K, batch_size]).cuda()
    eee = torch.zeros([grid[1], K_NUM_TILES, TILE_SIZE_M, batch_size]).cuda()

    my_kernel[grid](
        aaa = aaa,
        bbb = bbb,
        ccc = ccc,
        ddd = ddd,
        eee = eee,
        node_flows = node_flows, 
        element_flows = element_flows, 
        node_mars = node_mars, 
        element_mars = element_mars, 
        params = params, 
        chids = chids, 
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
        GROUP_SIZE_K = GROUP_SIZE_K
    )

    # nflows = node_flows[parids[0,0]:parids[0,1],:] # ccc
    # nmars = node_mars[parids[0,0]:parids[0,1],:]
    # epars = params[bbb[0,:,:]] # aaa
    # assert (epars - aaa[0,:,:]).abs().max() < 1e-4

    # log_n_fdm = nflows.log() - nmars
    # log_n_fdm_max = torch.max(log_n_fdm, dim = 0).values
    # n_fdm_sub = torch.exp(log_n_fdm - log_n_fdm_max[None,:]) # ddd
    # assert (n_fdm_sub[:,:BLOCK_B] - ddd[0,:,:BLOCK_B]).abs().max() < 1e-4

    # partial_flows = torch.matmul(epars, n_fdm_sub) # eee
    # # (partial_flows[:,:BLOCK_B].log() - eee[0,:,:BLOCK_B]).abs()

    # print((element_flows_ref[chids,:] - element_flows[chids,:]).abs().max())

    element_flows_ref[chids,143]
    element_flows[chids,143]

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()