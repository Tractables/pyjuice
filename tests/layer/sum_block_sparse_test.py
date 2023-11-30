import triton
import triton.language as tl
import torch
import numpy as np
import time


@triton.jit
def _forward_triton_kernel(node_mars_ptr, element_mars_ptr, params_ptr, 
                            nids_ptr, cids_ptr, pids_ptr, tot_n_nodes, 
                            tot_n_eles, n_nodes, n_edges: tl.constexpr, 
                            batch_size, n_nodes_per_block_m: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):

    # We use BLOCK_M to index over edges, and BLOCK_N to index over batches
    pid0 = tl.program_id(axis = 0)
    pid1 = tl.program_id(axis = 1)
    ne_start = pid0 * BLOCK_M
    b_start = pid1 * BLOCK_N

    # Id of edges processed by the current block (0.081ms)
    ne_offsets = ne_start + tl.arange(0, BLOCK_M)
    # Batch ids processed by the current block
    b_offsets = b_start + tl.arange(0, BLOCK_N)

    # Get node ids from `nids`
    n_start = ne_start // n_edges
    nid_offsets = n_start + tl.arange(0, n_nodes_per_block_m)
    n_ids = tl.load(nids_ptr + nid_offsets)

    # Get edge ids from `cids`
    cid_offsets = tl.view(ne_offsets, (n_edges, n_nodes_per_block_m))
    ch_ids = tl.load(cids_ptr + cid_offsets)
    # Use `ch_ids` to retrieve the corresponding element mars
    ele_offsets = ch_ids[None,:,:] * batch_size + b_offsets[:,None,None]
    ch_logps = tl.load(element_mars_ptr + ele_offsets) # `element_mars[cids]`

    # Get param ids from `pids`
    # Here we reuse `cid_offsets` and `cid_mask` thank to their similar structure
    par_ids = tl.load(pids_ptr + cid_offsets)

    # Use `par_ids` to retrieve the corresponding parameters
    ch_pars = tl.load(params_ptr + par_ids) # `params[pids]`

    # Take the max of the child mars
    ch_max_logp = tl.max(ch_logps, axis = 1) # `maxval`
    # Subtract the max from child mars
    ch_logps_sub_max = ch_logps - ch_max_logp[:,None,:]
    # Take exp
    ch_ps_sub_max = tl.exp(ch_logps_sub_max)

    # Sum node marginals (unnormalized)
    n_ps = tl.sum(ch_ps_sub_max * ch_pars[None,:,:], axis = 1)

    # Take log and subtract max vals
    n_logps = tl.log(tl.maximum(n_ps, 1e-10)) + ch_max_logp

    # Read out the target indices for `node_mars`
    nmar_offsets = n_ids[None,:] * batch_size + b_offsets[:,None]
    
    # Reshape seems to be necessary for certain combinations of (BLOCK_N, n_nodes_per_block_m)
    nmar_offsets = tl.view(nmar_offsets, (BLOCK_N * n_nodes_per_block_m,))
    n_logps = tl.view(n_logps, (BLOCK_N * n_nodes_per_block_m,))
    tl.store(node_mars_ptr + nmar_offsets, n_logps)


@triton.jit
def block_sparse_kernel(ddd, node_mars, element_mars, params, nids, cids_start, cids_increment, pids_start, pids_increment,
                        tot_n_nodes, tot_n_eles, layer_n_nodes, layer_n_edge_groups, batch_size,
                        BLOCK_M: tl.constexpr, GROUP_SIZE: tl.constexpr):
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1) # batch id

    # initialize pointers to `element_mars`
    node_start = tl.multiple_of(pid_m * layer_n_edge_groups * GROUP_SIZE, 8) # compiler hint
    offs_node = tl.arange(0, BLOCK_M) + node_start
    mask_node = offs_node < layer_n_nodes
    offs_edge = tl.arange(0, GROUP_SIZE)
    edge_start = tl.load(cids_start + offs_node, mask = mask_node, other = 0)
    emars_ptr = element_mars + pid_b * tot_n_eles + edge_start[:,None] + offs_edge[None,:]
    emars_ptr = tl.view(emars_ptr, (BLOCK_M, GROUP_SIZE))

    # initialize pointers to `params`
    param_start = tl.load(pids_start + offs_node, mask = mask_node, other = 0)
    params_ptr = params + param_start[:,None] + offs_edge[None,:]
    # params_ptr = params + offs_edge[:,None] + param_start[None,:]
    params_ptr = tl.view(params_ptr, (BLOCK_M, GROUP_SIZE))

    # Inner loop
    acc = tl.zeros((BLOCK_M,), dtype = tl.float32) - float("inf")

    cids_inc_ptr = cids_increment + offs_node
    pids_inc_ptr = pids_increment + offs_node
    for k in range(0, layer_n_edge_groups):
        emars = tl.load(emars_ptr, mask = mask_node[:,None])
        epars = tl.load(params_ptr, mask = mask_node[:,None])
        emars_max = tl.max(emars, axis = 1)
        emars = tl.exp(emars - emars_max[:,None])

        # nmars = tl.dot(emars, params)
        nmars = tl.sum(emars * epars, axis = 1)

        acc = tl.where(emars_max > acc, 
            tl.log(nmars + tl.exp(acc - emars_max)) + emars_max,
            tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc
        )

        cids_inc = tl.load(cids_inc_ptr, mask = mask_node)
        pids_inc = tl.load(pids_inc_ptr, mask = mask_node)
        emars_ptr += cids_inc
        params_ptr += pids_inc
        cids_inc_ptr += 1
        pids_inc_ptr += 1

    # Write back
    ns = tl.load(nids + offs_node, mask = mask_node)
    tl.store(node_mars + ns + pid_b * tot_n_nodes, tl.ravel(acc), mask = mask_node)


def main_baseline():
    data = np.load("temp.npz")

    device = torch.device("cuda:0")

    node_mars = torch.from_numpy(data["node_mars"]).to(device)
    node_mars2 = node_mars.clone()
    element_mars = torch.from_numpy(data["element_mars"]).to(device)
    params = torch.from_numpy(data["params"]).to(device)
    nids = torch.from_numpy(data["nids"]).to(device)
    cids = torch.from_numpy(data["cids"]).to(device)
    pids = torch.from_numpy(data["pids"]).to(device)
    tot_n_nodes = int(data["tot_n_nodes"])
    tot_n_eles = int(data["tot_n_eles"])
    n_nodes = int(data["n_nodes"])
    n_edges = int(data["n_edges"])
    batch_size = int(data["batch_size"])
    BLOCK_M = int(data["BLOCK_M"])
    BLOCK_N = int(data["BLOCK_N"])

    # ddd = torch.zeros([n_nodes * n_edges]).to(device)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(n_nodes * n_edges, BLOCK_M), triton.cdiv(batch_size, BLOCK_N), 1)

    ts = []
    for i in range(5):
        t0 = time.time()
        _forward_triton_kernel[grid](
            node_mars_ptr = node_mars, 
            element_mars_ptr = element_mars, 
            params_ptr = params,
            nids_ptr = nids, 
            cids_ptr = cids, 
            pids_ptr = pids,
            tot_n_nodes = tot_n_nodes,
            tot_n_eles = tot_n_eles,
            n_nodes = n_nodes,
            n_edges = n_edges,
            batch_size = batch_size,
            n_nodes_per_block_m = BLOCK_M // n_edges,
            BLOCK_M = BLOCK_M, 
            BLOCK_N = BLOCK_N
        )
        torch.cuda.synchronize()
        t1 = time.time()

        if i > 0:
            ts.append(t1 - t0)

    aveg_t, std_t = torch.tensor(ts).mean().item() * 1000, torch.tensor(ts).std().item() * 1000
    print(f"{aveg_t:.3f}±{std_t:.3f}ms")

    # node_mars_gt = node_mars.clone()
    # ch_mars = element_mars[cids]
    # maxval = ch_mars.max(dim = 1, keepdim = True).values
    # aaa = (((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)).sum(
    #     dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

    # bbb = node_mars[nids]

    # print(torch.max((aaa - bbb).abs()))


def main_blocksparse():

    GROUP_SIZE = 128

    data = np.load("temp.npz")

    device = torch.device("cuda:0")

    node_mars = torch.from_numpy(data["node_mars"]).permute(1, 0).contiguous().to(device)
    element_mars = torch.from_numpy(data["element_mars"]).permute(1, 0).contiguous().to(device)
    params = torch.from_numpy(data["params"]).to(device)

    # Convert `nids`, `cids`, and `pids` into block sparse format
    nids = torch.from_numpy(data["nids"]).to(device)
    cids = torch.from_numpy(data["cids"])
    pids = torch.from_numpy(data["pids"])

    cids = cids[:,::GROUP_SIZE].contiguous()
    pids = pids[:,::GROUP_SIZE].contiguous()

    cids_start = cids[:,0].contiguous().to(device)
    pids_start = pids[:,0].contiguous().to(device)
    cids_increment = torch.cat((cids[:,1:] - cids[:,:-1], cids[:,0:1] * 0), dim = 1).contiguous().to(device)
    pids_increment = torch.cat((pids[:,1:] - pids[:,:-1], pids[:,0:1] * 0), dim = 1).contiguous().to(device)

    tot_n_nodes = int(data["tot_n_nodes"])
    tot_n_eles = int(data["tot_n_eles"])
    layer_n_nodes = int(data["n_nodes"])
    layer_n_edges = int(data["n_edges"])
    batch_size = int(data["batch_size"])

    BLOCK_M = 16

    grid = (triton.cdiv(layer_n_nodes, BLOCK_M), batch_size)

    ddd = torch.zeros([layer_n_nodes, batch_size], dtype = torch.long, device = device)

    ts = []
    for i in range(5):
        t0 = time.time()
        block_sparse_kernel[grid](
            ddd,
            node_mars, 
            element_mars, 
            params, 
            nids, 
            cids_start, 
            cids_increment, 
            pids_start, 
            pids_increment,
            tot_n_nodes, 
            tot_n_eles, 
            layer_n_nodes, 
            layer_n_edge_groups = layer_n_edges // GROUP_SIZE, 
            batch_size = batch_size,
            BLOCK_M = BLOCK_M, 
            GROUP_SIZE = GROUP_SIZE
        )
        torch.cuda.synchronize()
        t1 = time.time()

        if i > 0:
            ts.append(t1 - t0)

    aveg_t, std_t = torch.tensor(ts).mean().item() * 1000, torch.tensor(ts).std().item() * 1000
    print(f"{aveg_t:.3f}±{std_t:.3f}ms")

    # import pdb; pdb.set_trace()


@triton.jit
def block_sparse_2d_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment, pids_start,
                           layer_n_edge_groups, batch_size, stride_pa, stride_pb, 
                           BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                           TILE_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    
    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node group id from `pid_m`
    ngroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
    ntile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

    # initialize pointers to `params`
    offs_node = tl.arange(0, TILE_SIZE_M)
    offs_edge = tl.arange(0, TILE_SIZE_K)
    par_start = tl.load(pids_start + ngroup_id * stride_pa + ntile_id * TILE_SIZE_M * stride_pb + offs_node * stride_pb)
    epars_ptr = params + par_start[:,None] + offs_edge[None,:]

    # initialize pointers to `element_mars`
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B 
    mask_batch = offs_batch < batch_size
    edge_start = tl.load(cids_start + ngroup_id * TILE_SIZE_K + offs_edge)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:]

    # Inner loop
    acc = tl.zeros((TILE_SIZE_M, BLOCK_B), dtype = tl.float32) - float("inf")

    cids_inc_ptr = cids_increment + ngroup_id * (layer_n_edge_groups * TILE_SIZE_K) + offs_edge
    for k in range(0, layer_n_edge_groups):
        epars = tl.load(epars_ptr)
        emars = tl.load(emars_ptr, mask = mask_batch[None,:])

        emars_max = tl.max(emars, axis = 0)[None,:]
        emars = tl.exp(emars - emars_max)
        epars = epars.to(tl.float16)
        emars = emars.to(tl.float16)
        nmars = tl.dot(epars, emars).to(tl.float32)

        # if TILE_SIZE_M < 16:
        #     epars = tl.view(tl.broadcast_to(epars[:,None,:], (TILE_SIZE_M, 16 // TILE_SIZE_M, TILE_SIZE_K)), (16, TILE_SIZE_K))
        #     nmars = tl.dot(epars, emars).to(tl.float32)
        #     nmars = tl.max(tl.view(nmars, (TILE_SIZE_M, 16 // TILE_SIZE_M, BLOCK_B)), axis = 1)

        acc = tl.where(emars_max > acc, 
            tl.log(nmars + tl.exp(acc - emars_max)) + emars_max,
            tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc
        )

        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[:,None] * batch_size
        cids_inc += TILE_SIZE_K

        epars_ptr += TILE_SIZE_K

    # Write back
    offs_nids = tl.load(nids + ngroup_id * GROUP_SIZE_M + ntile_id * TILE_SIZE_M + offs_node)
    offs_nmars = offs_nids[:,None] * batch_size + offs_batch[None,:]
    tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])


def main_blocksparse_2d():

    GROUP_SIZE_M = 32

    TILE_SIZE_M = 16
    TILE_SIZE_K = 64

    BLOCK_B = max(128, 16)

    data = np.load("temp.npz")

    device = torch.device("cuda:0")

    node_mars = torch.from_numpy(data["node_mars"]).to(device)
    element_mars = torch.from_numpy(data["element_mars"]).to(device)
    params = torch.from_numpy(data["params"]).to(device)

    # Convert `nids`, `cids`, and `pids` into block sparse format
    nids = torch.from_numpy(data["nids"])# .to(device)
    cids = torch.from_numpy(data["cids"])# .to(device)
    pids = torch.from_numpy(data["pids"])# .to(device)

    node_mars_gt = node_mars.clone()
    ch_mars = element_mars[cids]
    maxval = ch_mars.max(dim = 1, keepdim = True).values
    aaa = (((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)).sum(
        dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

    nids = nids.reshape(-1, GROUP_SIZE_M).contiguous().to(device)
    cids = cids[::GROUP_SIZE_M,:].reshape(nids.size(0), -1, TILE_SIZE_K).contiguous()
    pids_start = pids.reshape(nids.size(0), GROUP_SIZE_M, -1)[:,:,0].contiguous().to(device)

    cids_start = cids[:,0,:].contiguous().to(device)
    cids_increment = torch.cat((cids[:,1:,:] - cids[:,:-1,:], cids[:,0:1,:] * 0), dim = 1).contiguous().to(device)

    tot_n_nodes = int(data["tot_n_nodes"])
    tot_n_eles = int(data["tot_n_eles"])
    layer_n_nodes = int(data["n_nodes"])
    layer_n_edges = int(data["n_edges"])
    batch_size = int(data["batch_size"])

    layer_n_node_groups = layer_n_nodes // GROUP_SIZE_M
    layer_n_edge_groups = layer_n_edges // TILE_SIZE_K

    grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

    ts = []
    for i in range(50):
        # print("enter")
        t0 = time.time()
        block_sparse_2d_kernel[grid](
            node_mars, 
            element_mars, 
            params, 
            nids, 
            cids_start, 
            cids_increment, 
            pids_start,
            layer_n_edge_groups, 
            batch_size,
            stride_pa = pids_start.stride(0),
            stride_pb = pids_start.stride(1), # Do not provide pids.stride(2) since it is 1
            BLOCK_B = BLOCK_B,
            TILE_SIZE_K = TILE_SIZE_K,
            TILE_SIZE_M = TILE_SIZE_M,
            GROUP_SIZE_M = GROUP_SIZE_M
        )
        torch.cuda.synchronize()
        t1 = time.time()

        if i > 0:
            ts.append(t1 - t0)

    aveg_t, std_t = torch.tensor(ts).mean().item() * 1000, torch.tensor(ts).std().item() * 1000
    print(f"{aveg_t:.3f}±{std_t:.3f}ms")

    bbb = node_mars[nids]

    print(torch.max((aaa - bbb.flatten(0, 1)).abs()))

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    # main_baseline()
    # main_blocksparse()
    main_blocksparse_2d()