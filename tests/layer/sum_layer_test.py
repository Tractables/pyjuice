import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer, ProdLayer, SumLayer

import pytest

import triton
import triton.language as tl


@triton.jit
def _bk_triton_block_sparse_kernel(node_flows, element_flows, node_mars, element_mars, params, nids, cids_start, cids_increment,
                                    pids_start, pids_increment, local_ids, batch_size, partial_eval: tl.constexpr,
                                    BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                    TILE_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node group id from `pid_m`
    ngroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

    # Get the real node group id in the case of partial evaluation
    if partial_eval == 1:
        ngroup_id = tl.load(local_ids + ngroup_id)

    # Initialize pointers to `params`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    par_start = tl.load(pids_start + ngroup_id * TILE_SIZE_K + offs_edge)
    epars_ptr = params + \
        offs_node[:,None] + \
        par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    edge_start = tl.load(cids_start + ngroup_id * TILE_SIZE_K + offs_edge)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    eflows_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Initialize pointers to `node_flows`
    off_nids = tl.load(nids + ngroup_id)
    offs_nmfs = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nmars = tl.load(node_mars + offs_nmfs, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]
    nflows = tl.load(node_flows + offs_nmfs, mask = mask_batch[None,:])

    nmars_max = tl.max(nmars, axis = 0)
    nflows_div_mars = nflows / tl.exp(nmars - nmars_max[None,:])
    nflows_div_mars = nflows_div_mars.to(tl.float16)

    # Batch increment pointers
    pids_inc_ptr = pids_increment + ngroup_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + ngroup_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        epars = epars.to(tl.float16)
        aaa = tl.dot(tl.trans(epars), nflows_div_mars).to(tl.float32)
        bbb = aaa * (emars - nmars_max[None,:])

        tl.atomic_add(eflows_ptr, bbb, mask = mask_batch[None,:])
        # acc += bbb

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment `emars_ptr`
        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[:,None] * batch_size
        eflows_ptr += cids_inc[:,None] * batch_size
        cids_inc_ptr += TILE_SIZE_K


@triton.jit
def _bkp_triton_block_sparse_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids,
                                    local_ids, batch_size: tl.constexpr, n_edges: tl.constexpr, partial_eval: tl.constexpr,
                                    TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, TILE_SIZE_K: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                    GROUP_SIZE_M: tl.constexpr):

    pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node group id from `pid_m`
    ngroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

    # Get the real node group id in the case of partial evaluation
    if partial_eval == 1:
        ngroup_id = tl.load(local_ids + ngroup_id)

    # Batch offsets and mask
    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars`
    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + ngroup_id * n_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, TILE_SIZE_B]

    # Initialize pointers to `node_flows` and `node_mars`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + ngroup_id)
    offs_nmfs = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nmars_ptr = node_mars + offs_nmfs
    nflows_ptr = node_flows + offs_nmfs

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, TILE_SIZE_B]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, TILE_SIZE_B]
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, TILE_SIZE_B]

        nmars_max = tl.max(nmars, axis = 0)
        nflows_div_mars = nflows / tl.exp(nmars - nmars_max[None,:])
        nflows_div_mars = nflows_div_mars.to(tl.float16)

        emars = tl.exp(emars - nmars_max[None,:])
        emars = emars.to(tl.float16)

        pflows = tl.dot(nflows_div_mars, tl.trans(emars)).to(tl.float32)

        acc += pflows

        # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
        emars_ptr += TILE_SIZE_B
        nmars_ptr += TILE_SIZE_B
        nflows_ptr += TILE_SIZE_B

        # Update batch mask
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    par_start = tl.load(pids + ngroup_id * n_edges + offs_edge)
    epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
    
    epars = tl.load(params + epars_offsets)
    pflows = acc * epars

    tl.store(param_flows + epars_offsets, pflows)


def sum_layer_test():

    device = torch.device("cuda:0")

    group_size = 16
    batch_size = 16
    
    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)
        np2 = multiply(ni1, ni2)

        ns0 = summate(np0, num_node_groups = 2)
        ns1 = summate(np1, num_node_groups = 2)
        ns2 = summate(np2, num_node_groups = 2)

    input_layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = group_size)

    prod_layer = ProdLayer([np0, np1, np2])

    layer = SumLayer([ns0, ns1, ns2], global_nid_start = group_size,
                     param_ends = [1], tied_param_ids = [],
                     tied_param_group_ids = [], tied_param_ends = [],
                     ch_prod_layer_size = prod_layer.num_nodes + group_size)

    assert torch.all(layer.partitioned_nids[0] == torch.arange(group_size, 7 * group_size, group_size))
    assert torch.all(layer.partitioned_cids[0][0:2,0] == group_size)
    assert torch.all(layer.partitioned_cids[0][2:4,0] == 3 * group_size)
    assert torch.all(layer.partitioned_cids[0][4:6,0] == 5 * group_size)
    assert torch.all(layer.partitioned_cids[0][0:2,1] == group_size + 1)
    assert torch.all(layer.partitioned_cids[0][2:4,1] == 3 * group_size + 1)
    assert torch.all(layer.partitioned_cids[0][4:6,1] == 5 * group_size + 1)
    assert torch.all(layer.partitioned_pids[0][:,0] == torch.arange(group_size, (group_size * 2 * 6 + 1) * group_size, 2 * group_size * group_size) - group_size + 1)
    assert torch.all(layer.partitioned_pids[0][:,1] == torch.arange(group_size, (group_size * 2 * 6 + 1) * group_size, 2 * group_size * group_size) + 1)

    assert torch.all(layer.partitioned_chids[0] == torch.arange(group_size, 7 * group_size, group_size))
    assert torch.all(layer.partitioned_parids[0][0:2,0] == group_size)
    assert torch.all(layer.partitioned_parids[0][0:2,1] == 2 * group_size)
    assert torch.all(layer.partitioned_parids[0][2:4,0] == 3 * group_size)
    assert torch.all(layer.partitioned_parids[0][2:4,1] == 4 * group_size)
    assert torch.all(layer.partitioned_parids[0][4:6,0] == 5 * group_size)
    assert torch.all(layer.partitioned_parids[0][4:6,1] == 6 * group_size)
    assert torch.all(layer.partitioned_parpids[0][0,0] == 1)
    assert torch.all(layer.partitioned_parpids[0][1,0] == 1 + group_size**2)
    assert torch.all(layer.partitioned_parpids[0][0,1] == 1 + 2 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][1,1] == 1 + 3 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][2,0] == 1 + 4 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][3,0] == 1 + 5 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][2,1] == 1 + 6 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][3,1] == 1 + 7 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][4,0] == 1 + 8 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][5,0] == 1 + 9 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][4,1] == 1 + 10 * group_size**2)
    assert torch.all(layer.partitioned_parpids[0][5,1] == 1 + 11 * group_size**2)

    layer.to(device)

    ## Forward tests ##

    element_mars = torch.rand([group_size + 3 * 2 * 2 * group_size, batch_size]).log().to(device)
    element_mars[:group_size,:] = -float("inf")
    node_mars = torch.zeros([group_size + group_size * 2 * 3, batch_size]).to(device)

    params = torch.rand([1 + 3 * 4 * group_size * group_size]).to(device)

    layer(node_mars, element_mars, params)

    for i in range(group_size):
        for j in range(6):
            cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
            epars = params[layer.partitioned_pids[0][j,:]+i]
            assert torch.all(torch.abs(node_mars[(j+1)*group_size+i,:] - (epars[:,None] * cmars).sum(dim = 0).log()) < 1e-3)

    ## Backward tests ##

    node_flows = torch.rand([group_size + group_size * 2 * 3, batch_size]).to(device)
    element_flows = torch.zeros([group_size + 3 * 2 * 2 * group_size, batch_size]).to(device)

    param_flows = torch.zeros([1 + 3 * 4 * group_size * group_size]).to(device)

    layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows)

    chids = layer.partitioned_chids[0]
    parids = layer.partitioned_parids[0]
    parpids = layer.partitioned_parpids[0]

    num_ngroups = chids.size(0)
    num_egroups = parids.size(1)
    parids = (parids[:,:,None].repeat(1, 1, group_size) + torch.arange(0, group_size, device = parids.device)).reshape(num_ngroups, num_egroups * group_size)
    parpids = (parpids[:,:,None] + torch.arange(0, group_size * group_size, group_size, device = parids.device)).reshape(
        num_ngroups, num_egroups * group_size)

    for i in range(group_size):
        for j in range(6):
            nmars = node_mars[parids[j,:]].exp()
            nflows = node_flows[parids[j,:]]
            emars = element_mars[(j+1)*group_size+i,:].exp()
            epars = params[parpids[j,:]+i]
            eflows = (nflows * epars[:,None] * emars[None,:] / nmars).sum(dim = 0)

            import pdb; pdb.set_trace()

            assert torch.all(torch.abs(eflows - element_flows[(j+1)*group_size+i,:]) < 1e-3)
            
    import pdb; pdb.set_trace()


def speed_test():

    device = torch.device("cuda:0")

    group_size = 32
    num_vars = 28*28
    num_node_groups = 256 // group_size
    num_prod_nodes = 200

    batch_size = 512

    with juice.set_group_size(group_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(v, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 64)))

        nps = []
        for i in range(num_prod_nodes):
            v1 = random.randint(0, num_vars - 1)
            v2 = random.randint(0, num_vars - 1)
            if v1 == v2:
                if v1 == num_vars - 1:
                    v1 -= 2
                v2 = v1 + 1

            nps.append(multiply(nis[v1], nis[v2]))

        nodes = [summate(np, num_node_groups = num_node_groups) for np in nps]

    input_layer = InputLayer(nis, cum_nodes = group_size)

    prod_layer = ProdLayer(nps, layer_sparsity_tol = 0.1)

    layer = SumLayer(nodes, global_nid_start = group_size,
                         param_ends = [1], tied_param_ids = [],
                         tied_param_group_ids = [], tied_param_ends = [],
                         ch_prod_layer_size = prod_layer.num_nodes + group_size)

    # import pdb; pdb.set_trace()

    layer.to(device)

    node_mars = torch.zeros([group_size + group_size * num_node_groups * num_prod_nodes, batch_size]).to(device)
    element_mars = torch.rand([group_size + num_prod_nodes * group_size * num_node_groups, batch_size]).log().to(device)
    params = torch.rand([layer.partitioned_pids[0].max() + group_size]).to(device)

    # import pdb; pdb.set_trace()

    ## Forward tests ##

    layer(node_mars, element_mars, params)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(node_mars, element_mars, params)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Forward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 11.255ms.")
    print("--------------------------------------------------------------")

    # exit()

    node_flows = torch.rand([group_size + group_size * num_node_groups * num_prod_nodes, batch_size]).to(device)
    element_flows = torch.zeros([group_size + num_prod_nodes * group_size * num_node_groups, batch_size]).log().to(device)
    param_flows = torch.zeros([layer.partitioned_pids[0].max() + group_size]).to(device)

    layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows)
    torch.cuda.synchronize()
    t1 = time.time()
    backward_ms = (t1 - t0) / 100 * 1000

    print(f"Backward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 11.255ms.")
    print("--------------------------------------------------------------")

    exit()

    # import pdb; pdb.set_trace()

    nids = layer.partitioned_nids[0]
    cids_start, cids_increment, pids_start, pids_increment = layer._cached_fw_pcids[("block_sparse", 0, 64)]

    BLOCK_B = 128
    TILE_SIZE_K = 64
    K_NUM_TILES = layer.partitioned_cids[0].size(1) // TILE_SIZE_K
    TILE_SIZE_M = 32

    layer_n_nodes = nids.size(0) * layer.group_size
    
    grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))
        
    _bk_triton_block_sparse_kernel[grid](
        node_flows,
        element_flows,
        node_mars, 
        element_mars, 
        params, 
        nids, 
        cids_start,
        cids_increment, 
        pids_start,
        pids_increment,
        local_ids = None,
        batch_size = batch_size,
        partial_eval = 0,
        BLOCK_B = BLOCK_B,
        TILE_SIZE_K = TILE_SIZE_K,
        K_NUM_TILES = K_NUM_TILES,
        TILE_SIZE_M = TILE_SIZE_M,
        GROUP_SIZE_M = layer.group_size
    )

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        _bk_triton_block_sparse_kernel[grid](
            node_flows,
            element_flows,
            node_mars, 
            element_mars, 
            params, 
            nids, 
            cids_start,
            cids_increment, 
            pids_start,
            pids_increment,
            local_ids = None,
            batch_size = batch_size,
            partial_eval = 0,
            BLOCK_B = BLOCK_B,
            TILE_SIZE_K = TILE_SIZE_K,
            K_NUM_TILES = K_NUM_TILES,
            TILE_SIZE_M = TILE_SIZE_M,
            GROUP_SIZE_M = layer.group_size
        )
    torch.cuda.synchronize()
    t1 = time.time()
    backward_ms = (t1 - t0) / 100 * 1000

    print(f"bkbk: {backward_ms:.3f}ms.")

    nids = layer.partitioned_nids[0]
    cids = layer.partitioned_cids[0]
    pids = layer.partitioned_pids[0]

    param_flows = torch.zeros(params.size()).to(device)

    TILE_SIZE_B = 64
    TILE_SIZE_K = 64
    B_NUM_TILES = triton.cdiv(batch_size, TILE_SIZE_B)
    TILE_SIZE_M = 32

    n_edges = cids.size(1)

    layer_n_nodes = nids.size(0) * layer.group_size
    
    grid = (triton.cdiv(n_edges, TILE_SIZE_K), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

    # print("aaa")

    _bkp_triton_block_sparse_kernel[grid](
        node_flows, node_mars, element_mars, params, 
        param_flows, nids, cids, pids, local_ids = None, 
        batch_size = batch_size, n_edges = n_edges, partial_eval = 0,
        TILE_SIZE_B = TILE_SIZE_B, B_NUM_TILES = B_NUM_TILES, 
        TILE_SIZE_K = TILE_SIZE_K, TILE_SIZE_M = TILE_SIZE_M, 
        GROUP_SIZE_M = layer.group_size
    )

    t0 = time.time()
    # print("bbb")
    torch.cuda.synchronize()
    # print("ccc")
    for _ in range(100):
        _bkp_triton_block_sparse_kernel[grid](
            node_flows, node_mars, element_mars, params, 
            param_flows, nids, cids, pids, local_ids = None, 
            batch_size = batch_size, n_edges = n_edges, partial_eval = 0,
            TILE_SIZE_B = TILE_SIZE_B, B_NUM_TILES = B_NUM_TILES, 
            TILE_SIZE_K = TILE_SIZE_K, TILE_SIZE_M = TILE_SIZE_M, 
            GROUP_SIZE_M = layer.group_size
        )
        # print("ddd")
    torch.cuda.synchronize()
    t1 = time.time()
    backward_ms = (t1 - t0) / 100 * 1000

    # print("eee")

    print(f"bkpbkp: {backward_ms:.3f}ms.")


if __name__ == "__main__":
    torch.manual_seed(3890)
    # sum_layer_test()
    speed_test()