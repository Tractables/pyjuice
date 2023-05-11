import torch
import triton
import triton.language as tl
import time


@triton.jit
def _forward_kernel(node_mars_ptr, element_mars_ptr, params_ptr, 
                    nids_ptr, cids_ptr, pids_ptr,
                    tot_n_nodes: tl.constexpr, tot_n_eles: tl.constexpr,
                    tot_n_pars: tl.constexpr, 
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
    cid_mask = tl.broadcast_to(nid_mask[None,:], (n_edges, n_nodes_per_block_m))
    ch_ids = tl.load(cids_ptr + cid_offsets, mask = cid_mask, other = 0)

    # Use `ch_ids` to retrieve the corresponding element mars
    ele_offsets = tl.broadcast_to(ch_ids[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) * batch_size + \
        tl.broadcast_to(b_offsets[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
    ele_mask = tl.broadcast_to(nid_mask[None,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) & \
        tl.broadcast_to(b_mask[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
    ch_logps = tl.load(element_mars_ptr + ele_offsets, mask = ele_mask, other = 0) # `element_mars[cids]`

    # Take the max of the child mars
    ch_max_logp = tl.max(ch_logps, axis = 1) # `maxval`

    # Subtract the max from child mars
    ch_logps_sub_max = ch_logps - tl.broadcast_to(ch_max_logp[:,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m))

    # Take exp
    ch_ps_sub_max = tl.exp(ch_logps_sub_max)

    # Get param ids from `pids`
    # Here we reuse `cid_offsets` and `cid_mask` thank to their similar structure
    par_ids = tl.load(pids_ptr + cid_offsets, mask = cid_mask, other = 0)

    # Use `par_ids` to retrieve the corresponding parameters
    par_mask = tl.broadcast_to(nid_mask[None,:], (n_edges, n_nodes_per_block_m))
    ch_pars = tl.load(params_ptr + par_ids, mask = par_mask, other = 0) # `params[pids]`
    ch_pars = tl.broadcast_to(ch_pars[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m))

    # Sum node marginals (unnormalized)
    n_ps = tl.sum(ch_ps_sub_max * ch_pars, axis = 1)

    # Take log and subtract max vals
    n_logps = tl.log(tl.maximum(n_ps, 1e-10)) + ch_max_logp

    # Read out the target indices for `node_mars`
    nmar_offsets = tl.broadcast_to(n_ids[None,:], (BLOCK_N, n_nodes_per_block_m)) * batch_size + \
        tl.broadcast_to(b_offsets[:,None], (BLOCK_N, n_nodes_per_block_m))
    nmar_mask = tl.broadcast_to(nid_mask[None,:], (BLOCK_N, n_nodes_per_block_m)) & \
        tl.broadcast_to(b_mask[:,None], (BLOCK_N, n_nodes_per_block_m))
    
    tl.store(node_mars_ptr + nmar_offsets, n_logps, mask = nmar_mask)


def _forward_triton(node_mars: torch.Tensor, element_mars: torch.Tensor, 
                    params: torch.Tensor,
                    nids: torch.Tensor, cids: torch.Tensor,
                    pids: torch.Tensor, MAX_BLOCK_M = 256, MAX_BLOCK_N = 64):
    """
    This function is equivalent to running:
    ``` 
    ch_mars = element_mars[cids]
    maxval = ch_mars.max(dim = 1, keepdim = True).values
    node_mars[nids] = (((ch_mars - maxval).exp() * params[pids]).sum(
        dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)
    ```
    
    Parameters:
    `node_mars`:    Tensor[N, B]
    `element_mars`: Tensor[M, B]
    `params`:       Tensor[E]
    `nids`:         Tensor[n]
    `cids`:         Tensor[n, c]
    `pids`:         Tensor[n, c]
    """
    tot_n_nodes = node_mars.size(0)
    tot_n_eles = element_mars.size(0)
    tot_n_pars = params.size(0)
    n_nodes = nids.size(0)
    n_edges = cids.size(1)
    batch_size = node_mars.size(1)

    assert n_edges <= MAX_BLOCK_M, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
    assert params.dim() == 1, "Expecting a 1D `params`."

    BLOCK_M = MAX_BLOCK_M
    BLOCK_N = triton.next_power_of_2(min(MAX_BLOCK_N, batch_size))

    grid = (triton.cdiv(n_nodes * n_edges, BLOCK_M), triton.cdiv(batch_size, BLOCK_N), 1)

    _forward_kernel[grid](
        node_mars_ptr = node_mars, 
        element_mars_ptr = element_mars, 
        params_ptr = params,
        nids_ptr = nids, 
        cids_ptr = cids, 
        pids_ptr = pids,
        tot_n_nodes = tot_n_nodes,
        tot_n_eles = tot_n_eles,
        tot_n_pars = tot_n_pars,
        n_nodes = n_nodes, 
        n_edges = n_edges, 
        batch_size = batch_size, 
        n_nodes_per_block_m = BLOCK_M // n_edges,
        BLOCK_M = BLOCK_M, 
        BLOCK_N = BLOCK_N
    )

    return None


@torch.compile(mode = "reduce-overhead")
def forward_compile(node_mars: torch.Tensor, element_mars: torch.Tensor, 
                    params: torch.Tensor,
                    nids: torch.Tensor, cids: torch.Tensor,
                    pids: torch.Tensor):
    ch_mars = element_mars[cids]
    maxval = ch_mars.max(dim = 1, keepdim = True).values
    node_mars[nids] = (((ch_mars - maxval).exp() * params.unsqueeze(1)[pids]).sum(
        dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

    return None


if __name__ == "__main__":

    torch.manual_seed(10)

    device = torch.device("cuda:0")

    B = 543
    N = 845
    E = 32

    node_mars = torch.rand([1000, B]).to(device)
    element_mars = torch.rand([1200, B]).to(device)
    params = torch.rand([3000]).to(device)

    nids = torch.randperm(1000)[:N].to(device)
    # nids = torch.randint(0, 100, (N,)).to(device)
    cids = torch.randint(0, 1200, (N, E)).to(device)
    pids = torch.randint(0, 3000, (N, E)).to(device)

    fff = torch.zeros([10000]).to(device)

    _forward_triton(node_mars, element_mars, params, nids, cids, pids)

    ch_mars = element_mars[cids]
    maxval = ch_mars.max(dim = 1, keepdim = True).values
    nmars = (((ch_mars - maxval).exp() * params.unsqueeze(1)[pids]).sum(
        dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

    # aaa = fff[:64*4*16].reshape(64, 4, 16)[:11,:4,:13]
    # aaa = fff[:64*16].reshape(64, 16)[:11,:13]
    # print(aaa)
    # print(nids * B)
    # print(fff[:64*16].reshape(64, 16)[:12,:14].long())
    # print(torch.abs(node_mars.reshape(-1)[aaa.long()] - node_mars[nids]).max(), "e")
    # print(torch.abs(aaa - nmars).max())

    print(torch.abs(nmars - node_mars[nids]).max())

    '''node_mars = torch.rand([5000000, B]).to(device)
    element_mars = torch.rand([700000, B]).to(device)
    params = torch.rand([6567800]).to(device)

    nids = torch.randint(0, 5000000, (N,)).to(device)
    cids = torch.randint(0, 700000, (N, E)).to(device)
    pids = torch.randint(0, 6567800, (N, E)).to(device)

    _forward_triton(node_mars, element_mars, params, nids, cids, pids)

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(1000):
        _forward_triton(node_mars, element_mars, params, nids, cids, pids)
    torch.cuda.synchronize()
    t2 = time.time()
    print((t2 - t1) / 1000)
    # 0.008s

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(1000):
        forward_compile(node_mars, element_mars, params, nids, cids, pids)
    torch.cuda.synchronize()
    t2 = time.time()
    print((t2 - t1) / 1000)
    # 0.03s

    ch_mars = element_mars[cids]
    maxval = ch_mars.max(dim = 1, keepdim = True).values
    nmars = (((ch_mars - maxval).exp() * params.unsqueeze(1)[pids]).sum(
        dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

    print(torch.abs(nmars - node_mars[nids]).max())'''