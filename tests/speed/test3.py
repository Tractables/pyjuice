import torch
import triton
import triton.language as tl
import time


@triton.jit
def _backward_kernel(fff_ptr, node_flows_ptr, element_flows_ptr, params_ptr, 
                     node_mars_ptr, element_mars_ptr, param_flows_ptr,
                     chids_ptr, parids_ptr, parpids_ptr,
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

    # Node mask for future reuse
    n_start = ne_start // n_edges
    n_offsets = n_start + tl.arange(0, n_nodes_per_block_m)
    n_mask = n_offsets < n_nodes

    # Reusable ids for index tensors
    par_offsets = tl.reshape(ne_offsets, (n_edges, n_nodes_per_block_m))
    par_mask = tl.broadcast_to(n_mask[None,:], (n_edges, n_nodes_per_block_m)) 
    bpar_mask = tl.broadcast_to(n_mask[None,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) & \
        tl.broadcast_to(b_mask[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))

    # Get node ids from `parids` and retrieve the corresponding node flows and node mars
    node_ids = tl.load(parids_ptr + par_offsets, mask = par_mask, other = 0)
    node_offsets = tl.broadcast_to(node_ids[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) * batch_size + \
        tl.broadcast_to(b_offsets[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
    nflows = tl.load(node_flows_ptr + node_offsets, mask = bpar_mask, other = 0) # node_flows[parids]
    nmars = tl.load(node_mars_ptr + node_offsets, mask = bpar_mask, other = 0) # node_mars[parids]

    # Get param ids from `parpids` and retrieve the corresponding node params
    eparam_ids = tl.load(parpids_ptr + par_offsets, mask = par_mask, other = 0)
    eparams = tl.load(params_ptr + eparam_ids, mask = par_mask, other = 0)
    eparams = tl.broadcast_to(eparams[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) # params[parpids]

    # Get element ids from `cids` and retrieve the corresponding element mars
    ele_ids = tl.load(chids_ptr + n_offsets, mask = n_mask, other = 0)
    ele_offsets = tl.broadcast_to(ele_ids[None,:], (BLOCK_N, n_nodes_per_block_m)) * batch_size + \
        tl.broadcast_to(b_offsets[:,None], (BLOCK_N, n_nodes_per_block_m))
    ele_mask = tl.broadcast_to(n_mask[None,:], (BLOCK_N, n_nodes_per_block_m))
    emars = tl.load(element_mars_ptr + ele_offsets, mask = ele_mask, other = 0) # element_mars[chids]
    emars = tl.broadcast_to(emars[:,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) # element_mars[chids].unsqueeze(1)

    # Compute edge flows
    eflows = nflows * eparams * tl.exp(emars - nmars)

    sss = tl.reshape(tl.arange(0, BLOCK_N * n_edges * n_nodes_per_block_m), (BLOCK_N, n_edges, n_nodes_per_block_m))
    tl.store(fff_ptr + sss, eflows)

    # Store to `element_flows[chids]`
    cum_eflows = tl.sum(eflows, axis = 1) # [BLOCK_N, n_nodes_per_block_m]
    tl.store(element_flows_ptr + ele_offsets, cum_eflows, mask = ele_mask)

    # Compute parameter flows
    parflows = tl.sum(eflows, axis = 0) # [n_edges, n_nodes_per_block_m]
    # Here the `eparam_ids > 0` term masks out dummy edges
    parflow_mask = (eparam_ids > 0) & tl.broadcast_to(n_mask[None,:], (n_edges, n_nodes_per_block_m))
    curr_parflows = tl.load(param_flows_ptr + eparam_ids, mask = parflow_mask, other = 0)
    tl.store(param_flows_ptr + eparam_ids, curr_parflows + parflows, mask = parflow_mask)


def _backward_triton(fff, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                     params: torch.Tensor, node_mars: torch.Tensor, 
                     element_mars: torch.Tensor, param_flows: torch.Tensor, 
                     chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor, 
                     BLOCK_SIZE = 2**14, MAX_BLOCK_M = 512, MAX_BLOCK_N = 512):
    """
    This function is equivalent to running:
    ``` 
    element_flows[chids] = (node_flows[parids] * params[parpids] * \
        (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

    param_flows[seq_parpids] += (node_flows[parids] * params[parpids] * \
        (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 2)[seq_ids0, seq_ids1]
    ```
    
    Parameters:
    `node_flows`:    Tensor[N, B]
    `element_flows`: Tensor[M, B]
    `params`:        Tensor[E]
    `node_mars`:     Tensor[N, B]
    `element_mars`:  Tensor[M, B]
    `param_flows`:   Tensor[E]
    `chids`:         Tensor[n]
    `parids`:        Tensor[n, p]
    `parpids`:       Tensor[n, p]
    """
    tot_n_nodes = node_mars.size(0)
    tot_n_eles = element_mars.size(0)
    n_nodes = chids.size(0)
    n_edges = parids.size(1)
    batch_size = node_mars.size(1)

    assert n_edges <= MAX_BLOCK_M, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
    assert params.dim() == 1, "Expecting a 1D `params`."

    MIN_BLOCK_M = min(triton.next_power_of_2(n_edges), MAX_BLOCK_M)
    BLOCK_N = min(BLOCK_SIZE // MIN_BLOCK_M, MAX_BLOCK_N, triton.next_power_of_2(batch_size))
    BLOCK_M = min(BLOCK_SIZE // BLOCK_N, MAX_BLOCK_M)

    grid = (triton.cdiv(n_nodes * n_edges, BLOCK_M), triton.cdiv(batch_size, BLOCK_N), 1)

    _backward_kernel[grid](
        fff_ptr = fff,
        node_flows_ptr = node_flows,
        element_flows_ptr = element_flows,
        params_ptr = params,
        node_mars_ptr = node_mars, 
        element_mars_ptr = element_mars,
        param_flows_ptr = param_flows, 
        chids_ptr = chids, 
        parids_ptr = parids, 
        parpids_ptr = parpids,
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

    torch.manual_seed(10)

    device = torch.device("cuda:0")

    # B = 13
    # N = 11
    # E = 4

    B = 523
    N = 34
    E = 32

    node_mars = torch.rand([1000, B]).to(device)
    element_mars = torch.rand([1200, B]).to(device)
    params = torch.rand([3000]).to(device)
    param_flows = torch.zeros([3000]).to(device)

    node_flows = torch.rand([1000, B]).to(device)
    element_flows = torch.rand([1200, B]).to(device)

    chids = torch.randperm(1200)[:N].to(device)
    parids = torch.randint(1, 1000, (N, E)).to(device)
    parpids = torch.randint(1, 3000, (N, E)).to(device)

    fff = torch.zeros([10000]).to(device)

    _backward_triton(fff, node_flows, element_flows, params, node_mars, 
                     element_mars, param_flows, chids, 
                     parids, parpids)

    # aaa = fff[:64*4*8].reshape(64, 4, 8)[:3,:4,:5]
    # nsns = (node_flows[parids] * params.unsqueeze(1)[parpids] * \
    #     (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp())
    # print(aaa)
    # print(nsns)

    aaaa = (node_flows[parids] * params.unsqueeze(1)[parpids] * \
        (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

    # bbbb = (node_flows[parids] * params.unsqueeze(1)[parpids] * \
    #     (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 2)[seq_ids0, seq_ids1]

    print(torch.abs(element_flows[chids] - aaaa).max())
