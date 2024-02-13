import torch
import triton
import triton.language as tl

from typing import Optional


@triton.jit
def _cum_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, num_param_blocks, block_size, batch_size, 
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr):

    b_pid = tl.program_id(axis = 0)
    k_pid = tl.program_id(axis = 1)
    m_pid = tl.program_id(axis = 2)

    m_offsets = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < num_param_blocks

    k_offsets = k_pid * BLOCK_K + tl.arange(0, BLOCK_K)

    b_offsets = b_pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offsets < batch_size

    n_offsets = tl.load(node_ids_ptr + m_offsets, mask = m_mask, other = 0)
    reuse_offs = k_offsets[None,:,None] * batch_size + b_offsets[None,None,:]

    n_offsets = n_offsets[:,None,None] * (batch_size * block_size) + reuse_offs
    p_offsets = m_offsets[:,None,None] * (batch_size * block_size) + reuse_offs

    mask = m_mask[:,None,None] & b_mask[None,None,:]
    params = tl.load(params_ptr + p_offsets, mask = mask, other = 0)

    tl.atomic_add(cum_params_ptr + n_offsets, params, mask = mask)


@triton.jit
def _norm_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, node_nchs_ptr, num_param_blocks, block_size, 
                        batch_size, pseudocount, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr):

    b_pid = tl.program_id(axis = 0)
    k_pid = tl.program_id(axis = 1)
    m_pid = tl.program_id(axis = 2)

    m_offsets = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < num_param_blocks

    k_offsets = k_pid * BLOCK_K + tl.arange(0, BLOCK_K)

    b_offsets = b_pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offsets < batch_size

    n_offsets = tl.load(node_ids_ptr + m_offsets, mask = m_mask, other = 0)
    reuse_offs = k_offsets[None,:,None] * batch_size + b_offsets[None,None,:]

    nb_offsets = n_offsets[:,None,None] * (batch_size * block_size) + reuse_offs
    p_offsets = m_offsets[:,None,None] * (batch_size * block_size) + reuse_offs

    mask = m_mask[:,None,None] & b_mask[None,None,:]
    params = tl.load(params_ptr + p_offsets, mask = mask, other = 0)
    cum_params = tl.load(cum_params_ptr + nb_offsets, mask = mask, other = 1)
    nchs = tl.load(node_nchs_ptr + n_offsets, mask = m_mask, other = 1)[:,None,None]
    
    normed_params = (params + pseudocount / nchs) / (cum_params + pseudocount)
    tl.store(params_ptr + p_offsets, normed_params, mask = mask)


def normalize_ns_parameters(params: torch.Tensor, node_ids: torch.Tensor, block_size: int, ch_block_size: int, 
                            node_nchs: Optional[torch.Tensor] = None, pseudocount: float = 0.0):

    assert 3 <= params.dim() <= 4 and params.size(1) == block_size and params.size(2) == ch_block_size

    num_param_blocks = params.size(0)
    num_node_blocks = torch.max(node_ids).detach().cpu().item() + 1

    if node_nchs is None:
        node_nchs = torch.bincount(node_ids) * ch_block_size

    if node_ids.is_cuda:
        assert params.is_cuda, "Input `params` should be on GPU."

        if params.dim() == 3:
            params = params.unsqueeze(3)

        batch_size = params.size(3)

        cum_params = torch.zeros([num_node_blocks, block_size, batch_size], dtype = torch.float32, device = params.device)

        blockified_params = params.sum(2).contiguous()

        BLOCK_B = min(batch_size, 128)
        BLOCK_K = min(1024 // BLOCK_B, triton.next_power_of_2(block_size))
        BLOCK_M = min(1024 // (BLOCK_B * BLOCK_K), triton.next_power_of_2(num_param_blocks))

        grid = lambda meta: (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(block_size, BLOCK_K), triton.cdiv(num_param_blocks, BLOCK_M))

        _cum_params_kernel[grid](blockified_params, cum_params, node_ids, num_param_blocks, block_size, batch_size, BLOCK_M, BLOCK_K, BLOCK_B)
        _norm_params_kernel[grid](blockified_params, cum_params, node_ids, node_nchs, num_param_blocks, block_size, batch_size, pseudocount, BLOCK_M, BLOCK_K, BLOCK_B)

        params *= (blockified_params / (params.sum(2) + 1e-12)).unsqueeze(2)

    else:
        assert params.dim() == 3, "CPU version of `normalize_parameters` does not support `batch_size > 1` for now."

        with torch.no_grad():

            params = params.float()

            blockified_params = params.sum(dim = 2).contiguous()

            param_ids = torch.arange(0, num_param_blocks, dtype = torch.long, device = params.device)

            cum_matrix1 = torch.sparse_coo_tensor(
                torch.stack((node_ids, param_ids), dim = 0), 
                torch.ones([num_param_blocks], device = params.device), 
                (num_node_blocks, num_param_blocks)
            )
            node_buffer = torch.sparse.mm(cum_matrix1, blockified_params) + pseudocount

            node_buffer.reciprocal_()
            node_buffer = node_buffer.reshape(num_node_blocks * block_size, 1)

            param_ids = torch.arange(0, num_param_blocks * block_size, dtype = torch.long, device = params.device)
            flattened_node_ids = (node_ids.unsqueeze(1).repeat(1, block_size) * block_size + torch.arange(0, block_size, device = params.device)).reshape(-1)

            cum_matrix2 = torch.sparse_coo_tensor(
                torch.stack((param_ids, flattened_node_ids), dim = 0), 
                (blockified_params + pseudocount / node_nchs[node_ids].unsqueeze(1)).reshape(-1), 
                (num_param_blocks * block_size, num_node_blocks * block_size)
            )
            params_buffer = torch.sparse.mm(cum_matrix2, node_buffer).reshape(num_param_blocks, block_size)
            
            params *= (params_buffer / (blockified_params + 1e-12)).unsqueeze(2)