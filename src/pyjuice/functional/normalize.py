import torch
import triton
import triton.language as tl

from typing import Optional


@triton.jit
def _cum_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, tot_num_params, batch_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_params * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    params = tl.load(params_ptr + offsets, mask = mask, other = 0)

    tl.atomic_add(cum_params_ptr + n_offsets, params, mask = mask)


@triton.jit
def _norm_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, node_nchs_ptr, tot_num_params, 
                        batch_size, pseudocount, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_params * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    params = tl.load(params_ptr + offsets, mask = mask, other = 0)
    cum_params = tl.load(cum_params_ptr + n_offsets, mask = mask, other = 1)
    nchs = tl.load(node_nchs_ptr + n_offsets, mask = mask, other = 1)

    normed_params = (params + pseudocount / nchs) / (cum_params + pseudocount)
    tl.store(params_ptr + offsets, normed_params, mask = mask)


def normalize_parameters(params: torch.Tensor, node_ids: torch.Tensor, node_nchs: Optional[torch.Tensor] = None, pseudocount: float = 0.0):

    num_params = params.size(0)
    num_nodes = torch.max(node_ids).detach().cpu().item() + 1

    if node_nchs is None:
        node_nchs = torch.bincount(node_ids)

    if node_ids.is_cuda:
        assert params.is_cuda, "Input `params` should be on GPU."

        if params.dim() == 1:
            params = params.unsqueeze(1)

        batch_size = params.size(1)

        cum_params = torch.zeros([num_nodes, batch_size], dtype = torch.float32, device = params.device)

        grid1 = lambda meta: (triton.cdiv(num_params * batch_size, meta['BLOCK_SIZE']),)
        grid2 = lambda meta: (triton.cdiv(num_params * batch_size, meta['BLOCK_SIZE']),)

        _cum_params_kernel[grid1](params, cum_params, node_ids, num_params, batch_size, BLOCK_SIZE = 1024)
        _norm_params_kernel[grid2](params, cum_params, node_ids, node_nchs, num_params, batch_size, pseudocount, BLOCK_SIZE = 1024)

    else:
        assert params.dim() == 1, "CPU version of `normalize_parameters` does not support `batch_size > 1` for now."

        with torch.no_grad():

            params = params.float()

            param_ids = torch.arange(0, num_params, dtype = torch.long, device = params.device)

            cum_matrix1 = torch.sparse_coo_tensor(
                torch.stack((node_ids, param_ids), dim = 0), 
                params, (num_nodes, num_params)
            )
            node_buffer = torch.sparse.mm(cum_matrix1, torch.ones([num_params, 1], dtype = torch.float32, device = params.device)) + pseudocount

            node_buffer.reciprocal_()

            cum_matrix2 = torch.sparse_coo_tensor(
                torch.stack((param_ids, node_ids), dim = 0), 
                params + pseudocount / node_nchs[node_ids], (num_params, num_nodes)
            )
            params_buffer = torch.sparse.mm(cum_matrix2, node_buffer)
            params.data[:] = params_buffer[:,0]