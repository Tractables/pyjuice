import torch
import triton
import triton.language as tl

from typing import Optional


@triton.jit
def _aggregate_flows_kernel(param_flows_ptr, tied_param_flows_ptr, tied_param_ids_ptr, tied_param_group_ids_ptr,
                            num_params: tl.constexpr, batch_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_params * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    p_offsets = tl.load(tied_param_ids_ptr + param_offsets, mask = mask, other = 0)
    p_offsets = p_offsets * batch_size + batch_offsets

    g_offsets = tl.load(tied_param_group_ids_ptr + param_offsets, mask = mask, other = 0)
    g_offsets = g_offsets * batch_size + batch_offsets

    param_flows = tl.load(param_flows_ptr + p_offsets, mask = mask, other = 0)

    tl.atomic_add(tied_param_flows_ptr + g_offsets, param_flows, mask = mask)


@triton.jit
def _assign_flows_kernel(param_flows_ptr, tied_param_flows_ptr, tied_param_ids_ptr, tied_param_group_ids_ptr,
                         num_params: tl.constexpr, batch_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_params * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    p_offsets = tl.load(tied_param_ids_ptr + param_offsets, mask = mask, other = 0)
    p_offsets = p_offsets * batch_size + batch_offsets

    g_offsets = tl.load(tied_param_group_ids_ptr + param_offsets, mask = mask, other = 0)
    g_offsets = g_offsets * batch_size + batch_offsets

    gparam_flows = tl.load(tied_param_flows_ptr + g_offsets, mask = mask, other = 0)

    tl.store(param_flows_ptr + p_offsets, gparam_flows, mask = mask)


def tie_param_flows(param_flows: torch.Tensor, num_tied_params: int, 
                    tied_param_ids: torch.Tensor, tied_param_group_ids: torch.Tensor,
                    tied_param_flows: Optional[torch.Tensor] = None, BLOCK_SIZE: int = 1024):

    if param_flows.dim() == 1:
        param_flows = param_flows.unsqueeze(1)

    num_params = tied_param_ids.size(0)
    batch_size = param_flows.size(1)

    # Allocate buffer if not already
    if tied_param_flows is None:
        tied_param_flows = torch.zeros([num_tied_params, batch_size], device = param_flows.device)
    else:
        assert tied_param_flows.size(0) == num_tied_params and tied_param_flows.size(1) == batch_size, "Size of `tied_param_flows` is incorrect."
        tied_param_flows = tied_param_flows[:,:]

    if param_flows.is_cuda:
        assert tied_param_flows.is_cuda and tied_param_ids.is_cuda and tied_param_group_ids.is_cuda

        grid = lambda meta: (triton.cdiv(num_params * batch_size, meta['BLOCK_SIZE']),)

        _aggregate_flows_kernel[grid](
            param_flows_ptr = param_flows, 
            tied_param_flows_ptr = tied_param_flows,
            tied_param_ids_ptr = tied_param_ids, 
            tied_param_group_ids_ptr = tied_param_group_ids,
            num_params = num_params,
            batch_size = batch_size,
            BLOCK_SIZE = BLOCK_SIZE
        )

        grid = lambda meta: (triton.cdiv(num_params * batch_size, meta['BLOCK_SIZE']),)

        _assign_flows_kernel[grid](
            param_flows_ptr = param_flows, 
            tied_param_flows_ptr = tied_param_flows,
            tied_param_ids_ptr = tied_param_ids, 
            tied_param_group_ids_ptr = tied_param_group_ids,
            num_params = num_params,
            batch_size = batch_size,
            BLOCK_SIZE = BLOCK_SIZE
        )

    else:
        cum_matrix = torch.sparse_coo_tensor(
            torch.stack((tied_param_group_ids, tied_param_ids), dim = 0), 
            torch.ones([num_params], dtype = torch.float32, device = param_flows.device), 
            (num_tied_params, param_flows.size(0))
        )
        par_group_buffer = torch.sparse.mm(cum_matrix, param_flows) # [num_tied_params, B]

        param_flows[tied_param_ids] = par_group_buffer[tied_param_group_ids]

    return None