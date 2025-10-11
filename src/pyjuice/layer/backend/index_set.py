import torch
import triton
import triton.language as tl


@triton.jit
def _batched_index_set_kernel(target_tensor_ptr, ids_ptr, source_tensor_ptr, 
                              ids_size, batch_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ids_size * batch_size

    ids_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(ids_ptr + ids_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    source_vals = tl.load(source_tensor_ptr + offsets, mask = mask, other = 0)

    tl.store(target_tensor_ptr + n_offsets, source_vals, mask = mask)


def batched_index_set(target_tensor: torch.Tensor, ids: torch.Tensor, source_tensor: torch.Tensor,
                      BLOCK_SIZE: int = 1024):
    ids_size = ids.numel()
    batch_size = target_tensor.size(-1)

    grid = lambda meta: (triton.cdiv(ids_size * batch_size, meta['BLOCK_SIZE']),)
    _batched_index_set_kernel[grid](
        target_tensor_ptr = target_tensor, 
        ids_ptr = ids, 
        source_tensor_ptr = source_tensor,
        ids_size = ids_size, 
        batch_size = batch_size, 
        BLOCK_SIZE = BLOCK_SIZE
    )


@triton.jit
def _batched_index_cum_kernel(target_tensor_ptr, ids_ptr, source_tensor_ptr, 
                              ids_size, batch_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ids_size * batch_size

    ids_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(ids_ptr + ids_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    source_vals = tl.load(source_tensor_ptr + offsets, mask = mask, other = 0)
    target_vals = tl.load(target_tensor_ptr + n_offsets, mask = mask, other = 0)
    new_vals = target_vals + source_vals

    tl.store(target_tensor_ptr + n_offsets, new_vals, mask = mask)


def batched_index_cum(target_tensor: torch.Tensor, ids: torch.Tensor, source_tensor: torch.Tensor,
                      BLOCK_SIZE: int = 1024):
    ids_size = ids.numel()
    batch_size = target_tensor.size(-1)

    grid = lambda meta: (triton.cdiv(ids_size * batch_size, meta['BLOCK_SIZE']),)
    _batched_index_cum_kernel[grid](
        target_tensor_ptr = target_tensor, 
        ids_ptr = ids, 
        source_tensor_ptr = source_tensor,
        ids_size = ids_size, 
        batch_size = batch_size, 
        BLOCK_SIZE = BLOCK_SIZE
    )


@triton.jit
def _index_cum_kernel(target_tensor_ptr, ids_ptr, source_tensor_ptr, 
                      ids_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ids_size

    n_offsets = tl.load(ids_ptr + offsets, mask = mask, other = 0)

    source_vals = tl.load(source_tensor_ptr + offsets, mask = mask, other = 0)
    target_vals = tl.load(target_tensor_ptr + n_offsets, mask = mask, other = 0)
    new_vals = target_vals + source_vals

    tl.store(target_tensor_ptr + n_offsets, new_vals, mask = mask)


def index_cum(target_tensor: torch.Tensor, ids: torch.Tensor, source_tensor: torch.Tensor,
                      BLOCK_SIZE: int = 1024):
    ids_size = ids.numel()

    grid = lambda meta: (triton.cdiv(ids_size, meta['BLOCK_SIZE']),)
    _index_cum_kernel[grid](
        target_tensor_ptr = target_tensor, 
        ids_ptr = ids, 
        source_tensor_ptr = source_tensor,
        ids_size = ids_size, 
        BLOCK_SIZE = BLOCK_SIZE
    )