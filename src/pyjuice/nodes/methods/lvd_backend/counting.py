import torch
import triton
import triton.language as tl

from typing import Optional


@triton.jit
def _pairwise_count_kernel(data1_ptr, data2_ptr, pairwise_count_ptr, num_samples: tl.constexpr,
                           n_cls1: tl.constexpr, n_cls2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_samples

    cid1 = tl.load(data1_ptr + offsets, mask = mask, other = 0)
    cid2 = tl.load(data2_ptr + offsets, mask = mask, other = 0)
    cid = cid1 * n_cls2 + cid2

    tl.atomic_add(pairwise_count_ptr + cid, 1, mask = mask)


def get_pairwise_count(data1: torch.Tensor, data2: torch.Tensor, n_cls1: torch.Tensor, n_cls2: torch.Tensor, 
                       device: Optional[torch.device] = None):
    assert data1.min() >= 0 and data1.max() < n_cls1, f"Value range of `data1` exceeds limit: [Min: {data1.min().item()}, Max: {data1.max().item()}]."
    assert data2.min() >= 0 and data2.max() < n_cls2, f"Value range of `data2` exceeds limit: [Min: {data2.min().item()}, Max: {data2.max().item()}]."
    assert data1.size(0) == data2.size(0), "`data1` and `data2` must have the same number of examples."

    if device is not None:
        data1 = data1.to(device)
        data2 = data2.to(device)

    if data1.is_cuda:

        num_samples = data1.size(0)
        pairwise_count = torch.zeros([n_cls1, n_cls2], dtype = torch.long, device = data1.device)

        grid = lambda meta: (triton.cdiv(num_samples, meta['BLOCK_SIZE']),)

        _pairwise_count_kernel[grid](
            data1_ptr = data1, 
            data2_ptr = data2,
            pairwise_count_ptr = pairwise_count,
            num_samples = num_samples,
            n_cls1 = n_cls1,
            n_cls2 = n_cls2,
            BLOCK_SIZE = BLOCK_SIZE
        )

    else:
        pairwise_count = torch.bincount(data1 * n_cls2 + data2, minlength = n_cls1 * n_cls2)
        pairwise_count = pairwise_count.reshape(n_cls1, n_cls2)

    return pairwise_count