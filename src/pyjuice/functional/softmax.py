import torch
import triton
import triton.language as tl


@triton.jit
def _fw_cum_logits_kernel(logits_ptr, cum_weights_ptr, node_ids_ptr, tot_num_logits, batch_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_logits * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    logits = tl.load(logits_ptr + offsets, mask = mask, other = 0)
    logits = tl.exp(logits)

    tl.atomic_add(cum_weights_ptr + n_offsets, logits, mask = mask)


@triton.jit
def _fw_norm_logits_kernel(logits_ptr, targets_ptr, cum_weights_ptr, node_ids_ptr, tot_num_logits, 
                           batch_size, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_logits * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    logits = tl.load(logits_ptr + offsets, mask = mask, other = 0)
    cum_weights = tl.load(cum_weights_ptr + n_offsets, mask = mask, other = 1)

    normed_logits = tl.exp(logits) / cum_weights
    tl.store(targets_ptr + offsets, normed_logits, mask = mask)


@triton.jit
def _bp_cum_logits_kernel(grads_ptr, normed_values_ptr, cum_grads_ptr, node_ids_ptr, tot_num_logits, batch_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_logits * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    grads = tl.load(grads_ptr + offsets, mask = mask, other = 0)
    normed_values = tl.load(normed_values_ptr + offsets, mask = mask, other = 0)
    cum_grads = grads * normed_values

    tl.atomic_add(cum_grads_ptr + n_offsets, cum_grads, mask = mask)


@triton.jit
def _bp_norm_grads_p_kernel(grads_ptr, targets_ptr, normed_values_ptr, cum_grads_ptr, node_ids_ptr, tot_num_logits, 
                            batch_size, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_logits * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    grads = tl.load(grads_ptr + offsets, mask = mask, other = 0)
    normed_values = tl.load(normed_values_ptr + offsets, mask = mask, other = 0)
    cum_grads = tl.load(cum_grads_ptr + n_offsets, mask = mask, other = 1)

    grads = normed_values * (grads - cum_grads)
    tl.store(targets_ptr + offsets, grads, mask = mask)


@triton.jit
def _bp_norm_grads_logp_kernel(grads_ptr, targets_ptr, cum_grads_ptr, node_ids_ptr, tot_num_logits, 
                               batch_size, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tot_num_logits * batch_size

    param_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(node_ids_ptr + param_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    grads = tl.load(grads_ptr + offsets, mask = mask, other = 0)
    cum_grads = tl.load(cum_grads_ptr + n_offsets, mask = mask, other = 1)

    grads = grads - cum_grads
    tl.store(targets_ptr + offsets, grads, mask = mask)


def flat_softmax_fw(logits: torch.Tensor, node_ids: torch.Tensor, inplace: bool = False):

    num_logits = logits.size(0)
    num_nodes = torch.max(node_ids).detach().cpu().item() + 1

    if inplace:
        targets = logits
    else:
        targets = torch.empty_like(logits)

    assert logits.is_cuda, "Input `logits` should be on GPU."

    if logits.dim() == 1:
        logits = logits.unsqueeze(1)

    batch_size = logits.size(1)

    cum_weights = torch.zeros([num_nodes, batch_size], dtype = torch.float32, device = logits.device)

    grid1 = lambda meta: (triton.cdiv(num_logits * batch_size, meta['BLOCK_SIZE']),)
    grid2 = lambda meta: (triton.cdiv(num_logits * batch_size, meta['BLOCK_SIZE']),)

    _fw_cum_logits_kernel[grid1](logits, cum_weights, node_ids, num_logits, batch_size, BLOCK_SIZE = 1024)
    _fw_norm_logits_kernel[grid2](logits, targets, cum_weights, node_ids, num_logits, batch_size, BLOCK_SIZE = 1024)

    return targets


def flat_softmax_bp(grads: torch.Tensor, normed_values: torch.Tensor, node_ids: torch.Tensor, 
                    log_param_grad: bool = False, inplace: bool = False):

    num_logits = grads.size(0)
    num_nodes = torch.max(node_ids).detach().cpu().item() + 1

    if inplace:
        target_grads = grads
    else:
        target_grads = torch.empty_like(grads)

    assert grads.is_cuda, "Input `grads` should be on GPU."

    if grads.dim() == 1:
        grads = grads.unsqueeze(1)

    batch_size = grads.size(1)

    cum_grads = torch.zeros([num_nodes, batch_size], dtype = torch.float32, device = grads.device)

    grid1 = lambda meta: (triton.cdiv(num_logits * batch_size, meta['BLOCK_SIZE']),)
    grid2 = lambda meta: (triton.cdiv(num_logits * batch_size, meta['BLOCK_SIZE']),)

    _bp_cum_logits_kernel[grid1](grads, normed_values, cum_grads, node_ids, num_logits, batch_size, BLOCK_SIZE = 1024)
    if not log_param_grad:
        _bp_norm_grads_p_kernel[grid2](grads, target_grads, normed_values, cum_grads, node_ids, num_logits, batch_size, BLOCK_SIZE = 1024)
    else:
        _bp_norm_grads_logp_kernel[grid2](grads, target_grads, cum_grads, node_ids, num_logits, batch_size, BLOCK_SIZE = 1024)

    return target_grads
