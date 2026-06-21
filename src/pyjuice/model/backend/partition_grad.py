"""
Backend helpers for gradient-based PC optimization (SGD / Adam), used by `pyjuice.optim.SGD` and
`pyjuice.optim.Adam`. Everything here is ADDITIVE: it only reads existing `TensorCircuit` / layer
state and never modifies any existing pyjuice code path.

The central operation is :func:`eval_partition_grad`, which runs a forward/backward over the circuit
with the parameters treated as *unnormalized* (so the root marginal is the partition function `Z`) and
accumulates the partition-function flow into ``pc.param_flows`` (subtracting it when
``negate_pflows = True``). Combined with the data flow already in ``pc.param_flows`` (from
``pc.backward``), this yields the gradient of the *normalized* log-likelihood w.r.t. the log-parameters.

Ported from the research implementation `pc-mini-em/src/sgd/sgd_wrapper.py` and adapted to the current
pyjuice layer-group API. Restricted to categorical input layers (the partition forward marginalizes an
input node as the sum of its category parameters).
"""

import torch
import triton
import triton.language as tl

# In newer triton, the math functions moved (https://github.com/openai/triton/pull/3172).
if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math


# ---------------------------------------------------------------------------------------------------
# Input-layer partition-flow accumulation (num_cats <= 1024 and the > 1024 variant)
# ---------------------------------------------------------------------------------------------------

@triton.jit
def _acc_all_kernel(node_flows_ptr, pfids_ptr, pids_ptr, params_ptr, param_flows_ptr, nsid: tl.constexpr,
                    num_cats: tl.constexpr, batch_size: tl.constexpr, neg: tl.constexpr,
                    layer_num_nodes: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_C: tl.constexpr,
                    norm_params: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_nodes

    n_pfids = tl.load(pfids_ptr + offsets, mask = mask)
    n_pids = tl.load(pids_ptr + offsets, mask = mask)

    # The partition forward fills every batch column identically, so column 0 holds the node's
    # partition flow.
    nflows = tl.load(node_flows_ptr + (nsid + offsets) * batch_size, mask = mask)

    coffs = tl.arange(0, BLOCK_C)
    pfids = n_pfids[:, None] + coffs[None, :]
    pids = n_pids[:, None] + coffs[None, :]
    cmask = coffs < num_cats

    params = tl.load(params_ptr + pids, mask = mask[:, None])
    if norm_params:
        norm_param = params / (tl.sum(params, axis = 1)[:, None] + 1e-12)
    else:
        norm_param = params

    sflow = tl.exp(nflows[:, None]) * norm_param * batch_size
    if neg:
        tl.atomic_add(param_flows_ptr + pfids, -sflow, mask = (mask[:, None] & cmask[None, :]))
    else:
        tl.atomic_add(param_flows_ptr + pfids, sflow, mask = (mask[:, None] & cmask[None, :]))


@triton.jit
def _acc_all_kernel_large(node_flows_ptr, pfids_ptr, pids_ptr, params_ptr, param_flows_ptr, nsid: tl.constexpr,
                          num_cats: tl.constexpr, batch_size: tl.constexpr, neg: tl.constexpr,
                          layer_num_nodes: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_C: tl.constexpr,
                          nc: tl.constexpr, norm_params: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_nodes

    n_pfids = tl.load(pfids_ptr + offsets, mask = mask)
    n_pids = tl.load(pids_ptr + offsets, mask = mask)

    nflows = tl.load(node_flows_ptr + (nsid + offsets) * batch_size, mask = mask)

    coffs1 = tl.arange(0, BLOCK_C)
    cum_params = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
    for i in range(nc):
        pids = n_pids[:, None] + coffs1[None, :]
        cmask = coffs1 < num_cats
        params = tl.load(params_ptr + pids, mask = (mask[:, None] & cmask[None, :]), other = 0.0)
        cum_params += tl.sum(params, axis = 1)
        coffs1 += BLOCK_C

    coffs2 = tl.arange(0, BLOCK_C)
    for i in range(nc):
        pfids = n_pfids[:, None] + coffs2[None, :]
        pids = n_pids[:, None] + coffs2[None, :]
        cmask = coffs2 < num_cats
        params = tl.load(params_ptr + pids, mask = (mask[:, None] & cmask[None, :]), other = 0.0)
        if norm_params:
            norm_param = params / (cum_params[:, None] + 1e-12)
        else:
            norm_param = params

        sflow = tl.exp(nflows[:, None]) * norm_param * batch_size
        if neg:
            tl.atomic_add(param_flows_ptr + pfids, -sflow, mask = (mask[:, None] & cmask[None, :]))
        else:
            tl.atomic_add(param_flows_ptr + pfids, sflow, mask = (mask[:, None] & cmask[None, :]))
        coffs2 += BLOCK_C


def _accum_input_partition_flows(pc, layer, negate_pflows, norm_params):
    nsid, neid = layer._output_ind_range
    num_cats = layer.nodes[0].dist.num_cats
    batch_size = pc.node_mars.size(1)

    if num_cats <= 1024:
        BLOCK_SIZE = triton.next_power_of_2(max(2048 // num_cats, 1))
        BLOCK_C = triton.next_power_of_2(num_cats)
        grid = (triton.cdiv(neid - nsid, BLOCK_SIZE),)
        _acc_all_kernel[grid](
            pc.node_flows, layer.s_pfids, layer.s_pids, layer.params, layer.param_flows,
            nsid, num_cats = num_cats, batch_size = batch_size, neg = negate_pflows,
            layer_num_nodes = neid - nsid, BLOCK_SIZE = BLOCK_SIZE, BLOCK_C = BLOCK_C,
            norm_params = norm_params
        )
    else:
        BLOCK_SIZE = 8
        BLOCK_C = 256
        nc = triton.cdiv(num_cats, BLOCK_C)
        grid = (triton.cdiv(neid - nsid, BLOCK_SIZE),)
        _acc_all_kernel_large[grid](
            pc.node_flows, layer.s_pfids, layer.s_pids, layer.params, layer.param_flows,
            nsid, num_cats = num_cats, batch_size = batch_size, neg = negate_pflows,
            layer_num_nodes = neid - nsid, BLOCK_SIZE = BLOCK_SIZE, BLOCK_C = BLOCK_C, nc = nc,
            norm_params = norm_params
        )


# ---------------------------------------------------------------------------------------------------
# Partition-function gradient: forward/backward with UNNORMALIZED parameters
# ---------------------------------------------------------------------------------------------------

def eval_partition_grad(pc, negate_pflows: bool = True, input_layer_norm_params: bool = True):
    """
    Run a forward/backward treating ``pc.params`` as unnormalized, so the root marginal is the
    log-partition ``log Z``, and accumulate the partition-function flow into ``pc.param_flows`` (and
    each input layer's ``param_flows``). With ``negate_pflows = True`` the partition flow is
    subtracted, turning a ``param_flows`` that holds the data flow into the gradient of the normalized
    log-likelihood. Uses the same batch size as the most recent forward (``pc.node_mars.size(1)``).
    """
    B = pc.node_mars.size(1)

    # Forward pass (unnormalized): an input node's marginal is the sum of its category parameters.
    pc._init_buffer(name = "node_mars", shape = (pc.num_nodes, B), set_value = 0.0)
    pc._init_buffer(name = "element_mars", shape = (pc.num_elements, B), set_value = -float("inf"))

    for layer in pc.input_layer_group:
        nsid, neid = layer._output_ind_range
        num_cats = layer.nodes[0].dist.num_cats
        inds = layer.s_pids[:, None] + torch.arange(0, num_cats, device = pc.node_mars.device)[None, :]
        pc.node_mars[nsid:neid, :] = layer.params[inds].sum(dim = 1, keepdim = True).log()

    for layer_group in pc.inner_layer_groups:
        if layer_group.is_prod():
            layer_group(pc.node_mars, pc.element_mars)
        elif layer_group.is_sum():
            layer_group(pc.node_mars, pc.element_mars, pc.params,
                        force_use_fp32 = False, propagation_alg = "LL")
        else:
            raise ValueError(f"Unknown layer type {type(layer_group)}.")

    # Backward pass (log-space flows), root flow = log(1) = 0.
    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, B), set_value = -float("inf"))
    pc._init_buffer(name = "element_flows", shape = (pc.num_elements, B), set_value = -float("inf"))
    pc.node_flows[pc._root_node_range[0]:pc._root_node_range[1], :] = 0.0

    for layer_id in range(len(pc.inner_layer_groups) - 1, -1, -1):
        layer_group = pc.inner_layer_groups[layer_id]
        if layer_group.is_prod():
            layer_group.backward(pc.node_flows, pc.element_flows, logspace_flows = True)
        elif layer_group.is_sum():
            # Recompute the preceding product layer, then back-propagate the sum layer.
            pc.inner_layer_groups[layer_id - 1].forward(pc.node_mars, pc.element_mars, _for_backward = True)
            layer_group.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params,
                                 param_flows = pc.param_flows, allow_modify_flows = False,
                                 propagation_alg = "LL", logspace_flows = True, negate_pflows = negate_pflows)
        else:
            raise ValueError(f"Unknown layer type {type(layer_group)}.")

    for layer in pc.input_layer_group:
        _accum_input_partition_flows(pc, layer, negate_pflows = negate_pflows, norm_params = input_layer_norm_params)


# ---------------------------------------------------------------------------------------------------
# Input-layer log-space parameter update (SGD step): params <- exp(log(params) + lr * grad)
# ---------------------------------------------------------------------------------------------------

@triton.jit
def _sgd_input_kernel(params_ptr, param_flows_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr,
                      source_nids_ptr, constexprs_ptr, layer_num_source_nodes: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    lr = tl.load(constexprs_ptr)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_source_nodes

    local_offsets = tl.load(source_nids_ptr + offsets, mask = mask, other = 0)
    s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
    s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)
    s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
    num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

    max_num_cats = tl.max(num_cats, axis = 0)

    for cat_id in range(max_num_cats):
        cat_mask = mask & (cat_id < num_cats)
        param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
        flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
        new_param = tl.exp(tl.log(param) + lr * flow)
        tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)


def accum_input_tied_flows(layer):
    """Consolidate tied input-node flows onto their source positions (reuses the layer's own kernel)."""
    for i in range(len(layer.tied2source_nids)):
        pfid_start, num_par_flows, ch_pfids = layer.tied2source_nids[i]
        num_coalesced_blocks = ch_pfids.size(0)
        if num_coalesced_blocks > 1024:
            raise NotImplementedError("Unsupported number of coalesced parameter flows.")
        BLOCK_N = triton.next_power_of_2(num_coalesced_blocks)
        BLOCK_M = min(1024 // BLOCK_N, num_par_flows)
        grid = (triton.cdiv(num_par_flows, BLOCK_M),)
        layer._pflow_accum_kernel[grid](
            param_flows_ptr = layer.param_flows,
            pfid_start = pfid_start,
            ch_pfids_ptr = ch_pfids,
            num_coalesced_blocks = num_coalesced_blocks,
            num_par_flows = num_par_flows,
            BLOCK_M = BLOCK_M,
            BLOCK_N = BLOCK_N
        )


def sgd_input_layer_update(layer, lr: float):
    """Apply the log-space SGD update to an input layer's source parameters (no tied accumulation;
    callers consolidate tied flows beforehand)."""
    layer_num_source_nodes = layer.source_nids.size(0)
    constexprs = torch.tensor([lr], dtype = torch.float32, device = layer.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(layer_num_source_nodes, BLOCK_SIZE),)
    _sgd_input_kernel[grid](
        params_ptr = layer.params,
        param_flows_ptr = layer.param_flows,
        s_pids_ptr = layer.s_pids,
        s_pfids_ptr = layer.s_pfids,
        metadata_ptr = layer.metadata,
        s_mids_ptr = layer.s_mids,
        source_nids_ptr = layer.source_nids,
        constexprs_ptr = constexprs,
        layer_num_source_nodes = layer_num_source_nodes,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = 8
    )


# ---------------------------------------------------------------------------------------------------
# Gradient transforms (in-place): momentum and Adam
# ---------------------------------------------------------------------------------------------------

def momentum_update(gradients, m, t, momentum, dampening = 0.0, nesterov = False):
    with torch.no_grad():
        for idx, (grad, m_t) in enumerate(zip(gradients, m)):
            if m_t is None:
                m_t = torch.zeros_like(grad)
                m[idx] = m_t
            if t > 1:
                m_t[:] = momentum * m_t + (1.0 - dampening) * grad
            else:
                m_t[:] = grad
            if nesterov:
                grad[:] = grad + momentum * m_t
            else:
                grad[:] = m_t


def adam_update(gradients, m, v, t, beta1 = 0.9, beta2 = 0.95, epsilon = 1e-8):
    with torch.no_grad():
        for idx, (grad, m_t, v_t) in enumerate(zip(gradients, m, v)):
            if m_t is None:
                m_t = torch.zeros_like(grad)
                v_t = torch.zeros_like(grad)
                m[idx] = m_t
                v[idx] = v_t
            m_t[:] = beta1 * m_t + (1 - beta1) * grad
            v_t[:] = beta2 * v_t + (1 - beta2) * (grad ** 2)
            m_t_hat = m_t / (1 - beta1 ** t)
            v_t_hat = v_t / (1 - beta2 ** t)
            grad[:] = m_t_hat / (torch.sqrt(v_t_hat) + epsilon)
