import torch
import triton
import triton.language as tl
import pyjuice as juice
import time

import pytest


@triton.jit
def _fw_triton_sparse_kernel(node_mars, element_mars, mparams, nids, cids, pids,
                             local_ids, batch_size, partial_eval: tl.constexpr, num_edges: tl.constexpr, 
                             BLOCK_B: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    
    pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(axis = 1) # ID of size-`BLOCK_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, num_edges)
    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_ptr = mparams + par_start # [num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize and load edge mars
    edge_ids = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_ids[:,None] * batch_size + \
        offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

    # Compute max and subtract (only when using LL or GeneralLL propagation method)
    emars_max = tl.max(emars, axis = 0)
    emars = tl.exp(emars - emars_max[None,:])

    # Initialize pointers to `node_mars`
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + \
        off_nids * batch_size + \
        offs_batch

    # Inner loop
    for i in range(0, BLOCK_SIZE_M):
        epars = tl.load(epars_ptr)

        nmars = tl.where(emars_max == -float("inf"), -float("inf"), tl.log(tl.sum(emars * epars[:,None], axis = 0)) + emars_max)

        tl.store(nmars_ptr, nmars, mask = mask_batch)

        # Increment `epars_ptr`
        epars_ptr += 1

        # Increment `nmars_ptr`
        nmars_ptr += batch_size


# Define the LUT as a global tuple for compile-time unrolling
_LUT_RAW = (
    125,  573,  908, 1279, 1633, 1747, 2057, 2591,
    2898, 2858, 2896, 3017, 3220, 3521, 3922, 4427,
    4616, 4441, 4294, 4179, 4093, 4039, 4018, 4030,
    4078, 4157, 4272, 4425, 4610, 4834, 5103, 5404,
    5398, 5058, 4745, 4436, 4138, 3845, 3559, 3299,
    3028, 2776, 2542, 2315, 2094, 1888, 1686, 1499,
    1326, 1162, 1011,  865,  728,  612,  501,  399,
    311,  233,  168,  112,   68,   35,   13,    2
)

# Pack four 16-bit values into each 64-bit integer
_PACKED_LUT = tl.constexpr(tuple(
    _LUT_RAW[i*4] | (_LUT_RAW[i*4+1] << 16) | (_LUT_RAW[i*4+2] << 32) | (_LUT_RAW[i*4+3] << 48)
    for i in range(16)
))

@triton.jit
def lse_fixed_combine(a, b):
    a_max = tl.maximum(a, b)
    b_min = tl.minimum(a, b)
    sub = b_min - a_max
    shift_amt = -(sub >> 16)
    M = (65536 + (sub & 65535)) >> shift_amt
    lut_idx = (M >> 10) & 0x3F

    # # --- HIGH-PERFORMANCE B-TREE MULTIPLEXER ---
    # # 1. Split index into a 4-bit chunk index and a 2-bit sub-index
    # idx = lut_idx >> 2
    # sub_idx = lut_idx & 3

    # # 2. Extract routing bits for the binary tree
    # b0 = (idx & 1) != 0
    # b1 = (idx & 2) != 0
    # b2 = (idx & 4) != 0
    # b3 = (idx & 8) != 0

    # # 3. Parallel Binary Search Tree (Depth 4)
    # # Level 0 (Dispatched simultaneously by the SM)
    # v00 = tl.where(b0, _PACKED_LUT[1],  _PACKED_LUT[0])
    # v01 = tl.where(b0, _PACKED_LUT[3],  _PACKED_LUT[2])
    # v02 = tl.where(b0, _PACKED_LUT[5],  _PACKED_LUT[4])
    # v03 = tl.where(b0, _PACKED_LUT[7],  _PACKED_LUT[6])
    # v04 = tl.where(b0, _PACKED_LUT[9],  _PACKED_LUT[8])
    # v05 = tl.where(b0, _PACKED_LUT[11], _PACKED_LUT[10])
    # v06 = tl.where(b0, _PACKED_LUT[13], _PACKED_LUT[12])
    # v07 = tl.where(b0, _PACKED_LUT[15], _PACKED_LUT[14])

    # # Level 1
    # v10 = tl.where(b1, v01, v00)
    # v11 = tl.where(b1, v03, v02)
    # v12 = tl.where(b1, v05, v04)
    # v13 = tl.where(b1, v07, v06)

    # # Level 2
    # v20 = tl.where(b2, v11, v10)
    # v21 = tl.where(b2, v13, v12)

    # # Level 3 (Final packed int64 selection)
    # packed_val = tl.where(b3, v21, v20)

    # # 4. Extract the correct 16-bit segment from the 64-bit integer
    # bit_shift = (sub_idx * 16).to(tl.int64)
    # lut_val = (packed_val >> bit_shift) & 0xFFFF
    # lut_val = lut_val.to(tl.int32)
    # # -------------------------------------------

    # --- HIGH-PERFORMANCE POLYNOMIAL APPROXIMATION ---
    # Fits the curve y = -5.44x^2 + 342.72x + 2.16
    # Note: On Ampere/Hopper, FP32 FMA throughput is double INT32 throughput.
    # Casting to float32 for the math and back to int32 is strictly faster 
    # than an integer multiplication chain.
    idx_f = lut_idx.to(tl.float32)
    
    # Evaluates in 2 FMA clock cycles
    lut_val_f = -5.44 * idx_f * idx_f + 342.72 * idx_f + 2.16 
    
    lut_val = lut_val_f.to(tl.int32)
    # -------------------------------------------------

    res = a_max + M + lut_val

    return res


@triton.jit
def _fw_triton_sparse_lut_kernel(
    node_mars, element_mars, mparams, nids, cids, pids, local_ids,
    batch_size, partial_eval: tl.constexpr, num_edges: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(axis = 1) # ID of size-`BLOCK_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Initialize pointers to `params`
    offs_edge = tl.arange(0, num_edges)
    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_ptr = mparams + par_start # [num_edges]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize and load edge mars
    edge_ids = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_ids[:,None] * batch_size + \
        offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

    # Initialize pointers to `node_mars`
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + \
        off_nids * batch_size + \
        offs_batch

    # Inner loop
    for i in range(0, BLOCK_SIZE_M):
        epars = tl.load(epars_ptr)
        log_epars = tl.log(epars)
        
        # Combine edge weights with edge messages: x = log(weights) + messages
        log_val = log_epars[:,None] + emars # [num_edges, BLOCK_B]

        # Convert to fix-point representation (1.44269504089 is used to convert to base 2)
        log_val_fixed = (log_val * 1.44269504089 * 65536.0).to(tl.int32)

        # Parallel Tree Reduction across columns using our custom register-only combiner
        nmars_fixed = tl.reduce(log_val_fixed, axis = 0, combine_fn = lse_fixed_combine)

        # Convert final vector back to float32
        nmars = nmars_fixed.to(tl.float32) / 65536.0 * 0.69314718056

        tl.store(nmars_ptr, nmars, mask = mask_batch)

        # Increment `epars_ptr`
        epars_ptr += 1

        # Increment `nmars_ptr`
        nmars_ptr += batch_size


@triton.jit
def _fw_triton_sparse_lut_kernel2(
    node_mars, element_mars, mparams, nids, cids, pids, lut_ptr, local_ids,
    batch_size, partial_eval: tl.constexpr, num_edges: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(axis = 1) # ID of size-`BLOCK_SIZE_M` nodes

    offs_m = tl.arange(0, BLOCK_SIZE_M)

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m

    # Get the real node block id in the case of partial evaluation
    if partial_eval == 1:
        nblock_id = tl.load(local_ids + nblock_id)

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `params`
    pid_ptr = pids + nblock_id * num_edges

    # Initialize and load edge mars
    edge_ids_ptr = cids + nblock_id * num_edges

    # Inner loop
    acc_fixed = tl.zeros([BLOCK_SIZE_M, BLOCK_B], dtype = tl.int32) - 2000000000
    for i in range(0, num_edges):

        # Load emars
        edge_id = tl.load(edge_ids_ptr + i)
        emars_ptr = element_mars + \
            edge_id * batch_size + \
            offs_batch # [BLOCK_B]
        emars = tl.load(emars_ptr, mask = mask_batch) # [BLOCK_B]

        # Load epars
        par_start = tl.load(pid_ptr + i)
        epars_ptr = mparams + par_start + offs_m # [BLOCK_SIZE_M]
        epars = tl.load(epars_ptr)
        log_epars = tl.log(epars) # [BLOCK_SIZE_M]

        # Combine edge weights with edge messages: x = log(weights) + messages
        log_val = log_epars[:,None] + emars[None,:] # [BLOCK_SIZE_M, BLOCK_B]

        ## Compute logaddexp with the LUT function ##
        # Convert to fix-point representation (1.44269504089 is used to convert to base 2)
        log_val_fixed = (log_val * 1.44269504089 * 65536.0).to(tl.int32)

        val_max = tl.maximum(acc_fixed, log_val_fixed)
        val_min = tl.minimum(acc_fixed, log_val_fixed)
        
        sub = val_min - val_max
        
        # Bits shift
        shift_amt = -(sub >> 16)
        
        M = (65536 + (sub & 65535)) >> shift_amt
        lut_idx = (M >> 10) & 0x3F
        
        # Gather-load from the global LUT
        lut_val = tl.load(lut_ptr + lut_idx)
        
        # Convert to floating-point representation
        acc_fixed = val_max + M + lut_val
        
    nmars = acc_fixed.to(tl.float32) / 65536.0 * 0.69314718056

    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + \
        (off_nids + offs_m[:,None]) * batch_size + \
        offs_batch[None,:] # [BLOCK_SIZE_M, BLOCK_B]

    tl.store(nmars_ptr, nmars, mask = mask_batch[None,:])


def test_fw_kernels():

    device = torch.device("cuda:0")

    batch_size = 256
    num_edges = 64
    num_node_blocks = 16
    BLOCK_SIZE_M = 8

    node_mars1 = torch.zeros([num_node_blocks * BLOCK_SIZE_M, batch_size], device = device)
    node_mars2 = torch.zeros_like(node_mars1)
    element_mars = torch.rand([num_edges, batch_size], device = device).log()
    mparams = torch.ones([BLOCK_SIZE_M], device = device) / num_edges

    nids = torch.arange(0, num_node_blocks, device = device) * BLOCK_SIZE_M
    cids = torch.arange(0, num_edges)[None,:].repeat(num_node_blocks, 1).to(device).contiguous()
    pids = torch.zeros_like(cids)

    BLOCK_B = max(1024 // num_edges, 1)

    grid = (triton.cdiv(batch_size, BLOCK_B), num_node_blocks)
    _fw_triton_sparse_kernel[grid](
        node_mars1, element_mars, mparams, nids, cids, pids,
        local_ids = None, 
        batch_size = batch_size, 
        partial_eval = 0, 
        num_edges = num_edges, 
        BLOCK_B = BLOCK_B, 
        BLOCK_SIZE_M = BLOCK_SIZE_M
    )

    _fw_triton_sparse_lut_kernel[grid](
        node_mars2, element_mars, mparams, nids, cids, pids,
        local_ids = None, 
        batch_size = batch_size, 
        partial_eval = 0, 
        num_edges = num_edges, 
        BLOCK_B = BLOCK_B, 
        BLOCK_SIZE_M = BLOCK_SIZE_M
    )

    assert torch.all(torch.abs(node_mars1 - node_mars2) < 1e-2)


@pytest.mark.slow
def test_fw_kernels_speed():

    device = torch.device("cuda:0")

    batch_size = 256
    num_edges = 1024
    num_node_blocks = 32
    BLOCK_SIZE_M = 8

    node_mars1 = torch.zeros([num_node_blocks * BLOCK_SIZE_M, batch_size], device = device)
    node_mars2 = torch.zeros_like(node_mars1)
    element_mars = torch.rand([num_edges, batch_size], device = device).log()
    mparams = torch.ones([BLOCK_SIZE_M], device = device) / num_edges

    nids = torch.arange(0, num_node_blocks, device = device) * BLOCK_SIZE_M
    cids = torch.arange(0, num_edges)[None,:].repeat(num_node_blocks, 1).to(device).contiguous()
    pids = torch.zeros_like(cids)

    BLOCK_B = max(1024 // num_edges, 1)

    grid = (triton.cdiv(batch_size, BLOCK_B), num_node_blocks)
    _fw_triton_sparse_kernel[grid](
        node_mars1, element_mars, mparams, nids, cids, pids,
        local_ids = None, 
        batch_size = batch_size, 
        partial_eval = 0, 
        num_edges = num_edges, 
        BLOCK_B = BLOCK_B, 
        BLOCK_SIZE_M = BLOCK_SIZE_M
    )

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(100):
        _fw_triton_sparse_kernel[grid](
            node_mars1, element_mars, mparams, nids, cids, pids,
            local_ids = None, 
            batch_size = batch_size, 
            partial_eval = 0, 
            num_edges = num_edges, 
            BLOCK_B = BLOCK_B, 
            BLOCK_SIZE_M = BLOCK_SIZE_M
        )
    
    torch.cuda.synchronize()
    t1 = time.time()

    kernel1_ms = (t1 - t0) / 100 * 1000

    print(f"Kernel1 on average takes {kernel1_ms:.3f}ms.")
    print("---------------------------------------------------")

    BLOCK_B = max(1024 // num_edges, 1)

    grid = (triton.cdiv(batch_size, BLOCK_B), num_node_blocks)
    _fw_triton_sparse_lut_kernel[grid](
        node_mars1, element_mars, mparams, nids, cids, pids,
        local_ids = None, 
        batch_size = batch_size, 
        partial_eval = 0, 
        num_edges = num_edges, 
        BLOCK_B = BLOCK_B, 
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        num_warps = 1
    )

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(100):
        _fw_triton_sparse_lut_kernel[grid](
            node_mars1, element_mars, mparams, nids, cids, pids,
            local_ids = None, 
            batch_size = batch_size, 
            partial_eval = 0, 
            num_edges = num_edges, 
            BLOCK_B = BLOCK_B, 
            BLOCK_SIZE_M = BLOCK_SIZE_M,
            num_warps = 1
        )
    
    torch.cuda.synchronize()
    t1 = time.time()

    kernel2_ms = (t1 - t0) / 100 * 1000

    print(f"Kernel2 on average takes {kernel2_ms:.3f}ms.")
    print("---------------------------------------------------")

    lut_ptr = torch.tensor([
        125,  573,  908, 1279, 1633, 1747, 2057, 2591,
        2898, 2858, 2896, 3017, 3220, 3521, 3922, 4427,
        4616, 4441, 4294, 4179, 4093, 4039, 4018, 4030,
        4078, 4157, 4272, 4425, 4610, 4834, 5103, 5404,
        5398, 5058, 4745, 4436, 4138, 3845, 3559, 3299,
        3028, 2776, 2542, 2315, 2094, 1888, 1686, 1499,
        1326, 1162, 1011,  865,  728,  612,  501,  399,
        311,  233,  168,  112,   68,   35,   13,    2
    ], dtype = torch.int32, device = device)

    _fw_triton_sparse_lut_kernel2[grid](
        node_mars1, element_mars, mparams, nids, cids, pids, lut_ptr,
        local_ids = None, 
        batch_size = batch_size, 
        partial_eval = 0, 
        num_edges = num_edges, 
        BLOCK_B = BLOCK_B, 
        BLOCK_SIZE_M = BLOCK_SIZE_M
    )

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(100):
        _fw_triton_sparse_lut_kernel2[grid](
            node_mars1, element_mars, mparams, nids, cids, pids, lut_ptr,
            local_ids = None, 
            batch_size = batch_size, 
            partial_eval = 0, 
            num_edges = num_edges, 
            BLOCK_B = BLOCK_B, 
            BLOCK_SIZE_M = BLOCK_SIZE_M
        )
    
    torch.cuda.synchronize()
    t1 = time.time()

    kernel3_ms = (t1 - t0) / 100 * 1000

    print(f"Kernel3 on average takes {kernel3_ms:.3f}ms.")
    print("---------------------------------------------------")


if __name__ == "__main__":
    # test_fw_kernels()
    test_fw_kernels_speed()
