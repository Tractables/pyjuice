import torch
import triton
import triton.language as tl
import pyjuice as juice
import time

import pytest


@triton.jit
def test_kernel(node_mars_ptr, element_mars_ptr, params_ptr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    node_mars_ptr: [BLOCK_SIZE_N, BLOCK_SIZE_B]
    element_mars_ptr: [BLOCK_SIZE_M, BLOCK_SIZE_B]
    params_ptr: [BLOCK_SIZE_N, BLOCK_SIZE_M]
    """

    # In this test case, we only launch 1 thread-block, so both `pid_b` and `pid_m` are 0
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Load `element_mars`
    emars = tl.load(element_mars_ptr + tl.arange(0, BLOCK_SIZE_M)[:,None] * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[None,:])

    # Load `params`
    params = tl.load(params_ptr + tl.arange(0, BLOCK_SIZE_N)[:,None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None,:])

    ## A simple example of the main logsumexp computation part ##

    acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_B], dtype = tl.float32) - float("inf")

    emars_max = tl.max(emars, axis = 0)[None,:]
    emars_sub = tl.exp(emars - emars_max)

    nmars = tl.dot(params, emars_sub)

    acc = tl.where(emars_max > acc,
        tl.log(nmars + tl.exp(acc - emars_max) + 1e-24) + emars_max,
        tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc
    )

    # Write back
    tl.store(node_mars_ptr + tl.arange(0, BLOCK_SIZE_N)[:,None] * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[None,:], acc)


@triton.jit
def compute_lse_pair_fixed_with_lut_kernel(x_ptr, y_ptr, z_ptr, lut_ptr, seq_len, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = (offsets < seq_len)

    # Load `x` and `y`
    x = tl.load(x_ptr + offsets, mask = mask, other = 0.0)
    y = tl.load(y_ptr + offsets, mask = mask, other = 0.0)

    # Convert to fix-point representation (1.44269504089 is used to convert to base 2)
    x_fixed = (x * 1.44269504089 * 65536.0).to(tl.int32)
    y_fixed = (y * 1.44269504089 * 65536.0).to(tl.int32)

    x_max = tl.maximum(x_fixed, y_fixed)
    y_min = tl.minimum(x_fixed, y_fixed)
    
    sub = y_min - x_max
    
    # Bits shift
    shift_amt = -(sub >> 16)
    
    M = (65536 + (sub & 65535)) >> shift_amt
    lut_idx = (M >> 10) & 0x3F
    
    # Gather-load from the global LUT
    lut_val = tl.load(lut_ptr + lut_idx)
    
    # Convert to floating-point representation
    z = (x_max + M + lut_val).to(tl.float32) / 65536.0 * 0.69314718056

    tl.store(z_ptr + offsets, z, mask = mask)


# Define the LUT as a global tuple for compile-time unrolling
LUT_TUPLE = tl.constexpr((
    125,  573,  908, 1279, 1633, 1747, 2057, 2591,
    2898, 2858, 2896, 3017, 3220, 3521, 3922, 4427,
    4616, 4441, 4294, 4179, 4093, 4039, 4018, 4030,
    4078, 4157, 4272, 4425, 4610, 4834, 5103, 5404,
    5398, 5058, 4745, 4436, 4138, 3845, 3559, 3299,
    3028, 2776, 2542, 2315, 2094, 1888, 1686, 1499,
    1326, 1162, 1011,  865,  728,  612,  501,  399,
    311,  233,  168,  112,   68,   35,   13,    2
))

@triton.jit
def lse_fixed_combine(a, b):
    # # 1. Mask out identity values (padded elements)
    # # We use a large negative integer to represent -inf in our fixed-point space
    # INF_THRESHOLD = -1900000000
    # is_a_inf = a < INF_THRESHOLD
    # is_b_inf = b < INF_THRESHOLD

    # 2. Fixed-point LSE trick
    a_max = tl.maximum(a, b)
    b_min = tl.minimum(a, b)
    sub = b_min - a_max
    shift_amt = -(sub >> 16)
    M = (65536 + (sub & 65535)) >> shift_amt
    lut_idx = (M >> 10) & 0x3F

    # 3. Register multiplexer replacing tl.load
    # Unrolled at compile time; maps the LUT into the Instruction Cache
    lut_val = tl.zeros_like(lut_idx)
    for i in tl.static_range(64):
        lut_val = tl.where(lut_idx == i, LUT_TUPLE[i], lut_val)

    res = a_max + M + lut_val

    # # 4. Identity bypass to prevent accumulating fixed-point approximation errors from padding
    # res = tl.where(is_a_inf, b, res)
    # res = tl.where(is_b_inf, a, res)

    return res

@triton.jit
def compute_lse_pair_fixed_with_lut_matrix_kernel(x_ptr, y_ptr, seq_len, num_seqs, SEQ_LEN_NP2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)

    seq_offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    seq_mask = (seq_offsets < num_seqs)

    ind_offsets = tl.arange(0, SEQ_LEN_NP2)
    ind_mask = (ind_offsets < seq_len)

    # Load `x` and `y`
    x = tl.load(x_ptr + seq_offsets[:,None] * seq_len + ind_offsets[None,:], mask = (seq_mask[:,None] & ind_mask[None,:]), other = 0.0)

    # Convert to fix-point representation (1.44269504089 is used to convert to base 2)
    x_fixed = (x * 1.44269504089 * 65536.0).to(tl.int32)

    # Parallel Tree Reduction across columns using our custom register-only combiner
    z_fixed = tl.reduce(x_fixed, axis = 1, combine_fn = lse_fixed_combine)

    # Convert final vector back to float32
    z = z_fixed.to(tl.float32) / 65536.0 * 0.69314718056

    # Store
    tl.store(y_ptr + seq_offsets, z, mask = seq_mask)


@pytest.mark.slow
def test_logsumexp_kernel():

    device = torch.device("cuda:0")
    
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    node_mars = torch.zeros([BLOCK_SIZE_N, BLOCK_SIZE_B], device = device)
    element_mars = torch.rand([BLOCK_SIZE_M, BLOCK_SIZE_B], device = device).log()
    params = torch.rand([BLOCK_SIZE_N, BLOCK_SIZE_M], device = device)

    grid = (1, 1)
    test_kernel[grid](node_mars, element_mars, params, BLOCK_SIZE_B, BLOCK_SIZE_N, BLOCK_SIZE_M)

    ## This is my attempted implementation of the lse reduce function in triton ##

    LUT64 = torch.tensor([
        125,  573,  908, 1279, 1633, 1747, 2057, 2591,
        2898, 2858, 2896, 3017, 3220, 3521, 3922, 4427,
        4616, 4441, 4294, 4179, 4093, 4039, 4018, 4030,
        4078, 4157, 4272, 4425, 4610, 4834, 5103, 5404,
        5398, 5058, 4745, 4436, 4138, 3845, 3559, 3299,
        3028, 2776, 2542, 2315, 2094, 1888, 1686, 1499,
        1326, 1162, 1011,  865,  728,  612,  501,  399,
        311,  233,  168,  112,   68,   35,   13,    2
    ], dtype = torch.int32, device = device)

    x = torch.rand([10000], device = device).log()
    y = torch.rand([10000], device = device).log()
    z = torch.zeros([10000], device = device)

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(10000, BLOCK_SIZE),)
    compute_lse_pair_fixed_with_lut_kernel[grid](x, y, z, LUT64, 10000, BLOCK_SIZE)

    z_target = torch.logaddexp(x, y)

    assert torch.all(torch.abs(z - z_target) < 1e-2)

    x = torch.rand([1000,256], device = device).log()
    y = torch.rand([1000], device = device)

    BLOCK_SIZE = 4

    grid = (triton.cdiv(1000, BLOCK_SIZE),)
    compute_lse_pair_fixed_with_lut_matrix_kernel[grid](x, y, 256, 1000, 256, BLOCK_SIZE)

    y_target = torch.logsumexp(x, dim = 1)

    assert torch.all(torch.abs(y - y_target) < 1e-2)


@pytest.mark.slow
def test_lse_speed():

    device = torch.device("cuda:0")

    LUT64 = torch.tensor([
        125,  573,  908, 1279, 1633, 1747, 2057, 2591,
        2898, 2858, 2896, 3017, 3220, 3521, 3922, 4427,
        4616, 4441, 4294, 4179, 4093, 4039, 4018, 4030,
        4078, 4157, 4272, 4425, 4610, 4834, 5103, 5404,
        5398, 5058, 4745, 4436, 4138, 3845, 3559, 3299,
        3028, 2776, 2542, 2315, 2094, 1888, 1686, 1499,
        1326, 1162, 1011,  865,  728,  612,  501,  399,
        311,  233,  168,  112,   68,   35,   13,    2
    ], dtype = torch.int32, device = device)

    x = torch.rand([10000], device = device).log()
    y = torch.rand([10000], device = device).log()
    z = torch.zeros([10000], device = device)

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(10000, BLOCK_SIZE),)
    compute_lse_pair_fixed_with_lut_kernel[grid](x, y, z, LUT64, 10000, BLOCK_SIZE)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(100):
        compute_lse_pair_fixed_with_lut_kernel[grid](x, y, z, LUT64, 10000, BLOCK_SIZE)

    torch.cuda.synchronize()
    t1 = time.time()

    kernel1_ms = (t1 - t0) / 100 * 1000

    print(f"Kernel1 on average takes {kernel1_ms:.3f}ms.")
    print("---------------------------------------------------")

    z = torch.logaddexp(x, y)

    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(100):
        z = torch.logaddexp(x, y)

    torch.cuda.synchronize()
    t1 = time.time()

    kernel2_ms = (t1 - t0) / 100 * 1000

    print(f"Kernel2 on average takes {kernel2_ms:.3f}ms.")
    print("---------------------------------------------------")


if __name__ == "__main__":
    test_logsumexp_kernel()
    test_lse_speed()
