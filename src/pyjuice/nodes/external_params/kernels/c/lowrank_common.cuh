// Shared device helpers for the low-rank sum-layer kernels (forward and backward are separate
// translation units compiled into one extension).

#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ float logaddexp(float a, float b) {
    float m = fmaxf(a, b);
    // Forced to 0 when both are -inf, so the differences stay -inf (giving 0 after exp) rather than
    // becoming -inf - -inf = nan. Not exotic: it is exactly the vanishing-correction case.
    if (m == -INFINITY) return -INFINITY;
    return m + logf(expf(a - m) + expf(b - m));
}

// exp(num - den) and (num - den), both defined to vanish when either end is -inf. Needed because an
// unreachable node has logT = -inf and a vanishing factor has V = -inf, so the naive difference forms
// -inf - (-inf) = NaN -- which then spreads through logP into element_flows and every flow below it.
__device__ __forceinline__ float safe_exp_diff(float num, float den) {
    if (num == -INFINITY || den == -INFINITY) return 0.0f;
    return expf(num - den);
}

__device__ __forceinline__ float safe_log_diff(float num, float den) {
    if (num == -INFINITY || den == -INFINITY) return -INFINITY;
    return num - den;
}

// Online log-sum-exp accumulator: ONE exp per element and one log at the end, versus ~2 exp + 1 log for
// an incremental logaddexp. These reductions are latency-bound on that serially-dependent
// transcendental chain -- MEASURED at 6-16% of DRAM peak with small rank, i.e. nowhere near
// bandwidth-limited -- so cutting the transcendental count per element is what shortens them.
struct LSE { float m, s; };

__device__ __forceinline__ void lse_add(LSE &a, float x) {
    if (x == -INFINITY) return;
    if (x > a.m) {
        a.s = a.s * expf(a.m - x) + 1.0f;      // exp(-inf) = 0 on the first element, so s becomes 1
        a.m = x;
    } else {
        a.s += expf(x - a.m);
    }
}

__device__ __forceinline__ float lse_get(const LSE &a) {
    return (a.s == 0.0f) ? -INFINITY : a.m + logf(a.s);
}

// Combine two independent accumulators.
//
// WHY independent accumulators: `lse_add` is a serial dependency chain (load -> exp -> update -> next
// load), so a thread keeps only ONE load in flight. These reductions run at 7-16% of DRAM peak with
// small rank while moving a tenth of what the bus could deliver, on grids of ~4 warps per block -- too
// few warps to hide load latency, and too little ILP per warp to compensate. Splitting a reduction into
// LSE_WAYS chains that are merged at the end keeps LSE_WAYS loads and exp chains in flight at once
// without changing the arithmetic or the traffic.
__device__ __forceinline__ void lse_merge(LSE &a, const LSE &b) {
    if (b.s == 0.0f) return;
    if (b.m > a.m) {
        a.s = a.s * expf(a.m - b.m) + b.s;
        a.m = b.m;
    } else {
        a.s += b.s * expf(b.m - a.m);
    }
}

// 1: MEASURED, extra accumulators cost more in registers than they win in ILP here -- the forward's
// phase 1 went 0.47 -> 0.97 ms at four ways and did not recover at two, because fewer blocks stay
// resident on a grid that is already occupancy-starved. The latency win came from staging the
// rank-invariant loads in shared memory instead (see the kernels), not from unrolling.
#define LSE_WAYS 1

// Reduce one accumulator across a warp. Used to give the partial-reduction kernels a WARP per output
// instead of a thread: with one thread per output there are only `rank * batch` threads in the whole
// launch (256 at rank 4, batch 64), each walking a `n_tiles`-long chain of dependent loads, which cost
// those kernels ~0.6 ms at under 7% of peak -- the dominant cost of the pass once the streaming kernels
// were fixed. Lane-strided reads are fine here because the partials are tens of KB and L2-resident.
__device__ __forceinline__ void lse_warp_reduce(LSE &a) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        LSE b;
        b.m = __shfl_down_sync(0xffffffffu, a.m, off);
        b.s = __shfl_down_sync(0xffffffffu, a.s, off);
        lse_merge(a, b);
    }
}

// `element_flows` holds LOG-flows (the low-rank backward requires `logspace_flows`), so a correction
// combines with logaddexp rather than adding. CAS-based because there is no atomic logaddexp;
// uncontended in the common case of one parent block per child block, where it succeeds first try.
__device__ __forceinline__ void atomic_log_add(float* addr, float val) {
    if (val == -INFINITY) return;

    int* iaddr = reinterpret_cast<int*>(addr);
    int old = *iaddr, assumed;
    do {
        assumed = old;
        const float merged = logaddexp(__int_as_float(assumed), val);
        old = atomicCAS(iaddr, assumed, __float_as_int(merged));
    } while (assumed != old);
}
