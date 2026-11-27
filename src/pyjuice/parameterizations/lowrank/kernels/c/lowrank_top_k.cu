// `LowRank(top_k)` sum layers' FORWARD (`top_k.py`'s `_sum_values`): the node values of a node vector whose scale reads
// the token of the leaf beside it, made tile by tile, the tile never stored:
//
//     W_b[i, j] = 2^(phi'_b[i, j] + l2t[i, j] - m_b[i])        phi' = A' B, A' = log2 e A (fp16-rounded factors),
//                                                                l2t = log2 theta less its row's largest (fp16)
//     Z_b[i, s] = sum_j W_b[i, j] E_b[s, j]          U_b[i, s] = sum_j W_b[i, j] ev_b[j] E_b[s, j]
//     value_b[i] = log sum_s w_b[i, s] U_b[i, s] / Z_b[i, s] + c_b
//
// E = exp(G - its largest over the children) (a slot's largest 1), ev = exp(lc - c) (the children's values over their
// largest, c), w the slots' weights: where the leaf's token is MISSING its probabilities of the slots (`logw`), where it
// is OBSERVED one slot. The row's shifts -- log2 theta's largest, the running maximum m_b[i] kept over the children's
// chunks -- cancel in every ratio; they only keep W in range.
//
// PRECISION, measured on the best Codpula head against float64 (`check_tk_kernel_e2e.py`): phi' from fp16-rounded
// factors (one m16n8k16 MMA a 16 ranks), l2t in fp16 (its row's largest 0: the dominant entries near 0, where fp16 is
// finest), the masked samples' contraction W [ev E, E] in fp16 (m16n8k16, round to nearest: TF32's 10 bits) with each
// operand scaled to the top of fp16's range -- W by 2^14 (its reference lazy by 2^1 only), E by 2^15 (a slot's largest
// 1) --: ~3e-5 nats a masked token's read-out. The scales matter: a slot's weights E_b[s, j] W_b[i, j] peak where the
// row's and the slot's maxima do not meet, so its dominant products' factors sit far below both maxima; these keep
// their bits down to 2^-28 (W) and 2^-29 (E) of them. MEASURED (three transitions of the head, the operands' rounding
// emulated in float64): the node values' max and flow-weighted errors TF32's (8e-4, 1e-4 nats), Z and U within TF32's
// 9e-4 (relative) wherever the slot's posterior exceeds 1e-6 (2 binades of W's headroom given away: U 1.1e-3 there;
// 4: 3.7e-3); W and E as they are (maxima 1): values 1e-2 .. 6e-2 nats off; W lazy by 2^8 (times 2^7): values as
// TF32's, Z / U up to 1e-2 off at a posterior of 1e-4. A bf16 contraction moved the read-out by 1.4e-4 (over the 1e-4
// gate).
//
// SAMPLES SHARING A ROW OF TENSORS (`external_params_rows`: both passes of a conditional one batch) share the tile W,
// which their tensors alone determine: the WORK ITEMS (`tk_items_kernel`) gather them -- a MASKED item a sample whose
// token is missing with up to two observed samples reading the same row (its partners), an OBSERVED item two slots of a
// row and up to two of its observed samples. Without the map every sample reads its own row: the same items. The node
// vectors reading one leaf layer take theirs from one launch a pass (`top_k_items`, a CTA a leaf: they depend on the
// leaf's slots and masked rows alone) -- a launch a node vector was ~13.5 us of one CTA's scans on the chain each.
//
// TWO PERSISTENT LAUNCHES a node vector (`tk_sk_kernel`, stream-K: its notes below), one a kind of item:
//
//  * MASKED items (255 registers, one CTA an SM): 128 nodes (eight warps of 16) x one item, its children in chunks of
//    64 -- double-buffered cp.async stages of the row's B chunk (rank-major: `ldmatrix.trans` makes the MMA's
//    fragments), the chunk of E's slots 0 .. 63 in fp16 (`E16`, the prep's), E's rows of the rest slot and of the
//    partners' slots (fp32) and the samples' ev chunks. A warp keeps its A' fragments (made once from the caller's fp32
//    A), its accumulators, its rows' running maxima and l2t's fragments of the next chunk (read from global in FRAGMENT
//    ORDER, `LTh`: one 256-byte run a warp's load; the C operand the logits' MMAs start from). The masked sample's slots
//    0 .. 63 by fp16 MMAs (m16n8k16: half the TF32 m16n8k8 MMAs of the same cost each), the shared "rest" slot and the
//    partners' sums (their slot's E row, their ev) by FP32; ev on the A side (W ev beside W), so both contractions read
//    the same B operand E; the logits' accumulator fragments of two n-tiles ARE the MMA's A fragment (rows g, g + 8 x
//    children 2t, 2t + 1 of each: one conversion a pair), E's B fragments two slot tiles an `ldmatrix`.
//  * OBSERVED items: the same tile a slot (a row of tensors, its l2t's reads shared by the two slots), FP32 sums only
//    (z once where a slot's two samples take the same slot of E).
//
// MEASURED (RTX PRO 6000, one CTA an SM alone): the per-SM ingest (~62 GB/s from the L2) does not overlap the compute
// well -- the kernels' time ~ compute + their bytes' time -- so the bytes a CTA reads are cut where they can be: l2t
// fp16 and shared by two slots, A' made in the kernel, one barrier a chunk, the next stage's loads issued right after
// it. The conditional of the best Codpula head (48 blocks, 31 transitions): two passes 271 us a transition, one batch
// through the map 159 us; its masked slots by fp16 MMAs (E16 a stage: 22 KB a chunk, was 31) 152 -> 123 us.
//
// The PREP kernel (`tk_prep_kernel`, one launch a node vector) makes the other operands: a row of tensors each, B [Bx,
// KP, N] fp16 (ranks zero-padded to KP = 32, 64, 96 or 128), E [Bx, S, N] fp32 (TF32-rounded; S = 65) and its slots
// 0 .. 63 in fp16 times 2^15 (E16 [Bx, 64, N]: the masked forward's); a sample each, the children's values from the
// circuit's (ev [B, N], their largest c [B], -inf where every child's is) and the observed slots; l2t in fragment
// order; the masked samples' weights [Rm, M, SP] fp32 (SP = 72: aligned pairs).
//
// The BACKWARD (`top_k._sum_backward_terms`), from the node flows f, with a = f pi the flow into each (node, slot),
// alpha = a / U, beta = a / Z:
//
//     Gl[i, j] = W[i, j] (ev[j] X[i, j] - Y[i, j])      X = alpha E, Y = beta E          d / d (A B)
//     ef[j] = ev[j] sum_i W[i, j] X[i, j]               the element flows
//     dG[s, j] = E[s, j] sum_i (alpha[i, s] ev[j] - beta[i, s]) W[i, j],      dA = Gl B^T,      dB = A^T Gl
//
// takes from the forward's sums a STASH (`tk_sk_kernel`'s STASH ending: the forward's own launch where a backward may
// follow, else that launch again without the values): the slots' posterior pi (for the leaf beside the sum), pi / U,
// pi / Z, the partners' 1 / u, 1 / z, the shift h of W a sample (the tile's running maximum) -- no node flows in it,
// which are a factor a node's row: the flows kernel (`tk_flows_kernel`) puts them into the [B, M] terms, the passes
// multiply a masked sample's terms by its nodes'. Then two passes over the same work items, each a persistent launch
// a kind of item:
//
//  * COLS (`tk_bw_cols_kernel`: 128 children x an item, eight warps of 16, the nodes in chunks of 32): the transposed
//    tile -- W^T = B^T A'^T (+ l2t^T), X^T = E^T alpha^T, Y^T = E^T beta^T, the children the MMAs' M -- so the sums over
//    the nodes stay in the warp: the element flows, Gl^T summed over the item's samples, each times its sign (the
//    samples of a row of tensors share its gradients: `external_params_rows`' fold), dB^T += Gl^T A' (A' the forward's
//    fp16 operand), dG^T += (W ev)^T alpha - W^T beta; and Gl^T itself, stored (fp16, [items, N, M]). The slot
//    contractions (X, Y, dG) are fp16 MMAs (m16n8k16: half the instructions of TF32's m16n8k8, the same 10-bit
//    operands; fp16's narrower range met by powers of 2): the stash's fp16 alpha, beta rows times a node's d (its
//    largest alpha into [2^14, 2^15); 1 / d folded into the node flows), dG's A operand (W ev and -W, times the flows
//    and 1 / d) times a segment's power of 2 that takes its largest node factor to [2^6, 2^7) (A below 2^15: its
//    small values off the subnormals), taken out at the epilogue. MEASURED (the best Codpula head, 48 blocks, vs
//    float64): every gradient and EM statistic within the TF32 kernels' error (dG 9.6e-4 of its largest either way;
//    with the scale's largest factor at 1 instead, 2.6e-3); the pass 33.3 -> 31.6 ms.
//  * dA (`tk_bw_da_gemm_kernel`): a masked item's dA += Gl B^T, a GEMM of the stored Gl and the item's row of B (a CTA
//    128 nodes x the ranks, the children in chunks of 32, stream-K) -- its tile made once, by the cols pass (a dA pass
//    of its own made W, X and Y again in the forward's orientation: 1.8x the GEMM's MMAs, and the elementwise work
//    again); the masked items in groups, a group's Gl about 2/5 of the L2 (all of them overflowed it: the GEMM 2.5x
//    slower). An observed slot's dA (`tk_bw_da_kernel`): W again, Gl in the forward's orientation (their count is on
//    the device: a store for them sized by its bound, ~2 MB a sample, for a few slots).
//
// Gl is dA's and dB's fp16 MMA operand (m16n8k16 against the fp16 factors as they lie, the sums in fp32): |Gl[i, j]|
// <= f[i] a sample, so a power of 2 takes it into fp16's range -- a node's row by its samples' flows there (dA: the
// stored Gl, its rows' inverse scales beside it), the item by its samples' largest flows (dB: its sum runs over the
// nodes) -- and comes back out of the sums. An entry
// below 2^-14 of that bound loses bits (below 2^-24: all of them), what TF32 kept: MEASURED on the head's inputs
// against float64, dA and dB are what their TF32 operand gave to every printed digit (dA 1.5e-3 .. 4.4e-3, dB 1.1e-3
// .. 1.8e-3 of the gradient's norm: the forward's fp16 logits); the same inputs through both kernels: dA 4.4e-7, dB
// 7.9e-7 of the largest apart. The bound is the terms' -- W's shift cancels in them --, not their factors': W alone
// reaches 2^8 (the forward's lazy shift), so the scale (up to 2^126, for flows ~1e-35) multiplies the bounded product,
// last; the conversion saturates (`sat_h2`) where rounding takes a value past it.
//
// The cols' and the GEMM's sums are linear in the chunks: a CTA takes an equal share of the items' chunks (stream-K) and
// adds (red) each segment's sums into the destinations at its end -- the gradients into the rows of tensors the samples
// read, the element flows a sample each. MEASURED (the best Codpula head, 48 rows x both passes, 96 samples, RTX PRO
// 6000): dA 190 us, cols 287 us a transition (the Triton kernels before: 537 + 940); vs float64 dA 1.0e-3, dB 3.9e-4,
// dG 4.2e-4, the element flows 1.6e-4 of the largest (the forward's fp16 logits).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <type_traits>

namespace {

constexpr int TM = 128;                // nodes a CTA
constexpr int NW = 8;                  // warps a CTA (16 nodes each)
constexpr int TN = 64;                 // children a chunk
constexpr int BROW = TN + 8;           // halves a staged B row (ldmatrix rows conflict-free)
constexpr int SP = 72;                 // the masked rows' weights' row stride (aligned float2 pairs)
constexpr int NS8 = 8;                 // slot tiles (top_k = 64; the 65th slot by FP32 sums)
constexpr int RPC = 2;                 // observed rows a CTA
constexpr int SA = 72;                 // the backward's alpha / beta / counts row stride (slots 0 .. 63, 64 the rest)
constexpr int SH = 72;                 // the cols pass's fp16 alpha / beta rows (halves 0 .. 63 the slots, 64 .. 67 two
                                       // floats: the rest slot and 1 / d; 144 B, rows 16-B aligned)
constexpr int CN = 128;                // children a cols CTA (eight warps x 16)
constexpr int CM = 32;                 // nodes a cols chunk
constexpr int TBUF = 20;               // the cols epilogue's transpose buffer: a row's stride (16 children + pad)
constexpr int FE = TN + 12;            // dA's staged E row (floats: the slot pairs 2t, 2t + 1 conflict-free)
constexpr float LOG2E = 1.4426950408889634f;
// the masked forward's fp16 MMA operands' scales (log2): W (< 2^W_LAZY by its lazy reference) times 2^(15 - W_LAZY), E
// (a slot's largest 1) times 2^15 -- each operand at most 2^15, fp16's range used to its top; its sums handed on with
// the reference less S_OFF (W < 2^8 in their units: the range the backward's fp16 operands take W in)
constexpr int W_LAZY_I = 1, W_OFF_I = 15 - W_LAZY_I, E_OFF_I = 15, S_OFF_I = 8 - W_LAZY_I;
constexpr float W_LAZY = (float)W_LAZY_I, W_OFF = (float)W_OFF_I;
#define NEG (-CUDART_INF_F)

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// A launch of the chain of programmatic dependents (`launch_pdl`): the launch after it may be scheduled from here on (it
// waits for this one before reading anything), and this one waits for the launch before it -- so its reads of the
// chain's buffers come after this, by plain loads: no `__ldg` or `const __restrict__`, which ptxas may issue above the
// wait (a dependent can start while the launches before it still write)
__device__ __forceinline__ void pdl_entry() {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
}

__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ void ldsm4(unsigned (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// the first k-step: the accumulator starts from l2t's fragment (fp16 pairs: row g's, row g + 8's)
__device__ __forceinline__ void mma_f16_c(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, uint2 c) {
    const float2 c0 = __half22float2(*reinterpret_cast<const __half2*>(&c.x));
    const float2 c1 = __half22float2(*reinterpret_cast<const __half2*>(&c.y));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// x rounded to TF32 as `cvt.rna` rounds a finite non-negative x (to nearest, ties away): half a TF32 ulp added to its
// bits, the MMA ignoring the low 13 (`cvt.rna.tf32.f32` itself compiles to three instructions, the inf / NaN guard)
__device__ __forceinline__ unsigned to_tf32(float x) { return __float_as_uint(x) + 0x1000u; }
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
// 2^e with b 2^e in [2^14, 2^15) (e at most 126: its inverse a normal float; 1 where b is 0 or subnormal): what takes
// values bounded by b (>= 0, finite) into fp16's range -- an fp16 MMA operand of fp32 values, the scale taken back out
// of the fp32 sums
__device__ __forceinline__ float fp16_scale(float b) {
    const int ex = (__float_as_int(b) >> 23) & 0xFF;
    return ex == 0 ? 1.f : __int_as_float(min(253, 268 - ex) << 23);
}
// (lo, hi) as an fp16 pair, rounded to nearest, SATURATED at +-65504: a value past the bound its scale was made for
// (an inf would make the MMA's sums NaN) -- one F2FP, as the plain conversion
__device__ __forceinline__ unsigned sat_h2(float lo, float hi) {
    unsigned d;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ void red1(float* p, float v) {
    asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}
// p[0, 1] += (a, b) atomically: one vector reduction where the architecture has it (sm_90) and p is 8-byte aligned
__device__ __forceinline__ void red2(float* p, float a, float b, bool pair) {
#if __CUDA_ARCH__ >= 900
    if (pair) { asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory"); return; }
#endif
    red1(p, a); red1(p + 1, b);
}
// p[0 .. 3] += v atomically (p 16-byte aligned; one vector reduction where the architecture has it)
__device__ __forceinline__ void red4(float* p, float4 v) {
#if __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
#else
    red1(p, v.x); red1(p + 1, v.y); red1(p + 2, v.z); red1(p + 3, v.w);
#endif
}

// A gradient destination tile's first writer in a backward (`grad_init_row` sets its flag 0): it claims the tile
// (0 -> 1), stores its sums over every element of it rather than adding them, and publishes 2 (`tile_release`); every
// other writer waits for the 2 and adds into them. A flag 2 from the start, or no flag: every writer adds (the tile
// initialized already). The CTA's answer (every thread calls it: a barrier).
__device__ __forceinline__ bool tile_claim(int* f) {
    __shared__ int s_first;
    if (threadIdx.x == 0) {
        int first = 0;
        if (f != nullptr) {
            first = atomicCAS(f, 0, 1) == 0;
            if (!first) {
                int v;
                do {
                    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
                    if (v != 2) __nanosleep(32);
                } while (v != 2);
            }
        }
        s_first = first;
    }
    __syncthreads();
    return s_first != 0;
}

__device__ __forceinline__ void tile_release(int* f, bool first) {
    __syncthreads();                                                         // (every thread's stores of the tile issued)
    if (first && threadIdx.x == 0) {
        __threadfence();
        atomicExch(f, 2);
    }
}

// p[0 .. 3] = v, or += v (`add`)
__device__ __forceinline__ void put4(float* p, float4 v, bool add) {
    if (add) red4(p, v);
    else *reinterpret_cast<float4*>(p) = v;
}

__device__ __forceinline__ void put2(float* p, float a, float b, bool pair, bool add) {
    if (add) { red2(p, a, b, pair); return; }
    if (pair) *reinterpret_cast<float2*>(p) = make_float2(a, b);
    else { p[0] = a; p[1] = b; }
}

__device__ __forceinline__ void put1(float* p, float v, bool add) {
    if (add) red1(p, v);
    else *p = v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// A' fragments of one row from the caller's fp32 A [B, M, r] (`p0` row g's first rank): rows g, g + 8 of the warp's
// 16, ranks (2t, 2t + 1) and (2t + 8, 2t + 9) of each 16 -- log2 e A rounded to fp16, zero past r
template <int NK>
__device__ __forceinline__ void a_frags(unsigned (&af)[NK][4], const float* __restrict__ p0, int r, int t) {
    const float* p1 = p0 + 8L * r;
    #pragma unroll
    for (int ks = 0; ks < NK; ++ks) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = ks * 16 + h * 8 + 2 * t;
            const float x0 = k < r ? p0[k] : 0.f, x1 = k + 1 < r ? p0[k + 1] : 0.f;
            const float y0 = k < r ? p1[k] : 0.f, y1 = k + 1 < r ? p1[k + 1] : 0.f;
            af[ks][2 * h] = h2u(__floats2half2_rn(x0 * LOG2E, x1 * LOG2E));
            af[ks][2 * h + 1] = h2u(__floats2half2_rn(y0 * LOG2E, y1 * LOG2E));
        }
    }
}

// The logits of one chunk into Sx (l2t's fragments the C operand of the first k-step), from the staged B chunk Bs
template <int NK>
__device__ __forceinline__ void logits(float (&Sx)[8][4], const unsigned (&af)[NK][4], const __half* Bs, const uint2 (&lt)[8],
                                       int lane) {
    #pragma unroll
    for (int np = 0; np < 4; ++np) {
        unsigned bf[4];
        ldsm4t(bf, smem_u32(Bs + (lane & 15) * BROW + np * 16 + (lane >> 4) * 8));
        mma_f16_c(Sx[2 * np], af[0], bf[0], bf[1], lt[2 * np]);
        mma_f16_c(Sx[2 * np + 1], af[0], bf[2], bf[3], lt[2 * np + 1]);
    }
    #pragma unroll
    for (int ks = 1; ks < NK; ++ks) {
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
            unsigned bf[4];
            ldsm4t(bf, smem_u32(Bs + (ks * 16 + (lane & 15)) * BROW + np * 16 + (lane >> 4) * 8));
            mma_f16(Sx[2 * np], af[ks], bf[0], bf[1]);
            mma_f16(Sx[2 * np + 1], af[ks], bf[2], bf[3]);
        }
    }
}

// The rows' reference maxima (log2 units) moved -- `rescale(a0, a1)` applied to the accumulators -- only where the
// chunk's exceeds them by more than 2^lazy (LAZY: W = 2^(logits - reference) stays below 2^lazy, the sums far from
// fp32's range); then 2^off W = 2^(logits - reference + off) in place.
template <typename F>
__device__ __forceinline__ void exp_tile(float (&Sx)[8][4], float& m0, float& m1, F&& rescale, float off = 0.f,
                                         float lazy = 8.f) {
    float x0 = NEG, x1 = NEG;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        x0 = fmaxf(x0, fmaxf(Sx[nt][0], Sx[nt][1]));
        x1 = fmaxf(x1, fmaxf(Sx[nt][2], Sx[nt][3]));
    }
    x0 = quad_max(x0); x1 = quad_max(x1);
    const bool up0 = x0 > m0 + lazy, up1 = x1 > m1 + lazy;
    if (__any_sync(0xffffffffu, up0 || up1)) {
        rescale(up0 ? exp2f(m0 - x0) : 1.f, up1 ? exp2f(m1 - x1) : 1.f);
        m0 = up0 ? x0 : m0; m1 = up1 ? x1 : m1;
    }
    const float h0 = (m0 == NEG ? 0.f : m0) - off, h1 = (m1 == NEG ? 0.f : m1) - off;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        Sx[nt][0] = exp2f(Sx[nt][0] - h0); Sx[nt][1] = exp2f(Sx[nt][1] - h0);
        Sx[nt][2] = exp2f(Sx[nt][2] - h1); Sx[nt][3] = exp2f(Sx[nt][3] - h1);
    }
}

// a masked stage: { B chunk [KP][BROW] fp16, E16 chunk [64][BROW] fp16, E rows [3][TN] fp32, ev chunks [3][TN] fp32 }
template <int NK>
struct MaskedGeo {
    static constexpr int KP = 16 * NK;
    static constexpr int B_BYTES = KP * BROW * 2;
    static constexpr int H_BYTES = 8 * NS8 * BROW * 2;    // (the MMAs' slots, fp16: ldmatrix rows conflict-free)
    static constexpr int R_BYTES = 3 * TN * 4;            // (the rest slot's row, the partners' slots' rows)
    static constexpr int STAGE = B_BYTES + H_BYTES + R_BYTES + 3 * TN * 4;
    static constexpr int BYTES = 2 * STAGE;
};

template <int NK>
struct ObservedGeo {                   // a stage: 2 slots x { B chunk [KP][BROW] fp16, E rows [2][TN] fp32, ev [2][TN] fp32 }
    static constexpr int KP = 16 * NK;
    static constexpr int B_BYTES = KP * BROW * 2;
    static constexpr int ROW = B_BYTES + 4 * TN * 4;
    static constexpr int STAGE = 2 * ROW;
    static constexpr int BYTES = 2 * STAGE;
};

// A pass of a node vector is two persistent launches (`tk_sk_kernel`, STREAM-K) side by side on two streams, the masked
// items' tiles (128 nodes x an item) and the observed items', 16 chunks of 64 children each: the SMs split between the
// two kinds by their work (WM, WO: a chunk's cost each, measured in place) -- each launch a CTA an SM, the CTAs past
// its kind's share leaving at once (the counts are the device's) -- and each kind's tiles cut into one equal share a
// CTA: no wave's tail. (One launch of both kinds, weighted by their costs, was SLOWER: its code at the 255-register cap spilled
// in the observed loop, 13.7K instructions; the partners' sums stretched the masked tiles.) A tile cut between CTAs: its earlier parts'
// sums are PARTIALS (each CTA's at most one, in the scratch `ws`, a flag set), the CTA with its last chunk adds them in
// (each part's running maxima its own: rescaled to the larger) and makes the values. A CTA runs its partial first and
// waits on lower CTAs only (dispatched first): no deadlock, short waits.

constexpr int WM = 11, WO = 12;        // a masked chunk's cost and an observed one's (with the stash ending: the best
                                       // Codpula head's two kinds took 113 : 108 us a transition at 11 : 12; 5 : 5 left
                                       // the observed share 118 us, 5 : 6 the masked one 115)

struct MState {                         // a thread's sums of a masked tile: its rows g, g + 8
    float m0, m1, zr0, zr1, ur0, ur1;
    float Zc[NS8][4], Uc[NS8][4];
    float pz[2][2], pu[2][2];
};
struct OState {                         // ... of an observed tile: two slots x two samples x its rows
    float m[2][2], zs[2][2][2], us[2][2][2];
};
constexpr int WS_M = 6 + 8 * NS8 * 2 + 8, WS_O = 4 + 16;   // a partial's floats a thread: masked, observed

struct TkArgs {
    const float* A; const __half* BP; const uint2* LTh; const float* E; const float* ev; const float* cmax;
    const __half* E16;                  // [Bx, 64, N] E's slots 0 .. 63 times 2^15, fp16 (the masked items' MMAs)
    const int4* mitems; const int4* oitems; const int* noitems; const int* slot32; const float* logw;
    float* out; float* ws; int* flags;
    long nid0, ostride, sA; int r, M, N, S, Rm;                    // sA: A's batch stride (its rows contiguous)
    // the STASH ending's (what a backward takes from the forward's sums; `out` may be null then: no values): the
    // slots' posterior [Rm, M, SA] (a masked sample's), ao, bo [B, M], W's shift h [B, M] (every sample's); a masked
    // row's alpha, beta [Rm, M, SH] fp16: the slots times a node's d (a power of 2: `masked_stash`), the rest slot
    // times d and 1 / d in the row's tail; 1 / d again [Rm, M]
    float* acnt; float* ao; float* bo; float* hrow;
    __half* alpha16; __half* beta16; float* invd;
};

// A node's power of 2 d taking its largest alpha m into [2^14, 2^15) (fp16's range; beta <= alpha): 1 where m is 0
__device__ __forceinline__ float node_scale(float m) {
    if (!(m > 0.f) || !isfinite(m)) return 1.f;
    return ldexpf(1.f, max(-126, min(126, 14 - ilogbf(m))));
}

// the masked item y's sums over its tile x's chunks c0 .. c1 - 1
template <int NK>
__device__ __forceinline__ void masked_run(MState& St, const TkArgs& a, int y, int x, int c0, int c1,
                                           unsigned char* smem) {
    using G = MaskedGeo<NK>;
    constexpr int KP = G::KP;
    const int4 item = a.mitems[y];
    const int b = item.x, e = item.y, o1 = item.z, o2 = item.w;
    const int s1 = o1 >= 0 ? a.slot32[o1] : 0, s2 = o2 >= 0 ? a.slot32[o2] : 0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, S = a.S, nchunks = N / TN;
    const int node0 = x * TM + warp * 16 + g;

    constexpr int NB = KP * (TN / 8), NH = 8 * NS8 * (TN / 8), NR = 3 * (TN / 4), NSEG = NB + NH + NR + 3 * (TN / 4);
    auto load_seg = [&](int q, int st, int j0) {
        const unsigned bs = smem_u32(smem + st * G::STAGE);
        if (q < NB) {
            const int k = q / (TN / 8), s = q % (TN / 8);
            cp16(bs + (k * BROW + s * 8) * 2, a.BP + ((long)e * KP + k) * N + j0 + s * 8);
        } else if (q < NB + NH) {                                            // (the MMAs' slots, fp16)
            const int sl = (q - NB) / (TN / 8), s = (q - NB) % (TN / 8);
            cp16(bs + G::B_BYTES + (sl * BROW + s * 8) * 2, a.E16 + ((long)e * (8 * NS8) + sl) * N + j0 + s * 8);
        } else if (q < NB + NH + NR) {                                       // (the rest slot's row, the partners')
            const int v = (q - NB - NH) / (TN / 4), s = (q - NB - NH) % (TN / 4);
            if (v > 0 && (v == 1 ? o1 : o2) < 0) return;
            const int srow = v == 0 ? S - 1 : (v == 1 ? s1 : s2);
            cp16(bs + G::B_BYTES + G::H_BYTES + (v * TN + s * 4) * 4, a.E + ((long)e * S + srow) * N + j0 + s * 4);
        } else if (q < NSEG) {
            const int v = (q - NB - NH - NR) / (TN / 4), s = (q - NB - NH - NR) % (TN / 4);
            const int row = v == 0 ? b : (v == 1 ? o1 : o2);
            if (row >= 0)
                cp16(bs + G::B_BYTES + G::H_BYTES + G::R_BYTES + (v * TN + s * 4) * 4, a.ev + (long)row * N + j0 + s * 4);
        }
    };
    __syncthreads();                                                         // (the last tile's stages read by all)
    for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, 0, c0 * TN);
    cp_commit();

    unsigned af[NK][4];
    a_frags<NK>(af, a.A + (long)e * a.sA + (long)node0 * a.r, a.r, t);
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    uint2 lt[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + (c0 * 8 + nt) * 32);

    St.m0 = St.m1 = NEG;
    St.zr0 = St.zr1 = St.ur0 = St.ur1 = 0.f;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn)
        #pragma unroll
        for (int k = 0; k < 4; ++k) St.Zc[sn][k] = St.Uc[sn][k] = 0.f;
    #pragma unroll
    for (int k = 0; k < 2; ++k) St.pz[k][0] = St.pz[k][1] = St.pu[k][0] = St.pu[k][1] = 0.f;

    for (int jc = c0; jc < c1; ++jc) {
        const int st = (jc - c0) & 1;
        const bool more = jc + 1 < c1;
        cp_wait<0>();
        __syncthreads();                     // (this chunk's stage complete; the other free: every warp past the last chunk)
        if (more) {
            for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, st ^ 1, (jc + 1) * TN);
            cp_commit();
        }
        const unsigned char* p = smem + st * G::STAGE;
        const float* Rs = reinterpret_cast<const float*>(p + G::B_BYTES + G::H_BYTES);
        const float* Vs = reinterpret_cast<const float*>(p + G::B_BYTES + G::H_BYTES + G::R_BYTES);
        // (this lane's row of the slots' ldmatrix: matrices slots 16q .. +7, +8 .. +15 x children 16k .. +7, +8 .. +15)
        const unsigned hb = smem_u32(p + G::B_BYTES) + (((lane & 7) + ((lane >> 4) << 3)) * BROW + ((lane >> 3) & 1) * 8) * 2;

        float Sx[8][4];
        logits<NK>(Sx, af, reinterpret_cast<const __half*>(p), lt, lane);
        exp_tile(Sx, St.m0, St.m1, [&](float a0, float a1) {
            #pragma unroll
            for (int sn = 0; sn < NS8; ++sn) {
                St.Zc[sn][0] *= a0; St.Zc[sn][1] *= a0; St.Zc[sn][2] *= a1; St.Zc[sn][3] *= a1;
                St.Uc[sn][0] *= a0; St.Uc[sn][1] *= a0; St.Uc[sn][2] *= a1; St.Uc[sn][3] *= a1;
            }
            St.zr0 *= a0; St.ur0 *= a0; St.zr1 *= a1; St.ur1 *= a1;
            #pragma unroll
            for (int k = 0; k < 2; ++k) { St.pz[k][0] *= a0; St.pu[k][0] *= a0; St.pz[k][1] *= a1; St.pu[k][1] *= a1; }
        }, W_OFF, W_LAZY);

        #pragma unroll
        for (int ks = 0; ks < TN / 16; ++ks) {                                 // (a k-step of 16 children: n-tiles 2ks, 2ks + 1)
            if (more) {
                lt[2 * ks] = __ldg(ltp + ((jc + 1) * 8 + 2 * ks) * 32);
                lt[2 * ks + 1] = __ldg(ltp + ((jc + 1) * 8 + 2 * ks + 1) * 32);
            }
            float u[2][4];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float2 ec = *reinterpret_cast<const float2*>(Vs + (2 * ks + h) * 8 + 2 * t);
                u[h][0] = Sx[2 * ks + h][0] * ec.x; u[h][1] = Sx[2 * ks + h][1] * ec.y;
                u[h][2] = Sx[2 * ks + h][2] * ec.x; u[h][3] = Sx[2 * ks + h][3] * ec.y;
            }
            // the logits' fragments of two n-tiles ARE the m16n8k16 A fragment (rows g, g + 8 x children 2t, 2t + 1 of
            // each): W and W ev in fp16
            const float (&w0)[4] = Sx[2 * ks], (&w1)[4] = Sx[2 * ks + 1];
            const unsigned az[4] = {sat_h2(w0[0], w0[1]), sat_h2(w0[2], w0[3]), sat_h2(w1[0], w1[1]), sat_h2(w1[2], w1[3])};
            const unsigned au[4] = {sat_h2(u[0][0], u[0][1]), sat_h2(u[0][2], u[0][3]), sat_h2(u[1][0], u[1][1]),
                                    sat_h2(u[1][2], u[1][3])};
            #pragma unroll
            for (int q = 0; q < NS8 / 2; ++q) {                              // (two slot tiles a load)
                unsigned bf[4];
                ldsm4(bf, hb + (q * 16 * BROW + ks * 16) * 2);
                mma_f16(St.Zc[2 * q], az, bf[0], bf[1]);
                mma_f16(St.Uc[2 * q], au, bf[0], bf[1]);
                mma_f16(St.Zc[2 * q + 1], az, bf[2], bf[3]);
                mma_f16(St.Uc[2 * q + 1], au, bf[2], bf[3]);
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nt = 2 * ks + h;
                const float w00 = Sx[nt][0], w01 = Sx[nt][1], w10 = Sx[nt][2], w11 = Sx[nt][3];
                const float u00 = u[h][0], u01 = u[h][1], u10 = u[h][2], u11 = u[h][3];
                const float2 rr = *reinterpret_cast<const float2*>(Rs + nt * 8 + 2 * t);
                St.zr0 += w00 * rr.x + w01 * rr.y; St.zr1 += w10 * rr.x + w11 * rr.y;
                St.ur0 += u00 * rr.x + u01 * rr.y; St.ur1 += u10 * rr.x + u11 * rr.y;
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    if ((k == 0 ? o1 : o2) < 0) continue;
                    const float2 pr = *reinterpret_cast<const float2*>(Rs + (1 + k) * TN + nt * 8 + 2 * t);
                    const float2 pv = *reinterpret_cast<const float2*>(Vs + (1 + k) * TN + nt * 8 + 2 * t);
                    const float er0 = pv.x * pr.x, er1 = pv.y * pr.y;
                    St.pz[k][0] += w00 * pr.x + w01 * pr.y; St.pz[k][1] += w10 * pr.x + w11 * pr.y;
                    St.pu[k][0] += w00 * er0 + w01 * er1; St.pu[k][1] += w10 * er0 + w11 * er1;
                }
            }
        }
    }
    // (the sums handed on in the units of W = 2^(logits - reference) and E, the reference less S_OFF: the MMAs' sums
    // times 2^(S_OFF - W_OFF - E_OFF), the FP32 sums' times 2^(S_OFF - W_OFF) -- powers of 2, exact)
    constexpr float UZ = 1.f / (float)(1 << (W_OFF_I + E_OFF_I - S_OFF_I)), UR = 1.f / (float)(1 << (W_OFF_I - S_OFF_I));
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn)
        #pragma unroll
        for (int k = 0; k < 4; ++k) { St.Zc[sn][k] *= UZ; St.Uc[sn][k] *= UZ; }
    St.zr0 *= UR; St.zr1 *= UR; St.ur0 *= UR; St.ur1 *= UR;
    #pragma unroll
    for (int k = 0; k < 2; ++k) { St.pz[k][0] *= UR; St.pz[k][1] *= UR; St.pu[k][0] *= UR; St.pu[k][1] *= UR; }
    St.m0 = St.m0 == NEG ? NEG : St.m0 - S_OFF_I; St.m1 = St.m1 == NEG ? NEG : St.m1 - S_OFF_I;
}

// the masked item y's values over its tile x from its sums: log sum_s w U / Z + c; the partners' log u / z + c
__device__ __forceinline__ void masked_values(const MState& St, const TkArgs& a, int y, int x) {
    const int4 item = a.mitems[y];
    const int b = item.x, o1 = item.z, o2 = item.w;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int M = a.M, node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    const float* lw0 = a.logw + ((long)y * M + node0) * SP + 2 * t;
    const float* lw1 = lw0 + 8L * SP;
    float2 w0[NS8], w1[NS8];
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        w0[sn] = __ldg(reinterpret_cast<const float2*>(lw0 + sn * 8));
        w1[sn] = __ldg(reinterpret_cast<const float2*>(lw1 + sn * 8));
    }
    const float wr0 = __ldg(lw0 - 2 * t + 8 * NS8), wr1 = __ldg(lw1 - 2 * t + 8 * NS8);
    const float zr0 = quad_sum(St.zr0), zr1 = quad_sum(St.zr1), ur0 = quad_sum(St.ur0), ur1 = quad_sum(St.ur1);
    float R0 = 0.f, R1 = 0.f;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        if (St.Zc[sn][0] > 0.f) R0 += __expf(w0[sn].x) * (St.Uc[sn][0] / St.Zc[sn][0]);
        if (St.Zc[sn][1] > 0.f) R0 += __expf(w0[sn].y) * (St.Uc[sn][1] / St.Zc[sn][1]);
        if (St.Zc[sn][2] > 0.f) R1 += __expf(w1[sn].x) * (St.Uc[sn][2] / St.Zc[sn][2]);
        if (St.Zc[sn][3] > 0.f) R1 += __expf(w1[sn].y) * (St.Uc[sn][3] / St.Zc[sn][3]);
    }
    R0 = quad_sum(R0); R1 = quad_sum(R1);
    if (zr0 > 0.f) R0 += __expf(wr0) * (ur0 / zr0);
    if (zr1 > 0.f) R1 += __expf(wr1) * (ur1 / zr1);
    float q0[2], q1[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const float z0 = quad_sum(St.pz[k][0]), z1 = quad_sum(St.pz[k][1]);
        const float u0 = quad_sum(St.pu[k][0]), u1 = quad_sum(St.pu[k][1]);
        q0[k] = z0 > 0.f ? u0 / z0 : 0.f;
        q1[k] = z1 > 0.f ? u1 / z1 : 0.f;
    }
    if (t == 0) {
        float* out = a.out + a.nid0 * a.ostride;
        const float c = a.cmax[b];
        out[node0 * a.ostride + b] = (R0 > 0.f && c > NEG) ? __logf(R0) + c : NEG;
        out[node1 * a.ostride + b] = (R1 > 0.f && c > NEG) ? __logf(R1) + c : NEG;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int o = k == 0 ? o1 : o2;
            if (o < 0) continue;
            const float co = a.cmax[o];
            out[node0 * a.ostride + o] = (q0[k] > 0.f && co > NEG) ? __logf(q0[k]) + co : NEG;
            out[node1 * a.ostride + o] = (q1[k] > 0.f && co > NEG) ? __logf(q1[k]) + co : NEG;
        }
    }
}

// the observed item y's sums over its tile x's chunks c0 .. c1 - 1
template <int NK>
__device__ __forceinline__ void observed_run(OState& St, const TkArgs& a, int y, int x, int c0, int c1,
                                             unsigned char* smem) {
    using G = ObservedGeo<NK>;
    constexpr int KP = G::KP;
    const int4 sa = a.oitems[2 * y], sb = a.oitems[2 * y + 1];
    const int ek[2] = {sa.x, sb.x};
    const int rk[2][2] = {{sa.y, sa.z}, {sb.y, sb.z}};
    const bool two = sb.x >= 0;
    int sk[2][2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j) sk[k][j] = rk[k][j] >= 0 ? a.slot32[rk[k][j]] : 0;
    const bool same[2] = {rk[0][1] >= 0 && sk[0][0] == sk[0][1], rk[1][1] >= 0 && sk[1][0] == sk[1][1]};
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, t = lane & 3, g = lane >> 2;
    const int N = a.N, S = a.S, nchunks = N / TN;
    const int node0 = x * TM + warp * 16 + g;

    constexpr int NB = KP * (TN / 8), NSR = NB + 4 * (TN / 4), NSEG = 2 * NSR;
    auto load_seg = [&](int q, int st, int j0) {
        const int k = q / NSR, qq = q % NSR;
        const int ee = k == 0 ? ek[0] : ek[1];
        if (ee < 0) return;
        const unsigned bs = smem_u32(smem + st * G::STAGE + k * G::ROW);
        if (qq < NB) {
            const int kk = qq / (TN / 8), s = qq % (TN / 8);
            cp16(bs + (kk * BROW + s * 8) * 2, a.BP + ((long)ee * KP + kk) * N + j0 + s * 8);
        } else {
            const int v = (qq - NB) / (TN / 4), s = (qq - NB) % (TN / 4);      // v: E row 0, 1, ev 0, 1
            const int j = v & 1;
            const int row = k == 0 ? (j == 0 ? rk[0][0] : rk[0][1]) : (j == 0 ? rk[1][0] : rk[1][1]);
            if (row < 0) return;
            const int sl = k == 0 ? (j == 0 ? sk[0][0] : sk[0][1]) : (j == 0 ? sk[1][0] : sk[1][1]);
            const float* src = v < 2 ? a.E + ((long)ee * S + sl) * N : a.ev + (long)row * N;
            cp16(bs + G::B_BYTES + (v * TN + s * 4) * 4, src + j0 + s * 4);
        }
    };
    __syncthreads();
    for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, 0, c0 * TN);
    cp_commit();

    unsigned af[2][NK][4];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        if (k == 0 || two) a_frags<NK>(af[k], a.A + (long)ek[k] * a.sA + (long)node0 * a.r, a.r, t);
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    uint2 lt[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + (c0 * 8 + nt) * 32);

    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        St.m[k][0] = St.m[k][1] = NEG;
        #pragma unroll
        for (int j = 0; j < 2; ++j) St.zs[k][j][0] = St.zs[k][j][1] = St.us[k][j][0] = St.us[k][j][1] = 0.f;
    }

    for (int jc = c0; jc < c1; ++jc) {
        const int st = (jc - c0) & 1;
        const bool more = jc + 1 < c1;
        cp_wait<0>();
        __syncthreads();
        if (more) {
            for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, st ^ 1, (jc + 1) * TN);
            cp_commit();
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (k == 1 && !two) continue;                                    // (a constant trip count: the state in registers)
            const unsigned char* p = smem + st * G::STAGE + k * G::ROW;
            const float* Es = reinterpret_cast<const float*>(p + G::B_BYTES);
            float Sx[8][4];
            logits<NK>(Sx, af[k], reinterpret_cast<const __half*>(p), lt, lane);
            if (more && (k == 1 || !two)) {                                  // (l2t read by the last slot: the next chunk's)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + ((jc + 1) * 8 + nt) * 32);
            }
            exp_tile(Sx, St.m[k][0], St.m[k][1], [&](float a0, float a1) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    St.zs[k][j][0] *= a0; St.us[k][j][0] *= a0; St.zs[k][j][1] *= a1; St.us[k][j][1] *= a1;
                }
            });
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (rk[k][j] < 0) continue;
                const bool zj = !(j == 1 && same[k]);
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const float2 rr = *reinterpret_cast<const float2*>(Es + j * TN + nt * 8 + 2 * t);
                    const float2 e2 = *reinterpret_cast<const float2*>(Es + (2 + j) * TN + nt * 8 + 2 * t);
                    const float er0 = e2.x * rr.x, er1 = e2.y * rr.y;
                    if (zj) {
                        St.zs[k][j][0] += Sx[nt][0] * rr.x + Sx[nt][1] * rr.y;
                        St.zs[k][j][1] += Sx[nt][2] * rr.x + Sx[nt][3] * rr.y;
                    }
                    St.us[k][j][0] += Sx[nt][0] * er0 + Sx[nt][1] * er1;
                    St.us[k][j][1] += Sx[nt][2] * er0 + Sx[nt][3] * er1;
                }
            }
        }
    }
}

__device__ __forceinline__ void observed_values(const OState& St, const TkArgs& a, int y, int x) {
    const int4 sa = a.oitems[2 * y], sb = a.oitems[2 * y + 1];
    const int rk[2][2] = {{sa.y, sa.z}, {sb.y, sb.z}};
    const bool two = sb.x >= 0;
    int sk[2][2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j) sk[k][j] = rk[k][j] >= 0 ? a.slot32[rk[k][j]] : 0;
    const bool same[2] = {rk[0][1] >= 0 && sk[0][0] == sk[0][1], rk[1][1] >= 0 && sk[1][0] == sk[1][1]};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    float* out = a.out + a.nid0 * a.ostride;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = rk[k][j];
            const bool z_of_0 = j == 1 && same[k];                           // (a select, not an index: in registers)
            const float z0 = quad_sum(z_of_0 ? St.zs[k][0][0] : St.zs[k][j][0]);
            const float z1 = quad_sum(z_of_0 ? St.zs[k][0][1] : St.zs[k][j][1]);
            const float u0 = quad_sum(St.us[k][j][0]), u1 = quad_sum(St.us[k][j][1]);
            if (t == 0 && row >= 0 && (k == 0 || two)) {
                const float R0 = z0 > 0.f ? u0 / z0 : 0.f, R1 = z1 > 0.f ? u1 / z1 : 0.f;
                const float c = a.cmax[row];
                out[node0 * a.ostride + row] = (R0 > 0.f && c > NEG) ? __logf(R0) + c : NEG;
                out[node1 * a.ostride + row] = (R1 > 0.f && c > NEG) ? __logf(R1) + c : NEG;
            }
        }
    }
}

// The STASH ending of a masked item y's tile x -- what the backward takes from the forward's sums, the node flows not
// in it (they scale a node's row: `tk_flows_kernel`, the backward's kernels): the slots' posterior pi = w (U / Z) /
// sum_s w (U / Z), alpha = pi / U, beta = pi / Z (the masked sample's rows; the slot after the rest's: 1 where the node
// has mass, the place of its flow), its ao 1 where the node has mass (what the flows kernel reads); a partner's ao = 1 / u,
// bo = 1 / z where u / z > 0; W's shift h.
__device__ __forceinline__ void masked_stash(const MState& St, const TkArgs& a, int y, int x) {
    const int4 item = a.mitems[y];
    const int b = item.x, o1 = item.z, o2 = item.w;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int M = a.M, node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    const float* lw0 = a.logw + ((long)y * M + node0) * SP + 2 * t;
    const float* lw1 = lw0 + 8L * SP;
    auto ratio = [](float u, float z) { return z > 0.f ? u / z : 0.f; };
    auto sdiv = [](float x_, float y_) { return y_ > 0.f ? x_ / y_ : 0.f; };
    float R0 = 0.f, R1 = 0.f;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        const float2 u = __ldg(reinterpret_cast<const float2*>(lw0 + sn * 8));
        const float2 v = __ldg(reinterpret_cast<const float2*>(lw1 + sn * 8));
        R0 += __expf(u.x) * ratio(St.Uc[sn][0], St.Zc[sn][0]) + __expf(u.y) * ratio(St.Uc[sn][1], St.Zc[sn][1]);
        R1 += __expf(v.x) * ratio(St.Uc[sn][2], St.Zc[sn][2]) + __expf(v.y) * ratio(St.Uc[sn][3], St.Zc[sn][3]);
    }
    R0 = quad_sum(R0); R1 = quad_sum(R1);
    const float zr0 = quad_sum(St.zr0), zr1 = quad_sum(St.zr1), ur0 = quad_sum(St.ur0), ur1 = quad_sum(St.ur1);
    const float wr0 = __expf(__ldg(lw0 - 2 * t + 8 * NS8)) * ratio(ur0, zr0);
    const float wr1 = __expf(__ldg(lw1 - 2 * t + 8 * NS8)) * ratio(ur1, zr1);
    R0 += wr0; R1 += wr1;
    const float c0 = sdiv(1.f, R0), c1 = sdiv(1.f, R1);
    const long q0 = ((long)y * M + node0) * SA + 2 * t, q1 = q0 + 8L * SA;
    float al[2][2 * NS8], be[2][2 * NS8];                                // (the thread's 16 slots of its two nodes)
    float mx0 = 0.f, mx1 = 0.f;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        const float2 u = __ldg(reinterpret_cast<const float2*>(lw0 + sn * 8));
        const float2 v = __ldg(reinterpret_cast<const float2*>(lw1 + sn * 8));
        const float a00 = __expf(u.x) * ratio(St.Uc[sn][0], St.Zc[sn][0]) * c0;
        const float a01 = __expf(u.y) * ratio(St.Uc[sn][1], St.Zc[sn][1]) * c0;
        const float a10 = __expf(v.x) * ratio(St.Uc[sn][2], St.Zc[sn][2]) * c1;
        const float a11 = __expf(v.y) * ratio(St.Uc[sn][3], St.Zc[sn][3]) * c1;
        *reinterpret_cast<float2*>(a.acnt + q0 + sn * 8) = make_float2(a00, a01);
        *reinterpret_cast<float2*>(a.acnt + q1 + sn * 8) = make_float2(a10, a11);
        al[0][2 * sn] = sdiv(a00, St.Uc[sn][0]); al[0][2 * sn + 1] = sdiv(a01, St.Uc[sn][1]);
        al[1][2 * sn] = sdiv(a10, St.Uc[sn][2]); al[1][2 * sn + 1] = sdiv(a11, St.Uc[sn][3]);
        be[0][2 * sn] = sdiv(a00, St.Zc[sn][0]); be[0][2 * sn + 1] = sdiv(a01, St.Zc[sn][1]);
        be[1][2 * sn] = sdiv(a10, St.Zc[sn][2]); be[1][2 * sn + 1] = sdiv(a11, St.Zc[sn][3]);
        mx0 = fmaxf(mx0, fmaxf(al[0][2 * sn], al[0][2 * sn + 1]));
        mx1 = fmaxf(mx1, fmaxf(al[1][2 * sn], al[1][2 * sn + 1]));
    }
    // alpha, beta (the backward's fp16 operands): a node's slots times its d (`node_scale` of its largest alpha; beta
    // takes alpha's)
    const float d0 = node_scale(quad_max(mx0)), d1 = node_scale(quad_max(mx1));
    __half* const A16 = a.alpha16 + ((long)y * M + node0) * SH;
    __half* const B16 = a.beta16 + ((long)y * M + node0) * SH;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        const int s = sn * 8 + 2 * t;
        *reinterpret_cast<__half2*>(A16 + s) = __floats2half2_rn(al[0][2 * sn] * d0, al[0][2 * sn + 1] * d0);
        *reinterpret_cast<__half2*>(A16 + 8 * SH + s) = __floats2half2_rn(al[1][2 * sn] * d1, al[1][2 * sn + 1] * d1);
        *reinterpret_cast<__half2*>(B16 + s) = __floats2half2_rn(be[0][2 * sn] * d0, be[0][2 * sn + 1] * d0);
        *reinterpret_cast<__half2*>(B16 + 8 * SH + s) = __floats2half2_rn(be[1][2 * sn] * d1, be[1][2 * sn + 1] * d1);
    }
    float pz[2][2], pu[2][2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int h = 0; h < 2; ++h) { pz[k][h] = quad_sum(St.pz[k][h]); pu[k][h] = quad_sum(St.pu[k][h]); }
    if (t == 0) {
        const long r0 = q0 - 2 * t + 8 * NS8, r1 = q1 - 2 * t + 8 * NS8;          // (the rest slot)
        const float l0 = R0 > 0.f ? 1.f : 0.f, l1 = R1 > 0.f ? 1.f : 0.f;         // (the node has mass)
        a.acnt[r0] = wr0 * c0; a.acnt[r1] = wr1 * c1;
        a.acnt[r0 + 1] = l0; a.acnt[r1 + 1] = l1;
        // (the rows' tail: the rest slot times d, 1 / d)
        float* const ta0 = reinterpret_cast<float*>(A16 + 8 * NS8);
        float* const ta1 = reinterpret_cast<float*>(A16 + 8 * SH + 8 * NS8);
        float* const tb0 = reinterpret_cast<float*>(B16 + 8 * NS8);
        float* const tb1 = reinterpret_cast<float*>(B16 + 8 * SH + 8 * NS8);
        ta0[0] = sdiv(wr0 * c0, ur0) * d0; ta0[1] = 1.f / d0; ta1[0] = sdiv(wr1 * c1, ur1) * d1; ta1[1] = 1.f / d1;
        tb0[0] = sdiv(wr0 * c0, zr0) * d0; tb0[1] = 1.f / d0; tb1[0] = sdiv(wr1 * c1, zr1) * d1; tb1[1] = 1.f / d1;
        a.invd[(long)y * M + node0] = 1.f / d0; a.invd[(long)y * M + node1] = 1.f / d1;
        const float h0 = St.m0 == NEG ? 0.f : St.m0, h1 = St.m1 == NEG ? 0.f : St.m1;
        a.hrow[(long)b * M + node0] = h0; a.hrow[(long)b * M + node1] = h1;
        a.ao[(long)b * M + node0] = l0; a.ao[(long)b * M + node1] = l1;
        a.bo[(long)b * M + node0] = 0.f; a.bo[(long)b * M + node1] = 0.f;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int o = k == 0 ? o1 : o2;
            if (o < 0) continue;
            const float f0 = ratio(pu[k][0], pz[k][0]) > 0.f ? 1.f : 0.f;
            const float f1 = ratio(pu[k][1], pz[k][1]) > 0.f ? 1.f : 0.f;
            a.ao[(long)o * M + node0] = sdiv(f0, pu[k][0]); a.ao[(long)o * M + node1] = sdiv(f1, pu[k][1]);
            a.bo[(long)o * M + node0] = sdiv(f0, pz[k][0]); a.bo[(long)o * M + node1] = sdiv(f1, pz[k][1]);
            a.hrow[(long)o * M + node0] = h0; a.hrow[(long)o * M + node1] = h1;
        }
    }
}

// ... of an observed item y's tile x: each sample's ao = 1 / u, bo = 1 / z where u / z > 0, W's shift
__device__ __forceinline__ void observed_stash(const OState& St, const TkArgs& a, int y, int x) {
    const int4 sa = a.oitems[2 * y], sb = a.oitems[2 * y + 1];
    const int rk[2][2] = {{sa.y, sa.z}, {sb.y, sb.z}};
    const bool two = sb.x >= 0;
    int sk[2][2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j) sk[k][j] = rk[k][j] >= 0 ? a.slot32[rk[k][j]] : 0;
    const bool same[2] = {rk[0][1] >= 0 && sk[0][0] == sk[0][1], rk[1][1] >= 0 && sk[1][0] == sk[1][1]};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int M = a.M, node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    auto sdiv = [](float x_, float y_) { return y_ > 0.f ? x_ / y_ : 0.f; };
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = rk[k][j];
            const bool z_of_0 = j == 1 && same[k];
            const float z0 = quad_sum(z_of_0 ? St.zs[k][0][0] : St.zs[k][j][0]);
            const float z1 = quad_sum(z_of_0 ? St.zs[k][0][1] : St.zs[k][j][1]);
            const float u0 = quad_sum(St.us[k][j][0]), u1 = quad_sum(St.us[k][j][1]);
            if (t == 0 && row >= 0 && (k == 0 || two)) {
                const float f0 = sdiv(u0, z0) > 0.f ? 1.f : 0.f;
                const float f1 = sdiv(u1, z1) > 0.f ? 1.f : 0.f;
                a.ao[(long)row * M + node0] = sdiv(f0, u0); a.ao[(long)row * M + node1] = sdiv(f1, u1);
                a.bo[(long)row * M + node0] = sdiv(f0, z0); a.bo[(long)row * M + node1] = sdiv(f1, z1);
                a.hrow[(long)row * M + node0] = St.m[k][0] == NEG ? 0.f : St.m[k][0];
                a.hrow[(long)row * M + node1] = St.m[k][1] == NEG ? 0.f : St.m[k][1];
            }
        }
    }
}

// a partial's floats as one list, by value (`f(q, v, row)` returns the new value -- no member's address taken: a
// reference into the state kept it in local memory): the masked state's (rows -1: maxima; 0, 1: row g's, row g + 8's
// sums), the observed one's (maxima -1 .. -4; sums by their slot and row 0 .. 3)
template <typename F>
__device__ __forceinline__ void each_float(MState& S, F&& f) {
    S.m0 = f(0, S.m0, -1); S.m1 = f(1, S.m1, -1);
    S.zr0 = f(2, S.zr0, 0); S.zr1 = f(3, S.zr1, 1); S.ur0 = f(4, S.ur0, 0); S.ur1 = f(5, S.ur1, 1);
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            S.Zc[sn][k] = f(6 + (sn * 4 + k) * 2, S.Zc[sn][k], k >> 1);
            S.Uc[sn][k] = f(7 + (sn * 4 + k) * 2, S.Uc[sn][k], k >> 1);
        }
    }
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            S.pz[k][h] = f(6 + 8 * NS8 + (k * 2 + h) * 2, S.pz[k][h], h);
            S.pu[k][h] = f(7 + 8 * NS8 + (k * 2 + h) * 2, S.pu[k][h], h);
        }
    }
}
template <typename F>
__device__ __forceinline__ void each_float(OState& S, F&& f) {
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        S.m[k][0] = f(2 * k, S.m[k][0], -1);
        S.m[k][1] = f(2 * k + 1, S.m[k][1], -1);
    }
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int q = 4 + ((k * 2 + j) * 2 + h) * 2;
                S.zs[k][j][h] = f(q, S.zs[k][j][h], 2 * k + h);
                S.us[k][j][h] = f(q + 1, S.us[k][j][h], 2 * k + h);
            }
}

// the partial into this CTA's slot of the scratch, then its kind's flag
// a CTA's partial slot: the masked kind's from the scratch's start, the observed kind's after gridDim.x masked slots
__device__ __forceinline__ float* ws_slot(const TkArgs& a, int kp, bool masked) {
    return a.ws + (masked ? (long)kp * WS_M : (long)gridDim.x * WS_M + (long)kp * WS_O) * (NW * 32) + threadIdx.x;
}

template <typename St>
__device__ __forceinline__ void publish(St& S, const TkArgs& a, int* flags) {
    float* w = ws_slot(a, blockIdx.x, std::is_same<St, MState>::value);
    each_float(S, [&](int q, float v, int) { __stcg(w + q * (NW * 32), v); return v; });
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(flags + blockIdx.x, 1);
}

__device__ __forceinline__ void wait_flag(const int* f) {
    if (threadIdx.x == 0) {
        int v;
        do {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
            if (v == 0) __nanosleep(64);
        } while (v == 0);
    }
    __syncthreads();
}

// CTA kp's partial added into S: each row's sums rescaled to the larger of the two maxima
__device__ __forceinline__ void fold_in(MState& S, const TkArgs& a, int kp) {
    const float* w = ws_slot(a, kp, true);
    const float pm0 = __ldcg(w), pm1 = __ldcg(w + NW * 32);
    const float n0 = fmaxf(S.m0, pm0), n1 = fmaxf(S.m1, pm1);
    const float fs0 = n0 == NEG ? 1.f : exp2f(S.m0 - n0), fs1 = n1 == NEG ? 1.f : exp2f(S.m1 - n1);
    const float fp0 = n0 == NEG ? 1.f : exp2f(pm0 - n0), fp1 = n1 == NEG ? 1.f : exp2f(pm1 - n1);
    each_float(S, [&](int q, float v, int row) {
        if (row < 0) return v;
        return v * (row == 0 ? fs0 : fs1) + __ldcg(w + q * (NW * 32)) * (row == 0 ? fp0 : fp1);
    });
    S.m0 = n0; S.m1 = n1;
}
__device__ __forceinline__ void fold_in(OState& S, const TkArgs& a, int kp) {
    const float* w = ws_slot(a, kp, false);
    float fs0, fs1, fs2, fs3, fp0, fp1, fp2, fp3, nm0, nm1, nm2, nm3;
    auto pair = [&](float mine, float theirs, float& nm, float& fs, float& fp) {
        nm = fmaxf(mine, theirs);
        fs = nm == NEG ? 1.f : exp2f(mine - nm);
        fp = nm == NEG ? 1.f : exp2f(theirs - nm);
    };
    pair(S.m[0][0], __ldcg(w), nm0, fs0, fp0);
    pair(S.m[0][1], __ldcg(w + NW * 32), nm1, fs1, fp1);
    pair(S.m[1][0], __ldcg(w + 2 * NW * 32), nm2, fs2, fp2);
    pair(S.m[1][1], __ldcg(w + 3 * NW * 32), nm3, fs3, fp3);
    each_float(S, [&](int q, float v, int row) {
        if (row < 0) return v;
        const float a_ = row == 0 ? fs0 : (row == 1 ? fs1 : (row == 2 ? fs2 : fs3));
        const float b_ = row == 0 ? fp0 : (row == 1 ? fp1 : (row == 2 ? fp2 : fp3));
        return v * a_ + __ldcg(w + q * (NW * 32)) * b_;
    });
    S.m[0][0] = nm0; S.m[0][1] = nm1; S.m[1][0] = nm2; S.m[1][1] = nm3;
}

template <int NK, bool MASKED, bool STASH>
__global__ void __launch_bounds__(NW * 32, 1) tk_sk_kernel(const TkArgs a) {
    extern __shared__ __align__(128) unsigned char smem[];
    using St = typename std::conditional<MASKED, MState, OState>::type;
    const int nx = a.M / TM, nch = a.N / TN;
    const long Tm = (long)nx * a.Rm, To = (long)nx * (*a.noitems);
    // the SMs each kind takes: its share of both kinds' cost (all where the other has none), at most a CTA a tile -- a
    // kind's tiles split only where its share is short of them (a split's prologue and partial cost ~4 us a part)
    const long cm = Tm * WM, co = To * WO, G = gridDim.x;
    long Pm = cm == 0 ? 0 : (co == 0 ? G : max(1L, min(G - 1, (G * cm + (cm + co) / 2) / (cm + co))));
    long Po = G - Pm;
    Pm = min(Pm, Tm);
    Po = min(Po, To);
    const int P = (int)(MASKED ? Pm : Po), k = blockIdx.x;
    if (k >= P) return;
    const long T = MASKED ? Tm : To;
    const long W = T * nch;
    auto gstart = [&](long kk) -> long { return W * kk / P; };
    const long g0 = gstart(k), g1 = gstart(k + 1);
    if (g0 >= g1) return;
    const long ta = g0 / nch, tb = (g1 - 1) / nch;
    const int ca0 = (int)(g0 % nch), cb1 = (int)((g1 - 1) % nch) + 1;
    const bool part = cb1 < nch;                                             // (the share ends inside a tile)
    const long last = part ? tb - 1 : tb;                                    // (its last whole tile after the first)
    const int nmid = (int)(last > ta ? last - ta : 0);                       // (the whole tiles after the first)
    const int nseg = (part && tb == ta) ? 1 : (part ? 1 : 0) + 1 + nmid;
    int* flags = a.flags + (MASKED ? 0 : gridDim.x);                         // (each kind its own: by the grid, not the share)
    // the order: the partial (published at once: the next CTA waits on it), the whole tiles, the first tile ta last --
    // its last part waits on the lower CTAs' partials, done by then (first, it waited up to a tile's time: a share
    // starting at a tile's last chunk waited on its 15-chunk partial before its own work, 40 us a call)
    for (int si = 0; si < nseg; ++si) {
        long tt; int c0, c1, mode;                                           // mode 0 whole, 1 partial, 2 the tile's last part
        if (part && si == 0) {
            tt = tb; c0 = tb == ta ? ca0 : 0; c1 = cb1; mode = 1;
        } else {
            const int j = si - (part ? 1 : 0);
            if (j < nmid) { tt = ta + 1 + j; c0 = 0; c1 = nch; mode = 0; }
            else { tt = ta; c0 = ca0; c1 = ta == tb ? cb1 : nch; mode = ca0 > 0 ? 2 : 0; }
        }
        const int item = (int)(tt / nx), x = (int)(tt % nx);
        St S;
        if constexpr (MASKED) masked_run<NK>(S, a, item, x, c0, c1, smem);
        else observed_run<NK>(S, a, item, x, c0, c1, smem);
        if (mode == 1) {
            publish(S, a, flags);
            continue;
        }
        if (mode == 2)
            for (int kp = k - 1; kp >= 0; --kp) {
                const long s0 = gstart(kp);
                if (s0 >= gstart(kp + 1)) continue;                          // (an empty share)
                wait_flag(flags + kp);
                fold_in(S, a, kp);
                if (s0 <= tt * nch) break;                                   // (the tile's first part)
            }
        if constexpr (MASKED) {
            if (a.out != nullptr) masked_values(S, a, item, x);
            if constexpr (STASH) masked_stash(S, a, item, x);
        } else {
            if (a.out != nullptr) observed_values(S, a, item, x);
            if constexpr (STASH) observed_stash(S, a, item, x);
        }
    }
}

// The operands, one launch: blockIdx.y the task, blockIdx.x grid-stride over its items. B and G a row of tensors each
// (Bx of them), the children's values a sample each (B).
__global__ void tk_prep_kernel(
        const float* Bm, long sB, const float* Gt, long sG, const float* em,
        long em_stride,
        const long* eid, const bool* has, const float* theta,
        const float* logwin, __half* BP, float* E, __half* E16, float* ev,
        float* cmax, uint2* LTh, float* logw,
        int Bx, int B, int M, int N, int r, int KP, int S, int Rm, int* flags, int nflags) {
    pdl_entry();
    const int task = blockIdx.y;
    const long tid0 = (long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long)gridDim.x * blockDim.x;
    if (task == 0 && blockIdx.x == 0)                                        // (the forward's stream-K flags)
        for (int i = threadIdx.x; i < nflags; i += blockDim.x) flags[i] = 0;
    const long wid = tid0 >> 5, nwarp = nth >> 5;
    const int lane = threadIdx.x & 31;
    if (task == 0) {                                                         // B, fp16, its ranks padded
        const long total = (long)Bx * KP * (N / 2);
        for (long q = tid0; q < total; q += nth) {
            const int j2 = (int)(q % (N / 2));
            const long bk = q / (N / 2);
            const int k = (int)(bk % KP);
            const long bb = bk / KP;
            float2 v = make_float2(0.f, 0.f);
            if (k < r) v = reinterpret_cast<const float2*>(Bm + bb * sB + (long)k * N)[j2];
            reinterpret_cast<__half2*>(BP)[q] = __floats2half2_rn(v.x, v.y);
        }
    } else if (task == 1) {                                                  // E = exp(G - max) in TF32, a warp a row
        for (long row = wid; row < (long)Bx * S; row += nwarp) {
            const float* gr = Gt + (row / S) * sG + (row % S) * N;
            float mx = NEG;
            for (int j = lane; j < N; j += 32) mx = fmaxf(mx, gr[j]);
            mx = warp_max(mx);
            const float sh = isfinite(mx) ? mx : 0.f;
            // (the MMAs' slots also in fp16, times 2^E_OFF: the masked forward's operand, where it has items)
            const int sl = (int)(row % S);
            __half* hr = E16 != nullptr && sl < 8 * NS8 ? E16 + ((row / S) * (8 * NS8) + sl) * N : nullptr;
            for (int j = lane; j < N; j += 32) {
                const unsigned u = __float_as_uint(__expf(gr[j] - sh));
                const float x = __uint_as_float((u + 0x1000u) & ~0x1FFFu);
                E[row * N + j] = x;
                if (hr != nullptr) hr[j] = __float2half_rn(x * (float)(1 << E_OFF_I));
            }
        }
    } else if (task == 2) {                                                  // the children's values, a warp a sample
        for (long bb = wid; bb < B; bb += nwarp) {
            float mx = NEG;
            for (int j = lane; j < N; j += 32)
                mx = fmaxf(mx, (has == nullptr || has[j]) ? em[eid[j] * em_stride + bb] : NEG);
            mx = warp_max(mx);
            const float sh = isfinite(mx) ? mx : 0.f;
            for (int j = lane; j < N; j += 32) {
                const float v = (has == nullptr || has[j]) ? em[eid[j] * em_stride + bb] : NEG;
                ev[bb * N + j] = __expf(v - sh);
            }
            if (lane == 0) cmax[bb] = mx;
        }
    } else if (task == 3) {                                                  // l2t in fragment order, a CTA a 16-row tile
        if (theta == nullptr) return;                                        // (an earlier call's table reused)
        __shared__ float rmax[16];
        const int nch = N / TN;
        const int warp = threadIdx.x >> 5;
        for (int tile = blockIdx.x; tile < M / 16; tile += gridDim.x) {
            for (int rw = warp; rw < 16; rw += blockDim.x >> 5) {
                const float* tr = theta + ((long)tile * 16 + rw) * N;
                float mx = 0.f;
                for (int j = lane; j < N; j += 32) mx = fmaxf(mx, tr[j]);
                mx = warp_max(mx);
                if (lane == 0) rmax[rw] = mx > 0.f ? log2f(mx) : 0.f;
            }
            __syncthreads();
            for (int q = threadIdx.x; q < nch * 8 * 32; q += blockDim.x) {
                const int ln = q & 31, nt = (q >> 5) & 7, jc = q >> 8;
                const int g = ln >> 2, t = ln & 3;
                const long i = (long)tile * 16 + g;
                const int j = jc * TN + nt * 8 + 2 * t;
                const float2 u = *reinterpret_cast<const float2*>(theta + i * N + j);
                const float2 w = *reinterpret_cast<const float2*>(theta + (i + 8) * N + j);
                const float s0 = rmax[g], s1 = rmax[g + 8];
                uint2 o;
                o.x = h2u(__floats2half2_rn(log2f(u.x) - s0, log2f(u.y) - s0));
                o.y = h2u(__floats2half2_rn(log2f(w.x) - s1, log2f(w.y) - s1));
                LTh[((long)tile * nch * 8) * 32 + q] = o;
            }
            __syncthreads();
        }
    } else {                                                                 // the masked samples' weights, fp32, padded
        const long total = logwin != nullptr ? (long)Rm * M * SP : 0;      // (none to copy: given padded)
        for (long q = tid0; q < total; q += nth) {
            const int s = (int)(q % SP);
            logw[q] = s < S ? logwin[(q / SP) * S + s] : NEG;
        }
    }
}

// The work items (a CTA a node vector's): a masked item a masked sample, its row of tensors and up to two observed
// samples reading that row (its partners: the row's observed samples 2i, 2i + 1 for its i-th masked one); the row's
// other observed samples in slots of two, an observed item two slots. Every sample's place among its row's samples of
// its kind by scans in shared memory (deterministic: in the samples' order). `emap` the row each sample reads (null:
// its own); `pair` 0: no partners, a sample an item (the backward where a row's samples keep their gradients apart).
// `roff` (null: one node vector, Rm masked samples): CTA t the node vector reading leaf t, its masked samples rows
// [roff[t], roff[t + 1]), its slots slot64 [t][B], its items past the earlier leaves' (`items_offsets`).
__device__ __forceinline__ void items_offsets(const int* roff, int t, int B, long& mo, long& oo) {
    mo = 0; oo = 0;
    for (int u = 0; u < t; ++u) {
        const int rm = roff[u + 1] - roff[u];
        mo += max(rm, 1);
        oo += 2 * max((B - rm + 1) / 2, 1) + 1;
    }
}

__global__ void __launch_bounds__(1024) tk_items_kernel(
        const long* emap, const long* rows, const int* roff, const long* slot64,
        int4* mitems, int4* oitems, int* noitems, int* slot32,
        int B, int Rm, int pair) {
    pdl_entry();
    if (roff != nullptr) {
        const int t = blockIdx.x;
        long mo, oo;
        items_offsets(roff, t, B, mo, oo);
        rows += roff[t]; Rm = roff[t + 1] - roff[t];
        slot64 += (long)t * B; slot32 += (long)t * B;
        mitems += mo; oitems += oo; noitems += t;
    }
    extern __shared__ int sm[];
    int* se = sm;                                                            // [B] its group (pair: the row it reads)
    int* sk = se + B;                                                        // [B] its kind (1: masked)
    int* sp = sk + B;                                                        // [B] masked: its item
    int* sx = sp + B;                                                        // [B] its place among its row's of its kind
    int* sl = sx + B;                                                        // [B] observed: 1 a slot's first sample
    __shared__ int nslots;
    const int tid = threadIdx.x, nth = blockDim.x;
    for (int b = tid; b < B; b += nth) {
        se[b] = (emap != nullptr && pair) ? (int)emap[b] : b;
        sk[b] = 0;
        sl[b] = 0;
        slot32[b] = (int)slot64[b];
    }
    if (tid == 0) nslots = 0;
    __syncthreads();
    for (int q = tid; q < Rm; q += nth) {
        sk[rows[q]] = 1;
        sp[rows[q]] = q;
    }
    __syncthreads();
    for (int b = tid; b < B; b += nth) {
        const int e = se[b], k = sk[b];
        int ix = 0;
        for (int c = 0; c < b; ++c) ix += (se[c] == e && sk[c] == k);
        sx[b] = ix;
        if (k) mitems[sp[b]] = make_int4(b, emap != nullptr ? (int)emap[b] : b, -1, -1);
    }
    __syncthreads();
    for (int b = tid; b < B; b += nth) {
        if (sk[b]) continue;
        const int e = se[b], io = sx[b];
        int km = 0;
        for (int c = 0; c < B; ++c) km += (se[c] == e && sk[c]);
        if (io < 2 * km) {                                                   // a partner of the row's masked sample io / 2
            for (int c = 0; c < B; ++c)
                if (se[c] == e && sk[c] && sx[c] == io / 2) {
                    reinterpret_cast<int*>(mitems + sp[c])[2 + (io & 1)] = b;
                    break;
                }
        } else if (((io - 2 * km) & 1) == 0) {
            sl[b] = 1;
        }
    }
    __syncthreads();
    for (int b = tid; b < B; b += nth) {
        if (!sl[b]) continue;
        int s = 0;
        for (int c = 0; c < b; ++c) s += sl[c];
        int b2 = -1;
        for (int c = b + 1; c < B; ++c)
            if (se[c] == se[b] && !sk[c] && sx[c] == sx[b] + 1) {
                b2 = c;
                break;
            }
        oitems[s] = make_int4(emap != nullptr ? (int)emap[b] : b, b, b2, 0);
        atomicAdd(&nslots, 1);
    }
    __syncthreads();
    if (tid == 0) {
        const int ns = nslots;
        if (ns & 1) oitems[ns] = make_int4(-1, -1, -1, 0);
        *noitems = (ns + 1) / 2;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The backward's cols pass and dA's GEMM (their operands' stash is `tk_sk_kernel`'s STASH ending): the notes at the top.

struct BwArgs {                         // the cols pass's and dA's GEMM's (see `top_k_bw_grads`)
    const __half* AH;                    // [Bx, M, KP] fp16(log2 e A), the ranks zero-padded (the cols' staged chunks)
    const __half* BP;                    // [Bx, KP, N] fp16 (the prep's)
    const float* E;                      // [Bx, S, N] TF32-rounded (the prep's)
    const float* ev;                     // [B, N] the children's values over their largest (the prep's)
    const uint2* LTc;                    // l2t in the cols kernel's fragment order
    const int4* mitems; const int4* oitems; const int* noitems; const int* slot32;   // the prep's work items
    const __half* alpha16; const __half* beta16;   // [Rm, M, SH] a masked sample's alpha, beta (the stash's; its node
    const float* invd;                             // flows not in them): the slots times a node's d, the rest slot
                                                   // times d and 1 / d in the tail; [Rm, M] 1 / d again
    const float* ao; const float* bo;        // [B, M] an observed sample's
    const float* fl; const float* fmx;       // [B, M] a sample's node flows, [B] their largest (`tk_flows_kernel`)
    const float* hrow;                       // [B, M] W's shift (log2 units) a sample
    const int* drow; const float* sgn;       // [B] the row of the destinations a sample's gradients go to, its sign
    float* EF;                               // [B, N] the element flows (added into)
    const float* A; long sA;                 // the caller's A [Bx, M, r] (rows contiguous): the observed dA's A' fragments
    const uint2* LTh;                        // l2t in the forward's fragment order (the observed dA's)
    __half* GLT; float* GLS;                 // [m1 - m0, N, M] a masked item's Gl^T, dB's fp16 operand; [m1 - m0] its
                                             // scale's inverse (the cols pass's: dA's GEMM's operands)
    int m0, m1;                              // the masked items a launch takes (a group: its Gl about 2/5 of the L2)
    float* DA; long sDA;                     // [Bd, M, r] (rows contiguous; added into)
    float* DB; long sDB;                     // [Bd, r, N]
    float* DG; long sDG;                     // [Bd, S, N]
    // [Bd, M / GMN + N / CN] a destination row's tiles' flags -- its dA tiles (nodes by GMN), then its dB / dG tiles
    // (children by CN) -- where the masked passes' first writers store (`tile_claim`; set by `grad_init_row`),
    // null where every writer adds; dfa the dA tiles a row, dft all its tiles
    int* dflag; int dfa, dft;
    int r, M, N, S, Rm;
};

// --- dA of the observed slots (the masked items': a GEMM of the cols pass's Gl, below)

template <int NK>
struct DaGeo {   // a stage: B chunk [KP][BROW] fp16, the samples' slots' E rows [2][FE], their ev chunks [2][TN]
    static constexpr int KP = 16 * NK;
    static constexpr int B_BYTES = KP * BROW * 2;
    static constexpr int E_BYTES = 2 * FE * 4;
    static constexpr int STAGE = B_BYTES + E_BYTES + 2 * TN * 4;
    static constexpr int BYTES = 2 * STAGE;
};

// the logits of half a chunk (child tiles 4 hh .. 4 hh + 3) into Sx, l2t's fragments the first k-step's C operand
template <int NK>
__device__ __forceinline__ void logits4(float (&Sx)[4][4], const unsigned (&af)[NK][4], const __half* Bs,
                                        const uint2 (&lt)[8], int lane, int hh) {
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int np = 2 * hh + q;
        unsigned bf[4];
        ldsm4t(bf, smem_u32(Bs + (lane & 15) * BROW + np * 16 + (lane >> 4) * 8));
        mma_f16_c(Sx[2 * q], af[0], bf[0], bf[1], lt[2 * np]);
        mma_f16_c(Sx[2 * q + 1], af[0], bf[2], bf[3], lt[2 * np + 1]);
    }
    #pragma unroll
    for (int ks = 1; ks < NK; ++ks) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int np = 2 * hh + q;
            unsigned bf[4];
            ldsm4t(bf, smem_u32(Bs + (ks * 16 + (lane & 15)) * BROW + np * 16 + (lane >> 4) * 8));
            mma_f16(Sx[2 * q], af[ks], bf[0], bf[1]);
            mma_f16(Sx[2 * q + 1], af[ks], bf[2], bf[3]);
        }
    }
}

// One segment of the observed slots' dA pass: slot y's (oitems) node tile x, its children's chunks c0 .. c1 - 1 -- its
// samples' Gl, each times its sign, summed; dA added into the row at the end.
template <int NK>
__device__ __forceinline__ void da_segment(const BwArgs& a, int y, int x, int c0, int c1, unsigned char* smem) {
    using G = DaGeo<NK>;
    constexpr int KP = G::KP;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, M = a.M, S = a.S, r = a.r, nchunks = N / TN;
    const int4 it = a.oitems[y];
    const int e = it.x;
    int o[2] = {it.y, it.z};
    if (e < 0) return;
    const int hb = o[0];
    int so[2]; float sg[2];
    #pragma unroll
    for (int p = 0; p < 2; ++p) { so[p] = o[p] >= 0 ? a.slot32[o[p]] : 0; sg[p] = o[p] >= 0 ? a.sgn[o[p]] : 0.f; }
    const int node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    // the samples' slots' E rows (0, 1), their ev chunks (0, 1)
    constexpr int NB = KP * (TN / 8), NE = 2 * (TN / 4), NSEG = NB + NE + 2 * (TN / 4);
    auto load_seg = [&](int q, int st, int j0) {
        const unsigned bs = smem_u32(smem + st * G::STAGE);
        if (q < NB) {
            const int k = q / (TN / 8), s = q % (TN / 8);
            cp16(bs + (k * BROW + s * 8) * 2, a.BP + ((long)e * KP + k) * N + j0 + s * 8);
        } else if (q < NB + NE) {
            const int rr = (q - NB) / (TN / 4), s = (q - NB) % (TN / 4);
            if ((rr == 0 ? o[0] : o[1]) < 0) return;
            cp16(bs + G::B_BYTES + (rr * FE + s * 4) * 4, a.E + ((long)e * S + (rr == 0 ? so[0] : so[1])) * N + j0 + s * 4);
        } else if (q < NSEG) {
            const int v = (q - NB - NE) / (TN / 4), s = (q - NB - NE) % (TN / 4);
            const int row = v == 0 ? o[0] : o[1];
            if (row >= 0) cp16(bs + G::B_BYTES + G::E_BYTES + (v * TN + s * 4) * 4, a.ev + (long)row * N + j0 + s * 4);
        }
    };
    __syncthreads();
    for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, 0, c0 * TN);
    cp_commit();

    unsigned af[NK][4];
    a_frags<NK>(af, a.A + (long)e * a.sA + (long)node0 * r, r, t);
    const float h0 = a.hrow[(long)hb * M + node0], h1 = a.hrow[(long)hb * M + node1];
    float aop[2][2], bop[2][2];
    // Gl is dA's fp16 MMA operand: a node's row times a power of 2 (sc0, sc1) that takes its bound -- the flows of the
    // slot's samples there: |Gl[i, j]| <= f[i] a sample -- into fp16's range, in the samples' factors (applied LAST: W
    // alone reaches 2^8, the forward's lazy shift, and the scale 2^126 -- a flow ~1e-35 -- times it overflowed); the
    // row's sums (fp32) take it back out
    float bd0 = 0.f, bd1 = 0.f;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const long r0 = o[p] >= 0 ? (long)o[p] * M + node0 : 0;
        aop[p][0] = o[p] >= 0 ? a.ao[r0] : 0.f; aop[p][1] = o[p] >= 0 ? a.ao[r0 + 8] : 0.f;
        bop[p][0] = o[p] >= 0 ? a.bo[r0] : 0.f; bop[p][1] = o[p] >= 0 ? a.bo[r0 + 8] : 0.f;
        bd0 += o[p] >= 0 ? a.fl[r0] : 0.f; bd1 += o[p] >= 0 ? a.fl[r0 + 8] : 0.f;
    }
    const float sc0 = fp16_scale(bd0), sc1 = fp16_scale(bd1);
    const float sgs[2][2] = {{sg[0] * sc0, sg[0] * sc1}, {sg[1] * sc0, sg[1] * sc1}};
    float dA[2 * NK][4];
    #pragma unroll
    for (int q = 0; q < 2 * NK; ++q) dA[q][0] = dA[q][1] = dA[q][2] = dA[q][3] = 0.f;

    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    uint2 lt[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = *(ltp + (c0 * 8 + nt) * 32);

    for (int jc = c0; jc < c1; ++jc) {
        const int st = (jc - c0) & 1;
        const bool more = jc + 1 < c1;
        cp_wait<0>();
        __syncthreads();
        if (more) {
            for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, st ^ 1, (jc + 1) * TN);
            cp_commit();
        }
        const unsigned char* p = smem + st * G::STAGE;
        const __half* Bs = reinterpret_cast<const __half*>(p);
        const float* Es = reinterpret_cast<const float*>(p + G::B_BYTES);
        const float* Vs = reinterpret_cast<const float*>(p + G::B_BYTES + G::E_BYTES);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            float W[4][4];
            logits4<NK>(W, af, Bs, lt, lane, hh);
            if (more && hh == 1) {
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) lt[nt] = *(ltp + ((jc + 1) * 8 + nt) * 32);
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                W[q][0] = exp2f(W[q][0] - h0); W[q][1] = exp2f(W[q][1] - h0);
                W[q][2] = exp2f(W[q][2] - h1); W[q][3] = exp2f(W[q][3] - h1);
            }
            float gs[4][4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int jc0 = (4 * hh + q) * 8 + 2 * t;                     // the thread's children: jc0, jc0 + 1
                gs[q][0] = gs[q][1] = gs[q][2] = gs[q][3] = 0.f;
                #pragma unroll
                for (int pp = 0; pp < 2; ++pp) {
                    if (o[pp] < 0) continue;
                    const float2 es = *reinterpret_cast<const float2*>(Es + pp * FE + jc0);
                    const float2 eo = *reinterpret_cast<const float2*>(Vs + pp * TN + jc0);
                    gs[q][0] += W[q][0] * es.x * (aop[pp][0] * eo.x - bop[pp][0]) * sgs[pp][0];
                    gs[q][1] += W[q][1] * es.y * (aop[pp][0] * eo.y - bop[pp][0]) * sgs[pp][0];
                    gs[q][2] += W[q][2] * es.x * (aop[pp][1] * eo.x - bop[pp][1]) * sgs[pp][1];
                    gs[q][3] += W[q][3] * es.y * (aop[pp][1] * eo.y - bop[pp][1]) * sgs[pp][1];
                }
            }
            // dA [16 nodes x KP] += Gl B^T: Gl's fp16 fragment the A operand (k: the 16 children of two child tiles), B^T
            // from the staged chunk as it lies (rank-major: `ldmatrix`, two rank tiles a load)
            #pragma unroll
            for (int q = 0; q < 4; q += 2) {
                const unsigned ga[4] = {sat_h2(gs[q][0], gs[q][1]), sat_h2(gs[q][2], gs[q][3]),
                                        sat_h2(gs[q + 1][0], gs[q + 1][1]), sat_h2(gs[q + 1][2], gs[q + 1][3])};
                #pragma unroll
                for (int rt = 0; rt < 2 * NK; rt += 2) {
                    unsigned bf[4];
                    ldsm4(bf, smem_u32(Bs + ((rt + (lane >> 4)) * 8 + (lane & 7)) * BROW + (4 * hh + q) * 8 +
                                       ((lane >> 3) & 1) * 8));
                    mma_f16(dA[rt], ga, bf[0], bf[1]);
                    mma_f16(dA[rt + 1], ga, bf[2], bf[3]);
                }
            }
        }
    }
    // into the destination row, the rows' scales taken out: ranks 2t, 2t + 1 of each 8 (pairs where r is even)
    float* D0 = a.DA + (long)a.drow[hb] * a.sDA + (long)node0 * r;
    float* D1 = D0 + 8L * r;
    const bool pair = (r & 1) == 0;
    const float iv0 = 1.f / sc0, iv1 = 1.f / sc1;
    #pragma unroll
    for (int rt = 0; rt < 2 * NK; ++rt) {
        const int k = rt * 8 + 2 * t;
        if (k + 1 < r) {
            red2(D0 + k, dA[rt][0] * iv0, dA[rt][1] * iv0, pair); red2(D1 + k, dA[rt][2] * iv1, dA[rt][3] * iv1, pair);
        } else if (k < r) { red1(D0 + k, dA[rt][0] * iv0); red1(D1 + k, dA[rt][2] * iv1); }
    }
}

template <int NK>
__global__ void __launch_bounds__(NW * 32, 1) tk_bw_da_kernel(const BwArgs a) {
    extern __shared__ __align__(128) unsigned char smem[];
    pdl_entry();
    const int ntile = a.M / TM, nch = a.N / TN;
    const long U = 2L * (*a.noitems) * ntile * nch;
    const long P = gridDim.x, k = blockIdx.x;
    const long u0 = U * k / P, u1 = U * (k + 1) / P;
    for (long u = u0; u < u1;) {
        const long tt = u / nch;
        const int c0 = (int)(u % nch);
        const int c1 = (int)min((long)nch, c0 + (u1 - u));
        da_segment<NK>(a, (int)(tt / ntile), (int)(tt % ntile), c0, c1, smem);
        u += c1 - c0;
    }
}

// --- cols

template <int NK, bool MASKED>
struct ColsGeo {   // a stage: A' chunk [CM][AR] fp16, alpha, beta chunks [CM][SH] fp16 (masked), h, ao/bo of 2 samples [5][CM],
                   // the masked sample's node flows [CM]
    static constexpr int KP = 16 * NK, AR = KP + 8;
    static constexpr int A_BYTES = CM * AR * 2;
    static constexpr int AB_BYTES = MASKED ? CM * SH * 2 : 0;
    static constexpr int NV = MASKED ? 6 : 5;
    static constexpr int V_BYTES = NV * CM * 4;
    static constexpr int STAGE = A_BYTES + 2 * AB_BYTES + V_BYTES;
    // the dB^T and dG^T transposes, a buffer a warp: a row a rank, and (masked) a slot -- 64 of them whatever the ranks
    static constexpr int EROWS = MASKED && KP < 64 ? 64 : KP;
    static constexpr int EPI = NW * EROWS * TBUF * 4;
    static constexpr int BYTES = 2 * STAGE > EPI ? 2 * STAGE : EPI;
};

// One segment of the cols pass: item y's (masked: mitems; else an observed slot of oitems) children tile x, its node
// chunks c0 .. c1 - 1.
template <int NK, bool MASKED, bool GRADS>
__device__ __forceinline__ void cols_segment(const BwArgs& a, int y, int x, int c0, int c1, unsigned char* smem) {
    using G = ColsGeo<NK, MASKED>;
    constexpr int KP = G::KP, AR = G::AR;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, M = a.M, S = a.S;
    int e, bm = -1, o[2];
    if constexpr (MASKED) { const int4 it = a.mitems[y]; bm = it.x; e = it.y; o[0] = it.z; o[1] = it.w; }
    else { const int4 it = a.oitems[y]; e = it.x; o[0] = it.y; o[1] = it.z; }
    if (e < 0) return;                                                       // (an empty observed slot: the CTA's)
    const int hb = MASKED ? bm : o[0];
    int so[2]; float sg[2];
    #pragma unroll
    for (int p = 0; p < 2; ++p) { so[p] = o[p] >= 0 ? a.slot32[o[p]] : 0; sg[p] = o[p] >= 0 ? a.sgn[o[p]] : 0.f; }
    // Gl is dB's fp16 MMA operand (k the nodes): the item's times a power of 2 that takes its bound -- its samples'
    // largest flows: |Gl[i, j]| <= f[i] a sample -- into fp16's range, in the samples' signs (applied last, as dA's);
    // dB's sums take it back out
    float bd = MASKED ? a.fmx[bm] : 0.f;
    #pragma unroll
    for (int p = 0; p < 2; ++p) bd += o[p] >= 0 ? a.fmx[o[p]] : 0.f;
    const float sci = GRADS ? fp16_scale(bd) : 1.f;
    const float sm = MASKED ? a.sgn[bm] : 0.f, sms = sm * sci;
    const float sgs[2] = {sg[0] * sci, sg[1] * sci};
    const int jw = x * CN + warp * 16;                                       // the warp's first child
    // (a masked item's dB operand, Gl^T, stored -- dA's GEMM's, a group's items' --, and its scale's inverse, by the
    // segment that starts the item)
    __half* glt0 = MASKED && GRADS ? a.GLT + ((long)(y - a.m0) * N + jw + g) * M : nullptr;
    if (MASKED && GRADS && x == 0 && c0 == 0 && tid == 0) a.GLS[y - a.m0] = 1.f / sci;
    const int j0 = jw + g, j1 = j0 + 8;                                      // the thread's (rows of W^T)

    constexpr int NA = CM * (KP / 8), NAB = MASKED ? CM * (SH / 8) : 0, NV = CM / 4;
    constexpr int NSEG = NA + 2 * NAB + G::NV * NV;
    auto load_seg = [&](int qq, int st, int i0) {
        const unsigned bs = smem_u32(smem + st * G::STAGE);
        if (qq < NA) {
            const int row = qq / (KP / 8), s = qq % (KP / 8);
            cp16(bs + (row * AR + s * 8) * 2, a.AH + ((long)e * M + i0 + row) * KP + s * 8);
        } else if (qq < NA + 2 * NAB) {
            const int k2 = (qq - NA) / NAB, qr = (qq - NA) % NAB, row = qr / (SH / 8), s = qr % (SH / 8);
            const __half* src = (k2 == 0 ? a.alpha16 : a.beta16) + ((long)y * M + i0 + row) * SH + s * 8;
            cp16(bs + G::A_BYTES + k2 * G::AB_BYTES + (row * SH + s * 8) * 2, src);
        } else {
            // h, ao 0, bo 0, ao 1, bo 1, the masked sample's flows
            const int v = (qq - NA - 2 * NAB) / NV, s = (qq - NA - 2 * NAB) % NV;
            const float* src;
            if (v == 0) src = a.hrow + (long)hb * M;
            else if (v == 5) src = a.fl + (long)bm * M;
            else {
                const int op = v < 3 ? o[0] : o[1];                          // (selects: a run-time index would
                if (op < 0) return;                                          // keep o on the stack)
                src = ((v & 1) ? a.ao : a.bo) + (long)op * M;
            }
            cp16(bs + G::A_BYTES + 2 * G::AB_BYTES + (v * CM + s * 4) * 4, src + i0 + s * 4);
        }
    };
    __syncthreads();                                                         // (the last segment's buffers read by all)
    for (int qq = tid; qq < NSEG; qq += NW * 32) load_seg(qq, 0, c0 * CM);
    cp_commit();

    // B^T: the logits' A operand (rows: the thread's children; k: ranks 2t, 2t + 1 (+ 8) of each 16)
    unsigned bfr[NK][4];
    {
        const __half* B0 = a.BP + (long)e * KP * N;
        #pragma unroll
        for (int ks = 0; ks < NK; ++ks)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int k = ks * 16 + h * 8 + 2 * t;
                bfr[ks][2 * h] = h2u(__halves2half2(B0[(long)k * N + j0], B0[(long)(k + 1) * N + j0]));
                bfr[ks][2 * h + 1] = h2u(__halves2half2(B0[(long)k * N + j1], B0[(long)(k + 1) * N + j1]));
            }
    }
    const float* E0 = a.E + (long)e * S * N;
    // E^T: X^T's and Y^T's fp16 A operand (rows: the children; k: 16 slots a step; E <= 1 a slot's largest)
    unsigned efr[MASKED ? NS8 / 2 : 1][4];
    float er0 = 0.f, er1 = 0.f, evm0 = 0.f, evm1 = 0.f;
    if constexpr (MASKED) {
        #pragma unroll
        for (int ks = 0; ks < NS8 / 2; ++ks) {
            const int s = ks * 16 + 2 * t;
            efr[ks][0] = h2u(__floats2half2_rn(E0[(long)s * N + j0], E0[(long)(s + 1) * N + j0]));
            efr[ks][1] = h2u(__floats2half2_rn(E0[(long)s * N + j1], E0[(long)(s + 1) * N + j1]));
            efr[ks][2] = h2u(__floats2half2_rn(E0[(long)(s + 8) * N + j0], E0[(long)(s + 9) * N + j0]));
            efr[ks][3] = h2u(__floats2half2_rn(E0[(long)(s + 8) * N + j1], E0[(long)(s + 9) * N + j1]));
        }
        er0 = E0[(long)(8 * NS8) * N + j0]; er1 = E0[(long)(8 * NS8) * N + j1];
        evm0 = a.ev[(long)bm * N + j0]; evm1 = a.ev[(long)bm * N + j1];
    }
    float evo[2][2], eso[2][2];
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        evo[p][0] = o[p] >= 0 ? a.ev[(long)o[p] * N + j0] : 0.f;
        evo[p][1] = o[p] >= 0 ? a.ev[(long)o[p] * N + j1] : 0.f;
        eso[p][0] = o[p] >= 0 ? E0[(long)so[p] * N + j0] : 0.f;
        eso[p][1] = o[p] >= 0 ? E0[(long)so[p] * N + j1] : 0.f;
    }
    // dG's fp16 operands' scale cs, the segment's power of 2: the masked sample's node flows times 1 / d over its
    // nodes, below 2^7 after it -- A = W ev fl / d cs and -W fl / d cs then below 2^15 (W <= 2^8: fp16's range used,
    // its small values kept off the subnormals), B the stash's alpha, beta x d (< 2^15) -- taken back out (1 / cs) at
    // the epilogue
    float cs = 1.f;
    if constexpr (MASKED && GRADS) {
        __shared__ float s_mx[NW];
        float m_ = 0.f;
        for (int i = c0 * CM + tid; i < c1 * CM; i += NW * 32)
            m_ = fmaxf(m_, a.fl[(long)bm * M + i] * a.invd[(long)y * M + i]);
        #pragma unroll
        for (int w_ = 16; w_ >= 1; w_ >>= 1) m_ = fmaxf(m_, __shfl_xor_sync(0xffffffffu, m_, w_));
        if (lane == 0) s_mx[warp] = m_;
        __syncthreads();
        #pragma unroll
        for (int w_ = 0; w_ < NW; ++w_) m_ = fmaxf(m_, s_mx[w_]);
        cs = (m_ > 0.f && isfinite(m_)) ? ldexpf(1.f, max(-126, min(126, 6 - ilogbf(m_)))) : 1.f;
    }
    const float evc0 = evm0 * cs, evc1 = evm1 * cs;

    float dB[GRADS ? 2 * NK : 1][4], dG[GRADS && MASKED ? NS8 : 1][4];
    #pragma unroll
    for (int q = 0; q < (GRADS ? 2 * NK : 1); ++q) dB[q][0] = dB[q][1] = dB[q][2] = dB[q][3] = 0.f;
    #pragma unroll
    for (int q = 0; q < (GRADS && MASKED ? NS8 : 1); ++q) dG[q][0] = dG[q][1] = dG[q][2] = dG[q][3] = 0.f;
    float ef0 = 0.f, ef1 = 0.f, dgr0 = 0.f, dgr1 = 0.f;
    float po[2][2] = {{0.f, 0.f}, {0.f, 0.f}}, ro[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

    const int nchunks = M / CM;
    const uint2* ltp = a.LTc + ((long)(x * NW + warp) * (M / 8)) * 32 + lane;   // the warp's child tile, node tiles in order
    uint2 lt[4];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) lt[nt] = *(ltp + (c0 * 4 + nt) * 32);

    for (int ic = c0; ic < c1; ++ic) {
        const int st = (ic - c0) & 1;
        const bool more = ic + 1 < c1;
        cp_wait<0>();
        __syncthreads();
        if (more) {
            for (int qq = tid; qq < NSEG; qq += NW * 32) load_seg(qq, st ^ 1, (ic + 1) * CM);
            cp_commit();
        }
        const unsigned char* p = smem + st * G::STAGE;
        const __half* As = reinterpret_cast<const __half*>(p);
        const __half* Als = reinterpret_cast<const __half*>(p + G::A_BYTES);
        const __half* Bes = reinterpret_cast<const __half*>(p + G::A_BYTES + G::AB_BYTES);
        auto tail = [](const __half* row, int k) { return *reinterpret_cast<const float*>(row + 8 * NS8 + 2 * k); };
        const float* Vs = reinterpret_cast<const float*>(p + G::A_BYTES + 2 * G::AB_BYTES);

        // W^T [16 children x 32 nodes]: 4 node tiles, l2t's fragments the first k-step's C operand
        float W[4][4];
        #pragma unroll
        for (int np = 0; np < 2; ++np) {
            #pragma unroll
            for (int ks = 0; ks < NK; ++ks) {
                unsigned bf[4];
                const int bl = lane >> 3;
                ldsm4(bf, smem_u32(As + (16 * np + 8 * (bl >> 1) + (lane & 7)) * AR + ks * 16 + 8 * (bl & 1)));
                if (ks == 0) {
                    mma_f16_c(W[2 * np], bfr[0], bf[0], bf[1], lt[2 * np]);
                    mma_f16_c(W[2 * np + 1], bfr[0], bf[2], bf[3], lt[2 * np + 1]);
                } else {
                    mma_f16(W[2 * np], bfr[ks], bf[0], bf[1]);
                    mma_f16(W[2 * np + 1], bfr[ks], bf[2], bf[3]);
                }
            }
        }
        if (more) {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) lt[nt] = *(ltp + ((ic + 1) * 4 + nt) * 32);
        }
        // the shift a node: W = 2^(logits - h)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float2 h = *reinterpret_cast<const float2*>(Vs + nt * 8 + 2 * t);
            W[nt][0] = exp2f(W[nt][0] - h.x); W[nt][1] = exp2f(W[nt][1] - h.y);
            W[nt][2] = exp2f(W[nt][2] - h.x); W[nt][3] = exp2f(W[nt][3] - h.y);
        }

        unsigned ga[4];                                                      // (Gl^T's fp16 fragment: two node tiles)
        float wfe[4] = {0.f, 0.f, 0.f, 0.f};                                 // (the even node tile's wf: dG's pairs)
        // a node tile at a time (its operands consumed at once: the four tiles' kept the code over the register cap):
        // X^T, Y^T [16 children x 8 nodes] (masked) = E^T alpha^T, E^T beta^T, slots 0 .. 63 by MMAs, the rest by FMAs;
        // the elementwise terms (c: 0 (j0, i0), 1 (j0, i1), 2 (j1, i0), 3 (j1, i1); i0 = 8 nt + 2t, i1 = i0 + 1); dB^T,
        // dG^T's MMAs
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int i0 = nt * 8 + 2 * t;
            float gs[4] = {0.f, 0.f, 0.f, 0.f}, wf[4] = {0.f, 0.f, 0.f, 0.f};
            if constexpr (MASKED) {
                // X^T, Y^T times d (alpha, beta as staged: x d) by fp16 MMAs, two k-steps a load (rows: the nodes)
                float X[4] = {0.f, 0.f, 0.f, 0.f}, Y[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kp = 0; kp < NS8 / 4; ++kp) {
                    const int lr = (nt * 8 + (lane & 7)) * SH + (4 * kp + (lane >> 3)) * 8;
                    unsigned bf[4];
                    ldsm4(bf, smem_u32(Als + lr));
                    mma_f16(X, efr[2 * kp], bf[0], bf[1]);
                    mma_f16(X, efr[2 * kp + 1], bf[2], bf[3]);
                    if constexpr (GRADS) {
                        ldsm4(bf, smem_u32(Bes + lr));
                        mma_f16(Y, efr[2 * kp], bf[0], bf[1]);
                        mma_f16(Y, efr[2 * kp + 1], bf[2], bf[3]);
                    }
                }
                const float arx[2] = {tail(Als + i0 * SH, 0), tail(Als + (i0 + 1) * SH, 0)};     // (the rest slot x d)
                float brx[2] = {0.f, 0.f};
                if constexpr (GRADS) { brx[0] = tail(Bes + i0 * SH, 0); brx[1] = tail(Bes + (i0 + 1) * SH, 0); }
                // (the masked sample's terms, each linear in W: W times its nodes' flows -- not in alpha, beta -- and
                // 1 / d, which takes X, Y and the rest slot's x d back out)
                const float2 fn = *reinterpret_cast<const float2*>(Vs + 5 * CM + i0);
                const float fx = fn.x * tail(Als + i0 * SH, 1), fy = fn.y * tail(Als + (i0 + 1) * SH, 1);
                wf[0] = W[nt][0] * fx; wf[1] = W[nt][1] * fy; wf[2] = W[nt][2] * fx; wf[3] = W[nt][3] * fy;
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float w = wf[c];
                    const float erj = c < 2 ? er0 : er1, evj = c < 2 ? evm0 : evm1;
                    const float xx = X[c] + erj * arx[c & 1];
                    const float wev = w * evj, qv = wev * xx;
                    if (c < 2) ef0 += qv; else ef1 += qv;
                    if constexpr (GRADS) {
                        const float yy = Y[c] + erj * brx[c & 1];
                        gs[c] = (qv - w * yy) * sms;
                        if (c < 2) dgr0 += wev * arx[c & 1] - w * brx[c & 1];
                        else dgr1 += wev * arx[c & 1] - w * brx[c & 1];
                    }
                }
            }
            #pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                if (o[pp] < 0) continue;
                const float2 aov = *reinterpret_cast<const float2*>(Vs + (1 + 2 * pp) * CM + i0);
                const float2 bov = *reinterpret_cast<const float2*>(Vs + (2 + 2 * pp) * CM + i0);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float w = W[nt][c];
                    const float al = (c & 1) ? aov.y : aov.x, be = (c & 1) ? bov.y : bov.x;
                    const int jj = c >> 1;
                    po[pp][jj] += al * w;
                    if constexpr (GRADS) {
                        ro[pp][jj] += be * w;
                        gs[c] += w * eso[pp][jj] * (al * evo[pp][jj] - be) * sgs[pp];
                    }
                }
            }
            if constexpr (GRADS) {
                // dB^T [16 children x KP] += Gl^T A': Gl's fp16 fragment the A operand (k: the 16 nodes of two node tiles),
                // A' from the staged chunk by `ldmatrix.trans` (two rank tiles a load)
                ga[2 * (nt & 1)] = sat_h2(gs[0], gs[1]);
                ga[2 * (nt & 1) + 1] = sat_h2(gs[2], gs[3]);
                if constexpr (MASKED) {                                      // (rows j0, j1: nodes i0, i0 + 1)
                    *reinterpret_cast<unsigned*>(glt0 + (long)ic * CM + i0) = ga[2 * (nt & 1)];
                    *reinterpret_cast<unsigned*>(glt0 + 8L * M + (long)ic * CM + i0) = ga[2 * (nt & 1) + 1];
                }
                if (nt & 1) {
                    #pragma unroll
                    for (int rt = 0; rt < 2 * NK; rt += 2) {
                        unsigned bf[4];
                        ldsm4t(bf, smem_u32(As + ((nt - 1 + ((lane >> 3) & 1)) * 8 + (lane & 7)) * AR + (rt + (lane >> 4)) * 8));
                        mma_f16(dB[rt], ga, bf[0], bf[1]);
                        mma_f16(dB[rt + 1], ga, bf[2], bf[3]);
                    }
                }
                if constexpr (MASKED) {
                    // dG^T [16 children x 64 slots] x cs += (W ev)^T alpha - W^T beta over two node tiles (k: the even
                    // tile's 8 nodes, then the odd one's), fp16: A from wf (its flows and 1 / d in it) x cs, B the
                    // staged alpha, beta (x d) by `ldmatrix.trans` (two slot tiles a load)
                    if ((nt & 1) == 0) {
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) wfe[c] = wf[c];
                    } else {
                        const unsigned aw[4] = {sat_h2(wfe[0] * evc0, wfe[1] * evc0), sat_h2(wfe[2] * evc1, wfe[3] * evc1),
                                                sat_h2(wf[0] * evc0, wf[1] * evc0), sat_h2(wf[2] * evc1, wf[3] * evc1)};
                        const unsigned an[4] = {sat_h2(-wfe[0] * cs, -wfe[1] * cs), sat_h2(-wfe[2] * cs, -wfe[3] * cs),
                                                sat_h2(-wf[0] * cs, -wf[1] * cs), sat_h2(-wf[2] * cs, -wf[3] * cs)};
                        const int m4 = lane >> 3;
                        const int lr = ((nt - 1 + (m4 & 1)) * 8 + (lane & 7)) * SH + (m4 >> 1) * 8;
                        #pragma unroll
                        for (int sp = 0; sp < NS8; sp += 2) {
                            unsigned ba[4], bb[4];
                            ldsm4t(ba, smem_u32(Als + lr + sp * 8));
                            ldsm4t(bb, smem_u32(Bes + lr + sp * 8));
                            mma_f16(dG[sp], aw, ba[0], ba[1]);
                            mma_f16(dG[sp], an, bb[0], bb[1]);
                            mma_f16(dG[sp + 1], aw, ba[2], ba[3]);
                            mma_f16(dG[sp + 1], an, bb[2], bb[3]);
                        }
                    }
                }
            }
        }
    }

    // the segment's sums into their destinations
    ef0 = quad_sum(ef0); ef1 = quad_sum(ef1);
    #pragma unroll
    for (int pp = 0; pp < 2; ++pp) {
        po[pp][0] = quad_sum(po[pp][0]); po[pp][1] = quad_sum(po[pp][1]);
        if constexpr (GRADS) { ro[pp][0] = quad_sum(ro[pp][0]); ro[pp][1] = quad_sum(ro[pp][1]); }
    }
    if constexpr (GRADS && MASKED) { dgr0 = quad_sum(dgr0); dgr1 = quad_sum(dgr1); }
    if (t == 0) {
        if constexpr (MASKED) { red1(a.EF + (long)bm * N + j0, ef0); red1(a.EF + (long)bm * N + j1, ef1); }
        #pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
            if (o[pp] < 0) continue;
            red1(a.EF + (long)o[pp] * N + j0, eso[pp][0] * evo[pp][0] * po[pp][0]);
            red1(a.EF + (long)o[pp] * N + j1, eso[pp][1] * evo[pp][1] * po[pp][1]);
        }
    }
    if constexpr (GRADS) {
        const int drw = a.drow[hb];
        float* DGr = a.DG + (long)drw * a.sDG;
        // (a masked item's segment writes the whole dB / dG tile (drw, x): its first writer stores it, the others add --
        // after that store; the observed slots' terms, added, after the masked rows they land in: `tile_claim`)
        cp_wait<0>();
        int* flag = MASKED && a.dflag != nullptr ? a.dflag + (long)drw * a.dft + a.dfa + x : nullptr;
        const bool add = !(MASKED && tile_claim(flag));                      // (its barrier: every warp past the stages)
        if constexpr (!MASKED) __syncthreads();                              // (every warp past the stages)
        if constexpr (MASKED) {
            if (t == 0) {
                put1(DGr + (long)(8 * NS8) * N + j0, sm * er0 * dgr0, add);
                put1(DGr + (long)(8 * NS8) * N + j1, sm * er1 * dgr1, add);
            }
        }
        // dB^T and dG^T through a buffer a warp (transposed: a rank's / a slot's 16 children adjacent), v4 rows
        float* buf = reinterpret_cast<float*>(smem) + warp * (G::EROWS * TBUF);
        const float SC = 1.f / (LOG2E * sci);                                // (A' = log2 e A; Gl's scale)
        #pragma unroll
        for (int rt = 0; rt < 2 * NK; ++rt) {
            const int k = rt * 8 + 2 * t;
            buf[k * TBUF + g] = dB[rt][0] * SC; buf[(k + 1) * TBUF + g] = dB[rt][1] * SC;
            buf[k * TBUF + g + 8] = dB[rt][2] * SC; buf[(k + 1) * TBUF + g + 8] = dB[rt][3] * SC;
        }
        __syncwarp();
        float* DBr = a.DB + (long)drw * a.sDB + jw;
        for (int qq = lane; qq < a.r * 4; qq += 32) {
            const int k = qq >> 2, cc = qq & 3;
            put4(DBr + (long)k * N + 4 * cc, *reinterpret_cast<const float4*>(buf + k * TBUF + 4 * cc), add);
        }
        if constexpr (MASKED) {
            __syncwarp();
            #pragma unroll
            for (int sn = 0; sn < NS8; ++sn) {
                const int s = sn * 8 + 2 * t;
                buf[s * TBUF + g] = dG[sn][0]; buf[(s + 1) * TBUF + g] = dG[sn][1];
                buf[s * TBUF + g + 8] = dG[sn][2]; buf[(s + 1) * TBUF + g + 8] = dG[sn][3];
            }
            __syncwarp();
            // dG = E (.) [(W ev)^T alpha - W^T beta]^T, times the masked sample's sign (and 1 / cs: its operands' scale)
            const float smc = sm / cs;
            for (int qq = lane; qq < 8 * NS8 * 4; qq += 32) {
                const int s = qq >> 2, cc = qq & 3;
                const float4 v = *reinterpret_cast<const float4*>(buf + s * TBUF + 4 * cc);
                const float4 ee = *reinterpret_cast<const float4*>(E0 + (long)s * N + jw + 4 * cc);
                put4(DGr + (long)s * N + jw + 4 * cc,
                     make_float4(smc * ee.x * v.x, smc * ee.y * v.y, smc * ee.z * v.z, smc * ee.w * v.w), add);
            }
            __syncwarp();                                                    // (the warp's rows before its slots' adds)
        }
        if (t == 0) {
            #pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                if (o[pp] < 0) continue;
                red1(DGr + (long)so[pp] * N + j0, sg[pp] * eso[pp][0] * (evo[pp][0] * po[pp][0] - ro[pp][0]));
                red1(DGr + (long)so[pp] * N + j1, sg[pp] * eso[pp][1] * (evo[pp][1] * po[pp][1] - ro[pp][1]));
            }
        }
        if constexpr (MASKED) tile_release(flag, !add);
    }
}

template <int NK, bool MASKED, bool GRADS>
__global__ void __launch_bounds__(NW * 32, 1) tk_bw_cols_kernel(const BwArgs a) {
    extern __shared__ __align__(128) unsigned char smem[];
    pdl_entry();
    const int ntile = a.N / CN, nch = a.M / CM;
    const long items = MASKED ? a.m1 - a.m0 : 2L * (*a.noitems);
    const long U = items * ntile * nch;
    const long P = gridDim.x, k = blockIdx.x;
    const long u0 = U * k / P, u1 = U * (k + 1) / P;
    for (long u = u0; u < u1;) {
        const long tt = u / nch;
        const int c0 = (int)(u % nch);
        const int c1 = (int)min((long)nch, c0 + (u1 - u));
        cols_segment<NK, MASKED, GRADS>(a, (MASKED ? a.m0 : 0) + (int)(tt / ntile), (int)(tt % ntile), c0, c1, smem);
        u += c1 - c0;
    }
}



// --- dA: a GEMM of the items' Gl (the cols pass's, stored) and their rows' B

constexpr int GMN = 128, GKC = 64, GWN = 4;  // nodes a CTA (4 warps x 32), children a stage, warps a CTA

template <int NK>
struct GemmGeo {   // a stage: Gl^T chunk [GKC][GMN + 8] fp16, B chunk [KP][GKC + 8] fp16; three stages where they fit
    static constexpr int KP = 16 * NK, GTS = GMN + 8, BR = GKC + 8;
    static constexpr int G_BYTES = GKC * GTS * 2, B_BYTES = KP * BR * 2, STAGE = G_BYTES + B_BYTES;
    static constexpr int ST = 3 * STAGE <= 99 * 1024 ? 3 : 2;
    static constexpr int BYTES = ST * STAGE;
};

// the group's masked item yi (m0 + yi) node tile x, its children's chunks c0 .. c1 - 1: dA [128 x KP] += Gl B^T -- a
// warp 32 nodes x KP ranks, Gl^T's chunk by `ldmatrix.trans` (rows its nodes), B's as it lies --, Gl's scale taken
// out, added into the destination row at the end
template <int NK>
__device__ __forceinline__ void da_gemm_segment(const BwArgs& a, int yi, int x, int c0, int c1, unsigned char* smem) {
    using G = GemmGeo<NK>;
    constexpr int KP = G::KP, GTS = G::GTS, BR = G::BR, ST = G::ST;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, M = a.M, r = a.r;
    const int4 it = a.mitems[a.m0 + yi];
    const int e = it.y, hb = it.x;
    const __half* G0 = a.GLT + (long)yi * N * M + x * GMN;
    const __half* B0 = a.BP + (long)e * KP * N;
    auto load = [&](int st, int j0) {
        const unsigned s = smem_u32(smem + st * G::STAGE);
        for (int q = tid; q < GKC * (GMN / 8); q += GWN * 32) {
            const int row = q / (GMN / 8), c = q % (GMN / 8);
            cp16(s + (row * GTS + c * 8) * 2, G0 + (long)(j0 + row) * M + c * 8);
        }
        for (int q = tid; q < KP * (GKC / 8); q += GWN * 32) {
            const int row = q / (GKC / 8), c = q % (GKC / 8);
            cp16(s + G::G_BYTES + (row * BR + c * 8) * 2, B0 + (long)row * N + j0 + c * 8);
        }
    };
    __syncthreads();                                                         // (the last segment's stages read by all)
    #pragma unroll
    for (int st = 0; st < ST - 1; ++st) { if (c0 + st < c1) load(st, (c0 + st) * GKC); cp_commit(); }
    float acc[2][KP / 8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int n = 0; n < KP / 8; ++n) acc[mt][n][0] = acc[mt][n][1] = acc[mt][n][2] = acc[mt][n][3] = 0.f;
    for (int jc = c0; jc < c1; ++jc) {
        cp_wait<ST - 2>();
        __syncthreads();
        if (jc + ST - 1 < c1) load((jc - c0 + ST - 1) % ST, (jc + ST - 1) * GKC);
        cp_commit();
        const unsigned char* p = smem + ((jc - c0) % ST) * G::STAGE;
        const __half* Gs = reinterpret_cast<const __half*>(p);
        const __half* Bs = reinterpret_cast<const __half*>(p + G::G_BYTES);
        #pragma unroll
        for (int ks = 0; ks < GKC / 16; ++ks) {
            unsigned af[2][4];
            const int m = lane >> 3;                                         // (matrix m: children half m >> 1, nodes m & 1)
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                ldsm4t(af[mt], smem_u32(Gs + (ks * 16 + (m >> 1) * 8 + (lane & 7)) * GTS + warp * 32 + mt * 16 + (m & 1) * 8));
            #pragma unroll
            for (int n = 0; n < KP / 8; n += 2) {
                unsigned bf[4];
                ldsm4(bf, smem_u32(Bs + ((n + (lane >> 4)) * 8 + (lane & 7)) * BR + ks * 16 + ((lane >> 3) & 1) * 8));
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    mma_f16(acc[mt][n], af[mt], bf[0], bf[1]);
                    mma_f16(acc[mt][n + 1], af[mt], bf[2], bf[3]);
                }
            }
        }
    }
    // into the destination row, Gl's scale taken out: ranks 2t, 2t + 1 of each 8 (pairs where r is even) -- stored by
    // the tile's (row, x) first writer, added by the others (`tile_claim`)
    const int drw = a.drow[hb];
    float* DA = a.DA + (long)drw * a.sDA;
    int* flag = a.dflag != nullptr ? a.dflag + (long)drw * a.dft + x : nullptr;
    const bool add = !tile_claim(flag);
    const bool pair = (r & 1) == 0;
    const float iv = a.GLS[yi];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int node0 = x * GMN + warp * 32 + mt * 16 + g;
        float* D0 = DA + (long)node0 * r;
        float* D1 = D0 + 8L * r;
        #pragma unroll
        for (int n = 0; n < KP / 8; ++n) {
            const int k = n * 8 + 2 * t;
            if (k + 1 < r) {
                put2(D0 + k, acc[mt][n][0] * iv, acc[mt][n][1] * iv, pair, add);
                put2(D1 + k, acc[mt][n][2] * iv, acc[mt][n][3] * iv, pair, add);
            } else if (k < r) { put1(D0 + k, acc[mt][n][0] * iv, add); put1(D1 + k, acc[mt][n][2] * iv, add); }
        }
    }
    tile_release(flag, !add);
}

// a persistent launch over the group's masked items' node tiles' children chunks (stream-K: a CTA an equal share,
// each segment's sums added at its end)
template <int NK>
__global__ void __launch_bounds__(GWN * 32) tk_bw_da_gemm_kernel(const BwArgs a) {
    extern __shared__ __align__(128) unsigned char smem[];
    pdl_entry();
    const int ntile = a.M / GMN, nch = a.N / GKC;
    const long items = a.m1 - a.m0;
    const long U = items * ntile * nch;
    const long P = gridDim.x, k = blockIdx.x;
    const long u0 = U * k / P, u1 = U * (k + 1) / P;
    for (long u = u0; u < u1;) {
        const long tt = u / nch;
        const int c0 = (int)(u % nch);
        const int c1 = (int)min((long)nch, c0 + (u1 - u));
        da_gemm_segment<NK>(a, (int)(tt / ntile), (int)(tt % ntile), c0, c1, smem);
        u += c1 - c0;
    }
}

// --- EM statistics of the observed samples

// F+[i, j] = sum_b W_b[i, j] es_b[j] ev_b[j] ao_b[i],  F-[i, j] = sum_b W_b[i, j] es_b[j] bo_b[i] over the observed
// samples b (es_b: E's row of its slot): a CTA owns tiles of the parameters (128 nodes x two chunks of 64 children, the
// forward's warp tiling) and walks the samples -- stream-K over (tile, sample), each segment's sums added into FP, FM at
// its end by bulk reductions. l2t's fragments once a segment; the segment's walk (its samples, rows of tensors, slots)
// staged in shared memory once; a sample's B chunks (with its es, ev) staged by cp.async in a ring of mbarrier'd stages;
// its A' fragments from AH straight into registers (each warp its own 16 nodes), read once for both chunks (staging them
// in shared memory instead took 93 us a transition against 85). MEASURED (the best Codpula head, one transition, 48
// observed samples): 68.4 us with a barrier a sample, the walk's indices from global in the loop and reds; 64.1 with the
// walk staged; 60.8 with the ring; 53.8 with the stage copies' addresses a thread's constants (the register pressure
// down: no spill) and the bulk-reduced epilogue (red pairs of floats: 58.4).
struct EmArgs {
    const __half* AH; const __half* BP; const float* E; const float* ev; const uint2* LTh;
    const float* ao; const float* bo; const float* hrow; const int* slot; const int* emap;   // slot -1: masked
    const int* olist;                    // [nl] the samples to walk (-1: none; null: every sample, nl = B)
    float* FP; float* FM;
    int M, N, S, B, nl;
};

constexpr int ECH = 2;                 // an EM tile's children chunks

constexpr int EWL = NW * 32;           // an EM segment's places at most (its walk staged in shared memory, a thread a place)

__device__ __forceinline__ void mbar_init(unsigned bar, unsigned cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
// the arrive once the thread's cp.async copies before it land (the phase's count holds it: no increment)
__device__ __forceinline__ void mbar_cp_arrive(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}"
                 :: "r"(bar), "r"(parity) : "memory");
}

constexpr int ETR = ECH * TN + 8;      // the EM epilogue's staged row of sums (floats: 16-byte aligned, banks shifted a row)

template <int NK>
struct EmGeo {   // NST stages of ECH x { B chunk [KP][BROW] fp16, es and ev [2][TN] } (the epilogue's rows [TM][ETR] after
                 // them); the segment's walk [EWL] int4; the stages' full and empty mbarriers; the walk's per-warp counts
    static constexpr int KP = 16 * NK;
    static constexpr int B_BYTES = KP * BROW * 2;
    static constexpr int CHUNK = B_BYTES + 2 * TN * 4;
    static constexpr int STAGE = ECH * CHUNK;
    static constexpr int TAIL = EWL * 16 + 2 * 3 * 8 + NW * 4;
    static constexpr int NST = 3 * STAGE + TAIL <= 99 * 1024 ? 3 : 2;   // (three where they fit: ranks <= 96)
    static constexpr int STG = NST * STAGE > TM * ETR * 4 ? NST * STAGE : TM * ETR * 4;
    static constexpr int WL = STG, BAR = WL + EWL * 16, CNT = BAR + 2 * 3 * 8;
    static constexpr int BYTES = STG + TAIL;
};

// A segment of a CTA's walk: its tile's sums over the observed samples at places l0 .. l1 - 1 (at most EWL). The
// samples' stages in a ring of NST (`EmGeo`), each released by its warps (empty) and filled by every thread's copies
// (full) -- no CTA barrier a sample: a warp runs up to a sample ahead of the slowest (with a barrier a sample, the
// warps of a scheduler kept the same phase -- MMAs or FP32 together; 64 -> 59 us a transition without it, racing)
template <int NK>
__device__ __forceinline__ void em_segment(const EmArgs& a, int tile, int l0, int l1, unsigned char* smem) {
    using G = EmGeo<NK>;
    constexpr int KP = G::KP, NST = G::NST;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, M = a.M, S = a.S, nchunks = N / TN, ntc = nchunks / ECH;
    const int x = tile / ntc, c = (tile % ntc) * ECH, j0 = c * TN;           // (c: the tile's first chunk)
    const int node0 = x * TM + warp * 16 + g;
    int4* const wl = reinterpret_cast<int4*>(smem + G::WL);
    int* const wcnt = reinterpret_cast<int*>(smem + G::CNT);
    const unsigned full0 = smem_u32(smem + G::BAR), empty0 = full0 + 3 * 8;
    // the walk, staged once and compacted: the segment's walked samples (not the list's padding, not a masked one), in
    // order -- a sample, its row of tensors, its slot -- with no global index loads in the sample loop
    __syncthreads();                                   // (the last segment's stages, walk and barriers no longer in use)
    int4 rec = make_int4(-1, 0, -1, 0);
    if (tid < l1 - l0) {
        const int bb = a.olist != nullptr ? a.olist[l0 + tid] : l0 + tid;
        const int sl = bb >= 0 ? a.slot[bb] : -1;
        rec = make_int4(bb, bb >= 0 ? (a.emap != nullptr ? a.emap[bb] : bb) : 0, sl, 0);
    }
    const bool walked = rec.x >= 0 && rec.z >= 0;
    const unsigned bal = __ballot_sync(0xffffffffu, walked);
    if (lane == 0) wcnt[warp] = __popc(bal);
    if (tid < NST) { mbar_init(full0 + 8 * tid, NW * 32); mbar_init(empty0 + 8 * tid, NW); }
    __syncthreads();
    int nobs = 0, off = 0;
    #pragma unroll
    for (int w2 = 0; w2 < NW; ++w2) { const int cw = wcnt[w2]; off += w2 < warp ? cw : 0; nobs += cw; }
    if (walked) wl[off + __popc(bal & ((1u << lane) - 1))] = rec;
    __syncthreads();
    if (nobs == 0) return;                                                   // (no observed sample: the CTA's)
    // a sample's stage: its row's B chunks (a thread its 16 bytes of rows tid / 8 + 32 i: KP / 32 a chunk), its slot's E
    // row and its ev (threads 0 .. 63); every thread's copies, then its arrive on full
    static_assert(TN == 64 && NW * 32 == 256 && KP % 32 == 0, "the EM stage's copy layout");
    const unsigned so = ((tid >> 3) * BROW + (tid & 7) * 8) * 2;             // (the thread's B row, segment: in a chunk)
    const long go = (long)(tid >> 3) * N + (tid & 7) * 8;
    const int ech = (tid >> 5) & 1, ev_ = (tid >> 4) & 1, eseg = tid & 15;  // (tid < 64: chunk, E or ev, segment)
    auto load_stage = [&](int st, const int4 w) {
        const __half* bsrc = a.BP + (long)w.y * KP * N + j0 + go;
        const unsigned bs = smem_u32(smem + st * G::STAGE) + so;
        #pragma unroll
        for (int ch = 0; ch < ECH; ++ch)
            #pragma unroll
            for (int i = 0; i < KP / 32; ++i)
                cp16(bs + ch * G::CHUNK + i * 32 * BROW * 2, bsrc + ch * TN + (long)i * 32 * N);
        if (tid < 2 * ECH * (TN / 4)) {
            const float* src = ev_ == 0 ? a.E + ((long)w.y * S + w.z) * N : a.ev + (long)w.x * N;
            cp16(smem_u32(smem + st * G::STAGE + ech * G::CHUNK + G::B_BYTES) + (ev_ * TN + eseg * 4) * 4,
                 src + j0 + ech * TN + eseg * 4);
        }
        mbar_cp_arrive(full0 + 8 * st);
    };
    #pragma unroll
    for (int m = 0; m < NST - 1; ++m)
        if (m < nobs) load_stage(m, wl[m]);
    uint2 lt[ECH][8];
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    #pragma unroll
    for (int ch = 0; ch < ECH; ++ch)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) lt[ch][nt] = *(ltp + ((c + ch) * 8 + nt) * 32);
    float Fp[ECH][8][4], Fm[ECH][8][4];
    #pragma unroll
    for (int ch = 0; ch < ECH; ++ch)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            #pragma unroll
            for (int q = 0; q < 4; ++q) Fp[ch][nt][q] = Fm[ch][nt][q] = 0.f;
    // a sample's own operands -- its A' fragments (AH's half pairs), W's shift, ao, bo -- the next sample's loaded right
    // after the current one's last use: the stage wait hides their latency
    unsigned af[NK][4];
    float h0, h1, a0, a1, c0, c1;
    auto own = [&](const int4 ww) {
        const unsigned* p0 = reinterpret_cast<const unsigned*>(a.AH + ((long)ww.y * M + node0) * KP);
        const unsigned* p1 = p0 + 8 * KP / 2;
        #pragma unroll
        for (int ks = 0; ks < NK; ++ks) {
            af[ks][0] = p0[ks * 8 + t]; af[ks][1] = p1[ks * 8 + t];
            af[ks][2] = p0[ks * 8 + 4 + t]; af[ks][3] = p1[ks * 8 + 4 + t];
        }
        const long r0 = (long)ww.x * M + node0;
        h0 = a.hrow[r0]; h1 = a.hrow[r0 + 8];
        a0 = a.ao[r0]; a1 = a.ao[r0 + 8];
        c0 = a.bo[r0]; c1 = a.bo[r0 + 8];
    };
    own(wl[0]);
    for (int n = 0; n < nobs; ++n) {
        const int st = n % NST;
        mbar_wait(full0 + 8 * st, (n / NST) & 1);                           // (this sample's stage complete)
        #pragma unroll
        for (int ch = 0; ch < ECH; ++ch) {
            const unsigned char* sp = smem + st * G::STAGE + ch * G::CHUNK;
            const float* Vs = reinterpret_cast<const float*>(sp + G::B_BYTES);
            float W[8][4];
            logits<NK>(W, af, reinterpret_cast<const __half*>(sp), lt[ch], lane);
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const float2 es = *reinterpret_cast<const float2*>(Vs + nt * 8 + 2 * t);
                const float2 ev = *reinterpret_cast<const float2*>(Vs + TN + nt * 8 + 2 * t);
                const float w0 = exp2f(W[nt][0] - h0) * es.x, w1 = exp2f(W[nt][1] - h0) * es.y;
                const float w2 = exp2f(W[nt][2] - h1) * es.x, w3 = exp2f(W[nt][3] - h1) * es.y;
                Fp[ch][nt][0] += w0 * ev.x * a0; Fp[ch][nt][1] += w1 * ev.y * a0;
                Fp[ch][nt][2] += w2 * ev.x * a1; Fp[ch][nt][3] += w3 * ev.y * a1;
                Fm[ch][nt][0] += w0 * c0; Fm[ch][nt][1] += w1 * c0; Fm[ch][nt][2] += w2 * c1; Fm[ch][nt][3] += w3 * c1;
            }
        }
        // (the stage released by the warp: its last reads -- the B fragments the MMAs took, es and ev the products
        // took -- consumed above; its next copies, a full L2 round trip later)
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * st);
        if (n + 1 < nobs) own(wl[n + 1]);
        const int m = n + NST - 1;                                           // (the sample whose copies go out now)
        if (m < nobs) {
            const int sm = m % NST;
            if (m >= NST) mbar_wait(empty0 + 8 * sm, ((m / NST) - 1) & 1);   // (its stage released by every warp)
            load_stage(sm, wl[m]);
        }
    }
    // the sums into FP, FM: the tile staged in shared memory (the stages free once every warp is past its last sample),
    // F+ then F-, each of its rows added by one bulk reduction (the L2 adding 512 bytes an op; a red a pair of floats
    // each took 6.7 us of the 61 a transition)
    float* const ts = reinterpret_cast<float*>(smem);                        // [TM][ETR]
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        __syncthreads();                     // (the stages, or the last pass's rows, read by all)
        #pragma unroll
        for (int ch = 0; ch < ECH; ++ch)
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const float* F = pass == 0 ? Fp[ch][nt] : Fm[ch][nt];
                float* r0 = ts + (warp * 16 + g) * ETR + ch * TN + nt * 8 + 2 * t;
                *reinterpret_cast<float2*>(r0) = make_float2(F[0], F[1]);
                *reinterpret_cast<float2*>(r0 + 8 * ETR) = make_float2(F[2], F[3]);
            }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");     // (the rows, seen by the bulk reductions)
        __syncthreads();
        if (tid < TM) {
            float* dst = (pass == 0 ? a.FP : a.FM) + (long)(x * TM + tid) * N + j0;
            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                         :: "l"(dst), "r"(smem_u32(ts + tid * ETR)), "r"(ECH * TN * 4) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // (the rows read: free again)
        }
    }
}

template <int NK>
__global__ void __launch_bounds__(NW * 32, 1) tk_bw_em_obs_kernel(const EmArgs a) {
    extern __shared__ __align__(128) unsigned char smem[];
    pdl_entry();
    const int ntile = (a.M / TM) * (a.N / (TN * ECH));
    const long U = (long)ntile * a.nl, P = gridDim.x, k = blockIdx.x;
    const long u0 = U * k / P, u1 = U * (k + 1) / P;
    for (long u = u0; u < u1;) {
        const int tile = (int)(u / a.nl), l0 = (int)(u % a.nl);
        const int l1 = (int)min(min((long)a.nl, l0 + (u1 - u)), (long)l0 + EWL);     // (a segment's walk: EWL at most)
        em_segment<NK>(a, tile, l0, l1, smem);
        u += l1 - l0;
    }
}

// The backward's operands beside the prep's (one launch, blockIdx.y the task): AH [Bx, M, KP] = fp16(log2 e A), the
// ranks zero-padded (a thread 8 of a row's, one 16-byte store); l2t [M, N] fp16 = log2 theta less its row's largest
// (the forward's values, row-major; a warp a row)
__global__ void tk_bw_prep_kernel(const float* A, long sA, const float* theta,
                                  __half* AH, __half* L2T, int Bx, int M, int N, int r, int KP) {
    pdl_entry();
    const int task = blockIdx.y;
    const long tid0 = (long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long)gridDim.x * blockDim.x;
    if (task == 0) {
        const int G8 = KP / 8;
        const long total = (long)Bx * M * G8;
        for (long q = tid0; q < total; q += nth) {
            const long row = q / G8;
            const int k0 = (int)(q - row * G8) * 8;
            const float* src = A + (row / M) * sA + (row % M) * r;
            float v[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = k0 + u < r ? src[k0 + u] * LOG2E : 0.f;
            uint4 o;
            o.x = h2u(__floats2half2_rn(v[0], v[1])); o.y = h2u(__floats2half2_rn(v[2], v[3]));
            o.z = h2u(__floats2half2_rn(v[4], v[5])); o.w = h2u(__floats2half2_rn(v[6], v[7]));
            reinterpret_cast<uint4*>(AH)[q] = o;
        }
    } else {
        const int lane = threadIdx.x & 31;
        const long wid = tid0 >> 5, nwarp = nth >> 5;
        for (long i = wid; i < M; i += nwarp) {
            const float* tr = theta + i * N;
            float mx = 0.f;
            for (int j = lane; j < N; j += 32) mx = fmaxf(mx, tr[j]);
            mx = warp_max(mx);
            const float s = mx > 0.f ? log2f(mx) : 0.f;
            for (int j = lane; j < N; j += 32) L2T[i * N + j] = __float2half_rn(log2f(tr[j]) - s);
        }
    }
}

// l2t in the cols kernel's fragment order (`LTc`): a 16-child tile J, a node 8-tile I, a lane (g, t): {l2t[8I + 2t][16J
// + g], l2t[8I + 2t + 1][16J + g]}, the same at child 16J + g + 8 -- the C operand of W^T's first k-step, a warp's load
// one 256-byte run
__global__ void tk_bw_ltc_kernel(const __half* L2T, uint2* LTc, int M, int N) {
    pdl_entry();
    const long total = (long)(N / 16) * (M / 8) * 32;
    for (long q = (long)blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (long)gridDim.x * blockDim.x) {
        const int lane = (int)(q & 31), g = lane >> 2, t = lane & 3;
        const long JI = q >> 5;
        const int I = (int)(JI % (M / 8)), J = (int)(JI / (M / 8));
        const long i = 8L * I + 2 * t, j = 16L * J + g;
        uint2 o;
        o.x = h2u(__halves2half2(L2T[i * N + j], L2T[(i + 1) * N + j]));
        o.y = h2u(__halves2half2(L2T[i * N + j + 8], L2T[(i + 1) * N + j + 8]));
        LTc[q] = o;
    }
}

// The node flows into the stash, in place (the backward's first step). blockIdx.y the task, a CTA a sample (task 0)
// or a masked row (task 1): f = exp of the log-space rows nf [M][fld] (the circuit's buffer, a sample a column) --
// task 0: fl [B, M] the sample's flows where its ao is > 0 (a node with mass), fmx [B] their largest, and its ao, bo
// [B, M] times them (an observed sample's alpha, beta); task 1 (acnt given: the leaf's buffer): the masked row's flows
// into the slot after its posterior's, where the stash left 1.
// The destinations' tile flags before a backward's masked passes (`tile_claim`), a CTA a destination row d: 0 where a
// masked item writes the row (its first writer of a tile stores it), else 2 -- the row zeroed here, its dA [M r], dB
// [r N], dG [S N] (float4s: rows contiguous, 16 B aligned), for the observed passes to add into (none in a backward whose
// every row has a masked sample). The flows kernel's CTAs (`tk_flows_kernel`: the launch before the backward's).
struct GradInit {
    const int4* mitems; int nm; const int* drow; int* dflag; int ntiles, Bd;
    float* DA; long sDA, nA4; float* DB; long sDB, nB4; float* DG; long sDG, nG4;
};

__device__ __forceinline__ void grad_init_row(const GradInit& gi, int d) {
    int mine = 0;
    for (int i = threadIdx.x; i < gi.nm; i += blockDim.x) {
        const int b = gi.mitems[i].x;
        mine |= b >= 0 && gi.drow[b] == d;
    }
    const int has = __syncthreads_or(mine);
    for (int q = threadIdx.x; q < gi.ntiles; q += blockDim.x) gi.dflag[(long)d * gi.ntiles + q] = has ? 0 : 2;
    if (has) return;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4* pa = reinterpret_cast<float4*>(gi.DA + (long)d * gi.sDA);
    float4* pb = reinterpret_cast<float4*>(gi.DB + (long)d * gi.sDB);
    float4* pg = reinterpret_cast<float4*>(gi.DG + (long)d * gi.sDG);
    for (long q = threadIdx.x; q < gi.nA4; q += blockDim.x) pa[q] = z;
    for (long q = threadIdx.x; q < gi.nB4; q += blockDim.x) pb[q] = z;
    for (long q = threadIdx.x; q < gi.nG4; q += blockDim.x) pg[q] = z;
}

__global__ void tk_flows_kernel(const float* nf, long fld, float* ao, float* bo,
                                float* fl, float* fmx, const long* rows,
                                float* acnt, int B, int M, int Rm, float* EF, int N, const GradInit gi) {
    pdl_entry();
    __shared__ float wm[32];
    const int y = blockIdx.x, tid = threadIdx.x;
    if (blockIdx.y == 2) {                                                   // (the destinations' init: a row a CTA)
        if (y < gi.Bd) grad_init_row(gi, y);
        return;
    }
    if (blockIdx.y == 1) {
        if (y >= Rm) return;
        const long b = rows[y];
        for (int i = tid; i < M; i += blockDim.x) {
            float* c = acnt + ((long)y * M + i) * SA + 8 * NS8 + 1;
            *c = *c > 0.f ? __expf(nf[(long)i * fld + b]) : 0.f;
        }
        return;
    }
    if (y >= B) return;
    if (EF != nullptr)                                                       // (the element flows, added into after)
        for (int j = tid; j < N; j += blockDim.x) EF[(long)y * N + j] = 0.f;
    float mx = 0.f;
    for (int i = tid; i < M; i += blockDim.x) {
        const long q = (long)y * M + i;
        const float x = ao[q], f = x > 0.f ? __expf(nf[(long)i * fld + y]) : 0.f;
        fl[q] = f; ao[q] = x * f; bo[q] *= f;
        mx = fmaxf(mx, f);
    }
    mx = warp_max(mx);                                                       // (the CTA's largest: its warps', then theirs)
    if ((tid & 31) == 0) wm[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        mx = warp_max(tid < (int)(blockDim.x >> 5) ? wm[tid] : 0.f);
        if (tid == 0) fmx[y] = mx;
    }
}

// `kernel<<<grid, block, smem, st>>>(args...)` as a programmatic dependent of the launch before it (`pdl_entry`): its
// launch overlaps that one's tail -- the ~2 us between the backward's ~15 launches a node vector, back to back
template <typename... P, typename... A>
void launch_pdl(void (*kernel)(P...), dim3 grid, dim3 block, int smem, cudaStream_t st, A... args) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid; lc.blockDim = block; lc.dynamicSmemBytes = smem; lc.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kernel, static_cast<P>(args)...));
}

void check(const torch::Tensor& x, at::ScalarType dt, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == dt, "top_k: ", name,
                " must be a contiguous CUDA tensor of ", dt);
}

// a [batch, rows, n] tensor whose rows are contiguous and adjacent (any batch stride: a view of a group's tensor)
void check_rows(const torch::Tensor& x, at::ScalarType dt, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.scalar_type() == dt && x.stride(2) == 1 && x.stride(1) == x.size(2) &&
                x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "top_k: ", name,
                " must be a CUDA tensor of ", dt, " with contiguous rows, 16-byte aligned (batch stride a multiple of 4)");
}

int padded_rank(int r) { return r <= 32 ? 32 : (r <= 64 ? 64 : (r <= 96 ? 96 : 128)); }

template <typename K>
void smem_attr(K kern, int bytes, int dev, bool* done) {
    if (!done[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        done[dev] = true;
    }
}

template <int NK>
void da_launch(const BwArgs& args, int grid, int dev, cudaStream_t stream) {
    static bool done[64] = {};
    smem_attr(tk_bw_da_kernel<NK>, DaGeo<NK>::BYTES, dev, done);
    launch_pdl(tk_bw_da_kernel<NK>, dim3(grid), dim3(NW * 32), DaGeo<NK>::BYTES, stream, args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dA's GEMM of a group of masked items: a CTA a unit at most, as many CTAs as fit on the SMs
template <int NK>
void gemm_launch(const BwArgs& args, long items, int sms, int dev, cudaStream_t stream) {
    static bool done[64] = {};
    static int fit[64] = {};
    smem_attr(tk_bw_da_gemm_kernel<NK>, GemmGeo<NK>::BYTES, dev, done);
    if (fit[dev] == 0)
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit[dev], tk_bw_da_gemm_kernel<NK>, GWN * 32,
                                                                     GemmGeo<NK>::BYTES));
    const long units = items * (args.M / GMN) * (args.N / GKC);
    const int grid = (int)std::max(1L, std::min((long)sms * std::max(fit[dev], 1), units));
    launch_pdl(tk_bw_da_gemm_kernel<NK>, dim3(grid), dim3(GWN * 32), GemmGeo<NK>::BYTES, stream, args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NK, bool MASKED, bool GRADS>
void cols_launch(const BwArgs& args, int grid, int dev, cudaStream_t stream) {
    static bool done[64] = {};
    smem_attr(tk_bw_cols_kernel<NK, MASKED, GRADS>, ColsGeo<NK, MASKED>::BYTES, dev, done);
    launch_pdl(tk_bw_cols_kernel<NK, MASKED, GRADS>, dim3(grid), dim3(NW * 32), ColsGeo<NK, MASKED>::BYTES, stream, args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the backward's launches of a node vector, each a CTA an SM at most (a CTA a unit of work: the observed slots' count --
// on the device -- bounded by the observed samples'): the cols pass a kind of item a launch -- with gradients, the masked
// items in groups of `gsz` (their Gl about 2/5 of the L2), each group's dA's GEMM after its cols pass (its Gl read where
// that left it) --, then the observed slots' dA
template <int NK>
void bw_launches(BwArgs args, bool grads, int B, long gsz, int sms, int dev, cudaStream_t stream) {
    const long Rm = args.Rm, nob = B - Rm;
    auto grid = [&](long items, long units) { return (int)std::max(1L, std::min((long)sms, items * units)); };
    const long uda = (long)(args.M / TM) * (args.N / TN), ucols = (long)(args.N / CN) * (args.M / CM);
    if (grads) {
        for (long m0 = 0; m0 < Rm; m0 += gsz) {
            args.m0 = (int)m0; args.m1 = (int)std::min(Rm, m0 + gsz);
            cols_launch<NK, true, true>(args, grid(args.m1 - args.m0, ucols), dev, stream);
            gemm_launch<NK>(args, args.m1 - args.m0, sms, dev, stream);
        }
        if (nob > 0) {
            cols_launch<NK, false, true>(args, grid(nob, ucols), dev, stream);
            da_launch<NK>(args, grid(nob, uda), dev, stream);
        }
    } else {
        args.m0 = 0; args.m1 = (int)Rm;
        if (Rm > 0) cols_launch<NK, true, false>(args, grid(Rm, ucols), dev, stream);
        if (nob > 0) cols_launch<NK, false, false>(args, grid(nob, ucols), dev, stream);
    }
}

}  // namespace

// The operands of one node vector: B [Bx, r, N] and G [Bx, S, N] (fp32) a row of tensors each; the circuit's element
// values `em` [elements, B'] with every child's element id `eid` [N] (and `has` [N], or an empty tensor where every
// child is reached); theta [M, N] (fp32); a sample each (B of them): the observed slots `slot` [B], the row each reads
// `emap` [B] (an empty tensor: its own); the masked samples `rows` [Rm] and their weights `logw` [Rm, M, S] (float32;
// empty where none); `pair`: a row's samples in one item (else a sample an item); `lth` theta's l2t as an earlier
// call made it (int32 [M N / 2], its LTh: not made again), or an empty tensor; `items` the work items (mitems, oitems,
// noitems, slot32) `top_k_items` made for it with the others reading the leaf layer (pair), or empty (made here):
// (BP, E, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags, E16) -- E16 [Bx, 64, N] the slots 0 .. 63 of E
// in fp16, times 2^15: the masked forward's MMA operand (empty where no sample is masked).
std::vector<torch::Tensor> top_k_prep(torch::Tensor Bm, torch::Tensor G, torch::Tensor em, torch::Tensor eid,
                                      torch::Tensor has, torch::Tensor theta, torch::Tensor slot64, torch::Tensor emap,
                                      torch::Tensor rows, torch::Tensor logwin, bool pair, torch::Tensor lth,
                                      std::vector<torch::Tensor> items) {
    const c10::cuda::CUDAGuard guard(Bm.device());
    check_rows(Bm, at::kFloat, "B"); check_rows(G, at::kFloat, "G"); check(theta, at::kFloat, "theta");
    check(eid, at::kLong, "eid"); check(slot64, at::kLong, "slot"); check(rows, at::kLong, "rows");
    TORCH_CHECK(em.is_cuda() && em.scalar_type() == at::kFloat && em.dim() == 2 && em.stride(1) == 1,
                "top_k: the element values, a CUDA float32 [elements, batch] tensor, batch innermost");
    const int Bx = Bm.size(0), r = Bm.size(1), N = Bm.size(2), M = theta.size(0), S = G.size(1), Rm = rows.size(0);
    const int B = slot64.size(0);
    const bool mapped = emap.numel() > 0;
    TORCH_CHECK(G.size(0) == Bx && G.size(2) == N && theta.size(1) == N && eid.size(0) == N && (mapped || Bx == B),
                "top_k: shapes");
    if (mapped) { check(emap, at::kLong, "emap"); TORCH_CHECK(emap.size(0) == B, "top_k: emap [B]"); }
    TORCH_CHECK(M % TM == 0 && N % TN == 0 && r <= 128 && S == 8 * NS8 + 1,
                "top_k: nodes a multiple of 128, children of 64, rank <= 128, top_k = 64");
    TORCH_CHECK(B <= 4096, "top_k: at most 4096 samples (the work items' scans)");
    const bool full = has.numel() == 0;
    if (!full) TORCH_CHECK(has.is_cuda() && has.scalar_type() == at::kBool && has.numel() == N, "top_k: has [N] bool");
    // (the weights already in the operand's rows -- 72 wide, slots S .. 71 never read -- taken as they are)
    const bool padded = Rm > 0 && logwin.size(2) == SP;
    if (Rm > 0) {
        check(logwin, at::kFloat, "logw");
        TORCH_CHECK(logwin.size(0) == Rm && logwin.size(1) == M && (logwin.size(2) == S || padded),
                    "top_k: logw [Rm, M, S] or [Rm, M, 72]");
    }
    const int KP = padded_rank(r);
    auto f32 = Bm.options(), f16 = Bm.options().dtype(at::kHalf), i32 = Bm.options().dtype(at::kInt);
    auto BP = torch::empty({Bx, KP, N}, f16), E = torch::empty({Bx, S, N}, f32);
    auto E16 = torch::empty({Rm > 0 ? Bx : 0, 8 * NS8, N}, f16);              // (the masked forward's: none without)
    auto ev = torch::empty({B, N}, f32), cmax = torch::empty({B}, f32);
    const bool reuse = lth.numel() > 0;
    if (reuse) TORCH_CHECK(lth.is_cuda() && lth.scalar_type() == at::kInt && lth.is_contiguous() && lth.numel() == (long)M * N / 2,
                           "top_k: lth, an earlier call's LTh (int32 [M N / 2])");
    auto LTh = reuse ? lth : torch::empty({(long)M * N / 2}, i32);
    const long ocap = std::max((B - Rm + 1) / 2, 1);
    const bool given = !items.empty();
    if (given) {
        TORCH_CHECK(items.size() == 4 && pair, "top_k: given work items (mitems, oitems, noitems, slot32), paired");
        TORCH_CHECK(items[0].size(0) == std::max(Rm, 1) && items[1].size(0) == 2 * ocap + 1 && items[2].numel() == 1 &&
                    items[3].numel() == B, "top_k: the given work items' shapes");
        for (const auto& t : items) check(t, at::kInt, "items");
    }
    auto mitems = given ? items[0] : torch::empty({std::max(Rm, 1), 4}, i32);
    auto oitems = given ? items[1] : torch::empty({2 * ocap + 1, 4}, i32);
    auto noitems = given ? items[2] : torch::empty({1}, i32), slot32 = given ? items[3] : torch::empty({B}, i32);
    auto logw = padded ? logwin : torch::empty({std::max(Rm, 1), M, SP}, f32);
    // the forward's grid (each kind's launch): a CTA an SM, at least ~4 chunks a CTA (the observed items' count an
    // upper bound here); its flags, the masked launch's and the observed one's
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long chunks = (long)(M / TM) * (N / TN) * std::max((long)Rm, ocap);
    const int P = (int)std::max(2L, std::min((long)sms, chunks / 4));       // (two at least: a CTA for each kind)
    auto flags = torch::empty({2 * P}, i32);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_pdl(tk_prep_kernel, dim3(296, 5), dim3(256), 0, stream,
        Bm.data_ptr<float>(), Bm.stride(0), G.data_ptr<float>(), G.stride(0), em.data_ptr<float>(), em.stride(0),
        eid.data_ptr<long>(),
        full ? nullptr : has.data_ptr<bool>(), reuse ? nullptr : theta.data_ptr<float>(),
        Rm > 0 && !padded ? logwin.data_ptr<float>() : nullptr,
        reinterpret_cast<__half*>(BP.data_ptr<at::Half>()), E.data_ptr<float>(),
        Rm > 0 ? reinterpret_cast<__half*>(E16.data_ptr<at::Half>()) : nullptr, ev.data_ptr<float>(),
        cmax.data_ptr<float>(), reinterpret_cast<uint2*>(LTh.data_ptr<int>()), logw.data_ptr<float>(),
        Bx, B, M, N, r, KP, S, Rm, flags.data_ptr<int>(), 2 * P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!given) {
        static bool done[64] = {};
        const int ibytes = 5 * B * 4;
        if (ibytes > 48 * 1024) smem_attr(tk_items_kernel, ibytes, Bm.get_device(), done);
        launch_pdl(tk_items_kernel, dim3(1), dim3(1024), (int)ibytes, stream,
            mapped ? emap.data_ptr<long>() : nullptr, rows.data_ptr<long>(), (const int*)nullptr, slot64.data_ptr<long>(),
            reinterpret_cast<int4*>(mitems.data_ptr<int>()), reinterpret_cast<int4*>(oitems.data_ptr<int>()),
            noitems.data_ptr<int>(), slot32.data_ptr<int>(), B, Rm, pair ? 1 : 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (Rm == 0) mitems = mitems.narrow(0, 0, 0);
    return {BP, E, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags, E16};
}

// The paired work items of the T node vectors reading a leaf layer's T leaves, in one launch (`tk_items_kernel`, a CTA
// a leaf): leaf t's masked samples rows [offs[t], offs[t + 1]) of `rows` (int64; `roff` the offsets on the device,
// int32 [T + 1]), its observed slots `slots` [T, B] (int64), `emap` [B] the row each sample reads (empty: its own) --
// (mitems [sum max(Rm_t, 1), 4], oitems [sum (2 max((B - Rm_t + 1) / 2, 1) + 1), 4], noitems [T], slot32 [T, B]), a
// leaf's after the earlier leaves' (`top_k_prep`'s `items` the slices).
std::vector<torch::Tensor> top_k_items(torch::Tensor emap, torch::Tensor rows, torch::Tensor roff,
                                       std::vector<int64_t> offs, torch::Tensor slots) {
    const c10::cuda::CUDAGuard guard(slots.device());
    check(rows, at::kLong, "rows"); check(roff, at::kInt, "roff"); check(slots, at::kLong, "slots");
    const int T = (int)offs.size() - 1, B = (int)slots.size(1);
    TORCH_CHECK(T >= 1 && slots.dim() == 2 && slots.size(0) == T && roff.numel() == T + 1 && B <= 4096,
                "top_k: items of T leaves: slots [T, B], offsets T + 1, at most 4096 samples");
    const bool mapped = emap.numel() > 0;
    if (mapped) { check(emap, at::kLong, "emap"); TORCH_CHECK(emap.size(0) == B, "top_k: emap [B]"); }
    long nm = 0, no = 0;
    for (int t = 0; t < T; ++t) {
        const long rm = offs[t + 1] - offs[t];
        nm += std::max(rm, 1L);
        no += 2 * std::max((B - rm + 1) / 2, 1L) + 1;
    }
    auto i32 = slots.options().dtype(at::kInt);
    auto mitems = torch::empty({nm, 4}, i32), oitems = torch::empty({no, 4}, i32);
    auto noitems = torch::empty({T}, i32), slot32 = torch::empty({T, B}, i32);
    static bool done[64] = {};
    const int ibytes = 5 * B * 4;
    if (ibytes > 48 * 1024) smem_attr(tk_items_kernel, ibytes, slots.get_device(), done);
    launch_pdl(tk_items_kernel, dim3(T), dim3(1024), (int)ibytes, at::cuda::getCurrentCUDAStream(),
        mapped ? emap.data_ptr<long>() : nullptr, rows.numel() > 0 ? rows.data_ptr<long>() : nullptr,
        roff.data_ptr<int>(), slots.data_ptr<long>(), reinterpret_cast<int4*>(mitems.data_ptr<int>()),
        reinterpret_cast<int4*>(oitems.data_ptr<int>()), noitems.data_ptr<int>(), slot32.data_ptr<int>(), B, 0, 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {mitems, oitems, noitems, slot32};
}

// The sums' persistent launches (`tk_sk_kernel`): the masked items' and the observed items' beside it on a side stream
// (joined), each taking its kind's share of the SMs -- the forward's values and (STASH) the backward's stash.
template <bool STASH>
void sk_launch(const TkArgs& args, int KP, int P, int Rm, int B, int dev) {
    cudaStream_t main = at::cuda::getCurrentCUDAStream(), side = main;
    static cudaEvent_t fork[64] = {}, join[64] = {};
    const bool two = Rm > 0 && B > Rm;
    if (two) {
        if (fork[dev] == nullptr) {
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&fork[dev], cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&join[dev], cudaEventDisableTiming));
        }
        side = at::cuda::getStreamFromPool(false, dev).stream();
        C10_CUDA_CHECK(cudaEventRecord(fork[dev], main));
        C10_CUDA_CHECK(cudaStreamWaitEvent(side, fork[dev], 0));
    }
#define TK_CASE(NKV)                                                                                               \
    case NKV: {                                                                                                    \
        static bool dm[64] = {}, dob[64] = {};                                                                     \
        if (Rm > 0) {                                                                                              \
            smem_attr(tk_sk_kernel<NKV, true, STASH>, MaskedGeo<NKV>::BYTES, dev, dm);                             \
            tk_sk_kernel<NKV, true, STASH><<<P, NW * 32, MaskedGeo<NKV>::BYTES, main>>>(args);                     \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                        \
        }                                                                                                          \
        if (B > Rm) {                                                                                              \
            smem_attr(tk_sk_kernel<NKV, false, STASH>, ObservedGeo<NKV>::BYTES, dev, dob);                         \
            tk_sk_kernel<NKV, false, STASH><<<P, NW * 32, ObservedGeo<NKV>::BYTES, side>>>(args);                  \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                        \
        }                                                                                                          \
        break;                                                                                                     \
    }
    switch (KP / 16) {
        TK_CASE(2)
        TK_CASE(4)
        TK_CASE(6)
        TK_CASE(8)
        default: TORCH_CHECK(false, "top_k: ranks padded to 32, 64, 96 or 128");
    }
#undef TK_CASE
    if (two) {
        C10_CUDA_CHECK(cudaEventRecord(join[dev], side));
        C10_CUDA_CHECK(cudaStreamWaitEvent(main, join[dev], 0));
    }
}

TkArgs sk_args(const torch::Tensor& A, const torch::Tensor& BP, const torch::Tensor& E, const torch::Tensor& ev,
               const torch::Tensor& cmax, const torch::Tensor& LTh, const torch::Tensor& mitems,
               const torch::Tensor& oitems, const torch::Tensor& noitems, const torch::Tensor& slot32,
               const torch::Tensor& logw, const torch::Tensor& flags, const torch::Tensor& E16, const torch::Tensor& ws) {
    check_rows(A, at::kFloat, "A");
    check(E16, at::kHalf, "E16");
    const int r = A.size(2), KP = BP.size(1);
    TORCH_CHECK(KP == padded_rank(r) && BP.size(0) == A.size(0) && E16.size(0) == (mitems.size(0) > 0 ? E.size(0) : 0) &&
                E16.size(1) == 8 * NS8 && E16.size(2) == E.size(2), "top_k: A and the prep's operands disagree");
    TkArgs args{};
    args.A = A.data_ptr<float>(); args.BP = reinterpret_cast<const __half*>(BP.data_ptr<at::Half>());
    args.LTh = reinterpret_cast<const uint2*>(LTh.data_ptr<int>()); args.E = E.data_ptr<float>();
    args.E16 = reinterpret_cast<const __half*>(E16.data_ptr<at::Half>());
    args.ev = ev.data_ptr<float>(); args.cmax = cmax.data_ptr<float>();
    args.mitems = reinterpret_cast<const int4*>(mitems.data_ptr<int>());
    args.oitems = reinterpret_cast<const int4*>(oitems.data_ptr<int>()); args.noitems = noitems.data_ptr<int>();
    args.slot32 = slot32.data_ptr<int>(); args.logw = logw.data_ptr<float>(); args.ws = ws.data_ptr<float>();
    args.flags = flags.data_ptr<int>(); args.sA = A.stride(0);
    args.r = r; args.M = A.size(1); args.N = BP.size(2); args.S = E.size(1); args.Rm = (int)mitems.size(0);
    return args;
}

// One node vector's values from the caller's A [Bx, M, r] (fp32) and `top_k_prep`'s operands, into `out` [nodes,
// batch] from node `nid0` on (`sk_launch`; an empty `out`: no values). With `stash`, what a backward takes from the
// sums, returned (else nothing): (alpha, beta [max(Rm, 1), M, SH] fp16 and 1 / d [max(Rm, 1), M] -- a masked sample's,
// by its place in the prep's `rows`, the node's flow not in them --, the slots' posterior [max(Rm, 1), M, SA] (into
// `acnt_out` [Rm, M, SA] where given, its leaf's buffer), ao, bo [B, M], W's shift h [B, M]): `masked_stash`'s notes;
// `top_k_flows` puts the flows in.
std::vector<torch::Tensor> top_k_forward(torch::Tensor A, torch::Tensor BP, torch::Tensor E, torch::Tensor ev,
                                         torch::Tensor cmax, torch::Tensor LTh, torch::Tensor mitems,
                                         torch::Tensor oitems, torch::Tensor noitems, torch::Tensor slot32,
                                         torch::Tensor logw, torch::Tensor flags, torch::Tensor E16, torch::Tensor out,
                                         int64_t nid0, bool stash, torch::Tensor acnt_out) {
    const c10::cuda::CUDAGuard guard(A.device());
    const bool values = out.numel() > 0;
    const int M = A.size(1), B = ev.size(0), Rm = mitems.size(0);
    TORCH_CHECK(values || stash, "top_k: the values or the stash");
    if (values) {
        TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.size(1) == B,
                    "top_k: the node values, a contiguous CUDA float32 [nodes, batch] tensor");
        TORCH_CHECK(nid0 >= 0 && nid0 + M <= out.size(0), "top_k: the node vector's rows of the node values");
    }
    const int P = flags.size(0) / 2;
    auto o = A.options();
    auto ws = torch::empty({(long)P * (WS_M + WS_O) * NW * 32}, o);
    TkArgs args = sk_args(A, BP, E, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags, E16, ws);
    if (values) { args.out = out.data_ptr<float>(); args.nid0 = nid0; args.ostride = out.size(1); }
    if (!stash) {
        sk_launch<false>(args, BP.size(1), P, Rm, B, A.get_device());
        return {};
    }
    auto acnt = acnt_out.numel() > 0 ? acnt_out : torch::empty({std::max(Rm, 1), M, SA}, o);
    if (acnt_out.numel() > 0) {
        check(acnt_out, at::kFloat, "acnt_out");
        TORCH_CHECK(acnt_out.size(0) == Rm && acnt_out.size(1) == M && acnt_out.size(2) == SA, "top_k: acnt_out [Rm, M, 72]");
    }
    auto ao = torch::empty({B, M}, o), bo = torch::empty_like(ao), hrow = torch::empty_like(ao);
    auto alpha16 = torch::empty({std::max(Rm, 1), M, SH}, o.dtype(at::kHalf)), beta16 = torch::empty_like(alpha16);
    auto invd = torch::empty({std::max(Rm, 1), M}, o);
    args.acnt = acnt.data_ptr<float>(); args.ao = ao.data_ptr<float>(); args.bo = bo.data_ptr<float>();
    args.hrow = hrow.data_ptr<float>();
    args.alpha16 = reinterpret_cast<__half*>(alpha16.data_ptr<at::Half>());
    args.beta16 = reinterpret_cast<__half*>(beta16.data_ptr<at::Half>()); args.invd = invd.data_ptr<float>();
    sk_launch<true>(args, BP.size(1), P, Rm, B, A.get_device());
    return {alpha16, beta16, invd, acnt, ao, bo, hrow};
}

// The node flows into a stash, in place (`tk_flows_kernel`): `nf` [M, B] the node vector's log-space rows of the node
// flows (fp32, a sample a column), ao, bo [B, M]; `rows` [Rm] the masked samples and `acnt` their rows of the leaf's
// buffer [Rm, M, SA] (empty: none) -- the flow into the slot after the posterior's. Returns (fl [B, M] the samples'
// node flows, fmx [B] their largest).
std::vector<torch::Tensor> top_k_flows(torch::Tensor nf, torch::Tensor ao, torch::Tensor bo, torch::Tensor rows,
                                       torch::Tensor acnt, torch::Tensor EF, torch::Tensor mitems, torch::Tensor drow,
                                       torch::Tensor dflag, torch::Tensor DA, torch::Tensor DB, torch::Tensor DG) {
    const c10::cuda::CUDAGuard guard(ao.device());
    check(ao, at::kFloat, "ao"); check(bo, at::kFloat, "bo");
    const int B = ao.size(0), M = ao.size(1), Rm = acnt.numel() > 0 ? (int)rows.size(0) : 0;
    TORCH_CHECK(nf.is_cuda() && nf.scalar_type() == at::kFloat && nf.dim() == 2 && nf.size(0) == M && nf.size(1) == B &&
                nf.stride(1) == 1, "top_k: the node flows' log-space rows [M, B] (a sample a column)");
    if (Rm > 0) {
        check(rows, at::kLong, "rows"); check(acnt, at::kFloat, "acnt");
        TORCH_CHECK(acnt.size(0) == Rm && acnt.size(1) == M && acnt.size(2) == SA, "top_k: acnt [Rm, M, 72]");
    }
    auto fl = torch::empty_like(ao), fmx = torch::empty({B}, ao.options());
    // (EF [B, N] zeroed beside the flows where given; `dflag` non-empty: the gradient destinations' init, a slice of
    // its own -- `grad_init_row`, for `top_k_bw_grads`' masked passes' first writers)
    const int N = EF.numel() > 0 ? (int)EF.size(1) : 0;
    if (EF.numel() > 0) {
        check(EF, at::kFloat, "EF");
        TORCH_CHECK(EF.dim() == 2 && EF.size(0) == B, "top_k: EF [B, N]");
    }
    GradInit gi{};
    if (dflag.numel() > 0) {
        check(dflag, at::kInt, "dflag"); check(drow, at::kInt, "drow");
        check_rows(DA, at::kFloat, "dA"); check_rows(DB, at::kFloat, "dB"); check_rows(DG, at::kFloat, "dG");
        const long Bd = DA.size(0), Mg = DA.size(1), r = DA.size(2), Ng = DB.size(2), S = DG.size(1);
        TORCH_CHECK(DB.size(0) == Bd && DG.size(0) == Bd && Mg % GMN == 0 && Ng % CN == 0 &&
                    dflag.numel() == Bd * (Mg / GMN + Ng / CN) && drow.size(0) == B && mitems.dim() == 2 &&
                    mitems.size(1) == 4 && Mg * r % 4 == 0 && r * Ng % 4 == 0 && S * Ng % 4 == 0,
                    "top_k: the destinations' init: dflag [Bd, M / 128 + N / 128] int32, drow [B], the masked items");
        gi.mitems = reinterpret_cast<const int4*>(mitems.data_ptr<int>()); gi.nm = (int)mitems.size(0);
        gi.drow = drow.data_ptr<int>(); gi.dflag = dflag.data_ptr<int>(); gi.ntiles = (int)(Mg / GMN + Ng / CN);
        gi.Bd = (int)Bd;
        gi.DA = DA.data_ptr<float>(); gi.sDA = DA.stride(0); gi.nA4 = Mg * r / 4;
        gi.DB = DB.data_ptr<float>(); gi.sDB = DB.stride(0); gi.nB4 = r * Ng / 4;
        gi.DG = DG.data_ptr<float>(); gi.sDG = DG.stride(0); gi.nG4 = S * Ng / 4;
    }
    const int ny = gi.Bd > 0 ? 3 : (Rm > 0 ? 2 : 1);
    launch_pdl(tk_flows_kernel, dim3(std::max({B, Rm, gi.Bd}), ny), dim3(256), 0, at::cuda::getCurrentCUDAStream(),
        nf.data_ptr<float>(), nf.stride(0), ao.data_ptr<float>(), bo.data_ptr<float>(), fl.data_ptr<float>(),
        fmx.data_ptr<float>(), Rm > 0 ? rows.data_ptr<long>() : nullptr, Rm > 0 ? acnt.data_ptr<float>() : nullptr,
        B, M, Rm, EF.numel() > 0 ? EF.data_ptr<float>() : nullptr, N, gi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fl, fmx};
}

// The backward's operands beside the prep's: (AH [Bx, M, KP] = fp16(log2 e A), ranks zero-padded; l2t [M, N] fp16 (the
// forward's values, row-major); LTc: l2t in the cols kernel's fragment order) from A [Bx, M, r] and theta [M, N] --
// theta's two tables those of an earlier call where given (`l2t`, `ltc`; empty tensors: made here).
std::vector<torch::Tensor> top_k_bw_prep(torch::Tensor A, torch::Tensor theta, int64_t KP, torch::Tensor l2t,
                                         torch::Tensor ltc) {
    const c10::cuda::CUDAGuard guard(A.device());
    check_rows(A, at::kFloat, "A"); check(theta, at::kFloat, "theta");
    const int Bx = A.size(0), M = A.size(1), r = A.size(2), N = theta.size(1);
    TORCH_CHECK(theta.size(0) == M && KP == padded_rank(r) && M % CN == 0 && N % CN == 0, "top_k: shapes");
    // (theta's tables as an earlier call made them -- L2T, LTc --: not made again)
    const bool reuse = l2t.numel() > 0 && ltc.numel() > 0;
    if (reuse) {
        check(l2t, at::kHalf, "l2t"); check(ltc, at::kInt, "ltc");
        TORCH_CHECK(l2t.numel() == (long)M * N && ltc.numel() == (long)M * N / 2, "top_k: an earlier call's L2T, LTc");
    }
    auto AH = torch::empty({Bx, M, KP}, A.options().dtype(at::kHalf));
    auto L2T = reuse ? l2t : torch::empty({M, N}, A.options().dtype(at::kHalf));
    auto LTc = reuse ? ltc : torch::empty({(long)M * N / 2}, A.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_pdl(tk_bw_prep_kernel, dim3(296, reuse ? 1 : 2), dim3(256), 0, stream, A.data_ptr<float>(), A.stride(0),
        theta.data_ptr<float>(), reinterpret_cast<__half*>(AH.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(L2T.data_ptr<at::Half>()), Bx, M, N, r, (int)KP);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!reuse) {
        launch_pdl(tk_bw_ltc_kernel, dim3(296), dim3(256), 0, stream, reinterpret_cast<const __half*>(L2T.data_ptr<at::Half>()),
            reinterpret_cast<uint2*>(LTc.data_ptr<int>()), M, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {AH, L2T, LTc};
}

// The backward's passes of one node vector, from the operands (the prep's, `top_k_bw_prep`'s) and the stash's terms
// (the flows in them: `top_k_flows`; the masked samples' alpha, beta the stash's fp16 rows [Rm, M, SH] with 1 / d
// `invd` [Rm, M]): the element flows into EF [B, N]; with gradients (DA non-empty) dA, dB, dG, each
// sample's times its sign `sgn` [B], into the rows `drow` [B] of DA [Bd, M, r], DB [Bd, r, N], DG [Bd, S, N] (fp32,
// rows contiguous, batch strides a multiple of 4 floats) -- every destination ADDED into, or with `dflag` (int32
// [Bd, M / 128 + N / 128] as `top_k_flows` set it; empty: none) initialized: the masked passes' first writer of a
// destination tile stores it, the rows no masked item writes zeroed first (`tile_claim`, `grad_init_row`) --
// destinations handed over unzeroed then need no zeroing pass. A [Bx, M, r] the caller's. With gradients, a group's masked items' Gl (fp16,
// ~2 MB an item at N = M = 1024) is made for dA's GEMM, then freed.
void top_k_bw_grads(torch::Tensor A, torch::Tensor AH, torch::Tensor BP, torch::Tensor E, torch::Tensor ev,
                    torch::Tensor LTh, torch::Tensor LTc, torch::Tensor mitems, torch::Tensor oitems,
                    torch::Tensor noitems, torch::Tensor slot32, torch::Tensor alpha16, torch::Tensor beta16,
                    torch::Tensor invd, torch::Tensor ao, torch::Tensor bo, torch::Tensor hrow, torch::Tensor fl,
                    torch::Tensor fmx, torch::Tensor drow, torch::Tensor sgn, torch::Tensor EF, torch::Tensor DA,
                    torch::Tensor DB, torch::Tensor DG, torch::Tensor dflag) {
    const c10::cuda::CUDAGuard guard(A.device());
    check_rows(A, at::kFloat, "A"); check(EF, at::kFloat, "EF");
    const int M = A.size(1), r = A.size(2), KP = BP.size(1), N = BP.size(2), S = E.size(1), Rm = mitems.size(0);
    const int B = ev.size(0);
    const bool grads = DA.numel() > 0;
    TORCH_CHECK(AH.size(1) == M && AH.size(2) == KP && KP == padded_rank(r) && M % CN == 0 && N % CN == 0 &&
                EF.size(0) == B && EF.size(1) == N, "top_k: the backward's shapes");
    BwArgs args{};
    args.AH = reinterpret_cast<const __half*>(AH.data_ptr<at::Half>());
    args.BP = reinterpret_cast<const __half*>(BP.data_ptr<at::Half>()); args.E = E.data_ptr<float>();
    args.ev = ev.data_ptr<float>(); args.LTc = reinterpret_cast<const uint2*>(LTc.data_ptr<int>());
    args.mitems = reinterpret_cast<const int4*>(mitems.data_ptr<int>());
    args.oitems = reinterpret_cast<const int4*>(oitems.data_ptr<int>()); args.noitems = noitems.data_ptr<int>();
    args.slot32 = slot32.data_ptr<int>();
    if (Rm > 0) {                                                            // (the masked samples' alpha, beta: the stash's)
        check(alpha16, at::kHalf, "alpha16"); check(beta16, at::kHalf, "beta16"); check(invd, at::kFloat, "invd");
        TORCH_CHECK(alpha16.dim() == 3 && alpha16.size(0) >= Rm && alpha16.size(1) == M && alpha16.size(2) == SH &&
                    beta16.sizes() == alpha16.sizes() && invd.size(0) == alpha16.size(0) && invd.size(1) == M,
                    "top_k: the stash's alpha16, beta16 [Rm, M, 72] and 1 / d [Rm, M]");
        args.alpha16 = reinterpret_cast<const __half*>(alpha16.data_ptr<at::Half>());
        args.beta16 = reinterpret_cast<const __half*>(beta16.data_ptr<at::Half>()); args.invd = invd.data_ptr<float>();
    }
    args.ao = ao.data_ptr<float>(); args.bo = bo.data_ptr<float>(); args.hrow = hrow.data_ptr<float>();
    args.fl = fl.data_ptr<float>(); args.fmx = fmx.data_ptr<float>();
    args.EF = EF.data_ptr<float>(); args.A = A.data_ptr<float>(); args.sA = A.stride(0);
    args.LTh = reinterpret_cast<const uint2*>(LTh.data_ptr<int>());
    torch::Tensor GLT, GLS;
    long gsz = std::max(Rm, 1);
    if (grads && Rm > 0) {
        const long l2 = at::cuda::getCurrentDeviceProperties()->l2CacheSize, ib = (long)N * M * 2;
        gsz = std::max(1L, std::min((long)Rm, l2 * 2 / 5 / ib));
        const long ng = (Rm + gsz - 1) / gsz;
        gsz = (Rm + ng - 1) / ng;                                            // (the groups even)
        GLT = torch::empty({gsz, N, M}, A.options().dtype(at::kHalf));
        GLS = torch::empty({gsz}, A.options());
        args.GLT = reinterpret_cast<__half*>(GLT.data_ptr<at::Half>()); args.GLS = GLS.data_ptr<float>();
    }
    if (grads) {
        check(drow, at::kInt, "drow"); check(sgn, at::kFloat, "sgn");
        check_rows(DA, at::kFloat, "dA"); check_rows(DB, at::kFloat, "dB"); check_rows(DG, at::kFloat, "dG");
        TORCH_CHECK(DA.size(1) == M && DA.size(2) == r && DB.size(1) == r && DB.size(2) == N && DG.size(1) == S &&
                    DG.size(2) == N && drow.size(0) == B && sgn.size(0) == B, "top_k: the gradients' shapes");
        args.drow = drow.data_ptr<int>(); args.sgn = sgn.data_ptr<float>();
        args.DA = DA.data_ptr<float>(); args.sDA = DA.stride(0);
        args.DB = DB.data_ptr<float>(); args.sDB = DB.stride(0);
        args.DG = DG.data_ptr<float>(); args.sDG = DG.stride(0);
    }
    args.r = r; args.M = M; args.N = N; args.S = S; args.Rm = Rm;
    // (destinations handed over uninitialized -- `dflag` [Bd, M / GMN + N / CN] int32, initialized by `top_k_flows`, else
    // empty: every writer adds --: the masked passes' first writer of a tile stores it, rows no masked item writes
    // zeroed first)
    if (grads && dflag.numel() > 0) {
        const long Bd = DA.size(0);
        args.dfa = M / GMN; args.dft = M / GMN + N / CN;
        check(dflag, at::kInt, "dflag");
        TORCH_CHECK(dflag.numel() == Bd * args.dft && DB.size(0) == Bd && DG.size(0) == Bd &&
                    (long)M * r % 4 == 0 && (long)r * N % 4 == 0 && (long)S * N % 4 == 0,
                    "top_k: dflag [destination rows, M / 128 + N / 128] int32; rows of a multiple of 4 floats");
        args.dflag = dflag.data_ptr<int>();                              // (set by `top_k_flows`, the launch before)
    }
    // a launch a pass and a kind, each a CTA an SM (at most a CTA a unit: the observed slots' count an upper bound)
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount, dev = A.get_device();
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (KP / 16) {
        case 2: bw_launches<2>(args, grads, B, gsz, sms, dev, stream); break;
        case 4: bw_launches<4>(args, grads, B, gsz, sms, dev, stream); break;
        case 6: bw_launches<6>(args, grads, B, gsz, sms, dev, stream); break;
        case 8: bw_launches<8>(args, grads, B, gsz, sms, dev, stream); break;
        default: TORCH_CHECK(false, "top_k: ranks padded to 32, 64, 96 or 128");
    }
}

// The EM statistics of the observed samples (slot >= 0; -1 a masked one, skipped) added into FP, FM [M, N] (fp32):
// from the operands (AH, BP, E, ev, LTh) and the stash's ao, bo (the flows in them), hrow; emap the row a sample reads
// (empty: its own). `olist` the samples to walk, -1 padded (`top_k_em_index`; empty: every sample) -- the work split
// over them.
void top_k_bw_em_observed(torch::Tensor AH, torch::Tensor BP, torch::Tensor E, torch::Tensor ev, torch::Tensor LTh,
                          torch::Tensor ao, torch::Tensor bo, torch::Tensor hrow, torch::Tensor slot, torch::Tensor emap,
                          torch::Tensor olist, torch::Tensor FP, torch::Tensor FM) {
    const c10::cuda::CUDAGuard guard(AH.device());
    check(slot, at::kInt, "slot"); check(FP, at::kFloat, "FP"); check(FM, at::kFloat, "FM");
    const int M = AH.size(1), KP = AH.size(2), N = BP.size(2), S = E.size(1), B = ev.size(0);
    TORCH_CHECK(M % TM == 0 && N % (TN * ECH) == 0 && FP.size(0) == M && FP.size(1) == N && FM.sizes() == FP.sizes() &&
                slot.size(0) == B, "top_k: the EM statistics' shapes");
    EmArgs args{};
    args.AH = reinterpret_cast<const __half*>(AH.data_ptr<at::Half>());
    args.BP = reinterpret_cast<const __half*>(BP.data_ptr<at::Half>()); args.E = E.data_ptr<float>();
    args.ev = ev.data_ptr<float>(); args.LTh = reinterpret_cast<const uint2*>(LTh.data_ptr<int>());
    args.ao = ao.data_ptr<float>(); args.bo = bo.data_ptr<float>(); args.hrow = hrow.data_ptr<float>();
    args.slot = slot.data_ptr<int>(); args.emap = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    args.FP = FP.data_ptr<float>(); args.FM = FM.data_ptr<float>();
    args.M = M; args.N = N; args.S = S; args.B = B;
    if (emap.numel() > 0) check(emap, at::kInt, "emap");
    args.olist = nullptr; args.nl = B;
    if (olist.numel() > 0) {
        check(olist, at::kInt, "olist");
        args.olist = olist.data_ptr<int>(); args.nl = (int)olist.numel();
    }
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount, dev = AH.get_device();
    const long units = (long)(M / TM) * (N / (TN * ECH)) * args.nl;
    const int grid = (int)std::max(1L, std::min((long)sms, units));
    auto stream = at::cuda::getCurrentCUDAStream();
#define EM_CASE(NKV)                                                                                               \
    case NKV: {                                                                                                    \
        static bool done[64] = {};                                                                                 \
        smem_attr(tk_bw_em_obs_kernel<NKV>, EmGeo<NKV>::BYTES, dev, done);                                         \
        launch_pdl(tk_bw_em_obs_kernel<NKV>, dim3(grid), dim3(NW * 32), EmGeo<NKV>::BYTES, stream, args);         \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                            \
        break;                                                                                                     \
    }
    switch (KP / 16) {
        EM_CASE(2)
        EM_CASE(4)
        EM_CASE(6)
        EM_CASE(8)
        default: TORCH_CHECK(false, "top_k: ranks padded to 32, 64, 96 or 128");
    }
#undef EM_CASE
}

// ---------------------------------------------------------------------------------------------------------------------
// The backward's small passes: the EM statistics' sample lists, and the element flows' and the statistics' scatters.

// an exclusive scan of a pair over the block (every thread calls it; blockDim.x a multiple of 32); `tot` the sums
__device__ __forceinline__ int2 block_scan2(int2 v, int2& tot) {
    __shared__ int2 ws[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int2 x = v;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int p = __shfl_up_sync(~0u, x.x, d), q = __shfl_up_sync(~0u, x.y, d);
        if (lane >= d) { x.x += p; x.y += q; }
    }
    if (lane == 31) ws[w] = x;
    __syncthreads();
    if (w == 0) {
        int2 y = lane < nw ? ws[lane] : make_int2(0, 0);
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int p = __shfl_up_sync(~0u, y.x, d), q = __shfl_up_sync(~0u, y.y, d);
            if (lane >= d) { y.x += p; y.y += q; }
        }
        ws[lane] = y;
    }
    __syncthreads();
    tot = ws[nw - 1];
    const int2 before = w > 0 ? ws[w - 1] : make_int2(0, 0);
    return make_int2(before.x + x.x - v.x, before.y + x.y - v.y);
}

// (one CTA) the samples the EM statistics take -- `pmask` [B] (null: all) -- in the kernels' terms: SLOT [B] an observed
// one's slot (-1 else), LWIDX [B] a masked one's place in `rows` (-1 else), EMAP [B] the row a sample reads, MROWS
// [max(Rm, 1)] the masked ones in order then -1, NM [1] their count, OLIST [max(B - Rm, 1)] the observed ones in order
// then -1. kind [B] and nothing else in shared memory.
__global__ void __launch_bounds__(1024) tk_em_index_kernel(
        const int* slot32, const long* rows, const long* emap,
        const bool* pmask, int* SLOT, int* LWIDX, int* EMAP,
        int* MROWS, int* NM, int* OLIST, int B, int Rm) {
    pdl_entry();
    extern __shared__ int kind[];
    const int tid = threadIdx.x, nth = blockDim.x;
    for (int b = tid; b < B; b += nth) {
        kind[b] = 0;
        LWIDX[b] = -1;
        EMAP[b] = emap != nullptr ? (int)emap[b] : b;
    }
    __syncthreads();
    for (int q = tid; q < Rm; q += nth) {
        kind[rows[q]] = 1;
        LWIDX[rows[q]] = q;
    }
    __syncthreads();
    const int per = (B + nth - 1) / nth, b0 = min(B, tid * per), b1 = min(B, b0 + per);
    int2 cnt = make_int2(0, 0);                                              // (the thread's: masked, observed taken)
    for (int b = b0; b < b1; ++b)
        if (pmask == nullptr || pmask[b]) { if (kind[b]) ++cnt.x; else ++cnt.y; }
    int2 tot;
    int2 at = block_scan2(cnt, tot);
    for (int b = b0; b < b1; ++b) {
        const bool take = pmask == nullptr || pmask[b];
        SLOT[b] = take && !kind[b] ? slot32[b] : -1;
        if (take) { if (kind[b]) MROWS[at.x++] = b; else OLIST[at.y++] = b; }
    }
    for (int j = tot.x + tid; j < max(Rm, 1); j += nth) MROWS[j] = -1;
    for (int j = tot.y + tid; j < max(B - Rm, 1); j += nth) OLIST[j] = -1;
    if (tid == 0) NM[0] = tot.x;
}

std::vector<torch::Tensor> top_k_em_index(torch::Tensor slot32, torch::Tensor rows, torch::Tensor emap,
                                          torch::Tensor pmask) {
    const c10::cuda::CUDAGuard guard(slot32.device());
    check(slot32, at::kInt, "slot32"); check(rows, at::kLong, "rows");
    const int B = slot32.size(0), Rm = rows.numel();
    TORCH_CHECK(B <= 4096 && Rm <= B, "top_k: the EM statistics' lists of at most 4096 samples");
    if (emap.numel() > 0) { check(emap, at::kLong, "emap"); TORCH_CHECK(emap.numel() == B, "top_k: emap [B]"); }
    if (pmask.numel() > 0) { check(pmask, at::kBool, "pmask"); TORCH_CHECK(pmask.numel() == B, "top_k: pmask [B]"); }
    auto o = slot32.options();
    auto SLOT = torch::empty({B}, o), LWIDX = torch::empty({B}, o), EMAP = torch::empty({B}, o);
    auto MROWS = torch::empty({std::max(Rm, 1)}, o), NM = torch::empty({1}, o), OLIST = torch::empty({std::max(B - Rm, 1)}, o);
    launch_pdl(tk_em_index_kernel, dim3(1), dim3(1024), (int)(B * sizeof(int)), at::cuda::getCurrentCUDAStream(),
        slot32.data_ptr<int>(), rows.data_ptr<long>(), emap.numel() > 0 ? emap.data_ptr<long>() : nullptr,
        pmask.numel() > 0 ? pmask.data_ptr<bool>() : nullptr, SLOT.data_ptr<int>(), LWIDX.data_ptr<int>(),
        EMAP.data_ptr<int>(), MROWS.data_ptr<int>(), NM.data_ptr<int>(), OLIST.data_ptr<int>(), B, Rm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {SLOT, LWIDX, EMAP, MROWS, NM, OLIST};
}

// element_flows[kids[k], b] = logaddexp(element_flows[kids[k], b], log EF[b, cols[k]]): the children's element flows
// (log space, a sample a column) from the backward's EF [B, N] (linear), a 32 x 32 tile a CTA through shared memory
__global__ void tk_scatter_ef_kernel(const float* EF, const long* cols,
                                     const long* kids, float* ef, long ld, int B, int N, int nk) {
    pdl_entry();
    __shared__ float tile[32][33];
    const int k0 = blockIdx.x * 32, b0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
    const int kx = k0 + tx;
    const long cx = kx < nk ? cols[kx] : 0;
    for (int u = ty; u < 32; u += 8) {
        const int b = b0 + u;
        tile[u][tx] = (kx < nk && b < B) ? EF[(long)b * N + cx] : 0.f;
    }
    __syncthreads();
    for (int u = ty; u < 32; u += 8) {
        const int k = k0 + u, b = b0 + tx;
        if (k >= nk || b >= B) continue;
        float* p = ef + kids[k] * ld + b;
        const float x = *p, y = __logf(tile[tx][u]);
        const float m = fmaxf(x, y);
        *p = m == -INFINITY ? m : m + log1pf(__expf(-fabsf(x - y)));
    }
}

void top_k_scatter_ef(torch::Tensor EF, torch::Tensor cols, torch::Tensor kids, torch::Tensor element_flows) {
    const c10::cuda::CUDAGuard guard(EF.device());
    check(EF, at::kFloat, "EF"); check(cols, at::kLong, "cols"); check(kids, at::kLong, "kids");
    TORCH_CHECK(element_flows.is_cuda() && element_flows.scalar_type() == at::kFloat && element_flows.dim() == 2 &&
                element_flows.stride(1) == 1 && element_flows.size(1) == EF.size(0) && cols.numel() == kids.numel(),
                "top_k: the element flows [rows, B] fp32 and EF [B, N]");
    const int B = EF.size(0), N = EF.size(1), nk = cols.numel();
    if (nk == 0 || B == 0) return;
    dim3 grid((unsigned)((nk + 31) / 32), (unsigned)((B + 31) / 32)), block(32, 8);
    launch_pdl(tk_scatter_ef_kernel, grid, block, 0, at::cuda::getCurrentCUDAStream(),
        EF.data_ptr<float>(), cols.data_ptr<long>(), kids.data_ptr<long>(), element_flows.data_ptr<float>(),
        element_flows.stride(0), B, N, nk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The EM statistics onto a node vector's slots where its parameters lie block-major (`LowRank._scatter_add`'s grid:
// (i, j) at ((i / bs * Nb + j / cbs) * cbs + j % cbs) * bs + i % bs): pf[pf0 + .] += FP[i, j], and where `den` is
// given den[den0 + .] += FM[i, j] / theta (0 where theta = 0), theta = params[p0 + .] -- a 32 x 32 tile a CTA
__global__ void tk_scatter_em_kernel(const float* FP, const float* FM,
                                     const float* params, float* pf, float* den,
                                     long p0, long pf0, long den0, int M, int N, int Nb, int cbs, int bs) {
    pdl_entry();
    __shared__ float tp[32][33], tm[32][33];
    const int i0 = blockIdx.y * 32, j0 = blockIdx.x * 32, tx = threadIdx.x, ty = threadIdx.y;
    for (int u = ty; u < 32; u += 8) {
        const int i = i0 + u, j = j0 + tx;
        const bool in = i < M && j < N;
        if (pf != nullptr) tp[u][tx] = in ? FP[(long)i * N + j] : 0.f;
        if (den != nullptr) tm[u][tx] = in ? FM[(long)i * N + j] : 0.f;
    }
    __syncthreads();
    for (int u = ty; u < 32; u += 8) {
        const int i = i0 + tx, j = j0 + u;                                    // (i fastest: the slots' order)
        if (i >= M || j >= N) continue;
        const long o = ((long)(i / bs) * Nb + j / cbs) * cbs * bs + (long)(j % cbs) * bs + i % bs;
        if (pf != nullptr) pf[pf0 + o] += tp[tx][u];
        if (den != nullptr) {
            const float th = params[p0 + o];
            den[den0 + o] += th > 0.f ? tm[tx][u] / th : 0.f;
        }
    }
}

// F+ [M, N] added into the parameter flows `pf` from pf0 on (`pf` empty: none) and F- [M, N] as F- / theta into the
// denominators `den` from den0 on (`den` empty: none), in the parameters' block-major order
void top_k_scatter_em(torch::Tensor FP, torch::Tensor FM, torch::Tensor params, torch::Tensor pf, torch::Tensor den,
                      int64_t p0, int64_t pf0, int64_t den0, int64_t Nb, int64_t cbs, int64_t bs) {
    const bool wp = pf.numel() > 0, wd = den.numel() > 0;
    if (!wp && !wd) return;
    const torch::Tensor& ref = wp ? FP : FM;
    const c10::cuda::CUDAGuard guard(ref.device());
    if (wp) { check(FP, at::kFloat, "FP"); check(pf, at::kFloat, "param flows"); }
    if (wd) { check(FM, at::kFloat, "FM"); check(den, at::kFloat, "denominators"); check(params, at::kFloat, "params"); }
    const int M = ref.size(0), N = ref.size(1);
    TORCH_CHECK(M % bs == 0 && N % cbs == 0 && N == Nb * cbs && (!wp || !wd || FM.sizes() == FP.sizes()),
                "top_k: the statistics' grid");
    dim3 grid((unsigned)((N + 31) / 32), (unsigned)((M + 31) / 32)), block(32, 8);
    launch_pdl(tk_scatter_em_kernel, grid, block, 0, at::cuda::getCurrentCUDAStream(),
        wp ? FP.data_ptr<float>() : nullptr, wd ? FM.data_ptr<float>() : nullptr, wd ? params.data_ptr<float>() : nullptr,
        wp ? pf.data_ptr<float>() : nullptr, wd ? den.data_ptr<float>() : nullptr, p0, pf0, den0, M, N, (int)Nb,
        (int)cbs, (int)bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_prep", &top_k_prep, "a LowRank(top_k) sum node vector's operands and work items for `top_k_forward`");
    m.def("top_k_items", &top_k_items, "the paired work items of the node vectors reading a leaf layer, one launch");
    m.def("top_k_forward", &top_k_forward,
          "a LowRank(top_k) sum node vector's values and its backward's stash: a persistent stream-K launch for each "
          "kind of work item");
    m.def("top_k_flows", &top_k_flows, "the node flows into the forward's stash, in place; the flows, their largest");
    m.def("top_k_bw_prep", &top_k_bw_prep, "the backward's operands beside the prep's: fp16 A', l2t, l2t for the cols");
    m.def("top_k_bw_em_observed", &top_k_bw_em_observed,
          "the backward's EM statistics of the observed samples (a tile of the parameters a CTA, over the samples)");
    m.def("top_k_bw_grads", &top_k_bw_grads,
          "the backward's dA and cols passes: the element flows and the gradients, added into their destinations");
    m.def("top_k_em_index", &top_k_em_index, "the samples the backward's EM statistics take, in the kernels' terms");
    m.def("top_k_scatter_ef", &top_k_scatter_ef, "the children's element flows from the backward's EF (log space)");
    m.def("top_k_scatter_em", &top_k_scatter_em,
          "the EM statistics onto a node vector's slots of the parameter flows and the denominators (block-major)");
}
