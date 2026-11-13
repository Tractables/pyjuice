// The pass's end's two products, each ONE split-K GEMM whose operands are formed in it (the fused pass's end): over the
// rows (copy, sample, rank) of the exponentials the pass keeps -- Ua [M, rows] (the backward's exp(A - am)) and Vt
// [N, rows] (the forward's V) --
//   F+ = Ud Vd^T,  Ud[i, row] = Ua[i, row] exp(lf - log S + cm_b + am + shift_row),  Vd[j, row] = Vt[j, row] exp(w - cm_b)
//   W  = Un Vt^T,  Un[i, row] = Ua[i, row] exp(lf - log Z + am + shift_row)
// as `lowrank_mixture.cu`'s `kept_operands_kernel` forms them (the same guards: a node of no flow or no mass, no
// normalizer, or an operand entry 0: 0) -- but never written: the factors are split into a (node, sample)'s
// `exp(... + c_b)` and a row's `exp(shift_row - c_b)` (c_b the sample's largest shift over its ranks: <= 1, nothing past
// what the kept form would reach), tables written once (`pe_tables_kernel`, sample-major), and each GEMM stage of the
// kept operands scaled in place by them (fp32 products, one bf16 rounding: the kept form's but for the exp's split)
// before its MMAs. The GEMM: a CTA a (128 x 256 tile, K-split) item, the items split-major (a split's tiles together:
// its K-chunk shared in the L2); 8 warps (2 x 4, a 64 x 64 warp tile, 128 fp32 accumulators; 2 a sub-partition: up to
// 255 registers -- a 9th warp caps them at 168); thread 0 issuing the TMA boxes (k32 x 4 stages, the 64-byte swizzle);
// a stage scaled by every warp during the stage before's second MMA half (MEASURED: past its MMAs 8 points of the
// peak slower), then a named barrier; the splits' partials summed in split order through a turnstile a tile
// (deterministic). MEASURED (croissant, the reference HMM's shapes: M = N = 1024, rows 31 x B x 16, 47 splits; % of
// the mma.sync peak 443 TF/s): the plain GEMM 89% at B1024 / 80% at B256 (cuBLAS 73-79% / 80%), W 81% / 72%, F+ 68%
// / 64%.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <vector>
using namespace cute;
using bf16 = cutlass::bfloat16_t;

namespace lrpe {

constexpr int BM = 128, BN = 256, KT = 32, NW = 8, NTH = NW * 32, ST = 4;
constexpr int STAGE = (BM + BN) * KT * 2, ABYTES = BM * KT * 2;
constexpr int SMEM = ST * STAGE + 2 * ST * 8;
static_assert(SMEM <= 101376, "past sm_120's 99 KB a block");
using AtomK32 = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(bar)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n .reg .pred p;\n W_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra W_%=;\n}\n" ::"r"(a), "r"(phase) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((unsigned)__cvta_generic_to_shared(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t smem, const CUtensorMap* desc, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(smem), "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// ---------------- the copies' tables ----------------
// A copy's pointers and places (the kept operands' `KeptCopies`, and its slots in the two products): log Z, first node
// row, eid (its children's element rows), table, first, cv (its children's kept values), row0 (its rows of Ua / Vt),
// am's copy, its slot in F+ (-1: none), its slot in W (-1: none), the one-launch forward's tile maxima (0: none) and
// their node tiles
constexpr int CW = 12;
struct Copies { long long v[64 * CW]; };
struct Rows { long long r[64]; };                       // (a product's slots' first rows of Ua / Vt)

// The value of child `j` (element row `e`) at sample `b`: its two children's node values, or the kept `cv`
__device__ __forceinline__ float child_value(const float* cv, const int32_t* table, long e, long first, int j, int b,
                                             const float* node_mars, int B) {
    if (cv != nullptr) return cv[(long)j * B + b];
    const int32_t* ch = table + (e - first) * 2;
    return node_mars[(long)ch[0] * B + b] + node_mars[(long)ch[1] * B + b];
}

// cm_b of the copies in F+ (`kept_cmax_kernel`'s), in CMS slices of the children (a block 32 samples, its warps over
// every 8th child of its slice -- its partial into `cmp` [slot, slice, B]: `pe_tables_kernel` takes their max)
constexpr int CMS = 16;
__global__ void pe_cmax_kernel(const Copies copies, float* __restrict__ cmp, const float* __restrict__ node_mars, int N,
                               int B) {
    const long long* c = copies.v + CW * blockIdx.z;
    const int64_t* eid = reinterpret_cast<const int64_t*>(c[2]);
    const int32_t* table = reinterpret_cast<const int32_t*>(c[3]);
    const long first = (long)c[4];
    const float* cv = reinterpret_cast<const float*>(c[5]);
    const int qd = (int)c[8];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, b = blockIdx.x * 32 + lane;
    __shared__ float most[8][32];
    float m = -INFINITY;
    const int j0 = (int)((long)N * blockIdx.y / CMS), j1 = (int)((long)N * (blockIdx.y + 1) / CMS);
    const bool fwd = c[10] != 0;                         // (c_m from the forward's tile maxima: nothing to do)
    if (b < B && qd >= 0 && !fwd)
        for (int j = j0 + w; j < j1; j += 8) {
            const long e = eid[j];
            if (e < 0) continue;
            m = fmaxf(m, child_value(cv, table, e, first, j, b, node_mars, B));
        }
    most[w][lane] = m;
    __syncthreads();
    if (w != 0 || b >= B || qd < 0 || fwd) return;
    float sh = most[0][lane];
    for (int i = 1; i < 8; i++) sh = fmaxf(sh, most[i][lane]);
    cmp[((long)qd * CMS + blockIdx.y) * B + b] = sh;
}

// The tables of a copy (blockIdx.z), 32 rows (blockIdx.y) x 32 samples (blockIdx.x): its F+ slot's Ud factors fd [slot B
// + b, M] and Vd factors gd [slot B + b, N], its W slot's Un factors fn; and (row block 0) the rows' shift factors skd /
// skn [slot B K + b K + r]. Read sample-fastest, written row-fastest through the shared memory.
__global__ void pe_tables_kernel(const Copies copies, float* __restrict__ fd, float* __restrict__ gd,
                                 float* __restrict__ fn, float* __restrict__ skd, float* __restrict__ skn,
                                 const float* __restrict__ cmp, const float* __restrict__ am,
                                 const float* __restrict__ shb, const float* __restrict__ node_flows,
                                 const float* __restrict__ node_mars, int M, int N, int B, int K) {
    const long long* c = copies.v + CW * blockIdx.z;
    const float* lz = reinterpret_cast<const float*>(c[0]);
    const long nid0 = (long)c[1];
    const int64_t* eid = reinterpret_cast<const int64_t*>(c[2]);
    const int32_t* table = reinterpret_cast<const int32_t*>(c[3]);
    const long first = (long)c[4];
    const float* cv = reinterpret_cast<const float*>(c[5]);
    const long row0 = (long)c[6], acopy = (long)c[7];
    const int qd = (int)c[8], qn = (int)c[9];
    const int tx = threadIdx.x, ty = threadIdx.y;       // (32 x 8)
    const int b0 = blockIdx.x * 32, r0 = blockIdx.y * 32, b = b0 + tx;
    __shared__ float tile[3][32][33];
    __shared__ float csh[32], cmsh[32];                  // c_b: the sample's largest shift over its ranks; cm_b
    if (ty == 0) {
        float m = -INFINITY;
        if (b < B)
            for (int r = 0; r < K; r++) m = fmaxf(m, shb[row0 + (long)b * K + r]);
        csh[tx] = m > -INFINITY ? m : 0.f;
    } else if (ty == 1) {
        float m = -INFINITY;
        const float* cmo = reinterpret_cast<const float*>(c[10]);
        if (qd >= 0 && b < B) {
            if (cmo != nullptr)                          // (the forward's tile maxima: the same max over the children)
                for (int y = 0; y < (int)c[11]; y++) m = fmaxf(m, cmo[(long)y * B + b]);
            else
                for (int y = 0; y < CMS; y++) m = fmaxf(m, cmp[((long)qd * CMS + y) * B + b]);
        }
        cmsh[tx] = m;
    }
    __syncthreads();
    const float cb = csh[tx];
    const float cmb = cmsh[tx];
    for (int y = ty; y < 32; y += 8) {
        const int i = r0 + y;
        float vd = 0.f, vn = 0.f, vg = 0.f;
        if (b < B && i < M && (qd >= 0 || qn >= 0)) {
            const float lf = node_flows[(nid0 + i) * (long)B + b];
            const float z = lz[(long)i * B + b];
            const float ls = node_mars[(nid0 + i) * (long)B + b] + z;   // log S = log n + log Z
            const float a = am[(acopy * M + i) * B + b];
            if (qd >= 0 && lf > -INFINITY && ls > -INFINITY && cmb > -INFINITY) vd = expf(lf - ls + cmb + a + cb);
            if (qn >= 0 && lf > -INFINITY && z > -INFINITY && z < INFINITY) vn = expf(lf - z + a + cb);
        }
        if (qd >= 0 && b < B && i < N) {
            const long e = eid[i];
            if (e >= 0) {
                const float w = child_value(cv, table, e, first, i, b, node_mars, B);
                if (w > -INFINITY && cmb > -INFINITY) vg = expf(w - cmb);
            }
        }
        tile[0][y][tx] = vd; tile[1][y][tx] = vn; tile[2][y][tx] = vg;
    }
    __syncthreads();
    for (int y = ty; y < 32; y += 8) {                   // (row r0 + tx of sample b0 + y)
        const int bb = b0 + y, i = r0 + tx;
        if (bb >= B) continue;
        if (qd >= 0 && i < M) fd[((long)qd * B + bb) * M + i] = tile[0][tx][y];
        if (qn >= 0 && i < M) fn[((long)qn * B + bb) * M + i] = tile[1][tx][y];
        if (qd >= 0 && i < N) gd[((long)qd * B + bb) * N + i] = tile[2][tx][y];
    }
    if (blockIdx.y == 0 && b < B) {                      // the rows' shift factors exp(shift - c_b)
        for (int r = ty; r < K; r += 8) {
            const float s = shb[row0 + (long)b * K + r];
            const float f = s > -INFINITY ? expf(s - cb) : 0.f;
            if (qd >= 0) skd[((long)qd * B + b) * K + r] = f;
            if (qn >= 0) skn[((long)qn * B + b) * K + r] = f;
        }
    }
}

// ---------------- the GEMM: out = (A fA sK) (B fB)^T over the copies' rows, split-K ----------------
template <bool SB>
__global__ void __launch_bounds__(NTH, 1) pe_gemm_kernel(const __grid_constant__ CUtensorMap tA,
                                                       const __grid_constant__ CUtensorMap tB, float* __restrict__ out,
                                                       int* __restrict__ turn, const Rows row0, int M,
                                                       int N, int nks, int spc, int kper, int splits,
                                                       const float* __restrict__ fA, const float* __restrict__ fB,
                                                       const float* __restrict__ sK, int B, int lk) {
    extern __shared__ __align__(1024) unsigned char smem[];
    uint64_t* full = (uint64_t*)(smem + ST * STAGE);
    uint64_t* empty = full + ST;
    const int tid = threadIdx.x, lane = tid % 32;
    const int ntn = (N + BN - 1) / BN, ntiles = ((M + BM - 1) / BM) * ntn;
    const int tile = (int)blockIdx.x % ntiles, split = (int)blockIdx.x / ntiles;
    const int tm = tile / ntn, tn = tile % ntn;
    const int k0 = split * kper, nk = max(0, min(kper, nks - k0));
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    if (tid == 0) {
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], NW); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    // (the k-steps' copy slot q and k-step kk in it, counted on from the split's first: no divisions a k-step)
    int iq = k0 / spc, ikk = k0 - iq * spc;              // (thread 0: the next k-step to issue's)
    auto issue = [&](int g) {                            // (thread 0) the split's k-step g into stage g % ST, in order
        const int s = g % ST;
        const int col = (int)(row0.r[iq] + (long)ikk * KT);
        if (++ikk == spc) { ikk = 0; iq++; }
        const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(STAGE) : "memory");
        tma_load_2d(sbase + s * STAGE, &tA, col, tm * BM, bar);
        tma_load_2d(sbase + s * STAGE + ABYTES, &tB, col, tn * BN, bar);
    };
    if (tid == 0)
        for (int g = 0; g < min(ST, nk); g++) issue(g);
    // A thread's chunks (16 bytes: 8 k of a row) of a stage: A's rows tid / 4 + 64 u (u < 2), B's tid / 4 + 64 u (u <
    // 4), chunk c = tid % 4 of the row's 64 bytes (k 8c..8c+7: one sample, K % 8 == 0); their factors for k-step g,
    // loaded a stage ahead of their use (rows past M / N: 0 -- the boxes read zeros there)
    const int crow = tid / 4, cc = tid % 4;
    float fa[2], fb[4], sk[8];
    int fq = k0 / spc, fkk = k0 - fq * spc;              // (the next k-step to ask the factors of: in order)
    auto factors = [&](int g) {
        const long kg = (long)(k0 + g) * KT + 8 * cc;      // (the product's row of the chunk's first k)
        const long cb = (long)fq * B + ((fkk * KT + 8 * cc) >> lk);   // (its copy slot's sample: the rank 2^lk)
        if (++fkk == spc) { fkk = 0; fq++; }
        CUTE_UNROLL
        for (int u = 0; u < 2; u++) {
            const int i = tm * BM + crow + 64 * u;
            fa[u] = i < M ? fA[cb * M + i] : 0.f;
        }
        const float4 s0 = *(const float4*)&sK[kg];
        const float4 s1 = *(const float4*)&sK[kg + 4];
        sk[0] = s0.x; sk[1] = s0.y; sk[2] = s0.z; sk[3] = s0.w; sk[4] = s1.x; sk[5] = s1.y; sk[6] = s1.z; sk[7] = s1.w;
        if (SB) {
            CUTE_UNROLL
            for (int u = 0; u < 4; u++) {
                const int j = tn * BN + crow + 64 * u;
                fb[u] = j < N ? fB[cb * N + j] : 0.f;
            }
        }
    };
    auto scale_chunk = [&](uint32_t addr, const float (&f)[8]) {
        uint32_t v0, v1, v2, v3;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(addr));
        const uint32_t w[4] = {v0, v1, v2, v3};
        uint32_t o[4];
        CUTE_UNROLL
        for (int x = 0; x < 4; x++) {
            const float lo = __uint_as_float(w[x] << 16) * f[2 * x];
            const float hi = __uint_as_float(w[x] & 0xffff0000u) * f[2 * x + 1];
            __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
            o[x] = *reinterpret_cast<uint32_t*>(&p);
        }
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3])
                     : "memory");
    };
    auto scale = [&](int s) {                            // stage s's chunks, by the factors in hand
        const uint32_t st0 = sbase + s * STAGE;
        CUTE_UNROLL
        for (int u = 0; u < 2; u++) {
            const int r = crow + 64 * u;
            float f[8];
            CUTE_UNROLL
            for (int i = 0; i < 8; i++) f[i] = fa[u] * sk[i];
            scale_chunk(st0 + r * 64 + ((cc ^ ((r >> 1) & 3)) << 4), f);
        }
        if (SB) {
            CUTE_UNROLL
            for (int u = 0; u < 4; u++) {
                const int r = crow + 64 * u;
                float f[8];
                CUTE_UNROLL
                for (int i = 0; i < 8; i++) f[i] = fb[u];
                scale_chunk(st0 + ABYTES + r * 64 + ((cc ^ ((r >> 1) & 3)) << 4), f);
            }
        }
    };
    // ---- the MMAs: 2 x 4 warps, a 64 x 64 warp tile (the MMA tile 32 x 64: N 16 a warp, ldmatrix.x4 on B)
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_2, _4, _1>>{},
                                  Tile<Int<32>, Int<64>, _16>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    auto sAl = tile_to_shape(AtomK32{}, make_shape(Int<BM>{}, Int<KT>{}));
    auto sBl = tile_to_shape(AtomK32{}, make_shape(Int<BN>{}, Int<KT>{}));
    Tensor sA0 = make_tensor(make_smem_ptr((bf16*)smem), sAl);
    Tensor sB0 = make_tensor(make_smem_ptr((bf16*)(smem + ABYTES)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sA0);
    Tensor tCrB = thr.partition_fragment_B(sB0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bf16>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, bf16>{}, mma);
    auto tA_ = s2rA.get_thread_slice(tid);
    auto tB_ = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA_.retile_D(tCrA);
    Tensor tXrB = tB_.retile_D(tCrB);
    const auto tXsA0 = tA_.partition_S(sA0);
    const auto tXsB0 = tB_.partition_S(sB0);
    Tensor cC = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrC = thr.partition_fragment_C(cC);
    clear(tCrC);
    static_assert(size<2>(tCrA) == 2, "a k32 stage: two k16 blocks");
    auto head = [&](int s) {                             // a stage's first fragments (scaled, its barrier passed)
        const int so = s * (STAGE / 2);
        copy(s2rA, make_tensor(tXsA0.data() + so, tXsA0.layout())(_, _, 0), tXrA(_, _, 0));
        copy(s2rB, make_tensor(tXsB0.data() + so, tXsB0.layout())(_, _, 0), tXrB(_, _, 0));
    };
    if (nk > 0) {
        factors(0);
        mbar_wait(&full[0], 0);
        scale(0);
        if (nk > 1) factors(1);
        asm volatile("bar.sync 1, %0;" ::"n"(NTH) : "memory");
        head(0);
    }
    for (int t = 0; t < nk; t++) {
        const int s = t % ST, so = s * (STAGE / 2);
        auto tXsA = make_tensor(tXsA0.data() + so, tXsA0.layout());
        auto tXsB = make_tensor(tXsB0.data() + so, tXsB0.layout());
        copy(s2rA, tXsA(_, _, 1), tXrA(_, _, 1));
        copy(s2rB, tXsB(_, _, 1), tXrB(_, _, 1));
        cute::gemm(mma, tCrA(_, _, 0), tCrB(_, _, 0), tCrC);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // (the scaling's writes before the refill)
        if (lane == 0) mbar_arrive(&empty[s]);           // (the stage read: every fragment of it loaded)
        if (t + 1 < nk) {                                // the next stage scaled (its factors in hand), the one after's asked
            mbar_wait(&full[(t + 1) % ST], ((t + 1) / ST) & 1);
            scale((t + 1) % ST);
            if (t + 2 < nk) factors(t + 2);
        }
        if (tid == 0 && t >= 1 && t - 1 + ST < nk) {     // (the stage the k-step before read, refilled: no stall)
            mbar_wait(&empty[(t - 1) % ST], ((t - 1) / ST) & 1);
            issue(t - 1 + ST);
        }
        cute::gemm(mma, tCrA(_, _, 1), tCrB(_, _, 1), tCrC);
        if (t + 1 < nk) {
            asm volatile("bar.sync 1, %0;" ::"n"(NTH) : "memory");   // (every warp's chunks of it scaled)
            head((t + 1) % ST);
        }
    }
    // ---- the tile's sum, in split order (the turnstile: split s adds once split s - 1 has, split 0 stores; the last
    // sets it back to 0 for the next launch)
    if (tid == 0) {
        int v;
        do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(turn + tile) : "memory"); } while (v != split);
    }
    __syncthreads();
    // (through the L2: an SM's L1 may hold a stale line of the tile; a half's loads all issued before its stores --
    // stored in turn they are a round trip each, the compiler unable to tell the addresses apart)
    float* o = out + (long)tm * BM * N + (long)tn * BN;
    constexpr int NE = size(decltype(tCrC){}), HALF = NE / 2;
    static_assert(NE % 4 == 0, "pairs of adjacent columns, two halves");
    CUTE_UNROLL
    for (int h = 0; h < 2; h++) {
        float2 prev[HALF / 2];
        CUTE_UNROLL
        for (int x = 0; x < HALF / 2; x++) {
            const int i = h * HALF + 2 * x;              // (elements i, i + 1: columns c, c + 1 of row r)
            const int r = get<0>(tCcC(i)), cl = get<1>(tCcC(i));
            const bool in = tm * BM + r < M && tn * BN + cl + 1 < N && (N & 1) == 0;   // (pairs 8-byte aligned)
            prev[x] = make_float2(0.f, 0.f);
            if (split > 0 && in) prev[x] = __ldcg(reinterpret_cast<const float2*>(o + (long)r * N + cl));
        }
        CUTE_UNROLL
        for (int x = 0; x < HALF / 2; x++) {
            const int i = h * HALF + 2 * x;
            const int r = get<0>(tCcC(i)), cl = get<1>(tCcC(i));
            float* q = o + (long)r * N + cl;
            if (tm * BM + r < M && tn * BN + cl + 1 < N && (N & 1) == 0) {
                __stcg(reinterpret_cast<float2*>(q), make_float2(prev[x].x + tCrC(i), prev[x].y + tCrC(i + 1)));
            } else if (tm * BM + r < M) {                 // (an odd N's rows: the pair's columns one at a time)
                if (tn * BN + cl < N) __stcg(q, split == 0 ? tCrC(i) : __ldcg(q) + tCrC(i));
                if (tn * BN + cl + 1 < N) __stcg(q + 1, split == 0 ? tCrC(i + 1) : __ldcg(q + 1) + tCrC(i + 1));
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int nx = split + 1 == splits ? 0 : split + 1;
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(turn + tile), "r"(nx) : "memory");
    }
}

CUtensorMap kmap(const torch::Tensor& t, int rows_box) {
    CUtensorMap m;
    cuuint64_t gdim[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
    cuuint64_t gstride[1] = {(cuuint64_t)t.stride(0) * 2};
    cuuint32_t box[2] = {(cuuint32_t)KT, (cuuint32_t)rows_box};
    cuuint32_t es[2] = {1, 1};
    const CUresult rr = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), gdim, gstride, box,
                                               es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rr == CUDA_SUCCESS, "lowrank fused pass's end: cuTensorMapEncodeTiled failed (", (int)rr, ")");
    return m;
}

int sms() {
    static int n = 0;
    if (n == 0) {
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    }
    return n;
}

// One product into `out` [M, N] fp32: the copies' rows of `a` [M, rows] scaled by fA [slots B, M] x sK [slots B K], of
// `b` [N, rows] by fB [slots B, N] (or not: W), summed over the slots' rows (`row0` [slots]: each slot's B K rows)
void pe_product(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const Rows& row0, int slots,
             const torch::Tensor& fA, const c10::optional<torch::Tensor>& fB, const torch::Tensor& sK, int B, int K,
             torch::Tensor& turn, cudaStream_t st) {
    const int M = (int)a.size(0), N = (int)b.size(0);
    const int spc = B * K / KT, nks = slots * spc;
    int lk = 0;
    while ((1 << lk) < K) lk++;
    TORCH_CHECK((1 << lk) == K && K >= 8, "lowrank fused pass's end: the rank a power of 2, 8 or more");
    const int ntiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    // the splits: whole waves of the SMs, up to 8 of them, at least 128 k-steps a split (a wave's splits of a tile add
    // in turn at its end: its share of the wave kept small)
    const long waves = std::max<long>(1, std::min<long>(8, (long)nks * ntiles / ((long)sms() * 128)));
    const int splits = (int)std::max<long>(1, std::min<long>(nks, waves * sms() / ntiles));
    const int kper = (nks + splits - 1) / splits;
    TORCH_CHECK(turn.numel() >= ntiles, "lowrank fused pass's end: the turnstiles");
    const CUtensorMap ta = kmap(a, BM), tb = kmap(b, BN);
    auto go = [&](auto sbv) {
        constexpr bool SB_ = decltype(sbv)::value;
        auto k = pe_gemm_kernel<SB_>;
        static bool attr[64] = {};
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        if (dev >= 64 || !attr[dev]) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            if (dev < 64) attr[dev] = true;
        }
        k<<<dim3((unsigned)(ntiles * splits)), NTH, SMEM, st>>>(
            ta, tb, out.data_ptr<float>(), turn.data_ptr<int>(), row0, M, N, nks, spc, kper, splits,
            fA.data_ptr<float>(), SB_ ? fB->data_ptr<float>() : (const float*)nullptr, sK.data_ptr<float>(), B, lk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (fB.has_value()) go(std::true_type{});
    else go(std::false_type{});
}

void check(const torch::Tensor& t, const char* what, at::ScalarType dt, long numel) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous() && (numel < 0 || t.numel() >= numel),
                "lowrank fused pass's end: ", what);
}

}  // namespace lrpe
using namespace lrpe;

// The fused pass's end (the copies as `lowrank_kept_operands` takes them; `drow0[c]` >= 0: the copy in F+, `un[c]`: in
// W): F+ into `F` [M, N] and W into `W` [M, N] (fp32, written), neither operand written. `cmo[c]` (or none): the
// copy's one-launch forward's tile maxima [node tiles, B] (`lowrank_forward_om`'s), its c_m then theirs.
void lowrank_pass_end_fused(c10::optional<torch::Tensor> F, c10::optional<torch::Tensor> W, torch::Tensor ua,
                            torch::Tensor am, torch::Tensor vt, torch::Tensor shb, torch::Tensor node_flows,
                            torch::Tensor node_mars, std::vector<torch::Tensor> lz, std::vector<int64_t> nid0,
                            std::vector<c10::optional<torch::Tensor>> eid,
                            std::vector<c10::optional<torch::Tensor>> table, std::vector<int64_t> first,
                            std::vector<int64_t> row0, std::vector<int64_t> drow0, std::vector<int64_t> un, int64_t M,
                            int64_t K, std::vector<c10::optional<torch::Tensor>> cv,
                            std::vector<c10::optional<torch::Tensor>> cmo) {
    const int B = (int)node_flows.size(1), N = (int)vt.size(0);
    const long rows = ua.size(1), P = (long)lz.size(), R = (long)B * K;
    TORCH_CHECK(P <= 64, "lowrank fused pass's end: at most 64 copies");
    TORCH_CHECK(K >= 8 && (K & (K - 1)) == 0 && R % KT == 0,
                "lowrank fused pass's end: the rank a power of 2, 8 or more (whole 16-byte runs), B K % 32");
    TORCH_CHECK(ua.scalar_type() == torch::kBFloat16 && vt.scalar_type() == torch::kBFloat16 && ua.size(0) == M &&
                vt.size(1) == rows && ua.stride(1) == 1 && vt.stride(1) == 1 && ua.stride(0) % 8 == 0 &&
                vt.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(ua.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(vt.data_ptr()) & 15) == 0, "lowrank fused pass's end: Ua / Vt");
    check(am, "am", torch::kFloat32, -1);
    check(shb, "the shifts", torch::kFloat32, rows);
    check(node_flows, "node_flows", torch::kFloat32, -1);
    check(node_mars, "node_mars", torch::kFloat32, -1);
    const at::cuda::OptionalCUDAGuard guard(ua.device());
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK((long)cmo.size() == P, "lowrank fused pass's end: cmo a copy");
    Copies copies;
    std::vector<int64_t> rd, rn;
    bool cmax = false;                                   // (a copy in F+ whose c_m `pe_cmax` makes)
    for (long c = 0; c < P; c++) {
        check(lz[c], "lz", torch::kFloat32, (long)B * M);
        TORCH_CHECK(nid0[c] >= 0 && node_flows.size(0) >= nid0[c] + M, "lowrank fused pass's end: a copy's node rows");
        TORCH_CHECK(row0[c] >= 0 && row0[c] + R <= rows && row0[c] % R == 0, "lowrank fused pass's end: a copy's rows");
        const bool d = drow0[c] >= 0 && F.has_value(), n = un[c] != 0 && W.has_value();
        const bool kept = d && cv[c].has_value();
        TORCH_CHECK(!d || (eid[c].has_value() && (kept || table[c].has_value())),
                    "lowrank fused pass's end: a copy's data half");
        if (d) {
            check(*eid[c], "eid", torch::kInt64, N);
            if (kept) check(*cv[c], "the children's values", torch::kFloat32, (long)N * B);
            else check(*table[c], "table", torch::kInt32, -1);
        }
        long long* v = copies.v + CW * c;
        v[0] = (long long)lz[c].data_ptr<float>(); v[1] = (long long)nid0[c];
        v[2] = d ? (long long)eid[c]->data_ptr<int64_t>() : 0;
        v[3] = d && !kept ? (long long)table[c]->data_ptr<int32_t>() : 0;
        v[4] = (long long)first[c]; v[5] = kept ? (long long)cv[c]->data_ptr<float>() : 0;
        v[6] = (long long)row0[c]; v[7] = (long long)(row0[c] / R);
        v[8] = d ? (long long)rd.size() : -1; v[9] = n ? (long long)rn.size() : -1;
        const bool fw = d && cmo[c].has_value();
        if (fw) TORCH_CHECK(cmo[c]->is_cuda() && cmo[c]->is_contiguous() && cmo[c]->scalar_type() == torch::kFloat32 &&
                            cmo[c]->numel() % B == 0 && cmo[c]->numel() > 0, "lowrank fused pass's end: cmo [tiles, B]");
        v[10] = fw ? (long long)cmo[c]->data_ptr<float>() : 0; v[11] = fw ? (long long)(cmo[c]->numel() / B) : 0;
        cmax = cmax || (d && !fw);
        if (d) rd.push_back(row0[c]);
        if (n) rn.push_back(row0[c]);
    }
    const long Pd = (long)rd.size(), Pn = (long)rn.size();
    TORCH_CHECK(am.numel() >= (rows / R) * M * B, "lowrank fused pass's end: am");
    auto f32 = node_flows.options();
    auto i64 = node_flows.options().dtype(torch::kInt64);
    torch::Tensor cmp = torch::empty({std::max<long>(1, Pd) * CMS * B}, f32);
    torch::Tensor fd = torch::empty({std::max<long>(1, Pd * B * M)}, f32), gd = torch::empty({std::max<long>(1, Pd * B * N)}, f32);
    torch::Tensor fn = torch::empty({std::max<long>(1, Pn * B * M)}, f32);
    torch::Tensor skd = torch::empty({std::max<long>(1, Pd * R)}, f32), skn = torch::empty({std::max<long>(1, Pn * R)}, f32);
    if (cmax) {
        pe_cmax_kernel<<<dim3((B + 31) / 32, CMS, (unsigned)P), 256, 0, st>>>(copies, cmp.data_ptr<float>(),
                                                                              node_mars.data_ptr<float>(), N, B);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int rblocks = (int)((std::max<long>(M, N) + 31) / 32);
    pe_tables_kernel<<<dim3((B + 31) / 32, rblocks, (unsigned)P), dim3(32, 8), 0, st>>>(
        copies, fd.data_ptr<float>(), gd.data_ptr<float>(), fn.data_ptr<float>(), skd.data_ptr<float>(),
        skn.data_ptr<float>(), cmp.data_ptr<float>(), am.data_ptr<float>(), shb.data_ptr<float>(),
        node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), (int)M, N, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int ntiles = (int)(((M + BM - 1) / BM) * ((N + BN - 1) / BN));
    static std::vector<torch::Tensor> turns(64);           // (a device's turnstiles, zeroed once: each launch leaves 0)
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    TORCH_CHECK(dev < 64, "lowrank fused pass's end: device");
    if (!turns[dev].defined() || turns[dev].numel() < ntiles)
        turns[dev] = torch::zeros({std::max(ntiles, 64)}, node_flows.options().dtype(torch::kInt32));
    if (Pd > 0) {
        Rows r0;
        for (long q = 0; q < Pd; q++) r0.r[q] = rd[q];
        TORCH_CHECK(F->scalar_type() == torch::kFloat32 && F->is_contiguous() && F->size(0) == M && F->size(1) == N,
                    "lowrank fused pass's end: F");
        pe_product(*F, ua, vt, r0, (int)Pd, fd, gd, skd, B, (int)K, turns[dev], st);
    } else if (F.has_value()) {
        F->zero_();
    }
    if (Pn > 0) {
        Rows r0;
        for (long q = 0; q < Pn; q++) r0.r[q] = rn[q];
        TORCH_CHECK(W->scalar_type() == torch::kFloat32 && W->is_contiguous() && W->size(0) == M && W->size(1) == N,
                    "lowrank fused pass's end: W");
        pe_product(*W, ua, vt, r0, (int)Pn, fn, c10::nullopt, skn, B, (int)K, turns[dev], st);
    } else if (W.has_value()) {
        W->zero_();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_pass_end_fused", &lowrank_pass_end_fused,
          "the pass's end's F+ and W, each one split-K GEMM forming its operands from the kept exponentials");
}
