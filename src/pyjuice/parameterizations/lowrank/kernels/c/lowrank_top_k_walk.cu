// pyjuice: `juice.queries.sample`'s walk of a run of `LowRank(top_k)` levels (`lowrank/top_k_queries.py`, `walk`).
//
// A level: a sum layer of ONE node vector (M nodes over N children, the children elements of the product layer
// below) whose parameters read the token of the leaf beside it, and that product layer. ONE CTA a sample walks the
// levels top-down; at a level, the sample's node i (its frontier row's) draws
//
//     (s, j)  in proportion to  P(s | i) theta_b(j | i, s) [exp(c_j)]       theta_b(j | i, s) = W_ij E_sj / Z_is
//
// W_ij = theta[i, j] exp(A_i . B_j), E_sj = exp(G_sj), Z_is = sum_j W_ij E_sj; P(s | i) the leaf's probabilities of
// the sum's slots (its first K candidates one by one, the rest summed; an observed token's slot fixed); [exp(c_j)] the
// children's values where conditioning (a product's: its children's node_mars summed), U_is = sum_j W_ij E_sj
// exp(c_j) -- the slot's weight w_s U_is / Z_is --; then the leaf's candidate within the slot (the slot's own, else
// one of the rest by P(c | i); an observed token itself), into the leaf's record. The frontier row is cleared and the
// drawn product's children written into their rows, for the levels below (the same CTA: a barrier).
//
// A warp owns slots (lanes over the children: a slot's sums by shuffles alone); every draw a Gumbel-max over the
// counter-based uniforms of (seed, sample, level, draw, index). fp32 throughout.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int NT = 512;                     // threads a CTA (a sample)
constexpr int NWARP = NT / 32;
constexpr int NI = 24;                      // the levels' integer table's row (see top_k_queries.walk)
constexpr float NEG = -INFINITY;

struct Level {
    long row, nid0, M, N, r, re, Kl, lvid, K;
    long sA0, sA1, sA2, sB0, sB1, sB2, sG0, sG1, sG2, sU0, sU1, sU2, sD0, sD1, sD2;
    const float* A; const float* Bm; const float* G; const float* U; const float* D; const float* lth;
    const float* lpar; const long* spid; const long* eid; const long* cn; const long* cr; const long* place;
    int* cand;
};

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// a Gumbel(0, 1) draw keyed by (seed, key, idx)
__device__ __forceinline__ float gumbel(uint64_t seed, uint64_t key, uint64_t idx) {
    const uint64_t z = mix64(seed ^ mix64(key * 0x100000001B3ull ^ mix64(idx)));
    const float u = ((float)(z >> 40) + 0.5f) * (1.0f / 16777216.0f);
    return -__logf(-__logf(u));
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// (value, index) with the larger value (the smaller index on a tie)
__device__ __forceinline__ void warp_argmax(float& v, int& k) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int k2 = __shfl_xor_sync(0xffffffffu, k, o);
        if (v2 > v || (v2 == v && k2 < k)) { v = v2; k = k2; }
    }
}

__device__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = lane < NWARP ? red[lane] : NEG;
    return warp_max(v);
}

__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    v = lane < NWARP ? red[lane] : 0.f;
    return warp_sum(v);
}

__device__ int block_argmax(float v, int k, float* red, int* redk) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    warp_argmax(v, k);
    __syncthreads();
    if (lane == 0) { red[w] = v; redk[w] = k; }
    __syncthreads();
    v = lane < NWARP ? red[lane] : NEG;
    k = lane < NWARP ? redk[lane] : 0x7fffffff;
    warp_argmax(v, k);
    return k;
}

__device__ __forceinline__ float safe(float m) { return m == NEG ? 0.f : m; }

template <int CPT, int GPL = CPT * NT / 32>         // children a thread (N <= CPT * NT); a lane, a slot's
__global__ void __launch_bounds__(NT) tk_walk_kernel(
        const long* __restrict__ lvi, const long* __restrict__ lvp, int L, long* node_samples,
        const float* __restrict__ node_mars, int B, const long* __restrict__ ids, const float* __restrict__ lp,
        int ev_mapped, int nvars, int C, const long* __restrict__ emap, const int* __restrict__ seed_ptr,
        int cond, int has_place, int E_P, int maxN, int maxC) {
    extern __shared__ float sm[];
    float* w_s = sm;                                 // [maxN] exp(base - max)
    float* wv_s = w_s + maxN;                        // [maxN] w exp(c - max)
    float* lq_s = wv_s + maxN;                       // [maxC] log P(c | i)
    float* sc_s = lq_s + maxC;                       // [maxC + 1] the slots' log-weights, then their scores
    float* ab_s = sc_s + maxC + 1;                   // [128] A_i, then U_i
    float* red = ab_s + 256;                         // [NWARP]
    int* redk = reinterpret_cast<int*>(red + NWARP); // [NWARP]
    __shared__ Level lv;
    __shared__ long s_nid;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long b = blockIdx.x;
    const long xr = emap != nullptr ? emap[b] : b;
    const long er = ev_mapped ? xr : b;
    const uint64_t seed = (uint64_t)(uint32_t)seed_ptr[0];

    for (int l = 0; l < L; ++l) {
        if (tid == 0) {
            const long* qi = lvi + (long)l * NI;
            const long* qp = lvp + (long)l * NI;
            lv.row = qi[0]; lv.nid0 = qi[1]; lv.M = qi[2]; lv.N = qi[3]; lv.r = qi[4]; lv.re = qi[5]; lv.Kl = qi[6];
            lv.lvid = qi[7]; lv.K = qi[8];
            lv.sA0 = qi[9]; lv.sA1 = qi[10]; lv.sA2 = qi[11]; lv.sB0 = qi[12]; lv.sB1 = qi[13]; lv.sB2 = qi[14];
            lv.sG0 = qi[15]; lv.sG1 = qi[16]; lv.sG2 = qi[17]; lv.sU0 = qi[18]; lv.sU1 = qi[19]; lv.sU2 = qi[20];
            lv.sD0 = qi[21]; lv.sD1 = qi[22]; lv.sD2 = qi[23];
            lv.A = reinterpret_cast<const float*>(qp[0]); lv.Bm = reinterpret_cast<const float*>(qp[1]);
            lv.G = reinterpret_cast<const float*>(qp[2]); lv.U = reinterpret_cast<const float*>(qp[3]);
            lv.D = reinterpret_cast<const float*>(qp[4]); lv.lth = reinterpret_cast<const float*>(qp[5]);
            lv.lpar = reinterpret_cast<const float*>(qp[6]); lv.spid = reinterpret_cast<const long*>(qp[7]);
            lv.eid = reinterpret_cast<const long*>(qp[8]); lv.cn = reinterpret_cast<const long*>(qp[9]);
            lv.cr = reinterpret_cast<const long*>(qp[10]); lv.place = reinterpret_cast<const long*>(qp[11]);
            lv.cand = reinterpret_cast<int*>(qp[12]);
            s_nid = node_samples[lv.row * B + b];
        }
        __syncthreads();
        const long nid = s_nid;
        const bool live = nid >= lv.nid0 && nid < lv.nid0 + lv.M;
        const long i = live ? nid - lv.nid0 : 0;
        const int N = (int)lv.N, K = (int)lv.K, Kl = (int)lv.Kl, re = (int)lv.re, r = (int)lv.r;
        const uint64_t key = ((uint64_t)b * (uint64_t)L + (uint64_t)l) * 4u;
        int jstar = 0, cstar = 0;
        if (live) {
            // ---- the leaf's candidates: log P(c | i), then the slots' log-weights
            if (tid < re) ab_s[128 + tid] = lv.U[xr * lv.sU0 + i * lv.sU1 + tid * lv.sU2];
            __syncthreads();
            const long ev = (er * nvars + lv.lvid) * (long)C;
            const long sp = lv.spid[i];
            float lg = NEG;
            for (int c = tid; c < maxC; c += NT) lq_s[c] = NEG;
            if (tid < C) {
                const long tok = ids[ev + tid];
                const float th = lv.lpar[sp + tok];
                const int slc = tid < Kl ? tid : Kl;
                const float* dp = lv.D + xr * lv.sD0 + slc * lv.sD2;
                float t8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                int k = 0;
                for (; k + 7 < re; k += 8) {                           // (eight loads in flight)
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) t8[u] = fmaf(ab_s[128 + k + u], dp[(k + u) * lv.sD1], t8[u]);
                }
                for (; k < re; ++k) t8[0] = fmaf(ab_s[128 + k], dp[k * lv.sD1], t8[0]);
                const float tau = ((t8[0] + t8[1]) + (t8[2] + t8[3])) + ((t8[4] + t8[5]) + (t8[6] + t8[7]));
                lg = th > 0.f ? __logf(th) + lp[ev + tid] + tau : NEG;
            }
            const float mz = block_max(lg, red);
            const float zs = block_sum(lg == NEG ? 0.f : __expf(lg - safe(mz)), red);
            const float lz = zs > 0.f ? __logf(zs) + safe(mz) : 0.f;
            const float lq = lg == NEG ? NEG : lg - lz;
            if (tid < C) lq_s[tid] = lq;
            const bool rest = tid < C && tid >= K;
            const float mr = block_max(rest ? lq : NEG, red);
            const float rs = block_sum(rest && lq != NEG ? __expf(lq - safe(mr)) : 0.f, red);
            const float lrest = rs > 0.f ? __logf(rs) + safe(mr) : NEG;
            long pl = -2;
            if (has_place) pl = lv.place[b];
            for (int s = tid; s <= K; s += NT) {
                float v = s < K ? (s < C ? lq_s[s] : NEG) : lrest;
                if (pl != -2) v = (s == (pl < 0 ? 0 : (pl > K ? K : pl))) ? 0.f : NEG;
                sc_s[s] = v;
            }

            // ---- the children: base_j = log theta[i, j] + A_i . B_j (and c_j, conditioning)
            if (tid < r) ab_s[tid] = lv.A[xr * lv.sA0 + i * lv.sA1 + tid * lv.sA2];
            __syncthreads();
            float base[CPT], cj[CPT];
            bool okj[CPT];
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                const int j = tid + q * NT;
                okj[q] = j < N && lv.eid[j] >= 0;
                base[q] = okj[q] ? lv.lth[i * N + j] : NEG;
                cj[q] = 0.f;
            }
            const float* bp = lv.Bm + xr * lv.sB0;
            {
                // (every child of the thread, eight ranks at a time: 8 CPT loads in flight)
                float acc[CPT][8];
                #pragma unroll
                for (int q = 0; q < CPT; ++q) {
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) acc[q][u] = 0.f;
                }
                int k = 0;
                for (; k + 7 < r; k += 8) {
                    #pragma unroll
                    for (int q = 0; q < CPT; ++q) {
                        const int j = okj[q] ? tid + q * NT : 0;
                        const float* bj = bp + (long)j * lv.sB2;
                        #pragma unroll
                        for (int u = 0; u < 8; ++u) acc[q][u] = fmaf(ab_s[k + u], bj[(k + u) * lv.sB1], acc[q][u]);
                    }
                }
                for (; k < r; ++k) {
                    #pragma unroll
                    for (int q = 0; q < CPT; ++q) {
                        const int j = okj[q] ? tid + q * NT : 0;
                        acc[q][0] = fmaf(ab_s[k], bp[(long)j * lv.sB2 + k * lv.sB1], acc[q][0]);
                    }
                }
                #pragma unroll
                for (int q = 0; q < CPT; ++q)
                    if (okj[q]) base[q] += ((acc[q][0] + acc[q][1]) + (acc[q][2] + acc[q][3])) +
                                           ((acc[q][4] + acc[q][5]) + (acc[q][6] + acc[q][7]));
            }
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                const int j = tid + q * NT;
                if (!okj[q]) continue;
                if (cond) {
                    float v = 0.f;
                    for (int e = 0; e < E_P; ++e) {
                        const long cnode = lv.cn[(long)j * E_P + e];
                        if (cnode > 0) v += node_mars[cnode * B + b];
                    }
                    cj[q] = v;
                }
            }
            if (cond) {
                // each slot's log U / Z: w = exp(base - max), wv = w exp(c - max) in shared memory, a warp a slot
                float mb = NEG, mc = NEG;
                #pragma unroll
                for (int q = 0; q < CPT; ++q) {
                    mb = fmaxf(mb, base[q]);
                    if (okj[q]) mc = fmaxf(mc, cj[q]);
                }
                mb = safe(block_max(mb, red));
                mc = safe(block_max(mc, red));
                #pragma unroll
                for (int q = 0; q < CPT; ++q) {
                    const int j = tid + q * NT;
                    if (j < maxN) {
                        const float w = okj[q] ? __expf(base[q] - mb) : 0.f;
                        w_s[j] = w;
                        wv_s[j] = okj[q] ? w * __expf(cj[q] - mc) : 0.f;
                    }
                }
                __syncthreads();
                const float* gp = lv.G + xr * lv.sG0;
                for (int s = warp; s <= K; s += NWARP) {
                    const float* gs = gp + s * lv.sG1;
                    float gv[GPL];                                     // (the row's loads all in flight at once)
                    #pragma unroll
                    for (int q = 0; q < GPL; ++q) {
                        const int j = lane + 32 * q;
                        gv[q] = j < N ? gs[(long)j * lv.sG2] : NEG;
                    }
                    float mg = NEG;
                    #pragma unroll
                    for (int q = 0; q < GPL; ++q) {
                        const int j = lane + 32 * q;
                        if (j < N && w_s[j] > 0.f) mg = fmaxf(mg, gv[q]);
                    }
                    mg = safe(warp_max(mg));
                    float z = 0.f, u = 0.f;
                    #pragma unroll
                    for (int q = 0; q < GPL; ++q) {
                        const int j = lane + 32 * q;
                        const float w = j < N ? w_s[j] : 0.f;
                        if (w > 0.f) {
                            const float e = __expf(gv[q] - mg);
                            z = fmaf(e, w, z);
                            u = fmaf(e, wv_s[j], u);
                        }
                    }
                    z = warp_sum(z);
                    u = warp_sum(u);
                    if (lane == 0) sc_s[s] = (z > 0.f && u > 0.f && sc_s[s] != NEG) ? sc_s[s] + __logf(u) - __logf(z) : NEG;
                }
                __syncthreads();
            }

            // ---- the slot, then the child given it, then the leaf's candidate within the slot
            float bv = NEG;
            int bk = 0x7fffffff;
            for (int s = tid; s <= K; s += NT) {
                const float v = sc_s[s] == NEG ? NEG : sc_s[s] + gumbel(seed, key, (uint64_t)s);
                if (v > bv || (v == bv && s < bk)) { bv = v; bk = s; }
            }
            int sstar = block_argmax(bv, bk, red, redk);
            if (sstar < 0 || sstar > K) sstar = 0;                           // (no slot with mass: none on a path)
            const float* gsel = lv.G + xr * lv.sG0 + (long)sstar * lv.sG1;
            bv = NEG; bk = 0x7fffffff;
            #pragma unroll
            for (int q = 0; q < CPT; ++q) {
                const int j = tid + q * NT;
                if (!okj[q]) continue;
                float z = base[q] + gsel[j * lv.sG2] + (cond ? cj[q] : 0.f);
                if (z != NEG) z += gumbel(seed, key + 1, (uint64_t)j);
                if (z > bv || (z == bv && j < bk)) { bv = z; bk = j; }
            }
            jstar = block_argmax(bv, bk, red, redk);
            if (jstar < 0 || jstar >= N) jstar = 0;
            if (pl != -2) {
                cstar = pl < 0 ? 0 : (int)pl;
            } else if (sstar < K) {
                cstar = sstar;
            } else {
                bv = NEG; bk = 0x7fffffff;
                for (int c = K + tid; c < C; c += NT) {
                    const float v = lq_s[c] == NEG ? NEG : lq_s[c] + gumbel(seed, key + 2, (uint64_t)c);
                    if (v > bv || (v == bv && c < bk)) { bv = v; bk = c; }
                }
                cstar = block_argmax(bv, bk, red, redk);
                if (cstar == 0x7fffffff) cstar = K;
            }
            if (tid == 0) lv.cand[b] = cstar;
        }
        // ---- the level's row cleared, the drawn product's children into their rows
        __syncthreads();
        if (tid == 0) node_samples[lv.row * B + b] = -1;
        __syncthreads();
        if (live && tid < E_P) {
            const long cnode = lv.cn[(long)jstar * E_P + tid], crow = lv.cr[(long)jstar * E_P + tid];
            if (cnode > 0 && crow >= 0) node_samples[crow * B + b] = cnode;
        }
        __syncthreads();
    }
}

}  // namespace

// The walk of `L` levels (`top_k_queries.walk`'s tables `lvi`, `lvp` [L, 24] int64: the levels' integers and
// addresses) for every sample of `node_samples` [rows, B] (int64, the frontier: read and written); `node_mars`
// [nodes, B] float32 (conditioning; else empty); the evidence `ids` (int64) and `lp` (float32) [rows of evidence,
// nvars, C] -- of the external tensors' rows where `ev_mapped`, through `emap` [B] (int64; empty: a sample its own
// row) --; `seed` [1] int32; `E_P` the product layers' widest fan-in, `maxN` / `maxC` the widest level's children and
// the candidates (with the slots: K + 1 <= maxC + 1).
void top_k_walk(torch::Tensor lvi, torch::Tensor lvp, int64_t L, torch::Tensor node_samples, torch::Tensor node_mars,
                torch::Tensor ids, torch::Tensor lp, torch::Tensor emap, int64_t ev_mapped, torch::Tensor seed,
                bool cond, bool has_place, int64_t E_P, int64_t maxN, int64_t maxC) {
    const c10::cuda::CUDAGuard guard(node_samples.device());
    TORCH_CHECK(lvi.is_cuda() && lvi.scalar_type() == at::kLong && lvi.size(1) == NI && lvp.sizes() == lvi.sizes(),
                "top_k walk: the levels' tables [L, 24] int64");
    TORCH_CHECK(node_samples.scalar_type() == at::kLong && node_samples.is_contiguous(), "top_k walk: node_samples");
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() && lp.scalar_type() == at::kFloat &&
                lp.is_contiguous() && lp.sizes() == ids.sizes(), "top_k walk: ids int64, lp float32, [rows, nvars, C]");
    TORCH_CHECK(seed.scalar_type() == at::kInt && seed.numel() >= 1, "top_k walk: seed [1] int32");
    const int B = (int)node_samples.size(1), C = (int)ids.size(2);
    TORCH_CHECK(C <= NT && maxC >= C && maxN <= 4 * NT && E_P <= NT, "top_k walk: sizes (a candidate a thread)");
    if (cond) TORCH_CHECK(node_mars.scalar_type() == at::kFloat && node_mars.is_contiguous() &&
                          node_mars.size(1) == B, "top_k walk: node_mars [nodes, B] float32");
    if (emap.numel() > 0) TORCH_CHECK(emap.scalar_type() == at::kLong && emap.numel() == B, "top_k walk: emap [B]");
    const size_t smem = (size_t)(2 * maxN + 2 * maxC + 1 + 256 + 2 * NWARP) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    const long* ep = emap.numel() > 0 ? emap.data_ptr<long>() : nullptr;
    const float* nm = cond ? node_mars.data_ptr<float>() : nullptr;
#define WALK(CPT)                                                                                                   \
    {                                                                                                               \
        if (smem > 48 * 1024) cudaFuncSetAttribute(tk_walk_kernel<CPT>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                   (int)smem);                                                      \
        tk_walk_kernel<CPT><<<B, NT, smem, stream>>>(lvi.data_ptr<long>(), lvp.data_ptr<long>(), (int)L,            \
            node_samples.data_ptr<long>(), nm, B, ids.data_ptr<long>(), lp.data_ptr<float>(), (int)ev_mapped,       \
            (int)ids.size(1), C, ep, seed.data_ptr<int>(), (int)cond, (int)has_place, (int)E_P, (int)maxN, (int)maxC); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
    }
    if (maxN <= NT) WALK(1)
    else if (maxN <= 2 * NT) WALK(2)
    else WALK(4)
#undef WALK
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_walk", &top_k_walk, "a run of LowRank(top_k) levels drawn top-down, a CTA a sample");
}
