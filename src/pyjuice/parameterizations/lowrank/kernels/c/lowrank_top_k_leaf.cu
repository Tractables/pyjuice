// pyjuice: the passes of `LowRank(top_k)` soft-evidence leaves (`lowrank/top_k_leaf_fused.py`).
//
// A leaf node vector of M nodes over V categories; a sample's C candidates c (token ids_b(c), evidence lp_b(c)) and its
// tilt tau_b[i, s] = A_b[i] . B_b[:, s] (the tilt's slot s = min(c, Kl), Kl its top_k):
//
//     logit_b(i, c) = log theta[i, ids_b(c)] + lp_b(c) + tau_b[i, min(c, Kl)],     Z_b(i) = sum_c exp(logit_b(i, c))
//
//   tk_leaf_table     log2 theta^T [V, M] from the layer's parameters: a candidate's M nodes one contiguous read
//   tk_leaf_forward   a CTA a (sample, node vector, up to 1024 nodes): log2 Z, log2 of the bulk's sum (the candidates
//                     past the head H = max(K, Kl): no slot of their own), the observed token's place and log P(x | i)
//                     (the node values), and where the token is missing and a sum reads it the slots' log-weights (the
//                     sum's K slots one by one, the rest summed)
//   tk_leaf_backward  the same tiles: the evidence's gradient (summed over the nodes, into the caller's) and the tilt's
//                     (dtau [S, M] a sample: d / d tau) from the node flows (an observed token) or the sum's posterior
//                     counts of its slots (a missing token a sum reads)
//
// A warp takes 128 nodes (4 a lane: a candidate's row one float4 a lane) and walks the candidates. Every sum in fp32,
// log2 units, each node's exponentials against a lazily raised running max (raised only past 8).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <climits>

namespace {

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float FLOOR = -1e30f;     // a running max's start: finite, so a -inf term gives 2^-inf = 0, not NaN
constexpr int CH = 512;             // candidates staged in shared memory at once
constexpr int NW = 8;               // warps a CTA (128 nodes each)

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float4 ld4(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

__device__ __forceinline__ void st4(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

__device__ __forceinline__ float at(const float4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }

// a running log2-sum-exp2: the sum of 2^(x - m) over the terms so far
struct Lse { float m, s; };

__device__ __forceinline__ void lse_add(Lse& a, float x) {
    if (x > a.m + 8.f) { a.s *= ex2(a.m - x); a.m = x; }
    a.s += ex2(x - a.m);
}

__device__ __forceinline__ Lse lse_merge(const Lse& a, const Lse& b) {
    const float m = fmaxf(a.m, b.m);
    return {m, a.s * ex2(a.m - m) + b.s * ex2(b.m - m)};
}

__device__ __forceinline__ float lse_log2(const Lse& a) { return a.s > 0.f ? a.m + __log2f(a.s) : -INFINITY; }

// lane l ends with the warp's sum of p[l]
__device__ __forceinline__ float xreduce32(float (&p)[32], int lane) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
        const bool up = (lane & d) != 0;
#pragma unroll
        for (int u = 0; u < d; ++u) {
            const float send = up ? p[u] : p[u + d];
            const float keep = up ? p[u + d] : p[u];
            p[u] = keep + __shfl_xor_sync(0xffffffffu, send, d);
        }
    }
    return p[0];
}

__global__ void tk_leaf_table_kernel(const float* __restrict__ params, const long* __restrict__ spids,
                                     float* __restrict__ tab, int M, int V) {
    __shared__ float tile[32][33];
    const int v0 = blockIdx.x * 32, i0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
    for (int k = ty; k < 32; k += 8) {
        const int i = i0 + k, v = v0 + tx;
        tile[k][tx] = (i < M && v < V) ? params[spids[i] + v] : 1.f;
    }
    __syncthreads();
    for (int k = ty; k < 32; k += 8) {
        const int v = v0 + k, i = i0 + tx;
        if (v < V && i < M) tab[(long)v * M + i] = __log2f(tile[tx][k]);
    }
}

// what a tile reads: a node vector's table, tilt, evidence, token
struct Tile {
    const float* tab; long tab_stride; const int* nv_tab;       // log2 theta^T [ntab][V][M]; a node vector's table
    const float* tau; int Bx; const int* emap;                   // tilts [T][Bx][S][M] (log2; null: none); a sample's row
    const int* ids; const float* lp; int nvars; const int* nv_lvid;   // the evidence [B][nvars][C]; a node vector's
    const int* nv_row0;                                          // a node vector's first row of node values / flows
    int B, M, C, S, K, H;                                        // S = Kl + 1; H = min(C, max(K, Kl))
};

struct FwArgs {
    Tile g;
    const int* tok; const unsigned char* miss;                   // [T][B] the token, missing
    float* node_mars;                                            // [rows][B]
    float* lz; float* lb; int* place;                            // [T][B][M] log2 Z, log2 of the bulk's sum; [T][B]
    const int* wrow; float* logw;                                // [T][B] -> row of logw [R][M][K + 1] (-1: none)
};

__global__ void __launch_bounds__(NW * 32) tk_leaf_forward_kernel(const FwArgs a) {
    __shared__ int s_ids[CH];
    __shared__ float s_lp[CH];
    __shared__ int s_place;
    const Tile& g = a.g;
    const int b = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int blk = blockIdx.z * NW + (threadIdx.x >> 5);             // this warp's 128 nodes
    const bool act = blk * 128 < g.M;
    const int i0 = act ? blk * 128 + 4 * lane : 0;
    const long tb = (long)t * g.B + b;
    const int tok = a.tok[tb];
    const bool miss = a.miss[tb] != 0;
    const long e0 = ((long)b * g.nvars + g.nv_lvid[t]) * g.C;
    const float* tab = g.tab + (long)g.nv_tab[t] * g.tab_stride + i0;
    const int Kl = g.S - 1;
    const float* tau = g.tau == nullptr ? nullptr
                     : g.tau + ((long)t * g.Bx + (g.emap != nullptr ? g.emap[b] : b)) * g.S * g.M + i0;
    float4 tK = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tau != nullptr && act) tK = ld4(tau + (long)Kl * g.M);
    Lse own[4], mid[4], bulk[4];                                       // c < K; K <= c < H; c >= H
#pragma unroll
    for (int j = 0; j < 4; ++j) { own[j] = {FLOOR, 0.f}; mid[j] = {FLOOR, 0.f}; bulk[j] = {FLOOR, 0.f}; }
    if (threadIdx.x == 0) s_place = INT_MAX;

    for (int c0 = 0; c0 < g.C; c0 += CH) {
        const int n = min(CH, g.C - c0);
        __syncthreads();
        for (int k = threadIdx.x; k < n; k += NW * 32) {
            const int id = g.ids[e0 + c0 + k];
            s_ids[k] = id;
            s_lp[k] = g.lp[e0 + c0 + k] * LOG2E;
            if (!miss && id == tok) atomicMin(&s_place, c0 + k);
        }
        __syncthreads();
        if (!act) continue;
        int k = 0;
        for (; k < n && c0 + k < g.H; ++k) {                           // the head: slots of their own
            const int c = c0 + k;
            const float4 th = ld4(tab + (long)s_ids[k] * g.M);
            const float4 tc = tau != nullptr ? ld4(tau + (long)min(c, Kl) * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float l = s_lp[k];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float x = at(th, j) + l + at(tc, j);
                if (c < g.K) lse_add(own[j], x); else lse_add(mid[j], x);
            }
        }
        for (; k + 8 <= n; k += 8) {                                   // the bulk (the tilt's slot Kl: added after)
            float4 th[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) th[u] = ld4(tab + (long)s_ids[k + u] * g.M);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float l = s_lp[k + u];
                lse_add(bulk[0], th[u].x + l); lse_add(bulk[1], th[u].y + l);
                lse_add(bulk[2], th[u].z + l); lse_add(bulk[3], th[u].w + l);
            }
        }
        for (; k < n; ++k) {
            const float4 th = ld4(tab + (long)s_ids[k] * g.M);
            const float l = s_lp[k];
            lse_add(bulk[0], th.x + l); lse_add(bulk[1], th.y + l); lse_add(bulk[2], th.z + l); lse_add(bulk[3], th.w + l);
        }
    }
    if (!act) return;                                                  // (no barrier below)

    float lzs[4], lbs[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        bulk[j].m += at(tK, j);
        lzs[j] = lse_log2(lse_merge(lse_merge(own[j], mid[j]), bulk[j]));
        lbs[j] = lse_log2(bulk[j]);
    }
    st4(a.lz + tb * g.M + i0, lzs[0], lzs[1], lzs[2], lzs[3]);
    st4(a.lb + tb * g.M + i0, lbs[0], lbs[1], lbs[2], lbs[3]);
    const int p = s_place;
    if (blockIdx.z == 0 && threadIdx.x == 0) a.place[tb] = miss ? -2 : (p == INT_MAX ? -1 : p);

    if (!miss) {                                                       // the node values: log P(x | i)
        float v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        if (p != INT_MAX) {
            const float4 th = ld4(tab + (long)g.ids[e0 + p] * g.M);
            const float4 tc = tau == nullptr ? make_float4(0.f, 0.f, 0.f, 0.f) : p >= Kl ? tK : ld4(tau + (long)p * g.M);
            const float l = g.lp[e0 + p] * LOG2E;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = (at(th, j) + l + at(tc, j) - lzs[j]) * LN2;
        }
        const long r0 = g.nv_row0[t] + i0;
#pragma unroll
        for (int j = 0; j < 4; ++j) a.node_mars[(r0 + j) * g.B + b] = v[j];
    } else if (a.wrow != nullptr && a.wrow[tb] >= 0) {                // the slots' log-weights, as the sum reads them
        float* out = a.logw + ((long)a.wrow[tb] * g.M + i0) * (g.K + 1);
        for (int s = 0; s < g.K; ++s) {
            float x[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
            if (s < g.C) {
                const float4 th = ld4(tab + (long)g.ids[e0 + s] * g.M);
                const float4 tc = tau == nullptr ? make_float4(0.f, 0.f, 0.f, 0.f) : ld4(tau + (long)min(s, Kl) * g.M);
                const float l = g.lp[e0 + s] * LOG2E;
#pragma unroll
                for (int j = 0; j < 4; ++j) x[j] = (at(th, j) + l + at(tc, j) - lzs[j]) * LN2;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) out[j * (g.K + 1) + s] = x[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) out[j * (g.K + 1) + g.K] = (lse_log2(lse_merge(mid[j], bulk[j])) - lzs[j]) * LN2;
    }
}

struct BwArgs {
    Tile g;
    const float* lz; const float* lb; const int* place;          // the forward's
    const float* flows; int logspace;                            // node flows [rows][B]
    const int* wrow; const float* logw; const float* acnt;       // a missing token a sum reads: its row of the slots'
                                                                 // log-weights and posterior counts [R][M][K + 1]
    float* glp;                                                  // [B][nvars][C] (added into; null: not wanted)
    float* dtau;                                                 // [T][B][S][M] (written)
};

__global__ void __launch_bounds__(NW * 32) tk_leaf_backward_kernel(const BwArgs a) {
    __shared__ int s_ids[CH];
    __shared__ float s_lp[CH];
    const Tile& g = a.g;
    const int b = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int blk = blockIdx.z * NW + (threadIdx.x >> 5);
    const bool act = blk * 128 < g.M;
    const int i0 = act ? blk * 128 + 4 * lane : 0;
    const long tb = (long)t * g.B + b;
    const int Kl = g.S - 1;
    const int p = a.place[tb];                                         // >= 0 found, -1 none of the candidates, -2 missing
    const bool miss = p == -2;
    const int r = miss && a.wrow != nullptr ? a.wrow[tb] : -1;
    float* dt = a.dtau + tb * g.S * g.M + i0;
    if (p == -1 || (miss && r < 0)) {                                  // no gradient: a -inf value, or a token no one reads
        if (act) for (int s = 0; s < g.S; ++s) st4(dt + (long)s * g.M, 0.f, 0.f, 0.f, 0.f);
        return;
    }
    const long e0 = ((long)b * g.nvars + g.nv_lvid[t]) * g.C;
    const float* tab = g.tab + (long)g.nv_tab[t] * g.tab_stride + i0;
    const float* tau = g.tau == nullptr ? nullptr
                     : g.tau + ((long)t * g.Bx + (g.emap != nullptr ? g.emap[b] : b)) * g.S * g.M + i0;

    // a node's terms: dlogit(i, c) = P(i, c) g(i, c) + d(i, c) -- observed: g = -f, d = f at the token; missing: below K
    // g = -f, d = the slot's count, from K on g = a_K / P(rest) - f (the rest's count spread as P), d = 0
    float lzs[4], gH[4], gB[4], qB[4], fo[4], dR[4];
    const float* ac = nullptr;
    if (act) {
        const float4 tK = tau != nullptr ? ld4(tau + (long)Kl * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 z = ld4(a.lz + tb * g.M + i0), lbv = ld4(a.lb + tb * g.M + i0);
        if (!miss) {
            const long r0 = g.nv_row0[t] + i0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float f = a.flows[(r0 + j) * g.B + b];
                fo[j] = a.logspace ? ex2(f * LOG2E) : f;
                gH[j] = gB[j] = -fo[j];
            }
        } else {
            ac = a.acnt + ((long)r * g.M + i0) * (g.K + 1);
            const float* lw = a.logw + ((long)r * g.M + i0) * (g.K + 1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float f = 0.f;
                for (int s = 0; s <= g.K; ++s) f += ac[j * (g.K + 1) + s];
                const float pr = ex2(lw[j * (g.K + 1) + g.K] * LOG2E), aK = ac[j * (g.K + 1) + g.K];
                fo[j] = 0.f;
                gH[j] = -f;
                gB[j] = (pr > 0.f ? aK / pr : 0.f) - f;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lzs[j] = at(z, j);
            qB[j] = at(tK, j) - lzs[j];
            dR[j] = gB[j] * ex2(at(lbv, j) - lzs[j]);                  // the bulk's share of the rest slot
        }
    }

    const int Cend = a.glp != nullptr ? g.C : g.H;                    // (without the evidence's gradient: the head alone)
    for (int c0 = 0; c0 < Cend; c0 += CH) {
        const int n = min(CH, Cend - c0);
        __syncthreads();
        for (int k = threadIdx.x; k < n; k += NW * 32) {
            s_ids[k] = g.ids[e0 + c0 + k];
            s_lp[k] = g.lp[e0 + c0 + k] * LOG2E;
        }
        __syncthreads();
        if (!act) continue;
        for (int k0 = 0; k0 < n; k0 += 32) {
            float pt[32];
            const int cg = c0 + k0;
            if (cg < g.H) {                                            // a group with head candidates
#pragma unroll
                for (int u = 0; u < 32; ++u) {
                    const int k = k0 + u, c = cg + u;
                    pt[u] = 0.f;
                    if (k >= n) continue;
                    const float4 th = ld4(tab + (long)s_ids[k] * g.M);
                    const float l = s_lp[k];
                    if (c >= g.H) {
                        pt[u] = ex2(th.x + l + qB[0]) * gB[0] + ex2(th.y + l + qB[1]) * gB[1] +
                                ex2(th.z + l + qB[2]) * gB[2] + ex2(th.w + l + qB[3]) * gB[3];
                        continue;
                    }
                    const float4 tc = tau != nullptr ? ld4(tau + (long)min(c, Kl) * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
                    float dl[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float P = ex2(at(th, j) + l + at(tc, j) - lzs[j]);
                        const float d = miss ? (c < g.K ? ac[j * (g.K + 1) + c] : 0.f) : (c == p ? fo[j] : 0.f);
                        dl[j] = P * (c < g.K ? gH[j] : gB[j]) + d;
                    }
                    pt[u] = dl[0] + dl[1] + dl[2] + dl[3];
                    if (c < Kl) st4(dt + (long)c * g.M, dl[0], dl[1], dl[2], dl[3]);
                    else { dR[0] += dl[0]; dR[1] += dl[1]; dR[2] += dl[2]; dR[3] += dl[3]; }
                }
            } else {                                                   // the bulk
#pragma unroll
                for (int u0 = 0; u0 < 32; u0 += 8) {
                    float4 th[8];
#pragma unroll
                    for (int u = 0; u < 8; ++u)
                        th[u] = k0 + u0 + u < n ? ld4(tab + (long)s_ids[k0 + u0 + u] * g.M)
                                                : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const float l = k0 + u0 + u < n ? s_lp[k0 + u0 + u] : 0.f;
                        pt[u0 + u] = ex2(th[u].x + l + qB[0]) * gB[0] + ex2(th[u].y + l + qB[1]) * gB[1] +
                                     ex2(th[u].z + l + qB[2]) * gB[2] + ex2(th[u].w + l + qB[3]) * gB[3];
                    }
                }
            }
            const float tot = xreduce32(pt, lane);
            if (a.glp != nullptr && k0 + lane < n) atomicAdd(a.glp + e0 + cg + lane, tot);
        }
    }
    if (!act) return;
    if (!miss && p >= g.H) {                                           // the observed token past the head
        float s = fo[0] + fo[1] + fo[2] + fo[3];
#pragma unroll
        for (int d = 16; d >= 1; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
        if (lane == 0 && a.glp != nullptr) atomicAdd(a.glp + e0 + p, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) dR[j] += fo[j];
    }
    for (int s = min(g.C, Kl); s < Kl; ++s) st4(dt + (long)s * g.M, 0.f, 0.f, 0.f, 0.f);   // (slots past the candidates)
    st4(dt + (long)Kl * g.M, dR[0], dR[1], dR[2], dR[3]);
}

void check(const torch::Tensor& t, at::ScalarType dt, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), "top_k leaf: ", name,
                " must be a contiguous CUDA tensor of ", c10::toString(dt));
}

template <typename T> T* ptr_or_null(const torch::Tensor& t) { return t.numel() > 0 ? t.data_ptr<T>() : nullptr; }

Tile make_tile(const torch::Tensor& tab, const torch::Tensor& nv_tab, const torch::Tensor& tau, const torch::Tensor& emap,
               const torch::Tensor& ids, const torch::Tensor& lp, const torch::Tensor& nv_lvid,
               const torch::Tensor& nv_row0, int64_t B, int64_t S, int64_t K) {
    check(tab, at::kFloat, "tables"); check(nv_tab, at::kInt, "nv_tab"); check(ids, at::kInt, "ids");
    check(lp, at::kFloat, "lp"); check(nv_lvid, at::kInt, "nv_lvid"); check(nv_row0, at::kInt, "nv_row0");
    TORCH_CHECK(tab.dim() == 3 && ids.dim() == 3 && lp.sizes() == ids.sizes() && ids.size(0) == B, "top_k leaf: shapes");
    Tile g;
    const int M = (int)tab.size(2), C = (int)ids.size(2);
    TORCH_CHECK(M % 128 == 0, "top_k leaf: M % 128");
    g.tab = tab.data_ptr<float>(); g.tab_stride = tab.size(1) * tab.size(2); g.nv_tab = nv_tab.data_ptr<int>();
    g.tau = nullptr; g.Bx = (int)B;
    if (tau.numel() > 0) {
        check(tau, at::kFloat, "tau");
        TORCH_CHECK(tau.dim() == 4 && tau.size(2) == S && tau.size(3) == M, "top_k leaf: tau [T, Bx, S, M]");
        g.tau = tau.data_ptr<float>(); g.Bx = (int)tau.size(1);
    }
    g.emap = nullptr;
    if (emap.numel() > 0) { check(emap, at::kInt, "emap"); g.emap = emap.data_ptr<int>(); }
    g.ids = ids.data_ptr<int>(); g.lp = lp.data_ptr<float>(); g.nvars = (int)ids.size(1);
    g.nv_lvid = nv_lvid.data_ptr<int>(); g.nv_row0 = nv_row0.data_ptr<int>();
    g.B = (int)B; g.M = M; g.C = C; g.S = (int)S; g.K = (int)K;
    g.H = std::min<int>(C, std::max<int>((int)K, (int)S - 1));
    return g;
}

}  // namespace

// log2 theta^T [V, M] of a node vector's M nodes, node i's V probabilities at params[spids[i]:]
torch::Tensor top_k_leaf_table(torch::Tensor params, torch::Tensor spids, int64_t V) {
    check(params, at::kFloat, "params"); check(spids, at::kLong, "spids");
    const at::cuda::CUDAGuard guard(params.device());
    const int M = (int)spids.size(0);
    auto tab = torch::empty({V, M}, params.options());
    dim3 grid((unsigned)((V + 31) / 32), (unsigned)((M + 31) / 32)), block(32, 8);
    tk_leaf_table_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        params.data_ptr<float>(), spids.data_ptr<long>(), tab.data_ptr<float>(), M, (int)V);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return tab;
}

// the forward of T node vectors over B samples: writes their node values (observed tokens) into node_mars; returns
// (lz [T, B, M], lb [T, B, M], place [T, B], logw [R, M, K + 1]) -- R the rows wrow names (logw empty where none)
std::vector<torch::Tensor> top_k_leaf_forward(torch::Tensor tab, torch::Tensor nv_tab, torch::Tensor tau, torch::Tensor emap,
                                              torch::Tensor ids, torch::Tensor lp, torch::Tensor nv_lvid,
                                              torch::Tensor nv_row0, torch::Tensor tok, torch::Tensor miss,
                                              torch::Tensor wrow, int64_t R, int64_t S, int64_t K,
                                              torch::Tensor node_mars) {
    const at::cuda::CUDAGuard guard(tab.device());
    const int64_t T = nv_tab.size(0), B = tok.size(1);
    FwArgs a;
    a.g = make_tile(tab, nv_tab, tau, emap, ids, lp, nv_lvid, nv_row0, B, S, K);
    check(tok, at::kInt, "tok"); check(miss, at::kByte, "miss"); check(node_mars, at::kFloat, "node_mars");
    TORCH_CHECK(node_mars.size(1) == B, "top_k leaf: node_mars [rows, B]");
    const int M = a.g.M;
    auto f32 = tab.options();
    auto lz = torch::empty({T, B, M}, f32), lb = torch::empty({T, B, M}, f32);
    auto place = torch::empty({T, B}, tok.options());
    auto logw = torch::empty({std::max<int64_t>(R, 0), M, K + 1}, f32);
    a.tok = tok.data_ptr<int>(); a.miss = miss.data_ptr<uint8_t>(); a.node_mars = node_mars.data_ptr<float>();
    a.lz = lz.data_ptr<float>(); a.lb = lb.data_ptr<float>(); a.place = place.data_ptr<int>();
    a.wrow = nullptr; a.logw = nullptr;
    if (R > 0) { check(wrow, at::kInt, "wrow"); a.wrow = wrow.data_ptr<int>(); a.logw = logw.data_ptr<float>(); }
    dim3 grid((unsigned)B, (unsigned)T, (unsigned)((M + NW * 128 - 1) / (NW * 128)));
    tk_leaf_forward_kernel<<<grid, NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {lz, lb, place, logw};
}

// the backward of the same tiles: adds the evidence's gradient into glp [B, nvars, C] (empty: none); returns dtau
// [T, B, S, M]
torch::Tensor top_k_leaf_backward(torch::Tensor tab, torch::Tensor nv_tab, torch::Tensor tau, torch::Tensor emap,
                                  torch::Tensor ids, torch::Tensor lp, torch::Tensor nv_lvid, torch::Tensor nv_row0,
                                  torch::Tensor lz, torch::Tensor lb, torch::Tensor place, torch::Tensor flows,
                                  bool logspace, torch::Tensor wrow, torch::Tensor logw, torch::Tensor acnt,
                                  int64_t S, int64_t K, torch::Tensor glp) {
    const at::cuda::CUDAGuard guard(tab.device());
    const int64_t T = nv_tab.size(0), B = place.size(1);
    BwArgs a;
    a.g = make_tile(tab, nv_tab, tau, emap, ids, lp, nv_lvid, nv_row0, B, S, K);
    check(lz, at::kFloat, "lz"); check(lb, at::kFloat, "lb"); check(place, at::kInt, "place");
    check(flows, at::kFloat, "flows");
    if (glp.numel() > 0) {
        check(glp, at::kFloat, "glp");
        TORCH_CHECK(glp.sizes() == ids.sizes(), "top_k leaf: glp as lp");
    }
    const int M = a.g.M;
    auto dtau = torch::empty({T, B, S, M}, tab.options());
    a.lz = lz.data_ptr<float>(); a.lb = lb.data_ptr<float>(); a.place = place.data_ptr<int>();
    a.flows = flows.data_ptr<float>(); a.logspace = logspace ? 1 : 0;
    a.wrow = nullptr; a.logw = nullptr; a.acnt = nullptr;
    if (acnt.numel() > 0) {
        check(wrow, at::kInt, "wrow"); check(logw, at::kFloat, "logw"); check(acnt, at::kFloat, "acnt");
        TORCH_CHECK(acnt.sizes() == logw.sizes(), "top_k leaf: acnt as logw");
        a.wrow = wrow.data_ptr<int>(); a.logw = logw.data_ptr<float>(); a.acnt = acnt.data_ptr<float>();
    }
    a.glp = glp.numel() > 0 ? glp.data_ptr<float>() : nullptr; a.dtau = dtau.data_ptr<float>();
    dim3 grid((unsigned)B, (unsigned)T, (unsigned)((M + NW * 128 - 1) / (NW * 128)));
    tk_leaf_backward_kernel<<<grid, NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dtau;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_leaf_table", &top_k_leaf_table, "log2 theta^T of a soft-evidence leaf node vector");
    m.def("top_k_leaf_forward", &top_k_leaf_forward, "LowRank(top_k) leaves' forward");
    m.def("top_k_leaf_backward", &top_k_leaf_backward, "LowRank(top_k) leaves' backward");
}
