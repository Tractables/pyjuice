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
//                     counts of its slots (a missing token a sum reads: the slots' posterior a node, K + 1 of them, then
//                     the node's flow -- their product the counts); and the EM statistics (F+, F-) into transposed
//                     buffers [V, M] a block of parameters (`red.v4`: a lane's 4 nodes one atomic), F+ the posterior
//                     counts (the observed token's flow, a sum's counts, or f P where no one reads a missing token),
//                     F- f P
//   tk_leaf_fold      a transposed buffer of flows added into the layer's parameter flows
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

// p[0 .. 3] += (a, b, c, d) atomically (one vector reduction where the architecture has it)
__device__ __forceinline__ void red4(float* p, float a, float b, float c, float d) {
#if __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
#else
    atomicAdd(p, a); atomicAdd(p + 1, b); atomicAdd(p + 2, c); atomicAdd(p + 3, d);
#endif
}

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

__global__ void tk_leaf_fold_kernel(const float* __restrict__ fT, const long* __restrict__ spfids, float* __restrict__ pf,
                                    int M, int V, long off) {
    __shared__ float tile[32][33];
    const int v0 = blockIdx.x * 32, i0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
    for (int k = ty; k < 32; k += 8) {
        const int v = v0 + k, i = i0 + tx;
        tile[k][tx] = (v < V && i < M) ? fT[(long)v * M + i] : 0.f;
    }
    __syncthreads();
    for (int k = ty; k < 32; k += 8) {
        const int i = i0 + k, v = v0 + tx;
        if (v < V && i < M) pf[spfids[i] + off + v] += tile[tx][k];
    }
}

// what a tile reads: a node vector's table, tilt, evidence, token
struct Tile {
    const float* tab; long tab_stride; const int* nv_tab;       // log2 theta^T [ntab][V][M]; a node vector's table
    const float* tau; int Bx; const int* emap;                   // tilts [T][Bx][S][M] (log2; null: none); a sample's row
    const int* ids; const float* lp; int nvars; const int* nv_lvid;   // the evidence [B][nvars][C]; a node vector's
    const int* erow;                                             // the evidence's row a sample reads (null: its own)
    const int* nv_row0;                                          // a node vector's first row of node values / flows
    int B, M, C, S, K, H;                                        // S = Kl + 1; H = min(C, max(K, Kl))
};

struct FwArgs {
    Tile g;
    const int* tok; const unsigned char* miss;                   // [T][B] the token, missing
    float* node_mars;                                            // [rows][B]
    float* lz; float* lb; int* place;                            // [T][B][M] log2 Z, log2 of the bulk's sum; [T][B]
    const int* wrow; float* logw; int lws;                       // [T][B] -> row of logw [R][M][lws] (slots 0 .. K;
                                                                 // -1: none): lws = 72, the sums' operand stride
    const int* ord;                                              // [B] the samples by their evidence's row (null:
                                                                 // in their order)
};

__global__ void __launch_bounds__(NW * 32) tk_leaf_forward_kernel(const FwArgs a) {
    __shared__ int s_ids[CH];
    __shared__ float s_lp[CH];
    __shared__ int s_place;
    const Tile& g = a.g;
    // (a CTA a sample; with the evidence a row each, the samples in their rows' order: a row's samples run side by
    // side and read its candidates' table rows from the cache -- MEASURED faster than one CTA walking them for both)
    const int b = a.ord != nullptr ? a.ord[blockIdx.x] : blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31;
    const int blk = blockIdx.z * NW + (threadIdx.x >> 5);             // this warp's 128 nodes
    const bool act = blk * 128 < g.M;
    const int i0 = act ? blk * 128 + 4 * lane : 0;
    const long tb = (long)t * g.B + b;
    const int tok = a.tok[tb];
    const bool miss = a.miss[tb] != 0;
    const long e0 = ((long)(g.erow != nullptr ? g.erow[b] : b) * g.nvars + g.nv_lvid[t]) * g.C;
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
        float* out = a.logw + ((long)a.wrow[tb] * g.M + i0) * a.lws;
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
            for (int j = 0; j < 4; ++j) out[j * a.lws + s] = x[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) out[j * a.lws + g.K] = (lse_log2(lse_merge(mid[j], bulk[j])) - lzs[j]) * LN2;
    }
}

constexpr int LW_STRIDE = 72;          // the slots' log-weights' row stride (the sums' kernels' SP)

struct BwArgs {
    Tile g;
    const float* lz; const float* lb; const int* place;          // the forward's
    const float* flows; int logspace;                            // node flows [rows][B]
    const int* wrow; const float* logw; int lws; const float* acnt; int acs;   // a missing token a sum reads: its row of the slots'
                                                                 // log-weights [R][M][K + 1] and of the sum's posterior
                                                                 // [R][M][K + 2]: the slots', then the node's flow
    float* glp;                                                  // [B][nvars][C] (added into; null: not wanted)
    float* dtau;                                                 // [T][B][S][M] written, or (drow) [T][Bd][S][M] added
                                                                 // into: a sample's times dsgn into row drow; null: none
    const int* drow; const float* dsgn; int Bd;
    float* fpT; float* fmT;                                      // EM: F+, F- [ntab][V][M] (added into; null: none)
    const bool* pmask;                                           // [B] the samples the EM statistics take (null: all)
    const float* gsgn;                                           // [B] the evidence gradient's sign a sample (null: 1)
    const int* ord; const int* rank;                             // [B] the samples by their evidence's row and a
                                                                 // place's rank within its row (null: a sample a CTA)
};

// dtau's writes of a thread's 4 nodes: stores into the sample's row, or (fold) its values times its sign added into
// the row of tensors it read (the buffer zeroed; zeros then not written); none where no tilt gradient is wanted
struct DtauOut {
    float* p; float sg; bool fold;
    __device__ __forceinline__ void put(long off, float x0, float x1, float x2, float x3) const {
        if (p == nullptr) return;
        if (fold) red4(p + off, sg * x0, sg * x1, sg * x2, sg * x3);
        else st4(p + off, x0, x1, x2, x3);
    }
    __device__ __forceinline__ void zero(long off) const { if (p != nullptr && !fold) st4(p + off, 0.f, 0.f, 0.f, 0.f); }
};

// The backward of a CTA's samples: one (NS = 1), or (the evidence a row each) up to NS of a row, walking its candidates
// once -- the bulk's terms and the EM statistics summed over the samples before they are added, the head's a sample each
template <int NS>
__global__ void __launch_bounds__(NW * 32) tk_leaf_backward_kernel(const BwArgs a) {
    __shared__ int s_ids[CH];
    __shared__ float s_lp[CH];
    const Tile& g = a.g;
    const int t = blockIdx.y, lane = threadIdx.x & 31;
    int bs[NS], ns = 1;
    if constexpr (NS == 1) {
        bs[0] = blockIdx.x;
    } else {                                                           // (`ord` the samples by row, `rank` within it)
        const int q = blockIdx.x;
        if (a.rank[q] % NS != 0) return;                               // (the group's first place's CTA)
        bs[0] = a.ord[q];
        const int er = g.erow[bs[0]];
#pragma unroll
        for (int m = 1; m < NS; ++m) {
            const bool in = ns == m && q + m < g.B && g.erow[a.ord[q + m]] == er;
            bs[m] = in ? a.ord[q + m] : bs[0];
            ns += in ? 1 : 0;
        }
    }
    const int blk = blockIdx.z * NW + (threadIdx.x >> 5);
    const bool act = blk * 128 < g.M;
    const int i0 = act ? blk * 128 + 4 * lane : 0;
    const int Kl = g.S - 1;
    const bool fold = a.drow != nullptr;
    int p[NS], r[NS];
    bool miss[NS], em[NS], alone[NS], live[NS], grad[NS];
    float gs[NS];
    DtauOut dt[NS];
    bool any = false;
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        const int b = bs[m];
        const long tb = (long)t * g.B + b;
        p[m] = a.place[tb];                                            // >= 0 found, -1 none of the candidates, -2 missing
        miss[m] = p[m] == -2;
        r[m] = miss[m] && a.wrow != nullptr ? a.wrow[tb] : -1;
        dt[m] = DtauOut{a.dtau == nullptr ? nullptr
                        : a.dtau + ((long)t * (fold ? a.Bd : g.B) + (fold ? a.drow[b] : b)) * g.S * g.M + i0,
                        fold ? a.dsgn[b] : 1.f, fold};
        em[m] = m < ns && a.fmT != nullptr && p[m] != -1              // EM statistics: every token among the candidates
                && (a.pmask == nullptr || a.pmask[b]);
        alone[m] = miss[m] && r[m] < 0;                                // a missing token no one reads: its leaf is 1
        live[m] = m < ns && !(p[m] == -1 || (alone[m] && !em[m]));
        grad[m] = live[m] && !alone[m];                                // (a missing token no one reads: EM alone)
        gs[m] = a.gsgn != nullptr ? a.gsgn[b] : 1.f;                   // (the evidence gradient's sign: its row's)
        if (m < ns && !live[m] && act)                                 // no gradient: a -inf value, or a token no one reads
            for (int s = 0; s < g.S; ++s) dt[m].zero((long)s * g.M);
        any = any || live[m];
    }
    if (!any) return;
    const int b0 = bs[0];
    const long e0 = ((long)(g.erow != nullptr ? g.erow[b0] : b0) * g.nvars + g.nv_lvid[t]) * g.C;
    const float* tab = g.tab + (long)g.nv_tab[t] * g.tab_stride + i0;
    const float* tau = g.tau == nullptr ? nullptr
                     : g.tau + ((long)t * g.Bx + (g.emap != nullptr ? g.emap[b0] : b0)) * g.S * g.M + i0;
    // (one write of the tilt's gradient for the samples where they fold into one row; else a sample each)
    const bool merge = NS > 1 && fold && ns > 1 && a.drow[bs[0]] == a.drow[bs[NS - 1]];
    const DtauOut dtm{dt[0].p, 1.f, true};

    // a node's terms: dlogit(i, c) = P(i, c) g(i, c) + d(i, c) -- observed: g = -f, d = f at the token; missing: below K
    // g = -f, d = the slot's count, from K on g = a_K / P(rest) - f (the rest's count spread as P), d = 0. EM: F- += fm P,
    // F+ the posterior counts -- observed f at the token (after the walk), a missing token a sum reads its count below K
    // and cp P from K on, one no one reads f P (fm = cp = f, the leaf's flow). P is the row's: lz and the tilt its own.
    float lzs[4], qB[4], gH[NS][4], gB[NS][4], fo[NS][4], dR[NS][4], fm[NS][4], cp[NS][4];
    float gBs[4], fmE[4], cpE[4];                                      // (summed over the samples: the bulk, the EM)
    const float* ac[NS];
    bool emAny = false, cpAny = false;
    if (act) {
        const long tb0 = (long)t * g.B + b0;
        const float4 tK = tau != nullptr ? ld4(tau + (long)Kl * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 z = ld4(a.lz + tb0 * g.M + i0), lbv = ld4(a.lb + tb0 * g.M + i0);
#pragma unroll
        for (int j = 0; j < 4; ++j) { lzs[j] = at(z, j); qB[j] = at(tK, j) - lzs[j]; gBs[j] = fmE[j] = cpE[j] = 0.f; }
#pragma unroll
        for (int m = 0; m < NS; ++m) {
            ac[m] = nullptr;
#pragma unroll
            for (int j = 0; j < 4; ++j) fo[m][j] = fm[m][j] = cp[m][j] = gH[m][j] = gB[m][j] = dR[m][j] = 0.f;
            if (!live[m]) continue;
            const int b = bs[m];
            if (!miss[m] || alone[m]) {
                const long r0 = g.nv_row0[t] + i0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float f = a.flows[(r0 + j) * g.B + b];
                    fo[m][j] = a.logspace ? ex2(f * LOG2E) : f;
                    fm[m][j] = fo[m][j];
                    cp[m][j] = alone[m] ? fo[m][j] : 0.f;
                    gH[m][j] = gB[m][j] = alone[m] ? 0.f : -fo[m][j];
                }
            } else {
                ac[m] = a.acnt + ((long)r[m] * g.M + i0) * a.acs;
                const float* lw = a.logw + ((long)r[m] * g.M + i0) * a.lws;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float f = ac[m][j * a.acs + g.K + 1];
                    const float pr = ex2(lw[j * a.lws + g.K] * LOG2E), aK = f * ac[m][j * a.acs + g.K];
                    fo[m][j] = 0.f;
                    fm[m][j] = f;
                    cp[m][j] = pr > 0.f ? aK / pr : 0.f;
                    gH[m][j] = -f;
                    gB[m][j] = cp[m][j] - f;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dR[m][j] = gB[m][j] * ex2(at(lbv, j) - lzs[j]);        // the bulk's share of the rest slot
                if (grad[m]) gBs[j] += gs[m] * gB[m][j];
                if (em[m]) { fmE[j] += fm[m][j]; if (miss[m]) cpE[j] += cp[m][j]; }
            }
            emAny = emAny || em[m];
            cpAny = cpAny || (em[m] && miss[m]);
        }
    }

    bool gradAny = false, wide = false;
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        gradAny = gradAny || grad[m];
        wide = wide || (a.glp != nullptr && grad[m]) || em[m];
    }
    float* fpT = emAny ? a.fpT + (long)g.nv_tab[t] * g.tab_stride + i0 : nullptr;
    float* fmT = emAny ? a.fmT + (long)g.nv_tab[t] * g.tab_stride + i0 : nullptr;
    const int Cend = wide ? g.C : g.H;                                 // (only d / d tau: the head alone)
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
                        float P[4] = {ex2(th.x + l + qB[0]), ex2(th.y + l + qB[1]), ex2(th.z + l + qB[2]), ex2(th.w + l + qB[3])};
                        pt[u] = P[0] * gBs[0] + P[1] * gBs[1] + P[2] * gBs[2] + P[3] * gBs[3];
                        if (emAny) {
                            const long o = (long)s_ids[k] * g.M;
                            red4(fmT + o, fmE[0] * P[0], fmE[1] * P[1], fmE[2] * P[2], fmE[3] * P[3]);
                            if (cpAny) red4(fpT + o, cpE[0] * P[0], cpE[1] * P[1], cpE[2] * P[2], cpE[3] * P[3]);
                        }
                        continue;
                    }
                    const float4 tc = tau != nullptr ? ld4(tau + (long)min(c, Kl) * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
                    float P[4], fpS[4] = {0.f, 0.f, 0.f, 0.f}, dlS[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int j = 0; j < 4; ++j) P[j] = ex2(at(th, j) + l + at(tc, j) - lzs[j]);
                    float ptu = 0.f;
#pragma unroll
                    for (int m = 0; m < NS; ++m) {
                        if (!live[m]) continue;
                        float dl[4];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float d = miss[m] ? (c < g.K && !alone[m] ? fm[m][j] * ac[m][j * a.acs + c] : 0.f)
                                                    : (c == p[m] ? fo[m][j] : 0.f);
                            dl[j] = P[j] * (c < g.K ? gH[m][j] : gB[m][j]) + d;
                            if (em[m] && miss[m]) fpS[j] += alone[m] ? cp[m][j] * P[j] : (c < g.K ? d : cp[m][j] * P[j]);
                        }
                        if (!grad[m]) continue;
                        ptu += gs[m] * (dl[0] + dl[1] + dl[2] + dl[3]);
                        if (c < Kl) {
                            if (merge) {
#pragma unroll
                                for (int j = 0; j < 4; ++j) dlS[j] += dt[m].sg * dl[j];
                            } else {
                                dt[m].put((long)c * g.M, dl[0], dl[1], dl[2], dl[3]);
                            }
                        } else {
#pragma unroll
                            for (int j = 0; j < 4; ++j) dR[m][j] += dl[j];
                        }
                    }
                    pt[u] = ptu;
                    if (emAny) {
                        const long o = (long)s_ids[k] * g.M;
                        red4(fmT + o, fmE[0] * P[0], fmE[1] * P[1], fmE[2] * P[2], fmE[3] * P[3]);
                        if (cpAny) red4(fpT + o, fpS[0], fpS[1], fpS[2], fpS[3]);
                    }
                    if (merge && c < Kl) dtm.put((long)c * g.M, dlS[0], dlS[1], dlS[2], dlS[3]);
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
                        const float P0 = ex2(th[u].x + l + qB[0]), P1 = ex2(th[u].y + l + qB[1]);
                        const float P2 = ex2(th[u].z + l + qB[2]), P3 = ex2(th[u].w + l + qB[3]);
                        pt[u0 + u] = P0 * gBs[0] + P1 * gBs[1] + P2 * gBs[2] + P3 * gBs[3];
                        if (emAny && k0 + u0 + u < n) {
                            const long o = (long)s_ids[k0 + u0 + u] * g.M;
                            red4(fmT + o, fmE[0] * P0, fmE[1] * P1, fmE[2] * P2, fmE[3] * P3);
                            if (cpAny) red4(fpT + o, cpE[0] * P0, cpE[1] * P1, cpE[2] * P2, cpE[3] * P3);
                        }
                    }
                }
            }
            const float tot = xreduce32(pt, lane);
            if (a.glp != nullptr && gradAny && k0 + lane < n) atomicAdd(a.glp + e0 + cg + lane, tot);
        }
    }
    if (!act) return;
    float dRS[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        if (!live[m]) continue;
        if (em[m] && !miss[m]) red4(fpT + (long)g.ids[e0 + p[m]] * g.M, fo[m][0], fo[m][1], fo[m][2], fo[m][3]);
        if (!grad[m]) {                                                // (F+ at its token: the observed token's flow)
            for (int s = 0; s < g.S; ++s) dt[m].zero((long)s * g.M);
            continue;
        }
        if (!miss[m] && p[m] >= g.H) {                                 // the observed token past the head
            float sv = fo[m][0] + fo[m][1] + fo[m][2] + fo[m][3];
#pragma unroll
            for (int d = 16; d >= 1; d >>= 1) sv += __shfl_xor_sync(0xffffffffu, sv, d);
            if (lane == 0 && a.glp != nullptr) atomicAdd(a.glp + e0 + p[m], gs[m] * sv);
#pragma unroll
            for (int j = 0; j < 4; ++j) dR[m][j] += fo[m][j];
        }
        for (int s = min(g.C, Kl); s < Kl; ++s) dt[m].zero((long)s * g.M);        // (slots past the candidates)
        if (merge) {
#pragma unroll
            for (int j = 0; j < 4; ++j) dRS[j] += dt[m].sg * dR[m][j];
        } else {
            dt[m].put((long)Kl * g.M, dR[m][0], dR[m][1], dR[m][2], dR[m][3]);
        }
    }
    if (merge) dtm.put((long)Kl * g.M, dRS[0], dRS[1], dRS[2], dRS[3]);
}


// The tilt's gradients from dtau (`top_k_leaf_backward`'s, a row of destinations each: d / d tau [T, Bd, S, M]):
//     dA_t[b, i, r] += sum_s D[t, b, s, i] Bt_t[e, r, s],      dBt_t[b, r, s] += sum_i A_t[e, i, r] D[t, b, s, i]
// (e the row of tensors b reads), a CTA a (position, row): its nodes in chunks of 64 staged with their D columns, dA's
// chunk added at once, dBt's sums kept over the chunks -- one pass over D and A (cuBLAS's two batched products read D
// twice, at ~9-22 TF/s on these shapes). Exact fp32 (FMAs). re <= 64 ranks, S <= 65 slots.
constexpr int TR = 64, TS = 65, TC = 64, DST = TC + 1;   // ranks, slots, nodes a chunk, D's staged row stride
struct TiltArgs {
    const float* D; const long long* Ap; const long long* Bp; const long long* dAp; const long long* dBp;
    const int* emap;
    long sab, sam, sar, sbb, sbr, sbs, sdab, sdam, sdar, sdbb, sdbr, sdbs;
    int Bd, M, re, S;
};

constexpr int TILT_SMEM = (TS * TR + TC * TR + TS * DST) * 4;   // BtT, As, Ds (dynamic: past the 48 KB static cap)

__global__ void __launch_bounds__(256) tk_leaf_tilt_grad_kernel(const TiltArgs a) {
    extern __shared__ __align__(16) float tsm[];
    float (*BtT)[TR] = reinterpret_cast<float (*)[TR]>(tsm);                  // Bt_t[e] transposed: [s][r]
    float (*As)[TR] = reinterpret_cast<float (*)[TR]>(tsm + TS * TR);         // the chunk's A: [node][r]
    float (*Ds)[DST] = reinterpret_cast<float (*)[DST]>(tsm + TS * TR + TC * TR);   // the chunk's D: [s][node]
    const int tb = blockIdx.x, t = tb / a.Bd, b = tb % a.Bd;
    const int e = a.emap != nullptr ? a.emap[b] : b;
    const float* A = reinterpret_cast<const float*>(a.Ap[t]) + (long)e * a.sab;
    const float* Bt = reinterpret_cast<const float*>(a.Bp[t]) + (long)e * a.sbb;
    float* dA = reinterpret_cast<float*>(a.dAp[t]) + (long)b * a.sdab;
    float* dB = reinterpret_cast<float*>(a.dBp[t]) + (long)b * a.sdbb;
    const float* D = a.D + ((long)t * a.Bd + b) * a.S * a.M;
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int re = a.re, S = a.S, M = a.M;
    for (int q = tid; q < TS * TR; q += 256) {
        const int s = q / TR, r = q % TR;
        BtT[s][r] = (s < S && r < re) ? Bt[(long)r * a.sbr + (long)s * a.sbs] : 0.f;
    }
    float gB[4][4], gR[4];                                 // dBt: ranks tx*4.., slots ty*4..; the rest slot's (ty 0)
    #pragma unroll
    for (int u = 0; u < 4; ++u) { gR[u] = 0.f; gB[u][0] = gB[u][1] = gB[u][2] = gB[u][3] = 0.f; }
    for (int i0 = 0; i0 < M; i0 += TC) {
        __syncthreads();                                   // (the last chunk read by all; BtT written)
        for (int q = tid; q < TS * TC; q += 256) {
            const int s = q / TC, i = q % TC;
            Ds[s][i] = s < S ? D[(long)s * M + i0 + i] : 0.f;
        }
        for (int q = tid; q < TC * TR; q += 256) {
            const int i = q / TR, r = q % TR;
            As[i][r] = r < re ? A[(long)(i0 + i) * a.sam + (long)r * a.sar] : 0.f;
        }
        __syncthreads();
        // dA's chunk: nodes ty*4 .. +3, ranks tx*4 .. +3
        float g[4][4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) g[u][0] = g[u][1] = g[u][2] = g[u][3] = 0.f;
        for (int s = 0; s < TS; ++s) {
            const float4 bt = *reinterpret_cast<const float4*>(&BtT[s][tx * 4]);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float d = Ds[s][ty * 4 + u];
                g[u][0] += d * bt.x; g[u][1] += d * bt.y; g[u][2] += d * bt.z; g[u][3] += d * bt.w;
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float* o = dA + (long)(i0 + ty * 4 + u) * a.sdam;
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int r = tx * 4 + v;
                if (r < re) o[(long)r * a.sdar] += g[u][v];
            }
        }
        // dBt: over the chunk's nodes, ranks tx*4 .. +3, slots ty*4 .. +3 (and the rest slot: ty 0)
        for (int i = 0; i < TC; ++i) {
            const float4 av = *reinterpret_cast<const float4*>(&As[i][tx * 4]);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float d = Ds[ty * 4 + u][i];
                gB[0][u] += av.x * d; gB[1][u] += av.y * d; gB[2][u] += av.z * d; gB[3][u] += av.w * d;
            }
            if (ty == 0) {
                const float d = Ds[TS - 1][i];
                gR[0] += av.x * d; gR[1] += av.y * d; gR[2] += av.z * d; gR[3] += av.w * d;
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int r = tx * 4 + v;
        if (r >= re) continue;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int s = ty * 4 + u;
            if (s < S - 1) dB[(long)r * a.sdbr + (long)s * a.sdbs] += gB[v][u];
        }
        if (ty == 0) dB[(long)r * a.sdbr + (long)(S - 1) * a.sdbs] += gR[v];
    }
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
    TORCH_CHECK(tab.dim() == 3 && ids.dim() == 3 && lp.sizes() == ids.sizes(), "top_k leaf: shapes");
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
    // (the evidence a sample each, or -- the rows' batch -- a row of tensors each, read through the map)
    g.erow = nullptr;
    if (ids.size(0) != B) {
        TORCH_CHECK(g.emap != nullptr && ids.size(0) == g.Bx, "top_k leaf: the evidence of the samples' or the rows' batch");
        g.erow = g.emap;
    }
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
// (lz [T, B, M], lb [T, B, M], place [T, B], logw [R, M, max(K + 1, 72)] -- slots 0 .. K, the rows 72 wide as the sums'
// kernels read them (the rest not written) --) -- R the rows wrow names (logw empty where none)
std::vector<torch::Tensor> top_k_leaf_forward(torch::Tensor tab, torch::Tensor nv_tab, torch::Tensor tau, torch::Tensor emap,
                                              torch::Tensor ids, torch::Tensor lp, torch::Tensor nv_lvid,
                                              torch::Tensor nv_row0, torch::Tensor tok, torch::Tensor miss,
                                              torch::Tensor wrow, int64_t R, int64_t S, int64_t K,
                                              torch::Tensor node_mars, torch::Tensor ord) {
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
    auto logw = torch::empty({std::max<int64_t>(R, 0), M, std::max<int64_t>(K + 1, LW_STRIDE)}, f32);
    a.tok = tok.data_ptr<int>(); a.miss = miss.data_ptr<uint8_t>(); a.node_mars = node_mars.data_ptr<float>();
    a.lz = lz.data_ptr<float>(); a.lb = lb.data_ptr<float>(); a.place = place.data_ptr<int>();
    a.wrow = nullptr; a.logw = nullptr; a.lws = (int)logw.size(2);
    a.ord = nullptr;
    if (ord.numel() > 0) {
        check(ord, at::kInt, "ord");
        TORCH_CHECK(ord.numel() == B, "top_k leaf: ord [B]");
        a.ord = ord.data_ptr<int>();
    }
    if (R > 0) { check(wrow, at::kInt, "wrow"); a.wrow = wrow.data_ptr<int>(); a.logw = logw.data_ptr<float>(); }
    dim3 grid((unsigned)B, (unsigned)T, (unsigned)((M + NW * 128 - 1) / (NW * 128)));
    tk_leaf_forward_kernel<<<grid, NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {lz, lb, place, logw};
}

// the backward of the same tiles: adds the evidence's gradient into glp [B, nvars, C] (empty: none) and the EM
// statistics into fpT, fmT [ntab, V, M] (empty: none; of the samples `pmask` [B] bool holds, if not empty); returns
// dtau -- [T, B, S, M] a sample each, or where `drow` [B] (int32) is given each sample's times `dsgn` [B] summed into
// the row drow of [T, Bd, S, M]; empty where not `dtau_out`
torch::Tensor top_k_leaf_backward(torch::Tensor tab, torch::Tensor nv_tab, torch::Tensor tau, torch::Tensor emap,
                                  torch::Tensor ids, torch::Tensor lp, torch::Tensor nv_lvid, torch::Tensor nv_row0,
                                  torch::Tensor lz, torch::Tensor lb, torch::Tensor place, torch::Tensor flows,
                                  bool logspace, torch::Tensor wrow, torch::Tensor logw, torch::Tensor acnt,
                                  int64_t S, int64_t K, torch::Tensor glp, torch::Tensor fpT, torch::Tensor fmT,
                                  torch::Tensor pmask, torch::Tensor gsgn, bool dtau_out, torch::Tensor drow,
                                  torch::Tensor dsgn, int64_t Bd, torch::Tensor ord, torch::Tensor rank) {
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
    const bool fold = dtau_out && drow.numel() > 0;
    if (fold) {
        check(drow, at::kInt, "drow"); check(dsgn, at::kFloat, "dsgn");
        TORCH_CHECK(drow.numel() == B && dsgn.numel() == B && Bd > 0, "top_k leaf: drow, dsgn [B]");
    }
    auto dtau = !dtau_out ? torch::empty({0}, tab.options())
              : fold ? torch::zeros({T, Bd, S, M}, tab.options()) : torch::empty({T, B, S, M}, tab.options());
    a.drow = fold ? drow.data_ptr<int>() : nullptr; a.dsgn = fold ? dsgn.data_ptr<float>() : nullptr;
    a.Bd = fold ? (int)Bd : 0;
    a.lz = lz.data_ptr<float>(); a.lb = lb.data_ptr<float>(); a.place = place.data_ptr<int>();
    a.flows = flows.data_ptr<float>(); a.logspace = logspace ? 1 : 0;
    a.wrow = nullptr; a.logw = nullptr; a.lws = 0; a.acnt = nullptr; a.acs = 0;
    if (acnt.numel() > 0) {
        check(wrow, at::kInt, "wrow"); check(logw, at::kFloat, "logw"); check(acnt, at::kFloat, "acnt");
        TORCH_CHECK(acnt.size(0) == logw.size(0) && acnt.size(1) == logw.size(1) && acnt.size(2) >= K + 2 &&
                    logw.size(2) >= K + 1, "top_k leaf: acnt [R, M, >= K + 2], logw [R, M, >= K + 1]");
        a.wrow = wrow.data_ptr<int>(); a.logw = logw.data_ptr<float>(); a.acnt = acnt.data_ptr<float>();
        a.lws = (int)logw.size(2);
        a.acs = (int)acnt.size(2);
    }
    a.glp = glp.numel() > 0 ? glp.data_ptr<float>() : nullptr; a.dtau = dtau_out ? dtau.data_ptr<float>() : nullptr;
    a.fpT = nullptr; a.fmT = nullptr;
    if (fmT.numel() > 0) {
        check(fpT, at::kFloat, "fpT"); check(fmT, at::kFloat, "fmT");
        TORCH_CHECK(fpT.sizes() == tab.sizes() && fmT.sizes() == tab.sizes(), "top_k leaf: the flows' buffers as the tables");
        a.fpT = fpT.data_ptr<float>(); a.fmT = fmT.data_ptr<float>();
    }
    a.ord = nullptr; a.rank = nullptr;
    if (ord.numel() > 0) {                                  // (a row's samples a CTA: the evidence a row each)
        check(ord, at::kInt, "ord"); check(rank, at::kInt, "rank");
        TORCH_CHECK(a.g.erow != nullptr && ord.numel() == B && rank.numel() == B, "top_k leaf: ord, rank [B]");
        a.ord = ord.data_ptr<int>(); a.rank = rank.data_ptr<int>();
    }
    a.gsgn = nullptr;
    if (gsgn.numel() > 0) {
        check(gsgn, at::kFloat, "gsgn");
        TORCH_CHECK(gsgn.numel() == B, "top_k leaf: gsgn [B]");
        a.gsgn = gsgn.data_ptr<float>();
    }
    a.pmask = nullptr;
    if (pmask.numel() > 0) {
        check(pmask, at::kBool, "pmask");
        TORCH_CHECK(pmask.numel() == B, "top_k leaf: pmask [B]");
        a.pmask = pmask.data_ptr<bool>();
    }
    dim3 grid((unsigned)B, (unsigned)T, (unsigned)((M + NW * 128 - 1) / (NW * 128)));
    if (a.ord != nullptr) tk_leaf_backward_kernel<2><<<grid, NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(a);
    else tk_leaf_backward_kernel<1><<<grid, NW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dtau;
}

// pf[spfids[i] + off + v] += fT[v, i]: a block's flows [V, M] (transposed) into the layer's parameter flows
void top_k_leaf_fold(torch::Tensor fT, torch::Tensor spfids, torch::Tensor pf, int64_t off) {
    check(fT, at::kFloat, "fT"); check(spfids, at::kLong, "spfids"); check(pf, at::kFloat, "pf");
    const at::cuda::CUDAGuard guard(fT.device());
    const int V = (int)fT.size(0), M = (int)fT.size(1);
    TORCH_CHECK(spfids.size(0) == M, "top_k leaf: spfids [M]");
    dim3 grid((unsigned)((V + 31) / 32), (unsigned)((M + 31) / 32)), block(32, 8);
    tk_leaf_fold_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        fT.data_ptr<float>(), spfids.data_ptr<long>(), pf.data_ptr<float>(), M, V, off);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The tilt's gradients of the positions (`D` [T, Bd, S, M] contiguous; their factors and destinations by address,
// `ptrs` [4, T] int64: A_t, Bt_t, dA_t, dBt_t; every position's strides `strides` [12]: A's, Bt's, dA's, dBt's), ADDED
// into the destinations; `emap` [Bd] int32 the row of tensors a destination row reads (empty: its own)
void top_k_leaf_tilt_grads(torch::Tensor D, torch::Tensor ptrs, torch::Tensor strides, torch::Tensor emap, int64_t re) {
    const at::cuda::CUDAGuard guard(D.device());
    check(D, at::kFloat, "D"); check(ptrs, at::kLong, "ptrs");
    TORCH_CHECK(D.dim() == 4 && ptrs.size(0) == 4 && ptrs.size(1) == D.size(0) && strides.numel() == 12,
                "top_k leaf: the tilt's gradients' operands");
    const int T = D.size(0), Bd = D.size(1), S = D.size(2), M = D.size(3);
    TORCH_CHECK(re <= TR && S <= TS && M % TC == 0, "top_k leaf: the tilt's ranks <= 64, slots <= 65");
    auto st = strides.to(at::kCPU).contiguous();
    const long* sv = st.data_ptr<long>();
    TiltArgs a;
    a.D = D.data_ptr<float>();
    const long long* pp = reinterpret_cast<const long long*>(ptrs.data_ptr<long>());
    a.Ap = pp; a.Bp = pp + T; a.dAp = pp + 2 * T; a.dBp = pp + 3 * T;
    a.emap = emap.numel() > 0 ? emap.data_ptr<int>() : nullptr;
    a.sab = sv[0]; a.sam = sv[1]; a.sar = sv[2]; a.sbb = sv[3]; a.sbr = sv[4]; a.sbs = sv[5];
    a.sdab = sv[6]; a.sdam = sv[7]; a.sdar = sv[8]; a.sdbb = sv[9]; a.sdbr = sv[10]; a.sdbs = sv[11];
    a.Bd = Bd; a.M = M; a.re = (int)re; a.S = S;
    static bool done[64] = {};
    const int dev = D.get_device();
    if (!done[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(tk_leaf_tilt_grad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TILT_SMEM));
        done[dev] = true;
    }
    tk_leaf_tilt_grad_kernel<<<T * Bd, 256, TILT_SMEM, at::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_leaf_table", &top_k_leaf_table, "log2 theta^T of a soft-evidence leaf node vector");
    m.def("top_k_leaf_forward", &top_k_leaf_forward, "LowRank(top_k) leaves' forward");
    m.def("top_k_leaf_backward", &top_k_leaf_backward, "LowRank(top_k) leaves' backward");
    m.def("top_k_leaf_fold", &top_k_leaf_fold, "a transposed block of flows into the parameter flows");
    m.def("top_k_leaf_tilt_grads", &top_k_leaf_tilt_grads, "the leaves' tilt gradients from dtau, one pass");
}
