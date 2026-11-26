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
//                     the node's flow -- their product the counts); and for the EM statistics (F+, F-) the observed
//                     tokens' F+ (their flows, into the layer's parameter flows), the tile's entries (its candidates,
//                     keyed by their category) and its rows (log2 of the flows over Z)
//   tk_leaf_em        the EM statistics of the entries sorted by category: a CTA a run of them and 128 nodes, a
//                     category's F- = sum f P (F+ the posterior counts: a sum's counts, or cp P past its slots / f P
//                     where no one reads a missing token) over its entries, then a block of categories' sums added into
//                     the layer's parameter flows, a node's line at once -- no atomic a candidate, no V x M buffer
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
        // (no candidate with mass -- all of its evidence impossible: the normalizer 0, so that every log-probability
        // stays -inf and the backward's terms 2^(x - lz) are 0, not -inf - -inf)
        const float lz = lse_log2(lse_merge(lse_merge(own[j], mid[j]), bulk[j]));
        lzs[j] = lz == -INFINITY ? 0.f : lz;
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
    int* ekey; float* emr; int4* emeta; int* eplus; int Vb;      // EM (null: none): a tile's entries [T][B][C] (key
                                                                 // k Vb + v; preset: no entry), rows [T][B][4][M],
                                                                 // (sample, cp's first candidate, count rows) [T][B]
    float* pf; const long* spf;                                  // the observed tokens' F+ into pf[spf[k][i] + v]
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
    static_assert(NS <= 2, "the EM statistics' tiles: up to two count rows");
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
        em[m] = m < ns && a.ekey != nullptr && p[m] != -1             // EM statistics: every token among the candidates
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
    // EM: the tile's entries and its rows (`tk_leaf_em` reads them), where a sample's statistics are taken -- its
    // samples' cp from candidate 0 (a missing token no one reads) or K (a sum reads it: the slots' counts below K, by
    // their rows of the counts), none (C) where every token is observed
    const long gi = (long)t * g.B + blockIdx.x;
    bool emc = false, cpc = false;
    int cpf = g.C, hr[2] = {-1, -1};
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        emc = emc || em[m];
        if (em[m] && miss[m]) {
            cpc = true;
            cpf = alone[m] ? 0 : g.K;
            if (!alone[m]) hr[m] = r[m];
        }
    }
    if (emc && blockIdx.z == 0 && threadIdx.x == 0) {
        a.emeta[gi] = make_int4(b0, cpf, hr[0], hr[1]);
        if (cpc) *a.eplus = 1;
    }

    // a node's terms: dlogit(i, c) = P(i, c) g(i, c) + d(i, c) -- observed: g = -f, d = f at the token; missing: below K
    // g = -f, d = the slot's count, from K on g = a_K / P(rest) - f (the rest's count spread as P), d = 0. EM: F- += fm P,
    // F+ the posterior counts -- observed f at the token (after the walk), a missing token a sum reads its count below K
    // and cp P from K on, one no one reads f P (fm = cp = f, the leaf's flow). P is the row's: lz and the tilt its own.
    float lzs[4], qB[4], gH[NS][4], gB[NS][4], fo[NS][4], dR[NS][4], fm[NS][4], cp[NS][4];
    float gBs[4], fmE[4], cpE[4];                                      // (summed over the samples: the bulk, the EM)
    const float* ac[NS];
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
        }
        if (emc) {                                                     // the rows: log2 (fm, cp) - log2 Z, and with
            float* q = a.emr + gi * 4 * g.M + i0;                      // the bulk's tilt
            float w[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {                              // (log2 in fp64: no error past the fp32 flows')
                const float lf = fmE[j] > 0.f ? (float)log2((double)fmE[j]) : -INFINITY;
                const float lc = cpE[j] > 0.f ? (float)log2((double)cpE[j]) : -INFINITY;
                w[0][j] = lf - lzs[j]; w[1][j] = lf + qB[j]; w[2][j] = lc - lzs[j]; w[3][j] = lc + qB[j];
            }
#pragma unroll
            for (int x = 0; x < 4; ++x)
                if (x < 2 || cpc) st4(q + (long)x * g.M, w[x][0], w[x][1], w[x][2], w[x][3]);
        }
    }

    bool gradAny = false, wide = false;
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        gradAny = gradAny || grad[m];
        wide = wide || (a.glp != nullptr && grad[m]) || em[m];
    }
    int* ekey = emc && blockIdx.z == 0 ? a.ekey + gi * g.C : nullptr;
    const int kbase = g.nv_tab[t] * a.Vb;
    const int Cend = wide ? g.C : g.H;                                 // (only d / d tau: the head alone)
    for (int c0 = 0; c0 < Cend; c0 += CH) {
        const int n = min(CH, Cend - c0);
        __syncthreads();
        for (int k = threadIdx.x; k < n; k += NW * 32) {
            const int id = g.ids[e0 + c0 + k];
            s_ids[k] = id;
            s_lp[k] = g.lp[e0 + c0 + k] * LOG2E;
            if (ekey != nullptr) ekey[c0 + k] = kbase + id;
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
                        pt[u] = ex2(th.x + l + qB[0]) * gBs[0] + ex2(th.y + l + qB[1]) * gBs[1]
                              + ex2(th.z + l + qB[2]) * gBs[2] + ex2(th.w + l + qB[3]) * gBs[3];
                        continue;
                    }
                    const float4 tc = tau != nullptr ? ld4(tau + (long)min(c, Kl) * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
                    float P[4], dlS[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int j = 0; j < 4; ++j) P[j] = ex2(at(th, j) + l + at(tc, j) - lzs[j]);
                    float ptu = 0.f;
#pragma unroll
                    for (int m = 0; m < NS; ++m) {
                        if (!grad[m]) continue;
                        float dl[4];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float d = miss[m] ? (c < g.K ? fm[m][j] * ac[m][j * a.acs + c] : 0.f)
                                                    : (c == p[m] ? fo[m][j] : 0.f);
                            dl[j] = P[j] * (c < g.K ? gH[m][j] : gB[m][j]) + d;
                        }
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
                        pt[u0 + u] = ex2(th[u].x + l + qB[0]) * gBs[0] + ex2(th[u].y + l + qB[1]) * gBs[1]
                                   + ex2(th[u].z + l + qB[2]) * gBs[2] + ex2(th[u].w + l + qB[3]) * gBs[3];
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
        if (em[m] && !miss[m]) {                                       // F+ at its token: the observed token's flow
            const long v = g.ids[e0 + p[m]];
            const long* sp = a.spf + (long)g.nv_tab[t] * g.M + i0;
#pragma unroll
            for (int j = 0; j < 4; ++j) atomicAdd(a.pf + sp[j] + v, fo[m][j]);
        }
        if (!grad[m]) {
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

// The EM statistics of the backward's entries sorted by key (k Vb + v, Vb = V up to EB: a block of EB categories in one
// table; `offs` [nbins + 1] where each key's run starts), a CTA a run of EC of them and EN nodes, the run's blocks one
// after another: the block's log2 theta rows and sums in shared memory, its entries split among the warps (a warp a
// share, a lane 4 nodes; F- += 2^(log2 theta + l + log2 fm - log2 Z + tau), PLUS: F+, cp's terms from the tile's first
// candidate of them and below K a sum's counts of the slots), then the block's sums added into the layer's parameter
// flows (F- at spf[k][i] + V + v, F+ V before it), a node's EB categories at once (a block two runs share: both parts
// added). MEASURED (48 blocks): a CTA every 1024 nodes and a lane's 4 categories one red.v4 a node (a node's flows
// 1 MB apart) made those adds 2/3 of the pass; blocks of 16 (3 CTAs an SM) beat 32 (2) and 8
constexpr int EC = 512;             // entries a run
constexpr int EB = 16;              // categories a block: a node's 64 B of flows
constexpr int EN = 128;             // nodes a CTA (32 lanes x 4)

struct EmArgs {
    const int* skey; const long* sidx; const int* offs; int nbins;   // the entries sorted: keys, ids (tile C + c), runs
    const float* emr; const int4* emeta;                             // the backward's rows; (sample, cp, rows) a tile
    const int* eplus;                                                // any F+ term (a sample's token missing)
    const float* acnt; int acs;                                      // the sums' counts [R][M][acs] (null: none)
    const float* tab; long tab_stride;                               // log2 theta^T [ntab][V][M]
    const float* tau; int Bx; const int* emap;                       // tilts [T][Bx][S][M] (null: none); a sample's row
    const float* lp; const int* erow; int nvars; const int* nv_lvid;  // the evidence, as the tiles read it
    float* pf; const long* spf;                                      // the layer's parameter flows; [ntab][M] node i's
    int B, M, C, S, K, V, Vb;
    bool vec;                                                        // every block's flows 16 B aligned
};

__device__ __forceinline__ float4 add4(const float4& x, const float4& y) {
    return make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
}

template <bool PLUS>
__global__ void __launch_bounds__(NW * 32) tk_leaf_em_kernel(const EmArgs a) {
    __shared__ int s_g[EC], s_c[EC], s_k[EC];
    __shared__ float s_l[EC];
    __shared__ long s_t[EC];
    __shared__ __align__(16) float s_th[EB][EN];                       // the block's log2 theta rows
    __shared__ float s_sum[EB][EN + 1];                                // its sums (padded: read transposed)
    __shared__ int s_plus;
    if (PLUS && *a.eplus == 0) return;                                 // (every token observed: no F+ term)
    const int n = a.offs[a.nbins];                                     // (the entries: those before the unset keys)
    const int s0 = blockIdx.x * EC;
    if (s0 >= n) return;
    const int ne = min(EC, n - s0), Kl = a.S - 1;
    if (threadIdx.x == 0) s_plus = 0;
    __syncthreads();
    for (int x = threadIdx.x; x < ne; x += NW * 32) {                  // the run's entries: tile, candidate, l, tilt
        const long id = a.sidx[s0 + x];
        const int gi = (int)(id / a.C), c = (int)(id - (long)gi * a.C), t = gi / a.B;
        const int4 mt = a.emeta[gi];
        const long e0 = ((long)(a.erow != nullptr ? a.erow[mt.x] : mt.x) * a.nvars + a.nv_lvid[t]) * a.C;
        s_g[x] = gi; s_c[x] = c; s_k[x] = a.skey[s0 + x];
        s_l[x] = a.lp[e0 + c] * LOG2E;
        s_t[x] = a.tau != nullptr && c < Kl
               ? (((long)t * a.Bx + (a.emap != nullptr ? a.emap[mt.x] : mt.x)) * a.S + c) * a.M : -1;
        if (PLUS && (c >= mt.y || (c < a.K && (mt.z >= 0 || mt.w >= 0)))) s_plus = 1;
    }
    __syncthreads();
    if (PLUS && s_plus == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = blockIdx.y * EN, i0 = n0 + 4 * lane;               // the CTA's nodes; a lane's 4
    for (int x = 0; x < ne;) {
        const int b0 = s_k[x] / EB * EB, k = b0 / a.Vb, v0 = b0 - k * a.Vb;
        const int xe = min(a.offs[b0 + EB] - s0, ne);                  // the block's entries in the run
        const float* tb = a.tab + (long)k * a.tab_stride + (long)v0 * a.M + n0;
        for (int q = threadIdx.x; q < EB * EN / 4; q += NW * 32) {     // the block's rows staged, its sums zeroed
            const int r = q / (EN / 4), c4 = 4 * (q % (EN / 4));
            *reinterpret_cast<float4*>(&s_th[r][c4]) = v0 + r < a.V ? ld4(tb + (long)r * a.M + c4)
                                                                    : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            s_sum[r][c4] = s_sum[r][c4 + 1] = s_sum[r][c4 + 2] = s_sum[r][c4 + 3] = 0.f;
        }
        __syncthreads();
        // a warp's share of the block's entries, 8 rows in flight: a category's terms summed until it changes
        const int per = (xe - x + NW - 1) / NW, ya = x + warp * per, yb = min(ya + per, xe);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        int cur = -1;
        for (int y = ya; y < yb; y += 8) {
            float4 w[8];
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                w[z] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (y + z >= yb) continue;
                const long to = s_t[y + z];
                w[z] = ld4(a.emr + ((long)s_g[y + z] * 4 + (PLUS ? 2 : 0) + (to < 0 ? 1 : 0)) * a.M + i0);
                if (to >= 0) w[z] = add4(w[z], ld4(a.tau + to + i0));
            }
#pragma unroll
            for (int z = 0; z < 8; ++z) {
                const int e = y + z;
                if (e >= yb) break;
                const int r = s_k[e] - b0;
                if (r != cur) {
                    if (cur >= 0) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) atomicAdd(&s_sum[cur][4 * lane + j], acc[j]);
                    }
                    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
                    cur = r;
                }
                const float4 th = *reinterpret_cast<const float4*>(&s_th[r][4 * lane]);
                const float l = s_l[e];
                bool ex = true;
                if (PLUS) {                                            // F+: where a token is missing
                    const int c = s_c[e];
                    const int4 mt = a.emeta[s_g[e]];
                    ex = c >= mt.y;                                    // (cp P)
                    if (c < a.K) {                                     // (a sum's count of the slot: its flow times
                        const int hr[2] = {mt.z, mt.w};                // the slot's posterior)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            if (hr[h] < 0) continue;
                            const float* ac = a.acnt + ((long)hr[h] * a.M + i0) * a.acs;
#pragma unroll
                            for (int j = 0; j < 4; ++j) acc[j] += ac[j * a.acs + a.K + 1] * ac[j * a.acs + c];
                        }
                    }
                }
                if (ex) {
                    acc[0] += ex2(th.x + l + w[z].x); acc[1] += ex2(th.y + l + w[z].y);
                    acc[2] += ex2(th.z + l + w[z].z); acc[3] += ex2(th.w + l + w[z].w);
                }
            }
        }
        if (cur >= 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j) atomicAdd(&s_sum[cur][4 * lane + j], acc[j]);
        }
        __syncthreads();
        for (int q = threadIdx.x; q < EN * EB / 4; q += NW * 32) {     // the block's sums into the flows: a node's EB
            const int i = q / (EB / 4), c4 = 4 * (q % (EB / 4));       // categories, 4 a thread (none: no add)
            const float s[4] = {s_sum[c4][i], s_sum[c4 + 1][i], s_sum[c4 + 2][i], s_sum[c4 + 3][i]};
            if (s[0] == 0.f && s[1] == 0.f && s[2] == 0.f && s[3] == 0.f) continue;
            float* o = a.pf + a.spf[(long)k * a.M + n0 + i] + (PLUS ? 0 : a.V) + v0 + c4;
            if (a.vec) {
                red4(o, s[0], s[1], s[2], s[3]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (v0 + c4 + j < a.V) atomicAdd(o + j, s[j]);
            }
        }
        __syncthreads();                                               // (the block's staging reused)
        x = xe;
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
// statistics into the layer's parameter flows pf (empty: none; of the samples `pmask` [B] bool holds, if not empty),
// table k's node i's F+ at pf[spf[k, i] + v] and F- V past it (`spf_aligned`: every spf % 4 == 0); returns dtau --
// [T, B, S, M] a sample each, or where `drow` [B] (int32) is given each sample's times `dsgn` [B] summed into the row
// drow of [T, Bd, S, M]; empty where not `dtau_out`
torch::Tensor top_k_leaf_backward(torch::Tensor tab, torch::Tensor nv_tab, torch::Tensor tau, torch::Tensor emap,
                                  torch::Tensor ids, torch::Tensor lp, torch::Tensor nv_lvid, torch::Tensor nv_row0,
                                  torch::Tensor lz, torch::Tensor lb, torch::Tensor place, torch::Tensor flows,
                                  bool logspace, torch::Tensor wrow, torch::Tensor logw, torch::Tensor acnt,
                                  int64_t S, int64_t K, torch::Tensor glp, torch::Tensor pf, torch::Tensor spf,
                                  bool spf_aligned, torch::Tensor pmask, torch::Tensor gsgn, bool dtau_out,
                                  torch::Tensor drow, torch::Tensor dsgn, int64_t Bd, torch::Tensor ord,
                                  torch::Tensor rank) {
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
    // EM: the tiles' entries (a key a candidate, unset: none), rows and (sample, cp's first candidate, count rows)
    const bool em = pf.numel() > 0;
    const int64_t ntab = tab.size(0), V = tab.size(1), C = ids.size(2), Vb = (V + EB - 1) / EB * EB, nbins = ntab * Vb;
    torch::Tensor ekey, emr, emeta, eplus;
    a.ekey = nullptr; a.emr = nullptr; a.emeta = nullptr; a.eplus = nullptr; a.Vb = (int)Vb; a.pf = nullptr;
    a.spf = nullptr;
    if (em) {
        check(pf, at::kFloat, "pf"); check(spf, at::kLong, "spf");
        TORCH_CHECK(spf.dim() == 2 && spf.size(0) == ntab && spf.size(1) == M, "top_k leaf: spf [ntab, M]");
        TORCH_CHECK(T * B * C < INT_MAX && nbins < INT_MAX, "top_k leaf: the EM entries past int32");
        ekey = torch::full({T * B * C}, nbins, ids.options());
        emr = torch::empty({T * B, 4, M}, tab.options());
        emeta = torch::empty({T * B, 4}, ids.options());
        eplus = torch::zeros({1}, ids.options());
        a.ekey = ekey.data_ptr<int>(); a.emr = emr.data_ptr<float>();
        a.emeta = reinterpret_cast<int4*>(emeta.data_ptr<int>()); a.eplus = eplus.data_ptr<int>();
        a.pf = pf.data_ptr<float>(); a.spf = spf.data_ptr<long>();
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
    const auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)T, (unsigned)((M + NW * 128 - 1) / (NW * 128)));
    if (a.ord != nullptr) tk_leaf_backward_kernel<2><<<grid, NW * 32, 0, stream>>>(a);
    else tk_leaf_backward_kernel<1><<<grid, NW * 32, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!em) return dtau;
    // the entries by category (stable: a category's terms in the tiles' order), where each key's run starts
    auto sorted = torch::sort(ekey, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    auto skey = std::get<0>(sorted), sidx = std::get<1>(sorted);
    auto offs = torch::searchsorted(skey, torch::arange(nbins + 1, ids.options()), /*out_int32=*/true);
    EmArgs e;
    e.skey = skey.data_ptr<int>(); e.sidx = sidx.data_ptr<long>(); e.offs = offs.data_ptr<int>(); e.nbins = (int)nbins;
    e.emr = a.emr; e.emeta = a.emeta; e.eplus = a.eplus; e.acnt = a.acnt; e.acs = a.acs;
    e.tab = a.g.tab; e.tab_stride = a.g.tab_stride;
    e.tau = a.g.tau; e.Bx = a.g.Bx; e.emap = a.g.emap;
    e.lp = a.g.lp; e.erow = a.g.erow; e.nvars = a.g.nvars; e.nv_lvid = a.g.nv_lvid;
    e.pf = a.pf; e.spf = a.spf;
    e.B = (int)B; e.M = M; e.C = (int)C; e.S = (int)S; e.K = (int)K; e.V = (int)V; e.Vb = (int)Vb;
    e.vec = spf_aligned && V % 4 == 0 && reinterpret_cast<uintptr_t>(e.pf) % 16 == 0;
    dim3 egrid((unsigned)((T * B * C + EC - 1) / EC), (unsigned)(M / EN));     // (F-; F+ where a token is missing)
    tk_leaf_em_kernel<false><<<egrid, NW * 32, 0, stream>>>(e);
    tk_leaf_em_kernel<true><<<egrid, NW * 32, 0, stream>>>(e);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dtau;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_leaf_table", &top_k_leaf_table, "log2 theta^T of a soft-evidence leaf node vector");
    m.def("top_k_leaf_forward", &top_k_leaf_forward, "LowRank(top_k) leaves' forward");
    m.def("top_k_leaf_backward", &top_k_leaf_backward, "LowRank(top_k) leaves' backward and EM statistics");
}
