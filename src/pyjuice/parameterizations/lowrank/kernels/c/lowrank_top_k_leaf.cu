// pyjuice: the passes of `LowRank(top_k)` soft-evidence leaves (`lowrank/top_k_leaf_fused.py`).
//
// A leaf node vector of M nodes over V categories; a sample's C candidates c (token ids_b(c), evidence lp_b(c)) and its
// tilt tau_b[i, s] = A_b[i] . B_b[:, s] (the tilt's slot s = min(c, Kl), Kl its top_k):
//
//     logit_b(i, c) = log theta[i, ids_b(c)] + lp_b(c) + tau_b[i, min(c, Kl)],     Z_b(i) = sum_c exp(logit_b(i, c))
//
//   tk_leaf_table     log2 theta^T [V, M] from the layer's parameters: a candidate's M nodes one contiguous read
//   tk_leaf_forward   a CTA a (sample, node vector, up to 1024 nodes) -- or the samples reading one row of evidence
//                     (theirs alike but for the tokens): log2 Z, log2 of the bulk's sum (the candidates past the head
//                     H = max(K, Kl): no slot of their own), the observed token's place and log P(x | i) (the node
//                     values), and where the token is missing and a sum reads it the slots' log-weights (the sum's K
//                     slots one by one, the rest summed)
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
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#include <climits>

namespace {

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float FLOOR = -1e30f;     // a running max's start: finite, so a -inf term gives 2^-inf = 0, not NaN
constexpr int CH = 512;             // candidates staged in shared memory at once
constexpr int NW = 8;               // warps a CTA (128 nodes each)
constexpr int LW_STRIDE = 72;       // the slots' log-weights' row stride (the sums' kernels' SP)

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
    const int* ord; const int* rank;                             // [B] the samples by their evidence's row and a
                                                                 // place's rank within its row (null: a sample a CTA)
    const int* sids; const float* slp;                           // the candidates past the head of each tile [rows of
                                                                 // the evidence][nvars][C - H], by token id; their lp
    int pos_y;                                                   // the grid's positions in y, the samples in x (more
                                                                 // samples than grid.y holds); else the other way
};

constexpr int FB = 8;               // the bulk's candidates in flight a warp
constexpr int HB = 4;               // the head's (each with its tilt's row)

// The forward of a CTA's samples: one (NS = 1), or (the evidence a row each) up to NS of a row -- their tiles the same
// but for the tokens (the evidence and the tilt are the row's) -- walking the candidates ONCE: the normalizers are the
// group's, the node values and the slots' log-weights each sample's from them. The bulk first, by token id, then the
// head, its chunks last to first: the epilogue's head rows (the slots' log-weights) are the last read, and the head's
// first chunk the one staged. MEASURED (48 blocks, a row's two samples): 2.77 ms a sample a CTA -> 1.73 one walk a row
// -> 1.38 with the slots' log-weights stored as whole blocks of rows (1.10 without them: the walk ~5.7 TB/s of table
// rows from L2) -- with the dump's rows (each repeated 3 times: they share every candidate at a position). With the
// rows' evidence made distinct (the repeats' tokens permuted, as the real head's rows are) that kernel 2.29 ms:
// DRAM-bound; a sample's positions side by side (blockIdx.x the position: neighbours share 41% of their candidates)
// 1.79, and the bulk walked by token id (the CTAs side by side read the table rows they share at about the same time)
// 1.56 -- against 1.57 with the repeats (1.38 the samples side by side).
template <int NS>
__global__ void __launch_bounds__(NW * 32, 1) tk_leaf_forward_kernel(const FwArgs a) {
    __shared__ int s_ids[CH];
    __shared__ float s_lp[CH];
    __shared__ int s_place[NS];
    const Tile& g = a.g;
    const int t = a.pos_y ? blockIdx.y : blockIdx.x, qx = a.pos_y ? blockIdx.x : blockIdx.y, lane = threadIdx.x & 31;
    int bs[NS], ns = 1;
    if constexpr (NS == 1) {
        bs[0] = qx;
    } else {                                                           // (`ord` the samples by row, `rank` within it)
        const int q = qx;
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
    const int blk = blockIdx.z * NW + (threadIdx.x >> 5);             // this warp's 128 nodes
    const bool act = blk * 128 < g.M;
    const int i0 = act ? blk * 128 + 4 * lane : 0;
    int tok[NS];
    bool miss[NS];
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        const long tb = (long)t * g.B + bs[m];
        tok[m] = a.tok[tb];
        miss[m] = m >= ns || a.miss[tb] != 0;                          // (no token to find past the group's samples)
    }
    const int b0 = bs[0];
    const long e0 = ((long)(g.erow != nullptr ? g.erow[b0] : b0) * g.nvars + g.nv_lvid[t]) * g.C;
    const float* tab = g.tab + (long)g.nv_tab[t] * g.tab_stride + i0;
    const int Kl = g.S - 1;
    const float* tau = g.tau == nullptr ? nullptr
                     : g.tau + ((long)t * g.Bx + (g.emap != nullptr ? g.emap[b0] : b0)) * g.S * g.M + i0;
    float4 tK = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tau != nullptr && act) tK = ld4(tau + (long)Kl * g.M);
    Lse own[4], mid[4], bulk[4];                                       // c < K; K <= c < H; c >= H
#pragma unroll
    for (int j = 0; j < 4; ++j) { own[j] = {FLOOR, 0.f}; mid[j] = {FLOOR, 0.f}; bulk[j] = {FLOOR, 0.f}; }
    if (threadIdx.x < NS) s_place[threadIdx.x] = INT_MAX;
    __syncthreads();
    for (int c = threadIdx.x; c < g.C; c += NW * 32) {                // each sample's token's first place
        const int id = g.ids[e0 + c];
#pragma unroll
        for (int m = 0; m < NS; ++m)
            if (!miss[m] && id == tok[m]) atomicMin(&s_place[m], c);
    }
    // the bulk (the candidates past the head; the tilt's slot Kl: added after), by token id
    const int nb = g.C - g.H;
    const long eb = ((long)(g.erow != nullptr ? g.erow[b0] : b0) * g.nvars + g.nv_lvid[t]) * nb;
    for (int c0 = 0; c0 < nb; c0 += CH) {
        const int n = min(CH, nb - c0);
        __syncthreads();
        for (int k = threadIdx.x; k < n; k += NW * 32) {
            s_ids[k] = a.sids[eb + c0 + k];
            s_lp[k] = a.slp[eb + c0 + k] * LOG2E;
        }
        __syncthreads();
        if (!act) continue;
        int k = 0;
        for (; k + FB <= n; k += FB) {
            float4 th[FB];
#pragma unroll
            for (int u = 0; u < FB; ++u) th[u] = ld4(tab + (long)s_ids[k + u] * g.M);
#pragma unroll
            for (int u = 0; u < FB; ++u) {
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
    // the head (slots of their own), its chunks last to first: chunk 0 the one staged for the epilogue
    for (int ci = (g.H + CH - 1) / CH - 1; ci >= 0; --ci) {
        const int c0 = ci * CH, kh = min(CH, g.H - c0);
        __syncthreads();
        for (int k = threadIdx.x; k < kh; k += NW * 32) {
            s_ids[k] = g.ids[e0 + c0 + k];
            s_lp[k] = g.lp[e0 + c0 + k] * LOG2E;
        }
        __syncthreads();
        if (!act) continue;
        for (int k = 0; k < kh; k += HB) {
            float4 th[HB], tc[HB];
#pragma unroll
            for (int u = 0; u < HB; ++u) {
                const int c = c0 + min(k + u, kh - 1);
                th[u] = ld4(tab + (long)s_ids[c - c0] * g.M);
                tc[u] = tau != nullptr ? ld4(tau + (long)min(c, Kl) * g.M) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
            for (int u = 0; u < HB; ++u) {
                const int c = c0 + k + u;
                if (k + u >= kh) break;
                const float l = s_lp[k + u];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float x = at(th[u], j) + l + at(tc[u], j);
                    if (c < g.K) lse_add(own[j], x); else lse_add(mid[j], x);
                }
            }
        }
    }
    __syncthreads();                                                   // (the places found)
    const int hs = min(g.H, CH);                                       // (s_ids: the head's first hs candidates)
    // (from here the warps without nodes take part in the barriers only)
    float lzs[4], xr[4];
    if (act) {
        float lbs[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            bulk[j].m += at(tK, j);
            // (no candidate with mass -- all of its evidence impossible: the normalizer 0, so that every
            // log-probability stays -inf and the backward's terms 2^(x - lz) are 0, not -inf - -inf)
            const float lz = lse_log2(lse_merge(lse_merge(own[j], mid[j]), bulk[j]));
            lzs[j] = lz == -INFINITY ? 0.f : lz;
            lbs[j] = lse_log2(bulk[j]);
            xr[j] = (lse_log2(lse_merge(mid[j], bulk[j])) - lzs[j]) * LN2;
        }
        const long r0 = g.nv_row0[t] + i0;
#pragma unroll
        for (int m = 0; m < NS; ++m) {
            if (m >= ns) continue;
            const int b = bs[m];
            const long tb = (long)t * g.B + b;
            st4(a.lz + tb * g.M + i0, lzs[0], lzs[1], lzs[2], lzs[3]);
            st4(a.lb + tb * g.M + i0, lbs[0], lbs[1], lbs[2], lbs[3]);
            const int p = s_place[m];
            if (blockIdx.z == 0 && threadIdx.x == 0) a.place[tb] = miss[m] ? -2 : (p == INT_MAX ? -1 : p);
            if (miss[m]) continue;
            float v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};    // the node values: log P(x | i)
            if (p != INT_MAX) {
                const int id = p < hs ? s_ids[p] : g.ids[e0 + p];      // (the head's first chunk the one staged)
                const float l = p < hs ? s_lp[p] : g.lp[e0 + p] * LOG2E;
                const float4 th = ld4(tab + (long)id * g.M);
                const float4 tc = tau == nullptr ? make_float4(0.f, 0.f, 0.f, 0.f) : p >= Kl ? tK : ld4(tau + (long)p * g.M);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = (at(th, j) + l + at(tc, j) - lzs[j]) * LN2;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) a.node_mars[(r0 + j) * g.B + b] = v[j];
        }
    }
    int wr[NS];                                                        // the missing samples' rows of the weights
    bool anyw = false;
#pragma unroll
    for (int m = 0; m < NS; ++m) {
        wr[m] = m < ns && miss[m] && a.wrow != nullptr ? a.wrow[(long)t * g.B + bs[m]] : -1;
        anyw = anyw || wr[m] >= 0;
    }
    if (!anyw) return;                                                 // (the CTA's every thread alike)
    // The slots' log-weights, as the sum reads them -- the same for every missing sample of the group: made once. The
    // CTA's nodes 32 at a time: their [32][lws] block of rows made in shared memory (a thread a slot and 8 nodes: its
    // two table and tilt reads one 32-B sector each), then stored whole -- the rows' every 16 B, padding included
    // (MEASURED: each lane's own 4 nodes' 16-B stores, rows 288 B apart, cost 0.63 ms of 1.73 at 48 blocks)
    __shared__ float s_lz[NW * 128], s_xr[NW * 128];
    __shared__ __align__(16) float s_buf[32][LW_STRIDE];
    const int n0 = blockIdx.z * NW * 128, nloc = min(NW * 128, g.M - n0);
    if (act) {
        const int q = (threadIdx.x >> 5) * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) { s_lz[q + j] = lzs[j]; s_xr[q + j] = xr[j]; }
    }
    if (a.lws != LW_STRIDE) {                                         // (rows past the buffer: a node a thread)
        __syncthreads();
        for (int n = threadIdx.x; n < nloc; n += NW * 32) {
            for (int s = 0; s <= g.K; ++s) {
                float x = s_xr[n];
                if (s < g.K) {
                    x = -INFINITY;
                    if (s < g.C) {
                        const float* tr = g.tab + (long)g.nv_tab[t] * g.tab_stride + n0 + n;
                        const float l = s < hs ? s_lp[s] : g.lp[e0 + s] * LOG2E;
                        const float tc = tau == nullptr ? 0.f : tau[(long)min(s, Kl) * g.M + n0 + n - i0];
                        x = (tr[(long)(s < hs ? s_ids[s] : g.ids[e0 + s]) * g.M] + l + tc - s_lz[n]) * LN2;
                    }
                }
#pragma unroll
                for (int m = 0; m < NS; ++m)
                    if (wr[m] >= 0) a.logw[((long)wr[m] * g.M + n0 + n) * a.lws + s] = x;
            }
        }
        return;
    }
    for (int k = threadIdx.x; k < 32 * LW_STRIDE; k += NW * 32) (&s_buf[0][0])[k] = 0.f;   // (the padding)
    const float* tab0 = g.tab + (long)g.nv_tab[t] * g.tab_stride + n0;
    const float* tau0 = tau == nullptr ? nullptr : tau - i0 + n0;
    const int nq = (threadIdx.x & 3) * 8;                              // a thread's 8 nodes of the 32
    for (int p0 = 0; p0 < nloc; p0 += 32) {
        __syncthreads();                                               // (the buffer free; s_lz, s_xr made)
        for (int s = threadIdx.x >> 2; s < g.K; s += NW * 8) {
            float v[8];
            if (s < g.C) {
                const int id = s < hs ? s_ids[s] : g.ids[e0 + s];
                const float l = s < hs ? s_lp[s] : g.lp[e0 + s] * LOG2E;
                const float* tr = tab0 + (long)id * g.M + p0 + nq;
                const float4 a0 = ld4(tr), a1 = ld4(tr + 4);
                float4 c0 = make_float4(0.f, 0.f, 0.f, 0.f), c1 = c0;
                if (tau0 != nullptr) {
                    const float* tt = tau0 + (long)min(s, Kl) * g.M + p0 + nq;
                    c0 = ld4(tt); c1 = ld4(tt + 4);
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    v[u] = (at(a0, u) + l + at(c0, u) - s_lz[p0 + nq + u]) * LN2;
                    v[4 + u] = (at(a1, u) + l + at(c1, u) - s_lz[p0 + nq + 4 + u]) * LN2;
                }
            } else {
#pragma unroll
                for (int u = 0; u < 8; ++u) v[u] = -INFINITY;
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) s_buf[nq + u][s] = v[u];
        }
        if (threadIdx.x < 32) s_buf[threadIdx.x][g.K] = s_xr[p0 + threadIdx.x];
        __syncthreads();
        const int nf = 32 * a.lws / 4;                                 // (float4s of the block; lws % 4 = 0)
#pragma unroll
        for (int m = 0; m < NS; ++m) {
            if (wr[m] < 0) continue;
            float4* out = reinterpret_cast<float4*>(a.logw + ((long)wr[m] * g.M + n0 + p0) * a.lws);
            for (int f = threadIdx.x; f < nf; f += NW * 32)              // (the block as it lies: lws = LW_STRIDE)
                out[f] = reinterpret_cast<const float4*>(&s_buf[0][0])[f];
        }
    }
}

struct BwArgs {
    Tile g;
    const float* lz; const float* lb; const int* place;          // the forward's
    const float* flows; int logspace;                            // node flows [rows][B]
    const int* wrow; const float* logw; int lws; const float* acnt; int acs;   // a missing token a sum reads: its row of the slots'
                                                                 // log-weights [R][M][K + 1] and of the sum's posterior
                                                                 // [R][M][K + 2]: the slots', then the node's flow
    float* glp;                                                  // [B][nvars][C] (added into; null: not wanted)
    float* dtau;                                                 // [T][B][S][M] written, or (drow) [T][Bd][S][M]: a
                                                                 // sample's times dsgn into row drow; null: none
    const int* drow; const float* dsgn; int Bd;
    const bool* owned;                                           // [Bd] (drow) the rows a CTA holds every sample of:
                                                                 // stored whole, the others added into (null: all)
    int* ekey; float* emr; int4* emeta; int* eplus; int Vb;      // EM (null: none): a tile's entries [T][B][C] (key
                                                                 // k Vb + v; preset: no entry), rows [T][B][4][M],
                                                                 // (sample, cp's first candidate, count rows) [T][B]
    float* pf; const long* spf;                                  // the observed tokens' F+ into pf[spf[k][i] + v]
    const bool* pmask;                                           // [B] the samples the EM statistics take (null: all)
    const float* gsgn;                                           // [B] the evidence gradient's sign a sample (null: 1)
    const int* ord; const int* rank;                             // [B] the samples by their evidence's row and a
                                                                 // place's rank within its row (null: a sample a CTA)
    int pos_y;                                                   // (as the forward's)
    int headless;                                                // the head's walk another kernel's (no dtau, no head
                                                                 // evidence gradient here: `tk_leaf_head_grad_kernel`)
    float* nterm;                                                // (headless) a sample's node terms [T][B][5][M] for
                                                                 // it: g at the head, g past it, f, the counts' f, dR
};

// dtau's writes of a thread's 4 nodes, times the sample's sign (1 where nothing folds): stored into a row the CTA holds
// whole (OWN: the sample's own, or the row of tensors it read where no other CTA's sample folds into it), added into
// one CTA's share (ADD: the row zeroed before; zeros then not written), or left to the CTA's one writer of its samples'
// row (MERGED); none where no tilt gradient is wanted
enum : int { DT_OWN = 0, DT_ADD = 1, DT_MERGED = 2 };
struct DtauOut {
    float* p; float sg; int mode;
    __device__ __forceinline__ void put(long off, float x0, float x1, float x2, float x3) const {
        if (p == nullptr || mode == DT_MERGED) return;
        if (mode == DT_ADD) red4(p + off, sg * x0, sg * x1, sg * x2, sg * x3);
        else st4(p + off, sg * x0, sg * x1, sg * x2, sg * x3);
    }
    __device__ __forceinline__ void zero(long off) const {
        if (p != nullptr && mode == DT_OWN) st4(p + off, 0.f, 0.f, 0.f, 0.f);
    }
};

// The backward of a CTA's samples: one (NS = 1), or (the evidence a row each) up to NS of a row, walking its candidates
// once -- the bulk's terms and the EM statistics summed over the samples before they are added, the head's a sample each.
// The CTAs a sample's (a group's) positions side by side, blockIdx.x the position: the positions of a row share many of
// their candidates (their table rows then read from the cache) -- MEASURED (48 blocks, the rows' evidence made distinct:
// the dump's repeats' tokens permuted) 2.77 -> 2.29 ms against the samples side by side; with the dump's repeated rows
// (they share every candidate at a position) 2.32 -> 2.30
template <int NS>
__global__ void __launch_bounds__(NW * 32) tk_leaf_backward_kernel(const BwArgs a) {
    static_assert(NS <= 2, "the EM statistics' tiles: up to two count rows");
    __shared__ int s_ids[CH];
    __shared__ float s_lp[CH];
    const Tile& g = a.g;
    const int t = a.pos_y ? blockIdx.y : blockIdx.x, qx = a.pos_y ? blockIdx.x : blockIdx.y, lane = threadIdx.x & 31;
    int bs[NS], ns = 1;
    if constexpr (NS == 1) {
        bs[0] = qx;
    } else {                                                           // (`ord` the samples by row, `rank` within it)
        const int q = qx;
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
    // (one write of the tilt's gradient for the samples where they fold into one row -- stored where no other CTA's
    // sample folds into it --; else a sample each)
    const bool merge = NS > 1 && fold && ns > 1 && a.drow[bs[0]] == a.drow[bs[NS - 1]];
    const auto owns = [&](int b) { return !fold || (a.owned != nullptr && a.owned[a.drow[b]]); };
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
                        fold ? a.dsgn[b] : 1.f, merge ? DT_MERGED : owns(b) ? DT_OWN : DT_ADD};
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
    const DtauOut dtm{dt[0].p, 1.f, merge && owns(bs[0]) ? DT_OWN : DT_ADD};
    if (!any) {
        if (merge && act)                                              // (the merged row's zeros: its one writer's)
            for (int s = 0; s < g.S; ++s) dtm.zero((long)s * g.M);
        return;
    }
    const int b0 = bs[0];
    const long e0 = ((long)(g.erow != nullptr ? g.erow[b0] : b0) * g.nvars + g.nv_lvid[t]) * g.C;
    const float* tab = g.tab + (long)g.nv_tab[t] * g.tab_stride + i0;
    const float* tau = g.tau == nullptr ? nullptr
                     : g.tau + ((long)t * g.Bx + (g.emap != nullptr ? g.emap[b0] : b0)) * g.S * g.M + i0;
    // EM: the tile's entries and its rows (`tk_leaf_em` reads them), where a sample's statistics are taken -- its
    // samples' cp from candidate 0 (a missing token no one reads) or K (a sum reads it: the slots' counts below K, by
    // their rows of the counts), none (C) where every token is observed
    const long gi = (long)t * g.B + qx;
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
                    if (k >= n || (a.headless && c < g.H)) continue;       // (headless: the head's not here)
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
        if (a.nterm != nullptr) {                                      // (headless: the head's walk reads them)
            float* q = a.nterm + (((long)t * g.B + bs[m]) * 5) * g.M + i0;
            st4(q, gH[m][0], gH[m][1], gH[m][2], gH[m][3]);
            st4(q + g.M, gB[m][0], gB[m][1], gB[m][2], gB[m][3]);
            st4(q + 2 * g.M, fo[m][0], fo[m][1], fo[m][2], fo[m][3]);
            st4(q + 3 * g.M, fm[m][0], fm[m][1], fm[m][2], fm[m][3]);
            st4(q + 4 * g.M, dR[m][0], dR[m][1], dR[m][2], dR[m][3]);
        }
        for (int s = min(g.C, Kl); s < Kl; ++s) dt[m].zero((long)s * g.M);        // (slots past the candidates)
        if (merge) {
#pragma unroll
            for (int j = 0; j < 4; ++j) dRS[j] += dt[m].sg * dR[m][j];
        } else {
            dt[m].put((long)Kl * g.M, dR[m][0], dR[m][1], dR[m][2], dR[m][3]);
        }
    }
    if (merge) {
        dtm.put((long)Kl * g.M, dRS[0], dRS[1], dRS[2], dRS[3]);
        for (int s = min(g.C, Kl); s < Kl; ++s) dtm.zero((long)s * g.M);         // (slots past the candidates)
    }
}

// ------------------------------------------------------------------------- the head's walk in the tilt's gradients
//
// With the gradients folded into the rows of tensors and the evidence a row each, the samples folding into a row d all
// read its tile (its evidence and tilt): their d / d tau [S, M] is the head's walk over that tile -- the head's
// candidates' slots one each, the rest's slot the bulk's share (from the forward's lb) and the head's past Kl. A CTA a
// (position t, row d) makes it 64 nodes at a time in shared memory and takes the tilt's gradients from it there, as
// `tk_tilt_grad_kernel` does from a dtau read back -- dA_t[d] = D Bt_t[d]^T (written a chunk at a time), dBt_t[d] =
// A_t[d]^T D (summed over the chunks) -- with the head's candidates' evidence gradient beside (summed over the nodes:
// one add a candidate). The backward kernel then walks the bulk alone (`headless`): no dtau [T, Bd, S, M] made, written
// and read back.
constexpr int HG_TR = 64, HG_TS = 65;      // the tilt's ranks, slots (at most)
constexpr int HG_C = 64;                   // nodes a chunk
constexpr int HG_T = 256;                  // threads a CTA
constexpr int HG_BTS = HG_TR + 4;          // Bt^T's staged rows (16-B aligned)
constexpr int HG_NSM = 4;                  // samples whose node terms a chunk holds at once
constexpr int HG_MAXH = 80;                // the head's candidates (H = min(C, max(K, Kl)) <= 71: K + 2 <= 72)
constexpr int HG_MAXT = 64;                // positions a launch
constexpr int HG_CPT = (HG_MAXH + 15) / 16;   // a thread's candidates (every 16th)
constexpr int HG_SMEM = (HG_TS * HG_C + HG_C * HG_TR + 6 * HG_NSM * HG_C + 2 * HG_C) * 4
                      + HG_MAXH * 4 * 3 + 6 * HG_NSM * 4;

struct HgArgs {
    BwArgs b;                                                    // the backward's (its tiles, flows, counts, glp, ...)
    const int* rowstart;                                         // [Bd + 1] each row's first place in `ord`
    const float* A[HG_MAXT]; const float* Bt[HG_MAXT]; float* dA[HG_MAXT]; float* dB[HG_MAXT];   // a position's tensors
    long sab, sam, sar, sbb, sbr, sbs, sdab, sdam, sdar, sdbb, sdbr, sdbs;
    int t0, re;                                                  // the launch's first position; the ranks
    int store[HG_MAXT];                                          // a position's destinations stored, else added into
};

__device__ __forceinline__ int hg_swz(int q, int s) { return q ^ ((s >> 2) & 7); }   // node quad q's place in row s

__device__ __forceinline__ void hg_cp16(float* dst, const float* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void hg_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void hg_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

__global__ void __launch_bounds__(HG_T, 2) tk_leaf_head_grad_kernel(const HgArgs h) {
    extern __shared__ __align__(16) float sm[];
    const BwArgs& a = h.b;
    const Tile& g = a.g;
    float (*Ds)[HG_C] = reinterpret_cast<float (*)[HG_C]>(sm);                                      // [s][node] (swizzled)
    float (*As)[HG_C][HG_TR] = reinterpret_cast<float (*)[HG_C][HG_TR]>(sm + HG_TS * HG_C);        // [1][node][r]
    float* s_nt = sm + HG_TS * HG_C + HG_C * HG_TR;                         // node terms [6][NSM][C]: gH gB fo fm sg*dR
    float* s_lz = s_nt + 6 * HG_NSM * HG_C;                                 // [C] log2 Z; [C] the rest slot's sum
    float* s_rest = s_lz + HG_C;
    float* s_glp = s_rest + HG_C;                                           // [MAXH] the head's evidence gradient
    float* s_l = s_glp + HG_MAXH;                                           // [MAXH] the head's lp (log2)
    int* s_id = reinterpret_cast<int*>(s_l + HG_MAXH);                      // [MAXH] its ids
    int* s_si = s_id + HG_MAXH;                                             // [4][NSM] a group's samples: place, row
    float* s_sf = reinterpret_cast<float*>(s_si + 4 * HG_NSM);              // of the counts, grad; [2][NSM] signs
    const int tl = blockIdx.x, d = blockIdx.y, t = h.t0 + tl;               // (a position of the launch; a row)
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31;
    const int M = g.M, S = g.S, Kl = S - 1, H = g.H, K = g.K, re = h.re;
    const bool store = h.store[tl] != 0;
    float* dA = h.dA[tl] + (long)d * h.sdab;
    float* dB = h.dB[tl] + (long)d * h.sdbb;
    const int q0 = h.rowstart[d], q1 = h.rowstart[d + 1];                  // the row's samples: ord[q0 .. q1)
    if (q0 >= q1) {                                                         // (no sample reads the row: no gradient)
        if (store) {
            for (long k = tid; k < (long)M * re; k += HG_T) dA[(k / re) * h.sdam + (k % re) * h.sdar] = 0.f;
            for (int k = tid; k < re * S; k += HG_T) dB[(long)(k / S) * h.sdbr + (long)(k % S) * h.sdbs] = 0.f;
        }
        return;
    }
    const int bL = a.ord[q0];                                               // (the forward's lz, lb: the row's)
    const long e0 = ((long)d * g.nvars + g.nv_lvid[t]) * g.C;
    const float* tab = g.tab + (long)g.nv_tab[t] * g.tab_stride;
    const float* tau = g.tau + ((long)t * g.Bx + d) * S * M;
    const float* A = h.A[tl] + (long)d * h.sab;
    const float* Bt = h.Bt[tl] + (long)d * h.sbb;
    const int nq_t = tid & 15, cl = tid >> 4;                              // the items: a node quad, every 16th candidate

    auto issueA = [&](int st, int i0) {
        for (int k = tid; k < HG_C * (re / 4); k += HG_T) {
            const int n = k / (re / 4), c = k % (re / 4);
            hg_cp16(&As[st][n][4 * c], A + (long)(i0 + n) * h.sam + 4 * c);
        }
    };
    // the zeros past re, the head's candidates, the gradient's sums (Bt read through L1)
    for (int k = tid; k < HG_C * (HG_TR - re); k += HG_T) {
        const int n = k / (HG_TR - re), r = re + k % (HG_TR - re);
        As[0][n][r] = 0.f;
    }
    for (int k = tid; k < H; k += HG_T) {
        s_id[k] = g.ids[e0 + k];
        s_l[k] = g.lp[e0 + k] * LOG2E;
        s_glp[k] = 0.f;
    }
    const int nA = (w >> 1) * 4 + (l >> 3), rA = (w & 1) * 8 + (l & 7);    // dA: node quad, rank quad
    const int rB = (w >> 1) * 4 + (l >> 3), sB = (w & 1) * 8 + (l & 7);    // dBt: rank quad, slot quad
    const int r64 = tid & 63, p64 = tid >> 6;                               // the 65th slot: rank, quarter of nodes
    const bool liveA = 4 * rA < re;
    float gB[4][4], g64 = 0.f;
#pragma unroll
    for (int x = 0; x < 4; ++x) gB[x][0] = gB[x][1] = gB[x][2] = gB[x][3] = 0.f;
    const bool gl = a.glp != nullptr;

    const int nck = M / HG_C;
    for (int k = 0; k < nck; ++k) {
        const int st = 0, i0 = k * HG_C;
        __syncthreads();                                                    // (the previous chunk's Ds, As read)
        issueA(0, i0);
        hg_commit();
        for (int x = tid; x < HG_TS * HG_C; x += HG_T) (&Ds[0][0])[x] = 0.f;
        if (tid < HG_C) {
            s_lz[tid] = a.lz[((long)t * g.B + bL) * M + i0 + tid];
            s_rest[tid] = 0.f;
        }
        // the items' sums: a thread's candidates cl, cl + 16, .. (< H) at its 4 nodes
        float Dc[HG_CPT][4], gc[HG_CPT];
#pragma unroll
        for (int u = 0; u < HG_CPT; ++u) { Dc[u][0] = Dc[u][1] = Dc[u][2] = Dc[u][3] = 0.f; gc[u] = 0.f; }
        float rst[4] = {0.f, 0.f, 0.f, 0.f};                               // (the head past Kl: the rest's slot)
        for (int m0 = q0; m0 < q1; m0 += HG_NSM) {                          // the row's samples, NSM at a time
            const int nm = min(HG_NSM, q1 - m0);
            __syncthreads();                                                // (the previous group's node terms read)
            {                                                               // the group's node terms: a thread a
                const int n = tid & (HG_C - 1), mm = tid / HG_C;            // (node, sample)
                float tgH = 0.f, tgB = 0.f, tfo = 0.f, tfm = 0.f, tdr = 0.f;
                if (mm < nm) {
                    const int b = a.ord[m0 + mm];
                    const long tb = (long)t * g.B + b;
                    const int p = a.place[tb];
                    const bool miss = p == -2;
                    const int r = miss && a.wrow != nullptr ? a.wrow[tb] : -1;
                    const bool grad = p != -1 && !(miss && r < 0);
                    if (n == 0) {
                        s_si[0 * HG_NSM + mm] = p; s_si[1 * HG_NSM + mm] = r; s_si[2 * HG_NSM + mm] = grad ? 1 : 0;
                        s_sf[0 * HG_NSM + mm] = a.dsgn[b]; s_sf[1 * HG_NSM + mm] = a.gsgn != nullptr ? a.gsgn[b] : 1.f;
                    }
                    if (grad && a.nterm != nullptr) {                      // (the backward kernel's: coalesced)
                        const float* q = a.nterm + (tb * 5) * M + i0 + n;
                        tgH = q[0]; tgB = q[M]; tfo = q[2 * M]; tfm = q[3 * M];
                        tdr = q[4 * M] * a.dsgn[b];
                    } else if (grad) {
                        const float lzn = a.lz[tb * M + i0 + n], lbn = a.lb[tb * M + i0 + n];
                        if (!miss) {
                            const float f = a.flows[(long)(g.nv_row0[t] + i0 + n) * g.B + b];
                            tfo = a.logspace ? ex2(f * LOG2E) : f;
                            tgH = tgB = -tfo;
                        } else {
                            const float* ac = a.acnt + ((long)r * M + i0 + n) * a.acs;
                            const float f = ac[K + 1];
                            const float pr = ex2(a.logw[((long)r * M + i0 + n) * a.lws + K] * LOG2E), aK = f * ac[K];
                            tfm = f;
                            tgH = -f;
                            tgB = (pr > 0.f ? aK / pr : 0.f) - f;
                        }
                        tdr = tgB * ex2(lbn - lzn) + (!miss && p >= H ? tfo : 0.f);   // the bulk's share; the token past
                        tdr *= a.dsgn[b];                                             // the head
                    }
                }
                s_nt[(0 * HG_NSM + mm) * HG_C + n] = tgH; s_nt[(1 * HG_NSM + mm) * HG_C + n] = tgB;
                s_nt[(2 * HG_NSM + mm) * HG_C + n] = tfo; s_nt[(3 * HG_NSM + mm) * HG_C + n] = tfm;
                s_nt[(4 * HG_NSM + mm) * HG_C + n] = tdr;
            }
            __syncthreads();
            if (tid < HG_C) {                                               // the rest slot: the group's dR, signed
                float v = 0.f;
                for (int mm = 0; mm < nm; ++mm) v += s_nt[(4 * HG_NSM + mm) * HG_C + tid];
                s_rest[tid] += v;
            }
            // the items: P at the thread's candidates and nodes, each sample's dlogit
            const float4 z = *reinterpret_cast<const float4*>(&s_lz[4 * nq_t]);
#pragma unroll
            for (int u = 0; u < HG_CPT; ++u) {
                const int c = cl + 16 * u;
                if (c >= H) break;
                const float4 th = ld4(tab + (long)s_id[c] * M + i0 + 4 * nq_t);
                const float4 tc = ld4(tau + (long)min(c, Kl) * M + i0 + 4 * nq_t);
                const float lc = s_l[c];
                float P[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) P[j] = ex2(at(th, j) + lc + at(tc, j) - at(z, j));
                for (int mm = 0; mm < nm; ++mm) {
                    if (s_si[2 * HG_NSM + mm] == 0) continue;               // (no gradient)
                    const int p = s_si[0 * HG_NSM + mm], r = s_si[1 * HG_NSM + mm];
                    const bool miss = p == -2;
                    const float sg = s_sf[0 * HG_NSM + mm], gs = s_sf[1 * HG_NSM + mm];
                    float dl[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int n = 4 * nq_t + j;
                        const float gH = s_nt[(0 * HG_NSM + mm) * HG_C + n], gBv = s_nt[(1 * HG_NSM + mm) * HG_C + n];
                        float dd = 0.f;
                        if (miss) {
                            if (c < K) dd = s_nt[(3 * HG_NSM + mm) * HG_C + n] * a.acnt[((long)r * M + i0 + n) * a.acs + c];
                        } else if (c == p) {
                            dd = s_nt[(2 * HG_NSM + mm) * HG_C + n];
                        }
                        dl[j] = P[j] * (c < K ? gH : gBv) + dd;
                    }
                    gc[u] += gs * (dl[0] + dl[1] + dl[2] + dl[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (c < Kl) Dc[u][j] += sg * dl[j];
                        else rst[j] += sg * dl[j];
                    }
                }
            }
        }
        // the chunk's D: the head's slots, the rest's (dR's and the head past Kl)
#pragma unroll
        for (int u = 0; u < HG_CPT; ++u) {
            const int c = cl + 16 * u;
            if (c >= H || c >= Kl) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) Ds[c][hg_swz(nq_t, c) * 4 + j] = Dc[u][j];
        }
        if (H > Kl) {
#pragma unroll
            for (int j = 0; j < 4; ++j) atomicAdd(&s_rest[4 * nq_t + j], rst[j]);
        }
        if (gl) {                                                           // the head's evidence gradient: over the
#pragma unroll                                                              // node quads (16 lanes), then the chunks
            for (int u = 0; u < HG_CPT; ++u) {
                float v = gc[u];
#pragma unroll
                for (int o = 8; o >= 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                const int c = cl + 16 * u;
                if (nq_t == 0 && c < H) s_glp[c] += v;
            }
        }
        __syncthreads();
        if (tid < HG_C) Ds[Kl][hg_swz(tid >> 2, Kl) * 4 + (tid & 3)] = s_rest[tid];
        hg_wait<0>();
        __syncthreads();                                                    // (Ds made; As landed)

        // dA's chunk: nodes 4 nA .. + 3, ranks 4 rA .. + 3 (as `tk_tilt_grad_kernel`)
        {
            float* o = dA + (long)(i0 + 4 * nA) * h.sdam + (long)(4 * rA) * h.sdar;
            float gg[4][4];
            if (store || !liveA) {
#pragma unroll
                for (int u = 0; u < 4; ++u) gg[u][0] = gg[u][1] = gg[u][2] = gg[u][3] = 0.f;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float4 v = *reinterpret_cast<const float4*>(o + (long)u * h.sdam);
                    gg[u][0] = v.x; gg[u][1] = v.y; gg[u][2] = v.z; gg[u][3] = v.w;
                }
            }
#pragma unroll 5
            for (int s = 0; s < S; ++s) {
                const float4 dv4 = *reinterpret_cast<const float4*>(&Ds[s][hg_swz(nA, s) * 4]);
                float4 bt = make_float4(0.f, 0.f, 0.f, 0.f);                // (Bt_t[d][4 rA .. + 3][s])
                if (liveA) bt = make_float4(__ldg(Bt + (long)(4 * rA) * h.sbr + (long)s * h.sbs),
                                            __ldg(Bt + (long)(4 * rA + 1) * h.sbr + (long)s * h.sbs),
                                            __ldg(Bt + (long)(4 * rA + 2) * h.sbr + (long)s * h.sbs),
                                            __ldg(Bt + (long)(4 * rA + 3) * h.sbr + (long)s * h.sbs));
                const float dv[4] = {dv4.x, dv4.y, dv4.z, dv4.w}, bv[4] = {bt.x, bt.y, bt.z, bt.w};
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int v = 0; v < 4; ++v) gg[u][v] = fmaf(dv[u], bv[v], gg[u][v]);
            }
            if (liveA) {
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    *reinterpret_cast<float4*>(o + (long)u * h.sdam) = make_float4(gg[u][0], gg[u][1], gg[u][2], gg[u][3]);
            }
        }
        // dBt: ranks 4 rB .. + 3 x slots 4 sB .. + 3 over the chunk's nodes, four at a time
#pragma unroll 2
        for (int iq = 0; iq < HG_C / 4; ++iq) {
            float av[4][4], dv[4][4];                                       // [node][rank], [slot][node]
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float4 x = *reinterpret_cast<const float4*>(&As[st][4 * iq + v][4 * rB]);
                av[v][0] = x.x; av[v][1] = x.y; av[v][2] = x.z; av[v][3] = x.w;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int s = 4 * sB + u;
                const float4 x = *reinterpret_cast<const float4*>(&Ds[s][hg_swz(iq, s) * 4]);
                dv[u][0] = x.x; dv[u][1] = x.y; dv[u][2] = x.z; dv[u][3] = x.w;
            }
#pragma unroll
            for (int v = 0; v < 4; ++v)
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int u = 0; u < 4; ++u) gB[x][u] = fmaf(av[v][x], dv[u][v], gB[x][u]);
        }
        if (S == HG_TS) {                                                   // the 65th slot: rank r64, a node quarter
#pragma unroll 4
            for (int j = 0; j < HG_C / 4; ++j) {
                const int n = (HG_C / 4) * p64 + j;
                g64 = fmaf(As[st][n][r64], Ds[HG_TS - 1][hg_swz(n >> 2, HG_TS - 1) * 4 + (n & 3)], g64);
            }
        }
    }
    __syncthreads();
    // dBt's tile; the 65th slot's quarters summed through a free buffer; the head's evidence gradient
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int r = 4 * rB + x;
        if (r >= re) continue;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int s = 4 * sB + u;
            if (s >= S || s >= HG_TS - 1) continue;
            float* q = dB + (long)r * h.sdbr + (long)s * h.sdbs;
            *q = store ? gB[x][u] : *q + gB[x][u];
        }
    }
    if (S == HG_TS) {
        float* red = &As[0][0][0];
        red[p64 * HG_TR + r64] = g64;
        __syncthreads();
        if (tid < re) {
            const float v = ((red[tid] + red[HG_TR + tid]) + red[2 * HG_TR + tid]) + red[3 * HG_TR + tid];
            float* q = dB + (long)tid * h.sdbr + (long)(HG_TS - 1) * h.sdbs;
            *q = store ? v : *q + v;
        }
    }
    if (gl)
        for (int c = tid; c < H; c += HG_T) atomicAdd(a.glp + e0 + c, s_glp[c]);
}

// The folded dtau [T][Bd][S][M]'s rows (a CTA a (row d, t)): `owned[d]` whether one CTA of the backward holds every
// sample folding into d -- the places of `ord` in groups of NS of an evidence row by their `rank`, else (null) a
// sample a CTA -- and stores it whole; the others' (no sample, or samples in more than one CTA: added into) zeroed
template <int NS>
__global__ void __launch_bounds__(256) tk_dtau_rows_kernel(const int* drow, const int* ord, const int* rank, int B,
                                                           int Bd, long rowsz, bool* owned, float* dtau) {
    __shared__ int s_lo[8], s_hi[8];
    const int d = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int lo = INT_MAX, hi = -1;
    for (int q = threadIdx.x; q < B; q += blockDim.x) {
        const int b = ord != nullptr ? ord[q] : q;
        if (drow[b] != d) continue;
        const int grp = ord != nullptr ? q - rank[q] % NS : q;
        lo = min(lo, grp); hi = max(hi, grp);
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o)); hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    if (lane == 0) { s_lo[w] = lo; s_hi[w] = hi; }
    __syncthreads();
    lo = s_lo[0]; hi = s_hi[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) { lo = min(lo, s_lo[i]); hi = max(hi, s_hi[i]); }
    const bool own = lo == hi;                                         // (no sample: INT_MAX against -1)
    if (t == 0 && threadIdx.x == 0) owned[d] = own;
    if (own) return;
    float4* row = reinterpret_cast<float4*>(dtau + ((long)t * Bd + d) * rowsz);
    for (long i = threadIdx.x; i < rowsz / 4; i += blockDim.x) row[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}

// The EM statistics of the backward's entries sorted by key (k Vb + v, Vb = V up to EB: a block of EB categories in one
// table; `offs` [nbins + 1] where each key's run starts), a CTA a run of EC of them and EN nodes, the run's blocks one
// after another: the block's log2 theta rows and sums in shared memory, its entries split among the warps (a warp a
// share, a lane 4 nodes; F- += 2^(log2 theta + l + log2 fm - log2 Z + tau), PLUS: F+, cp's terms from the tile's first
// candidate of them and below K a sum's counts of the slots), then the block's sums added into the layer's parameter
// flows (F- at spf[k][i] + V + v, F+ V before it), a node's EB categories at once (a block two runs share: both parts
// added). MEASURED (48 blocks): a CTA every 1024 nodes and a lane's 4 categories one red.v4 a node (a node's flows
// 1 MB apart) made those adds 2/3 of the pass; blocks of 16 (3 CTAs an SM) beat 32 (2) and 8. F- staged from the
// entries' records (`tk_leaf_em_rec_kernel`) and a block's rows loaded a block ahead: 2.00 -> 1.74 ms (DISTINCT=1 rows).
// REFUTED (same shapes): a warp a run with its sums in registers and a red.v4 a node at each block of 4 categories
// (2.13-3.04 ms: 16-B reds into rows 1 MB apart), and a CTA a whole block of categories (2.87: frequent blocks' tails)
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

// The records of the entries sorted by key, the F- kernel's staging (one 16-B read an entry, not its chain of four):
// (key, the row of `emr` its term takes -- the tile's first (the head's: its tilt added) or second (the bulk's) --,
// l in log2 units, its tilt's row of tau [T][Bx][S] (-1: the bulk's, in the second row already))
__global__ void __launch_bounds__(256) tk_leaf_em_rec_kernel(const EmArgs a, int4* __restrict__ rec) {
    const int k = blockIdx.x * 256 + threadIdx.x;
    if (k >= a.offs[a.nbins]) return;
    const long id = a.sidx[k];
    const int gi = (int)(id / a.C), c = (int)(id - (long)gi * a.C), t = gi / a.B, Kl = a.S - 1;
    const int b = a.emeta[gi].x;
    const long e0 = ((long)(a.erow != nullptr ? a.erow[b] : b) * a.nvars + a.nv_lvid[t]) * a.C;
    const int to = a.tau != nullptr && c < Kl ? ((t * a.Bx + (a.emap != nullptr ? a.emap[b] : b)) * a.S + c) : -1;
    rec[k] = make_int4(a.skey[k], gi * 4 + (to < 0 ? 1 : 0), __float_as_int(a.lp[e0 + c] * LOG2E), to);
}

template <bool PLUS>
__global__ void __launch_bounds__(NW * 32) tk_leaf_em_kernel(const EmArgs a, const int4* __restrict__ rec) {
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
        if constexpr (!PLUS) {                                         // (F-: their records; s_g the term's row)
            const int4 r = rec[s0 + x];
            s_k[x] = r.x; s_g[x] = r.y; s_l[x] = __int_as_float(r.z);
            s_t[x] = r.w >= 0 ? (long)r.w * a.M : -1;
            continue;
        }
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
    // a block's rows: 2 float4 a thread (EB x EN / 4 = 512 of them), loaded a block ahead into registers
    static_assert(EB * EN / 4 == 2 * NW * 32, "a block's rows two float4 a thread");
    float4 nxt[2];
    const auto load_rows = [&](int bb) {
        const int k = bb / a.Vb, v0 = bb - k * a.Vb;
        const float* tb = a.tab + (long)k * a.tab_stride + (long)v0 * a.M + n0;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int q = threadIdx.x + u * NW * 32, r = q / (EN / 4), c4 = 4 * (q % (EN / 4));
            nxt[u] = v0 + r < a.V ? ld4(tb + (long)r * a.M + c4) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        }
    };
    load_rows(s_k[0] / EB * EB);
    for (int x = 0; x < ne;) {
        const int b0 = s_k[x] / EB * EB, k = b0 / a.Vb, v0 = b0 - k * a.Vb;
        const int xe = min(a.offs[b0 + EB] - s0, ne);                  // the block's entries in the run
#pragma unroll
        for (int u = 0; u < 2; ++u) {                                  // the block's rows staged, its sums zeroed
            const int q = threadIdx.x + u * NW * 32, r = q / (EN / 4), c4 = 4 * (q % (EN / 4));
            *reinterpret_cast<float4*>(&s_th[r][c4]) = nxt[u];
            s_sum[r][c4] = s_sum[r][c4 + 1] = s_sum[r][c4 + 2] = s_sum[r][c4 + 3] = 0.f;
        }
        __syncthreads();
        if (xe < ne) load_rows(s_k[xe] / EB * EB);                     // (the next block's, in flight meanwhile)
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
                w[z] = ld4(a.emr + (PLUS ? (long)s_g[y + z] * 4 + 2 + (to < 0 ? 1 : 0) : (long)s_g[y + z]) * a.M + i0);
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
// kernels read them (the rest zeros, or not written past 72) --) -- R the rows wrow names (logw empty where none).
// `ord`, `rank` [B] int32 (the evidence a row each): the samples by their row and each place's rank within it -- a
// CTA the samples of a row two at a time (empty: a sample a CTA); `sids`, `slp` [rows of ids, nvars, C - H] each
// tile's candidates past the head H = min(C, max(K, S - 1)) sorted by token id and their lp (`top_k_leaf_fused`)
std::vector<torch::Tensor> top_k_leaf_forward(torch::Tensor tab, torch::Tensor nv_tab, torch::Tensor tau, torch::Tensor emap,
                                              torch::Tensor ids, torch::Tensor lp, torch::Tensor nv_lvid,
                                              torch::Tensor nv_row0, torch::Tensor tok, torch::Tensor miss,
                                              torch::Tensor wrow, int64_t R, int64_t S, int64_t K,
                                              torch::Tensor node_mars, torch::Tensor ord, torch::Tensor rank,
                                              torch::Tensor sids, torch::Tensor slp) {
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
    a.ord = nullptr; a.rank = nullptr;
    if (ord.numel() > 0) {                                  // (a row's samples a CTA: the evidence a row each)
        check(ord, at::kInt, "ord"); check(rank, at::kInt, "rank");
        TORCH_CHECK(a.g.erow != nullptr && ord.numel() == B && rank.numel() == B, "top_k leaf: ord, rank [B]");
        a.ord = ord.data_ptr<int>(); a.rank = rank.data_ptr<int>();
    }
    if (R > 0) { check(wrow, at::kInt, "wrow"); a.wrow = wrow.data_ptr<int>(); a.logw = logw.data_ptr<float>(); }
    const int nb = a.g.C - a.g.H;                           // (the bulk by token id: every tile's, beside its lp)
    TORCH_CHECK(sids.dim() == 3 && slp.sizes() == sids.sizes() && sids.size(0) == ids.size(0) &&
                sids.size(1) == ids.size(1) && sids.size(2) == nb, "top_k leaf: sids, slp [rows, nvars, C - H]");
    a.sids = nullptr; a.slp = nullptr;
    if (nb > 0) {
        check(sids, at::kInt, "sids"); check(slp, at::kFloat, "slp");
        a.sids = sids.data_ptr<int>(); a.slp = slp.data_ptr<float>();
    }
    a.pos_y = B > 65535 ? 1 : 0;                            // (a sample's positions side by side, where grid.y holds B)
    const unsigned gz = (unsigned)((M + NW * 128 - 1) / (NW * 128));
    const dim3 grid = a.pos_y ? dim3((unsigned)B, (unsigned)T, gz) : dim3((unsigned)T, (unsigned)B, gz);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (a.ord != nullptr) tk_leaf_forward_kernel<2><<<grid, NW * 32, 0, stream>>>(a);
    else tk_leaf_forward_kernel<1><<<grid, NW * 32, 0, stream>>>(a);
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
                                  torch::Tensor rank, std::vector<int64_t> tptrs, std::vector<int64_t> tstrides,
                                  torch::Tensor rowstart, int64_t re, std::vector<bool> tstores) {
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
    // the head's walk in the tilt's gradients (`tptrs` given: the evidence a row each, the gradients folded): no dtau
    const bool hg = !tptrs.empty();
    if (hg) {
        TORCH_CHECK(fold && ids.size(0) != B && (int64_t)tptrs.size() == 4 * T && tstrides.size() == 12 &&
                    (int64_t)tstores.size() == T &&
                    T <= HG_MAXT && re <= HG_TR && re % 4 == 0 && S <= HG_TS && M % HG_C == 0 &&
                    std::min<int64_t>(ids.size(2), std::max<int64_t>(K, S - 1)) <= HG_MAXH && rowstart.numel() == Bd + 1,
                    "top_k leaf: the head's walk in the tilt's gradients: its shapes");
        check(rowstart, at::kInt, "rowstart");
    }
    auto dtau = !dtau_out || hg ? torch::empty({0}, tab.options())
              : fold ? torch::empty({T, Bd, S, M}, tab.options()) : torch::empty({T, B, S, M}, tab.options());
    a.headless = hg ? 1 : 0;
    at::cuda::CUDAEvent hg_fork, hg_join;
    torch::Tensor nterm;
    a.nterm = nullptr;
    if (hg) {
        nterm = torch::empty({T, B, 5, M}, tab.options());
        a.nterm = nterm.data_ptr<float>();
    }
    a.drow = fold ? drow.data_ptr<int>() : nullptr; a.dsgn = fold ? dsgn.data_ptr<float>() : nullptr;
    a.Bd = fold ? (int)Bd : 0;
    a.owned = nullptr;
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
    a.glp = glp.numel() > 0 ? glp.data_ptr<float>() : nullptr;
    a.dtau = dtau_out && !hg ? dtau.data_ptr<float>() : nullptr;
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
    torch::Tensor owned;
    if (fold && !hg) {                                      // (the rows a CTA stores whole; the others zeroed)
        TORCH_CHECK(M % 4 == 0, "top_k leaf: nodes a multiple of 4");
        owned = torch::empty({Bd}, tab.options().dtype(at::kBool));
        a.owned = owned.data_ptr<bool>();
        const dim3 rgrid((unsigned)Bd, (unsigned)T);
        if (a.ord != nullptr)
            tk_dtau_rows_kernel<2><<<rgrid, 256, 0, stream>>>(a.drow, a.ord, a.rank, (int)B, (int)Bd, (long)S * M,
                                                               owned.data_ptr<bool>(), a.dtau);
        else
            tk_dtau_rows_kernel<1><<<rgrid, 256, 0, stream>>>(a.drow, nullptr, nullptr, (int)B, (int)Bd, (long)S * M,
                                                               owned.data_ptr<bool>(), a.dtau);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    a.pos_y = B > 65535 ? 1 : 0;                            // (a sample's positions side by side, where grid.y holds B)
    const unsigned gz = (unsigned)((M + NW * 128 - 1) / (NW * 128));
    const dim3 grid = a.pos_y ? dim3((unsigned)B, (unsigned)T, gz) : dim3((unsigned)T, (unsigned)B, gz);
    if (a.ord != nullptr) tk_leaf_backward_kernel<2><<<grid, NW * 32, 0, stream>>>(a);
    else tk_leaf_backward_kernel<1><<<grid, NW * 32, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (hg) {                                               // (the head's walk and the tilt's gradients: a CTA a
        HgArgs hga;                                         // position and a row of the destinations)
        hga.b = a;
        hga.rowstart = rowstart.data_ptr<int>();
        for (int k = 0; k < T; ++k) {
            hga.A[k] = reinterpret_cast<const float*>(tptrs[k]); hga.Bt[k] = reinterpret_cast<const float*>(tptrs[T + k]);
            hga.dA[k] = reinterpret_cast<float*>(tptrs[2 * T + k]); hga.dB[k] = reinterpret_cast<float*>(tptrs[3 * T + k]);
        }
        hga.sab = tstrides[0]; hga.sam = tstrides[1]; hga.sar = tstrides[2];
        hga.sbb = tstrides[3]; hga.sbr = tstrides[4]; hga.sbs = tstrides[5];
        hga.sdab = tstrides[6]; hga.sdam = tstrides[7]; hga.sdar = tstrides[8];
        hga.sdbb = tstrides[9]; hga.sdbr = tstrides[10]; hga.sdbs = tstrides[11];
        TORCH_CHECK(hga.sar == 1 && hga.sdar == 1, "top_k leaf: the tilt's tensors' ranks contiguous");
        hga.t0 = 0; hga.re = (int)re;
        for (int k = 0; k < T; ++k) hga.store[k] = tstores[k] ? 1 : 0;
        static bool hg_attr[64] = {};
        const int dev = tab.get_device();
        if (!hg_attr[dev]) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(tk_leaf_head_grad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                HG_SMEM));
            hg_attr[dev] = true;
        }
        // (on a stream of its own, beside the EM statistics: its CTAs and theirs share the SMs)
        const auto side = at::cuda::getStreamFromPool(false, dev);
        hg_fork.record(stream);
        hg_fork.block(side);
        tk_leaf_head_grad_kernel<<<dim3((unsigned)T, (unsigned)Bd), HG_T, HG_SMEM, side>>>(hga);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hg_join.record(side);
    }
    const auto join = [&]() { if (hg) hg_join.block(stream); };
    if (!em) {
        join();
        return dtau;
    }
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
    // (F-: from the entries' records; F+ where a token is missing)
    auto rec = torch::empty({T * B * C, 4}, ids.options());
    int4* recp = reinterpret_cast<int4*>(rec.data_ptr<int>());
    tk_leaf_em_rec_kernel<<<(unsigned)((T * B * C + 255) / 256), 256, 0, stream>>>(e, recp);
    dim3 egrid((unsigned)((T * B * C + EC - 1) / EC), (unsigned)(M / EN));
    tk_leaf_em_kernel<false><<<egrid, NW * 32, 0, stream>>>(e, recp);
    tk_leaf_em_kernel<true><<<egrid, NW * 32, 0, stream>>>(e, recp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    join();
    return dtau;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_leaf_table", &top_k_leaf_table, "log2 theta^T of a soft-evidence leaf node vector");
    m.def("top_k_leaf_forward", &top_k_leaf_forward, "LowRank(top_k) leaves' forward");
    m.def("top_k_leaf_backward", &top_k_leaf_backward, "LowRank(top_k) leaves' backward and EM statistics");
}
