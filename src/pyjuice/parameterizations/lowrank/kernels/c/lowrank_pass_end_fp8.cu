// The "sum_exp" form's pass's end in e4m3 (`LowRank(flow_precision = "fp8")`): the operands of its two products --
// the data half's `Ud Vd^T` (the parameter flows' F+ = theta * Ud Vd^T) and the normalizer half's `Un Vt^T` (the
// denominators' W) -- written in e4m3 under a power-of-2 scale a row, for an e4m3 product with fp32 accumulation
// (`lowrank/virtual.py`: `torch._scaled_mm`, the rows' and columns' scales applied to its output).
//
// The operands are `lowrank_mixture.cu`'s `kept_operands_kernel`'s, from the exponentials the pass keeps (that file's
// header and the kernel's comment have them):
//     Ud[i, row'] = Ua[i, row] exp(lf - log S + cm_b + shA + shB)     Vd[j, row'] = Vt[j, row] exp(c_b[j] - cm_b)
//     Un[i, row]  = Ua[i, row] exp(lf - log Z + shA + shB)            Vt8[j, row] = Vt[j, row]
// every entry >= 0. A row's scale is 2^e, e the least with the row's largest entry over 2^e at most 448 (e4m3's
// largest), so its largest entry lands in (224, 448]; an entry is stored as e4m3(x 2^-e), rounded to nearest even,
// saturating. A row's largest entry over every copy of the pass takes a pass of its own (`fp8_rowmax_kernel`: the
// operands' threads, a warp's lanes of a row combined, an atomic max a segment), then the scales, then the operands
// (`fp8_operands_kernel`: `kept_operands_kernel`'s threads, the stores 8-byte runs). The children's maxima `cm_b` are `kept_cmax_kernel`'s (the same kernel as that file's).
//
// The scales, [Ud's M | Un's M | Vd's N | Vt's N] floats, 2^e with e clamped to [-126, 126]: a row of entries under
// 2^-118 keeps the clamped scale (its entries then e4m3 subnormals or 0). A row with no entry (0 everywhere): scale 1.

#include <torch/extension.h>
#include <vector>
#include <type_traits>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int COPIES_PER_LAUNCH = 64;           // (`lowrank_mixture.cu`'s)
constexpr int WARPS = 32;
constexpr int OPERAND_THREADS = WARPS * 32;
constexpr int RUN = 8;
constexpr int THREADS = 256;

inline int grid_of(long total) { return (int)((total + THREADS - 1) / THREADS); }
int run_grid(long items, long B, long K) { return grid_of(items * ((B * K + RUN - 1) / RUN)); }

inline __device__ void run_at(int t0, int K, int& b, int& r) { b = t0 / K; r = t0 - b * K; }
inline __device__ void run_step(int K, int& b, int& r) { if (++r == K) { r = 0; b++; } }

inline __device__ void load_run(const float* p, float (&v)[RUN]) {
    const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
inline __device__ void load_run(const __nv_bfloat16* p, float (&v)[RUN]) {
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    const unsigned u[4] = {a.x, a.y, a.z, a.w};
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        v[2 * q] = __uint_as_float(u[q] << 16);
        v[2 * q + 1] = __uint_as_float(u[q] & 0xffff0000u);
    }
}
inline __device__ void load_run_n(const __nv_bfloat16* p, float (&v)[RUN], int n, bool vec) {
    if (vec) {
        load_run(p, v);
    } else {
        #pragma unroll
        for (int q = 0; q < RUN; q++) v[q] = q < n ? __bfloat162float(p[q]) : 0.0f;
    }
}
inline __device__ void load_run_n(const float* p, float (&v)[RUN], int n, bool vec) {
    if (vec) {
        load_run(p, v);
    } else {
        #pragma unroll
        for (int q = 0; q < RUN; q++) v[q] = q < n ? p[q] : 0.0f;
    }
}

// The value of child `j` (element row `e`) at sample `b`: its two children's node values, or the kept `cv`
__device__ __forceinline__ float child_value(const float* cv, const int32_t* table, long e, long first, int j, int b,
                                             const float* node_mars, int B) {
    if (cv != nullptr) return cv[(long)j * B + b];
    const int32_t* ch = table + (e - first) * 2;
    return node_mars[(long)ch[0] * B + b] + node_mars[(long)ch[1] * B + b];
}

// `lowrank_mixture.cu`'s KeptCopies: log Z, first node row, eid, table, first, row0, drow0, Un, cv
struct KeptCopies { long long rows[COPIES_PER_LAUNCH * 9]; };

// cm_b of a launch's copies with a data half (`lowrank_mixture.cu`'s `kept_cmax_kernel`)
__global__ void kept_cmax_kernel(const KeptCopies copies, float* __restrict__ cm, const float* __restrict__ node_mars,
                                 int N, int B) {
    const long long* c = copies.rows + 9 * blockIdx.z;
    const int64_t* eid = reinterpret_cast<const int64_t*>(c[2]);
    const int32_t* table = reinterpret_cast<const int32_t*>(c[3]);
    const long first = (long)c[4];
    const float* cv = reinterpret_cast<const float*>(c[8]);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, b = blockIdx.x * 32 + lane;
    __shared__ float most[WARPS][32];
    float m = -INFINITY;
    if (b < B && c[6] >= 0) {
        for (int j = w; j < N; j += WARPS) {
            const long e = eid[j];
            if (e < 0) continue;
            m = fmaxf(m, child_value(cv, table, e, first, j, b, node_mars, B));
        }
    }
    most[w][lane] = m;
    __syncthreads();
    if (w != 0 || b >= B) return;
    float sh = most[0][lane];
    for (int i = 1; i < WARPS; i++) sh = fmaxf(sh, most[i][lane]);
    cm[(long)blockIdx.z * B + b] = sh;
}

// A child's run of Vd (`dh`) and its Vt (as loaded), a node's runs of Ud (`dh`) and Un (`un`): the operands' values,
// `kept_operands_kernel`'s expressions
struct RunCtx {
    const float* lz; long nid0; const int64_t* eid; const int32_t* table; long first, row0, drow0; const float* cv;
    bool dh, un;
};
inline __device__ RunCtx run_ctx(const long long* c) {
    RunCtx k;
    k.lz = reinterpret_cast<const float*>(c[0]); k.nid0 = (long)c[1];
    k.eid = reinterpret_cast<const int64_t*>(c[2]); k.table = reinterpret_cast<const int32_t*>(c[3]);
    k.first = (long)c[4]; k.row0 = (long)c[5]; k.drow0 = (long)c[6]; k.cv = reinterpret_cast<const float*>(c[8]);
    k.dh = k.drow0 >= 0; k.un = c[7] != 0;
    return k;
}
inline __device__ void child_run(const RunCtx& k, const float (&vin)[RUN], float (&v)[RUN], int x, int t0, int n,
                                 const float* cmb, const float* node_mars, int B, int K) {
    int b, r;
    run_at(t0, K, b, r);
    const long e = k.eid[x];
    int bb = -1;
    float w = 0.0f, m = 0.0f;                                      // (the run's sample's: loaded once)
    #pragma unroll
    for (int q = 0; q < RUN; q++) {
        float y = 0.0f;
        if (q < n && e >= 0) {
            if (b != bb) { w = child_value(k.cv, k.table, e, k.first, x, b, node_mars, B); m = cmb[b]; bb = b; }
            y = (w > -INFINITY && m > -INFINITY && vin[q] != 0.0f) ? vin[q] * expf(w - m) : 0.0f;
        }
        v[q] = y;
        run_step(K, b, r);
    }
}
inline __device__ void node_run(const RunCtx& k, const float (&uin)[RUN], const float (&ha)[RUN], const float (&hb)[RUN],
                                float (&d)[RUN], float (&v)[RUN], int x, int t0, int n, const float* cmb,
                                const float* node_flows, const float* node_mars, int B, int K) {
    int b, r;
    run_at(t0, K, b, r);
    int bb = -1;
    float lf = 0.0f, z = 0.0f, ls = 0.0f;                          // (the run's sample's: loaded once)
    #pragma unroll
    for (int q = 0; q < RUN; q++) {
        float yd = 0.0f, yn = 0.0f;
        if (q < n && uin[q] != 0.0f) {
            if (b != bb) {
                lf = node_flows[(k.nid0 + x) * (long)B + b];
                z = k.lz[(long)x * B + b];
                ls = node_mars[(k.nid0 + x) * (long)B + b] + z;    // log S = log n + log Z
                bb = b;
            }
            const float h = ha[q] + hb[q];
            if (k.dh && lf > -INFINITY && ls > -INFINITY && cmb[b] > -INFINITY)
                yd = uin[q] * expf(lf - ls + cmb[b] + h);
            if (k.un && lf > -INFINITY && z > -INFINITY && z < INFINITY) yn = uin[q] * expf(lf - z + h);
        }
        d[q] = yd; v[q] = yn;
        run_step(K, b, r);
    }
}

// The scale 2^e of a row whose largest entry is `m`: e the least with m 2^-e <= 448 (= 1.75 2^8), clamped to
// [-126, 126] (2^e and 2^-e normal floats); m 0 (or under fp32's normal range): 1
__device__ __forceinline__ float fp8_scale(float m) {
    int e = 0;
    if (m > 0.0f) {
        const int ex = ilogbf(m);
        e = (m <= ldexpf(1.75f, ex)) ? ex - 8 : ex - 7;
        e = max(-126, min(126, e));
    }
    return __int_as_float((127 + e) << 23);
}
__device__ __forceinline__ float inv_pow2(float s) { return __int_as_float((254 << 23) - __float_as_int(s)); }

// Each operand row's largest entry over a launch's copies, `fp8_operands_kernel`'s threads: a thread its run's, a
// warp's lanes of one row (consecutive threads: a row's runs) combined, the segment's first lane one atomic max a role
// into `mx` [Ud M | Un M | Vd N | Vt N] (zeroed before the first launch; entries >= 0: their bits order as unsigned)
__global__ void fp8_rowmax_kernel(const KeptCopies copies, unsigned* __restrict__ mx, const __nv_bfloat16* __restrict__ ua,
                                  const float* __restrict__ sha, const __nv_bfloat16* __restrict__ vt,
                                  const float* __restrict__ shb, const float* __restrict__ cm, long rows,
                                  const float* __restrict__ node_flows, const float* __restrict__ node_mars, int M,
                                  int N, int B, int K, bool vec) {
    const RunCtx k = run_ctx(copies.rows + 9 * blockIdx.z);
    const int R = B * K, runs = (R + RUN - 1) / RUN;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = t < (long)(N + M) * runs && (k.dh || k.un);
    const bool of_v = t < (long)N * runs;
    const long u = of_v ? t : t - (long)N * runs;
    const int x = (int)(u / runs), t0 = (int)(u % runs) * RUN;
    const float* cmb = cm + (long)blockIdx.z * B;
    float m1 = 0.0f, m2 = 0.0f;                      // a child's Vd, Vt; a node's Ud, Un
    if (live) {
        const int n = min(RUN, R - t0);
        if (of_v) {
            float vin[RUN], v[RUN];
            load_run_n(vt + (long)x * rows + k.row0 + t0, vin, n, vec);
            if (k.dh) {
                child_run(k, vin, v, x, t0, n, cmb, node_mars, B, K);
                #pragma unroll
                for (int q = 0; q < RUN; q++) m1 = fmaxf(m1, v[q]);
            }
            if (k.un) {
                #pragma unroll
                for (int q = 0; q < RUN; q++) m2 = fmaxf(m2, q < n ? vin[q] : 0.0f);
            }
        } else {
            float uin[RUN], ha[RUN], hb[RUN], d[RUN], v[RUN];
            load_run_n(ua + (long)x * rows + k.row0 + t0, uin, n, vec);
            load_run_n(sha + k.row0 + t0, ha, n, vec);
            load_run_n(shb + k.row0 + t0, hb, n, vec);
            node_run(k, uin, ha, hb, d, v, x, t0, n, cmb, node_flows, node_mars, B, K);
            #pragma unroll
            for (int q = 0; q < RUN; q++) { m1 = fmaxf(m1, d[q]); m2 = fmaxf(m2, v[q]); }
        }
    }
    const int key = live ? (of_v ? x : N + x) : -1, lane = threadIdx.x & 31;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {               // (a row's lanes contiguous: lane L ends with [L, L + 31]'s of it)
        const float a1 = __shfl_down_sync(0xffffffffu, m1, o), a2 = __shfl_down_sync(0xffffffffu, m2, o);
        const int kk = __shfl_down_sync(0xffffffffu, key, o);
        if (lane + o < 32 && kk == key) { m1 = fmaxf(m1, a1); m2 = fmaxf(m2, a2); }
    }
    const int prev = __shfl_up_sync(0xffffffffu, key, 1);
    if (key < 0 || (lane > 0 && prev == key)) return;
    unsigned* a = of_v ? mx + 2 * M + x : mx + x;           // (Vd's or Ud's)
    unsigned* c = of_v ? mx + 2 * M + N + x : mx + M + x;   // (Vt's or Un's)
    if (m1 > 0.0f) atomicMax(a, __float_as_uint(m1));
    if (m2 > 0.0f) atomicMax(c, __float_as_uint(m2));
}

// The rows' scales (`fp8_scale`) from their maxima
__global__ void fp8_scales_kernel(const unsigned* __restrict__ mx, float* __restrict__ scales, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) scales[i] = fp8_scale(__uint_as_float(mx[i]));
}

// `out [M, N] *= sa[i] sb[j]` (powers of 2: exact)
__global__ void scale_rows_cols_kernel(float* __restrict__ out, const float* __restrict__ sa, const float* __restrict__ sb,
                                       long M, int N, bool vec) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vec) {
        if (t >= M * (N / 4)) return;
        const long i = t / (N / 4);
        const int j = (int)(t % (N / 4)) * 4;
        float4 v = reinterpret_cast<float4*>(out)[t];
        const float a = sa[i];
        v.x *= a * sb[j]; v.y *= a * sb[j + 1]; v.z *= a * sb[j + 2]; v.w *= a * sb[j + 3];
        reinterpret_cast<float4*>(out)[t] = v;
    } else {
        if (t >= M * N) return;
        out[t] *= sa[t / N] * sb[t % N];
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// The in-place V operand (`lowrank_mixture.cu`'s `inplace_operand_kernel<.., false>`, its operations: the same bits),
// each (child, copy, sample)'s largest V over its ranks besides, as stored -- `g [N, rows / K]`, the column the (copy,
// sample)'s first row over K --: what the children's rows' maxima take in place of a pass over V (`fp8_gmax_kernel`).
struct ICopies { long long p[COPIES_PER_LAUNCH]; long long sb[COPIES_PER_LAUNCH]; };
constexpr int ITHREADS = 256, INW = ITHREADS / 32, IKEEP = 64;

inline __device__ void store_as(float* p, float v) { *p = v; }
inline __device__ void store_as(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
inline __device__ float as_stored(const float*, float v) { return v; }
inline __device__ float as_stored(const __nv_bfloat16*, float v) { return __bfloat162float(__float2bfloat16(v)); }

template <typename TO, int KC>
__device__ __forceinline__ void store_kc(TO* o, const float (&v)[KC]) {
    if constexpr (std::is_same<TO, __nv_bfloat16>::value && KC % 8 == 0) {
        #pragma unroll
        for (int q = 0; q < KC / 8; q++) {
            __nv_bfloat162 h[4];
            #pragma unroll
            for (int x = 0; x < 4; x++) h[x] = __floats2bfloat162_rn(v[8 * q + 2 * x], v[8 * q + 2 * x + 1]);
            *reinterpret_cast<uint4*>(o + 8 * q) = *reinterpret_cast<const uint4*>(h);
        }
    } else if constexpr (std::is_same<TO, float>::value && KC % 4 == 0) {
        #pragma unroll
        for (int q = 0; q < KC / 4; q++)
            *reinterpret_cast<float4*>(o + 4 * q) = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    } else {
        #pragma unroll
        for (int r = 0; r < KC; r++) store_as(o + r, v[r]);
    }
}

template <typename TO, int KC>
__global__ void __launch_bounds__(ITHREADS) inplace_g_kernel(const ICopies copies, TO* __restrict__ out,
                                                             float* __restrict__ shift, long rows, long row0,
                                                             int X, int B, float* __restrict__ g, long gcols) {
    static_assert(KC <= 32 && ITHREADS % KC == 0, "a warp's lanes the ranks' maxima");
    const int b = blockIdx.x, c = blockIdx.y, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const float* f = reinterpret_cast<const float*>(copies.p[c]) + (long)b * copies.sb[c];
    const long at = row0 + (long)c * KC * B + (long)b * KC;           // the (copy, sample)'s first row
    __shared__ float red[KC][INW];
    __shared__ float mk[KC];
    constexpr int IKB = IKEEP / KC;                                // children kept a rank
    float keep[KC][IKB];                                           // B[b, r, x] at r X + x
    float mr[KC];
    #pragma unroll
    for (int r = 0; r < KC; r++) {
        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < IKB; k++) {
            const int x = t + ITHREADS * k;
            keep[r][k] = x < X ? f[(long)r * X + x] : -INFINITY;
            m = fmaxf(m, keep[r][k]);
        }
        for (int x = t + ITHREADS * IKB; x < X; x += ITHREADS) m = fmaxf(m, f[(long)r * X + x]);
        for (int off = 16; off >= 1; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        mr[r] = m;
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < KC; r++) red[r][w] = mr[r];
    }
    __syncthreads();
    if (t < KC) {
        float x = red[t][0];
        for (int i = 1; i < INW; i++) x = fmaxf(x, red[t][i]);
        mk[t] = x;
        shift[at + t] = x;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < KC; r++) mr[r] = mk[r];
    #pragma unroll
    for (int k = 0; k < IKB; k++) {
        const int x = t + ITHREADS * k;
        if (x < X) {
            float v[KC];
            #pragma unroll
            for (int r = 0; r < KC; r++) v[r] = keep[r][k] > -INFINITY ? expf(keep[r][k] - mr[r]) : 0.0f;
            store_kc<TO, KC>(out + (long)x * rows + at, v);
            float mv = 0.0f;
            #pragma unroll
            for (int r = 0; r < KC; r++) mv = fmaxf(mv, as_stored(out, v[r]));
            g[(long)x * gcols + at / KC] = mv;
        }
    }
    for (int x = t + ITHREADS * IKB; x < X; x += ITHREADS) {
        float v[KC];
        #pragma unroll
        for (int r = 0; r < KC; r++) {
            const float y = f[(long)r * X + x];
            v[r] = y > -INFINITY ? expf(y - mr[r]) : 0.0f;
        }
        store_kc<TO, KC>(out + (long)x * rows + at, v);
        float mv = 0.0f;
        #pragma unroll
        for (int r = 0; r < KC; r++) mv = fmaxf(mv, as_stored(out, v[r]));
        g[(long)x * gcols + at / KC] = mv;
    }

}


// Each operand row's largest entry from the children's (row, copy, sample) maxima `gv` (V's twin's) and, a node's,
// from its runs of `Ua` (the largest log of Ua exp(shA + shB) over a (node, sample)'s ranks, the exponent apart):
// a thread a (row, sample) of a launch's copy -- the first N B a child's (Vd, Vt), the M B after them a node's (Ud,
// Un) --, the operands' expressions with the rank's factors at their largest; a warp's lanes of one row combined, one
// atomic max a segment (`fp8_rowmax_kernel`'s)
__global__ void fp8_gmax_kernel(const KeptCopies copies, unsigned* __restrict__ mx, const __nv_bfloat16* __restrict__ ua,
                                const float* __restrict__ sha, const float* __restrict__ shb, long rows,
                                const float* __restrict__ gv, long gcols, const float* __restrict__ cm,
                                const float* __restrict__ node_flows, const float* __restrict__ node_mars, int M, int N,
                                int B, int K, bool vec) {
    const RunCtx k = run_ctx(copies.rows + 9 * blockIdx.z);
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = t < (long)(N + M) * B && (k.dh || k.un);
    const bool of_v = t < (long)N * B;
    const long u = of_v ? t : t - (long)N * B;
    const int x = (int)(u / B), b = (int)(u % B);
    const long col = k.row0 / K + b;
    const float* cmb = cm + (long)blockIdx.z * B;
    float m1 = 0.0f, m2 = 0.0f;
    if (live) {
        if (of_v) {
            const float vmax = gv[(long)x * gcols + col];
            if (k.dh) {
                const long e = k.eid[x];
                if (e >= 0 && vmax != 0.0f) {
                    const float w = child_value(k.cv, k.table, e, k.first, x, b, node_mars, B), m = cmb[b];
                    if (w > -INFINITY && m > -INFINITY) m1 = vmax * expf(w - m);
                }
            }
            if (k.un) m2 = vmax;
        } else {
            // (the (node, sample)'s largest log of Ua exp(shA + shB) over its K ranks, a run of Ua)
            const long r0 = k.row0 + (long)b * K;
            const __nv_bfloat16* up = ua + (long)x * rows + r0;
            float lg = -INFINITY;
            if (vec) {                                   // (K a multiple of RUN, the runs aligned: 16-byte loads)
                for (int r = 0; r < K; r += RUN) {
                    float y[RUN], ha[RUN], hb[RUN];
                    load_run(up + r, y); load_run(sha + r0 + r, ha); load_run(shb + r0 + r, hb);
                    #pragma unroll
                    for (int q = 0; q < RUN; q++)
                        if (y[q] > 0.0f) lg = fmaxf(lg, __logf(y[q]) + ha[q] + hb[q]);
                }
            } else {
                for (int r = 0; r < K; r++) {
                    const float y = __bfloat162float(up[r]);
                    if (y > 0.0f) lg = fmaxf(lg, __logf(y) + sha[r0 + r] + shb[r0 + r]);
                }
            }
            if (lg > -INFINITY) {
                const float lf = node_flows[(k.nid0 + x) * (long)B + b];
                const float z = k.lz[(long)x * B + b];
                const float ls = node_mars[(k.nid0 + x) * (long)B + b] + z;
                if (k.dh && lf > -INFINITY && ls > -INFINITY && cmb[b] > -INFINITY) m1 = expf(lf - ls + cmb[b] + lg);
                if (k.un && lf > -INFINITY && z > -INFINITY && z < INFINITY) m2 = expf(lf - z + lg);
            }
        }
    }
    const int key = live ? (of_v ? x : N + x) : -1, lane = threadIdx.x & 31;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float a1 = __shfl_down_sync(0xffffffffu, m1, o), a2 = __shfl_down_sync(0xffffffffu, m2, o);
        const int kk = __shfl_down_sync(0xffffffffu, key, o);
        if (lane + o < 32 && kk == key) { m1 = fmaxf(m1, a1); m2 = fmaxf(m2, a2); }
    }
    const int prev = __shfl_up_sync(0xffffffffu, key, 1);
    if (key < 0 || (lane > 0 && prev == key)) return;
    unsigned* a = of_v ? mx + 2 * M + x : mx + x;
    unsigned* c = of_v ? mx + 2 * M + N + x : mx + M + x;
    if (m1 > 0.0f) atomicMax(a, __float_as_uint(m1));
    if (m2 > 0.0f) atomicMax(c, __float_as_uint(m2));
}

inline __device__ unsigned fp8x2(float a, float b) {
    return (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}
// A run's `n` values times `inv` in e4m3 from `p` on: one 8-byte store where every run is whole and aligned (`vec`)
inline __device__ void store_run8(uint8_t* p, const float (&v)[RUN], float inv, int n, bool vec) {
    if (vec) {
        uint2 w;
        w.x = fp8x2(v[0] * inv, v[1] * inv) | (fp8x2(v[2] * inv, v[3] * inv) << 16);
        w.y = fp8x2(v[4] * inv, v[5] * inv) | (fp8x2(v[6] * inv, v[7] * inv) << 16);
        *reinterpret_cast<uint2*>(p) = w;
    } else {
        for (int q = 0; q < n; q++) p[q] = (uint8_t)__nv_cvt_float_to_fp8(v[q] * inv, __NV_SATFINITE, __NV_E4M3);
    }
}

// The operands in e4m3, `kept_operands_kernel`'s threads: the first `N runs` of a copy a child's run -- Vd's (a copy
// with a data half) and Vt's copy (a copy with rows of Un) --, the `M runs` after them a node's, Ud's and Un's
__global__ void fp8_operands_kernel(const KeptCopies copies, uint8_t* __restrict__ Ud, uint8_t* __restrict__ Vd,
                                    uint8_t* __restrict__ Un, uint8_t* __restrict__ Vt8,
                                    const float* __restrict__ scales, const __nv_bfloat16* __restrict__ ua,
                                    const float* __restrict__ sha, const __nv_bfloat16* __restrict__ vt,
                                    const float* __restrict__ shb, const float* __restrict__ cm, long rows, long drows,
                                    const float* __restrict__ node_flows, const float* __restrict__ node_mars, int M,
                                    int N, int B, int K, bool vec) {
    const RunCtx k = run_ctx(copies.rows + 9 * blockIdx.z);
    const int R = B * K, runs = (R + RUN - 1) / RUN;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)(N + M) * runs) return;
    if (!(k.dh || k.un)) return;
    const bool of_v = t < (long)N * runs;
    const long u = of_v ? t : t - (long)N * runs;
    const int x = (int)(u / runs), t0 = (int)(u % runs) * RUN, n = min(RUN, R - t0);
    const float* cmb = cm + (long)blockIdx.z * B;
    if (of_v) {
        float vin[RUN], v[RUN];
        load_run_n(vt + (long)x * rows + k.row0 + t0, vin, n, vec);
        if (k.dh) {
            child_run(k, vin, v, x, t0, n, cmb, node_mars, B, K);
            store_run8(Vd + (long)x * drows + k.drow0 + t0, v, inv_pow2(scales[2 * M + x]), n, vec);
        }
        if (k.un) store_run8(Vt8 + (long)x * rows + k.row0 + t0, vin, inv_pow2(scales[2 * M + N + x]), n, vec);
    } else {
        float uin[RUN], ha[RUN], hb[RUN], d[RUN], v[RUN];
        load_run_n(ua + (long)x * rows + k.row0 + t0, uin, n, vec);
        load_run_n(sha + k.row0 + t0, ha, n, vec);
        load_run_n(shb + k.row0 + t0, hb, n, vec);
        node_run(k, uin, ha, hb, d, v, x, t0, n, cmb, node_flows, node_mars, B, K);
        if (k.dh) store_run8(Ud + (long)x * drows + k.drow0 + t0, d, inv_pow2(scales[x]), n, vec);
        if (k.un) store_run8(Un + (long)x * rows + k.row0 + t0, v, inv_pow2(scales[M + x]), n, vec);
    }
}

void check(const torch::Tensor& t, const char* name, torch::ScalarType st, long numel) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "lowrank fp8: `", name, "` must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == st, "lowrank fp8: `", name, "` has the wrong dtype");
    TORCH_CHECK(numel < 0 || t.numel() == numel, "lowrank fp8: `", name, "` has ", t.numel(),
                " elements, expected ", numel);
}

bool whole_runs(const torch::Tensor& op, long B, long K, int align) {
    return (B * K) % RUN == 0 && op.size(1) % RUN == 0 && (reinterpret_cast<uintptr_t>(op.data_ptr()) % align) == 0;
}

}  // namespace

// `lowrank_kept_operands` (`lowrank_mixture.cu`) with the operands in e4m3 under a power-of-2 scale a row: `Ud8`,
// `Vd8` [M | N, data rows], `Un8`, `Vt8` [M | N, rows] (Float8_e4m3fn; Un8 and Vt8 both or neither -- a copy with rows
// of Un writes its columns of both), `scales` [2 M + 2 N] float32 (their rows' scales: Ud, Un, Vd, Vt); the other
// arguments `lowrank_kept_operands`'s. Three phases, every launch's in turn: the children's maxima, the rows' maxima
// over every copy (then the scales), the operands.
void lowrank_kept_operands_fp8(c10::optional<torch::Tensor> Ud, c10::optional<torch::Tensor> Vd,
                               c10::optional<torch::Tensor> Un, c10::optional<torch::Tensor> Vt8, torch::Tensor scales,
                               torch::Tensor ua, torch::Tensor sha, torch::Tensor vt, torch::Tensor shb,
                               torch::Tensor node_flows, torch::Tensor node_mars, std::vector<torch::Tensor> lz,
                               std::vector<int64_t> nid0, std::vector<c10::optional<torch::Tensor>> eid,
                               std::vector<c10::optional<torch::Tensor>> table, std::vector<int64_t> first,
                               std::vector<int64_t> row0, std::vector<int64_t> drow0, std::vector<int64_t> un, int64_t M,
                               int64_t K, std::vector<c10::optional<torch::Tensor>> cv, c10::optional<torch::Tensor> gv) {
    check(node_flows, "node_flows", torch::kFloat32, -1);
    check(node_mars, "node_mars", torch::kFloat32, node_flows.numel());
    const size_t P = lz.size();
    TORCH_CHECK(nid0.size() == P && eid.size() == P && table.size() == P && first.size() == P && row0.size() == P &&
                drow0.size() == P && un.size() == P && cv.size() == P, "lowrank fp8: the copies' lists do not match");
    TORCH_CHECK(ua.is_cuda() && ua.is_contiguous() && ua.scalar_type() == torch::kBFloat16 && ua.dim() == 2 &&
                ua.size(0) == M && vt.is_cuda() && vt.is_contiguous() && vt.scalar_type() == torch::kBFloat16 &&
                vt.dim() == 2 && vt.size(1) == ua.size(1), "lowrank fp8: Ua [M, rows], Vt [N, rows] bf16");
    const int B = node_flows.size(1), N = vt.size(0);
    const long rows = ua.size(1), drows = Ud.has_value() ? Ud->size(1) : 0;
    check(sha, "sha", torch::kFloat32, rows); check(shb, "shb", torch::kFloat32, rows);
    check(scales, "scales", torch::kFloat32, 2 * M + 2 * (long)N);
    auto f8 = [&](c10::optional<torch::Tensor>& t, long r, long c, const char* name) {
        if (t.has_value())
            TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat8_e4m3fn &&
                        t->dim() == 2 && t->size(0) == r && t->size(1) == c, "lowrank fp8: `", name,
                        "` must be e4m3 [", r, ", ", c, "]");
    };
    TORCH_CHECK(Ud.has_value() == Vd.has_value() && Un.has_value() == Vt8.has_value(),
                "lowrank fp8: Ud with Vd, Un with Vt8");
    f8(Ud, M, drows, "Ud"); f8(Vd, N, drows, "Vd"); f8(Un, M, rows, "Un"); f8(Vt8, N, rows, "Vt8");
    if (P == 0 || rows == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    bool vec = whole_runs(ua, B, K, 16) && whole_runs(vt, B, K, 16) && (!Un.has_value() || (whole_runs(*Un, B, K, 8) &&
               whole_runs(*Vt8, B, K, 8))) && (!Ud.has_value() || (whole_runs(*Ud, B, K, 8) && whole_runs(*Vd, B, K, 8))) &&
               (reinterpret_cast<uintptr_t>(sha.data_ptr()) & 15) == 0 &&
               (reinterpret_cast<uintptr_t>(shb.data_ptr()) & 15) == 0;
    for (size_t c = 0; c < P; c++) vec = vec && row0[c] % RUN == 0 && (drow0[c] < 0 || drow0[c] % RUN == 0);
    torch::Tensor cm = torch::empty({(long)P * B}, node_flows.options());
    torch::Tensor mx = torch::empty({2 * M + 2 * (long)N}, node_flows.options().dtype(torch::kInt32));
    C10_CUDA_CHECK(cudaMemsetAsync(mx.data_ptr(), 0, mx.numel() * 4, stream));
    std::vector<KeptCopies> launches;
    std::vector<int> counts;
    for (size_t p0 = 0; p0 < P; p0 += COPIES_PER_LAUNCH) {
        const int n = (int)std::min<size_t>(COPIES_PER_LAUNCH, P - p0);
        KeptCopies copies;
        bool any_d = false;
        for (int i = 0; i < n; i++) {
            const size_t c = p0 + i;
            check(lz[c], "lz", torch::kFloat32, (long)B * M);
            TORCH_CHECK(nid0[c] >= 0 && node_flows.size(0) >= nid0[c] + M, "lowrank fp8: a copy's node rows");
            TORCH_CHECK(row0[c] >= 0 && row0[c] + (long)B * K <= rows, "lowrank fp8: a copy's rows of Ua");
            const bool d = drow0[c] >= 0;
            const bool kept = d && cv[c].has_value();
            TORCH_CHECK(!d || (Ud.has_value() && eid[c].has_value() && (kept || table[c].has_value()) &&
                               drow0[c] + (long)B * K <= drows), "lowrank fp8: a copy's data half");
            TORCH_CHECK(!un[c] || Un.has_value(), "lowrank fp8: a copy's rows of Un without Un");
            if (d) {
                check(*eid[c], "eid", torch::kInt64, N);
                if (kept) check(*cv[c], "the children's values", torch::kFloat32, (long)N * B);
                else check(*table[c], "table", torch::kInt32, -1);
            }
            any_d = any_d || d;
            long long* row = copies.rows + 9 * i;
            row[0] = (long long)lz[c].data_ptr<float>(); row[1] = (long long)nid0[c];
            row[2] = d ? (long long)eid[c]->data_ptr<int64_t>() : 0;
            row[3] = d && !kept ? (long long)table[c]->data_ptr<int32_t>() : 0;
            row[4] = (long long)first[c]; row[5] = (long long)row0[c]; row[6] = (long long)drow0[c];
            row[7] = (long long)(un[c] != 0); row[8] = kept ? (long long)cv[c]->data_ptr<float>() : 0;
        }
        if (any_d) {
            kept_cmax_kernel<<<dim3((B + 31) / 32, 1, n), OPERAND_THREADS, 0, stream>>>(
                copies, cm.data_ptr<float>() + (long)p0 * B, node_mars.data_ptr<float>(), N, B);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        launches.push_back(copies);
        counts.push_back(n);
    }
    auto u8 = [](c10::optional<torch::Tensor>& t) {
        return t.has_value() ? reinterpret_cast<uint8_t*>(t->data_ptr()) : (uint8_t*)nullptr;
    };
    const auto* uap = reinterpret_cast<const __nv_bfloat16*>(ua.data_ptr());
    const auto* vtp = reinterpret_cast<const __nv_bfloat16*>(vt.data_ptr());
    unsigned* mxp = reinterpret_cast<unsigned*>(mx.data_ptr());
    const bool from_g = gv.has_value();
    if (from_g) {
        check(*gv, "gv", torch::kFloat32, (long)N * (rows / K));
        TORCH_CHECK(rows % K == 0, "lowrank fp8: rows of whole (copy, sample)s");
    }
    for (size_t l = 0; from_g && l < launches.size(); l++) {
        fp8_gmax_kernel<<<dim3(grid_of((long)(N + M) * B), 1, counts[l]), THREADS, 0, stream>>>(
            launches[l], mxp, uap, sha.data_ptr<float>(), shb.data_ptr<float>(), rows, gv->data_ptr<float>(), rows / K,
            cm.data_ptr<float>() + (long)l * COPIES_PER_LAUNCH * B, node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), (int)M, N, B, (int)K, vec && K % RUN == 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    for (size_t l = 0; !from_g && l < launches.size(); l++) {
        fp8_rowmax_kernel<<<dim3(run_grid(N + M, B, K), 1, counts[l]), THREADS, 0, stream>>>(
            launches[l], mxp, uap, sha.data_ptr<float>(), vtp, shb.data_ptr<float>(),
            cm.data_ptr<float>() + (long)l * COPIES_PER_LAUNCH * B, rows, node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), (int)M, N, B, (int)K, vec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    fp8_scales_kernel<<<grid_of(2 * M + 2 * (long)N), THREADS, 0, stream>>>(mxp, scales.data_ptr<float>(),
                                                                          (int)(2 * M + 2 * (long)N));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (size_t l = 0; l < launches.size(); l++) {
        fp8_operands_kernel<<<dim3(run_grid(N + M, B, K), 1, counts[l]), THREADS, 0, stream>>>(
            launches[l], u8(Ud), u8(Vd), u8(Un), u8(Vt8), scales.data_ptr<float>(), uap, sha.data_ptr<float>(), vtp,
            shb.data_ptr<float>(), cm.data_ptr<float>() + (long)l * COPIES_PER_LAUNCH * B, rows, drows,
            node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), (int)M, N, B, (int)K, vec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


namespace {
template <int KC>
void inplace_g_launch(torch::Tensor& out, torch::Tensor& shift, torch::Tensor& g,
                      std::vector<torch::Tensor>& fs, int64_t B) {
    const long rows = out.size(1), P = (long)fs.size();
    const int X = (int)out.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    for (long p0 = 0; p0 < P; p0 += COPIES_PER_LAUNCH) {
        const int n = (int)std::min<long>(COPIES_PER_LAUNCH, P - p0);
        ICopies copies;
        for (int i = 0; i < n; i++) {
            const torch::Tensor& f = fs[p0 + i];
            TORCH_CHECK(f.is_cuda() && f.scalar_type() == torch::kFloat32 && f.dim() == 3 && f.size(0) == B &&
                        f.size(1) == KC && f.size(2) == X &&
                        f.stride(2) == 1 && f.stride(1) == f.size(2) && f.stride(0) >= f.size(1) * f.size(2),
                        "lowrank fp8: a factor in place is fp32 B [B, K, N], a sample's block contiguous");
            copies.p[i] = (long long)f.data_ptr<float>();
            copies.sb[i] = (long long)f.stride(0);
        }
        inplace_g_kernel<__nv_bfloat16, KC><<<dim3((unsigned)B, (unsigned)n), ITHREADS, 0, stream>>>(
            copies, static_cast<__nv_bfloat16*>(out.data_ptr()), shift.data_ptr<float>(), rows, p0 * KC * B, X, (int)B,
            g.data_ptr<float>(), rows / KC);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void inplace_g(torch::Tensor& out, torch::Tensor& shift, torch::Tensor& g, std::vector<torch::Tensor>& fs, int64_t B,
               int64_t K) {
    const long rows = (long)fs.size() * B * K;
    check(shift, "shift", torch::kFloat32, rows);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2 && out.size(1) == rows &&
                out.scalar_type() == torch::kBFloat16, "lowrank fp8: the operand bf16 [X, the copies' rows]");
    check(g, "g", torch::kFloat32, out.size(0) * (rows / std::max<int64_t>(K, 1)));
    if (rows == 0) return;
#define GO(KC_) if (K == KC_) { inplace_g_launch<KC_>(out, shift, g, fs, B); return; }
    GO(4) GO(8) GO(16) GO(32)
#undef GO
    TORCH_CHECK(false, "lowrank fp8: the operands in place serve ranks 4, 8, 16, 32; got ", K);
}
}  // namespace

// `lowrank_z_operand_in_place` (`lowrank_mixture.cu`: the same bits) and each (child, copy, sample)'s largest V over its
// ranks into `gv` [N, rows / K]
void lowrank_z_operand_in_place_g(torch::Tensor Vt, torch::Tensor shift, torch::Tensor gv, std::vector<torch::Tensor> fb,
                                  int64_t B, int64_t K) {
    inplace_g(Vt, shift, gv, fb, B, K);
}

// `out [M, N] *= sa[i] sb[j]`, `sa` [M], `sb` [N] the rows' and the columns' power-of-2 scales (float32, CUDA)
void lowrank_scale_rows_cols(torch::Tensor out, torch::Tensor sa, torch::Tensor sb) {
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat32 && out.dim() == 2,
                "lowrank fp8: the product fp32 [M, N], contiguous");
    const long M = out.size(0);
    const int N = (int)out.size(1);
    check(sa, "sa", torch::kFloat32, M); check(sb, "sb", torch::kFloat32, N);
    if (M == 0 || N == 0) return;
    const bool vec = N % 4 == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0;
    scale_rows_cols_kernel<<<grid_of(vec ? M * (N / 4) : M * N), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<float>(), sa.data_ptr<float>(), sb.data_ptr<float>(), M, N, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_scale_rows_cols", &lowrank_scale_rows_cols, "out *= sa[:, None] * sb[None, :] (powers of 2)");
    m.def("lowrank_z_operand_in_place_g", &lowrank_z_operand_in_place_g,
          "lowrank_z_operand_in_place, and each (child, copy, sample)'s largest V over its ranks");
    m.def("lowrank_kept_operands_fp8", &lowrank_kept_operands_fp8,
          "the pass's end operands from the kept exponentials in e4m3, a power-of-2 scale a row (and Vt's e4m3 copy)");
}
