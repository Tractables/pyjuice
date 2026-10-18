// The `input @ block` top-down step on compact values (`kernels/prod_top_down.py`, `_input_block_top_down_kernel`):
// the right operand R over [t + 1, t2) of a launch's children,
//     O_R[q, j] = log sum over the parents, and over the entries (i, class) leading to q at boundary t, of
//                 mass + O_out[i, j],
// a sample's terms those of its live columns i (a dead one's parent lane holds nothing).
//
// Two kernels per launch:
//   `ibtd_terms_kernel`, a warp per (boundary t of the launch, sample, live column q at t + 1), 32 entries at a time:
//   the entries of q's predecessor list that the sample sums -- i live at t, and the observed class if the sample
//   observes t --, in the list's order, as (i's place among the sample's live columns, class) packed in 32 bits (the
//   class in the low `xb`, as few as the classes need) at q's own place in a copy of the list per sample (`terms`), and
//   their number (`cnt`); every program of the main kernel would otherwise
//   walk the whole list, mostly dead entries under evidence. The last CTA of a (boundary, sample) then orders its live
//   columns by their terms, most first (`perm`): a warp's lanes sum about as many (a column of many predecessors next
//   to columns of few kept 3 of 4 threads waiting, without evidence) -- unless the sample's lanes are few;
//   `ibtd_kernel`, a CTA per (node tile's children, sample) -- or per 8 samples, a warp each, when a sample's lanes are
//   few --: stages the children's parents' lanes of the sample and the parents' class masses (or an observed token's
//   log-probability) in shared memory, children minor, then a thread per (lane, G children) sums the lane's terms for
//   each of them -- in the Triton kernel's order and with its arithmetic (`tri_exp`, `tri_log`), so the values are the
//   same bit for bit whichever thread sums them -- and writes it, or log-adds it to the value there (`add`). Parents
//   whose values the query keeps once are read by slot, every entry of the list (`once`), as the Triton kernel does.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>

#define WTILE 16           // a node tile's rows (`common.NODE_TILE`)
#define WIPT 2             // items per thread at once when the sums run over several parents
#define TW 256             // `ibtd_terms_kernel`: threads, a live column per warp
#define SORT_BINS 256      // live columns ordered by min(terms, 255)

namespace {

struct WIdx {              // an index table of either width
    const void* p;
    int wide;
    __device__ __forceinline__ long long operator[](long long i) const {
        return wide ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
    }
};

WIdx widx(const torch::Tensor& t) {
    TORCH_CHECK(t.scalar_type() == torch::kInt64 || t.scalar_type() == torch::kInt32, "index tables: int32 or int64");
    return WIdx{t.data_ptr(), t.scalar_type() == torch::kInt64 ? 1 : 0};
}

const int* i32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.is_contiguous(), name, ": int32, contiguous");
    return t.data_ptr<int>();
}

// Triton's `tl.exp` on fp32: ex2.approx of x log2(e)
__device__ __forceinline__ float tri_exp(float x) {
    float r;
    asm("{\n\t.reg .f32 s;\n\tmul.f32 s, %1, 0f3FB8AA3B;\n\tex2.approx.f32 %0, s;\n\t}" : "=f"(r) : "f"(x));
    return r;
}

// Triton's `tl.log` on fp32: libdevice's `__nv_logf` (ftz), as Triton inlines it
__device__ __forceinline__ float tri_log(float a) {
    float r;
    asm("{\n\t"
        ".reg .pred p1, p2, p3;\n\t"
        ".reg .b32 x, j, k, m, e, adj, f, q, w;\n\t"
        "setp.lt.f32 p1, %1, 0f00800000;\n\t"
        "mul.f32 x, %1, 0f4B000000;\n\t"
        "selp.f32 x, x, %1, p1;\n\t"
        "selp.f32 adj, 0fC1B80000, 0f00000000, p1;\n\t"
        "add.s32 j, x, -1059760811;\n\t"
        "and.b32 k, j, -8388608;\n\t"
        "sub.s32 m, x, k;\n\t"
        "cvt.rn.f32.s32 e, k;\n\t"
        "fma.rn.ftz.f32 e, e, 0f34000000, adj;\n\t"
        "add.f32 f, m, 0fBF800000;\n\t"
        "fma.rn.ftz.f32 q, 0fBE055027, f, 0f3E1039F6;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBDF8CDCC;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0f3E0F2955;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBE2AD8B9;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0f3E4CED0B;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBE7FFF22;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0f3EAAAA78;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBF000000;\n\t"
        "mul.f32 q, f, q;\n\t"
        "fma.rn.ftz.f32 q, q, f, f;\n\t"
        "fma.rn.ftz.f32 w, e, 0f3F317218, q;\n\t"
        "setp.lt.u32 p2, x, 2139095040;\n\t"
        "@!p2 fma.rn.ftz.f32 w, x, 0f7F800000, 0f7F800000;\n\t"
        "setp.eq.f32 p3, x, 0f00000000;\n\t"
        "selp.f32 %0, 0fFF800000, w, p3;\n\t"
        "}" : "=f"(r) : "f"(a));
    return r;
}

// `prod._lse_step_one_exp`, as Triton compiles it
__device__ __forceinline__ void lse_one_exp(float& m, float& acc, float v) {
    float e = tri_exp(0.0f - fabsf(v - m));
    float up;
    asm("fma.rn.f32 %0, %1, %2, 0f3F800000;" : "=f"(up) : "f"(acc), "f"(e));
    float next = v > m ? up : acc + e;
    acc = v == -CUDART_INF_F ? acc : next;
    m = fmaxf(m, v);
}

// `common._logaddexp_tile`
__device__ __forceinline__ float log_add(float a, float b) {
    float hi = fmaxf(a, b);
    float s = hi == -CUDART_INF_F ? 0.0f : hi;
    return hi == -CUDART_INF_F ? -CUDART_INF_F : tri_log(tri_exp(a - s) + tri_exp(b - s)) + s;
}

// A class's CODE at a boundary (`kernels/transitions.py`, EnumeratedTokens): a stored class's column of the class-mass
// table, or -1 - e for the e-th enumerated token live at the boundary (its mass its token's own probability). The
// staged masses of a child run over the Cs stored columns, then the boundary's enumerated tokens: code x sits at
// `x >= 0 ? x : Cs + (-1 - x)`, which is what a packed term carries.
__device__ __forceinline__ int code_index(int x, int Cs) {
    return x >= 0 ? x : Cs + (-1 - x);
}

struct TermArgs {
    const int* tlist;      // [T]: the launch's boundaries t
    WIdx ecum;             // [T + 1]: the entries of the boundaries before, of one sample's copy
    const int* pred_ptr;   // [n, W + 1]
    const int* pred_col;
    const int* pred_x;     // every entry's class code
    const int* code;       // [n, C]: every class's code at every boundary
    WIdx obs;              // [B, n]: observed classes, -1 where missing
    const int* ccount;     // [B * (n + 1)]
    WIdx cstart;
    const int* ccolumns;
    const int* crank;      // [B, n + 1, W]
    int* terms;
    int* cnt;              // [T, B, W]: by live column's place
    int* perm;             // [T, B, W]: the live columns' places, most terms first (none: not ordered)
    int* done;             // [T, B]: CTAs done (0 between launches)
    int B, n, W, C, Cs, xb;   // xb: a term's mass-index bits
};

__global__ void __launch_bounds__(TW) ibtd_terms_kernel(TermArgs a) {
    __shared__ int bins[SORT_BINS];
    __shared__ int last;
    const int ti = blockIdx.z, b = blockIdx.y, n = a.n, W = a.W;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t = a.tlist[ti];
    const long long row_q = (long long)b * (n + 1) + t + 1, sb = (long long)ti * a.B + b;
    const int cq = a.ccount[row_q];
    const int rq = blockIdx.x * (TW / 32) + warp;
    if (rq < cq) {
        const int q = a.ccolumns[a.cstart[row_q] + rq];
        const int* pp = a.pred_ptr + (long long)t * (W + 1);
        const int e0 = pp[q], e1 = pp[q + 1];
        const long long E = a.ecum[ti + 1] - a.ecum[ti];
        int* out = a.terms + a.ecum[ti] * a.B + (long long)b * E + (e0 - pp[0]);
        const int oc = (int)a.obs[(long long)b * n + t];
        const int code_oc = oc >= 0 ? a.code[(long long)t * a.C + oc] : -1;   // the observed class's code
        const int* rank = a.crank + ((long long)b * (n + 1) + t) * W;
        int kept = 0;
        for (int base = e0; base < e1; base += 32) {
            const int e = base + lane;
            bool ok = false;
            int pk = 0;
            if (e < e1) {
                const int x = a.pred_x[e];
                if (oc < 0 || x == code_oc) {
                    const int ri = rank[a.pred_col[e]];
                    ok = ri >= 0;
                    pk = (int)(((unsigned)ri << a.xb) | (unsigned)code_index(x, a.Cs));
                }
            }
            const unsigned m = __ballot_sync(0xffffffffu, ok);
            if (ok) out[kept + __popc(m & ((1u << lane) - 1u))] = pk;
            kept += __popc(m);
        }
        if (lane == 0) a.cnt[sb * W + rq] = kept;
    }
    // the last CTA of the (boundary, sample): its live columns by their terms, most first
    if (a.perm == nullptr) return;
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(a.done + sb, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (tid == 0) a.done[sb] = 0;
    for (int i = tid; i < SORT_BINS; i += TW) bins[i] = 0;
    __syncthreads();
    const int* cnt = a.cnt + sb * W;
    for (int r = tid; r < cq; r += TW) atomicAdd(&bins[SORT_BINS - 1 - min(__ldcg(cnt + r), SORT_BINS - 1)], 1);
    __syncthreads();
    if (warp == 0) {                                                      // the bins' firsts: an exclusive scan
        int v[SORT_BINS / 32], sum = 0;
#pragma unroll
        for (int k = 0; k < SORT_BINS / 32; ++k) { v[k] = bins[lane * (SORT_BINS / 32) + k]; sum += v[k]; }
        int incl = sum;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= off) incl += y;
        }
        int run = incl - sum;
#pragma unroll
        for (int k = 0; k < SORT_BINS / 32; ++k) { bins[lane * (SORT_BINS / 32) + k] = run; run += v[k]; }
    }
    __syncthreads();
    int* perm = a.perm + sb * W;
    for (int r = tid; r < cq; r += TW)
        perm[atomicAdd(&bins[SORT_BINS - 1 - min(__ldcg(cnt + r), SORT_BINS - 1)], 1)] = r;
}

struct IbArgs {
    const float* element_flows;
    float* temp_flows;
    float* node_flows;
    const float* node_mars;
    const float* class_mars;
    WIdx obs;
    const int* mass_row;
    const int* mass_stride;
    const int* pred_ptr;
    const int* pred_col;
    const int* pred_x;     // every entry's class code
    const int* code;       // [n, C]: every class's code at every boundary
    const int* tok_ptr;    // [n + 1]: the enumerated tokens live at boundary t are tok[tok_ptr[t] .. tok_ptr[t + 1]]
    const int* tok;
    const float* tokmaj;   // [V, num_rows]: every token's probability under every parameter row
    const int* prow;       // [rows of class_mars]: every row's parameter row
    const float* lognorm;  // [rows of class_mars]: every row's log Z (per-sample masses)
    const float* enum_logw;   // [B or 1, enumerated tokens + 1]: every enumerated token's log weight per sample
    const int* width;
    const int* iv_of;
    WIdx iv_info;
    const int* iv_table;
    const int* once_iv;
    WIdx coff;
    const int* ccount;
    WIdx cstart;
    const int* ccolumns;
    const int* children;   // [children, 4]: row, region, add, -
    const int* tiles;      // [tiles * 16]
    const int* ptr;
    const int* parents;    // [parents, 7]: out row, input node, t, child row, child region, t2, -
    WIdx reg_offset, reg_width, reg_first;
    const int* t2i;        // [n]: a boundary's place in the launch's (`ibtd_terms_kernel`)
    WIdx ecum;
    const int* terms;
    const int* cnt;
    const int* perm;
    long long out_first, out_width, temp_width;
    long long team_floats;   // a team's shared memory (floats, a multiple of 4)
    int B, n, W, C, Cs, mw, num_rows, logw_stride, input_start, subs, tn, lp_ld, xb;   // mw: staged masses a child
    int sample, logw, c_kind, out_temp, c_full, out_full;
};

template <int G>
__device__ __forceinline__ void load_g(const float* p, float* v) {       // G consecutive floats (16 B aligned at 4)
    if constexpr (G == 4) {
        const float4 f = *reinterpret_cast<const float4*>(p);
        v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
    } else {
#pragma unroll
        for (int c = 0; c < G; ++c) v[c] = p[c];
    }
}

#define WTEAMS 8           // samples a CTA of warp teams takes (256 threads)

// A CTA per (TN children of a node tile, sample), or with `WARPS` per (TN children, 8 samples), a warp each (a sample's
// lanes few: one CTA each would leave most threads idle); a thread per (lane, G of the children) item
template <int G, bool WARPS>
__global__ void __launch_bounds__(256) ibtd_kernel(IbArgs a) {
    extern __shared__ float sm[];
    __shared__ int s_c[WTILE];                 // the CTA's children (-1: none)
    __shared__ long long s_dst[WTILE];         // where a child's row starts (`node_flows` / `temp_flows`)
    __shared__ long long s_op[WARPS ? WTEAMS : 1][WTILE];     // where its parent's row starts (a team's parent r)
    __shared__ long long s_mrow[WARPS ? WTEAMS : 1][WTILE];   // its parent's class masses (the team's sample's)
    const int n = a.n, B = a.B, W = a.W, Cs = a.Cs, MW = a.mw, TN = a.tn;
    const int team = WARPS ? threadIdx.x >> 5 : 0;                        // a team sums one sample
    const int tid = WARPS ? threadIdx.x & 31 : threadIdx.x, nt = WARPS ? 32 : blockDim.x;
    const int tile = blockIdx.x / a.subs, sub = blockIdx.x - tile * a.subs;
    const int b = WARPS ? blockIdx.y * (blockDim.x >> 5) + team : blockIdx.y;
    const int c0 = a.tiles[(long long)tile * WTILE];                   // the tile's first: its interval, parents, add
    const int f0 = a.ptr[c0], R = a.ptr[c0 + 1] - f0;
    const int t = a.parents[(long long)f0 * 7 + 2], t2 = a.parents[(long long)f0 * 7 + 5];
    const int add = a.children[(long long)c0 * 4 + 2];
    const int kc = a.iv_of[(t + 1) * (n + 1) + t2], kp = a.iv_of[t * (n + 1) + t2];
    if (threadIdx.x < TN) {
        const int c = a.tiles[(long long)tile * WTILE + sub * TN + threadIdx.x];
        s_c[threadIdx.x] = c;
        if (c >= 0) {
            const long long row = a.children[(long long)c * 4], reg = a.children[(long long)c * 4 + 1];
            s_dst[threadIdx.x] = a.c_kind == 0 ? a.reg_offset[reg] + (row - a.reg_first[reg]) * a.reg_width[reg]
                                               : row * a.temp_width;
        }
    }
    __syncthreads();                                                      // (the last barrier of the whole CTA)
    if (b >= B) return;
    const long long lane0 = a.coff[(long long)kc * (B + 1) + b];
    const long long lc = a.coff[(long long)kc * (B + 1) + b + 1] - lane0;   // the sample's lanes of a child's row
    if (lc <= 0) return;
    const long long base_p = a.coff[(long long)kp * (B + 1) + b];
    const long long row_t = (long long)b * (n + 1) + t, row_e = (long long)b * (n + 1) + t2;
    const int ci = a.ccount[row_t], ce = a.ccount[row_e];
    const long long st_t = a.cstart[row_t], st_q = a.cstart[row_t + 1], st_j = a.cstart[row_e];
    const int lp = ci * ce;
    const int X = t2 == n ? 1 : a.width[t2];
    const long long cpos = a.c_full ? -1 : a.iv_info[(long long)kc * 3 + 1];
    const long long opos = a.out_full ? -1 : a.iv_info[(long long)kp * 3 + 1];
    const bool once = a.once_iv[kp] != 0;
    const int oc = (int)a.obs[(long long)b * n + t];
    const int* pp = a.pred_ptr + (long long)t * (W + 1);
    const int ti = a.t2i[t];
    const long long sb = (long long)ti * B + b;
    const int* terms = a.terms + a.ecum[ti] * B + (long long)b * (a.ecum[ti + 1] - a.ecum[ti]) - pp[0];
    const int* cnt = a.cnt + sb * W;
    const int* perm = a.perm != nullptr ? a.perm + sb * W : nullptr;
    const float* src = a.out_temp ? a.temp_flows : a.element_flows;
    float* O = sm + (long long)team * a.team_floats;                     // [lp_ld][TN]: the parents' lanes
    float* M = O + (long long)a.lp_ld * TN;                               // [MW][TN]: their masses, by code index
    float* own = M + (long long)MW * TN;                                  // [TN]: an observed token's log-prob
    long long* op_ = s_op[team];
    long long* mrow = s_mrow[team];
    const int tk = a.tok_ptr[t], ecount = a.tok_ptr[t + 1] - tk;          // the boundary's enumerated tokens
    const int code_oc = oc >= 0 ? a.code[(long long)t * a.C + oc] : -1;
    auto sync = [&]() {
        if constexpr (WARPS) __syncwarp();
        else __syncthreads();
    };

    // the children's parents r: their rows, masses and lanes of the sample in shared memory (-inf for no child)
    auto stage = [&](int r) {
        if (tid < TN) {
            const int c = s_c[tid];
            own[tid] = -CUDART_INF_F;
            if (c >= 0) {
                const long long p = (long long)a.ptr[c] + r;
                const long long row = a.parents[p * 7], u = a.parents[p * 7 + 1];
                op_[tid] = a.out_temp ? row * a.temp_width : (row - a.out_first) * a.out_width;
                const long long k = u - a.input_start;
                mrow[tid] = (long long)a.mass_row[k] + (a.sample ? (long long)b * a.mass_stride[k] : 0);
                if (oc >= 0) own[tid] = a.node_mars[u * B + b];
            }
        }
        sync();
        if (!once)
            for (int e = tid; e < TN * lp; e += nt) {
                const int k = e / lp, l = e - k * lp;
                O[(long long)l * TN + k] = s_c[k] >= 0 ? src[op_[k] + base_p + l] : -CUDART_INF_F;
            }
        if (oc < 0)
            for (int e = tid; e < TN * MW; e += nt) {                    // stored columns, then enumerated tokens
                const int k = e / MW, x = e - k * MW;
                float v = -CUDART_INF_F;
                if (s_c[k] >= 0) {
                    if (x < Cs) {
                        v = a.class_mars[mrow[k] * Cs + x];
                    } else if (x - Cs < ecount) {                        // (`_entry_mass`'s arithmetic, in its order)
                        const int ei = x - Cs;
                        v = tri_log(a.tokmaj[(long long)a.tok[tk + ei] * a.num_rows + a.prow[mrow[k]]]);
                        if (a.logw) v = v + a.enum_logw[(long long)b * a.logw_stride + tk + ei] - a.lognorm[mrow[k]];
                    }
                }
                M[x * TN + k] = v;
            }
        sync();
    };

    // item e: its lane (the sample's live columns at t + 1 in `perm`'s order, then the exit's) and children
    const int NG = TN / G;
    auto item = [&](long long e, int& l, int& q, int& j, int& k0) {
        const int g = (int)(e % NG);
        const long long rest = e / NG;
        const int rj = (int)(rest % ce), rq = perm != nullptr ? perm[rest / ce] : (int)(rest / ce);
        l = rq * ce + rj;
        q = a.ccolumns[st_q + rq];
        j = a.ccolumns[st_j + rj];
        k0 = g * G;
    };

    // an item's terms of parent r (the staged one), for each of its G children
    auto accumulate = [&](long long e, float* mx, float* acc) {
        int l, q, j, k0;
        item(e, l, q, j, k0);
        const int rq = l / ce, rj = l - rq * ce;
        if (cpos >= 0 && a.iv_table[cpos + (long long)q * X + j] < 0) return;     // a hole: -inf
        float ov[G], mv[G];
        if (oc >= 0) load_g<G>(own + k0, mv);
        if (!once) {
            const float* ob = O + (long long)rj * TN + k0;
            const int* ent = terms + pp[q];
            const int m = cnt[rq];
            const int xmask = (1 << a.xb) - 1;
            for (int s = 0; s < m; ++s) {
                const int pk = ent[s], ri = (int)((unsigned)pk >> a.xb), x = pk & xmask;
                if (opos >= 0 && a.iv_table[opos + (long long)a.ccolumns[st_t + ri] * X + j] < 0) continue;
                load_g<G>(ob + (long long)ri * ce * TN, ov);
                if (oc < 0) load_g<G>(M + x * TN + k0, mv);
#pragma unroll
                for (int c = 0; c < G; ++c) lse_one_exp(mx[c], acc[c], mv[c] + ov[c]);
            }
        } else {                                                          // the one block kept once, by slot
            for (int s = pp[q]; s < pp[q + 1]; ++s) {
                const int i = a.pred_col[s], x = a.pred_x[s];
                if (oc >= 0 && x != code_oc) continue;
                long long sl = (long long)i * X + j;
                if (opos >= 0) sl = a.iv_table[opos + sl];
                if (sl < 0) continue;
                if (oc < 0) load_g<G>(M + code_index(x, Cs) * TN + k0, mv);
#pragma unroll
                for (int c = 0; c < G; ++c)
                    if (s_c[k0 + c] >= 0) lse_one_exp(mx[c], acc[c], mv[c] + src[op_[k0 + c] + sl]);
            }
        }
    };

    auto store = [&](long long e, const float* mx, const float* acc) {                // (a hole: no term, -inf)
        int l, q, j, k0;
        item(e, l, q, j, k0);
#pragma unroll
        for (int c = 0; c < G; ++c) {
            if (s_c[k0 + c] < 0) continue;
            float val = tri_log(acc[c]) + (mx[c] == -CUDART_INF_F ? 0.0f : mx[c]);
            float* dst = (a.c_kind == 0 ? a.node_flows : a.temp_flows) + s_dst[k0 + c] + lane0 + l;
            if (add) val = log_add(*dst, val);
            *dst = val;
        }
    };

    const long long items = lc * NG;
    if (R == 1) {
        stage(0);
        for (long long e = tid; e < items; e += nt) {
            float mx[G], acc[G];
#pragma unroll
            for (int c = 0; c < G; ++c) { mx[c] = -CUDART_INF_F; acc[c] = 0.0f; }
            accumulate(e, mx, acc);
            store(e, mx, acc);
        }
        return;
    }
    for (long long e0 = 0; e0 < items; e0 += (long long)nt * WIPT) {
        float mx[WIPT][G], acc[WIPT][G];
#pragma unroll
        for (int s = 0; s < WIPT; ++s)
#pragma unroll
            for (int c = 0; c < G; ++c) { mx[s][c] = -CUDART_INF_F; acc[s][c] = 0.0f; }
        for (int r = 0; r < R; ++r) {
            sync();                                                       // (the previous parent's values read)
            stage(r);
#pragma unroll
            for (int s = 0; s < WIPT; ++s) {
                const long long e = e0 + (long long)s * nt + tid;
                if (e < items) accumulate(e, mx[s], acc[s]);
            }
        }
#pragma unroll
        for (int s = 0; s < WIPT; ++s) {
            const long long e = e0 + (long long)s * nt + tid;
            if (e < items) store(e, mx[s], acc[s]);
        }
    }
}

}  // namespace

// Shared memory of a CTA of `tn` children: their parents' lanes (`lp_ld` each), masses (`mw`: the stored columns and a
// boundary's most enumerated tokens), observed log-probabilities (`prod_top_down._input_block_cuda` sizes the CTAs by
// the same rule)
static int64_t ibtd_smem(int64_t tn, int64_t lp_ld, int64_t mw) {
    return (4 * tn * (lp_ld + mw + 1) + 15) / 16 * 16;
}

// The most dynamic shared memory a CTA of `ibtd_kernel<G, WARPS>` may take (the device's opt-in less the kernel's
// static), the attribute set once
template <int G, bool WARPS>
static int64_t ibtd_allowed() {
    static const int64_t most = [] {
        const void* f = (const void*)ibtd_kernel<G, WARPS>;
        int dev = 0, optin = 0;
        cudaFuncAttributes fa;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, f));
        const int dyn = optin - (int)fa.sharedSizeBytes;
        C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
        return (int64_t)dyn;
    }();
    return most;
}

// The most dynamic shared memory of a CTA of one child (a sample's CTA, not warp teams): `_input_block_cuda`'s limit
// for the CTAs of parent rows too wide for 45 KB
int64_t ibtd_smem_max() {
    return ibtd_allowed<1, false>();
}

// One launch: `tiles` (int32, 16 per node tile) with `subs` CTAs of `tn` children per tile (`threads` each), every
// sample (`warps`: 8 a CTA, a warp each); `tlist` / `t2i` / `ecum` the launch's boundaries (see TermArgs), `terms` /
// `cnt` / `perm` / `done` the work space (B x ecum[T], T x B x W, T x B x W, T x B int32; `done` zero), `sort`: the
// live columns summed in `perm`'s order (or in their own); `cq_max` a sample's most live columns at t + 1 over them,
// `lp_ld` at least a sample's lanes of a parent's row
void ibtd_cuda(torch::Tensor element_flows, torch::Tensor temp_flows, torch::Tensor node_flows,
               torch::Tensor node_mars, torch::Tensor obs_class, torch::Tensor class_mars, torch::Tensor mass_row,
               torch::Tensor mass_stride, torch::Tensor pred_ptr, torch::Tensor pred_col, torch::Tensor pred_x,
               torch::Tensor width, torch::Tensor iv_of, torch::Tensor iv_info, torch::Tensor iv_table,
               torch::Tensor once_iv, torch::Tensor coff, torch::Tensor ccount, torch::Tensor cstart,
               torch::Tensor ccolumns, torch::Tensor crank, torch::Tensor children, torch::Tensor tiles,
               torch::Tensor ptr, torch::Tensor parents, torch::Tensor reg_offset, torch::Tensor reg_width,
               torch::Tensor reg_first, torch::Tensor tlist, torch::Tensor t2i, torch::Tensor ecum,
               torch::Tensor terms, torch::Tensor cnt, torch::Tensor perm, torch::Tensor done, int64_t out_first,
               int64_t out_width, int64_t temp_width, int64_t B, int64_t n, int64_t W, int64_t C,
               int64_t input_start, int64_t subs, int64_t tn, int64_t threads, bool warps, bool sort,
               int64_t lp_ld, int64_t cq_max, bool sample, int64_t c_kind, bool out_temp, bool c_full,
               bool out_full, torch::Tensor code, torch::Tensor tok_ptr, torch::Tensor tok, torch::Tensor tokmaj,
               torch::Tensor prow, torch::Tensor lognorm, torch::Tensor enum_logw, int64_t Cs, int64_t mw,
               int64_t num_rows, int64_t logw_stride, bool logw) {
    TORCH_CHECK(node_flows.scalar_type() == torch::kFloat32 && element_flows.scalar_type() == torch::kFloat32 &&
                temp_flows.scalar_type() == torch::kFloat32 && node_mars.scalar_type() == torch::kFloat32 &&
                class_mars.scalar_type() == torch::kFloat32 && tokmaj.scalar_type() == torch::kFloat32 &&
                lognorm.scalar_type() == torch::kFloat32 && enum_logw.scalar_type() == torch::kFloat32,
                "ibtd_cuda: fp32 values");
    TORCH_CHECK(tn >= 1 && tn <= WTILE && subs * tn <= WTILE && (tn < 4 || tn % 4 == 0),
                "ibtd_cuda: 1, 2 or a multiple of 4 children a CTA, at most 16 a tile");
    TORCH_CHECK(threads == 128 || threads == 256, "ibtd_cuda: 128 or 256 threads");
    TORCH_CHECK(!warps || threads / 32 <= WTEAMS, "ibtd_cuda: at most 8 warp teams");
    TORCH_CHECK(mw >= Cs && mw >= 1, "ibtd_cuda: the staged masses hold every stored column");
    int xb = 1;                                                     // a term's mass-index bits, its place the rest
    while ((1LL << xb) < mw) ++xb;
    TORCH_CHECK(xb < 32 && W < (1LL << (32 - xb)), "ibtd_cuda: a term packs (place, mass index) in 32 bits");
    const int64_t ntiles = tiles.numel() / WTILE;
    const int T = (int)tlist.numel();
    if (ntiles == 0 || B == 0 || T == 0 || cq_max == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    TermArgs ta{i32(tlist, "tlist"), widx(ecum), i32(pred_ptr, "pred_ptr"), i32(pred_col, "pred_col"),
                i32(pred_x, "pred_x"), i32(code, "code"), widx(obs_class), i32(ccount, "ccount"), widx(cstart),
                i32(ccolumns, "ccolumns"), i32(crank, "crank"), terms.data_ptr<int>(), cnt.data_ptr<int>(),
                sort ? perm.data_ptr<int>() : nullptr, done.data_ptr<int>(), (int)B, (int)n, (int)W, (int)C, (int)Cs,
                xb};
    ibtd_terms_kernel<<<dim3((unsigned)((cq_max + TW / 32 - 1) / (TW / 32)), (unsigned)B, (unsigned)T), TW, 0,
                        stream>>>(ta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    IbArgs a;
    a.element_flows = element_flows.data_ptr<float>();
    a.temp_flows = temp_flows.data_ptr<float>();
    a.node_flows = node_flows.data_ptr<float>();
    a.node_mars = node_mars.data_ptr<float>();
    a.class_mars = class_mars.data_ptr<float>();
    a.obs = widx(obs_class);
    a.mass_row = i32(mass_row, "mass_row");
    a.mass_stride = i32(mass_stride, "mass_stride");
    a.pred_ptr = i32(pred_ptr, "pred_ptr");
    a.pred_col = i32(pred_col, "pred_col");
    a.pred_x = i32(pred_x, "pred_x");
    a.code = i32(code, "code");
    a.tok_ptr = i32(tok_ptr, "tok_ptr");
    a.tok = i32(tok, "tok");
    a.tokmaj = tokmaj.data_ptr<float>();
    a.prow = i32(prow, "prow");
    a.lognorm = lognorm.data_ptr<float>();
    a.enum_logw = enum_logw.data_ptr<float>();
    a.width = i32(width, "width");
    a.iv_of = i32(iv_of, "iv_of");
    a.iv_info = widx(iv_info);
    a.iv_table = i32(iv_table, "iv_table");
    a.once_iv = i32(once_iv, "once_iv");
    a.coff = widx(coff);
    a.ccount = i32(ccount, "ccount");
    a.cstart = widx(cstart);
    a.ccolumns = i32(ccolumns, "ccolumns");
    a.children = i32(children, "children");
    a.tiles = i32(tiles, "tiles");
    a.ptr = i32(ptr, "ptr");
    a.parents = i32(parents, "parents");
    a.reg_offset = widx(reg_offset);
    a.reg_width = widx(reg_width);
    a.reg_first = widx(reg_first);
    a.t2i = i32(t2i, "t2i");
    a.ecum = widx(ecum);
    a.terms = terms.data_ptr<int>();
    a.cnt = cnt.data_ptr<int>();
    a.perm = sort ? perm.data_ptr<int>() : nullptr;
    a.out_first = out_first;
    a.out_width = out_width;
    a.temp_width = temp_width;
    a.B = (int)B;
    a.n = (int)n;
    a.W = (int)W;
    a.C = (int)C;
    a.Cs = (int)Cs;
    a.mw = (int)mw;
    a.num_rows = (int)num_rows;
    a.logw_stride = (int)logw_stride;
    a.input_start = (int)input_start;
    a.subs = (int)subs;
    a.tn = (int)tn;
    a.lp_ld = (int)lp_ld;
    a.xb = xb;
    a.sample = sample;
    a.logw = logw;
    a.c_kind = (int)c_kind;
    a.out_temp = out_temp;
    a.c_full = c_full;
    a.out_full = out_full;
    a.team_floats = ibtd_smem(tn, lp_ld, mw) / 4;
    const int64_t smem = 4 * a.team_floats * (warps ? threads / 32 : 1);
    const dim3 grid((unsigned)(ntiles * subs), (unsigned)(warps ? (B + threads / 32 - 1) / (threads / 32) : B));
#define IBTD_LAUNCH(GG, WW) \
    do { \
        TORCH_CHECK(smem <= 45 * 1024 || smem <= (ibtd_allowed<GG, WW>()), "ibtd_cuda: ", smem, \
                    " bytes of dynamic shared memory, more than the device takes"); \
        ibtd_kernel<GG, WW><<<grid, (unsigned)threads, (size_t)smem, stream>>>(a); \
    } while (0)
    if (warps) {
        if (tn % 4 == 0) IBTD_LAUNCH(4, true);
        else if (tn == 2) IBTD_LAUNCH(2, true);
        else IBTD_LAUNCH(1, true);
    } else {
        if (tn % 4 == 0) IBTD_LAUNCH(4, false);
        else if (tn == 2) IBTD_LAUNCH(2, false);
        else IBTD_LAUNCH(1, false);
    }
#undef IBTD_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
