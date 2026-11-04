// The class weights of `input @ block` steps with one exit column and one step a destination, on compact values
// under evidence pruning (`kernels/score.py`, `_class_weight_kernel`'s class walk), as CUDA kernels: destination
// (position t, column k) of sample b and class x gets, stored to (or log-added into) its row of `dfa_class_flow`,
//     log sum over the columns i at t where x leads to q, i live for the sample, of  exp(O_out[i, 0] + R[q, 0])
// -- the Triton kernel's terms in its order (the class's columns ascending) with its arithmetic
// (`prod._lse_step_one_exp`, `tl.log(acc * tl.exp(m - shift)) + shift`), so the weights are its own bit for bit.
//
// MEASURED (ncu, HMM-4096 x hard_0 B8, a 1.5 ms launch): the Triton kernel is bound by L2-to-SM sectors (77% of the
// L2's peak, 5 of every 32 bytes used): its lanes walk different class lists (a warp's list load touches 32 sectors) and
// gather O_out and R per destination from global memory, ~44 instructions a term. Here:
//   * `cw_prep_kernel` (a warp per (position, sample, class lane)): the lane's class list at t, its entries with i
//     live and a successor, packed (i | q << 16) and interleaved by 32 lanes -- a warp's entry is one 128-byte line;
//   * `cw_stage_kernel` (a CTA per (TN destinations of a node tile, sample)): the destinations' O_out over the columns
//     at t and R over the columns at t + 1 staged in shared memory once, destination-minor (`Os[i][d]`, `Rs[q][d]`: one
//     vector load serves the TN destinations), then the CTA's warps walk their class lanes' entries independently --
//     no barrier after the staging (a CTA-wide barrier between a class's entries is what made a thread-per-class kernel
//     wait for the CTA's longest list).
// Values kept once (the outputs' interval, a block operand's region) are staged by slot, the others by compact lane, as
// `_parent_lane` reads them.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>

#include "tri_math.cuh"

#define CW_TILE 16         // a node tile's rows (`common.NODE_TILE`)

namespace {

using namespace tri;

struct PrepArgs {
    const int* tlist;                            // [T]: the launch's positions
    WIdx cw_col;                                 // per position: its first target column
    const int* cw_count;                         // per column: its live classes ...
    const int* cw_cls;                           // ... in order [N, C]
    const int* cls_ptr;                          // [n, C + 1]
    const int* cls_col;
    const int* cls_succ;
    const int8_t* live;                          // [B, n + 1, W]
    unsigned* tab;                               // [T, B, lanes / 32, maxl, 32]
    int* cnt;                                    // [T, B, lanes]
    int B, n, W, C, lanes, maxl;
};

__global__ void __launch_bounds__(128) cw_prep_kernel(PrepArgs a) {
    // a warp per (position, sample, class lane): the lane's list 32 entries at a time, the kept ones placed in order by
    // a ballot's prefix count (a thread a lane walked its list one dependent load after another: 43 us a launch)
    const int ti = blockIdx.z, b = blockIdx.y, lid = threadIdx.x & 31;
    const int lane = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (lane >= a.lanes) return;
    const int t = a.tlist[ti];
    const long long tb = (long long)ti * a.B + b;
    const long long nb = a.cw_col[t] + b;
    int k = 0;
    if (lane < a.cw_count[nb]) {
        const int cls = a.cw_cls[nb * a.C + lane];
        const int* p = a.cls_ptr + (long long)t * (a.C + 1);
        const int e0 = p[cls], e1 = p[cls + 1];
        const int8_t* lv = a.live + ((long long)b * (a.n + 1) + t) * a.W;
        unsigned* out = a.tab + (tb * (a.lanes / 32) + lane / 32) * (long long)a.maxl * 32 + (lane & 31);
        for (int e = e0; e < e1; e += 32) {
            const int ee = e + lid;
            int i = 0, q = -1;
            if (ee < e1) { i = a.cls_col[ee]; q = a.cls_succ[ee]; }
            const bool keep = ee < e1 && q >= 0 && lv[i] != 0;          // (no term: -inf in the Triton kernel)
            const unsigned m = __ballot_sync(0xffffffffu, keep);
            if (keep) out[(long long)(k + __popc(m & ((1u << lid) - 1))) * 32] = (unsigned)i | ((unsigned)q << 16);
            k += __popc(m);
        }
    }
    if (lid == 0) a.cnt[tb * a.lanes + lane] = k;
}

struct StageArgs {
    float* cw;                                   // `dfa_class_flow`
    WIdx cw_col, cw_K, cw_off;
    const int* cw_count;
    const unsigned* tab;
    const int* cnt;
    const int* t2i;                              // [n + 1]: a position's index in the launch's list
    const float* element_flows;
    const float* temp_flows;
    const float* node_mars;
    const float* prior_mars;
    const float* temp;
    WIdx width, iv_of, iv_info, iv_table, once_iv;
    WIdx count, start;                           // the liveness lists
    const int* columns;
    const int* crank;                            // [B, n + 1, W]
    WIdx coff;                                   // [intervals, B + 1]
    const int* dests;                            // [D, 2]: position, column
    const int* tiles;                            // [tiles, 16]
    const int* ptr;                              // [D + 1]
    const int* parents;                          // [P, 7]
    WIdx reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior;
    long long out_first, out_width, temp_width, tflow_width;
    int B, n, W, C, subs, lanes, maxl, s_kind, out_temp, out_full, s_full, mars, store, wt_max;
};

template <int TN> struct Vec;
template <> struct Vec<1> { using T = float; __device__ static float at(const T& v, int) { return v; } };
template <> struct Vec<2> { using T = float2; __device__ static float at(const T& v, int d) { return d ? v.y : v.x; } };
template <> struct Vec<4> {
    using T = float4;
    __device__ static float at(const T& v, int d) { return d == 0 ? v.x : d == 1 ? v.y : d == 2 ? v.z : v.w; }
};

template <int TN, int NT>
__global__ void __launch_bounds__(NT) cw_stage_kernel(StageArgs a) {
    extern __shared__ __align__(16) float smem[];
    __shared__ int s_k[TN], s_real[TN], s_own[TN];
    __shared__ const float* s_op[TN];
    __shared__ const float* s_rp[TN];
    using V = typename Vec<TN>::T;
    const int tid = threadIdx.x, n = a.n, B = a.B, W = a.W;
    const int tile = blockIdx.x / a.subs, sub = blockIdx.x - tile * a.subs, b = blockIdx.y;
    const int d0 = a.tiles[(long long)tile * CW_TILE];
    const int t = a.dests[(long long)d0 * 2];
    const int f0 = a.ptr[d0];
    const int t2 = a.parents[(long long)f0 * 7 + 5];                   // (one step a destination: the host checks)
    const int Wt1 = (int)a.width[t + 1];
    float* Os = smem;                                                   // [width at t][TN]
    float* Rs = smem + (long long)a.wt_max * TN;                        // [width at t + 1][TN]
    const long long kp = a.iv_of[(long long)t * (n + 1) + t2], kr = a.iv_of[(long long)(t + 1) * (n + 1) + t2];
    const bool once = a.once_iv[kp] != 0;                               // the outputs kept once: by slot
    const long long opos = a.out_full ? -1 : a.iv_info[kp * 3 + 1];
    const long long rpos = a.s_full ? -1 : a.iv_info[kr * 3 + 1];
    const long long RS = a.iv_info[kr * 3];
    const long long row_e = (long long)b * (n + 1) + t2;
    const long long ce = a.count[row_e];                                // the exit boundary's live columns ...
    const int rj0 = a.crank[row_e * W];                                 // ... and the exit column's rank
    const long long base_p = a.coff[kp * (B + 1) + b], base_r = a.coff[kr * (B + 1) + b];
    if (tid < TN) {
        const int d = a.tiles[(long long)tile * CW_TILE + sub * TN + tid];
        s_real[tid] = d >= 0;
        s_k[tid] = d >= 0 ? a.dests[(long long)d * 2 + 1] : 0;
        const long long p = (long long)(d >= 0 ? a.ptr[d] : f0);
        const long long orow = a.parents[p * 7];
        s_op[tid] = a.out_temp ? a.temp_flows + orow * a.tflow_width : a.element_flows + (orow - a.out_first) * a.out_width;
        const long long srow = a.parents[p * 7 + 3], reg = a.parents[p * 7 + 4];
        const float* rp;
        bool own;
        if (a.s_kind == 0) {                                            // a sum node's row: its region, or kept once
            const long long k = srow - a.reg_first[reg];
            const bool shared = a.reg_prior[reg] != 0;
            rp = shared ? a.prior_mars + a.prior_offset[reg] + k * a.prior_width[reg]
                        : a.node_mars + a.reg_offset[reg] + k * a.reg_width[reg];
            own = !shared;
        } else {                                                        // a scratch row
            rp = a.temp + srow * a.temp_width;
            own = srow >= 0;
        }
        if (!a.mars && own) rp += (long long)b * RS;                    // dense inside values: the sample's block
        s_rp[tid] = rp;
        s_own[tid] = own;
    }
    __syncthreads();
    // O_out[i, 0] over the sample's live columns at t (the walk reads no other), by slot or compact lane
    const long long row_t = (long long)b * (n + 1) + t;
    const int ct = (int)a.count[row_t];
    const long long st = a.start[row_t];
    for (int ri = tid; ri < ct; ri += NT) {
        const int i = a.columns[st + ri];
        const long long sl = a.out_full ? (long long)i : (opos < 0 ? (long long)i : a.iv_table[opos + i]);
        long long so = -1;
        if (sl >= 0) so = once ? sl : (rj0 >= 0 ? base_p + (long long)ri * ce + rj0 : -1);   // (ri: i's rank)
#pragma unroll
        for (int d = 0; d < TN; ++d) Os[(long long)i * TN + d] = (s_real[d] && so >= 0) ? s_op[d][so] : -CUDART_INF_F;
    }
    // R[q, 0] over every column at t + 1: the sample's own values by compact lane (-inf where q is dead), values kept
    // once by slot
    const int* rank_t1 = a.crank + (row_t + 1) * W;
    for (int q = tid; q < Wt1; q += NT) {
        const long long sl = a.s_full ? (long long)q : (rpos < 0 ? (long long)q : a.iv_table[rpos + q]);
        const int rq = rank_t1[q];
        const long long lane = (sl >= 0 && rq >= 0 && rj0 >= 0) ? base_r + (long long)rq * ce + rj0 : -1;
#pragma unroll
        for (int d = 0; d < TN; ++d) {
            const long long sr = sl < 0 ? -1 : (a.mars ? (s_own[d] ? lane : sl) : sl);
            Rs[(long long)q * TN + d] = (s_real[d] && sr >= 0) ? s_rp[d][sr] : -CUDART_INF_F;
        }
    }
    __syncthreads();
    // the walk: a thread a class lane at a time, its live entries from the table (a warp's 32 lanes: one line)
    const long long tb = (long long)a.t2i[t] * B + b;
    const long long nb = a.cw_col[t] + b;
    const int nlive = a.cw_count[nb];
    const unsigned* tab = a.tab + tb * (a.lanes / 32) * (long long)a.maxl * 32;
    const int* cnt = a.cnt + tb * a.lanes;
    const long long Kt = a.cw_K[t];
    float* cwrow0 = a.cw + a.cw_off[nb];
    for (int lane = tid; lane < nlive; lane += NT) {
        float mx[TN], acc[TN];
#pragma unroll
        for (int d = 0; d < TN; ++d) { mx[d] = -CUDART_INF_F; acc[d] = 0.0f; }
        const unsigned* tp = tab + (long long)(lane / 32) * a.maxl * 32 + (lane & 31);
        const int m = cnt[lane];
        int k = 0;
        for (; k + 4 <= m; k += 4) {                                    // four entries' loads in flight
            unsigned e[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) e[u] = __ldg(tp + (long long)(k + u) * 32);
            V ov[4], rv[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                ov[u] = reinterpret_cast<const V*>(Os)[e[u] & 0xffffu];
                rv[u] = reinterpret_cast<const V*>(Rs)[e[u] >> 16];
            }
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int d = 0; d < TN; ++d) lse_one_exp(mx[d], acc[d], Vec<TN>::at(ov[u], d) + Vec<TN>::at(rv[u], d));
        }
        for (; k < m; ++k) {
            const unsigned e = __ldg(tp + (long long)k * 32);
            const V ov = reinterpret_cast<const V*>(Os)[e & 0xffffu], rv = reinterpret_cast<const V*>(Rs)[e >> 16];
#pragma unroll
            for (int d = 0; d < TN; ++d) lse_one_exp(mx[d], acc[d], Vec<TN>::at(ov, d) + Vec<TN>::at(rv, d));
        }
        float* row = cwrow0 + (long long)lane * Kt;
#pragma unroll
        for (int d = 0; d < TN; ++d) {
            if (!s_real[d]) continue;
            const float sh = mx[d] == -CUDART_INF_F ? 0.0f : mx[d];
            const float val = tri_log(__fmul_rn(acc[d], tri_exp(mx[d] - sh))) + sh;
            row[s_k[d]] = a.store ? val : log_add(row[s_k[d]], val);
        }
    }
}

// The dynamic shared memory a kernel may take beyond the default 48 KB: the device's opt-in limit less its static
// shared memory
template <int TN, int NT>
size_t stage_smem_limit() {
    static size_t limit = 0;                                            // (per instantiation, once a process)
    if (limit == 0) {
        int dev = 0, optin = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        cudaFuncAttributes fa;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, cw_stage_kernel<TN, NT>));
        limit = (size_t)optin - fa.sharedSizeBytes;
    }
    return limit;
}

template <int TN, int NT>
void launch_stage(const StageArgs& a, dim3 grid, size_t smem, cudaStream_t stream) {
    static size_t raised = 48 * 1024;                                   // (per instantiation: the limit set so far)
    const size_t limit = stage_smem_limit<TN, NT>();
    TORCH_CHECK(smem <= limit, "cw_class_cuda: the rows do not fit in shared memory");
    if (smem > raised) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(cw_stage_kernel<TN, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem));
        raised = smem;
    }
    cw_stage_kernel<TN, NT><<<grid, NT, smem, stream>>>(a);
}

}  // namespace


// The shared memory a CTA of `tn` destinations takes for row widths `wt` (at t) and `wt1` (at t + 1)
int64_t cw_stage_smem(int64_t tn, int64_t wt, int64_t wt1) {
    return tn * (wt + wt1) * 4;
}

// One class-weight launch (`kernels/score.py`, `_class_weight_cuda`): `tiles` (int32, 16 a node tile) in `subs` CTAs
// of `tn` (1, 2 or 4) destinations each, every sample, `threads` a CTA (128 or 256); `tlist` / `t2i` the launch's
// positions; `tab` / `cnt` the entry tables' work space ([T, B, lanes / 32, maxl, 32] / [T, B, lanes] int32; `lanes`
// a multiple of 32, at least any column's live classes, `maxl` at least any class list at the positions); `wt_max` /
// `wt1_max` the widest boundaries at t and t + 1; `s_kind` the block operand's kind (0: a sum node's row, else a
// scratch row), `mars` whether the inside values are compact too, `store` whether the weights are stored
void cw_class_cuda(torch::Tensor cw, torch::Tensor cw_col, torch::Tensor cw_K, torch::Tensor cw_count,
                   torch::Tensor cw_cls, torch::Tensor cw_off, torch::Tensor element_flows, torch::Tensor temp_flows,
                   torch::Tensor node_mars, torch::Tensor prior_mars, torch::Tensor temp, torch::Tensor width,
                   torch::Tensor iv_of, torch::Tensor iv_info, torch::Tensor iv_table, torch::Tensor once_iv,
                   torch::Tensor cls_ptr, torch::Tensor cls_col, torch::Tensor cls_succ, torch::Tensor live,
                   torch::Tensor count, torch::Tensor start, torch::Tensor columns, torch::Tensor crank,
                   torch::Tensor coff, torch::Tensor dests, torch::Tensor tiles, torch::Tensor ptr,
                   torch::Tensor parents, torch::Tensor reg_offset, torch::Tensor reg_width, torch::Tensor reg_first,
                   torch::Tensor prior_offset, torch::Tensor prior_width, torch::Tensor reg_prior,
                   torch::Tensor tlist, torch::Tensor t2i, torch::Tensor tab, torch::Tensor cnt, int64_t out_first,
                   int64_t out_width, int64_t temp_width, int64_t tflow_width, int64_t B, int64_t n, int64_t W,
                   int64_t C, int64_t lanes, int64_t maxl, int64_t wt_max, int64_t wt1_max, int64_t subs, int64_t tn,
                   int64_t threads, int64_t s_kind, bool out_temp, bool out_full, bool s_full, bool mars, bool store) {
    TORCH_CHECK(cw.scalar_type() == torch::kFloat32 && element_flows.scalar_type() == torch::kFloat32 &&
                temp_flows.scalar_type() == torch::kFloat32 && node_mars.scalar_type() == torch::kFloat32 &&
                prior_mars.scalar_type() == torch::kFloat32 && temp.scalar_type() == torch::kFloat32,
                "cw_class_cuda: fp32 values");
    TORCH_CHECK(live.scalar_type() == torch::kInt8, "cw_class_cuda: int8 live flags");
    TORCH_CHECK(tn == 1 || tn == 2 || tn == 4, "cw_class_cuda: 1, 2 or 4 destinations a CTA");
    TORCH_CHECK(threads == 128 || threads == 256, "cw_class_cuda: 128 or 256 threads a CTA");
    TORCH_CHECK(subs * tn <= CW_TILE && lanes % 32 == 0 && W < 65535, "cw_class_cuda: tiles, lanes, columns");
    TORCH_CHECK(tab.numel() >= tlist.numel() * B * lanes * maxl && cnt.numel() >= tlist.numel() * B * lanes,
                "cw_class_cuda: the entry tables' work space");
    const int64_t ntiles = tiles.numel() / CW_TILE;
    if (ntiles == 0 || B == 0 || tlist.numel() == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    PrepArgs p{i32(tlist, "tlist"), widx(cw_col), i32(cw_count, "cw_count"), i32(cw_cls, "cw_cls"),
               i32(cls_ptr, "cls_ptr"), i32(cls_col, "cls_col"), i32(cls_succ, "cls_succ"), live.data_ptr<int8_t>(),
               reinterpret_cast<unsigned*>(tab.data_ptr<int>()), cnt.data_ptr<int>(), (int)B, (int)n, (int)W,
               (int)C, (int)lanes, (int)maxl};
    cw_prep_kernel<<<dim3((unsigned)((lanes + 3) / 4), (unsigned)B, (unsigned)tlist.numel()), 128, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    StageArgs a;
    a.cw = cw.data_ptr<float>();
    a.cw_col = widx(cw_col); a.cw_K = widx(cw_K); a.cw_off = widx(cw_off);
    a.cw_count = i32(cw_count, "cw_count");
    a.tab = reinterpret_cast<const unsigned*>(tab.data_ptr<int>()); a.cnt = cnt.data_ptr<int>();
    a.t2i = i32(t2i, "t2i");
    a.element_flows = element_flows.data_ptr<float>(); a.temp_flows = temp_flows.data_ptr<float>();
    a.node_mars = node_mars.data_ptr<float>(); a.prior_mars = prior_mars.data_ptr<float>();
    a.temp = temp.data_ptr<float>();
    a.width = widx(width); a.iv_of = widx(iv_of); a.iv_info = widx(iv_info); a.iv_table = widx(iv_table);
    a.once_iv = widx(once_iv);
    a.count = widx(count); a.start = widx(start); a.columns = i32(columns, "columns");
    a.crank = i32(crank, "crank"); a.coff = widx(coff);
    a.dests = i32(dests, "dests"); a.tiles = i32(tiles, "tiles"); a.ptr = i32(ptr, "ptr");
    a.parents = i32(parents, "parents");
    a.reg_offset = widx(reg_offset); a.reg_width = widx(reg_width); a.reg_first = widx(reg_first);
    a.prior_offset = widx(prior_offset); a.prior_width = widx(prior_width); a.reg_prior = widx(reg_prior);
    a.out_first = out_first; a.out_width = out_width; a.temp_width = temp_width; a.tflow_width = tflow_width;
    a.B = (int)B; a.n = (int)n; a.W = (int)W; a.C = (int)C; a.subs = (int)subs; a.lanes = (int)lanes;
    a.maxl = (int)maxl; a.s_kind = (int)s_kind; a.out_temp = out_temp; a.out_full = out_full; a.s_full = s_full;
    a.mars = mars; a.store = store; a.wt_max = (int)wt_max;
    const size_t smem = (size_t)cw_stage_smem(tn, wt_max, wt1_max);
    const dim3 grid((unsigned)(ntiles * subs), (unsigned)B);
#define CW_LAUNCH(TN_, NT_) launch_stage<TN_, NT_>(a, grid, smem, stream)
    if (threads == 256) {
        if (tn == 4) CW_LAUNCH(4, 256); else if (tn == 2) CW_LAUNCH(2, 256); else CW_LAUNCH(1, 256);
    } else {
        if (tn == 4) CW_LAUNCH(4, 128); else if (tn == 2) CW_LAUNCH(2, 128); else CW_LAUNCH(1, 128);
    }
#undef CW_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cw_class_cuda", &cw_class_cuda, "a class-weight launch of input @ block steps on compact values, staged");
    m.def("cw_stage_smem", &cw_stage_smem, "the shared memory of a class-weight CTA");
}
