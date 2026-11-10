// The elementwise halves of `LowRank`' "sum_exp" form computed on a VIRTUAL BATCH.
//
// The mixture scale `g_b[i, j] = sum_r exp(A_b[i, r]) exp(B_b[r, j])` makes the node's unnormalized value and its
// normalizer
//
//     S_b[i] = sum_r exp(A_b[i, r]) * sum_j theta[i, j] * exp(B_b[r, j]) * c_b[j]
//     Z_b[i] = sum_r exp(A_b[i, r]) * sum_j theta[i, j] * exp(B_b[r, j])
//
// the first of which is, per rank r, a plain sum node over the children's values rescaled by `exp(B_b[r, :])`. So every
// (sample, rank) pair is a column of a virtual batch, the plain sum-layer kernels run on it unchanged, the shared
// parameters read once for all of them, and what is specific to the parameterization is elementwise:
//
//     expand        vem[child, b K + r] = element_mars[child, b] + B_b[r, child]
//     (plain forward on the virtual buffers:  vnm[node, v] = log sum_j theta[node, j] exp(vem[j, v]))
//     collapse      node_mars[node, b] = logsumexp_r(A_b[node, r] + vnm[node, b K + r])                 (log S)
//                                      - logsumexp_r(A_b[node, r] + log Z_r[node, (copy, b, r)])        (log Z)
//
// The rank's normalizer `Z_r[i, v] = sum_j theta[i, j] exp(B_b[r, j])` reads no child values: for a node vector and all
// of its tied copies at once, before the first of them runs, it is one matrix product over the rows `v = (copy, b, r)`,
// `Zt = theta @ exp(B - shift)` (`z_operand`, each row shifted by its largest `B`); its readers take the log and the
// shift back, `log Z_r = log Zt + shift` (the same operations wherever: the same bits). A copy's rows are its columns'
// order, a sample's ranks adjacent.
//
// and backward, with the virtual node values taken as 0 (a virtual node's flow is then the sum of its edges' flows
// whatever its value was), the DATA half alone through the plain kernels:
//
//     expand_bw     vnf[node, b K + r] = log f_b[node] + A_b[node, r] - log S_b[node]
//     (plain backward: virtual element flows `vef` -- stored, whatever the buffer held --; the parameter flows `F+`)
//     collapse_bw   element_flows[child, b] = logsumexp_r vef[child, b K + r]
//
// with the factors' gradients beside: `dA` from the two responsibilities (each rank's share of S and of Z, formed from
// the virtual node values the forward kept on the layer and from the pass's log normalizers), the data half's part
// of `dB` from the virtual element flows.
//
// THE NORMALIZER HALF reads no child values: for a node vector and all of its tied copies at once, at the end of the
// pass, it is two matrix products over the rows `v = (copy, sample, rank)`,
//
//     U[v, i] = f_b[i] exp(A_b[i, r]) / Z_b[i]        V[v, j] = exp(B_b[r, j])
//     W = U^T V             (the denominator statistic, `F- = theta * W`, free of the parameters)
//     dB[v, j] -= V[v, j] * (U theta)[v, j]           (the normalizer half's part of dLL / dB)
//
// V is the forward's (`z_operand`, row `v` shifted by `max_j B_b[r, :]`), `u_operand` writes U, the shift multiplying its
// row as it divides V's, so V <= 1 and U stays within fp32 (log f + A - log Z + max B <= -log theta_min); the products are
// the caller's (cuBLAS, bf16 operands with fp32 accumulation or fp32), `normalizer_grads` takes `V * (U theta)` off
// each copy's `dB`.
//
// THE DATA HALF'S PARAMETER FLOWS are one product too, where a layer's product layer is fused in (its children's values
// are at hand at the pass's end, from `node_mars`): with `U[v, i] = f_b[i] exp(A_b[i, r]) / S_b[i]` and `V[v, j] = c_b[j]
// exp(B_b[r, j])` (`data_operands`, the shift likewise), `F+ = theta * (U^T V)`. The node vector's other flows, those of
// the elements, need the layer above's and stay the plain kernels' on the virtual columns.
//
// Layouts: the circuit's buffers `[rows, batch]`, batch innermost; the virtual ones `[rows, columns]`, a sample's K
// ranks adjacent from column `b K` of a half -- the element ones as tall as the circuit's, the node ones one LAYER's
// (its rows after a few dummy rows, the plain kernels given the layer's tables shifted to match: `nid0` is a node
// vector's first row in the circuit's buffers, `vn0` in the virtual ones); the factors AS STAGED, batch innermost too
// -- `A [M, K, batch]`,
// `B [K, N, batch]` (`LowRank.storage_shapes`: the staging copy is the transpose from the caller's
// `[batch, M, K]` and `[batch, K, N]`), the gradients the same way; per-node state `[M, batch]`; the products' operands
// `[nodes or children, rows]`, a copy's rows `(b, r)` as the columns. One thread a (row, sample), adjacent threads
// adjacent samples, its K ranks in a loop: a warp's accesses of the circuit's buffers, the factors and the state are
// to consecutive addresses, those of the virtual ones and of the log normalizers a run of K a lane.

#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

constexpr int THREADS = 256;

inline int grid_of(long total) { return (int)((total + THREADS - 1) / THREADS); }

// KC values from `p` on, 16-byte aligned, in KC / 4 loads
template <int KC>
inline __device__ void load_run4(const float* __restrict__ p, float (&v)[KC]) {
    #pragma unroll
    for (int q = 0; q < KC / 4; q++) {
        const float4 x = reinterpret_cast<const float4*>(p)[q];
        v[4 * q] = x.x; v[4 * q + 1] = x.y; v[4 * q + 2] = x.z; v[4 * q + 3] = x.w;
    }
}

// ... and stored, KC / 4 stores
template <int KC>
inline __device__ void store_run4(float* __restrict__ p, const float (&v)[KC]) {
    #pragma unroll
    for (int q = 0; q < KC / 4; q++)
        reinterpret_cast<float4*>(p)[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
}

// `expand` for thread `t` (a child, a sample): `vd` the virtual buffer, `cd` columns wide. `vals`: the element values --
// or, FUSED (the product layer below fused in), the NODE values: a child's value read as the sum of its two children's
// (`table[e - first]`, the product group's rows of child node rows; a product of one child has the dummy node, of value
// 0, as its second) -- what the product layer's forward would have written to `element_mars`, which is then neither
// launched nor read.
template <bool FUSED, int KC = 0>
inline __device__ void expand_at(long t, float* vd, long cd, const float* vals, const int32_t* table, long first,
                                 const float* fb, const int64_t* eid, int N, int B, int K) {
    const int b = (int)(t % B);
    const int j = (int)(t / B);
    const long e = eid[j];
    if (e < 0) return;                                                  // a child no edge reaches

    float c;
    if (FUSED) {
        const int32_t* ch = table + (e - first) * 2;
        c = vals[(long)ch[0] * B + b] + vals[(long)ch[1] * B + b];
    } else {
        c = vals[e * B + b];
    }
    float* outd = vd + e * cd + (long)b * K;
    const float* src = fb + (long)j * B + b;                            // B[r, j, b]: r at a stride of N B
    const long sr = (long)N * B;
    if constexpr (KC == 0) {
        for (int r = 0; r < K; r++) outd[r] = c + src[r * sr];
    } else {                                                            // (every load first; the run 16 bytes a store)
        float v[KC];
        #pragma unroll
        for (int r = 0; r < KC; r++) v[r] = c + src[r * sr];
        store_run4(outd, v);
    }
}

template <bool FUSED>
__global__ void expand_kernel(float* vd, long cd, const float* __restrict__ vals, const int32_t* __restrict__ table,
                              long first, const float* __restrict__ fb, const int64_t* __restrict__ eid, int N, int B,
                              int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long)N * B) expand_at<FUSED>(t, vd, cd, vals, table, first, fb, eid, N, B, K);
}

// logsumexp_r (a[r sa] + y[r sy]): `a` the factor's K ranks at a stride of `sa` (the batch), `y` K values at `sy`
inline __device__ float lse_with(const float* __restrict__ a, long sa, const float* __restrict__ y, long sy, int K) {
    float m = -INFINITY;
    for (int r = 0; r < K; r++) m = fmaxf(m, a[r * sa] + y[r * sy]);
    if (!(m > -INFINITY)) return -INFINITY;
    float s = 0.0f;
    for (int r = 0; r < K; r++) s += expf(a[r * sa] + y[r * sy] - m);
    return m + logf(s);
}

// A rank's log normalizer from the product `zt` and its row's shift `sh`
inline __device__ float log_z_of(float zt, float sh) { return logf(zt) + sh; }

// logsumexp_r (a[r sa] + log Z_r): log Z_r from the product's K values `zt` and their rows' shifts `sh`, both runs
inline __device__ float lse_with_z(const float* __restrict__ a, long sa, const float* __restrict__ zt,
                                   const float* __restrict__ sh, int K) {
    float m = -INFINITY;
    for (int r = 0; r < K; r++) m = fmaxf(m, a[r * sa] + log_z_of(zt[r], sh[r]));
    if (!(m > -INFINITY)) return -INFINITY;
    float s = 0.0f;
    for (int r = 0; r < K; r++) s += expf(a[r * sa] + log_z_of(zt[r], sh[r]) - m);
    return m + logf(s);
}

// A (node, sample)'s two sums over the ranks, `ls = lse_r(A_r + log S_r)` (`ys`: log S_r) and `z = lse_r(A_r + log
// Z_r)` (`yz`, `sh`: the normalizers' product and its rows' shifts), `a` the factor's K ranks at a stride of `sa`. KC
// the rank at compile time -- every value of the three runs loaded first, 16 bytes a load, held in registers: these
// kernels run few threads (a node vector's nodes x the batch), each one's loads and exponentials a chain where K is
// known at run time only (MEASURED, HMM 1024 x 1024, rank 16, batch 64: the collapse 15 us a layer so) -- or 0. The
// same operations in the same order either way: the same bits.
// `kd` given: each rank's SHARE DIFFERENCE there, `D_r = exp(A_r + log S_r - ls) - exp(A_r + log Z_r - z)` (0 where
// the node has no mass), what the backward's dA takes (`dLL / dA_r = f D_r`) -- kept by the forward, so the backward
// reads neither the factors, the virtual node values nor the normalizers for it.
template <int KC>
inline __device__ void collapse_sums(const float* __restrict__ a, long sa, const float* __restrict__ ys,
                                     const float* __restrict__ yz, const float* __restrict__ sh, int K, float& ls,
                                     float& z, float* __restrict__ kd = nullptr) {
    if constexpr (KC == 0) {
        ls = lse_with(a, sa, ys, 1, K);
        z = lse_with_z(a, sa, yz, sh, K);
        if (kd != nullptr) {
            const bool live = ls > -INFINITY && z > -INFINITY;
            for (int r = 0; r < K; r++)
                kd[r] = live ? expf(a[r * sa] + ys[r] - ls) - expf(a[r * sa] + log_z_of(yz[r], sh[r]) - z) : 0.0f;
        }
    } else {
        float av[KC], sv[KC], zv[KC], hv[KC];
        load_run4(ys, sv); load_run4(yz, zv); load_run4(sh, hv);
        #pragma unroll
        for (int r = 0; r < KC; r++) av[r] = a[r * sa];
        float ms = -INFINITY, mz = -INFINITY;
        #pragma unroll
        for (int r = 0; r < KC; r++) {
            sv[r] = av[r] + sv[r];
            zv[r] = av[r] + log_z_of(zv[r], hv[r]);
            ms = fmaxf(ms, sv[r]);
            mz = fmaxf(mz, zv[r]);
        }
        float ss = 0.0f, sz = 0.0f;
        #pragma unroll
        for (int r = 0; r < KC; r++) {
            ss += expf(sv[r] - ms);
            sz += expf(zv[r] - mz);
        }
        ls = ms > -INFINITY ? ms + logf(ss) : -INFINITY;
        z = mz > -INFINITY ? mz + logf(sz) : -INFINITY;
        if (kd != nullptr) {
            const bool live = ls > -INFINITY && z > -INFINITY;
            float dv[KC];
            #pragma unroll
            for (int r = 0; r < KC; r++) dv[r] = live ? expf(sv[r] - ls) - expf(zv[r] - z) : 0.0f;
            store_run4(kd, dv);
        }
    }
}

// `zt` [M, rows]: the normalizers' product, `zt[i, row0 + b K + r]`, row0 this copy's first row, `shift` [rows] its
// rows' shifts; `vnm` the virtual node values (log S_r: row `vn0 + i`, column `b K + r`, `cols` wide); `kd` (as wide,
// the same rows) the share differences the backward's dA takes, kept by the layer (`virtual._kept`).
template <int KC>
__global__ void collapse_kernel(float* __restrict__ node_mars, const float* __restrict__ vnm, float* __restrict__ kd,
                                long cols, const float* __restrict__ fa, float* __restrict__ lz,
                                const float* __restrict__ zt, const float* __restrict__ shift, long rows, long row0,
                                long nid0, long vn0, int M, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)M * B) return;
    const int b = (int)(t % B);
    const int i = (int)(t / B);

    const float* yd = vnm + (vn0 + i) * cols + (long)b * K;             // the virtual node values: K in a run
    const float* yz = zt + (long)i * rows + row0 + (long)b * K;         // the normalizers: K in a run, their shifts
    const long at = (long)i * K * B + b;                                // A[i, r, b]: r at a stride of B
    float ls, z;
    collapse_sums<KC>(fa + at, B, yd, yz, shift + row0 + (long)b * K, K, ls, z, kd + (vn0 + i) * cols + (long)b * K);
    // (Z = 0, a node whose parameters are all 0: no mass, -inf. The plain kernels give such a row's virtual values
    // -FLT_MAX, not -inf, so `ls` alone does not tell it: `ls - z` would be +inf.)
    node_mars[(nid0 + i) * B + b] = (ls > -INFINITY && z > -INFINITY) ? ls - z : -INFINITY;
    lz[(long)i * B + b] = z;
}

// The forward's `collapse` of a node vector and the `expand` of the NEXT layer's children (the product layer between
// fused in), ONE launch: the first `M B` threads are `collapse_kernel`'s, the `N B` after them the fused `expand`'s
// for a node vector of the next layer (rank `KN`, its forward buffer `vem`, `cw` wide). A product's child that is one of
// THIS node vector's is not read from `node_mars` -- which the other threads are writing -- but collapsed again from
// the virtual values: the same operations, the same bits. Every other child is a row computed before this layer (the
// caller's condition).
template <int KC>
__global__ void collapse_expand_kernel(float* node_mars, const float* __restrict__ vnm, float* __restrict__ kd,
                                       long cols, const float* __restrict__ fa, float* __restrict__ lz,
                                       const float* __restrict__ zt, const float* __restrict__ shift, long rows,
                                       long row0, long nid0, long vn0, int M, int K,
                                       float* __restrict__ vem, long cw, const int32_t* __restrict__ table, long first,
                                       const float* __restrict__ fb, const int64_t* __restrict__ eid, int N, int KN,
                                       int B) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long)M * B) {
        const int b = (int)(t % B);
        const int i = (int)(t / B);
        const float* yd = vnm + (vn0 + i) * cols + (long)b * K;
        const float* yz = zt + (long)i * rows + row0 + (long)b * K;
        const long at = (long)i * K * B + b;
        float ls, z;
        collapse_sums<KC>(fa + at, B, yd, yz, shift + row0 + (long)b * K, K, ls, z, kd + (vn0 + i) * cols + (long)b * K);
        node_mars[(nid0 + i) * B + b] = (ls > -INFINITY && z > -INFINITY) ? ls - z : -INFINITY;
        lz[(long)i * B + b] = z;
        return;
    }
    const long u = t - (long)M * B;
    if (u >= (long)N * B) return;
    const int b = (int)(u % B);
    const int j = (int)(u / B);
    const long e = eid[j];
    if (e < 0) return;

    const int32_t* ch = table + (e - first) * 2;
    float v[2];
    for (int k = 0; k < 2; k++) {
        const long row = ch[k];
        if (row >= nid0 && row < nid0 + M) {                            // one of the nodes collapsed in this launch
            const float* yd = vnm + (row - nid0 + vn0) * cols + (long)b * K;
            const float* yz = zt + (row - nid0) * rows + row0 + (long)b * K;
            float ls, z;
            collapse_sums<KC>(fa + (row - nid0) * (long)K * B + b, B, yd, yz, shift + row0 + (long)b * K, K, ls, z);
            v[k] = (ls > -INFINITY && z > -INFINITY) ? ls - z : -INFINITY;
        } else {
            v[k] = node_mars[row * B + b];
        }
    }
    const float c = v[0] + v[1];
    float* outd = vem + e * cw + (long)b * KN;
    const float* src = fb + (long)j * B + b;                            // the layer above's B[r, j, b]
    const long sr = (long)N * B;
    for (int r = 0; r < KN; r++) outd[r] = c + src[r * sr];
}

// `expand_bw` for thread `t` (a node, a sample); GRAD: and dLL / dA = f D_r, D the share differences the forward kept
// (`kd`: row `vn0 + i`, column `b K + r`, `cols` wide)
template <bool GRAD, int KC>
inline __device__ void expand_bw_at(long t, float* vnfd, const float* node_flows, const float* node_mars,
                                    const float* fa, const float* lz, const float* kd, long cols, float* grad_a,
                                    long nid0, long vn0, int M, int B, int K) {
    const int b = (int)(t % B);
    const int i = (int)(t / B);

    const long row = nid0 + i;
    const long at = (long)i * K * B + b;                                // A[i, r, b]: r at a stride of B
    const float lf = node_flows[row * B + b];
    const float z = lz[(long)i * B + b];
    const float ls = node_mars[row * B + b] + z;                        // log S = log n + log Z
    float* outd = vnfd + ((vn0 + i) * (long)B + b) * K;
    const bool live = (lf > -INFINITY) && (ls > -INFINITY);
    const float* dk = kd + (vn0 + i) * cols + (long)b * K;
    if constexpr (KC == 0) {
        for (int r = 0; r < K; r++) outd[r] = live ? lf + fa[at + r * B] - ls : -INFINITY;
        if (GRAD && live) {
            // dLL / dA[i, r] = f * (the rank's share of S - its share of Z); (live: S and so Z have mass)
            const float ef = expf(lf);
            for (int r = 0; r < K; r++) grad_a[at + r * B] += ef * dk[r];
        }
    } else {                                             // (collapse_sums': every load first, the runs 16 bytes a load)
        float av[KC];
        #pragma unroll
        for (int r = 0; r < KC; r++) av[r] = fa[at + r * B];
        #pragma unroll
        for (int q = 0; q < KC / 4; q++)
            reinterpret_cast<float4*>(outd)[q] = live ? make_float4(lf + av[4 * q] - ls, lf + av[4 * q + 1] - ls,
                                                                   lf + av[4 * q + 2] - ls, lf + av[4 * q + 3] - ls)
                                                      : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        if (GRAD && live) {
            const float ef = expf(lf);
            float dv[KC], gv[KC];
            load_run4(dk, dv);
            #pragma unroll
            for (int r = 0; r < KC; r++) gv[r] = grad_a[at + r * B];
            #pragma unroll
            for (int r = 0; r < KC; r++) grad_a[at + r * B] = gv[r] + ef * dv[r];
        }
    }
}

// A layer's backward before its plain calls, ONE launch: the first `N B` threads `expand` (the virtual child values
// of the data half), the `M B` after them `expand_bw` (the virtual node flows of the data half, `dA`). Neither reads
// what the other writes.
template <bool GRAD, bool FUSED, int KC>
__global__ void backward_pre_kernel(float* vd, long cd, const float* vals, const int32_t* __restrict__ table,
                                    long first, const float* __restrict__ fb, const int64_t* __restrict__ eid, int N,
                                    float* __restrict__ vnfd, const float* __restrict__ node_flows,
                                    const float* node_mars, const float* __restrict__ fa,
                                    const float* __restrict__ lz, const float* __restrict__ kd, long cols,
                                    float* __restrict__ grad_a, long nid0, long vn0, int M, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long)N * B) {
        expand_at<FUSED, KC>(t, vd, cd, vals, table, first, fb, eid, N, B, K);
    } else if (t - (long)N * B < (long)M * B) {
        expand_bw_at<GRAD, KC>(t - (long)N * B, vnfd, node_flows, node_mars, fa, lz, kd, cols, grad_a, nid0, vn0, M, B,
                               K);
    }
}

// GRAD: and the data half's part of dLL / dB. FUSED: the product layer below fused in -- `flows` are the NODE flows,
// and an element's flow is written as its two children's (`table[e - first]`): what the product layer's backward
// writes for children of one parent element each, which is then not launched.
template <bool GRAD, bool FUSED, int KC>
__global__ void collapse_bw_kernel(float* __restrict__ flows, const float* __restrict__ vefd,
                                   const int64_t* __restrict__ eid, float* __restrict__ grad_b,
                                   const int32_t* __restrict__ table, long first, int N, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)N * B) return;
    const int b = (int)(t % B);
    const int j = (int)(t / B);
    const long e = eid[j];
    if (e < 0) return;

    const float* f = vefd + (e * (long)B + b) * K;
    constexpr int KR = KC > 0 ? KC : 1;
    float fv[KR];                                                       // (KC: the run in registers, read once)
    if constexpr (KC > 0) load_run4(f, fv);
    auto at = [&](int r) { if constexpr (KC > 0) return fv[r]; else return f[r]; };
    float m = -INFINITY;
    #pragma unroll
    for (int r = 0; r < (KC > 0 ? KC : K); r++) m = fmaxf(m, at(r));
    float lef = -INFINITY;
    if (m > -INFINITY) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < (KC > 0 ? KC : K); r++) s += expf(at(r) - m);
        lef = m + logf(s);
    }
    if (FUSED) {
        const int32_t* ch = table + (e - first) * 2;
        flows[(long)ch[0] * B + b] = lef;
        flows[(long)ch[1] * B + b] = lef;
    } else {
        flows[e * B + b] = lef;
    }

    if (GRAD) {
        // dLL / dB[r, j] += the data half's virtual element flow (the normalizer half's comes off at the pass's end)
        float* g = grad_b + (long)j * B + b;
        #pragma unroll
        for (int r = 0; r < (KC > 0 ? KC : K); r++) g[(long)r * N * B] += expf(at(r));
    }
}

inline __device__ float as_float(float x) { return x; }
inline __device__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// The operands of the pass's matrix products, stored TRANSPOSED: `Ut` [M, rows], `Vt` [N, rows], a copy's rows
// `(copy, sample, rank)` -- from the copy's first row `row0` on, `row0 + b K + r`: a sample's K ranks adjacent, as the
// virtual batch's columns are (a tile of the fused stream forward reads a run of V a child, its samples' every rank).
// ONE launch for all the copies of a node vector: their tensors go BY VALUE in the launch's parameters (as the staging
// transpose's pairs: nothing to upload, nothing kept by address), `blockIdx.z` picks a copy; at most COPIES_PER_LAUNCH
// copies a launch, more in as many launches. The shapes of launch:
//   - a row's SHIFT, the largest of its terms over the children, takes a block over every child: the data half's
//     (`data_shift`) a copy's rank's 32 samples, `WARPS` warps of 32 lanes over every `WARPS`th child (a load of a
//     factor one line; bound by the latency of a warp's loads, one child after another); the normalizer's in its
//     operand's own launch (`z_operand`);
//   - the operands U and the data half's V: a thread a (child or node, RUN consecutive rows of a copy), adjacent threads
//     adjacent runs -- a thread's stores one 16-byte run (bf16) where a copy's rows are whole runs, a warp's one
//     contiguous run;
//   - the gradients' `normalizer_grads`: a thread a (child, sample), lanes the samples.
constexpr int COPIES_PER_LAUNCH = 64;
constexpr int WARPS = 32;
constexpr int OPERAND_THREADS = WARPS * 32;
constexpr int RUN = 8;

// The (sample, rank) of the row `t0` of a copy, and of the row after
inline __device__ void run_at(int t0, int K, int& b, int& r) { b = t0 / K; r = t0 - b * K; }
inline __device__ void run_step(int K, int& b, int& r) { if (++r == K) { r = 0; b++; } }

// A run's `n` values from `p` on: one vector store where every run is whole and aligned (`vec`), else one by one
inline __device__ void store_run(float* p, const float (&v)[RUN], int n, bool vec) {
    if (vec) {
        reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
        reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
    } else {
        for (int q = 0; q < n; q++) p[q] = v[q];
    }
}
inline __device__ unsigned bf16x2(float lo, float hi) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
inline __device__ void store_run(__nv_bfloat16* p, const float (&v)[RUN], int n, bool vec) {
    if (vec) {
        *reinterpret_cast<uint4*>(p) = make_uint4(bf16x2(v[0], v[1]), bf16x2(v[2], v[3]), bf16x2(v[4], v[5]),
                                                  bf16x2(v[6], v[7]));
    } else {
        for (int q = 0; q < n; q++) p[q] = __float2bfloat16(v[q]);
    }
}

// The backward's normalizer half: `Ut[i, row] = f_b[i] exp(A[i, r, b] - log Z_b[i] + shift)` (0 where the node has no
// flow), with the rows and the shifts the forward's `z_operand` gave `Vt`.
struct UCopies { long long rows[COPIES_PER_LAUNCH * 4]; };      // a copy's A, log Z, first node row and first row

template <typename TO>
__global__ void u_operand_kernel(const UCopies copies, TO* __restrict__ Ut, const float* __restrict__ shift, long rows,
                                 const float* __restrict__ node_flows, int M, int B, int K, bool vec) {
    const long long* c = copies.rows + 4 * blockIdx.z;
    const float* fa = reinterpret_cast<const float*>(c[0]);
    const float* lz = reinterpret_cast<const float*>(c[1]);
    const long nid0 = (long)c[2], row0 = (long)c[3];
    const int R = B * K, runs = (R + RUN - 1) / RUN;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)M * runs) return;
    const int i = (int)(t / runs), t0 = (int)(t % runs) * RUN, n = min(RUN, R - t0);
    const long at = row0 + t0;
    int b, r;
    run_at(t0, K, b, r);
    float v[RUN];
    #pragma unroll
    for (int q = 0; q < RUN; q++) {
        float y = 0.0f;
        if (q < n) {
            const float lf = node_flows[(nid0 + i) * (long)B + b];
            const float z = lz[(long)i * B + b];
            const bool live = (lf > -INFINITY) && (z < INFINITY);
            y = live ? expf(lf + fa[((long)i * K + r) * B + b] - z + shift[at + q]) : 0.0f;
        }
        v[q] = y;
        run_step(K, b, r);
    }
    store_run(Ut + (long)i * rows + at, v, n, vec);
}

// The forward's normalizers: each row's `shift`, the largest `B[r, :, b]`, and `Vt[j, row] = exp(B[r, j, b] - shift)`
// (0 where B is -inf); the product `Zt = theta @ Vt` [M, rows] is then each rank's normalizer under its row's shift. The
// same kernel makes the backward's `exp(A - shift)` (`lowrank_a_operand`): a factor's element (x, r, b) at
// `x sx + r sr + b`, `x` a child of B or a node of A.
struct ZCopies { long long fb[COPIES_PER_LAUNCH]; };

// ONE launch: a block is 8 samples of a copy and 4 of its ranks, a lane one (sample, rank) -- the rank innermost: a
// warp's loads of a rank 8 samples in one 32-byte sector, its stores a child 8 runs of 4 rows --, `WARPS` warps over
// every `WARPS`th child; a lane's first `ZKEEP` children held in registers from the max to the operand, so the factors
// are read once (more children than `WARPS ZKEEP`: the rest read again). MEASURED (HMM 1024 x 1024, rank 16, batch 64,
// 31 copies, us): this 166; one launch for the shifts and one for the operand (a thread a run of 8 rows) 207 -- the
// factors read twice from DRAM --; a block of 32 samples of one rank (lanes the samples: 2-byte stores a row apart)
// 429, of 16 samples x 2 ranks 221, of 8 x 4 reading the factors again 187, of 8 samples x 16 ranks (4 a lane: 8-byte
// stores, one run a warp) 237 -- 248 blocks, 1.3 waves. (The rows (copy, rank, sample) took 142: a warp's stores a
// child one 64-byte run.)
inline __device__ void store_as(float* p, float v) { *p = v; }
inline __device__ void store_as(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

constexpr int ZKEEP = 32;

template <typename TO>
__global__ void __launch_bounds__(OPERAND_THREADS, 1) z_operand_kernel(const ZCopies copies, TO* __restrict__ Vt,
                                                                      float* __restrict__ shift, long rows, long row0,
                                                                      int N, int B, int K, long sx, long sr) {
    const float* fb = reinterpret_cast<const float*>(copies.fb[blockIdx.z]);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int b = blockIdx.x * 8 + lane / 4;
    const int r = blockIdx.y * 4 + lane % 4;
    const bool on = b < B && r < K;
    __shared__ float most[WARPS][32];

    const float* src = fb + (on ? (long)r * sr + b : 0);              // B[r, j, b]: j at a stride of sx
    float keep[ZKEEP];
    float m = -INFINITY;
    #pragma unroll
    for (int k = 0; k < ZKEEP; k++) {
        const int j = w + WARPS * k;
        keep[k] = on && j < N ? src[(long)j * sx] : -INFINITY;
        m = fmaxf(m, keep[k]);
    }
    if (on) for (int j = w + WARPS * ZKEEP; j < N; j += WARPS) m = fmaxf(m, src[(long)j * sx]);
    most[w][lane] = m;
    __syncthreads();
    m = most[0][lane];
    for (int i = 1; i < WARPS; i++) m = fmaxf(m, most[i][lane]);
    if (!on) return;
    const long at = row0 + (long)blockIdx.z * K * B + (long)b * K + r;
    if (w == 0) shift[at] = m;
    TO* out = Vt + at;
    #pragma unroll
    for (int k = 0; k < ZKEEP; k++) {
        const int j = w + WARPS * k;
        if (j < N) store_as(out + (long)j * rows, keep[k] > -INFINITY ? expf(keep[k] - m) : 0.0f);
    }
    for (int j = w + WARPS * ZKEEP; j < N; j += WARPS) {
        const float x = src[(long)j * sx];
        store_as(out + (long)j * rows, x > -INFINITY ? expf(x - m) : 0.0f);
    }
}

// The data half's operands, for the copies whose product layer is fused in (their parameter flows are one product at the
// pass's end): `Ut[i, row] = f_b[i] exp(A[i, r, b]) / S_b[i] exp(shift)`, `Vt[j, row] = c_b[j] exp(B[r, j, b]) exp(-shift)`,
// the shift the row's largest log of `c_b[j] exp(B[r, j, b])`, `c_b[j]` the value of the product at child `j`, formed from
// its two children's node values -- which, unlike `element_mars`, the layers below have not overwritten at the pass's end.
// `S_b[i]`: `node_mars` holds `log S - log Z`. Two launches, as the normalizer's: the shifts (`data_shift`, into a
// scratch of the launch's rows), then both operands. (A child no edge reaches: 0.)
struct DataCopies { long long rows[COPIES_PER_LAUNCH * 7]; };   // A, log Z, B, first node row, children's rows, table, first

__global__ void data_shift_kernel(const DataCopies copies, float* __restrict__ shd, const float* __restrict__ node_mars,
                                  int N, int B, int K) {
    const long long* c = copies.rows + 7 * blockIdx.z;
    const float* fb = reinterpret_cast<const float*>(c[2]);
    const int64_t* eid = reinterpret_cast<const int64_t*>(c[4]);
    const int32_t* table = reinterpret_cast<const int32_t*>(c[5]);
    const long first = (long)c[6];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r = blockIdx.y, b = blockIdx.x * 32 + lane;
    __shared__ float most[WARPS][32];

    const float* brow = fb + (long)r * N * B + b;
    float m = -INFINITY;
    if (b < B) {
        for (int j = w; j < N; j += WARPS) {
            const long e = eid[j];
            if (e < 0) continue;
            const int32_t* ch = table + (e - first) * 2;
            m = fmaxf(m, node_mars[(long)ch[0] * B + b] + node_mars[(long)ch[1] * B + b] + brow[(long)j * B]);
        }
    }
    most[w][lane] = m;
    __syncthreads();
    if (w != 0 || b >= B) return;
    float sh = most[0][lane];
    for (int i = 1; i < WARPS; i++) sh = fmaxf(sh, most[i][lane]);
    shd[(long)blockIdx.z * K * B + (long)b * K + r] = sh;
}

// The first `N runs` threads of a copy write `Vt`'s runs, the `M runs` after them `Ut`'s
template <typename TO>
__global__ void data_operands_kernel(const DataCopies copies, TO* __restrict__ Ut, TO* __restrict__ Vt,
                                     const float* __restrict__ shd, long rows, long row0,
                                     const float* __restrict__ node_flows, const float* __restrict__ node_mars, int M,
                                     int N, int B, int K, bool vec) {
    const long long* c = copies.rows + 7 * blockIdx.z;
    const float* fa = reinterpret_cast<const float*>(c[0]);
    const float* lz = reinterpret_cast<const float*>(c[1]);
    const float* fb = reinterpret_cast<const float*>(c[2]);
    const long nid0 = (long)c[3];
    const int64_t* eid = reinterpret_cast<const int64_t*>(c[4]);
    const int32_t* table = reinterpret_cast<const int32_t*>(c[5]);
    const long first = (long)c[6];
    const int R = B * K, runs = (R + RUN - 1) / RUN;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)(N + M) * runs) return;
    const bool of_v = t < (long)N * runs;
    const long u = of_v ? t : t - (long)N * runs;
    const int x = (int)(u / runs), t0 = (int)(u % runs) * RUN, n = min(RUN, R - t0);   // x: a child (V), a node (U)
    const float* sh = shd + (long)blockIdx.z * R + t0;
    const long at = row0 + (long)blockIdx.z * R + t0;
    int b, r;
    run_at(t0, K, b, r);
    float v[RUN];
    if (of_v) {
        const long e = eid[x];
        const int32_t* ch = table + (e >= 0 ? e - first : 0) * 2;
        #pragma unroll
        for (int q = 0; q < RUN; q++) {
            float y = 0.0f;
            if (q < n && e >= 0) {
                const float w = node_mars[(long)ch[0] * B + b] + node_mars[(long)ch[1] * B + b] +
                                fb[((long)r * N + x) * B + b];
                y = w > -INFINITY ? expf(w - sh[q]) : 0.0f;
            }
            v[q] = y;
            run_step(K, b, r);
        }
        store_run(Vt + (long)x * rows + at, v, n, vec);
    } else {
        #pragma unroll
        for (int q = 0; q < RUN; q++) {
            float y = 0.0f;
            if (q < n) {
                const float lf = node_flows[(nid0 + x) * (long)B + b];
                const float ls = node_mars[(nid0 + x) * (long)B + b] + lz[(long)x * B + b];   // log S = log n + log Z
                const bool live = (lf > -INFINITY) && (ls > -INFINITY);
                y = live ? expf(lf + fa[((long)x * K + r) * B + b] - ls + sh[q]) : 0.0f;
            }
            v[q] = y;
            run_step(K, b, r);
        }
        store_run(Ut + (long)x * rows + at, v, n, vec);
    }
}

// dB[r, j, b] -= Vt[j, row] * Gt[j, row]: every copy's normalizer part off its gradient, one launch, a thread a (child,
// sample), its K ranks in runs of RUN -- lanes adjacent samples: a warp's read-modify-write of dB one line a rank, its
// loads of a run 16 bytes a lane (MEASURED, HMM 1024 x 1024, rank 16, batch 64, 31 copies: 316 us; a thread a run of 8
// rows 346; the rows (copy, rank, sample), lanes the samples, 322). A copy's buffer (0 where it has none) and first row
// by value, no two copies of a launch sharing a buffer (the launcher).
struct NormalizerGrads { long long at[COPIES_PER_LAUNCH * 2]; };

// A run's values of an operand and of `Gt` (`vec`: whole runs, aligned: one 16-byte load of the operand's, two of Gt's)
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

template <typename TO>
__global__ void normalizer_grads_kernel(const NormalizerGrads grads, const TO* __restrict__ Vt,
                                        const float* __restrict__ Gt, long rows, int N, int B, int K, bool vec) {
    float* grad_b = reinterpret_cast<float*>(grads.at[2 * blockIdx.z]);
    const long row0 = (long)grads.at[2 * blockIdx.z + 1];
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (grad_b == nullptr || t >= (long)N * B) return;
    const int b = (int)(t % B), j = (int)(t / B);
    const long at = (long)j * rows + row0 + (long)b * K;
    for (int r0 = 0; r0 < K; r0 += RUN) {
        const int n = min(RUN, K - r0);
        float v[RUN], gv[RUN];
        if (vec) {
            load_run(Vt + at + r0, v);
            load_run(Gt + at + r0, gv);
        } else {
            #pragma unroll
            for (int q = 0; q < RUN; q++) {
                v[q] = q < n ? as_float(Vt[at + r0 + q]) : 0.0f;
                gv[q] = q < n ? Gt[at + r0 + q] : 0.0f;
            }
        }
        #pragma unroll
        for (int q = 0; q < RUN; q++)
            if (q < n) grad_b[((long)(r0 + q) * N + j) * B + b] -= v[q] * gv[q];
    }
}

void check(const torch::Tensor& t, const char* name, torch::ScalarType st, long numel) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "lowrank mixture: `", name, "` must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == st, "lowrank mixture: `", name, "` has the wrong dtype");
    TORCH_CHECK(numel < 0 || t.numel() == numel, "lowrank mixture: `", name, "` has ", t.numel(),
                " elements, expected ", numel);
}

}  // namespace


namespace {

// What `expand` is given: `rows`, the element rows its buffer must hold
void check_expand(const torch::Tensor& vd, const torch::Tensor& fb, const torch::Tensor& eid, long rows, int B, int N,
                  int64_t K) {
    check(vd, "vem", torch::kFloat32, -1);
    check(fb, "B", torch::kFloat32, (long)B * K * N); check(eid, "eid", torch::kInt64, N);
    TORCH_CHECK(vd.size(0) >= rows && vd.size(1) == (long)B * K,
                "lowrank mixture: the virtual element buffer does not hold ", (long)B * K, " columns of ", rows, " rows");
}

long check_children(const torch::Tensor& table, int64_t first) {
    check(table, "children", torch::kInt32, -1);
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    return first + table.size(0);
}

}  // namespace

void lowrank_expand(torch::Tensor vd, torch::Tensor element_mars, torch::Tensor fb, torch::Tensor eid, int64_t K) {
    const int B = element_mars.size(1), N = eid.size(0);
    check(element_mars, "element_mars", torch::kFloat32, -1);
    check_expand(vd, fb, eid, element_mars.size(0), B, N, K);
    if (N == 0 || B == 0) return;
    expand_kernel<false><<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        vd.data_ptr<float>(), (long)vd.size(1), element_mars.data_ptr<float>(), nullptr, 0, fb.data_ptr<float>(),
        eid.data_ptr<int64_t>(), N, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lowrank_expand_fused(torch::Tensor vd, torch::Tensor node_mars, torch::Tensor table, int64_t first,
                          torch::Tensor fb, torch::Tensor eid, int64_t K) {
    const int B = node_mars.size(1), N = eid.size(0);
    check(node_mars, "node_mars", torch::kFloat32, -1);
    check_expand(vd, fb, eid, check_children(table, first), B, N, K);
    if (N == 0 || B == 0) return;
    expand_kernel<true><<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        vd.data_ptr<float>(), (long)vd.size(1), node_mars.data_ptr<float>(), table.data_ptr<int32_t>(), (long)first,
        fb.data_ptr<float>(), eid.data_ptr<int64_t>(), N, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

// The rank the collapse's kernels take at compile time (16, every run 16-byte aligned), else 0
int collapse_rank(int64_t K, std::initializer_list<const torch::Tensor*> runs) {
    if (K != 16) return 0;
    for (const torch::Tensor* t : runs)
        if ((reinterpret_cast<uintptr_t>(t->data_ptr()) & 15) != 0 || t->size(-1) % 4 != 0) return 0;
    return 16;
}

// The normalizers' product `zt` [M, rows] and its rows' shifts [rows], holding this copy's `B K` rows from `row0` on
void check_normalizers(const torch::Tensor& zt, const torch::Tensor& shift, int64_t row0, long M, long BK) {
    check(zt, "Z", torch::kFloat32, -1); check(shift, "shift", torch::kFloat32, -1);
    TORCH_CHECK(zt.dim() == 2 && zt.size(0) == M && shift.numel() == zt.size(1) && row0 >= 0 && row0 + BK <= zt.size(1),
                "lowrank mixture: the normalizers do not hold this copy's rows");
}

void check_collapse(const torch::Tensor& node_mars, const torch::Tensor& vnm, const torch::Tensor& kd,
                    const torch::Tensor& fa, const torch::Tensor& lz, const torch::Tensor& zt,
                    const torch::Tensor& shift, int64_t row0, int64_t nid0, int64_t vn0, int64_t M, int64_t K) {
    const int B = node_mars.size(1);
    check(node_mars, "node_mars", torch::kFloat32, -1); check(vnm, "vnm", torch::kFloat32, -1);
    check(kd, "share differences", torch::kFloat32, -1);
    TORCH_CHECK(kd.sizes() == vnm.sizes(), "lowrank mixture: the kept share differences as the virtual node values");
    check(fa, "A", torch::kFloat32, (long)B * M * K); check(lz, "lz", torch::kFloat32, (long)B * M);
    TORCH_CHECK(nid0 >= 0 && vn0 >= 0 && vnm.size(0) >= vn0 + M && vnm.size(1) == (long)B * K &&
                node_mars.size(0) >= nid0 + M,
                "lowrank mixture: the node rows ", nid0, " + ", M, " (virtual ", vn0, ") do not fit the buffers");
    check_normalizers(zt, shift, row0, M, (long)B * K);
}

}  // namespace

// `zt` [M, rows]: the normalizers' product, `shift` [rows] its rows' shifts (`lowrank_z_operand`), this copy's rows
// from `row0` on; `kd` (as `vnm`) the share differences the backward's dA takes, written
void lowrank_collapse(torch::Tensor node_mars, torch::Tensor vnm, torch::Tensor kd, torch::Tensor fa, torch::Tensor lz,
                      torch::Tensor zt, torch::Tensor shift, int64_t row0, int64_t nid0, int64_t vn0, int64_t M,
                      int64_t K) {
    check_collapse(node_mars, vnm, kd, fa, lz, zt, shift, row0, nid0, vn0, M, K);
    const int B = node_mars.size(1);
    if (M == 0 || B == 0) return;
    auto go = [&](auto kc) {
        collapse_kernel<decltype(kc)::value><<<grid_of((long)M * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), vnm.data_ptr<float>(), kd.data_ptr<float>(), (long)vnm.size(1), fa.data_ptr<float>(),
            lz.data_ptr<float>(), zt.data_ptr<float>(), shift.data_ptr<float>(), (long)zt.size(1), (long)row0,
            (long)nid0, (long)vn0, (int)M, B, (int)K);
    };
    if (collapse_rank(K, {&vnm, &kd, &zt, &shift}) == 16 && row0 % 4 == 0) go(std::integral_constant<int, 16>{});
    else go(std::integral_constant<int, 0>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// `lowrank_collapse` of a node vector and `lowrank_expand_fused` of a node vector of the next layer (its factors `fb`
// of rank `KC`, its children `eid`, its forward buffer `vem`), one launch; the next layer's children that are this
// node vector's are taken from the collapse itself, every other one from `node_mars`.
void lowrank_collapse_expand(torch::Tensor node_mars, torch::Tensor vnm, torch::Tensor kd, torch::Tensor fa,
                             torch::Tensor lz, torch::Tensor zt, torch::Tensor shift, int64_t row0, int64_t nid0,
                             int64_t vn0, int64_t M, int64_t K, torch::Tensor vem, torch::Tensor table, int64_t first,
                             torch::Tensor fb, torch::Tensor eid, int64_t KC) {
    check_collapse(node_mars, vnm, kd, fa, lz, zt, shift, row0, nid0, vn0, M, K);
    const int B = node_mars.size(1), N = eid.size(0);
    check(vem, "vem", torch::kFloat32, -1); check(table, "children", torch::kInt32, -1);
    check(fb, "B", torch::kFloat32, (long)B * KC * N); check(eid, "eid", torch::kInt64, N);
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    TORCH_CHECK(vem.size(0) >= first + table.size(0) && vem.size(1) == (long)B * KC,
                "lowrank mixture: the virtual element buffer does not hold the next layer's columns");
    if ((M == 0 && N == 0) || B == 0) return;
    auto go = [&](auto kc) {
        collapse_expand_kernel<decltype(kc)::value><<<grid_of((long)(M + N) * B), THREADS, 0,
                                                      at::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), vnm.data_ptr<float>(), kd.data_ptr<float>(), (long)vnm.size(1),
            fa.data_ptr<float>(), lz.data_ptr<float>(), zt.data_ptr<float>(), shift.data_ptr<float>(), (long)zt.size(1),
            (long)row0, (long)nid0, (long)vn0, (int)M, (int)K, vem.data_ptr<float>(), (long)vem.size(1),
            table.data_ptr<int32_t>(), (long)first, fb.data_ptr<float>(), eid.data_ptr<int64_t>(), N, (int)KC, B);
    };
    if (collapse_rank(K, {&vnm, &kd, &zt, &shift}) == 16 && row0 % 4 == 0) go(std::integral_constant<int, 16>{});
    else go(std::integral_constant<int, 0>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

template <bool FUSED>
void collapse_bw_launch(torch::Tensor& flows, torch::Tensor& vefd, torch::Tensor& eid,
                        c10::optional<torch::Tensor>& grad_b, const int32_t* table, long first, int64_t K) {
    const int B = flows.size(1), N = eid.size(0);
    check(flows, "flows", torch::kFloat32, -1); check(vefd, "vef (data)", torch::kFloat32, -1);
    check(eid, "eid", torch::kInt64, N);
    TORCH_CHECK(vefd.size(1) == (long)B * K, "lowrank mixture: `vef` does not match the flows");
    if (N == 0 || B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = grid_of((long)N * B);
    if (grad_b.has_value()) check(*grad_b, "grad B", torch::kFloat32, (long)B * K * N);
    float* gp = grad_b.has_value() ? grad_b->data_ptr<float>() : nullptr;
    auto go = [&](auto grad, auto kc) {
        collapse_bw_kernel<decltype(grad)::value, FUSED, decltype(kc)::value><<<grid, THREADS, 0, stream>>>(
            flows.data_ptr<float>(), vefd.data_ptr<float>(), eid.data_ptr<int64_t>(), gp, table, first, N, B, (int)K);
    };
    const bool k16 = collapse_rank(K, {&vefd});
    if (gp) { if (k16) go(std::true_type{}, std::integral_constant<int, 16>{}); else go(std::true_type{}, std::integral_constant<int, 0>{}); }
    else { if (k16) go(std::false_type{}, std::integral_constant<int, 16>{}); else go(std::false_type{}, std::integral_constant<int, 0>{}); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void lowrank_collapse_bw(torch::Tensor element_flows, torch::Tensor vefd, torch::Tensor eid,
                         c10::optional<torch::Tensor> grad_b, int64_t K) {
    TORCH_CHECK(vefd.size(0) >= element_flows.size(0), "lowrank mixture: `vef` does not match the element flows");
    collapse_bw_launch<false>(element_flows, vefd, eid, grad_b, nullptr, 0, K);
}

// ... with the product layer below fused in: the element flows written as their two children's NODE flows
void lowrank_collapse_bw_fused(torch::Tensor node_flows, torch::Tensor vefd, torch::Tensor eid, torch::Tensor table,
                               int64_t first, c10::optional<torch::Tensor> grad_b, int64_t K) {
    check(table, "children", torch::kInt32, -1);
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    TORCH_CHECK(vefd.size(0) >= first + table.size(0), "lowrank mixture: `vef` does not hold the products' rows");
    collapse_bw_launch<true>(node_flows, vefd, eid, grad_b, table.data_ptr<int32_t>(), (long)first, K);
}

namespace {

// `vals`: the element values, or with a table of children (FUSED) the node values the products' are read from
template <bool FUSED>
void backward_pre_launch(torch::Tensor& vemd, const float* vals, long rows, const int32_t* table, long first,
                         torch::Tensor& fb, torch::Tensor& eid, torch::Tensor& vnfd, torch::Tensor& node_flows,
                         torch::Tensor& node_mars, torch::Tensor& fa, torch::Tensor& lz, torch::Tensor& kd,
                         c10::optional<torch::Tensor>& grad_a, int64_t nid0, int64_t vn0, int64_t M, int64_t K) {
    const int B = node_mars.size(1), N = eid.size(0);
    check(vemd, "vem (data)", torch::kFloat32, -1); check(fb, "B", torch::kFloat32, (long)B * K * N);
    check(eid, "eid", torch::kInt64, N);
    check(vnfd, "vnf (data)", torch::kFloat32, -1); check(node_flows, "node_flows", torch::kFloat32, -1);
    check(node_mars, "node_mars", torch::kFloat32, -1);
    check(fa, "A", torch::kFloat32, (long)B * M * K); check(lz, "lz", torch::kFloat32, (long)B * M);
    check(kd, "kept share differences", torch::kFloat32, -1);
    const long w = (long)B * K;
    TORCH_CHECK(kd.dim() == 2 && kd.size(0) >= vn0 + M && kd.size(1) == w,
                "lowrank mixture: the kept share differences do not hold this node vector's rows");
    TORCH_CHECK(vemd.size(0) >= rows && vemd.size(1) == w, "lowrank mixture: the virtual element buffer does not hold ",
                w, " columns of ", rows, " rows");
    TORCH_CHECK(nid0 >= 0 && vn0 >= 0 && vnfd.size(0) >= vn0 + M && vnfd.size(1) == w && node_flows.size(1) == B &&
                node_flows.size(0) >= nid0 + M && node_mars.size(0) >= nid0 + M,
                "lowrank mixture: the node rows ", nid0, " + ", M, " (virtual ", vn0, ") do not fit the buffers");
    if (grad_a.has_value()) check(*grad_a, "grad A", torch::kFloat32, (long)B * M * K);
    if ((N == 0 && M == 0) || B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = grid_of((long)(N + M) * B);
    const bool k16 = collapse_rank(K, {&kd, &vnfd, &vemd}) == 16;
#define LOWRANK_PRE(GRAD, PTR) if (k16) LOWRANK_PRE_K(GRAD, PTR, 16); else LOWRANK_PRE_K(GRAD, PTR, 0)
#define LOWRANK_PRE_K(GRAD, PTR, KC_)                                                                               \
    backward_pre_kernel<GRAD, FUSED, KC_><<<grid, THREADS, 0, stream>>>(                                            \
        vemd.data_ptr<float>(), (long)vemd.size(1), vals, table, first, fb.data_ptr<float>(),                       \
        eid.data_ptr<int64_t>(), N, vnfd.data_ptr<float>(), node_flows.data_ptr<float>(),                           \
        node_mars.data_ptr<float>(), fa.data_ptr<float>(), lz.data_ptr<float>(), kd.data_ptr<float>(), w, PTR,      \
        (long)nid0, (long)vn0, (int)M, B, (int)K)
    if (grad_a.has_value()) { LOWRANK_PRE(true, grad_a->data_ptr<float>()); }
    else { LOWRANK_PRE(false, nullptr); }
#undef LOWRANK_PRE
#undef LOWRANK_PRE_K
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// A layer's backward before its plain calls: the data half's virtual child values (`vemd`) and virtual node flows
// (`vnfd`), and dLL / dA = f D from the share differences the forward kept (`kd`), one launch
void lowrank_backward_pre(torch::Tensor vemd, torch::Tensor element_mars, torch::Tensor fb, torch::Tensor eid,
                          torch::Tensor vnfd, torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor fa,
                          torch::Tensor lz, torch::Tensor kd, c10::optional<torch::Tensor> grad_a, int64_t nid0,
                          int64_t vn0, int64_t M, int64_t K) {
    check(element_mars, "element_mars", torch::kFloat32, -1);
    TORCH_CHECK(element_mars.size(1) == node_mars.size(1), "lowrank mixture: element and node values of two batches");
    backward_pre_launch<false>(vemd, element_mars.data_ptr<float>(), element_mars.size(0), nullptr, 0, fb, eid, vnfd,
                               node_flows, node_mars, fa, lz, kd, grad_a, nid0, vn0, M, K);
}

// ... with the product layer below fused in: the element values read from their two children's node values
void lowrank_backward_pre_fused(torch::Tensor vemd, torch::Tensor table, int64_t first, torch::Tensor fb,
                                torch::Tensor eid, torch::Tensor vnfd, torch::Tensor node_flows,
                                torch::Tensor node_mars, torch::Tensor fa, torch::Tensor lz, torch::Tensor kd,
                                c10::optional<torch::Tensor> grad_a, int64_t nid0, int64_t vn0, int64_t M, int64_t K) {
    const long rows = check_children(table, first);
    backward_pre_launch<true>(vemd, node_mars.data_ptr<float>(), rows, table.data_ptr<int32_t>(), (long)first, fb,
                              eid, vnfd, node_flows, node_mars, fa, lz, kd, grad_a, nid0, vn0, M, K);
}

namespace {

// The operand launches: a thread a (row of the operand, run of a copy's rows), `RUN` rows a run
int run_grid(long items, long B, long K) { return grid_of(items * ((B * K + RUN - 1) / RUN)); }

// Whether every run is a whole one at a 16-byte boundary: a copy's rows whole runs, the operand's rows too, its storage
// aligned (one vector store a run)
bool whole_runs(const torch::Tensor& op, long B, long K) {
    return (B * K) % RUN == 0 && op.size(1) % RUN == 0 && (reinterpret_cast<uintptr_t>(op.data_ptr()) & 15) == 0;
}

template <typename TO>
void u_operand_launch(torch::Tensor& Ut, torch::Tensor& shift, torch::Tensor& node_flows, std::vector<torch::Tensor>& fa,
                      std::vector<torch::Tensor>& lz, std::vector<int64_t>& nid0, std::vector<int64_t>& row0, int64_t M,
                      int64_t K) {
    const int B = node_flows.size(1);
    const long rows = Ut.size(1), P = (long)fa.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    bool vec = whole_runs(Ut, B, K);
    for (long c = 0; c < P; c++) vec = vec && row0[c] % RUN == 0;
    for (long p0 = 0; p0 < P; p0 += COPIES_PER_LAUNCH) {
        const int n = (int)std::min<long>(COPIES_PER_LAUNCH, P - p0);
        UCopies copies;
        for (int i = 0; i < n; i++) {
            const long c = p0 + i;
            check(fa[c], "A", torch::kFloat32, (long)B * M * K); check(lz[c], "lz", torch::kFloat32, (long)B * M);
            TORCH_CHECK(nid0[c] >= 0 && node_flows.size(0) >= nid0[c] + M, "lowrank mixture: a copy's node rows");
            TORCH_CHECK(row0[c] >= 0 && row0[c] + (long)B * K <= rows, "lowrank mixture: a copy's rows");
            copies.rows[4 * i] = (long long)fa[c].data_ptr<float>();
            copies.rows[4 * i + 1] = (long long)lz[c].data_ptr<float>();
            copies.rows[4 * i + 2] = (long long)nid0[c];
            copies.rows[4 * i + 3] = (long long)row0[c];
        }
        u_operand_kernel<TO><<<dim3(run_grid(M, B, K), 1, n), THREADS, 0, stream>>>(
            copies, static_cast<TO*>(Ut.data_ptr()), shift.data_ptr<float>(), rows, node_flows.data_ptr<float>(), (int)M,
            B, (int)K, vec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// `of_a`: the factors are A [M, K, B] (the backward's `exp(A - shift)`), else B [K, N, B]
template <typename TO>
void z_operand_launch(torch::Tensor& Vt, torch::Tensor& shift, std::vector<torch::Tensor>& fb, int64_t B, int64_t K,
                      bool of_a) {
    const int N = Vt.size(0);
    const long sx = of_a ? K * B : B, sr = of_a ? B : (long)N * B;
    const long rows = Vt.size(1), P = (long)fb.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    for (long p0 = 0; p0 < P; p0 += COPIES_PER_LAUNCH) {
        const int n = (int)std::min<long>(COPIES_PER_LAUNCH, P - p0);
        ZCopies copies;
        for (int i = 0; i < n; i++) {
            check(fb[p0 + i], of_a ? "A" : "B", torch::kFloat32, (long)B * K * N);
            copies.fb[i] = (long long)fb[p0 + i].data_ptr<float>();
        }
        z_operand_kernel<TO><<<dim3((unsigned)((B + 7) / 8), (unsigned)((K + 3) / 4), (unsigned)n), OPERAND_THREADS,
                               0, stream>>>(
            copies, static_cast<TO*>(Vt.data_ptr()), shift.data_ptr<float>(), rows, p0 * K * B, N, (int)B, (int)K, sx,
            sr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <typename TO>
void data_operands_launch(torch::Tensor& Ut, torch::Tensor& Vt, torch::Tensor& node_flows, torch::Tensor& node_mars,
                          std::vector<torch::Tensor>& fa, std::vector<torch::Tensor>& lz,
                          std::vector<torch::Tensor>& fb, std::vector<int64_t>& nid0, std::vector<torch::Tensor>& eid,
                          std::vector<torch::Tensor>& table, std::vector<int64_t>& first, int64_t M, int64_t K) {
    const int B = node_flows.size(1), N = Vt.size(0);
    const long rows = Ut.size(1), P = (long)fa.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool vec = whole_runs(Ut, B, K) && whole_runs(Vt, B, K);
    // the rows' shifts of a launch
    torch::Tensor shd = torch::empty({(long)std::min<long>(COPIES_PER_LAUNCH, P) * B * K}, node_flows.options());
    for (long p0 = 0; p0 < P; p0 += COPIES_PER_LAUNCH) {
        const int n = (int)std::min<long>(COPIES_PER_LAUNCH, P - p0);
        DataCopies copies;
        for (int i = 0; i < n; i++) {
            const long c = p0 + i;
            check(fa[c], "A", torch::kFloat32, (long)B * M * K); check(lz[c], "lz", torch::kFloat32, (long)B * M);
            check(fb[c], "B", torch::kFloat32, (long)B * K * N); check(eid[c], "eid", torch::kInt64, N);
            check(table[c], "table", torch::kInt32, -1);
            TORCH_CHECK(nid0[c] >= 0 && node_flows.size(0) >= nid0[c] + M, "lowrank mixture: a copy's node rows");
            long long* row = copies.rows + 7 * i;
            row[0] = (long long)fa[c].data_ptr<float>(); row[1] = (long long)lz[c].data_ptr<float>();
            row[2] = (long long)fb[c].data_ptr<float>(); row[3] = (long long)nid0[c];
            row[4] = (long long)eid[c].data_ptr<int64_t>(); row[5] = (long long)table[c].data_ptr<int32_t>();
            row[6] = (long long)first[c];
        }
        data_shift_kernel<<<dim3((B + 31) / 32, (int)K, n), OPERAND_THREADS, 0, stream>>>(
            copies, shd.data_ptr<float>(), node_mars.data_ptr<float>(), N, B, (int)K);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        data_operands_kernel<TO><<<dim3(run_grid(N + M, B, K), 1, n), THREADS, 0, stream>>>(
            copies, static_cast<TO*>(Ut.data_ptr()), static_cast<TO*>(Vt.data_ptr()), shd.data_ptr<float>(), rows,
            p0 * K * B, node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), (int)M, N, B, (int)K, vec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <typename TO>
void normalizer_grads_launch(std::vector<c10::optional<torch::Tensor>>& grads, std::vector<int64_t>& row0,
                             torch::Tensor& Vt, torch::Tensor& Gt, int64_t B, int64_t K) {
    const int N = Vt.size(0);
    const long rows = Vt.size(1), P = (long)grads.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    // (a sample's ranks whole runs of RUN at 16-byte boundaries)
    bool vec = K % RUN == 0 && whole_runs(Vt, B, K) && whole_runs(Gt, B, K);
    for (long c = 0; c < P; c++) vec = vec && row0[c] % RUN == 0;
    for (long p0 = 0; p0 < P; ) {
        NormalizerGrads at_copy;
        int n = 0;
        for (; n < COPIES_PER_LAUNCH && p0 + n < P; n++) {
            const c10::optional<torch::Tensor>& g = grads[p0 + n];
            if (g.has_value()) check(*g, "grad B", torch::kFloat32, (long)B * K * N);
            TORCH_CHECK(row0[p0 + n] >= 0 && row0[p0 + n] + (long)B * K <= rows, "lowrank mixture: a copy's rows");
            const long long where = g.has_value() ? (long long)g->data_ptr<float>() : 0;
            // A buffer two copies share (`tie_external`) takes their subtractions a launch each: in one launch they
            // would race (load, subtract, store), the kernel's blocks being unordered
            bool shared = false;
            for (int i = 0; i < n && where != 0; i++) shared = shared || at_copy.at[2 * i] == where;
            if (shared) break;
            at_copy.at[2 * n] = where;
            at_copy.at[2 * n + 1] = (long long)row0[p0 + n];
        }
        normalizer_grads_kernel<TO><<<dim3(grid_of((long)N * B), 1, (unsigned)n), THREADS, 0, stream>>>(
            at_copy, static_cast<const TO*>(Vt.data_ptr()), Gt.data_ptr<float>(), rows, N, (int)B, (int)K, vec);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        p0 += n;
    }
}

}  // namespace

// The normalizer half's `Ut` [M, rows] of a node vector's copies, float32 or bfloat16 (as allocated), `shift` [rows] the
// forward's (`lowrank_z_operand`); from each copy's node flows (rows `nid0[c]` on), factors and kept `log Z`, into its rows
// from `row0[c]` on
void lowrank_u_operand(torch::Tensor Ut, torch::Tensor shift, torch::Tensor node_flows, std::vector<torch::Tensor> fa,
                       std::vector<torch::Tensor> lz, std::vector<int64_t> nid0, std::vector<int64_t> row0, int64_t M,
                       int64_t K) {
    check(node_flows, "node_flows", torch::kFloat32, -1);
    TORCH_CHECK(Ut.is_cuda() && Ut.is_contiguous() && Ut.dim() == 2 && Ut.size(0) == M && fa.size() == lz.size()
                && fa.size() == nid0.size() && fa.size() == row0.size(),
                "lowrank mixture: the normalizer operands do not match the copies");
    check(shift, "shift", torch::kFloat32, Ut.size(1));
    if (Ut.numel() == 0 || fa.empty()) return;
    if (Ut.scalar_type() == torch::kBFloat16) u_operand_launch<__nv_bfloat16>(Ut, shift, node_flows, fa, lz, nid0, row0, M, K);
    else if (Ut.scalar_type() == torch::kFloat32) u_operand_launch<float>(Ut, shift, node_flows, fa, lz, nid0, row0, M, K);
    else TORCH_CHECK(false, "lowrank mixture: the normalizer operands must be float32 or bfloat16");
}

// The forward's normalizers of a node vector's copies (`fb`, their `B`): `Vt` [N, copies B K], float32 or bfloat16 (as
// allocated), and `shift` [copies B K] float32, each row's largest `B`; a copy's rows `(b, r)`
void lowrank_z_operand(torch::Tensor Vt, torch::Tensor shift, std::vector<torch::Tensor> fb, int64_t B, int64_t K) {
    const long rows = (long)fb.size() * B * K;
    check(shift, "shift", torch::kFloat32, rows);
    TORCH_CHECK(Vt.is_cuda() && Vt.is_contiguous() && Vt.dim() == 2 && Vt.size(1) == rows,
                "lowrank mixture: `Vt` does not match the copies");
    if (rows == 0) return;
    if (Vt.scalar_type() == torch::kBFloat16) z_operand_launch<__nv_bfloat16>(Vt, shift, fb, B, K, false);
    else if (Vt.scalar_type() == torch::kFloat32) z_operand_launch<float>(Vt, shift, fb, B, K, false);
    else TORCH_CHECK(false, "lowrank mixture: `Vt` must be float32 or bfloat16");
}

// The backward's `Ut` [M, copies B K] = exp(A - shift), bf16 or fp32 (as allocated), `shift` [copies B K] each row's
// largest A over the nodes: the one-launch backward's operand (`lowrank_stream.cu`), the rows as the normalizers'
void lowrank_a_operand(torch::Tensor Ut, torch::Tensor shift, std::vector<torch::Tensor> fa, int64_t B, int64_t K) {
    const long rows = (long)fa.size() * B * K;
    check(shift, "shift", torch::kFloat32, rows);
    TORCH_CHECK(Ut.is_cuda() && Ut.is_contiguous() && Ut.dim() == 2 && Ut.size(1) == rows,
                "lowrank mixture: `Ut` does not match the copies");
    if (rows == 0) return;
    if (Ut.scalar_type() == torch::kBFloat16) z_operand_launch<__nv_bfloat16>(Ut, shift, fa, B, K, true);
    else if (Ut.scalar_type() == torch::kFloat32) z_operand_launch<float>(Ut, shift, fa, B, K, true);
    else TORCH_CHECK(false, "lowrank mixture: `Ut` must be float32 or bfloat16");
}

// The data half's operands of the copies whose product layer is fused in, as `lowrank_u_operand`'s: `eid` each
// copy's children's element rows, `table` / `first` its product layer's children table and first element row
void lowrank_data_operands(torch::Tensor Ut, torch::Tensor Vt, torch::Tensor node_flows, torch::Tensor node_mars,
                           std::vector<torch::Tensor> fa, std::vector<torch::Tensor> lz, std::vector<torch::Tensor> fb,
                           std::vector<int64_t> nid0, std::vector<torch::Tensor> eid, std::vector<torch::Tensor> table,
                           std::vector<int64_t> first, int64_t M, int64_t K) {
    check(node_flows, "node_flows", torch::kFloat32, -1); check(node_mars, "node_mars", torch::kFloat32, node_flows.numel());
    const long rows = (long)fa.size() * node_flows.size(1) * K;
    TORCH_CHECK(Ut.is_cuda() && Ut.is_contiguous() && Vt.is_cuda() && Vt.is_contiguous()
                && Ut.scalar_type() == Vt.scalar_type() && Ut.dim() == 2 && Vt.dim() == 2 && Ut.size(1) == rows
                && Vt.size(1) == rows && Ut.size(0) == M && fa.size() == lz.size() && fa.size() == fb.size()
                && fa.size() == nid0.size() && fa.size() == eid.size() && fa.size() == table.size()
                && fa.size() == first.size(), "lowrank mixture: the data operands do not match the copies");
    if (rows == 0) return;
    if (Ut.scalar_type() == torch::kBFloat16)
        data_operands_launch<__nv_bfloat16>(Ut, Vt, node_flows, node_mars, fa, lz, fb, nid0, eid, table, first, M, K);
    else if (Ut.scalar_type() == torch::kFloat32)
        data_operands_launch<float>(Ut, Vt, node_flows, node_mars, fa, lz, fb, nid0, eid, table, first, M, K);
    else TORCH_CHECK(false, "lowrank mixture: the data operands must be float32 or bfloat16");
}

// Each copy's `dB` [K, N, B] minus `Vt * Gt` on its rows (`Gt` [N, copies B K] float32: `theta^T U`); None: a copy
// without a gradient
void lowrank_normalizer_grads(std::vector<c10::optional<torch::Tensor>> grads, std::vector<int64_t> row0, torch::Tensor Vt,
                              torch::Tensor Gt, int64_t B, int64_t K) {
    check(Gt, "theta^T U", torch::kFloat32, Vt.numel());
    TORCH_CHECK(Vt.is_cuda() && Vt.is_contiguous() && Vt.dim() == 2 && Gt.dim() == 2 && Gt.size(0) == Vt.size(0)
                && grads.size() == row0.size(), "lowrank mixture: `Vt` does not match the copies");
    if (Vt.scalar_type() == torch::kBFloat16) normalizer_grads_launch<__nv_bfloat16>(grads, row0, Vt, Gt, B, K);
    else if (Vt.scalar_type() == torch::kFloat32) normalizer_grads_launch<float>(grads, row0, Vt, Gt, B, K);
    else TORCH_CHECK(false, "lowrank mixture: `Vt` must be float32 or bfloat16");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_expand", &lowrank_expand, "virtual child values: element values + B, a column per (sample, rank)");
    m.def("lowrank_expand_fused", &lowrank_expand_fused,
          "virtual child values with the product layer below fused in: each the sum of its two children's node values + B");
    m.def("lowrank_collapse", &lowrank_collapse,
          "node values from the virtual ones: logsumexp over ranks with A, - log Z (from the normalizers' product)");
    m.def("lowrank_collapse_expand", &lowrank_collapse_expand,
          "`lowrank_collapse` of a node vector and `lowrank_expand_fused` of the next layer's, one launch");
    m.def("lowrank_collapse_bw", &lowrank_collapse_bw,
          "element flows from the virtual ones (and the data half's part of dLL/dB)");
    m.def("lowrank_backward_pre", &lowrank_backward_pre,
          "the backward's virtual child values and virtual node flows (and dLL/dA from the kept responsibilities)");
    m.def("lowrank_backward_pre_fused", &lowrank_backward_pre_fused,
          "`lowrank_backward_pre` with the element values read from their two children's node values");
    m.def("lowrank_collapse_bw_fused", &lowrank_collapse_bw_fused,
          "`lowrank_collapse_bw` with the element flows written as their two children's node flows");
    m.def("lowrank_u_operand", &lowrank_u_operand, "the normalizer half's U^T of a node vector's copies, at the forward's shifts");
    m.def("lowrank_z_operand", &lowrank_z_operand, "the forward's normalizers: V^T of a node vector's copies, each row's shift");
    m.def("lowrank_a_operand", &lowrank_a_operand,
          "the backward's exp(A - shift) of a node vector's copies (the one-launch backward's operand), each row's shift");
    m.def("lowrank_data_operands", &lowrank_data_operands,
          "the data half's U^T and V^T of the copies whose product layer is fused in, for F+ as one matrix product");
    m.def("lowrank_normalizer_grads", &lowrank_normalizer_grads, "each copy's dLL/dB minus V * (U theta) on its rows");
}
