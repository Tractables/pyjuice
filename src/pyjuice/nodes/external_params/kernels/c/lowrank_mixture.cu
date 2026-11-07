// The elementwise halves of `LowRankSumParams`' "sum_exp" form computed on a VIRTUAL BATCH.
//
// The mixture scale `g_b[i, j] = sum_r exp(A_b[i, r]) exp(B_b[r, j])` makes the node's unnormalized value and its
// normalizer
//
//     S_b[i] = sum_r exp(A_b[i, r]) * sum_j theta[i, j] * exp(B_b[r, j]) * c_b[j]
//     Z_b[i] = sum_r exp(A_b[i, r]) * sum_j theta[i, j] * exp(B_b[r, j])
//
// each of which is, per rank r, a plain sum node: over the children's values rescaled by `exp(B_b[r, :])`, and over
// `exp(B_b[r, :])` alone. So every (sample, rank) pair is two columns of a virtual batch -- a DATA column and a
// NORMALIZER column -- the plain sum-layer kernels run on it unchanged, the shared parameters read once for all of
// them, and what is specific to the parameterization is elementwise:
//
//     expand        data:        vem[child, b K + r] = element_mars[child, b] + B_b[r, child]
//                   normalizer:  vem[child, b K + r] = B_b[r, child]
//     (plain forward on the virtual buffers:  vnm[node, v] = log sum_j theta[node, j] exp(vem[j, v]))
//     collapse      node_mars[node, b] = logsumexp_r(A_b[node, r] + vnm_data[node, b K + r])            (log S)
//                                      - logsumexp_r(A_b[node, r] + vnm_norm[node, b K + r])            (log Z)
//
// and backward, with the virtual node values taken as 0 (a virtual node's flow is then the sum of its edges' flows
// whatever its value was), the DATA half alone through the plain kernels:
//
//     expand_bw     vnf[node, b K + r] = log f_b[node] + A_b[node, r] - log S_b[node]
//     (plain backward: virtual element flows `vef` -- stored, whatever the buffer held --; the parameter flows `F+`)
//     collapse_bw   element_flows[child, b] = logsumexp_r vef[child, b K + r]
//
// with the factors' gradients beside: `dA` from the two responsibilities the forward kept (each rank's share of S and
// of Z), the data half's part of `dB` from the virtual element flows.
//
// THE NORMALIZER HALF reads no child values: for a node vector and all of its tied copies at once, at the end of the
// pass, it is two matrix products over the rows `v = (copy, sample, rank)`,
//
//     U[v, i] = f_b[i] exp(A_b[i, r]) / Z_b[i]        V[v, j] = exp(B_b[r, j])
//     W = U^T V             (the denominator statistic, `F- = theta * W`, free of the parameters)
//     dB[v, j] -= V[v, j] * (U theta)[v, j]           (the normalizer half's part of dLL / dB)
//
// `normalizer_operands` writes U and V -- row `v` shifted by `max_j B_b[r, :]`, the shift multiplying U's row and
// dividing V's, so V <= 1 and U stays within fp32 (log f + A - log Z + max B <= -log theta_min); the products are
// the caller's (cuBLAS, bf16 operands with fp32 accumulation or fp32), `normalizer_grads` takes `V * (U theta)` off
// each copy's `dB`.
//
// Layouts: the circuit's buffers `[rows, batch]`, batch innermost; the virtual ones `[rows, columns]`, a sample's K
// ranks adjacent from column `b K` of a half; the factors as staged, `A [batch, M, K]`, `B [batch, K, N]`; per-node
// state `[batch, M]` and `[batch, M, K]`. One thread a (row, sample), its K ranks in a loop: every read and write of
// the virtual buffers and of `A` is K adjacent floats.

#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

constexpr int THREADS = 256;

inline int grid_of(long total) { return (int)((total + THREADS - 1) / THREADS); }

// `expand` for thread `t` (a child, a sample). `vd` / `vz`: the data and the normalizer half's buffers (one buffer in
// the forward, the normalizer's columns after the data's), `cd` / `cz` columns wide, the half starting at column `od`
// / `oz`, a child's row `eid[j]` in the data half's and `eidz[j]` in the normalizer's (another copy's rows where that
// one hosts the normalizer half). `vals`: the element values -- or, FUSED (the product layer below fused in), the
// NODE values: a child's value read as the sum of its two children's (`table[e - first]`, the product group's rows of
// child node rows; a product of one child has the dummy node, of value 0, as its second) -- what the product
// layer's forward would have written to `element_mars`, which is then neither launched nor read.
template <bool FUSED>
inline __device__ void expand_at(long t, float* vd, long cd, long od, float* vz, long cz, long oz, const float* vals,
                                 const int32_t* table, long first, const float* fb, const int64_t* eid,
                                 const int64_t* eidz, int N, int B, int K) {
    const int b = (int)(t % B);
    const int j = (int)(t / B);
    const long e = eid[j];
    if (e < 0) return;                                                  // a child no edge reaches
    const long ez = (vz == nullptr) ? 0 : eidz[j];

    float c;
    if (FUSED) {
        const int32_t* ch = table + (e - first) * 2;
        c = vals[(long)ch[0] * B + b] + vals[(long)ch[1] * B + b];
    } else {
        c = vals[e * B + b];
    }
    float* outd = vd + e * cd + od + (long)b * K;
    const float* src = fb + (long)b * K * N + j;
    if (vz == nullptr) {                                                // (the backward: the data half alone)
        for (int r = 0; r < K; r++) outd[r] = c + src[(long)r * N];
        return;
    }
    float* outz = vz + ez * cz + oz + (long)b * K;
    for (int r = 0; r < K; r++) {
        const float v = src[(long)r * N];
        outd[r] = c + v;
        outz[r] = v;
    }
}

template <bool FUSED>
__global__ void expand_kernel(float* vd, long cd, long od, float* vz, long cz, long oz, const float* __restrict__ vals,
                              const int32_t* __restrict__ table, long first, const float* __restrict__ fb,
                              const int64_t* __restrict__ eid, const int64_t* __restrict__ eidz, int N, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long)N * B) expand_at<FUSED>(t, vd, cd, od, vz, cz, oz, vals, table, first, fb, eid, eidz, N, B, K);
}

inline __device__ float lse_with(const float* __restrict__ a, const float* __restrict__ y, int K) {
    float m = -INFINITY;
    for (int r = 0; r < K; r++) m = fmaxf(m, a[r] + y[r]);
    if (!(m > -INFINITY)) return -INFINITY;
    float s = 0.0f;
    for (int r = 0; r < K; r++) s += expf(a[r] + y[r] - m);
    return m + logf(s);
}

__global__ void collapse_kernel(float* __restrict__ node_mars, const float* __restrict__ vnm, long cols,
                                const float* __restrict__ fa, float* __restrict__ lz, float* __restrict__ lrn,
                                float* __restrict__ lrz, long nid0, int M, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)M * B) return;
    const int b = (int)(t % B);
    const int i = (int)(t / B);

    const float* yd = vnm + (nid0 + i) * cols + (long)b * K;            // the data half's virtual node values
    const float* yz = yd + (long)B * K;                                 // the normalizer half's
    const long at = ((long)b * M + i) * K;
    const float* a = fa + at;
    const float ls = lse_with(a, yd, K);
    const float z = lse_with(a, yz, K);
    node_mars[(nid0 + i) * B + b] = (ls > -INFINITY) ? ls - z : -INFINITY;
    lz[(long)b * M + i] = z;

    // log of each rank's share of S and of Z: what the backward splits the node's flow by
    for (int r = 0; r < K; r++) {
        lrn[at + r] = (ls > -INFINITY) ? a[r] + yd[r] - ls : -INFINITY;
        lrz[at + r] = (z > -INFINITY) ? a[r] + yz[r] - z : -INFINITY;
    }
}

// The forward's `collapse` of a node vector and the `expand` of the NEXT layer's children (the product layer between
// fused in), ONE launch: the first `M B` threads are `collapse_kernel`'s, the `N B` after them the fused `expand`'s
// for a node vector of the next layer (rank `KC`, its forward buffer `vem`, `cw` wide: the data half's columns, then
// the normalizer half's). A product's child that is one of THIS node vector's is not read from `node_mars` -- which
// the other threads are writing -- but collapsed again from the virtual values: the same operations, the same bits.
// Every other child is a row computed before this layer (the caller's condition).
__global__ void collapse_expand_kernel(float* node_mars, const float* __restrict__ vnm, long cols,
                                       const float* __restrict__ fa, float* __restrict__ lz, float* __restrict__ lrn,
                                       float* __restrict__ lrz, long nid0, int M, int K,
                                       float* __restrict__ vem, long cw, const int32_t* __restrict__ table, long first,
                                       const float* __restrict__ fb, const int64_t* __restrict__ eid, int N, int KC,
                                       int B) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long)M * B) {
        const int b = (int)(t % B);
        const int i = (int)(t / B);
        const float* yd = vnm + (nid0 + i) * cols + (long)b * K;
        const float* yz = yd + (long)B * K;
        const long at = ((long)b * M + i) * K;
        const float* a = fa + at;
        const float ls = lse_with(a, yd, K);
        const float z = lse_with(a, yz, K);
        node_mars[(nid0 + i) * B + b] = (ls > -INFINITY) ? ls - z : -INFINITY;
        lz[(long)b * M + i] = z;
        for (int r = 0; r < K; r++) {
            lrn[at + r] = (ls > -INFINITY) ? a[r] + yd[r] - ls : -INFINITY;
            lrz[at + r] = (z > -INFINITY) ? a[r] + yz[r] - z : -INFINITY;
        }
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
            const float* yd = vnm + row * cols + (long)b * K;
            const float* yz = yd + (long)B * K;
            const float* a = fa + ((long)b * M + (row - nid0)) * K;
            const float ls = lse_with(a, yd, K);
            const float z = lse_with(a, yz, K);
            v[k] = (ls > -INFINITY) ? ls - z : -INFINITY;
        } else {
            v[k] = node_mars[row * B + b];
        }
    }
    const float c = v[0] + v[1];
    float* outd = vem + e * cw + (long)b * KC;
    float* outz = outd + (long)B * KC;
    const float* src = fb + (long)b * KC * N + j;
    for (int r = 0; r < KC; r++) {
        const float w = src[(long)r * N];
        outd[r] = c + w;
        outz[r] = w;
    }
}

// `expand_bw` for thread `t` (a node, a sample); GRAD: and dLL / dA from the kept shares
template <bool GRAD>
inline __device__ void expand_bw_at(long t, float* vnfd, float* vnfz, long cz, long oz, long nid0z,
                                    const float* node_flows, const float* node_mars, const float* fa,
                                    const float* lz, const float* lrn, const float* lrz, float* grad_a, long nid0,
                                    int M, int B, int K) {
    const int b = (int)(t % B);
    const int i = (int)(t / B);

    const long row = nid0 + i;
    const long at = ((long)b * M + i) * K;
    const float lf = node_flows[row * B + b];
    const float z = lz[(long)b * M + i];
    const float ls = node_mars[row * B + b] + z;                        // log S = log n + log Z
    float* outd = vnfd + (row * (long)B + b) * K;
    const bool live = (lf > -INFINITY) && (ls > -INFINITY);
    for (int r = 0; r < K; r++) outd[r] = live ? lf + fa[at + r] - ls : -INFINITY;
    if (vnfz != nullptr) {                                              // (the normalizer half, where it is wanted)
        float* outz = vnfz + (nid0z + i) * cz + oz + (long)b * K;
        for (int r = 0; r < K; r++) outz[r] = live ? lf + fa[at + r] - z : -INFINITY;
    }

    if (GRAD && live) {
        // dLL / dA[i, r] = f * (the rank's share of S - its share of Z)
        for (int r = 0; r < K; r++) grad_a[at + r] += expf(lf + lrn[at + r]) - expf(lf + lrz[at + r]);
    }
}

// A layer's backward before its plain calls, ONE launch: the first `N B` threads `expand` (the virtual child values
// of the data half), the `M B` after them `expand_bw` (the virtual node flows of the data half, `dA`). Neither reads
// what the other writes.
template <bool GRAD, bool FUSED>
__global__ void backward_pre_kernel(float* vd, long cd, const float* vals, const int32_t* __restrict__ table,
                                    long first, const float* __restrict__ fb, const int64_t* __restrict__ eid, int N,
                                    float* __restrict__ vnfd, const float* __restrict__ node_flows,
                                    const float* node_mars, const float* __restrict__ fa,
                                    const float* __restrict__ lz, const float* __restrict__ lrn,
                                    const float* __restrict__ lrz, float* __restrict__ grad_a, long nid0, int M,
                                    int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (long)N * B) {
        expand_at<FUSED>(t, vd, cd, 0, nullptr, 0, 0, vals, table, first, fb, eid, nullptr, N, B, K);
    } else if (t - (long)N * B < (long)M * B) {
        expand_bw_at<GRAD>(t - (long)N * B, vnfd, nullptr, 0, 0, 0, node_flows, node_mars, fa, lz, lrn, lrz,
                           grad_a, nid0, M, B, K);
    }
}

// GRAD: and the data half's part of dLL / dB. FUSED: the product layer below fused in -- `flows` are the NODE flows,
// and an element's flow is written as its two children's (`table[e - first]`): what the product layer's backward
// writes for children of one parent element each, which is then not launched.
template <bool GRAD, bool FUSED>
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
    float m = -INFINITY;
    for (int r = 0; r < K; r++) m = fmaxf(m, f[r]);
    float lef = -INFINITY;
    if (m > -INFINITY) {
        float s = 0.0f;
        for (int r = 0; r < K; r++) s += expf(f[r] - m);
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
        float* g = grad_b + (long)b * K * N + j;
        for (int r = 0; r < K; r++) g[(long)r * N] += expf(f[r]);
    }
}

inline __device__ float as_float(float x) { return x; }
inline __device__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }
inline __device__ void store_as(float* p, float v) { *p = v; }
inline __device__ void store_as(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// The normalizer half's operands of one copy, rows `row0 + b K + r` of `U` [rows, M] and `V` [rows, N]: one block a
// row -- the row's shift `max_j B_b[r, j]` by a block reduction, then V = exp(B - shift), U = f exp(A - log Z + shift)
// (0 where the node has no flow)
template <typename TO>
__global__ void normalizer_rows_kernel(TO* __restrict__ U, TO* __restrict__ V, long row0,
                                       const float* __restrict__ node_flows, long nid0, const float* __restrict__ fa,
                                       const float* __restrict__ lz, const float* __restrict__ fb, int M, int N,
                                       int B, int K) {
    __shared__ float part[THREADS / 32];
    const int v = blockIdx.x;                                           // b K + r
    const int b = v / K, r = v % K;
    const float* brow = fb + ((long)b * K + r) * N;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, brow[j]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = (threadIdx.x < blockDim.x / 32) ? part[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (threadIdx.x == 0) part[0] = m;
    }
    __syncthreads();
    const float shift = part[0];
    TO* vrow = V + (row0 + v) * (long)N;
    for (int j = threadIdx.x; j < N; j += blockDim.x) store_as(vrow + j, expf(brow[j] - shift));
    TO* urow = U + (row0 + v) * (long)M;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        const float lf = node_flows[(nid0 + i) * (long)B + b];
        const float z = lz[(long)b * M + i];
        const bool live = (lf > -INFINITY) && (z < INFINITY);
        store_as(urow + i, live ? expf(lf + fa[((long)b * M + i) * K + r] - z + shift) : 0.0f);
    }
}

// dB_b[r, j] -= V[row0 + b K + r, j] * G[row0 + b K + r, j]: a copy's normalizer part off its gradient
template <typename TO>
__global__ void normalizer_grads_kernel(float* __restrict__ grad_b, const TO* __restrict__ V,
                                        const float* __restrict__ G, long row0, long n, int N) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const long at = row0 * (long)N + t;
    grad_b[t] -= as_float(V[at]) * G[at];
}

void check(const torch::Tensor& t, const char* name, torch::ScalarType st, long numel) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "lowrank mixture: `", name, "` must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == st, "lowrank mixture: `", name, "` has the wrong dtype");
    TORCH_CHECK(numel < 0 || t.numel() == numel, "lowrank mixture: `", name, "` has ", t.numel(),
                " elements, expected ", numel);
}

}  // namespace


namespace {

// What `expand` is given: `rows`, the element rows its buffers must hold
void check_expand(const torch::Tensor& vd, int64_t od, const torch::Tensor& vz, int64_t oz, const torch::Tensor& fb,
                  const torch::Tensor& eid, const torch::Tensor& eidz, long rows, int B, int N, int64_t K) {
    check(vd, "vem (data)", torch::kFloat32, -1); check(vz, "vem (normalizer)", torch::kFloat32, -1);
    check(fb, "B", torch::kFloat32, (long)B * K * N); check(eid, "eid", torch::kInt64, N);
    check(eidz, "eid (normalizer)", torch::kInt64, N);
    const long w = (long)B * K;
    TORCH_CHECK(vd.size(0) >= rows && vz.size(0) >= rows && od >= 0 && oz >= 0 && od + w <= vd.size(1)
                && oz + w <= vz.size(1),
                "lowrank mixture: the virtual element buffers do not hold ", w, " columns at ", od, " and ", oz);
}

long check_children(const torch::Tensor& table, int64_t first) {
    check(table, "children", torch::kInt32, -1);
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    return first + table.size(0);
}

}  // namespace

void lowrank_expand(torch::Tensor vd, int64_t od, torch::Tensor vz, int64_t oz, torch::Tensor element_mars,
                    torch::Tensor fb, torch::Tensor eid, torch::Tensor eidz, int64_t K) {
    const int B = element_mars.size(1), N = eid.size(0);
    check(element_mars, "element_mars", torch::kFloat32, -1);
    check_expand(vd, od, vz, oz, fb, eid, eidz, element_mars.size(0), B, N, K);
    if (N == 0 || B == 0) return;
    expand_kernel<false><<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        vd.data_ptr<float>(), (long)vd.size(1), (long)od, vz.data_ptr<float>(), (long)vz.size(1), (long)oz,
        element_mars.data_ptr<float>(), nullptr, 0, fb.data_ptr<float>(), eid.data_ptr<int64_t>(),
        eidz.data_ptr<int64_t>(), N, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lowrank_expand_fused(torch::Tensor vd, int64_t od, torch::Tensor vz, int64_t oz, torch::Tensor node_mars,
                          torch::Tensor table, int64_t first, torch::Tensor fb, torch::Tensor eid, torch::Tensor eidz,
                          int64_t K) {
    const int B = node_mars.size(1), N = eid.size(0);
    check(node_mars, "node_mars", torch::kFloat32, -1);
    check_expand(vd, od, vz, oz, fb, eid, eidz, check_children(table, first), B, N, K);
    if (N == 0 || B == 0) return;
    expand_kernel<true><<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        vd.data_ptr<float>(), (long)vd.size(1), (long)od, vz.data_ptr<float>(), (long)vz.size(1), (long)oz,
        node_mars.data_ptr<float>(), table.data_ptr<int32_t>(), (long)first, fb.data_ptr<float>(),
        eid.data_ptr<int64_t>(), eidz.data_ptr<int64_t>(), N, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lowrank_collapse(torch::Tensor node_mars, torch::Tensor vnm, torch::Tensor fa, torch::Tensor lz,
                      torch::Tensor lrn, torch::Tensor lrz, int64_t nid0, int64_t M, int64_t K) {
    const int B = node_mars.size(1);
    check(node_mars, "node_mars", torch::kFloat32, -1); check(vnm, "vnm", torch::kFloat32, -1);
    check(fa, "A", torch::kFloat32, (long)B * M * K); check(lz, "lz", torch::kFloat32, (long)B * M);
    check(lrn, "lrn", torch::kFloat32, (long)B * M * K); check(lrz, "lrz", torch::kFloat32, (long)B * M * K);
    TORCH_CHECK(vnm.size(0) >= nid0 + M && vnm.size(1) == 2 * (long)B * K && node_mars.size(0) >= nid0 + M,
                "lowrank mixture: the node rows ", nid0, " + ", M, " do not fit the buffers");
    if (M == 0 || B == 0) return;
    collapse_kernel<<<grid_of((long)M * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), vnm.data_ptr<float>(), (long)vnm.size(1), fa.data_ptr<float>(),
        lz.data_ptr<float>(), lrn.data_ptr<float>(), lrz.data_ptr<float>(), (long)nid0, (int)M, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// `lowrank_collapse` of a node vector and `lowrank_expand_fused` of a node vector of the next layer (its factors `fb`
// of rank `KC`, its children `eid`, its forward buffer `vem`), one launch; the next layer's children that are this
// node vector's are taken from the collapse itself, every other one from `node_mars`.
void lowrank_collapse_expand(torch::Tensor node_mars, torch::Tensor vnm, torch::Tensor fa, torch::Tensor lz,
                             torch::Tensor lrn, torch::Tensor lrz, int64_t nid0, int64_t M, int64_t K,
                             torch::Tensor vem, torch::Tensor table, int64_t first, torch::Tensor fb,
                             torch::Tensor eid, int64_t KC) {
    const int B = node_mars.size(1), N = eid.size(0);
    check(node_mars, "node_mars", torch::kFloat32, -1); check(vnm, "vnm", torch::kFloat32, -1);
    check(fa, "A", torch::kFloat32, (long)B * M * K); check(lz, "lz", torch::kFloat32, (long)B * M);
    check(lrn, "lrn", torch::kFloat32, (long)B * M * K); check(lrz, "lrz", torch::kFloat32, (long)B * M * K);
    TORCH_CHECK(vnm.size(0) >= nid0 + M && vnm.size(1) == 2 * (long)B * K && node_mars.size(0) >= nid0 + M,
                "lowrank mixture: the node rows ", nid0, " + ", M, " do not fit the buffers");
    check(vem, "vem", torch::kFloat32, -1); check(table, "children", torch::kInt32, -1);
    check(fb, "B", torch::kFloat32, (long)B * KC * N); check(eid, "eid", torch::kInt64, N);
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    TORCH_CHECK(vem.size(0) >= first + table.size(0) && vem.size(1) == 2 * (long)B * KC,
                "lowrank mixture: the virtual element buffer does not hold the next layer's columns");
    if ((M == 0 && N == 0) || B == 0) return;
    collapse_expand_kernel<<<grid_of((long)(M + N) * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), vnm.data_ptr<float>(), (long)vnm.size(1), fa.data_ptr<float>(),
        lz.data_ptr<float>(), lrn.data_ptr<float>(), lrz.data_ptr<float>(), (long)nid0, (int)M, (int)K,
        vem.data_ptr<float>(), (long)vem.size(1), table.data_ptr<int32_t>(), (long)first, fb.data_ptr<float>(),
        eid.data_ptr<int64_t>(), N, (int)KC, B);
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
    if (grad_b.has_value()) {
        check(*grad_b, "grad B", torch::kFloat32, (long)B * K * N);
        collapse_bw_kernel<true, FUSED><<<grid, THREADS, 0, stream>>>(
            flows.data_ptr<float>(), vefd.data_ptr<float>(), eid.data_ptr<int64_t>(), grad_b->data_ptr<float>(),
            table, first, N, B, (int)K);
    } else {
        collapse_bw_kernel<false, FUSED><<<grid, THREADS, 0, stream>>>(
            flows.data_ptr<float>(), vefd.data_ptr<float>(), eid.data_ptr<int64_t>(), nullptr, table, first, N, B,
            (int)K);
    }
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
                         torch::Tensor& node_mars, torch::Tensor& fa, torch::Tensor& lz, torch::Tensor& lrn,
                         torch::Tensor& lrz, c10::optional<torch::Tensor>& grad_a, int64_t nid0, int64_t M, int64_t K) {
    const int B = node_mars.size(1), N = eid.size(0);
    check(vemd, "vem (data)", torch::kFloat32, -1); check(fb, "B", torch::kFloat32, (long)B * K * N);
    check(eid, "eid", torch::kInt64, N);
    check(vnfd, "vnf (data)", torch::kFloat32, -1); check(node_flows, "node_flows", torch::kFloat32, -1);
    check(node_mars, "node_mars", torch::kFloat32, -1);
    check(fa, "A", torch::kFloat32, (long)B * M * K); check(lz, "lz", torch::kFloat32, (long)B * M);
    check(lrn, "lrn", torch::kFloat32, (long)B * M * K); check(lrz, "lrz", torch::kFloat32, (long)B * M * K);
    const long w = (long)B * K;
    TORCH_CHECK(vemd.size(0) >= rows && vemd.size(1) == w, "lowrank mixture: the virtual element buffer does not hold ",
                w, " columns of ", rows, " rows");
    TORCH_CHECK(vnfd.size(0) >= nid0 + M && vnfd.size(1) == w && node_flows.size(1) == B,
                "lowrank mixture: the node rows ", nid0, " + ", M, " do not fit the buffers");
    if (grad_a.has_value()) check(*grad_a, "grad A", torch::kFloat32, (long)B * M * K);
    if ((N == 0 && M == 0) || B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = grid_of((long)(N + M) * B);
#define LOWRANK_PRE(GRAD, PTR)                                                                                      \
    backward_pre_kernel<GRAD, FUSED><<<grid, THREADS, 0, stream>>>(                                                 \
        vemd.data_ptr<float>(), (long)vemd.size(1), vals, table, first, fb.data_ptr<float>(),                       \
        eid.data_ptr<int64_t>(), N, vnfd.data_ptr<float>(), node_flows.data_ptr<float>(),                           \
        node_mars.data_ptr<float>(), fa.data_ptr<float>(), lz.data_ptr<float>(), lrn.data_ptr<float>(),             \
        lrz.data_ptr<float>(), PTR, (long)nid0, (int)M, B, (int)K)
    if (grad_a.has_value()) { LOWRANK_PRE(true, grad_a->data_ptr<float>()); }
    else { LOWRANK_PRE(false, nullptr); }
#undef LOWRANK_PRE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// A layer's backward before its plain calls: the data half's virtual child values (`vemd`) and virtual node flows
// (`vnfd`), and dLL / dA from the responsibilities the forward kept, one launch
void lowrank_backward_pre(torch::Tensor vemd, torch::Tensor element_mars, torch::Tensor fb, torch::Tensor eid,
                          torch::Tensor vnfd, torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor fa,
                          torch::Tensor lz, torch::Tensor lrn, torch::Tensor lrz, c10::optional<torch::Tensor> grad_a,
                          int64_t nid0, int64_t M, int64_t K) {
    check(element_mars, "element_mars", torch::kFloat32, -1);
    TORCH_CHECK(element_mars.size(1) == node_mars.size(1), "lowrank mixture: element and node values of two batches");
    backward_pre_launch<false>(vemd, element_mars.data_ptr<float>(), element_mars.size(0), nullptr, 0, fb, eid, vnfd,
                               node_flows, node_mars, fa, lz, lrn, lrz, grad_a, nid0, M, K);
}

// ... with the product layer below fused in: the element values read from their two children's node values
void lowrank_backward_pre_fused(torch::Tensor vemd, torch::Tensor table, int64_t first, torch::Tensor fb,
                                torch::Tensor eid, torch::Tensor vnfd, torch::Tensor node_flows,
                                torch::Tensor node_mars, torch::Tensor fa, torch::Tensor lz, torch::Tensor lrn,
                                torch::Tensor lrz, c10::optional<torch::Tensor> grad_a, int64_t nid0, int64_t M,
                                int64_t K) {
    const long rows = check_children(table, first);
    backward_pre_launch<true>(vemd, node_mars.data_ptr<float>(), rows, table.data_ptr<int32_t>(), (long)first, fb,
                              eid, vnfd, node_flows, node_mars, fa, lz, lrn, lrz, grad_a, nid0, M, K);
}

namespace {

template <typename TO>
void normalizer_launch(torch::Tensor& U, torch::Tensor& V, torch::Tensor& node_flows,
                       std::vector<torch::Tensor>& fa, std::vector<torch::Tensor>& lz, std::vector<torch::Tensor>& fb,
                       std::vector<int64_t>& nid0, int64_t M, int64_t K) {
    const int B = node_flows.size(1), N = V.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    long row0 = 0;
    for (size_t c = 0; c < fa.size(); c++) {
        check(fa[c], "A", torch::kFloat32, (long)B * M * K); check(lz[c], "lz", torch::kFloat32, (long)B * M);
        check(fb[c], "B", torch::kFloat32, (long)B * K * N);
        TORCH_CHECK(nid0[c] >= 0 && node_flows.size(0) >= nid0[c] + M, "lowrank mixture: a copy's node rows");
        normalizer_rows_kernel<TO><<<B * (int)K, THREADS, 0, stream>>>(
            static_cast<TO*>(U.data_ptr()), static_cast<TO*>(V.data_ptr()), row0, node_flows.data_ptr<float>(),
            (long)nid0[c],
            fa[c].data_ptr<float>(), lz[c].data_ptr<float>(), fb[c].data_ptr<float>(), (int)M, N, B, (int)K);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        row0 += (long)B * K;
    }
}

template <typename TO>
void normalizer_grads_launch(std::vector<c10::optional<torch::Tensor>>& grads, torch::Tensor& V, torch::Tensor& G,
                             int64_t B, int64_t K) {
    const int N = V.size(1);
    const long n = (long)B * K * N;
    auto stream = at::cuda::getCurrentCUDAStream();
    for (size_t c = 0; c < grads.size(); c++) {
        if (!grads[c].has_value()) continue;
        check(*grads[c], "grad B", torch::kFloat32, n);
        normalizer_grads_kernel<TO><<<grid_of(n), THREADS, 0, stream>>>(
            grads[c]->data_ptr<float>(), static_cast<const TO*>(V.data_ptr()), G.data_ptr<float>(), (long)c * B * K, n,
            N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

// The normalizer half's operands of a node vector's copies: `U` [copies B K, M] and `V` [copies B K, N], float32 or
// bfloat16 (as allocated), from each copy's node flows (rows `nid0[c]` on), factors and kept `log Z`
void lowrank_normalizer_operands(torch::Tensor U, torch::Tensor V, torch::Tensor node_flows,
                                 std::vector<torch::Tensor> fa, std::vector<torch::Tensor> lz,
                                 std::vector<torch::Tensor> fb, std::vector<int64_t> nid0, int64_t M, int64_t K) {
    check(node_flows, "node_flows", torch::kFloat32, -1);
    TORCH_CHECK(U.is_cuda() && U.is_contiguous() && V.is_cuda() && V.is_contiguous() && U.scalar_type() == V.scalar_type()
                && U.dim() == 2 && V.dim() == 2 && U.size(0) == V.size(0) && U.size(1) == M
                && U.size(0) == (long)fa.size() * node_flows.size(1) * K
                && fa.size() == lz.size() && fa.size() == fb.size() && fa.size() == nid0.size(),
                "lowrank mixture: the normalizer operands do not match the copies");
    if (U.numel() == 0 && V.numel() == 0) return;
    if (U.scalar_type() == torch::kBFloat16) normalizer_launch<__nv_bfloat16>(U, V, node_flows, fa, lz, fb, nid0, M, K);
    else if (U.scalar_type() == torch::kFloat32) normalizer_launch<float>(U, V, node_flows, fa, lz, fb, nid0, M, K);
    else TORCH_CHECK(false, "lowrank mixture: the normalizer operands must be float32 or bfloat16");
}

// Each copy's `dB` [B, K, N] minus `V * G` on its rows (`G` [copies B K, N] float32: `U theta`); None: a copy without
// a gradient
void lowrank_normalizer_grads(std::vector<c10::optional<torch::Tensor>> grads, torch::Tensor V, torch::Tensor G,
                              int64_t B, int64_t K) {
    check(G, "U theta", torch::kFloat32, V.numel());
    TORCH_CHECK(V.is_cuda() && V.is_contiguous() && V.dim() == 2 && G.dim() == 2 && G.size(1) == V.size(1)
                && V.size(0) == (long)grads.size() * B * K, "lowrank mixture: `V` does not match the copies");
    if (V.scalar_type() == torch::kBFloat16) normalizer_grads_launch<__nv_bfloat16>(grads, V, G, B, K);
    else if (V.scalar_type() == torch::kFloat32) normalizer_grads_launch<float>(grads, V, G, B, K);
    else TORCH_CHECK(false, "lowrank mixture: `V` must be float32 or bfloat16");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_expand", &lowrank_expand, "virtual child values: element values + B, a column per (sample, rank)");
    m.def("lowrank_expand_fused", &lowrank_expand_fused,
          "virtual child values with the product layer below fused in: each the sum of its two children's node values + B");
    m.def("lowrank_collapse", &lowrank_collapse, "node values from the virtual ones: logsumexp over ranks with A, - log Z");
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
    m.def("lowrank_normalizer_operands", &lowrank_normalizer_operands,
          "the normalizer half's U and V of a node vector's copies, for the two matrix products");
    m.def("lowrank_normalizer_grads", &lowrank_normalizer_grads, "each copy's dLL/dB minus V * (U theta) on its rows");
}
