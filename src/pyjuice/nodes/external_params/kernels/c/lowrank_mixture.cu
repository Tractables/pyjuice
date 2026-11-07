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
// whatever its value was):
//
//     expand_bw     data:        vnf[node, b K + r] = log f_b[node] + A_b[node, r] - log S_b[node]
//                   normalizer:  vnf[node, b K + r] = log f_b[node] + A_b[node, r] - log Z_b[node]
//     (plain backward on each half: virtual element flows `vef` -- stored, whatever the buffer held --; the data
//      half's parameter flows are `F+`, the normalizer half's are the denominator flow `F-`, kept divided by the
//      parameters: `take`)
//     collapse_bw   element_flows[child, b] = logsumexp_r vef_data[child, b K + r]
//
// with the factors' gradients beside: `dA` from the two responsibilities the forward kept (each rank's share of S and
// of Z), `dB` from the virtual element flows, data minus normalizer.
//
// THE NORMALIZER HALF OF SEVERAL TIED COPIES IN ONE LAUNCH. It reads no child values, and tied copies share the
// parameters, so the backward's normalizer columns of several copies can be the columns of ONE plain backward: that
// of a "host" copy, the others' columns written into the host's rows at their own column offsets (`expand` and
// `expand_bw` take the normalizer half's rows and offset apart from the data half's; `sub_b` reads a copy's part of
// the result back). The data half stays each layer's own: it reads the layer's children.
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

// `vd` / `vz`: the data and the normalizer half's buffers (one buffer in the forward, the normalizer's columns after
// the data's), `cd` / `cz` columns wide, the half starting at column `od` / `oz`, a child's row `eid[j]` in the data
// half's and `eidz[j]` in the normalizer's (another copy's rows where that one hosts the normalizer half).
__global__ void expand_kernel(float* vd, long cd, long od, float* vz, long cz, long oz,
                              const float* __restrict__ element_mars, const float* __restrict__ fb,
                              const int64_t* __restrict__ eid, const int64_t* __restrict__ eidz, int N, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)N * B) return;
    const int b = (int)(t % B);
    const int j = (int)(t / B);
    const long e = eid[j];
    if (e < 0) return;                                                  // a child no edge reaches
    const long ez = eidz[j];

    const float c = element_mars[e * B + b];
    float* outd = vd + e * cd + od + (long)b * K;
    float* outz = vz + ez * cz + oz + (long)b * K;
    const float* src = fb + (long)b * K * N + j;
    for (int r = 0; r < K; r++) {
        const float v = src[(long)r * N];
        outd[r] = c + v;
        outz[r] = v;
    }
}

// The forward's `expand` with the product layer below fused in: a child's value read as the sum of its two children's
// node values (`table[e - first]`, the product group's rows of child node rows; a product of one child has the dummy
// node, of value 0, as its second) -- what the product layer's forward would have written to `element_mars`, which
// is then neither launched nor read.
__global__ void expand_fused_kernel(float* vd, long cd, long od, float* vz, long cz, long oz,
                                    const float* __restrict__ node_mars, const int32_t* __restrict__ table, long first,
                                    const float* __restrict__ fb, const int64_t* __restrict__ eid,
                                    const int64_t* __restrict__ eidz, int N, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)N * B) return;
    const int b = (int)(t % B);
    const int j = (int)(t / B);
    const long e = eid[j];
    if (e < 0) return;
    const long ez = eidz[j];

    const int32_t* ch = table + (e - first) * 2;
    const float c = node_mars[(long)ch[0] * B + b] + node_mars[(long)ch[1] * B + b];
    float* outd = vd + e * cd + od + (long)b * K;
    float* outz = vz + ez * cz + oz + (long)b * K;
    const float* src = fb + (long)b * K * N + j;
    for (int r = 0; r < K; r++) {
        const float v = src[(long)r * N];
        outd[r] = c + v;
        outz[r] = v;
    }
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
// fused in), ONE launch: the first `M B` threads are `collapse_kernel`'s, the `N B` after them `expand_fused_kernel`'s
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

template <bool GRAD>
__global__ void expand_bw_kernel(float* __restrict__ vnfd, float* __restrict__ vnfz, long cz, long oz, long nid0z,
                                 const float* __restrict__ node_flows, const float* __restrict__ node_mars,
                                 const float* __restrict__ fa, const float* __restrict__ lz,
                                 const float* __restrict__ lrn, const float* __restrict__ lrz,
                                 float* __restrict__ grad_a, long nid0, int M, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)M * B) return;
    const int b = (int)(t % B);
    const int i = (int)(t / B);

    const long row = nid0 + i;
    const long at = ((long)b * M + i) * K;
    const float lf = node_flows[row * B + b];
    const float z = lz[(long)b * M + i];
    const float ls = node_mars[row * B + b] + z;                        // log S = log n + log Z
    float* outd = vnfd + (row * (long)B + b) * K;
    float* outz = vnfz + (nid0z + i) * cz + oz + (long)b * K;
    const bool live = (lf > -INFINITY) && (ls > -INFINITY);
    for (int r = 0; r < K; r++) {
        outd[r] = live ? lf + fa[at + r] - ls : -INFINITY;
        outz[r] = live ? lf + fa[at + r] - z : -INFINITY;
    }

    if (GRAD && live) {
        // dLL / dA[i, r] = f * (the rank's share of S - its share of Z)
        for (int r = 0; r < K; r++) grad_a[at + r] += expf(lf + lrn[at + r]) - expf(lf + lrz[at + r]);
    }
}

// GRAD 0: the element flows alone; 1: and the data half's part of dLL / dB; 2: and the normalizer half's too (read
// from `vefz`, this layer's rows, columns as the data half's). FUSED: the product layer below fused in -- `flows` are
// the NODE flows, and an element's flow is written as its two children's (`table[e - first]`): what the product
// layer's backward writes for children of one parent element each, which is then not launched.
template <int GRAD, bool FUSED>
__global__ void collapse_bw_kernel(float* __restrict__ flows, const float* __restrict__ vefd,
                                   const float* __restrict__ vefz, const int64_t* __restrict__ eid,
                                   float* __restrict__ grad_b, const int32_t* __restrict__ table, long first,
                                   int N, int B, int K) {
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
        // dLL / dB[r, j] = the data half's virtual element flow - the normalizer half's
        float* g = grad_b + (long)b * K * N + j;
        if (GRAD == 2) {
            const float* fz = vefz + (e * (long)B + b) * K;
            for (int r = 0; r < K; r++) g[(long)r * N] += expf(f[r]) - expf(fz[r]);
        } else {
            for (int r = 0; r < K; r++) g[(long)r * N] += expf(f[r]);
        }
    }
}

// dLL / dB[b, r, j] -= exp(vefz[eidz[j], oz + b K + r]): a copy's normalizer part, from the host's rows
__global__ void sub_b_kernel(float* __restrict__ grad_b, const float* __restrict__ vefz, long cz, long oz,
                             const int64_t* __restrict__ eidz, int N, int B, int K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)N * B) return;
    const int b = (int)(t % B);
    const int j = (int)(t / B);
    const long e = eidz[j];
    if (e < 0) return;
    const float* fz = vefz + e * cz + oz + (long)b * K;
    float* g = grad_b + (long)b * K * N + j;
    for (int r = 0; r < K; r++) g[(long)r * N] -= expf(fz[r]);
}

// dst += src / theta; src = 0: the normalizer half's parameter flows -- `theta * W`, the plain kernels' -- taken into
// the denominator's statistic `W`, which is kept free of the parameters (a parameter of 0 has no flow: nothing)
inline __device__ float as_float(float x) { return x; }
inline __device__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename TP>
__global__ void take_kernel(float* __restrict__ dst, float* __restrict__ src, const TP* __restrict__ theta, long n) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const float th = as_float(theta[t]);
    if (th > 0.0f) dst[t] += src[t] / th;
    src[t] = 0.0f;
}

void check(const torch::Tensor& t, const char* name, torch::ScalarType st, long numel) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "lowrank mixture: `", name, "` must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == st, "lowrank mixture: `", name, "` has the wrong dtype");
    TORCH_CHECK(numel < 0 || t.numel() == numel, "lowrank mixture: `", name, "` has ", t.numel(),
                " elements, expected ", numel);
}

}  // namespace


void lowrank_expand(torch::Tensor vd, int64_t od, torch::Tensor vz, int64_t oz, torch::Tensor element_mars,
                    torch::Tensor fb, torch::Tensor eid, torch::Tensor eidz, int64_t K) {
    const int B = element_mars.size(1), N = eid.size(0);
    check(vd, "vem (data)", torch::kFloat32, -1); check(vz, "vem (normalizer)", torch::kFloat32, -1);
    check(element_mars, "element_mars", torch::kFloat32, -1);
    check(fb, "B", torch::kFloat32, (long)B * K * N); check(eid, "eid", torch::kInt64, N);
    check(eidz, "eid (normalizer)", torch::kInt64, N);
    const long cd = vd.size(1), cz = vz.size(1), w = (long)B * K;
    TORCH_CHECK(vd.size(0) >= element_mars.size(0) && vz.size(0) >= element_mars.size(0)
                && od >= 0 && oz >= 0 && od + w <= cd && oz + w <= cz,
                "lowrank mixture: the virtual element buffers do not hold ", w, " columns at ", od, " and ", oz);
    if (N == 0 || B == 0) return;
    expand_kernel<<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        vd.data_ptr<float>(), cd, (long)od, vz.data_ptr<float>(), cz, (long)oz, element_mars.data_ptr<float>(),
        fb.data_ptr<float>(), eid.data_ptr<int64_t>(), eidz.data_ptr<int64_t>(), N, B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lowrank_expand_fused(torch::Tensor vd, int64_t od, torch::Tensor vz, int64_t oz, torch::Tensor node_mars,
                          torch::Tensor table, int64_t first, torch::Tensor fb, torch::Tensor eid, torch::Tensor eidz,
                          int64_t K) {
    const int B = node_mars.size(1), N = eid.size(0);
    check(vd, "vem (data)", torch::kFloat32, -1); check(vz, "vem (normalizer)", torch::kFloat32, -1);
    check(node_mars, "node_mars", torch::kFloat32, -1); check(table, "children", torch::kInt32, -1);
    check(fb, "B", torch::kFloat32, (long)B * K * N); check(eid, "eid", torch::kInt64, N);
    check(eidz, "eid (normalizer)", torch::kInt64, N);
    const long cd = vd.size(1), cz = vz.size(1), w = (long)B * K;
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    TORCH_CHECK(vd.size(0) >= first + table.size(0) && vz.size(0) >= first + table.size(0)
                && od >= 0 && oz >= 0 && od + w <= cd && oz + w <= cz,
                "lowrank mixture: the virtual element buffers do not hold ", w, " columns at ", od, " and ", oz);
    if (N == 0 || B == 0) return;
    expand_fused_kernel<<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        vd.data_ptr<float>(), cd, (long)od, vz.data_ptr<float>(), cz, (long)oz, node_mars.data_ptr<float>(),
        table.data_ptr<int32_t>(), (long)first, fb.data_ptr<float>(), eid.data_ptr<int64_t>(),
        eidz.data_ptr<int64_t>(), N, B, (int)K);
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

void lowrank_expand_bw(torch::Tensor vnfd, torch::Tensor vnfz, int64_t oz, int64_t nid0z, torch::Tensor node_flows,
                       torch::Tensor node_mars, torch::Tensor fa, torch::Tensor lz, torch::Tensor lrn,
                       torch::Tensor lrz, c10::optional<torch::Tensor> grad_a, int64_t nid0, int64_t M, int64_t K) {
    const int B = node_mars.size(1);
    check(vnfd, "vnf (data)", torch::kFloat32, -1); check(vnfz, "vnf (normalizer)", torch::kFloat32, -1);
    check(node_flows, "node_flows", torch::kFloat32, -1); check(node_mars, "node_mars", torch::kFloat32, -1);
    check(fa, "A", torch::kFloat32, (long)B * M * K); check(lz, "lz", torch::kFloat32, (long)B * M);
    check(lrn, "lrn", torch::kFloat32, (long)B * M * K); check(lrz, "lrz", torch::kFloat32, (long)B * M * K);
    const long cz = vnfz.size(1), w = (long)B * K;
    TORCH_CHECK(vnfd.size(0) >= nid0 + M && vnfd.size(1) == w && vnfz.size(0) >= nid0z + M && nid0z >= 0
                && oz >= 0 && oz + w <= cz && node_flows.size(1) == B,
                "lowrank mixture: the node rows ", nid0, " / ", nid0z, " + ", M, " do not fit the buffers");
    if (M == 0 || B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (grad_a.has_value()) {
        check(*grad_a, "grad A", torch::kFloat32, (long)B * M * K);
        expand_bw_kernel<true><<<grid_of((long)M * B), THREADS, 0, stream>>>(
            vnfd.data_ptr<float>(), vnfz.data_ptr<float>(), cz, (long)oz, (long)nid0z, node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), fa.data_ptr<float>(), lz.data_ptr<float>(), lrn.data_ptr<float>(),
            lrz.data_ptr<float>(), grad_a->data_ptr<float>(), (long)nid0, (int)M, B, (int)K);
    } else {
        expand_bw_kernel<false><<<grid_of((long)M * B), THREADS, 0, stream>>>(
            vnfd.data_ptr<float>(), vnfz.data_ptr<float>(), cz, (long)oz, (long)nid0z, node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), fa.data_ptr<float>(), lz.data_ptr<float>(), lrn.data_ptr<float>(),
            lrz.data_ptr<float>(), nullptr, (long)nid0, (int)M, B, (int)K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

template <bool FUSED>
void collapse_bw_launch(torch::Tensor& flows, torch::Tensor& vefd, torch::Tensor& eid,
                        c10::optional<torch::Tensor>& vefz, c10::optional<torch::Tensor>& grad_b,
                        const int32_t* table, long first, int64_t K) {
    const int B = flows.size(1), N = eid.size(0);
    check(flows, "flows", torch::kFloat32, -1); check(vefd, "vef (data)", torch::kFloat32, -1);
    check(eid, "eid", torch::kInt64, N);
    TORCH_CHECK(vefd.size(1) == (long)B * K, "lowrank mixture: `vef` does not match the flows");
    if (N == 0 || B == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = grid_of((long)N * B);
    if (!grad_b.has_value()) {
        collapse_bw_kernel<0, FUSED><<<grid, THREADS, 0, stream>>>(
            flows.data_ptr<float>(), vefd.data_ptr<float>(), nullptr, eid.data_ptr<int64_t>(), nullptr, table, first,
            N, B, (int)K);
    } else {
        check(*grad_b, "grad B", torch::kFloat32, (long)B * K * N);
        if (vefz.has_value()) {
            check(*vefz, "vef (normalizer)", torch::kFloat32, vefd.numel());
            collapse_bw_kernel<2, FUSED><<<grid, THREADS, 0, stream>>>(
                flows.data_ptr<float>(), vefd.data_ptr<float>(), vefz->data_ptr<float>(), eid.data_ptr<int64_t>(),
                grad_b->data_ptr<float>(), table, first, N, B, (int)K);
        } else {
            collapse_bw_kernel<1, FUSED><<<grid, THREADS, 0, stream>>>(
                flows.data_ptr<float>(), vefd.data_ptr<float>(), nullptr, eid.data_ptr<int64_t>(),
                grad_b->data_ptr<float>(), table, first, N, B, (int)K);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void lowrank_collapse_bw(torch::Tensor element_flows, torch::Tensor vefd, torch::Tensor eid,
                         c10::optional<torch::Tensor> vefz, c10::optional<torch::Tensor> grad_b, int64_t K) {
    TORCH_CHECK(vefd.size(0) >= element_flows.size(0), "lowrank mixture: `vef` does not match the element flows");
    collapse_bw_launch<false>(element_flows, vefd, eid, vefz, grad_b, nullptr, 0, K);
}

// ... with the product layer below fused in: the element flows written as their two children's NODE flows
void lowrank_collapse_bw_fused(torch::Tensor node_flows, torch::Tensor vefd, torch::Tensor eid, torch::Tensor table,
                               int64_t first, c10::optional<torch::Tensor> vefz, c10::optional<torch::Tensor> grad_b,
                               int64_t K) {
    check(table, "children", torch::kInt32, -1);
    TORCH_CHECK(table.dim() == 2 && table.size(1) == 2, "lowrank mixture: products of two children only");
    TORCH_CHECK(vefd.size(0) >= first + table.size(0), "lowrank mixture: `vef` does not hold the products' rows");
    collapse_bw_launch<true>(node_flows, vefd, eid, vefz, grad_b, table.data_ptr<int32_t>(), (long)first, K);
}

void lowrank_sub_b(torch::Tensor grad_b, torch::Tensor vefz, int64_t oz, torch::Tensor eidz, int64_t B, int64_t K) {
    const int N = eidz.size(0);
    check(grad_b, "grad B", torch::kFloat32, (long)B * K * N); check(vefz, "vef (normalizer)", torch::kFloat32, -1);
    check(eidz, "eid (normalizer)", torch::kInt64, N);
    TORCH_CHECK(oz >= 0 && oz + (long)B * K <= vefz.size(1), "lowrank mixture: `vef` does not hold the columns at ", oz);
    if (N == 0 || B == 0) return;
    sub_b_kernel<<<grid_of((long)N * B), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_b.data_ptr<float>(), vefz.data_ptr<float>(), (long)vefz.size(1), (long)oz, eidz.data_ptr<int64_t>(),
        N, (int)B, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// A layer's backward before its plain calls, one host call: `lowrank_expand` and `lowrank_expand_bw`
void lowrank_backward_pre(torch::Tensor vemd, torch::Tensor vemz, int64_t oz, torch::Tensor element_mars,
                          torch::Tensor fb, torch::Tensor eid, torch::Tensor eidz,
                          torch::Tensor vnfd, torch::Tensor vnfz, int64_t nid0z,
                          torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor fa, torch::Tensor lz,
                          torch::Tensor lrn, torch::Tensor lrz, c10::optional<torch::Tensor> grad_a,
                          int64_t nid0, int64_t M, int64_t K) {
    lowrank_expand(vemd, 0, vemz, oz, element_mars, fb, eid, eidz, K);
    lowrank_expand_bw(vnfd, vnfz, oz, nid0z, node_flows, node_mars, fa, lz, lrn, lrz, grad_a, nid0, M, K);
}

// ... with the product layer below fused in (`lowrank_expand_fused`: the element values read from their children)
void lowrank_backward_pre_fused(torch::Tensor vemd, torch::Tensor vemz, int64_t oz, torch::Tensor table, int64_t first,
                                torch::Tensor fb, torch::Tensor eid, torch::Tensor eidz,
                                torch::Tensor vnfd, torch::Tensor vnfz, int64_t nid0z,
                                torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor fa, torch::Tensor lz,
                                torch::Tensor lrn, torch::Tensor lrz, c10::optional<torch::Tensor> grad_a,
                                int64_t nid0, int64_t M, int64_t K) {
    lowrank_expand_fused(vemd, 0, vemz, oz, node_mars, table, first, fb, eid, eidz, K);
    lowrank_expand_bw(vnfd, vnfz, oz, nid0z, node_flows, node_mars, fa, lz, lrn, lrz, grad_a, nid0, M, K);
}

// `lowrank_sub_b` for the copies a host holds, copy `s` at column `s * B * K` (None: a copy without gradients)
void lowrank_sub_b_hosted(std::vector<c10::optional<torch::Tensor>> grads, torch::Tensor vefz, torch::Tensor eidz,
                          int64_t B, int64_t K) {
    for (size_t s = 0; s < grads.size(); s++)
        if (grads[s].has_value()) lowrank_sub_b(*grads[s], vefz, (int64_t)s * B * K, eidz, B, K);
}

void lowrank_take(torch::Tensor dst, torch::Tensor src, torch::Tensor theta) {
    check(dst, "dst", torch::kFloat32, src.numel()); check(src, "src", torch::kFloat32, -1);
    TORCH_CHECK(theta.is_cuda() && theta.is_contiguous() && theta.numel() == src.numel(),
                "lowrank mixture: the parameters of the flows taken: ", theta.numel(), " for ", src.numel());
    const long n = src.numel();
    if (n == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (theta.scalar_type() == torch::kBFloat16) {
        take_kernel<__nv_bfloat16><<<grid_of(n), THREADS, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), (const __nv_bfloat16*)theta.data_ptr(), n);
    } else {
        TORCH_CHECK(theta.scalar_type() == torch::kFloat32, "lowrank mixture: parameters in fp32 or bf16");
        take_kernel<float><<<grid_of(n), THREADS, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), theta.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_expand", &lowrank_expand, "virtual child values: element values + B, a column per (sample, rank)");
    m.def("lowrank_expand_fused", &lowrank_expand_fused,
          "virtual child values with the product layer below fused in: each the sum of its two children's node values + B");
    m.def("lowrank_collapse", &lowrank_collapse, "node values from the virtual ones: logsumexp over ranks with A, - log Z");
    m.def("lowrank_collapse_expand", &lowrank_collapse_expand,
          "`lowrank_collapse` of a node vector and `lowrank_expand_fused` of the next layer's, one launch");
    m.def("lowrank_expand_bw", &lowrank_expand_bw, "virtual node flows (and dLL/dA from the kept responsibilities)");
    m.def("lowrank_collapse_bw", &lowrank_collapse_bw, "element flows from the virtual ones (and their part of dLL/dB)");
    m.def("lowrank_sub_b", &lowrank_sub_b, "a copy's normalizer part of dLL/dB, from the host's virtual element flows");
    m.def("lowrank_take", &lowrank_take, "dst += src / theta, src = 0");
    m.def("lowrank_backward_pre", &lowrank_backward_pre, "`lowrank_expand` and `lowrank_expand_bw`, one call");
    m.def("lowrank_backward_pre_fused", &lowrank_backward_pre_fused,
          "`lowrank_expand_fused` and `lowrank_expand_bw`, one call");
    m.def("lowrank_collapse_bw_fused", &lowrank_collapse_bw_fused,
          "`lowrank_collapse_bw` with the element flows written as their two children's node flows");
    m.def("lowrank_sub_b_hosted", &lowrank_sub_b_hosted, "`lowrank_sub_b` for every copy a host holds");
}
