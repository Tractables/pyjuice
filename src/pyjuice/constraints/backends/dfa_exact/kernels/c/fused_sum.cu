// The dfa_exact backend's fused sums on tensor cores, for the launches their pre-passes serve (`kernels/sum.py`):
//   forward (`fused_sum`): node blocks of 32 nodes over the same children, read in chunks of 32 children;
//   top-down (`fused_sum_top_down`, TD): child blocks of 32 children under the same parents, read in chunks of a
//   parent node block's 32 nodes -- the forward with children and nodes exchanged (a parent's weights to a child
//   block, 32 x 32, read node-major).
// Every chunk's shifted exponentials `e = exp(x - t)` and its maximum per column `t` are computed beforehand
// (`_forward_prepass_kernel` / `_top_down_prepass_kernel`, in `pre_buf`, the parents' less their log Z; at TF32, e on
// the TF32 grid already). With external parameters (BlockScale, see `..external_params.block_scale`) every (row block,
// chunk) reads one log-scale cell s of the column's sample -- staged per (block, sample) where the tile's columns are of
// few samples (TILE_SAMPLES), else per column; forward, every node's log Z of the sample is taken off.
//
// Per output tile -- 4 row blocks (128 rows) x TN columns (64 to 128, the launch's), a CTA of 8 warps, a warp one
// block's 32 rows x TN / 2 columns -- and share of the chunks (grid z): a first pass gives every (row block, column)
// its shift
//     T = max over the share's chunks of t + s,
// then, per chunk, the chunk's weights (4 x 32 x 32), e (32 x TN), t and s are copied into shared memory a chunk
// ahead (cp.async, two stages: the products of one under the copies of the next), every warp's products are summed
// from zero over the chunk on the tensor cores (TF32, or 3 x TF32 for fp32-level products, from zero every 8: the
// tensor cores' sums truncate) and added times the chunk's factors f = exp(t + s - T) <= 1 (every column's computed once
// a warp, by one lane, and shuffled to its sums' lanes; at TF32 a chunk's sums are added under the next chunk's
// products, so a warp meets the next barrier with its products in flight) -- the same per-chunk shift as the Triton
// kernels' online log-sum-exp:
//     out = log(acc) + T (- log Z)        (-inf where nothing is live),
// written to the block's rows (top-down: element rows, or an aliased sum's rows).
// Pruned (the region's columns in the liveness' lane order), a tile past the live columns is written -inf.
//
// A launch of few tiles splits every tile's chunks in shares (`kernels/sum.py`, `_cuda_split`): every share stores its
// sums and shifts in the split buffer, the first one the tile's columns and samples too, and the merge kernel
// combines them in order -- out = log(sum of acc exp(T - T')) + T' (T' their maximum).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>

#define NBT 4              // row blocks per CTA
#define BS 32              // rows per block, reduction rows per chunk
#define THREADS 256
#define SPLIT_TILES 4096   // the split buffer's counters (`kernels/sum.py`, SPLIT_TILES): not used here
#define MAX_SHARE_CELLS 256    // external parameters: the most chunks a share (their cells in shared memory)
#define MAX_SHARE_TABLE 128    // weights not in one run per block: the most chunks a share (their firsts too)
#define TILE_SAMPLES 8         // external parameters: a tile's columns of at most this many samples (a range: lanes
                               // are in sample order) stage their scales per (block, sample), not per column


// An index table of either width (the backend's tables are int32 or int64)
struct Idx {
    const void* p;
    int wide;
    __device__ __forceinline__ long long operator[](long long i) const {
        return wide ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
    }
};

static Idx idx_of(const torch::Tensor& t) {
    TORCH_CHECK(t.scalar_type() == torch::kInt64 || t.scalar_type() == torch::kInt32, "index tables: int32 or int64");
    return Idx{t.data_ptr(), t.scalar_type() == torch::kInt64 ? 1 : 0};
}


__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int bytes) {     // the rest zero-filled
    unsigned s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool valid) {
    unsigned s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ unsigned to_tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_tf32(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A tile of TN columns: a stage of shared memory holds a chunk's weights, e, t and (external parameters) s. The
// weights' rows are their memory's (32 floats: forward one child's nodes, top-down one child's parent nodes), padded:
// A_LD 40 (forward: the products read a column of them), 36 (top-down: a row).
template <int TN, bool TD>
struct Tile {
    static constexpr int A_LD = TD ? 36 : 40;
    static constexpr int B_LD = TN + 8;                // a row of a B tile (one reduction row's columns), padded
    static constexpr int J = TN / 16;                  // a warp's tiles of 8 columns
    static constexpr int A_ST = NBT * BS * A_LD, B_ST = BS * B_LD;
    static constexpr int ST = A_ST + B_ST + TN + NBT * TN;
    static constexpr int SHARE = NBT * BS * TN + NBT * TN;          // a share's sums and shifts in the split buffer
    static constexpr int HEAD = 4 * TN;                             // a tile's columns (int64) and samples first
};


struct Args {
    float* out0;                      // forward: node_mars; TD: element_flows
    float* out1;                      // TD: node_flows (aliased sums' rows)
    const float* weights;
    const long long* wbase;           // [blocks, chunks]: a chunk's 32 x 32 weights' first
    long long wstride;                // affine: chunk ch's at the block's first + wstride ch
    const float* pre;                 // e[chunk row, k] at (chunk 32 + row) * pre_ld + k; the chunks' t from pre_cm on
    long long pre_ld, pre_cm;
    Idx rows, rreg, colreg, sel;      // forward: nids, nb_reg, nb_reg; TD: cb_row, cb_reg, cb_ivreg
    Idx reg_offset, reg_width, reg_first, reg_slots;
    Idx perm, reg_perm, reg_live;     // prune: the lane order
    long long elem_first, elem_width; // TD: the element rows
    const float* scales;              // EXT: the external parameters' log-scales
    const int* cells;                 // [blocks, chunks]: every chunk's cell
    long long sstride;                // floats per sample (0: every sample's)
    const float* lz;                  // EXT forward: log Z by node row, lzstride per sample
    long long lzstride;
    Idx coff, reg_ivk;                // lanes: compact rows, every lane's sample by a binary search
    int Bq, steps;
    int B, num_blocks, nch;
    int lanes, prune, indexed;        // (uniform: outside the products)
    int affine;                       // every block's weights one run
    float* split_buf;                 // split: every tile's head and shares
    int split, share;                 // the shares of every tile's chunks (1: none) and the chunks of each
    long long per_tile;               // floats of a tile in the split buffer
};


// The CTA's blocks (-1: none) and where their rows go; its columns' region, columns and live columns: shared by both
// kernels
struct Where {
    long long nbs[NBT];
    float* out[NBT];                  // a block's first row
    long long ow[NBT], lzrow[NBT];    // its rows' width; forward: its first node (log Z)
    long long reg, ncols, num_live;
};

template <bool TD>
__device__ __forceinline__ void where_of(const Args& a, Where& w) {
    for (int b = 0; b < NBT; ++b) {
        int i = blockIdx.x * NBT + b;
        const long long nb = i < a.num_blocks ? (a.indexed ? a.sel[i] : (long long)i) : -1;
        w.nbs[b] = nb;
        w.out[b] = nullptr;
        w.ow[b] = w.lzrow[b] = 0;
        if (nb < 0) continue;
        const long long row = a.rows[nb], r = a.rreg[nb];
        if (TD && r < 0) {                              // element rows
            w.out[b] = a.out0 + (row - a.elem_first) * a.elem_width;
            w.ow[b] = a.elem_width;
        } else {                                        // a sum region's rows
            w.ow[b] = a.reg_width[r];
            w.out[b] = (TD ? a.out1 : a.out0) + a.reg_offset[r] + (row - a.reg_first[r]) * w.ow[b];
        }
        w.lzrow[b] = row;
    }
    w.reg = a.colreg[w.nbs[0]];                         // (the blocks share their chunks: one region of columns)
    w.ncols = (long long)a.B * a.reg_slots[w.reg];
    w.num_live = a.prune ? (long long)a.reg_live[w.reg] : w.ncols;
}

__device__ __forceinline__ void put(const Where& w, int b, long long r, long long col, float o) {
    w.out[b][r * w.ow[b] + col] = o;
}


template <bool EXT, bool X3, int TN, bool TD>
__global__ void __launch_bounds__(THREADS) fused_sum_kernel(Args a) {
    using TT = Tile<TN, TD>;
    extern __shared__ float smem[];
    float* Ts = smem + 2 * TT::ST;                                  // [NBT][TN] the shifts
    long long* Cs = reinterpret_cast<long long*>(Ts + NBT * TN);   // [TN] the columns
    int* Ss = reinterpret_cast<int*>(Cs + TN);                      // [TN] their samples
    int* Cl = Ss + TN;                                              // EXT: [NBT][share] the chunks' cells
    long long* Wb = reinterpret_cast<long long*>(Cl + (EXT ? NBT * a.share : 0));   // !affine: [NBT][share]
    __shared__ Where w;
    __shared__ int s_lo, s_hi;                                      // EXT: the tile's columns' samples, a range

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        where_of<TD>(a, w);
        s_lo = 0x7fffffff;
        s_hi = -1;
    }
    __syncthreads();
    const long long col0 = (long long)blockIdx.y * TN;
    if (col0 >= w.ncols) return;
    const int ch0 = blockIdx.z * a.share, ch1 = min(a.nch, ch0 + a.share);   // this share's chunks

    if (a.prune && col0 >= w.num_live) {                // a tile of dead columns: -inf, nothing read (its first
        if (blockIdx.z != 0) return;                    // share; not merged)
        for (int i = tid; i < NBT * BS * TN; i += THREADS) {
            int b = i / (BS * TN), m = (i / TN) % BS, c = i % TN;
            long long k = col0 + c;
            if (w.nbs[b] >= 0 && k < w.ncols)
                w.out[b][m * w.ow[b] + a.perm[a.reg_perm[w.reg] + k]] = -CUDART_INF_F;
        }
        return;
    }
    for (int c = tid; c < TN; c += THREADS) {           // every column and its sample
        long long k = col0 + c;
        long long col = k < w.ncols ? (a.prune ? a.perm[a.reg_perm[w.reg] + k] : k) : 0;
        int s = 0;
        if (EXT && a.sstride != 0) {
            if (a.lanes) {                              // the last sample whose lanes start at or before it
                long long base = a.reg_ivk[w.reg] * (a.Bq + 1);
                int lo = 0, hi = a.Bq;
                for (int st = 0; st < a.steps; ++st) {
                    int mid = (lo + hi + 1) / 2;
                    bool go = a.coff[base + mid] <= col;
                    lo = go ? mid : lo;
                    hi = go ? hi : mid - 1;
                }
                s = lo;
            } else {
                s = (int)(col / a.reg_slots[w.reg]);
            }
        }
        Cs[c] = col;
        Ss[c] = s;
        if (EXT && k < w.ncols) {
            atomicMin(&s_lo, s);
            atomicMax(&s_hi, s);
        }
    }
    for (int i = tid; i < NBT * a.share; i += THREADS) {  // the share's cells and (not one run) first weights
        const int b = i / a.share, ch = ch0 + i % a.share;
        const bool v = w.nbs[b] >= 0 && ch < ch1;
        if (EXT) Cl[i] = v ? a.cells[w.nbs[b] * a.nch + ch] : 0;
        if (!a.affine) Wb[i] = v ? a.wbase[w.nbs[b] * a.nch + ch] : 0;
    }
    long long w0[NBT];                                  // affine: every block's first weight
    #pragma unroll
    for (int b = 0; b < NBT; ++b) w0[b] = a.affine && w.nbs[b] >= 0 ? a.wbase[w.nbs[b] * a.nch] : 0;
    __syncthreads();
    const int lo = s_lo, ns = s_hi - s_lo + 1;          // EXT: the scales per (block, sample), when few samples
    const bool few = EXT && ns <= TILE_SAMPLES;

    auto load = [&](int ch, int st) {                   // chunk ch into stage st
        float* As = smem + st * TT::ST;
        float* Bs = As + TT::A_ST;
        float* tS = Bs + TT::B_ST;
        float* sS = tS + TN;
        #pragma unroll
        for (int b = 0; b < NBT; ++b) {                 // the weights' 32 rows, 16 B at a time: 4 copies a thread
            const int r = tid / (BS / 4), q4 = (tid % (BS / 4)) * 4;
            const bool v = w.nbs[b] >= 0;
            const long long first = !v ? 0 : a.affine ? w0[b] + (long long)ch * a.wstride : Wb[b * a.share + ch - ch0];
            cp_async16(As + (b * BS + r) * TT::A_LD + q4, a.weights + first + r * BS + q4, v ? 16 : 0);
        }
        for (int i = tid; i < BS * TN / 4; i += THREADS) {              // e: past the columns, zeros
            const int k = i / (TN / 4), c4 = (i % (TN / 4)) * 4;
            const long long pos = col0 + c4;
            const int bytes = (int)max(0LL, min(16LL, 4 * (w.ncols - pos)));
            cp_async16(Bs + k * TT::B_LD + c4, a.pre + ((long long)ch * BS + k) * a.pre_ld + (bytes ? pos : 0), bytes);
        }
        for (int i = tid; i < TN / 4; i += THREADS) {                   // t
            const long long pos = col0 + 4 * i;
            const int bytes = (int)max(0LL, min(16LL, 4 * (w.ncols - pos)));
            cp_async16(tS + 4 * i, a.pre + a.pre_cm + (long long)ch * a.pre_ld + (bytes ? pos : 0), bytes);
        }
        if (EXT && few)                                                 // s per (block, sample)
            for (int i = tid; i < NBT * TILE_SAMPLES; i += THREADS) {
                const int b = i / TILE_SAMPLES, r = i % TILE_SAMPLES;
                const bool v = w.nbs[b] >= 0 && r < ns;
                cp_async4(sS + i, a.scales + (v ? (long long)(lo + r) * a.sstride + Cl[b * a.share + ch - ch0] : 0), v);
            }
        else if (EXT)                                                   // s per (block, column)
            for (int i = tid; i < NBT * TN; i += THREADS) {
                const int b = i / TN, c = i % TN;
                const bool v = w.nbs[b] >= 0 && col0 + c < w.ncols;
                cp_async4(sS + i, a.scales + (v ? (long long)Ss[c] * a.sstride + Cl[b * a.share + ch - ch0] : 0), v);
            }
    };
    load(ch0, 0);                                       // (under pass 1)
    cp_commit();

    {   // pass 1: T = max over the share's chunks of t + s -- a thread's one or two (block, column)s together (their
        // loads in flight)
        static_assert(NBT * TN >= THREADS && NBT * TN <= 2 * THREADS, "a thread's one or two (block, column)s");
        const int i1 = tid + THREADS;
        const int b0 = tid / TN, c0 = tid % TN, b1 = i1 / TN, c1 = i1 % TN;
        const bool v0 = w.nbs[b0] >= 0 && col0 + c0 < w.num_live;
        const bool v1 = i1 < NBT * TN && w.nbs[min(b1, NBT - 1)] >= 0 && col0 + c1 < w.num_live;
        const float* tp0 = a.pre + a.pre_cm + col0 + c0;
        const float* tp1 = a.pre + a.pre_cm + col0 + c1;
        const float* sp0 = a.scales + (long long)Ss[c0] * a.sstride;
        const float* sp1 = a.scales + (long long)Ss[c1] * a.sstride;
        float T0 = -CUDART_INF_F, T1 = -CUDART_INF_F;
        #pragma unroll 8
        for (int ch = ch0; ch < ch1; ++ch) {
            if (v0) {
                float t = tp0[(long long)ch * a.pre_ld];
                if (EXT) t += sp0[Cl[b0 * a.share + ch - ch0]];
                T0 = fmaxf(T0, t);
            }
            if (v1) {
                float t = tp1[(long long)ch * a.pre_ld];
                if (EXT) t += sp1[Cl[b1 * a.share + ch - ch0]];
                T1 = fmaxf(T1, t);
            }
        }
        Ts[tid] = T0;
        if (i1 < NBT * TN) Ts[i1] = T1;
    }

    float acc[2][TT::J][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < TT::J; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    const int wb = warp >> 1, wc = (warp & 1) * (TN / 2);   // the warp's block and first column
    const bool fl = lane < TN / 4;                      // lane l: the factors of the warp's columns wc + 2 l, + 1
    int slot[2];                                        // EXT, few samples: their samples
    #pragma unroll
    for (int h = 0; h < 2; ++h) slot[h] = few && fl ? max(Ss[wc + 2 * lane + h] - lo, 0) : 0;

    // A chunk: its copies waited for, the next chunk's issued under its products, its sums into P and the lane's two
    // factors into G (shuffled to the fold)
    auto chunk = [&](int ch, float (&P)[2][TT::J][4], float (&G)[2]) {
        const int st = (ch - ch0) & 1;
        cp_wait_all();                                  // this chunk's copies in
        __syncthreads();                                // (and visible; the other stage's products done)
        if (ch + 1 < ch1) load(ch + 1, st ^ 1);         // the next chunk's under this one's products
        cp_commit();
        const float* A = smem + st * TT::ST + wb * BS * TT::A_LD;
        const float* Bt = smem + st * TT::ST + TT::A_ST;
        const float* tS = Bt + TT::B_ST;
        const float* sS = tS + TN + wb * (few ? TILE_SAMPLES : TN);
        const float s1 = EXT && ns == 1 ? sS[0] : 0.f;   // (one sample: one scale for every column)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = wc + 2 * lane + h;
            G[h] = 0.f;
            if (fl) {
                const float T = Ts[wb * TN + c];
                const float t = !EXT ? tS[c] : tS[c] + (ns == 1 ? s1 : few ? sS[slot[h]] : sS[c]);
                G[h] = T != -CUDART_INF_F && t != -CUDART_INF_F ? __expf(t - T) : 0.f;
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < TT::J; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) P[i][j][q] = 0.f;
        #pragma unroll
        for (int k8 = 0; k8 < BS / 8; ++k8) {
            unsigned ah[2][4], al[2][4];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {               // A (rows r, r + 8; reduction rows kk, kk + 4)
                const int r = i * 16 + (lane >> 2), kk = k8 * 8 + (lane & 3);
                const int at = TD ? r * TT::A_LD + kk : kk * TT::A_LD + r;      // (row r, reduction row kk)
                constexpr int R8 = TD ? 8 * TT::A_LD : 8, K4 = TD ? 4 : 4 * TT::A_LD;
                const float x[4] = {A[at], A[at + R8], A[at + K4], A[at + R8 + K4]};
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    ah[i][q] = to_tf32(x[q]);
                    if (X3) al[i][q] = to_tf32(x[q] - __uint_as_float(ah[i][q]));
                }
            }
            #pragma unroll
            for (int j = 0; j < TT::J; ++j) {
                const int c = wc + j * 8 + (lane >> 2), kk = k8 * 8 + (lane & 3);
                const float y0 = Bt[kk * TT::B_LD + c], y1 = Bt[(kk + 4) * TT::B_LD + c];
                if (X3) {                               // (from zero every 8: the small terms first)
                    const unsigned bh0 = to_tf32(y0), bh1 = to_tf32(y1);
                    const unsigned bl0 = to_tf32(y0 - __uint_as_float(bh0)), bl1 = to_tf32(y1 - __uint_as_float(bh1));
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        float pp[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_tf32(pp, al[i], bh0, bh1);
                        mma_tf32(pp, ah[i], bl0, bl1);
                        mma_tf32(pp, ah[i], bh0, bh1);
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) P[i][j][q] += pp[q];
                    }
                } else {                                // (e on the TF32 grid already)
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) mma_tf32(P[i][j], ah[i], __float_as_uint(y0), __float_as_uint(y1));
                }
            }
        }
    };
    auto fold = [&](const float (&P)[2][TT::J][4], const float (&G)[2]) {
        #pragma unroll
        for (int j = 0; j < TT::J; ++j)                 // (q 0, 2: column h 0; 1, 3: h 1)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float f = __shfl_sync(0xffffffffu, G[h], 4 * j + (lane & 3));   // the column's lane
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    acc[i][j][h] = fmaf(P[i][j][h], f, acc[i][j][h]);
                    acc[i][j][h + 2] = fmaf(P[i][j][h + 2], f, acc[i][j][h + 2]);
                }
            }
    };
    if (X3) {                                           // (the registers: a chunk's products, then its fold)
        for (int ch = ch0; ch < ch1; ++ch) {
            float p[2][TT::J][4], F[2];
            chunk(ch, p, F);
            fold(p, F);
        }
    } else {                                            // a chunk's sums folded under the next chunk's products, two
        float p0[2][TT::J][4], p1[2][TT::J][4], F0[2], F1[2];  // sets in turn (a warp meets the barrier with its
        for (int ch = ch0; ch < ch1; ch += 2) {                 // products in flight)
            chunk(ch, p0, F0);
            if (ch > ch0) fold(p1, F1);
            if (ch + 1 == ch1) {                        // (an odd count's last)
                fold(p0, F0);
                break;
            }
            chunk(ch + 1, p1, F1);
            fold(p0, F0);
        }
        if ((ch1 - ch0) % 2 == 0) fold(p1, F1);
    }
    cp_wait_all();

    if (a.split > 1) {                                  // this share's sums and shifts; the merge kernel's out
        float* base = a.split_buf + SPLIT_TILES + ((long long)blockIdx.y * gridDim.x + blockIdx.x) * a.per_tile;
        if (blockIdx.z == 0)
            for (int c = tid; c < TN; c += THREADS) {
                reinterpret_cast<long long*>(base)[c] = Cs[c];
                reinterpret_cast<int*>(base + 2 * TN)[c] = Ss[c];
            }
        float* mine = base + TT::HEAD + (long long)blockIdx.z * TT::SHARE;
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < TT::J; ++j) {           // (q 0, 1 and 2, 3: two columns of a row)
                const int m = i * 16 + (lane >> 2), c = wc + j * 8 + (lane & 3) * 2;
                *reinterpret_cast<float2*>(mine + (wb * BS + m) * TN + c) = make_float2(acc[i][j][0], acc[i][j][1]);
                *reinterpret_cast<float2*>(mine + (wb * BS + m + 8) * TN + c) =
                    make_float2(acc[i][j][2], acc[i][j][3]);
            }
        for (int i = tid; i < NBT * TN; i += THREADS) mine[NBT * BS * TN + i] = Ts[i];
        return;
    }
    if (w.nbs[wb] < 0) return;                          // log(acc) + T (- log Z)
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < TT::J; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int m = i * 16 + (lane >> 2) + (q >= 2 ? 8 : 0);
                const int c = wc + j * 8 + (lane & 3) * 2 + (q & 1);
                if (col0 + c < w.ncols) {
                    const float v = acc[i][j][q];
                    float o = -CUDART_INF_F;
                    if (v > 0.f) {
                        o = __logf(v) + Ts[wb * TN + c];
                        if (EXT && !TD) o -= a.lz[(long long)Ss[c] * a.lzstride + w.lzrow[wb] + m];
                    }
                    put(w, wb, m, Cs[c], o);
                }
            }
}


// A split launch's outputs: a thread an output of a tile (grid z: the tile's outputs, 256 a CTA), every share's
// sums and shifts in order
template <bool EXT, int TN, bool TD>
__global__ void __launch_bounds__(THREADS) fused_sum_merge_kernel(Args a) {
    using TT = Tile<TN, TD>;
    __shared__ Where w;
    if (threadIdx.x == 0) where_of<TD>(a, w);
    __syncthreads();
    const int i = blockIdx.z * THREADS + threadIdx.x, r = i / TN, c = i % TN, b = r / BS, m = r % BS;
    const long long col0 = (long long)blockIdx.y * TN;
    if (r >= NBT * BS || w.nbs[b] < 0 || col0 + c >= w.ncols || (a.prune && col0 >= w.num_live)) return;
    const float* base = a.split_buf + SPLIT_TILES + ((long long)blockIdx.y * gridDim.x + blockIdx.x) * a.per_tile;
    const long long col = reinterpret_cast<const long long*>(base)[c];
    const float* sh = base + TT::HEAD;
    float T = -CUDART_INF_F;
    for (int z = 0; z < a.split; ++z) T = fmaxf(T, sh[(long long)z * TT::SHARE + NBT * BS * TN + b * TN + c]);
    float v = 0.f;
    if (T != -CUDART_INF_F)
        for (int z = 0; z < a.split; ++z) {             // (in order)
            const float* s = sh + (long long)z * TT::SHARE;
            const float t = s[NBT * BS * TN + b * TN + c];
            if (t != -CUDART_INF_F) v = fmaf(s[r * TN + c], __expf(t - T), v);
        }
    float o = -CUDART_INF_F;
    if (v > 0.f) {
        o = __logf(v) + T;
        if (EXT && !TD)
            o -= a.lz[(long long)reinterpret_cast<const int*>(base + 2 * TN)[c] * a.lzstride + w.lzrow[b] + m];
    }
    put(w, b, m, col, o);
}


static size_t smem_bytes(int tn, bool td, bool ext, int share, bool affine) {
    const int st = NBT * BS * (td ? 36 : 40) + BS * (tn + 8) + tn + NBT * tn;
    return (2 * st + NBT * tn) * sizeof(float) + tn * (sizeof(long long) + sizeof(int)) +
           (ext ? NBT * share * sizeof(int) : 0) + (affine ? 0 : NBT * share * sizeof(long long));
}

// A kernel may take as much shared memory as a block may have (set once)
template <bool EXT, bool X3, int TN, bool TD>
static const void* allowed() {
    static const void* k = [] {
        const void* f = (const void*)fused_sum_kernel<EXT, X3, TN, TD>;
        int dev = 0, most = 0;
        cudaFuncAttributes fa;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, f));
        C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            most - (int)fa.sharedSizeBytes));   // (less the static)
        return f;
    }();
    return k;
}

template <bool EXT, bool X3, int TN, bool TD>
static void launch(const Args& a, dim3 grid, cudaStream_t stream) {
    allowed<EXT, X3, TN, TD>();
    const size_t bytes = smem_bytes(TN, TD, EXT, a.share, a.affine);
    fused_sum_kernel<EXT, X3, TN, TD><<<grid, THREADS, bytes, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (a.split > 1) {
        dim3 mgrid(grid.x, grid.y, (NBT * BS * TN + THREADS - 1) / THREADS);
        fused_sum_merge_kernel<EXT, TN, TD><<<mgrid, THREADS, 0, stream>>>(a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <bool EXT, bool X3, bool TD>
static void launch_tn(const Args& a, int tn, dim3 grid, cudaStream_t stream) {
    switch (tn) {
        case 64: return launch<EXT, X3, 64, TD>(a, grid, stream);
        case 80: return launch<EXT, X3, 80, TD>(a, grid, stream);
        case 96: return launch<EXT, X3, 96, TD>(a, grid, stream);
        case 112: return launch<EXT, X3, 112, TD>(a, grid, stream);
        case 128: return launch<EXT, X3, 128, TD>(a, grid, stream);
    }
    TORCH_CHECK(false, "tiles of 64, 80, 96, 112 or 128 columns");
}


// One launch of either direction: the tables as `kernels/sum.py` passes them (`_cuda_forward`, `_cuda_top_down`)
void fused_sum_cuda(bool td, torch::Tensor out0, torch::Tensor out1, torch::Tensor weights, torch::Tensor wbase,
                    int64_t wstride, torch::Tensor pre, int64_t pre_ld, int64_t pre_cm, torch::Tensor rows,
                    torch::Tensor rreg, torch::Tensor colreg, torch::Tensor sel,
                    torch::Tensor reg_offset, torch::Tensor reg_width, torch::Tensor reg_first, torch::Tensor reg_slots,
                    torch::Tensor perm, torch::Tensor reg_perm, torch::Tensor reg_live, int64_t elem_first,
                    int64_t elem_width, torch::Tensor scales, torch::Tensor cells, int64_t sstride, torch::Tensor lz,
                    int64_t lzstride, torch::Tensor coff, torch::Tensor reg_ivk, int64_t Bq, int64_t steps, int64_t B,
                    int64_t num_blocks, int64_t nch, int64_t max_cols, bool ext, bool shared, bool lanes, bool prune,
                    bool indexed, bool x3, bool affine, int64_t tn, torch::Tensor split_buf, int64_t split) {
    TORCH_CHECK(out0.is_cuda() && out0.scalar_type() == torch::kFloat32 && out1.scalar_type() == torch::kFloat32,
                "outputs: fp32 on the device");
    TORCH_CHECK(wbase.scalar_type() == torch::kInt64 && cells.scalar_type() == torch::kInt32,
                "wbase int64, cells int32");
    Args a;
    a.out0 = out0.data_ptr<float>();
    a.out1 = out1.data_ptr<float>();
    a.weights = weights.data_ptr<float>();
    a.wbase = reinterpret_cast<const long long*>(wbase.data_ptr<int64_t>());
    a.wstride = wstride;
    a.pre = pre.data_ptr<float>();
    a.pre_ld = pre_ld;
    a.pre_cm = pre_cm;
    a.rows = idx_of(rows);
    a.rreg = idx_of(rreg);
    a.colreg = idx_of(colreg);
    a.sel = idx_of(sel);
    a.reg_offset = idx_of(reg_offset);
    a.reg_width = idx_of(reg_width);
    a.reg_first = idx_of(reg_first);
    a.reg_slots = idx_of(reg_slots);
    a.perm = idx_of(perm);
    a.reg_perm = idx_of(reg_perm);
    a.reg_live = idx_of(reg_live);
    a.elem_first = elem_first;
    a.elem_width = elem_width;
    a.scales = scales.data_ptr<float>();
    a.cells = cells.data_ptr<int>();
    a.sstride = sstride;
    a.lz = lz.data_ptr<float>();
    a.lzstride = lzstride;
    a.coff = idx_of(coff);
    a.reg_ivk = idx_of(reg_ivk);
    a.Bq = (int)Bq;
    a.steps = (int)steps;
    a.B = (int)B;
    a.num_blocks = (int)num_blocks;
    a.nch = (int)nch;
    a.lanes = lanes;
    a.prune = prune;
    a.indexed = indexed;
    a.affine = affine;
    if (!ext || shared) a.sstride = 0;              // (one sample's external parameters for every column)
    a.share = (int)((nch + split - 1) / split);
    a.split = (int)((nch + a.share - 1) / a.share);  // (every share some chunks)
    TORCH_CHECK(!ext || a.share <= MAX_SHARE_CELLS, "external parameters: at most 256 chunks a share");
    TORCH_CHECK(affine || a.share <= MAX_SHARE_TABLE, "weights not one run per block: at most 128 chunks a share");
    const int stn = (int)(4 * tn + (long long)a.split * (NBT * BS * tn + NBT * tn));
    a.per_tile = (stn + 3) / 4 * 4;
    a.split_buf = split_buf.data_ptr<float>();
    dim3 grid((unsigned)((num_blocks + NBT - 1) / NBT), (unsigned)((max_cols + tn - 1) / tn), (unsigned)a.split);
    if (grid.x == 0 || grid.y == 0) return;
    TORCH_CHECK(a.split == 1 || split_buf.numel() >= SPLIT_TILES + (long long)grid.x * grid.y * a.per_tile,
                "the split buffer is too small");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int t = (int)tn;
#define DISPATCH(E, X) \
    if (ext == E && x3 == X) return td ? launch_tn<E, X, true>(a, t, grid, stream) : launch_tn<E, X, false>(a, t, grid, stream);
    DISPATCH(false, false) DISPATCH(false, true) DISPATCH(true, false) DISPATCH(true, true)
}


// How many CTAs of a direction's kernel an SM holds at once (for the split's waves)
int64_t ctas_per_sm(bool td, bool ext, bool x3, int64_t tn, int64_t share, bool affine) {
    const size_t bytes = smem_bytes((int)tn, td, ext, (int)share, affine);
    const void* k = nullptr;
#define K_OF(T, E, X, N) if (td == T && ext == E && x3 == X && tn == N) k = allowed<E, X, N, T>();
#define K_TN(T, E, X) K_OF(T, E, X, 64) K_OF(T, E, X, 80) K_OF(T, E, X, 96) K_OF(T, E, X, 112) K_OF(T, E, X, 128)
    K_TN(false, false, false) K_TN(false, false, true) K_TN(false, true, false) K_TN(false, true, true)
    K_TN(true, false, false) K_TN(true, false, true) K_TN(true, true, false) K_TN(true, true, true)
    TORCH_CHECK(k != nullptr, "tiles of 64, 80, 96, 112 or 128 columns");
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, THREADS, bytes));
    return n;
}


// liveness.cu
bool liveness_fits(int64_t n, int64_t W, int64_t smax);
void liveness_cuda(torch::Tensor live, torch::Tensor back, torch::Tensor obs_class, torch::Tensor next_col,
                   torch::Tensor entry_ptr, torch::Tensor entry_succ, torch::Tensor width, int64_t n, int64_t W,
                   int64_t C, int64_t smax, int64_t ps, int64_t ss);

// walk.cu
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
               bool out_full);
int64_t ibtd_smem_max();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sum_cuda", &fused_sum_cuda,
          "a fused sum launch (forward, or top-down) of blocks of 32 over the same chunks, from the pre-pass's");
    m.def("ctas_per_sm", &ctas_per_sm, "how many CTAs of a direction's kernel an SM holds at once");
    m.def("liveness_fits", &liveness_fits, "whether the shared-memory liveness sweep takes an automaton");
    m.def("liveness_cuda", &liveness_cuda, "the liveness sweep, one CTA per sample in shared memory");
    m.def("ibtd_cuda", &ibtd_cuda, "an input @ block top-down launch on compact values, a sample's live terms");
    m.def("ibtd_smem_max", &ibtd_smem_max, "the most dynamic shared memory of an input @ block top-down CTA of one child");
}
