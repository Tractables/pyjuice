// The dfa_exact backend's fused sums on tensor cores, for the launches their pre-passes serve (`kernels/sum.py`):
//   forward (`fused_sum`): node blocks of 32 nodes over the same children, read in chunks of 32 children;
//   top-down (`fused_sum_top_down`, TD): child blocks of 32 children under the same parents, read in chunks of a
//   parent node block's 32 nodes -- the forward with children and nodes exchanged (a parent's weights to a child
//   block, 32 x 32, read node-major).
// Every chunk's shifted exponentials `e = exp(x - t)` and its maximum per column `t` are computed beforehand
// (`_forward_prepass_kernel` / `_top_down_prepass_kernel`, in `pre_buf`, the parents' less their log Z; at TF32, e on
// the TF32 grid already), e in the order the tensor cores read it: a chunk's 32 rows x 8 columns at a time, every
// lane's two values of a k8 step side by side (`_frag_offsets` there), so a B fragment is one 8-byte read. At TF32
// the weights may come the same way (FA): a copy of every 32 x 32 tile in A-fragment order, on the TF32 grid
// (`fragment_weights`), a fragment one 16-byte read and no conversion. At bfloat16 (BF) both operands are bf16 in
// the fragment order of mma.sync m16n8k16 -- the weights a bf16 copy, e the pre-passes' in the first half of their
// buffer -- and a chunk is two k16 steps; the sums, factors and fold are fp32 as at TF32. With external parameters
// (BlockScale, see `..external_params.block_scale`) every (row block, chunk) reads one log-scale cell s of the column's
// sample, from the session's scales stored sample-minor (a cell's samples side by side): a chunk stages every block's
// scales of the tile's samples -- a range, its columns being in sample order -- 16 bytes at a time, or, a range wider
// than the tile (samples without a column between), a scale per column; forward, every node's log Z of the sample is
// taken off.
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
#include <cuda_bf16.h>

#define NBT 4              // row blocks per CTA
#define BS 32              // rows per block, reduction rows per chunk
#define THREADS 256
#define SPLIT_TILES 4096   // the split buffer's counters (`kernels/sum.py`, SPLIT_TILES): not used here
#define MAX_SHARE_CELLS 256    // external parameters: the most chunks a share (their cells in shared memory)
#define MAX_SHARE_TABLE 128    // weights not in one run per block: the most chunks a share (their firsts too)
#define OFF_SAMPLES 255        // external parameters, lanes: up to this many samples, the column's interval's lane
                               // offsets are read once into shared memory and its columns' samples found there


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

// D += A B on bf16 operands (m16n8k16: A 16 x 16, B 16 x 8, a register two values), fp32 sums
__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A tile of TN columns: a stage of shared memory holds a chunk's weights, e, t and (external parameters) s. The
// weights' rows are their memory's (32 floats: forward one child's nodes, top-down one child's parent nodes), padded:
// A_LD 40 (forward: the products read a column of them), 36 (top-down: a row) -- or (FA) a block's 1024 in fragment
// order. e: the tile's TN / 8 groups of 8 columns, 256 floats each, in fragment order.
template <int TN, bool TD>
struct Tile {
    static constexpr int A_LD = TD ? 36 : 40;
    static constexpr int J = TN / 16;                  // a warp's tiles of 8 columns
    static constexpr int SL = TN + 4;                  // a block's scales: its samples from a multiple of 4, or columns
    static constexpr int A_ST = NBT * BS * A_LD, B_ST = BS * TN;
    static constexpr int ST = A_ST + B_ST + TN + NBT * SL;
    static constexpr int SHARE = NBT * BS * TN + NBT * TN;          // a share's sums and shifts in the split buffer
    static constexpr int HEAD = 4 * TN;                             // a tile's columns (int64) and samples first
};


struct Args {
    float* out0;                      // forward: node_mars; TD: element_flows
    float* out1;                      // TD: node_flows (aliased sums' rows)
    const float* weights;
    const float* wfrag;               // FA / BF: the weights' copy in fragment order (`fragment_weights`), same
                                      // element offsets; BF: bf16, e too (bf16 from pre on)
    const long long* wbase;           // [blocks, chunks]: a chunk's 32 x 32 weights' first
    long long wstride;                // affine: chunk ch's at the block's first + wstride ch
    const float* pre;                 // a chunk's e from chunk 32 pre_ld on, in fragment order; the chunks' t from
                                      // pre_cm on, chunk ch's column k at ch pre_ld + k
    long long pre_ld, pre_cm;
    Idx rows, rreg, colreg, sel;      // forward: nids, nb_reg, nb_reg; TD: cb_row, cb_reg, cb_ivreg
    Idx reg_offset, reg_width, reg_first, reg_slots;
    Idx perm, reg_perm, reg_live;     // prune: the lane order
    long long elem_first, elem_width; // TD: the element rows
    const float* scales;              // EXT: the external parameters' log-scales, sample-minor: cell k of sample s at
    const int* cells;                 // k sstride + s (sstride 0: every sample's, cell k at k); [blocks, chunks]: every
    long long sstride;                // chunk's cell
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
    // The pre-pass, launched ahead of the sums on the same stream (`prepass_kernel`): 0 none (`pre` filled by the
    // Triton pre-pass), 1 forward -- `values` element_mars, `cids` the children (row 0 of the blocks' table, `num_edges`
    // of them) --, 2 top-down -- `values` node_flows, the parents of the first child block (`par_ptr` at the block, its
    // slices `sl` to a block, `par_nid` / `par_reg`), less the column's sample's log Z (EXT). `frag`: e as the products
    // read it, 1 fp32 (`round_tf32`: on the TF32 grid; fp32-level products read it as it is), 2 bf16
    const float* values;
    float* pre_w;                     // `pre`, written
    Idx cids, par_ptr, par_nid, par_reg;
    int sl, prepass, frag, round_tf32;
    long long num_edges;
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


// The pre-pass of chunk `blockIdx.x` and a tile of PRE_COLS columns from `blockIdx.y PRE_COLS`: the chunk's 32 rows
// -- forward the children `cids[32 ch + i]` (element rows), top-down the parent entry's node block (TD: less the
// column's sample's log Z, EXT) --, their maximum per column t, and e = exp(x - t) in the order the sums read it
// (`_frag_offsets` / `_frag_offsets_bf16` in `kernels/sum.py`: a group of 8 columns' 32 rows contiguous, the tile's
// groups from `32 k0` of the chunk's rows), t at `pre_cm + ch pre_ld + k`. Pruned, the live columns (the region's lane
// order) alone are read, the rest -inf: e 0, as the Triton pre-pass writes them.
#define PRE_COLS 64

template <bool TD, bool EXT, int FRAG>
__global__ void __launch_bounds__(THREADS) prepass_kernel(Args a) {
    __shared__ float X[BS][PRE_COLS + 1];
    __shared__ float Tc[PRE_COLS];
    __shared__ long long cols[PRE_COLS];
    __shared__ int samp[PRE_COLS];
    const int tid = threadIdx.x, ch = blockIdx.x;
    const long long k0 = (long long)blockIdx.y * PRE_COLS;
    const long long s0 = a.indexed ? a.sel[0] : 0;                   // (every block's columns: one region)
    const long long reg = a.colreg[TD ? s0 : 0];
    const long long ncols = (long long)a.B * a.reg_slots[reg];
    if (k0 >= ncols) return;
    const long long num_live = a.prune ? (long long)a.reg_live[reg] : ncols;
    long long nid0 = 0, width = 0, base = 0;
    if (TD) {                                                        // the parent entry's node block
        const long long p = a.par_ptr[s0 / a.sl] + ch;
        nid0 = a.par_nid[p];
        const long long r = a.par_reg[p];
        width = a.reg_width[r];
        base = a.reg_offset[r] + (nid0 - a.reg_first[r]) * width;
    }
    const bool search = TD && EXT && a.lzstride != 0 && a.lanes;
    const long long obase = search ? a.reg_ivk[reg] * (a.Bq + 1) : 0;
    for (int c = tid; c < PRE_COLS; c += THREADS) {                  // the columns and their samples
        const long long k = k0 + c;
        const bool in = k < num_live;
        const long long col = in ? (a.prune ? a.perm[a.reg_perm[reg] + k] : k) : 0;
        cols[c] = in ? col : -1;
        int s = 0;
        if (TD && EXT && a.lzstride != 0 && in) {
            if (search) {
                int lo = 0, hi = a.Bq;
                for (int st = 0; st < a.steps; ++st) {
                    const int mid = (lo + hi + 1) / 2;
                    const bool go = a.coff[obase + mid] <= col;
                    lo = go ? mid : lo;
                    hi = go ? hi : mid - 1;
                }
                s = lo;
            } else {
                s = (int)(col / a.reg_slots[reg]);
            }
        }
        samp[c] = s;
    }
    __syncthreads();
    for (int i = tid; i < BS * PRE_COLS; i += THREADS) {             // x
        const int r = i / PRE_COLS, c = i % PRE_COLS;
        const long long col = cols[c];
        float x = -CUDART_INF_F;
        if (col >= 0) {
            if (TD) {
                x = a.values[base + r * width + col];
                if (EXT) x -= a.lz[(long long)samp[c] * a.lzstride + nid0 + r];
            } else {
                const long long e = (long long)ch * BS + r;
                if (e < a.num_edges) {
                    const long long cid = a.cids[e];
                    if (cid >= a.elem_first) x = a.values[(cid - a.elem_first) * a.elem_width + col];
                }
            }
        }
        X[r][c] = x;
    }
    __syncthreads();
    for (int c = tid; c < PRE_COLS; c += THREADS) {                  // t
        float t = -CUDART_INF_F;
        for (int r = 0; r < BS; ++r) t = fmaxf(t, X[r][c]);
        Tc[c] = t;
        const long long k = k0 + c;
        if (k < ncols) a.pre_w[a.pre_cm + (long long)ch * a.pre_ld + k] = t;
    }
    __syncthreads();
    const long long ebase = (long long)ch * BS * a.pre_ld + k0 * BS;   // e: the tile's groups, contiguous
    for (int i = tid; i < BS * PRE_COLS; i += THREADS) {
        const int g = i / 256, o = i % 256;
        int r, c;
        if (FRAG == 2) {                                              // (m16n8k16 B fragments, four values a lane)
            r = (o / 128) * 16 + ((o % 4) / 2) * 8 + ((o % 16) / 4) * 2 + (o % 2);
            c = g * 8 + (o % 128) / 16;
        } else {                                                      // (m16n8k8 B fragments, two values a lane)
            r = (o / 64) * 8 + (o % 2) * 4 + (o % 8) / 2;
            c = g * 8 + (o % 64) / 8;
        }
        if (k0 + c >= ncols) continue;
        const float t = Tc[c];
        const float e = __expf(X[r][c] - (t == -CUDART_INF_F ? 0.f : t));
        if (FRAG == 2) reinterpret_cast<__nv_bfloat16*>(a.pre_w)[ebase + i] = __float2bfloat16_rn(e);
        else a.pre_w[ebase + i] = a.round_tf32 ? __uint_as_float(to_tf32(e)) : e;
    }
}


template <bool EXT, bool X3, bool FA, bool BF, int TN, bool TD>
__global__ void __launch_bounds__(THREADS) fused_sum_kernel(Args a) {
    using TT = Tile<TN, TD>;
    extern __shared__ float smem[];
    float* Ts = smem + 2 * TT::ST;                                  // [NBT][TN] the shifts
    long long* Cs = reinterpret_cast<long long*>(Ts + NBT * TN);   // [TN] the columns
    int* Ss = reinterpret_cast<int*>(Cs + TN);                      // [TN] their samples
    int* Cl = Ss + TN;                                              // EXT: [NBT][share] the chunks' cells
    long long* Off = reinterpret_cast<long long*>(Cl + (EXT ? NBT * a.share : 0));  // EXT: lanes' offsets
    long long* Wb = Off + (EXT ? OFF_SAMPLES + 1 : 0);              // !affine: [NBT][share]
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
    // lanes: a column's sample is the last whose lanes start at or before it (the interval's offsets, Bq + 1 of them)
    const bool search = EXT && a.sstride != 0 && a.lanes, near = search && a.Bq <= OFF_SAMPLES;
    const long long obase = search ? a.reg_ivk[w.reg] * (a.Bq + 1) : 0;
    if (near)                                           // (read once)
        for (int i = tid; i <= a.Bq; i += THREADS) Off[i] = a.coff[obase + i];
    for (int i = tid; i < NBT * a.share; i += THREADS) {  // the share's cells and (not one run) first weights
        const int b = i / a.share, ch = ch0 + i % a.share;
        const bool v = w.nbs[b] >= 0 && ch < ch1;
        if (EXT) Cl[i] = v ? a.cells[w.nbs[b] * a.nch + ch] : 0;
        if (!a.affine) Wb[i] = v ? a.wbase[w.nbs[b] * a.nch + ch] : 0;
    }
    long long w0[NBT];                                  // affine: every block's first weight
    #pragma unroll
    for (int b = 0; b < NBT; ++b) w0[b] = a.affine && w.nbs[b] >= 0 ? a.wbase[w.nbs[b] * a.nch] : 0;
    if (near) __syncthreads();                          // (the offsets in)
    for (int c = tid; c < TN; c += THREADS) {           // every column and its sample
        long long k = col0 + c;
        long long col = k < w.ncols ? (a.prune ? a.perm[a.reg_perm[w.reg] + k] : k) : 0;
        int s = 0;
        if (EXT && a.sstride != 0) {
            if (search) {
                int lo = 0, hi = a.Bq;
                for (int st = 0; st < a.steps; ++st) {
                    int mid = (lo + hi + 1) / 2;
                    bool go = (near ? Off[mid] : a.coff[obase + mid]) <= col;
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
    __syncthreads();
    const int lo4 = s_lo & ~3, span = s_hi - lo4 + 1;   // EXT: the tile's samples, from a multiple of 4
    const bool ranged = EXT && a.sstride != 0 && span <= TT::SL;    // (else a scale per column)
    const int q4 = (span + 3) / 4;                      // ranged: a block's copies of 16 bytes

    // The copies a thread issues every chunk and their invariants -- which blocks are live (a mask), the thread's
    // place in a weight tile and its first weight of each block (affine: advanced by wstride a chunk; else the share's
    // table), its e copies (E_COPIES of 16 bytes, THREADS copies apart; a bit each: within the tile and its column one
    // of the launch's), its t copy (its bytes and columns), its scale copy (a block's, a block's 4 samples', or a
    // (block, column)'s) -- settled once where the registers allow (DESC), so that a chunk's load adds the chunk's
    // offsets alone: settled every chunk, with the arguments reloaded and divisions by q4, they ran ~800 instructions
    // a warp a chunk for 15 copies, 0.75 of a chunk's 1.1 us (Codpula hard_0 B8, tiles of 16: the copies themselves
    // were never waited on). The three-pass products past 64 columns and the fp32 operands at 128 sit at the register
    // cap: their descriptors are settled per chunk, as before, rather than spilled.
    constexpr bool DESC = BF || TN <= (X3 ? 64 : 112);
    constexpr int E_N = BF ? BS * TN / 8 : BS * TN / 4;              // the tile's e copies (8 bf16 or 4 floats each)
    constexpr int E_COPIES = (E_N + THREADS - 1) / THREADS;          // a thread's
    constexpr int S_COPIES = (NBT * TN + THREADS - 1) / THREADS;     // scales by (block, column): a thread's copies
    const long long ncols = w.ncols;
    const int wr = tid / (BS / 4), wq = (tid % (BS / 4)) * 4;         // (weights as they are: the thread's row, 4 columns)
    const int wt_off = BF ? 8 * tid : FA ? 4 * tid : wr * BS + wq;    // the thread's elements in a weight tile
    const int wd_off = FA ? 4 * tid : wr * TT::A_LD + wq;             // and in a stage's (BF: bf16 units, wt_off)
    #pragma unroll
    for (int b = 0; b < NBT; ++b) w0[b] += wt_off;                    // (affine: the thread's first weight of chunk 0)
    const int s_mode = !EXT ? 0 : a.sstride == 0 ? 1 : ranged ? 2 : 3;   // 1 a block's, 2 a block's samples, 3 by column
    unsigned vmask = 0, emask = 0, smask = 0;                         // the live blocks, the e copies and the scale
    long long t_pos = 0;                                              // copies by column that are; t: the thread's 4
    int t_bytes = 0;                                                  // columns and their bytes
    int s_b = 0, s_k = 0;                                             // mode 2: the thread's block and first sample
    bool s_v = false;
    int s_slot[S_COPIES];                                             // mode 3: each copy's sample
    auto settle = [&]() {
        vmask = 0;
        #pragma unroll
        for (int b = 0; b < NBT; ++b) vmask |= (w.nbs[b] >= 0 ? 1u : 0u) << b;
        emask = 0;
        #pragma unroll
        for (int m = 0; m < E_COPIES; ++m) {
            const int i = tid + THREADS * m;
            const long long col = col0 + 8 * (i / (BF ? 32 : 64)) + (i % 16) / 2;
            if (i < E_N && col < ncols) emask |= 1u << m;
        }
        t_pos = 0;
        t_bytes = 0;
        if (tid < TN / 4) {
            t_pos = col0 + 4 * tid;
            t_bytes = (int)max(0LL, min(16LL, 4 * (ncols - t_pos)));
            if (t_bytes == 0) t_pos = 0;
        }
        s_b = s_k = 0;
        s_v = false;
        if (s_mode == 2 && tid < NBT * q4) {
            s_b = tid / q4;
            s_k = 4 * (tid % q4);
            s_v = (vmask >> s_b) & 1;
        } else if (s_mode == 1 && tid < NBT) {
            s_v = (vmask >> tid) & 1;
        }
        smask = 0;
        #pragma unroll
        for (int m = 0; m < S_COPIES; ++m) {
            const int i = tid + THREADS * m, b = i / TN, c = i % TN;
            s_slot[m] = s_mode == 3 && i < NBT * TN ? Ss[c] : 0;
            if (s_mode == 3 && i < NBT * TN && ((vmask >> b) & 1) && col0 + c < ncols) smask |= 1u << m;
        }
    };
    if (DESC) settle();

    auto load = [&](int ch, int st) {                   // chunk ch into stage st: the thread's copies at its offsets
        if (!DESC) settle();                            // (at the register cap: settled every chunk)
        float* As = smem + (st ? TT::ST : 0);
        float* Bs = As + TT::A_ST;
        float* tS = Bs + TT::B_ST;
        float* sS = tS + TN;
        const int k = ch - ch0;
        const long long wch = (long long)ch * a.wstride;
        #pragma unroll
        for (int b = 0; b < NBT; ++b) {                 // the weights' 32 rows, 16 B at a time: 4 copies a thread
            const long long first = a.affine ? w0[b] + wch : Wb[b * a.share + k] + wt_off;
            const int bytes = (vmask >> b) & 1 ? 16 : 0;
            if (BF) {                                   // (the bf16 copy: the tile's 1024 in a row, 2 KB)
                if (tid < 128)
                    cp_async16(reinterpret_cast<__nv_bfloat16*>(As + b * BS * TT::A_LD) + wt_off,
                               reinterpret_cast<const __nv_bfloat16*>(a.wfrag) + first, bytes);
            } else if (FA)                              // (the copy: the tile's 1024 floats in a row)
                cp_async16(As + b * BS * TT::A_LD + wd_off, a.wfrag + first, bytes);
            else
                cp_async16(As + b * BS * TT::A_LD + wd_off, a.weights + first, bytes);
        }
        const long long ebase = (long long)ch * BS * a.pre_ld + col0 * BS;   // e: the tile's groups in a row; a
        #pragma unroll                                                       // copy one column's 8 bf16 or 4 floats
        for (int m = 0; m < E_COPIES; ++m) {
            const int i = tid + THREADS * m;
            if (i < E_N) {
                const bool v = (emask >> m) & 1;
                if (BF)
                    cp_async16(reinterpret_cast<__nv_bfloat16*>(Bs) + 8 * i,
                               reinterpret_cast<const __nv_bfloat16*>(a.pre) + ebase + (v ? 8 * i : 0), v ? 16 : 0);
                else
                    cp_async16(Bs + 4 * i, a.pre + ebase + (v ? 4 * i : 0), v ? 16 : 0);
            }
        }
        if (tid < TN / 4)                                             // t
            cp_async16(tS + 4 * tid, a.pre + a.pre_cm + (long long)ch * a.pre_ld + t_pos, t_bytes);
        if (EXT) {                                                    // s
            if (s_mode == 1) {                                        // every sample's: a block's
                if (tid < NBT) cp_async4(sS + tid * TT::SL, a.scales + (s_v ? Cl[tid * a.share + k] : 0), s_v);
            } else if (s_mode == 2) {                                 // a block's samples, 4 a copy
                if (tid < NBT * q4) {
                    const long long at = s_v ? (long long)Cl[s_b * a.share + k] * a.sstride + lo4 + s_k : 0;
                    cp_async16(sS + s_b * TT::SL + s_k, a.scales + at, s_v ? 16 : 0);
                }
            } else {                                                  // a (block, column)'s
                #pragma unroll
                for (int m = 0; m < S_COPIES; ++m) {
                    const int i = tid + THREADS * m, b = i / TN, c = i % TN;
                    if (i < NBT * TN) {
                        const bool v = (smask >> m) & 1;
                        const long long at = v ? (long long)Cl[b * a.share + k] * a.sstride + s_slot[m] : 0;
                        cp_async4(sS + b * TT::SL + c, a.scales + at, v);
                    }
                }
            }
        }
    };
    load(ch0, 0);                                       // (under pass 1)
    cp_commit();

    {   // pass 1: T = max over the share's chunks of t + s -- a thread's one or two (block, column)s together, which of
        // them are live tested once, outside the loops (a loop without a test: its loads in flight together)
        static_assert(NBT * TN <= 2 * THREADS, "a thread's one or two (block, column)s");   // (narrow tiles: idle threads)
        const int i1 = tid + THREADS;
        const int b0 = min(tid / TN, NBT - 1), c0 = tid % TN, b1 = i1 / TN, c1 = i1 % TN;
        const bool v0 = tid < NBT * TN && w.nbs[b0] >= 0 && col0 + c0 < w.num_live;
        const bool v1 = i1 < NBT * TN && w.nbs[min(b1, NBT - 1)] >= 0 && col0 + c1 < w.num_live;
        const float* tp0 = a.pre + a.pre_cm + col0 + c0;
        const float* tp1 = a.pre + a.pre_cm + col0 + c1;
        const long long cm = a.sstride != 0 ? a.sstride : 1;          // (sample-minor: a warp's columns side by side)
        const float* sp0 = a.scales + (a.sstride != 0 ? Ss[c0] : 0);
        const float* sp1 = a.scales + (a.sstride != 0 ? Ss[c1] : 0);
        float T0 = -CUDART_INF_F, T1 = -CUDART_INF_F;
        const int* cl0 = Cl + b0 * a.share - ch0;
        const int* cl1 = Cl + min(b1, NBT - 1) * a.share - ch0;
        if (v0 && v1) {
            #pragma unroll 16
            for (int ch = ch0; ch < ch1; ++ch) {
                float t0 = tp0[(long long)ch * a.pre_ld], t1 = tp1[(long long)ch * a.pre_ld];
                if (EXT) {
                    t0 += sp0[cl0[ch] * cm];
                    t1 += sp1[cl1[ch] * cm];
                }
                T0 = fmaxf(T0, t0);
                T1 = fmaxf(T1, t1);
            }
        } else if (v0) {
            #pragma unroll 16
            for (int ch = ch0; ch < ch1; ++ch) {
                float t0 = tp0[(long long)ch * a.pre_ld];
                if (EXT) t0 += sp0[cl0[ch] * cm];
                T0 = fmaxf(T0, t0);
            }
        } else if (v1) {
            #pragma unroll 16
            for (int ch = ch0; ch < ch1; ++ch) {
                float t1 = tp1[(long long)ch * a.pre_ld];
                if (EXT) t1 += sp1[cl1[ch] * cm];
                T1 = fmaxf(T1, t1);
            }
        }
        if (tid < NBT * TN) Ts[tid] = T0;
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
    int slot[2];                                        // EXT: the lane's two columns' scales in a block's
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int c = wc + 2 * lane + h;
        slot[h] = !EXT || !fl || a.sstride == 0 ? 0 : ranged ? min(max(Ss[c] - lo4, 0), TT::SL - 1) : c;
    }
    __syncthreads();                                    // (pass 1's shifts in: a factor lane's two, read once)
    float Tl[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) Tl[h] = DESC && fl ? Ts[wb * TN + wc + 2 * lane + h] : -CUDART_INF_F;

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
        const float* sS = tS + TN + wb * TT::SL;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = wc + 2 * lane + h;
            G[h] = 0.f;
            if (fl) {
                const float T = DESC ? Tl[h] : Ts[wb * TN + c];
                const float t = !EXT ? tS[c] : tS[c] + sS[slot[h]];
                G[h] = T != -CUDART_INF_F && t != -CUDART_INF_F ? __expf(t - T) : 0.f;
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < TT::J; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) P[i][j][q] = 0.f;
        if (BF) {                                       // two k16 steps: an A fragment one 16-byte read, a B one 8
            #pragma unroll
            for (int k16 = 0; k16 < BS / 16; ++k16) {
                unsigned ab[2][4];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 v = reinterpret_cast<const uint4*>(A)[(k16 * 2 + i) * 32 + lane];
                    ab[i][0] = v.x;
                    ab[i][1] = v.y;
                    ab[i][2] = v.z;
                    ab[i][3] = v.w;
                }
                #pragma unroll
                for (int j = 0; j < TT::J; ++j) {
                    const uint2 y = reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(Bt) +
                                                                   (wc / 8 + j) * 256 + k16 * 128)[lane];
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) mma_bf16(P[i][j], ab[i], y.x, y.y);
                }
            }
        } else
        #pragma unroll
        for (int k8 = 0; k8 < BS / 8; ++k8) {
            unsigned ah[2][4], al[2][4];
            if (FA) {                                   // A in fragment order, on the TF32 grid: one 16-byte read
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float4 v = reinterpret_cast<const float4*>(A)[(k8 * 2 + i) * 32 + lane];
                    ah[i][0] = __float_as_uint(v.x);
                    ah[i][1] = __float_as_uint(v.y);
                    ah[i][2] = __float_as_uint(v.z);
                    ah[i][3] = __float_as_uint(v.w);
                }
            } else
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
            for (int j = 0; j < TT::J; ++j) {           // B (column wc + 8 j + lane / 4; rows kk, kk + 4): 8 bytes
                const float2 y = reinterpret_cast<const float2*>(Bt + (wc / 8 + j) * 256 + k8 * 64)[lane];
                const float y0 = y.x, y1 = y.y;
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
    const int st = NBT * BS * (td ? 36 : 40) + BS * tn + tn + NBT * (tn + 4);
    return (2 * st + NBT * tn) * sizeof(float) + tn * (sizeof(long long) + sizeof(int)) +
           (ext ? NBT * share * sizeof(int) + (OFF_SAMPLES + 1) * sizeof(long long) : 0) +
           (affine ? 0 : NBT * share * sizeof(long long));
}

// A kernel may take as much shared memory as a block may have (set once)
template <bool EXT, bool X3, bool FA, bool BF, int TN, bool TD>
static const void* allowed() {
    static const void* k = [] {
        const void* f = (const void*)fused_sum_kernel<EXT, X3, FA, BF, TN, TD>;
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

template <bool TD, bool EXT>
static void launch_prepass(const Args& a, unsigned col_tiles, cudaStream_t stream) {
    dim3 pgrid((unsigned)a.nch, col_tiles);
    if (a.frag == 2) prepass_kernel<TD, EXT, 2><<<pgrid, THREADS, 0, stream>>>(a);
    else prepass_kernel<TD, EXT, 1><<<pgrid, THREADS, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool EXT, bool X3, bool FA, bool BF, int TN, bool TD>
static void launch(const Args& a, dim3 grid, unsigned pre_tiles, cudaStream_t stream) {
    if (a.prepass) launch_prepass<TD, EXT>(a, pre_tiles, stream);
    allowed<EXT, X3, FA, BF, TN, TD>();
    const size_t bytes = smem_bytes(TN, TD, EXT, a.share, a.affine);
    fused_sum_kernel<EXT, X3, FA, BF, TN, TD><<<grid, THREADS, bytes, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (a.split > 1) {
        dim3 mgrid(grid.x, grid.y, (NBT * BS * TN + THREADS - 1) / THREADS);
        fused_sum_merge_kernel<EXT, TN, TD><<<mgrid, THREADS, 0, stream>>>(a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <bool EXT, bool X3, bool FA, bool BF, bool TD>
static void launch_tn(const Args& a, int tn, dim3 grid, unsigned pre_tiles, cudaStream_t stream) {
    switch (tn) {
        case 16: return launch<EXT, X3, FA, BF, 16, TD>(a, grid, pre_tiles, stream);
        case 32: return launch<EXT, X3, FA, BF, 32, TD>(a, grid, pre_tiles, stream);
        case 64: return launch<EXT, X3, FA, BF, 64, TD>(a, grid, pre_tiles, stream);
        case 80: return launch<EXT, X3, FA, BF, 80, TD>(a, grid, pre_tiles, stream);
        case 96: return launch<EXT, X3, FA, BF, 96, TD>(a, grid, pre_tiles, stream);
        case 112: return launch<EXT, X3, FA, BF, 112, TD>(a, grid, pre_tiles, stream);
        case 128: return launch<EXT, X3, FA, BF, 128, TD>(a, grid, pre_tiles, stream);
    }
    TORCH_CHECK(false, "tiles of 16, 32, 64, 80, 96, 112 or 128 columns");
}


// One launch of either direction: the tables as `kernels/sum.py` passes them (`_cuda_forward`, `_cuda_top_down`)
void fused_sum_cuda(bool td, torch::Tensor out0, torch::Tensor out1, torch::Tensor weights, torch::Tensor wfrag,
                    torch::Tensor wbase, int64_t wstride, torch::Tensor pre, int64_t pre_ld, int64_t pre_cm,
                    torch::Tensor rows, torch::Tensor rreg, torch::Tensor colreg, torch::Tensor sel,
                    torch::Tensor reg_offset, torch::Tensor reg_width, torch::Tensor reg_first, torch::Tensor reg_slots,
                    torch::Tensor perm, torch::Tensor reg_perm, torch::Tensor reg_live, int64_t elem_first,
                    int64_t elem_width, torch::Tensor scales, torch::Tensor cells, int64_t sstride, torch::Tensor lz,
                    int64_t lzstride, torch::Tensor coff, torch::Tensor reg_ivk, int64_t Bq, int64_t steps, int64_t B,
                    int64_t num_blocks, int64_t nch, int64_t max_cols, bool ext, bool shared, bool lanes, bool prune,
                    bool indexed, bool x3, bool affine, int64_t tn, torch::Tensor split_buf, int64_t split,
                    torch::Tensor values, torch::Tensor cids, torch::Tensor par_ptr, torch::Tensor par_nid,
                    torch::Tensor par_reg, int64_t sl, int64_t prepass, int64_t frag, int64_t num_edges) {
    TORCH_CHECK(out0.is_cuda() && out0.scalar_type() == torch::kFloat32 && out1.scalar_type() == torch::kFloat32,
                "outputs: fp32 on the device");
    TORCH_CHECK(prepass == 0 || (values.scalar_type() == torch::kFloat32 && (frag == 1 || frag == 2)),
                "the pre-pass: fp32 values, e in fp32 or bf16");
    TORCH_CHECK(wbase.scalar_type() == torch::kInt64 && cells.scalar_type() == torch::kInt32,
                "wbase int64, cells int32");
    const bool copy = wfrag.numel() > 0;            // (empty: the weights as they are)
    const bool bf = copy && wfrag.scalar_type() == torch::kBFloat16, fa = copy && !bf;   // (bf16: bf16 operands)
    TORCH_CHECK(!copy || (!x3 && wfrag.numel() == weights.numel() &&
                          (bf || wfrag.scalar_type() == torch::kFloat32)),
                "fragment-ordered weights: fp32 (TF32) or bf16, as many as the weights, not at fp32");
    Args a;
    a.out0 = out0.data_ptr<float>();
    a.out1 = out1.data_ptr<float>();
    a.weights = weights.data_ptr<float>();
    a.wfrag = copy ? reinterpret_cast<const float*>(wfrag.data_ptr()) : nullptr;
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
    a.prepass = (int)prepass;
    a.frag = (int)frag;
    a.round_tf32 = !x3 && frag == 1;
    a.num_edges = num_edges;
    a.sl = (int)sl;
    a.values = prepass ? values.data_ptr<float>() : nullptr;
    a.pre_w = pre.data_ptr<float>();
    a.cids = idx_of(cids);
    a.par_ptr = idx_of(par_ptr);
    a.par_nid = idx_of(par_nid);
    a.par_reg = idx_of(par_reg);
    const unsigned pre_tiles = (unsigned)((max_cols + PRE_COLS - 1) / PRE_COLS);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int t = (int)tn;
#define DISPATCH(E, X, F, BF) if (ext == E && x3 == X && fa == F && bf == BF) \
        return td ? launch_tn<E, X, F, BF, true>(a, t, grid, pre_tiles, stream) \
                  : launch_tn<E, X, F, BF, false>(a, t, grid, pre_tiles, stream);
    DISPATCH(false, false, false, false) DISPATCH(false, true, false, false) DISPATCH(true, false, false, false)
    DISPATCH(true, true, false, false) DISPATCH(false, false, true, false) DISPATCH(true, false, true, false)
    DISPATCH(false, false, false, true) DISPATCH(true, false, false, true)
}


// The TF32 weights in the order the fused sums' A fragments read them (FA): every 32 x 32 tile `wbase` names (a CTA
// one), rounded to TF32 as the products round them (`cvt.rna`) and stored at the tile's own place in `frag`, a
// fragment's four values side by side -- fragment (k8, i) of lane l at (2 k8 + i) 128 + 4 l: rows i 16 + l / 4 (+ 8),
// reduction rows k8 8 + l % 4 (+ 4); a row is a node (forward, the tile's column) or a child (TD, its row)
template <bool TD>
__global__ void __launch_bounds__(THREADS) fragment_weights_kernel(const float* w, float* frag,
                                                                   const long long* wbase) {
    const long long first = wbase[blockIdx.x];
    const int f = 4 * threadIdx.x, lane = (f >> 2) & 31, i = (f >> 7) & 1, k8 = f >> 8;
    const int m = i * 16 + (lane >> 2), kk = k8 * 8 + (lane & 3);
    const float* t = w + first;
    float4 v;                                           // (q 0: m, kk; 1: m + 8, kk; 2: m, kk + 4; 3: m + 8, kk + 4)
    v.x = __uint_as_float(to_tf32(TD ? t[m * BS + kk] : t[kk * BS + m]));
    v.y = __uint_as_float(to_tf32(TD ? t[(m + 8) * BS + kk] : t[kk * BS + m + 8]));
    v.z = __uint_as_float(to_tf32(TD ? t[m * BS + kk + 4] : t[(kk + 4) * BS + m]));
    v.w = __uint_as_float(to_tf32(TD ? t[(m + 8) * BS + kk + 4] : t[(kk + 4) * BS + m + 8]));
    reinterpret_cast<float4*>(frag + first)[threadIdx.x] = v;
}

// The bf16 copy (BF): the weights rounded to bf16 (to nearest even) in m16n8k16's A-fragment order -- fragment
// (k16, i) of lane l at ((2 k16 + i) 32 + l) 8, its registers 0-3 two values each: rows i 16 + l / 4 (registers 1, 3:
// + 8), reduction rows k16 16 + 2 (l % 4), + 1 (registers 2, 3: + 8); a thread four values (half a lane's)
template <bool TD>
__global__ void __launch_bounds__(THREADS) fragment_weights_bf16_kernel(const float* w, __nv_bfloat16* frag,
                                                                        const long long* wbase) {
    const long long first = wbase[blockIdx.x];
    const int f = 4 * threadIdx.x, lane = (f >> 3) & 31, i = (f >> 8) & 1, k16 = f >> 9;
    const int g = lane >> 2, t = lane & 3;
    const float* tile = w + first;
    __align__(8) __nv_bfloat16 v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int r = (f & 7) + q, reg = r >> 1, half = r & 1;
        const int m = i * 16 + g + ((reg & 1) ? 8 : 0), kk = k16 * 16 + 2 * t + half + ((reg & 2) ? 8 : 0);
        v[q] = __float2bfloat16_rn(TD ? tile[m * BS + kk] : tile[kk * BS + m]);
    }
    reinterpret_cast<uint2*>(frag + first)[threadIdx.x] = *reinterpret_cast<const uint2*>(v);
}

void fragment_weights(torch::Tensor weights, torch::Tensor frag, torch::Tensor wbase, bool td) {
    const bool bf = frag.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 && (bf || frag.scalar_type() == torch::kFloat32) &&
                frag.numel() == weights.numel() && wbase.scalar_type() == torch::kInt64 && wbase.is_contiguous(),
                "fp32 weights, their copy fp32 (TF32) or bf16, int64 firsts");
    const long long n = wbase.numel();
    TORCH_CHECK(n < (1LL << 31), "at most 2^31 - 1 tiles");
    if (n == 0) return;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const long long* wb = reinterpret_cast<const long long*>(wbase.data_ptr<int64_t>());
    const float* w = weights.data_ptr<float>();
    if (bf) {
        __nv_bfloat16* f = reinterpret_cast<__nv_bfloat16*>(frag.data_ptr());
        if (td)
            fragment_weights_bf16_kernel<true><<<(unsigned)n, THREADS, 0, stream>>>(w, f, wb);
        else
            fragment_weights_bf16_kernel<false><<<(unsigned)n, THREADS, 0, stream>>>(w, f, wb);
    } else {
        float* f = frag.data_ptr<float>();
        if (td)
            fragment_weights_kernel<true><<<(unsigned)n, THREADS, 0, stream>>>(w, f, wb);
        else
            fragment_weights_kernel<false><<<(unsigned)n, THREADS, 0, stream>>>(w, f, wb);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// How many CTAs of a direction's kernel an SM holds at once (for the split's waves)
int64_t ctas_per_sm(bool td, bool ext, bool x3, bool fa, bool bf, int64_t tn, int64_t share, bool affine) {
    const size_t bytes = smem_bytes((int)tn, td, ext, (int)share, affine);
    const void* k = nullptr;
#define K_OF(T, E, X, F, B, N) \
    if (td == T && ext == E && x3 == X && fa == F && bf == B && tn == N) k = allowed<E, X, F, B, N, T>();
#define K_TN(T, E, X, F, B) \
    K_OF(T, E, X, F, B, 16) K_OF(T, E, X, F, B, 32) \
    K_OF(T, E, X, F, B, 64) K_OF(T, E, X, F, B, 80) K_OF(T, E, X, F, B, 96) K_OF(T, E, X, F, B, 112) \
    K_OF(T, E, X, F, B, 128)
#define K_TD(T) K_TN(T, false, false, false, false) K_TN(T, false, true, false, false) \
    K_TN(T, true, false, false, false) K_TN(T, true, true, false, false) K_TN(T, false, false, true, false) \
    K_TN(T, true, false, true, false) K_TN(T, false, false, false, true) K_TN(T, true, false, false, true)
    K_TD(false) K_TD(true)
    TORCH_CHECK(k != nullptr, "tiles of 16, 32, 64, 80, 96, 112 or 128 columns (fragment-ordered weights: TF32 or bf16)");
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, THREADS, bytes));
    return n;
}


// liveness.cu
bool liveness_fits(int64_t n, int64_t W, int64_t smax);
void liveness_wide_cuda(torch::Tensor live, torch::Tensor back, torch::Tensor obs_class, torch::Tensor next_col,
                        torch::Tensor delta, torch::Tensor state_id, torch::Tensor col_of_state,
                        torch::Tensor entry_ptr, torch::Tensor entry_succ, torch::Tensor width, int64_t n, int64_t W,
                        int64_t C, int64_t K, int64_t ps, int64_t ss, bool homogeneous, bool every_class,
                        torch::Tensor scratch);
void liveness_cuda(torch::Tensor live, torch::Tensor back, torch::Tensor obs_class, torch::Tensor next_col,
                   torch::Tensor delta, torch::Tensor state_id, torch::Tensor col_of_state, torch::Tensor entry_ptr,
                   torch::Tensor entry_succ, torch::Tensor width, int64_t n, int64_t W, int64_t C, int64_t K,
                   int64_t smax, int64_t ps, int64_t ss, bool homogeneous);

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
               bool out_full, torch::Tensor code, torch::Tensor tok_ptr, torch::Tensor eidx, torch::Tensor eprobs,
               torch::Tensor prow, torch::Tensor lognorm, torch::Tensor enum_logw, int64_t Cs, int64_t mw,
               int64_t num_enum, int64_t logw_stride, bool logw, torch::Tensor pred_class, torch::Tensor allowed,
               int64_t a_stride, bool classes);
int64_t ibtd_smem_max();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sum_cuda", &fused_sum_cuda,
          "a fused sum launch (forward, or top-down) of blocks of 32 over the same chunks, from the pre-pass's");
    m.def("ctas_per_sm", &ctas_per_sm, "how many CTAs of a direction's kernel an SM holds at once");
    m.def("fragment_weights", &fragment_weights,
          "the weights' tiles a launch reads, in its A-fragment order: TF32 (an fp32 copy) or bf16 (a bf16 copy)");
    m.def("liveness_fits", &liveness_fits, "whether the shared-memory liveness sweep takes an automaton");
    m.def("liveness_cuda", &liveness_cuda, "the liveness sweep, one CTA per sample in shared memory");
    m.def("liveness_wide_cuda", &liveness_wide_cuda,
          "the liveness sweep of an automaton too wide for shared memory, one CTA per sample and direction");
    m.def("ibtd_cuda", &ibtd_cuda, "an input @ block top-down launch on compact values, a sample's live terms");
    m.def("ibtd_smem_max", &ibtd_smem_max, "the most dynamic shared memory of an input @ block top-down CTA of one child");
}
