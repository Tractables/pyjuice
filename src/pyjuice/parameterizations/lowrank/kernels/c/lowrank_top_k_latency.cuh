// pyjuice: the LATENCY mode of a stack's forward (`TopKStackPlan`, `lowrank/top_k_stack.py`): the small batches'.
//
// A sum position's value is, for a node i and a sample b (its row of tensors e),
//
//     v_i = log sum_s q_is U_is (masked: q_is = w_is / Z_is, every slot)    v_i = log U_i / Z_i (observed: its slot)
//
//     Z_is = sum_j W_ij E_sj      U_is = sum_j W_ij E_sj exp(c_j)           W_ij = theta_ij exp(A_i . B_j)
//
// Z (and W's reference, the rows' largest logits) reads the tensors alone, U the children's values c_j too: the
// chain. The throughput mode (`tk_sk_kernel`) makes both a position at once, its tiles' 16 chunks shared out among a
// few CTAs that fold their partials in turn -- ~26 us a position at 2 samples, latency all of it. Here:
//
// * OFF the chain, one launch for every position (`tk_lat_zh_kernel`, after the leaf's forward: an observed sample's
//   slot is its token's place): each (position, sample, tile)'s Z, its 16 chunks in one pass -- masked: every slot, by
//   the fp16 MMAs of W and E the throughput mode takes (E16, W times 2^W_OFF at a reference raised as the chunks come,
//   the throughput mode's lazy rule: `exp_tile`), the rest slot by fp32 sums, and EACH CHUNK'S REFERENCE kept (R), so
//   that the chain's W of a chunk is made of the same fp16 operands to the bit -- U and Z alike rounded (MEASURED,
//   every token masked, each likelihood 1: Z's W at another reference than U's, the log-likelihoods' error 5e-5
//   against 1e-6); observed: its slot by fp32 sums (W at a lazy reference: fp32 sums alike either way) -- and the
//   reference (h, log2) the sums end at.
// * ON the chain, one launch a position (`tk_lat_step_kernel`, PDL: the launch after it starts as soon as this one's
//   chain-free part is done), a CTA a (chunk of 64 children, tile of 128 nodes, sample): before the wait for the
//   position below, W of its chunk (the logits' MMAs) at the chunk's reference (masked: R; observed: h); after it, its
//   children's values (shifted by their largest in the chunk: no position-wide maximum), W exp(c) -- masked: in fp16
//   into the slots' MMAs, as the throughput mode makes U --, and at once its share of the value's sum, R_i = sum_s
//   q_is U_is (Z complete: off the chain; U at the chunk's reference, times 2^(R - h)) -- one float a node. The
//   tile's last CTA (a count) adds its 16 shares (their shifts to the largest, in the chunks' order: no atomics on
//   values) into the values.
//
// The same operands and roundings as the throughput mode's (the fp16 factors, W and W exp(c) as fp16 MMA operands for
// the masked slots, fp32 sums elsewhere); the sums reassociated (U a chunk at a time, each chunk's children shifted by
// its own largest). Every sample's own work items (a sample a CTA: no tile shared by the samples reading a row).

// a position's slot sums off the chain: each reader's operands (`TopKStackPlan`)
struct LatZh {
    const float* A; long sA; const __half* BP; const __half* E16; const uint2* LTh;
    float* h;                          // [B, M] the sums' reference (log2)
    float* Z;                          // [B, M, SP] masked: slots 0 .. 64; observed: [0]
    float* R;                          // [B, N / TN, M] masked: each chunk's W's reference (log2)
    int r, M, N, S, t;                 // t: its leaf's place among the layer's (`wrow`, `slot32` rows)
};

// (masked or observed, a sample's: its row of the slot weights, -1 where observed; its slot where observed)
__device__ __forceinline__ void lat_kind(const int* wrow, const int* slot32, int t, int B, int b, int& wr, int& sl) {
    wr = wrow[(long)t * B + b];
    sl = wr >= 0 ? 0 : slot32[(long)t * B + b];
}

// the stage's segments: the B chunk [KP][BROW] fp16, then E16's rows -- every slot's (masked) or the sample's own slot's
// (observed, row 0) -- of the chunk's 64 children
template <int KP>
__device__ __forceinline__ void lat_load(unsigned bs, const __half* BP, const __half* E16, int e, int S, int N, int j0,
                                         bool masked, int sl, int tid) {
    constexpr int NB = KP * (TN / 8);
    const int nseg = NB + (masked ? (8 * NS8 + 1) : 1) * (TN / 8);
    for (int q = tid; q < nseg; q += NW * 32) {
        if (q < NB) {
            const int k = q / (TN / 8), s = q % (TN / 8);
            cp16(bs + (k * BROW + s * 8) * 2, BP + ((long)e * KP + k) * N + j0 + s * 8);
        } else {
            const int qq = q - NB, row = qq / (TN / 8), s = qq % (TN / 8);
            cp16(bs + KP * BROW * 2 + (row * BROW + s * 8) * 2,
                 E16 + ((long)e * S + (masked ? row : sl)) * N + j0 + s * 8);
        }
    }
    cp_commit();
}

// grid (M / TM tiles, B samples, readers): a CTA a tile of a sample's position, its 16 chunks in turn
template <int NK>
__global__ void __launch_bounds__(NW * 32) tk_lat_zh_kernel(const LatZh* __restrict__ readers,
                                                            const int* __restrict__ wrow, const int* __restrict__ slot32,
                                                            const int* __restrict__ emap, int B) {
    extern __shared__ __align__(128) unsigned char smem[];
    using G = MaskedGeo<NK>;
    constexpr int KP = G::KP;
    const LatZh a = readers[blockIdx.z];
    const int x = blockIdx.x, b = blockIdx.y;
    const int e = emap != nullptr ? emap[b] : b;
    int wr, sl;
    lat_kind(wrow, slot32, a.t, B, b, wr, sl);
    const bool masked = wr >= 0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, nchunks = N / TN;
    const int node0 = x * TM + warp * 16 + g, node1 = node0 + 8;

    unsigned af[NK][4];
    a_frags<NK>(af, a.A + (long)e * a.sA + (long)node0 * a.r, a.r, t);
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    uint2 lt[8];
    const unsigned hoff = (((lane & 7) + ((lane >> 4) << 3)) * BROW + ((lane >> 3) & 1) * 8) * 2;

    float m0 = NEG, m1 = NEG, zr0 = 0.f, zr1 = 0.f;
    float* const rk = a.R + (long)b * nchunks * a.M + node0;    // (masked: chunk jc's reference of node0 at jc * M)
    lat_load<KP>(smem_u32(smem), a.BP, a.E16, e, a.S, N, 0, masked, sl, tid);
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + nt * 32);
    float Zc[NS8][4];
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn)
        #pragma unroll
        for (int k = 0; k < 4; ++k) Zc[sn][k] = 0.f;
    for (int jc = 0; jc < nchunks; ++jc) {
        const int st = jc & 1;
        const bool more = jc + 1 < nchunks;
        cp_wait<0>();
        __syncthreads();                     // (this chunk's stage complete; the other free: every warp past the last)
        if (more)
            lat_load<KP>(smem_u32(smem + (st ^ 1) * G::STAGE), a.BP, a.E16, e, a.S, N, (jc + 1) * TN, masked, sl, tid);
        const unsigned char* p = smem + st * G::STAGE;
        const __half* Hs = reinterpret_cast<const __half*>(p + G::B_BYTES);
        float Sx[8][4];
        logits<NK>(Sx, af, reinterpret_cast<const __half*>(p), lt, lane);
        if (more) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + ((jc + 1) * 8 + nt) * 32);
        }
        if (masked) {
            // (W at a reference raised as the chunks come -- the throughput mode's rule: below 2^(W_OFF + W_LAZY),
            // fp16's range -- the sums so far moved with it; the chunk's reference kept: the chain step's W the same)
            exp_tile(Sx, m0, m1, [&](float a0, float a1) {
                #pragma unroll
                for (int sn = 0; sn < NS8; ++sn) {
                    Zc[sn][0] *= a0; Zc[sn][1] *= a0; Zc[sn][2] *= a1; Zc[sn][3] *= a1;
                }
                zr0 *= a0; zr1 *= a1;
            }, W_OFF, W_LAZY);
            if (t == 0) {
                rk[(long)jc * a.M] = m0;
                rk[(long)jc * a.M + 8] = m1;
            }
            const __half* Rs = Hs + 8 * NS8 * BROW;
            const unsigned hb = smem_u32(Hs) + hoff;
            #pragma unroll
            for (int ks = 0; ks < TN / 16; ++ks) {
                const float (&w0)[4] = Sx[2 * ks], (&w1)[4] = Sx[2 * ks + 1];
                const unsigned az[4] = {sat_h2(w0[0], w0[1]), sat_h2(w0[2], w0[3]), sat_h2(w1[0], w1[1]),
                                        sat_h2(w1[2], w1[3])};
                #pragma unroll
                for (int q = 0; q < NS8 / 2; ++q) {
                    unsigned bf[4];
                    ldsm4(bf, hb + (q * 16 * BROW + ks * 16) * 2);
                    mma_f16(Zc[2 * q], az, bf[0], bf[1]);
                    mma_f16(Zc[2 * q + 1], az, bf[2], bf[3]);
                }
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int nt = 2 * ks + hh;
                    const float2 rr = __half22float2(*reinterpret_cast<const __half2*>(Rs + nt * 8 + 2 * t));
                    zr0 += Sx[nt][0] * rr.x + Sx[nt][1] * rr.y;
                    zr1 += Sx[nt][2] * rr.x + Sx[nt][3] * rr.y;
                }
            }
        } else {
            exp_tile(Sx, m0, m1, [&](float a0, float a1) { zr0 *= a0; zr1 *= a1; });     // (fp32: lazy 8, no offset)
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const float2 rr = __half22float2(*reinterpret_cast<const __half2*>(Hs + nt * 8 + 2 * t));
                zr0 += Sx[nt][0] * rr.x + Sx[nt][1] * rr.y;
                zr1 += Sx[nt][2] * rr.x + Sx[nt][3] * rr.y;
            }
        }
    }
    zr0 = quad_sum(zr0); zr1 = quad_sum(zr1);
    float* z0 = a.Z + ((long)b * a.M + node0) * SP;
    float* z1 = z0 + 8L * SP;
    if (masked) {
        #pragma unroll
        for (int sn = 0; sn < NS8; ++sn) {
            *reinterpret_cast<float2*>(z0 + sn * 8 + 2 * t) = make_float2(Zc[sn][0], Zc[sn][1]);
            *reinterpret_cast<float2*>(z1 + sn * 8 + 2 * t) = make_float2(Zc[sn][2], Zc[sn][3]);
        }
    }
    if (t == 0) {
        z0[masked ? 8 * NS8 : 0] = zr0; z1[masked ? 8 * NS8 : 0] = zr1;
        a.h[(long)b * a.M + node0] = m0; a.h[(long)b * a.M + node1] = m1;
    }
}

// a position's chain step: its reader's operands (`TopKStackPlan::forward`)
struct LatStep {
    const float* A; long sA; const __half* BP; const __half* E16; const uint2* LTh;
    const float* h; const float* Z; const float* R; const float* logw;
    ChildValues cv; const bool* has;   // the children's values (element values at eid, or their products': pairs)
    float* out; long nid0, ostride;    // the node values (the circuit's buffer)
    int r, M, N, S, t;
    const int* wrow; const int* slot32; const int* emap; int B;
    float* part; float* pmax; int* cnt;    // the tiles' shares [B, tiles, chunks, TM], their shifts, their counts
};

// grid (N / TN chunks, M / TM tiles, B samples)
template <int NK>
__global__ void __launch_bounds__(NW * 32, 2) tk_lat_step_kernel(const LatStep a) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ int s_last;
    using G = MaskedGeo<NK>;
    constexpr int KP = G::KP;
    const int c = blockIdx.x, x = blockIdx.y, b = blockIdx.z;
    const int nchunks = gridDim.x, tiles = gridDim.y;
    const int e = a.emap != nullptr ? a.emap[b] : b;
    int wr, sl;
    lat_kind(a.wrow, a.slot32, a.t, a.B, b, wr, sl);
    const bool masked = wr >= 0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, j0 = c * TN;
    const int node0 = x * TM + warp * 16 + g, node1 = node0 + 8;

    // ---- before the wait: what the tensors alone make (no launch of the chain writes any of it)
    lat_load<KP>(smem_u32(smem), a.BP, a.E16, e, a.S, N, j0, masked, sl, tid);
    unsigned af[NK][4];
    a_frags<NK>(af, a.A + (long)e * a.sA + (long)node0 * a.r, a.r, t);
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8 + c * 8) * 32 + lane;
    uint2 lt[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + nt * 32);
    // (the sums' reference: Z's; W's of this chunk -- masked: the chunk's own, observed: the same)
    const float hm0 = a.h[(long)b * a.M + node0], hm1 = a.h[(long)b * a.M + node1];
    float cr0 = hm0, cr1 = hm1;
    if (masked) {
        const float* rk = a.R + ((long)b * nchunks + c) * a.M + node0;
        cr0 = __ldg(rk); cr1 = __ldg(rk + 8);
    }
    cp_wait<0>();
    __syncthreads();
    float Sx[8][4];
    logits<NK>(Sx, af, reinterpret_cast<const __half*>(smem), lt, lane);
    {                                                                        // W at the reference the sums made it at
        const float off = masked ? W_OFF : 0.f;
        const float h0 = (cr0 == NEG ? 0.f : cr0) - off, h1 = (cr1 == NEG ? 0.f : cr1) - off;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            Sx[nt][0] = exp2f(Sx[nt][0] - h0); Sx[nt][1] = exp2f(Sx[nt][1] - h0);
            Sx[nt][2] = exp2f(Sx[nt][2] - h1); Sx[nt][3] = exp2f(Sx[nt][3] - h1);
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();

    // ---- after it: the chunk's children (a lane two of them, every warp all 64: their largest each warp's own)
    const int ja = j0 + 2 * lane;
    const float v0 = (a.has == nullptr || a.has[ja]) ? a.cv.at(a.cv.ref(ja), b) : NEG;
    const float v1 = (a.has == nullptr || a.has[ja + 1]) ? a.cv.at(a.cv.ref(ja + 1), b) : NEG;
    // (the slots' weights over their sums, fp32: loaded beside the children's values)
    const float* z0 = a.Z + ((long)b * a.M + node0) * SP;
    const float* z1 = z0 + 8L * SP;
    const float mc = warp_max(fmaxf(v0, v1));
    const float shv = mc == NEG ? 0.f : mc;
    const float e0 = __expf(v0 - shv), e1 = __expf(v1 - shv);
    float2 ev[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        ev[nt].x = __shfl_sync(0xffffffffu, e0, nt * 4 + t);
        ev[nt].y = __shfl_sync(0xffffffffu, e1, nt * 4 + t);
    }
    const __half* Hs = reinterpret_cast<const __half*>(smem + G::B_BYTES);
    float R0, R1;
    if (masked) {
        const float* lw0 = a.logw + ((long)wr * a.M + node0) * SP;
        const float* lw1 = lw0 + 8L * SP;
        float Uc[NS8][4];
        #pragma unroll
        for (int sn = 0; sn < NS8; ++sn)
            #pragma unroll
            for (int k = 0; k < 4; ++k) Uc[sn][k] = 0.f;
        float ur0 = 0.f, ur1 = 0.f;
        const __half* Rs = Hs + 8 * NS8 * BROW;
        const unsigned hb = smem_u32(Hs) + (((lane & 7) + ((lane >> 4) << 3)) * BROW + ((lane >> 3) & 1) * 8) * 2;
        #pragma unroll
        for (int ks = 0; ks < TN / 16; ++ks) {
            float u[2][4];
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float2 ec = ev[2 * ks + hh];
                u[hh][0] = Sx[2 * ks + hh][0] * ec.x; u[hh][1] = Sx[2 * ks + hh][1] * ec.y;
                u[hh][2] = Sx[2 * ks + hh][2] * ec.x; u[hh][3] = Sx[2 * ks + hh][3] * ec.y;
            }
            const unsigned au[4] = {sat_h2(u[0][0], u[0][1]), sat_h2(u[0][2], u[0][3]), sat_h2(u[1][0], u[1][1]),
                                    sat_h2(u[1][2], u[1][3])};
            #pragma unroll
            for (int q = 0; q < NS8 / 2; ++q) {
                unsigned bf[4];
                ldsm4(bf, hb + (q * 16 * BROW + ks * 16) * 2);
                mma_f16(Uc[2 * q], au, bf[0], bf[1]);
                mma_f16(Uc[2 * q + 1], au, bf[2], bf[3]);
            }
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int nt = 2 * ks + hh;
                const float2 rr = __half22float2(*reinterpret_cast<const __half2*>(Rs + nt * 8 + 2 * t));
                ur0 += u[hh][0] * rr.x + u[hh][1] * rr.y;
                ur1 += u[hh][2] * rr.x + u[hh][3] * rr.y;
            }
        }
        // R = sum_s exp(logw_s) U_s / Z_s over the thread's slots, then the quad's (and the rest slot's)
        float p0 = 0.f, p1 = 0.f;
        #pragma unroll
        for (int sn = 0; sn < NS8; ++sn) {
            const float2 wa = __ldg(reinterpret_cast<const float2*>(lw0 + sn * 8 + 2 * t));
            const float2 wb = __ldg(reinterpret_cast<const float2*>(lw1 + sn * 8 + 2 * t));
            const float2 za = __ldg(reinterpret_cast<const float2*>(z0 + sn * 8 + 2 * t));
            const float2 zb = __ldg(reinterpret_cast<const float2*>(z1 + sn * 8 + 2 * t));
            if (za.x > 0.f) p0 += __expf(wa.x) * (Uc[sn][0] / za.x);
            if (za.y > 0.f) p0 += __expf(wa.y) * (Uc[sn][1] / za.y);
            if (zb.x > 0.f) p1 += __expf(wb.x) * (Uc[sn][2] / zb.x);
            if (zb.y > 0.f) p1 += __expf(wb.y) * (Uc[sn][3] / zb.y);
        }
        R0 = quad_sum(p0); R1 = quad_sum(p1);
        ur0 = quad_sum(ur0); ur1 = quad_sum(ur1);
        const float zra = __ldg(z0 + 8 * NS8), zrb = __ldg(z1 + 8 * NS8);
        if (zra > 0.f) R0 += __expf(__ldg(lw0 + 8 * NS8)) * (ur0 / zra);
        if (zrb > 0.f) R1 += __expf(__ldg(lw1 + 8 * NS8)) * (ur1 / zrb);
        // (U at the chunk's reference, Z at the sums' last: the share times 2^(cr - h), at most 1)
        R0 *= cr0 == NEG ? 0.f : exp2f(cr0 - hm0);
        R1 *= cr1 == NEG ? 0.f : exp2f(cr1 - hm1);
    } else {
        float u0 = 0.f, u1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float2 rr = __half22float2(*reinterpret_cast<const __half2*>(Hs + nt * 8 + 2 * t));
            const float er0 = ev[nt].x * rr.x, er1 = ev[nt].y * rr.y;
            u0 += Sx[nt][0] * er0 + Sx[nt][1] * er1;
            u1 += Sx[nt][2] * er0 + Sx[nt][3] * er1;
        }
        u0 = quad_sum(u0); u1 = quad_sum(u1);
        const float za = __ldg(z0), zb = __ldg(z1);
        R0 = za > 0.f ? u0 / za : 0.f;
        R1 = zb > 0.f ? u1 / zb : 0.f;
    }
    // the chunk's share, then the tile's last chunk adds the 16 (in the chunks' order) into the values
    const long tb = (long)b * tiles + x;
    float* const share = a.part + tb * nchunks * TM;
    if (t == 0) {
        share[c * TM + node0 - x * TM] = R0;
        share[c * TM + node1 - x * TM] = R1;
    }
    if (tid == 0) a.pmax[tb * nchunks + c] = mc;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(a.cnt + tb, 1) == nchunks - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (tid < TM) {
        const float* pm = a.pmax + tb * nchunks;
        float M = NEG;
        for (int k = 0; k < nchunks; ++k) M = fmaxf(M, __ldcg(pm + k));
        float R = 0.f;
        if (M != NEG)
            for (int k = 0; k < nchunks; ++k) {
                const float mk = __ldcg(pm + k);
                if (mk != NEG) R += __expf(mk - M) * __ldcg(share + k * TM + tid);
            }
        a.out[(a.nid0 + (long)x * TM + tid) * a.ostride + b] = (R > 0.f && M != NEG) ? M + __logf(R) : NEG;
    }
    if (tid == 0) a.cnt[tb] = 0;                                             // (for the next position's)
}
