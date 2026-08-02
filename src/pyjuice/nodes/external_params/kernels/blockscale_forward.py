"""
Portable Triton forward for the per-block multiplicative gate (`BlockScaleSumParams`).

The counterpart of `kernels/c/blockscale_forward.cu`, and it exists for REACH rather than speed. The
two CUDA forward forks between them require a CUDA toolchain -- the CuTe/TMA one needs nvcc, CUTLASS
and sm_90+, the small-batch one needs nvcc -- so on a machine that cannot build them this
parameterization was refused outright rather than merely running slowly. Triton compiles wherever
torch does.

It is offered as a FALLBACK and not as an autotune candidate: a shape either CUDA fork serves keeps
exactly the candidates, the measurement and the pick it had before this existed, and never pays to
compile a kernel it will not use.

A second property comes free and is worth naming, because it is what the CUDA forks had to be taught:
this kernel reads `nids`, `cids` and `pids` PER LANE rather than deriving addresses from a per-tile
base, so it imposes no layout requirements whatsoever. Contiguity, `block_size` striding, padding and
block-sparsity are all simply non-issues -- a padded slot's `cids` is the dummy `-inf` element and its
`pids` the dummy zero parameter, so the lane contributes nothing without being special-cased.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def _bs_triton_fw_kernel(node_mars, element_mars, mparams, ext, gate, log_z,
                         nids, cids, pids,
                         batch_size: tl.constexpr, num_edges: tl.constexpr,
                         BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr,
                         K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                         BLOCK_SIZE_M: tl.constexpr, TL_DOT: tl.constexpr,
                         NODE_CBS: tl.constexpr, GATE_CBS: tl.constexpr,
                         gate_stride: tl.constexpr, ext_base, WRITE_LOGZ: tl.constexpr = 1,
                         pid_m_offset = 0):
    """
    PORTABLE forward for the per-block multiplicative gate: `node_mars = log N - log Z`.

    This exists for REACH, not for speed. The other two forward forks are both CUDA -- the CuTe one
    needs nvcc, CUTLASS and sm_90+, the small-batch one needs nvcc -- so on a machine without a CUDA
    toolchain `BlockScaleSumParams` was refused outright rather than merely being slow. Triton
    compiles wherever torch does, so this makes the parameterization usable everywhere and leaves the
    fast forks to be chosen on shapes where they apply.

    It also has no layout requirements. It reads `nids`, `cids` and `pids` PER LANE rather than
    deriving addresses from a per-tile base, so it needs neither contiguous children nor
    `block_size`-strided parameters, and padding is inert for the same reason it is in the plain sum
    layer: a padded slot's `cids` is the dummy element (`-inf`) and its `pids` the dummy parameter
    (zero), so the lane contributes nothing without being special-cased. Whatever the compiled
    topology is, this serves it.

    The maths is the CuTe fork's, with both accumulators carried in the same online form:

        N[m,b] = sum_e theta[e,m] * exp(element_mars[c(e),b] + log phi[g(e),b])
        Z[m,b] = sum_e theta[e,m] * exp(                       log phi[g(e),b])

    `Z` reuses the same `A` operand and drops `element_mars`, exactly as the CuTe fork does, so the
    two differ only in the exponent. Both are stabilized per BATCH COLUMN -- the running max has no
    `m` dependence, since it is a max over edges -- which is what keeps the rescale one multiply per
    column rather than one per element.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1) + pid_m_offset

    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    off_nids = tl.load(nids + nblock_id)

    # Running (max, linear sum) for both accumulators. The max is per batch column only.
    mn = tl.zeros([BLOCK_B], dtype = tl.float32) - float("inf")
    mz = tl.zeros([BLOCK_B], dtype = tl.float32) - float("inf")
    ln = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)
    lz = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        offs_edge = tl.arange(0, TILE_SIZE_K) + k * TILE_SIZE_K
        mask_edge = offs_edge < num_edges

        edge_start = tl.load(cids + nblock_id * num_edges + offs_edge,
                             mask = mask_edge, other = 0)
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge,
                            mask = mask_edge, other = 0)

        # The gate row of each edge, BOUNDED by the table's width: `num_edges` is padded up to a
        # power of two while the table is only `ext_max_n_eblks` wide, so an unbounded column can
        # read the next row's gate -- a valid `>= 0` offset that would survive the `ghas` test below.
        offs_gcol = offs_edge // NODE_CBS
        gbase = tl.load(gate + nblock_id * gate_stride + offs_gcol,
                        mask = mask_edge & (offs_gcol < gate_stride), other = -1)
        grow = gbase + ext_base + ((offs_edge % NODE_CBS) // GATE_CBS)
        ghas = gbase >= 0

        lphi = tl.load(ext + grow[:,None] * batch_size + offs_batch[None,:],
                       mask = ghas[:,None] & mask_batch[None,:], other = 0.0)
        lphi = tl.where(ghas[:,None], lphi, -float("inf"))          # [TILE_SIZE_K, BLOCK_B]

        emars = tl.load(element_mars + edge_start[:,None] * batch_size + offs_batch[None,:],
                        mask = mask_edge[:,None] & mask_batch[None,:], other = -float("inf"))
        epars = tl.load(mparams + par_start[None,:] + offs_node[:,None],
                        mask = mask_edge[None,:], other = 0.0)       # [TILE_SIZE_M, TILE_SIZE_K]

        # ---- N: the gate folds onto `element_mars`, so a padded or gateless edge is -inf and drops
        v = emars + lphi
        tmx = tl.max(v, axis = 0)                                   # [BLOCK_B]
        nmn = tl.maximum(mn, tmx)
        # Guarded for the all--inf column, where the difference would be inf - inf = NaN.
        rescale = tl.where(nmn == -float("inf"), 0.0, tl.exp(mn - nmn))
        vexp = tl.where(nmn[None,:] == -float("inf"), 0.0, tl.exp(v - nmn[None,:]))
        if TL_DOT == 1:
            ln = ln * rescale[None,:] + tl.dot(epars, vexp)
        else:
            ln = ln * rescale[None,:] + tl.sum(epars[:,:,None] * vexp[None,:,:], axis = 1)
        mn = nmn

        # ---- Z: the same contraction with `element_mars` dropped
        tmz = tl.max(lphi, axis = 0)
        nmz = tl.maximum(mz, tmz)
        rescale_z = tl.where(nmz == -float("inf"), 0.0, tl.exp(mz - nmz))
        zexp = tl.where(nmz[None,:] == -float("inf"), 0.0, tl.exp(lphi - nmz[None,:]))
        if TL_DOT == 1:
            lz = lz * rescale_z[None,:] + tl.dot(epars, zexp)
        else:
            lz = lz * rescale_z[None,:] + tl.sum(epars[:,:,None] * zexp[None,:,:], axis = 1)
        mz = nmz

    log_n = tl.where((mn[None,:] == -float("inf")) | (ln <= 0.0), -float("inf"),
                     tl.log(ln + 1e-38) + mn[None,:])
    log_zz = tl.where((mz[None,:] == -float("inf")) | (lz <= 0.0), -float("inf"),
                      tl.log(lz + 1e-38) + mz[None,:])
    out = tl.where((log_n == -float("inf")) | (log_zz == -float("inf")), -float("inf"),
                   log_n - log_zz)

    offs = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    tl.store(node_mars + offs, out, mask = mask_batch[None,:])
    if WRITE_LOGZ:
        offs_z = ((nblock_id * BLOCK_SIZE_M + offs_node[:,None]) * batch_size
                  + offs_batch[None,:])
        tl.store(log_z + offs_z, log_zz, mask = mask_batch[None,:])
