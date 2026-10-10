"""
The lifted sum layer. A sum node and its children (products over the same scope) keep blocks of the same
shape, so a sum acts on every column independently: it is pyjuice's sum over ``B * slots`` columns, where
``slots`` is the block size of the node group's scope. The edges are read from the sum layer's own partitions
(``partitioned_nids / cids / pids``): node ``i`` of the node block at ``nids[k]`` adds child element
``cids[k, e]`` with weight ``params[pids[k, e] + i]``.

Two paths:

* DENSE node blocks -- children a contiguous run of element rows, weights strided by the block size (an HMM's
  transitions) -- are a matrix product with weights read in place from the parameter buffer: a node block's
  weights are the ``[E, BS]`` matrix at ``params[pids[k, 0]]``. The children are shifted by their column
  maximum and exponentiated once, multiplied by cuBLAS, and the log taken. In fp32 this is cuBLAS's exact fp32
  product (measured faster than pyjuice's own sum layer at every batch size on a 4096-state HMM, with ~100x
  smaller error);
* every other node block runs the fused Triton kernel: an online log-sum-exp over chunks of children, with
  the products on tensor cores. Padded edges point at pyjuice's dummy element rows, which lie below the
  product layer group's region, and are skipped.
"""

import contextlib

import torch
import triton
import triton.language as tl

#: The fused kernel's tiles (TILE_M, TILE_E, TILE_B, num_warps, num_stages) per product precision, from a sweep on
#: a 4096-state HMM layer (4 x 1024 nodes, 4096 children) at 69 to 8832 columns.
_TILES = {"tf32": (128, 32, 128, 8, 3), "tf32x3": (64, 32, 64, 4, 3), "bf16x3": (64, 32, 64, 4, 3)}

#: Dense node blocks smaller than this run the fused kernel: one cuBLAS call per node block only pays for itself
#: on large blocks (a partition of many small blocks is one fused launch instead).
DENSE_MIN_BLOCK = 64


@triton.jit
def _lse_dot_step(m, acc, x, w, PRECISION: tl.constexpr):
    """One chunk of children of the online log-sum-exp: ``x`` [children, columns] in log space, ``w`` [nodes,
    children] the weights."""
    m_new = tl.maximum(m, tl.max(x, axis = 0))
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc * scale[None, :] + tl.dot(w, tl.exp(x - shift[None, :]), input_precision = PRECISION)


@triton.jit
def _lifted_sum_kernel(node_mars, element_mars, weights, nids, cids, pids, nb_reg,
                       reg_offset, reg_width, reg_first, reg_slots, a_rows, a_regs, a_pids,
                       elem_first, elem_width, B, num_edges, num_alias, num_mtiles,
                       BS: tl.constexpr, TILE_M: tl.constexpr, TILE_E: tl.constexpr, TILE_A: tl.constexpr,
                       TILE_B: tl.constexpr, PRECISION: tl.constexpr, ALIAS: tl.constexpr):
    pid_nm = tl.program_id(0)                                     # (node block, tile of its nodes)
    pid_c = tl.program_id(1)                                      # tile of columns
    pid_n = pid_nm // num_mtiles
    mt = pid_nm % num_mtiles
    reg = tl.load(nb_reg + pid_n)
    ncols = B * tl.load(reg_slots + reg)
    col0 = pid_c * TILE_B
    if col0 < ncols:
        cols = col0 + tl.arange(0, TILE_B)
        cmask = cols < ncols
        offs_m = mt * TILE_M + tl.arange(0, TILE_M)
        mmask = offs_m < BS
        m = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE_M, TILE_B], dtype = tl.float32)
        for e0 in range(0, num_edges, TILE_E):
            offs_e = e0 + tl.arange(0, TILE_E)
            emask = offs_e < num_edges
            cid = tl.load(cids + pid_n * num_edges + offs_e, mask = emask, other = 0).to(tl.int64)
            pid = tl.load(pids + pid_n * num_edges + offs_e, mask = emask, other = 0).to(tl.int64)
            valid = emask & (cid >= elem_first)
            x = tl.load(element_mars + (cid - elem_first)[:, None] * elem_width + cols[None, :],
                        mask = valid[:, None] & cmask[None, :], other = float("-inf"))           # [TILE_E, TILE_B]
            w = tl.load(weights + pid[None, :] + offs_m[:, None],
                        mask = mmask[:, None] & valid[None, :], other = 0.0)                    # [TILE_M, TILE_E]
            m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
        if ALIAS:
            # children that copy a sum: read in the sum's own row (row a_rows within region a_regs)
            for a0 in range(0, num_alias, TILE_A):
                offs_a = a0 + tl.arange(0, TILE_A)
                amask = offs_a < num_alias
                row = tl.load(a_rows + pid_n * num_alias + offs_a, mask = amask, other = -1)
                areg = tl.load(a_regs + pid_n * num_alias + offs_a, mask = amask, other = 0)
                pid = tl.load(a_pids + pid_n * num_alias + offs_a, mask = amask, other = 0).to(tl.int64)
                valid = amask & (row >= 0)
                base = tl.load(reg_offset + areg, mask = valid, other = 0) + \
                    row.to(tl.int64) * tl.load(reg_width + areg, mask = valid, other = 0)
                base = tl.multiple_of(base, 16)                       # regions start and rows are padded to 16 floats
                x = tl.load(node_mars + base[:, None] + cols[None, :], mask = valid[:, None] & cmask[None, :],
                            other = float("-inf"))                                             # [TILE_A, TILE_B]
                w = tl.load(weights + pid[None, :] + offs_m[:, None],
                            mask = mmask[:, None] & valid[None, :], other = 0.0)                # [TILE_M, TILE_A]
                m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
        out = tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)[None, :]
        nid = tl.load(nids + pid_n).to(tl.int64)
        width = tl.load(reg_width + reg)
        base = tl.load(reg_offset + reg) + (nid - tl.load(reg_first + reg)) * width
        tl.store(node_mars + base + offs_m.to(tl.int64)[:, None] * width + cols[None, :], out,
                 mask = mmask[:, None] & cmask[None, :])


def fused_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, nids: torch.Tensor,
              cids: torch.Tensor, pids: torch.Tensor, nb_reg: torch.Tensor, regions: dict, elem_first: int,
              elem_width: int, batch_size: int, block_size: int, max_cols: int, precision: str, aliased = None):
    """
    Node blocks of a sum layer partition with the fused Triton kernel, into ``node_mars``.

    :param nids, cids, pids: the node blocks' tables (see the module docstring)
    :param nb_reg: [num node blocks] the sum region every node block's rows belong to
    :param regions: ``offset``, ``width``, ``first`` and ``slots`` of every sum region, as device tensors
    :param elem_first, elem_width: the first row and row width of the product layer group's region
    :param max_cols: the most columns any of the node blocks has (``batch_size`` x its slots)
    :param aliased: None, or the node blocks' children that copy a sum and are read in the sum's own row:
        ``(rows, regions, pids)``, each [num node blocks, num aliased] int32 (``rows`` within the region, ``-1`` for
        padding; ``pids`` as for ``cids``)
    """
    num_blocks, num_edges = cids.shape
    tm, te, tb, warps, stages = _TILES[precision]
    TILE_M = min(tm, max(16, triton.next_power_of_2(block_size)))
    TILE_E = min(te, max(16, triton.next_power_of_2(num_edges)))
    TILE_B = min(tb, max(16, triton.next_power_of_2(max_cols)))
    num_mtiles = triton.cdiv(block_size, TILE_M)
    a_rows, a_regs, a_pids = aliased if aliased is not None else (cids, cids, cids)
    num_alias = a_rows.size(1) if aliased is not None else 0
    TILE_A = min(te, max(16, triton.next_power_of_2(num_alias)))
    grid = (num_blocks * num_mtiles, triton.cdiv(max_cols, TILE_B))
    _lifted_sum_kernel[grid](node_mars, element_mars, params, nids, cids, pids, nb_reg,
                             regions["offset"], regions["width"], regions["first"], regions["slots"], a_rows,
                             a_regs, a_pids, elem_first, elem_width, batch_size, num_edges, num_alias, num_mtiles,
                             BS = block_size, TILE_M = TILE_M, TILE_E = TILE_E, TILE_A = TILE_A, TILE_B = TILE_B,
                             PRECISION = precision, ALIAS = aliased is not None, num_warps = warps,
                             num_stages = stages)


# -------------------------------------------------------------------------------------------------
# Dense node blocks
# -------------------------------------------------------------------------------------------------

def dense_groups(nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor, block_size: int, elem_first: int):
    """
    Split a partition's node blocks into dense ones, grouped by their run of children, and the rest.

    :returns: ``(groups, rest)``: ``groups`` maps the first child row to ``[(node block index, nid, pid0)]`` for
        node blocks whose children are ``E`` contiguous element rows and whose weights are ``params[pid0 + e * BS
        + i]``; ``rest`` is a long tensor of the other node blocks' indices
    """
    E = cids.size(1)
    e = torch.arange(E, device = cids.device)
    dense = (cids == cids[:, :1] + e).all(dim = 1) & (pids == pids[:, :1] + e * block_size).all(dim = 1) & \
        (cids[:, 0] >= elem_first)
    if block_size < DENSE_MIN_BLOCK:
        dense[:] = False
    groups = {}
    for k, ok, nid, c0, p0 in zip(range(cids.size(0)), dense.tolist(), nids.tolist(), cids[:, 0].tolist(),
                                  pids[:, 0].tolist()):
        if ok:
            groups.setdefault(c0, []).append((k, nid, p0))
    rest = torch.nonzero(~dense).flatten()
    return groups, rest


@contextlib.contextmanager
def _matmul_precision(precision: str):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def dense_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, groups: dict, E: int,
              block_size: int, region: tuple, elem_first: int, elem_width: int, ncols: int, precision: str):
    """
    Dense node blocks of one sum region (see :func:`dense_groups`), into ``node_mars``: per run of children,
    ``log(W_k @ exp(X - m)) + m`` with ``m`` the column maximum of the children ``X``.

    :param region: ``(offset, width, first)`` of the node blocks' sum region (Python ints)
    """
    offset, width, first = region
    BS = block_size
    elems = element_mars[:element_mars.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_mars[offset:offset + (node_mars.numel() - offset) // width * width].view(-1, width)
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            x = elems[c0 - elem_first:c0 - elem_first + E, :ncols]
            m = x.amax(dim = 0, keepdim = True)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            ex = torch.exp(x - m)
            for _, nid, p0 in blocks:
                w = params[p0:p0 + E * BS].view(E, BS).t()                         # [BS, E], in place
                out = nodes[nid - first:nid - first + BS, :ncols]
                torch.mm(w, ex, out = out)
                out.log_().add_(m)
