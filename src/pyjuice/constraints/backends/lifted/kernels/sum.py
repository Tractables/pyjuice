"""
The lifted sum layer. A sum node and its children (products over the same scope) keep blocks of the same
shape, so a sum acts on every column independently: it is pyjuice's sum over ``B * slots`` columns, where
``slots`` is the block size of the node group's scope. The edges are read from the sum layer's own partitions
(``partitioned_nids / cids / pids``): node ``i`` of the node block at ``nids[k]`` adds child element
``cids[k, e]`` with weight ``params[pids[k, e] + i]``.

Under evidence pruning (:mod:`.live`) only the live columns are computed, and the dead ones set to -inf. Two
paths:

* DENSE node blocks -- children a contiguous run of element rows, weights strided by the block size (an HMM's
  transitions) -- are a matrix product with weights read in place from the parameter buffer: a node block's
  weights are the ``[E, BS]`` matrix at ``params[pids[k, 0]]``. The children are shifted by their column
  maximum and exponentiated once (one pass), multiplied by cuBLAS, and the log taken and the maximum added back
  (one pass over the product, in place). In fp32 this is cuBLAS's exact fp32
  product (measured faster than pyjuice's own sum layer at every batch size on a 4096-state HMM, with ~100x
  smaller error). Node blocks that read the same children from consecutive rows can instead multiply their
  weights stacked into one ``[K * BS, E]`` matrix (:func:`stacked_weights`, a copy) in one product;
  :mod:`..forward` decides when the copy pays for itself;
* every other node block runs the fused Triton kernel: an online log-sum-exp over chunks of children, with
  the products on tensor cores. Padded edges point at pyjuice's dummy element rows, which lie below the
  product layer group's region, and are skipped.
"""

import contextlib
from typing import Optional

import torch
import triton
import triton.language as tl

from .prod import grid_chunks

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
                       reg_offset, reg_width, reg_first, reg_slots, a_rows, a_regs, a_pids, perm, reg_perm, reg_live,
                       elem_first, elem_width, B, num_edges, num_alias, num_mtiles, pid0,
                       BS: tl.constexpr, TILE_M: tl.constexpr, TILE_E: tl.constexpr, TILE_A: tl.constexpr,
                       TILE_B: tl.constexpr, PRECISION: tl.constexpr, ALIAS: tl.constexpr, PRUNE: tl.constexpr):
    """Pruned (``PRUNE``), the program's columns are positions in the region's interval's order of columns (live
    ones first, see :mod:`.live`): a tile past the live ones is all -inf, without reading anything."""
    pid_nm = tl.program_id(0)                                     # (node block, tile of its nodes)
    pid_c = pid0 + tl.program_id(1)                               # tile of columns
    pid_n = pid_nm // num_mtiles
    mt = pid_nm % num_mtiles
    reg = tl.load(nb_reg + pid_n)
    ncols = B * tl.load(reg_slots + reg)
    col0 = pid_c * TILE_B
    if col0 < ncols:
        k = col0 + tl.arange(0, TILE_B)
        kmask = k < ncols
        n_edges, n_alias = num_edges, num_alias
        if PRUNE:
            num_live = tl.load(reg_live + reg)
            cols = tl.load(perm + tl.load(reg_perm + reg) + k, mask = kmask, other = 0)
            cmask = kmask & (k < num_live)
            n_edges = tl.where(col0 < num_live, num_edges, 0)                # a dead tile reads nothing
            n_alias = tl.where(col0 < num_live, num_alias, 0)
        else:
            cols = k
            cmask = kmask
        offs_m = mt * TILE_M + tl.arange(0, TILE_M)
        mmask = offs_m < BS
        m = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE_M, TILE_B], dtype = tl.float32)
        for e0 in range(0, n_edges, TILE_E):
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
            for a0 in range(0, n_alias, TILE_A):
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
                 mask = mmask[:, None] & kmask[None, :])


def fused_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, nids: torch.Tensor,
              cids: torch.Tensor, pids: torch.Tensor, nb_reg: torch.Tensor, regions: dict, elem_first: int,
              elem_width: int, batch_size: int, block_size: int, max_cols: int, precision: str, aliased = None,
              pruning = None):
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
    :param pruning: None, or the query's :class:`~.live.Pruning`
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
    perm, reg_perm, reg_live = (pruning.perm, pruning.reg_first, pruning.reg_live) if pruning is not None else \
        (cids, cids, cids)
    for first, size in grid_chunks(triton.cdiv(max_cols, TILE_B), 1):
        _lifted_sum_kernel[(num_blocks * num_mtiles, size)](
            node_mars, element_mars, params, nids, cids, pids, nb_reg, regions["offset"], regions["width"],
            regions["first"], regions["slots"], a_rows, a_regs, a_pids, perm, reg_perm, reg_live, elem_first,
            elem_width, batch_size, num_edges, num_alias, num_mtiles, first, BS = block_size, TILE_M = TILE_M,
            TILE_E = TILE_E, TILE_A = TILE_A, TILE_B = TILE_B, PRECISION = precision, ALIAS = aliased is not None,
            PRUNE = pruning is not None, num_warps = warps, num_stages = stages)


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


#: The dense path's elementwise passes: (rows, columns) per program.
_EW_TILE = (32, 128)


@triton.jit
def _shifted_exp_kernel(x, ex, m, rows, ncols, x_ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``ex = exp(x - m)`` (``ex`` contiguous [rows, ncols]), ``m`` the column maxima of ``x``; a column of -inf
    is shifted by 0 and stays all 0."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    shift = tl.where(shift == float("-inf"), 0.0, shift)
    r = r.to(tl.int64)
    v = tl.load(x + r[:, None] * x_ld + c[None, :], mask = mask, other = float("-inf"))
    tl.store(ex + r[:, None] * ncols + c[None, :], tl.exp(v - shift[None, :]), mask = mask)


@triton.jit
def _log_add_kernel(out, m, rows, ncols, ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``out = log(out) + m`` in place, ``m`` the column maxima :func:`_shifted_exp_kernel` shifted by (a column of
    -inf has a product of 0, and log 0 - inf = -inf: no guard needed)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    ptr = out + r.to(tl.int64)[:, None] * ld + c[None, :]
    tl.store(ptr, tl.log(tl.load(ptr, mask = mask, other = 1.0)) + shift[None, :], mask = mask)


@triton.jit
def _log_scatter_kernel(out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R: tl.constexpr,
                        TILE_C: tl.constexpr):
    """``out[:, c] = log(prod[:, rank[c]]) + m[rank[c]]`` for every live column ``c`` of ``out``, -inf for the others
    (``prod`` [rows, num_live], the products of the live columns)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    k = tl.load(rank + c, mask = cmask, other = -1)
    alive = k >= 0
    shift = tl.load(m + k, mask = cmask & alive, other = 0.0)
    r = r.to(tl.int64)
    v = tl.load(prod + r[:, None] * num_live + k[None, :], mask = mask & alive[None, :], other = 1.0)
    tl.store(out + r[:, None] * ld + c[None, :], tl.where(alive[None, :], tl.log(v) + shift[None, :], float("-inf")),
             mask = mask)


def stacked_weights(params: torch.Tensor, p0s, E: int, block_size: int) -> torch.Tensor:
    """``[K * block_size, E]``: the weights of K dense node blocks over the same ``E`` children (node block ``k``'s
    ``[E, block_size]`` matrix at ``params[p0s[k]]``), transposed and stacked in the order of ``p0s`` -- a copy, as
    the blocks' matrices are not one strided view of the parameter buffer."""
    return torch.cat([params[p0:p0 + E * block_size].view(E, block_size).t() for p0 in p0s], dim = 0)


def _products(blocks, params, E, BS, stacked):
    """``(weights [rows, E], first node row)`` of every matrix product of a run of node blocks: one with their
    ``stacked`` weights (the blocks then occupy consecutive rows), else one per block, read in place."""
    if stacked is not None:
        return [(stacked, blocks[0][1])]
    return [(params[p0:p0 + E * BS].view(E, BS).t(), nid) for _, nid, p0 in blocks]


def dense_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, groups: dict, E: int,
              block_size: int, region: tuple, elem_first: int, elem_width: int, ncols: int, precision: str,
              live = None, stacked: dict = None):
    """
    Dense node blocks of one sum region (see :func:`dense_groups`), into ``node_mars``: per run of children,
    ``log(W_k @ exp(X - m)) + m`` with ``m`` the column maximum of the children ``X``.

    :param region: ``(offset, width, first)`` of the node blocks' sum region (Python ints)
    :param live: None (every column), or the region's interval's ``(perm, rank, num_live)`` (see
        :meth:`~.live.Pruning.segment`): the live columns are gathered and multiplied, the others set to -inf
    :param stacked: first child row -> the :func:`stacked_weights` of that run's node blocks, which occupy
        consecutive rows in the order listed: one product for the run instead of one per block
    """
    offset, width, first = region
    BS = block_size
    stacked = stacked or {}
    elems = element_mars[:element_mars.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_mars[offset:offset + (node_mars.numel() - offset) // width * width].view(-1, width)
    TILE_R, TILE_C = _EW_TILE
    num_ctiles = triton.cdiv(ncols, TILE_C)
    if live is not None:
        _dense_sum_live(nodes, elems, params, groups, E, BS, first, elem_first, ncols, precision, live, stacked)
        return
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            x = elems[c0 - elem_first:c0 - elem_first + E, :ncols]
            m = x.amax(dim = 0)
            ex = torch.empty(E, ncols, dtype = x.dtype, device = x.device)
            _shifted_exp_kernel[(triton.cdiv(E, TILE_R) * num_ctiles,)](x, ex, m, E, ncols, elem_width, num_ctiles,
                                                                       TILE_R = TILE_R, TILE_C = TILE_C)
            for w, nid in _products(blocks, params, E, BS, stacked.get(c0)):
                rows = w.size(0)
                out = nodes[nid - first:nid - first + rows, :ncols]
                torch.mm(w, ex, out = out)
                _log_add_kernel[(triton.cdiv(rows, TILE_R) * num_ctiles,)](out, m, rows, ncols, width, num_ctiles,
                                                                          TILE_R = TILE_R, TILE_C = TILE_C)


def _dense_sum_live(nodes, elems, params, groups, E, BS, first, elem_first, ncols, precision, live, stacked):
    perm, rank, num_live = live
    width = nodes.size(1)
    TILE_R, TILE_C = _EW_TILE
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            x = torch.index_select(elems[c0 - elem_first:c0 - elem_first + E, :ncols], 1, perm[:num_live])  # [E, live]
            m = x.amax(dim = 0)
            live_tiles = triton.cdiv(num_live, TILE_C)
            if num_live > 0:
                _shifted_exp_kernel[(triton.cdiv(E, TILE_R) * live_tiles,)](x, x, m, E, num_live, num_live, live_tiles,
                                                                           TILE_R = TILE_R, TILE_C = TILE_C)
            for w, nid in _products(blocks, params, E, BS, stacked.get(c0)):
                rows = w.size(0)
                prod = torch.mm(w, x)                                              # [rows, live]
                num_ctiles = triton.cdiv(ncols, TILE_C)
                _log_scatter_kernel[(triton.cdiv(rows, TILE_R) * num_ctiles,)](
                    nodes[nid - first:nid - first + rows], prod, m, rank, rows, ncols, width, num_live, num_ctiles,
                    TILE_R = TILE_R, TILE_C = TILE_C)


# -------------------------------------------------------------------------------------------------
# The top-down pass: a sum's children from the sum
# -------------------------------------------------------------------------------------------------

@triton.jit
def _logaddexp_tile(a, b):
    hi = tl.maximum(a, b)
    s = tl.where(hi == float("-inf"), 0.0, hi)
    return tl.where(hi == float("-inf"), float("-inf"), tl.log(tl.exp(a - s) + tl.exp(b - s)) + s)


@triton.jit
def _lifted_sum_top_down_kernel(element_flows, node_flows, weights, cb_row, cb_reg, cb_add, cb_ivreg, cb_idx, par_ptr,
                               par_nid, par_reg, par_pid, reg_offset, reg_width, reg_first, reg_slots, perm, reg_perm,
                               reg_live, elem_first, elem_width, B, pid0, BS: tl.constexpr, CBS: tl.constexpr,
                               TILE_C: tl.constexpr, TILE_I: tl.constexpr, TILE_B: tl.constexpr,
                               PRECISION: tl.constexpr, PRUNE: tl.constexpr, INDEXED: tl.constexpr):
    """One program per (child block, tile of its children, tile of columns): ``O_child[c, col] = log sum over the
    block's parent node blocks and their nodes i of w[c, i] exp(O_node[i, col])``, an online log-sum-exp over tiles
    of parent nodes with the products on tensor cores -- the forward's fused kernel with children and nodes
    exchanged (``w[c, i] = weights[pid + c * BS + i]``). A child block is element rows (``cb_reg`` -1), written to
    ``element_flows``, or a sum node block's rows (an aliased copy), written to ``node_flows``; written, or log-added
    to what is there (``cb_add``). Pruned, the columns are the interval's live ones first, as in the forward.
    ``INDEXED``: the child blocks are those of the plan's index list ``cb_idx``."""
    pid_cm = tl.program_id(0)
    pid_c = pid0 + tl.program_id(1)
    cb = pid_cm // ((CBS + TILE_C - 1) // TILE_C)
    ct = pid_cm % ((CBS + TILE_C - 1) // TILE_C)
    if INDEXED:
        cb = tl.load(cb_idx + cb)
    ivreg = tl.load(cb_ivreg + cb)
    ncols = B * tl.load(reg_slots + ivreg)
    col0 = pid_c * TILE_B
    if col0 < ncols:
        k = col0 + tl.arange(0, TILE_B)
        kmask = k < ncols
        if PRUNE:
            num_live = tl.load(reg_live + ivreg)
            cols = tl.load(perm + tl.load(reg_perm + ivreg) + k, mask = kmask, other = 0)
            cmask = kmask & (k < num_live)
            skip = col0 >= num_live                                           # a dead tile reads nothing
        else:
            cols = k
            cmask = kmask
            skip = False
        offs_c = ct * TILE_C + tl.arange(0, TILE_C)
        cm = offs_c < CBS
        m = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE_C, TILE_B], dtype = tl.float32)
        p0 = tl.load(par_ptr + cb)
        p1 = tl.load(par_ptr + cb + 1)
        if not skip:
            for p in range(p0, p1):
                nid = tl.load(par_nid + p).to(tl.int64)
                reg = tl.load(par_reg + p)
                pid = tl.load(par_pid + p).to(tl.int64)
                base = tl.load(reg_offset + reg) + (nid - tl.load(reg_first + reg)) * tl.load(reg_width + reg)
                width = tl.load(reg_width + reg)
                for i0 in range(0, BS, TILE_I):
                    offs_i = i0 + tl.arange(0, TILE_I)
                    im = offs_i < BS
                    w = tl.load(weights + pid + offs_c[:, None] * BS + offs_i[None, :],
                                mask = cm[:, None] & im[None, :], other = 0.0)                # [TILE_C, TILE_I]
                    x = tl.load(node_flows + base + offs_i.to(tl.int64)[:, None] * width + cols[None, :],
                                mask = im[:, None] & cmask[None, :], other = float("-inf"))     # [TILE_I, TILE_B]
                    m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
        out = tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)[None, :]
        row = tl.load(cb_row + cb).to(tl.int64)
        creg = tl.load(cb_reg + cb)
        add = tl.load(cb_add + cb)
        if creg < 0:                                                          # element rows
            ptr = element_flows + (row - elem_first + offs_c.to(tl.int64))[:, None] * elem_width + cols[None, :]
        else:                                                                 # an aliased sum's rows
            crow = row - tl.load(reg_first + creg) + offs_c.to(tl.int64)
            ptr = node_flows + tl.load(reg_offset + creg) + crow[:, None] * tl.load(reg_width + creg) + cols[None, :]
        mask = cm[:, None] & kmask[None, :]
        if add:
            out = _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), out)
        tl.store(ptr, out, mask = mask)


def fused_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor, tables: dict,
                      regions: dict, elem_first: int, elem_width: int, batch_size: int, block_size: int, cbs: int,
                      max_cols: int, precision: str, pruning = None, idx: Optional[torch.Tensor] = None):
    """
    The top-down values of the children of a sum layer's fused node blocks (:func:`_lifted_sum_top_down_kernel`).

    :param tables: child-block-major: ``cb_row``, ``cb_reg`` (-1 for element rows), ``cb_add``, ``cb_ivreg`` (a region
        of the children's interval, for pruning), ``par_ptr``, ``par_nid``, ``par_reg``, ``par_pid`` (int32 / int64
        device tensors, see :mod:`..backward`)
    :param cbs: the child block size of these children
    :param max_cols: the most columns any child block has (``batch_size`` x its slots)
    :param idx: None (every child block), or int32 on the device: the plan's index list of the child blocks to compute
    """
    num_blocks = tables["cb_row"].numel() if idx is None else idx.numel()
    tm, te, tb, warps, stages = _TILES[precision]
    TILE_C = min(tm, max(16, triton.next_power_of_2(cbs)))
    TILE_I = min(te, max(16, triton.next_power_of_2(block_size)))
    TILE_B = min(tb, max(16, triton.next_power_of_2(max_cols)))
    num_ctiles = triton.cdiv(cbs, TILE_C)
    perm, reg_perm, reg_live = (pruning.perm, pruning.reg_first, pruning.reg_live) if pruning is not None else \
        (tables["cb_row"], tables["cb_row"], tables["cb_row"])
    for first, size in grid_chunks(triton.cdiv(max_cols, TILE_B), 1):
        _lifted_sum_top_down_kernel[(num_blocks * num_ctiles, size)](
            element_flows, node_flows, params, tables["cb_row"], tables["cb_reg"], tables["cb_add"], tables["cb_ivreg"],
            tables["cb_row"] if idx is None else idx, tables["par_ptr"], tables["par_nid"], tables["par_reg"],
            tables["par_pid"], regions["offset"], regions["width"], regions["first"], regions["slots"], perm, reg_perm,
            reg_live, elem_first, elem_width, batch_size, first, BS = block_size, CBS = cbs, TILE_C = TILE_C,
            TILE_I = TILE_I, TILE_B = TILE_B, PRECISION = precision, PRUNE = pruning is not None,
            INDEXED = idx is not None, num_warps = warps, num_stages = stages)


@triton.jit
def _log_accumulate_kernel(out, prod, m, rows, ncols, ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``out = logaddexp(out, log(prod) + m)`` (``prod`` contiguous [rows, ncols])."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    r = r.to(tl.int64)
    v = tl.log(tl.load(prod + r[:, None] * ncols + c[None, :], mask = mask, other = 0.0)) + shift[None, :]
    ptr = out + r[:, None] * ld + c[None, :]
    tl.store(ptr, _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), v), mask = mask)


@triton.jit
def _log_scatter_accumulate_kernel(out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R: tl.constexpr,
                                   TILE_C: tl.constexpr):
    """``out[:, c] = logaddexp(out[:, c], log(prod[:, rank[c]]) + m[rank[c]])`` for every live column ``c``; dead
    columns untouched (they hold -inf from the first writer)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    k = tl.load(rank + c, mask = cmask, other = -1)
    alive = k >= 0
    mask = (r < rows)[:, None] & (cmask & alive)[None, :]
    shift = tl.load(m + k, mask = cmask & alive, other = 0.0)
    r = r.to(tl.int64)
    v = tl.log(tl.load(prod + r[:, None] * num_live + k[None, :], mask = mask, other = 0.0)) + shift[None, :]
    ptr = out + r[:, None] * ld + c[None, :]
    tl.store(ptr, _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), v), mask = mask)


def dense_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor, blocks, E: int,
                      block_size: int, region: tuple, c0: int, elem_first: int, elem_width: int, ncols: int,
                      precision: str, add: bool, live = None, stacked = None, piece = None):
    """
    The top-down values of the ``E`` children ``c0 ..`` of a run of dense node blocks of one sum region:
    ``O_elem = log(W^T @ exp(O_node - m)) + m`` with ``m`` the column maximum of the node blocks' top-down values,
    written to ``element_flows``, or log-added to it (``add``); ``W^T`` is the parameter buffer's own ``[E, BS]``
    view per block (``stacked``: the blocks' :func:`stacked_weights`, transposed for free).

    :param blocks: ``[(k, nid, p0)]`` as in :func:`dense_sum`'s groups (consecutive rows when ``stacked``)
    :param region: ``(offset, width, first)`` of the node blocks' sum region
    :param live: None, or the interval's ``(perm, rank, num_live)`` (see :meth:`~.live.Pruning.segment`)
    :param piece: None, or ``(lo, hi)``: only the children ``c0 + lo .. c0 + hi``
    """
    offset, width, first = region
    BS = block_size
    lo, hi = piece if piece is not None else (0, E)
    elems = element_flows[:element_flows.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_flows[offset:offset + (node_flows.numel() - offset) // width * width].view(-1, width)
    out = elems[c0 + lo - elem_first:c0 + hi - elem_first]
    E = hi - lo
    TILE_R, TILE_C = _EW_TILE
    products = [(stacked.t()[lo:hi], blocks[0][1], stacked.size(0))] if stacked is not None else \
        [(params[p0 + lo * BS:p0 + hi * BS].view(E, BS), nid, BS) for _, nid, p0 in blocks]
    with _matmul_precision(precision):
        for wt, nid, rows in products:
            x = nodes[nid - first:nid - first + rows, :ncols]
            if live is None:
                m = x.amax(dim = 0)
                ex = torch.empty(rows, ncols, dtype = x.dtype, device = x.device)
                num_ctiles = triton.cdiv(ncols, TILE_C)
                _shifted_exp_kernel[(triton.cdiv(rows, TILE_R) * num_ctiles,)](x, ex, m, rows, ncols, width, num_ctiles,
                                                                              TILE_R = TILE_R, TILE_C = TILE_C)
                if add:
                    prod = torch.mm(wt, ex)
                    _log_accumulate_kernel[(triton.cdiv(E, TILE_R) * num_ctiles,)](
                        out, prod, m, E, ncols, elem_width, num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C)
                else:
                    torch.mm(wt, ex, out = out[:, :ncols])
                    _log_add_kernel[(triton.cdiv(E, TILE_R) * num_ctiles,)](out, m, E, ncols, elem_width, num_ctiles,
                                                                           TILE_R = TILE_R, TILE_C = TILE_C)
            else:
                perm, rank, num_live = live
                xl = torch.index_select(x, 1, perm[:num_live])                                  # [rows, live]
                m = xl.amax(dim = 0)
                live_tiles = triton.cdiv(num_live, TILE_C)
                if num_live > 0:
                    _shifted_exp_kernel[(triton.cdiv(rows, TILE_R) * live_tiles,)](
                        xl, xl, m, rows, num_live, num_live, live_tiles, TILE_R = TILE_R, TILE_C = TILE_C)
                prod = torch.mm(wt, xl)                                                         # [E, live]
                num_ctiles = triton.cdiv(ncols, TILE_C)
                kernel = _log_scatter_accumulate_kernel if add else _log_scatter_kernel
                kernel[(triton.cdiv(E, TILE_R) * num_ctiles,)](out, prod, m, rank, E, ncols, elem_width, num_live,
                                                              num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C)
            add = True                                                        # the run's later blocks add to the first


@triton.jit
def _fill_dead_rows_kernel(node_flows, rows, reg_offset, reg_width, reg_first, num_tiles, TILE: tl.constexpr):
    """Sum rows of ``node_flows`` set to -inf over their whole width: one program per (``(row, region)`` of ``rows``,
    tile of ``TILE`` columns)."""
    k = tl.program_id(0) // num_tiles
    row = tl.load(rows + k * 2)
    reg = tl.load(rows + k * 2 + 1)
    width = tl.load(reg_width + reg)
    base = node_flows + tl.load(reg_offset + reg) + (row - tl.load(reg_first + reg)).to(tl.int64) * width
    c = (tl.program_id(0) % num_tiles) * TILE + tl.arange(0, TILE)
    tl.store(base + c, tl.full([TILE], float("-inf"), dtype = tl.float32), mask = c < width)


def fill_dead_rows(node_flows: torch.Tensor, rows: torch.Tensor, regions: dict, max_width: int):
    """
    The top-down value -inf on the sum rows no chain of edges from the root reaches (no launch writes them).

    :param rows: int32 [D, 2], ``(row, region)`` per dead row
    :param regions: ``offset``, ``width`` and ``first`` of every sum region (device tensors)
    :param max_width: the widest of their regions' rows
    """
    TILE = 1024
    num_tiles = triton.cdiv(max_width, TILE)
    _fill_dead_rows_kernel[(rows.size(0) * num_tiles,)](node_flows, rows, regions["offset"], regions["width"],
                                                        regions["first"], num_tiles, TILE = TILE)
