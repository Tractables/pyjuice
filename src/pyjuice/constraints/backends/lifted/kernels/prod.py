"""
The lifted product layer. A product chains its children's blocks in scope order: a log-space matrix product
over the boundary columns they share. An input child at position ``t`` is never materialized as a block in
the hot path: class ``c`` of its token moves column ``i`` at boundary ``t`` to column ``next_col[t, i, c]``.

Two forms, following the plan's patterns:

* ``input_suffix`` (every product of a right-linear PC, e.g. an HMM): an input at ``a`` followed by a node
  over ``[a + 1, n - 1]``. Its block has one column per state at ``a`` (the exit is the sequence end), and
  each entry is a sum over the token classes -- a Triton kernel;
* ``chain``: any other product, chained child by child with batched matrix products. A first version, in
  PyTorch: it materializes input children's blocks and loops over groups of products that share their
  children's kinds and boundaries.
"""

import contextlib

import torch
import triton
import triton.language as tl


@triton.jit
def _input_suffix_kernel(element_mars, node_mars, input_mars, class_mars, obs_class, next_col, width,
                         out_rows, in_rows, suf_rows, starts, suf_reg, reg_offset, reg_width, reg_first,
                         elem_first, elem_width, B, n, W, C, input_start, TILE: tl.constexpr):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    a = tl.load(starts + r)
    ncols = tl.load(width + a) * B
    col0 = pid_c * TILE
    if col0 < ncols:
        cols = col0 + tl.arange(0, TILE)
        cmask = cols < ncols
        i = cols // B                                             # entry column at boundary a
        b = cols % B                                              # sample
        u = tl.load(in_rows + r).to(tl.int64)
        s = tl.load(suf_rows + r).to(tl.int64)
        sreg = tl.load(suf_reg + r)
        s_base = tl.load(reg_offset + sreg) + (s - tl.load(reg_first + sreg)) * tl.load(reg_width + sreg)
        oc = tl.load(obs_class + b * n + a, mask = cmask, other = -1)    # observed token's class, -1 if missing
        nc_base = (a * W + i) * C

        # an observed token: its own log-probability, then the suffix from the column its class leads to
        obs = cmask & (oc >= 0)
        j_obs = tl.load(next_col + nc_base + oc, mask = obs, other = -1)
        v_obs = tl.load(input_mars + u * B + b, mask = obs, other = float("-inf")) + \
            tl.load(node_mars + s_base + j_obs.to(tl.int64) * B + b, mask = obs & (j_obs >= 0), other = float("-inf"))

        # a missing token: log-sum over the classes of class mass + suffix
        miss = cmask & (oc < 0)
        m = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for c in range(C):
            j = tl.load(next_col + nc_base + c, mask = miss, other = -1)
            v = tl.load(class_mars + (u - input_start) * C + c) + \
                tl.load(node_mars + s_base + j.to(tl.int64) * B + b, mask = miss & (j >= 0), other = float("-inf"))
            m_new = tl.maximum(m, v)
            scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
            shift = tl.where(m_new == float("-inf"), 0.0, m_new)
            acc = acc * scale + tl.exp(v - shift)
            m = m_new
        v_miss = tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)

        out = (tl.load(out_rows + r).to(tl.int64) - elem_first) * elem_width
        tl.store(element_mars + out + cols, tl.where(oc >= 0, v_obs, v_miss), mask = cmask)


def input_suffix(element_mars, node_mars, input_mars, class_mars, obs_class, next_col, width, table, regions,
                 elem_first: int, elem_width: int, batch_size: int, n: int, input_start: int, max_cols: int):
    """The ``input_suffix`` products of one product layer (``table``: out_rows, in_rows, suf_rows, starts,
    suf_reg) into ``element_mars``."""
    out_rows, in_rows, suf_rows, starts, suf_reg = table
    W, C = next_col.size(1), next_col.size(2)
    TILE = min(256, max(16, triton.next_power_of_2(max_cols)))
    grid = (out_rows.numel(), triton.cdiv(max_cols, TILE))
    _input_suffix_kernel[grid](element_mars, node_mars, input_mars, class_mars, obs_class, next_col, width,
                               out_rows, in_rows, suf_rows, starts, suf_reg,
                               regions["offset"], regions["width"], regions["first"],
                               elem_first, elem_width, batch_size, n, W, C, input_start, TILE = TILE, num_warps = 4)


# -------------------------------------------------------------------------------------------------
# Chains (PyTorch, first version)
# -------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _fp32_matmuls():
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def _shift(x, dim):
    s = x.amax(dim = dim, keepdim = True)
    return torch.where(torch.isfinite(s), s, torch.zeros_like(s))


def transition_table(next_col, t: int, n: int, width) -> torch.Tensor:
    """[W_t, C, W_{t+1}] 0/1: class ``c`` moves column ``i`` at boundary ``t`` to column ``j`` (the exit at
    boundary ``n`` is one column, every active state there being accepting)."""
    wi = width[t]
    wj = 1 if t + 1 == n else width[t + 1]
    nc = next_col[t, :wi].long()                                                        # [wi, C]
    T = torch.zeros(wi, nc.size(1), wj, device = nc.device)
    ii, cc = torch.nonzero(nc >= 0, as_tuple = True)
    T[ii, cc, 0 if t + 1 == n else nc[ii, cc]] = 1.0
    return T


def _sum_block(node_mars, regions, rows, regs, t0, t1, n, width, B):
    """[R, W_t0, W_t1, B] block of sum children over boundaries (t0, t1) (a sequence end is one column)."""
    wi = width[t0]
    wj = 1 if t1 == n else width[t1]
    base = regions["offset"][regs] + (rows - regions["first"][regs]) * regions["width"][regs]       # [R]
    i = torch.arange(wi, device = rows.device)
    j = torch.arange(wj, device = rows.device)
    b = torch.arange(B, device = rows.device)
    idx = base[:, None, None, None] + ((i[:, None] * wj + j[None, :]) * B)[None, :, :, None] + b[None, None, None, :]
    return node_mars[idx]


def _input_block(input_mars, class_mars, obs_class, T, rows, t, B, input_start):
    """[R, W_t, W_{t+1}, B] block of input children at position t, from its transition table ``T``."""
    C = T.size(1)
    oc = obs_class[:, t].long()                                                          # [B], -1 if missing
    free = class_mars[rows - input_start][:, None, :]                                    # [R, 1, C]
    own = torch.where(torch.arange(C, device = oc.device)[None, :] == oc[:, None],       # observed: only its class
                      input_mars[rows][:, :, None], -float("inf"))                       # [R, B, C]
    lmass = torch.where((oc >= 0)[None, :, None], own, free)
    s = _shift(lmass, -1)
    with _fp32_matmuls():
        block = torch.einsum("rbc,icj->rijb", torch.exp(lmass - s), T)
    return torch.log(block) + s.permute(0, 2, 1)[:, :, None, :]


def _log_matmul(x, y):
    """log(exp(x) @ exp(y)) over the middle boundary, per row and sample: [R, I, M, B] x [R, M, J, B]."""
    x, y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)                                 # [R, B, I, M], [R, B, M, J]
    sx, sy = _shift(x, -1), _shift(y, -2)
    with _fp32_matmuls():
        out = torch.log(torch.exp(x - sx) @ torch.exp(y - sy)) + sx + sy
    return out.permute(0, 2, 3, 1)


def chain(element_mars, node_mars, input_mars, class_mars, obs_class, groups, transitions, regions, width,
          elem_first: int, elem_width: int, batch_size: int, n: int, input_start: int):
    """
    The ``chain`` products of one product layer into ``element_mars``.

    :param groups: one entry per set of products sharing their children's kinds and boundaries:
        ``(out_rows, [(kind, rows, regs, t0, t1), ...])`` with the children in scope order
    :param transitions: boundary ``t`` -> :func:`transition_table` (for input children)
    :param width: the number of columns per boundary, as Python ints
    """
    B = batch_size
    for out_rows, children in groups:
        block = None
        for kind, rows, regs, t0, t1 in children:
            if kind == "input":
                x = _input_block(input_mars, class_mars, obs_class, transitions[t0], rows, t0, B, input_start)
            else:
                x = _sum_block(node_mars, regions, rows, regs, t0, t1, n, width, B)
            block = x if block is None else _log_matmul(block, x)
        R, wi, wj, _ = block.shape
        base = (out_rows - elem_first) * elem_width
        cols = torch.arange(wi * wj * B, device = block.device)
        element_mars[base[:, None] + cols[None, :]] = block.reshape(R, -1)
