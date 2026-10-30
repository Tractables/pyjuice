"""
Mini-batch EM's sum-parameter update by node blocks, in the parameters' own layout.

A node block's parameters (and their flows) are one contiguous ``[children, nodes]`` array -- nodes innermost --, and
`par_update.em_par_update_kernel` walks them per (node, run of 32 children): it reads and WRITES 32 values
``block_size`` floats apart, one per 32-byte sector, through offsets it loads from tables, so nothing coalesces. The
reads survive it (`cum_pflow_kernel`, which only reads, runs at ~1 TB/s); the writes do not -- each partial sector costs
the DRAM a read and a write, and the update ran 6.3x above its bandwidth floor (MEASURED, HCLT-1024: 24.5 ms
against 3.9).

Here a program takes a tile of consecutive nodes of a node block, rows of consecutive values that coalesce, and computes
every node's total (`cum_pflow_kernel`'s sum, without its atomics) and then every parameter's update, the expression of
`em_par_update_kernel` element for element:

* `_em_node_block_one_pass_kernel` for tiles of at most `ONE_PASS_MAX` children: the tile read once and kept while its
  totals are summed -- the update at its floor (MEASURED, HCLT-1024, 1024 children: fp32 3.95 ms against 3.88; bf16
  with the master copy 5.39 against 5.2);
* `_em_block_kernel` for a node block of 32-64 nodes and at most `ONE_PASS_MAX` children whose parameters have a master
  copy: a program per whole block, its contiguous ``[children, nodes]`` slab walked twice in linear steps of
  `BLOCK_STEP` children -- the totals, then the update, the second walk read back from the L2 (32 warps a program: at
  most two programs an SM, the slabs in flight well inside the L2). The one-pass kernel's master-copy tile is 8 nodes
  (a 16-byte row of bf16 values; wider tiles spill), 86% of the bytes' floor on HCLT-1024 (block 32: 1911 us against
  a copy of the bytes 1644; block 64: 1888), where this form is at it (1642 / 1711; MEASURED, toast; fp32's one-pass
  tile is 16 nodes and at the floor already, 1259 / 1263 against 1234, so fp32 keeps it);
* the split kernels for any number of children, a program per (node tile, `SPLIT_CHUNK` children): the chunks'
  partial totals (`_em_split_partials_kernel`), summed in chunk order (`_em_split_merge_kernel`), then the update
  (`_em_split_update_kernel`) -- the flows read twice. A two-pass kernel whose programs were the node tiles alone, each
  walking every child, ran 64 programs on HMM-4096 (block 1024, 4096 children): 2.4-2.8x above the bytes' floor
  (MEASURED, toast, mini_batch_em: bf16 with the master copy 675 us, fp32 419, against a copy of the bytes 243 / 194);
  split by chunks of 256, 293 / 155 (chunks of 128-1024 and node tiles of 32 within 10%). HCLT-1024 (1024 children,
  the one-pass form): fp32 5.06 ms, bf16 6.41 forced to the split form.

Which one, and the one-pass tile, is a fixed rule (`_form`), not a timed choice: the forms sum a node's flows in
different orders (~1e-10 apart, thousands of a layer's values differing in the last bits), and the update must come out
to the same bits wherever it runs on the same values -- DDP's ranks each update from the same reduced flows, and with
the form timed per process two of them could pick differently and drift apart (tests/optim/bf16_params_ddp_test.py, on
busy GPUs). A node's total is summed in one order (the atomics added in none): the parameters agree with the per-block
kernels' to the last bits of fp32, and with themselves run to run and rank to rank to the bit.

The same kernels serve the fp32 master copy of parameters stored below fp32 (`params_master`): the old value read
through `params_master._em_par_update_master_kernel`'s external-write check, both copies written.

Node blocks of fewer than `MIN_BLOCK_SIZE` nodes (a node's children a few floats apart, the per-block kernels' reads and
writes then sharing sectors) keep `par_update.em_par_update` / `params_master.em_par_update_master`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl


#: Node blocks of at least this many nodes take the node-block kernel; smaller ones the per-block kernels
MIN_BLOCK_SIZE = 16

#: The split kernels' tile: nodes a program takes (a power of two), children per step, children a program (a power of
#: two; the chunks' partial totals summed in chunk order) -- a node block's programs = node tiles x chunks (see the
#: module docstring)
BLOCK_N = 64
BLOCK_C = 32
SPLIT_CHUNK = 256

#: The whole-block kernel's children a step and warps (see the module docstring), and the blocks it takes: nodes within
#: ``BLOCK_FORM_NODES`` (a slab of at most 64 x 1024 floats, 256 KB, so that two programs an SM on 188 SMs hold under
#: 100 MB of the L2 between the walks)
BLOCK_STEP = 128
BLOCK_WARPS = 32
BLOCK_FORM_NODES = (32, 64)

#: The most children the one-pass kernel holds (a tile's whole column of flows in registers)
ONE_PASS_MAX = 1024

#: The one-pass kernel's tile (nodes, warps) without and with a master copy: MEASURED on HCLT-1024 (1024 children) among
#: (16, 8), (8, 16), (8, 8), (16, 16), (4, 8) -- fp32 16 nodes on 8 warps, bf16 with the master copy 8 on 16 (the master
#: copy's second read and write); the others spilled (40+ ms against 3.95)
ONE_PASS_TILE = (16, 8)
ONE_PASS_TILE_MASTER = (8, 16)


@triton.jit
def _em_node_block_one_pass_kernel(stored_ptr, master, param_flows, cum_pflows, t_par, t_pflow, t_rows, t_stride,
                                   t_nodes, t_gnid, step_size, pseudocount, HAS_MASTER: tl.constexpr,
                                   KEEP_ZERO: tl.constexpr, NCH: tl.constexpr, BLOCK_N: tl.constexpr):
    """Program t: the tile's ``t_nodes[t]`` nodes, from parameter ``t_par[t]`` / flow ``t_pflow[t]``, ``t_rows[t]``
    children (at most ``NCH``, a power of two), rows ``t_stride[t]`` (the node block's size) apart; their totals stored
    at ``cum_pflows[t_gnid[t]:]`` (the per-block kernels' buffer, as they leave it). The tile's flows read once, kept
    while their totals are summed. ``stored_ptr``: the parameters as stored (an argument named ``params`` breaks Triton
    3.7's binder)."""
    t = tl.program_id(0)
    par0 = tl.load(t_par + t).to(tl.int64)
    pf0 = tl.load(t_pflow + t).to(tl.int64)
    nch = tl.load(t_rows + t)
    nch32 = nch.to(tl.int32)
    stride = tl.load(t_stride + t).to(tl.int64)
    nv = tl.load(t_nodes + t).to(tl.int32)
    gnid = tl.load(t_gnid + t).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, NCH)
    mask = (offs_c < nch32)[:, None] & (offs_n < nv)[None, :]
    rel = offs_c.to(tl.int64)[:, None] * stride + offs_n[None, :]
    pflows = tl.load(param_flows + pf0 + rel, mask = mask, other = 0)
    if HAS_MASTER:
        stored = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0)
        mval = tl.load(master + par0 + rel, mask = mask, other = 0)
        old = tl.where(stored != mval.to(stored.dtype), stored.to(tl.float32), mval)
    else:
        old = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0).to(tl.float32)
    if KEEP_ZERO:
        shifted = pflows + pseudocount / nch
        tot = tl.sum(tl.where(mask & ~(old < 1e-12), shifted, 0.0), axis = 0)
        tl.store(cum_pflows + gnid + offs_n, tot, mask = offs_n < nv)
        new_param = shifted / tot[None, :]
        upd = tl.where(old < 1e-12, 0.0, (1.0 - step_size) * old + step_size * new_param)
    else:
        tot = tl.sum(pflows, axis = 0)
        tl.store(cum_pflows + gnid + offs_n, tot, mask = offs_n < nv)
        new_param = (pflows + pseudocount / nch) / (tot[None, :] + pseudocount)
        upd = (1.0 - step_size) * old + step_size * new_param
    if HAS_MASTER:
        tl.store(master + par0 + rel, upd, mask = mask)
    tl.store(stored_ptr + par0 + rel, upd.to(stored_ptr.dtype.element_ty), mask = mask)


@triton.jit
def _em_block_kernel(stored_ptr, master, param_flows, cum_pflows, t_par, t_pflow, t_rows, t_stride, t_nodes,
                          t_gnid, step_size, pseudocount, HAS_MASTER: tl.constexpr, KEEP_ZERO: tl.constexpr,
                          BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr):
    """Program t: a whole node block (the one-pass kernel's tables, a tile the block), its flows walked twice,
    ``BLOCK_C`` children a step -- the totals, then the update; the block's ``[children, nodes]`` slab is contiguous, so
    a step is one linear stream, and the second walk reads it back from the L2 (see `_form`)."""
    t = tl.program_id(0)
    par0 = tl.load(t_par + t).to(tl.int64)
    pf0 = tl.load(t_pflow + t).to(tl.int64)
    nch = tl.load(t_rows + t)                                          # the nodes' children (int64, as `nchs`)
    nch32 = nch.to(tl.int32)
    stride = tl.load(t_stride + t).to(tl.int64)
    nv = tl.load(t_nodes + t).to(tl.int32)
    gnid = tl.load(t_gnid + t).to(tl.int64)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < nv
    offs_c0 = tl.arange(0, BLOCK_C)

    # The nodes' totals: `cum_pflow_kernel`'s sum (with `keep_zero_params`, over the children not at zero, each flow
    # plus its share of the pseudocount)
    tot = tl.zeros([BLOCK_N], dtype = tl.float32)
    for c0 in range(0, nch32, BLOCK_C):
        offs_c = c0 + offs_c0
        mask = (offs_c < nch32)[:, None] & mask_n[None, :]
        rel = offs_c.to(tl.int64)[:, None] * stride + offs_n[None, :]
        pflows = tl.load(param_flows + pf0 + rel, mask = mask, other = 0)
        if KEEP_ZERO:
            if HAS_MASTER:
                stored = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0)
                mval = tl.load(master + par0 + rel, mask = mask, other = 0)
                old = tl.where(stored != mval.to(stored.dtype), stored.to(tl.float32), mval)
            else:
                old = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0).to(tl.float32)
            pflows += pseudocount / nch
            tot += tl.sum(tl.where(mask & ~(old < 1e-12), pflows, 0.0), axis = 0)      # (the original's test)
        else:
            tot += tl.sum(pflows, axis = 0)
    tl.store(cum_pflows + gnid + offs_n, tot, mask = mask_n)

    # The update, as `em_par_update_kernel` computes it
    for c0 in range(0, nch32, BLOCK_C):
        offs_c = c0 + offs_c0
        mask = (offs_c < nch32)[:, None] & mask_n[None, :]
        rel = offs_c.to(tl.int64)[:, None] * stride + offs_n[None, :]
        pflows = tl.load(param_flows + pf0 + rel, mask = mask, other = 0)
        if KEEP_ZERO:
            new_param = (pflows + pseudocount / nch) / tot[None, :]
        else:
            new_param = (pflows + pseudocount / nch) / (tot[None, :] + pseudocount)
        if HAS_MASTER:
            stored = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0)
            mval = tl.load(master + par0 + rel, mask = mask, other = 0)
            old = tl.where(stored != mval.to(stored.dtype), stored.to(tl.float32), mval)
        else:
            old = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0).to(tl.float32)
        if KEEP_ZERO:
            upd = tl.where(old < 1e-12, 0.0, (1.0 - step_size) * old + step_size * new_param)
        else:
            upd = (1.0 - step_size) * old + step_size * new_param
        if HAS_MASTER:
            tl.store(master + par0 + rel, upd, mask = mask)
        tl.store(stored_ptr + par0 + rel, upd.to(stored_ptr.dtype.element_ty), mask = mask)


@triton.jit
def _em_split_partials_kernel(stored_ptr, master, param_flows, partials, t_par, t_pflow, t_rows, t_stride, t_nodes,
                              t_gnid, G, pseudocount, HAS_MASTER: tl.constexpr, KEEP_ZERO: tl.constexpr,
                              CHUNK: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr):
    """Program (t, c): the partial totals of tile ``t``'s nodes (the one-pass kernel's tables) over its children
    ``[c * CHUNK, (c + 1) * CHUNK)`` -- `cum_pflow_kernel`'s sum over them (with ``keep_zero_params``, over the children
    not at zero, each flow plus its share of the pseudocount) -- at ``partials[c * G + gnid + node]`` (``G`` the
    circuit's nodes); `_em_split_merge_kernel` sums the chunks."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    nch = tl.load(t_rows + t)
    nch32 = nch.to(tl.int32)
    if c * CHUNK < nch32:
        par0 = tl.load(t_par + t).to(tl.int64)
        pf0 = tl.load(t_pflow + t).to(tl.int64)
        stride = tl.load(t_stride + t).to(tl.int64)
        nv = tl.load(t_nodes + t).to(tl.int32)
        gnid = tl.load(t_gnid + t).to(tl.int64)
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < nv
        offs_c0 = c * CHUNK + tl.arange(0, BLOCK_C)
        tot = tl.zeros([BLOCK_N], dtype = tl.float32)
        for c0 in range(0, CHUNK, BLOCK_C):
            offs_c = c0 + offs_c0
            mask = (offs_c < nch32)[:, None] & mask_n[None, :]
            rel = offs_c.to(tl.int64)[:, None] * stride + offs_n[None, :]
            pflows = tl.load(param_flows + pf0 + rel, mask = mask, other = 0)
            if KEEP_ZERO:
                if HAS_MASTER:
                    stored = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0)
                    mval = tl.load(master + par0 + rel, mask = mask, other = 0)
                    old = tl.where(stored != mval.to(stored.dtype), stored.to(tl.float32), mval)
                else:
                    old = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0).to(tl.float32)
                pflows += pseudocount / nch
                tot += tl.sum(tl.where(mask & ~(old < 1e-12), pflows, 0.0), axis = 0)
            else:
                tot += tl.sum(pflows, axis = 0)
        tl.store(partials + c.to(tl.int64) * G + gnid + offs_n, tot, mask = mask_n)


@triton.jit
def _em_split_merge_kernel(partials, cum_pflows, t_rows, t_nodes, t_gnid, G, CHUNK: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    """Program t: tile ``t``'s nodes' totals, their chunks' partials summed in chunk order (one order: the same bits
    on every rank), at ``cum_pflows[gnid:]`` (the per-block kernels' buffer, as they leave it)."""
    t = tl.program_id(0)
    nch32 = tl.load(t_rows + t).to(tl.int32)
    nv = tl.load(t_nodes + t).to(tl.int32)
    gnid = tl.load(t_gnid + t).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < nv
    tot = tl.zeros([BLOCK_N], dtype = tl.float32)
    for c in range(0, tl.cdiv(nch32, CHUNK)):
        tot += tl.load(partials + c.to(tl.int64) * G + gnid + offs_n, mask = mask_n, other = 0)
    tl.store(cum_pflows + gnid + offs_n, tot, mask = mask_n)


@triton.jit
def _em_split_update_kernel(stored_ptr, master, param_flows, cum_pflows, t_par, t_pflow, t_rows, t_stride, t_nodes,
                            t_gnid, step_size, pseudocount, HAS_MASTER: tl.constexpr, KEEP_ZERO: tl.constexpr,
                            CHUNK: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr):
    """Program (t, c): the update of tile ``t``'s parameters over its children ``[c * CHUNK, (c + 1) * CHUNK)`` from the
    totals at ``cum_pflows``, as `em_par_update_kernel` computes it."""
    t = tl.program_id(0)
    c = tl.program_id(1)
    nch = tl.load(t_rows + t)
    nch32 = nch.to(tl.int32)
    if c * CHUNK < nch32:
        par0 = tl.load(t_par + t).to(tl.int64)
        pf0 = tl.load(t_pflow + t).to(tl.int64)
        stride = tl.load(t_stride + t).to(tl.int64)
        nv = tl.load(t_nodes + t).to(tl.int32)
        gnid = tl.load(t_gnid + t).to(tl.int64)
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < nv
        tot = tl.load(cum_pflows + gnid + offs_n, mask = mask_n, other = 1.0)
        offs_c0 = c * CHUNK + tl.arange(0, BLOCK_C)
        for c0 in range(0, CHUNK, BLOCK_C):
            offs_c = c0 + offs_c0
            mask = (offs_c < nch32)[:, None] & mask_n[None, :]
            rel = offs_c.to(tl.int64)[:, None] * stride + offs_n[None, :]
            pflows = tl.load(param_flows + pf0 + rel, mask = mask, other = 0)
            if KEEP_ZERO:
                new_param = (pflows + pseudocount / nch) / tot[None, :]
            else:
                new_param = (pflows + pseudocount / nch) / (tot[None, :] + pseudocount)
            if HAS_MASTER:
                stored = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0)
                mval = tl.load(master + par0 + rel, mask = mask, other = 0)
                old = tl.where(stored != mval.to(stored.dtype), stored.to(tl.float32), mval)
            else:
                old = tl.load(stored_ptr + par0 + rel, mask = mask, other = 0).to(tl.float32)
            if KEEP_ZERO:
                upd = tl.where(old < 1e-12, 0.0, (1.0 - step_size) * old + step_size * new_param)
            else:
                upd = (1.0 - step_size) * old + step_size * new_param
            if HAS_MASTER:
                tl.store(master + par0 + rel, upd, mask = mask)
            tl.store(stored_ptr + par0 + rel, upd.to(stored_ptr.dtype.element_ty), mask = mask)


class _Tables:
    """A circuit's node blocks for the update, structure only: those of at least `MIN_BLOCK_SIZE` nodes (each its first
    parameter and flow, children, size, and its `ns`'s parameter range start, for exclusions), cut into tiles per tile
    width and grouped by children (kept per exclusion); the per-block tables of the others."""

    def __init__(self, pc):
        blocks, gnid = [], 0
        for ns in pc.root_ns:
            if not ns.is_sum() or ns.is_tied():
                continue
            par_start, pflow_start = ns._param_range[0], ns._param_flow_range[0]
            bs, cbs = ns.block_size, ns.ch_block_size
            num_edges = torch.bincount(ns.edge_ids[0, :].cpu(), minlength = ns.num_node_blocks).numpy()
            for ng in range(ns.num_node_blocks):
                nch = int(num_edges[ng]) * cbs
                if bs >= MIN_BLOCK_SIZE and nch > 0:
                    blocks.append((par_start, pflow_start, nch, bs, ns._param_range[0], gnid))
                par_start += bs * nch
                pflow_start += bs * nch
                gnid += bs                          # (the global node ids `compile_par_update_fn` gives)
        self.blocks = np.asarray(blocks, dtype = np.int64).reshape(-1, 6)
        self.device = pc.params.device
        self.cache = dict()
        self.partials = None                       # the split kernels' scratch, grown to the largest launch

    def scratch(self, floats: int) -> torch.Tensor:
        if self.partials is None or self.partials.numel() < floats:
            self.partials = torch.empty(floats, dtype = torch.float32, device = self.device)
        return self.partials

    def groups(self, pc, exclude: Sequence) -> tuple:
        """``(groups, per-block kwargs)`` without the parameter ranges ``exclude`` (each a whole `ns`'s): ``groups``
        ``{(children bucket, nodes bucket): blocks}`` -- the children's bucket the power of two at or above the blocks'
        children up to `ONE_PASS_MAX`, None past it; the nodes' the power of two at or above the blocks' size up to
        `BLOCK_N`, None past it --, every one's tiles made per width on demand (`tiles`)."""
        key = tuple(sorted(exclude))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        b = self.blocks
        keep = np.ones([b.shape[0]], dtype = bool)
        for ps, pe in key:
            keep &= ~((b[:, 4] >= ps) & (b[:, 4] < pe))
        b = b[keep]
        bucket = np.where(b[:, 2] <= ONE_PASS_MAX, 1 << np.ceil(np.log2(np.maximum(b[:, 2], 1))).astype(np.int64), 0)
        nbucket = np.where(b[:, 3] <= BLOCK_N, 1 << np.ceil(np.log2(np.maximum(b[:, 3], 1))).astype(np.int64), 0)
        groups = {}
        for v, w in sorted(set(zip(bucket.tolist(), nbucket.tolist()))):
            groups[(None if v == 0 else v, None if w == 0 else w)] = {"blocks": b[(bucket == v) & (nbucket == w)],
                                                                     "tiles": {}}
        par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = \
            pc.par_update_kwargs
        small = blk_intervals < MIN_BLOCK_SIZE
        for ps, pe in key:
            small &= ~((par_start_ids >= ps) & (par_start_ids < pe))
        sel = torch.nonzero(small).flatten()
        kwargs = [par_start_ids[sel].contiguous(), pflow_start_ids[sel].contiguous(), blk_sizes[sel].contiguous(),
                  blk_intervals[sel].contiguous(), global_nids[sel].contiguous(), nchs[sel].contiguous(),
                  cum_pflows, metadata]
        self.cache[key] = hit = (groups, kwargs)
        return hit

    def tiles(self, group: dict, width: int) -> list:
        """The group's tiles of up to ``width`` nodes, as the kernels take them: [first parameter, first flow, children,
        row stride, nodes, first global node id] (device tensors)."""
        t = group["tiles"].get(width)
        if t is None:
            rows = []
            for par, pflow, nch, bs, _, gnid in group["blocks"].tolist():
                for n0 in range(0, bs, width):
                    rows.append((par + n0, pflow + n0, nch, bs, min(width, bs - n0), gnid + n0))
            a = np.asarray(rows, dtype = np.int64).reshape(-1, 6)
            t = group["tiles"][width] = [torch.from_numpy(a[:, k].copy()).to(self.device) for k in range(6)]
        return t


def _tables(pc) -> _Tables:
    t = pc.__dict__.get("_em_node_block_tables")
    if t is None or t.device != pc.params.device:
        t = pc.__dict__["_em_node_block_tables"] = _Tables(pc)
    return t


def _form(nch_bucket, nodes_bucket, has_master: bool) -> tuple:
    """The kernel form for a group of node blocks of ``nch_bucket`` children (None: past `ONE_PASS_MAX`) and
    ``nodes_bucket`` nodes (None: past `BLOCK_N`), as `_launch` takes it: a function of the shape alone, so every
    process updating the same values gets the same bits."""
    if nch_bucket is None:
        return ("split", BLOCK_N, 4)
    if has_master and nodes_bucket is not None and BLOCK_FORM_NODES[0] <= nodes_bucket <= BLOCK_FORM_NODES[1]:
        return ("block", nodes_bucket, BLOCK_WARPS)
    width, warps = ONE_PASS_TILE_MASTER if has_master else ONE_PASS_TILE
    return ("one", width, warps)


def _launch(cfg, tables, group, nch_bucket, stored, master, param_flows, cum_pflows, step_size, pseudocount, keep_zero):
    """One group's update with ``cfg``: ``("split", BLOCK_N, 4)`` the split kernels (`SPLIT_CHUNK` children a program,
    the partial totals merged), ``("block", nodes, warps)`` the whole-block kernel (a tile the block), ``("one", nodes,
    warps)`` the one-pass one (its children ``nch_bucket``)."""
    kind, width, warps = cfg
    tiles = tables.tiles(group, width)
    if tiles[0].numel() == 0:
        return
    m = master if master is not None else stored
    if kind == "split":
        G = cum_pflows.numel()
        chunks = -(-int(group["blocks"][:, 2].max()) // SPLIT_CHUNK)
        partials = tables.scratch(chunks * G)
        t_par, t_pflow, t_rows, t_stride, t_nodes, t_gnid = tiles
        _em_split_partials_kernel[(tiles[0].numel(), chunks)](
            stored, m, param_flows, partials, *tiles, G, pseudocount, HAS_MASTER = master is not None,
            KEEP_ZERO = keep_zero, CHUNK = SPLIT_CHUNK, BLOCK_C = BLOCK_C, BLOCK_N = width, num_warps = warps)
        _em_split_merge_kernel[(tiles[0].numel(),)](partials, cum_pflows, t_rows, t_nodes, t_gnid, G,
                                                    CHUNK = SPLIT_CHUNK, BLOCK_N = width, num_warps = 1)
        _em_split_update_kernel[(tiles[0].numel(), chunks)](
            stored, m, param_flows, cum_pflows, *tiles, step_size, pseudocount, HAS_MASTER = master is not None,
            KEEP_ZERO = keep_zero, CHUNK = SPLIT_CHUNK, BLOCK_C = BLOCK_C, BLOCK_N = width, num_warps = warps)
    elif kind == "block":
        _em_block_kernel[(tiles[0].numel(),)](
            stored, m, param_flows, cum_pflows, *tiles, step_size, pseudocount, HAS_MASTER = master is not None,
            KEEP_ZERO = keep_zero, BLOCK_C = BLOCK_STEP, BLOCK_N = width, num_warps = warps)
    else:
        _em_node_block_one_pass_kernel[(tiles[0].numel(),)](
            stored, m, param_flows, cum_pflows, *tiles, step_size, pseudocount, HAS_MASTER = master is not None,
            KEEP_ZERO = keep_zero, NCH = nch_bucket, BLOCK_N = width, num_warps = warps)


def em_update(pc, param_flows: torch.Tensor, step_size: float, pseudocount: float, keep_zero_params: bool,
              exclude: Sequence = (), master: Optional[torch.Tensor] = None) -> None:
    """Mini-batch EM's update of ``pc.params`` (and of their fp32 master copy ``master`` when given) from
    ``param_flows``, every sum node not in the parameter ranges ``exclude``: the node-block kernels for blocks of at
    least `MIN_BLOCK_SIZE` nodes (which of them per group of children: `_form`), the per-block kernels for the
    others."""
    from .par_update import em_par_update
    from .params_master import em_par_update_master
    tables = _tables(pc)
    groups, small = tables.groups(pc, exclude)
    stored = pc.params.data
    step_size, pseudocount, keep_zero = float(step_size), float(pseudocount), bool(keep_zero_params)
    # Every node's total in `par_update_kwargs`' buffer, as the per-block kernels leave it -- their call first: it
    # zeroes the whole buffer (excluded nodes: 0), the node-block kernels then write their own nodes'
    cum_pflows = pc.par_update_kwargs[6]
    if small[0].numel() > 0:
        if master is None:
            em_par_update(stored, param_flows, small, step_size = step_size, pseudocount = pseudocount,
                          keep_zero_params = keep_zero)
        else:
            em_par_update_master(master, stored, param_flows, small, step_size, pseudocount, keep_zero)
    elif cum_pflows is not None:
        cum_pflows.zero_()
    if cum_pflows is None:
        cum_pflows = torch.zeros([pc.par_update_kwargs[7]["tot_num_nodes"]], dtype = torch.float32,
                                 device = stored.device)
    for (nch_bucket, nodes_bucket), group in groups.items():
        cfg = _form(nch_bucket, nodes_bucket, master is not None)
        _launch(cfg, tables, group, nch_bucket, stored, master, param_flows, cum_pflows, step_size, pseudocount,
                keep_zero)
