"""
The launches of a query's top-down product steps: their tiles, the tuner's choices among them, what each kernel is told.
"""

from __future__ import annotations

from typing import Optional
import torch

from pyjuice.layer.kernels import autotune
from pyjuice.utils.util import host_cdiv, host_next_power_of_2
from ..common import NODE_TILE, SUM, grid_chunks, tensor, pdl_launch
from ..transitions import enumerated_args
from ..prod import compact_tiles
from ..prod.block_block import EXACT_BELOW
from ..prod.run import (BLOCK_BLOCK_PROGRAMS_PER_SM, BLOCK_BLOCK_TILES, PRODUCT_PRECISION, SKIP_EMPTY_TILES,
                        _flat_tile, _tile)
from .block_block import _block_block_top_down_compact_kernel, _block_block_top_down_kernel
from .input_block import _block_input_top_down_kernel, _input_block_top_down_kernel
from .launches import _copy_top_down_kernel
from .walk import _input_block_cuda


#: children (of a node tile) and (children x lanes) per program of the ``input @ block`` and ``block @ input``
#: top-down kernels, at most, and its warps: many small programs (Codpula diffusion, 4096 latents: 8 x 16 lanes 0.92
#: ms per call, 16 x 16 1.09, 16 x 64 1.89; 1 x 128, every child its own walk, 2.46)
TOP_DOWN_NODES = 8
TOP_DOWN_TILE_LANES = 128
TOP_DOWN_WARPS = 4


def run_top_down(launches, bufs, prog, regions, element_flows, temp_flows, node_flows, node_mars, masses, obs_class,
                live, elem_first: int, elem_width: int, B: int, prior: dict, skip, exact: bool, once_iv: torch.Tensor,
                lanes = None, compact = None, tflow_width = None, inside_regions = None, inside: bool = False,
                allowed: Optional[torch.Tensor] = None):
    """
    The top-down pass of one product layer's stage: every :class:`TopDownLaunch` in ``launches``, each with the plan's
    index list of the children it computes (int32 on the device) or None for every one.

    :param bufs: ``(temp, temp_width)``, the layer's scratch rows, holding the folds' inside values (recomputed)
    :param regions: ``offset``, ``width``, ``first`` and ``slots`` of every sum region (device tensors) of the buffer
        the children are written to (``node_flows``, or ``prior_flows`` for a batch of one)
    :param element_flows, temp_flows, node_flows: the top-down buffers (``temp_flows`` as wide as ``temp``)
    :param masses: the query's :class:`~..state.ClassMasses` (per sample, the ``SAMPLE`` kernels)
    :param prior: where the sum operands the plan keeps once are (see :func:`~.common._block_ptr`)
    :param skip: the top-down pass's :class:`~.prod.SkipMasks` (its triples are the parents' ``(interval A, B)`` specs)
    :param exact: see :func:`~.prod.run_products`
    :param once_iv: int32 [intervals] (device, by ``prog.intervals``' numbers): 1 where the query keeps an interval's
        top-down values once, for a batch of one: a parent there is read with sample stride 0
    :param lanes: None, or under evidence pruning the query's :class:`~.live.LaneOrder` ``(perm, reg_first,
        reg_live)``: a sum child's lanes run in that order, its dead ones only set to -inf
    :param compact: None, or the query's :class:`~.compact.CompactLayout`: the per-sample values (children and
        parents) are compact -- ``regions``, ``elem_width`` and ``tflow_width`` are their rows'; the children are
        written, and the parents read, a lane per pair of live columns
    :param tflow_width: the scratch rows' width in ``temp_flows`` (None: ``bufs``')
    :param inside_regions: the sum regions' tables in ``node_mars`` (None: ``regions``), for the inside values a
        ``block @ block`` launch reads
    :param inside: whether those inside values are compact too (by ``compact``'s layout; ``inside_regions`` and
        ``bufs`` are their rows')
    :param allowed: None, or the classes a session's per-sample inputs leave each sample
        (:attr:`~.live.Liveness.allowed`), by which the CUDA ``input @ block`` walk drops terms
        (:func:`_input_block_cuda`)
    """
    temp, temp_width = bufs
    tflow_width = temp_width if tflow_width is None else tflow_width
    inside_regions = regions if inside_regions is None else inside_regions
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  out_first = elem_first, out_width = elem_width, temp_width = tflow_width)
    n, W, C = prog.n, prog.W, prog.C
    ivs = prog.intervals
    prune, live = live is not None, live if live is not None else obs_class
    tt, grouped = prog.trans_top_down, prog.trans_top_down.grouped
    margs = prog.mass_args()                                              # the query's transition masses
    mpos = tuple(margs[k] for k in ("trans", "mbase", "mcount", "mrank", "bbase", "mb_stride", "mrank_stride"))
    sample = masses.batch > 1
    if compact is not None:
        cl = compact
        lists = dict(coff = cl.off, ccount = cl.count, cstart = cl.start, ccolumns = cl.columns, csteps = cl.steps)
    else:
        lists = dict(coff = live, ccount = live, cstart = live, ccolumns = live, csteps = 0)
    for la, item in launches:
        idx = tensor(item)
        Cn = la.children.size(0) if idx is None else idx.numel()
        indexed = dict(children_idx = la.children if idx is None else idx, INDEXED = idx is not None)
        at = 3 * la.number
        lanes_max = compact.lanes(la.host["child_ivs"]) if compact is not None else B * prog.td_most[at + 2]   # a child's most
        if la.form == "copy":
            c_kind, out_temp = la.kinds
            tile = _tile(max(lanes_max, 1))
            for first, size in grid_chunks(host_cdiv(lanes_max, tile), 1):
                _copy_top_down_kernel[(Cn, size)](
                    element_flows, temp_flows, node_flows, ivs.of, ivs.info, once_iv, lists["coff"],
                    children = la.children, ptr = la.ptr, parents = la.parents, B = B, n = n, pid0 = first,
                    C_KIND = c_kind, OUT_TEMP = out_temp, COMPACT = compact is not None, TILE = tile, num_warps = 4,
                    **indexed, **common)
        elif la.form in ("input_block", "block_input"):
            c_kind, out_temp, (c_full, out_full) = la.kinds
            ordered = prune and lanes is not None and c_kind == SUM and compact is None
            tiles, height = (la.tiles, la.tile_height) if idx is None else (idx, item.height)
            if la.form == "input_block" and compact is not None and lanes_max > 0 and \
                    _input_block_cuda(la, tiles, height, prog, compact, element_flows, temp_flows, node_flows,
                                      node_mars, masses, obs_class, once_iv, common, B, allowed):
                continue
            tn = min(height, TOP_DOWN_NODES)
            tile = max(16, min(_flat_tile(max(lanes_max, 1)), TOP_DOWN_TILE_LANES // tn))
            kernel = _input_block_top_down_kernel if la.form == "input_block" else _block_input_top_down_kernel
            pdl = pdl_launch(element_flows.device)
            tables = (tt.pred_ptr, tt.pred_col, prog.mass_x("pred_x")) if la.form == "input_block" else \
                (tt.entry_ptr, tt.entry_succ, prog.mass_x("entry_x"))
            for first, size in grid_chunks(host_cdiv(lanes_max, tile), 1):
                kernel[(tiles.numel() // NODE_TILE * (height // tn), size)](
                    element_flows, temp_flows, node_flows, node_mars, obs_class, prog.next_col_arg, prog.delta,
                    prog.state_id, prog.col_of_state, masses.table,
                    masses.row, masses.stride if sample else masses.row, *mpos, *tables, prog.width_t,
                    ivs.of, ivs.info, ivs.table, once_iv, live,
                    *(lanes if ordered else (live, live, live)), crank = compact.rank if compact is not None else live,
                    children = la.children, tiles = tiles, ptr = la.ptr, parents = la.parents, B = B, n = n, W = W,
                    C = C, K = prog.K, input_start = prog.input_start, pid0 = first, C_KIND = c_kind,
                    OUT_TEMP = out_temp, HOMOGENEOUS = prog.homogeneous,
                    GROUPED = grouped, PRUNE = prune, ORDERED = ordered, COMPACT = compact is not None,
                    C_FULL = c_full, OUT_FULL = out_full, SAMPLE = sample, SAMPLE_MASSES = margs["SAMPLE_MASSES"],
                    TN = tn, SUBS = height // tn, TILE = tile, PDL = pdl, launch_pdl = pdl,
                    num_warps = TOP_DOWN_WARPS, **common, **lists, **enumerated_args(prog, masses))
        elif compact is not None:                    # block @ block on compact values, a sample's rectangle at a time
            c_kind, o_kind, out_temp = la.kinds
            ends = prog.iv_ends_np[la.host["child_ivs"]]
            rmax, cmax = int(compact.max_count[ends[:, 0]].max()), int(compact.max_count[ends[:, 1]].max())
            TM, TN, TK, warps = compact_tiles(rmax, cmax)
            tiles = host_cdiv(rmax, TM) * host_cdiv(cmax, TN)
            if tiles == 0:
                continue
            split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, Cn * B))
            group = host_cdiv(tiles, max(1, split))
            split = host_cdiv(tiles, group)
            for first, size in grid_chunks(Cn * B * split, 0):
                _block_block_top_down_compact_kernel[(size,)](
                    element_flows, temp_flows, node_flows, node_mars, prior["mars"], temp, prog.width_t, ivs.of,
                    ivs.info, ivs.table, once_iv, compact.off, compact.count, compact.start, compact.columns,
                    la.children, ptr = la.ptr, parents = la.parents, mars_offset = inside_regions["offset"],
                    mars_width = inside_regions["width"], prior_offset = prior["offset"], prior_width = prior["width"],
                    reg_prior = prior["reg_prior"], tflow_width = tflow_width, B = B, n = n, SPLIT = split,
                    GROUP = group, exact_below = EXACT_BELOW, pid0 = first, C_KIND = c_kind, O_KIND = o_kind,
                    OUT_TEMP = out_temp, SIDE_L = la.side == "L", MARS = inside, TM = TM, TN = TN, TK = TK,
                    PRECISION = PRODUCT_PRECISION, num_warps = warps, num_stages = 1, **indexed,
                    **dict(common, temp_width = temp_width))
        else:
            c_kind, o_kind, out_temp = la.kinds
            side_l = la.side == "L"

            E_max, X_max = prog.td_most[at], prog.td_most[at + 1]

            def block_block(cfg, nf, tf, la = la, Cn = Cn, indexed = indexed, c_kind = c_kind, o_kind = o_kind,
                            out_temp = out_temp, side_l = side_l, E_max = E_max, X_max = X_max):
                TM, TN, TK, warps = cfg
                tiles = host_cdiv(E_max, TM) * host_cdiv(X_max, TN)
                split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, Cn * B))
                group = host_cdiv(tiles, max(1, split))
                split = host_cdiv(tiles, group)
                for first, size in grid_chunks(Cn * B * split, 0):
                    _block_block_top_down_kernel[(size,)](
                        element_flows, tf, nf, node_mars, prior["mars"], temp, prog.width_t, ivs.of, ivs.info,
                        ivs.table, once_iv, live, la.children, ptr = la.ptr, parents = la.parents,
                        skip_bits = skip.bits, prior_offset = prior["offset"], prior_width = prior["width"],
                        reg_prior = prior["reg_prior"], B = B, n = n, W = W, TILES_I = skip.tiles_i,
                        TILES_J = skip.tiles_j, WORDS = skip.words, SPLIT = split, GROUP = group,
                        exact_below = EXACT_BELOW, pid0 = first, C_KIND = c_kind, O_KIND = o_kind, OUT_TEMP = out_temp,
                        SIDE_L = side_l, TM = TM, TN = TN, TK = TK, SUB_I = TM // skip.tm, SUB_J = TN // skip.tn,
                        SKIP = SKIP_EMPTY_TILES, PRUNE = prune, EXACT = exact, PRECISION = PRODUCT_PRECISION,
                        num_warps = warps, num_stages = 1, tflow_width = tflow_width, **indexed,
                        **dict(common, temp_width = temp_width))

            # a child reached before log-adds to its value, so the launch is not idempotent: the candidates are timed
            # into throwaway copies of the outputs (and not at all when those cannot be allocated)
            key = ("lifted_block_block_top_down", la.kinds, la.side, prune, exact, SKIP_EMPTY_TILES, PRODUCT_PRECISION,
                   B, host_next_power_of_2(Cn), E_max, X_max)
            cfg = BLOCK_BLOCK_TILES[0]
            if autotune.should_tune(key, len(BLOCK_BLOCK_TILES)):
                nf_s, tf_s = autotune.scratch_like(node_flows), autotune.scratch_like(temp_flows)
                if nf_s is not None and tf_s is not None:
                    cfg = autotune.pick(key, BLOCK_BLOCK_TILES, lambda c: block_block(c, nf_s, tf_s))
                del nf_s, tf_s
            else:
                cfg = autotune.pick(key, BLOCK_BLOCK_TILES, None)
            block_block(cfg, node_flows, temp_flows)
