"""
The class weights of a query's boundaries: the walks and rows they take, and their launches.
"""

from __future__ import annotations

from typing import Optional
import torch

from pyjuice.utils.util import host_cdiv, host_next_power_of_2
from ..common import IDENTITY, NODE_TILE, grid_chunks, tensor, pdl_launch
from ..c import get_class_weights_module
from .class_rows import ClassWeightRows
from .class_weight_kernels import (CLASS_WEIGHT_CUDA, CLASS_WEIGHT_CUDA_CLASSES, _class_weight_cuda,
                                   _class_weight_kernel, _long_walk)


#: :func:`_class_weight_kernel`'s walk: 0, every program chooses (its lanes' longest class list against the columns
#: it would walk otherwise); 1, the classes' columns; 2, the sample's columns (tests)
CLASS_WEIGHT_WALK = 0


#: destinations (of a node tile), and destinations x lanes (sample, class) x other-boundary columns, per
#: :func:`_class_weight_kernel` program, at most (Codpula diffusion, 4096 latents: 16 x 32 lanes 1.09 ms per call,
#: 8 x 32 1.16, 8 x 128 1.34, 16 x 128 1.65; 1 x 128, every destination its own walk, 2.29)
CLASS_WEIGHT_NODES = 16
CLASS_WEIGHT_LANES = 512


def class_weight_walk(rows: ClassWeightRows, prog, liveness, compact = None):
    """``rows``' longest walk at every boundary (:attr:`ClassWeightRows.walk`: the most live columns of any sample
    there), the class weights' kernel choice (:data:`CLASS_WEIGHT_CUDA_WALK`): a compact query's layout's
    (``compact.max_count``, on the host already), else its ``liveness``' lists' counts copied behind the launches, or
    every column without pruning."""
    n = prog.n
    if compact is not None:
        rows._walk = compact.max_count.tolist()
    elif liveness is not None:
        host = torch.empty(n + 1, dtype = torch.int64, pin_memory = True)
        host.copy_(liveness.lists()[0].view(liveness.B, n + 1).amax(0), non_blocking = True)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(prog.dev.index))
        rows._walk_copy = (host, done)
    else:
        rows._walk = prog.width[:n + 1]


def run_class_weights(launches, cw: dict, prog, bufs, element_flows, temp_flows, node_mars, liveness, elem_first: int,
                      elem_width: int, B: int, prior: dict, once_iv: torch.Tensor, regions: dict, compact = None,
                      tflow_width: Optional[int] = None, inside: bool = False):
    """
    The class weights of one product layer stage's input operands at the query's target positions: every
    :class:`ClassWeightLaunch` in ``launches`` as ``(launch, index list of destinations (None: all), store)`` --
    ``store``: the launch writes each of its destinations first in the query, and stores the weights there; the others
    log-add them.

    :param cw: ``buf`` (``dfa_class_flow``: -inf where a launch log-adds and nothing was written yet), ``rows`` (its
        :class:`ClassWeightRows`: every target column's live classes, and where their rows are), and per position its
        first target column (``col``, int64 [n], -1 for a position no sample targets) and its row group's rows (``K``,
        int64 [n])
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows (the folds' inside values)
    :param liveness: None, or the query's :class:`~.live.Liveness` (its live columns' lists are built on first use)
    :param prior: where the sum operands kept once are (see :func:`~.common._block_ptr`)
    :param once_iv: the plan's per-interval flag of top-down values kept once (read with sample stride 0)
    :param compact: None, or the query's :class:`~.compact.CompactLayout`: the outputs' top-down values are compact,
        ``elem_width`` and ``tflow_width`` the widths of their rows (``regions``: the sum operands' inside values')
    :param tflow_width: the scratch rows' width in ``temp_flows`` (None: ``bufs``')
    :param inside: whether the other operands' per-sample inside values are compact too (by ``compact``'s layout;
        ``regions`` and ``bufs`` are their rows')
    """
    temp, temp_width = bufs
    tflow_width = temp_width if tflow_width is None else tflow_width
    n, W, C = prog.n, prog.W, prog.C
    ivs, tt = prog.intervals, prog.trans
    prune = liveness is not None
    lists = liveness.lists() if prune else (cw["K"],) * 3
    ctables = (compact.off, compact.rank) if compact is not None else (cw["K"],) * 2
    rows = cw["rows"]
    TC = min(64, host_next_power_of_2(max(rows.most, 1)))                   # (a lane a live class of a column)
    mod = get_class_weights_module() if CLASS_WEIGHT_CUDA and prune and compact is not None else None
    for la, item, store in launches:
        idx = tensor(item)
        tiles, height = (la.tiles, la.tile_height) if idx is None else (idx, item.height)
        s_kind, out_temp, (out_full, *s_full) = la.kinds
        identity = la.form == "input_block" and s_kind == IDENTITY
        J_max = prog.cw_most[la.number]
        if mod is not None and la.form == "input_block" and not identity and J_max == 1 and \
                C <= CLASS_WEIGHT_CUDA_CLASSES and prog.W < 65535 and _long_walk(la, rows) and \
                _class_weight_cuda(mod, la, tiles, height, cw, rows, prog, (temp, once_iv), element_flows, temp_flows,
                                   node_mars, liveness, compact, prior, regions, elem_first, elem_width, temp_width,
                                   tflow_width, B, s_kind, out_temp, out_full, all(s_full), inside, store):
            continue
        TJ = 1 if identity else min(64, host_next_power_of_2(J_max))
        TN = min(height, CLASS_WEIGHT_NODES, max(1, CLASS_WEIGHT_LANES // (TC * TJ)))   # (powers of 2)
        TB = max(1, min(host_next_power_of_2(B), 128 // TC, CLASS_WEIGHT_LANES // (TC * TJ * TN)))
        lanes = TN * TB * TC * TJ
        warps = 1 if lanes <= 128 else 2 if lanes <= 512 else 4
        pdl = pdl_launch(element_flows.device)
        for first, size in grid_chunks(host_cdiv(B, TB), 1):
            _class_weight_kernel[(tiles.numel() // NODE_TILE * (height // TN), size, host_cdiv(max(rows.most, 1), TC))](
                cw["buf"], cw["col"], cw["K"], rows.count, rows.cls, rows.off, element_flows, temp_flows, node_mars,
                prior["mars"], temp, prog.next_col_arg, prog.delta, prog.state_id, prog.col_of_state, tt.cls_ptr,
                tt.cls_col, tt.cls_succ,
                liveness.live if prune else cw["K"], prog.width_t,
                ivs.of, ivs.info, ivs.table, once_iv, *lists, *ctables,
                la.dests, tiles, la.ptr, la.parents, regions["offset"], regions["width"], regions["first"],
                prior["offset"], prior["width"], prior["reg_prior"], elem_first, elem_width, temp_width, tflow_width,
                B, n, W, C, prog.K, first, BLOCK_INPUT = la.form == "block_input", S_KIND = s_kind,
                OUT_TEMP = out_temp,
                PRUNE = prune, OUT_FULL = out_full, S_FULL = all(s_full), UNIFORM = not prune or TB == 1,
                COMPACT = compact is not None, MARS = inside, STORE = store, WALK = CLASS_WEIGHT_WALK, TN = TN,
                HOMOGENEOUS = prog.homogeneous,
                SUBS = height // TN, TB = TB, TC = TC, TJ = TJ, PDL = pdl, launch_pdl = pdl, num_warps = warps)
