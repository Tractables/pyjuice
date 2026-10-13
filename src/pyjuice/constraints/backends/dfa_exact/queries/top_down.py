"""
The lifted top-down pass: every sum node's TOP-DOWN value in ``node_flows`` (laid out as ``node_mars``), from the root
down -- the log of what ``p(C, e)`` multiplies the node's inside value by, over the positions outside its scope: its
log outside value, or pyjuice's top-down probability (:func:`~pyjuice.model.backend.eval_top_down_probs`) with the
evidence and the automaton's columns. The lifted circuit is a smooth, decomposable circuit (the PC times the
automaton's), so this is its backward pass, the pyjuice flow of a node being ``exp(node_mars + node_flows - log p(C,
e))``; the top-down value is kept rather than the flow because it depends only on the evidence outside the node's
scope, which a cache can key on.

Layer by layer from the top, mirroring :mod:`.forward`, over the tables of
:class:`~.top_down_program.TopDownProgram`:

* a sum layer gives its children (the products below) their top-down values: dense node blocks by the transposed
  matrix product on the parameter buffer's own weight views, the others by a child-major kernel over pyjuice's edge
  tables (:func:`~.sum.dense_sum_top_down`, :func:`~.sum.fused_sum_top_down`); the values go to
  ``element_flows`` -- ``element_mars``'s storage, whose product insides the pass never reads -- or, for a product
  that was a copy of a sum (aliased, never materialized), straight to that sum's rows;
* a product layer gives its children their top-down values from its own: the transposed contractions of
  :mod:`.prod_top_down`, reading the siblings' inside values (sum rows, or a fold's scratch row, which the
  layer's fold steps compute again first).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from . import forward as F
from ..compiled.tables import _align
from ..kernels.common import tensor
from ..kernels.prod import run_products as products
from ..kernels.prod_top_down import run_top_down
from ..kernels.score import run_class_weights
from ..kernels.sum import dense_sum_top_down, fill_dead_rows, fused_sum_top_down, stacked_weights



def run_backward(backend, bufs: dict, flows: dict, plan, x: torch.Tensor, missing: torch.Tensor,
                 obs_class: torch.Tensor, liveness, precision: str, visit: Optional[Callable[[int], None]] = None,
                 class_weights: Optional[dict] = None, class_mars: Optional[torch.Tensor] = None):
    """
    The top-down values a query plan computes (``plan.top_down``, see :mod:`.plans`), after
    :func:`~.forward.run_forward` filled the inside values of ``bufs`` under ``plan``: the per-sample ones into
    ``flows["node_flows"]``, those kept once into ``prior_flows`` (:meth:`~.state.State.prior_flows`), for a batch
    of one. Products' values pass through ``flows["element_flows"]`` and the folds' through
    the layer's scratch rows (per sample, or a batch of one in the same rows).

    :param visit: None, or called with a product layer group's index once its products' top-down values are in
        ``element_flows`` (per sample, or for a batch of one where the plan keeps them once), before the next group's
        overwrite them: where the values of the products of the targets' inputs are read
    :param class_weights: None, or where the class weights of the target positions' input nodes go, every sample's
        (see :func:`~.score.run_class_weights`; the plan must be a scoring one, see :func:`~.query_plan.
        top_down_plan`)
    :param class_mars: the query's class masses if its forward has them (:func:`~.forward.run_forward`'s), else None
    """
    pc, prog, td = backend.pc, backend.program(), plan.top_down
    out = backend.top_down_program()
    B = x.size(0)
    lay = bufs["layout"]
    node_mars, element_mars = bufs["node_mars"], bufs["element_mars"]
    node_flows, element_flows = flows["node_flows"], flows["element_flows"]
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots)
    prior = backend.state.prior_buffer() if plan.prior else None
    srcs = dict(mars = node_mars, offset = regions["offset"], width = regions["width"], reg_prior = plan.reg_prior)
    if prior is not None:
        srcs.update(mars = prior["mars"], offset = prior["offset_t"], width = prior["width_t"])
    pf = backend.state.prior_flows(td.once) if td.once.any() else None
    if pf is not None:                                   # where the values kept once go
        once = dict(offset = pf["offset_t"], width = pf["width_t"], first = prog.reg_first, slots = prog.reg_slots)
    dot = F.PRECISIONS[precision]
    live = liveness.live if liveness is not None else None
    lanes = (liveness.lanes.perm, liveness.lanes.reg_first, liveness.lanes.reg_live) if liveness is not None else None
    exact = liveness is not None
    if class_mars is None:
        class_mars = backend.state.class_masses()
    prog.transitions(class_mars, backend.state.class_mars_version)
    stacks, kept = backend.state.weight_stacks()

    r = prog.root_region                                 # the root: top-down value 0 (log 1) on its slot
    computed, root_once = td.root
    if computed:
        dest, offset, width, cols = (pf["flows"], pf["offset"][r], pf["width"][r], prog.slots[r]) if root_once else \
            (node_flows, lay["sum_offsets"][r], lay["sum_widths"][r], B * prog.slots[r])
        base = offset + (prog.root_rows[0] - prog.firsts[r]) * width
        for row in range(prog.root_rows[1] - prog.root_rows[0]):
            dest[base + row * width:base + row * width + cols] = 0.0
    if td.dead is not None:                              # rows no launch writes, of the regions computed per sample
        fill_dead_rows(node_flows, td.dead, regions, max(lay["sum_widths"][r] for r in td.dead_regions))

    for kind, prod_index, layers in td.steps:
        elem_first = backend.element_regions[prod_index][0]
        elem_width = lay["element_widths"][prod_index]
        if kind == "sum":
            for block_size, dense_runs, fused_groups in layers:
                for part in (0, 1):                      # once, then per sample
                    if part == 0 and not (dense_runs[0] or fused_groups[0]):
                        continue
                    nf, regs, nb = (pf["flows"], once, 1) if part == 0 else (node_flows, regions, B)
                    offsets, widths = (pf["offset"], pf["width"]) if part == 0 else \
                        (lay["sum_offsets"], lay["sum_widths"])
                    for r, c0, E, piece, blocks, key, add in dense_runs[part]:
                        ncols = nb * prog.slots[r]
                        cols = None
                        if part == 1 and liveness is not None and \
                                F.prune_dense(ncols, liveness.lanes.num_live(prog.reg_iv_host[r])):
                            cols = liveness.lanes.segment(prog.reg_iv_host[r], ncols)
                        stacked = None
                        width_ = cols[2] if cols is not None else ncols
                        if key is not None and width_ <= F.STACK_MAX_COLUMNS[precision] and \
                                (kept or prog.stack_uses[key] > 1):
                            if key not in stacks:
                                stacks[key] = stacked_weights(pc.params, key[2], key[0], block_size)
                            stacked = stacks[key]
                        dense_sum_top_down(element_flows, nf, pc.params, blocks, E, block_size,
                                          (offsets[r], widths[r], prog.firsts[r]), c0, elem_first, elem_width, ncols,
                                          dot, add, live = cols, stacked = stacked, piece = piece)
                    for cbs, tables, idx in fused_groups[part]:
                        fused_sum_top_down(element_flows, nf, pc.params, tables, regs, elem_first, elem_width, nb,
                                          block_size, cbs, max_cols = nb * tables["max_slots"], precision = dot,
                                          liveness = liveness if part == 1 else None, idx = tensor(idx))
            if visit is not None:
                visit(prod_index)
        else:
            for folds, stages, num_temps, temp_slots in layers:
                temp, temp_flows, temp_width = node_mars, node_mars, 0
                if num_temps > 0:
                    temp_width = _align(B * temp_slots)
                    temp = backend.state.storage_view("product_scratch", num_temps * temp_width)
                    temp_flows = backend.state.storage_view("product_scratch_flows", num_temps * temp_width)
                if folds[0]:                             # the folds' inside values, for a batch of one ...
                    products(folds[0], (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             prog.no_evidence, None, elem_first, elem_width, 1, srcs, exact = True)
                if folds[1]:                             # ... and per sample
                    products(folds[1], (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             obs_class, live, elem_first, elem_width, B, srcs, exact = exact)
                for once_launches, sample_launches, cw_launches in stages:
                    if once_launches:
                        run_top_down(once_launches, (temp, temp_width), prog, once, element_flows, temp_flows,
                                     pf["flows"], node_mars, class_mars, prog.no_evidence, None, elem_first,
                                     elem_width, 1, srcs, out.skip, True, td.once_iv)
                    if sample_launches:
                        run_top_down(sample_launches, (temp, temp_width), prog, regions, element_flows, temp_flows,
                                     node_flows, node_mars, class_mars, obs_class, live, elem_first, elem_width, B,
                                     srcs, out.skip, exact, td.once_iv, lanes = lanes)
                    if cw_launches and class_weights is not None:
                        run_class_weights(cw_launches, class_weights, prog, (temp, temp_width), element_flows,
                                          temp_flows, node_mars, liveness, elem_first, elem_width, B, srcs, td.once_iv,
                                          regions)
    if pf is not None:
        pf["valid"] |= td.todo
