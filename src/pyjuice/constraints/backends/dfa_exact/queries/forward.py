"""
The lifted forward pass: ``log p(C, e)`` for a batch of evidence, in the backend's buffers (:class:`~.state.State`).

Layer by layer, as :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of
every observed token, the class masses of missing ones are summed per token class (by the backend's state, see
:meth:`~.state.State.class_masses`), then every product layer group chains its children's blocks
(:mod:`.prod`) and every sum layer group sums its children column by column (:mod:`.sum`).
The root keeps a single value per sample. Like a :class:`TensorCircuit`, the pass is driven by tables: no
kernel knows the circuit's shape. A query runs the part of them its plan (:mod:`.plans`) says, and the values
no sample's evidence reaches once for all of them.

Everything the pass reads besides the buffers and the parameters is built once per backend
(:class:`~.program.Program`), from the layout's tables and the PC's own layer metadata.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..compiled.tables import _align
from ..kernels.prod import run_products as products
from ..kernels.sum import dense_sum, fused_sum, stacked_weights


#: Matrix-product precision of the kernels: "fp32" (three TF32 products, fp32-level accuracy) or "tf32".
PRECISIONS = {"fp32": "tf32x3", "tf32": "tf32"}

#: The fewest columns (batch x slots) a dense sum region needs before evidence pruning pays with more than one live
#: column (see :func:`prune_dense`).
DENSE_PRUNE_MIN_COLUMNS = 48

#: Dense node blocks over the same children multiply their stacked weights in one product (see
#: :func:`~.sum.stacked_weights`) only up to this many columns per precision (the live ones under evidence
#: pruning); wider, one product per block. From an HMM4096 sweep (4 x 1024 rows) and lifted marginals on ex1-ex4 and
#: 13.8k- / 49k-state automata: stacked wins in TF32 at every width (x1.04-1.46) but loses 1% at 49k columns; in fp32
#: it wins up to ~2k (x1.1-1.3) and loses 1-3% wider (cuBLAS's fp32 kernel for the 4096-row shape, and the log pass
#: no longer finding a block's product in L2).
STACK_MAX_COLUMNS = {"fp32": 2048, "tf32": 16384}


def prune_dense(ncols: int, num_live: int) -> bool:
    """
    Whether a dense sum region with ``ncols`` columns (batch x slots), ``num_live`` of them live under evidence
    pruning, multiplies only its live ones -- exact either way. From the library's dense products at an HMM4096 size
    (4096 x 4096 stacked weights, fp32 and tf32, 4 to 512 columns, forward and top-down): one live column (or none)
    always pays -- a matrix-vector product, x1.07-6.2; otherwise pruning pays once the product has left the
    weight-read floor (from ``DENSE_PRUNE_MIN_COLUMNS`` columns) and at least half the columns are dead (x1.2-2.4
    with a quarter live), while below, the gather and scatter cost more than they save (x0.74-0.98). Reading
    ``num_live`` is the query's one wait for the device (see :class:`~.live.Pruning`).
    """
    return num_live < ncols and (num_live <= 1 or (ncols >= DENSE_PRUNE_MIN_COLUMNS and 2 * num_live <= ncols))


def missing_tensor(missing_mask: Optional[torch.Tensor], B: int, n: int, dev) -> torch.Tensor:
    """The forward pass's ``missing`` input, [B, n] bool on ``dev``, from a query's ``missing_mask`` (None, [n] or
    [B, n])."""
    if missing_mask is None:
        return torch.zeros(B, n, dtype = torch.bool, device = dev)
    missing = missing_mask.to(dev).bool()
    if missing.dim() == 1:
        missing = missing[None, :].expand(B, n)
    if missing.shape != (B, n):
        raise ValueError(f"`missing_mask` must be [{n}] or [{B}, {n}], got {list(missing_mask.shape)}.")
    return missing.contiguous()


def run_forward(backend, bufs: dict, plan, x: torch.Tensor, missing: torch.Tensor, obs_class: torch.Tensor, pruning,
                precision: str):
    """
    The inside values a query plan computes (see :mod:`.plans`): the per-sample ones into ``node_mars``, and
    those kept once that ``prior_mars`` does not hold yet, for a batch of one, into ``prior_mars``. Layer by layer, as
    :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of every observed token, then
    every product layer and every sum layer run their parts of the plan.

    :param bufs: the batch's :meth:`~.state.State.buffers`
    :param x: [B, n] token ids (any valid id where missing)
    :param missing: [B, n] bool, True where a token is missing
    :param obs_class: [B, n] int32, every observed token's class, -1 where missing
    :param pruning: None, or the query's :class:`~.live.Pruning`
    :param precision: "fp32" or "tf32" for the sum layers' matrix products
    :returns: the class masses the pass read (:meth:`~.state.State.class_masses`; computed
        for this query outside :func:`pyjuice.fast_inference`), for the query's later passes
    """
    pc, prog = backend.pc, backend.program()
    B = x.size(0)
    lay = bufs["layout"]
    node_mars, element_mars = bufs["node_mars"], bufs["element_mars"]
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots)
    prior = backend.state.prior_buffer() if plan.prior else None
    if prior is not None:                                              # where the values kept once are
        once = dict(offset = prior["offset_t"], width = prior["width_t"], first = prog.reg_first, slots = prog.reg_slots)
        srcs = dict(mars = prior["mars"], offset = prior["offset_t"], width = prior["width_t"], reg_prior = plan.reg_prior)
    else:                                                              # none: never read
        srcs = dict(mars = node_mars, offset = regions["offset"], width = regions["width"], reg_prior = plan.reg_prior)
    dot = PRECISIONS[precision]
    live = pruning.live if pruning is not None else None

    for layer in pc.input_layer_group:                                 # the input region: pyjuice's own layout
        layer(x.permute(1, 0), bufs["input_mars"], missing_mask = missing)
    class_mars = backend.state.class_masses()
    prog.transitions(class_mars, backend.state.class_mars_version)
    stacks, kept = backend.state.weight_stacks()

    for kind, prod_index, layers in plan.steps:
        elem_first = backend.element_regions[prod_index][0]
        elem_width = lay["element_widths"][prod_index]
        if kind == "prod":
            for prior_stages, sample_stages, num_temps, temp_slots in layers:
                temp, temp_width = node_mars, 0                      # no scratch rows: never read
                if num_temps > 0:
                    temp_width = _align(B * temp_slots)
                    temp = backend.state.storage_view("product_scratch", num_temps * temp_width)
                if prior_stages:                                     # nothing observed in their scopes
                    products(prior_stages, (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             prog.no_evidence, None, elem_first, elem_width, 1, srcs, exact = True)
                if sample_stages:
                    products(sample_stages, (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             obs_class, live, elem_first, elem_width, B, srcs, exact = pruning is not None)
        else:
            for block_size, prior_parts, sample_parts in layers:
                if prior_parts:
                    _sums(backend, prior_parts, block_size, prior["mars"], element_mars, once, prior["offset"],
                          prior["width"], elem_first, elem_width, 1, None, stacks, kept, precision)
                if sample_parts:
                    _sums(backend, sample_parts, block_size, node_mars, element_mars, regions, lay["sum_offsets"],
                          lay["sum_widths"], elem_first, elem_width, B, pruning, stacks, kept, precision)
    return class_mars


def _sums(backend, parts, block_size: int, nodes: torch.Tensor, element_mars: torch.Tensor, regions: dict, offsets,
          widths, elem_first: int, elem_width: int, B: int, pruning, stacks: dict, kept: bool, precision: str):
    """A sum layer's parts a query plan runs (``(dense runs, num_edges, fused)``: the runs ``(region, c0, blocks,
    key)``, and None or ``(fused part, index list of its node blocks or None for all, count, most slots)``, see
    :func:`~.plans.query_plan`) into ``nodes`` -- ``node_mars``, or ``prior_mars`` with ``B`` 1 and no pruning --
    whose sum regions ``regions`` (device tensors) and ``offsets`` / ``widths`` (Python ints) describe."""
    prog, pc = backend.program(), backend.pc
    dot = PRECISIONS[precision]
    for runs, num_edges, fused in parts:
        for r, c0, blocks, key in runs:
            region = (offsets[r], widths[r], prog.firsts[r])
            ncols = B * prog.slots[r]
            cols = None                                                  # every column live
            if pruning is not None and prune_dense(ncols, pruning.num_live[prog.reg_iv_host[r]]):
                cols = pruning.segment(prog.reg_iv_host[r], ncols)
            stacked = None                                  # stacked if narrow enough and the copy is kept or reused
            width = cols[2] if cols is not None else ncols                                # the product's columns
            if key is not None and width <= STACK_MAX_COLUMNS[precision] and (kept or prog.stack_uses[key] > 1):
                if key not in stacks:
                    stacks[key] = stacked_weights(pc.params, key[2], num_edges, block_size)
                stacked = {c0: stacks[key]}
            dense_sum(nodes, element_mars, pc.params, {c0: blocks}, num_edges, block_size, region, elem_first,
                      elem_width, ncols, dot, live = cols, stacked = stacked)
        if fused is not None:
            (nids, cids, pids, nb_reg, _, aliased), sel, count, max_slots = fused
            fused_sum(nodes, element_mars, pc.params, nids, cids, pids, nb_reg, regions, elem_first, elem_width, B,
                      block_size, max_cols = B * max_slots, precision = dot, aliased = aliased, pruning = pruning,
                      sel = None if sel is None else sel.tensor, count = count)
