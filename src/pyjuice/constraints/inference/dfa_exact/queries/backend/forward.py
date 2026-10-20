"""
The lifted forward pass: ``log p(C, e)`` for a batch of evidence, in the backend's buffers (:class:`~.state.State`).

Layer by layer, as :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of
every observed token, the class masses of missing ones are summed per token class (by the backend's state, see
:meth:`~.state.State.masses`; per sample under a decoder session's soft evidence), then every product layer group
chains its children's blocks (:mod:`.prod`) and every sum layer group sums its children column by column
(:mod:`.sum`).
The root keeps a single value per sample. Like a :class:`TensorCircuit`, the pass is driven by tables: no
kernel knows the circuit's shape. A query runs the part of them its plan (:mod:`.plans`) says, and the values
no sample's evidence reaches once for all of them.

Everything the pass reads besides the buffers and the parameters is built once per backend
(:class:`~.program.Program`), from the layout's tables and the PC's own layer metadata.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .....distributions import lookup
from ...compiled.tables import _align
from ...kernels.prod import run_products as products
from ...external_params import block_scale
from ...kernels.compact import fresh_counts, fresh_order
from ...kernels.sum import dense_sum, fused_sum, stacked_weights
from .plans import kept_plan


#: Matrix-product precision of the kernels per query precision: "fp32" (three TF32 products, fp32-level accuracy),
#: "tf32", or "bf16" -- the CUDA fused sums' and the dense sums' live-column products on bf16 operands
#: (:func:`~...kernels.sum.fused_sum`, :func:`~...kernels.sum.dense_sum`), every other kernel's as at "tf32".
PRECISIONS = {"fp32": "tf32x3", "tf32": "tf32", "bf16": "tf32"}

#: The fewest columns (batch x slots) a dense sum region needs before evidence pruning pays with more than one live
#: column (see :func:`prune_dense`).
DENSE_PRUNE_MIN_COLUMNS = 48

#: Dense node blocks over the same children multiply their stacked weights in one product (see
#: :func:`~.sum.stacked_weights`) only up to this many columns per precision (the live ones under evidence
#: pruning); wider, one product per block. From an HMM4096 sweep (4 x 1024 rows) and lifted marginals on ex1-ex4 and
#: 13.8k- / 49k-state automata: stacked wins in TF32 at every width (x1.04-1.46) but loses 1% at 49k columns; in fp32
#: it wins up to ~2k (x1.1-1.3) and loses 1-3% wider (cuBLAS's fp32 kernel for the 4096-row shape, and the log pass
#: no longer finding a block's product in L2).
STACK_MAX_COLUMNS = {"fp32": 2048, "tf32": 16384, "bf16": 16384}


def prune_dense(ncols: int, num_live: int) -> bool:
    """
    Whether a dense sum region with ``ncols`` columns (batch x slots), ``num_live`` of them live under evidence
    pruning, multiplies only its live ones -- exact either way. From the library's dense products at an HMM4096 size
    (4096 x 4096 stacked weights, fp32 and tf32, 4 to 512 columns, forward and top-down): one live column (or none)
    always pays -- a matrix-vector product, x1.07-6.2; otherwise pruning pays once the product has left the
    weight-read floor (from ``DENSE_PRUNE_MIN_COLUMNS`` columns) and at least half the columns are dead (x1.2-2.4
    with a quarter live), while below, the gather and scatter cost more than they save (x0.74-0.98). Reading
    ``num_live`` waits for the device (see :meth:`~.live.LaneOrder.num_live`).
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


def gate_context(backend, external) -> Optional[dict]:
    """The gates a query's gated sum layers read (:mod:`...external_params.block_scale`): None
    without, else ``values`` and ``lz`` (the session's log gates and log normalizers), ``minor`` (the log gates
    sample-minor, :func:`~...external_params.block_scale.sample_minor`, for the CUDA sums) and ``layers`` (bool per
    gated layer: those given gates)."""
    gates = None if external is None else external.gates
    if gates is None:
        return None
    lz = block_scale.log_normalizers(backend.pc, backend.program().gate_cells, gates)
    return dict(values = gates.values, lz = lz, minor = block_scale.sample_minor(gates), layers = gates.layers)


def part_gates(gates: Optional[dict], gate: Optional[dict], compact, prog, once: bool) -> Optional[dict]:
    """A fused sum part's gates for :func:`~...kernels.sum.fused_sum` / :func:`~...kernels.sum.fused_sum_top_down` under
    a query's ``gates`` (:func:`gate_context`), or None when its layer is not gated: ``gate`` the part's tables
    (``block_scale.edge_tables``); ``compact`` the query's :class:`~...kernels.compact.CompactLayout` if the part's rows
    are compact (a lane's sample found in its offsets); ``once``: the part computes values kept once, for a batch of
    one -- so the gates are every sample's."""
    if gates is None or gate is None or not gates["layers"][gate["layer"]]:
        return None
    shared = gates["values"].size(0) == 1
    if once and not shared:
        raise RuntimeError("Internal error: values kept once under gates that differ per sample.")
    if max(gates["values"].numel(), gates["lz"].numel()) >= 1 << 31:          # the kernels address them in int32
        raise NotImplementedError("Gates of 2^31 floats and more (a batch's of every gate or node) are not supported.")
    lanes = None
    if compact is not None:
        if not hasattr(prog, "reg_iv_all_t"):
            prog.reg_iv_all_t = torch.from_numpy(prog.reg_iv_all).to(prog.dev)
        lanes = (compact.off, prog.reg_iv_all_t, compact.B, compact.steps)
    return dict(values = gates["values"], gstride = 0 if shared else gates["values"].size(1), lz = gates["lz"],
                lzstride = 0 if shared else gates["lz"].size(1), minor = gates["minor"],
                mstride = 0 if shared else gates["minor"].size(1), lanes = lanes, idx = gate.get("idx"),
                a_idx = gate.get("a_idx"), uniform = gate["uniform"], a_uniform = gate.get("a_uniform", {}),
                ck = gate.get("ck"), node_gate = gate["node_gate"])


def run_forward(backend, bufs: dict, plan, x: torch.Tensor, missing: torch.Tensor, obs_class: torch.Tensor, liveness,
                precision: str, compact = None, masses = None, gates: Optional[dict] = None,
                hold: Optional[dict] = None):
    """
    The inside values a query plan computes (see :mod:`.plans`): the per-sample ones into ``node_mars``, and
    those kept once that ``prior_mars`` does not hold yet, for a batch of one, into ``prior_mars``. Layer by layer, as
    :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of every observed token (or
    their distributions' modules, ``input_values``, see :mod:`pyjuice.constraints.distributions`), then every product
    layer and every sum layer run their parts of the plan.

    :param bufs: the batch's :meth:`~.state.State.buffers`
    :param x: [B, n] token ids (any valid id where missing)
    :param missing: [B, n] bool, True where a token is missing
    :param obs_class: [B, n] int32, every observed token's class, -1 where missing
    :param liveness: None, or the query's :class:`~.live.Liveness`, the intervals it computes per sample ordered (unless
        they are compact)
    :param precision: "fp32", "tf32" or "bf16" for the sum layers' matrix products (see :data:`PRECISIONS`)
    :param compact: None, or the query's :class:`~...kernels.compact.CompactLayout`: the per-sample inside values are
        compact where it lays them out (``mars``; ``bufs`` are laid out by it, see :meth:`~.state.State.buffers`)
    :param masses: the query's class masses (:meth:`~.state.State.masses`; None: the state's, under no per-sample
        inputs)
    :param gates: None, or the gates of the query's gated sum layers (:func:`gate_context`)
    :param hold: None, or (compact) the regions some rows hold the inside values of (``regions``, bool [sum regions],
        re-gathered into ``node_mars`` already, see :func:`~...kernels.compact.partial_carry`) and per region the rows
        that compute theirs (``fresh``, bool [sum regions, B]): their sums compute those rows' lanes alone (the dense
        runs multiply them, the fused sums take them as their columns, :func:`~...kernels.compact.fresh_order`), and so
        do the products whose values those sums alone read (:func:`~...kernels.compact.fresh_counts`)
    :returns: the class masses the pass read (computed for this query outside :func:`pyjuice.fast_inference`), for the
        query's later passes
    """
    pc, prog = backend.pc, backend.program()
    B = x.size(0)
    lay = bufs["layout"]
    node_mars, element_mars = bufs["node_mars"], bufs["element_mars"]
    compact = compact if compact is not None and compact.mars is not None else None
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots if compact is None else compact.lanes_t)   # compact: B = 1, slots = lanes
    prior = backend.state.prior_buffer(plan) if plan.prior else None     # (extents for the values it computes)
    if prior is not None:                                              # where the values kept once are
        once = dict(offset = prior["offset_t"], width = prior["width_t"], first = prog.reg_first, slots = prog.reg_slots)
        srcs = dict(mars = prior["mars"], offset = prior["offset_t"], width = prior["width_t"], reg_prior = plan.reg_prior)
    else:                                                              # none: never read
        srcs = dict(mars = node_mars, offset = regions["offset"], width = regions["width"], reg_prior = plan.reg_prior)
    dot = PRECISIONS[precision]
    live = liveness.live if liveness is not None else None
    succ_lists = None                                                  # entry loops under pruning: a column's entries
    if liveness is not None and (prog.trans.grouped or prog.entry_loop):         # whose successor is live, where few are
        succ_lists = lambda: liveness.successor_lists(backend.state)   # (built by the first launch that can take them)

    if hold is not None and compact is not None:       # the fused sums' order of the rows computing their values,
        built = {}                                      # built by the first launch reading it

        def order():
            if "order" not in built:
                built["order"] = fresh_order(compact, hold["regions"], hold["fresh"])
            return built["order"]

        def counts():
            if "counts" not in built:
                built["counts"] = fresh_counts(compact, hold["regions"], hold["fresh"])
            return built["counts"]

        hold = dict(hold, ivs = np.unique(prog.reg_iv_all[hold["regions"]]), order = order, counts = counts)
    else:
        hold = None
    masses = backend.state.masses() if masses is None else masses
    for k, layer in enumerate(pc.input_layer_group):                   # the input region: pyjuice's own layout
        values = getattr(lookup(layer.dist), "input_values", None)
        if values is not None:
            values(layer, x, missing, bufs["input_mars"], masses.sample.get(k))
        else:
            layer(x.permute(1, 0), bufs["input_mars"], missing_mask = missing)
    prog.transitions(masses)
    stacks, kept = backend.state.weight_stacks()

    for kind, prod_index, layers in plan.steps:
        elem_first = backend.element_regions[prod_index][0]
        elem_width = lay["element_widths"][prod_index]
        if kind == "prod":
            for prior_stages, sample_stages, num_temps, temp_slots in layers:
                temp, temp_width = node_mars, 0                      # no scratch rows: never read
                if num_temps > 0:
                    temp_width = scratch_width(B, temp_slots, sample_stages, compact)
                    temp = backend.state.storage_view("product_scratch", num_temps * temp_width)
                if prior_stages:                                     # nothing observed in their scopes
                    products(prior_stages, (temp, temp_width), prog, regions, element_mars, node_mars, masses,
                             prog.no_evidence, None, elem_first, elem_width, 1, srcs, exact = True)
                if sample_stages:
                    products(sample_stages, (temp, temp_width), prog, regions, element_mars, node_mars, masses,
                             obs_class, live, elem_first, elem_width, B, srcs, exact = liveness is not None,
                             compact = compact, succ_lists = succ_lists,
                             hold = None if hold is None else hold["counts"])
        else:
            for block_size, prior_parts, sample_parts in layers:
                if prior_parts:
                    _sums(backend, prior_parts, block_size, prior["mars"], element_mars, once, prior["offset"],
                          prior["width"], elem_first, elem_width, 1, None, stacks, kept, precision, gates = gates,
                          once = True)
                if sample_parts:
                    _sums(backend, sample_parts, block_size, node_mars, element_mars, regions, lay["sum_offsets"],
                          lay["sum_widths"], elem_first, elem_width, B, liveness, stacks, kept, precision, compact,
                          gates = gates, hold = hold if compact is not None else None)
    return masses


def kept_values(backend, plan, replan, forward, valid: torch.Tensor):
    """
    The values kept once a query reads (:attr:`~.plans.QueryPlan.prior_read`) must be held together while it runs.
    When they do not fit their pool (:class:`~...state.PriorPool`) beside its checkpoints, the ones it computes are
    computed first in pieces, the lowest depths first (:meth:`~...state.PriorPool.piece`), each by a plan of its own
    (:func:`~.plans.kept_plan`) that ``forward(plan)`` runs (:func:`run_forward` with the query's arguments), the pool
    evicting between them all but the values the query reads -- ``valid`` (the state's, see
    :meth:`~...state.State.prior_validity`) marked as they are computed; the query's plan is then made again
    (``replan()``) over the values held, and so on while its values do not fit: every piece leaves fewer to compute
    (the pieces keep what the query's closure reads). When even the lowest depth does not fit, and the query's values
    do not fit without the checkpoints either, the pool refuses the query.

    :returns: the query's plan, ``plan`` itself when its values fit
    """
    if not plan.prior:
        return plan
    pool, prog = backend.state.prior_pool(), backend.program()
    prior = plan.prior_read | plan.todo.numpy()                     # (kept once: what it reads and computes)
    while not pool.holds(plan.prior_read):
        piece = pool.piece(plan)
        if piece is None:
            if pool.holds(plan.prior_read, checkpoints = False):     # the checkpoints go
                break
            raise pool._too_small(int(pool.iv_floats[plan.prior_read].sum()))
        sub = kept_plan(prog, piece, prior | valid.numpy(), plan.prior_read & valid.numpy())
        forward(sub)
        valid |= sub.todo
        plan = replan()
    return plan


def scratch_width(B: int, temp_slots: int, sample_stages, compact) -> int:
    """The width of a product layer's scratch rows (floats): a batch's blocks of ``temp_slots`` -- or ``compact`` (a
    :class:`~...kernels.compact.CompactLayout`), the compact values of the folds ``sample_stages`` (lists of ``(launch,
    cut)``, see :mod:`...kernels.prod`) compute per sample, or one sample's block kept once."""
    if compact is None:
        return _align(B * temp_slots)
    ivs = [launch.out_ivs for launches in sample_stages for launch, _ in launches if launch.out_temp]
    return compact.row_width(np.concatenate(ivs) if ivs else np.zeros(0, dtype = np.int64), temp_slots)


def _sums(backend, parts, block_size: int, nodes: torch.Tensor, element_mars: torch.Tensor, regions: dict, offsets,
          widths, elem_first: int, elem_width: int, B: int, liveness, stacks: dict, kept: bool, precision: str,
          compact = None, gates: Optional[dict] = None, once: bool = False, hold: Optional[dict] = None):
    """A sum layer's parts a query plan runs (``(dense runs, num_edges, fused)``: the runs ``(region, c0, blocks,
    key)``, and None or ``(fused part, index list of its node blocks or None for all, count, most slots)``, see
    :func:`~.plans.query_plan`) into ``nodes`` -- ``node_mars``, or ``prior_mars`` with ``B`` 1 and no pruning
    (``once``) -- whose sum regions ``regions`` (device tensors) and ``offsets`` / ``widths`` (host, a value per
    region) describe. ``compact``: the query's :class:`~...kernels.compact.CompactLayout`, by which the rows are
    compact -- a batch of one whose slots are the lanes (``regions``' ``slots``), every lane computed. ``gates``,
    ``hold``: see :func:`run_forward`."""
    prog, pc = backend.program(), backend.pc
    dot = PRECISIONS[precision]
    for runs, num_edges, fused in parts:
        for r, c0, blocks, key in runs:
            region = (int(offsets[r]), int(widths[r]), prog.firsts[r])
            ncols = B * prog.slots[r]
            cols = None                                                  # every column live
            keep = False
            if compact is not None:                                      # a compact row's lanes, in order
                ncols = int(compact.total[prog.reg_iv_all[r]])
                if ncols == 0:
                    continue
                ident = compact.identity(ncols)
                cols = (ident, ident, ncols)
                fresh = hold["order"]().dense(int(prog.reg_iv_all[r]), ncols) \
                    if hold is not None and hold["regions"][r] else None   # the lanes of the rows computing theirs
                if fresh is not None:
                    if fresh[2] == 0:
                        continue
                    cols, keep = fresh, True
            elif liveness is not None and prune_dense(ncols, liveness.lanes.num_live(prog.reg_iv_host[r])):
                cols = liveness.lanes.segment(prog.reg_iv_host[r], ncols)
            stacked = None                                  # stacked if narrow enough and the copy is kept or reused
            width = cols[2] if cols is not None else ncols                                # the product's columns
            if key is not None and width <= STACK_MAX_COLUMNS[precision] and (kept or prog.stack_uses[key] > 1):
                if key not in stacks:
                    stacks[key] = stacked_weights(pc.params, key[2], num_edges, block_size)
                stacked = {c0: stacks[key]}
            dense_sum(nodes, element_mars, pc.params, {c0: blocks}, num_edges, block_size, region, elem_first,
                      elem_width, ncols, dot, live = cols, stacked = stacked, bf16 = precision == "bf16",
                      copies = stacks, keep = keep)
        if fused is not None:
            (nids, cids, pids, nb_reg, _, aliased, gate), sel, count, max_slots, ivs = fused
            max_cols = B * max_slots if compact is None else compact.lanes(ivs)          # (its regions' most)
            if max_cols == 0:
                continue
            order = None                                # (some rows hold their values: their lanes left as they are)
            if hold is not None and np.isin(ivs, hold["ivs"]).any():
                order = hold["order"]()
            fused_sum(nodes, element_mars, pc.params, nids, cids, pids, nb_reg,
                      regions if order is None else dict(regions, slots = order.slots), elem_first, elem_width,
                      B if compact is None else 1, block_size, max_cols = max_cols, precision = dot, aliased = aliased,
                      liveness = liveness if compact is None else order, sel = None if sel is None else sel.tensor,
                      count = count, gated = part_gates(gates, gate, compact, prog, once),
                      frags = stacks if kept or precision == "bf16" else None, bf16 = precision == "bf16")
