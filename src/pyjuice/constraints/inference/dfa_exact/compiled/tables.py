"""
The PC's side of the lifted values: the rows and boundaries the lifted passes read for every node of the circuit
(:func:`build_pc_tables`), and where every region of the lifted buffers sits for a batch (:func:`buffer_layout`). Both
depend only on the PC's structure and the constraint's layout (:mod:`.layout`).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .layout import BoundaryLayout


# -------------------------------------------------------------------------------------------------
# The PC side
# -------------------------------------------------------------------------------------------------

def build_pc_tables(structure, layout: BoundaryLayout, pc) -> Dict[str, Any]:
    """
    Everything the lifted forward reads about the circuit, as int32 tensors on the PC's device.

    A node over the scope ``[a, b]`` keeps, for every sample, a block of (columns at boundary ``a``) x
    (columns at boundary ``b + 1``) values, where a sequence end counts as one column: the entry of a
    prefix is the initial state alone, and the exit of a suffix is summed over the accepting states. Only
    the pairs the automaton joins are stored -- the block's slots (:class:`Reachability`).
    Only sum and product nodes keep blocks in the PC's buffers: in pyjuice a sum's children are always
    products (`SumNodes` puts a one-child product above an input child), so input nodes are only ever
    read by products.

    Rows are taken from the PC's own layers (every layer numbers its node groups consecutively, in the
    order of its `nodes` list) and checked against the node groups' record of them, which describes the
    circuit their graph was compiled into LAST: a graph compiled again into another circuit is refused
    rather than read with that circuit's rows.

    :param structure: the PC's :class:`~pyjuice.constraints.structure.PCStructure`
    :param layout: the constraint's :class:`BoundaryLayout` over the PC's variables
    :param pc: the compiled PC

    :returns: a dict with

        * ``columns_per_sample`` -- S, the most slots any node keeps per sample;
        * ``product_rows`` -- one ``(out_rows [R], child_rows [R, k], boundaries [R, k + 1])`` per product
          layer, in ``pc.inner_layer_groups`` order: every product node's element row, its children's node
          rows in scope order, and the boundaries before, between and after them (``-1`` padding for rows
          with fewer than ``k`` children). Nothing about the circuit's shape is singled out: the lifted
          product kernels read every product from this one table;
        * ``input_range`` -- the ``node_mars`` rows ``(first, last + 1)`` of all input nodes;
        * ``root_rows`` -- the ``node_mars`` rows ``(first, last + 1)`` of the root nodes;
        * ``sum_regions`` -- one ``(first_row, end_row, slots)`` per sum node group, in row order: its
          ``node_mars`` rows and the slots its block takes per sample;
        * ``element_regions`` -- one ``(first_row, end_row, slots)`` per product layer group, in
          ``pc.inner_layer_groups`` order: its ``element_mars`` rows and the most slots any of its
          products takes per sample.

        :func:`buffer_layout` turns the regions into offsets and widths for a batch size.
    """
    n = structure.num_vars
    info_of = {info.ns: info for info in structure.nodes}
    reach = layout.reachability

    def block(a, b):
        return reach.slots(a, b + 1)

    # rows of every node group, from the layers (node_mars for inputs and sums, element_mars for products)
    first_row, stale = {}, []

    def number(layer, row):
        for ns in layer.nodes:
            first_row[ns] = row
            if tuple(getattr(ns, "_output_ind_range", ())) != (row, row + ns.num_nodes):
                stale.append(ns)
            row += ns.num_nodes

    for layer in pc.input_layer_group:
        number(layer, layer._output_ind_range[0])
    for lg in pc.inner_layer_groups:
        for layer in lg.layers:
            number(layer, layer._layer_nid_range[0])
    if stale:
        from ....errors import ConstraintCompileError
        raise ConstraintCompileError(
            f"The PC's node groups no longer describe this circuit: {len(stale)} of them record other rows than "
            f"the circuit's layers. This happens when the same node graph has been compiled again into "
            f"another circuit. Compile the constraint against the circuit compiled last, or compile a fresh "
            f"copy of the nodes.")

    S = 1
    for info in structure.nodes:
        if info.kind != "input":
            (a, b), = info.scope_runs
            S = max(S, block(a, b))

    dev = pc.device
    product_rows = []
    for lg in pc.inner_layer_groups:
        if not lg.is_prod():
            continue
        for layer in lg.layers:
            parts = []
            for ns in layer.nodes:
                (a, b), = info_of[ns].scope_runs
                order = sorted(range(len(ns.chs)), key = lambda k: info_of[ns.chs[k]].scope_runs[0][0])
                chs = [ns.chs[k] for k in order]
                starts = [info_of[cs].scope_runs[0][0] for cs in chs]
                child_rows = torch.stack([first_row[cs] + _product_child_index(ns, k) for cs, k in zip(chs, order)],
                                         dim = 1)                                       # [num_nodes, k]
                bounds = torch.tensor(starts + [b + 1]).expand(ns.num_nodes, -1)
                out = first_row[ns] + torch.arange(ns.num_nodes)
                parts.append((out, child_rows, bounds))
            product_rows.append(_concat_padded(parts, dev))

    # buffer regions: every sum node group keeps exactly its own block per sample; a product layer group
    # (scratch, consumed by the next sum layer) keeps its largest product's block
    slots = lambda ns: block(*info_of[ns].scope_runs[0])
    sum_regions = sorted((first_row[ns], first_row[ns] + ns.num_nodes, slots(ns))
                         for lg in pc.inner_layer_groups if lg.is_sum() for layer in lg.layers for ns in layer.nodes)
    element_regions = []
    for lg in pc.inner_layer_groups:
        if lg.is_prod():
            rows = [layer._layer_nid_range for layer in lg.layers]
            element_regions.append((min(r[0] for r in rows), max(r[1] for r in rows),
                                    max(slots(ns) for layer in lg.layers for ns in layer.nodes)))

    ranges = [layer._output_ind_range for layer in pc.input_layer_group]
    return dict(columns_per_sample = S, product_rows = product_rows,
                input_range = (min(r[0] for r in ranges), max(r[1] for r in ranges)),
                root_rows = tuple(pc._root_node_range),
                sum_regions = tuple(sum_regions), element_regions = tuple(element_regions))


#: Every region starts, and every sum and product row is padded, to a multiple of this many floats (64 bytes),
#: which the kernels need to run with the column count as a run-time argument at full speed.
ALIGN = 16


def _align(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


def buffer_layout(input_range, sum_regions, element_regions, batch_size: int) -> Dict[str, Any]:
    """
    Where every region of the lifted buffers sits for a batch of ``batch_size`` samples (in floats).

    ``node_mars`` holds, one region after the other, each starting on an :data:`ALIGN` boundary:

    * the input region, at offset 0: rows ``0 .. input_end`` (pyjuice's own row numbers, so that an input layer
      writes it as it writes a :class:`TensorCircuit`'s ``node_mars``, and a kernel reads input row ``u`` of sample
      ``b`` at ``node_mars[u * batch_size + b]``), ``batch_size`` columns, the log-probability of every observed
      token;
    * one region per sum node group: its rows, ``align(batch_size * slots)`` columns, sample-major (slot ``s``
      of sample ``b`` in column ``b * slots + s``: every sample's block is contiguous).

    ``element_mars`` holds the products of one product layer group at a time, ``align(batch_size * slots)``
    columns per row; its size is that of the largest group.

    :returns: a dict with ``input_offset``, ``sum_offsets`` and ``sum_widths`` (one per sum
        region; row ``r`` of region ``i`` starts at ``sum_offsets[i] + (r - first_row) * sum_widths[i]``),
        ``node_size``, ``element_widths`` (one per product layer group; row ``r`` starts at
        ``(r - first_row) * width``) and ``element_size``
    """
    B = int(batch_size)
    input_start, input_end = input_range
    offset = _align(input_end * B)
    sum_offsets, sum_widths = [], []
    for first, end, slots in sum_regions:
        width = _align(B * slots)
        sum_offsets.append(offset); sum_widths.append(width)
        offset += (end - first) * width                                       # stays aligned
    element_widths = [_align(B * slots) for _, _, slots in element_regions]
    element_size = max(((end - first) * w for (first, end, _), w in zip(element_regions, element_widths)), default = 0)
    return dict(input_offset = 0, sum_offsets = sum_offsets, sum_widths = sum_widths,
                node_size = offset, element_widths = element_widths, element_size = element_size)


def _product_child_index(ns, k) -> torch.Tensor:
    """[num_nodes] the node of child ``k`` (within that child's group) that each product node multiplies."""
    if ns.is_block_sparse():
        off = torch.arange(ns.num_nodes) % ns.block_size
        return ns.edge_ids[torch.arange(ns.num_nodes) // ns.block_size, k].cpu() * ns.chs[k].block_size + off
    return ns.edge_ids[:, k].cpu()


def _concat_padded(rows, dev):
    """Stack (out [r], child_rows [r, k_i], bounds [r, k_i + 1]) parts, padding to the largest k with -1."""
    k = max(child.size(1) for _, child, _ in rows)
    pad = lambda x, width: torch.nn.functional.pad(x, (0, width - x.size(1)), value = -1)
    out = torch.cat([o for o, _, _ in rows])
    child = torch.cat([pad(c, k) for _, c, _ in rows])
    bounds = torch.cat([pad(bd, k + 1) for _, _, bd in rows])
    return tuple(t.to(dev, torch.int32) for t in (out, child, bounds))
