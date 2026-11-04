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
    from .circuit import circuit_rows                # (it imports the kernels, which import this module)
    rows = circuit_rows(structure, pc)
    reach = layout.reachability
    return dict(columns_per_sample = max([1] + [reach.slots(a, b) for a, b in rows.node_ivs]),
                product_rows = rows.on_device(pc.device), input_range = rows.input_range, root_rows = rows.root_rows,
                sum_regions = tuple((first, end, reach.slots(a, b)) for first, end, a, b in rows.sum_groups),
                element_regions = tuple((first, end, max(reach.slots(a, b) for a, b in ivs))
                                        for first, end, ivs in rows.product_groups))


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
