"""
The PC's side of the lifted values: the rows and boundaries the lifted passes read for every node of the circuit
(:func:`build_pc_tables`), and where every region of the lifted buffers sits for a batch (:func:`buffer_layout`). Both
depend only on the PC's structure and the constraint's layout (:mod:`.layout`).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .layout import BoundaryLayout


# -------------------------------------------------------------------------------------------------
# The PC side
# -------------------------------------------------------------------------------------------------

class Regions:
    """The sum regions of a backend, ``(first_row, end_row, slots)`` each, as a read-only sequence of tuples (Python
    ints), and ``array``: int64 [regions, 3]. A circuit may have tens of thousands of them, and a new constraint new ones."""

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray):
        self.array = array

    def __len__(self) -> int:
        return self.array.shape[0]

    def __getitem__(self, r: int) -> tuple:
        first, end, slots = self.array[r].tolist()
        return first, end, slots

    def __iter__(self):
        return map(tuple, self.array.tolist())

    def __array__(self, dtype = None, copy = None):
        return self.array if dtype is None else self.array.astype(dtype, copy = False)


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
        * ``sum_regions`` -- one ``(first_row, end_row, slots)`` per sum node group, in row order
          (:class:`Regions`): its ``node_mars`` rows and the slots its block takes per sample;
        * ``element_regions`` -- one ``(first_row, end_row, slots)`` per product layer group, in
          ``pc.inner_layer_groups`` order: its ``element_mars`` rows and the most slots any of its
          products takes per sample.

        :func:`buffer_layout` turns the regions into offsets and widths for a batch size.
    """
    from .circuit import circuit_rows                # (it imports the kernels, which import this module)
    rows = circuit_rows(structure, pc)
    reach = layout.reachability
    reach.prepare(rows.node_ivs)
    slots = np.fromiter((reach.slots(a, b) for a, b in rows.node_ivs), dtype = np.int64, count = len(rows.node_ivs))
    return dict(columns_per_sample = max(1, int(slots.max(initial = 1))),
                product_rows = rows.on_device(pc.device), input_range = rows.input_range, root_rows = rows.root_rows,
                sum_regions = Regions(np.concatenate([rows.sum_rows, slots[rows.sum_iv][:, None]], axis = 1)),
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
    regions = np.asarray(sum_regions, dtype = np.int64).reshape(-1, 3)
    widths = (B * regions[:, 2] + ALIGN - 1) // ALIGN * ALIGN
    ends = _align(input_end * B) + np.cumsum((regions[:, 1] - regions[:, 0]) * widths)   # (stays aligned)
    node_size = int(ends[-1]) if ends.size else _align(input_end * B)
    sum_offsets = np.concatenate([[_align(input_end * B)], ends[:-1]]) if ends.size else ends
    element_widths = [_align(B * slots) for _, _, slots in element_regions]
    element_size = max(((end - first) * w for (first, end, _), w in zip(element_regions, element_widths)), default = 0)
    return dict(input_offset = 0, sum_offsets = sum_offsets.tolist(), sum_widths = widths.tolist(),
                node_size = node_size, element_widths = element_widths, element_size = element_size)
