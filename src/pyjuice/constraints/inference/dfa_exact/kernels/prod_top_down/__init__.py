"""
The lifted product layer's top-down pass (see :mod:`.queries.backend.top_down`): the top-down value of every child of
every product, from the products' own (``element_flows``, or ``temp_flows`` for a fold's scratch row) and the other
children's inside values. One kernel per contraction of :mod:`.prod`, each the transpose of its forward:

* ``copy``: the child's top-down value is the copy's;
* ``input @ block`` (``out[i, j] = sum_c mass(i, c) R[next(t, i, c), j]``): ``O_R[q, j] = sum_(i, c) -> q mass(i, c)
  O_out[i, j]``, a gather over the predecessors of ``q`` at boundary ``t`` (:func:`_input_block_top_down_kernel`);
* ``block @ input`` (``out[i, j] = sum_(q, c) -> j L[i, q] mass(q, c)``): ``O_L[i, q] = sum_c mass(q, c)
  O_out[i, next(t, q, c)]``, a gather over the successors of ``q`` (:func:`_block_input_top_down_kernel`);
* ``block @ block`` (``out = L @ R``): ``O_L = O_out @ R^T`` and ``O_R = L^T @ O_out``, log-space matrix products with
  one operand read transposed (:func:`_block_block_top_down_kernel`), skipping the (output tile, shared chunk) pairs
  the automaton cannot join and recomputing exactly the tiles whose entries the shifts flush, as the forward does.

A child's value is the log-sum over every parent product (several, in a PD) of its contribution: the tables are
CHILD-MAJOR (:class:`TopDownLaunch`), one program owning a child's tile and looping over its parents, so no
accumulation needs atomics or scratch. Across launches -- a sum node is a child of products in several layers --
the first launch to reach a child writes its value and the later ones log-add to it (``add`` per child), so the
buffers are never filled beforehand.
"""

from . import launches, input_block, block_block, walk, run
from .launches import TopDownLaunch
from .run import run_top_down
