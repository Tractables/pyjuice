"""
The lifted product layer. A product chains its children's blocks in scope order: a log-space product over the
boundary columns that adjacent children share. There is one table-driven implementation for every circuit
shape. :class:`~pyjuice.constraints.inference.dfa_exact.compiled.program.Program` folds each product into binary steps,
right to left. Each step is one of three contractions, chosen by what its two operands are, never by the
circuit's shape:

* ``input @ block`` (:func:`_input_block_kernel`): an input at position ``t``, then a block from ``t + 1``.
  Class ``c`` of the token moves column ``i`` at ``t`` to ``next_col[t, i, c]``, so this is a gather over
  the successors of ``i``. The block may be the identity, which materializes the input's own block (a product
  with a single input child, or an input next to another input);
* ``block @ input`` (:func:`_block_input_kernel`): the mirror case, a gather over the predecessors of each
  column at ``t + 1``;
* ``block @ block`` (:func:`_block_block_kernel`): a per-sample log-space matrix product over the shared
  boundary, skipping the (output tile, shared chunk) pairs the automaton cannot join (:class:`SkipMasks`) -- or, the
  values compact (:mod:`.compact`), over the sample's live columns alone (:func:`_block_block_compact_kernel`).

Blocks are either sum nodes' rows in ``node_mars`` or intermediate results in a scratch buffer. Outputs go
to the product layer's ``element_mars`` rows, or to scratch when another step reads them. A row holds one
block per sample, sample-major: sample ``b``'s block in columns ``[b * slots, (b + 1) * slots)``. A sum node whose
scope no sample observes may keep one block for every sample instead, in ``prior_mars`` (the query plan says which,
see :mod:`.plans`). A block keeps
only the (entry column ``i``, exit column ``j``) pairs the automaton joins, in row-major order: slot ``i * X +
j`` when it joins them all, else through its interval's tables (:class:`~.dfa_exact.compiled.reachability.IntervalTables`). An input child
is read only as the log-probability of its observed token (pyjuice's own input layers wrote it) or, when the
token is missing, through its transition masses (:func:`transition_masses`): for a column and one of its
successors, the log-mass of the token classes that lead there -- so a missing token costs a gather per
successor, not per class. Under evidence pruning (``live``, see :mod:`.live`), a dead slot is -inf and costs no
work.
"""

from . import launches, input_block, block_block, skip_masks, run
from .launches import Launch, Cut
from .block_block import compact_tiles
from .skip_masks import SkipMasks, skip_masks, skip_masks_for
from .run import run_products
