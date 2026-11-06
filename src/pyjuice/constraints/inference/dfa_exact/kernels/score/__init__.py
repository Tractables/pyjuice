"""
The conditionals' kernels (see :mod:`.queries.conditional`): the class weights of the target positions' input nodes,
read off the top-down pass, and the scoring of every token from them.

Token ``v`` of class ``c`` at position ``t`` moves column ``i`` to ``next_col[t, i, c]`` with probability ``p_u(v)`` at
input node ``u``, so ``p(x_t = v, e, C) = sum_u p_u(v) A_u[c]``, with ``A_u[c] = sum_i exp(O_u[i, next_col[t, i, c]])``
the CLASS WEIGHTS of ``u``'s top-down (log outside) block ``O_u``. An input node has no block of its own in the lifted
circuit: it is an operand of a product's step (:mod:`.prod`), and its top-down block is the transpose of that step,
from the step's output's top-down value and the other operand's inside value:

* ``input @ block`` (``out[i, j] = sum_q In[i, q] R[q, j]``): ``A[c] = sum_(i, j) exp(O_out[i, j] + R[next(t, i, c), j])``;
  the identity block (a product of one input, or an input next to another): ``A[c] = sum_i exp(O_out[i, next(t, i,
  c)])``;
* ``block @ input`` (``out[i0, q] = sum_i L[i0, i] In[i, q]``): ``A[c] = sum_(i0, i) exp(L[i0, i] + O_out[i0, next(t, i,
  c)])``,

a sequence end's successors being its one column. Tied input nodes share their parameters' row: the weights are summed
per (position, row of the class masses), :data:`dfa_class_flow` -- ``[target positions x samples, classes, rows of the
position's row group]`` per row group, log -- and scored per row group against the rows' token-major probabilities:
``log p(x_t = v | e, C) = log sum_k T[v, k] A[k, class(v)] - log p(e, C)``, with ``p(e, C) = sum_(k, c) A[k, c]
mass_k(c)``: the stored classes' terms from the class masses (no second pass over the tokens), an enumerated class's
(one token, no column in the masses) from its token's unnormalized score, the dot the scoring computes anyway.
"""

from . import tables, class_weight_kernels, class_rows, class_weights, group
from .tables import ScoreTables, score_tables
from .class_weight_kernels import ClassWeightLaunch, class_weight_launch
from .class_rows import ClassWeightRows, class_weight_rows
from .class_weights import class_weight_walk, run_class_weights
from .set_weights import combine_set_weights, most_live_columns, set_weight_lists, set_weight_rows, set_weights_always, set_weights_apply
from .group import score_group
