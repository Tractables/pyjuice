"""
The ``dfa_exact`` backend: exact queries under a constraint with a finite-state form (an automaton over token classes in
which every accepted string has one accepting run, :meth:`~pyjuice.constraints.Constraint.automaton`), on a PC whose
node scopes are contiguous runs of positions.

THE IDEA: lift the automaton's states into the PC's node values. A node over positions ``a .. b-1`` holds, per sample, a
BLOCK: for every COLUMN ``i`` at boundary ``a`` (an automaton state before the PC reads ``x_a``) and ``j`` at boundary
``b``, the log-probability of the scope's evidence together with the automaton moving from ``i`` to ``j`` on it. Then

* an input node at position ``t`` moves column ``i`` to ``next_col[t, i, c]`` with the mass of its token's class ``c``:
  the observed token's, or every class's (the CLASS MASSES, ``log sum_{v in c} p(v)``) when the token is missing;
* a product chains its children's blocks in scope order: a log-space matrix product over the boundary two adjacent
  children share (``input @ block``, ``block @ input`` and ``block @ block`` steps, :mod:`.kernels.prod`);
* a sum adds its children's blocks slot by slot: pyjuice's sum layer, run column by column (:mod:`.kernels.sum`);
* the root, over every position, enters at the initial state and exits summed over the accepting ones: ``log p(C, e)``.

The lifted circuit is again smooth and decomposable (the PC times the automaton), so its top-down pass gives every
node's outside values, and from those of the inputs at a target position the conditionals ``log p(x_t = v | e, C)``.

WHAT MAKES IT FAST

* DFA sparsity: a boundary keeps only the states active there (reachable from the start, still able to accept); a block
  only the (entry, exit) pairs the automaton joins; a missing token's classes that lead to the same successor are
  summed once when there are many per successor; ``block @ block`` skips the tiles no pair joins
  (:mod:`.compiled.layout`, :mod:`.kernels.transitions`).
* Evidence pruning: per sample, the columns its evidence leaves possible (LIVE); the other slots hold -inf and cost
  nothing, and a dense sum multiplies the live columns alone when that pays (:mod:`.kernels.live`).
* Query plans: a query computes only the values its targets need; those no sample's evidence reaches are computed
  once for the batch, and inside :func:`pyjuice.fast_inference` once across queries; values a caching engine says the
  buffers hold (``held``) are not computed. A plan is index lists into the passes' fixed launch tables, built by numba
  planners and cached by the query's inputs (:mod:`.queries.plans`, :mod:`.queries.planners`).
* Kept across queries inside :func:`pyjuice.fast_inference`: the class masses, the values no evidence reaches, the
  stacked weights of dense sums and the token tables the conditionals score against, in bf16 if allowed
  (:mod:`.state`).

MODULES, by when they take effect

* :mod:`.backend` -- :class:`DFAExactBackend`: which pairs it serves, what it builds, its queries; :mod:`.state` -- the
  buffers its queries run in and what it keeps across them.
* ``compiled/`` -- built once per (constraint, PC): :mod:`.compiled.layout` (the automaton over the positions),
  :mod:`.compiled.tables` (the PC's rows, the buffers' layout), :mod:`.compiled.program` and
  :mod:`.compiled.top_down_program` (the launch tables of the forward and top-down passes).
* ``queries/`` -- run by every query: :mod:`.queries.marginal` and :mod:`.queries.conditional`; what a query computes,
  :mod:`.queries.plans` (built by :mod:`.queries.planners`); the passes, :mod:`.queries.forward` and
  :mod:`.queries.top_down`.
* ``kernels/`` -- the Triton kernels with their launchers: products (:mod:`.kernels.prod`, :mod:`.kernels.transitions`)
  and their top-down pass (:mod:`.kernels.prod_top_down`), sums both ways (:mod:`.kernels.sum`), evidence pruning
  (:mod:`.kernels.live`), the conditionals' class weights and scoring (:mod:`.kernels.score`), shared helpers
  (:mod:`.kernels.common`).
"""

from .backend import DFAExactBackend
