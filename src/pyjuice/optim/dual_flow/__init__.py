"""
The conditional dual-flow M-step's kernels, a backend of the optimizers: `theta <- normalize(theta * F+ / F-)` for every
parameterization that requests the denominator flow (`SumParameterization.compute_em_correction` drives them).

* :mod:`.dual_em` -- the two passes (the per-node sums, the update), `F-` formed in-kernel by its `DENOM_MODE`;
* :mod:`.dual_em_master` -- the same passes for a circuit storing its parameters below fp32, its fp32 master copy's
  sync and store fused in.
"""
