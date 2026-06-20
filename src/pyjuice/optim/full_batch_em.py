from typing import Optional

from .optim import CircuitOptimizer


class FullBatchEM(CircuitOptimizer):
    """
    Full-batch EM. Accumulate parameter flows over the *entire* dataset, then perform a single exact
    EM M-step (``step_size = 1.0``). Use one :func:`step` per epoch::

        opt = juice.optim.FullBatchEM(pc, pseudocount = 0.01)
        for epoch in range(num_epochs):
            for x in loader:
                lls = pc(x)
                lls.mean().backward()   # flows accumulate over the whole epoch
            opt.step()                  # one exact EM update, then reset

    See :class:`CircuitOptimizer` for the constructor arguments.
    """

    def step(self, step_size: Optional[float] = None):
        # Full-batch EM is an exact M-step (step_size = 1.0); `step_size` can override it if desired.
        self._sync_flows()
        ss = 1.0 if step_size is None else step_size
        self.pc.mini_batch_em(step_size = ss, pseudocount = self.pseudocount,
                              keep_zero_params = self.keep_zero_params)
        self.zero_flows()
