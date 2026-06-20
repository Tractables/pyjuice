import torch
from typing import Optional

from pyjuice.model import TensorCircuit

from .optim import CircuitOptimizer


class MiniBatchEM(CircuitOptimizer):
    """
    Mini-batch EM. Perform an EM update every ``niters_per_update`` minibatches, blending the old and
    newly-estimated parameters by ``step_size``::

        opt = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.01)
        for x in loader:
            lls = pc(x)
            lls.mean().backward()
            opt.step()

    :param step_size: EM step size in ``(0, 1]``; ``params <- (1 - step_size) * params + step_size * new_params``
    :type step_size: float

    :param niters_per_update: number of minibatches to accumulate per EM update (default 1)
    :type niters_per_update: int

    The remaining arguments are those of :class:`CircuitOptimizer`.
    """

    def __init__(self, pc: TensorCircuit, step_size: float = 0.1, niters_per_update: int = 1,
                 pseudocount: float = 0.0, keep_zero_params: bool = False,
                 ddp: bool = False, ddp_dtype: Optional[torch.dtype] = None, ddp_group = None):

        assert 0.0 < step_size <= 1.0, "`step_size` should be in (0, 1]."
        assert niters_per_update >= 1, "`niters_per_update` should be a positive integer."

        super().__init__(pc, pseudocount = pseudocount, keep_zero_params = keep_zero_params,
                         ddp = ddp, ddp_dtype = ddp_dtype, ddp_group = ddp_group)

        self.step_size = step_size
        self.niters_per_update = niters_per_update
        self._iter = 0

    def step(self, step_size: Optional[float] = None):
        self._iter += 1
        if self._iter % self.niters_per_update != 0:
            return   # still accumulating this update window

        self._sync_flows()
        ss = self.step_size if step_size is None else step_size
        self.pc.mini_batch_em(step_size = ss, pseudocount = self.pseudocount,
                              keep_zero_params = self.keep_zero_params)
        self.zero_flows()
