import torch
from typing import Optional

from pyjuice.model import TensorCircuit
from pyjuice.model.backend.partition_grad import momentum_update

from .gradient import GradientOptimizer


class SGD(GradientOptimizer):
    """
    Stochastic gradient ascent on the PC log-likelihood, with optional momentum. Operates on the
    log-parameters (``param <- exp(log(param) + lr * grad)``) and renormalizes after every update; see
    :class:`pyjuice.optim.gradient.GradientOptimizer` for the full update pipeline.

    Usage mirrors the EM optimizers::

        opt = juice.optim.SGD(pc, lr = 0.01, momentum = 0.9)
        for x in loader:
            lls = pc(x)
            lls.mean().backward()
            opt.step()

    :param lr: learning rate
    :param momentum: momentum coefficient in ``[0, 1)``; ``0`` disables momentum
    :type momentum: float

    The remaining arguments are those of :class:`GradientOptimizer`.
    """

    def __init__(self, pc: TensorCircuit, lr: float = 0.01, momentum: float = 0.0,
                 niters_per_update: int = 1, grad_clip: float = 5.0, max_param: float = 2000.0,
                 renorm_pseudocount: float = 1e-6, ddp: bool = False,
                 ddp_dtype: Optional[torch.dtype] = None, ddp_group = None):

        assert 0.0 <= momentum < 1.0, "`momentum` should be in [0, 1)."

        super().__init__(pc, lr = lr, niters_per_update = niters_per_update, grad_clip = grad_clip,
                         max_param = max_param, renorm_pseudocount = renorm_pseudocount,
                         ddp = ddp, ddp_dtype = ddp_dtype, ddp_group = ddp_group)

        self.momentum = momentum
        self._m = None   # momentum buffers (one per gradient tensor)

    def _transform_gradient(self):
        if self.momentum <= 0.0:
            return
        if self._m is None:
            self._m = [None for _ in self._gradient_tensors()]
        momentum_update(self._gradient_tensors(), self._m, self._num_updates, self.momentum)
