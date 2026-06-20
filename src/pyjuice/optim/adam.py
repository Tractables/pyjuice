import torch
from typing import Optional

from pyjuice.model import TensorCircuit
from pyjuice.model.backend.partition_grad import adam_update

from .gradient import GradientOptimizer


class Adam(GradientOptimizer):
    """
    Adam optimizer for PC log-likelihood. Maintains per-parameter first/second moment estimates with
    bias correction, operates on the log-parameters, and renormalizes after every update; see
    :class:`pyjuice.optim.gradient.GradientOptimizer` for the full update pipeline.

    Usage mirrors the EM optimizers::

        opt = juice.optim.Adam(pc, lr = 0.01, beta1 = 0.9, beta2 = 0.95)
        for x in loader:
            lls = pc(x)
            lls.mean().backward()
            opt.step()

    :param lr: learning rate
    :param beta1: first-moment decay rate (default 0.9)
    :param beta2: second-moment decay rate (default 0.95)
    :param eps: numerical-stability constant (default 1e-8)

    The remaining arguments are those of :class:`GradientOptimizer`.
    """

    def __init__(self, pc: TensorCircuit, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.95,
                 eps: float = 1e-8, niters_per_update: int = 1, grad_clip: float = 5.0,
                 max_param: float = 2000.0, renorm_pseudocount: float = 1e-6, ddp: bool = False,
                 ddp_dtype: Optional[torch.dtype] = None, ddp_group = None):

        assert 0.0 <= beta1 < 1.0, "`beta1` should be in [0, 1)."
        assert 0.0 <= beta2 < 1.0, "`beta2` should be in [0, 1)."

        super().__init__(pc, lr = lr, niters_per_update = niters_per_update, grad_clip = grad_clip,
                         max_param = max_param, renorm_pseudocount = renorm_pseudocount,
                         ddp = ddp, ddp_dtype = ddp_dtype, ddp_group = ddp_group)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = None   # first-moment buffers
        self._v = None   # second-moment buffers

    def _transform_gradient(self):
        grads = self._gradient_tensors()
        if self._m is None:
            self._m = [None for _ in grads]
            self._v = [None for _ in grads]
        adam_update(grads, self._m, self._v, self._num_updates,
                    beta1 = self.beta1, beta2 = self.beta2, epsilon = self.eps)
