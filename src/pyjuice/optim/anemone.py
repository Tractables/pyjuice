import torch
from typing import Optional

from pyjuice.model import TensorCircuit

from .optim import CircuitOptimizer


class Anemone(CircuitOptimizer):
    """
    The Anemone optimizer: scaled mini-batch EM with momentum.

    It accumulates parameter flows over ``niters_per_update`` minibatches and then performs a
    *flow-rescaled* EM update (``mini_batch_em(..., step_size_rescaling = True)``, i.e. the
    "mini_em_scaled" objective that normalizes by the accumulated flow mass). When ``momentum > 0``,
    the accumulated flows are first passed through a bias-corrected exponential moving average (the
    same scheme as Adam-style momentum) before the update::

        opt = juice.optim.Anemone(pc, step_size = 0.4, momentum = 0.9,
                                  niters_per_update = 8, ddp = True)
        for x in loader:
            lls = pc(x)
            lls.mean().backward()
            opt.step()              # fires the update every `niters_per_update` minibatches

    The momentum is applied to both ``pc.param_flows`` and each input layer's ``param_flows`` as::

        f       <- (1 - momentum) * f
        buffer  <- momentum * buffer + f
        f       <- buffer / (1 - momentum ** (update_count + 1))   # bias correction

    :param step_size: EM step size in ``(0, 1]`` (used by the rescaled update)
    :type step_size: float

    :param momentum: momentum coefficient in ``[0, 1)``; ``0`` disables momentum
    :type momentum: float

    :param niters_per_update: number of minibatches to accumulate per EM update (default 1)
    :type niters_per_update: int

    The remaining arguments are those of :class:`CircuitOptimizer` (``pseudocount`` defaults to
    ``1e-6`` here, matching typical Anemone training).
    """

    def __init__(self, pc: TensorCircuit, step_size: float = 0.4, momentum: float = 0.9,
                 niters_per_update: int = 1, pseudocount: float = 1e-6, keep_zero_params: bool = False,
                 ddp: bool = False, ddp_dtype: Optional[torch.dtype] = None, ddp_group = None):

        assert 0.0 < step_size <= 1.0, "`step_size` should be in (0, 1]."
        assert 0.0 <= momentum < 1.0, "`momentum` should be in [0, 1)."
        assert niters_per_update >= 1, "`niters_per_update` should be a positive integer."

        super().__init__(pc, pseudocount = pseudocount, keep_zero_params = keep_zero_params,
                         ddp = ddp, ddp_dtype = ddp_dtype, ddp_group = ddp_group)

        self.step_size = step_size
        self.momentum = momentum
        self.niters_per_update = niters_per_update

        self._iter = 0
        self._num_updates = 0          # number of EM updates performed (for momentum bias correction)
        self._momentum_sum = None      # EMA buffer for pc.param_flows
        self._momentum_input = None    # EMA buffers for the input layers' param_flows

    def step(self, step_size: Optional[float] = None):
        self._iter += 1
        if self._iter % self.niters_per_update != 0:
            return   # still accumulating this update window

        self._sync_flows()

        if self.momentum > 0.0:
            self._apply_momentum()

        ss = self.step_size if step_size is None else step_size
        self.pc.mini_batch_em(step_size = ss, pseudocount = self.pseudocount,
                              keep_zero_params = self.keep_zero_params, step_size_rescaling = True)
        self.zero_flows()
        self._num_updates += 1

    def _apply_momentum(self):
        m = self.momentum
        bias_correction = 1.0 - m ** (self._num_updates + 1)

        with torch.no_grad():
            # Lazily allocate the EMA buffers to match the flow tensors.
            if self._momentum_sum is None:
                self._momentum_sum = torch.zeros_like(self.pc.param_flows)
            if self._momentum_input is None:
                self._momentum_input = [torch.zeros_like(layer.param_flows)
                                        for layer in self.pc.input_layer_group]

            def _ema(flows, buffer):
                flows.mul_(1.0 - m)
                buffer.mul_(m).add_(flows)
                flows.copy_(buffer).div_(bias_correction)

            _ema(self.pc.param_flows, self._momentum_sum)
            for layer, buffer in zip(self.pc.input_layer_group, self._momentum_input):
                _ema(layer.param_flows, buffer)
