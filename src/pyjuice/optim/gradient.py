import torch
from typing import Optional

from pyjuice.model import TensorCircuit
from pyjuice.model.backend.par_update import sgd_par_update
from pyjuice.model.backend.parflow_fusing import compute_cum_par_flows
from pyjuice.model.backend.partition_grad import (
    eval_partition_grad, accum_input_tied_flows, sgd_input_layer_update
)

from .optim import CircuitOptimizer


class GradientOptimizer(CircuitOptimizer):
    """
    Base class for gradient-based PC optimizers (:class:`SGD`, :class:`Adam`).

    Whereas the EM optimizers consume the backward ``param_flows`` directly, a gradient optimizer needs
    the gradient of the *normalized* log-likelihood. Each :func:`step` (after the user's forward/backward
    that populates ``pc.param_flows`` with the data flow):

    1. all-reduces the data flow across DDP ranks (if ``ddp = True``);
    2. subtracts the partition-function flow (:func:`eval_partition_grad`) so ``pc.param_flows`` becomes
       the gradient of the normalized LL w.r.t. the log-parameters;
    3. averages the gradient over the samples seen this update window and clips it to ``+/- grad_clip``;
    4. consolidates tied-parameter gradients;
    5. transforms the gradient in place (subclass hook -- momentum / Adam moments);
    6. applies a log-space update ``param <- exp(log(param) + lr * grad)`` to sum and input layers and
       clamps the (unnormalized) parameters to ``<= max_param``;
    7. renormalizes the circuit (partition pass + a single EM M-step) so the parameters are a valid
       normalized PC again -- hence ``pc(x)`` after ``step`` gives a correct log-likelihood.

    Only categorical input layers are supported (the partition forward marginalizes an input node as
    the sum of its category parameters).

    :param lr: learning rate of the log-space parameter update
    :param niters_per_update: number of minibatches to accumulate per update (default 1)
    :param grad_clip: gradient values are clipped to ``[-grad_clip, grad_clip]`` (default 5.0)
    :param max_param: unnormalized parameters are clamped to this maximum (default 2000.0)
    :param renorm_pseudocount: pseudocount used by the renormalization EM step (default 1e-6)

    The remaining arguments (``ddp`` / ``ddp_dtype`` / ``ddp_group``) are those of :class:`CircuitOptimizer`.
    """

    def __init__(self, pc: TensorCircuit, lr: float = 0.01, niters_per_update: int = 1,
                 grad_clip: float = 5.0, max_param: float = 2000.0, renorm_pseudocount: float = 1e-6,
                 ddp: bool = False, ddp_dtype: Optional[torch.dtype] = None, ddp_group = None):

        assert lr > 0.0, "`lr` should be positive."
        assert niters_per_update >= 1, "`niters_per_update` should be a positive integer."

        for layer in pc.input_layer_group:
            assert hasattr(layer.nodes[0].dist, "num_cats"), \
                "Gradient optimizers (SGD / Adam) currently support only categorical input layers."

        super().__init__(pc, pseudocount = renorm_pseudocount, keep_zero_params = False,
                         ddp = ddp, ddp_dtype = ddp_dtype, ddp_group = ddp_group)

        self.lr = lr
        self.niters_per_update = niters_per_update
        self.grad_clip = grad_clip
        self.max_param = max_param
        self.renorm_pseudocount = renorm_pseudocount

        self._iter = 0
        self._num_updates = 0
        self._samples_consumed = 0

    def _gradient_tensors(self):
        return [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

    def _transform_gradient(self):
        # No-op for plain (no-momentum) gradient descent; subclasses override.
        pass

    def _finalize_gradient(self, samples_consumed):
        with torch.no_grad():
            self.pc.param_flows[:] /= samples_consumed
            for layer in self.pc.input_layer_group:
                layer.param_flows[:] /= samples_consumed

            # Clip the gradient.
            self.pc.param_flows.clamp_(min = -self.grad_clip, max = self.grad_clip)
            for layer in self.pc.input_layer_group:
                layer.param_flows.clamp_(min = -self.grad_clip, max = self.grad_clip)

        # Consolidate tied-parameter gradients onto their source positions.
        compute_cum_par_flows(self.pc.param_flows, self.pc.parflow_fusing_kwargs)
        for layer in self.pc.input_layer_group:
            accum_input_tied_flows(layer)

    def _clamp_params(self):
        with torch.no_grad():
            self.pc.params.clamp_(max = self.max_param)
            for layer in self.pc.input_layer_group:
                layer.params.clamp_(max = self.max_param)

    def step(self, lr: Optional[float] = None):
        self._iter += 1
        self._samples_consumed += self.pc.node_mars.size(1)
        if self._iter % self.niters_per_update != 0:
            return   # still accumulating this update window

        lr_eff = self.lr if lr is None else lr

        # (1) all-reduce the accumulated data flow across DDP ranks.
        self._sync_flows()

        # (2) subtract the partition flow -> gradient of the normalized log-likelihood.
        eval_partition_grad(self.pc, negate_pflows = True)

        # (3,4) average + clip + consolidate tied gradients.
        self._finalize_gradient(self._samples_consumed)

        # (5) momentum / Adam moments.
        self._num_updates += 1
        self._transform_gradient()

        # (6) log-space parameter update + clamp.
        sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs,
                       lr = lr_eff, keep_zero_params = False)
        for layer in self.pc.input_layer_group:
            sgd_input_layer_update(layer, lr = lr_eff)
        self._clamp_params()

        # (7) renormalize the (now unnormalized) circuit back onto the normalized manifold.
        self.zero_flows()
        eval_partition_grad(self.pc, negate_pflows = False)
        self.pc.mini_batch_em(step_size = 1.0, pseudocount = self.renorm_pseudocount)

        self.zero_flows()
        self._samples_consumed = 0
