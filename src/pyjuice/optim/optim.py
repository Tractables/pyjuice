import torch
from typing import Optional

from pyjuice.model import TensorCircuit


class CircuitOptimizer():
    """
    Base class for PC parameter optimizers.

    A PC optimizer drives Expectation-Maximization (EM) training of a :class:`TensorCircuit`. Unlike
    a :class:`torch.optim.Optimizer`, it does NOT operate on ``nn.Parameter`` gradients: the "gradient"
    of a PC is its set of *parameter flows*, accumulated into ``pc.param_flows`` (and each input
    layer's ``param_flows``) by the backward pass. An optimizer's :func:`step` consumes those flows to
    perform an EM update of ``pc.params``.

    The intended loop keeps forward/backward in your own code and lets the optimizer handle the update::

        opt = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.01)
        for x in loader:
            lls = pc(x)
            lls.mean().backward()   # accumulates flows into pc.param_flows
            opt.step()              # EM update, then resets the flow accumulator

    :func:`step` resets the flow accumulator after each update, so you never need to call
    :func:`zero_flows` manually in the common case. Concrete optimizers (:class:`FullBatchEM`,
    :class:`MiniBatchEM`, :class:`Anemone`) live in their own modules and define how the accumulated
    flows are turned into a parameter update.

    :param pc: the PC to optimize
    :type pc: TensorCircuit

    :param pseudocount: Laplace-smoothing pseudocount added to the parameter flows during the update
    :type pseudocount: float

    :param keep_zero_params: if ``True``, parameters that are exactly zero stay zero (no pseudocount)
    :type keep_zero_params: bool

    :param ddp: if ``True``, all-reduce the parameter flows across the ``torch.distributed`` process
        group before every update (via :func:`TensorCircuit.sync_param_flows`). No-op when distributed
        is not initialized / world size is 1.
    :type ddp: bool

    :param ddp_dtype: optional reduce dtype for the DDP all-reduce (e.g. ``torch.bfloat16`` to halve
        communication on bandwidth-bound interconnects); the stored flows stay float32.
    :type ddp_dtype: Optional[torch.dtype]

    :param ddp_group: optional ``torch.distributed`` process group for the all-reduce
    """

    def __init__(self, pc: TensorCircuit, pseudocount: float = 0.0, keep_zero_params: bool = False,
                 ddp: bool = False, ddp_dtype: Optional[torch.dtype] = None, ddp_group = None):

        self.pc = pc
        self.pseudocount = pseudocount
        self.keep_zero_params = keep_zero_params

        self.ddp = ddp
        self.ddp_dtype = ddp_dtype
        self.ddp_group = ddp_group

        # The backward pass should ACCUMULATE flows across the minibatches of an update window; the
        # optimizer resets the accumulator itself after each update.
        self.pc._optim_hyperparams["flows_memory"] = 1.0

        # Start from a clean flow accumulator.
        self.zero_flows()

    def zero_flows(self):
        """
        Reset the parameter-flow accumulator (``pc.param_flows`` and every input layer's
        ``param_flows``). Called automatically at the end of every :func:`step`, so it is rarely
        needed explicitly.
        """
        self.pc.init_param_flows(flows_memory = 0.0)

    def _sync_flows(self):
        # All-reduce the accumulated flows (and the EM normalizer ``_cum_flow``) across DDP ranks.
        if self.ddp:
            self.pc.sync_param_flows(dtype = self.ddp_dtype, group = self.ddp_group)

    def step(self, step_size: Optional[float] = None):
        """
        Consume the accumulated parameter flows to perform one EM update, then reset the accumulator.

        :param step_size: if given, overrides the optimizer's default step size for this step only;
            pass a per-step value here to reproduce a learning-rate schedule without a scheduler.
        :type step_size: Optional[float]
        """
        raise NotImplementedError("`step` must be implemented by a CircuitOptimizer subclass.")
