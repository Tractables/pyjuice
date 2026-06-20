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

    :param sync_every: DDP synchronization cadence, in EM updates. ``1`` (default) reduces the parameter
        flows every update -- exact synchronous DDP. ``> 1`` runs ``sync_every`` *local* EM updates per
        rank (each on its own data shard, no flow reduction) and then averages the *parameters* across
        ranks (Local-SGD): the all-reduce happens ``sync_every`` times less often. Only meaningful when
        ``ddp = True``. NOTE: ``sync_every > 1`` is a different optimizer than the synchronous one
        (averaging params after local updates is not the same as one update on averaged flows), so its
        convergence should be validated.
    :type sync_every: int
    """

    def __init__(self, pc: TensorCircuit, pseudocount: float = 0.0, keep_zero_params: bool = False,
                 ddp: bool = False, ddp_dtype: Optional[torch.dtype] = None, ddp_group = None,
                 sync_every: int = 1):

        assert sync_every >= 1, "`sync_every` should be a positive integer."

        self.pc = pc
        self.pseudocount = pseudocount
        self.keep_zero_params = keep_zero_params

        self.ddp = ddp
        self.ddp_dtype = ddp_dtype
        self.ddp_group = ddp_group

        self.sync_every = sync_every
        self._update_count = 0   # number of EM updates performed (drives the Local-SGD sync cadence)

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

    def _average_params(self):
        # Local-SGD: replace each rank's params by their arithmetic mean across the process group. The
        # mean of normalized PC params is itself normalized (per sum node the children sum to 1 on every
        # rank, so their mean sums to 1), so no renormalization is needed. No-op without DDP.
        if not self.ddp:
            return
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        ws = dist.get_world_size(self.ddp_group)
        if ws <= 1:
            return

        tensors = [self.pc.params]
        for layer in self.pc.input_layer_group:
            p = getattr(layer, "params", None)
            if p is not None:
                tensors.append(p)

        with torch.no_grad():
            for p in tensors:
                if self.ddp_dtype is not None and p.dtype != self.ddp_dtype:
                    buf = p.to(self.ddp_dtype)
                    dist.all_reduce(buf, op = dist.ReduceOp.SUM, group = self.ddp_group)
                    p.copy_(buf)
                else:
                    dist.all_reduce(p, op = dist.ReduceOp.SUM, group = self.ddp_group)
                p.div_(ws)

    def _post_update_sync(self):
        # Called by each subclass's `step` AFTER the local EM update. In Local-SGD mode
        # (``sync_every > 1``) the params are averaged across ranks every ``sync_every`` updates;
        # otherwise nothing happens here (the flows were already reduced before the update).
        self._update_count += 1
        if self.sync_every > 1 and self._update_count % self.sync_every == 0:
            self._average_params()

    def step(self, step_size: Optional[float] = None):
        """
        Consume the accumulated parameter flows to perform one EM update, then reset the accumulator.

        :param step_size: if given, overrides the optimizer's default step size for this step only;
            pass a per-step value here to reproduce a learning-rate schedule without a scheduler.
        :type step_size: Optional[float]
        """
        raise NotImplementedError("`step` must be implemented by a CircuitOptimizer subclass.")
