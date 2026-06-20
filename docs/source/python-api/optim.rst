pyjuice.optim
=============

.. currentmodule:: pyjuice.optim

EM-based optimizers for training PCs. Unlike a :class:`torch.optim.Optimizer`, a PC optimizer does
not operate on ``nn.Parameter`` gradients: the backward pass accumulates *parameter flows* into the
circuit, and the optimizer's ``step`` consumes those flows to perform an EM update. The training loop
keeps forward/backward in your own code::

    opt = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.01)
    for x in loader:
        lls = pc(x)
        lls.mean().backward()   # accumulates parameter flows
        opt.step()              # EM update, then resets the flow accumulator

.. autoclass:: pyjuice.optim.CircuitOptimizer

    .. automethod:: pyjuice.optim.CircuitOptimizer.zero_flows
    .. automethod:: pyjuice.optim.CircuitOptimizer.step

.. autoclass:: pyjuice.optim.FullBatchEM

    .. automethod:: pyjuice.optim.FullBatchEM.step

.. autoclass:: pyjuice.optim.MiniBatchEM

    .. automethod:: pyjuice.optim.MiniBatchEM.step

.. autoclass:: pyjuice.optim.Anemone

    .. automethod:: pyjuice.optim.Anemone.step
