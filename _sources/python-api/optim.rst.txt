pyjuice.optim
=============

.. currentmodule:: pyjuice.optim

PyTorch-style optimizer and learning-rate scheduler for training PCs. They mirror the
:class:`torch.optim` API, so a PC training loop reads like a standard PyTorch loop
(``zero_grad`` / ``step``), while optionally also updating attached neural-network parameters.

.. autoclass:: pyjuice.optim.CircuitOptimizer

    .. automethod:: pyjuice.optim.CircuitOptimizer.zero_grad
    .. automethod:: pyjuice.optim.CircuitOptimizer.step
    .. automethod:: pyjuice.optim.CircuitOptimizer.state_dict
    .. automethod:: pyjuice.optim.CircuitOptimizer.load_state_dict

.. autoclass:: pyjuice.optim.CircuitScheduler

    .. automethod:: pyjuice.optim.CircuitScheduler.step
