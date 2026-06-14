pyjuice.TensorCircuit
=====================

.. autoclass:: pyjuice.TensorCircuit

Inference
---------

.. automethod:: pyjuice.TensorCircuit.forward
.. automethod:: pyjuice.TensorCircuit.backward
.. automethod:: pyjuice.TensorCircuit.set_propagation_alg

Learning
--------

.. automethod:: pyjuice.TensorCircuit.mini_batch_em
.. automethod:: pyjuice.TensorCircuit.init_param_flows
.. automethod:: pyjuice.TensorCircuit.zero_param_flows
.. automethod:: pyjuice.TensorCircuit.cumulate_flows
.. automethod:: pyjuice.TensorCircuit.update_parameters
.. automethod:: pyjuice.TensorCircuit.update_param_flows

Inspection
----------

.. automethod:: pyjuice.TensorCircuit.get_node_mars
.. automethod:: pyjuice.TensorCircuit.get_node_flows
.. automethod:: pyjuice.TensorCircuit.get_node_params
.. automethod:: pyjuice.TensorCircuit.get_node_param_flows
.. automethod:: pyjuice.TensorCircuit.print_statistics

Partial Evaluation
------------------

.. automethod:: pyjuice.TensorCircuit.enable_partial_evaluation
.. automethod:: pyjuice.TensorCircuit.disable_partial_evaluation
