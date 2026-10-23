"""
The input layers' part of the queries in :mod:`pyjuice.queries`, one module per distribution, named as in
:mod:`pyjuice.nodes.distributions`. ``marginal`` and ``conditional`` run the circuit's own inner layers; at its input
layers they call the module of each layer's distribution, found by the layer's ``dist_signature``. Each module provides

* ``forward(layer, inputs, node_mars, missing_mask = None, **kwargs)``: the layer's node values under the query's data
  (``[B, num_vars]`` hard evidence, or ``[B, num_vars, num_cats]`` soft evidence where the module takes it), written
  into its rows of ``node_mars``;
* ``backward(layer, inputs, node_flows, node_mars, params = None, **kwargs) -> torch.Tensor``: the conditional
  probabilities of the query's target variables (``target_vars`` in ``kwargs``), from the flows of the layer's nodes.

A layer whose distribution has no module here is refused: the query then needs its own ``fw_input_fn`` /
``bk_input_fn``.
"""

from . import categorical, discrete_logistic, external_categorical, softevi_categorical


MODULES = {
    "Categorical": categorical,
    "DiscreteLogistic": discrete_logistic,
    "ExternProductCategorical": external_categorical,
    "ExternSoftEvidenceCategorical": softevi_categorical,
}


def forward(layer, inputs, node_mars, **kwargs) -> None:
    """The forward pass of the input layer ``layer``, by its distribution's module."""
    module = MODULES.get(layer.dist_signature)
    if module is None:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)} for the forward pass. Please implement and provide your own `fw_input_fn`.")

    module.forward(layer, inputs, node_mars, **kwargs)


def backward(layer, inputs, node_flows, node_mars, **kwargs):
    """The conditional probabilities the input layer ``layer`` gives, by its distribution's module."""
    module = MODULES.get(layer.dist_signature)
    if module is None:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)} for the backward pass. Please implement and provide your own `bk_input_fn`.")

    return module.backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
