"""
What :class:`~pyjuice.model.tensorcircuit.TensorCircuit` and its mixins share: the device and grad mode a pass runs
under (:func:`device_grad_controller`), the walk over the layers (:func:`layer_iterator`), and the autograd hook that
runs the backward of a forward taken under autograd (:func:`_pc_model_backward_hook`).
"""

from __future__ import annotations

import torch
from contextlib import contextmanager


def _pc_model_backward_hook(grad, pc, inputs, record_cudagraph, apply_cudagraph, propagation_alg, **kwargs):
    grad = grad.permute(1, 0)
    pc.backward(
        inputs = inputs,
        ll_weights = grad / grad.sum() * grad.size(1),
        compute_param_flows = pc._optim_hyperparams["compute_param_flows"], 
        flows_memory = pc._optim_hyperparams["flows_memory"],
        record_cudagraph = record_cudagraph,
        apply_cudagraph = apply_cudagraph,
        propagation_alg = propagation_alg,
        **kwargs
    )

    return None


def layer_iterator(pc, reverse = False, ret_layer_groups = False, ignore_input_layers = False):
    if not reverse:
        if ret_layer_groups:
            if not ignore_input_layers:
                yield pc.input_layer_group

            for layer_group in pc.inner_layer_groups:
                yield layer_group

        else:
            if not ignore_input_layers:
                for layer in pc.input_layer_group:
                    yield layer

            for layer_group in pc.inner_layer_groups:
                for layer in layer_group:
                    yield layer
    else:
        if ret_layer_groups:
            for layer_group in pc.inner_layer_groups[::-1]:
                yield layer_group

            if not ignore_input_layers:
                yield pc.input_layer_group

        else:
            for layer_group in pc.inner_layer_groups[::-1]:
                for layer in layer_group:
                    yield layer

            if not ignore_input_layers:
                for layer in pc.input_layer_group:
                    yield layer


@contextmanager
def device_grad_controller(device, no_grad = True):
    device_type = device.type
    if device_type == "cpu":
        if no_grad:
            with torch.no_grad():
                yield
        else:
            yield
    else:
        # Pass the `torch.device` straight through: it handles an index-less CUDA device
        # (`torch.device("cuda")`, `.index is None`) by selecting the current device, whereas
        # `f"cuda:{device.index}"` would build the invalid string "cuda:None".
        with torch.cuda.device(device):
            if no_grad:
                with torch.no_grad():
                    yield
            else:
                yield
