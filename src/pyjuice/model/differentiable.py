"""
Log-likelihoods differentiable in the per-sample tensors of a call (:class:`_DifferentiableLLs`): when a tensor handed to
`pc(...)` requires grad -- an external parameter (`external_params` / `sum_external_params`) or an input distribution's
evidence that has a gradient (`categorical_evidence_logp`, ...: a keyword whose `<name>_grad` the distribution takes) --
the forward returns log-likelihoods whose `backward` reaches them, so a network that made them trains by plain autograd:

.. code-block:: python

    ll = pc(x, external_params = {"tr": (A, B)}, categorical_evidence_logp = lp)          # A, B, lp require grad
    (ll_all - ll_vis).sum().backward()                                                    # any signs, any weights

The gradient of `sum_b w_b log p_b` for the upstream weights `w`: ONE backward of the circuit (log-space flows) at |w|,
each sample's gradients -- the per-sample ones a backward writes -- times the sign of its weight (a sample's flows all
scale with its own weight). Where the circuit's latest pass is not this call's forward (another forward or backward
ran since: the circuit keeps the node values of its latest forward only) the backward runs the forward again first. With
`external_params_rows` the samples' gradients are added into the rows of the tensors they read. The circuit's own
parameters get no gradient here (`compute_param_flows = False`): their EM statistics are a backward of their own, on
the pass they train on.

A conditional in one call -- both of its passes one batch, the samples of a pass reading the same rows of tensors:

.. code-block:: python

    rows = torch.arange(n).repeat(2)                                                      # (each pass: rows 0 .. n - 1)
    ll = pc(x.repeat(2, 1), missing_mask = torch.cat([no_mask, m]), external_params = ext, external_params_rows = rows,
            categorical_evidence_logp = lp.repeat(2, 1, 1), differentiable = True)
    (ll[:n] - ll[n:]).sum().backward()                                                    # one forward, one backward
"""

from __future__ import annotations

import torch

from pyjuice.layer.parameterized_sum_layer import EXTERNAL_PARAMS_KWARG, EXTERNAL_PARAMS_GRAD_KWARG, \
    EXTERNAL_PARAMS_ROWS_KWARG


def _evidence_names(pc) -> set:
    """The input layers' call keywords that have a gradient keyword of their own (`<name>_grad`)."""
    names = pc.__dict__.get("_diff_evidence_names")
    if names is None:
        names = set()
        for layer in pc.input_layer_group:
            kws = set(getattr(layer.dist, "call_kwargs", ()))
            names |= {k[:-len("_grad")] for k in kws if k.endswith("_grad") and k[:-len("_grad")] in kws}
        pc.__dict__["_diff_evidence_names"] = names
    return names


def differentiable_entries(pc, given: dict) -> list:
    """`[(kind, key, slot, tensor)]`: the tensors of the call's keywords `given` that require grad -- `("ext", group or
    node, slot, t)` for an external parameter, `("evi", name, None, t)` for evidence."""
    out = []
    ext = given.get(EXTERNAL_PARAMS_KWARG)
    if isinstance(ext, dict):
        for key, tensors in ext.items():
            ts = (tensors,) if torch.is_tensor(tensors) else tuple(tensors)
            for slot, t in enumerate(ts):
                if torch.is_tensor(t) and t.requires_grad:
                    out.append(("ext", key, slot, t))
    for name in _evidence_names(pc):
        t = given.get(name)
        if torch.is_tensor(t) and t.requires_grad:
            out.append(("evi", name, None, t))
    return out


def _detached(given: dict) -> dict:
    """The call's keywords with every tensor detached (the external parameters' too)."""
    out = {}
    for k, v in given.items():
        if k == EXTERNAL_PARAMS_KWARG and isinstance(v, dict):
            out[k] = {key: (t.detach() if torch.is_tensor(t) else tuple(x.detach() for x in t)) for key, t in v.items()}
        else:
            out[k] = v.detach() if torch.is_tensor(v) else v
    return out


class _DifferentiableLLs(torch.autograd.Function):
    """`lls` as an op of the call's tensors that require grad (`differentiable_entries`); see the module's notes."""

    @staticmethod
    def forward(ctx, lls, pc, inputs, given, entries, *tensors):
        ctx.pc, ctx.inputs, ctx.given, ctx.entries = pc, inputs, given, entries
        ctx.run = pc.__dict__.get("_num_runs", 0)                        # (this forward's place among the circuit's passes)
        return lls.clone()

    @staticmethod
    def backward(ctx, grad):
        pc, inputs, entries = ctx.pc, ctx.inputs, ctx.entries
        given = _detached(ctx.given)
        assert grad.dim() == 2 and grad.size(1) == 1, "Differentiable log-likelihoods: circuits with one root node."
        w = grad[:, 0].float()
        ext = given.get(EXTERNAL_PARAMS_KWARG) or {}
        rows = given.get(EXTERNAL_PARAMS_ROWS_KWARG)
        out = [torch.zeros_like(t) for _, _, _, t in entries]
        if not bool((w != 0).any()):
            return (None, None, None, None, None, *out)
        rest = {k: v for k, v in given.items() if k != EXTERNAL_PARAMS_KWARG}
        B = w.size(0)
        per_sample = lambda t: t.new_zeros((B,) + tuple(t.shape[1:])) if rows is not None else torch.zeros_like(t)
        ext_grads = {}
        for kind, key, _, _ in entries:
            if kind == "ext" and key not in ext_grads:
                ts = ext[key]
                ext_grads[key] = per_sample(ts) if torch.is_tensor(ts) else tuple(per_sample(t) for t in ts)
        evi_grads = {f"{key}_grad": torch.zeros_like(given[key]) for kind, key, _, _ in entries if kind == "evi"}
        with torch.no_grad():
            if pc.__dict__.get("_num_runs", 0) != ctx.run:
                pc(inputs, **given)                                       # (another pass ran since: this call's values)
            pc.backward(inputs, ll_weights = w.abs(), compute_param_flows = False, logspace_flows = True,
                        allow_modify_flows = False, **rest, **evi_grads,
                        **({EXTERNAL_PARAMS_GRAD_KWARG: ext_grads} if ext_grads else {}))
        sign = torch.where(w < 0, -1.0, 1.0)
        for i, (kind, key, slot, t) in enumerate(entries):
            if kind == "ext":
                g = ext_grads[key]
                g = g if torch.is_tensor(g) else g[slot]
            else:
                g = evi_grads[f"{key}_grad"]
            g = (g * sign.view(-1, *([1] * (g.dim() - 1))).to(g.dtype)).to(out[i].dtype)
            if kind == "ext" and rows is not None:
                out[i].index_add_(0, rows.to(device = g.device, dtype = torch.long), g)
            else:
                out[i].add_(g)
        return (None, None, None, None, None, *out)
