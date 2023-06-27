from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Sequence, Dict, Optional

from pyjuice.nodes import InputNodes
from pyjuice.nodes.distributions import DiscreteLogistic
from pyjuice.layer.input_layer import InputLayer
from pyjuice.utils.grad_fns import ReverseGrad


"""
Implementation of discretized Logistic distribution (https://en.wikipedia.org/wiki/Logistic_distribution)
"""
class DiscreteLogisticLayer(InputLayer, nn.Module):
    
    def __init__(self, nodes: Sequence[InputNodes], cum_nodes: int = 0) -> None:
        nn.Module.__init__(self)
        InputLayer.__init__(self, nodes)

        # Parse input `nodes`
        self.vars = []
        self.node_sizes = []
        self.node_input_range = []
        self.node_bin_sizes = []
        layer_num_nodes = 0
        for ns in self.nodes:
            assert len(ns.scope) == 1, "DiscreteLogisticLayer only support uni-variate categorical distributions."
            assert isinstance(ns.dist, DiscreteLogistic), f"Adding a `{type(ns.dist)}` node to a `DiscreteLogistic`."

            self.vars.append(next(iter(ns.scope)))
            self.node_sizes.append(ns.num_nodes)
            self.node_input_range.append(
                [ns.dist.input_range[0], ns.dist.input_range[1]]
            )
            self.node_bin_sizes.append(ns.dist.bin_size)

            ns._output_ind_range = (cum_nodes, cum_nodes + ns.num_nodes)
            cum_nodes += ns.num_nodes
            layer_num_nodes += ns.num_nodes

        self._output_ind_range = (cum_nodes - layer_num_nodes, cum_nodes)

        self.num_nodes = layer_num_nodes

        d2vids = torch.empty([len(self.nodes)], dtype = torch.long)
        vrangeslow = torch.tensor([r[0] for r in self.node_input_range])
        vrangeshigh = torch.tensor([r[1] for r in self.node_input_range])
        vhbinsizes = torch.tensor(self.node_bin_sizes) / 2.0
        vids = torch.empty([layer_num_nodes], dtype = torch.long)
        vids2d = torch.tensor(self.vars)
        n_start = 0
        for idx, ns in enumerate(self.nodes):
            n_end = n_start + ns.num_nodes

            d2vids[idx] = self.vars[idx]
            vids[n_start:n_end] = idx

            n_start = n_end

            if ns._source_node is not None:
                raise NotImplementedError()

        self.register_buffer("d2vids", d2vids)
        self.register_buffer("vrangeslow", vrangeslow.unsqueeze(1))
        self.register_buffer("vrangeshigh", vrangeshigh.unsqueeze(1))
        self.register_buffer("vhbinsizes", vhbinsizes.unsqueeze(1))
        self.register_buffer("vids", vids)
        self.register_buffer("vids2d", vids2d)

        # Initialize parameters
        self._init_params()

        self.param_flows_size = self.mus.size(0) * 2

        # Buffers for backward pass
        self._backward_buffer = dict()

        # Batch size of parameters in the previous forward pass
        self._param_batch_size = 1

    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, params: Optional[Dict] = None, skip_logsumexp: bool = False):
        """
        data: [num_vars, B]
        node_mars: [num_nodes, B]
        """
        super(DiscreteLogisticLayer, self).forward(params is not None)

        B = data.size(1)

        if params is None:
            mus = self.mus
            log_scales = self.log_scales
        else:
            mus = params["mus"].permute(1, 0)
            log_scales = params["log_scales"].permute(1, 0)

        if mus.dim() == 1:
            mus = mus.unsqueeze(1)
            log_scales = log_scales.unsqueeze(1)

        log_scales = log_scales.clip(min = -5.0)

        self._param_batch_size = mus.size(1)

        l_boundaries = torch.empty([self.num_nodes, B], dtype = torch.float32, device = data.device)
        r_boundaries = torch.empty([self.num_nodes, B], dtype = torch.float32, device = data.device)

        if skip_logsumexp:
            raise NotImplementedError()
        else:
            self._dense_forward_pass(data, node_mars, mus, log_scales, l_boundaries, r_boundaries)

        return None

    def backward(self, data: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor, params: Optional[Dict] = None):
        """
        data: [num_vars, B]
        node_flows: [num_nodes, B]
        node_mars: [num_nodes, B]
        """
        B = data.size(1)

        if params is None:
            mus = self.mus
            log_scales = self.log_scales
        else:
            mus = params["mus"].permute(1, 0)
            log_scales = params["log_scales"].permute(1, 0)

        if mus.dim() == 1:
            mus = mus.unsqueeze(1)
            log_scales = log_scales.unsqueeze(1)
        
        self._dense_backward_pass(data, node_flows, node_mars, mus, log_scales)

        self._backward_buffer.clear()
        
        return None

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        if not self._used_external_params:
            with torch.no_grad():
                flows = self.param_flows
                if flows is None:
                    return None
                self.mus.data += step_size * flows[:self.mus.size(0)]
                self.log_scales.data += step_size * flows[self.mus.size(0):] / torch.norm(flows[self.mus.size(0):]).clip(min = 5.0) * 5.0

                self.mus.data.clamp(min = 0.0, max = 1.0)
                self.log_scales.data.clamp(min = -4.0, max = 2.0)

    def get_param_specs(self):
        return {"mus": torch.Size([self.mus.size(0)]), "log_scales": torch.Size([self.log_scales.size(0)])}

    def _dense_forward_pass(self, data: torch.Tensor, node_mars: torch.Tensor, mus: torch.Tensor, log_scales: torch.Tensor,
                            l_boundaries: torch.Tensor, r_boundaries: torch.Tensor):
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]

        scaled_data = (data[self.d2vids] - self.vrangeslow) / (self.vrangeshigh - self.vrangeslow)
        l_b = ((scaled_data[self.vids] - self.vhbinsizes[self.vids] - mus) / log_scales.exp()).detach()
        r_b = ((scaled_data[self.vids] + self.vhbinsizes[self.vids] - mus) / log_scales.exp()).detach()
        with torch.enable_grad():
            l_b.requires_grad = True
            r_b.requires_grad = True
            l_b.retain_grad()
            r_b.retain_grad()

            self._backward_buffer["l_b"] = l_b
            self._backward_buffer["r_b"] = r_b

            mars = self._log_min_exp(F.logsigmoid(r_b), F.logsigmoid(l_b))
            mars = torch.where(scaled_data[self.vids] < 0.01, F.logsigmoid(l_b), mars)
            mars = torch.where(scaled_data[self.vids] > 0.99, self._log_min_exp(0.0, F.logsigmoid(r_b)), mars)

            self._backward_buffer["mars"] = mars

        node_mars[sid:eid,:] = mars

        return None

    def _dense_backward_pass(self, data: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor, mus: torch.Tensor, log_scales: torch.Tensor):
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]

        maxval = node_mars[sid:eid,:].max(dim = 0)[0]
        grads = node_flows[sid:eid,:] / (node_mars[sid:eid,:] - maxval.unsqueeze(0)).exp()

        self._backward_buffer["mars"].backward(grads)

        l_b, r_b = self._backward_buffer["l_b"], self._backward_buffer["r_b"]

        if self.param_flows.dim() == 1:
            self.param_flows[:self.num_nodes] = (-(l_b.grad + r_b.grad) / log_scales.exp()).sum(dim = 1)
            self.param_flows[self.num_nodes:] = -(l_b.grad * l_b.data + r_b.grad * r_b.data).sum(dim = 1)
        else:
            self.param_flows[:self.num_nodes,:] = -(l_b.grad + r_b.grad) / log_scales.exp()
            self.param_flows[self.num_nodes:,:] = -(l_b.grad * l_b.data + r_b.grad * r_b.data)

        return None

    @staticmethod
    def _log_min_exp(a: torch.Tensor, b: torch.Tensor, epsilon = 1e-8):
        return a + torch.log(1 - torch.exp(b - a) + epsilon)

    @staticmethod
    def _log_min_exp_grad(grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon = 1e-8):
        exp_min = torch.exp(b - a)
        h = exp_min / (1.0 - exp_min + epsilon)
        return grad * (1.0 + h), -grad * h

    def _init_params(self, perturbation: float = 0.2):
        """
        Initialize the parameters with random values
        """
        mus = torch.rand([self.num_nodes])
        log_scales = torch.rand([self.num_nodes]) * -perturbation

        self.mus = nn.Parameter(mus)
        self.log_scales = nn.Parameter(log_scales)

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.mus.requires_grad = False
        self.log_scales.requires_grad = False

    def update_parameters(self):
        raise NotImplementedError()

    @staticmethod
    def _hook_params(grad_hook_idx: int, _inputs: List, layer_params: Dict):
        while len(_inputs) < grad_hook_idx + 2:
            _inputs.append(None)

        with torch.enable_grad():
            _inputs[grad_hook_idx] = ReverseGrad.apply(layer_params["mus"].permute(1, 0))
            _inputs[grad_hook_idx+1] = ReverseGrad.apply(layer_params["log_scales"].permute(1, 0))

        return grad_hook_idx + 2

    def _hook_param_grads(self, grad_hook_idx: int, _inputs: List, _inputs_grad: List):
        while len(_inputs_grad) < grad_hook_idx + 2:
            _inputs_grad.append(None)

        # Gradients of `mus`
        if _inputs[grad_hook_idx] is not None and _inputs[grad_hook_idx].requires_grad:
            if self.param_flows.dim() == 1:
                grads = self.param_flows[:self.num_nodes]
            else:
                grads = self.param_flows[:self.num_nodes,:]

            _inputs_grad[grad_hook_idx] = grads / torch.norm(grads, dim = 0, keepdim = True).clip(min = 5.0) * 5.0

        # Gradients of `log_scales`
        if _inputs[grad_hook_idx+1] is not None and _inputs[grad_hook_idx+1].requires_grad:
            if self.param_flows.dim() == 1:
                grads = self.param_flows[self.num_nodes:]
            else:
                grads = self.param_flows[self.num_nodes:,:]

            _inputs_grad[grad_hook_idx+1] = grads / torch.norm(grads, dim = 0, keepdim = True).clip(min = 5.0) * 5.0

        return grad_hook_idx + 2

    def _hook_input_grads(self, _inputs: List, _inputs_grad: List):

        # Gradients of `input`
        if _inputs[0] is not None and _inputs[0].requires_grad:

            if self.param_flows.dim() == 1:
                raise NotImplementedError()
            else:
                grads = -self.param_flows[:self.num_nodes,:]

            targets = torch.zeros([_inputs[0].size(0), grads.size(1)], device = grads.device)

            grid = lambda meta: (triton.cdiv(self.num_nodes * targets.size(1), meta['BLOCK_SIZE']),)
            self._accum_grad_kernel[grid](grads, targets, self.vids, self.vids2d, self.num_nodes, targets.size(1), BLOCK_SIZE = 1024)

            if _inputs_grad[0] is None:
                _inputs_grad[0] = targets
            else:
                _inputs_grad[0] += targets

    @staticmethod
    @triton.jit
    def _accum_grad_kernel(grads_ptr, targets_ptr, vids_ptr, vids2d_ptr, tot_num_nodes, batch_size, BLOCK_SIZE: tl.constexpr):

        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tot_num_nodes * batch_size

        node_offsets = offsets // batch_size
        batch_offsets = offsets % batch_size

        n_offsets = tl.load(vids_ptr + node_offsets, mask = mask, other = 0)
        v_offsets = tl.load(vids2d_ptr + n_offsets, mask = mask, other = 0)
        v_offsets = v_offsets * batch_size + batch_offsets

        grads = tl.load(grads_ptr + offsets, mask = mask, other = 0)

        tl.store(grads_ptr + offsets, v_offsets, mask = mask)
        tl.atomic_add(targets_ptr + v_offsets, grads, mask = mask)