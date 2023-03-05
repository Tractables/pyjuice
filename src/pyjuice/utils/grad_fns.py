import torch
from torch.autograd import Function


class ReverseGrad(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -grad_out


class PseudoHookFunc(Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.num_vars = len(args)
        ctx.save_for_backward(*args)
        return args[-1]

    @staticmethod
    def backward(ctx, grad):
        return tuple(grad if i == ctx.num_vars-1 else torch.zeros_like(ctx.saved_tensors[i]) for i in range(ctx.num_vars))
