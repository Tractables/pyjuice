import torch
from torch.autograd import Function


class ReverseGrad(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -grad_out