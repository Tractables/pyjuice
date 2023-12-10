import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer, ProdLayer, SumLayer

import pytest


import triton
import triton.language as tl


@triton.jit
def kernel1(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(axis = 0)

    offs_a = tl.arange(0, M)[:,None] * N + tl.arange(0, N)[None,:]
    aa = tl.load(a + offs_a) # .to(tl.bfloat16)

    offs_b = tl.arange(0, N)[:,None] * K + tl.arange(0, K)[None,:]
    bb = tl.load(b + offs_b) # .to(tl.bfloat16)

    cc = tl.dot(aa, bb, allow_tf32 = True) # .to(tl.float32)

    offs_c = tl.arange(0, M)[:,None] * K + tl.arange(0, K)[None,:]
    tl.store(c + offs_c, cc)


@triton.jit
def kernel2(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(axis = 0)

    offs_a = tl.arange(0, M)[:,None] * N + tl.arange(0, N)[None,:]
    aa = tl.load(a + offs_a) # .to(tl.bfloat16)

    offs_b = tl.arange(0, N)[:,None] * K + tl.arange(0, K)[None,:]
    bb = tl.load(b + offs_b) # .to(tl.bfloat16)

    cc = tl.sum(aa[:,:,None] * bb[None,:,:], axis = 1) # .to(tl.float32)

    # cc = tl.dot(aa, bb)

    offs_c = tl.arange(0, M)[:,None] * K + tl.arange(0, K)[None,:]
    tl.store(c + offs_c, cc)


@triton.jit
def kernel3(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(axis = 0)

    offs_a = tl.arange(0, M)[:,None] * N + tl.arange(0, N)[None,:]
    aa = tl.load(a + offs_a)

    offs_b = tl.arange(0, N)[:,None] * K + tl.arange(0, K)[None,:]
    bb = tl.load(b + offs_b)

    aa = tl.view(tl.broadcast_to(aa[:,None,:], (M, 16 // M, N)), (16, N))
    cc = tl.dot(aa, bb)
    cc = tl.max(tl.view(cc, (M, 16 // M, N)), axis = 1)

    offs_c = tl.arange(0, M)[:,None] * K + tl.arange(0, K)[None,:]
    tl.store(c + offs_c, cc)


if __name__ == "__main__":
    import time

    M = 16
    N = 16
    K = 16

    a = torch.rand([M, N]).cuda()
    b = torch.rand([N, K]).cuda()
    c = torch.rand([M, K]).cuda()

    grid = (1000,)

    kernel1[grid](a, b, c, M, N, K)

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for _ in range(100):
    #     kernel1[grid](a, b, c, M, N, K)
    #     torch.cuda.synchronize()
    # t1 = time.time()

    # print((t1 - t0) / 100 * 1000)

    kernel2[grid](a, b, c, M, N, K)

    # torch.cuda.synchronize()
    # t0 = time.time()
    # for _ in range(100):
    #     kernel2[grid](a, b, c, M, N, K)
    #     torch.cuda.synchronize()
    # t1 = time.time()

    # print((t1 - t0) / 100 * 1000)

    # cc = torch.matmul(a, b)

    # print((c - cc).abs().max())