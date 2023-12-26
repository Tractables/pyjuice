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
    aa = tl.load(a + offs_a).to(tl.float16)

    offs_b = tl.arange(0, N)[:,None] * K + tl.arange(0, K)[None,:]
    bb = tl.load(b + offs_b).to(tl.float16)

    cc = tl.dot(aa, bb).to(tl.float32)

    offs_c = tl.arange(0, M)[:,None] * K + tl.arange(0, K)[None,:]
    tl.store(c + offs_c, cc)


@triton.jit
def kernel2(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(axis = 0)

    offs_a = tl.arange(0, M)[:,None] * N + tl.arange(0, N)[None,:]
    aa = tl.load(a + offs_a)#.to(tl.float16)

    offs_b = tl.arange(0, N)[:,None] * K + tl.arange(0, K)[None,:]
    bb = tl.load(b + offs_b)#.to(tl.float16)

    cc = tl.sum(aa[:,:,None] * bb[None,:,:], axis = 1)#.to(tl.float32)

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
    c = torch.zeros([M, K]).cuda()

    grid = (400,)

    kernel1[grid](a, b, c, M, N, K)

    aaa = [item for item in kernel1.cache[0].values()]
    raw_kernel = aaa[0][(grid[0], 1, 1)]

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack = True) as prof:
        for i in range(50):
            # kernel1[grid](a, b, c, M, N, K)
            raw_kernel(a, b, c)

    prof.export_chrome_trace("trace.json")

    import pdb; pdb.set_trace()