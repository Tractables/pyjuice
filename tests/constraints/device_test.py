"""
A constrained circuit on a GPU other than the process's current one: Triton launches kernels on the current device, so
every query pins it (:meth:`TensorCircuit.to` does not change it). Before that: "an illegal memory access", at a later
launch. In a child process, for two GPUs (the suite's workers see one each).
"""
import os
import subprocess
import sys

import pytest

_CHILD = r"""
import sys
sys.setrecursionlimit(1 << 20)
import torch
import pyjuice as juice, pyjuice.constraints as jc

V, n = 6, 6

def build(dev):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = n, num_latents = 64, num_emits = V)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(dev)

def answers(cc, dev):
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, V, (4, n), generator = g).to(dev)
    missing = (torch.rand(4, n, generator = g) < 0.6).to(dev)
    with cc.decoder() as d:
        session = d.marginal(data, missing), d.conditional(data, missing)
    return cc.marginal(data, missing), cc.conditional(data, missing), *session

dfa = jc.DFA.contains([[1, 2]], V)
assert torch.cuda.current_device() == 0
on_one = answers(jc.compile(dfa, build("cuda:1")), "cuda:1")      # compiled there, the current device 0
assert torch.cuda.current_device() == 0
moved = jc.compile(dfa, build("cuda:0"))
answers(moved, "cuda:0")
got = answers(moved.to("cuda:1"), "cuda:1")                       # moved there
for a, b in zip(got, on_one):
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
print("CHILD_OK")
"""


def _two_gpus():
    """Two GPUs of the suite's pool for a child of its own, or None."""
    pool = [g for g in os.environ.get("PYJUICE_TEST_GPU_POOL", "").split(",") if g]
    return pool[:2] if len(pool) >= 2 else None


@pytest.mark.skipif(_two_gpus() is None, reason = "requires two GPUs in PYJUICE_TEST_GPU_POOL")
def test_queries_on_a_gpu_other_than_the_current_one():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(_two_gpus())
    env.pop("PYJUICE_TEST_GPU_POOL", None)
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output = True, text = True, timeout = 600, env = env)
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"
