"""
A circuit moved to another device after it ran (`TensorCircuit.to`): the tensors its layers' passes derived from their
tables -- a sum layer's index tables, CUDA operands and scratch buffers, an external-parameter layer's plans, an input
layer's parameter flows -- must follow it, or the next pass hands the kernels pointers into the old device (an illegal
memory access, or "Pointer argument cannot be accessed from Triton").
"""

import os
import sys
import subprocess

import pytest
import torch

import pyjuice as juice
from pyjuice.layer import SumLayer


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEV = torch.device("cuda:0")


def _num_real_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True)
        return len(out.stdout.strip().splitlines())
    except (FileNotFoundError, OSError):
        return 0


def _two_gpus():
    """Two GPU ids for a child process, or None: the conftest pins every test process to ONE GPU before torch is
    imported, so `torch.cuda.device_count()` is 1 here however many there are -- read the pool it published instead
    (as `tests/queries/sample/sample_scope_plan_test.py` does), preferring ids past the worker count."""
    pool = os.environ.get("PYJUICE_TEST_GPU_POOL")
    gpus = [g for g in pool.split(",") if g] if pool else [str(i) for i in range(_num_real_gpus())]
    if len(gpus) < 2:
        return None
    n_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    idle = gpus[n_workers:]
    return idle[:2] if len(idle) >= 2 else gpus[-2:]


# Runs with two GPUs visible; its assertions are its own (the parent sees one GPU).
_CHILD = r"""
import sys
sys.setrecursionlimit(100000)
import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams, LowRankSumParams

C = 6
d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")


def chain(dist_fn, external = None):
    with juice.set_block_size(16):
        ch = inputs(0, num_node_blocks = 2, dist = dist_fn())
        trans = []
        for t in range(1, 6):
            ext = dict()
            if external == "blockscale":
                ext = dict(external_params = BlockScaleSumParams(ch_block_size = 16))
            elif external == "lowrank":
                ext = dict(external_params = LowRankSumParams(rank = 4))
            ch = summate(multiply(ch, inputs(t, num_node_blocks = 2, dist = dist_fn())), num_node_blocks = 2, **ext)
            trans.append(ch)
        root = summate(multiply(ch), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, trans


def build(kind, dev):
    torch.manual_seed(0)
    trans = None
    if kind == "hmm":
        ns = juice.structures.HMM(seq_length = 12, num_latents = 64, num_emits = C)
    elif kind == "hclt":
        x = torch.randint(0, C, (256, 10))
        ns = juice.structures.HCLT(x.float(), num_latents = 32, input_dist = dists.Categorical(num_cats = C))
        ns.init_parameters(perturbation = 2.0)
    elif kind == "pd":
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 16, split_intervals = 2,
                                 input_dist = dists.Categorical(num_cats = C))
        ns.init_parameters(perturbation = 2.0)
    elif kind == "softevi":
        ns, _ = chain(lambda: dists.SoftEvidenceCategorical(num_cats = C))
    else:
        ns, trans = chain(lambda: dists.Categorical(num_cats = C), external = kind)
    return juice.compile(ns, verbose = False).to(dev), trans


def run(pc, trans, kind, dev, B, V):
    g = torch.Generator().manual_seed(B)
    x = torch.randint(0, C, (B, V), generator = g).to(dev)
    miss = torch.zeros(B, V, dtype = torch.bool, device = dev)
    miss[:, 2] = True
    kw = {}
    if trans is not None:
        params = {}
        for ns in trans:
            ts = tuple((torch.randn(s, generator = g) * 0.5 - 0.5).to(dev) for s in ns.external_params.tensor_shapes(ns, B))
            params[ns] = ts if len(ts) > 1 else ts[0]
        kw = dict(sum_external_params = params)
    if kind == "softevi":
        kw = dict(categorical_evidence_logp = (torch.randn(B, V, C, generator = g) * 2).to(dev))
    lls = pc(x, missing_mask = miss, **kw).detach().clone()
    pc.backward(x, flows_memory = 0.0, missing_mask = miss, **({} if kind == "softevi" else kw))
    cond = juice.queries.conditional(pc, data = x, missing_mask = miss, target_vars = [2], **kw)
    return lls, pc.param_flows.clone(), cond


for kind, B, V in [("hmm", 4, 12), ("hmm", 32, 12), ("hclt", 4, 10), ("hclt", 32, 10), ("pd", 32, 16),
                   ("softevi", 4, 6), ("softevi", 32, 6), ("blockscale", 32, 6), ("lowrank", 32, 6)]:
    torch.cuda.set_device(d0)
    pc, trans = build(kind, d0)
    run(pc, trans, kind, d0, B, V)
    torch.cuda.set_device(d1)
    twin, twin_trans = build(kind, d1)
    want = run(twin, twin_trans, kind, d1, B, V)
    pc.to(d1)
    got = run(pc, trans, kind, d1, B, V)
    assert torch.equal(got[0], want[0]), (kind, B, "lls")
    # The backward's CUDA and Triton arms, picked per layer by timing, agree to ~1e-6; the conditional's add atomically
    assert torch.allclose(got[1], want[1], rtol = 1e-5, atol = 1e-6), (kind, B, "param flows")
    assert torch.allclose(got[2], want[2], atol = 1e-6), (kind, B, "conditional")
print("CHILD_OK")
"""


@pytest.mark.skipif(_two_gpus() is None, reason = "requires >= 2 GPUs")
def test_a_circuit_moved_after_running_computes_as_one_compiled_there():
    """Every kind of layer -- plain sum layers at small and large batch, an HCLT, a PD, BlockScale and LowRank sums,
    SoftEvidenceCategorical inputs: a forward, a backward and a conditional on one GPU, `pc.to` another, the same
    there -- against a twin compiled on the second GPU (log-likelihoods to the bit). Before the fix: an illegal memory
    access at the first pass after the move."""
    gpus = _two_gpus()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env.pop("PYJUICE_TEST_GPU_POOL", None)
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output = True, text = True, timeout = 900, env = env)
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, (
        f"a moved circuit failed on GPUs {gpus} (exit {proc.returncode})\n"
        f"--- child stdout ---\n{proc.stdout}\n--- child stderr ---\n{proc.stderr}")


@cuda_only
def test_a_sum_layer_to_drops_its_derived_tensors():
    """`SumLayer.to` returns the layer and forgets the tensors derived from its tables, which the next pass rebuilds
    to the same values."""
    torch.manual_seed(0)
    pc = juice.compile(juice.structures.HMM(seq_length = 8, num_latents = 32, num_emits = 5), verbose = False).to(DEV)
    x = torch.randint(0, 5, (16, 8), device = DEV)
    want = pc(x).detach().clone()
    layers = [layer for lg in pc.inner_layer_groups for layer in lg if isinstance(layer, SumLayer)]
    assert any(layer._cached_fw_pcids for layer in layers)
    for layer in layers:
        assert layer.to(DEV) is layer
        assert not layer._cached_fw_pcids and not layer._cached_fw_cuda and layer._bk_ele_scratch is None
    assert torch.equal(pc(x), want)
