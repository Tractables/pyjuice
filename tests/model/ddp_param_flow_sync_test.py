import os
import socket
import subprocess

import pytest
import torch


def _num_real_gpus():
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so this sees the true count even though the test
    # conftest pins each worker to a single visible GPU.
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True)
        return len([l for l in out.stdout.strip().splitlines() if l.strip()])
    except Exception:
        return 0


def _free_port():
    s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close()
    return p


def _ddp_gpu_pair(world_size = 2):
    # Pick a worker-specific block of `world_size` GPUs from the pool the conftest exposes, so that
    # under `pytest -n` different (multi-GPU) tests land on different GPUs rather than all grabbing
    # 0,1. Falls back to the physical GPUs when run outside the conftest. Returns None if too few.
    pool = os.environ.get("PYJUICE_TEST_GPU_POOL")
    gpus = [g for g in pool.split(",") if g] if pool else [str(i) for i in range(_num_real_gpus())]
    if len(gpus) < world_size:
        return None
    wid = int(os.environ.get("PYJUICE_TEST_WORKER_ID", "0"))
    start = (wid * world_size) % len(gpus)
    return [gpus[(start + i) % len(gpus)] for i in range(world_size)]


def _build_small_pc(device):
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists
    torch.manual_seed(0)   # identical structure + data across ranks
    xb = torch.randint(0, 8, (1000, 16), device = device)
    ns = juice.structures.HCLT(xb.float(), num_latents = 64, num_bins = 8,
                               input_dist = dists.Categorical(num_cats = 8))
    pc = juice.compile(ns); pc.to(device)
    return pc


def _ddp_worker(rank, world_size, port):
    # Triton path only (skip the optional CUDA-kernel JIT for a fast test).
    os.environ["PYJUICE_DISABLE_CUDA_KERNELS"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    import torch.distributed as dist
    dist.init_process_group("nccl", rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    pc = _build_small_pc(device)
    x = torch.randint(0, 8, (32, 16), device = device)   # identical batch on every rank (same seed)

    def bwd():
        pc.init_param_flows(flows_memory = 0.0)
        pc(x, propagation_alg = "LL")
        pc.backward(x, flows_memory = 1.0, allow_modify_flows = False,
                    propagation_alg = "LL", logspace_flows = True)

    # All ranks compute identical flows P -> a SUM all-reduce must give world_size * P everywhere.
    # IMPORTANT: do every collective first, then assert (an assert between collectives would deadlock
    # the other ranks that proceed to the next collective).

    bwd(); P = pc.param_flows.clone()
    pc.sync_param_flows()                              # fp32 SUM
    pf_fp32 = pc.param_flows.clone()

    bwd(); P2 = pc.param_flows.clone()
    pc.sync_param_flows(dtype = torch.bfloat16)        # bf16 SUM
    pf_bf16 = pc.param_flows.clone()

    pc._cum_flow = float(rank + 1)                     # distinct per rank -> SUM is sum(1..world_size)
    pc.sync_param_flows()
    cum_flow = pc._cum_flow

    dist.barrier()   # all collectives done; safe to assert now

    m = P.abs() > P.abs().max() * 1e-4
    assert torch.allclose(pf_fp32[m], world_size * P[m], rtol = 1e-4, atol = 1e-6), \
        f"[rank {rank}] fp32 sync_param_flows mismatch"

    m2 = P2.abs() > P2.abs().max() * 1e-4
    rel = ((pf_bf16[m2] - world_size * P2[m2]).abs() / (world_size * P2[m2].abs())).max().item()
    assert rel < 2e-2, f"[rank {rank}] bf16 sync_param_flows rel error too large: {rel:.3e}"

    assert abs(cum_flow - sum(range(1, world_size + 1))) < 1e-6, \
        f"[rank {rank}] _cum_flow sync mismatch: {cum_flow}"

    dist.destroy_process_group()


@pytest.mark.skipif(_ddp_gpu_pair() is None, reason = "requires >= 2 GPUs")
def test_sync_param_flows_ddp():
    import torch.multiprocessing as mp

    world_size = 2
    # The conftest pins this process to a single GPU; expose this worker's >=2-GPU subset to the
    # spawned ranks so each rank gets its own device and parallel workers stay on distinct GPUs.
    # (Spawned processes re-read CUDA_VISIBLE_DEVICES at torch import.)
    pair = _ddp_gpu_pair(world_size)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(pair)
    try:
        mp.spawn(_ddp_worker, args = (world_size, _free_port()), nprocs = world_size, join = True)
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_sync_param_flows_noop_without_dist():
    # Without an initialized process group, the helper must be a safe no-op (leaves flows untouched).
    device = torch.device("cuda:0")
    pc = _build_small_pc(device)
    x = torch.randint(0, 8, (32, 16), device = device)
    pc.init_param_flows(flows_memory = 0.0)
    pc(x, propagation_alg = "LL")
    pc.backward(x, flows_memory = 1.0, allow_modify_flows = False,
                propagation_alg = "LL", logspace_flows = True)

    before = pc.param_flows.clone()
    cum_before = pc._cum_flow
    pc.sync_param_flows()                          # no dist -> no-op
    pc.sync_param_flows(dtype = torch.bfloat16)    # no dist -> no-op
    assert torch.equal(pc.param_flows, before)
    assert pc._cum_flow == cum_before


if __name__ == "__main__":
    test_sync_param_flows_noop_without_dist()
    test_sync_param_flows_ddp()
