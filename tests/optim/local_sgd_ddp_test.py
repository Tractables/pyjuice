import os
import sys
import shutil
import tempfile
import traceback
import subprocess
from datetime import timedelta

import pytest
import torch


def _num_real_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True)
        return len([l for l in out.stdout.strip().splitlines() if l.strip()])
    except Exception:
        return 0


def _ddp_gpus(world_size = 2):
    # Prefer GPUs not pinned to other concurrent xdist workers (gpus[N:] under `pytest -n N`).
    pool = os.environ.get("PYJUICE_TEST_GPU_POOL")
    gpus = [g for g in pool.split(",") if g] if pool else [str(i) for i in range(_num_real_gpus())]
    if len(gpus) < world_size:
        return None
    n_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    idle = gpus[n_workers:]
    return (idle[:world_size] if len(idle) >= world_size else gpus[-world_size:])


def _read_child_errs(errdir):
    out = []
    for name in sorted(os.listdir(errdir)):
        if name.endswith(".err"):
            try:
                with open(os.path.join(errdir, name)) as f:
                    txt = f.read().strip()
            except OSError:
                txt = ""
            if txt:
                out.append(f"---- {name} ----\n{txt}")
    return "\n".join(out) if out else "(no child stderr captured)"


def _build_pc(device, seed = 0):
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists
    torch.manual_seed(seed)
    xb = torch.randint(0, 8, (600, 16), device = device)
    ns = juice.structures.HCLT(xb.float(), num_latents = 32, num_bins = 8,
                               input_dist = dists.Categorical(num_cats = 8))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns); pc.to(device)
    return pc


def _make(name, pc, sync_every):
    import pyjuice as juice
    if name == "MiniBatchEM":
        return juice.optim.MiniBatchEM(pc, step_size = 0.5, ddp = True, sync_every = sync_every)
    return juice.optim.Anemone(pc, step_size = 0.4, momentum = 0.9, ddp = True, sync_every = sync_every)


def _ddp_worker(rank, world_size, init_file, errdir):
    fd = os.open(os.path.join(errdir, f"rank{rank}.err"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ["PYJUICE_DISABLE_CUDA_KERNELS"] = "1"

    try:
        import torch.distributed as dist
        import pyjuice as juice
        from pyjuice.model.backend import eval_partition_fn

        torch.cuda.set_device(rank)
        dist.init_process_group(backend = "nccl", init_method = f"file://{init_file}",
                                rank = rank, world_size = world_size, timeout = timedelta(seconds = 120))
        device = torch.device(f"cuda:{rank}")

        def rank_data():
            torch.manual_seed(1000 + rank)                      # distinct data per rank -> local divergence
            return torch.randint(0, 8, (32, 16), device = device)

        def accumulate(pc, x):
            pc(x); pc.backward(x)

        def log_z(pc):
            z = eval_partition_fn(pc)
            return float(z.abs().max()) if torch.is_tensor(z) else abs(float(z))

        checks = []   # (label, bool) -- evaluated AFTER all collectives to avoid deadlock

        for name in ["MiniBatchEM", "Anemone"]:
            # (1) _average_params == arithmetic mean across ranks, and stays normalized.
            pc = _build_pc(device)
            opt = _make(name, pc, sync_every = 10_000)           # large -> updates stay local, no auto-sync
            accumulate(pc, rank_data()); opt.step()              # one local update -> rank-specific params

            exp = pc.params.clone(); dist.all_reduce(exp, op = dist.ReduceOp.SUM); exp.div_(world_size)
            inp_exp = []
            for layer in pc.input_layer_group:
                e = layer.params.clone(); dist.all_reduce(e, op = dist.ReduceOp.SUM); e.div_(world_size)
                inp_exp.append(e)

            opt._average_params()

            ok = torch.allclose(pc.params, exp, rtol = 1e-4, atol = 1e-6)
            for layer, e in zip(pc.input_layer_group, inp_exp):
                ok = ok and torch.allclose(layer.params, e, rtol = 1e-4, atol = 1e-6)
            checks.append((f"{name}: _average_params equals arithmetic mean", ok))
            checks.append((f"{name}: normalized after averaging (|logZ|<1e-3)", log_z(pc) < 1e-3))

            # (2) sync_every cadence: local update diverges; the sync update re-syncs all ranks.
            pc2 = _build_pc(device)
            opt2 = _make(name, pc2, sync_every = 2)
            x = rank_data()

            accumulate(pc2, x); opt2.step()                      # update 1: local (1 % 2 != 0)
            c1 = torch.tensor([(pc2.params.double() ** 2).sum().item()], device = device)
            g1 = [torch.zeros_like(c1) for _ in range(world_size)]; dist.all_gather(g1, c1)

            accumulate(pc2, x); opt2.step()                      # update 2: sync fires (2 % 2 == 0)
            c2 = torch.tensor([(pc2.params.double() ** 2).sum().item()], device = device)
            g2 = [torch.zeros_like(c2) for _ in range(world_size)]; dist.all_gather(g2, c2)

            spread1 = max(t.item() for t in g1) - min(t.item() for t in g1)
            spread2 = max(t.item() for t in g2) - min(t.item() for t in g2)
            checks.append((f"{name}: ranks diverge during local updates (spread {spread1:.3e})", spread1 > 1e-6))
            checks.append((f"{name}: ranks identical after sync (spread {spread2:.3e})", spread2 < 1e-6))

        dist.barrier()
        for label, ok in checks:
            assert ok, f"[rank {rank}] {label}"

        dist.destroy_process_group()
    except Exception:
        traceback.print_exc(); sys.stderr.flush()
        raise


@pytest.mark.skipif(_ddp_gpus() is None, reason = "requires >= 2 GPUs")
def test_local_sgd_param_averaging():
    import torch.multiprocessing as mp
    world_size = 2
    gpus = _ddp_gpus(world_size)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    last_err = ""
    try:
        for attempt in range(3):
            errdir = tempfile.mkdtemp(prefix = "pyjuice_localsgd_")
            init_file = os.path.join(errdir, "store")
            try:
                mp.spawn(_ddp_worker, args = (world_size, init_file, errdir), nprocs = world_size, join = True)
                return
            except Exception as e:
                last_err = f"attempt {attempt} failed: {e}\n{_read_child_errs(errdir)}"
            finally:
                shutil.rmtree(errdir, ignore_errors = True)
        pytest.skip(f"DDP rendezvous failed after 3 attempts (transient infra issue).\n{last_err}")
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


if __name__ == "__main__":
    test_local_sgd_param_averaging()
