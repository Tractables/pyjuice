"""
The optimizers under DDP on a circuit storing its sum parameters in bf16 (`param_dtype`), training through the fp32
master copy (`params_master`), two ranks on two GPUs, each with its own data:

* synchronous DDP (`MiniBatchEM` and `SGD` with `ddp = True`): the ranks' master copies and stored parameters stay
  identical to the bit, and follow a twin storing the same values in fp32 (the drift the bf16 forward makes);
* Local-SGD (`MiniBatchEM`, `sync_every` > 1): `_average_params` makes every rank's master copy the ranks' arithmetic mean,
  stores its rounding, and leaves the circuit normalized.

Assertion failures fail the test; only a failed rendezvous (an infrastructure hiccup) is retried, then skipped.
"""
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


def _build_pc(device, dtype):
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists
    torch.manual_seed(0)
    xb = torch.randint(0, 8, (600, 16))
    ns = juice.structures.HCLT(xb.float(), num_latents = 32, num_bins = 8, input_dist = dists.Categorical(num_cats = 8))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(device)
    return pc if dtype is torch.bfloat16 else pc.set_param_dtype(torch.float32)     # (both: the bf16 values)


def _ddp_worker(rank, world_size, init_file, errdir):
    fd = os.open(os.path.join(errdir, f"rank{rank}.err"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ["PYJUICE_DISABLE_CUDA_KERNELS"] = "1"                 # (the optimizers' logic is under test, not kernels)
    try:
        import torch.distributed as dist
        import pyjuice as juice
        from pyjuice.model.backend import eval_partition_fn

        torch.cuda.set_device(rank)
        dist.init_process_group(backend = "nccl", init_method = f"file://{init_file}",
                                rank = rank, world_size = world_size, timeout = timedelta(seconds = 120))
        device = torch.device(f"cuda:{rank}")
        BF16 = torch.bfloat16

        def rank_data(step):
            torch.manual_seed(1000 + 10 * step + rank)                  # distinct data per rank
            return torch.randint(0, 8, (64, 16), device = device)

        def gathered(t):
            parts = [torch.empty_like(t) for _ in range(world_size)]
            dist.all_gather(parts, t.contiguous())
            return parts

        checks = []                                                     # (label, ok): asserted after the collectives

        # (1) synchronous DDP
        makers = {"MiniBatchEM": lambda pc: juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1, ddp = True),
                  "SGD": lambda pc: juice.optim.SGD(pc, lr = 0.1, ddp = True)}
        for name, make in makers.items():
            runs = {}
            for dtype in (torch.float32, BF16):
                pc = _build_pc(device, dtype)
                opt = make(pc)
                for step in range(4):
                    x = rank_data(step)
                    pc(x).mean().backward()
                    opt.step()
                runs[dtype] = pc
            pc16, pc32 = runs[BF16], runs[torch.float32]
            masters, stored = gathered(pc16.params_master), gathered(pc16.params.data)
            checks.append((f"{name}: master copies identical across ranks", all(torch.equal(m, masters[0]) for m in masters)))
            checks.append((f"{name}: stored params identical across ranks", all(torch.equal(p, stored[0]) for p in stored)))
            checks.append((f"{name}: params are the master's rounding", torch.equal(pc16.params, pc16.params_master.to(BF16))))
            nz = pc32.params > 1e-6
            rel = ((pc16.params_master - pc32.params).abs() / pc32.params)[nz]
            checks.append((f"{name}: master follows the fp32 twin (median rel {float(rel.median()):.2e})", float(rel.median()) < 1e-2))

        # (2) Local-SGD: one local update per rank, then the parameters averaged
        pc = _build_pc(device, BF16)
        opt = juice.optim.MiniBatchEM(pc, step_size = 0.5, pseudocount = 0.1, ddp = True, sync_every = 10_000)
        pc(rank_data(9)).mean().backward(); opt.step()                 # rank-specific master copies
        expected = pc.params_master.clone(); dist.all_reduce(expected); expected.div_(world_size)
        before = gathered(pc.params_master)
        spread = max(float((m - before[0]).abs().max()) for m in before)
        opt._average_params()
        checks.append((f"Local-SGD: ranks diverged before averaging (spread {spread:.2e})", spread > 0.0))
        checks.append(("Local-SGD: master is the ranks' mean", torch.allclose(pc.params_master, expected, rtol = 1e-6, atol = 1e-8)))
        checks.append(("Local-SGD: params are the master's rounding", torch.equal(pc.params, pc.params_master.to(BF16))))
        z = eval_partition_fn(pc)
        z = float(z.abs().max()) if torch.is_tensor(z) else abs(float(z))
        checks.append((f"Local-SGD: normalized after averaging (|log Z| {z:.2e})", z < 2e-2))   # (bf16-stored values)

        dist.barrier()
        for label, ok in checks:
            assert ok, f"[rank {rank}] {label}"
        dist.destroy_process_group()
    except Exception:
        traceback.print_exc(); sys.stderr.flush()
        raise


@pytest.mark.skipif(_ddp_gpus() is None, reason = "requires >= 2 GPUs")
def test_optimizers_under_ddp_at_bf16():
    import torch.multiprocessing as mp
    world_size = 2
    gpus = _ddp_gpus(world_size)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    last_err = ""
    try:
        for attempt in range(3):
            errdir = tempfile.mkdtemp(prefix = "pyjuice_bf16_ddp_")
            init_file = os.path.join(errdir, "store")
            try:
                mp.spawn(_ddp_worker, args = (world_size, init_file, errdir), nprocs = world_size, join = True)
                return
            except Exception as e:
                errs = _read_child_errs(errdir)
                # Only a failed rendezvous / connection is retried; any other error of a rank (a failed check, a bug)
                # fails the test
                infra = any(k in errs for k in ("DistStoreError", "Timed out", "timed out", "rendezvous", "ConnectionRefused",
                                                "Connection reset", "Address already in use"))
                if "Traceback" in errs and not infra:
                    pytest.fail(f"a rank failed:\n{errs}")
                last_err = f"attempt {attempt} failed: {e}\n{errs}"
            finally:
                shutil.rmtree(errdir, ignore_errors = True)
        pytest.skip(f"DDP rendezvous failed after 3 attempts (transient infra issue).\n{last_err}")
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev
