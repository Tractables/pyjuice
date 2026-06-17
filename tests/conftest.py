import os
import subprocess

"""
This script ensures pytest uses different GPUs to launch runtests for different workers.
Example usage:
py```
pytest -v -n 4
```
"""

def pytest_configure(config):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker_id = int(worker.replace("gw", "")) if worker.startswith("gw") else 0

    # Determine the pool of GPUs to distribute workers across.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        # Respect an externally-provided restriction (e.g. `CUDA_VISIBLE_DEVICES=4,5,6,7 pytest -n 4`):
        # pick from that set rather than overriding it with absolute indices.
        gpus = [g.strip() for g in visible.split(",") if g.strip() != ""]
    else:
        # No restriction provided: enumerate all physical GPUs WITHOUT initializing CUDA.
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            num_gpus = len(out.stdout.strip().splitlines())
        except FileNotFoundError:
            num_gpus = 1
        gpus = [str(i) for i in range(max(num_gpus, 1))]

    # Pin this worker to a single GPU from the pool.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker_id % len(gpus)]

    # CPU limits — env vars before torch import
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    # Only now is it safe to import torch
    import torch
    torch.set_num_threads(8)