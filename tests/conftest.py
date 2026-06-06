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

    # Count GPUs WITHOUT initializing CUDA
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        num_gpus = len(out.stdout.strip().splitlines())
    except FileNotFoundError:
        num_gpus = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id % num_gpus)

    # CPU limits — env vars before torch import
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    # Only now is it safe to import torch
    import torch
    torch.set_num_threads(8)