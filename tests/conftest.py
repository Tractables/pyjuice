import os
import json
import subprocess

"""
This script ensures pytest uses different GPUs to launch runtests for different workers.
Example usage:
py```
pytest -v -n 4
```

It also load-balances tests across workers (= across GPUs). pytest-xdist's default `--dist load`
scheduler deals tests out in *collection order* and refills greedily as workers finish; if the
slowest tests are collected last, one worker ends up hoarding them while the others idle (this is
why heavy tests appeared to pile onto GPU0). We fix this with LPT ("longest processing time first")
scheduling: per-test wall-times are recorded to `tests/.test_durations` and, on the next run, the
collection is reordered slowest-first so the greedy scheduler spreads the heavy tests over all
workers. Run the suite once to seed the cache; every run after that is balanced.
"""

# Cache of per-test wall-times, written by the xdist controller at session end and read back to
# reorder the collection. Lives next to this conftest so it is found regardless of the CWD.
_DURATIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".test_durations")
_durations_seen = {}   # nodeid -> seconds (this run; accumulates setup+call+teardown)

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

    # Expose the full pool (and this worker's id) so multi-GPU tests, which need >=2 visible GPUs,
    # can select a worker-specific subset instead of the single pinned GPU above -- keeping such
    # tests spread across distinct GPUs under `pytest -n`.
    os.environ["PYJUICE_TEST_GPU_POOL"] = ",".join(gpus)
    os.environ["PYJUICE_TEST_WORKER_ID"] = str(worker_id)

    # CPU limits — env vars before torch import
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"

    # Only now is it safe to import torch
    import torch
    torch.set_num_threads(8)


def pytest_collection_modifyitems(session, config, items):
    # LPT scheduling: reorder the collected tests slowest-first using the cached durations, so the
    # greedy `--dist load` scheduler deals the heavy tests out first and spreads them across workers
    # (= GPUs) instead of leaving them to pile onto whichever worker grabs them last.
    #
    # This hook runs in BOTH the controller and every xdist worker; they all read the same file and
    # apply the same stable sort, so they agree on the collection order (xdist requires this).
    # Unknown tests (no cached time yet) are sorted FIRST so a brand-new/uncached test is treated as
    # potentially-heavy and dealt out early -- the conservative choice for balancing.
    try:
        with open(_DURATIONS_PATH) as f:
            durations = json.load(f)
    except (OSError, ValueError):
        return   # no cache yet (first run) -> leave collection order untouched

    INF = float("inf")
    items.sort(key = lambda it: durations.get(it.nodeid, INF), reverse = True)


def pytest_runtest_logreport(report):
    # Accumulate each test's wall-time (setup + call + teardown). Under xdist the controller receives
    # a logreport for every test the workers run, so recording here captures the whole suite from one
    # process; under serial runs it simply records locally.
    _durations_seen[report.nodeid] = _durations_seen.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session, exitstatus):
    # Persist the durations once, from the controller only (workers have `workerinput`), merging with
    # any existing cache so a partial run (e.g. `-k`, a single file) doesn't wipe other tests' times.
    if hasattr(session.config, "workerinput"):
        return   # an xdist worker -> the controller writes the file
    if not _durations_seen:
        return
    merged = {}
    try:
        with open(_DURATIONS_PATH) as f:
            merged = json.load(f)
    except (OSError, ValueError):
        pass
    merged.update(_durations_seen)
    try:
        with open(_DURATIONS_PATH, "w") as f:
            json.dump(merged, f, indent = 0, sort_keys = True)
    except OSError:
        pass