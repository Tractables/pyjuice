import pytest
import torch

from pyjuice.layer.kernels import autotune


# `autotune.best_of` measures in two stages: one untimed and one timed run per candidate, then the full
# `warmup` / `reps` only for the candidates within `PRUNE` of the fastest. GPU time is simulated with
# `torch.cuda._sleep` (spin cycles), so the candidates' relative speeds are known.


def _candidate(cycles, calls, name):
    def run():
        calls[name] = calls.get(name, 0) + 1
        torch.cuda._sleep(cycles)
    return name, run


def test_a_much_slower_candidate_runs_twice_and_loses():
    calls = {}
    cands = [_candidate(2_000_000, calls, "ref"), _candidate(40_000_000, calls, "slow"),
             _candidate(2_000_000, calls, "tie")]
    assert autotune.best_of(cands, warmup = 3, reps = 7) == "ref"          # a tie keeps the reference
    assert calls["slow"] == 2, f"the pruned candidate ran {calls['slow']} times"
    assert calls["ref"] == calls["tie"] == 10, calls                        # 1 + 1 + (3 - 2) + 7, as before


def test_a_pruned_reference_loses_to_the_fast_candidate():
    calls = {}
    cands = [_candidate(40_000_000, calls, "ref"), _candidate(2_000_000, calls, "fast")]
    assert autotune.best_of(cands, warmup = 3, reps = 7) == "fast"
    assert calls["ref"] == 2 and calls["fast"] == 10, calls


def test_a_close_win_still_needs_the_margin():
    calls = {}
    # 1.5x faster: inside PRUNE, so both are measured in full, and well past MARGIN
    cands = [_candidate(3_000_000, calls, "ref"), _candidate(2_000_000, calls, "better")]
    assert autotune.best_of(cands, warmup = 3, reps = 7) == "better"
    assert calls["ref"] == calls["better"] == 10, calls


def test_candidates_that_cannot_run_are_skipped():
    def broken():
        raise RuntimeError("cannot launch")
    calls = {}
    assert autotune.best_of([("ref", broken), _candidate(2_000_000, calls, "ok")]) == "ok"
    assert autotune.best_of([("ref", broken)]) is None
