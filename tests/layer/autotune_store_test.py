"""
The tuner's winners on disk (:class:`pyjuice.layer.kernels.autotune._Store`): a later process starts from them, on the same GPU
model and sources alone, and nothing it reads there can take it off course.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from pyjuice.layer.kernels import autotune as A
from pyjuice.utils.kernel_launcher import triton_jit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "the tuner keys its choices by device")


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """A process of its own, as far as the tuner can tell, with its winners kept in a folder of its own."""
    monkeypatch.setenv(A.PERSIST_ENV, "1")
    monkeypatch.setenv(A.DIR_ENV, str(tmp_path))
    fresh_process(monkeypatch)
    monkeypatch.setattr(A, "ENABLED", True)
    monkeypatch.setattr(A, "_FORGET", False)
    torch.zeros(1, device = "cuda")
    return tmp_path


def fresh_process(monkeypatch):
    monkeypatch.setattr(A, "_CACHE", {})
    monkeypatch.setattr(A, "_STORES", {})
    monkeypatch.setattr(A, "_LOOKED_UP", set())


def counted(winner, monkeypatch):
    """`best_of` as `winner` says, the calls counted."""
    calls = []

    def best_of(candidates, *args, **kwargs):
        calls.append([c for c, _ in candidates])
        return winner
    monkeypatch.setattr(A, "best_of", best_of)
    return calls


def lines_of(folder):
    out = []
    for name in os.listdir(folder):
        with open(os.path.join(folder, name)) as f:
            out += [json.loads(line) for line in f]
    return out


def test_a_later_process_starts_from_what_an_earlier_one_measured(store_dir, monkeypatch):
    key = ("shape", 4, True, None, torch.float32, (2, 64))
    calls = counted((8, 16), monkeypatch)
    assert A.pick(key, [(4, 32), (8, 16)], lambda c: None) == (8, 16) and len(calls) == 1
    assert A.pick(key, [(4, 32), (8, 16)], lambda c: None) == (8, 16) and len(calls) == 1      # (as ever, within a process)
    fresh_process(monkeypatch)
    assert A.cached(key) == (8, 16) and not A.should_tune(key, 2)
    fresh_process(monkeypatch)
    assert not A.should_tune(key, 2)                           # (no scratch to set up for it)
    fresh_process(monkeypatch)
    assert A.pick(key, [(4, 32), (8, 16)], None) == (8, 16) and len(calls) == 1
    other = key[:-1] + ((2, 128),)
    assert A.cached(other) is None and A.should_tune(other, 2)
    assert A.pick(other, [(4, 32), (8, 16)], lambda c: None) == (8, 16) and len(calls) == 2


def test_a_gate_starts_from_it_too(store_dir, monkeypatch):
    key = ("gate", 1)
    calls = counted(("cuda", 3), monkeypatch)
    arms = [(("triton", -1), lambda: None), (("cuda", 3), lambda: None)]
    assert A.choose(key, arms) == ("cuda", 3) and len(calls) == 1
    fresh_process(monkeypatch)
    assert A.choose(key, arms) == ("cuda", 3) and len(calls) == 1
    monkeypatch.setattr(A, "ENABLED", False)                   # (a gate measures whatever the switch says: it is read too)
    fresh_process(monkeypatch)
    assert A.choose(key, arms) == ("cuda", 3) and len(calls) == 1


def test_it_takes_another_process(store_dir, tmp_path_factory):
    """The real thing: a kernel in the key, a Triton kernel's name reading the same in another process."""
    script = textwrap.dedent("""
        import torch, triton.language as tl
        from pyjuice.layer.kernels import autotune as A
        from pyjuice.utils.kernel_launcher import triton_jit
        @triton_jit
        def kernel(x, TILE: tl.constexpr):
            pass
        torch.zeros(1, device = "cuda")
        runs = []
        def bench(c):
            runs.append(c)
        print(A.pick((kernel, 8, torch.float32), [1, 2], bench), len(runs))
    """)
    path = tmp_path_factory.mktemp("script") / "run.py"
    path.write_text(script)
    outs = [subprocess.run([sys.executable, str(path)], capture_output = True, text = True, env = dict(os.environ))
            for _ in range(2)]
    for out in outs:
        assert out.returncode == 0, out.stderr[-2000:]
    first, runs = outs[0].stdout.split()[-2:]
    assert int(runs) > 0
    assert outs[1].stdout.split()[-2:] == [first, "0"]


def test_keys_read_the_same_in_any_process_or_are_not_kept(store_dir, monkeypatch):
    @triton_jit
    def kernel(x):
        pass

    def plain():
        pass
    text = A._key_text((kernel, plain, "a", None, 1.5, False, torch.int32))
    assert "0x" not in text and "test_keys_read_the_same_in_any_process_or_are_not_kept.<locals>.kernel" in text and "torch.int32" in text

    class Spy:                                                 # (no name that says what it is: an address, if anything)
        pass
    calls = counted("b", monkeypatch)
    assert A._key_text((Spy(), 1)) is None
    key = (Spy(), 1)
    assert A.pick(key, ["a", "b"], lambda c: None) == "b" and len(calls) == 1
    assert lines_of(store_dir) == []


def test_what_cannot_be_read_back_as_itself_is_not_kept(store_dir, monkeypatch):
    winner = object()
    counted(winner, monkeypatch)
    assert A.pick(("odd", 1), ["a", winner], lambda c: None) is winner
    assert lines_of(store_dir) == []


def test_a_cut_line_or_a_stranger_in_the_file_changes_nothing(store_dir, monkeypatch):
    key = ("shape", 1)
    calls = counted("b", monkeypatch)
    A.pick(key, ["a", "b"], lambda c: None)
    path = A._STORES[torch.cuda.current_device()].path
    with open(path, "a") as f:
        f.write(json.dumps([A._key_text(("evil", 1)), "__import__('os').system('false')"]) + "\n")
        f.write(json.dumps([A._key_text(("broken", 1)), "(1, "]) + "\n")
        f.write('["cut sh')
    fresh_process(monkeypatch)
    assert A.pick(key, ["a", "b"], lambda c: None) == "b" and len(calls) == 1
    assert A.pick(("evil", 1), ["a", "b"], lambda c: None) == "b" and len(calls) == 2           # (as for a key never seen)
    assert A.pick(("broken", 1), ["a", "b"], lambda c: None) == "b" and len(calls) == 3


def test_a_choice_not_offered_any_more_is_measured_again(store_dir, monkeypatch):
    key = ("shape", 2)
    calls = counted("b", monkeypatch)
    A.pick(key, ["a", "b"], lambda c: None)
    fresh_process(monkeypatch)
    counted("c", monkeypatch)
    assert A.pick(key, ["a", "c"], lambda c: None) == "c"
    fresh_process(monkeypatch)
    assert A.pick(key, ["a", "c"], None) == "c"                # (the last one counts)
    assert len(calls) == 1


def test_nothing_that_ran_is_nothing_to_go_by(store_dir, monkeypatch):
    calls = counted(None, monkeypatch)
    assert A.pick(("nothing ran", 1), ["a", "b"], lambda c: None) == "a"
    assert lines_of(store_dir) == [] and len(calls) == 1


def test_another_version_of_the_code_measures_again_and_old_files_go(store_dir, monkeypatch):
    key = ("shape", 3)
    calls = counted("b", monkeypatch)
    A.pick(key, ["a", "b"], lambda c: None)
    monkeypatch.setattr(A, "KEPT_VERSIONS", 3)
    for version in range(6):
        monkeypatch.setattr(A, "_source_digest", lambda version = version: f"version {version}")
        fresh_process(monkeypatch)
        assert A.pick(key, ["a", "b"], lambda c: None) == "b"
    assert len(calls) == 7
    assert len(os.listdir(store_dir)) == 3


def test_switched_off_nothing_is_read_or_kept(store_dir, monkeypatch):
    key = ("shape", 5)
    calls = counted("b", monkeypatch)
    A.pick(key, ["a", "b"], lambda c: None)
    fresh_process(monkeypatch)
    monkeypatch.setattr(A, "ENABLED", False)                   # autotuning off: every site its default, as ever
    assert A.cached(key) is None and A.pick(key, ["a", "b"], lambda c: None) == "a" and not A.should_tune(key, 2)
    monkeypatch.setattr(A, "ENABLED", True)
    monkeypatch.setenv(A.PERSIST_ENV, "0")                     # keeping winners off: measured, not read, not written
    before = lines_of(store_dir)
    fresh_process(monkeypatch)
    assert A.cached(key) is None and A.should_tune(key, 2)
    assert A.pick(key, ["a", "b"], lambda c: None) == "b" and len(calls) == 2
    assert lines_of(store_dir) == before


def test_clearing_the_cache_forgets_the_disk_too(store_dir, monkeypatch):
    key = ("shape", 6)
    calls = counted("b", monkeypatch)
    A.pick(key, ["a", "b"], lambda c: None)
    A.set_autotune(True, clear_cache = True)
    assert A.cached(key) is None and lines_of(store_dir) == []
    assert A.pick(key, ["a", "b"], lambda c: None) == "b" and len(calls) == 2
    assert len(lines_of(store_dir)) == 1


def test_a_folder_that_cannot_be_written_is_no_error(store_dir, monkeypatch):
    monkeypatch.setenv(A.DIR_ENV, str(store_dir / "a file" / "below"))
    (store_dir / "a file").write_text("")
    fresh_process(monkeypatch)
    counted("b", monkeypatch)
    assert A.pick(("shape", 7), ["a", "b"], lambda c: None) == "b"
