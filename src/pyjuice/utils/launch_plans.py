"""
Launch plans: what a layer call launches, decided once per call signature and replayed after.

An eager pass spends most of its host time deciding, not launching: a sum layer's backward walks three methods
deep through tile heuristics, cache lookups, mode checks and the argument dict of each launch before the launch
itself -- MEASURED (a tied gated HMM-2048 x4, 15 layers, B 64, bf16) 45 us of host time a layer of which the two
launches it makes are ~20, and the circuit's step 3.0 ms of host time against 1.5 ms of GPU work. For a given
call signature -- the batch size, the flags, the dtypes, the device -- those decisions come out the same every
time, so a plan keeps their result: one LAUNCHER per unit of work, a callable that takes the call's tensors and
launches exactly what the full path launches for them.

The contract that keeps it correct:

  * A launcher receives every tensor that can change between calls as an ARGUMENT. What it captures is fixed for
    the key: the layer's compiled tables, the tuned configs, sizes, flags. (Never the tensors of the call it was
    recorded in: a buffer reallocated by a later call would be written through a stale address.)
  * The full path runs inside `LaunchPlans.record(tag)`, and each converted launch site adds its launcher with its
    tag (`LaunchPlans.add`) to the innermost recording of that tag open then -- so recordings of two levels (a gated
    layer's whole call, and the standard backward within it, which plans itself too) never mix their launchers. The
    owner stores the plan only if the recording holds exactly the launchers it expects -- one per unit of work --
    so a call that went through a site nobody converted is never planned: it fails closed.
  * An owner recording around another owner's call captures that call's launches through sites of its own, which
    the inner call's converted sites reach (a gated layer's hooks, called from the sum layer's). So an inner
    recording that comes out incomplete (`store`), or an inner call in a regime it does not record, fails every
    recording around it as well (`LaunchPlans.fail`): its unconverted launch reached no outer site either.
  * The owner keys a plan on everything its decisions read, and plans only the regimes it has converted.
  * Nothing is recorded or replayed during a CUDA graph capture: a capture runs the full path, unchanged.

`ENABLED` (`PYJUICE_LAUNCH_PLANS=0` turns it off) makes every call take the full path.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import torch


ENABLED = os.environ.get("PYJUICE_LAUNCH_PLANS", "1") != "0"


class _Recording:
    """One call's full path being recorded: its tag, the launchers its converted sites added, and whether a site that
    has none of its own ran (`LaunchPlans.fail`)."""
    __slots__ = ("tag", "launchers", "failed")

    def __init__(self, tag):
        self.tag = tag
        self.launchers = []
        self.failed = False


class LaunchPlans:
    """
    One owner's launch plans -- key -> its launchers, `MAX_PLANS` at most --, `LaunchPlans.of(owner)`. The recordings
    open now are the class's, shared by every owner -- a launch site adds to whichever call is being recorded without
    knowing whose it is -- and per thread: a thread running another circuit's pass meanwhile never adds to them.
    """
    MAX_PLANS = 32              # a few batch sizes x a few flag sets; a sweep over many must not grow it unbounded

    _local = threading.local()  # `.open`: this thread's recordings open now, innermost last

    __slots__ = ("plans",)

    def __init__(self):
        self.plans = dict()

    @staticmethod
    def of(owner) -> "LaunchPlans":
        """`owner`'s plans (made on first use; kept in its `__dict__`: an `nn.Module`'s attribute protocol costs ~1 us a
        call)."""
        lp = owner.__dict__.get("_launch_plans")
        if lp is None:
            lp = owner.__dict__["_launch_plans"] = LaunchPlans()
        return lp

    def lookup(self, key):
        """The plan stored for `key`, or None (also when plans are off or a capture is underway)."""
        if not ENABLED or key is None:
            return None
        plan = self.plans.get(key)
        if plan is not None and _capturing():
            return None
        return plan

    def store(self, key, rec: _Recording, expected: int) -> None:
        """Keep `rec`'s launchers as the plan for `key` if it holds exactly `expected` of them and nothing marked it
        failed (and no capture is underway). Incomplete, it fails the recordings still open around it."""
        complete = not rec.failed and len(rec.launchers) == expected
        if not complete:
            LaunchPlans.fail()
        if not ENABLED or key is None or not complete or _capturing():
            return
        if len(self.plans) >= self.MAX_PLANS:
            self.plans.clear()
        self.plans[key] = tuple(rec.launchers)

    def clear(self) -> None:
        """Drop every plan (a device move: the launchers captured the tables of the old device)."""
        self.plans.clear()

    @classmethod
    @contextmanager
    def record(cls, tag = None):
        """Open a recording with `tag` for the full path of one call; yields it."""
        rec = _Recording(tag)
        stack = cls._stack()
        stack.append(rec)
        try:
            yield rec
        finally:
            stack.pop()

    @classmethod
    def add(cls, launcher, tag = None) -> None:
        """Add `launcher` to the innermost recording of `tag` open now, if one is."""
        for rec in reversed(getattr(cls._local, "open", None) or ()):
            if rec.tag == tag:
                rec.launchers.append(launcher)
                return

    @classmethod
    def recording(cls, tag = None) -> bool:
        """Whether a recording of `tag` is open now (a site builds its launcher only then)."""
        return any(rec.tag == tag for rec in getattr(cls._local, "open", None) or ())

    @classmethod
    def nested(cls) -> bool:
        """Whether any recording is open now."""
        return bool(getattr(cls._local, "open", None))

    @classmethod
    def fail(cls) -> None:
        """Mark every recording open now as not replayable: a launch no site of theirs captured (a site without a
        launcher of its own, or an inner call that did not record itself completely)."""
        for rec in getattr(cls._local, "open", None) or ():
            rec.failed = True

    @classmethod
    def _stack(cls) -> list:
        stack = getattr(cls._local, "open", None)
        if stack is None:
            stack = cls._local.open = []
        return stack


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
