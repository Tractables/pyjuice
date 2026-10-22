"""
Decoding sessions (:meth:`ConstrainedCircuit.decoder`) on the dfa_exact backend
(:class:`~pyjuice.constraints.inference.dfa_exact.queries.decoder.DFAExactDecoder`): a scenario's decoder calls give
what the circuit's own calls give -- left to right, any order, beams reordered and expanded, a new row, a position
observed then missing again, other targets after rows moved; candidates, targets as positions, a mask id, the marginal
-- through every sum path; they hold what the evidence left unchanged, and only that, moving it when rows move
(compact, only the inside values a call reads: the others stay parked);
anything else using the buffers, or another precision, starts afresh, and so does a reset (parameter updates seen); an
HMM's decoding gives the circuit's rows bit for bit, and the reference's; (slow) it is faster than the circuit's own calls on an HMM4096. The session is a
fast_inference scope (its options reach it, it nests), one per circuit at a time, usable only inside its ``with``,
with query defaults a call overrides.
"""
import random
import time

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact import state as S
from pyjuice.constraints.inference.dfa_exact.kernels import compact as CP
from pyjuice.constraints.inference.dfa_exact.kernels.blocks import move_blocks, move_table
from pyjuice.constraints.inference.dfa_exact.kernels.live import liveness
from pyjuice.constraints.inference.dfa_exact.queries import decoder as QD
from pyjuice.utils.fast_inference import is_active

V = 3
KINDS = {"hmm": 6, "pd": 8, "hand_left": 5, "left_linear": 6}                  # see conftest.py


@pytest.fixture(params = ["fused", "dense", "dense_unpruned"])
def sum_path(request, monkeypatch):
    """As conditional_test.py's: the fused sum kernels, or the dense path with its products on the live columns only or
    on every column -- which reads the slots a held block computed under less evidence left finite."""
    from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
    from pyjuice.constraints.inference.dfa_exact.queries.backend import forward as F
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", (lambda ncols, live: False) if request.param == "dense_unpruned" else
                        (lambda ncols, live: live < ncols))
    return request.param


class Spy:
    """Every call's held values (None, or ``(inside, top_down)``, bool per sum region; None too when the call computed
    them again, held in the other storage than the one it picked), the block moves and those of live lanes only, the
    compact re-gathers of inside (``"mars"``) and top-down (``"flows"``) values, and the storage each call used."""

    def __init__(self, monkeypatch):
        self.held, self.moves, self.live_moves, self.regathers, self.storage = [], 0, 0, {"mars": 0, "flows": 0}, []
        self.plans, self.dropped = [], []
        conditional, marginal, move_blocks, regather = QD.conditional, QD.marginal, QD.move_blocks, CP.regather

        def record(held, report):
            self.held.append(None if held is None or report.get("dropped") else held)
            self.storage.append(report.get("storage"))
            self.plans.append(report.get("plan"))
            self.dropped.append(bool(report.get("dropped")))

        def spy_conditional(*args, held = None, report = None, **kwargs):
            report = {} if report is None else report
            out = conditional(*args, held = held, report = report, **kwargs)
            record(None if held is None else (held["inside"].copy(), held["top_down"].copy()), report)
            return out

        def spy_marginal(*args, held = None, report = None, **kwargs):
            report = {} if report is None else report
            out = marginal(*args, held = held, report = report, **kwargs)
            record(None if held is None else (held.copy(), np.zeros_like(held)), report)
            return out

        def spy_move_blocks(*args, **kwargs):
            self.moves += 1
            self.live_moves += kwargs.get("live") is not None
            return move_blocks(*args, **kwargs)

        def spy_regather(*args, rows = "flows", **kwargs):
            self.regathers[rows] += 1
            return regather(*args, rows = rows, **kwargs)

        monkeypatch.setattr(QD, "conditional", spy_conditional)
        monkeypatch.setattr(QD, "marginal", spy_marginal)
        monkeypatch.setattr(QD, "move_blocks", spy_move_blocks)
        monkeypatch.setattr(CP, "regather", spy_regather)

    def anything(self, call):
        return self.held[call] is not None and (self.held[call][0].any() or self.held[call][1].any())


def nan_new_storage(monkeypatch):
    """Every storage the backends' states allocate from now on is filled with NaN (see State.storage_view): a slot read
    before it is written -- computed, moved or re-gathered -- shows."""
    real = S.State.storage_view

    def storage_view(self, name, numel):
        old = self._storage.get(name)
        out = real(self, name, numel)
        if self._storage[name] is not old:
            self._storage[name].fill_(float("nan"))
        return out

    monkeypatch.setattr(S.State, "storage_view", storage_view)


def scopes(cc):
    """[sum regions, 2]: every sum region's scope ``[a, b)``."""
    prog = cc.backend.program()
    return np.asarray([prog.sum_ivs[prog.reg_iv_host[r]] for r in range(len(cc.backend.sum_regions))])


def assert_close(got, want, n, where = None):
    """Log-probabilities: the same -inf entries, the finite ones within fp32 rounding."""
    got, want = got.cpu(), want.cpu()
    assert got.shape == want.shape and not torch.isnan(got).any(), where
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), where
    fin = torch.isfinite(want)
    assert ((got[fin] - want[fin]).abs() <= 1e-5 * want[fin].abs() + n * 1e-6).all(), where


def left_to_right(n, B = 4, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    return [dict(data = data[:, :t], targets = [t]) for t in range(n)]


def any_order(n, B = 4, seed = 1):
    """Every sample unmasks its positions in an order of its own, one per call."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    order = torch.stack([torch.randperm(n, generator = g) for _ in range(B)])
    steps = []
    for s in range(n):
        missing = torch.ones(B, n, dtype = torch.bool)
        missing.scatter_(1, order[:, :s], False)
        steps.append(dict(data = data, missing_mask = missing))
    return steps


def beams(n, seed = 2):
    """Rows reordered, duplicated and dropped between calls, each then observing the next position."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (2, n), generator = g)
    steps = []
    for t in range(n):
        steps.append(dict(data = data[:, :t].clone(), targets = [t]))
        idx = torch.randint(0, data.size(0), (2 + t % 3,), generator = g)
        data = data[idx].clone()
        data[:, t] = torch.randint(0, V, (data.size(0),), generator = g)
    return steps


def new_and_forgotten(n, seed = 3):
    """A row replaced by a new prompt; a position observed, then missing again."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (3, n), generator = g)
    steps = [dict(data = data[:, :2]), dict(data = data[:, :3])]
    fresh = data.clone()
    fresh[1] = torch.randint(0, V, (n,), generator = g)
    steps.append(dict(data = fresh[:, :3]))
    missing = torch.zeros(3, n, dtype = torch.bool)
    missing[:, 3:] = True
    missing[0, 1] = True                                                      # observed before, missing now
    steps.append(dict(data = fresh, missing_mask = missing))
    return steps


def candidates_and_mask_id(n, B = 3, seed = 4):
    """Candidates per sample, and the evidence marked with a mask id of 7."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    steps = []
    for t in range(n):
        masked = data.clone()
        masked[:, t:] = 7
        steps.append(dict(data = masked, mask_id = 7, candidates = torch.randint(-1, V, (B, 4), generator = g)))
    return steps


def other_targets_after_a_move(n, seed = 5):
    """Nothing newly observed; the rows rotated every other call, the targets alternating: a call needs values the
    call before it, which moved only its own, did not."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (3, 2), generator = g)
    steps = []
    for k in range(6):
        if k % 2:
            data = data[torch.tensor([1, 2, 0])]
        steps.append(dict(data = data, targets = [2 if k % 2 == 0 else n - 1]))
    return steps


SCENARIOS = {"left to right": left_to_right, "any order": any_order, "beams": beams,
             "new and forgotten": new_and_forgotten, "candidates and mask id": candidates_and_mask_id,
             "other targets after a move": other_targets_after_a_move}


def ask(target, query, step):
    """A step's query at fp32 (where a decoder's calls are the circuit's own to the bit)."""
    if query == "marginal":
        return target.marginal(step["data"], step.get("missing_mask"), precision = "fp32",
                               **{k: step[k] for k in ("mask_id", "storage") if k in step})
    return target.conditional(**{"precision": "fp32", **step})


@pytest.mark.parametrize("storage, park", [("dense", "default"), ("compact", "default"), ("compact", "always")])
@pytest.mark.parametrize("query", ["conditional", "marginal"])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_decoders_calls_are_the_circuits(kind, scenario, query, storage, park, sum_path, build_pc, monkeypatch):
    """A scenario's calls through a decoder give the circuit's own calls' results (in a fast_inference scope of
    their own, dense). The storage values move into is filled with NaN: a query reading a slot before writing it would
    show. Compact (``storage``), the values held, inside and top-down, are re-gathered by the call into storage filled
    with NaN as well -- every one, or (``park`` "always", whatever their size) those it reads alone, the others
    left where they are (``compact_buffers``)."""
    monkeypatch.setattr(QD, "_empty", lambda numel, device: torch.full((numel,), float("nan"), device = device))
    if storage == "compact":
        nan_new_storage(monkeypatch)
    if park == "always":
        monkeypatch.setattr(CP, "PARK_MIN_FLOATS", 0)
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    steps = SCENARIOS[scenario](n)
    if query == "marginal":
        steps = [{k: v for k, v in s.items() if k in ("data", "missing_mask", "mask_id")} for s in steps]
    with juice.fast_inference():
        want = [ask(cc, query, s) for s in steps]
    with cc.decoder(storage = storage, precision = "fp32") as decoder:
        got = [ask(decoder, query, s) for s in steps]
    for i, (a, b) in enumerate(zip(got, want)):
        assert_close(a, b, n, where = f"step {i}")


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_a_compact_call_keeps_the_inside_values_it_has_rows_for(kind, build_pc):
    """Compact, a call keeps the inside values it has rows for, held or computed, and no other: none in an HMM's
    prefix decoding (every per-sample scope holds the prefix's last position, the others are kept once: node_mars
    holds the input rows alone), some in a PD's."""
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    kept_any = False
    with cc.decoder(storage = "compact") as decoder:
        for s in left_to_right(n)[1:]:
            decoder.conditional(**s)
            kept = decoder._kept
            assert not (kept.inside & ~kept.compact.mars["regions"]).any()
            kept_any |= bool(kept.inside.any())
        assert kept_any == (kind == "pd")


@pytest.mark.parametrize("scenario", ["left to right", "any order", "beams"])
def test_a_compact_call_moves_the_held_values_it_reads(scenario, build_pc, monkeypatch):
    """Compact, a call holding enough (here: any amount) re-gathers into its rows the held inside values its launches
    read (``read_regions``), and of the others those of every storage but one: the one holding the most stays
    (``parked``) until a call reads them -- a PD's do when every row extends its prefix (left to right, beams). A value
    is in one place only, and one moved is kept (not computed again); the calls give the circuit's results, the storage
    values move into filled with NaN."""
    from pyjuice.constraints.inference.dfa_exact.queries.backend.plans import read_regions
    nan_new_storage(monkeypatch)
    monkeypatch.setattr(CP, "PARK_MIN_FLOATS", 0)                       # (values of every size)
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    steps = SCENARIOS[scenario](n)
    calls, regather, conditional = [], CP.regather, QD.conditional

    def spy_regather(prog, old, src, new, dst, regions, parents, dst_offset = None, rows = "flows"):
        if rows == "mars":
            calls[-1]["moved"] |= regions
        return regather(prog, old, src, new, dst, regions, parents, dst_offset = dst_offset, rows = rows)

    def spy_conditional(*args, held = None, report = None, **kwargs):
        report = {} if report is None else report
        calls.append(dict(moved = np.zeros(len(cc.backend.sum_regions), dtype = bool)))
        out = conditional(*args, held = held, report = report, **kwargs)
        calls[-1].update(held = None if held is None else held["inside"].copy(), plan = report.get("plan"),
                         partial = report.get("partial"))
        return out

    monkeypatch.setattr(CP, "regather", spy_regather)
    monkeypatch.setattr(QD, "conditional", spy_conditional)
    with juice.fast_inference():
        want = [cc.conditional(precision = "fp32", **s) for s in steps]
    parked = []
    with cc.decoder(storage = "compact", precision = "fp32") as decoder:
        for i, s in enumerate(steps):
            assert_close(decoder.conditional(**s), want[i], n, where = f"step {i}")
            parked.append(np.any([p["inside"] for p in decoder._kept.parked], axis = 0) if decoder._kept.parked
                          else np.zeros(len(cc.backend.sum_regions), dtype = bool))
            assert len(decoder._kept.parked) <= 1 and not (parked[-1] & decoder._kept.inside).any()
            assert not (calls[-1]["moved"] & ~decoder._kept.inside).any()      # what it moved, kept: never recomputed
    prog = cc.backend.program()
    for call in calls:
        if call["held"] is not None:                                     # what it reads, of what it holds (and the
            read = call["held"] & read_regions(prog, call["plan"])       # regions some of its rows hold)
            moved = call["moved"] if call["partial"] is None else call["moved"] & ~call["partial"]
            assert not (moved & ~call["held"]).any() and not (read & ~call["moved"]).any()
    if scenario != "any order":
        assert any(p.any() for p in parked)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_the_rows_that_hold_a_regions_values_get_them_back(kind, sum_path, build_pc, monkeypatch):
    """Compact, every row unmasking a position of its own per call: a region some rows newly observe a position of is
    computed again, and the values the previous call left are re-gathered into its rows first -- those of the rows
    observing nothing new on its scope are what its launches compute for them (the others' are computed over; here
    every row's, ``hold`` withheld from the forward pass)."""
    from pyjuice.constraints.inference.dfa_exact.queries import conditional as QC
    from pyjuice.constraints.inference.dfa_exact.queries.backend import forward as F
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc.backend.program()
    seen, checked = {}, []
    partial_carry, run_forward = QC.partial_carry, F.run_forward

    def spy_partial_carry(prog, plan, carry):
        out = partial_carry(prog, plan, carry)
        seen.update(partial = out[1], fresh = None if carry is None else carry.get("fresh"))
        return out

    def spy_run_forward(backend, bufs, plan, x, missing, obs_class, lv, precision, compact = None, masses = None,
                        gates = None, hold = None):
        before = bufs["node_mars"].clone() if compact is not None and seen.get("partial") is not None else None
        out = run_forward(backend, bufs, plan, x, missing, obs_class, lv, precision, compact, masses, gates = gates)
        if before is not None:
            offset, width = compact.rows("mars")
            for r in np.nonzero(seen["partial"])[0]:
                off = compact.off[prog.reg_iv_all[r]].cpu().numpy()
                for b in np.nonzero(~seen["fresh"][r])[0]:
                    lanes = np.arange(off[b], off[b + 1])
                    at = offset[r] + np.arange(prog.reg_rows[r])[:, None] * width[r] + lanes[None, :]
                    at = torch.from_numpy(at.ravel()).to(before.device)
                    assert_close(before[at], bufs["node_mars"][at], n, where = f"region {r}, row {b}")
                    checked.append(lanes.size)
        return out

    monkeypatch.setattr(QC, "partial_carry", spy_partial_carry)
    monkeypatch.setattr(F, "run_forward", spy_run_forward)
    with cc.decoder(storage = "compact", precision = "fp32") as decoder:
        for s in any_order(n):
            seen.clear()
            decoder.conditional(**s)
    assert sum(checked) > 0                                              # (some rows held some region's values)


@pytest.mark.parametrize("path", ["dense", "fused"])
@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_sums_compute_the_lanes_of_the_rows_computing_theirs(kind, path, build_pc, monkeypatch):
    """Compact, every row unmasking a position of its own per call: the sums of a region some rows hold the inside
    values of compute the other rows' lanes alone, the held ones re-gathered and left as they are -- a dense run
    multiplies those columns, a fused part takes them as its columns (``fresh_order``) -- and the calls give the
    circuit's results, the storage values are re-gathered into filled with NaN."""
    from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
    from pyjuice.constraints.inference.dfa_exact.queries.backend import forward as F
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 if path == "dense" else 1 << 30)
    nan_new_storage(monkeypatch)
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    runs, orders, dense_sum, fused_sum, fresh_order = [], [], F.dense_sum, F.fused_sum, F.fresh_order

    def spy_dense_sum(*args, live = None, keep = False, **kwargs):
        if keep:
            runs.append(live[2] < args[9])                              # (its lanes multiplied, of its row's)
        return dense_sum(*args, live = live, keep = keep, **kwargs)

    def spy_fresh_order(layout, regions, fresh):
        out = fresh_order(layout, regions, fresh)
        orders.append(out)                                               # (some region's lanes, fewer than its row's)
        runs.append(bool((out.slots.cpu().numpy() < layout.total[layout.prog.reg_iv_all]).any()))
        return out

    def spy_fused_sum(*args, liveness = None, **kwargs):
        assert liveness is None or liveness is orders[-1]               # (compact: no lane order but that one)
        return fused_sum(*args, liveness = liveness, **kwargs)

    monkeypatch.setattr(F, "dense_sum", spy_dense_sum)
    monkeypatch.setattr(F, "fused_sum", spy_fused_sum)
    monkeypatch.setattr(F, "fresh_order", spy_fresh_order)
    steps = any_order(n)
    with juice.fast_inference():
        want = [cc.conditional(precision = "fp32", **s) for s in steps]
    with cc.decoder(storage = "compact", precision = "fp32") as decoder:
        for i, s in enumerate(steps):
            assert_close(decoder.conditional(**s), want[i], n, where = f"step {i}")
    assert runs and any(runs)


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_a_compact_decoder_over_a_wide_batch(kind, build_pc, monkeypatch):
    """24 rows left to right, then beams: compact, the held values' blocks span many tiles of the copies that carry
    them over into a call's layout (a tile of 16 floats here); the rows are the circuit's own."""
    from pyjuice.constraints.inference.dfa_exact.kernels import compact as CP
    monkeypatch.setattr(CP, "COPY_TILE", 16)
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    for steps in (left_to_right(n, B = 24), beams(n)):
        with juice.fast_inference():
            want = [cc.conditional(precision = "fp32", **s) for s in steps]
        with cc.decoder(storage = "compact", precision = "fp32") as decoder:
            got = [decoder.conditional(**s) for s in steps]
        for i, (a, b) in enumerate(zip(got, want)):
            assert_close(a, b, n, where = f"step {i}")


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_a_call_holds_what_the_evidence_left_unchanged(kind, storage, build_pc, monkeypatch):
    """Left to right, a call newly observes the position before its target: it holds inside values only of scopes
    without it, top-down values only of scopes with it (none newly observed outside), and some of them; in place,
    while reordered beams move theirs (the lanes live in the previous call). The marginal holds inside values alone
    -- none on an HMM, whose per-sample scopes are suffixes holding the new position. Compact, the same, but nothing
    moves: the values held, inside (a PD's) and top-down, are re-gathered into the call's layout."""
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    a, b = scopes(cc).T
    spy = Spy(monkeypatch)
    with cc.decoder(storage = storage) as decoder:
        for s in left_to_right(n):
            decoder.conditional(**s)
    assert spy.held[0] is None and spy.moves == 0                  # left to right: every row stays put
    for t, held in enumerate(spy.held[1:], 1):
        inside, top_down = held
        assert not (inside & (a <= t - 1) & (t - 1 < b)).any(), t
        assert not (top_down & ~((a <= t - 1) & (t - 1 < b))).any(), t
    assert any(held[1].any() for held in spy.held[1:])
    if kind == "pd":                                                # an HMM's inside values are kept once
        assert any(held[0].any() for held in spy.held[1:])
    spy.held.clear()
    with cc.decoder(storage = storage) as decoder:
        for s in beams(n):
            decoder.conditional(**s)
    assert any(held is not None and held[1].any() for held in spy.held[1:])   # top-down values held across moves
    if storage == "compact":
        assert spy.moves == 0 and spy.regathers["flows"] > 0 and (spy.regathers["mars"] > 0) == (kind == "pd")
    elif kind == "pd":
        assert spy.live_moves > 0                                   # the previous call's live lanes only
    spy.held.clear()
    with cc.decoder(storage = storage) as decoder:
        for s in left_to_right(n):
            decoder.marginal(s["data"])
    assert all(not held[1].any() for held in spy.held[1:])
    assert any(spy.anything(c) for c in range(1, n)) == (kind == "pd")   # an HMM's scopes are suffixes


@pytest.mark.parametrize("kind, query", [("pd", "conditional"), ("pd_prod_dominated", "conditional"),
                                         ("hand", "conditional"), ("hand", "marginal")])   # (a PD's marginals: compact)
def test_an_auto_decoder_switching_storage_computes_its_values_again(kind, query, build_pc, monkeypatch):
    """``storage = "auto"`` (the default), any-order decoding with a 48-state automaton: compact, then dense once every
    sample observes one position (its live columns joined in few pairs), then compact again -- a call that picks the
    other storage than the previous one computes the values held in that one again (into storage filled with NaN), and
    keeps no other value than those it computed. The rows are the circuit's own."""
    nan_new_storage(monkeypatch)
    rng = random.Random(0)
    K, n = 48, {"pd": 8, "pd_prod_dominated": 8, "hand": 5}[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    steps = any_order(n)
    with juice.fast_inference():
        want = [ask(cc, query, dict(s, storage = "dense")) for s in steps]
    spy = Spy(monkeypatch)
    reg_iv = cc.backend.program().reg_iv_np
    got = []
    with cc.decoder(precision = "fp32") as decoder:
        for s in steps:
            got.append(ask(decoder, query, s))
            if spy.dropped[-1]:                                         # only what the call computed is kept
                plan = spy.plans[-1]
                assert not (decoder._kept.inside & ~plan.sample[reg_iv]).any()
                assert plan.top_down is None or not (decoder._kept.top_down & ~plan.top_down.sample).any()
    for i, (a, b) in enumerate(zip(got, want)):
        assert_close(a, b, n, where = f"step {i}")
    used = spy.storage
    assert used[0] == "compact" and "dense" in used and used[-1] == "compact"
    switches = [c for c in range(1, n) if used[c] != used[c - 1]]
    assert all(not spy.anything(c) for c in switches)                  # each switch: nothing held
    assert any(spy.anything(c) for c in range(1, n) if c not in switches)   # ... else held


def test_a_call_that_computed_its_held_values_again_keeps_only_what_it_computed(build_pc):
    """A call that computed again the values held in the other storage than the one it picked (``report["dropped"]``)
    keeps none of them, only what it computed (here nothing); one that did not keeps them."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    ones = np.ones(len(cc.backend.sum_regions), dtype = bool)
    with cc.decoder() as decoder:
        decoder._after(4, "fp32", "auto", (ones, ones), {"storage": "dense", "dropped": True}, None)
        assert not decoder._kept.inside.any() and not decoder._kept.top_down.any()
        decoder._after(4, "fp32", "auto", (ones, ones), {"storage": "dense"}, None)
        assert decoder._kept.inside.all() and decoder._kept.top_down.all() and decoder._kept.storage == "dense"


def test_anything_else_using_the_buffers_or_another_precision_starts_afresh(build_pc, monkeypatch):
    """A circuit's own query between two calls, inside the session or not, overwrites the values the decoder would
    hold, and a call in another precision or storage computes them otherwise: the next call holds nothing, and is
    right."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    steps = left_to_right(n)
    other = left_to_right(n, seed = 9)
    with juice.fast_inference():
        want = [cc.conditional(precision = "fp32", **s) for s in steps]
    spy = Spy(monkeypatch)
    with cc.decoder(storage = "dense", precision = "fp32") as decoder:
        for t in range(4):
            decoder.conditional(**steps[t])
        assert spy.anything(3)
        cc.conditional(precision = "fp32", **other[4])                                  # the same regions, other values
        assert_close(decoder.conditional(**steps[4]), want[4], n)
        assert spy.held[-1] is None
        cc.marginal(other[5]["data"])
        assert_close(decoder.conditional(**steps[5]), want[5], n)
        assert spy.held[-1] is None
        decoder.conditional(**steps[6], precision = "tf32")
        assert_close(decoder.conditional(**steps[7]), want[7], n)
        assert spy.held[-1] is None
    with cc.decoder(storage = "dense", precision = "fp32") as decoder:
        for t in range(4):
            decoder.conditional(**steps[t])
        decoder.conditional(**steps[4], storage = "compact")
        assert spy.held[-1] is None
        assert_close(decoder.conditional(**steps[5]), want[5], n)
        assert spy.held[-1] is None


@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_reset_starts_afresh(storage, build_pc, monkeypatch):
    """``decoder.reset()``: the next call holds nothing and is right, the one after holds again; parameters updated in
    place and then a reset: the next call answers under the new ones, where without the reset it would not; outside
    the ``with``, a reset is refused."""
    import pyjuice.constraints.inference.dfa_exact.backend as BE
    monkeypatch.setattr(BE, "ENUMERATE_MAX_TOKENS", 0)   # every class stored: the kept class masses carry the old
    n = KINDS["pd"]                                       # parameters (an enumerated class's kept value is the token
    pc, other = build_pc("pd", n, V, seed = 0), build_pc("pd", n, V, seed = 1)   # table, which these rows barely
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)                           # depend on)
    steps = left_to_right(n)
    with juice.fast_inference():
        want = [cc.conditional(precision = "fp32", **s) for s in steps]
    spy = Spy(monkeypatch)
    with cc.decoder(storage = storage, precision = "fp32") as decoder:
        for t in range(4):
            decoder.conditional(**steps[t])
        assert spy.anything(3)
        decoder.reset()
        assert is_active()                                          # (still the session's scope)
        assert_close(decoder.conditional(**steps[4]), want[4], n)
        assert spy.held[-1] is None
        assert_close(decoder.conditional(**steps[5]), want[5], n)
        assert spy.anything(-1)
        with torch.no_grad():                                       # parameters updated in place
            pc.params.copy_(other.params)
            for layer, src in zip(pc.input_layer_group, other.input_layer_group):
                layer.params.copy_(src.params)
        stale = decoder.conditional(**steps[6])
        decoder.reset()
        fresh = decoder.conditional(**steps[6])
    new = cc.conditional(precision = "fp32", **steps[6])
    assert_close(fresh, new, n)
    fin = torch.isfinite(new)
    assert (stale[fin] - new[fin]).abs().max() > 1e-2                # (the session kept the old parameters' values)
    with pytest.raises(RuntimeError, match = "inside its `with` block"):
        decoder.reset()


def test_evidence_on_the_device(build_pc):
    """Data, missing masks, targets and candidates on the GPU (``input_ids`` as a language model's loop has them): the
    rows the same evidence on the host gives, bit for bit."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    cuda = lambda v: v.cuda() if isinstance(v, torch.Tensor) else v
    for scenario in ("any order", "beams", "candidates and mask id"):
        steps = SCENARIOS[scenario](n)
        with cc.decoder() as decoder:
            want = [decoder.conditional(**s) for s in steps]
        with cc.decoder() as decoder:
            got = [decoder.conditional(**{k: cuda(v) for k, v in s.items()}) for s in steps]
        assert all(torch.equal(a, b) for a, b in zip(got, want)), scenario


def test_moving_blocks():
    """Regions of many rows (several programs' worth) and columns (several tiles), in two buffers, at offsets and row
    widths of their own: gathered by parent (rows duplicated and dropped), then placed as they are."""
    g = torch.Generator().manual_seed(0)
    B_old, B, slots, rows = 5, 170, (7, 3), (41, 18)                 # 170 x 7 columns: two tiles
    src = [torch.randn(3000, generator = g).cuda() for _ in range(2)]
    parents = torch.randint(0, B_old, (B,), generator = g)
    gather = [(1000, B_old * slots[0] + 3, 0, B * slots[0], slots[0], rows[0], 0),        # both into one scratch
              (1500, B_old * slots[1] + 3, rows[0] * B * slots[0], B * slots[1], slots[1], rows[1], 1)]
    temp = torch.full((sum(r * B * k for r, k in zip(rows, slots)),), float("nan"), device = "cuda")
    move_blocks(src, (temp, temp), torch.tensor(move_table(*np.asarray(gather).T), device = "cuda"), B, max(rows),
                max(slots), parents = parents.cuda())
    for so, sw, to, _, k, r, which in gather:
        old = src[which][so:so + r * sw].view(r, sw)[:, :B_old * k].view(r, B_old, k)
        assert torch.equal(temp[to:to + r * B * k].view(r, B, k), old[:, parents.cuda()])
    place = [(to, B * k, do, B * k + 9, k, r, which) for (_, _, to, _, k, r, which), do in zip(gather, (3, 100))]
    dst = [torch.full((do + r * dw,), float("nan"), device = "cuda") for _, _, do, dw, _, r, _ in place]
    move_blocks((temp, temp), dst, torch.tensor(move_table(*np.asarray(place).T), device = "cuda"), B, max(rows),
                max(slots))
    for to, _, do, dw, k, r, which in place:
        assert torch.equal(dst[which][do:].view(r, dw)[:, :B * k], temp[to:to + r * B * k].view(r, B * k))


def test_moving_live_lanes_only(build_pc):
    """Given the source call's liveness (its flags: a lane is live where both ends of its slot's pair are), a
    destination lane gets its parent's value where that lane is live, and -inf where it is dead -- over an interval
    that keeps every pair and one that does not, against the liveness's own lane order."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    prog, layout = cc.backend.program(), cc.backend.layout
    g = torch.Generator().manual_seed(1)
    B_old, B = 5, 170
    data = torch.randint(0, V, (B_old, n), generator = g)
    missing = torch.rand(B_old, n, generator = g) < 0.5
    x = torch.where(missing, 0, data).cuda()
    oc = torch.where(missing.cuda(), -1, cc.backend.token_classes.token_class[x]).contiguous()
    lv = liveness(prog, oc, B_old)
    order = lv.order(np.ones(len(prog.sum_ivs), dtype = bool))
    slots = np.asarray(prog.sum_iv_slots)
    pick = lambda full: max((k for k, (a, b) in enumerate(prog.sum_ivs)
                             if (layout.reachability.positions(a, b) is None) == full), key = lambda k: slots[k])
    ks, rows = (pick(True), pick(False)), (41, 18)
    S = [int(slots[k]) for k in ks]
    src = [torch.randn(1000 + r * (B_old * s + 3), generator = g).cuda() for r, s in zip(rows, S)]
    parents = torch.randint(0, B_old, (B,), generator = g)
    moves = [(1000, B_old * S[0] + 3, 0, B * S[0], S[0], rows[0], 0, *prog.sum_ivs[ks[0]]),
             (1000, B_old * S[1] + 3, rows[0] * B * S[0], B * S[1], S[1], rows[1], 1, *prog.sum_ivs[ks[1]])]
    dst = torch.full((sum(r * B * s for r, s in zip(rows, S)),), float("nan"), device = "cuda")
    move_blocks(src, (dst, dst), torch.tensor(move_table(*np.asarray(moves).T[:7], np.asarray(moves)[:, 7:]),
                                              device = "cuda"),
                B, max(rows), max(S), parents = parents.cuda(), live = lv.live, prog = prog)
    rank = order.rank.cpu()
    for (so, sw, do, _, s, r, which, _, _), k in zip(moves, ks):
        old = src[which][so:so + r * sw].view(r, sw)[:, :B_old * s].view(r, B_old, s).cpu()
        alive = (rank[order.first[k]:order.first[k] + B_old * s] >= 0).view(B_old, s)[parents]          # [B, s]
        want = torch.where(alive, old[:, parents], float("-inf"))
        assert torch.equal(dst[do:do + r * B * s].view(r, B, s).cpu(), want), k
        assert alive.any() and not alive.all()


@pytest.mark.parametrize("kind", ["hmm", "hmm_untied", "hmm_block_sparse"])
def test_hmm_decoding_gives_the_circuits_rows(kind, sum_path, build_pc, reference):
    """An HMM decoded left to right: every step's rows are those of the circuit's own query, bit for bit (the held
    values are the ones it computes), and the reference's."""
    n = 8
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    data = torch.randint(0, V, (4, n), generator = torch.Generator().manual_seed(0))
    with juice.fast_inference():
        want = [cc.conditional(data[:, :t], targets = [t], precision = "fp32") for t in range(n)]
    with cc.decoder(precision = "fp32") as decoder:
        got = [decoder.conditional(data[:, :t], targets = [t]) for t in range(n)]
    for t in range(n):
        assert torch.equal(got[t], want[t]), t
        missing = torch.zeros(4, n, dtype = torch.bool)
        missing[:, t:] = True
        targets = torch.zeros_like(missing)
        targets[:, t] = True
        assert_close(got[t], reference.conditional(cc, data, missing, targets = targets), n)


def test_the_session_is_a_fast_inference_scope(build_pc):
    """Open inside, closed after; its options reach the scope (bf16 token tables); it nests in a caller's scope."""
    n = KINDS["hmm"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    data = torch.randint(0, V, (2, n))
    assert not is_active()
    with cc.decoder(allow_bf16_params = True) as decoder:
        assert is_active()
        decoder.conditional(data[:, :2])
        assert all(t.dtype == torch.bfloat16 for t in cc.backend.state._token_tables.values())
    assert not is_active() and cc.backend.state._token_tables is None
    with juice.fast_inference():
        with cc.decoder() as decoder:
            decoder.conditional(data[:, :2])
        assert is_active()


@pytest.mark.parametrize("option", ["auto", True, False])
def test_the_bf16_token_table_option(option, build_pc):
    """``bf16_token_table``: the conditionals' token tables in bfloat16 -- "auto" (the default) at tf32 only -- the
    scores within ~1e-2 of the fp32 table's; never outside the session."""
    n = KINDS["hmm"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    data = torch.randint(0, V, (2, n), generator = torch.Generator().manual_seed(0))
    for precision in ("fp32", "tf32"):
        want = cc.conditional(data[:, :2], precision = precision)
        kw = {} if option == "auto" else dict(bf16_token_table = option)
        with cc.decoder(**kw) as decoder:
            got = decoder.conditional(data[:, :2], precision = precision)
            dtypes = {t.dtype for t in cc.backend.state._token_tables.values()}
        bf16 = option is True or (option == "auto" and precision == "tf32")
        assert dtypes == {torch.bfloat16 if bf16 else torch.float32}, (precision, dtypes)
        fin = torch.isfinite(want)
        assert torch.equal(fin, torch.isfinite(got)) and ((got - want)[fin].abs() < (1e-2 if bf16 else 1e-5)).all()
        assert cc.backend.state.bf16_tokens is False


def test_one_session_at_a_time_inside_its_with(build_pc):
    n = KINDS["hmm"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    data = torch.randint(0, V, (2, n))
    decoder = cc.decoder()
    with pytest.raises(RuntimeError, match = "with"):
        decoder.conditional(data[:, :2])
    with decoder:
        with pytest.raises(RuntimeError, match = "open decoder"):
            with cc.decoder():
                pass
    with pytest.raises(RuntimeError, match = "with"):
        decoder.conditional(data[:, :2])
    with pytest.raises(RuntimeError, match = "once"):
        with decoder:
            pass
    with cc.decoder() as other:                                               # a new one opens fine
        other.marginal(data[:, :2])


def test_query_defaults_and_a_calls_own_options(build_pc):
    """Options other than fast_inference's are defaults for every call (precision, mask_id); a call's own wins (the
    token table fp32 as the circuit's own queries score it: not the decoder's default at tf32)."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V, (3, n), generator = g)
    data[:, n // 2:] = 7
    with juice.fast_inference():
        tf32 = cc.conditional(data, mask_id = 7, precision = "tf32")
        fp32 = cc.conditional(data, mask_id = 7, precision = "fp32")
    with cc.decoder(precision = "tf32", mask_id = 7, bf16_token_table = False) as decoder:
        assert torch.equal(decoder.conditional(data), tf32)
        assert torch.equal(decoder.conditional(data, precision = "fp32"), fp32)
        assert torch.equal(decoder.conditional(torch.where(data == 7, -1, data), mask_id = -1), tf32)


def test_conditionals_default_to_bf16_and_marginals_to_fp32(build_pc, monkeypatch):
    """A conditional's precision defaults to "bf16" and a marginal's to "fp32", the circuit's queries and a decoder's
    alike: a session mixing them computes again at each switch (its values are the other precision's), and reuses
    across them under a precision of its own."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    steps = left_to_right(n)
    with juice.fast_inference():
        assert torch.equal(cc.conditional(**steps[3]), cc.conditional(**steps[3], precision = "bf16"))
        assert torch.equal(cc.marginal(steps[3]["data"]), cc.marginal(steps[3]["data"], precision = "fp32"))
    spy = Spy(monkeypatch)
    for options in ({}, dict(precision = "fp32")):
        spy.held.clear()
        with cc.decoder(storage = "dense", **options) as decoder:
            for t in range(4):
                decoder.conditional(**steps[t])
            decoder.marginal(steps[4]["data"])
            decoder.conditional(**steps[5])
        assert spy.anything(3)                                          # conditionals after conditionals
        assert spy.anything(4) == spy.anything(5) == bool(options)      # the switches: a precision of its own


@pytest.mark.slow
def test_decoding_speed(capsys):
    """An HMM4096 over 50257 tokens and 32 positions, under a 60-state, 10-class automaton (three keywords, all
    required -- the size of Ctrl-G's first example's), decoded left to right at batch 1, 16 and 128, and as beams (rows
    permuted every step) at 16, through a decoder and through the circuit's own calls inside fast_inference (fp32 and
    bf16 token tables): the same rows, and the decoder's steps are faster. Prints the milliseconds per step."""
    n, Vb = 32, 50257
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = n, num_latents = 4096, num_emits = Vb)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to("cuda")
    dfa = jc.DFA.contains([[11, 12, 13]], Vb) & jc.DFA.contains([[21, 22, 23, 24]], Vb) & \
        jc.DFA.contains([[31, 32]], Vb)
    cc = jc.compile(dfa, pc)
    report = []
    for B, beams in ((1, False), (16, False), (128, False), (16, True)):
        g = torch.Generator().manual_seed(B)
        data = torch.randint(0, Vb, (B, n), generator = g).cuda()
        order = [torch.arange(B)]
        for t in range(1, n):
            order.append(order[-1][torch.randperm(B, generator = g)] if beams else order[-1])
        for bf16 in (False, True):
            times, rows = {}, {}
            for mode in ("decoder", "circuit"):
                ts = []
                for rep in range(2):                                        # the second pass: plans kept
                    scope = cc.decoder(allow_bf16_params = bf16) if mode == "decoder" else \
                        juice.fast_inference(allow_bf16_params = bf16)
                    with scope as decoder:
                        query = decoder if mode == "decoder" else cc
                        for t in range(n):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            out = query.conditional(data[order[t]][:, :t], targets = [t], precision = "fp32")
                            torch.cuda.synchronize()
                            if rep == 1:
                                ts.append(time.perf_counter() - t0)
                                rows[mode, t] = out
                times[mode] = 1e3 * float(np.median(ts[8:]))
            for t in range(n):
                assert_close(rows["decoder", t], rows["circuit", t], n, where = t)
            assert times["decoder"] < times["circuit"]
            report.append(f"B={B:4d}{' beams' if beams else '      '} {'bf16' if bf16 else 'fp32'} table: decoder "
                          f"{times['decoder']:.3f} ms/step, circuit {times['circuit']:.3f} ms/step")
    with capsys.disabled():
        print("\n" + "\n".join(report))
