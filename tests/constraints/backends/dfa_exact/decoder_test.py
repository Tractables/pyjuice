"""
Decoding sessions (:meth:`ConstrainedCircuit.decoder`) on the dfa_exact backend
(:class:`~pyjuice.constraints.backends.dfa_exact.queries.decoder.DFAExactDecoder`): a scenario's decoder calls give
what the circuit's own calls give -- left to right, any order, beams reordered and expanded, a new row, a position
observed then missing again, other targets after rows moved; candidates, targets as positions, a mask id, the marginal
-- through every sum path; they hold what the evidence left unchanged, and only that, moving it when rows move;
anything else using the buffers, or another precision, starts afresh; an HMM's decoding gives the circuit's rows bit
for bit, and the reference's; (slow) it is faster than the circuit's own calls on an HMM4096. The session is a
fast_inference scope (its options reach it, it nests), one per circuit at a time, usable only inside its ``with``,
with query defaults a call overrides.
"""
import time

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.backends.dfa_exact.kernels.blocks import move_blocks, move_table
from pyjuice.constraints.backends.dfa_exact.kernels.live import liveness
from pyjuice.constraints.backends.dfa_exact.queries import decoder as QD
from pyjuice.utils.fast_inference import is_active

V = 3
KINDS = {"hmm": 6, "pd": 8, "hand_left": 5, "left_linear": 6}                  # see conftest.py


@pytest.fixture(params = ["fused", "dense", "dense_unpruned"])
def sum_path(request, monkeypatch):
    """As conditional_test.py's: the fused sum kernels, or the dense path with its products on the live columns only or
    on every column -- which reads the slots a held block computed under less evidence left finite."""
    from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
    from pyjuice.constraints.backends.dfa_exact.queries import forward as F
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", (lambda ncols, live: False) if request.param == "dense_unpruned" else
                        (lambda ncols, live: live < ncols))
    return request.param


class Spy:
    """Every call's held values (None, or ``(inside, top_down)``, bool per sum region), and the block moves and those
    of live lanes only."""

    def __init__(self, monkeypatch):
        self.held, self.moves, self.live_moves = [], 0, 0
        conditional, marginal, move_blocks = QD.conditional, QD.marginal, QD.move_blocks

        def spy_conditional(*args, held = None, **kwargs):
            self.held.append(None if held is None else (held["inside"].copy(), held["top_down"].copy()))
            return conditional(*args, held = held, **kwargs)

        def spy_marginal(*args, held = None, **kwargs):
            self.held.append(None if held is None else (held.copy(), np.zeros_like(held)))
            return marginal(*args, held = held, **kwargs)

        def spy_move_blocks(*args, **kwargs):
            self.moves += 1
            self.live_moves += kwargs.get("live") is not None
            return move_blocks(*args, **kwargs)

        monkeypatch.setattr(QD, "conditional", spy_conditional)
        monkeypatch.setattr(QD, "marginal", spy_marginal)
        monkeypatch.setattr(QD, "move_blocks", spy_move_blocks)

    def anything(self, call):
        return self.held[call] is not None and (self.held[call][0].any() or self.held[call][1].any())


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
    if query == "marginal":
        return target.marginal(step["data"], step.get("missing_mask"), **({"mask_id": step["mask_id"]}
                                                                          if "mask_id" in step else {}))
    return target.conditional(**step)


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("query", ["conditional", "marginal"])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_decoders_calls_are_the_circuits(kind, scenario, query, storage, sum_path, build_pc, monkeypatch):
    """A scenario's calls through a decoder give the circuit's own calls' results (in a fast_inference scope of
    their own, dense). The storage values move into is filled with NaN: a query reading a slot before writing it would
    show. Compact (``storage``), the top-down values held are re-gathered by the call."""
    monkeypatch.setattr(QD, "_empty", lambda numel, device: torch.full((numel,), float("nan"), device = device))
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    steps = SCENARIOS[scenario](n)
    if query == "marginal":
        steps = [{k: v for k, v in s.items() if k in ("data", "missing_mask", "mask_id")} for s in steps]
    with juice.fast_inference():
        want = [ask(cc, query, s) for s in steps]
    with cc.decoder(storage = storage) as decoder:
        got = [ask(decoder, query, s) for s in steps]
    for i, (a, b) in enumerate(zip(got, want)):
        assert_close(a, b, n, where = f"step {i}")


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_a_compact_decoder_over_a_wide_batch(kind, build_pc, monkeypatch):
    """24 rows left to right, then beams: compact, the held values' blocks span many tiles of the copies that carry
    them over into a call's layout (a tile of 16 floats here); the rows are the circuit's own."""
    from pyjuice.constraints.backends.dfa_exact.kernels import compact as CP
    monkeypatch.setattr(CP, "COPY_TILE", 16)
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    for steps in (left_to_right(n, B = 24), beams(n)):
        with juice.fast_inference():
            want = [cc.conditional(**s) for s in steps]
        with cc.decoder(storage = "compact") as decoder:
            got = [decoder.conditional(**s) for s in steps]
        for i, (a, b) in enumerate(zip(got, want)):
            assert_close(a, b, n, where = f"step {i}")


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_a_call_holds_what_the_evidence_left_unchanged(kind, storage, build_pc, monkeypatch):
    """Left to right, a call newly observes the position before its target: it holds inside values only of scopes
    without it, top-down values only of scopes with it (none newly observed outside), and some of them; in place,
    while reordered beams move theirs (the lanes live in the previous call). The marginal holds inside values alone
    -- none on an HMM, whose per-sample scopes are suffixes holding the new position. Compact, the same: the top-down
    values held are re-gathered into the call's layout, the inside values move."""
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
    if kind == "pd":
        assert spy.live_moves > 0                                   # the previous call's live lanes only
    spy.held.clear()
    with cc.decoder(storage = storage) as decoder:
        for s in left_to_right(n):
            decoder.marginal(s["data"])
    assert all(not held[1].any() for held in spy.held[1:])
    assert any(spy.anything(c) for c in range(1, n)) == (kind == "pd")   # an HMM's scopes are suffixes


def test_anything_else_using_the_buffers_or_another_precision_starts_afresh(build_pc, monkeypatch):
    """A circuit's own query between two calls, inside the session or not, overwrites the values the decoder would
    hold, and a call in another precision or storage computes them otherwise: the next call holds nothing, and is
    right."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    steps = left_to_right(n)
    other = left_to_right(n, seed = 9)
    with juice.fast_inference():
        want = [cc.conditional(**s) for s in steps]
    spy = Spy(monkeypatch)
    with cc.decoder() as decoder:
        for t in range(4):
            decoder.conditional(**steps[t])
        assert spy.anything(3)
        cc.conditional(**other[4])                                  # the same regions, other values
        assert_close(decoder.conditional(**steps[4]), want[4], n)
        assert spy.held[-1] is None
        cc.marginal(other[5]["data"])
        assert_close(decoder.conditional(**steps[5]), want[5], n)
        assert spy.held[-1] is None
        decoder.conditional(**steps[6], precision = "tf32")
        assert_close(decoder.conditional(**steps[7]), want[7], n)
        assert spy.held[-1] is None
    with cc.decoder() as decoder:
        for t in range(4):
            decoder.conditional(**steps[t])
        decoder.conditional(**steps[4], storage = "compact")
        assert spy.held[-1] is None
        assert_close(decoder.conditional(**steps[5]), want[5], n)
        assert spy.held[-1] is None


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
        want = [cc.conditional(data[:, :t], targets = [t]) for t in range(n)]
    with cc.decoder() as decoder:
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
    """Options other than fast_inference's are defaults for every call (precision, mask_id); a call's own wins."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V, (3, n), generator = g)
    data[:, n // 2:] = 7
    with juice.fast_inference():
        tf32 = cc.conditional(data, mask_id = 7, precision = "tf32")
        fp32 = cc.conditional(data, mask_id = 7)
    with cc.decoder(precision = "tf32", mask_id = 7) as decoder:
        assert torch.equal(decoder.conditional(data), tf32)
        assert torch.equal(decoder.conditional(data, precision = "fp32"), fp32)
        assert torch.equal(decoder.conditional(torch.where(data == 7, -1, data), mask_id = -1), tf32)


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
                            out = query.conditional(data[order[t]][:, :t], targets = [t])
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
