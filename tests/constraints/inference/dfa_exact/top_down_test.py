"""
The lifted top-down pass (:mod:`pyjuice.constraints.inference.dfa_exact.queries.backend.top_down`): every sum node's
top-down value (its log outside value) against the test reference's (``reference.top_down``), on every kind of PC in
conftest.py's ``PC_KINDS``, under constraints and every pattern of evidence (pruned, and unpruned where nothing is
observed), through both sum paths, and where block @ block's shifts flush entries; run twice, the values are identical.
Under query plans (:mod:`~pyjuice.constraints.inference.dfa_exact.queries.backend.plans`): with every policy of keeping
values once, a query computes exactly the values its targets need; values kept once accumulate across queries,
append-only; held values are not computed again. Compact, the ``input @ block`` steps' CUDA walk (``kernels/c/walk.cu``)
gives the Triton kernel's values bit for bit, in every case it has.
"""
import contextlib
import math
import random

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.constraints.inference.dfa_exact.kernels import prod_top_down
from pyjuice.constraints.inference.dfa_exact.kernels.c import get_module
from pyjuice.constraints.inference.dfa_exact.queries.backend import forward as F
from pyjuice.constraints.inference.dfa_exact.queries.backend.top_down import run_backward
from pyjuice.constraints.inference.dfa_exact.kernels.compact import compact_layout
from pyjuice.constraints.inference.dfa_exact.kernels.live import liveness
from pyjuice.constraints.inference.dfa_exact.queries.backend.plans import (per_sample_intervals, prior_regions,
                                                                         query_plan, top_down_prior)

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8,
         "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5,
         "hand_shared": 4, "hand_fold": 4}                                                      # see conftest.py
PATTERNS = ["none", "all", "prefix", "suffix", "scattered", "mixed"]
# keep nothing once; as a query decides; every input-independent value (fast_inference); every input-independent
# top-down value and no inside value (so the inside values its launches for a batch of one read must be kept once)
POLICIES = ["sample", "default", "kept", "top_down"]
# no block @ block product: compact top-down values are dense's bit for bit through the fused sums (a block @ block one
# sums the live shared columns only, in other groups)
BIT_EXACT_KINDS = ["hmm", "hmm_untied", "hmm_block_sparse", "left_linear", "hand_left"]
CONSTRAINTS = {
    "contains": lambda: jc.DFA.contains([[1, 2]], V),
    "either": lambda: jc.DFA.contains([[2]], V) | jc.DFA.contains([[0, 0]], V),
}


def evidence(pattern, n, B = 6, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.ones(B, n, dtype = torch.bool)
    for b in range(B):
        p = PATTERNS[b % 5 + 1] if pattern == "mixed" else pattern
        if p == "all":
            missing[b] = False
        elif p == "prefix":
            missing[b, :n // 2] = False
        elif p == "suffix":
            missing[b, n // 2:] = False
        elif p == "scattered":
            missing[b] = torch.rand(n, generator = g) < 0.5
    return data, missing


@pytest.fixture(params = ["fused", "dense", "dense_unpruned"])
def sum_path(request, monkeypatch):
    """The fused sum kernels, or the dense path with its products on the live columns only or on every column (the
    dead ones read too: they must hold -inf)."""
    from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", (lambda ncols, live: False) if request.param == "dense_unpruned" else
                        (lambda ncols, live: live < ncols))
    return request.param


def top_down_blocks(cc, data, missing, targets = None, policy = "default", held = None, nan_fill = True,
                    visit = None, storage = "dense"):
    """
    The forward then the top-down pass of a query under its plan (``policy``: see ``POLICIES``; "kept" runs inside
    :func:`pyjuice.fast_inference`), node_flows filled with NaN first (``nan_fill``): ``(blocks, plan)``, ``blocks``
    every sum region's rows of the buffer its values are in -- [rows, B * slots], those kept once repeated per sample
    (cloned). ``visit(prod_index, bufs, flows, plan)``: see
    :func:`~pyjuice.constraints.inference.dfa_exact.queries.backend.top_down.run_backward`. ``storage = "compact"``: the
    per-sample values compact, expanded to [rows, B * slots] (-inf off the lanes; every lane written, -inf at a pair the
    automaton does not join), the inside values too.
    """
    with juice.fast_inference() if policy == "kept" else contextlib.nullcontext():
        prog, td, dev = cc.backend.program(), cc.backend.top_down_program(), cc.pc.params.device
        B = data.size(0)
        missing = missing.to(dev)
        observed = (~missing).any(dim = 0).cpu().numpy()
        x = torch.where(missing, 0, data.to(dev).long())
        obs_class = torch.where(missing, -1, cc.backend.token_classes.token_class[x]).contiguous()
        lv = liveness(prog, obs_class, B) if observed.any() or storage == "compact" else None
        valid, kept = cc.backend.state.prior_validity()
        if policy in ("sample", "top_down"):
            prior = np.zeros(prog.iv_bounds.shape[0], dtype = bool)
            once = top_down_prior(prog, td, observed, B, True, lv) if policy == "top_down" else \
                np.zeros(td.iv_bounds.shape[0], dtype = bool)
        else:
            prior = prior_regions(prog, observed, B, kept, lv)
            once = top_down_prior(prog, td, observed, B, kept, lv)
        pf = cc.backend.state.prior_flows(np.zeros(len(cc.backend.sum_regions), dtype = bool))
        plan = query_plan(prog, observed, None if targets is None else targets.cpu().numpy(), valid.numpy(), prior,
                          top_down = dict(program = td, once = once, valid = pf["valid"].copy(), held = held))
        compact = storage == "compact"
        if lv is not None and not compact:
            lv.order(per_sample_intervals(prog, plan))
        cl = compact_layout(cc.backend, lv, plan.top_down.per_sample,
                            inside = plan.sample[prog.reg_iv_np]) if compact else None
        bufs = cc.backend.state.buffers(B, layout = cl.mars if compact else None)
        flows = cc.backend.state.flows(B, bufs, cl)
        if nan_fill:
            flows["node_flows"].fill_(float("nan"))         # every needed slot must be written
        with torch.no_grad():
            F.run_forward(cc.backend, bufs, plan, x, missing, obs_class, lv, "fp32", cl)
            valid |= plan.todo
            run_backward(cc.backend, bufs, flows, plan, x, missing, obs_class, lv, "fp32",
                         visit = None if visit is None else lambda pi: visit(pi, bufs, flows, plan), compact = cl)
        lay, nf, pf = bufs["layout"], flows["node_flows"], cc.backend.state._prior_flows_state
        out = []
        for r, (first, end, slots) in enumerate(cc.backend.sum_regions):
            if plan.top_down.once[r]:
                o, w = pf["offset"][r], pf["width"][r]
                out.append(pf["flows"][o:o + (end - first) * w].view(end - first, w)[:, :slots].repeat(1, B).clone())
            elif cl is not None:
                out.append(expand(cc, cl, nf, r) if plan.top_down.per_sample[r] else
                           torch.full((end - first, B * slots), float("nan"), device = dev))
            else:
                off, wid = lay["sum_offsets"][r], lay["sum_widths"][r]
                out.append(nf[off:off + (end - first) * wid].view(end - first, wid)[:, :B * slots].clone())
        return out, plan


def expand(cc, cl, nf, r):
    """Sum region ``r``'s compact rows in ``nf`` (``cl``: the query's CompactLayout) as [rows, B * slots]: every
    sample's lanes at their pairs' slots, -inf at the slots without a lane; every lane written, -inf where its pair is
    not joined."""
    prog, n, B = cc.backend.program(), cc.n, cl.B
    first, end, slots = cc.backend.sum_regions[r]
    a, e = prog.sum_ivs[prog.reg_iv_host[r]]
    k, rows, width = int(prog.reg_iv_all[r]), end - first, cl.width[r]
    block = nf[cl.offset[r]:cl.offset[r] + rows * width].view(rows, width)
    lanes = int(cl.total[k])
    assert not torch.isnan(block[:, :lanes]).any(), r
    pos = cc.backend.layout.reachability.positions(a, e)
    X = 1 if e == n else int(cc.backend.layout.width[e])
    count, start, columns = cl.count.cpu().view(B, n + 1), cl.start.cpu().view(B, n + 1), cl.columns.cpu()
    off = cl.off[k].cpu()
    out = torch.full((rows, B * slots), -float("inf"), device = block.device)
    for b in range(B):
        ca, ce = int(count[b, a]), int(count[b, e])
        ia = columns[int(start[b, a]):int(start[b, a]) + ca].long()
        je = columns[int(start[b, e]):int(start[b, e]) + ce].long()
        i, j = ia[:, None].expand(ca, ce).flatten(), je[None, :].expand(ca, ce).flatten()
        slot = i * X + j if pos is None else pos.cpu()[i, j].long()
        lane = int(off[b]) + torch.arange(ca * ce)
        joined = slot >= 0
        assert torch.isneginf(block[:, lane[~joined].to(block.device)]).all(), (r, b)
        out[:, (b * slots + slot[joined]).to(block.device)] = block[:, lane[joined].to(block.device)]
    return out


def needed_regions(cc, targets):
    """bool [sum regions]: those whose scope meets a target (all for None)."""
    prog = cc.backend.program()
    out = []
    for r in range(len(cc.backend.sum_regions)):
        a, b = prog.sum_ivs[prog.reg_iv_host[r]]
        out.append(targets is None or bool(targets[a:b].any()))
    return np.asarray(out)


def compact(cc, block, a, b, B):
    """A reference top-down block [N, B, W_a, W_b] as the library keeps it: [N, B * slots], joined pairs row-major per
    sample, a sequence end's columns as one -- a top-down value is the same at every accepting column (the root adds
    them up), so it is any of them, not their sum (which an inside value is)."""
    if b == cc.n:
        assert torch.equal(torch.isneginf(block), torch.isneginf(block[..., :1]).expand_as(block))
        fin = torch.isfinite(block)
        assert ((block - block[..., :1])[fin].abs() <= 1e-4 * block[fin].abs() + 1e-4).all()
        block = block[..., :1]
    pos = cc.backend.layout.reachability.positions(a, b)
    flat = block.flatten(2)
    if pos is not None:
        flat = flat[:, :, torch.nonzero(pos.flatten() >= 0).flatten().to(flat.device)]
    return flat.reshape(block.size(0), -1)


def live_slots(cc, pruning_live, a, b, B):
    """bool [B * slots]: the live slots of the interval [a, b) per sample (all when not pruned)."""
    n, W = cc.n, pruning_live.size(2)
    layout = cc.backend.layout
    pos = layout.reachability.positions(a, b)
    X = 1 if b == n else int(layout.width[b])
    pairs = torch.arange(int(layout.width[a]) * X) if pos is None else torch.nonzero(pos.flatten() >= 0).flatten()
    i, j = pairs // X, pairs % X
    live = pruning_live.cpu()
    out = []
    for s in range(B):
        out.append((live[s, a, i] != 0) & (live[s, b, j] != 0))
    return torch.cat(out)


def assert_top_down_close(cc, got, want, data, missing, regions = None):
    """Every sum region's top-down values (:func:`top_down_blocks`) against the reference's, on the live slots (of the
    ``regions`` marked, bool per sum region, if given)."""
    prog, n, B = cc.backend.program(), cc.n, data.size(0)
    dev = cc.pc.params.device
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).contiguous()
    pruning = liveness(prog, obs_class, B) if (~missing).any() else None
    firsts = [first for first, _, _ in cc.backend.sum_regions]
    for ns, block in want.items():
        if not ns.is_sum():
            continue
        r = firsts.index(ns._output_ind_range[0])
        if regions is not None and not regions[r]:
            continue
        a, b = prog.sum_ivs[prog.reg_iv_host[r]]
        w = compact(cc, block, a, b, B).float()
        g = got[r]
        assert g.shape == w.shape
        # dead slots (pruned) are -inf or unwritten; samples the constraint rules out have no live slot
        if pruning is not None:
            ls = live_slots(cc, pruning.live, a, b, B).to(dev)
            g, w = g[:, ls], w[:, ls]
        assert not torch.isnan(g).any(), r
        assert torch.equal(torch.isneginf(g), torch.isneginf(w)), r
        fin = torch.isfinite(w)
        tol = 1e-5 * w[fin].abs() + n * 1e-6
        assert ((g[fin] - w[fin]).abs() <= tol).all(), (r, (g[fin] - w[fin]).abs().max())


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", list(CONSTRAINTS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_top_down_values_match_the_reference(kind, name, pattern, sum_path, build_pc, reference):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    assert_top_down_close(cc, top_down_blocks(cc, data, missing)[0], reference.top_down(cc, data, missing), data,
                          missing)


@pytest.mark.parametrize("kind", list(KINDS))
def test_dead_slots_hold_neg_inf(kind, sum_path, build_pc, reference):
    """A batch whose lanes (sample, slot) span several programs per node, under an observed prefix (one live column
    per sample at the boundaries in it): a program of dead lanes only (a sum child's, in the pruning order) still
    writes -inf over the NaN, as a dead slot's value is a term of live ones (the dense sum path on every column reads
    it); the live slots match the reference."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    prog, dev = cc.backend.program(), cc.pc.params.device
    data, missing = evidence("prefix", n, B = 48)
    B = data.size(0)
    got, plan = top_down_blocks(cc, data, missing, policy = "sample")
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).contiguous()
    pruning = liveness(prog, obs_class, B)
    for r in range(len(cc.backend.sum_regions)):
        a, b = prog.sum_ivs[prog.reg_iv_host[r]]
        assert torch.isneginf(got[r][:, ~live_slots(cc, pruning.live, a, b, B).to(dev)]).all(), r
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated", "hand", "hmm"])
def test_top_down_values_survive_a_wide_dynamic_range(kind, storage, build_pc, reference, monkeypatch):
    """The construction of liveness_test.py's ``test_marginals_survive_a_wide_dynamic_range`` (tokens 1 and 2 have
    probability e^-50 under "contains 1 2", so a block's entries span far more than fp32's exponent): the top-down
    pass's transposed block @ block contractions flush the small entries as the forward's do unless the tile is
    computed again exactly -- dense, or compact (one shared column at a time). Against the reference in float64."""
    n = KINDS[kind]
    pc = build_pc(kind, n, V)
    with torch.no_grad():
        for layer in pc.input_layer_group:                                 # P(0) ~ 1, P(1) = P(2) = e^-50 everywhere
            table = layer.params.view(-1, V)
            table[:, 1:] = math.exp(-50.0)
            table[:, 0] = 1.0 - 2 * math.exp(-50.0)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    t = n // 2
    data = torch.zeros(4, n, dtype = torch.long)
    data[2, t - 1] = 1                                                      # ... owing "2": state 1
    data[3, 0], data[3, 1] = 1, 2                                           # ... satisfied: state 2
    missing = torch.arange(n)[None, :] >= torch.tensor([0, t, t, t])[:, None]   # sample 0: nothing observed
    monkeypatch.setitem(reference.top_down.__globals__, "DTYPE", torch.float64)
    assert_top_down_close(cc, top_down_blocks(cc, data, missing, storage = storage)[0],
                          reference.top_down(cc, data, missing), data, missing)


@pytest.mark.parametrize("kind", ["pd_prod_dominated", "hand_permuted", "hmm"])
def test_top_down_values_are_deterministic(kind, build_pc):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    a = top_down_blocks(cc, data, missing)[0]
    b = top_down_blocks(cc, data, missing)[0]
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", list(KINDS))
def test_top_down_values_under_every_policy(kind, policy, pattern, build_pc, reference):
    """Values kept once (in prior_flows, read with sample stride 0) or per sample, as the policy decides: the same."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    got, plan = top_down_blocks(cc, data, missing, policy = policy)
    if policy == "sample":
        assert not plan.top_down.once.any()
    elif policy == "kept" and pattern == "none":
        assert plan.top_down.once.all()                 # nothing observed: every value is input-independent
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", list(KINDS))
def test_targets_compute_what_they_need(kind, policy, seed, build_pc, reference):
    """A query computes the top-down values of the nodes whose scope meets a target, and leaves every other one
    alone: per-sample rows still NaN, nothing kept once."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n, seed = seed)
    g = torch.Generator().manual_seed(seed)
    targets = torch.zeros(n, dtype = torch.bool)
    targets[torch.randperm(n, generator = g)[:1 + seed]] = True
    got, plan = top_down_blocks(cc, data, missing, targets = targets, policy = policy)
    need = needed_regions(cc, targets.numpy())
    assert not (plan.top_down.once & ~need).any()
    assert not (plan.top_down.sample & ~need).any()
    for r in np.nonzero(~need)[0]:
        assert torch.isnan(got[r]).all(), r                # untouched
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing, regions = need)


@pytest.mark.parametrize("kind", ["pd", "hmm", "left_linear", "hand_shared"])
def test_values_kept_once_accumulate(kind, build_pc, reference):
    """Inside one fast_inference scope, two queries observing different sides: the second keeps the first's values
    once where they still apply (computing none of them again, leaving their rows where they are) and adds its own
    after them; every value matches the reference."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    with juice.fast_inference():
        data, first = evidence("prefix", n)
        got, plan = top_down_blocks(cc, data, first)
        assert_top_down_close(cc, got, reference.top_down(cc, data, first), data, first)
        pf = cc.backend.state._prior_flows_state
        kept, offsets, size = pf["valid"].copy(), list(pf["offset"]), pf["size"]
        assert kept.any()
        data, second = evidence("suffix", n, seed = 1)
        got, plan = top_down_blocks(cc, data, second)
        assert_top_down_close(cc, got, reference.top_down(cc, data, second), data, second)
        assert not (plan.top_down.todo & kept).any()     # held from the first query
        for r in np.nonzero(kept)[0]:
            assert pf["offset"][r] == offsets[r]         # append-only
        new = plan.top_down.todo & ~kept
        assert all(pf["offset"][r] >= size for r in np.nonzero(new)[0])


@pytest.mark.parametrize("kind", ["pd", "hmm", "hand_shared"])
def test_held_values_are_not_computed(kind, build_pc, reference):
    """A query told the buffers hold some per-sample values (a caching engine's decision) computes none of them and
    reads them where they are: after a first query of the same evidence, the second holds the nodes whose scope starts
    at 0 (the root and its kin, parents of the others) and computes only the others, written over with NaN in between;
    every value matches."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    prog = cc.backend.program()
    data, missing = evidence("mixed", n)
    top_down_blocks(cc, data, missing, policy = "sample")
    held = np.asarray([prog.sum_ivs[prog.reg_iv_host[r]][0] == 0 for r in range(len(cc.backend.sum_regions))])
    state, B = cc.backend.state, data.size(0)
    lay, nf = state.batch_layout(B), state.flows(B, state.buffers(B))["node_flows"]
    for r, (first, end, _) in enumerate(cc.backend.sum_regions):
        if not held[r]:
            nf[lay["sum_offsets"][r]:lay["sum_offsets"][r] + (end - first) * lay["sum_widths"][r]] = float("nan")
    got, plan = top_down_blocks(cc, data, missing, policy = "sample", held = held, nan_fill = False)
    assert not (plan.top_down.sample & held).any()
    assert plan.top_down.sample[~held].any()
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", list(KINDS))
def test_products_of_target_inputs_get_their_values(kind, policy, build_pc, reference):
    """A target's scoring reads the top-down values of the products with an input at it: each is computed (where no
    child needs it too) and in element_flows when its group is visited -- against the reference."""
    n = KINDS[kind]
    pc = build_pc(kind, n, V)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    prog, dev = cc.backend.program(), pc.params.device
    data, missing = evidence("mixed", n)
    B = data.size(0)
    targets = torch.zeros(n, dtype = torch.bool)
    targets[n // 2] = True
    want = reference.top_down(cc, data, missing)
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).contiguous()
    pruning = liveness(prog, obs_class, B) if (~missing).any() else None
    groups = [lg for lg in pc.inner_layer_groups if lg.is_prod()]
    of = prog.intervals.of.cpu().numpy()
    checked = []

    def visit(pi, bufs, flows, plan):
        first, _, _ = cc.backend.element_regions[pi]
        width = bufs["layout"]["element_widths"][pi]
        once = plan.top_down.once_iv.cpu().numpy()
        for layer in groups[pi]:
            for ns in layer.nodes:
                if not any(c.is_input() and targets[c.scope.to_list()[0]] for c in ns.chs):
                    continue
                scope = sorted(ns.scope.to_list())
                a, b = scope[0], scope[-1] + 1
                S = cc.backend.layout.reachability.slots(a, b)
                lo, hi = ns._output_ind_range
                rows = flows["element_flows"][(lo - first) * width:(hi - first) * width].view(hi - lo, width)
                g = rows[:, :S].repeat(1, B) if once[of[a * (n + 1) + b]] else rows[:, :B * S]
                w = compact(cc, want[ns], a, b, B).float()
                if pruning is not None:
                    ls = live_slots(cc, pruning.live, a, b, B).to(dev)
                    g, w = g[:, ls], w[:, ls]
                fin = torch.isfinite(w)
                assert torch.equal(torch.isneginf(g), torch.isneginf(w)), (pi, lo)
                assert ((g[fin] - w[fin]).abs() <= 1e-5 * w[fin].abs() + n * 1e-6).all(), (pi, lo)
                checked.append(lo)

    top_down_blocks(cc, data, missing, targets = targets, policy = policy, visit = visit)
    assert checked


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("policy", ["default", "top_down", "kept"])
def test_values_kept_once_read_their_inside_values_once(policy, storage, build_pc, reference):
    """A top-down value kept once reads inside values for a batch of one -- its siblings', and a fold's operands' --
    which are kept once too, never read off one sample's pruned ones: hand_fold (whose s01 reads the fold of s2 and s3)
    with prefixes leaving the automaton in each of its states at position 2, so the samples' live columns differ.
    Compact, the folds' scratch rows hold both a batch of one's values and the samples' compact ones."""
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand_fold", 4, V))
    data = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0]]).repeat(16, 1)   # 48 samples: rows wider than
    missing = (torch.arange(4) >= 2)[None, :].expand(48, 4).contiguous()            # padding, either way
    got, plan = top_down_blocks(cc, data, missing, policy = policy, storage = storage)
    assert plan.top_down.once.any()
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)


def live_columns_of(cc, got, data, missing):
    """Every sum region's blocks (:func:`top_down_blocks`) on their live slots only."""
    prog, B, dev = cc.backend.program(), data.size(0), cc.pc.params.device
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).contiguous()
    lv = liveness(prog, obs_class, B)
    return [g[:, live_slots(cc, lv.live, *prog.sum_ivs[prog.reg_iv_host[r]], B).to(dev)] for r, g in enumerate(got)]


@pytest.mark.parametrize("pattern", ["none", "prefix", "suffix", "scattered", "mixed"])
@pytest.mark.parametrize("name", list(CONSTRAINTS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_compact_top_down_values_match_the_reference(kind, name, pattern, sum_path, build_pc, reference):
    """Kept compact (``storage = "compact"``, see kernels/compact.py): every lane written, the reference's values on
    the live slots, and the dense storage's -- bit for bit through the fused sums, which compute each lane alone, but
    for block @ block products (``BIT_EXACT_KINDS``)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    got = top_down_blocks(cc, data, missing, storage = "compact")[0]
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)
    dense = top_down_blocks(cc, data, missing)[0]
    for r, (g, d) in enumerate(zip(live_columns_of(cc, got, data, missing), live_columns_of(cc, dense, data, missing))):
        assert torch.equal(torch.isneginf(g), torch.isneginf(d)), r
        if sum_path == "fused" and kind in BIT_EXACT_KINDS:
            assert torch.equal(g, d), r


@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", list(KINDS))
def test_compact_values_under_every_policy(kind, policy, pattern, build_pc, reference):
    """Compact per-sample values next to values kept once (read from and written to a batch of one's blocks, in rows
    as wide as both need): the reference's. Nothing observed, every value input-independent: inside fast_inference
    ("kept") no region has per-sample values, and node_flows no row."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence(pattern, n)
    got, plan = top_down_blocks(cc, data, missing, policy = policy, storage = "compact")
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)
    if policy == "kept" and pattern == "none":
        assert not plan.top_down.per_sample.any()


@pytest.mark.parametrize("kind", list(KINDS))
def test_compact_values_of_targets_only(kind, build_pc, reference):
    """Compact, a query computes the values its targets need and leaves the others alone (no rows at all)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    targets = torch.zeros(n, dtype = torch.bool)
    targets[n // 2] = True
    got, plan = top_down_blocks(cc, data, missing, targets = targets, policy = "sample", storage = "compact")
    need = needed_regions(cc, targets.numpy())
    assert not (plan.top_down.sample & ~need).any()
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing, regions = need)


@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated"])
def test_wide_automata(kind, storage, build_pc, reference):
    """A 48-state automaton with 10 accepting columns at the sequence end, samples observing nothing among others: a
    sample's rectangle of live columns spans several tiles of a compact block @ block product each way (more than 32
    live columns at a boundary); the values against the reference."""
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    dev = cc.pc.params.device
    x = torch.where(missing, 0, data).to(dev)
    lv = liveness(cc.backend.program(), torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).int(),
                  data.size(0))
    assert int(lv.lists()[0].max()) > 32
    got = top_down_blocks(cc, data, missing, storage = storage)[0]
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)


#: the circuits of the walk's tests: conftest.py's kinds with ``input @ block`` steps
WALK_KINDS = {"hmm": 6, "hmm_block_sparse": 6, "hand": 5, "hand_left": 5, "hand_fold": 4, "walks": 6, "walks_unit": 6}


def walk_pc(kind, build_pc, input_dist = None):
    return build_pc(kind, WALK_KINDS[kind], V, input_dist = input_dist)


@pytest.fixture
def walk_launches(monkeypatch):
    """Every ``input @ block`` launch offered to the CUDA kernels (switched on): whether they took it, and what it is
    made of."""
    seen = []
    real = prod_top_down._input_block_cuda

    def record(la, tiles, height, prog, compact, *args):
        taken = real(la, tiles, height, prog, compact, *args)
        if not prod_top_down.INPUT_BLOCK_CUDA:                           # (the Triton kernel's run)
            return taken
        masses, obs_class, once_iv = args[4], args[5], args[6]
        c_kind, out_temp, (c_full, out_full) = la.kinds
        seen.append(dict(taken = taken is not None, c_kind = c_kind, out_temp = out_temp, c_full = c_full,
                         out_full = out_full, parents = int(np.diff(la.host["ptr"]).max()), height = height,
                         once = bool(once_iv.cpu().numpy()[np.asarray(la.host["parent_ivs"])].any()),
                         add = bool(la.host["children"][:, 2].any()), sample = masses.batch > 1,
                         observed = bool((obs_class >= 0).any()), missing = bool((obs_class < 0).any()),
                         **(taken or {})))
        return taken

    monkeypatch.setattr(prod_top_down, "_input_block_cuda", record)
    return seen


def both_walks(monkeypatch, run):
    """``run()`` on the Triton kernel, then on the CUDA kernels."""
    monkeypatch.setattr(prod_top_down, "INPUT_BLOCK_CUDA", False)
    want = run()
    monkeypatch.setattr(prod_top_down, "INPUT_BLOCK_CUDA", True)
    return want, run()


def assert_same_bits(got, want):
    for r, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape and torch.equal(g.view(torch.int32), w.view(torch.int32)), r


#: CTAs of the CUDA walk: a node tile's children (at most 16) or fewer, a sample's (128 threads or 256) or 8 samples'
#: (a warp each) -- ``INPUT_BLOCK_CUDA_NODES``, ``_WIDE``, ``_WARP_ITEMS``
WALK_CTAS = {"default": (16, 512, 64), "8_wide": (8, 0, 0), "2": (2, 512, 0), "1_warps": (1, 512, 1 << 20),
             "16_warps": (16, 512, 1 << 20)}


def set_walk_ctas(monkeypatch, cta):
    for name, value in zip(("NODES", "WIDE", "WARP_ITEMS"), WALK_CTAS[cta]):
        monkeypatch.setattr(prod_top_down, "INPUT_BLOCK_CUDA_" + name, value)


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
@pytest.mark.parametrize("cta", list(WALK_CTAS))
@pytest.mark.parametrize("policy", ["default", "top_down"])
@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("name", list(CONSTRAINTS))
@pytest.mark.parametrize("kind", list(WALK_KINDS))
def test_cuda_walk_gives_the_triton_values(kind, name, pattern, policy, cta, build_pc, walk_launches, monkeypatch):
    """Compact, the ``input @ block`` launches run on the CUDA kernels (every one, CTAs of every size) and give the
    Triton kernel's top-down values bit for bit: the same terms in the same order, the same arithmetic."""
    set_walk_ctas(monkeypatch, cta)
    n = WALK_KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), walk_pc(kind, build_pc))
    data, missing = evidence(pattern, n)
    want, got = both_walks(monkeypatch, lambda: top_down_blocks(cc, data, missing, policy = policy,
                                                                storage = "compact")[0])
    assert_same_bits(got, want)
    assert all(la["taken"] for la in walk_launches)


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
@pytest.mark.parametrize("codes", [1, 2])
@pytest.mark.parametrize("cta", ["default", "1_warps"])
@pytest.mark.parametrize("pattern", ["none", "prefix", "mixed"])
@pytest.mark.parametrize("name", list(CONSTRAINTS))
@pytest.mark.parametrize("kind", list(WALK_KINDS))
def test_cuda_walk_with_deferred_units_gives_the_triton_values(kind, name, pattern, cta, codes, build_pc, walk_launches,
                                                               monkeypatch):
    """A CTA staging the masses of 1 or 2 codes a unit (``INPUT_BLOCK_CUDA_CODES``): a unit whose terms read more
    defers to the kernel staging every code, with the kept-once parents' -- with 1 code, that kernel's one CTA summing
    every deferred unit in turn -- the Triton kernel's values bit for bit."""
    set_walk_ctas(monkeypatch, cta)
    monkeypatch.setattr(prod_top_down, "INPUT_BLOCK_CUDA_CODES", codes)
    monkeypatch.setattr(prod_top_down, "INPUT_BLOCK_CUDA_DEFER_CTAS", 1 if codes == 1 else 0)
    n = WALK_KINDS[kind]
    cc = jc.compile(CONSTRAINTS[name](), walk_pc(kind, build_pc))
    data, missing = evidence(pattern, n)
    want, got = both_walks(monkeypatch, lambda: top_down_blocks(cc, data, missing, storage = "compact")[0])
    assert_same_bits(got, want)
    assert all(la["taken"] and la["codes"] <= codes for la in walk_launches)


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
@pytest.mark.parametrize("kind", ["hmm", "walks"])
def test_cuda_walk_on_per_sample_class_masses(kind, build_pc, walk_launches, monkeypatch):
    """Soft evidence inputs (class masses of each sample's own) in a decoder session, left to right: the conditionals
    of the Triton kernel's bit for bit."""
    n = WALK_KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), walk_pc(kind, build_pc, input_dist = dists.SoftEvidenceCategorical))
    g = torch.Generator().manual_seed(0)
    B = 5
    data = torch.randint(0, V, (B, n), generator = g).cuda()
    E = (torch.randn(B, n, V, generator = g) * 2).cuda()

    def session():
        with cc.decoder(categorical_evidence_logp = E, storage = "compact") as decoder:
            return [decoder.conditional(data, (torch.arange(n, device = data.device) >= t)[None, :].expand(B, n))
                    for t in range(n)]

    want, got = both_walks(monkeypatch, session)
    assert_same_bits(got, want)
    assert any(la["taken"] and la["sample"] for la in walk_launches)


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
def test_cuda_walk_on_wide_automata(build_pc, walk_launches, monkeypatch):
    """A 48-state automaton, samples observing nothing among others: tens of live columns a sample (their terms
    ordered, most first), 256 threads a CTA; the Triton kernel's values bit for bit."""
    rng = random.Random(0)
    K, n = 48, 6
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc("hmm_block_sparse", n, V))
    data, missing = evidence("mixed", n)
    want, got = both_walks(monkeypatch, lambda: top_down_blocks(cc, data, missing, storage = "compact")[0])
    assert_same_bits(got, want)
    assert all(la["taken"] for la in walk_launches)


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
def test_cuda_walk_past_45k_of_shared_memory(build_pc, walk_launches, monkeypatch):
    """A random 16000-state automaton over 32 tokens: a sample observing nothing keeps most states live within three
    tokens, its parents' rows too wide for 45 KB of shared memory -- CTAs of one child with the device's more; the
    Triton kernel's values bit for bit."""
    g = torch.Generator().manual_seed(0)
    K, Vw, n = 16000, 32, 5
    dfa = jc.DFA.from_dense(Vw, torch.randint(0, K, (K, Vw), generator = g), 0,
                            torch.randperm(K, generator = g)[:K // 2].tolist())
    cc = jc.compile(dfa, build_pc("hmm_block_sparse", n, Vw))
    data = torch.randint(0, Vw, (2, n), generator = g)
    missing = torch.tensor([[True] * n, [True, False, True, True, False]])
    want, got = both_walks(monkeypatch, lambda: top_down_blocks(cc, data, missing, storage = "compact")[0])
    assert_same_bits(got, want)
    assert all(la["taken"] for la in walk_launches)
    assert max(la["smem"] for la in walk_launches) > 45 * 1024


@pytest.mark.skipif(get_module() is None, reason = "the CUDA kernels are unavailable")
def test_cuda_walk_reaches_every_case(build_pc, walk_launches, monkeypatch):
    """The walk tests' circuits and queries reach every case of the CUDA kernels: children in sum rows and in fold
    rows, parents in element rows and fold rows, intervals joining every pair or not (child's and parent's), several
    parents a child, parents kept once, children log-added to, observed and missing classes, node tiles of 1, 4 and 16
    rows, CTAs of every number of children, of 128 and 256 threads, of a sample or a warp per sample."""
    for kind in WALK_KINDS:
        n = WALK_KINDS[kind]
        for name in CONSTRAINTS:
            cc = jc.compile(CONSTRAINTS[name](), walk_pc(kind, build_pc))
            for pattern, policy in (("mixed", "default"), ("prefix", "top_down")):
                data, missing = evidence(pattern, n)
                top_down_blocks(cc, data, missing, policy = policy, storage = "compact")
    for cta in WALK_CTAS:
        set_walk_ctas(monkeypatch, cta)
        for kind in ("walks", "hmm"):                                     # node tiles of 4 rows and of 16
            cc = jc.compile(CONSTRAINTS["contains"](), walk_pc(kind, build_pc))
            data, missing = evidence("mixed", 6)
            top_down_blocks(cc, data, missing, storage = "compact")
    taken = [la for la in walk_launches if la["taken"]]
    cases = {key: {la[key] for la in taken} for key in taken[0]}
    for key, want in dict(c_kind = {0, 1}, out_temp = {False, True}, c_full = {False, True},
                          out_full = {False, True}, once = {False, True}, add = {False, True}, observed = {True},
                          missing = {True}, height = {1, 4, 16}, tn = {16, 8, 4, 2, 1}, threads = {128, 256},
                          warps = {False, True}).items():
        assert want <= cases[key], (key, cases[key])
    assert max(cases["parents"]) > 1

