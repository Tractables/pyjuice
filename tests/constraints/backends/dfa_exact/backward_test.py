"""
The lifted top-down pass (:mod:`pyjuice.constraints.backends.dfa_exact.backward`): every sum node's top-down value (its
log outside value) against the test reference's (``reference.top_down``), on every kind of PC in conftest.py's
``PC_KINDS``, under constraints and every pattern of evidence (pruned, and unpruned where nothing is observed), through
both sum paths, and where block @ block's shifts flush entries; run twice, the values are identical. Under query plans
(:mod:`~pyjuice.constraints.backends.dfa_exact.query_plan`): with every policy of keeping values once, a query computes
exactly the values its targets need; values kept once accumulate across queries, append-only; held values are not
computed again.
"""
import contextlib
import math

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.backends.dfa_exact import forward as F
from pyjuice.constraints.backends.dfa_exact.backward import run_backward
from pyjuice.constraints.backends.dfa_exact.kernels.live import prune
from pyjuice.constraints.backends.dfa_exact.query_plan import prior_regions, query_plan, top_down_prior

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8,
         "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5,
         "hand_shared": 4, "hand_fold": 4}                                                      # see conftest.py
PATTERNS = ["none", "all", "prefix", "suffix", "scattered", "mixed"]
# keep nothing once; as a query decides; every input-independent value (fast_inference); every input-independent
# top-down value and no inside value (so the inside values its launches for a batch of one read must be kept once)
POLICIES = ["sample", "default", "kept", "top_down"]
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
    from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "prune_dense", (lambda ncols, live: False) if request.param == "dense_unpruned" else
                        (lambda ncols, live: live < ncols))
    return request.param


def top_down_blocks(cc, data, missing, targets = None, policy = "default", held = None, nan_fill = True,
                    visit = None):
    """
    The forward then the top-down pass of a query under its plan (``policy``: see ``POLICIES``; "kept" runs inside
    :func:`pyjuice.fast_inference`), node_flows filled with NaN first (``nan_fill``): ``(blocks, plan)``, ``blocks``
    every sum region's rows of the buffer its values are in -- [rows, B * slots], those kept once repeated per sample
    (cloned). ``visit(prod_index, bufs, flows, plan)``: see :func:`~pyjuice.constraints.backends.dfa_exact.backward.
    run_backward`.
    """
    with juice.fast_inference() if policy == "kept" else contextlib.nullcontext():
        prog, td, dev = cc._lifted_program(), cc._top_down_program(), cc.pc.params.device
        B = data.size(0)
        missing = missing.to(dev)
        observed = (~missing).any(dim = 0).cpu().numpy()
        x = torch.where(missing, 0, data.to(dev).long())
        obs_class = torch.where(missing, -1, cc.token_classes.token_class[x]).contiguous()
        pruning = prune(prog, obs_class, B) if observed.any() else None
        valid, kept = cc._prior_validity()
        if policy in ("sample", "top_down"):
            prior = np.zeros(prog.iv_bounds.shape[0], dtype = bool)
            once = top_down_prior(prog, td, observed, B, True, pruning) if policy == "top_down" else \
                np.zeros(td.iv_bounds.shape[0], dtype = bool)
        else:
            prior = prior_regions(prog, observed, B, kept, pruning)
            once = top_down_prior(prog, td, observed, B, kept, pruning)
        pf = cc._prior_flows(np.zeros(len(cc.sum_regions), dtype = bool))
        plan = query_plan(prog, observed, None if targets is None else targets.cpu().numpy(), valid.numpy(), prior,
                          top_down = dict(program = td, once = once, valid = pf["valid"].copy(), held = held))
        bufs = cc._buffers(B)
        flows = cc._flows(B, bufs)
        if nan_fill:
            flows["node_flows"].fill_(float("nan"))         # every needed slot must be written
        with torch.no_grad():
            F.run_forward(cc, bufs, plan, x, missing, obs_class, pruning, "fp32")
            valid |= plan.todo
            run_backward(cc, bufs, flows, plan, x, missing, obs_class, pruning, "fp32",
                         visit = None if visit is None else lambda pi: visit(pi, bufs, flows, plan))
        lay, nf, pf = bufs["layout"], flows["node_flows"], cc._prior_flows_state
        out = []
        for r, (first, end, slots) in enumerate(cc.sum_regions):
            if plan.top_down.once[r]:
                o, w = pf["offset"][r], pf["width"][r]
                out.append(pf["flows"][o:o + (end - first) * w].view(end - first, w)[:, :slots].repeat(1, B).clone())
            else:
                off, wid = lay["sum_offsets"][r], lay["sum_widths"][r]
                out.append(nf[off:off + (end - first) * wid].view(end - first, wid)[:, :B * slots].clone())
        return out, plan


def needed_regions(cc, targets):
    """bool [sum regions]: those whose scope meets a target (all for None)."""
    prog = cc._lifted_program()
    out = []
    for r in range(len(cc.sum_regions)):
        a, b = prog.sum_ivs[prog.reg_iv_host[r]]
        out.append(targets is None or bool(targets[a:b].any()))
    return np.asarray(out)


def compact(cc, block, a, b, B):
    """A reference block [N, B, W_a, W_b] as the library keeps it: [N, B * slots], joined pairs row-major per sample,
    a sequence end's columns summed."""
    if b == cc.n:
        block = torch.logsumexp(block, dim = -1, keepdim = True)
    pos = cc.layout.reachability.positions(a, b)
    flat = block.flatten(2)
    if pos is not None:
        flat = flat[:, :, torch.nonzero(pos.flatten() >= 0).flatten().to(flat.device)]
    return flat.reshape(block.size(0), -1)


def live_slots(cc, pruning_live, a, b, B):
    """bool [B * slots]: the live slots of the interval [a, b) per sample (all when not pruned)."""
    n, W = cc.n, pruning_live.size(2)
    pos = cc.layout.reachability.positions(a, b)
    X = 1 if b == n else int(cc.layout.width[b])
    pairs = torch.arange(int(cc.layout.width[a]) * X) if pos is None else torch.nonzero(pos.flatten() >= 0).flatten()
    i, j = pairs // X, pairs % X
    live = pruning_live.cpu()
    out = []
    for s in range(B):
        out.append((live[s, a, i] != 0) & (live[s, b, j] != 0))
    return torch.cat(out)


def assert_top_down_close(cc, got, want, data, missing, regions = None):
    """Every sum region's top-down values (:func:`top_down_blocks`) against the reference's, on the live slots (of the
    ``regions`` marked, bool per sum region, if given)."""
    prog, n, B = cc._lifted_program(), cc.n, data.size(0)
    dev = cc.pc.params.device
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.token_classes.token_class[x]).contiguous()
    pruning = prune(prog, obs_class, B) if (~missing).any() else None
    firsts = [first for first, _, _ in cc.sum_regions]
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
    prog, dev = cc._lifted_program(), cc.pc.params.device
    data, missing = evidence("prefix", n, B = 48)
    B = data.size(0)
    got, plan = top_down_blocks(cc, data, missing, policy = "sample")
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.token_classes.token_class[x]).contiguous()
    pruning = prune(prog, obs_class, B)
    for r in range(len(cc.sum_regions)):
        a, b = prog.sum_ivs[prog.reg_iv_host[r]]
        assert torch.isneginf(got[r][:, ~live_slots(cc, pruning.live, a, b, B).to(dev)]).all(), r
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)


@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated", "hand", "hmm"])
def test_top_down_values_survive_a_wide_dynamic_range(kind, build_pc, reference, monkeypatch):
    """The construction of pruning_test.py's ``test_marginals_survive_a_wide_dynamic_range`` (tokens 1 and 2 have
    probability e^-50 under "contains 1 2", so a block's entries span far more than fp32's exponent): the top-down
    pass's transposed block @ block contractions flush the small entries as the forward's do unless the tile is
    computed again exactly. Against the reference in float64."""
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
    assert_top_down_close(cc, top_down_blocks(cc, data, missing)[0], reference.top_down(cc, data, missing), data,
                          missing)


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
        pf = cc._prior_flows_state
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
    prog = cc._lifted_program()
    data, missing = evidence("mixed", n)
    top_down_blocks(cc, data, missing, policy = "sample")
    held = np.asarray([prog.sum_ivs[prog.reg_iv_host[r]][0] == 0 for r in range(len(cc.sum_regions))])
    lay, nf = cc._layout(data.size(0)), cc._flows(data.size(0), cc._buffers(data.size(0)))["node_flows"]
    for r, (first, end, _) in enumerate(cc.sum_regions):
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
    prog, dev = cc._lifted_program(), pc.params.device
    data, missing = evidence("mixed", n)
    B = data.size(0)
    targets = torch.zeros(n, dtype = torch.bool)
    targets[n // 2] = True
    want = reference.top_down(cc, data, missing)
    x = torch.where(missing, 0, data).to(dev)
    obs_class = torch.where(missing.to(dev), -1, cc.token_classes.token_class[x]).contiguous()
    pruning = prune(prog, obs_class, B) if (~missing).any() else None
    groups = [lg for lg in pc.inner_layer_groups if lg.is_prod()]
    of = prog.intervals.of.cpu().numpy()
    checked = []

    def visit(pi, bufs, flows, plan):
        first, _, _ = cc.element_regions[pi]
        width = bufs["layout"]["element_widths"][pi]
        once = plan.top_down.once_iv.cpu().numpy()
        for layer in groups[pi]:
            for ns in layer.nodes:
                if not any(c.is_input() and targets[c.scope.to_list()[0]] for c in ns.chs):
                    continue
                scope = sorted(ns.scope.to_list())
                a, b = scope[0], scope[-1] + 1
                S = cc.layout.reachability.slots(a, b)
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


@pytest.mark.parametrize("policy", ["default", "top_down", "kept"])
def test_values_kept_once_read_their_inside_values_once(policy, build_pc, reference):
    """A top-down value kept once reads inside values for a batch of one -- its siblings', and a fold's operands' --
    which are kept once too, never read off one sample's pruned ones: hand_fold (whose s01 reads the fold of s2 and s3)
    with prefixes leaving the automaton in each of its states at position 2, so the samples' live columns differ."""
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hand_fold", 4, V))
    data = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0]])
    missing = (torch.arange(4) >= 2)[None, :].expand(3, 4).contiguous()
    got, plan = top_down_blocks(cc, data, missing, policy = policy)
    assert plan.top_down.once.any()
    assert_top_down_close(cc, got, reference.top_down(cc, data, missing), data, missing)
