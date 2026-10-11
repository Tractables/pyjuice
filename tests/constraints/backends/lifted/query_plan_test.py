"""
Query plans (:mod:`pyjuice.constraints.backends.lifted.query_plan`) and the input-independent values they keep in
``prior_mars``:

* a plan cuts every launch, dense run and fused part to exactly the values the query needs, input-dependent ones per
  sample and input-independent ones not yet kept once -- against the definitions, from the scopes, on every kind of
  PC; every product step carries its product's interval;
* the input-independent values are those of the scope with nothing observed (the reference with everything
  missing), and a planned forward computes the needed input-dependent values as the full one does, leaving the
  others untouched;
* they are kept inside :func:`pyjuice.fast_inference` (the second query computes none, and gets the same marginal)
  and recomputed by every query outside it, following in-place parameter updates.

Marginals under every pattern of evidence, which run on these plans, are tested in ``pruning_test.py``.
"""
import types

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted import forward as F
from pyjuice.constraints.backends.lifted.kernels.live import prune
from pyjuice.constraints.backends.lifted.kernels.prod import TEMP
from pyjuice.constraints.backends.lifted.query_plan import free_regions, prior_regions, query_plan

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8,
         "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5}         # see conftest.py


def compiled(build_pc, kind):
    n = KINDS[kind]
    return jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V)), n


def prefixes(n, lengths, seed = 0):
    """A batch whose sample b observes its first lengths[b] tokens (decoding), the rest missing."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (len(lengths), n), generator = g)
    missing = torch.arange(n)[None, :] >= torch.tensor(lengths)[:, None]
    return data, missing


def out_interval(launch, rows):
    """Every step's output interval ``(entry, exit)`` (host rows, by contraction)."""
    if launch.form == "input_block":
        return rows[:, 2], rows[:, 5]
    if launch.form == "block_input":
        return rows[:, 3], rows[:, 5] + 1
    if launch.form == "block_block":
        return rows[:, 5], rows[:, 7]
    return rows[:, 3], rows[:, 4]


def temp_reads(launch, rows):
    """The scratch rows every step reads: [S, 2], -1 where an operand is not scratch."""
    out = torch.full((rows.size(0), 2), -1, dtype = torch.long)
    k = launch.kinds
    if launch.form == "input_block" and k[0] == TEMP:
        out[:, 0] = rows[:, 3]
    elif launch.form == "block_input" and k[0] == TEMP:
        out[:, 0] = rows[:, 1]
    elif launch.form == "block_block":
        if k[0] == TEMP:
            out[:, 0] = rows[:, 1]
        if k[1] == TEMP:
            out[:, 1] = rows[:, 3]
    elif launch.form == "copy" and k[0] == TEMP:
        out[:, 0] = rows[:, 1]
    return out


def out_temp(launch):
    return launch.kinds[2] if launch.form == "block_block" else launch.kinds[1]


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_step_carries_its_products_interval(kind, build_pc):
    """A step writing the layer's element row is its product: its interval is its output's. A step writing scratch
    is a fold of the product whose later step reads that row: both carry the same interval."""
    cc, _ = compiled(build_pc, kind)
    prog = cc._lifted_program()
    for k, _, layers in prog.steps:
        if k != "prod":
            continue
        for stages, _, _ in layers:
            written, read = {}, {}                                          # scratch row -> interval
            for launches in stages:
                for launch in launches:
                    ivs = [prog.sum_ivs[i] for i in launch.iv.tolist()]
                    a, b = out_interval(launch, launch.rows)
                    for s, iv in enumerate(ivs):
                        if out_temp(launch):
                            written[int(launch.rows[s, 0])] = iv
                            assert iv[1] == int(b[s])                     # a fold ends where its product ends
                        else:
                            assert iv == (int(a[s]), int(b[s]))
                        for r in temp_reads(launch, launch.rows)[s].tolist():
                            if r >= 0:
                                read[r] = iv
            assert written == read


def flags(cc, observed, targets, valid, prior):
    """Per sum interval, from the definitions: (computed per sample, computed once) -- needed and not kept once;
    needed, kept once and not yet held."""
    prog = cc._lifted_program()
    out = []
    for k, (a, b) in enumerate(prog.sum_ivs):
        needed = targets is None or bool(targets[:a].any() or targets[b:].any())
        out.append((needed and not bool(prior[k]), needed and bool(prior[k]) and not bool(valid[k])))
    return out


def rng_cases(n, seed):
    """(observed, targets or None, valid, generator) cases, as numpy."""
    for observed, targets, valid, g in cases(n, seed):
        yield observed.numpy(), None if targets is None else targets.numpy(), valid.numpy(), g


def cases(n, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(4):
        observed = torch.rand(n, generator = g) < 0.4
        targets = (torch.rand(n, generator = g) < 0.3) & ~observed
        valid = torch.rand(n * n, generator = g) < 0.5
        yield observed, None, valid, g
        yield observed, targets, valid, g


@pytest.mark.parametrize("kind", list(KINDS))
def test_plans_cut_exactly_the_needed_steps(kind, build_pc):
    """Every launch, dense run and fused part of a plan holds exactly the steps and node blocks whose product's (or
    region's) values the query computes: the input-dependent needed ones per sample, the input-independent needed
    ones not yet kept once."""
    cc, n = compiled(build_pc, kind)
    prog = cc._lifted_program()
    for observed, targets, valid, g in rng_cases(n, seed = 1):
        valid = valid[:len(prog.sum_ivs)]
        free = free_regions(prog, observed)
        prior = free & (torch.rand(free.size, generator = g) < 0.7).numpy()   # some of the free intervals kept once
        prior = free & prog.iv_inside[prior].any(axis = 0)                  # ... and every interval inside one
        want = flags(cc, observed, targets, valid, prior)
        plan = query_plan(prog, observed, targets, valid, prior)
        assert np.array_equal(plan.reg_prior.cpu().numpy().astype(bool), prior[prog.reg_iv_host])
        assert plan.todo.tolist() == [once for _, once in want]
        for (kind_, _, layers), (_, _, cut) in zip(prog.steps, plan.steps):
            if kind_ == "prod":
                for (stages, _, _), (once, sample, _, _) in zip(layers, cut):
                    for variant, side in ((sample, 0), (once, 1)):
                        got = sorted(tuple(r) for launches in variant for l in launches for r in l.rows.tolist())
                        exp = sorted(tuple(r) for launches in stages for l in launches
                                     for r, i in zip(l.rows.tolist(), l.iv.tolist()) if want[i][side])
                        assert got == exp
            else:
                for (_, parts), (_, once, sample) in zip(layers, cut):
                    for variant, side in ((sample, 0), (once, 1)):
                        got = sorted((r, c0) for dense, _, _ in variant for (r, c0) in dense)
                        exp = sorted((r, c0) for dense, _, _ in parts for (r, c0) in dense
                                     if want[prog.reg_iv_host[r]][side])
                        assert got == exp
                        got = sorted(int(x) for _, _, fused in variant if fused is not None for x in fused[0].tolist())
                        exp = sorted(int(x) for _, _, fused in parts if fused is not None
                                     for x, r in zip(fused[0].tolist(), fused[3].tolist())
                                     if want[prog.reg_iv_host[r]][side])
                        assert got == exp


def test_plans_are_cached(build_pc):
    cc, n = compiled(build_pc, "pd")
    prog = cc._lifted_program()
    observed, valid = np.zeros(n, dtype = bool), np.zeros(len(prog.sum_ivs), dtype = bool)
    observed[:2] = True
    prior = free_regions(prog, observed)
    plan = query_plan(prog, observed, None, valid, prior)
    assert query_plan(prog, observed.copy(), None, valid.copy(), prior.copy()) is plan
    valid[:] = True
    assert query_plan(prog, observed, None, valid, prior) is not plan        # nothing left to compute once


def compact(cc, block, a, b):
    """A reference block [N, B, W_a, W_b] as the library keeps it: [N, B, slots], the joined pairs in row-major order,
    a sequence end's columns summed."""
    if b == cc.n:
        block = torch.logsumexp(block, dim = -1, keepdim = True)
    pos = cc.layout.reachability.positions(a, b)
    flat = block.flatten(2)
    return flat if pos is None else flat[:, :, torch.nonzero(pos.flatten() >= 0).flatten().to(flat.device)]


def prior_rows(cc, r):
    """Sum region r's rows in prior_mars, its slots."""
    first, end, slots = cc.sum_regions[r]
    buf = cc._prior_buffer()
    off, wid = buf["offset"][r], buf["width"][r]
    return buf["mars"][off:off + (end - first) * wid].view(end - first, wid)[:, :slots]


def assert_close(got, want):
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert ((got[fin] - want[fin]).abs() <= 1e-5 * want[fin].abs() + 1e-5).all(), (got - want).abs().max()


@pytest.mark.parametrize("kind", list(KINDS))
def test_input_independent_values_are_those_with_nothing_observed(kind, build_pc, reference):
    """After a marginal, the regions whose scope no sample observes are held once, in prior_mars: the reference's
    inside values with every token missing. Two batches: one decoding in lockstep (the regions after the longest
    prefix), one observing suffixes (the regions before them: a left-linear chain's are all prefixes)."""
    cc, n = compiled(build_pc, kind)
    prog = cc._lifted_program()
    want = reference.inside(cc, torch.zeros(1, n, dtype = torch.long), torch.ones(1, n, dtype = torch.bool))
    decoding = prefixes(n, [n // 3, n // 2, n // 2])
    data, missing = prefixes(n, [n // 2, n // 3])
    seen = 0
    for data, missing in (decoding, (data, missing.flip(1))):            # prefixes observed, then suffixes
        with juice.fast_inference():                                        # kept: every free region once
            assert_close(cc.marginal(data, missing), reference.marginal(cc, data, missing))
        observed = (~missing).any(dim = 0)
        for ns, block in want.items():
            if not ns.is_sum():
                continue
            r = [first for first, _, _ in cc.sum_regions].index(ns._output_ind_range[0])
            a, b = prog.sum_ivs[prog.reg_iv_host[r]]
            if not observed[a:b].any():
                assert_close(prior_rows(cc, r), compact(cc, block, a, b)[:, 0])
                seen += 1
    assert seen > 0


def planned_forward(cc, data, missing, targets):
    """The forward of a conditional's plan (its values only, nothing read off): returns the plan and buffers."""
    prog, dev = cc._lifted_program(), cc.pc.params.device
    missing = missing.to(dev)
    observed = (~missing).any(dim = 0).cpu().numpy()
    valid, _ = cc._prior_validity()
    plan = query_plan(prog, observed, None if targets is None else targets.numpy(), valid.numpy(),
                      free_regions(prog, observed))                         # every free region once
    x = torch.where(missing, 0, data.to(dev).long())
    obs_class = torch.where(missing, -1, cc.token_classes.token_class[x]).contiguous()
    pruning = prune(prog, obs_class, data.size(0)) if observed.any() else None
    bufs = cc._buffers(data.size(0))
    with torch.no_grad():
        F.run_forward(cc, bufs, plan, x, missing, obs_class, pruning, "fp32")
    valid |= plan.todo
    return plan, bufs


def sample_rows(cc, bufs, r, B):
    first, end, slots = cc.sum_regions[r]
    lay = bufs["layout"]
    off, wid = lay["sum_offsets"][r], lay["sum_widths"][r]
    return bufs["node_mars"][off:off + (end - first) * wid].view(end - first, wid)[:, :B * slots]


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_planned_forward_computes_exactly_the_needed_values(kind, build_pc):
    """The inside values a conditional needs -- of the regions whose scope misses one of its targets -- are those the
    full forward of the same evidence computes; the others are not computed (their rows keep the NaN they were filled
    with)."""
    cc, n = compiled(build_pc, kind)
    prog = cc._lifted_program()
    g = torch.Generator().manual_seed(2)
    B = 3
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.6
    for t in range(n):
        targets = torch.zeros(n, dtype = torch.bool)
        targets[t] = True
        missing_t = missing | targets[None, :]                              # a target is missing
        planned_forward(cc, data, missing_t, None)                          # every value, as a marginal
        full = [sample_rows(cc, cc._buffers(B), r, B).clone() for r in range(len(cc.sum_regions))]
        bufs = cc._buffers(B)
        bufs["node_mars"].fill_(float("nan"))
        planned_forward(cc, data, missing_t, targets)
        obs_t = (~missing_t).any(dim = 0)
        for r in range(len(cc.sum_regions)):
            a, b = prog.sum_ivs[prog.reg_iv_host[r]]
            if not obs_t[a:b].any():                                        # input-independent: prior_mars
                continue
            got = sample_rows(cc, bufs, r, B)
            if a <= t < b:                                                  # not needed
                assert torch.isnan(got).all(), (kind, t, r)
            else:                                                           # the same evidence as the full forward's
                assert_close(got, full[r])


def count_once(monkeypatch):
    """Record, per marginal, whether its plan computes input-independent values."""
    seen = []

    def spy(*args):
        plan = query_plan(*args)
        seen.append(bool(plan.todo.any()))
        return plan

    monkeypatch.setattr(F, "query_plan", spy)
    return seen


def test_input_independent_values_are_kept_inside_fast_inference(build_pc, monkeypatch):
    cc, n = compiled(build_pc, "hmm")
    data, missing = prefixes(n, [0] * 7 + [2])                             # a batch mostly unobserved: kept once pays
    seen = count_once(monkeypatch)
    cold = [cc.marginal(data, missing) for _ in range(2)]
    assert seen == [True, True]                                             # outside: every query computes them
    with juice.fast_inference():
        kept = [cc.marginal(data, missing) for _ in range(2)]
    assert seen[2:] == [True, False]                                        # inside: the first query only
    assert torch.equal(kept[0], kept[1]) and torch.allclose(kept[0], cold[0], rtol = 1e-6, atol = 1e-6)
    cc.marginal(data, missing)
    assert seen[4:] == [True]                                               # the scope's exit dropped them


def test_input_independent_values_follow_parameter_updates(build_pc, reference):
    """An in-place parameter update between queries (outside fast_inference) is seen by the next query."""
    n = 5
    pc, other = build_pc("hand", n, V, seed = 0), build_pc("hand", n, V, seed = 1)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    data, missing = prefixes(n, [0] * 7 + [2])
    before = cc.marginal(data, missing)
    with torch.no_grad():
        pc.params.copy_(other.params)
        for layer, src in zip(pc.input_layer_group, other.input_layer_group):
            layer.params.copy_(src.params)
    after = cc.marginal(data, missing)
    assert (after - before).abs().max() > 1e-2
    assert_close(after, reference.marginal(cc, data, missing))


def test_nothing_observed_reads_the_root_once(build_pc, reference):
    """With every token missing in every sample, every value is input-independent: the per-sample buffers are not
    written, and every sample reads the same root."""
    cc, n = compiled(build_pc, "pd")
    data, missing = prefixes(n, [0, 0, 0])
    bufs = cc._buffers(3)
    bufs["node_mars"].fill_(float("nan"))
    got = cc.marginal(data, missing)
    assert_close(got, reference.marginal(cc, data, missing))
    lay = bufs["layout"]
    assert torch.isnan(bufs["node_mars"][lay["sum_offsets"][0]:]).all()


def test_kept_once_when_it_pays(build_pc):
    """Outside fast_inference, a free interval is kept once when its block costs less than the samples' live columns
    (and so is every interval inside one of those); inside, every free interval is."""
    cc, n = compiled(build_pc, "pd")
    prog = cc._lifted_program()
    slots = prog.iv_slots
    a, b = prog.iv_bounds[:, 0], prog.iv_bounds[:, 1]
    for lengths in ([0], [0, 0, 0], [n // 2, n // 2], [n // 2] + [0] * 7, [1, 2, 3]):
        data, missing = prefixes(n, lengths)
        observed = (~missing).any(dim = 0).numpy()
        free = free_regions(prog, observed)
        assert np.array_equal(free, np.array([not observed[lo:hi].any() for lo, hi in prog.sum_ivs]))
        dev = cc.pc.params.device
        x = torch.where(missing, 0, data).to(dev)
        obs_class = torch.where(missing.to(dev), -1, cc.token_classes.token_class[x]).contiguous()
        pruning = prune(prog, obs_class, len(lengths)) if observed.any() else None
        live = np.array(pruning.num_live) if pruning is not None else len(lengths) * slots
        worth = free & (live > slots)
        closed = free & np.array([bool((worth & (a <= a[k]) & (b >= b[k])).any()) for k in range(len(slots))])
        assert np.array_equal(prior_regions(prog, observed, len(lengths), False, pruning), closed)
        assert np.array_equal(prior_regions(prog, observed, len(lengths), True, pruning), free)
        if lengths == [0]:
            assert not closed.any()                                         # one sample, nothing observed: as before
        if lengths == [0, 0, 0]:
            assert closed.all()                                             # nothing observed, 3 samples: all once
        if lengths == [n // 2, n // 2]:                                      # two pinned prefixes: the intervals at
            assert not closed[torch.tensor([lo == n // 2 for lo, _ in prog.sum_ivs])].any()   # the pin do not pay

    # the rule itself, on live counts set by hand: a block kept once pays when the samples' live slots exceed it (not
    # equal it), and takes every interval inside it along, since its launch reads them for a batch of one
    nothing = np.zeros(n, dtype = bool)                                      # nothing observed: every interval free
    k = max(range(len(slots)), key = lambda m: (b[m] - a[m], -a[m]))        # a longest interval
    inside = (a >= a[k]) & (b <= b[k])
    assert inside.sum() > 1
    live = np.zeros(len(slots), dtype = np.int64)
    live[k] = slots[k]
    assert not prior_regions(prog, nothing, 2, False, types.SimpleNamespace(num_live = live.tolist())).any()
    live[k] += 1
    assert np.array_equal(prior_regions(prog, nothing, 2, False, types.SimpleNamespace(num_live = live.tolist())), inside)
    assert not prior_regions(prog, nothing, 1, False, None).any()           # one sample: never, and no counts read
