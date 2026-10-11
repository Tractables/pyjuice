"""
The lifted top-down pass (:mod:`pyjuice.constraints.backends.lifted.backward`): every sum node's top-down value (its
log outside value) against the test reference's (``reference.top_down``), on every kind of PC in conftest.py's
``PC_KINDS``, under constraints and every pattern of evidence (pruned, and unpruned where nothing is observed), through
both sum paths, and where block @ block's shifts flush entries; run twice, the values are identical.
"""
import math

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted import forward as F
from pyjuice.constraints.backends.lifted.backward import run_backward
from pyjuice.constraints.backends.lifted.kernels.live import prune
from pyjuice.constraints.backends.lifted.query_plan import prior_regions, query_plan

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8,
         "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5,
         "hand_shared": 4}                                                                      # see conftest.py
PATTERNS = ["none", "all", "prefix", "suffix", "scattered", "mixed"]
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


@pytest.fixture(params = ["fused", "dense"])
def sum_path(request, monkeypatch):
    from pyjuice.constraints.backends.lifted.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 << 30 if request.param == "fused" else 1)
    monkeypatch.setattr(F, "DENSE_PRUNE_MIN_COLUMNS", 0)
    return request.param


def top_down_blocks(cc, data, missing, precision = "fp32"):
    """The forward then the full top-down pass; every sum region's rows of node_flows, [rows, B * slots] (cloned)."""
    prog, dev = cc._lifted_program(), cc.pc.params.device
    B, n = data.size(0), cc.n
    missing = missing.to(dev)
    observed = (~missing).any(dim = 0).cpu().numpy()
    x = torch.where(missing, 0, data.to(dev).long())
    obs_class = torch.where(missing, -1, cc.token_classes.token_class[x]).contiguous()
    pruning = prune(prog, obs_class, B) if observed.any() else None
    valid, kept = cc._prior_validity()
    plan = query_plan(prog, observed, None, valid.numpy(), prior_regions(prog, observed, B, kept, pruning))
    bufs = cc._buffers(B)
    flows = cc._flows(B, bufs)
    flows["node_flows"].fill_(float("nan"))                                 # every needed slot must be written
    with torch.no_grad():
        F.run_forward(cc, bufs, plan, x, missing, obs_class, pruning, precision)
        valid |= plan.todo
        run_backward(cc, bufs, flows, plan, x, missing, obs_class, pruning, precision)
    lay, nf = bufs["layout"], flows["node_flows"]
    out = []
    for r, (first, end, slots) in enumerate(cc.sum_regions):
        off, wid = lay["sum_offsets"][r], lay["sum_widths"][r]
        out.append(nf[off:off + (end - first) * wid].view(end - first, wid)[:, :B * slots].clone())
    return out


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


def assert_top_down_close(cc, got, want, data, missing):
    """Every sum region's top-down values (:func:`top_down_blocks`) against the reference's, on the live slots."""
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
    assert_top_down_close(cc, top_down_blocks(cc, data, missing), reference.top_down(cc, data, missing), data,
                          missing)


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
    assert_top_down_close(cc, top_down_blocks(cc, data, missing), reference.top_down(cc, data, missing), data,
                          missing)


@pytest.mark.parametrize("kind", ["pd_prod_dominated", "hand_permuted", "hmm"])
def test_top_down_values_are_deterministic(kind, build_pc):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence("mixed", n)
    a = top_down_blocks(cc, data, missing)
    b = top_down_blocks(cc, data, missing)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
