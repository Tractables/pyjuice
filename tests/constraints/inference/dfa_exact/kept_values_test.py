"""
The inside values a session keeps once (``prior_mars``, :class:`~pyjuice.constraints.inference.dfa_exact.state.
PriorPool`) under a memory budget: a decoder whose budget holds a few of them at a time gives the rows of one that
holds them all, bit for bit -- left to right and as beams, conditionals and marginals --, evicting and computing
again (each value a few times a session at most: once, and once more as its segment is entered); every value a pass
reads has an extent while it runs; a budget too small for one call's values is refused, with the sizes (a PD's first
call reads nearly every one: no budget below their total serves it); the option is checked; without a budget nothing
is evicted.
"""
import numpy as np
import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact import state as S

V = 3


def steps(n, beams = False, B = 4, seed = 0):
    """Left-to-right calls over ``n`` positions (rows permuted every call when ``beams``): ``(data, missing, targets)``."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    out = []
    for t in range(n):
        rows = torch.randperm(B, generator = g) if beams and t > 0 else torch.arange(B)
        missing = torch.arange(n)[None, :].expand(B, n) >= t
        out.append((data[rows].cuda(), missing.cuda(), [t]))
    return out


def full_bytes(cc):
    """The bytes of every sum region kept once, as a batch of one lays them out."""
    return cc.backend.state.batch_layout(1)["node_size"] * 4


def spy_allocations(monkeypatch, record):
    """Every :meth:`PriorPool.allocate`: the intervals the plan computes appended to ``record``, and the pool checked:
    every value the plan reads has an extent after it, where the table says, inside the buffer."""
    real = S.PriorPool.allocate

    def allocate(self, plan):
        out = real(self, plan)
        for iv in np.nonzero(plan.prior_read)[0]:
            for r in self.iv_regions[iv]:
                assert int(r) in self.extents, (iv, r)
        for r, (off, size) in self.extents.items():
            assert out["offset"][r] == off and off + size <= self.buffer.numel()
        record.append(int(plan.todo.sum()))
        return out

    monkeypatch.setattr(S.PriorPool, "allocate", allocate)


@pytest.mark.parametrize("beams, share, bound", [(False, 0.5, 3), (True, 0.5, 3), (False, 0.25, 4)])
def test_a_budget_gives_the_rows_of_the_full_storage(beams, share, bound, build_pc, monkeypatch):
    """An HMM over 32 positions, left to right: a call reads the suffixes after its prefix, two values of 32. A pool
    of half the values holds the checkpoints and a segment (every value computed once more at most); one of a
    quarter does not (the stride of the least estimated recomputation), and recomputes more."""
    n = 32
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    calls = steps(n, beams)
    full = []
    spy_allocations(monkeypatch, full)
    with cc.decoder(precision = "fp32") as decoder:
        want = [decoder.conditional(x, m, targets = tg) for x, m, tg in calls]
        want.append(decoder.marginal(*calls[n // 2][:2]))
        pool = cc.backend.state._prior_pool
        assert pool.evictions == 0 and pool.buffer.numel() * 4 == full_bytes(cc) and pool.stride == 1
    K = sum(full)                                          # (each value kept once computed once)
    assert 0 < K <= len(cc.backend.program().sum_ivs)
    budget = int(full_bytes(cc) * share)
    small = []
    spy_allocations(monkeypatch, small)
    with cc.decoder(precision = "fp32", memory_budget = budget) as decoder:
        got = [decoder.conditional(x, m, targets = tg) for x, m, tg in calls]
        got.append(decoder.marginal(*calls[n // 2][:2]))
        pool = cc.backend.state._prior_pool
        assert pool.buffer.numel() * 4 <= budget + 64 and pool.evictions > 0 and pool.stride > 1
        held = len(pool.extents)
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert held < len(cc.backend.sum_regions)
    assert K < sum(small) <= bound * K, (K, sum(small), small)   # (computed again as their segments are entered)


def test_a_budget_too_small_for_one_call_is_refused(build_pc):
    """A PD's first call reads nearly every value kept once (the top-down values it keeps once read their siblings'
    inside values): a budget below their total refuses it, saying what they take; the full total serves it."""
    n = 16
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    x, m, tg = steps(n)[0]
    with pytest.raises(RuntimeError, match = "values kept once that this query reads take .* more than the"):
        with cc.decoder(precision = "fp32", memory_budget = full_bytes(cc) * 9 // 10) as decoder:
            decoder.conditional(x, m, targets = tg)
    with cc.decoder(precision = "fp32", memory_budget = full_bytes(cc)) as decoder:
        decoder.conditional(x, m, targets = tg)
        assert cc.backend.state._prior_pool.evictions == 0
    with pytest.raises(ValueError, match = "memory_budget"):
        cc.decoder(memory_budget = 0)
    with pytest.raises(ValueError, match = "memory_budget"):
        cc.decoder(memory_budget = 2.5)
