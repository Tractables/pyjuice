"""
Compact storage of the per-sample top-down values (:mod:`pyjuice.constraints.backends.dfa_exact.kernels.compact`): the
layout's rectangles, ranks and rows against the liveness flags. The values themselves: top_down_test.py's and
conditional_test.py's compact tests.
"""
import numpy as np
import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.backends.dfa_exact.kernels.compact import compact_layout
from pyjuice.constraints.backends.dfa_exact.kernels.live import liveness

V = 3
KINDS = {"hmm": 6, "pd": 8, "hand_permuted": 5, "left_linear": 6}                     # see conftest.py


def obs_classes(cc, data, missing):
    dev = cc.pc.params.device
    x = torch.where(missing, 0, data).to(dev)
    return torch.where(missing.to(dev), -1, cc.backend.token_classes.token_class[x]).int().contiguous()


@pytest.mark.parametrize("kind", list(KINDS))
def test_rectangles_ranks_and_rows(kind, build_pc):
    """Every interval's lanes per sample are its live columns at the entry times those at the exit, sample after
    sample (``off``, ``total``); every column's rank is its place among its sample's live ones at the boundary, -1 if
    dead; the regions given get rows as wide as their interval's lanes (aligned), one after the other, the others none
    -- with samples whose evidence breaks the constraint (no lane) and an odd batch."""
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc.backend.program()
    g = torch.Generator().manual_seed(0)
    B = 37
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.6
    data[:3], missing[:3] = 1, False                                     # "1 1 ..." never contains "1 2"
    regions = np.random.default_rng(0).random(len(cc.backend.sum_regions)) < 0.5
    lv = liveness(prog, obs_classes(cc, data, missing), B)
    cl = compact_layout(cc.backend, lv, regions)
    live, rank = lv.live.cpu(), cl.rank.cpu()
    widths = [int(w) for w in cc.backend.layout.width.tolist()[:n]] + [1]
    cnt = torch.zeros(B, n + 1, dtype = torch.long)
    for b in range(B):
        for t, w in enumerate(widths):
            flags = live[b, t, :w] != 0
            cnt[b, t] = int(flags.sum())
            assert torch.equal(rank[b, t, :w], torch.where(flags, torch.cumsum(flags, 0) - 1, -1).int()), (b, t)
    assert int(cnt[:3, :].prod(dim = 1).sum()) == 0                     # the broken samples: nothing live somewhere
    off = cl.off.cpu()
    for k, (a, e) in enumerate(prog.all_ivs):
        want = torch.cat([torch.zeros(1, dtype = torch.long), torch.cumsum(cnt[:, a] * cnt[:, e], 0)])
        assert torch.equal(off[k], want), (a, e)
        assert cl.total[k] == want[-1]
    lanes = cl.total[prog.reg_iv_all]
    want_width = np.where(regions, (lanes + 15) // 16 * 16, 0)
    assert cl.width == want_width.tolist()
    rows = prog.reg_rows
    assert cl.offset == np.concatenate([[0], np.cumsum(rows * want_width)[:-1]]).tolist()
    assert cl.size == int((rows * want_width).sum())
    assert torch.equal(cl.lanes_t.cpu(), torch.from_numpy(lanes))
    assert cl.steps >= int(np.ceil(np.log2(B + 1)))


def test_product_rows_hold_compact_values_or_a_block_kept_once(build_pc):
    """A product group's rows are as wide as its products' widest compact row, or one sample's block (a value kept
    once), whichever is wider; ``element_size`` is what they take."""
    n = KINDS["hmm"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    prog = cc.backend.program()
    data = torch.zeros(4, n, dtype = torch.long)
    missing = torch.arange(n)[None, :].expand(4, n) >= n // 2
    lv = liveness(prog, obs_classes(cc, data, missing), 4)
    cl = compact_layout(cc.backend, lv, np.ones(len(cc.backend.sum_regions), dtype = bool))
    for pi, (first, end, slots) in enumerate(cc.backend.element_regions):
        ivs = prog.elem_ivs[pi]
        assert cl.elem_width[pi] == (max(int(cl.total[ivs].max()), slots) + 15) // 16 * 16
    assert cl.element_size == max((end - first) * w for (first, end, _), w in
                                  zip(cc.backend.element_regions, cl.elem_width))
