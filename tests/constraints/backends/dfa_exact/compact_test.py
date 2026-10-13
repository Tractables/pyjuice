"""
Compact storage of the per-sample values (:mod:`pyjuice.constraints.backends.dfa_exact.kernels.compact`): the
layout's rectangles, ranks and rows against the liveness flags. The values themselves: forward_test.py's,
top_down_test.py's and conditional_test.py's compact tests.
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
    rng = np.random.default_rng(0)
    regions, inside = (rng.random(len(cc.backend.sum_regions)) < 0.5 for _ in range(2))
    lv = liveness(prog, obs_classes(cc, data, missing), B)
    cl = compact_layout(cc.backend, lv, regions, inside = inside)
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
    assert cl.width.tolist() == want_width.tolist()
    rows = prog.reg_rows
    assert cl.offset.tolist() == np.concatenate([[0], np.cumsum(rows * want_width)[:-1]]).tolist()
    assert cl.size == int((rows * want_width).sum())
    assert torch.equal(cl.lanes_t.cpu(), torch.from_numpy(lanes))
    assert torch.equal(cl.offset_t.cpu(), torch.tensor(cl.offset)) and torch.equal(cl.width_t.cpu(),
                                                                                  torch.tensor(cl.width))
    assert cl.steps >= int(np.ceil(np.log2(B + 1)))
    # the inside values: the input rows first (a value per sample), then the regions given, as the top-down values'
    mars, inputs = cl.mars, (cc.backend.input_range[1] * B + 15) // 16 * 16
    want_width = np.where(inside, (lanes + 15) // 16 * 16, 0)
    assert mars["sum_widths"].tolist() == want_width.tolist()
    assert mars["sum_offsets"].tolist() == (inputs + np.concatenate([[0], np.cumsum(rows * want_width)[:-1]])).tolist()
    assert mars["node_size"] == inputs + int((rows * want_width).sum())
    assert torch.equal(mars["sum_offsets_t"].cpu(), torch.tensor(mars["sum_offsets"]))
    assert torch.equal(mars["sum_widths_t"].cpu(), torch.tensor(mars["sum_widths"]))
    assert mars["lanes"] == int(lanes[inside].max(initial = 0))


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_product_groups_intervals_in_one_table(kind, build_pc):
    """The Program's table of the product groups' intervals: a row per group, its products' intervals, then padding one
    past the last interval -- which a query's widths read as an interval without lanes."""
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc.backend.program()
    assert prog.elem_iv_pad.shape[0] == len(cc.backend.element_regions)
    for pi, row in enumerate(prog.elem_iv_pad.tolist()):
        ivs = prog.elem_ivs.get(pi, np.zeros(0, dtype = np.int64)).tolist()
        assert row[:len(ivs)] == ivs and set(row[len(ivs):]) <= {len(prog.all_ivs)}, pi


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_product_rows_hold_compact_values_or_a_block_kept_once(kind, observed, build_pc):
    """A product group's rows are as wide as the widest compact row of its products whose regions keep values per
    sample, or one sample's block (a value kept once), whichever is wider -- the top-down values' (``elem_width``) and
    the inside values' (``mars``), each for their own regions, with a prefix observed or nothing; ``element_size`` is
    what they take."""
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc.backend.program()
    B = 4
    data = torch.zeros(B, n, dtype = torch.long)
    missing = torch.arange(n)[None, :].expand(B, n) >= (n // 2 if observed else 0)
    rng = np.random.default_rng(1)
    regions, inside = (rng.random(len(cc.backend.sum_regions)) < 0.5 for _ in range(2))
    lv = liveness(prog, obs_classes(cc, data, missing), B)
    cl = compact_layout(cc.backend, lv, regions, inside = inside)
    groups = cc.backend.element_regions
    for marks, widths, size in ((regions, cl.elem_width, cl.element_size),
                                (inside, cl.mars["element_widths"], cl.mars["element_size"])):
        ivs = set(prog.reg_iv_all[marks].tolist())
        for pi, (first, end, slots) in enumerate(groups):
            lanes = max([int(cl.total[k]) for k in prog.elem_ivs.get(pi, ()) if k in ivs], default = 0)
            assert widths[pi] == (max(lanes, slots) + 15) // 16 * 16, pi
        assert size == max((end - first) * w for (first, end, _), w in zip(groups, widths))
