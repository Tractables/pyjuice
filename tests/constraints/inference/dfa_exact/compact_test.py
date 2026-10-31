"""
Compact storage of the per-sample values (:mod:`pyjuice.constraints.inference.dfa_exact.kernels.compact`): the
layout's rectangles, ranks and rows against the liveness flags. The values themselves: forward_test.py's,
top_down_test.py's and conditional_test.py's compact tests.
"""
import numpy as np
import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels.compact import compact_layout
from pyjuice.constraints.inference.dfa_exact.kernels.live import liveness

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


def _reference_layout(total, regions, inside, reg_iv, reg_rows, pad, slots, elem_rows, align, base):
    """compact_layout's host arrays as numpy computes them (the layout's helpers before `_layout_host`)."""
    aligned = lambda x: (np.asarray(x, dtype = np.int64) + align - 1) // align * align
    lanes = total[reg_iv]

    def of(keep):
        width = np.where(keep, aligned(lanes), 0)
        offset = np.zeros(len(width), dtype = np.int64)
        np.cumsum((reg_rows * width)[:-1], out = offset[1:])
        ivs = np.zeros(len(total), dtype = bool)
        ivs[reg_iv[keep]] = True
        group = np.where(np.append(ivs, False)[pad], np.append(total, 0)[pad], 0).max(axis = 1)
        widths = aligned(np.maximum(group, slots))
        return (width, offset, ivs, widths, int((elem_rows * widths).max(initial = 0)), int((reg_rows * width).sum()),
                int(lanes[keep].max(initial = 0)))

    return lanes, of(regions), of(inside) if inside is not None else None


@pytest.mark.parametrize("seed", range(6))
def test_the_layouts_host_arrays_are_numpys(seed):
    """`_layout_host` (numba) gives the arrays and sizes numpy's formulas give -- every region's rows' width and place,
    its intervals, every product group's width, the sizes and most lanes, of the top-down values and (``mars``) the
    inside ones after the input rows -- and the device's arrays in one buffer, 16 bytes apart; odd and even numbers of
    regions, some regions or none kept, groups padded past the last interval."""
    from pyjuice.constraints.inference.dfa_exact.kernels.compact import _layout_host
    rng = np.random.default_rng(seed)
    K, R, G, P = int(rng.integers(1, 40)), int(rng.integers(1, 30)), int(rng.integers(1, 8)), int(rng.integers(1, 6))
    total = rng.integers(0, 300, K).astype(np.int64)
    reg_iv = rng.integers(0, K, R).astype(np.int64)
    reg_rows = rng.integers(1, 64, R).astype(np.int64)
    pad = rng.integers(0, K + 1, (G, P)).astype(np.int64)                # (K: padding, an interval without lanes)
    slots, elem_rows = rng.integers(0, 100, G).astype(np.int64), rng.integers(1, 32, G).astype(np.int64)
    for regions, inside in ((rng.random(R) < 0.5, rng.random(R) < 0.5), (np.zeros(R, bool), None),
                            (np.ones(R, bool), np.zeros(R, bool))):
        base = int(rng.integers(0, 100)) * 16
        mars = inside is not None
        got = _layout_host(total, regions, inside if mars else regions, mars, reg_iv, reg_rows, pad, slots, elem_rows,
                           16, base)
        lanes, want, want_m = _reference_layout(total, regions, inside, reg_iv, reg_rows, pad, slots, elem_rows, 16,
                                                base)
        buf, step = got[0], got[1]
        assert step % 2 == 0 and step >= R and buf.size == (5 if mars else 3) * step
        assert np.array_equal(got[2], lanes)
        for g, w in zip(got[3:10], want):
            assert np.array_equal(np.asarray(g), np.asarray(w)), (g, w)
        rows = [want[1], want[0], lanes]
        if mars:
            m_width, m_offset, m_ivs, m_groups, m_size_e, m_size, m_most = want_m
            assert np.array_equal(got[10], m_width) and np.array_equal(got[11], base + m_offset)
            assert np.array_equal(got[12], m_ivs) and np.array_equal(got[13], m_groups)
            assert (got[14], got[15], got[16]) == (m_size_e, base + m_size, m_most)
            rows += [base + m_offset, m_width]
        for i, a in enumerate(rows):
            assert np.array_equal(buf[i * step:i * step + R], a) and not buf[i * step + R:(i + 1) * step].any()


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
