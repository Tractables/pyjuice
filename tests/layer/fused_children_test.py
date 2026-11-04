"""
The sum layers' fused forward and backward (`compute_children`): a sum kernel that forms its children's element values
from their products' child node values (`pyjuice.layer.fused_children.element_children`) instead of reading
`element_mars` -- so a decoder session need not run the product layers' forward first.

* the children table: for every element row, the node rows its product reads, the dummy node past them;
* the fused forward against the product layers' forward followed by the plain sum forward -- to the bit where the
  group's products have up to two children, to fp32 rounding beyond (the product kernel's `tl.sum` adds in a
  layout-dependent order) -- with `element_mars` poisoned (NaN) to show it is not read, on HMM (two children a
  product), HCLT, PD and RAT-SPN (some of three or four), and circuits of products over 3 and 5 children (padding);
  whole partitions and runs; bf16 and fp32 dots; the layers a circuit fuses (`fuses_children`) and the refusals
  outside the `tlmm` regime -- with tile tuning pinned off (the bit comparison holds at the heuristic tiles);
* the fused forward's own tuning: its candidates (taller node tiles at 8 warps), a whole launch measuring them and a
  launch of runs not, the values at every candidate;
* the fused backward the same way, against the product forward followed by the plain element-flow backward.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.fused_children import element_children
from pyjuice.layer.kernels import autotune

DEV = torch.device("cuda:0")
NAN = float("nan")


@pytest.fixture
def no_tuning():
    """Tile tuning pinned off: the fused kernel and the plain one then run the same heuristic tiles (a tuned tile
    shape changes the dot's reduction layout, ~1e-7, which the bit comparisons do not allow for)."""
    was = autotune.set_autotune(False)
    yield
    autotune.set_autotune(was)


def _hclt(latents, cats = 16, V = 64):
    torch.manual_seed(0)
    x = torch.randint(0, cats, (512, V))
    ns = juice.structures.HCLT(x.float(), num_latents = latents, input_dist = dists.Categorical(num_cats = cats))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), V, cats


def _pd():
    torch.manual_seed(0)
    ns = juice.structures.PD(data_shape = (8, 8), num_latents = 32, split_intervals = (2, 2),
                             input_dist = dists.Categorical(num_cats = 16))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), 64, 16


def _rat():
    torch.manual_seed(0)
    ns = juice.structures.RAT_SPN(num_vars = 64, num_latents = 32, depth = 3, num_repetitions = 4, num_pieces = 2,
                                  input_dist = dists.Categorical(num_cats = 8))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), 64, 8


def _hmm():
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 12, num_latents = 64, num_emits = 16)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), 12, 16


def _wide(k):
    """Sums over products of `k` children each (k = 3, 5: the table pads to 4, 8), two layers deep."""
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ins = [juice.inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 8)) for v in range(2 * k)]
        s1 = juice.summate(juice.multiply(*ins[:k]), num_node_blocks = 2)
        s2 = juice.summate(juice.multiply(*ins[k:]), num_node_blocks = 2)
        s3 = juice.summate(juice.multiply(s1, s2), num_node_blocks = 2)
    root = juice.summate(juice.multiply(s3), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(DEV), 2 * k, 8


CIRCUITS = {"hclt32": lambda: _hclt(32), "hclt128": lambda: _hclt(128), "pd": _pd, "rat": _rat, "hmm": _hmm,
            "k3": lambda: _wide(3), "k5": lambda: _wide(5)}


def _sum_layers(pc):
    return [(k, layer) for k, lg in enumerate(pc.inner_layer_groups) if lg.is_sum() for layer in lg]


def _products_below(pc, k):
    for prod in pc.inner_layer_groups[k - 1]:
        prod.forward(pc.node_mars, pc.element_mars)


def _same(got, want, K, gates):
    """Against the Triton path (the CUDA gates pinned off): bit-identical where every product of the group has at most
    two children (one add), else to the rounding a 1-ulp difference in an element value makes through a bf16 dot
    (MEASURED up to 4.5e-4). With the gates live the plain path may run a CUDA kernel, which agrees with the Triton
    kernel the fused one copies to ~2e-3 by design."""
    if not gates and K <= 2:
        return torch.equal(got, want)
    fin = torch.isfinite(want)
    return bool(torch.equal(torch.isfinite(got), fin) and torch.allclose(got[fin], want[fin], atol = 5e-3, rtol = 0.0))


@pytest.fixture(params = [False, True])
def gates(request, monkeypatch):
    """The sum layers' CUDA gates pinned off (the plain path is the Triton kernel the fused one copies), or live."""
    if not request.param:
        monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
        monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", False)
    return request.param


@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_children_table(name):
    """Every element row's value from the table (sum of its children's node values) equals what the product layers
    write; the dummy element's row is all dummy nodes; padding is the dummy node."""
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (8, V), device = DEV)
    pc(x)
    tables = element_children(pc)
    d = pc.num_dummy_eles                                           # the dummy block: rows [0, d), -inf, no product's
    assert bool(torch.isinf(pc.element_mars[:d]).all())
    for k, lg in enumerate(pc.inner_layer_groups):
        if not lg.is_prod():
            assert tables[k] is None
            continue
        table, first = tables[k]
        assert first == d and table.shape[1] & (table.shape[1] - 1) == 0
        em = torch.full_like(pc.element_mars, NAN)
        for layer in lg:
            layer.forward(pc.node_mars, em)                          # this group's values (the rows are shared)
        got = pc.node_mars[table].sum(dim = 1)                       # [n_k, B] (torch's add order, not the kernel's)
        want = em[d:d + table.shape[0]]
        assert torch.allclose(got, want, atol = 1e-4, rtol = 1e-6) and torch.equal(torch.isinf(got), torch.isinf(want)), k
    # The kernels keep only row 0 -inf by its id: the padding of every sum layer's tables points there and nowhere
    # else in the dummy block
    for _, layer in _sum_layers(pc):
        for cids in layer.partitioned_cids:
            assert bool(((cids == 0) | (cids >= d)).all())
        for chids in layer.partitioned_chids:
            assert bool((chids >= d).all())


def _fused_cases(pc, B):
    """The sum layers whose every forward partition fuses at B (and their layer-group index)."""
    return [(k, layer) for k, layer in _sum_layers(pc) if all(layer.fuses_children(B))]


@pytest.mark.parametrize("B", [16, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_fused_forward_is_the_product_then_sum_forward(name, B, gates, no_tuning):
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)                                                           # every value in place, kernels compiled
    tables = element_children(pc)
    cases = _fused_cases(pc, B)
    assert cases, "no layer fuses"
    for k, layer in cases:
        a, b = layer.nid_range
        _products_below(pc, k)
        layer.forward(pc.node_mars, pc.element_mars, pc.params)
        want = pc.node_mars[a:b].clone()
        pc.element_mars[1:] = NAN                                    # not read by the fused kernel
        pc.node_mars[a:b] = NAN
        layer.forward(pc.node_mars, pc.element_mars, pc.params, compute_children = tables[k - 1])
        K = tables[k - 1][0].shape[1]
        assert _same(pc.node_mars[a:b], want, K, gates), (name, k)
        assert torch.isnan(pc.element_mars[1:]).all()                # and not written
        # runs of rows: the first row and the second half of partition 0
        n = layer.partitioned_nids[0].numel()
        for runs in ([(0, 1)], [(n // 2, n)]):
            pc.node_mars[a:b] = NAN
            layer.forward(pc.node_mars, pc.element_mars, pc.params, compute_children = tables[k - 1],
                          partition_runs = [runs] + [[]] * (layer.num_fw_partitions - 1))
            rows = torch.zeros(b - a, dtype = torch.bool, device = DEV)
            for s, e in runs:
                for r in layer.partitioned_nids[0][s:e].tolist():
                    rows[r - a:r - a + layer.block_size] = True
            assert _same(pc.node_mars[a:b][rows], want[rows], K, gates) and torch.isnan(pc.node_mars[a:b][~rows]).all()
        _products_below(pc, k)                                       # restore for the layers above
        layer.forward(pc.node_mars, pc.element_mars, pc.params)


def test_fp32_dots_fuse_too(gates, no_tuning):
    torch.manual_seed(0)
    x = torch.randint(0, 16, (512, 64))
    ns = juice.structures.HCLT(x.float(), num_latents = 64, input_dist = dists.Categorical(num_cats = 16))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, precision = "fp32").to(DEV)
    data = torch.randint(0, 16, (32, 64), device = DEV)
    pc(data)
    tables = element_children(pc)
    for k, layer in _fused_cases(pc, 32):
        a, b = layer.nid_range
        _products_below(pc, k)
        layer.forward(pc.node_mars, pc.element_mars, pc.params, precision = "fp32")
        want = pc.node_mars[a:b].clone()
        pc.node_mars[a:b] = NAN
        layer.forward(pc.node_mars, pc.element_mars, pc.params, precision = "fp32", compute_children = tables[k - 1])
        assert _same(pc.node_mars[a:b], want, tables[k - 1][0].shape[1], gates)


@pytest.mark.parametrize("B", [16, 128])
def test_the_fused_forward_tunes_its_own_tiles(B, monkeypatch):
    """A whole launch of the fused forward measures its own candidates -- the heuristic tiles, then 2x and 4x the node
    tile at 8 warps (within the block) -- keyed apart from the plain kernel's; a launch of runs takes the default
    without measuring; every candidate's values agree with the plain path to the rounding of a bf16 dot."""
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS["hclt128"]()
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    tables = element_children(pc)
    k, layer = max(_fused_cases(pc, B), key = lambda kl: kl[1].partitioned_nids[0].numel())
    a, b = layer.nid_range
    _products_below(pc, k)
    layer.forward(pc.node_mars, pc.element_mars, pc.params)
    want = pc.node_mars[a:b].clone()
    tk, tm, bb = layer._fw_block_sparse_tiles(layer.partitioned_cids[0].size(1), B, 0, False)
    picks = []
    real_pick = autotune.pick

    def spy(key, candidates, bench, *args, **kwargs):
        picks.append((key, list(candidates)))
        for c in candidates:                                         # every candidate runs and agrees
            pc.node_mars[a:b] = NAN
            bench(c)
            got = pc.node_mars[a:b]
            assert torch.allclose(got, want, atol = 5e-3, rtol = 0.0) and torch.equal(torch.isinf(got), torch.isinf(want)), c
        return real_pick(key, candidates, bench, *args, **kwargs)

    monkeypatch.setattr(autotune, "pick", spy)
    was = autotune.set_autotune(True, clear_cache = True)
    try:
        n = layer.partitioned_nids[0].numel()
        layer.forward(pc.node_mars, pc.element_mars, pc.params, compute_children = tables[k - 1],
                      partition_runs = [[(0, max(1, n // 2))]] + [[]] * (layer.num_fw_partitions - 1))
        assert picks == []                                           # runs: the default, nothing measured
        layer.forward(pc.node_mars, pc.element_mars, pc.params, compute_children = tables[k - 1])
        key, cands = picks[0]
        assert key[0] is SL.fw_fused._fw_triton_block_sparse_tlmm_fused_kernel and key[-1] == (tm, bb, None)
        assert cands == [(tm, bb, None)] + [(m, bb, 8) for m in (tm * 2, tm * 4) if m <= layer.block_size]
        assert len(cands) >= 2 and autotune.cached(key) in cands
        assert torch.allclose(pc.node_mars[a:b], want, atol = 5e-3, rtol = 0.0)
    finally:
        autotune.set_autotune(was, clear_cache = True)


def test_what_fuses_and_what_refuses():
    """Below the `tlmm` regime (small blocks, small batch) nothing fuses and `compute_children` raises; the dense
    cuBLAS path does not fuse either; MPE propagation does not."""
    pc, V, C = _hclt(32)
    tables = element_children(pc)
    x = torch.randint(0, C, (4, V), device = DEV)
    pc(x)
    layers = [layer for _, layer in _sum_layers(pc)]
    assert not any(any(layer.fuses_children(4)) for layer in layers)            # batch 4: the csmm2 regime
    assert all(all(layer.fuses_children(16)) for layer in layers if layer.block_size >= 16)
    assert not any(any(layer.fuses_children(16, propagation_alg = "MPE")) for layer in layers)
    with pytest.raises(ValueError, match = "tlmm"):
        layers[0].forward(pc.node_mars, pc.element_mars, pc.params, compute_children = tables[0])
    # the dense path: a 1024-block layer in its gate
    torch.manual_seed(0)
    ins = [juice.inputs(v, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = 4)) for v in range(2)]
    s = juice.summate(juice.multiply(*ins), num_node_blocks = 1, block_size = 1024)
    root = juice.summate(juice.multiply(s), num_node_blocks = 1, block_size = 1)
    root.init_parameters()
    dense_pc = juice.compile(root, verbose = False).to(DEV)
    dense_layer = next(l for _, l in _sum_layers(dense_pc) if l.block_size == 1024)
    assert dense_layer.fuses_children(64) == [True]                  # 1024 edges x 64 is below the dense path's work
    assert dense_layer.fuses_children(512) == [False]                 # in its gate
    with pytest.raises(ValueError, match = "dense"):
        y = torch.randint(0, 4, (512, 2), device = DEV)
        dense_pc(y)
        dense_layer.forward(dense_pc.node_mars, dense_pc.element_mars, dense_pc.params,
                            compute_children = element_children(dense_pc)[0])


def _bk_cases(pc, B):
    return [(k, layer) for k, layer in _sum_layers(pc) if all(layer.fuses_children_backward(B))]


@pytest.mark.parametrize("logspace", [True, False])
@pytest.mark.parametrize("B", [16, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_fused_backward_is_the_product_forward_then_element_flows(name, B, logspace, gates):
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    pc.backward(x, compute_param_flows = False, logspace_flows = logspace)
    tables = element_children(pc)
    kw = dict(logspace_flows = logspace, allow_modify_flows = False, propagation_alg = "LL")
    nf, ef, nm, em = pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars
    cases = _bk_cases(pc, B)
    assert cases, "no layer fuses"
    for k, layer in cases:
        starts = [c.tolist() for c in layer.partitioned_chids]
        widths = list(layer.cs_block_sizes)
        lo = min(min(s) for s in starts)
        hi = max(max(s) + w for s, w in zip(starts, widths))
        covered = torch.zeros(hi - lo, dtype = torch.bool, device = DEV)   # this layer's child rows (not contiguous)
        for s_, w in zip(starts, widths):
            for r in s_:
                covered[r - lo:r - lo + w] = True
        _products_below(pc, k)                                       # the values the plain kernel reads
        ef[lo:hi] = NAN
        layer.backward(nf, ef, nm, em, pc.params, **kw)
        want = ef[lo:hi].clone()
        em[1:] = NAN                                                 # not read by the fused kernel
        ef[lo:hi] = NAN
        layer.backward(nf, ef, nm, em, pc.params, compute_children = tables[k - 1], **kw)
        K = tables[k - 1][0].shape[1]
        assert _same(ef[lo:hi][covered], want[covered], K, gates), (name, k)
        assert torch.isnan(ef[lo:hi][~covered]).all()
        # a run: the first row of partition 0
        ef[lo:hi] = NAN
        layer.backward(nf, ef, nm, em, pc.params, compute_children = tables[k - 1],
                       partition_runs = [[(0, 1)]] + [[]] * (layer.num_bk_partitions - 1), **kw)
        r = starts[0][0]
        assert _same(ef[r:r + widths[0]], want[r - lo:r - lo + widths[0]], K, gates)
        assert torch.isnan(ef[lo:r]).all() and torch.isnan(ef[r + widths[0]:hi]).all()
        _products_below(pc, k)
        ef[lo:hi] = want


def test_the_backward_refuses_what_it_cannot_fuse():
    pc, V, C = _hclt(32)
    tables = element_children(pc)
    table = next(t for t in tables if t is not None)
    x = torch.randint(0, C, (16, V), device = DEV)
    pc(x)
    pc.backward(x, compute_param_flows = False)
    layer = next(l for _, l in _sum_layers(pc) if l.block_size >= 16)
    with pytest.raises(ValueError, match = "element flows only"):
        layer.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params,
                       param_flows = torch.zeros(1, device = DEV), compute_children = table)
    with pytest.raises(ValueError, match = "element flows only"):
        layer.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params,
                       allow_modify_flows = True, compute_children = table)
    assert not any(any(l.fuses_children_backward(16, propagation_alg = "MPE")) for _, l in _sum_layers(pc))
    small = next(l for _, l in _sum_layers(pc) if l.block_size == 1)              # the root: the sparse mode
    assert small.fuses_children_backward(16) == [False] * small.num_bk_partitions
    with pytest.raises(ValueError, match = "block-sparse"):
        small.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params,
                       compute_children = table, logspace_flows = True)
