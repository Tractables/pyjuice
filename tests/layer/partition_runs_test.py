"""
`partition_runs`: a sum or product layer's launch of some rows of a partition's tables gives those rows the bits of
the whole partition's launch and leaves every other row as it was -- on every path the layers dispatch to:

* the sum forward's block-sparse Triton kernels (tlmm, csmm1, csmm2), its CUDA tlmm and small-batch kernels, the
  dense cuBLAS path (one child group) and both sparse kernels;
* the element-flow backward's block-sparse Triton kernels (dot, csmm2), its CUDA tlmm and small-batch kernels and the
  small sparse kernel, in log-space and linear flows;
* the product kernel and its large-edge one, forward and backward;

with runs at a partition's start, middle and end, one row, several runs and all of them, the layer's other partitions
given none (skipped). Each case asserts that the paths it means to take were taken (spies on the launch sites), with
the CUDA gates' timed choices forced both ways. And what takes no runs: a dense partition of several child groups
(`forward_runs_ok`), parameter flows, `allow_modify_flows`, an external-parameter layer.

Not reached: the element-flow backward's large sparse kernel (more than 32768 child blocks in a partition) and the
product kernel of Triton 2.0.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import ExternalParamsSumLayer
from pyjuice.layer import prod_layer as PL
from pyjuice.layer import sum_layer as SL
from pyjuice.nodes import LowRankSumParams

DEV = torch.device("cuda:0")
NAN = float("nan")


class _Spy:
    """A Triton kernel that notes its launches."""
    def __init__(self, kernel, name, seen):
        self.kernel, self.name, self.seen = kernel, name, seen

    def __getitem__(self, grid):
        self.seen.add(self.name)
        return self.kernel[grid]

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def _spy_fn(fn, name, seen):
    def spied(*args, **kwargs):
        seen.add(name)
        return fn(*args, **kwargs)
    return spied


TRITON = ((SL.fw_bsparse, ("_fw_triton_block_sparse_tlmm_kernel", "_fw_triton_block_sparse_csmm1_kernel",
                           "_fw_triton_block_sparse_csmm2_kernel")),
          (SL.fw_sparse, ("_fw_triton_sparse_kernel", "_fw_triton_large_sparse_kernel")),
          (SL.bk_ele_bsparse, ("_bk_triton_block_sparse_ele_kernel", "_bk_triton_block_sparse_ele_csmm2_kernel")),
          (SL.bk_ele_sparse, ("_bk_triton_sparse_ele_kernel", "_bk_triton_large_sparse_ele_kernel")),
          (PL.kernels, ("_forward_backward_kernel_2d", "_forward_backward_kernel_large")))
CUDA = ("tlmm_forward_sum", "smallbatch_forward_sum", "ele_backward_sum", "smallbatch_ele_backward_sum",
        "tlmm_forward_sum_pipelined", "ele_backward_sum_pipelined")


@pytest.fixture
def hits(monkeypatch):
    """The launch sites a test reaches, by name (the Triton kernels', `cuda.<launcher>`, `dense`)."""
    seen = set()
    for module, names in TRITON:
        for name in names:
            monkeypatch.setattr(module, name, _Spy(getattr(module, name), name, seen))
    for name in CUDA:
        monkeypatch.setattr(SL.cuda_kernels, name, _spy_fn(getattr(SL.cuda_kernels, name), "cuda." + name, seen))
    monkeypatch.setattr(SL.SumLayer, "_forward_dense", _spy_fn(SL.SumLayer._forward_dense, "dense", seen))
    return seen


class _Autotune:
    """The sum layer's `autotune` with the CUDA gates' timed choices forced: their first CUDA candidate (``cuda``) or
    the Triton reference -- the same for a whole launch and for a launch of some rows, which reads it back."""
    GATES = ("sum_fw_tlmm_cuda", "sum_fw_smallbatch_cuda")

    def __init__(self, real, cuda: bool):
        self.real, self.cuda, self.forced = real, cuda, {}

    def __getattr__(self, name):
        return getattr(self.real, name)

    def _gate(self, key):
        return isinstance(key, tuple) and len(key) > 0 and key[0] in self.GATES

    def cached(self, key):
        return self.forced.get(key) if self._gate(key) else self.real.cached(key)

    def choose(self, key, candidates):                       # the tlmm gate: `(choice, run)` pairs
        if not self._gate(key):
            return self.real.choose(key, candidates)
        choices = [c for c, _ in candidates]
        self.forced[key] = next((c for c in choices if c[0] == "cuda"), choices[0]) if self.cuda else choices[0]
        return self.forced[key]

    def pick(self, key, candidates, bench, *args, **kwargs):   # the small-batch gate: choices
        if not self._gate(key):
            return self.real.pick(key, candidates, bench, *args, **kwargs)
        self.forced[key] = next((c for c in candidates if c[0] == "cuda"), candidates[0]) if self.cuda \
                           else candidates[0]
        return self.forced[key]

    def best_of(self, candidates, *args, **kwargs):           # the element-flow gates: `(choice, run)` pairs
        choices = [c for c, _ in candidates]
        if all(isinstance(c, tuple) and c[0] in ("triton", "cuda", "cuda_pipe", "cuda_prepass") for c in choices):
            return next((c for c in choices if c[0] == "cuda"), choices[0]) if self.cuda else choices[0]
        return self.real.best_of(candidates, *args, **kwargs)


def run_sets(n):
    """Runs of rows of an `n`-row partition: its first, last and middle row, its halves, several runs, all of it."""
    sets = [[(0, 1)], [(n - 1, n)], [(0, n)]]
    if n >= 3:
        sets += [[(n // 2, n // 2 + 1)], [(0, n // 2)], [(n // 2, n)]]
    if n >= 6:
        sets.append([(0, 1), (2, n - 3), (n - 1, n)])
    out = []
    for s in sets:
        if s not in out:
            out.append(s)
    return out


def covered(starts, width, runs, lo, size):
    """The buffer rows (from `lo`, `size` of them) that rows `runs` of a table write, row `r` covering
    `[starts[r], starts[r] + width)`."""
    mask = torch.zeros(size, dtype = torch.bool, device = DEV)
    for a, b in runs:
        for r in range(a, b):
            mask[starts[r] - lo:starts[r] - lo + width] = True
    return mask


def only(p, runs, num_partitions):
    return [runs if q == p else [] for q in range(num_partitions)]


def check_sum_forward(layer, nm, em, params, precision):
    a, b = layer.nid_range
    layer.forward(nm, em, params, precision = precision)
    ref = nm[a:b].clone()
    P = layer.num_fw_partitions
    for p in range(P):
        starts = layer.partitioned_nids[p].tolist()
        for runs in run_sets(len(starts)):
            nm[a:b] = NAN
            layer.forward(nm, em, params, precision = precision, partition_runs = only(p, runs, P))
            want = covered(starts, layer.block_size, runs, a, b - a)
            assert torch.equal(nm[a:b][want], ref[want]), ("sum forward", p, runs)
            assert nm[a:b][~want].isnan().all(), ("sum forward wrote other rows", p, runs)
    nm[a:b] = ref


def check_sum_backward(layer, nf, ef, nm, em, params, logspace):
    P = layer.num_bk_partitions
    rows = [(layer.partitioned_chids[p].tolist(), layer.cs_block_sizes[p]) for p in range(P)]
    lo = min(min(s) for s, _ in rows)
    hi = max(max(s) + w for s, w in rows)
    kw = dict(logspace_flows = logspace, allow_modify_flows = False, propagation_alg = "LL")
    ef[lo:hi] = NAN
    layer.backward(nf, ef, nm, em, params, **kw)
    ref = ef[lo:hi].clone()
    for p in range(P):
        starts, width = rows[p]
        for runs in run_sets(len(starts)):
            ef[lo:hi] = NAN
            layer.backward(nf, ef, nm, em, params, partition_runs = only(p, runs, P), **kw)
            want = covered(starts, width, runs, lo, hi - lo)
            assert torch.equal(ef[lo:hi][want], ref[want]), ("element flows", p, runs)
            assert ef[lo:hi][~want].isnan().all(), ("element flows of other rows", p, runs)


def check_prod_forward(layer, nm, em):
    a, b = layer._layer_nid_range
    layer.forward(nm, em)
    ref = em[a:b].clone()
    P = layer.num_fw_partitions
    for p in range(P):
        starts = layer.partitioned_nids[p].tolist()
        for runs in run_sets(len(starts)):
            em[a:b] = NAN
            layer.forward(nm, em, partition_runs = only(p, runs, P))
            want = covered(starts, layer.block_size, runs, a, b - a)
            assert torch.equal(em[a:b][want], ref[want]), ("product forward", p, runs)
            assert em[a:b][~want].isnan().all(), ("product forward wrote other rows", p, runs)
    em[a:b] = ref


def check_prod_backward(layer, nf, ef, logspace):
    """The children's flows the layer adds to: from `-inf` (log-space) or 0, the rows not computed NaN."""
    P = layer.num_bk_partitions
    rows = [layer.partitioned_u_cids[p].tolist() for p in range(P)]
    bs = layer.block_size
    lo, hi = min(min(r) for r in rows), max(max(r) for r in rows) + bs
    a, b = layer._layer_nid_range
    g = torch.Generator(device = DEV).manual_seed(0)
    vals = torch.randn(b - a, ef.size(1), generator = g, device = DEV)
    ef[a:b] = vals if logspace else vals.exp()
    start = -float("inf") if logspace else 0.0
    every = torch.zeros(hi - lo, dtype = torch.bool, device = DEV)
    for p in range(P):
        every |= covered(rows[p], bs, [(0, len(rows[p]))], lo, hi - lo)
    nf[lo:hi] = NAN
    nf[lo:hi][every] = start
    layer.backward(nf, ef, logspace_flows = logspace)
    ref = nf[lo:hi].clone()
    for p in range(P):
        for runs in run_sets(len(rows[p])):
            want = covered(rows[p], bs, runs, lo, hi - lo)
            nf[lo:hi] = NAN
            nf[lo:hi][want] = start
            layer.backward(nf, ef, logspace_flows = logspace, partition_runs = only(p, runs, P))
            assert torch.equal(nf[lo:hi][want], ref[want]), ("children's flows", p, runs)
            assert nf[lo:hi][~want].isnan().all(), ("children's flows of other rows", p, runs)


def check_circuit(pc, x, logspace = True):
    """Every sum and product layer of `pc` and every partition, under evidence `x` (its own forward and backward)."""
    pc(x)
    pc.backward(x, compute_param_flows = False, logspace_flows = logspace)
    nm, em, nf, ef, params = pc.node_mars, pc.element_mars, pc.node_flows, pc.element_flows, pc.params
    for k, lg in enumerate(pc.inner_layer_groups):
        for layer in lg:
            if lg.is_prod():
                check_prod_forward(layer, nm, em)
                check_prod_backward(layer, nf.clone(), ef.clone(), logspace)
            else:
                for prod in pc.inner_layer_groups[k - 1]:
                    prod.forward(nm, em)                     # this layer's children
                check_sum_forward(layer, nm, em, params, pc.precision)
                check_sum_backward(layer, nf, ef, nm, em, params, logspace)


def hclt(latents):
    torch.manual_seed(0)
    x = torch.randint(0, 16, (512, 64))
    ns = juice.structures.HCLT(x.float(), num_latents = latents, input_dist = dists.Categorical(num_cats = 16))
    ns.init_parameters()
    return juice.compile(ns, verbose = False).to(DEV), 64, 16


def pd():
    torch.manual_seed(0)
    ns = juice.structures.PD(data_shape = (8, 8), num_latents = 16, split_intervals = (2, 2),
                             input_dist = dists.Categorical(num_cats = 16))
    ns.init_parameters()
    return juice.compile(ns, verbose = False).to(DEV), 64, 16


def wide(n = 2100):
    """An input of variable 0 under `n` products, each with its own input of variable 1 (the large product backward:
    `n` parents), under one sum; beside it `n` single-node sums over their own variables (a partition of `n` rows: the
    large sparse forward) under one product (`n` children: the large product forward)."""
    torch.manual_seed(0)

    def leaf(v):
        return juice.inputs(v, num_node_blocks = 2, block_size = 1, dist = dists.Categorical(num_cats = 4))

    shared = leaf(0)
    left = juice.summate(*[juice.multiply(shared, leaf(1)) for _ in range(n)], num_node_blocks = 1, block_size = 1)
    sums = [juice.summate(juice.multiply(leaf(v)), num_node_blocks = 1, block_size = 1) for v in range(2, n + 2)]
    right = juice.summate(juice.multiply(*sums), num_node_blocks = 1, block_size = 1)
    root = juice.summate(juice.multiply(left, right), num_node_blocks = 1, block_size = 1)
    root.init_parameters()
    return juice.compile(root, verbose = False).to(DEV), n + 2, 4


TL, C1, C2 = ("_fw_triton_block_sparse_tlmm_kernel", "_fw_triton_block_sparse_csmm1_kernel",
              "_fw_triton_block_sparse_csmm2_kernel")
SP, ELE, ELE2, SPELE = ("_fw_triton_sparse_kernel", "_bk_triton_block_sparse_ele_kernel",
                        "_bk_triton_block_sparse_ele_csmm2_kernel", "_bk_triton_sparse_ele_kernel")
PROD = "_forward_backward_kernel_2d"

SB_FW, SB_ELE, TLMM, ELE_CUDA = ("cuda.smallbatch_forward_sum", "cuda.smallbatch_ele_backward_sum",
                                  "cuda.tlmm_forward_sum", "cuda.ele_backward_sum")

# (circuit, batch, the paths taken with the CUDA gates forced to Triton, and forced to CUDA)
CASES = {
    "hclt32-b2": (lambda: hclt(32), 2, {C2, ELE2, SP, SPELE, PROD}, {SB_FW, SB_ELE, SP, SPELE, PROD}),
    "hclt32-b8": (lambda: hclt(32), 8, {C1, ELE, SP, SPELE, PROD}, {SB_FW, SB_ELE, SP, SPELE, PROD}),
    "hclt128-b16": (lambda: hclt(128), 16, {TL, ELE, SP, SPELE, PROD}, {SB_FW, ELE, SP, SPELE, PROD}),
    "hclt128-b64": (lambda: hclt(128), 64, {TL, ELE, SP, SPELE, PROD}, {TLMM, ELE_CUDA, SP, SPELE, PROD}),
    "pd-b2": (pd, 2, {SP, SPELE, PROD}, {SP, SPELE, PROD}),
    "pd-b8": (pd, 8, {C1, ELE, SP, SPELE, PROD}, {C1, ELE, SP, SPELE, PROD}),
    "pd-b16": (pd, 16, {TL, ELE, SP, SPELE, PROD}, {TL, ELE, SP, SPELE, PROD}),
}

AVAILABLE = {TLMM: "is_available", SB_FW: "smallbatch_fw_is_available", ELE_CUDA: "ele_is_available",
             SB_ELE: "smallbatch_ele_is_available"}


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_runs_get_the_whole_partitions_bits(case, cuda, hits, monkeypatch):
    build, B, triton_paths, cuda_paths = CASES[case]
    monkeypatch.setattr(SL, "autotune", _Autotune(SL.autotune, cuda))
    pc, V, C = build()
    torch.manual_seed(1)
    check_circuit(pc, torch.randint(0, C, (B, V), device = DEV))
    if cuda:                                      # (a CUDA kernel this machine lacks: the Triton one, unasserted)
        expected = {p for p in cuda_paths if p not in AVAILABLE or getattr(SL.cuda_kernels, AVAILABLE[p])()}
    else:
        expected = triton_paths
    assert expected <= hits, sorted(expected - hits)
    if not cuda:
        assert not any(h.startswith("cuda.") for h in hits), sorted(h for h in hits if h.startswith("cuda."))


@pytest.mark.parametrize("case", ["hclt32-b8", "pd-b2", "hclt128-b64"])
def test_runs_of_linear_flows(case, hits):
    build, B, _, _ = CASES[case]
    pc, V, C = build()
    torch.manual_seed(1)
    check_circuit(pc, torch.randint(0, C, (B, V), device = DEV), logspace = False)
    assert {PROD, SPELE} <= hits


@pytest.mark.slow                                       # (MEASURED: 23 s -- the large kernels need > 2048 rows)
def test_runs_of_the_large_kernels(hits):
    pc, V, C = wide()
    torch.manual_seed(1)
    check_circuit(pc, torch.randint(0, C, (2, V), device = DEV))
    assert {"_fw_triton_large_sparse_kernel", "_forward_backward_kernel_large", SPELE} <= hits


def test_runs_of_a_dense_partition_of_one_child_group(hits, monkeypatch):
    """Three node blocks over one product group: the dense path's one-group plan, each block its own product."""
    torch.manual_seed(0)
    ins = [juice.inputs(v, num_node_blocks = 1, block_size = 128, dist = dists.Categorical(num_cats = 4))
           for v in range(2)]
    prod = juice.multiply(*ins)
    sums = [juice.summate(prod, num_node_blocks = 1, block_size = 64) for _ in range(3)]
    root = juice.summate(*[juice.multiply(s) for s in sums], num_node_blocks = 1, block_size = 1)
    root.init_parameters()
    pc = juice.compile(root, verbose = False).to(DEV)
    monkeypatch.setattr(SL, "_DENSE_MIN_BLOCK_SIZE", 64)          # bring the small layer into the gate
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", 128 * 16)
    layer = next(l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg if l.block_size == 64)
    assert layer.partitioned_nids[0].numel() == 3 and layer.forward_runs_ok(32) == [True]
    check_circuit(pc, torch.randint(0, 4, (32, 2), device = DEV))
    assert "dense" in hits


def test_a_first_launch_of_some_rows_leaves_the_others():
    """A launch of some rows that comes before any whole launch of its shape -- nothing tuned for it yet -- writes
    those rows only: the sum forward takes its default tiles untuned, the bits of an untuned whole launch (a tuned
    one may differ at bf16 rounding: its dots take bf16 operands stabilized per tile), and the other passes time the
    whole partition on scratch, a choice the whole launch after reuses (the same bits)."""
    pc, V, C = hclt(32)
    B = 43                                                        # a batch no other test takes: nothing cached
    g = torch.Generator(device = DEV).manual_seed(0)
    nm, em, nf = (torch.randn(n, B, device = DEV, generator = g) - 3
                  for n in (pc.num_nodes, pc.num_elements, pc.num_nodes))
    kw = dict(logspace_flows = True, allow_modify_flows = False, propagation_alg = "LL")
    for lg in pc.inner_layer_groups:
        for layer in lg:
            runs = [(0, 1)]
            if lg.is_prod():
                a, b = layer._layer_nid_range
                out = em.clone()
                out[a:b] = NAN
                layer.forward(nm, out, partition_runs = only(0, runs, layer.num_fw_partitions))
                want = covered(layer.partitioned_nids[0].tolist(), layer.block_size, runs, a, b - a)
                ref = em.clone()
                layer.forward(nm, ref)
                assert out[a:b][~want].isnan().all() and torch.equal(out[a:b][want], ref[a:b][want])

                starts = layer.partitioned_u_cids[0].tolist()
                lo, hi = min(starts), max(starts) + layer.block_size
                want = covered(starts, layer.block_size, runs, lo, hi - lo)
                out = nf.clone()
                out[lo:hi] = NAN
                out[lo:hi][want] = -float("inf")
                layer.backward(out, em, logspace_flows = True, partition_runs = only(0, runs, layer.num_bk_partitions))
                ref = nf.clone()
                ref[lo:hi][want] = -float("inf")
                layer.backward(ref, em, logspace_flows = True)
                assert out[lo:hi][~want].isnan().all() and torch.equal(out[lo:hi][want], ref[lo:hi][want])
                continue
            a, b = layer.nid_range
            out = nm.clone()
            out[a:b] = NAN
            layer.forward(out, em, pc.params, partition_runs = only(0, runs, layer.num_fw_partitions))
            want = covered(layer.partitioned_nids[0].tolist(), layer.block_size, runs, a, b - a)
            ref = nm.clone()
            tuning = SL.autotune.set_autotune(False)                  # the whole launch untuned, as the first one
            try:
                layer.forward(ref, em, pc.params)
            finally:
                SL.autotune.set_autotune(tuning)
            assert out[a:b][~want].isnan().all() and torch.equal(out[a:b][want], ref[a:b][want])

            starts, width = layer.partitioned_chids[0].tolist(), layer.cs_block_sizes[0]
            lo, hi = min(starts), max(starts) + width
            out = torch.full_like(em, NAN)
            layer.backward(nf, out, nm, em, pc.params, partition_runs = only(0, runs, layer.num_bk_partitions), **kw)
            want = covered(starts, width, runs, lo, hi - lo)
            ref = torch.full_like(em, NAN)
            layer.backward(nf, ref, nm, em, pc.params, **kw)
            assert out[lo:hi][~want].isnan().all() and torch.equal(out[lo:hi][want], ref[lo:hi][want])


def test_a_dense_partition_of_several_child_groups_takes_no_runs(monkeypatch):
    """Node blocks over their own children (as in HCLT) are one batched product on the dense path: a subset would
    change its shape, so `forward_runs_ok` refuses them and the forward raises."""
    torch.manual_seed(0)
    ins = [juice.inputs(v, num_node_blocks = 1, block_size = 128, dist = dists.Categorical(num_cats = 4))
           for v in range(6)]
    sums = [juice.summate(juice.multiply(ins[2 * g], ins[2 * g + 1]), num_node_blocks = 1, block_size = 64)
            for g in range(3)]
    root = juice.summate(juice.multiply(*sums), num_node_blocks = 1, block_size = 1)
    root.init_parameters()
    pc = juice.compile(root, verbose = False).to(DEV)
    monkeypatch.setattr(SL, "_DENSE_MIN_BLOCK_SIZE", 64)
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", 128 * 16)
    layer = next(l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg if l.block_size == 64)
    assert layer.forward_runs_ok(32) == [False]
    assert layer.forward_runs_ok(8) == [True]                     # below the dense path's batch: the kernels
    x = torch.randint(0, 4, (32, 6), device = DEV)
    pc(x)
    with pytest.raises(ValueError, match = "computed whole"):
        layer.forward(pc.node_mars, pc.element_mars, pc.params, partition_runs = [[(0, 1)]])


def test_what_takes_no_runs():
    with juice.set_block_size(4):
        ni0 = juice.inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = juice.summate(juice.multiply(ni0), num_node_blocks = 2, external_params = LowRankSumParams(rank = 3))
        root = juice.summate(juice.multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters()
    pc = juice.compile(root, verbose = False).to(DEV)
    x = torch.randint(0, 5, (16, 1), device = DEV)
    pc(x)
    pc.backward(x, compute_param_flows = False)
    nm, em, nf, ef = pc.node_mars, pc.element_mars, pc.node_flows, pc.element_flows
    gated = next(l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg if isinstance(l, ExternalParamsSumLayer))
    with pytest.raises(ValueError, match = "whole partitions"):
        gated.forward(nm, em, pc.params, partition_runs = [[(0, 1)]] * gated.num_fw_partitions)
    with pytest.raises(ValueError, match = "whole partitions"):
        gated.backward(nf, ef, nm, em, pc.params, partition_runs = [[(0, 1)]] * gated.num_bk_partitions)
    plain = next(l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg if not isinstance(l, ExternalParamsSumLayer))
    runs = [[(0, 1)]] * plain.num_bk_partitions
    with pytest.raises(ValueError, match = "element flows"):
        plain.backward(nf, ef, nm, em, pc.params, param_flows = torch.zeros(1, device = DEV), partition_runs = runs)
    with pytest.raises(ValueError, match = "element flows"):
        plain.backward(nf, ef, nm, em, pc.params, allow_modify_flows = True, partition_runs = runs)
