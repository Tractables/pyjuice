"""
The sum layer's forward precision modes (`pyjuice.layer.sum_layer.PRECISIONS`), against a float64 forward:

* "auto" keeps today's kernels except dense layers in the cuBLAS gate (`_DENSE_MIN_BLOCK_SIZE`, `_DENSE_MIN_WORK`),
  which take TF32 products;
* "tf32" never uses bf16 products; "fp32" gives fp32-level products (`tf32x3` in the Triton kernel, exact fp32
  cuBLAS on dense layers);
* the dense cuBLAS path runs exactly when its gate holds (LL, no tempering, dense node blocks, sizes);
* on that path, node blocks with their own, evenly strided children (HCLT) are one batched product, run in pieces
  when `_DENSE_SCRATCH` is small, and any other multi-group layout is left to the kernels;
* the backward's element and parameter flows match float64 in each mode, and "fp32" takes `tf32x3` dots in both flow
  kernels (the other modes launch them as before);
* an unknown mode is refused.
"""
import collections.abc

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import sum_layer as SL

DEV = torch.device("cuda:0")


def hmm_sum_layer(latents, seed = 0):
    torch.manual_seed(seed)
    pc = juice.compile(juice.structures.HMM(seq_length = 3, num_latents = latents, num_emits = 16),
                       verbose = False).to(DEV)
    return pc, [l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg.layers][0]


def grouped_sum_layer(groups = 3, block_size = 64, children = 128, seed = 0):
    """
    A sum layer of `groups` node blocks, each over its own product block of `children` nodes: one child group per
    block, as in HCLT. `children` differs from the block size, so that the two cannot be confused, and spans
    several row tiles of the dense kernels.
    """
    torch.manual_seed(seed)
    ins = [juice.inputs(v, num_node_blocks = 1, block_size = children, dist = dists.Categorical(num_cats = 4))
           for v in range(2 * groups)]
    sums = [juice.summate(juice.multiply(ins[2 * g], ins[2 * g + 1]), num_node_blocks = 1, block_size = block_size)
            for g in range(groups)]
    pc = juice.compile(juice.summate(juice.multiply(*sums), num_node_blocks = 1, block_size = 1), verbose = False).to(DEV)
    return pc, [l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg.layers][0]


def run(pc, layer, batch, seed = 0, prepare = None, **kw):
    """
    The layer's forward on random children (`prepare(x)` may edit them first); returns (its node rows, the float64
    forward of the same).
    """
    nids, cids, pids = layer.partitioned_nids[0], layer.partitioned_cids[0], layer.partitioned_pids[0]
    NB = cids.size(0)
    BS = layer.block_size
    first = int(cids[cids > 0].min())
    R = int(cids.max()) - first + 1                                 # child rows (a node block may read a subset)
    g = torch.Generator(device = DEV).manual_seed(seed)
    x = torch.randn(R, batch, device = DEV, generator = g) * 3 - 5
    if prepare is not None:
        prepare(x)
    node_mars = torch.zeros(pc.num_nodes, batch, device = DEV)
    element_mars = torch.full((pc.num_elements, batch), -float("inf"), device = DEV)
    element_mars[first:first + R] = x
    layer.forward(node_mars, element_mars, pc.params, **kw)
    W = torch.zeros(NB * BS, R, dtype = torch.float64, device = DEV)
    i = torch.arange(BS, device = DEV)
    for k in range(NB):
        W[k * BS + i[:, None], (cids[k] - first)[None, :]] += pc.params[pids[k][None, :] + i[:, None]].double()
    m = x.double().max(0, keepdim = True).values
    exact = torch.log(W @ torch.exp(x.double() - m)) + m
    n0 = int(nids.min())
    return node_mars[n0:n0 + NB * BS].double(), exact


def run_backward(pc, layer, batch, seed = 0, **kw):
    """
    The layer's backward (linear flows) on random children and parent flows; returns (element flows, parameter
    flows) and the float64 values of the same, from the same `node_mars`.
    """
    nids, cids, pids, pfids = (layer.partitioned_nids[0], layer.partitioned_cids[0], layer.partitioned_pids[0],
                               layer.partitioned_pfids[0])
    NB, E, BS = cids.size(0), cids.size(1), layer.block_size
    first = int(cids[cids > 0].min())
    R = int(cids.max()) - first + 1
    n0 = int(nids.min())
    g = torch.Generator(device = DEV).manual_seed(seed)
    x = torch.randn(R, batch, device = DEV, generator = g) * 3 - 5
    node_mars = torch.zeros(pc.num_nodes, batch, device = DEV)
    element_mars = torch.full((pc.num_elements, batch), -float("inf"), device = DEV)
    element_mars[first:first + R] = x
    layer.forward(node_mars, element_mars, pc.params)
    node_flows = torch.zeros(pc.num_nodes, batch, device = DEV)
    node_flows[n0:n0 + NB * BS] = torch.rand(NB * BS, batch, device = DEV, generator = g)
    element_flows = torch.zeros(pc.num_elements, batch, device = DEV)
    param_flows = torch.zeros(int(pfids.max()) + BS, device = DEV)
    layer.backward(node_flows, element_flows, node_mars, element_mars, pc.params, param_flows, **kw)

    i = torch.arange(BS, device = DEV)[None, :, None]
    n = (torch.arange(NB, device = DEV)[:, None, None] * BS + i).expand(NB, BS, E)       # parent row
    c = (cids - first)[:, None, :].expand(NB, BS, E).clamp(min = 0)                      # child row
    w = pc.params[pids[:, None, :] + i].double() * (cids > 0)[:, None, :]                # padded edges: 0
    F, NM = node_flows[n0:n0 + NB * BS].double(), node_mars[n0:n0 + NB * BS].double()
    T = F[n] * w[..., None] * torch.exp(x.double()[c] - NM[n])                           # [NB, BS, E, batch]
    eref = torch.zeros(R, batch, dtype = torch.float64, device = DEV).index_add_(0, c.reshape(-1), T.reshape(-1, batch))
    pref = torch.zeros(param_flows.numel(), dtype = torch.float64, device = DEV)
    pref.index_add_(0, (pfids[:, None, :] + i).reshape(-1), T.sum(-1).reshape(-1))
    return (element_flows[first:first + R].double(), param_flows.double()), (eref, pref)


class _CountedKwargs(collections.abc.Mapping):
    """Launch keyword arguments that count how many launches unpack them."""
    def __init__(self, **kw):
        self.kw, self.unpacked = kw, 0
    def __getitem__(self, key):
        return self.kw[key]
    def __iter__(self):
        return iter(self.kw)
    def __len__(self):
        return len(self.kw)
    def keys(self):
        self.unpacked += 1
        return self.kw.keys()


#: max |log value - float64| per mode, with margin over the measured (5e-4 TF32, 3e-6 fp32 on a 4096-latent layer)
TOL = {"tf32": 2e-3, "fp32": 2e-5}


@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
def test_dense_path_accuracy(precision, monkeypatch):
    pc, layer = hmm_sum_layer(64)
    monkeypatch.setattr(SL, "_DENSE_MIN_BLOCK_SIZE", 64)          # bring a small layer into the gate
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", 64 * 32)
    calls = []
    monkeypatch.setattr(layer, "_forward_dense", lambda *a, **k: (calls.append(k["exact"]), SL.SumLayer._forward_dense(layer, *a, **k)))
    got, exact = run(pc, layer, 48, precision = precision)
    assert calls == [precision == "fp32"]
    assert (got - exact).abs().max() <= TOL["fp32" if precision == "fp32" else "tf32"]


@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
@pytest.mark.parametrize("per_run", [3, 2])                         # all 3 groups at once; runs of 2 and 1
def test_batched_dense_path_accuracy(precision, per_run, monkeypatch):
    pc, layer = grouped_sum_layer()
    nids, cids, pids = layer.partitioned_nids[0], layer.partitioned_cids[0], layer.partitioned_pids[0]
    blocks, E, c0, G = layer._dense_blocks(0, nids, cids, pids)
    assert (G, len(blocks), E, layer.block_size) == (3, 3, 128, 64) # one batched product over [3, ...] views
    monkeypatch.setattr(SL, "_DENSE_MIN_BLOCK_SIZE", 64)
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", E * 32)
    monkeypatch.setattr(SL, "_DENSE_SCRATCH", per_run * E * 48)
    calls = []
    monkeypatch.setattr(layer, "_forward_dense", lambda *a, **k: (calls.append(k["exact"]), SL.SumLayer._forward_dense(layer, *a, **k)))

    def per_group_maxima(x):
        x[E - 32:E] += 100                                          # group 0's max in its last rows: missing it overflows
        x[E:2 * E, 0] = -float("inf")                               # group 1 has no mass in column 0 (max -inf)
        x[2 * E:] -= 200                                            # group 2 far below: another group's max underflows

    got, exact = run(pc, layer, 48, precision = precision, prepare = per_group_maxima)
    assert calls == [precision == "fp32"]
    finite = exact.isfinite()
    assert (~finite).sum() == layer.block_size and torch.equal(got == -float("inf"), ~finite)
    assert (got[finite] - exact[finite]).abs().max() <= TOL["fp32" if precision == "fp32" else "tf32"]


def test_dense_plans():
    """`_dense_blocks` on made-up tables: which layouts the dense path takes, and how."""
    pc, layer = hmm_sum_layer(64)
    BS = E = 64
    e = torch.arange(E, device = DEV)

    def plan(blocks, partition_id):                                 # blocks: [(nid, first child, first param)]
        nids = torch.tensor([nid for nid, _, _ in blocks], device = DEV)
        cids = torch.stack([c0 + e for _, c0, _ in blocks])
        pids = torch.stack([p0 + e * BS for _, _, p0 in blocks])
        return layer._dense_blocks(partition_id, nids, cids, pids)

    # one group: both blocks read children 1..64, exponentiated once
    assert plan([(100, 1, 0), (164, 1, E * BS)], 1000) == ([(100, 0), (164, E * BS)], E, 1, 1)
    # a group per block, evenly strided (listed out of order): one batched product
    assert plan([(164, 65, E * BS), (100, 1, 0)], 1001) == ([(100, 0), (164, E * BS)], E, 1, 2)
    # a gap between the children, the node rows or the parameters: the kernels
    assert plan([(100, 1, 0), (164, 129, E * BS)], 1002) is None
    assert plan([(100, 1, 0), (228, 65, E * BS)], 1003) is None
    assert plan([(100, 1, 0), (164, 65, 2 * E * BS)], 1004) is None
    # two groups of two blocks each: the kernels
    assert plan([(100, 1, 0), (164, 1, E * BS), (228, 65, 2 * E * BS), (292, 65, 3 * E * BS)], 1005) is None


@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
def test_triton_path_accuracy(precision):
    pc, layer = hmm_sum_layer(64)                                   # below the gate: the Triton kernels
    got, exact = run(pc, layer, 48, precision = precision)
    err = (got - exact).abs().max()
    if precision == "auto":
        assert err <= 2e-2                                          # bf16 products
    else:
        assert err <= TOL[precision]


def test_the_gate(monkeypatch):
    pc, layer = hmm_sum_layer(64)
    monkeypatch.setattr(SL, "_DENSE_MIN_BLOCK_SIZE", 64)
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", 64 * 32)            # 64 children x batch 32
    taken = []
    monkeypatch.setattr(layer, "_forward_dense", lambda *a, **k: (taken.append(True), SL.SumLayer._forward_dense(layer, *a, **k)))
    run(pc, layer, 48)
    assert taken == [True]                                          # in the gate
    run(pc, layer, 16)                                              # 64 x 16: below `_DENSE_MIN_WORK`
    run(pc, layer, 48, propagation_alg = "MPE")                     # not LL
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", 64 * 64)            # 64 x 48: too little work
    run(pc, layer, 48)
    assert taken == [True]


@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
def test_backward_accuracy(precision, monkeypatch):
    pc, layer = hmm_sum_layer(64)                                   # batch 48: the block-sparse dot kernels
    tf32x3 = _CountedKwargs(DOT_TF32X3 = True)
    monkeypatch.setattr(SL, "_TF32X3_DOT", tf32x3)
    (eflows, pflows), (eref, pref) = run_backward(pc, layer, 48, precision = precision)
    if precision == "fp32":
        assert tf32x3.unpacked >= 2                                 # the element- and the parameter-flow kernel
    else:
        assert tf32x3.unpacked == 0
    tol = TOL["fp32" if precision == "fp32" else "tf32"]
    for got, ref in ((eflows, eref), (pflows, pref)):
        assert (got - ref).abs().max() <= tol * ref.abs().max()


def test_removed_layer_flags():
    pc, layer = hmm_sum_layer(16)
    for name in ("force_use_bf16", "force_use_fp32"):
        with pytest.raises(TypeError, match = f"{name}.*precision"):   # set: refused
            run(pc, layer, 16, **{name: True})
        run(pc, layer, 16, **{name: False})                          # unset (their old default): ignored


def test_block_sparse_layers_are_not_dense(monkeypatch):
    from functools import partial
    from pyjuice.nodes.methods.edge_constructors import block_sparse_rnd_blk_edge_constructor
    torch.manual_seed(0)
    pc = juice.compile(juice.structures.HMM(seq_length = 3, num_latents = 64, num_emits = 16, block_size = 16,
                                            sum_edge_ids_constructor = partial(block_sparse_rnd_blk_edge_constructor,
                                                                               num_chs_per_block = 2)),
                       verbose = False).to(DEV)
    layer = [l for lg in pc.inner_layer_groups if lg.is_sum() for l in lg.layers][0]
    monkeypatch.setattr(SL, "_DENSE_MIN_BLOCK_SIZE", 16)
    monkeypatch.setattr(SL, "_DENSE_MIN_WORK", 16)
    nids, cids, pids = layer.partitioned_nids[0], layer.partitioned_cids[0], layer.partitioned_pids[0]
    e = torch.arange(cids.size(1), device = DEV)
    contiguous = bool(((cids == cids[:, :1] + e).all() & (pids == pids[:, :1] + e * layer.block_size).all()).item())
    assert not contiguous                                           # two random child blocks per node block
    assert layer._dense_blocks(0, nids, cids, pids) is None
    got, exact = run(pc, layer, 32, precision = "fp32")            # the Triton kernel, fp32-level
    assert (got - exact).abs().max() <= TOL["fp32"]


def test_unknown_precision_is_refused():
    pc, layer = hmm_sum_layer(16)
    with pytest.raises(ValueError, match = "precision"):
        run(pc, layer, 16, precision = "bf16")
