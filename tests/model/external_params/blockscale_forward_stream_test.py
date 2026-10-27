"""
The split forward (`blockscale_forward_stream.cu`) against float64 recomputed from what the layer was given: every
fitting (tile, split) on one node block and several, node blocks of 64 (the 64-row tiles) and 128, gate widths 4 / 16
/ 64 (two chunks of gates: `log Z`'s running max), rows of 1536 children (a split past the pre-pass's 32 chunks), a
ragged layer of two partitions (padded edge blocks) and one whose widest row is not a power of two (past the gate
table's width), batches 64 / 128 / 192; with `log phi` -inf on some
gates and columns (whole columns of a node block: `Z` 0, the node -inf), -inf exactly where float64 has it; nothing
outside the layer's rows written; the same bits run to run. Then the plan: offered at bf16 (its pick through
`forward_layer` matches float64), never at fp32.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams
import pyjuice.layer.kernels.c as ck
import pyjuice.nodes.external_params.kernels.c as ext_c
from pyjuice.nodes.external_params.blockscale.forward import _ForwardMixin
from pyjuice.nodes.external_params.blockscale.common import _buffer_kwarg


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

RAGGED = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]])
# the widest row three child blocks: the compiled edge count is padded past the gate table's width
NP2 = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [0, 1, 2, 0, 1, 1, 2, 3]])

# name: (node blocks, node block size, child blocks, child block size, gate width, edge ids or None)
SHAPES = {
    "one_block":    (1, 128, 2, 64, 16, None),
    "four_blocks":  (4, 64, 4, 64, 64, None),
    "gate_4":       (2, 128, 4, 64, 4, None),
    "two_chunks":   (1, 128, 8, 64, 4, None),        # 128 gates: log Z in two chunks
    "long_rows":    (1, 128, 24, 64, 64, None),      # 1536 children: one split past the pre-pass's 32 chunks
    "ragged_split": (4, 128, 4, 64, 16, RAGGED),
    "ragged_np2":   (4, 128, 4, 64, 16, NP2),
}


def _compile(name, dtype):
    nb, bs, nc, cbs, gate_cbs, edge_ids = SHAPES[name]
    torch.manual_seed(0)
    with juice.set_block_size(cbs):
        ni = [inputs(v, num_node_blocks = nc, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        prod = multiply(*ni)
    kw = dict(edge_ids = edge_ids) if edge_ids is not None else {}
    ns = summate(prod, num_node_blocks = nb, block_size = bs,
                 external_params = BlockScaleSumParams(ch_block_size = gate_cbs), **kw)
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = dtype).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    return pc, ns, layer


def _force_stream(monkeypatch):
    """The plan's measured choice made the split forward wherever it is a candidate."""
    real = ck.autotune

    def pick(cands, *a, **kw):
        for key, _ in cands:
            if key[0] == "stream":
                return key
        return real(cands, *a, **kw)
    monkeypatch.setattr(ck, "autotune", pick)


def _run(name, batch, dtype, monkeypatch):
    """Compile, run one forward through the plan (the split forward forced), and capture the layer's inputs."""
    pc, ns, layer = _compile(name, dtype)
    _force_stream(monkeypatch)
    captured = []
    real = _ForwardMixin.forward_layer

    def wrapped(self, layer_, ns_tensors, node_mars, element_mars, params, **kw):
        ext = kw.get(_buffer_kwarg(), None)
        if ext is not None and layer_ is layer:
            captured.append(dict(bs = self, em = element_mars.clone(), ext = ext.clone(), params = params,
                                 before = node_mars.clone()))
        out = real(self, layer_, ns_tensors, node_mars, element_mars, params, **kw)
        if ext is not None and layer_ is layer:
            captured[-1]["after"] = node_mars.clone()
        return out
    monkeypatch.setattr(_ForwardMixin, "forward_layer", wrapped)
    dev = torch.device("cuda:0")
    g = torch.Generator(device = "cpu").manual_seed(batch)
    data = torch.randint(0, NUM_CATS, [batch, 2], generator = g).to(dev)
    phi = (torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], generator = g) * 0.7).to(dev)
    pc(data, sum_external_params = {ns: phi})
    torch.cuda.synchronize()
    assert len(captured) == 1, "the gated layer ran other than once"
    return pc, ns, layer, captured[0]


def _reference(calls, em, ext, params, batch):
    """float64 `(node rows, node_mars, log Z)` per partition, from the launch tuples and the inputs (the fork's -inf
    rules: -inf where Z is 0 or N is -inf)."""
    dev = em.device
    out = []
    ninf = -float("inf")
    for (nids, cids, pids, gate, _, block_size, num_edges, node_cbs, gate_cbs, _, ext_base, _) in calls:
        rows = nids.numel()
        pos = torch.arange(num_edges, device = dev)
        cid = cids[:, pos // 64] + pos % 64                                                # [rows, E]
        th = params[(pids[:, pos // 64] + (pos % 64) * block_size)[:, :, None]
                    + torch.arange(block_size, device = dev)].double()                     # [rows, E, bs]
        j, d = pos // node_cbs, (pos % node_cbs) // gate_cbs
        gb = torch.full((rows, num_edges), -1, dtype = torch.long, device = dev)
        inside = j < gate.size(1)
        gb[:, inside] = gate[:, j[inside]]
        lphi = torch.where((gb >= 0)[:, :, None], ext.view(-1, batch)[gb.clamp(min = 0) + ext_base + d],
                           ninf).double()                                                 # [rows, E, B]
        s = em.view(-1, batch)[cid].double() + lphi
        M = s.amax(1, keepdim = True)
        Ms = torch.where(torch.isinf(M), 0.0, M)
        logN = torch.log(torch.einsum("rem,reb->rmb", th, torch.exp(s - Ms))) + Ms
        mz = lphi.amax(1, keepdim = True)
        mzs = torch.where(torch.isinf(mz), 0.0, mz)
        logZ = torch.log(torch.einsum("rem,reb->rmb", th, torch.exp(lphi - mzs))) + mzs
        nm = torch.where(torch.isinf(logN) | torch.isinf(logZ), ninf, logN - logZ)
        nrow = (nids[:, None] + torch.arange(block_size, device = dev)).reshape(-1)
        out.append((nrow, nm.reshape(rows * block_size, batch), logZ.reshape(rows * block_size, batch)))
    return out


def _codes(mod, calls, batch):
    bs_, node_cbs, gate_cbs = calls[0][5], calls[0][7], calls[0][8]
    cfgs = [tuple(int(v) for v in c) for c in mod.configs()]
    codes = []
    for i in mod.fitting_configs(bs_, batch, node_cbs, gate_cbs):
        kt = cfgs[i][2]
        kmin = min(c[6] // kt for c in calls)
        codes += [int(i) * 16 + s for s in (1, 2, 3, 5, 8) if s <= kmin]
    return codes


@cuda_only
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("batch", [64, 128, 192])
def test_every_tile_and_split_gives_float64s_forward(name, batch, monkeypatch):
    mod = ext_c.get_stream_module()
    assert mod is not None, "blockscale_forward_stream.cu did not compile"
    _, bs_, _, cbs, gate_cbs, _ = SHAPES[name]
    if not mod.fitting_configs(bs_, batch, cbs, gate_cbs):
        pytest.skip("no tile divides this node block and batch")
    pc, ns, layer, cap = _run(name, batch, torch.bfloat16, monkeypatch)
    if name == "ragged_split":
        assert layer.num_fw_partitions == 2, "the ragged layer is no longer split in two"
    plan = layer._bs_fw_plans[batch][0]
    assert plan[1] == "blockscale_forward_stream", f"the split forward was not offered: {plan[1]}"
    calls, bs, params = plan[2], cap["bs"], cap["params"]
    codes = _codes(mod, calls, batch)
    if not codes:
        pytest.skip("no tile divides this node block and batch")

    # -inf log phi: one gate's row on a few columns, and every gate of the first partition's rows on column 5
    ext = cap["ext"].clone().view(-1, batch)
    nids, gate, ext_base, n_child = calls[0][0], calls[0][3], calls[0][10], calls[0][7] // calls[0][8]
    g0 = int(gate[0, 0]) + ext_base
    ext[g0, :3] = -float("inf")
    rows0 = [int(gate[r, jj]) + ext_base + dd for r in range(gate.size(0)) for jj in range(gate.size(1))
             if int(gate[r, jj]) >= 0 for dd in range(n_child)]
    ext[rows0, 5] = -float("inf")
    ext = ext.view(-1)
    em = cap["em"]

    ref = _reference(calls, em, ext, params, batch)
    assert any(bool(torch.isinf(r[1]).any()) for r in ref), "no -inf node values to check"
    st = layer._bs_bw_state
    sig = [bs._sigma(layer, params, pid, st["block_size"], st["gate_cbs"])[0] for pid in range(len(calls))]

    for code in codes:
        outs = []
        for rep in range(2):
            nm = cap["before"].clone()
            lzs = []
            for pid, args in enumerate(calls):
                lz = torch.full_like(args[4], float("nan"))
                tuned = tuple(args[:4]) + (lz,) + tuple(args[5:-1]) + (code,)
                mod.blockscale_forward_stream(nm, em, params, ext, *tuned, sig[pid])
                lzs.append(lz)
            torch.cuda.synchronize()
            outs.append((nm, lzs))
        (nm, lzs), (nm2, lzs2) = outs
        what = f"config {code // 16} split {code % 16}"
        assert torch.equal(nm, nm2) and all(torch.equal(a, b) for a, b in zip(lzs, lzs2)), \
            f"{what}: two equal launches gave different bits"
        mine = torch.zeros(nm.size(0), dtype = torch.bool, device = nm.device)
        for (nrow, want, want_lz), lz in zip(ref, lzs):
            mine[nrow] = True
            for got, ref_, tol, label in ((nm[nrow].double(), want, 2e-3, "node values"),
                                          (lz.view(-1, batch).double(), want_lz, 1e-4, "log Z")):
                assert not bool(torch.isnan(got).any() | torch.isposinf(got).any()), f"{what}: {label} NaN or +inf"
                assert torch.equal(torch.isneginf(got), torch.isneginf(ref_)), \
                    f"{what}: {label} -inf other than where float64 has it"
                fin = torch.isfinite(ref_)
                d = float((got[fin] - ref_[fin]).abs().max())
                assert d < tol, f"{what}: {label} off float64 by {d}"
        assert torch.equal(nm[~mine], cap["before"][~mine]), f"{what}: rows outside the layer were written"


@cuda_only
@pytest.mark.parametrize("name", ["four_blocks", "ragged_split"])
def test_the_plans_pick_through_forward_layer(name, monkeypatch):
    """The forward through the plan -- each partition's `sigma` from `_sigma` -- is float64's."""
    batch = 128
    pc, ns, layer, cap = _run(name, batch, torch.bfloat16, monkeypatch)
    plan = layer._bs_fw_plans[batch][0]
    assert plan[1] == "blockscale_forward_stream"
    for nrow, want, _ in _reference(plan[2], cap["em"], cap["ext"], cap["params"], batch):
        got = cap["after"][nrow].double()
        assert not bool(torch.isnan(got).any()) and torch.equal(torch.isneginf(got), torch.isneginf(want))
        fin = torch.isfinite(want)
        assert float((got[fin] - want[fin]).abs().max()) < 2e-3


@cuda_only
def test_not_offered_at_fp32(monkeypatch):
    batch = 128
    pc, ns, layer, cap = _run("one_block", batch, torch.float32, monkeypatch)
    assert layer._bs_fw_plans[batch][0][1] != "blockscale_forward_stream", \
        "fp32 parameters were sent to the bf16-only split forward"
