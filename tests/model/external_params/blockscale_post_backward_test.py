"""
`_bs_post_backward_kernel` -- the log-Z half of the gate gradient, the denominator flow and the `-log Z` shift in one
launch per partition -- against float64 recomputed from what the launch was given (its inputs captured at
`_post_fused`, the gate rows found through the gate table): the gradient's log-Z term and `W` to fp32-level rounding,
`node_mars` to the bit (one fp32 subtraction), nothing else written; the gradient the same from run to run (the node
tiles' shares summed in order) and the ticket counters back to 0.

The shapes are the kernel's regimes: one node block and several, a ragged layer of two partitions (padded gate cells)
and one whose widest row is not a power of two (the gate table's bound), a node block of 8 (no `tl.dot`: the
broadcast-sum branch), two and 32 gate tiles (the shift by the first alone), batches that are not a multiple of the
batch tile (atomic batch slices), and each of the gradient alone, the denominator alone and both.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams
import pyjuice.nodes.external_params.blockscale.backward as bs_backward


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

SPLIT = torch.tensor([[0] * 8 + [1] * 3 + [2] * 3 + [3] * 3, list(range(8)) + [0, 1, 2] * 3])
# the widest row three child blocks, not a power of two: the compiled edge count is padded past the gate table's width
NP2 = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [0, 1, 2, 0, 1, 1, 2, 3]])

# name: (node blocks, node block size, child blocks, child block size, gate width, edge ids or None)
SHAPES = {
    "one_block":    (1, 128, 1, 128, 8, None),
    "four_blocks":  (4, 64, 4, 64, 16, None),
    "ragged_split": (4, 64, 8, 32, 8, SPLIT),
    "block_8":      (4, 8, 4, 8, 4, None),
    "many_gates":   (1, 64, 4, 64, 4, None),            # 64 gates: two gate tiles
    # 1024 gates, 32 gate tiles: the shift by the first alone. With two, both tiles' programs ran at once and read
    # `node_mars` before either wrote it, so a shift by every tile went unseen
    "gate_tiles":   (1, 64, 64, 64, 4, None),
    "ragged_np2":   (4, 64, 4, 32, 8, NP2),
}


def _compile(name, apply_z_correction):
    nb, bs, nc, cbs, gate_cbs, edge_ids = SHAPES[name]
    torch.manual_seed(0)
    with juice.set_block_size(cbs):
        ni = [inputs(v, num_node_blocks = nc, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        prod = multiply(*ni)
    kw = dict(edge_ids = edge_ids) if edge_ids is not None else {}
    ns = summate(prod, num_node_blocks = nb, block_size = bs,
                 external_params = BlockScaleSumParams(ch_block_size = gate_cbs,
                                                       apply_z_correction = apply_z_correction), **kw)
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    return pc, ns, layer


def _capture(monkeypatch):
    """Every `_post_fused` call: what it was given, and the buffers it writes before and after."""
    calls = []
    real = bs_backward._BackwardMixin._post_fused

    def wrapped(self, layer, state, node_flows, node_mars, params, external_params, grad_ext, denom_param_flows):
        before = (node_mars.clone(), None if grad_ext is None else grad_ext.clone(),
                  None if denom_param_flows is None else denom_param_flows.clone())
        real(self, layer, state, node_flows, node_mars, params, external_params, grad_ext, denom_param_flows)
        after = (node_mars.clone(), None if grad_ext is None else grad_ext.clone(),
                 None if denom_param_flows is None else denom_param_flows.clone())
        calls.append(dict(bs = self, layer = layer, state = state, nf = node_flows.clone(), params = params,
                          ext = external_params.clone(), before = before, after = after))
    monkeypatch.setattr(bs_backward._BackwardMixin, "_post_fused", wrapped)
    return calls


def _reference(call):
    """float64 `(node_mars, gradient, denominator flows)` after the call, from its inputs and the buffers before it."""
    bs, layer, state, nf = call["bs"], call["layer"], call["state"], call["nf"]
    nm, grad, dpf = (None if t is None else t.clone() for t in call["before"])
    block_size, gate_cbs, batch = state["block_size"], state["gate_cbs"], state["batch_size"]
    n_child_gates, ext_base = state["node_cbs"] // gate_cbs, state["ext_base"]
    ext = call["ext"].view(-1, batch).double()
    ar = torch.arange(block_size, device = nf.device)
    grad = None if grad is None else grad.double()
    rows_g = None if grad is None else grad.view(-1, batch)                     # (the buffer is flat; rows of a gate)
    W_all = None if dpf is None else dpf.double()
    for pid, (nids, log_z, rows) in enumerate(state["shift_args"]):
        lz_all = log_z.view(rows * block_size, batch)
        gb = layer.ext_slots[0][pid].long().cpu()
        if grad is not None:
            sigma, n_gates = bs._sigma(layer, call["params"], pid, block_size, gate_cbs)
        else:
            sigma, n_gates = None, bs.denom_gate_geometry(layer)[pid][2]
        W = None if W_all is None else bs._denom_slice(layer, W_all, pid)
        for r in range(rows):
            nrows = int(nids[r]) + ar
            zrows = r * block_size + ar
            nm[nrows] = nm[nrows] - lz_all[zrows]                               # the one fp32 subtraction
            v = torch.exp(nf[nrows].double() - lz_all[zrows].double())          # [M, B]
            # each gate's row in `ext`: the table's entry for its edge block (within the table's width) plus its place
            j = torch.arange(n_gates) // n_child_gates
            base = torch.full((n_gates,), -1, dtype = torch.long)
            inside = j < gb.size(1)
            base[inside] = gb[r, j[inside]]
            has = base >= 0
            grow = (base + ext_base + torch.arange(n_gates) % n_child_gates)[has].to(nf.device)
            phi = torch.exp(ext[grow])                                          # [G', B]
            gidx = torch.nonzero(has).flatten().to(nf.device)
            if grad is not None:
                term = sigma[zrows][:, gidx].double().t() @ v                   # [G', B]
                rows_g[grow] -= phi * term
            if W is not None:
                W[zrows[:, None], gidx[None, :]] += v @ phi.t()                 # [M, G']
    return nm, grad, W_all


@cuda_only
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("batch", [64, 100, 8])
@pytest.mark.parametrize("want_grad,apply_z_correction", [(True, True), (True, False), (False, True)])
def test_one_launch_gives_float64s_post_backward(name, batch, want_grad, apply_z_correction, monkeypatch):
    if name == "block_8" and batch < 16:
        pytest.skip("a gated layer of block size 8 has no parameter-flow backward below batch 16 (the sum layer "
                    "refuses it)")
    if name == "ragged_np2" and batch < 16:
        pytest.skip("below batch 16 the small-batch parameter-flow fork refuses a row of more edge blocks than the "
                    "gate table's width")
    pc, ns, layer = _compile(name, apply_z_correction)
    if name == "ragged_split":
        assert layer.num_fw_partitions == 2, "the ragged layer is no longer split in two"
    dev = torch.device("cuda:0")
    g = torch.Generator(device = "cpu").manual_seed(batch)
    data = torch.randint(0, NUM_CATS, [batch, 2], generator = g).to(dev)
    phi = (torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], generator = g) * 0.7).to(dev)

    calls = _capture(monkeypatch)
    for _ in range(2):
        pc(data, sum_external_params = {ns: phi})
        pc.backward(data, flows_memory = 0.0, compute_external_grads = want_grad)
    assert len(calls) == 2, "the fused launch did not run once a backward"
    call = calls[0]
    nm_ref, grad_ref, w_ref = _reference(call)
    nm, grad, dpf = call["after"]

    assert torch.equal(nm, nm_ref), "`node_mars` after the shift is not node_mars - log Z, or other rows moved"
    # What the launch added, against what float64 adds (the gradient already holds the element kernel's term, of nearly
    # the log-Z term's size, so their difference is no scale for this one's rounding)
    for got, ref, base, what in ((grad, grad_ref, call["before"][1], "the gradient's log-Z term"),
                                 (dpf, w_ref, call["before"][2], "the denominator flows")):
        if ref is None:
            assert got is None
            continue
        added, want = got.double() - base.double(), ref - base.double()
        scale = float(want.abs().max())
        assert scale > 0, f"{what}: nothing was written"
        d = float((added - want).abs().max()) / scale
        assert d < 1e-5, f"{what} off float64 by {d} (relative to its largest entry)"
    if want_grad:
        # the same from run to run, as the log-Z kernel's was: node tiles add their gradient in order, not atomically
        assert torch.equal(calls[1]["after"][1], grad), "two equal backwards gave different gradients"
    for _, counters, _ in getattr(call["bs"], "_bs_post_scratch", {}).values():
        assert int(counters.abs().sum()) == 0, "a ticket counter was left set"
