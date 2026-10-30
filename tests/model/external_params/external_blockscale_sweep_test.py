"""
Block-scale forward AND backward across many shapes, against one vectorized float64 reference.

The companion file (`external_blockscale_test.py`) pins behaviours -- invariants, boundaries, the EM
loop -- on a handful of shapes. This one does the opposite: one check, run over a wide grid of
`(num_latents, block_size, gate ch_block_size, batch)`, because most of what can go wrong in these
kernels is shape-dependent. Tile sizes, the `BMAX` register state, the number of parent blocks a child
sees, and which of the four fork kernels the launcher picks are all functions of the shape, and each
combination exercises a different path through the gate tables.

DELIBERATELY SELF-CONTAINED. The reference here shares no helper with the other file: an error in a
shared `_effective` would cancel out of both sides and neither suite would notice. It is also fully
vectorized over the batch -- the other file's reference loops, which is fine for six shapes and far too
slow for sixty.

Kept light: tiny circuits (a handful of KB of parameters), two variables, one gated layer. The cost is
dominated by compiling one PC per shape, so the grid is chosen for path coverage rather than size.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _cute_available():
    if not torch.cuda.is_available():
        return False
    from pyjuice.nodes.external_params.kernels.c import get_cute_module
    try:
        return get_cute_module() is not None
    except Exception:
        return False


needs_cute = pytest.mark.skipif(
    not _cute_available(),
    reason = "compares the CuTe/TMA fork (nvcc + CUTLASS + sm_90+) against the oracle; the "
             "Triton fork that covers a machine without it is pinned separately")


def _build(num_latents, block_size, gate_cbs, seed):
    torch.manual_seed(seed)
    nb = num_latents // block_size
    with juice.set_block_size(block_size):
        ni = [inputs(v, num_node_blocks = nb, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        prod = multiply(*ni)
        ns = summate(prod, num_node_blocks = nb,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, ns, prod


def _reference(pc, ns, prod, phi, gate_cbs, batch):
    """
    float64 node values and flows under the effective per-sample parameters, vectorized over the batch.

    Straight from the definition, with nothing borrowed from the implementation's index tables:

        eff[e][b,n,c]   = phi[b, nb_e, cb_e * n_child_gates + c // gate_cbs] * theta[e][n,c]
        Z[b, nb, n]     = sum over the edge blocks of node block nb, and over their children, of eff
        theta_b         = eff / Z
        node value      = log sum_e sum_c theta_b * exp(element_mars[child, b])
        edge_flow       = exp(node_flows) * theta_b * exp(element_mars - node_mars)

    `node_mars` is the node's value under the EFFECTIVE parameters and so already carries 1/Z; it is
    paired with theta_b here and must not also be shifted by log Z.

    The only loop is over edge blocks, of which these circuits have at most a handful.
    """
    dev = pc.node_mars.device
    bs, cbs = ns.block_size, ns.ch_block_size
    E = ns.edge_ids.size(1)
    nid0 = ns._output_ind_range[0]
    elo = prod._output_ind_range[0]
    n_child_gates = cbs // gate_cbs

    theta = pc.get_node_params(ns).double()                       # [E, bs, cbs]
    lphi = phi.double()                                           # [B, Nk, Ck]
    nb_of = ns.edge_ids[0].tolist()
    cb_of = ns.edge_ids[1].tolist()

    # per-sample effective (unnormalized) parameters of every edge block
    eff = []
    for e in range(E):
        g = lphi[:, nb_of[e],
                 cb_of[e] * n_child_gates:(cb_of[e] + 1) * n_child_gates]      # [B, n_child_gates]
        g = g.repeat_interleave(gate_cbs, dim = 1).exp()                        # [B, cbs]
        eff.append(theta[e][None, :, :] * g[:, None, :])                        # [B, bs, cbs]

    # normalizer: over all edge blocks incident to the same node block
    Z = {}
    for nb in sorted(set(nb_of)):
        Z[nb] = sum(eff[e].sum(dim = 2) for e in range(E) if nb_of[e] == nb)    # [B, bs]

    nm = pc.node_mars[:, :batch].double()
    em = pc.element_mars[:, :batch].double()
    nf = pc.node_flows[:, :batch].double()

    val = {nb: torch.zeros([batch, bs], dtype = torch.float64, device = dev)
           for nb in set(nb_of)}
    ef = torch.zeros([pc.element_flows.size(0), batch], dtype = torch.float64, device = dev)
    pf = torch.zeros([E, bs, cbs], dtype = torch.float64, device = dev)

    for e in range(E):
        nb, cb = nb_of[e], cb_of[e]
        nrows = nid0 + nb * bs + torch.arange(bs, device = dev)
        crows = elo + cb * cbs + torch.arange(cbs, device = dev)

        tb = eff[e] / Z[nb][:, :, None]                                        # [B, bs, cbs] theta_b
        e_em = em[crows].T                                                     # [B, cbs]

        val[nb] += (tb * e_em[:, None, :].exp()).sum(dim = 2)                  # [B, bs]

        w = (nf[nrows].T.exp()[:, :, None] * tb
             * (e_em[:, None, :] - nm[nrows].T[:, :, None]).exp())             # [B, bs, cbs]
        pf[e] += w.sum(dim = 0)
        ef[crows] += w.sum(dim = 1).T

    node_vals = torch.full_like(nm, -float("inf"))
    for nb in val:
        node_vals[nid0 + nb * bs: nid0 + (nb + 1) * bs] = val[nb].T.log()

    return node_vals, ef.log(), pf


# The grid. Each row is a distinct path: which fork runs, how many parent blocks a child block has,
# how many gates tile a child block, and the register state the small-batch kernels size to the batch.
# `(128, 64)` and `(64, 64)` have `num_edges % 128 != 0`, so no CUDA param fork applies and the
# Triton one has to carry them -- the shapes that motivated writing it.
LARGE = [(K, bs, g, b)
         for K, bs in ((128, 128), (256, 128), (256, 256), (512, 128), (128, 64), (64, 64))
         for g in (4, 8, 16, 32, 64)
         for b in (64, 128)
         if g <= bs and g <= 64]

SMALL = [(K, K, g, b)
         for K in (128, 256, 512)
         for g in (4, 8, 16, 32, 64)
         for b in (1, 2, 3, 8, 15)]


@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,gate_cbs,batch", LARGE)
def test_sweep_large_batch(num_latents, block_size, gate_cbs, batch):
    """The CuTe/TMA forks, over tile sizes, gate widths and parent-block counts."""
    _check(num_latents, block_size, gate_cbs, batch, tol_val = 2e-3, tol_ef = 3e-3, tol_pf = 5e-3)


@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,gate_cbs,batch", SMALL)
def test_sweep_small_batch(num_latents, block_size, gate_cbs, batch, monkeypatch):
    """The plain-CUDA forks. These are fp32 throughout, so the bar is far tighter -- the tensor-core small-batch twins
    (`test_sweep_small_batch_tensor_cores`, `test_sweep_small_batch_high_precision`) taken out, as if their extensions
    had not built."""
    import pyjuice.nodes.external_params.kernels.c as bs_c
    monkeypatch.setattr(bs_c, "get_sb_mma_module", lambda: None)
    monkeypatch.setattr(bs_c, "get_sb_ele_mma_module", lambda: None)
    _check(num_latents, block_size, gate_cbs, batch, tol_val = 1e-5, tol_ef = 1e-5, tol_pf = 1e-4)


SMALL_TC = [(K, K, g, b) for K in (128, 512) for g in (8, 64) for b in (16, 33, 63)]   # (one bf16 an operand from 16)
SMALL_HP = [(K, K, g, b) for K in (128, 512) for g in (8, 64) for b in (1, 3, 8, 15)]  # (hi + lo operands below 16)


def _force_twins(monkeypatch, hp):
    """An autotune that takes the tensor-core small-batch forward and element backward wherever offered (asserting both
    were, in the form `hp` says), the real one for every other choice."""
    import pyjuice.layer.kernels.c as ck
    real = ck.autotune
    taken = set()

    def forced(cands, *args, **kwargs):
        keys = [k for k, _ in cands]
        # (the forward's pick: among the batch-tiled kernel's arms; the element plan's: beside the plain "triton")
        for kind, mark in (("sbmma", lambda k: isinstance(k, tuple) and k[0] == "sbbt"), ("sbm", lambda k: k == "triton")):
            twin = [k for k in keys if isinstance(k, tuple) and k[0] == kind]
            if any(mark(k) for k in keys):
                assert twin, f"the tensor-core small-batch {kind} was not offered"
                assert all(bool(k[1] & 128) == hp for k in twin), f"{kind}: {twin}"
                taken.add(kind)
                return twin[0]
        return real(cands, *args, **kwargs)
    monkeypatch.setattr(ck, "autotune", forced)
    return taken


@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,gate_cbs,batch", SMALL_TC)
def test_sweep_small_batch_tensor_cores(num_latents, block_size, gate_cbs, batch, monkeypatch):
    """The tensor-core small-batch forward and element backward (blockscale_smallbatch_forward_mma.cu,
    blockscale_smallbatch_ele_mma.cu) forced: bf16 operands, so the large batch's bar."""
    taken = _force_twins(monkeypatch, hp = False)
    _check(num_latents, block_size, gate_cbs, batch, tol_val = 2e-3, tol_ef = 3e-3, tol_pf = 5e-3)
    assert taken == {"sbmma", "sbm"}, taken


@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,gate_cbs,batch", SMALL_HP)
def test_sweep_small_batch_high_precision(num_latents, block_size, gate_cbs, batch, monkeypatch):
    """Below batch 16 the same twins in their high-precision form (every operand a bf16 hi + lo pair, ~2^-16 a term):
    near the fp32 forks' bar."""
    taken = _force_twins(monkeypatch, hp = True)
    _check(num_latents, block_size, gate_cbs, batch, tol_val = 1e-4, tol_ef = 1e-4, tol_pf = 1e-3)
    assert taken == {"sbmma", "sbm"}, taken


def _check(num_latents, block_size, gate_cbs, batch, tol_val, tol_ef, tol_pf, seed = 0):
    dev = torch.device("cuda:0")
    root, ns, prod = _build(num_latents, block_size, gate_cbs, seed)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(seed + 7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    phi = torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], device = dev) * 1.5

    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)

    ref_val, ref_ef, ref_pf = _reference(pc, ns, prod, phi, gate_cbs, batch)

    lo, hi = ns._output_ind_range
    live = torch.isfinite(ref_val[lo:hi])
    d_val = float((pc.node_mars[lo:hi, :batch].double()[live] - ref_val[lo:hi][live]).abs().max())

    live = torch.isfinite(ref_ef)
    d_ef = float((pc.element_flows[:, :batch].double()[live] - ref_ef[live]).abs().max())

    ns.update_param_flows(pc.param_flows)
    got_pf = ns.get_param_flows().double().to(dev)
    d_pf = float(((got_pf - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())

    del pc
    torch.cuda.empty_cache()

    assert d_val < tol_val, f"node values off by {d_val}"
    assert d_ef < tol_ef, f"element flows off by {d_ef}"
    assert d_pf < tol_pf, f"param flows off by {d_pf} (relative)"
