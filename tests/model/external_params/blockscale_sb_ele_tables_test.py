"""
The gated small-batch element backward (kernels/c/blockscale_smallbatch_backward.cu, `blockscale_sb_ele_backward`) on
layers of SEVERAL parent blocks, which it addresses through the per-tile parent / parameter tables (its gate per parent
block flushed where a block ends, whether a block spans tiles or a tile spans blocks): the plan offers it there, and
forced to each config it gives the Triton kind's element flows AND gate gradients, at batches 1 to 15, fp32 and bf16,
gate widths 8 / 16 / 32, log-gates -inf in places. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.c as ck

DEV = torch.device("cuda:0")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _sb_module():
    if not torch.cuda.is_available():
        return None
    from pyjuice.nodes.external_params.kernels.c import get_sb_bw_module
    try:
        return get_sb_bw_module()
    except Exception:
        return None


needs_kernels = pytest.mark.skipif(_sb_module() is None, reason = "needs the small-batch block-scale backward (nvcc)")


def _ghmm(L, T, gcb, dt, V = 16):
    torch.manual_seed(0)
    nb = 4
    gated = []
    with juice.set_block_size(L // nb):
        ns = inputs(0, num_node_blocks = nb, dist = dists.Categorical(num_cats = V))
        src, esrc = None, None
        for t in range(1, T):
            emit = esrc.duplicate(t, tie_params = True) if esrc is not None else None
            if emit is None:
                emit = esrc = inputs(t, num_node_blocks = nb, dist = dists.Categorical(num_cats = V))
            if src is not None:
                ns = src.duplicate(multiply(ns, emit), tie_params = True)
            else:
                ns = src = summate(multiply(ns, emit), num_node_blocks = nb,
                                   external_params = BlockScaleSumParams(ch_block_size = gcb))
            gated.append(ns)
        ns = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = dt).to(DEV)
    return pc, gated, T, V


class _Force:
    """`ck.autotune` with the block-scale BACKWARD's element kind forced (its Triton kind is the string "triton");
    the forward's picks go to the real tuner."""

    def __init__(self, real):
        self.real, self.kind, self.offered = real, None, []

    def __call__(self, cands, warmup = 3, reps = 7):
        keys = [k for k, _ in cands]
        if "triton" not in keys:
            return self.real(cands, warmup, reps)
        self.offered.append(keys)
        assert self.kind in keys, (self.kind, keys)
        return self.kind


def _run(pc, phi, x, force, kind):
    force.kind = kind
    for lg in pc.inner_layer_groups:
        for layer in lg:
            for a in [k for k in layer.__dict__ if k.startswith("_bs_")]:
                layer.__dict__.pop(a, None)                         # every cached plan: the forced kind is re-chosen
    pc(x, sum_external_params = phi)
    pc.backward(x, compute_param_flows = False, sum_external_params = phi)
    ef = pc.element_flows.detach().clone()
    grads = [pc.get_external_params_grad(g)[0].detach().clone() for g in phi]
    return ef, grads


@needs_kernels
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("shape", ["L1024-g32", "L512-g16", "L2048-g8"])
def test_offered_on_several_parent_blocks_and_matches_triton(shape, dtype, monkeypatch):
    L, gcb = {"L1024-g32": (1024, 32), "L512-g16": (512, 16), "L2048-g8": (2048, 8)}[shape]
    pc, gated, T, V = _ghmm(L, 4, gcb, DTYPES[dtype])
    force = _Force(ck.autotune)
    monkeypatch.setattr(ck, "autotune", force)
    for B in (1, 2, 5, 8, 15):
        torch.manual_seed(B)
        phi = {g: torch.randn(g.external_params.tensor_shapes(g, B)[0], device = DEV) * 2 for g in gated}
        for t in phi.values():
            t[torch.rand(t.shape, device = DEV) < 0.1] = -float("inf")
        x = torch.randint(0, V, (B, T), device = DEV)
        ef_t, gr_t = _run(pc, phi, x, force, "triton")
        assert any(any(isinstance(k, tuple) and k[0] == "sb" for k in keys) for keys in force.offered), \
            "the small-batch element backward is not offered"
        for cfg in range(3):
            ef, gr = _run(pc, phi, x, force, ("sb", cfg))
            fin = torch.isfinite(ef_t)
            assert not torch.isnan(ef).any() and torch.equal(torch.isfinite(ef), fin), (B, cfg)
            assert (ef[fin] - ef_t[fin]).abs().max() < 1e-3, (B, cfg)
            for a, b in zip(gr, gr_t):
                assert (a - b).abs().max() < 1e-2 * max(1.0, b.abs().max().item()), (B, cfg)
