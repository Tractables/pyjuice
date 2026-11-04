"""
The batch-tiled small-batch block-scale forward (kernels/c/blockscale_smallbatch_forward_bt.cu: the parameters loaded
once per BT batch columns) against the per-sample small-batch kernel it tiles -- `node_mars` of every gated layer and
the log-likelihoods, at every config, on gated HMMs of three gate widths at batches 1 to 33 (tile tails included), fp32
and bf16 parameters, log-gates -inf in places and in whole columns; and the plan offers its configs beside the
per-sample kernel's and takes a forced pick. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.c as ck

DEV = torch.device("cuda:0")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _modules():
    if not torch.cuda.is_available():
        return None, None
    from pyjuice.nodes.external_params.kernels.c import get_sb_module, get_sb_bt_module
    try:
        return get_sb_module(), get_sb_bt_module()
    except Exception:
        return None, None


needs_kernels = pytest.mark.skipif(any(m is None for m in _modules()),
                                   reason = "needs the small-batch block-scale forwards (nvcc)")


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
    """`ck.autotune` with the block-scale forward's pick forced to a kind (and config)."""

    def __init__(self, real):
        self.real, self.kind, self.offered = real, None, []

    def __call__(self, cands, warmup = 3, reps = 7):
        keys = [k for k, _ in cands]
        self.offered.append(keys)
        if self.kind is None:
            return self.real(cands, warmup, reps)
        want = [k for k in keys if k[0] == self.kind[0] and (self.kind[1] is None or k[1] == self.kind[1])]
        assert want, (self.kind, keys)
        return want[0]


def _run(pc, phi, x, force, kind):
    force.kind = kind
    for lg in pc.inner_layer_groups:
        for layer in lg:
            layer.__dict__.pop("_bs_fw_plans", None)                  # a fresh plan takes the forced kind
    ll = pc(x, sum_external_params = phi)
    return ll.detach().clone(), pc.node_mars.detach().clone()


@needs_kernels
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("shape", ["L512-g32", "L1024-g16", "L256-g8"])
def test_every_config_matches_the_per_sample_kernel(shape, dtype, monkeypatch):
    L, gcb = {"L512-g32": (512, 32), "L1024-g16": (1024, 16), "L256-g8": (256, 8)}[shape]
    dt = DTYPES[dtype]
    pc, gated, T, V = _ghmm(L, 5, gcb, dt)
    force = _Force(ck.autotune)
    monkeypatch.setattr(ck, "autotune", force)
    from pyjuice.nodes.external_params.kernels.c import get_sb_bt_module
    n_cfg = len(get_sb_bt_module().blockscale_sb_bt_configs())
    for B in (1, 3, 8, 16, 33):
        torch.manual_seed(B)
        phi = {g: torch.randn(g.external_params.tensor_shapes(g, B)[0], device = DEV) * 2 for g in gated}
        for g, t in phi.items():                                      # log-gates -inf in places and in a whole column
            t[torch.rand(t.shape, device = DEV) < 0.1] = -float("inf")
            if B > 2:
                t[1] = -float("inf")
        x = torch.randint(0, V, (B, T), device = DEV)
        ll_sb, nm_sb = _run(pc, phi, x, force, ("sb", 0))
        assert any(any(k[0] == "sbbt" for k in keys) for keys in force.offered), "the batch-tiled configs are not offered"
        for cfg in range(n_cfg):
            ll, nm = _run(pc, phi, x, force, ("sbbt", cfg))
            assert not torch.isnan(nm).any(), (B, cfg)
            fin = torch.isfinite(nm_sb)
            assert torch.equal(torch.isfinite(nm), fin), (B, cfg)
            assert (nm[fin] - nm_sb[fin]).abs().max() < 1e-3, (B, cfg)
            lf = torch.isfinite(ll_sb)
            assert torch.equal(torch.isfinite(ll), lf) and (ll[lf] - ll_sb[lf]).abs().max() < 1e-3, (B, cfg)
