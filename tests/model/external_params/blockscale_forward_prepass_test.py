"""
The pre-pass block-scale forward (kernels/c/blockscale_forward_prepass.cu: a pre-pass, the merge factors and a
streaming kernel) against the CuTe fork it reproduces -- bit-identical, `node_mars` and `log Z`, at every config valid
for a layer against every config of the fork, on each gated layer's real inputs: gated HMMs of one edge tile, 7
(not a multiple of the consumer groups), 16 and 64, of one to four node blocks; gate widths 4 to 512 (past the edge
tile); fp32 and bf16 parameters; log-gates -inf in places and in whole columns, children -inf; padded, block-sparse
and two-node layers. Then the plan: it offers the configs, a forward taking them gives the fork's log-likelihoods and
the backward its flows; and it declines what it does not serve. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.c as ck
from pyjuice.nodes.external_params.blockscale import forward as bs_forward
from pyjuice.nodes.external_params.blockscale.common import _buffer_kwarg

DEV = torch.device("cuda:0")
NAN = float("nan")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CATS = 16


def _modules():
    if not torch.cuda.is_available():
        return None, None
    from pyjuice.nodes.external_params.kernels.c import get_cute_module, get_prepass_module
    try:
        return get_cute_module(), get_prepass_module()
    except Exception:
        return None, None


needs_kernels = pytest.mark.skipif(any(m is None for m in _modules()),
                                   reason = "needs the CuTe fork and the pre-pass forward (nvcc + CUTLASS + sm_90+)")


# --------------------------------------------------------------------------------- circuits

def _hmm(L, nb, gcb, T = 4):
    """A gated HMM: T - 1 transitions with a gate, node blocks of L / nb."""
    bs = L // nb
    with juice.set_block_size(bs):
        ns = inputs(0, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS))
        gated = []
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS))
            ns = summate(multiply(ns, emit), num_node_blocks = nb,
                         external_params = BlockScaleSumParams(ch_block_size = gcb))
            gated.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    return root, gated, T


def _eids(rows):
    return torch.tensor([[i for i, chs in enumerate(rows) for _ in chs], [c for chs in rows for c in chs]])


def _topology(rows, bs, ch_bs, gcb, n_ch = 4):
    """One gated sum layer, its node blocks' child blocks `rows` (padded where they differ, block-sparse where they
    are not adjacent), over a product of two inputs."""
    with juice.set_block_size(ch_bs):
        ni = [inputs(v, num_node_blocks = n_ch, dist = dists.Categorical(num_cats = CATS)) for v in range(2)]
        prod = multiply(*ni)
    ns = summate(prod, num_node_blocks = len(rows), edge_ids = _eids(rows), block_size = bs,
                 external_params = BlockScaleSumParams(ch_block_size = gcb))
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    return root, [ns], 2


def _two(gcb, bs = 64, nb = 4):
    """Two gated nodes compiled into ONE sum layer (each its own slab of the staging buffer)."""
    with juice.set_block_size(bs):
        ni = [inputs(v, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS)) for v in range(4)]
        p0, p1 = multiply(ni[0], ni[1]), multiply(ni[2], ni[3])
    s0 = summate(p0, num_node_blocks = nb, block_size = bs, external_params = BlockScaleSumParams(ch_block_size = gcb))
    s1 = summate(p1, num_node_blocks = nb, block_size = bs, external_params = BlockScaleSumParams(ch_block_size = gcb))
    root = summate(multiply(s0, s1), num_node_blocks = 1, block_size = 1)
    return root, [s0, s1], 4


PAD = [[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
PAD_BIG = [list(range(8))] + [list(range(7))] * 7
SPARSE = [[0, 2], [0, 2], [1, 3], [1, 3]]

# (name, builder, params dtype, batch, -inf log-gates, -inf children)
CASES = [
    ("hmm64_g16", lambda: _hmm(64, 1, 16), "bf16", 64, False, False),          # one edge tile
    ("hmm256_g64", lambda: _hmm(256, 1, 64), "bf16", 64, False, False),
    ("hmm256_g8_fp32", lambda: _hmm(256, 1, 8), "fp32", 128, True, False),
    ("hmm448x7_g64", lambda: _hmm(448, 7, 64), "bf16", 64, True, False),       # 7 edge tiles (blocks of 64)
    ("hmm1024_g4", lambda: _hmm(1024, 1, 4), "bf16", 128, True, True),
    ("hmm1024_g16_fp32", lambda: _hmm(1024, 1, 16), "fp32", 64, False, True),
    ("hmm512x2_g32", lambda: _hmm(512, 2, 32), "bf16", 192, True, False),
    ("hmm1024x4_g128_fp32", lambda: _hmm(1024, 4, 128), "fp32", 64, True, True),
    ("hmm4096x4_g8", lambda: _hmm(4096, 4, 8, T = 3), "bf16", 64, True, False),  # 64 edge tiles
    ("hmm512_g512", lambda: _hmm(512, 1, 512), "bf16", 64, True, False),
    ("pad", lambda: _topology(PAD, 64, 64, 8), "bf16", 64, True, False),
    ("pad_fp32", lambda: _topology(PAD, 64, 64, 16), "fp32", 128, False, True),
    ("pad_big", lambda: _topology(PAD_BIG, 128, 64, 16, n_ch = 8), "bf16", 128, True, False),
    ("sparse", lambda: _topology(SPARSE, 64, 64, 8), "fp32", 64, True, False),
    ("two_nodes", lambda: _two(8), "bf16", 128, True, False),
]


def _instance(builder, dt, B, neg_gates, neg_children, seed = 0):
    torch.manual_seed(seed)
    root, gated, V = builder()
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = DTYPES[dt]).to(DEV)
    torch.manual_seed(seed + 1)
    phi = {}
    for i, ns in enumerate(gated):
        p = torch.randn(ns.external_params.tensor_shapes(ns, B)[0], device = DEV) * 3.0
        if neg_gates:
            p[torch.rand_like(p) < 0.3] = -float("inf")
            p[: max(1, B // 8)] = -float("inf")             # samples whose gates are all -inf
        phi[ns] = p
    x = torch.randint(0, CATS, (B, V), device = DEV)
    if neg_children:                                         # observations the emissions rule out
        with torch.no_grad():
            for layer in pc.input_layer_group:
                prm = layer.params.view(-1)
                prm[: prm.numel() // 4] = 0.0
    return pc, phi, x


def _pick(prefer):
    """An autotune that picks the first candidate of kind `prefer` (else the first), recording every key."""
    seen = []

    def autotune(cands, *a, **k):
        keys = [key for key, _ in cands]
        seen.extend(keys)
        return next((key for key in keys if isinstance(key, tuple) and key[0] == prefer), keys[0])

    return autotune, seen


def _bits(t):
    return t.contiguous().view(torch.int32)


def _gated_layers(pc):
    return [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers if getattr(l, "_bs_fw_plans", None)]


# --------------------------------------------------------------------------------- the kernel

@needs_kernels
@pytest.mark.parametrize("name,builder,dt,B,neg_gates,neg_children", CASES, ids = [c[0] for c in CASES])
def test_every_config_matches_the_fork_to_the_bit(name, builder, dt, B, neg_gates, neg_children, monkeypatch):
    fork, pre = _modules()
    pc, phi, x = _instance(builder, dt, B, neg_gates, neg_children)
    captured = []
    orig = bs_forward._ForwardMixin.forward_layer

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs):
        ext = kwargs.get(_buffer_kwarg(), None)
        em = element_mars.clone()
        out = orig(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs)
        if ext is not None:
            captured.append((layer, em, ext.clone(), params, node_mars))
        return out

    monkeypatch.setattr(bs_forward._ForwardMixin, "forward_layer", forward_layer)
    autotune, _ = _pick("cute")
    monkeypatch.setattr(ck, "autotune", autotune)
    pc(x, sum_external_params = phi)
    torch.cuda.synchronize()

    checked = 0
    for layer, em, ext, params, node_mars in captured:
        mod, fname, calls = layer._bs_fw_plans[B][0]
        assert fname == "blockscale_forward", f"{name}: the fork does not serve this layer ({fname})"
        for args in calls:
            (nids, cids, pids, gate, log_z, bs, ne, node_cbs, gate_cbs, n_node_gates, ext_base, _c) = args
            ref = None
            for cfg in fork.fitting_configs(bs, B, gate_cbs):
                nm, lz = torch.full_like(node_mars, NAN), torch.full_like(log_z, NAN)
                fork.blockscale_forward(nm, em, params, ext, nids, cids, pids, gate, lz, bs, ne, node_cbs, gate_cbs,
                                        n_node_gates, ext_base, cfg)
                if ref is None:
                    ref = (nm, lz)
                else:                                    # (the fork's own configs agree: the bar is theirs)
                    assert torch.equal(_bits(nm), _bits(ref[0])) and torch.equal(_bits(lz), _bits(ref[1]))
            valid = pre.fitting_configs(bs, B, node_cbs, gate_cbs)
            assert len(valid) > 0, f"{name}: no pre-pass config serves the layer"
            for cfg in valid:
                nm, lz = torch.full_like(node_mars, NAN), torch.full_like(log_z, NAN)
                pre.blockscale_forward_prepass(nm, em, params, ext, nids, cids, pids, gate, lz, bs, ne, node_cbs,
                                               gate_cbs, n_node_gates, ext_base, cfg)
                tile = tuple(pre.configs()[cfg])
                assert torch.equal(_bits(nm), _bits(ref[0])), f"{name}: config {cfg} {tile}: node_mars differ"
                assert torch.equal(_bits(lz), _bits(ref[1])), f"{name}: config {cfg} {tile}: log Z differs"
                checked += 1
    assert checked > 0


# --------------------------------------------------------------------------------- the plan

@needs_kernels
@pytest.mark.parametrize("name,builder,dt,B,neg_gates,neg_children",
                         [c for c in CASES if c[0] in ("hmm1024_g4", "pad_fp32", "two_nodes")],
                         ids = ["hmm1024_g4", "pad_fp32", "two_nodes"])
def test_a_forward_taking_it_gives_the_forks_results(name, builder, dt, B, neg_gates, neg_children, monkeypatch):
    """The plan offers every config that serves the layer; a forward taking the pre-pass copy gives the fork's
    log-likelihoods and log Z to the bit, and the backward after it the fork's flows."""
    _, pre = _modules()
    pc, phi, x = _instance(builder, dt, B, neg_gates, neg_children)

    def run(prefer):
        for layer in _gated_layers(pc):
            layer._bs_fw_plans.clear()
        autotune, seen = _pick(prefer)
        monkeypatch.setattr(ck, "autotune", autotune)
        lls = pc(x, sum_external_params = phi).clone()
        zs = [lz.clone() for layer in _gated_layers(pc) for (_n, lz, _r) in layer._bs_bw_state["shift_args"]]
        fnames = {layer._bs_fw_plans[B][0][1] for layer in _gated_layers(pc)}
        pc.backward(x, flows_memory = 0.0)
        return lls, zs, pc.param_flows.clone(), fnames, seen

    lls_a, zs_a, pf_a, fn_a, _ = run("cute")
    lls_c, zs_c, pf_c, fn_c, _ = run("cute")
    lls_b, zs_b, pf_b, fn_b, seen = run("pre")
    assert fn_a == {"blockscale_forward"} and fn_b == {"blockscale_forward_prepass"}, (fn_a, fn_b)
    for layer in _gated_layers(pc):
        mod, fname, calls = layer._bs_fw_plans[B][0]
        a = calls[0]
        offered = {key[1] for key in seen if isinstance(key, tuple) and key[0] == "pre"}
        assert offered == set(pre.fitting_configs(a[5], B, a[7], a[8])), offered
    assert torch.equal(_bits(lls_a), _bits(lls_b)), f"{name}: log-likelihoods differ"
    assert all(torch.equal(_bits(u), _bits(v)) for u, v in zip(zs_a, zs_b)), f"{name}: log Z differs"
    # Within the fork's own run-to-run spread, with a floor: the backward's atomics vary its bits, and two runs that
    # happened to agree proved nothing (a third differed by an ulp in a loaded suite).
    spread = float((pf_a - pf_c).abs().max())
    tol = 4 * spread + 1e-6 * float(pf_a.abs().max())
    assert float((pf_a - pf_b).abs().max()) <= tol, f"{name}: the backward's parameter flows differ"


@needs_kernels
def test_the_switch_leaves_it_out(monkeypatch):
    pc, phi, x = _instance(lambda: _hmm(256, 1, 16), "bf16", 64, False, False)
    monkeypatch.setattr(bs_forward, "FORWARD_PREPASS", False)
    autotune, seen = _pick("pre")
    monkeypatch.setattr(ck, "autotune", autotune)
    pc(x, sum_external_params = phi)
    assert not any(isinstance(k, tuple) and k[0] == "pre" for k in seen) and seen, seen
    assert {layer._bs_fw_plans[64][0][1] for layer in _gated_layers(pc)} != {"blockscale_forward_prepass"}


@needs_kernels
@pytest.mark.parametrize("label,builder", [
    ("child blocks of 32", lambda: _topology(PAD, 64, 32, 8)),     # an edge tile spans two runs of children
    ("gate width 2", lambda: _hmm(256, 1, 2)),
])
def test_it_declines_what_it_does_not_serve(label, builder, monkeypatch):
    _, pre = _modules()
    pc, phi, x = _instance(builder, "bf16", 64, False, False)
    autotune, seen = _pick("pre")
    monkeypatch.setattr(ck, "autotune", autotune)
    pc(x, sum_external_params = phi)
    assert seen and not any(isinstance(k, tuple) and k[0] == "pre" for k in seen), f"{label}: {seen}"
