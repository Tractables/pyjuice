"""
The pre-pass block-scale element-flow backward (kernels/c/blockscale_ele_backward_prepass.cu: a pre-pass and a
streaming kernel) against float64 -- the element flows and d LL / d log phi, at every config valid for a layer, with
the gradient on and off, on each gated layer's real inputs captured from a backward: gated HMMs of one k-tile, 7 (not a
multiple of the consumer groups), 16 and 64, of one to four blocks; gate widths 4 to 512 (past the element tile, and
past the element block); fp32 and bf16 parameters; log-gates -inf in places and in whole columns, children -inf;
padded (phantom k-tiles), block-sparse, two-node layers and child blocks of 32. Then the plan: it offers the configs,
and a backward taking them gives the Triton fork's parameter flows and gate gradients within the forks' rounding; the
switch; and it declines what it does not serve. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.c as ck
from pyjuice.nodes.external_params.blockscale import backward as bs_backward

DEV = torch.device("cuda:0")
NAN = float("nan")
NEG = -float("inf")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CATS = 16


def _module():
    if not torch.cuda.is_available():
        return None
    from pyjuice.nodes.external_params.kernels.c import get_ele_prepass_module
    try:
        return get_ele_prepass_module()
    except Exception:
        return None


needs_kernels = pytest.mark.skipif(_module() is None,
                                   reason = "needs the pre-pass element backward (nvcc + CUTLASS + sm_80+)")


# --------------------------------------------------------------------------------- circuits

def _hmm(L, nb, gcb, T = 4):
    """A gated HMM: T - 1 transitions with a gate, blocks of L / nb."""
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
    ("hmm64_g16", lambda: _hmm(64, 1, 16), "bf16", 64, False, False),          # one k-tile
    ("hmm256_g64", lambda: _hmm(256, 1, 64), "bf16", 64, False, False),
    ("hmm256_g8_fp32", lambda: _hmm(256, 1, 8), "fp32", 128, True, False),
    ("hmm448x7_g64", lambda: _hmm(448, 7, 64), "bf16", 64, True, False),       # 7 k-tiles (blocks of 64)
    ("hmm1024_g4", lambda: _hmm(1024, 1, 4), "bf16", 128, True, True),
    ("hmm1024_g16_fp32", lambda: _hmm(1024, 1, 16), "fp32", 64, False, True),
    ("hmm512x2_g32", lambda: _hmm(512, 2, 32), "bf16", 192, True, False),
    ("hmm1024x4_g128_fp32", lambda: _hmm(1024, 4, 128), "fp32", 64, True, True),
    ("hmm4096x4_g8", lambda: _hmm(4096, 4, 8, T = 3), "bf16", 64, True, False),  # 64 k-tiles
    ("hmm512_g512", lambda: _hmm(512, 1, 512), "bf16", 64, True, False),
    ("pad", lambda: _topology(PAD, 64, 64, 8), "bf16", 64, True, False),
    ("pad_fp32", lambda: _topology(PAD, 64, 64, 16), "fp32", 128, False, True),
    ("pad_big", lambda: _topology(PAD_BIG, 128, 64, 16, n_ch = 8), "bf16", 128, True, False),
    ("sparse", lambda: _topology(SPARSE, 64, 64, 8), "fp32", 64, True, False),
    ("two_nodes", lambda: _two(8), "bf16", 128, True, False),
    ("child32", lambda: _topology(PAD, 64, 32, 8), "bf16", 64, True, False),   # element blocks of 32 rows
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
            p[torch.rand_like(p) < 0.3] = NEG
            p[: max(1, B // 8)] = NEG                        # samples whose gates are all -inf
        phi[ns] = p
    x = torch.randint(0, CATS, (B, V), device = DEV)
    if neg_children:                                         # observations the emissions rule out
        with torch.no_grad():
            for layer in pc.input_layer_group:
                prm = layer.params.view(-1)
                prm[: prm.numel() // 4] = 0.0
    return pc, phi, x


_REAL_AUTOTUNE = ck.autotune


def _pick(prefer):
    """An autotune that picks the first candidate of kind `prefer` (a name, or a tuple's tag) without a trial, and
    leaves a choice without one to the real autotune (which drops a candidate that raises: the CuTe parameter fork
    refuses a gate wider than its subtile); it records every key."""
    seen = []

    def autotune(cands, *a, **k):
        keys = [key for key, _ in cands]
        seen.extend(keys)
        hit = next((key for key in keys if key == prefer or (isinstance(key, tuple) and key[0] == prefer)), None)
        return hit if hit is not None else _REAL_AUTOTUNE(cands, *a, **k)

    return autotune, seen


def _gated_layers(pc):
    return [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers if getattr(l, "_bs_bw_gate_cache", None)]


def _ele_plans(pc):
    return [v for layer in _gated_layers(pc) for k, v in layer._bs_bw_gate_cache.items()
            if isinstance(k, tuple) and k[0] == "eleplan"]


def _backward(pc, phi, x):
    grads = {ns: torch.zeros_like(p) for ns, p in phi.items()}
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 0.0, sum_external_params_grad = grads)
    torch.cuda.synchronize()
    return grads


# --------------------------------------------------------------------------------- the kernel

def _reference(em, nf, nm, prm, ext, chids, ebase, pbase, gate, B, blk, cbs, knt, gc, ext_base):
    """The element flows (the blocks' rows; NaN elsewhere) and the kernel's share of d LL / d log phi, in float64."""
    em, nf, nm, prm, ext = (t.double() for t in (em, nf, nm, prm, ext))
    ext = ext.view(-1, B)
    ef = torch.full(em.shape, NAN, dtype = torch.float64, device = em.device)
    g = torch.zeros(ext.shape, dtype = torch.float64, device = em.device)
    rows, k = torch.arange(cbs, device = em.device), torch.arange(64, device = em.device)
    for eb in range(chids.numel()):
        c0 = int(chids[eb])
        emb = em[c0:c0 + cbs]
        terms = []
        for t in range(knt):
            gb = int(gate[eb, t])
            if gb < 0:
                continue
            e0, p0 = int(ebase[eb, t]), int(pbase[eb, t])
            lf = torch.where(nm[e0:e0 + 64] == NEG, torch.full_like(nf[e0:e0 + 64], NEG), nf[e0:e0 + 64] - nm[e0:e0 + 64])
            mxe = lf.max(0).values
            mxe = torch.where(torch.isfinite(mxe), mxe, torch.zeros_like(mxe))
            P = prm[p0 + rows[:, None] * blk + k[None, :]]
            grow = gb + ext_base + rows // gc
            lt = torch.log(P @ torch.exp(lf - mxe)) + mxe[None, :] + ext[grow]
            terms.append(lt)
            g.index_add_(0, grow, torch.exp(lt + emb))
        ef[c0:c0 + cbs] = emb + (torch.logsumexp(torch.stack(terms), 0) if terms else NEG)
    return ef, g.view(-1)


@needs_kernels
@pytest.mark.parametrize("name,builder,dt,B,neg_gates,neg_children", CASES, ids = [c[0] for c in CASES])
def test_every_config_matches_float64(name, builder, dt, B, neg_gates, neg_children, monkeypatch):
    mod = _module()
    pc, phi, x = _instance(builder, dt, B, neg_gates, neg_children)
    captured = []
    real = mod.blockscale_ele_backward_prepass

    def capture(tgt, em, nf, nm, prm, ext, chids, ebase, pbase, gate, grad, *sizes):
        captured.append((tgt, em.clone(), nf.clone(), nm.clone(), prm.clone(), ext.clone(), chids, ebase, pbase,
                         gate, grad, sizes))
        return real(tgt, em, nf, nm, prm, ext, chids, ebase, pbase, gate, grad, *sizes)

    monkeypatch.setattr(mod, "blockscale_ele_backward_prepass", capture)
    autotune, _ = _pick("elep")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    assert captured, f"{name}: the pre-pass fork served no layer"

    checked = 0
    for tgt, em, nf, nm, prm, ext, chids, ebase, pbase, gate, grad, sizes in captured:
        batch, blk, cbs, knt, gc, ext_base, wg, _cfg = sizes
        ref_ef, ref_g = _reference(em, nf, nm, prm, ext, chids, ebase, pbase, gate, batch, blk, cbs, knt, gc, ext_base)
        rows = ~torch.isnan(ref_ef).all(1)
        valid = list(mod.fitting_configs(cbs, batch, gc, knt))
        assert valid, f"{name}: no config serves the layer"
        for cfg in valid:
            for write_grad in (1, 0):
                ef = torch.full_like(tgt, NAN)
                g = torch.zeros_like(ext) if write_grad else ext.new_empty(0)
                real(ef, em, nf, nm, prm, ext, chids, ebase, pbase, gate, g, batch, blk, cbs, knt, gc, ext_base,
                     write_grad, cfg)
                torch.cuda.synchronize()
                what = f"{name}: config {cfg} {tuple(mod.configs()[cfg])} (gradient {'on' if write_grad else 'off'})"
                r, y = ref_ef[rows], ef[rows].double()
                known = ~torch.isnan(r)
                assert not torch.isnan(y[known]).any(), f"{what}: NaN element flows"
                assert torch.equal(r[known] == NEG, y[known] == NEG), f"{what}: the -inf pattern differs"
                fin = known & torch.isfinite(r)
                if bool(fin.any()):
                    d = float((r[fin] - y[fin]).abs().max())
                    assert d < 1e-3, f"{what}: element flows off by {d:.2e}"
                if write_grad:
                    assert not torch.isnan(g).any(), f"{what}: NaN gradient"
                    scale = float(ref_g.abs().max())
                    if scale > 0:
                        d = float((g.double().view(-1) - ref_g).abs().max()) / scale
                        assert d < 1e-3, f"{what}: the gradient off by {d:.2e} (relative)"
                checked += 1
    assert checked > 0


# --------------------------------------------------------------------------------- the plan

# (no samples whose gates are all -inf: their flows and gradients are NaN in either fork and would compare nothing)
PLAN_CASES = [
    ("hmm4096x4_g8", lambda: _hmm(4096, 4, 8, T = 3), "bf16", 64, False, False),
    ("hmm1024_g16_fp32", lambda: _hmm(1024, 1, 16), "fp32", 64, False, True),
    ("pad_fp32", lambda: _topology(PAD, 64, 64, 16), "fp32", 128, False, True),
    ("two_nodes", lambda: _two(8), "bf16", 128, False, False),
    ("child32", lambda: _topology(PAD, 64, 32, 8), "bf16", 64, False, False),
]


@needs_kernels
@pytest.mark.parametrize("name,builder,dt,B,neg_gates,neg_children", PLAN_CASES, ids = [c[0] for c in PLAN_CASES])
def test_a_backward_taking_it_matches_triton(name, builder, dt, B, neg_gates, neg_children, monkeypatch):
    """The plan offers every config that serves the layer; a backward taking the pre-pass fork gives the Triton fork's
    parameter flows (where they depend on the element flows: the HMMs' lower layers) and gate gradients within the two
    forks' rounding -- Triton's TF32 dots, this fork's fp16 operands; the gradient is a difference of two terms of
    nearly equal size, so its bar is external_blockscale_grad_test.py's between a fork and the float64 reference."""
    mod = _module()
    pc, phi, x = _instance(builder, dt, B, neg_gates, neg_children)
    fitting, real_fitting = [], mod.fitting_configs

    def record(*a):
        out = real_fitting(*a)
        fitting.extend(out)
        return out

    monkeypatch.setattr(mod, "fitting_configs", record)

    def run(prefer):
        for layer in _gated_layers(pc):
            for k in [k for k in layer._bs_bw_gate_cache if isinstance(k, tuple) and k[0] == "eleplan"]:
                del layer._bs_bw_gate_cache[k]
        autotune, seen = _pick(prefer)
        monkeypatch.setattr(ck, "autotune", autotune)
        grads = _backward(pc, phi, x)
        return pc.param_flows.clone(), {ns: g.clone() for ns, g in grads.items()}, {p["kind"] for p in _ele_plans(pc)}, seen

    pf_t, g_t, kinds_t, _ = run("triton")
    pf_p, g_p, kinds_p, seen = run("elep")
    assert kinds_t == {"triton"}, kinds_t
    assert kinds_p and all(isinstance(k, tuple) and k[0] == "elep" for k in kinds_p), kinds_p
    offered = {k[1] for k in seen if isinstance(k, tuple) and k[0] == "elep"}
    assert offered and offered == set(fitting), (offered, set(fitting))

    fin = torch.isfinite(pf_t) & torch.isfinite(pf_p)
    assert torch.equal(torch.isfinite(pf_t), torch.isfinite(pf_p)), f"{name}: the parameter flows' finite pattern"
    d = float((pf_t[fin] - pf_p[fin]).abs().max() / pf_t[fin].abs().max())
    assert d < 2e-3, f"{name}: the parameter flows differ by {d:.2e} (relative)"
    for ns in g_t:
        # (samples whose gates are all -inf have a log-likelihood of -inf: their gradient is NaN in either fork)
        nan = torch.isnan(g_t[ns])
        assert torch.equal(nan, torch.isnan(g_p[ns])), f"{name}: the gate gradient's NaN pattern differs"
        scale = float(g_t[ns][~nan].abs().max())
        d = float((g_t[ns] - g_p[ns])[~nan].abs().max()) / max(scale, 1e-30)
        assert d < 2e-2, f"{name}: the gate gradient differs by {d:.2e} (relative)"


@needs_kernels
def test_the_switch_leaves_it_out(monkeypatch):
    pc, phi, x = _instance(lambda: _hmm(256, 1, 16), "bf16", 64, False, False)
    monkeypatch.setattr(bs_backward, "ELE_PREPASS", False)
    autotune, seen = _pick("elep")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    assert seen and not any(isinstance(k, tuple) and k[0] == "elep" for k in seen), seen
    assert not any(isinstance(p["kind"], tuple) and p["kind"][0] == "elep" for p in _ele_plans(pc))


@needs_kernels
@pytest.mark.parametrize("label,builder,B", [
    ("gate width 2", lambda: _hmm(256, 1, 2), 64),
    ("batch 32", lambda: _hmm(256, 1, 16), 32),
    ("parent blocks of 32", lambda: _topology(PAD, 32, 64, 8), 64),   # a k-tile of 64 parents spans two blocks
])
def test_it_declines_what_it_does_not_serve(label, builder, B, monkeypatch):
    pc, phi, x = _instance(builder, "bf16", B, False, False)
    autotune, seen = _pick("elep")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    plans = _ele_plans(pc)
    assert plans, f"{label}: no element plan was built"
    assert not any(isinstance(k, tuple) and k[0] == "elep" for k in seen), f"{label}: {seen}"
    assert not any(isinstance(p["kind"], tuple) and p["kind"][0] == "elep" for p in plans), f"{label}"
