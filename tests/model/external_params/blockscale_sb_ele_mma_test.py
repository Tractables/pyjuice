"""
The tensor-core small-batch block-scale element-flow backward (kernels/c/blockscale_smallbatch_ele_mma.cu: per 64-parent
tile a product under the tile's own stabilizer, folded in under the parent block's gates) against float64 -- the
element flows and d LL / d log phi at every arm the dispatch offers a layer (config x split: ranges of whole parent
blocks and ranges slicing one), the gradient on and off,
on each gated layer's real inputs captured from a backward: batches 1 to 63 (partial batch tiles; below 16 the
high-precision form, held to the fp32 bar), parent blocks of 64
to 1024 (one to sixteen tiles a block), one to sixteen gates a child tile and gates wider than the child block; fp32
parameters (hi + lo operands) and bf16 ones; log-gates -inf in places and in whole columns, children -inf; padded
(phantom k-tiles), block-sparse and two-node layers. Then the plan: a backward taking the fork gives the Triton fork's
parameter flows and gate gradients within the forks' rounding, the same gradient bits twice, and it declines what it
does not serve. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.c as ck
from pyjuice.utils.launch_plans import LaunchPlans

DEV = torch.device("cuda:0")
NAN = float("nan")
NEG = -float("inf")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CATS = 16
# element flows (log space, absolute) and gradients (relative to the largest) against float64: bf16 parameters are
# exact and the exponentials one bf16 (2^-9 a term); fp32 parameters both operands hi + lo (~2^-16), and so the
# high-precision form's exponentials ("hp", code bit 7: below batch 16)
TOL = {"bf16": 2.5e-3, "fp32": 1e-4, "hp": 1e-4}


def _ext():
    if not torch.cuda.is_available():
        return None
    import pyjuice.nodes.external_params.kernels.c as ext_c
    try:
        return ext_c if ext_c.get_sb_ele_mma_module() is not None else None
    except Exception:
        return None


needs_kernels = pytest.mark.skipif(_ext() is None,
                                   reason = "needs the tensor-core small-batch element backward (nvcc + CUTLASS + sm_80+)")


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
    ("hmm1024x16_g32_bf16", lambda: _hmm(1024, 16, 32), "bf16", 48, True, False),      # parent blocks of 64: 16 a row
    ("hmm1024x16_g32_fp32", lambda: _hmm(1024, 16, 32), "fp32", 20, False, True),
    ("hmm256_g8_fp32", lambda: _hmm(256, 1, 8), "fp32", 33, True, False),              # a block of 4 tiles, 16+ gates
    ("hmm1024_g16_bf16", lambda: _hmm(1024, 1, 16), "bf16", 16, False, True),          # one block of 16 tiles
    ("hmm1024x4_g128_fp32", lambda: _hmm(1024, 4, 128), "fp32", 63, True, True),
    ("hmm4096x4_g8_bf16", lambda: _hmm(4096, 4, 8, T = 3), "bf16", 32, True, False),   # 16 gates a 128-row tile
    ("hmm512_g512_bf16", lambda: _hmm(512, 1, 512), "bf16", 17, True, False),          # one gate an element block
    ("pad_bf16", lambda: _topology(PAD, 64, 64, 8), "bf16", 40, True, False),           # phantom k-tiles
    ("pad_big_fp32", lambda: _topology(PAD_BIG, 128, 64, 16, n_ch = 8), "fp32", 24, True, False),
    ("sparse_fp32", lambda: _topology(SPARSE, 64, 64, 8), "fp32", 31, True, False),
    ("two_nodes_bf16", lambda: _two(8), "bf16", 50, True, False),
    # below 16: the high-precision form
    ("hmm1024x16_g32_bf16_b4", lambda: _hmm(1024, 16, 32), "bf16", 4, True, False),
    ("hmm1024x4_g128_fp32_b7", lambda: _hmm(1024, 4, 128), "fp32", 7, True, True),
    ("hmm4096x4_g8_bf16_b8", lambda: _hmm(4096, 4, 8, T = 3), "bf16", 8, True, False),
    ("pad_bf16_b1", lambda: _topology(PAD, 64, 64, 8), "bf16", 1, False, False),
    ("sparse_bf16_b15", lambda: _topology(SPARSE, 64, 64, 8), "bf16", 15, True, True),
    ("two_nodes_bf16_b13", lambda: _two(8), "bf16", 13, True, False),
]


def _instance(builder, dt, B, neg_gates, neg_children, seed = 0):
    torch.manual_seed(seed)
    root, gated, V = builder()
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = DTYPES[dt]).to(DEV)
    torch.manual_seed(seed + 1)
    phi = {}
    for ns in gated:
        p = torch.randn(ns.external_params.tensor_shapes(ns, B)[0], device = DEV) * 3.0
        if neg_gates:
            p[torch.rand_like(p) < 0.3] = NEG
            p[: max(1, B // 8)] = NEG                        # samples whose gates are all -inf
        phi[ns] = p
    x = torch.randint(0, CATS, (B, V), device = DEV)
    if neg_children:
        with torch.no_grad():
            gen = torch.Generator(device = DEV).manual_seed(seed + 2)
            for layer in pc.input_layer_group:
                prm = layer.params.view(-1)
                prm[torch.rand(prm.shape, generator = gen, device = prm.device) < 0.25] = 0.0
    return pc, phi, x


_REAL_AUTOTUNE = ck.autotune


def _pick(prefer):
    """An autotune that picks the first candidate of kind `prefer` (a name, or a tuple's tag) without a trial, leaving a
    choice without one to the real autotune; it records every key."""
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


def _reset(pc):
    for layer in _gated_layers(pc):
        for k in [k for k in layer._bs_bw_gate_cache if isinstance(k, tuple) and k[0] == "eleplan"]:
            del layer._bs_bw_gate_cache[k]
        LaunchPlans.of(layer).clear()                       # (its plans replay the choice dropped above)


# --------------------------------------------------------------------------------- the kernel

def _reference(em, nf, nm, prm, ext, chids, ebase, pbase, gate, B, blk, cbs, ne, gc, ext_base):
    """The element flows (the blocks' rows; NaN elsewhere) and the kernel's share of d LL / d log phi, in float64: per
    64-parent tile under its parent block's gate row (`gate[eb, t // tiles a block]`)."""
    em, nf, nm, prm, ext = (t.double() for t in (em, nf, nm, prm, ext))
    ext = ext.view(-1, B)
    ef = torch.full(em.shape, NAN, dtype = torch.float64, device = em.device)
    g = torch.zeros(ext.shape, dtype = torch.float64, device = em.device)
    rows, k = torch.arange(cbs, device = em.device), torch.arange(64, device = em.device)
    tpb = blk // 64
    for eb in range(chids.numel()):
        c0 = int(chids[eb])
        emb = em[c0:c0 + cbs]
        terms = []
        for t in range(ne // 64):
            gb = int(gate[eb, t // tpb])
            if gb < 0:
                continue
            e0, p0 = int(ebase[eb, t]), int(pbase[eb, t])
            lf = torch.where(nm[e0:e0 + 64] == NEG, torch.full_like(nf[e0:e0 + 64], NEG),
                             nf[e0:e0 + 64] - nm[e0:e0 + 64])
            mxe = lf.max(0).values
            mxe = torch.where(torch.isfinite(mxe), mxe, torch.zeros_like(mxe))
            P = prm[p0 + rows[:, None] * blk + k[None, :]]
            grow = gb + ext_base + rows // gc
            lt = torch.log(P @ torch.exp(lf - mxe)) + mxe[None, :] + ext[grow]
            terms.append(lt)
            g.index_add_(0, grow, torch.exp(lt + emb))
        ef[c0:c0 + cbs] = emb + (torch.logsumexp(torch.stack(terms), 0) if terms else NEG)
    return ef, g.view(-1)


def _capture(monkeypatch):
    mod = _ext().get_sb_ele_mma_module()
    captured, real = [], mod.blockscale_sb_ele_backward_mma

    def capture(tgt, em, nf, nm, prm, ext, chids, ebase, pbase, gate, grad, gpart, scratch, *sizes):
        captured.append((tgt, em.clone(), nf.clone(), nm.clone(), prm.clone(), ext.clone(), chids, ebase, pbase, gate,
                         sizes))
        return real(tgt, em, nf, nm, prm, ext, chids, ebase, pbase, gate, grad, gpart, scratch, *sizes)

    monkeypatch.setattr(mod, "blockscale_sb_ele_backward_mma", capture)
    return mod, real, captured


@needs_kernels
@pytest.mark.parametrize("name,builder,dt,B,neg_gates,neg_children", CASES, ids = [c[0] for c in CASES])
def test_every_arm_matches_float64(name, builder, dt, B, neg_gates, neg_children, monkeypatch):
    ext_c = _ext()
    pc, phi, x = _instance(builder, dt, B, neg_gates, neg_children)
    mod, real, captured = _capture(monkeypatch)
    autotune, _ = _pick("sbm")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    assert captured, f"{name}: the tensor-core fork served no layer"

    checked = 0
    for tgt, em, nf, nm, prm, ext, chids, ebase, pbase, gate, sizes in captured:
        batch, blk, cbs, ne, gc, ext_base, _code = sizes
        ref_ef, ref_g = _reference(em, nf, nm, prm, ext, chids, ebase, pbase, gate, batch, blk, cbs, ne, gc, ext_base)
        rows = ~torch.isnan(ref_ef).all(1)
        arms = ext_c.blockscale_sb_ele_mma_arms(cbs, blk, batch, ne, chids.numel(), gc, prm.dtype == torch.float32)
        assert arms, f"{name}: no arm serves the layer"
        assert all(bool(code & 128) == (batch < 16) for code in arms), f"{name}: the high-precision form iff below 16"
        for code in arms:
            tol = TOL["hp" if code & 128 else "fp32" if prm.dtype == torch.float32 else "bf16"]
            for write_grad in (1, 0):
                ef = torch.full_like(tgt, NAN)
                g = torch.zeros_like(ext) if write_grad else ext.new_empty(0)
                # (shares a unit: a parent block, or a range where ranges slice the blocks; NaN-poisoned, every one written)
                gpart = torch.full((chids.numel() * cbs * max(ne // blk, code >> 8) * batch,), NAN, device = DEV)
                real(ef, em, nf, nm, prm, ext, chids, ebase, pbase, gate, g, gpart, ext.new_empty(0), batch, blk, cbs,
                     ne, gc, ext_base, code)
                torch.cuda.synchronize()
                what = (f"{name}: config {code & 127} {tuple(mod.blockscale_sb_ele_mma_configs()[code & 127])}"
                        f"{' hp' if code & 128 else ''} split {code >> 8} (gradient {'on' if write_grad else 'off'})")
                r, y = ref_ef[rows], ef[rows].double()
                known = ~torch.isnan(r)
                assert not torch.isnan(y[known]).any(), f"{what}: NaN element flows"
                assert torch.equal(r[known] == NEG, y[known] == NEG), f"{what}: the -inf pattern differs"
                fin = known & torch.isfinite(r)
                assert bool(fin.any()), f"{what}: no finite element flow to compare"
                d = float((r[fin] - y[fin]).abs().max())
                assert d < tol, f"{what}: element flows off by {d:.2e}"
                if write_grad:
                    assert not torch.isnan(g).any(), f"{what}: NaN gradient"
                    scale = float(ref_g.abs().max())
                    assert scale > 0, f"{what}: the reference gradient is zero: nothing compared"
                    d = float((g.double().view(-1) - ref_g).abs().max()) / scale
                    assert d < tol, f"{what}: the gradient off by {d:.2e} (relative)"
                checked += 1
    assert checked > 0


# --------------------------------------------------------------------------------- the plan

# (no samples whose gates are all -inf: their flows and gradients are NaN in either fork and would compare nothing)
PLAN_CASES = [
    ("hmm1024x16_g32_bf16", lambda: _hmm(1024, 16, 32), "bf16", 48, False, False),
    ("hmm1024x4_g128_fp32", lambda: _hmm(1024, 4, 128), "fp32", 33, False, True),
    ("pad_big_fp32", lambda: _topology(PAD_BIG, 128, 64, 16, n_ch = 8), "fp32", 24, False, False),
    ("two_nodes_bf16", lambda: _two(8), "bf16", 16, False, False),
    ("hmm1024x16_g32_bf16_b8", lambda: _hmm(1024, 16, 32), "bf16", 8, False, False),   # the high-precision form
]


@needs_kernels
@pytest.mark.parametrize("name,builder,dt,B,neg_gates,neg_children", PLAN_CASES, ids = [c[0] for c in PLAN_CASES])
def test_a_backward_taking_it_matches_triton(name, builder, dt, B, neg_gates, neg_children, monkeypatch):
    """A backward taking the tensor-core fork gives the Triton fork's parameter flows (where they depend on the element
    flows: the HMMs' lower layers) and gate gradients within the two forks' rounding (Triton's TF32 dots; the gradient a
    difference of nearly equal terms, external_blockscale_grad_test.py's bar between a fork and the reference), and the
    same gradient bits on a second backward (its shares summed in order, no atomics across a gate's children)."""
    pc, phi, x = _instance(builder, dt, B, neg_gates, neg_children)

    def run(prefer):
        _reset(pc)
        autotune, seen = _pick(prefer)
        monkeypatch.setattr(ck, "autotune", autotune)
        grads = _backward(pc, phi, x)
        return pc.param_flows.clone(), {ns: g.clone() for ns, g in grads.items()}, {p["kind"] for p in _ele_plans(pc)}

    pf_t, g_t, kinds_t = run("triton")
    pf_p, g_p, kinds_p = run("sbm")
    assert kinds_t == {"triton"}, kinds_t
    assert kinds_p and all(isinstance(k, tuple) and k[0] == "sbm" for k in kinds_p), kinds_p
    fin = torch.isfinite(pf_t) & torch.isfinite(pf_p)
    assert torch.equal(torch.isfinite(pf_t), torch.isfinite(pf_p)), f"{name}: the parameter flows' finite pattern"
    d = float((pf_t[fin] - pf_p[fin]).abs().max() / pf_t[fin].abs().max())
    assert d < 3e-3, f"{name}: the parameter flows differ by {d:.2e} (relative)"
    for ns in g_t:
        nan = torch.isnan(g_t[ns])
        assert torch.equal(nan, torch.isnan(g_p[ns])), f"{name}: the gate gradient's NaN pattern differs"
        scale = float(g_t[ns][~nan].abs().max())
        d = float((g_t[ns] - g_p[ns])[~nan].abs().max()) / max(scale, 1e-30)
        assert d < 2e-2, f"{name}: the gate gradient differs by {d:.2e} (relative)"

    # the same backward again (the plan kept): the same gradient bits
    grads2 = _backward(pc, phi, x)
    for ns in g_p:
        assert torch.equal(torch.nan_to_num(g_p[ns], nan = 7.0), torch.nan_to_num(grads2[ns], nan = 7.0)), \
            f"{name}: the gate gradient differs between two equal backwards"


@needs_kernels
@pytest.mark.parametrize("label,builder,B", [
    ("batch 64", lambda: _hmm(256, 4, 16), 64),
    ("parent blocks of 32", lambda: _topology(PAD, 32, 64, 8), 32),      # a 64-parent tile spans two blocks
    ("child blocks of 32", lambda: _topology(PAD, 64, 32, 8), 32),       # under the smallest child tile
])
def test_it_declines_what_it_does_not_serve(label, builder, B, monkeypatch):
    pc, phi, x = _instance(builder, "bf16", B, False, False)
    autotune, seen = _pick("sbm")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    plans = _ele_plans(pc)
    assert plans, f"{label}: no element plan was built"
    assert not any(isinstance(k, tuple) and k[0] == "sbm" for k in seen), f"{label}: {seen}"
    assert not any(isinstance(p["kind"], tuple) and p["kind"][0] == "sbm" for p in plans), f"{label}"
