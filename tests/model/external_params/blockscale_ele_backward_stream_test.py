"""
The split block-scale element-flow backward (kernels/c/blockscale_ele_backward_stream.cu: a pre-pass, a split GEMM and
a merge) against float64 -- the element flows and d LL / d log phi at every fitting (tile, split), the gradient on and
off, on each gated layer's real inputs captured from a backward (bf16 parameters): gated HMMs of one k-tile, 7 (blocks
of 64 rows), 16 and 64, of one to four blocks; gate widths 4 (the merge's 4-child groups) to 512 (past the merge's
64-child chunk, the whole element block); log-gates -inf in places and in whole columns, children -inf; padded (phantom
k-tiles), block-sparse and two-node layers; ranges cut inside a parent block (pieces) and across several. Nothing
outside the element blocks' rows written; the same bits run to run. The piece tables against a direct construction.
Then the plan: it offers the split backward at bf16 and a backward taking it gives the Triton fork's parameter flows and
gate gradients within the forks' rounding, tied gates too (one gate tensor, its gradient rows added by every layer); it
is never offered at fp32 or where it does not fit. Skipped where the CUDA kernels cannot build.
"""
import os
import sys

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.c as ck
from pyjuice.utils.launch_plans import LaunchPlans
from pyjuice.nodes.external_params.blockscale.backward import _BackwardMixin

# (a sibling module: the pre-pass fork's circuits, capture helpers and float64 reference)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockscale_ele_backward_prepass_test import (_hmm, _topology, _two, _instance, _reference, _pick,  # noqa: E402
                                                  _gated_layers, _ele_plans, _backward, PAD, PAD_BIG, SPARSE, NAN,
                                                  NEG, CATS)


def _module():
    if not torch.cuda.is_available():
        return None
    from pyjuice.nodes.external_params.kernels.c import get_ele_stream_module
    try:
        return get_ele_stream_module()
    except Exception:
        return None


needs_kernels = pytest.mark.skipif(_module() is None,
                                   reason = "needs the split element backward (nvcc + CUTLASS + sm_80+)")

SPLITS = (1, 2, 3, 4, 5, 6, 8)


def _tied_hmm(L, nb, gcb, T = 4):
    """A gated HMM whose transitions and gates are all the first's (`tie_external`): one gate tensor, its gradient
    rows written by every layer."""
    with juice.set_block_size(L // nb):
        ns = inputs(0, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS))
        src = None
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = nb, dist = dists.Categorical(num_cats = CATS))
            if src is None:
                ns = src = summate(multiply(ns, emit), num_node_blocks = nb,
                                   external_params = BlockScaleSumParams(ch_block_size = gcb, tie_external = True))
            else:
                ns = src.duplicate(multiply(ns, emit), tie_params = True)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    return root, [src], T

# (name, builder, batch, -inf log-gates, -inf children); bf16 parameters throughout
CASES = [
    ("hmm64_g16", lambda: _hmm(64, 1, 16), 128, False, False),                # one k-tile, element blocks of 64
    ("hmm256_g64", lambda: _hmm(256, 1, 64), 64, False, False),
    ("hmm448x7_g64", lambda: _hmm(448, 7, 64), 128, True, False),             # 7 parent blocks of 64: 7 runs
    ("hmm1024_g4", lambda: _hmm(1024, 1, 4), 128, True, True),                # 4-child groups
    ("hmm1024_g16", lambda: _hmm(1024, 1, 16), 192, False, True),             # 16 k-tiles, one run
    ("hmm512x2_g32", lambda: _hmm(512, 2, 32), 128, True, False),
    ("hmm1024x4_g128", lambda: _hmm(1024, 4, 128), 128, True, True),          # gates of two 64-child chunks
    ("hmm2048x4_g64", lambda: _hmm(2048, 4, 64, T = 3), 256, True, False),    # 32 k-tiles, 4 runs
    ("hmm4096x4_g8", lambda: _hmm(4096, 4, 8, T = 3), 128, True, False),      # 64 k-tiles
    ("hmm512_g512", lambda: _hmm(512, 1, 512), 64, True, False),              # a gate the whole element block
    ("pad", lambda: _topology(PAD, 128, 128, 8), 128, True, False),           # phantom k-tiles
    ("pad_big", lambda: _topology(PAD_BIG, 128, 128, 16, n_ch = 8), 128, True, False),
    ("sparse", lambda: _topology(SPARSE, 128, 128, 8), 128, True, False),
    ("two_nodes", lambda: _two(8, bs = 128), 128, True, False),
]


def _capture(pc, phi, x, monkeypatch):
    """One backward through the plan with the split backward preferred; each layer's inputs as `_eles_code` saw them
    (the gate table among them), cloned."""
    captured = []
    real = _BackwardMixin._eles_code

    def record(self, eles_mod, ctx, gate_tile, scr, ext, ext_base, gate_cbs, fit):
        captured.append(dict(
            em = ctx["element_mars"].clone(), nf = ctx["node_flows"].clone(), nm = ctx["node_mars"].clone(),
            prm = ctx["params"].clone(), ext = ext.clone(), chids = ctx["chids"], ebase = ctx["ele_ebase"],
            pbase = ctx["ele_pbase"], gate = gate_tile, sizes = (ctx["batch_size"], ctx["block_size"],
                                                                 ctx["cs_block_size"], ctx["K_NUM_TILES"],
                                                                 gate_cbs, ext_base), fit = list(fit)))
        return real(self, eles_mod, ctx, gate_tile, scr, ext, ext_base, gate_cbs, fit)

    monkeypatch.setattr(_BackwardMixin, "_eles_code", record)
    autotune, _ = _pick("eles")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    return captured


def _launch(mod, c, ef, g, cfg, split, tb, write_grad):
    batch, blk, cbs, knt, gc, ext_base = c["sizes"]
    mod.blockscale_ele_backward_stream(ef, c["em"], c["nf"], c["nm"], c["prm"], c["ext"], c["chids"], c["ebase"],
                                       c["pbase"], g, batch, blk, cbs, knt, gc, ext_base, write_grad,
                                       cfg * 64 + split, tb["tables"], tb["ngrp"], tb["npc"])


@needs_kernels
@pytest.mark.parametrize("name,builder,B,neg_gates,neg_children", CASES, ids = [c[0] for c in CASES])
def test_every_tile_and_split_matches_float64(name, builder, B, neg_gates, neg_children, monkeypatch):
    mod = _module()
    pc, phi, x = _instance(builder, "bf16", B, neg_gates, neg_children)
    captured = _capture(pc, phi, x, monkeypatch)
    assert captured, f"{name}: the split backward served no layer"
    cfgs = [tuple(int(v) for v in c) for c in mod.configs()]
    pmax = int(mod.max_pieces())

    checked = 0
    for c in captured:
        batch, blk, cbs, knt, gc, ext_base = c["sizes"]
        ref_ef, ref_g = _reference(c["em"], c["nf"], c["nm"], c["prm"], c["ext"], c["chids"], c["ebase"], c["pbase"],
                                   c["gate"], batch, blk, cbs, knt, gc, ext_base)
        rows = ~torch.isnan(ref_ef).all(1)
        assert c["fit"] == [int(i) for i in mod.fitting_configs(cbs, batch, gc)]
        for cfg in c["fit"]:
            for split in SPLITS:
                tb = _BackwardMixin._eles_tables(c["gate"], c["ebase"], cfgs[cfg][2], split, pmax)
                if tb is None:
                    continue
                for write_grad in (1, 0):
                    ef = torch.full_like(c["em"], NAN)
                    g = torch.zeros_like(c["ext"]) if write_grad else c["ext"].new_empty(0)
                    _launch(mod, c, ef, g, cfg, split, tb, write_grad)
                    torch.cuda.synchronize()
                    what = (f"{name}: config {cfg} {cfgs[cfg]} split {split} ({tb['npc']} pieces, "
                            f"{tb['grep'].numel()} groups; gradient {'on' if write_grad else 'off'})")
                    assert torch.isnan(ef[~rows]).all(), f"{what}: rows outside the element blocks written"
                    r, y = ref_ef[rows], ef[rows].double()
                    known = ~torch.isnan(r)
                    assert not torch.isnan(y[known]).any(), f"{what}: NaN element flows"
                    assert not (y[known] == float("inf")).any(), f"{what}: +inf element flows"
                    assert torch.equal(r[known] == NEG, y[known] == NEG), f"{what}: the -inf pattern differs"
                    fin = known & torch.isfinite(r)
                    assert bool(fin.any()), f"{what}: no finite element flow to compare"
                    d = float((r[fin] - y[fin]).abs().max())
                    assert d < 1e-4, f"{what}: element flows off by {d:.2e}"
                    if write_grad:
                        assert torch.isfinite(g).all(), f"{what}: a gradient entry not finite"
                        scale = float(ref_g.abs().max())
                        assert scale > 0, f"{what}: the reference gradient is zero: nothing compared"
                        d = float((g.double().view(-1) - ref_g).abs().max()) / scale
                        assert d < 1e-4, f"{what}: the gradient off by {d:.2e} (relative)"
                        ef2, g2 = torch.full_like(c["em"], NAN), torch.zeros_like(c["ext"])
                        _launch(mod, c, ef2, g2, cfg, split, tb, 1)
                        torch.cuda.synchronize()
                        bits = lambda t: t.view(torch.int32)
                        assert torch.equal(bits(ef2), bits(ef)) and torch.equal(bits(g2), bits(g)), \
                            f"{what}: bits differ run to run"
                    checked += 1
    assert checked > 0


def _direct_tables(gate, kt, split):
    """The pieces by walking an element block's gated k-steps one at a time (eles_check.py's construction)."""
    sub = 64 // kt
    out = []
    for row in gate.cpu().tolist():
        steps = [(t * sub + s, row[t]) for t in range(len(row)) if row[t] >= 0 for s in range(sub)]
        n, pieces, rng = len(steps), [], []
        for q in range(split):
            a, b = q * n // split, (q + 1) * n // split
            lo, i = len(pieces), a
            while i < b:
                j = i
                while j + 1 < b and steps[j + 1][1] == steps[i][1] and steps[j + 1][0] == steps[j][0] + 1:
                    j += 1
                pieces.append((steps[i][0], steps[j][0] + 1, steps[i][1]))
                i = j + 1
            rng.append((lo, len(pieces)))
        out.append((pieces, rng))
    return out


@pytest.mark.parametrize("kt", [64, 32])
@pytest.mark.parametrize("split", SPLITS)
def test_the_piece_tables_match_a_direct_construction(kt, split):
    """Runs of one gate, a gap (an ungated k-tile) between two runs of one gate, an element block without gates, an
    element block of one k-tile, equal element blocks (one group), equal parents but other pieces (another)."""
    gate = torch.tensor([[5, 5, 5, 9, 9, -1, 9, 9, 2, 2],
                         [5, 5, 5, 9, 9, -1, 9, 9, 2, 2],
                         [-1] * 10,
                         [7] + [-1] * 9,
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
                         [5, 5, 5, 9, 9, 9, 9, 9, 2, 2]])
    ebase = torch.arange(gate.numel()).view(gate.shape) * 64
    ebase[1] = ebase[0]                                       # element blocks 0 and 1: equal parents and pieces
    ebase[6] = ebase[0]                                       # 6: block 0's parents, its tile 5 gated
    pmax = 16
    tb = _BackwardMixin._eles_tables(gate, ebase, kt, split, pmax)
    direct = _direct_tables(gate, kt, split)
    if max(len(p) for p, _ in direct) > pmax:
        assert tb is None
        return
    assert tb is not None
    assert tb["npc"] == max(1, max(len(p) for p, _ in direct))
    for eb, (pieces, rng) in enumerate(direct):
        got = [(int(tb["pc_k0"][eb, p]), int(tb["pc_k1"][eb, p]), int(tb["pc_gate"][eb, p])) for p in range(pmax)]
        assert got[:len(pieces)] == pieces, (eb, got, pieces)
        assert all(x == (0, 0, -1) for x in got[len(pieces):]), (eb, got)
        assert [(int(tb["sp_lo"][eb, q]), int(tb["sp_hi"][eb, q])) for q in range(split)] == rng, eb
    ugrp = tb["ugrp"].tolist()
    assert ugrp[0] == ugrp[1] and len(set(ugrp[1:])) == len(ugrp) - 1, ugrp
    assert [ugrp[e] for e in tb["grep"].tolist()] == list(range(len(tb["grep"])))


# --------------------------------------------------------------------------------- the plan

# (no samples whose gates are all -inf: their flows and gradients are NaN in either fork and would compare nothing)
PLAN_CASES = [
    ("hmm4096x4_g8", lambda: _hmm(4096, 4, 8, T = 3), 128, False, False),
    ("hmm1024x4_g128", lambda: _hmm(1024, 4, 128), 128, False, True),
    ("pad", lambda: _topology(PAD, 128, 128, 16), 128, False, True),
    ("two_nodes", lambda: _two(8, bs = 128), 128, False, False),
    ("tied_hmm1024x4_g16", lambda: _tied_hmm(1024, 4, 16), 128, False, False),
]


@needs_kernels
@pytest.mark.parametrize("name,builder,B,neg_gates,neg_children", PLAN_CASES, ids = [c[0] for c in PLAN_CASES])
def test_a_backward_taking_it_matches_triton(name, builder, B, neg_gates, neg_children, monkeypatch):
    """The plan offers the split backward at bf16; a backward taking it gives the Triton fork's parameter flows (where
    they depend on the element flows: the HMMs' lower layers) and gate gradients within the forks' rounding -- the
    bar of blockscale_ele_backward_prepass_test.py's same test (Triton's TF32 dots dominate it)."""
    pc, phi, x = _instance(builder, "bf16", B, neg_gates, neg_children)

    def run(prefer):
        for layer in _gated_layers(pc):
            for k in [k for k in layer._bs_bw_gate_cache if isinstance(k, tuple) and k[0] == "eleplan"]:
                del layer._bs_bw_gate_cache[k]
            LaunchPlans.of(layer).clear()                  # (its plans replay the choice dropped above)
        autotune, seen = _pick(prefer)
        monkeypatch.setattr(ck, "autotune", autotune)
        grads = _backward(pc, phi, x)
        return (pc.param_flows.clone(), {ns: g.clone() for ns, g in grads.items()}, {p["kind"] for p in _ele_plans(pc)},
                seen)

    pf_t, g_t, kinds_t, _ = run("triton")
    pf_s, g_s, kinds_s, seen = run("eles")
    assert kinds_t == {"triton"}, kinds_t
    assert kinds_s and all(isinstance(k, tuple) and k[0] == "eles" for k in kinds_s), kinds_s
    assert any(isinstance(k, tuple) and k[0] == "eles" for k in seen), seen

    fin = torch.isfinite(pf_t) & torch.isfinite(pf_s)
    assert torch.equal(torch.isfinite(pf_t), torch.isfinite(pf_s)), f"{name}: the parameter flows' finite pattern"
    d = float((pf_t[fin] - pf_s[fin]).abs().max() / pf_t[fin].abs().max())
    assert d < 2e-3, f"{name}: the parameter flows differ by {d:.2e} (relative)"
    for ns in g_t:
        nan = torch.isnan(g_t[ns])
        assert torch.equal(nan, torch.isnan(g_s[ns])), f"{name}: the gate gradient's NaN pattern differs"
        scale = float(g_t[ns][~nan].abs().max())
        d = float((g_t[ns] - g_s[ns])[~nan].abs().max()) / max(scale, 1e-30)
        assert d < 2e-2, f"{name}: the gate gradient differs by {d:.2e} (relative)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_the_plans_trials_charge_a_candidates_extra_host_time():
    """Two candidates with the same GPU work (~150 us) and host work hidden behind it, one 80 us slower on the host: one
    call's event time cannot tell them apart; charged the host time it costs beyond the other's, the slower loses."""
    import time
    from pyjuice.nodes.external_params.blockscale.backward import _host_extra, _charged

    def make(spin):
        def run():
            torch.cuda._sleep(400000)
            end = time.perf_counter() + spin
            while time.perf_counter() < end:
                pass
        return run

    runs = {"slow_host": make(80e-6), "fast_host": make(0.0)}
    extra = _host_extra(runs)
    assert extra["fast_host"] == 0.0 and extra["slow_host"] > 50e-6, extra
    assert ck.autotune([(k, _charged(runs[k], extra[k])) for k in runs]) == "fast_host"


@needs_kernels
@pytest.mark.parametrize("label,builder,dt,B", [
    ("fp32 parameters", lambda: _hmm(1024, 1, 16), "fp32", 128),
    ("gate width 2", lambda: _hmm(256, 1, 2), "bf16", 128),
    ("batch 32", lambda: _hmm(256, 1, 16), "bf16", 32),
    ("element blocks of 32", lambda: _topology(PAD, 64, 32, 8), "bf16", 128),
    ("parent blocks of 32", lambda: _topology(PAD, 32, 64, 8), "bf16", 128),   # a k-tile of 64 parents: two blocks
])
def test_it_declines_what_it_does_not_serve(label, builder, dt, B, monkeypatch):
    pc, phi, x = _instance(builder, dt, B, False, False)
    autotune, seen = _pick("eles")
    monkeypatch.setattr(ck, "autotune", autotune)
    _backward(pc, phi, x)
    plans = _ele_plans(pc)
    assert plans, f"{label}: no element plan was built"
    assert not any(isinstance(k, tuple) and k[0] == "eles" for k in seen), f"{label}: {seen}"
    assert not any(isinstance(p["kind"], tuple) and p["kind"][0] == "eles" for p in plans), f"{label}"
