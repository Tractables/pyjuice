"""
The pre-pass CUDA element-flow backward (c/ele_backward_sum_prepass.cu: the parents' log ratios' maxima and
exponentials once per element block, then a streaming kernel over the parameter tiles) against the CUDA element-flow
kernel it reproduces -- bit-identical at every config valid for the partition, at fp32 and bf16 parameters, on HCLT-128
and HMMs of 128 and 896 latents (2 and 14 parent tiles: fewer than the stages, and refills ending part-way through a
round), at B 64 and 128; on columns whose parents are dead (-inf node values) in an element block's first parent tile,
have no flow (-inf) in its last, or are dead in all of them; and the layer's dispatch, told the gate chose it, reaches
it on a whole backward and on runs, and offers its configs beside the other kernels'. Skipped where the CUDA kernels
cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.utils.launch_plans as lp
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as ck

DEV = torch.device("cuda:0")
NAN = float("nan")
NINF = -float("inf")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _hclt(latents, dt, cats = 16, V = 64, block_size = None):
    torch.manual_seed(0)
    x = torch.randint(0, cats, (512, V))
    ns = juice.structures.HCLT(x.float(), num_latents = latents, input_dist = dists.Categorical(num_cats = cats),
                               **({} if block_size is None else dict(block_size = block_size)))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dt).to(DEV), V, cats


def _hmm(latents, dt, T = 8):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = latents, num_emits = 16)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dt).to(DEV), T, 16


CIRCUITS = {"hclt128": lambda dt: _hclt(128, dt), "hmm128": lambda dt: _hmm(128, dt),
            "hmm896": lambda dt: _hmm(896, dt, T = 4)}


def _available():
    return ck.ele_is_available() and ck.ele_prepass_is_available()


def _state(name, dts, B, monkeypatch):
    pc, V, C = CIRCUITS[name](DTYPES[dts])
    x = torch.randint(0, C, (B, V), device = DEV)
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", False)
    pc(x)
    pc.backward(x, compute_param_flows = False)                     # the backward's caches, Triton throughout
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", True)
    return pc


def _partitions(pc, B):
    """(layer, chids, cs, ebase, pbase, rows) of every partition the CUDA element-flow kernels take at batch B."""
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            for sig, (ebase, pbase, ok) in list(layer._cached_bk_ele_cuda.items()):
                if not ok or sig[2] != 64:
                    continue
                chids, cs = layer.partitioned_chids[sig[1]], int(layer.cs_block_sizes[sig[1]])
                if (cs % 128 or B % 64
                        or not layer._cuda_param_loads_aligned(pc.params, ("ele", sig), pbase, layer.block_size)):
                    continue
                rows = (chids[:, None] + torch.arange(cs, device = DEV)[None, :]).reshape(-1)
                yield layer, chids, cs, ebase, pbase, rows


def _plain(pc, layer, chids, cs, ebase, pbase, rows, B):
    ef = torch.full_like(pc.element_flows, NAN)
    ck.ele_backward_sum(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, ebase, pbase, B,
                        layer.block_size, cs, ebase.size(1))
    return ef[rows].clone()


def _check_configs(pc, layer, chids, cs, ebase, pbase, rows, B, want, label):
    tiles, tix, shared = ck.tile_index(ebase)
    for cfg in ck.ele_prepass_valid_configs(cs, B):
        ef = torch.full_like(pc.element_flows, NAN)
        ck.ele_backward_sum_prepass(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, tiles, tix,
                                    pbase, B, layer.block_size, cs, ebase.size(1), cfg, shared)
        assert torch.equal(ef[rows], want), (label, cfg)


@pytest.mark.parametrize("dts", list(DTYPES))
@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_prepass_element_backward_is_the_cuda_backward_to_the_bit(name, B, dts, monkeypatch):
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    pc = _state(name, dts, B, monkeypatch)
    checked = 0
    for layer, chids, cs, ebase, pbase, rows in _partitions(pc, B):
        want = _plain(pc, layer, chids, cs, ebase, pbase, rows, B)
        _check_configs(pc, layer, chids, cs, ebase, pbase, rows, B, want, (name, dts))
        checked += 1
    assert checked > 0, "no partition of the circuit takes the CUDA path"


@pytest.mark.parametrize("dts", list(DTYPES))
@pytest.mark.parametrize("name", ["hclt128", "hmm896"])
def test_the_prepass_element_backward_on_dead_columns(name, dts, monkeypatch):
    """Parents dead (-inf node values) in an element block's first parent tile (the merge's alpha = 0 from then on),
    without flow (-inf) in its last (beta = 0 there), or dead in every tile (-inf flows), against the plain kernel to
    the bit."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    B = 64
    pc = _state(name, dts, B, monkeypatch)
    checked = 0
    for layer, chids, cs, ebase, pbase, rows in _partitions(pc, B):
        saved_m, saved_f = pc.node_mars.clone(), pc.node_flows.clone()
        knt = ebase.size(1)
        tile_rows = lambda kt: (ebase[:, kt, None] + torch.arange(64, device = DEV)[None, :]).reshape(-1)
        pc.node_mars[tile_rows(0), 3] = NINF
        pc.node_flows[tile_rows(knt - 1), 5] = NINF
        for kt in range(knt):
            pc.node_mars[tile_rows(kt), 7] = NINF
        pc.node_flows[tile_rows(0)[::3], 9] = NINF                     # some parents of a tile: an ordinary column
        want = _plain(pc, layer, chids, cs, ebase, pbase, rows, B)
        assert (want[:, 7] == NINF).all()
        _check_configs(pc, layer, chids, cs, ebase, pbase, rows, B, want, (name, dts, "dead columns"))
        pc.node_mars.copy_(saved_m); pc.node_flows.copy_(saved_f)
        checked += 1
    assert checked > 0, "no partition of the circuit takes the CUDA path"


def _float64_block(pc, bs, cs, chid, parids_k, parpids_k):
    """Element block `k`'s flows in float64 from the layer's COMPILED tables (`partitioned_parids` / `parpids`: a parent
    block's first node row and first parameter per child block, padding 0 -- not the kernels' derived tables, which a
    wrong derivation would feed to both sides): element_mars + log sum_e params[m, e] exp(lf[e, b])."""
    real = parids_k != 0
    ar = torch.arange(bs, device = DEV)
    e_rows = (parids_k[real][:, None] + ar).reshape(-1)                           # the parents
    nf, nm = pc.node_flows.double()[e_rows], pc.node_mars.double()[e_rows]
    lf = torch.where(nm == NINF, torch.full_like(nm, NINF), nf - nm)              # [E, B]
    pidx = (parpids_k[real][:, None] + ar).reshape(-1)                            # element 0's parameter per parent
    par = pc.params.double()[pidx[None, :] + torch.arange(cs, device = DEV)[:, None] * bs]     # [cs, E]
    rows = chid + torch.arange(cs, device = DEV)
    return pc.element_mars.double()[rows] + torch.logsumexp(par.log()[:, :, None] + lf[None], dim = 1)   # [cs, B]


@pytest.mark.parametrize("dts", list(DTYPES))
@pytest.mark.parametrize("B", [64, 128])
def test_the_prepass_element_backward_on_32_node_blocks(B, dts, monkeypatch):
    """Partitions of 32-node child blocks under 32-node parent blocks (an HCLT's), which no other CUDA kernel takes: a
    parent tile spans two parameter blocks; the 32-row configs against each other to the bit and against float64; the
    pre-pass over the DISTINCT parent tiles (a variable's 8 child blocks share theirs); the dispatch, told the gate chose
    one of them, reaching it."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    pc, V, C = _hclt(256, DTYPES[dts], block_size = 32)                # 8 blocks a variable: they share the pre-pass
    x = torch.randint(0, C, (B, V), device = DEV)
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", False)
    pc(x)
    pc.backward(x, compute_param_flows = False)                     # the backward's caches, Triton throughout
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", True)
    checked = 0
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            for sig, (ebase, pbase, plain_ok) in list(layer._cached_bk_ele_cuda.items()):
                pbase_full, pre_ok = layer._cached_bk_ele_pre[sig]
                chids, cs = layer.partitioned_chids[sig[1]], int(layer.cs_block_sizes[sig[1]])
                if sig[2] != 64 or cs != 32 or not pre_ok:
                    continue
                assert not plain_ok and layer.block_size == 32 and pbase_full.size(1) == 2 * ebase.size(1)
                assert layer._cuda_param_loads_aligned(pc.params, ("ele", sig), pbase_full, layer.block_size)
                tiles, tix, shared = ck.tile_index(ebase)
                assert shared and tiles.numel() * 8 == ebase.numel() and torch.equal(tiles[tix.long()], ebase), "8 share"
                rows = (chids[:, None] + torch.arange(cs, device = DEV)[None, :]).reshape(-1)
                outs = []
                for cfg in ck.ele_prepass_valid_configs(cs, B):
                    ef = torch.full_like(pc.element_flows, NAN)
                    ck.ele_backward_sum_prepass(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, tiles,
                                                tix, pbase_full, B, layer.block_size, cs, ebase.size(1), cfg, shared)
                    outs.append(ef[rows].clone())
                assert outs and all(torch.equal(o, outs[0]) for o in outs[1:]), (dts, B, "the configs differ")
                parids, parpids = layer.partitioned_parids[sig[1]], layer.partitioned_parpids[sig[1]]
                for k in range(chids.numel()):
                    ref = _float64_block(pc, layer.block_size, cs, int(chids[k]), parids[k].long(), parpids[k].long())
                    got = outs[0][k * cs:(k + 1) * cs].double()
                    fin = torch.isfinite(ref)
                    assert torch.equal(fin, torch.isfinite(got)), (dts, B, k, "finite pattern")
                    assert float((got[fin] - ref[fin]).abs().max()) < 1e-2, (dts, B, k, "against float64")
                cfg0 = ck.ele_prepass_valid_configs(cs, B)[0]
                with monkeypatch.context() as mp:
                    mp.setattr(autotune, "best_of", lambda cands, *a, **k: ("cuda_prepass", cfg0))
                    layer._cached_bk_ele_choice.clear()
                    ef = torch.full_like(pc.element_flows, NAN)
                    layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, logspace_flows = True)
                    assert torch.equal(ef[rows], outs[0])
                layer._cached_bk_ele_choice.clear()
                checked += 1
    assert checked > 0, "no partition of 32-node blocks takes the pre-pass kernel"


@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_dispatch_reaches_the_prepass_element_backward(name, monkeypatch):
    """The layer's backward, told the gate chose the pre-pass kernel, over all its partitions against the plain CUDA
    kernel's, whole and on runs of partition 0; and the gate's candidates hold its configs (none where its extension
    is not available)."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    B = 64
    pc = _state(name, "bf16", B, monkeypatch)
    monkeypatch.setattr(lp, "ENABLED", False)          # (every call below must take the dispatch, not replay a plan)
    checked = 0
    kw = dict(logspace_flows = True)
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            parts = list(p for p in _partitions(pc, B) if p[0] is layer)
            if not parts:
                continue
            cs0 = int(layer.cs_block_sizes[0])
            cfg0 = ck.ele_prepass_valid_configs(cs0, B)[0]
            with monkeypatch.context() as mp:
                mp.setattr(autotune, "best_of", lambda cands, *a, **k: ("cuda", 0))
                layer._cached_bk_ele_choice.clear()
                ef = torch.full_like(pc.element_flows, NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                want = ef.clone()
                mp.setattr(autotune, "best_of", lambda cands, *a, **k: ("cuda_prepass", cfg0))
                layer._cached_bk_ele_choice.clear()
                ef.fill_(NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                assert torch.equal(torch.nan_to_num(ef, nan = 7.0), torch.nan_to_num(want, nan = 7.0)), name
                n = layer.partitioned_chids[0].numel()
                ef.fill_(NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params,
                               partition_runs = [[(0, 1), (n // 2, n)]] + [[]] * (layer.num_bk_partitions - 1), **kw)
                rows = torch.cat([layer.partitioned_chids[0][s:e] for s, e in ((0, 1), (n // 2, n))])
                rows = (rows[:, None] + torch.arange(cs0, device = DEV)[None, :]).reshape(-1)
                assert torch.equal(ef[rows], want[rows])
                available = ck.ele_prepass_is_available
                for on in (True, False):
                    asked = []
                    mp.setattr(ck, "ele_prepass_is_available", available if on else (lambda: False))
                    mp.setattr(autotune, "best_of",
                               lambda cands, *a, **k: asked.append([c for c, _ in cands]) or ("cuda", 0))
                    layer._cached_bk_ele_choice.clear()
                    layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                    offered = [c for cands in asked for c in cands if c[0] == "cuda_prepass"]
                    want_offered = [("cuda_prepass", c) for c in ck.ele_prepass_valid_configs(cs0, B)] if on else []
                    assert offered[:len(want_offered)] == want_offered and (on or offered == [])
                layer._cached_bk_ele_choice.clear()
            checked += 1
    assert checked > 0, "no layer of the circuit takes the CUDA path"
