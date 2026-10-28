"""
The pre-pass CUDA tlmm forward (c/tlmm_forward_sum_prepass.cu: the element tiles' maxima and exponentials once per
node block, then a streaming kernel over the parameter tiles) against the CUDA tlmm kernel it reproduces --
bit-identical at every tile config valid for the layer, at fp32 and bf16 parameters, on HCLT-128 and HMMs of 64 latents
(one edge tile), 128 (two, fewer than the stages), 448 and 896 (7 and 14), at B 64 and 128; on columns -inf in a node
block's first edge tile, in its last, or in all of them (the merge's three cases); and the layer's dispatch, told the
gate chose it, reaches it on a whole launch and on runs, and offers its configs beside the other kernels'. Skipped where
the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as ck

DEV = torch.device("cuda:0")
NAN = float("nan")
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


# edge tiles a node block: HCLT-128 2, HMM-64 1, HMM-128 2 (fewer than the stages), HMM-448 7 (blocks of 64), HMM-896
# 14 (blocks of 128): the last two refill every stage and end part-way through a round of them
CIRCUITS = {"hclt128": lambda dt: _hclt(128, dt), "hmm64": lambda dt: _hmm(64, dt, T = 4),
            "hmm128": lambda dt: _hmm(128, dt), "hmm448": lambda dt: _hmm(448, dt, T = 4),
            "hmm896": lambda dt: _hmm(896, dt, T = 4)}


def _available():
    return ck.is_available() and ck.pipe_is_available() and ck.prepass_is_available()


def _cuda_layers(pc, B):
    """(layer, nids, ebase, pbase) of every sum layer the CUDA tlmm kernels take at batch B (aligned param loads)."""
    sig = ("block_sparse", 0, 64)
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            if sig not in layer._cached_fw_cuda or not layer._cached_fw_cuda[sig][2]:
                continue
            ebase, pbase, _ = layer._cached_fw_cuda[sig]
            if not layer._cuda_param_loads_aligned(pc.params, ("fw", sig), pbase, layer.block_size):
                continue
            if not ck.prepass_valid_configs(layer.block_size, B):
                continue
            yield layer, layer.partitioned_nids[0], ebase, pbase


def _reference(pc, layer, nids, ebase, pbase, B):
    """The layer's rows as the plain CUDA kernel writes them (the pipelined copy where no plain config fits)."""
    a, b = layer.nid_range
    bs, knt = layer.block_size, ebase.size(1)
    pc.node_mars[a:b] = NAN
    plain = ck.valid_configs(bs, B)
    if plain:
        ck.tlmm_forward_sum(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, B, bs, knt, plain[0])
    else:
        ck.tlmm_forward_sum_pipelined(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, B, bs, knt,
                                      ck.pipe_valid_configs(bs, B)[0])
    return pc.node_mars[a:b].clone()


def _check_configs(pc, layer, nids, ebase, pbase, B, want, label):
    a, b = layer.nid_range
    bs, knt = layer.block_size, ebase.size(1)
    tiles, tix, shared = ck.tile_index(ebase)
    for cfg in ck.prepass_valid_configs(bs, B):
        pc.node_mars[a:b] = NAN
        ck.tlmm_forward_sum_prepass(pc.node_mars, pc.element_mars, pc.params, nids, tiles, tix, pbase, B, bs, knt, cfg,
                                    shared)
        assert torch.equal(pc.node_mars[a:b], want), (label, cfg)


@pytest.mark.parametrize("dts", list(DTYPES))
@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_prepass_forward_is_the_cuda_forward_to_the_bit(name, B, dts, monkeypatch):
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS[name](DTYPES[dts])
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for layer, nids, ebase, pbase in _cuda_layers(pc, B):
        want = _reference(pc, layer, nids, ebase, pbase, B)
        _check_configs(pc, layer, nids, ebase, pbase, B, want, (name, dts))
        checked += 1
        # the dispatch, told the gate chose the first config: a whole launch and runs (the stub stands only around
        # the two calls -- every tuned launch goes through `best_of` -- and its choices leave the cache with it)
        a, b = layer.nid_range
        n, bs = nids.numel(), layer.block_size
        cfg0 = ck.prepass_valid_configs(bs, B)[0]
        was = autotune.set_autotune(True, clear_cache = True)
        try:
            with monkeypatch.context() as mp:
                mp.setattr(SL, "FORWARD_SUM_CUDA", True)
                mp.setattr(autotune, "best_of", lambda cands, *args, **kw: ("cuda_prepass", cfg0))
                pc.node_mars[a:b] = NAN
                layer.forward(pc.node_mars, pc.element_mars, pc.params)
                assert torch.equal(pc.node_mars[a:b], want)
                pc.node_mars[a:b] = NAN
                runs = [(0, 1), (n // 2, n)]
                layer.forward(pc.node_mars, pc.element_mars, pc.params,
                              partition_runs = [runs] + [[]] * (layer.num_fw_partitions - 1))
                rows = torch.zeros(b - a, dtype = torch.bool, device = DEV)
                for s, e in runs:
                    for r in nids[s:e].tolist():
                        rows[r - a:r - a + bs] = True
                assert torch.equal(pc.node_mars[a:b][rows], want[rows]) and torch.isnan(pc.node_mars[a:b][~rows]).all()
        finally:
            autotune.set_autotune(was, clear_cache = True)
        pc.node_mars[a:b] = want
    assert checked > 0, "no layer of the circuit takes the CUDA path"


@pytest.mark.parametrize("dts", list(DTYPES))
@pytest.mark.parametrize("name", ["hclt128", "hmm128", "hmm896"])
def test_the_prepass_forward_on_dead_columns(name, dts, monkeypatch):
    """Columns whose children are all -inf in a node block's first edge tile (the merge's alpha = 0 from then on), in
    its last (beta = 0 there), or in every tile (-inf out), against the plain kernel to the bit."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS[name](DTYPES[dts])
    B = 64
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for layer, nids, ebase, pbase in _cuda_layers(pc, B):
        saved = pc.element_mars.clone()
        tile_rows = lambda kt: (ebase[:, kt, None] + torch.arange(64, device = DEV)[None, :]).reshape(-1)
        knt = ebase.size(1)
        pc.element_mars[tile_rows(0), 3] = -float("inf")
        pc.element_mars[tile_rows(knt - 1), 5] = -float("inf")
        for kt in range(knt):
            pc.element_mars[tile_rows(kt), 7] = -float("inf")
        pc.element_mars[tile_rows(0)[::3], 9] = -float("inf")          # some rows of a tile: an ordinary column
        want = _reference(pc, layer, nids, ebase, pbase, B)
        assert torch.isinf(want[:, 7]).all() and (want[:, 7] < 0).all()
        if knt > 1:
            assert torch.isfinite(want[:, 3]).all() and torch.isfinite(want[:, 5]).all()
        _check_configs(pc, layer, nids, ebase, pbase, B, want, (name, dts, "dead columns"))
        pc.element_mars.copy_(saved)
        checked += 1
    assert checked > 0, "no layer of the circuit takes the CUDA path"


@pytest.mark.parametrize("B", [64, 128])
def test_the_gate_offers_the_prepass_configs(B, monkeypatch):
    """The CUDA gate's candidates hold the pre-pass kernel's configs beside the plain and pipelined ones and Triton, and
    none where its extension is not available."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    pc, V, C = _hclt(128, torch.float32)
    x = torch.randint(0, C, (B, V), device = DEV)
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", True)
    available = ck.prepass_is_available
    for on in (True, False):
        monkeypatch.setattr(ck, "prepass_is_available", available if on else (lambda: False))
        monkeypatch.setattr(autotune, "_CACHE", dict())
        asked = []

        def first(candidates, *args, **kwargs):
            asked.append([choice for choice, _ in candidates])
            return candidates[0][0]

        monkeypatch.setattr(autotune, "best_of", first)
        pc(x)
        tlmm = [c for c in asked if any(ch[0] == "cuda" for ch in c)]
        assert tlmm, "no tlmm gate measured"
        for cands in tlmm:
            assert cands[0] == ("triton", -1)
            want = [("cuda_prepass", c) for c in ck.prepass_valid_configs(128, B)] if on else []
            assert [ch for ch in cands if ch[0] == "cuda_prepass"] == want


def _float64_block(pc, bs, cids_k, pids_k):
    """Node block `k`'s rows in float64 from the layer's COMPILED tables (`partitioned_cids` / `pids`: an edge's child
    row and node 0's parameter, padding 0 -- not the kernels' derived tables): log sum_e params[m, e] exp(emars[e, b])."""
    real = cids_k != 0
    par = pc.params.double()[pids_k[real][:, None] + torch.arange(bs, device = DEV)]              # [E, bs]
    return torch.logsumexp(par.log()[:, :, None] + pc.element_mars.double()[cids_k[real]][:, None, :], dim = 0)


@pytest.mark.parametrize("dts", list(DTYPES))
@pytest.mark.parametrize("B", [64, 128])
def test_the_prepass_forward_on_32_node_blocks(B, dts, monkeypatch):
    """Layers of 32-node blocks (an HCLT's), which no other CUDA kernel takes: the 32-row configs against each other to
    the bit and against float64; the pre-pass over the DISTINCT element tiles (a variable's 8 blocks share theirs); the
    dispatch, told the gate chose one of them, reaching it."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = _hclt(256, DTYPES[dts], block_size = 32)                # 8 blocks a variable: they share the pre-pass
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for layer, nids, ebase, pbase in _cuda_layers(pc, B):
        assert layer.block_size == 32 and not ck.valid_configs(32, B) and not ck.pipe_valid_configs(32, B)
        tiles, tix, shared = ck.tile_index(ebase)
        assert shared and tiles.numel() * 8 == ebase.numel() and torch.equal(tiles[tix.long()], ebase), "8 share a tile"
        a, b = layer.nid_range
        bs, knt = 32, ebase.size(1)
        outs = []
        for cfg in ck.prepass_valid_configs(bs, B):
            pc.node_mars[a:b] = NAN
            ck.tlmm_forward_sum_prepass(pc.node_mars, pc.element_mars, pc.params, nids, tiles, tix, pbase, B, bs, knt,
                                        cfg, shared)
            outs.append(pc.node_mars[a:b].clone())
        assert outs and all(torch.equal(o, outs[0]) for o in outs[1:]), (dts, B, "the configs differ")   # (B 128: two)
        cids, pids = layer.partitioned_cids[0], layer.partitioned_pids[0]
        for k in range(nids.numel()):
            r0 = int(nids[k]) - a
            ref, got = _float64_block(pc, bs, cids[k].long(), pids[k].long()), outs[0][r0:r0 + bs].double()
            fin = torch.isfinite(ref)
            assert torch.equal(fin, torch.isfinite(got)), (dts, B, k, "finite pattern")
            assert float((got[fin] - ref[fin]).abs().max()) < 1e-2, (dts, B, k, "against float64")
        cfg0 = ck.prepass_valid_configs(bs, B)[0]
        was = autotune.set_autotune(True, clear_cache = True)
        try:
            with monkeypatch.context() as mp:
                mp.setattr(SL, "FORWARD_SUM_CUDA", True)
                mp.setattr(autotune, "best_of", lambda cands, *args, **kw: ("cuda_prepass", cfg0))
                pc.node_mars[a:b] = NAN
                layer.forward(pc.node_mars, pc.element_mars, pc.params)
                assert torch.equal(pc.node_mars[a:b], outs[0])
        finally:
            autotune.set_autotune(was, clear_cache = True)
        pc.node_mars[a:b] = outs[0]
        checked += 1
    assert checked > 0, "no layer of 32-node blocks takes the pre-pass kernel"
