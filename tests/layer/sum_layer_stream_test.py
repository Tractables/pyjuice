"""
The stream CUDA tlmm forward (c/tlmm_forward_sum_stream.cu: a pre-pass over a node block's children per split range
under one stabilizer, shared by the node blocks of equal children; a GEMM of mbarrier'd cp.async stages; a merge of the
ranges) against float64 from the compiled tables and against the pre-pass kernel it stands beside -- every fitting tile
config at every split arm, bf16 parameters, on HMMs of 128 (two edge tiles), 448 (blocks of 64, 7 tiles) and 896
latents (14 tiles), 512 latents in blocks of 256 (the 256-row tiles) and HCLT-128 and -256 at blocks of 64 and 32 (a variable's blocks sharing their children: a
128-row tile over 2 or 4 of them), at B 64 and 128; columns whose children are
all -inf in a range or in every range; and the layer's dispatch, told the gate chose it, reaching it on a whole launch
and on runs, and its settled replay. Skipped where the CUDA kernels cannot build.
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


def _hclt(latents, cats = 16, V = 64, block_size = None):
    torch.manual_seed(0)
    x = torch.randint(0, cats, (512, V))
    ns = juice.structures.HCLT(x.float(), num_latents = latents, input_dist = dists.Categorical(num_cats = cats),
                               **({} if block_size is None else dict(block_size = block_size)))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(DEV), V, cats


def _hmm(latents, T = 8, block_size = None):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = latents, num_emits = 16,
                              **({} if block_size is None else dict(block_size = block_size)))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(DEV), T, 16


CIRCUITS = {"hclt128": lambda: _hclt(128), "hclt256_b64": lambda: _hclt(256, block_size = 64),
            "hclt256_b32": lambda: _hclt(256, block_size = 32),
            "hmm128": lambda: _hmm(128), "hmm448": lambda: _hmm(448, T = 4), "hmm896": lambda: _hmm(896, T = 4),
            "hmm512_b256": lambda: _hmm(512, T = 4, block_size = 256)}     # (blocks of 256: the 256-row tiles)


def _available():
    return ck.is_available() and ck.prepass_is_available() and ck.stream_is_available()


def _cuda_layers(pc, B):
    """(layer, nids, ebase, pbase) of every sum layer the stream forward takes at batch B."""
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
            if not ck.stream_arms(layer.block_size, B, ebase.size(1), layer.partitioned_nids[0].numel(),
                                  ck.row_groups(ebase)[2]):
                continue
            yield layer, layer.partitioned_nids[0], ebase, pbase


def _float64(pc, layer, nids, B):
    """The layer's rows in float64 from the compiled tables (`partitioned_cids` / `pids`: an edge's child row and node
    0's parameter, padding 0): log sum_e params[m, e] exp(emars[e, b])."""
    a, b = layer.nid_range
    bs = layer.block_size
    cids, pids = layer.partitioned_cids[0].long(), layer.partitioned_pids[0].long()
    out = torch.empty(b - a, B, dtype = torch.float64, device = DEV)
    for k in range(nids.numel()):
        real = cids[k] != 0
        par = pc.params.double()[pids[k][real][:, None] + torch.arange(bs, device = DEV)]
        r0 = int(nids[k]) - a
        out[r0:r0 + bs] = torch.logsumexp(par.log()[:, :, None] + pc.element_mars.double()[cids[k][real]][:, None, :],
                                          dim = 0)
    return out


def _prepass(pc, layer, nids, ebase, pbase, B):
    a, b = layer.nid_range
    tiles = ck.tile_index(ebase)
    pc.node_mars[a:b] = NAN
    ck.tlmm_forward_sum_prepass(pc.node_mars, pc.element_mars, pc.params, nids, tiles[0], tiles[1], pbase, B,
                                layer.block_size, ebase.size(1), ck.prepass_valid_configs(layer.block_size, B)[0],
                                tiles[2])
    return pc.node_mars[a:b].clone()


def _check_arms(pc, layer, nids, ebase, pbase, B, ref, pre, label):
    """Every arm's rows: the -inf pattern of float64's, within 1e-2 of it (bf16 exponentials and parameters) and within
    5e-3 of the pre-pass kernel's (the same operands under another stabilizer)."""
    a, b = layer.nid_range
    groups = ck.row_groups(ebase)
    fin = torch.isfinite(ref)
    for code in ck.stream_arms(layer.block_size, B, ebase.size(1), nids.numel(), groups[2]):
        pc.node_mars[a:b] = NAN
        ck.tlmm_forward_sum_stream(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, groups, B,
                                   layer.block_size, ebase.size(1), code)
        got = pc.node_mars[a:b]
        assert torch.equal(fin, torch.isfinite(got)) and (got[~fin] == NINF).all(), (label, code, "the -inf pattern")
        assert float((got.double()[fin] - ref[fin]).abs().max()) < 1e-2, (label, code, "against float64")
        assert float((got[fin] - pre[fin]).abs().max()) < 5e-3, (label, code, "against the pre-pass kernel")


@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_stream_forward_against_float64_and_the_prepass_kernel(name, B, monkeypatch):
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for layer, nids, ebase, pbase in _cuda_layers(pc, B):
        ref = _float64(pc, layer, nids, B)
        pre = _prepass(pc, layer, nids, ebase, pbase, B)
        ugrp, grep, grouped = ck.row_groups(ebase)
        share = {"hclt128": 1, "hclt256_b64": 4, "hclt256_b32": 8}.get(name)   # (a variable's blocks share children)
        if share is not None:
            assert grep.numel() * share == nids.numel() and grouped == min(share, 4), (name, "blocks sharing children")
        else:                                                           # an HMM: every block the same children
            assert grep.numel() == 1, (name, "one group")
        _check_arms(pc, layer, nids, ebase, pbase, B, ref, pre, (name, B))
        a, b = layer.nid_range
        pc.node_mars[a:b] = pre
        checked += 1
    assert checked > 0, "no layer of the circuit takes the stream forward"


@pytest.mark.parametrize("name", ["hmm896", "hclt256_b64", "hclt256_b32"])
def test_the_stream_forward_on_dead_columns(name, monkeypatch):
    """Columns whose children are all -inf in a range (its M -inf: the range adds nothing), in every range (-inf out),
    or in some rows of a tile only, against float64 and the pre-pass kernel."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS[name]()
    B = 64
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for layer, nids, ebase, pbase in _cuda_layers(pc, B):
        saved = pc.element_mars.clone()
        knt = ebase.size(1)
        tile_rows = lambda kt: (ebase[:, kt, None] + torch.arange(64, device = DEV)[None, :]).reshape(-1)
        pc.element_mars[tile_rows(0), 3] = NINF                         # the first tile: dead in the first range
        pc.element_mars[tile_rows(knt - 1), 5] = NINF
        for kt in range(knt):
            pc.element_mars[tile_rows(kt), 7] = NINF                    # every child: -inf out
        pc.element_mars[tile_rows(0)[::3], 9] = NINF
        ref = _float64(pc, layer, nids, B)
        assert (ref[:, 7] == NINF).all()
        pre = _prepass(pc, layer, nids, ebase, pbase, B)
        _check_arms(pc, layer, nids, ebase, pbase, B, ref, pre, (name, "dead columns"))
        pc.element_mars.copy_(saved)
        checked += 1
    assert checked > 0, "no layer of the circuit takes the stream forward"


def _check_runs(got, want, rows, label):
    """A launch over runs: the rows outside untouched (NaN), those inside the arm's -- or, a run too short for the
    partition's grouped tiles, the pre-pass kernel's: the same -inf pattern, within 5e-3."""
    assert torch.isnan(got[~rows]).all(), (label, "rows outside the runs")
    g, w = got[rows], want[rows]
    fin = torch.isfinite(w)
    assert torch.equal(torch.isfinite(g), fin) and (g[~fin] == NINF).all(), (label, "the -inf pattern")
    assert float((g[fin] - w[fin]).abs().max()) < 5e-3, (label, "the runs' rows")


@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name, arm", [("hmm448", -1), ("hclt256_b64", 0)])
def test_the_dispatch_reaches_the_stream_forward(name, arm, B, monkeypatch):
    """The layer's forward, told the gate chose a stream arm (an HMM's last: a split where one is offered; an HCLT's
    first: a 128-row tile over 2 of a variable's 64-node blocks, which the 1-block run cannot take), gives the arm's
    rows -- whole and on runs -- and replays it settled (`_fw_fast_launch`); without the stream module the gate offers
    no stream arm."""
    if not _available():
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for layer, nids, ebase, pbase in _cuda_layers(pc, B):
        a, b = layer.nid_range
        bs, knt = layer.block_size, ebase.size(1)
        groups = ck.row_groups(ebase)
        code = ck.stream_arms(bs, B, knt, nids.numel(), groups[2])[arm]
        if name == "hclt256_b64":
            assert not ck.stream_run_fits(code, bs, 1, 1), "the 1-block run must need the pre-pass kernel"
        pc.node_mars[a:b] = NAN
        ck.tlmm_forward_sum_stream(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, groups, B, bs, knt, code)
        want = pc.node_mars[a:b].clone()
        was = autotune.set_autotune(True, clear_cache = True)
        try:
            with monkeypatch.context() as mp:
                mp.setattr(SL, "FORWARD_SUM_CUDA", True)
                asked = []
                real_best_of = autotune.best_of
                # the sum layer's gate (a stream arm among its candidates) is told to take the arm; every other tuner
                # (the whole circuit's forward below runs the product layers' too) tunes as usual
                mp.setattr(autotune, "best_of", lambda cands, *args, **kw: asked.append([c for c, _ in cands]) or
                           (("cuda_stream", code) if any(c[0] == "cuda_stream" for c, _ in cands)
                            else real_best_of(cands, *args, **kw)))
                pc.node_mars[a:b] = NAN
                layer.forward(pc.node_mars, pc.element_mars, pc.params)
                assert torch.equal(pc.node_mars[a:b], want)
                offered = [c for cands in asked for c in cands if c[0] == "cuda_stream"]
                assert offered == [("cuda_stream", c) for c in ck.stream_arms(bs, B, knt, nids.numel(), groups[2])]
                n = nids.numel()
                runs = [(0, 1), (n // 2, n)]
                pc.node_mars[a:b] = NAN
                layer.forward(pc.node_mars, pc.element_mars, pc.params,
                              partition_runs = [runs] + [[]] * (layer.num_fw_partitions - 1))
                rows = torch.zeros(b - a, dtype = torch.bool, device = DEV)
                for s, e in runs:
                    for r in nids[s:e].tolist():
                        rows[r - a:r - a + bs] = True
                _check_runs(pc.node_mars[a:b], want, rows, (name, B))
                # settled: the whole circuit's forward records the stream launch and replays it
                pc(x)
                fast = [e for e in layer._fw_fast.values() for e in e]
                assert fast and fast[0][1] == ("cuda_stream", code), "the stream launch was not recorded"
                # the replay against the full dispatch inside the circuit (the product layers share `element_mars`, so
                # this layer's children are there only while its turn runs): a second, settled full dispatch -- every
                # tuner's choice cached -- gives the rows the replay must repeat
                layer._fw_fast.clear()
                pc(x)
                want_now = pc.node_mars[a:b].clone()
                assert layer._fw_fast, "the settled full dispatch did not record the launch"
                orig_fast_launch = SL.SumLayer._fw_fast_launch
                replayed = []
                def spy(self, *args, **kw):
                    r = orig_fast_launch(self, *args, **kw)
                    replayed.append((self, r))
                    return r
                mp.setattr(SL.SumLayer, "_fw_fast_launch", spy)
                pc.node_mars[a:b] = NAN
                pc(x)
                assert any(s is layer and r for s, r in replayed), "the settled launch did not replay"
                assert torch.equal(pc.node_mars[a:b], want_now), "the settled replay"
                mp.setattr(ck, "stream_is_available", lambda: False)
                autotune.set_autotune(True, clear_cache = True)
                asked.clear()
                mp.setattr(autotune, "best_of", lambda cands, *args, **kw: asked.append([c for c, _ in cands]) or
                           (("triton", -1) if any(c == ("triton", -1) for c, _ in cands)
                            else real_best_of(cands, *args, **kw)))
                layer._fw_fast.clear()
                layer.forward(pc.node_mars, pc.element_mars, pc.params)
                assert asked and not any(c[0] == "cuda_stream" for cands in asked for c in cands)
        finally:
            autotune.set_autotune(was, clear_cache = True)
        pc.node_mars[a:b] = want
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------------------------------------------------
# the stream ELEMENT backward (c/ele_backward_sum_stream.cu): the same design over a partition's child blocks -- the
# pre-pass over a block's parents' log ratios (shared by the blocks of equal parents), the GEMM, the merge adding
# element_mars -- against float64 from the compiled parent tables and the pre-pass element kernel
# ---------------------------------------------------------------------------------------------------------------------

def _ele_available():
    return ck.ele_is_available() and ck.ele_prepass_is_available() and ck.ele_stream_is_available()


def _ele_state(name, B, monkeypatch):
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", False)
    pc(x)
    pc.backward(x, compute_param_flows = False)                     # the backward's caches, Triton throughout
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", True)
    return pc


def _ele_partitions(pc, B):
    """(layer, sig, chids, cs, ebase, pbase, rows, groups) of every partition the stream element backward takes at
    batch B (`pbase`: the pre-pass tables' per-tile parameter offsets)."""
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            for sig, (ebase, _, _) in list(layer._cached_bk_ele_cuda.items()):
                if sig[2] != 64 or sig not in layer._cached_bk_ele_pre:
                    continue
                pbase, pre_ok = layer._cached_bk_ele_pre[sig]
                chids, cs = layer.partitioned_chids[sig[1]], int(layer.cs_block_sizes[sig[1]])
                if not pre_ok or not layer._cuda_param_loads_aligned(pc.params, ("ele", sig), pbase, layer.block_size):
                    continue
                groups = ck.row_groups(ebase)
                if not ck.ele_stream_arms(cs, B, layer.block_size, ebase.size(1), chids.numel(), groups[2]):
                    continue
                rows = (chids[:, None] + torch.arange(cs, device = DEV)[None, :]).reshape(-1)
                yield layer, sig, chids, cs, ebase, pbase, rows, groups


def _ele_float64(pc, layer, sig, chids, cs):
    """The partition's element flows in float64 from the layer's COMPILED tables (`partitioned_parids` / `parpids`: a
    parent block's first node row and first parameter per child block, padding 0): element_mars + log sum_e params[m, e]
    exp(lf[e, b]), lf = node_flows - node_mars (-inf where node_mars is)."""
    bs = layer.block_size
    parids, parpids = layer.partitioned_parids[sig[1]].long(), layer.partitioned_parpids[sig[1]].long()
    ar = torch.arange(bs, device = DEV)
    out = []
    for k in range(chids.numel()):
        real = parids[k] != 0
        e_rows = (parids[k][real][:, None] + ar).reshape(-1)
        nf, nm = pc.node_flows.double()[e_rows], pc.node_mars.double()[e_rows]
        lf = torch.where(nm == NINF, torch.full_like(nm, NINF), nf - nm)
        pidx = (parpids[k][real][:, None] + ar).reshape(-1)
        par = pc.params.double()[pidx[None, :] + torch.arange(cs, device = DEV)[:, None] * bs]
        rows = int(chids[k]) + torch.arange(cs, device = DEV)
        out.append(pc.element_mars.double()[rows] + torch.logsumexp(par.log()[:, :, None] + lf[None], dim = 1))
    return torch.cat(out)


def _ele_prepass(pc, layer, chids, cs, ebase, pbase, rows, B):
    tiles = ck.tile_index(ebase)
    cfgs = ck.ele_prepass_valid_configs(cs, B)
    assert cfgs, "the pre-pass element kernel takes no config here"
    ef = torch.full_like(pc.element_flows, NAN)
    ck.ele_backward_sum_prepass(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, tiles[0], tiles[1],
                                pbase, B, layer.block_size, cs, ebase.size(1), cfgs[0], tiles[2])
    return ef[rows].clone()


def _check_ele_arms(pc, layer, chids, cs, ebase, pbase, rows, groups, B, ref, pre, label):
    """Every arm's rows: the -inf pattern of float64's, within 1e-2 of it (bf16 exponentials and parameters) and within
    5e-3 of the pre-pass element kernel's (fp16 exponentials under per-tile stabilizers)."""
    fin = torch.isfinite(ref)
    for code in ck.ele_stream_arms(cs, B, layer.block_size, ebase.size(1), chids.numel(), groups[2]):
        ef = torch.full_like(pc.element_flows, NAN)
        ck.ele_backward_sum_stream(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, ebase, pbase,
                                   groups, B, layer.block_size, cs, ebase.size(1), code)
        got = ef[rows]
        assert torch.equal(fin, torch.isfinite(got)) and (got[~fin] == NINF).all(), (label, code, "the -inf pattern")
        assert float((got.double()[fin] - ref[fin]).abs().max()) < 1e-2, (label, code, "against float64")
        assert float((got[fin] - pre[fin]).abs().max()) < 5e-3, (label, code, "against the pre-pass kernel")


@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_stream_element_backward_against_float64_and_the_prepass_kernel(name, B, monkeypatch):
    if not _ele_available():
        pytest.skip("the CUDA kernels are not available here")
    pc = _ele_state(name, B, monkeypatch)
    checked = 0
    for layer, sig, chids, cs, ebase, pbase, rows, groups in _ele_partitions(pc, B):
        ref = _ele_float64(pc, layer, sig, chids, cs)
        pre = _ele_prepass(pc, layer, chids, cs, ebase, pbase, rows, B)
        assert float((pre.double()[torch.isfinite(ref)] - ref[torch.isfinite(ref)]).abs().max()) < 1e-2, "the pre-pass"
        if name.startswith("hmm"):                                      # every child block the same parents
            assert groups[1].numel() == 1, (name, "one group")
        _check_ele_arms(pc, layer, chids, cs, ebase, pbase, rows, groups, B, ref, pre, (name, B, sig))
        checked += 1
    assert checked > 0, "no partition of the circuit takes the stream element backward"


@pytest.mark.parametrize("name", ["hmm896", "hclt256_b64", "hclt256_b32"])
def test_the_stream_element_backward_on_dead_columns(name, monkeypatch):
    """Columns whose parents are all dead (node_mars -inf) in a range (its M -inf: the range adds nothing), in every
    range (-inf out), or in some parents of a tile only, against float64 and the pre-pass kernel."""
    if not _ele_available():
        pytest.skip("the CUDA kernels are not available here")
    B = 64
    pc = _ele_state(name, B, monkeypatch)
    checked = 0
    for layer, sig, chids, cs, ebase, pbase, rows, groups in _ele_partitions(pc, B):
        saved = pc.node_mars.clone()
        knt = ebase.size(1)
        tile_rows = lambda kt: (ebase[:, kt, None] + torch.arange(64, device = DEV)[None, :]).reshape(-1)
        pc.node_mars[tile_rows(0), 3] = NINF                            # the first tile: dead in the first range
        pc.node_mars[tile_rows(knt - 1), 5] = NINF
        for kt in range(knt):
            pc.node_mars[tile_rows(kt), 7] = NINF                       # every parent: -inf out
        pc.node_mars[tile_rows(0)[::3], 9] = NINF
        ref = _ele_float64(pc, layer, sig, chids, cs)
        assert (ref[:, 7] == NINF).all()
        pre = _ele_prepass(pc, layer, chids, cs, ebase, pbase, rows, B)
        _check_ele_arms(pc, layer, chids, cs, ebase, pbase, rows, groups, B, ref, pre, (name, "dead columns"))
        pc.node_mars.copy_(saved)
        checked += 1
    assert checked > 0, "no partition of the circuit takes the stream element backward"


@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_dispatch_reaches_the_stream_element_backward(name, monkeypatch):
    """The layer's backward, told the gate chose a stream arm, gives the arm's rows -- whole and on runs of partition 0
    --; the gate offers every arm of `ele_stream_arms`, and none without the stream module."""
    if not _ele_available():
        pytest.skip("the CUDA kernels are not available here")
    B = 64
    pc = _ele_state(name, B, monkeypatch)
    monkeypatch.setattr(lp, "ENABLED", False)          # (every call below must take the dispatch, not replay a plan)
    kw = dict(logspace_flows = True)
    real_best_of = autotune.best_of
    checked = 0
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            parts = [p for p in _ele_partitions(pc, B) if p[0] is layer and p[1][1] == 0]
            if not parts:
                continue
            _, sig, chids, cs, ebase, pbase, rows, groups = parts[0]
            arms = ck.ele_stream_arms(cs, B, layer.block_size, ebase.size(1), chids.numel(), groups[2])
            want = torch.full_like(pc.element_flows, NAN)
            ck.ele_backward_sum_stream(want, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, ebase,
                                       pbase, groups, B, layer.block_size, cs, ebase.size(1), arms[0])
            want = want[rows].clone()
            with monkeypatch.context() as mp:
                asked = []
                # each partition's gate takes its own first stream arm; a gate without one (a partition the stream does
                # not serve) tunes as usual
                mp.setattr(autotune, "best_of", lambda cands, *a, **k: asked.append([c for c, _ in cands]) or
                           next((c for c, _ in cands if c[0] == "cuda_stream"), None) or real_best_of(cands, *a, **k))
                layer._cached_bk_ele_choice.clear()
                ef = torch.full_like(pc.element_flows, NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                assert torch.equal(ef[rows], want), (name, "the whole launch")
                assert [c for c in asked[0] if c[0] == "cuda_stream"] == [("cuda_stream", c) for c in arms], "the arms"
                n = chids.numel()
                ef.fill_(NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params,
                               partition_runs = [[(0, 1), (n // 2, n)]] + [[]] * (layer.num_bk_partitions - 1), **kw)
                run_rows = torch.cat([chids[s:e] for s, e in ((0, 1), (n // 2, n))])
                run_rows = (run_rows[:, None] + torch.arange(cs, device = DEV)[None, :]).reshape(-1)
                _check_runs(ef[rows], want, torch.isin(rows, run_rows), (name, "element runs"))
                mp.setattr(ck, "ele_stream_is_available", lambda: False)
                asked.clear()
                mp.setattr(autotune, "best_of", lambda cands, *a, **k: asked.append([c for c, _ in cands]) or
                           next((c for c, _ in cands if c[0] in ("cuda", "cuda_prepass")), None)
                           or real_best_of(cands, *a, **k))
                layer._cached_bk_ele_choice.clear()
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                assert asked and not any(c[0] == "cuda_stream" for cands in asked for c in cands)
            layer._cached_bk_ele_choice.clear()
            checked += 1
    assert checked > 0, "no layer of the circuit takes the stream element backward"
