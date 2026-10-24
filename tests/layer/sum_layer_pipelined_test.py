"""
The warp-specialized CUDA tlmm forward (c/tlmm_forward_sum_pipelined.cu: producer warps stream the operands a tile
ahead of the MMA warps) against the CUDA tlmm kernel it copies -- bit-identical at every tile config valid for the
layer, on HCLT-128 and HMM at B 64 and 128; and the layer's dispatch, told the gate chose it, reaches it on a whole
launch and on runs. Skipped where the CUDA kernels cannot build.
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


def _hclt(latents, cats = 16, V = 64):
    torch.manual_seed(0)
    x = torch.randint(0, cats, (512, V))
    ns = juice.structures.HCLT(x.float(), num_latents = latents, input_dist = dists.Categorical(num_cats = cats))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), V, cats


def _hmm():
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 8, num_latents = 128, num_emits = 16)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), 8, 16


CIRCUITS = {"hclt128": lambda: _hclt(128), "hmm": _hmm}


@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_pipelined_forward_is_the_cuda_forward_to_the_bit(name, B, monkeypatch):
    if not (ck.is_available() and ck.pipe_is_available()):
        pytest.skip("the CUDA kernels are not available here")
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    pc(x)
    checked = 0
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            sig = ("block_sparse", 0, 64)
            if sig not in layer._cached_fw_cuda or not layer._cached_fw_cuda[sig][2]:
                continue
            plain, pipe = ck.valid_configs(layer.block_size, B), ck.pipe_valid_configs(layer.block_size, B)
            if not plain or not pipe:
                continue
            ebase, pbase, _ = layer._cached_fw_cuda[sig]
            nids = layer.partitioned_nids[0]
            n, knt, bs = nids.numel(), ebase.size(1), layer.block_size
            a, b = layer.nid_range
            pc.node_mars[a:b] = NAN
            ck.tlmm_forward_sum(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, B, bs, knt, plain[0])
            want = pc.node_mars[a:b].clone()
            for cfg in pipe:
                pc.node_mars[a:b] = NAN
                ck.tlmm_forward_sum_pipelined(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, B, bs, knt, cfg)
                assert torch.equal(pc.node_mars[a:b], want), (name, cfg)
                checked += 1
            # the dispatch, told the gate chose the first config: a whole launch and runs (the stub stands only around
            # the two calls -- every tuned launch goes through `best_of` -- and its choices leave the cache with it)
            was = autotune.set_autotune(True, clear_cache = True)
            try:
                with monkeypatch.context() as mp:
                    mp.setattr(SL, "FORWARD_SUM_CUDA", True)
                    mp.setattr(autotune, "best_of", lambda cands, *args, **kw: ("cuda_pipe", pipe[0]))
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


@pytest.mark.parametrize("B", [64, 128])
@pytest.mark.parametrize("name", list(CIRCUITS))
def test_the_pipelined_element_backward_is_the_cuda_backward_to_the_bit(name, B, monkeypatch):
    """The warp-specialized element-flow backward (c/ele_backward_sum_pipelined.cu) against the CUDA element-flow
    kernel it copies, on every row it writes; and the layer's dispatch, told the gate chose it, on a whole launch and on
    runs."""
    if not (ck.ele_is_available() and ck.ele_pipe_is_available()):
        pytest.skip("the CUDA kernels are not available here")
    pc, V, C = CIRCUITS[name]()
    x = torch.randint(0, C, (B, V), device = DEV)
    monkeypatch.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", False)
    pc(x)
    pc.backward(x, compute_param_flows = False)                     # the backward's caches, Triton throughout
    checked = 0
    for lg in pc.inner_layer_groups:
        if not lg.is_sum():
            continue
        for layer in lg:
            for sig, (ebase, pbase, ok) in list(layer._cached_bk_ele_cuda.items()):
                if not ok or sig[2] != 64:
                    continue
                part = sig[1]
                chids, cs, bs = layer.partitioned_chids[part], int(layer.cs_block_sizes[part]), layer.block_size
                if cs % 128 or B % 64:
                    continue
                n, knt = chids.numel(), ebase.size(1)
                rows = (chids[:, None] + torch.arange(cs, device = DEV)[None, :]).reshape(-1)
                ef = torch.full_like(pc.element_flows, NAN)
                ck.ele_backward_sum(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, ebase, pbase, B, bs, cs, knt)
                want = ef[rows].clone()
                ef.fill_(NAN)
                ck.ele_backward_sum_pipelined(ef, pc.element_mars, pc.node_flows, pc.node_mars, pc.params, chids, ebase, pbase, B, bs, cs, knt)
                assert torch.equal(ef[rows], want), (name, sig)
                checked += 1
            # the dispatch, told the gate chose the pipelined kernel: the layer's backward over all its partitions
            # against the plain CUDA kernel's, whole and on runs of partition 0
            if not layer._cached_bk_ele_cuda or not any(ok for _, _, ok in layer._cached_bk_ele_cuda.values()):
                continue
            kw = dict(logspace_flows = True)
            with monkeypatch.context() as mp:
                mp.setattr(SL, "BACKWARD_ELE_FLOW_CUDA", True)
                mp.setattr(autotune, "best_of", lambda cands, *args, **k: ("cuda", 0))
                layer._cached_bk_ele_choice.clear()
                ef = torch.full_like(pc.element_flows, NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                want = ef.clone()
                mp.setattr(autotune, "best_of", lambda cands, *args, **k: ("cuda_pipe", 0))
                layer._cached_bk_ele_choice.clear()
                ef.fill_(NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params, **kw)
                assert torch.equal(torch.nan_to_num(ef, nan = 7.0), torch.nan_to_num(want, nan = 7.0)), name
                n = layer.partitioned_chids[0].numel()
                ef.fill_(NAN)
                layer.backward(pc.node_flows, ef, pc.node_mars, pc.element_mars, pc.params,
                               partition_runs = [[(0, 1), (n // 2, n)]] + [[]] * (layer.num_bk_partitions - 1), **kw)
                cs = int(layer.cs_block_sizes[0])
                rows = torch.cat([layer.partitioned_chids[0][s:e] for s, e in ((0, 1), (n // 2, n))])
                rows = (rows[:, None] + torch.arange(cs, device = DEV)[None, :]).reshape(-1)
                assert torch.equal(ef[rows], want[rows])
                layer._cached_bk_ele_choice.clear()
    assert checked > 0, "no layer of the circuit takes the CUDA path"


def test_the_gate_offers_the_pipelined_configs(monkeypatch):
    """The plain CUDA gate's candidates hold the pipelined kernel's configs beside the plain ones and Triton."""
    if not (ck.is_available() and ck.pipe_is_available()):
        pytest.skip("the CUDA kernels are not available here")
    pc, V, C = _hclt(128)
    B = 128
    x = torch.randint(0, C, (B, V), device = DEV)
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", True)
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
        assert [ch for ch in cands if ch[0] == "cuda_pipe"] == [("cuda_pipe", c) for c in ck.pipe_valid_configs(128, B)]
