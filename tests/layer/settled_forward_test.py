"""
The layers' settled forwards (`SumLayer._fw_fast`, `ProdLayer._fw_fast`): a plain whole forward whose launches the full
dispatch recorded relaunches them without it, to the bit -- and goes back through the full dispatch whenever that
could decide differently: the tuner's cache cleared or deciding anew, the CUDA kernels switched off, a call that is not
plain (runs, another precision), the circuit moved. Skipped where the CUDA kernels cannot build. And the
circuit's inner loop, which calls a layer group at its `forward`, still running module hooks.
"""
import pytest
import torch

import pyjuice as juice
from pyjuice.layer import SumLayer, ProdLayer
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as ck

DEV = torch.device("cuda:0")


def _hmm(dt, latents = 128):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 8, num_latents = latents, num_emits = 16)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dt).to(DEV)


def _counted(monkeypatch, cls, name):
    calls = []
    real = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(cls, name, wrapped)
    return calls


def _layers(pc, cls):
    return [layer for group in pc.inner_layer_groups for layer in group if isinstance(layer, cls)]


@pytest.mark.parametrize("dts", ["fp32", "bf16"])
def test_a_settled_forward_skips_the_dispatch_to_the_bit(dts, monkeypatch):
    if not (ck.is_available() and ck.prepass_is_available()):
        pytest.skip("the CUDA kernels are not available here")
    pc = _hmm({"fp32": torch.float32, "bf16": torch.bfloat16}[dts])
    x = torch.randint(0, 16, (64, 8), device = DEV)
    want = pc(x).detach().clone()                                   # (the full dispatch: tunes, records)
    sums, prods = _layers(pc, SumLayer), _layers(pc, ProdLayer)
    # the layers on a tuned CUDA tlmm / 2-d product launch record (the root's block of 1 takes the sparse kernel)
    settled_s, settled_p = [l for l in sums if l._fw_fast], [l for l in prods if l._fw_fast]
    assert len(settled_s) >= len(sums) - 1 and len(settled_p) == len(prods)
    s_calls, p_calls = _counted(monkeypatch, SumLayer, "_forward"), _counted(monkeypatch, ProdLayer, "_forward_backward")
    got = pc(x).detach()
    assert len(s_calls) == len(sums) - len(settled_s) and p_calls == []     # the settled ones took no dispatch
    assert torch.equal(got, want)
    # without the fast paths, the same bits
    monkeypatch.setattr(SumLayer, "_fw_fast_launch", lambda self, *a: False)
    monkeypatch.setattr(ProdLayer, "_fw_fast_launch", lambda self, *a: False)
    s_calls.clear(); p_calls.clear()
    assert torch.equal(pc(x).detach(), want)
    assert len(s_calls) == sum(l.num_fw_partitions for l in sums) and len(p_calls) == sum(l.num_fw_partitions for l in prods)


def test_the_settled_forward_follows_the_tuner_and_the_switches(monkeypatch):
    if not (ck.is_available() and ck.prepass_is_available()):
        pytest.skip("the CUDA kernels are not available here")
    pc = _hmm(torch.bfloat16)
    x = torch.randint(0, 16, (64, 8), device = DEV)
    pc(x)
    sums, prods = _layers(pc, SumLayer), _layers(pc, ProdLayer)
    unsettled = len([l for l in sums if not l._fw_fast])
    s_calls, p_calls = _counted(monkeypatch, SumLayer, "_forward"), _counted(monkeypatch, ProdLayer, "_forward_backward")
    # the tuner's cache cleared, and it decides differently this time (the Triton reference): every settled sum layer
    # back through its dispatch, to the Triton kernel; then settled again on the new choice (the sum layers on Triton
    # record nothing: their dispatch every call)
    twin = _hmm(torch.bfloat16)
    with monkeypatch.context() as mp:
        mp.setattr(SL, "FORWARD_SUM_CUDA", False)
        triton_out = twin(x).detach()
    was = autotune.set_autotune(True, clear_cache = True)
    try:
        with monkeypatch.context() as mp:
            mp.setattr(autotune, "best_of", lambda cands, *a, **k: next((c for c, _ in cands if c == ("triton", -1)), cands[0][0]))
            s_calls.clear(); p_calls.clear()
            assert torch.equal(pc(x).detach(), triton_out)
            assert len(s_calls) == sum(l.num_fw_partitions for l in sums)
            assert len(p_calls) == sum(l.num_fw_partitions for l in prods)
            s_calls.clear(); p_calls.clear()
            assert torch.equal(pc(x).detach(), triton_out)
            assert len(s_calls) == sum(l.num_fw_partitions for l in sums) and p_calls == []
    finally:
        autotune.set_autotune(was, clear_cache = True)
    pc(x)                                                           # (tuned and recorded again)
    # the CUDA kernels switched off: the sum layers take the Triton kernel, as a circuit that never used CUDA does
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
    s_calls.clear()
    got = pc(x).detach()
    assert len(s_calls) == sum(l.num_fw_partitions for l in sums)
    assert torch.equal(got, triton_out)
    monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", True)
    # a call that is not plain records nothing and replays nothing
    s_calls.clear()
    layer = sums[0]
    before = dict(layer._fw_fast)
    a, b = layer.nid_range
    layer.forward(pc.node_mars, pc.element_mars, pc.params, precision = "fp32")
    assert len(s_calls) == layer.num_fw_partitions and layer._fw_fast.keys() == before.keys() | {(64, torch.bfloat16, "fp32")} \
        or layer._fw_fast.keys() == before.keys()                     # (fp32 products: no CUDA tlmm, nothing recorded)
    n = layer.partitioned_nids[0].numel()
    s_calls.clear()
    layer.forward(pc.node_mars, pc.element_mars, pc.params, partition_runs = [[(0, n)]] + [[]] * (layer.num_fw_partitions - 1))
    assert len(s_calls) == 1


def test_a_device_move_drops_the_settled_launches():
    if not (ck.is_available() and ck.prepass_is_available()):
        pytest.skip("the CUDA kernels are not available here")
    pc = _hmm(torch.float32)
    x = torch.randint(0, 16, (64, 8), device = DEV)
    want = pc(x).detach().clone()
    layers = _layers(pc, SumLayer) + _layers(pc, ProdLayer)
    assert sum(bool(layer._fw_fast) for layer in layers) >= len(layers) - 1
    pc.to("cpu")
    assert not any(layer._fw_fast for layer in layers)
    pc.to(DEV)
    assert torch.equal(pc(x).detach(), want)


def test_the_inner_loop_keeps_module_hooks():
    """The circuit's inner loop calls a layer group at its `forward` only when calling it would do nothing more: a hook
    on a group, or a global module hook, still runs."""
    pc = _hmm(torch.float32)
    x = torch.randint(0, 16, (64, 8), device = DEV)
    want = pc(x).detach().clone()
    seen = []
    group = pc.inner_layer_groups[0]
    handle = group.register_forward_hook(lambda module, args, out: seen.append("group"))
    try:
        assert torch.equal(pc(x).detach(), want)
    finally:
        handle.remove()
    assert seen == ["group"]
    seen.clear()
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda module, args, out: seen.append(type(module).__name__))
    try:
        assert torch.equal(pc(x).detach(), want)
    finally:
        handle.remove()
    assert seen.count("LayerGroup") == len(pc.inner_layer_groups)
    seen.clear()
    pc(x)
    assert seen == []
