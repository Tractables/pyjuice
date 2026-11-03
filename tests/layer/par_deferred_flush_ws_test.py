"""
The deferred flush with its operand tiles loaded by a producer warp (layer/kernels/c/par_deferred_fp16_flush_ws.cu)
against par_deferred_fp16.cu's flush: the flows to the bit, every config of each, the L2's bulk adds and the threads'
read-add-write, bf16 and fp32 parameters, one and several node blocks and slots, slots of 64 to 512 columns, operand
values from 0 to full fp16 range with power-of-two row scales far apart. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

DEV = torch.device("cuda:0")


def _mods():
    if not torch.cuda.is_available():
        return None
    import pyjuice.layer.kernels.c as ck
    try:
        mods = (ck.get_par_deferred_module(), ck.get_par_deferred_flush_ws_module())
    except Exception:
        return None
    return mods if all(m is not None for m in mods) else None


needs_kernels = pytest.mark.skipif(_mods() is None, reason = "needs the deferred parameter-flow kernels (nvcc + CUTLASS)")

# (node blocks, block size, edges a node block, a slot's columns, slots, slots flushed)
SHAPES = [(1, 128, 128, 64, 2, 2), (2, 256, 384, 128, 4, 3), (1, 128, 256, 512, 2, 1), (2, 128, 128, 256, 3, 3),
          (4, 128, 128, 128, 8, 8)]


def _case(nbk, blk, ne, W, S, dtype, aligned, seed):
    g = torch.Generator(device = DEV).manual_seed(seed)
    # the captures' fp16 values: a row's max in [2^14, 2^15), smaller ones down to subnormals, some zero
    def operand(rows):
        v = torch.rand(nbk, rows, S * W, device = DEV, generator = g) * 32767.0
        v = v * torch.exp2(-torch.randint(0, 40, v.shape, device = DEV, generator = g).float())
        v[torch.rand(v.shape, device = DEV, generator = g) < 0.1] = 0.0
        return v.half()
    A, Bop = operand(blk), operand(ne)
    # the rows' power-of-two scales, far apart (the fold's 2^-30 and their product within float range)
    fA = torch.exp2(torch.randint(-40, 40, (nbk, S, blk), device = DEV, generator = g).float())
    fB = torch.exp2(torch.randint(-40, 40, (nbk, S, ne), device = DEV, generator = g).float())
    # each (node block, edge)'s run of `blk` flows and parameters, in a shuffled order; `aligned`: 16-byte runs
    off = 0 if aligned else 1
    runs = torch.randperm(nbk * ne, device = DEV, generator = g).view(nbk, ne)
    pfids = (runs * blk + off).long()
    pids = (torch.randperm(nbk * ne, device = DEV, generator = g).view(nbk, ne) * blk).long()
    pf = torch.randn(nbk * ne * blk + off + 3, device = DEV, generator = g)
    params = (torch.rand(nbk * ne * blk, device = DEV, generator = g) + 0.01).to(dtype)
    return pf, A, Bop, fA, fB, pfids, pids, params


@needs_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids = ["bf16", "fp32"])
@pytest.mark.parametrize("bulk", [True, False], ids = ["bulk_adds", "threads_add"])
@pytest.mark.parametrize("nbk,blk,ne,W,S,slots", SHAPES, ids = [f"nb{s[0]}_blk{s[1]}_e{s[2]}_W{s[3]}_s{s[5]}" for s in SHAPES])
def test_the_producer_warp_flush_adds_the_flushs_bits(nbk, blk, ne, W, S, slots, bulk, dtype):
    base, ws = _mods()
    pf, A, Bop, fA, fB, pfids, pids, params = _case(nbk, blk, ne, W, S, dtype, bulk, seed = nbk * 1000 + blk + W)
    ref = pf.clone()
    base.flush(ref, A, Bop, fA, fB, pfids, pids, params, slots, 0, bulk)
    assert not torch.equal(ref, pf), "the flush added nothing"
    for cfg, (bm, bn, bk) in enumerate(ws.flush_configs()):
        if blk % bm or ne % bn or W % bk:
            continue
        out = pf.clone()
        ws.flush(out, A, Bop, fA, fB, pfids, pids, params, slots, cfg, bulk)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int32), ref.view(torch.int32)), f"config {cfg}: the flows differ"


@needs_kernels
def test_the_flushs_configs_add_the_same_bits():
    """The reference above: every config of par_deferred_fp16.cu's flush that fits gives its config 0's bits."""
    base, _ = _mods()
    nbk, blk, ne, W, S, slots = 2, 128, 128, 128, 3, 3
    pf, A, Bop, fA, fB, pfids, pids, params = _case(nbk, blk, ne, W, S, torch.bfloat16, True, seed = 7)
    ref = pf.clone()
    base.flush(ref, A, Bop, fA, fB, pfids, pids, params, slots, 0, True)
    for cfg, (bm, bn, bk) in enumerate(base.flush_configs()):
        if blk % bm or ne % bn or W % bk:
            continue
        out = pf.clone()
        base.flush(out, A, Bop, fA, fB, pfids, pids, params, slots, cfg, True)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int32), ref.view(torch.int32)), f"config {cfg}: the flows differ"


@needs_kernels
def test_a_group_picks_and_runs_the_producer_warp_flush(monkeypatch):
    """A tied group's flush at the producer-warp copy (`_Group._pick_cfg`'s tuner steered to it): that module's flush
    runs, and the same circuit's backward with the group's flush set to the flush module's 128 x 128 config gives the
    group's flows to the bit (one circuit: two circuits' other tuners may take arms of other bits)."""
    import pyjuice as juice
    import pyjuice.layer.kernels.c as ck
    from pyjuice.layer import deferred_par as dpar
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    base, ws = _mods()
    real_tune = ck.autotune
    monkeypatch.setattr(ck, "autotune", lambda cands, *a, **k: next(
        (key for key, _ in cands if isinstance(key, tuple) and key[0] is ws), None) or real_tune(cands, *a, **k))
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 6, num_latents = 1024, num_emits = 32, homogeneous = True, block_size = 1024)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    x = torch.randint(0, 32, (128, 6), device = DEV, generator = torch.Generator(device = DEV).manual_seed(1))
    calls, real = [], ws.flush
    monkeypatch.setattr(ws, "flush", lambda *a: (calls.append(1), real(*a))[1])

    def backward():
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
        torch.cuda.synchronize()

    backward()
    (group,) = [g for g in dpar._groups_of(pc.param_flows, create = False).values() if g.cfg is not None]
    assert group.cfg[0] is ws and calls, "the group did not take the producer-warp flush"
    lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
    backward()
    a = pc.param_flows[lo:hi].clone()
    group.cfg = (base, base.flush_configs().index([128, 128, 64]))
    backward()
    assert torch.equal(pc.param_flows[lo:hi].view(torch.int32), a.view(torch.int32)), "the two flush modules' flows differ"
