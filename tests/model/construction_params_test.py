"""
A new circuit's parameters: normalized on the GPU from a size on, to what the CPU gives and to the same bits on every
run; no random vector drawn for slots the nodes overwrite, while slots no node writes still start random.
"""

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.model.build as build
import pyjuice.nodes.distributions as dists
from pyjuice.model.backend import normalize_parameters
from pyjuice.nodes import inputs, multiply, summate

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _circuit(kind):
    torch.manual_seed(0)
    if kind == "hmm":            # (64 children a node, past a table entry's 32: a node's total takes several entries)
        ns = juice.structures.HMM(seq_length = 6, num_latents = 64, num_emits = 8, homogeneous = False)
    elif kind == "tied":
        ns = juice.structures.HMM(seq_length = 6, num_latents = 64, num_emits = 8, homogeneous = True)
    elif kind == "hclt":
        x = torch.randint(0, 8, (256, 12))
        ns = juice.structures.HCLT(x.float(), num_latents = 16, input_dist = dists.Categorical(num_cats = 8))
    else:
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 8, split_intervals = 2,
                                 input_dist = dists.Categorical(num_cats = 4))
    return ns


def _reference(pc, raw, pseudocount):
    """The normalization in float64, from the tables: an entry is up to 32 children of one node."""
    ps, _, bsz, bint, gnid, nchs, _, _ = pc.par_update_kwargs
    p = raw.double().numpy().copy()
    tot = np.zeros(int(gnid.max()) + 1)
    entries = list(zip(ps.tolist(), bsz.tolist(), bint.tolist(), gnid.tolist()))
    for s, b, i, n in entries:
        tot[n] += p[s + np.arange(b) * i].sum()
    out = p.copy()
    for s, b, i, n in entries:
        idx = s + np.arange(b) * i
        out[idx] = (p[idx] + pseudocount / float(nchs[n])) / (tot[n] + pseudocount)
    return torch.from_numpy(out)


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "tied", "hclt", "pd"])
@pytest.mark.parametrize("pseudocount", [0.0, 0.1])
def test_on_the_gpu_it_is_the_cpus_normalization_and_repeats_to_the_bit(kind, pseudocount):
    pc = juice.compile(_circuit(kind), verbose = False)
    torch.manual_seed(1)
    raw = torch.exp(torch.rand(pc.params.shape) * -4.0)
    raw[:pc.num_dummy_params] = 0.0

    on_cpu = raw.clone()
    normalize_parameters(on_cpu, pc.par_update_kwargs, pseudocount)
    runs = []
    for _ in range(4):
        on_gpu = raw.clone()
        pc._normalize_parameters(on_gpu, pseudocount = pseudocount, use_cuda = True)
        runs.append(on_gpu)

    assert not on_gpu.is_cuda, "the parameters stay where they were"
    assert all(torch.equal(r, runs[0]) for r in runs), "the normalization differs run to run"
    want = _reference(pc, raw, pseudocount)
    assert float((on_cpu.double() - want).abs().max()) < 2e-6
    assert float((runs[0].double() - want).abs().max()) < 2e-6
    assert torch.equal(runs[0][:pc.num_dummy_params], raw[:pc.num_dummy_params])


@cuda_only
def test_a_nodes_total_adds_the_entries_in_the_tables_order():
    """fp32 addition does not associate, so the order of the adds decides the bits; a run of the same tensors may keep
    an atomic order for a while, and repeating cannot tell. One value an entry, its first child's (so an entry's sum is
    that value, however `tl.sum` reduces), spanning 20 nats: each node's total must be the fold in the table's order."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    pc = juice.compile(juice.structures.HMM(seq_length = 6, num_latents = 256, num_emits = 8, homogeneous = False),
                       verbose = False)
    ps, gnid = pc.par_update_kwargs[0], pc.par_update_kwargs[4]
    assert torch.bincount(gnid).max() >= 8, "nodes taking several entries"
    torch.manual_seed(2)
    values = torch.exp(torch.rand(ps.shape) * 20 - 10)
    raw = torch.zeros(pc.params.shape)
    raw[ps] = values
    want = np.zeros(int(gnid.max()) + 1, dtype = np.float32)
    np.add.at(want, gnid.numpy(), values.numpy())

    tables = tuple(x.to(dev) if torch.is_tensor(x) else x for x in pc.par_update_kwargs)
    normalize_parameters(raw.to(dev), tables, 0.0)
    assert torch.equal(tables[6].cpu(), torch.from_numpy(want))


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "tied", "hclt", "pd"])
def test_a_circuit_compiled_either_way_has_the_same_parameters(kind, monkeypatch):
    pcs = []
    for threshold in (0, 2 ** 62):
        monkeypatch.setattr(build, "GPU_NORMALIZE_MIN_PARAMS", threshold)
        pcs.append(juice.compile(_circuit(kind), verbose = False))
    gpu, cpu = pcs
    assert float((gpu.params.data.double() - cpu.params.data.double()).abs().max()) < 2e-6
    for a, b in zip(gpu.input_layer_group, cpu.input_layer_group):
        assert torch.equal(a.params, b.params)


@cuda_only
@pytest.mark.parametrize("flags,threshold,on_gpu", [(dict(), 2 ** 62, False), (dict(), 0, True),
                                                    (dict(disable_gpu_compilation = True), 0, False),
                                                    (dict(force_gpu_compilation = True), 2 ** 62, True)],
                         ids = ["small", "large", "disabled", "forced"])
def test_where_it_runs(flags, threshold, on_gpu, monkeypatch):
    seen = []
    real = build.normalize_parameters
    monkeypatch.setattr(build, "normalize_parameters", lambda p, *a, **k: seen.append(p.is_cuda) or real(p, *a, **k))
    monkeypatch.setattr(build, "GPU_NORMALIZE_MIN_PARAMS", threshold)
    pc = juice.compile(_circuit("hmm"), verbose = False, **flags)
    assert seen == [on_gpu]
    assert not pc.params.is_cuda


@cuda_only
def test_no_random_vector_is_drawn_for_slots_the_nodes_overwrite(monkeypatch):
    root = _circuit("hmm")
    sizes, real = [], torch.rand
    monkeypatch.setattr(torch, "rand", lambda size, *a, **k: sizes.append(list(size)) or real(size, *a, **k))
    pc = juice.compile(root, verbose = False)
    assert sizes, "the nodes' own parameters are drawn"
    assert [pc.num_sum_params] not in sizes


@cuda_only
def test_slots_no_node_writes_still_start_random(monkeypatch):
    """A tied node whose source is outside the circuit: its range belongs to no node the circuit walks."""
    torch.manual_seed(0)
    with juice.set_block_size(4):
        a = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = 4))
        b = inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = 4))
        source = summate(multiply(a, b), num_node_blocks = 1)
        c = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = 4))
        d = inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = 4))
        copy = source.duplicate(multiply(c, d), tie_params = True)
        root = summate(multiply(copy), num_node_blocks = 1, block_size = 1)
    pc = juice.compile(root, verbose = False)
    lo, hi = copy._param_range
    theta = pc.params.data[lo:hi]
    assert torch.all(theta > 0) and torch.all(torch.isfinite(theta))
    assert theta.unique().numel() > 1
