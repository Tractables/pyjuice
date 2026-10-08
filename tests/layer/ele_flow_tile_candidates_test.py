import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.layer.sum_layer as sl
from pyjuice.layer.kernels import autotune


# The element-flow backward's autotuner picks among tile configs `(TILE_SIZE_K, TILE_SIZE_M, BLOCK_B)` by
# timing, so ANY of them may run on some GPU. Force each in turn and require it to be no less accurate
# than the heuristic's own config, against the exact fp32 sparse kernels: the configs regroup the parent
# reduction (in the tensor-core dot regime too), so they differ from each other by rounding, not by more
# than any of them already differs from the exact flows. That includes the second tile family (a
# different `TILE_SIZE_K`), which the test checks is actually on offer.


def _structures():
    torch.manual_seed(0)
    yield "hmm", juice.structures.GeneralizedHMM(seq_length = 6, num_latents = 128, homogeneous = True,
                                                 input_dist = dists.Categorical(num_cats = 20)), 20
    x = torch.randint(0, 16, (500, 12))
    yield "hclt", juice.structures.HCLT(x, num_latents = 128, input_dist = dists.Categorical(num_cats = 16)), 16
    yield "pd", juice.structures.PD(data_shape = (4, 4), num_latents = 64, split_intervals = (2, 2),
                                    input_dist = dists.Categorical(num_cats = 8)), 8


@pytest.mark.parametrize("batch_size", [17, 33, 100])
def test_every_element_flow_tile_config_gives_the_same_flows(batch_size, monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(sl, "BACKWARD_ELE_FLOW_CUDA", False)     # the Triton configs are what is tuned
    forced = [0]
    offered = []
    real_pick = autotune.pick

    ele_kernels = [getattr(sl.bk_ele_bsparse, name) for name in dir(sl.bk_ele_bsparse)
                   if name.startswith("_bk_triton_block_sparse") and "ele" in name]

    def pick(key, candidates, bench, *args, **kwargs):
        if any(key[0] is k for k in ele_kernels):
            offered.append(list(candidates))
            return candidates[min(forced[0], len(candidates) - 1)]
        return real_pick(key, candidates, bench, *args, **kwargs)

    monkeypatch.setattr(autotune, "pick", pick)
    for name, ns, num_cats in _structures():
        pc = juice.compile(ns, verbose = False).to(device)
        data = torch.randint(0, num_cats, (batch_size, pc.num_vars), device = device)

        pc.init_param_flows(flows_memory = 0.0)
        pc(data, mode = "sparse")
        pc.backward(data, mode = "sparse", allow_modify_flows = False, logspace_flows = True)
        exact_nf, exact_pf = pc.node_flows.double().clone(), pc.param_flows.double().clone()
        finite = torch.isfinite(exact_nf)

        out = []
        c = 0
        while c == 0 or c < max(len(cands) for cands in out[0][2]):      # every config any layer offers
            forced[0] = c
            offered.clear()
            pc.init_param_flows(flows_memory = 0.0)
            pc(data)
            pc.backward(data, allow_modify_flows = False, logspace_flows = True)
            nf, pf = pc.node_flows.double(), pc.param_flows.double()
            assert torch.equal(torch.isfinite(nf), finite), f"{name} batch {batch_size} config {c}: -inf pattern"
            out.append(((nf - exact_nf)[finite].abs().max().item(),
                        (pf - exact_pf).abs().max().item() / exact_pf.abs().max().item(),
                        [list(o) for o in offered]))
            c += 1

        families = {cfg[0] for cands in out[0][2] for cfg in cands}
        if name == "hmm":
            assert len(families) > 1, f"{name}: only one edge tile on offer ({families}) -- the test is vacuous"
        ref_nf_err, ref_pf_err = out[0][0], out[0][1]
        for c, (nf_err, pf_err, _) in enumerate(out[1:], start = 1):
            assert nf_err <= 1.5 * ref_nf_err + 1e-6, \
                f"{name} batch {batch_size} config {c}: node-flow error {nf_err:.2e} vs the heuristic's {ref_nf_err:.2e}"
            assert pf_err <= 1.5 * ref_pf_err + 1e-6, \
                f"{name} batch {batch_size} config {c}: param-flow error {pf_err:.2e} vs the heuristic's {ref_pf_err:.2e}"
