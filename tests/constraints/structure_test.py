import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.constraints.structure import analyze_structure


def hmm(n, seed = 0):
    torch.manual_seed(seed)
    return juice.structures.HMM(seq_length = n, num_latents = 4, num_emits = 5)


def fragmented_circuit(input_dist = None):
    """root over {0,1,2,3} = sum(prod(sum(prod(x0, x2)), sum(prod(x1, x3)))): the inner scopes {0,2}
    and {1,3} each have two runs."""
    mk = lambda v, d = None: inputs(v, num_node_blocks = 1, block_size = 2,
                                    dist = d if d is not None else dists.Categorical(num_cats = 3))
    x0, x1, x2, x3 = mk(0), mk(1, input_dist), mk(2), mk(3)
    s02 = summate(multiply(x0, x2), num_node_blocks = 1, block_size = 2)
    s13 = summate(multiply(x1, x3), num_node_blocks = 1, block_size = 2)
    return summate(multiply(s02, s13), num_node_blocks = 1, block_size = 1)


def test_hmm_is_right_linear():
    n = 6
    st = analyze_structure(hmm(n))
    assert st.num_vars == n and st.right_linear and st.contiguous and st.max_runs == 1
    assert st.unsupported == ()
    for info in st.nodes:
        (a, b), = info.scope_runs
        if info.kind == "input":
            assert a == b
        else:
            assert b == n - 1                               # every sum / product covers a suffix
            assert info.shape == ("whole" if a == 0 else "suffix")
    assert st.nodes[-1].shape == "whole"                    # the root comes last


def test_1d_pd_is_contiguous_but_not_right_linear():
    n = 8
    ns = juice.structures.PD(data_shape = (n,), num_latents = 4, split_intervals = 1,
                             input_node_params = {"num_cats": 5})
    st = analyze_structure(ns)
    assert st.contiguous and st.max_runs == 1 and not st.right_linear
    shapes = {info.shape for info in st.nodes if info.kind != "input"}
    assert {"interval", "prefix", "suffix", "whole"} <= shapes
    # several products split the same interval at different points (PD is not structured decomposable)
    splits = {}
    for info in st.nodes:
        if info.kind == "prod":
            chs = tuple(sorted(analyze_structure(ns).node(c).scope_runs for c in info.ns.chs))
            splits.setdefault(info.scope_runs, set()).add(chs)
    assert any(len(s) > 1 for s in splits.values())


def test_hclt_and_a_hand_built_circuit_are_fragmented():
    torch.manual_seed(0)
    x = torch.randint(0, 5, (256, 8))
    st = analyze_structure(juice.structures.HCLT(x, num_latents = 4, input_node_params = {"num_cats": 5}))
    assert not st.contiguous and st.max_runs > 1

    st = analyze_structure(fragmented_circuit())
    runs = {info.scope_runs: info.shape for info in st.nodes}
    assert runs[((0, 0), (2, 2))] == "fragmented" and runs[((1, 1), (3, 3))] == "fragmented"
    assert runs[((0, 0),)] == "prefix" and runs[((1, 1),)] == "interval" and runs[((3, 3),)] == "suffix"
    assert runs[((0, 3),)] == "whole"
    assert st.max_runs == 2 and not st.contiguous and not st.right_linear


def test_signature_ignores_parameters_but_not_structure():
    a, b = hmm(6, seed = 0), hmm(6, seed = 1)
    pa, pb = juice.compile(a), juice.compile(b)
    assert not torch.equal(pa.params, pb.params)            # different parameters ...
    assert analyze_structure(pa).signature == analyze_structure(pb).signature   # ... same structure
    assert analyze_structure(a).signature == analyze_structure(pa).signature
    assert analyze_structure(hmm(7)).signature != analyze_structure(a).signature
    assert analyze_structure(fragmented_circuit()).signature != analyze_structure(a).signature
    # cached per compiled PC
    assert analyze_structure(pa) is analyze_structure(pa)


def test_unsupported_features_are_reported():
    st = analyze_structure(fragmented_circuit(input_dist = dists.Gaussian(mu = 0.0, sigma = 1.0)))
    assert len(st.unsupported) == 1
    ns, reason = st.unsupported[0]
    assert ns.is_input() and "Gaussian" in reason
