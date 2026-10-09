import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
import pyjuice.constraints as jc
from pyjuice.constraints.structure import analyze_structure
from pyjuice.constraints.backends.lifted import plan
from pyjuice.constraints.backends.lifted.plan import build_layout


V = 5


def hmm(n, seed = 0):
    torch.manual_seed(seed)
    return juice.compile(juice.structures.HMM(seq_length = n, num_latents = 4, num_emits = V), verbose = False)


def fragmented_circuit(input_dist = None):
    """root over {0,1,2,3} = sum(prod(sum(prod(x0, x2)), sum(prod(x1, x3)))): the inner scopes {0,2}
    and {1,3} each have two runs."""
    mk = lambda v, d = None: inputs(v, num_node_blocks = 1, block_size = 2,
                                    dist = d if d is not None else dists.Categorical(num_cats = V))
    x0, x1, x2, x3 = mk(0), mk(1, input_dist), mk(2), mk(3)
    s02 = summate(multiply(x0, x2), num_node_blocks = 1, block_size = 2)
    s13 = summate(multiply(x1, x3), num_node_blocks = 1, block_size = 2)
    return juice.compile(summate(multiply(s02, s13), num_node_blocks = 1, block_size = 1), verbose = False)


class OnlyAccepts(jc.Constraint):
    """A constraint with membership only (no automaton)."""

    def accepts(self, tokens):
        return True

    def _fingerprint_payload(self):
        return ()


def refusal(constraint, pc, **kwargs):
    with pytest.raises(jc.ConstraintCompileError) as e:
        jc.compile(constraint, pc, **kwargs)
    return str(e.value)


def test_hmm_compiles_with_the_lifted_backend():
    n = 6
    pc = hmm(n)
    c = jc.DFA.contains([[1, 2]], vocab_size = V)
    cc = jc.compile(c, pc)

    assert isinstance(cc, jc.ConstrainedCircuit)
    assert cc.pc is pc and cc.constraint is c and cc.structure is analyze_structure(pc)
    assert cc.backend == "lifted" and cc.exact and cc.n == n and cc.satisfiable
    assert cc.num_states == c.automaton().num_states and cc.num_classes == c.automaton().num_classes
    assert torch.equal(cc.width_per_boundary, build_layout(c.automaton(), n).width)
    assert cc.max_width == int(cc.width_per_boundary.max())

    counts = cc.shape_counts
    num_inner = sum(info.kind != "input" for info in cc.structure.nodes)
    assert sum(counts.values()) == num_inner
    assert counts["whole"] >= 1 and counts["suffix"] > 0
    assert counts["prefix"] == counts["interval"] == counts["fragmented"] == 0
    assert cc.compile_time_s > 0
    assert "lifted" in repr(cc) and cc.info()["width_per_boundary"] == cc.width_per_boundary.tolist()


def test_1d_pd_compiles_with_interval_scopes():
    pc = juice.compile(juice.structures.PD(data_shape = (8,), num_latents = 4, split_intervals = 1,
                                           input_node_params = {"num_cats": V}), verbose = False)
    cc = jc.compile(jc.DFA.contains([[1, 2]], vocab_size = V), pc)
    assert cc.shape_counts["interval"] > 0 and cc.shape_counts["fragmented"] == 0


def test_columns_and_memory_on_a_hand_counted_circuit():
    # every group below has 2 nodes except the root (1); under "contains 1 1" the widths per boundary
    # are 1, 2, 3, 3, 2, 1, so the blocks (entry x exit columns, a sequence end counting as one) are:
    #   prod(x1, x2), s12 over [1, 2]         -> interval, 2 x 3 = 6
    #   prod(x0, s12), s02 over [0, 2]        -> prefix,   1 x 3 = 3
    #   prod(x3, x4), s34 over [3, 4]         -> suffix,   3 x 1 = 3
    #   prod(s02, s34), root over [0, 4]      -> whole,    1
    mk = lambda *a: summate(multiply(*a), num_node_blocks = 1, block_size = 2)
    x = [inputs(v, num_node_blocks = 1, block_size = 2, dist = dists.Categorical(num_cats = 3)) for v in range(5)]
    s02 = mk(x[0], mk(x[1], x[2]))
    pc = juice.compile(summate(multiply(s02, mk(x[3], x[4])), num_node_blocks = 1, block_size = 1), verbose = False)
    cc = jc.compile(jc.DFA.contains([[1, 1]], vocab_size = 3), pc)
    assert cc.shape_counts == dict(whole = 2, suffix = 2, prefix = 2, interval = 2, fragmented = 0)
    assert cc.width_per_boundary.tolist() == [1, 2, 3, 3, 2, 1] and cc.num_classes == 2
    assert cc.columns_per_sample == 6
    # per sample: the sum node groups' blocks (2 x 6 + 2 x 3 + 2 x 3 + 1 x 1 = 25), the largest product layer
    # (x1 * x2 and x3 * x4: 4 nodes x 6 = 24) and the input rows (2 dummy rows, then the 10 input nodes)
    assert cc.input_range == (2, 12)
    assert cc.bytes_per_sample == 4 * (25 + 24 + 12)


def test_fragmented_scopes_are_refused_with_a_clear_message():
    msg = refusal(jc.DFA.contains([[1, 2]], vocab_size = V), fragmented_circuit())
    assert "the node group over variables {0, 2} has 2 separate runs" in msg
    assert "4 node groups are fragmented, with up to 2 runs" in msg
    assert "K = 3 states and r = 2 that is 81 per node" in msg

    torch.manual_seed(0)
    hclt = juice.compile(juice.structures.HCLT(torch.randint(0, V, (256, 12)), num_latents = 4,
                                               input_node_params = {"num_cats": V}), verbose = False)
    max_runs = analyze_structure(hclt).max_runs
    assert max_runs > 1
    msg = refusal(jc.DFA.contains([[1, 2]], vocab_size = V), hclt)
    assert f"has {max_runs} separate runs" in msg and "index order 0..11" in msg


def test_unsupported_input_nodes_are_refused():
    pc = juice.compile(juice.structures.GeneralizedHMM(seq_length = 4, num_latents = 4, homogeneous = False,
                                                       input_dist = dists.Gaussian(mu = 0.0, sigma = 1.0)),
                       verbose = False)
    msg = refusal(jc.DFA.anything(V), pc)
    assert "4 node groups are not supported: input distribution Gaussian" in msg


def test_a_vocabulary_mismatch_is_refused():
    msg = refusal(jc.DFA.contains([[1, 2]], vocab_size = V + 2), hmm(6))
    assert f"vocab_size {V + 2}" in msg and f"variables {{0..5}} have num_cats {V}" in msg


def test_a_constraint_without_an_automaton_is_refused():
    msg = refusal(OnlyAccepts(V), hmm(6))
    assert "no `automaton` capability" in msg and "it supports: accepts" in msg


def test_every_reason_is_reported_at_once():
    msg = refusal(OnlyAccepts(V), fragmented_circuit(input_dist = dists.Gaussian(mu = 0.0, sigma = 1.0)))
    lines = [line for line in msg.splitlines() if line.startswith("  - ")]
    assert len(lines) == 3
    assert "no `automaton` capability" in lines[0]
    assert "Gaussian" in lines[1]
    assert "separate runs" in lines[2] and "K = " not in lines[2]       # no automaton, so no state count


def test_an_unknown_backend_or_wrong_types_are_refused():
    pc = hmm(6)
    with pytest.raises(jc.ConstraintCompileError, match = "known backends: 'lifted'"):
        jc.compile(jc.DFA.anything(V), pc, backend = "smc")
    with pytest.raises(TypeError, match = "Constraint"):
        jc.compile("abc", pc)
    with pytest.raises(TypeError, match = "pyjuice.compile"):
        jc.compile(jc.DFA.anything(V), pc.root_ns)


def test_an_unsatisfiable_constraint_compiles():
    n = 4
    for c in (jc.DFA.nothing(V), jc.DFA.contains([[1, 2, 3, 4, 1]], vocab_size = V)):    # pattern longer than n
        cc = jc.compile(c, hmm(n))
        assert not cc.satisfiable
        assert cc.max_width == 0 and cc.width_per_boundary.tolist() == [0] * (n + 1)


def test_with_pc_rebinds_to_the_same_structure_only():
    pa, pb = hmm(6, seed = 0), hmm(6, seed = 1)
    assert not torch.equal(pa.params, pb.params)
    cc = jc.compile(jc.DFA.contains([[1, 2]], vocab_size = V), pa)

    cb = cc.with_pc(pb)
    assert cb.pc is pb and cb.structure is analyze_structure(pb)
    assert cb.constraint is cc.constraint and cb.automaton is cc.automaton and cb.layout is cc.layout

    with pytest.raises(jc.ConstraintCompileError, match = "structure differs"):
        cc.with_pc(hmm(7))
    with pytest.raises(TypeError, match = "pyjuice.compile"):
        cc.with_pc(pb.root_ns)


def test_compiling_releases_the_pcs_activation_buffers():
    pc = hmm(6).to(torch.device("cuda:0"))
    x = torch.randint(0, V, (4, 6), device = pc.device)
    pc.init_param_flows(flows_memory = 0.0)
    lls = pc(x).clone()
    pc.backward(x, allow_modify_flows = False)
    param_flows = pc.param_flows.clone()
    cc = jc.compile(jc.DFA.contains([[1, 2]], vocab_size = V), pc)
    assert not hasattr(pc, "node_mars") and not hasattr(pc, "element_flows")
    assert torch.equal(pc.param_flows, param_flows)                 # EM statistics are kept
    assert torch.equal(pc(x), lls)                                  # the PC still works on its own
    cc.with_pc(pc)                                                  # rebinding releases them again
    assert not hasattr(pc, "node_mars")


def test_the_pc_is_fixed_once_compiled():
    pc = hmm(6)                                                    # on the CPU
    cc = jc.compile(jc.DFA.contains([[1, 2]], vocab_size = V), pc)
    with pytest.raises(AttributeError):
        cc.pc = hmm(6)                                             # switching PCs goes through `with_pc`
    with pytest.raises(NotImplementedError):
        cc.marginal()                                              # (queries are not implemented yet)
    pc.to(torch.device("cuda:0"))                                  # the PC moved on its own
    for query in (cc.marginal, cc.conditional, cc.sample, cc.decoder):
        with pytest.raises(RuntimeError, match = "moved from cpu to cuda:0 on its own.*cc.to"):
            query()
    with pytest.raises(NotImplementedError):                       # recompiling (or with_pc) fixes it
        cc.with_pc(pc).marginal()


def test_to_moves_the_pc_and_the_tables_together():
    cc = jc.compile(jc.DFA.contains([[1, 2]], vocab_size = V), hmm(6))
    for device in (torch.device("cuda:0"), torch.device("cpu"), "cuda:0"):
        assert cc.to(device) is cc
        want = torch.device(device)
        assert cc.pc.params.device.type == want.type
        assert all(t.device == cc.pc.params.device for layer in cc.product_rows for rows in layer.values() for t in rows)
        with pytest.raises(NotImplementedError):                   # the guard passes: queries may run
            cc.marginal()


def test_the_plan_does_not_depend_on_parameters():
    c = jc.DFA.contains([[1, 2]], vocab_size = V)
    pa, pb = hmm(6, seed = 0), hmm(6, seed = 1)
    plan._CACHE.clear()
    ca = jc.compile(c, pa)
    tables = {k: v.clone() for k, v in vars(ca.layout).items() if isinstance(v, torch.Tensor)}
    info = {k: v for k, v in ca.info().items() if k != "compile_time_s"}

    with torch.no_grad():
        pa.params.uniform_(0.1, 1.0)                    # change the parameters in place
    plan._CACHE.clear()                                 # build the layout again, from scratch
    for cc in (jc.compile(c, pa), jc.compile(c, pb)):
        assert cc.layout is not ca.layout
        for k, v in tables.items():
            assert torch.equal(getattr(cc.layout, k), v), k
        assert {k: v for k, v in cc.info().items() if k != "compile_time_s"} == info
        plan._CACHE.clear()
    # ... and the first compiled object is untouched
    for k, v in tables.items():
        assert torch.equal(getattr(ca.layout, k), v), k


def test_compile_is_exported_without_shadowing_a_module():
    import pyjuice.constraints.compiler as compiler_module         # a module, not the function
    assert compiler_module.compile is jc.compile and juice.constraints.compile is jc.compile
