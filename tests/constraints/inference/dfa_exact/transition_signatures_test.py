"""
Transition masses by column signature (kernels/transition_signatures.py, ``Program.signature_layout``): a query's one
set of transition masses built once per distinct column instead of once per boundary column. On an HMM long enough
that boundaries repeat their columns: the layout is taken; a decoder session (l2r and diffusion, dense and compact
values) gives the per-column layout's results to the bit and the float64 reference's to strong_check's bars; a
budget that the per-column masses exceed and the signature ones do not switches the query to the grouped kernels,
still within the bars; per-sample masses (Codpula's soft evidence) keep the per-column layout.
"""
import importlib.util
import os

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels import transition_signatures as TS

_spec = importlib.util.spec_from_file_location("strong_check_test", os.path.join(os.path.dirname(__file__), "strong_check_test.py"))
strong = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strong)
Float64Reference, check_against_reference, run_session, schedule, small_case, hmm_pc, keyword_dfa = (
    strong.Float64Reference, strong.check_against_reference, strong.run_session, strong.schedule, strong.small_case,
    strong.hmm_pc, strong.keyword_dfa)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def long_hmm(n = 24, latents = 64, V = 40, B = 4, seed = 0):
    """small_case's HMM and automaton over ``n`` positions: past the shortest accepted string, boundaries repeat their
    columns. (Its few classes do not make the grouped tables worth it: the tests force them, ``group_worth``.)"""
    phrases = [[1, 2], [3]]
    pc = hmm_pc(n, latents, V, seed)
    dfa = keyword_dfa(V, phrases, forbidden = 0)
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(4, V, (B, n), generator = g)
    data[:, 2:4], data[:, 5] = torch.tensor(phrases[0]), phrases[1][0]
    return pc, dfa, data


@cuda_only
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("mode", ["l2r", "diffusion"])
def test_signature_masses_give_the_per_column_ones_to_the_bit(mode, storage, monkeypatch):
    pc, dfa, data = long_hmm()
    cc = jc.compile(dfa, pc)
    monkeypatch.setattr(cc.backend.program(), "group_worth", True)
    masks = schedule(mode, data.size(0), data.size(1))
    monkeypatch.setattr(TS, "SIGNATURE_MASSES", True)
    check_against_reference(cc, Float64Reference(pc, dfa, data.size(1)), data, masks, None, storage, "fp32")
    prog = cc.backend.program()
    assert prog.trans.grouped and prog.trans_sig is not None, "the signature layout was not taken"
    assert prog.trans_sig.saving >= TS.SIGNATURE_MIN_SAVING
    sig = run_session(cc, data, masks, None, storage, "fp32")
    monkeypatch.setattr(TS, "SIGNATURE_MASSES", False)
    col = run_session(cc, data, masks, None, storage, "fp32")
    assert prog.trans_sig is None
    for i in sig:
        for u, v in zip(sig[i], col[i]):
            assert torch.equal(u, v), f"call {i}: the signature masses differ from the per-column ones"


@cuda_only
@pytest.mark.parametrize("mode", ["l2r", "diffusion"])
def test_a_budget_only_the_signature_masses_fit_takes_the_grouped_kernels(mode, monkeypatch):
    """The per-column masses over the budget, the signature ones under it: ungrouped without signatures, grouped with
    them -- both within the float64 reference's bars (the two paths sum alike terms in different orders)."""
    pc, dfa, data = long_hmm()
    cc = jc.compile(dfa, pc)
    prog = cc.backend.program()
    monkeypatch.setattr(prog, "group_worth", True)
    masks = schedule(mode, data.size(0), data.size(1))
    ref = Float64Reference(pc, dfa, data.size(1))
    monkeypatch.setattr(TS, "SIGNATURE_MASSES", True)
    run_session(cc, data, masks[:1], None, "compact", "fp32")
    sig = prog.trans_sig
    assert sig is not None and sig.size < prog.trans_size
    monkeypatch.setattr(prog, "trans_budget", (sig.size + prog.trans_size) // 2)
    for on, grouped in ((False, False), (True, True)):
        monkeypatch.setattr(TS, "SIGNATURE_MASSES", on)
        check_against_reference(cc, ref, data, masks, None, "compact", "fp32")
        assert prog.trans.grouped == grouped, (on, prog.trans.grouped)


@cuda_only
def test_per_sample_masses_keep_the_per_column_layout(monkeypatch):
    pc, dfa, data, inputs = small_case("codpula")
    cc = jc.compile(dfa, pc)
    monkeypatch.setattr(TS, "SIGNATURE_MASSES", True)
    run_session(cc, data, schedule("l2r", data.size(0), data.size(1)), inputs, "compact", "fp32")
    assert cc.backend.program().trans_sig is None
