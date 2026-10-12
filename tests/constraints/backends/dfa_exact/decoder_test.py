"""
Decoding sessions (:meth:`ConstrainedCircuit.decoder`) on the dfa_exact backend: a scenario's decoder calls give what
the circuit's own calls give -- left to right, any order, beams reordered and expanded, a new row, a position observed
then missing again; candidates, targets as positions, a mask id, the marginal -- and the session is a fast_inference
scope (its options reach it, it nests), one per circuit at a time, usable only inside its ``with``, with query
defaults a call overrides.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.utils.fast_inference import is_active

V = 3
KINDS = {"hmm": 6, "pd": 8, "hand_left": 5, "left_linear": 6}                  # see conftest.py


def assert_close(got, want, n, where = None):
    """Log-probabilities: the same -inf entries, the finite ones within fp32 rounding."""
    got, want = got.cpu(), want.cpu()
    assert got.shape == want.shape and not torch.isnan(got).any(), where
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), where
    fin = torch.isfinite(want)
    assert ((got[fin] - want[fin]).abs() <= 1e-5 * want[fin].abs() + n * 1e-6).all(), where


def left_to_right(n, B = 4, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    return [dict(data = data[:, :t], targets = [t]) for t in range(n)]


def any_order(n, B = 4, seed = 1):
    """Every sample unmasks its positions in an order of its own, one per call."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    order = torch.stack([torch.randperm(n, generator = g) for _ in range(B)])
    steps = []
    for s in range(n):
        missing = torch.ones(B, n, dtype = torch.bool)
        missing.scatter_(1, order[:, :s], False)
        steps.append(dict(data = data, missing_mask = missing))
    return steps


def beams(n, seed = 2):
    """Rows reordered, duplicated and dropped between calls, each then observing the next position."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (2, n), generator = g)
    steps = []
    for t in range(n):
        steps.append(dict(data = data[:, :t].clone(), targets = [t]))
        idx = torch.randint(0, data.size(0), (2 + t % 3,), generator = g)
        data = data[idx].clone()
        data[:, t] = torch.randint(0, V, (data.size(0),), generator = g)
    return steps


def new_and_forgotten(n, seed = 3):
    """A row replaced by a new prompt; a position observed, then missing again."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (3, n), generator = g)
    steps = [dict(data = data[:, :2]), dict(data = data[:, :3])]
    fresh = data.clone()
    fresh[1] = torch.randint(0, V, (n,), generator = g)
    steps.append(dict(data = fresh[:, :3]))
    missing = torch.zeros(3, n, dtype = torch.bool)
    missing[:, 3:] = True
    missing[0, 1] = True                                                      # observed before, missing now
    steps.append(dict(data = fresh, missing_mask = missing))
    return steps


def candidates_and_mask_id(n, B = 3, seed = 4):
    """Candidates per sample, and the evidence marked with a mask id of 7."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    steps = []
    for t in range(n):
        masked = data.clone()
        masked[:, t:] = 7
        steps.append(dict(data = masked, mask_id = 7, candidates = torch.randint(-1, V, (B, 4), generator = g)))
    return steps


SCENARIOS = {"left to right": left_to_right, "any order": any_order, "beams": beams,
             "new and forgotten": new_and_forgotten, "candidates and mask id": candidates_and_mask_id}


def ask(target, query, step):
    if query == "marginal":
        return target.marginal(step["data"], step.get("missing_mask"), **({"mask_id": step["mask_id"]}
                                                                          if "mask_id" in step else {}))
    return target.conditional(**step)


@pytest.mark.parametrize("query", ["conditional", "marginal"])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_decoders_calls_are_the_circuits(kind, scenario, query, build_pc):
    n = KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    steps = SCENARIOS[scenario](n)
    if query == "marginal":
        steps = [{k: v for k, v in s.items() if k in ("data", "missing_mask", "mask_id")} for s in steps]
    with juice.fast_inference():
        want = [ask(cc, query, s) for s in steps]
    with cc.decoder() as decoder:
        got = [ask(decoder, query, s) for s in steps]
    for i, (a, b) in enumerate(zip(got, want)):
        assert_close(a, b, n, where = f"step {i}")


def test_the_session_is_a_fast_inference_scope(build_pc):
    """Open inside, closed after; its options reach the scope (bf16 token tables); it nests in a caller's scope."""
    n = KINDS["hmm"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    data = torch.randint(0, V, (2, n))
    assert not is_active()
    with cc.decoder(allow_bf16_params = True) as decoder:
        assert is_active()
        decoder.conditional(data[:, :2])
        assert all(t.dtype == torch.bfloat16 for t in cc.backend.state._token_tables.values())
    assert not is_active() and cc.backend.state._token_tables is None
    with juice.fast_inference():
        with cc.decoder() as decoder:
            decoder.conditional(data[:, :2])
        assert is_active()


def test_one_session_at_a_time_inside_its_with(build_pc):
    n = KINDS["hmm"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    data = torch.randint(0, V, (2, n))
    decoder = cc.decoder()
    with pytest.raises(RuntimeError, match = "with"):
        decoder.conditional(data[:, :2])
    with decoder:
        with pytest.raises(RuntimeError, match = "open decoder"):
            with cc.decoder():
                pass
    with pytest.raises(RuntimeError, match = "with"):
        decoder.conditional(data[:, :2])
    with pytest.raises(RuntimeError, match = "once"):
        with decoder:
            pass
    with cc.decoder() as other:                                               # a new one opens fine
        other.marginal(data[:, :2])


def test_query_defaults_and_a_calls_own_options(build_pc):
    """Options other than fast_inference's are defaults for every call (precision, mask_id); a call's own wins."""
    n = KINDS["pd"]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("pd", n, V))
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V, (3, n), generator = g)
    data[:, n // 2:] = 7
    with juice.fast_inference():
        tf32 = cc.conditional(data, mask_id = 7, precision = "tf32")
        fp32 = cc.conditional(data, mask_id = 7)
    with cc.decoder(precision = "tf32", mask_id = 7) as decoder:
        assert torch.equal(decoder.conditional(data), tf32)
        assert torch.equal(decoder.conditional(data, precision = "fp32"), fp32)
        assert torch.equal(decoder.conditional(torch.where(data == 7, -1, data), mask_id = -1), tf32)
