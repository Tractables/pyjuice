"""Regression tests for the soft-evidence caches being keyed by tensor IDENTITY, not by address.

Both `_fw_linear_evidence` and `_build_dense_index` memoize a derived table on the layer so that the
two `pc()` calls of one step -- and the forward and backward of one step -- share a single build. That
memo used to be keyed on `(data_ptr(), _version, shape)`, which is not an identity: the caching
allocator hands a freed block straight back out, so under `torch.no_grad()`, where a step's evidence
dies as soon as the forward returns, the next call's tensors land on the same addresses with the same
shapes and a fresh `_version` of 0. The key then matched byte for byte while the data was entirely
different, and the circuit was silently evaluated against the PREVIOUS call's evidence -- which can
return a positive log-likelihood.

Under autograd the graph keeps the evidence alive, so the addresses differ and the collision never
arises. That is why this only ever appeared in evaluation loops.
"""

import torch
import pytest

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes.distributions.softevi_categorical import (
    _evidence_cache_key, _evidence_cache_hit, _fw_linear_evidence,
)


def _realloc_at_same_address(shape, dtype, device, make):
    """`(first, second)` where `second` reuses the block `first` was freed from, or `None`.

    The caching allocator hands back the most recently freed block of a matching size, so this is
    reliable in practice; it is still reported rather than asserted, since it is an allocator policy
    and not a guarantee.
    """
    first = make()
    ptr = first.data_ptr()
    del first
    second = make()
    return second if second.data_ptr() == ptr else None


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_evidence_cache_key_rejects_recycled_address(device):
    """A new tensor on a recycled address must MISS, even though every scalar in the old key matches."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device(device)

    a = torch.zeros([8, 4, 16], device = dev)
    key = _evidence_cache_key(a)
    assert _evidence_cache_hit(key, (a,)), "the same live tensor must hit"

    ptr = a.data_ptr()
    del a
    b = torch.ones([8, 4, 16], device = dev)

    # The old `(data_ptr, _version, shape)` key would be identical here whenever the allocator recycles.
    if b.data_ptr() == ptr:
        assert b._version == 0
        assert not _evidence_cache_hit(key, (b,)), \
            "a different tensor reusing the freed address must not be served from the cache"
    assert _evidence_cache_hit(_evidence_cache_key(b), (b,))


def test_evidence_cache_key_catches_inplace_mutation():
    """The same object mutated in place must miss -- `_version` still has to be part of the key."""
    a = torch.zeros([4, 4])
    key = _evidence_cache_key(a)
    assert _evidence_cache_hit(key, (a,))

    a.add_(1.0)
    assert not _evidence_cache_hit(key, (a,))


def test_fw_linear_evidence_not_served_across_a_recycled_block():
    """`exp(evidence)` for a recycled address must be recomputed, not served from the previous call."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device("cuda:0")

    class _Layer:
        pass

    layer = _Layer()

    first = torch.full([4, 2, 32], -1.0, device = dev)
    got = _fw_linear_evidence(layer, {"categorical_evidence_logp": first})
    assert torch.allclose(got, first.exp())

    ptr = first.data_ptr()
    del first, got

    second = torch.full([4, 2, 32], -3.0, device = dev)
    got = _fw_linear_evidence(layer, {"categorical_evidence_logp": second})
    if second.data_ptr() == ptr:
        assert not torch.allclose(got, torch.full_like(got, torch.tensor(-1.0).exp().item())), \
            "the cache served the previous call's exp(evidence) for a recycled address"
    assert torch.allclose(got, second.exp())


def test_fw_linear_evidence_still_hits_for_the_same_tensor():
    """The memo must still do its job: one build shared by repeated calls with the same tensor."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device("cuda:0")

    class _Layer:
        pass

    layer = _Layer()
    evi = torch.randn([4, 2, 32], device = dev)

    a = _fw_linear_evidence(layer, {"categorical_evidence_logp": evi})
    b = _fw_linear_evidence(layer, {"categorical_evidence_logp": evi})
    assert a is b, "a repeated call with the same live tensor must be served from the cache"


def test_softevi_forward_correct_across_recycled_evidence_blocks():
    """End to end: a no-grad loop whose evidence tensors recycle addresses must stay exact.

    This is the shape of the evaluation loop that surfaced the bug -- fresh evidence every call, freed
    before the next allocation -- and the log-likelihood of a fully observed categorical is a closed
    form, so every iteration is checked against it rather than against a neighbouring iteration.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    device = torch.device("cuda:0")

    batch_size, num_vars, num_cats = 16, 8, 512

    nis = [juice.inputs(v, num_nodes = 1,
                        dist = dists.SoftEvidenceCategorical(num_cats = num_cats,
                                                             _dual_flow_backward = False))
           for v in range(num_vars)]
    ns = juice.summate(juice.multiply(*nis), num_nodes = 1)
    ns.init_parameters(perturbation = 0.0)

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)

    for it in range(12):
        with torch.no_grad():
            logits = torch.randn([batch_size, num_vars, num_cats], device = device) * (1.0 + it)
            logp = torch.log_softmax(logits, dim = 2)
            del logits

            target = logp.gather(2, data.unsqueeze(2)).squeeze(2).sum(dim = 1)
            lls = pc(data, categorical_evidence_logp = logp).view(-1)

            assert torch.isfinite(lls).all(), f"iteration {it}: non-finite log-likelihood"
            assert (lls <= 1e-3).all(), \
                f"iteration {it}: positive log-likelihood {float(lls.max()):+.4e}"
            assert torch.all((lls - target).abs() < 1e-2), \
                f"iteration {it}: max deviation {float((lls - target).abs().max()):.4e}"
            del logp, lls, target


if __name__ == "__main__":
    torch.manual_seed(4343442)
    torch.cuda.manual_seed(5434)
    test_evidence_cache_key_rejects_recycled_address("cpu")
    test_evidence_cache_key_rejects_recycled_address("cuda:0")
    test_evidence_cache_key_catches_inplace_mutation()
    test_fw_linear_evidence_not_served_across_a_recycled_block()
    test_fw_linear_evidence_still_hits_for_the_same_tensor()
    test_softevi_forward_correct_across_recycled_evidence_blocks()
