"""
A query's inputs in the one form backends read (:mod:`pyjuice.constraints.evidence`): the evidence from ``mask_id``,
prefixes and ``missing_mask``; targets given as positions; candidates in every form.
"""
import pytest
import numpy as np
import torch

from pyjuice.constraints.evidence import candidate_tensor, evidence, target_mask

n = 5


def test_mask_id_marks_the_missing_tokens():
    data = torch.tensor([[0, -1, 2, -1, 4], [1, 1, -1, 0, 0]])
    got, missing = evidence(data, None, -1, n)
    assert torch.equal(got, data)
    assert torch.equal(missing, data == -1)
    got, missing = evidence(data, None, 2, n)                                 # another mask id
    assert torch.equal(missing, data == 2)


def test_a_prefix_is_padded_and_the_rest_is_missing():
    data = torch.tensor([[3, 1], [0, 2]])
    got, missing = evidence(data, None, -1, n)
    assert got.shape == (2, n) and torch.equal(got[:, :2], data)
    assert torch.equal(missing, torch.tensor([[False, False, True, True, True]] * 2))
    got, missing = evidence(data, None, 7, n)
    assert (got[:, 2:] == 7).all() and missing[:, 2:].all() and not missing[:, :2].any()


def test_missing_mask_adds_to_the_mask_id():
    data = torch.tensor([[0, -1, 2, 3, 4], [1, 1, 1, 0, 0]])
    shared = torch.tensor([True, False, False, False, True])
    _, missing = evidence(data, shared, -1, n)
    assert torch.equal(missing, (data == -1) | shared[None])
    per_sample = torch.rand(2, n) < 0.5
    _, missing = evidence(data, per_sample, -1, n)
    assert torch.equal(missing, (data == -1) | per_sample)
    _, missing = evidence(data[:, :3], shared, -1, n)                         # a prefix and a mask
    assert torch.equal(missing, torch.tensor([[True, True, False, True, True], [True, False, False, True, True]]))


def test_wrong_shapes_raise():
    with pytest.raises(ValueError, match = "data"):
        evidence(torch.zeros(2, n + 1, dtype = torch.long), None, -1, n)
    with pytest.raises(ValueError, match = "data"):
        evidence(torch.zeros(n, dtype = torch.long), None, -1, n)
    with pytest.raises(ValueError, match = "missing_mask"):
        evidence(torch.zeros(2, n, dtype = torch.long), torch.zeros(2, n + 1, dtype = torch.bool), -1, n)
    with pytest.raises(ValueError, match = "missing_mask"):
        evidence(torch.zeros(2, n, dtype = torch.long), torch.zeros(3, n, dtype = torch.bool), -1, n)


def test_targets_as_positions_are_the_mask():
    assert target_mask(None, 2, n) is None
    mask = torch.rand(2, n) < 0.5
    assert target_mask(mask, 2, n) is mask                                    # a mask is passed on as it is
    want = torch.zeros(2, n, dtype = torch.bool)
    want[:, [1, 3]] = True
    assert torch.equal(target_mask([1, 3], 2, n), want)
    assert torch.equal(target_mask(torch.tensor([3, 1]), 2, n), want)
    want = torch.zeros(2, n, dtype = torch.bool)
    want[0, 1] = want[1, 0] = want[1, 2] = True
    assert torch.equal(target_mask(torch.tensor([[1, -1], [0, 2]]), 2, n), want)                 # -1 pads
    want = torch.zeros(2, n, dtype = torch.bool)
    want[:, 2] = True
    assert torch.equal(target_mask([2, -1], 2, n), want)
    with pytest.raises(ValueError, match = "targets"):
        target_mask([n], 2, n)
    with pytest.raises(ValueError, match = "targets"):
        target_mask(torch.zeros(3, 1, dtype = torch.long), 2, n)
    with pytest.raises(TypeError, match = "targets"):
        target_mask(torch.zeros(2), 2, n)


@pytest.mark.parametrize("seed", range(4))
def test_targets_as_positions_are_numpys_mask(seed):
    """Positions ([k] for every sample, or [B, k]; -1 pads; repeated, any integer dtype) make numpy's mask, and a
    position out of 0..n-1 anywhere is refused."""
    rng = np.random.default_rng(seed)
    B, n_, k = int(rng.integers(1, 6)), int(rng.integers(1, 40)), int(rng.integers(1, 9))
    for shape in ((k,), (B, k)):
        for dtype in (np.int64, np.int32, np.uint8):
            pos = rng.integers(-1 if dtype != np.uint8 else 0, n_, shape).astype(dtype)
            want = np.zeros((B, n_ + 1), dtype = bool)
            if len(shape) == 1:
                want[:, pos.astype(np.int64)] = True
            else:
                want[np.arange(B)[:, None], pos.astype(np.int64)] = True
            got = target_mask(pos, B, n_)
            assert got.dtype == torch.bool and torch.equal(got, torch.from_numpy(want[:, :n_]))
        bad = rng.integers(0, n_, shape)
        bad.flat[int(rng.integers(0, bad.size))] = n_ if rng.random() < 0.5 else -2
        with pytest.raises(ValueError, match = "targets"):
            target_mask(bad, B, n_)
    assert not target_mask(np.zeros((B, 0), dtype = np.int64), B, n_).any()          # no position
    assert not target_mask(np.zeros(0, dtype = np.int64), B, n_).any()


def test_candidates_in_every_form():
    assert candidate_tensor(None, 2, n) is None
    shared = torch.tensor([4, 0, 7])
    got = candidate_tensor(shared, 2, n)
    assert got.shape == (2, n, 3) and (got == shared).all() and got.stride()[:2] == (0, 0)       # a view
    per_sample = torch.tensor([[1, 2], [3, -1]])
    got = candidate_tensor(per_sample, 2, n)
    assert got.shape == (2, n, 2) and torch.equal(got[:, 3], per_sample) and got.stride(1) == 0
    full = torch.randint(0, 9, (2, n, 4))
    assert candidate_tensor(full, 2, n) is full
    with pytest.raises(ValueError, match = "candidates"):
        candidate_tensor(torch.zeros(3, 2, dtype = torch.long), 2, n)
    with pytest.raises(ValueError, match = "candidates"):
        candidate_tensor(torch.zeros(2, n + 1, 2, dtype = torch.long), 2, n)
    with pytest.raises(TypeError, match = "candidates"):
        candidate_tensor(torch.zeros(4), 2, n)
