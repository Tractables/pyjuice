"""
Row matching between a decoder's calls (:func:`pyjuice.constraints.decoder.match_rows`): which previous row every row
continues -- appended tokens, reordered and duplicated beams, any-order unmasking -- and none for a new row or one that
no longer observes a position it did; ties go to the first row; chunking changes nothing.
"""
import torch

from pyjuice.constraints import decoder as D
from pyjuice.constraints.decoder import match_rows


def ev(rows):
    """``(data, missing)`` from rows of tokens, None where missing."""
    data = torch.tensor([[0 if v is None else v for v in r] for r in rows])
    missing = torch.tensor([[v is None for v in r] for r in rows])
    return data, missing


def test_appended_tokens_continue_every_row():
    parents, new = match_rows(*ev([[1, 2, 3, None], [0, 0, 1, None]]), *ev([[1, 2, None, None], [0, 0, None, None]]))
    assert parents.tolist() == [0, 1]
    assert new.tolist() == [[False, False, True, False]] * 2


def test_reordered_and_duplicated_beams():
    previous = ev([[1, 2, None], [0, 1, None]])
    parents, new = match_rows(*ev([[0, 1, 2], [1, 2, 0], [1, 2, 1]]), *previous)
    assert parents.tolist() == [1, 0, 0]
    assert new.tolist() == [[False, False, True]] * 3
    parents, _ = match_rows(*ev([[1, 2, None]]), *previous)                   # a beam dropped, nothing observed
    assert parents.tolist() == [0]


def test_a_new_row_and_an_unobserved_position_have_no_parent():
    previous = ev([[1, 2, 3, None]])
    parents, new = match_rows(*ev([[2, 2, None, None], [1, None, 3, 0], [1, 2, 3, None]]), *previous)
    assert parents.tolist() == [-1, -1, 0]
    assert not new[:2].any() and not new[2].any()                            # nothing newly observed on row 2 either


def test_any_order_unmasking():
    parents, new = match_rows(*ev([[0, 2, None, 1], [None, 2, 3, 1]]), *ev([[None, 2, None, 1]]))
    assert parents.tolist() == [0, 0]
    assert new.tolist() == [[True, False, False, False], [False, False, True, False]]


def test_ties_go_to_the_first_row():
    parents, _ = match_rows(*ev([[1, 2, 0]]), *ev([[None, None, None], [1, None, None], [1, None, None]]))
    assert parents.tolist() == [0]


def test_chunks_change_nothing(monkeypatch):
    g = torch.Generator().manual_seed(0)
    previous_data, previous_missing = torch.randint(0, 2, (9, 6), generator = g), torch.rand(9, 6, generator = g) < 0.3
    data, missing = torch.randint(0, 2, (40, 6), generator = g), torch.rand(40, 6, generator = g) < 0.3
    src = torch.randint(0, 9, (20,), generator = g)                           # half the rows continue a previous one
    data[:20], missing[:20] = previous_data[src], previous_missing[src] & (torch.rand(20, 6, generator = g) < 0.5)
    want = match_rows(data, missing, previous_data, previous_missing)
    monkeypatch.setattr(D, "MATCH_ELEMENTS", 1)                               # one row at a time
    got = match_rows(data, missing, previous_data, previous_missing)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    assert (want[0] >= 0).any() and (want[0] < 0).any()                       # both cases occur
