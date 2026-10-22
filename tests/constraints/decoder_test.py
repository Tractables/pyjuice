"""
Row matching between a decoder's calls (:func:`pyjuice.constraints.decoder.match_rows`): which previous row every row
continues -- appended tokens, reordered and duplicated beams, any-order unmasking -- and none for a new row or one that
no longer observes a position it did; ties go to the row at its own index, then the first; random rows as a
brute force matches them.
"""
import numpy as np

from pyjuice.constraints.decoder import match_rows


def ev(rows):
    """``(data, missing)`` from rows of tokens, None where missing."""
    data = np.array([[0 if v is None else v for v in r] for r in rows], dtype = np.int64)
    missing = np.array([[v is None for v in r] for r in rows], dtype = bool)
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


def test_ties_go_to_the_row_at_its_own_index_then_the_first():
    previous = ev([[None, None, None], [1, None, None], [1, None, None]])
    parents, _ = match_rows(*ev([[1, 2, 0], [1, 0, 0], [1, 1, 1], [1, 2, 2]]), *previous)
    assert parents.tolist() == [0, 1, 2, 0]                     # row 3 has no previous row at its index
    parents, _ = match_rows(*ev([[2, 2, 0], [0, 0, 0]]), *previous)
    assert parents.tolist() == [0, 0]                           # row 1 does not continue previous row 1


def test_random_rows_match_a_brute_force():
    """Rows continuing random previous rows (own index or not), new rows, rows dropping an observed position."""
    rng = np.random.default_rng(0)
    previous_data, previous_missing = rng.integers(0, 2, (9, 6)), rng.random((9, 6)) < 0.3
    data, missing = rng.integers(0, 2, (40, 6)), rng.random((40, 6)) < 0.3
    src = rng.integers(0, 9, 20)                                              # half the rows continue a previous one
    data[:20], missing[:20] = previous_data[src], previous_missing[src] & (rng.random((20, 6)) < 0.5)
    parents, new = match_rows(data, missing, previous_data, previous_missing)
    for b in range(40):
        fits = [p for p in range(9) if all(previous_missing[p, j] or (not missing[b, j] and data[b, j] ==
                                                                     previous_data[p, j]) for j in range(6))]
        assert parents[b] == (b if b in fits else fits[0] if fits else -1), b
        assert new[b].tolist() == ([bool(previous_missing[parents[b], j] and not missing[b, j]) for j in range(6)]
                                   if fits else [False] * 6), b
    assert (parents >= 0).any() and (parents < 0).any() and (parents[:9] == np.arange(9)).any()
