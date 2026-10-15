"""Fixtures shared by every test of :mod:`pyjuice.constraints`."""
import pytest


@pytest.fixture(autouse = True)
def nan_fresh_storage(monkeypatch):
    """The dfa_exact backend's new storage filled with NaN (it is not filled in use, see ``state.FRESH_FILL``): a
    query reading a slot no kernel wrote gives NaN, and fails the test."""
    from pyjuice.constraints.backends.dfa_exact import state
    monkeypatch.setattr(state, "FRESH_FILL", float("nan"))
