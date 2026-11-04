"""A layer's unique scopes, found with a set: a `BitSet`'s hash must agree with its `__eq__` (blind to trailing zeros)."""

from types import SimpleNamespace

from pyjuice.layer.layer import Layer
from pyjuice.utils import BitSet


def _longer(*values, byte_length = 16):
    b = BitSet(byte_length)
    for v in values:
        b.add(v)
    return b


def test_equal_scopes_of_other_byte_lengths_hash_alike():
    a, b = BitSet.from_array([1, 9]), _longer(1, 9)
    assert a.byte_length != b.byte_length
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_a_layers_scopes_are_the_unique_ones_in_the_order_first_seen():
    def ns(scope):
        return SimpleNamespace(scope = scope, block_size = 4)
    x, y, z = BitSet.from_array([5]), BitSet.from_array([0, 1]), BitSet.from_array([70])
    layer = Layer([ns(x), ns(y), ns(_longer(5)), ns(z), ns(_longer(0, 1)), ns(x)])
    assert layer.scopes == [x, y, z]
    assert all(s is o for s, o in zip(layer.scopes, [x, y, z]))
