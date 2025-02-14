from __future__ import annotations

from typing import List
from copy import deepcopy

class BitSet(object):
    def __init__(self, byte_length = 1):
        self.values = bytearray(b"\x00" * byte_length)
        self.byte_length = byte_length

        self.length = 0

    @classmethod
    def from_array(cls, arr: List):
        b = BitSet(1)
        for value in arr:
            b.add(value)

        return b

    def to_list(self):
        vals = []
        for i in range(self.byte_length):
            for j in range(8):
                if (self.values[i] & (1 << (7 - j))) != 0:
                    vals.append(i * 8 + j)
        
        return vals

    def add(self, value):
        if self.hasitem(value):
            return

        if value >= self.byte_length * 8:
            new_length = (value // 8 + 1)
            self.values += bytearray(b"\x00" * (new_length - self.byte_length))
            self.byte_length = new_length

        self.values[value // 8] |= (1 << (7 - value % 8))

        self.length += 1

    def remove(self, value):
        if not self.hasitem(value):
            return

        self.values[value // 8] &= (0xff ^ (1 << (7 - value % 8)))

        self.length -= 1

    def hasitem(self, value):
        if value >= self.byte_length * 8:
            return False

        return (self.values[value // 8] & (1 << (7 - value % 8))) != 0

    def contains_all(self, other: BitSet):
        return len(self & other) == len(other)

    def contains_any(self, other: BitSet):
        return len(self & other) > 0

    def __and__(self, other: BitSet):
        if self.byte_length > other.byte_length:
            b = deepcopy(self)
            for i in range(other.byte_length):
                b.values[i] &= other.values[i]
            for i in range(other.byte_length, self.byte_length):
                b.values[i] = 0
        else:
            b = deepcopy(other)
            for i in range(self.byte_length):
                b.values[i] &= self.values[i]
            for i in range(self.byte_length, other.byte_length):
                b.values[i] = 0

        b.length = BitSet._count_ones(b.values)

        return b

    def __or__(self, other: BitSet):
        max_byte_length = max(self.byte_length, other.byte_length)
        b = BitSet(max_byte_length)

        for i in range(self.byte_length):
            b.values[i] |= self.values[i]
        for i in range(other.byte_length):
            b.values[i] |= other.values[i]

        b.length = BitSet._count_ones(b.values)

        return b

    def __eq__(self, other: BitSet):
        min_byte_length = min(self.byte_length, other.byte_length)
        for i in range(min_byte_length):
            if self.values[i] != other.values[i]:
                return False

        if self.byte_length > other.byte_length:
            for i in range(min_byte_length, self.byte_length):
                if self.values[i] != 0:
                    return False
        elif self.byte_length < other.byte_length:
            for i in range(min_byte_length, other.byte_length):
                if other.values[i] != 0:
                    return False

        return True

    def __iter__(self):
        for i in range(self.byte_length):
            for j in range(8):
                if (self.values[i] & (1 << (7 - j))) != 0:
                    yield i * 8 + j

    @staticmethod
    def _count_ones(arr: bytearray):
        count = 0
        for i in range(len(arr)):
            u = arr[i]
            u_count = u - ((u >> 1) & 3681400539) - ((u >> 2) & 1227133513)
            count += ((u_count + (u_count >> 3)) & 3340530119) % 63

        return count

    def __len__(self):
        return self.length

    def __repr__(self):
        if self.length <= 16:
            return "BitSet([" + ",".join([str(v) for v in self]) + "])"
        else:
            return "BitSet(num_elements={})".format(self.length)

    def __hash__(self):
        return hash(bytes(self.values))