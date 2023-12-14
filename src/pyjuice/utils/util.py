from __future__ import annotations

import math


def max_cdf_power_of_2(val: int):
    count = 0
    while True:
        halfval = val // 2

        if halfval * 2 != val:
            break

        val = halfval
        count += 1

    return 2 ** count
