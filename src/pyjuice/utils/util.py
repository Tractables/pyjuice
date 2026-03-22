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


def max_power_of_2_factor(n):
    if n == 0:
        return 0
    if n % 2 != 0:
        return 1

    power_of_2 = 1
    while n % 2 == 0:
        power_of_2 *= 2
        n //= 2  # Use integer division

    return power_of_2
