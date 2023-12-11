import numpy as np
from numba import njit, prange


@njit(parallel = True)
def ff(a, b):
    for i in prange(10000000000):
        a[i%1000000000] = b[i%1000000000]


if __name__ == "__main__":
    a = np.random.uniform(size = [1000000000])
    b = np.random.uniform(size = [1000000000])

    ff(a, b)

    import time
    t0 = time.time()
    ff(a, b)
    t1 = time.time()
    print(t1 - t0)