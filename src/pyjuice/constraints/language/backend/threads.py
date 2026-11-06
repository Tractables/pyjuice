"""
How many threads the numba kernels here run on. They read and write memory, and gain little past a few threads; and a team waits for
its last thread: with a thread for every core of a machine, the thread that shares its core with another process's holds up all of
them for a time slice. Measured on 128 cores of which 11 busy: an empty parallel region takes 6 ms with 128 threads, 30 us with 64, 9 us
with 16 (71 us if the team has gone to sleep).
"""

from functools import wraps

from numba import get_num_threads, set_num_threads

#: the most threads a kernel here is given
MOST_THREADS = 16


def few_threads(function):
    """``function``, its kernels on at most :data:`MOST_THREADS` threads."""

    @wraps(function)
    def run(*args, **kwargs):
        was = get_num_threads()
        if was <= MOST_THREADS:
            return function(*args, **kwargs)
        set_num_threads(MOST_THREADS)
        try:
            return function(*args, **kwargs)
        finally:
            set_num_threads(was)
    return run
