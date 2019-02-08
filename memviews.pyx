"""
memviews.pyx
"""

import math
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def summer(double[::1] mv):
    """
    Sums the array.
    """

    cdef:
        int i, arr_len
        double arr_sum = 0

    arr_len = mv.shape[0]
    for i in range(arr_len):
        arr_sum += mv[i]

    return arr_sum


def make_delicious_pi(double[::1] mv):
    """
    Make the entire double array delicious.

    Modifies the array in-place.
    """

    mv[...] = math.pi
