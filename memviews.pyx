"""
memviews.pyx
"""

import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def summer(double[:] mv):
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
