#cython: boundscheck=False
#cython: wraparound=False
"""
carrays.pyx
"""

import cython
import numpy as np
from libc.stdlib cimport malloc


def identity_2d_long(size_t N, size_t M):
    """
    Initialize an NxM identity matrix of longs.
    """

    # Allocate memory.
    cdef long *arr = <long*>malloc(N * M * sizeof(long))
    
    # Define a typed memoryview.
    # It is column-major by default.
    cdef long[:, ::1] mv = <long[:N, :M]>arr

    # Now we can work with the memoryview more conveniently.
    cdef int i, j
    for i in range(N):
        for j in range(M):
            if i == j:
                mv[i, j] = 1
            else:
                mv[i, j] = 0

    # We must wrap the memoryview as a numpy array before returning.
    # Note that this does not copy anything.
    # It's fine not to do this, but we would be returning just a
    # memoryview.
    return np.asarray(mv)
