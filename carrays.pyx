"""
carrays.pyx
"""

import cython
from libc.stdlib cimport malloc


def identity_2d_long(size_t N, size_t M):
    """
    Initialize an NxM identity matrix of longs.
    """

    cdef long *arr = <long*>malloc(N * M * sizeof(long))
