"""
carrays.pyx
"""

import cython
import numpy as np
cimport numpy as cnp  # numpy's C interface
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
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


cdef class _finalizer:
    """
    This object owns some memory and deallocates it when the time is right.
    """
    cdef void *_data
    def __dealloc__(self):
        print("Deallocating...")
        if self._data is not NULL:
            free(self._data)


cdef void set_base(cnp.ndarray arr, void *carr):
    """
    A convenience function that creates a finalizer and makes it responsible for some C array
    by setting the array's base to the finalizer.
    """
    cdef _finalizer f = _finalizer()
    f._data = <void*>carr
    cnp.set_array_base(arr, f)


def identity_2d_long_wcleanup(size_t N, size_t M):
    """
    Initialize an NxM identity matrix of longs and
    register for automatic cleanup.
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

    # Construct ndarray and set its base attribute to a
    # _finalizer object.
    cdef cnp.ndarray nd_arr = np.asarray(mv)
    #set_base(nd_arr, arr)  # ERROR this isn't working.

    return nd_arr
