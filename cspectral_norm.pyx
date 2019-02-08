"""
cspectral_norm.py

Cythonized implementation of spectral norm calculations.
"""

from cython cimport cdivision, boundscheck, wraparound
from array import array
from math import sqrt
from libc.stdlib cimport malloc, free

N_POWER_ITER = 10

@cdivision(True)
cdef inline double A(int i, int j):
    """Elements of the matrix we will work with."""
    return 1.0 / (((i + j) * (i + j + 1) >> 1) + i + 1)


@boundscheck(False)
@wraparound(False)
cdef void A_times_u(double[::1] u, double[::1] v):
    """
    v = Au

    where A is NxN,
          u, v are Nx1
    """
    
    cdef:
        int n, i, j
        double partial_sum

    n = u.shape[0]

    for i in range(n):
        partial_sum = 0.
        for j in range(n):
            partial_sum += A(i,j) * u[j]

        v[i] = partial_sum


@boundscheck(False)
@wraparound(False)
cdef void At_times_u(double[::1] u, double[::1] v):
    """
    v = A^{T}u

    where AT^{T} is the conjugate transpose of A.
    """

    cdef:
        int n, i, j
        double partial_sum

    n = u.shape[0]

    for i in range(n):
        partial_sum = 0.
        for j in range(n):
            partial_sum += A(j,i) * u[j]

        v[i] = partial_sum


cdef void B_times_u(double[::1] u, double[::1] out, double[::1] tmp):
    """
    Bu = A^{T}Au

    Since the spectral norm of A is the square root of the principal
    eigenvalue of B = A^{T}A.
    """

    A_times_u(u, tmp)
    At_times_u(tmp, out)

@boundscheck(False)
def spectral_norm(int n):
    """
    The spectral norm of an infinite matrix A truncated to n rows and n columns.
    """

    #cdef double *u_arr = <double*>malloc(n * sizeof(double))
    #cdef double *v_arr = <double*>malloc(n * sizeof(double))
    #cdef double *tmp_arr = <double*>malloc(n * sizeof(double))

    #cdef double[::1] u = <double[:n]>u_arr
    #cdef double[::1] v = <double[:n]>v_arr
    #cdef double[::1] tmp = <double[:n]>tmp_arr

    u = array("d", [1.0] * n)
    v = array("d", [1.0] * n)
    tmp = array("d", [0.0] * n)

    # Use the power iteration to converge u to the principal eigenvector.
    for _ in range(N_POWER_ITER):
        B_times_u(u, v, tmp)
        B_times_u(v, u, tmp)

    # Finally, calculate the principal eigenvalue.
    vBv = vv = 0

    #cdef int i
    #for i in range(n):
    #    vBv += u[i] * v[i]
    #    vv += v[i] * v[i]

    for ue, ve in zip(u, v):
        vBv += ue * ve
        vv  += ve *ve

    #free(u_arr)
    #free(v_arr)
    #free(tmp_arr)

    return sqrt(vBv / vv)
