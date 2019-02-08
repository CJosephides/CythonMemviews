"""
spectral_norm.py

Pure-python implementation of spectral norm calculations.
"""

import sys
from array import array
from math import sqrt

N_POWER_ITER = 10

def A(i, j):
    """Elements of the matrix we will work with."""
    return 1.0 / (((i + j) * (i + j + 1) >> 1) + i + 1)


def A_times_u(u, v):
    """
    v = Au

    where A is NxN,
          u, v are Nx1
    """

    n = len(u)

    for i in range(n):
        partial_sum = 0.
        for j in range(n):
            partial_sum += A(i,j) * u[j]

        v[i] = partial_sum


def At_times_u(u, v):
    """
    v = A^{T}u

    where AT^{T} is the conjugate transpose of A.
    """

    n = len(u)

    for i in range(n):
        partial_sum = 0.
        for j in range(n):
            partial_sum += A(j,i) * u[j]

        v[i] = partial_sum


def B_times_u(u, out, tmp):
    """
    Bu = A^{T}Au

    Since the spectral norm of A is the square root of the principal
    eigenvalue of B = A^{T}A.
    """

    A_times_u(u, tmp)
    At_times_u(tmp, out)


def spectral_norm(n):
    """
    The spectral norm of an infinite matrix A truncated to n rows and n columns.
    """

    u = array("d", [1.0] * n)
    v = array("d", [1.0] * n)
    tmp = array("d", [0.0] * n)

    # Use the power iteration to converge u to the principal eigenvector.
    for _ in range(N_POWER_ITER):
        B_times_u(u, v, tmp)
        B_times_u(v, u, tmp)

    # Finally, calculate the principal eigenvalue.
    vBv = vv = 0

    for ue, ve in zip(u, v):
        vBv += ue * ve
        vv  += ve *ve

    return sqrt(vBv / vv)


if __name__ == "__main__":
    n = int(sys.argv[1])
    spec_norm = spectral_norm(n)
    print("%0.9f" % spec_norm)
