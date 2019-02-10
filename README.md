# Working with memoryviews in Cython

## Prerequisites

We need numpy and cython for these examples.

## Memoryviews

```
python3 setup_memviews.py build_ext --inplace
```

In python:

```
import memviews
import numpy as np
na = np.linspace(0, 1000, 1000+1)
na_sum = memviews.summer(na)
```

On my machine, this is actually two times faster than

```
na.sum()
```

for some, very strange, reason.

### Ellipsis

We have to be careful to use the ellipsis operator, as we did with `mv[...] = math.pi` when we wish to modify
a memory view in place. To do otherwise -- for example `mv = math.pi` -- would mean that we wish to make a copy
of the data and set to pi. This follows the convention of standard python lists.

### Trading safety for performance

The `cython.boundscheck` and `cython.wraparound` decorators respectively absolve Cython from doing explicit checks on
array bounds and disable negative indexing. We can also set these compiler directives for the entire module through the
header:

```
#cython: boundscheck=False
#cython:wraparound=False
```

or globally enable these at compile time through the `--directive` flag.

### Strided vs contiguous arrays

The syntax

```
def summer(double[::1] mv)
```

declares double as a C-contiguous (or **column-major**) array. It is usually more efficient to work with C-contiguous
arrays, since this is the numpy standard. It is also required when working with external C/C++ libraries. If not
specified, and had we instead simply used `double[:] mv`, then we would be declaring mv to be a fully-strded array,
which is slower to work with.

To declare a multidimensional array as a C-contiguous array, use, for example:

```
double[:, :, ::1]
```

for a three-dimensional array.

## C arrays

The example `identity_2d_long` shows how we can dynamically allocate space for a two-dimensional matrix on the heap.

**Note: unlike modern C, we have to explicitly cast the void pointer return by malloc to the type that we are assigning.**

Be very careful with the imports and cimports. Also, don't forget to type the index cdefs.

The current implementation, even with disabled check directives, is not quite as fast as numpy (1.4 vs 1.8 usec). I'm not sure why.

### Automatic deallocation of memory

I couldn't get this to work. Attempting to `set_array_base` seg faults.

The [official Cython documents](https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html) suggest that we use the C-API functions directly instead of the low-level C functions.

### Note on usage

The more typical case is tu use numpy arrays to manage data (and memory!) and the use the basic features of typed memoryviews to efficiently access and modify these numpy arrays from cython.

## Spectral Norm of a matrix

This is a longer example showing how to use C-optimized routines in Cython for a common linear algebra calculation. Most of this code is taken from "Cython: A guide for python programmers."

`spectral_norm.py` consists of the pure-python implementation.

In IPython, use `%run -p ./spectral_norm.py 300` to see a profiled run. Our code spends >90% of its time in `A`, `A_times_u`, and `At_times_u`.

Alternatively, from the shell:

`python3 -m cProfile -s tottime spectral_norm.py 300 | less`

should show the same thing.

On my machine, the pure python implementation takes 1.265 seconds.

### No-frill Cythonize

Merely cythonizing the pure-python implementation, as in `cspectral_norm.pyx` and set up in `setup_cspectral_norm.py` results in a nearly three-fold speed increase (0.58 seconds)!

### Type information

We modified the `A` function by adding static type information. We also inlined the function.
After compiling and looking at the annotation file, we saw that the `return` line was still yellow.
The reason for this was the division-by-zero precaution. We turned that off by 
decorating the function with `cdivision(True)`, which tells the compiler not to worry
about this eventuality.

The total runtime is now 0.134 seconds. This is an order of magnitude faster than pure python and about four times faster than our naive cython code.

### Using typed memoryviews

We modified `A_times_u` by declaring the `u` and `v` arguments as one-dimensional, C-contiguous, typed memory views.
Further, we provided type information for the function's variables.

This modification reduces our runtime by another order of magnitude: 0.015 seconds. We could just leave it here, by why stop now?

### Using numpy

We can use numpy to manage our initial arrays. However, I find that the overhead of importing numpy
almost ten times larger than running the entire calculation!

Here, using `array.array` is alright, particularly because we don't use multi-dimensional arrays.

### cdef

We gain very little by converting `A_times_u` and `At_times_u` to `cdef`ed functions.

Note, however, that our Cython code can still call these functions, even though they will
not be available to python.

### Dynamical array allocation

We gain very little by dynamically allocating the `u`, `v`, and `tmp` arrays, and we also have to remember to free this
memory when we are done with it!

We are already using the efficient `array.array` through the buffer protocol, so dynamically allocating here really is unnecessary.

## General remarks

A nice feedback loop when working with Cython is to annotate the generated C code by passing `annotate=True` to the
`cythonize` call in the setup file. We then inspect the highlighted lines and take steps to optimize them.
We clear out the build, recompile, and repeat.

## Parallelization

We can paralellize loops like so:

```
from cython.parallel import prange

cdef int i
cdef int n = 30
cdef int sum = 0

for i in prange(n, nogil=True):
    sum += i

print(sum)
```

Example with a typed memoryview (e.g. a NumPy array):

```
from cython.parallel import prange

def func(double[:] x, double alpha):
    cdef Py_ssize_t i

    for i in prange(x.shape[0]):
        x[i] = alpha * x[i]
```

Refer to https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html for more examples. 


