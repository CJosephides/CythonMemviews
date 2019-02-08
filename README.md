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

Be very careful with the imports and cimports.
