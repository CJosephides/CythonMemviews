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

### Trading safety for performance

The `cython.boundscheck` and `cython.wraparound` decorators respectively absolve Cython from doing explicit checks on
array bounds and disable negative indexing. We can also set these compiler directives for the entire module through the
header:

```
#cython: boundscheck=False
#cython:wraparound=False
```

or globally enable these at compile time through the `--directive` flag.
