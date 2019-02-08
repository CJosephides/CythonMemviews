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
