"""
setup_cspectral_norm.py
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name="cspectral_norm",
      ext_modules=cythonize("cspectral_norm.pyx", annotate=True))
