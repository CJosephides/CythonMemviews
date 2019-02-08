"""
setup.py
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("memviews",
                ["memviews.pyx"])

setup(name="memviews",
      ext_modules=cythonize(ext))
