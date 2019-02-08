"""
setup_carrays.py
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("carrays",
                ["carrays.pyx"],
                include_dirs=['.', get_include()])

setup(name="carrays",
      ext_modules=cythonize(ext))
