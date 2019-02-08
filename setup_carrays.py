"""
setup_carrays.py
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("carrays",
                ["carrays.pyx"])

setup(name="carrays",
      ext_modules=cythonize(ext))
