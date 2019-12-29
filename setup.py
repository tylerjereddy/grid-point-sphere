import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

# currently using:
# python setup.py build_ext --inplace

setup(
    ext_modules = cythonize("slerp.pyx"),
    include_dirs=[np.get_include()]
)
