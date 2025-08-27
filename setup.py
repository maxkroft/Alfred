from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension('_rv_func', ['_rv_func.pyx'], 
                  include_dirs=[numpy.get_include()], 
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    )
)