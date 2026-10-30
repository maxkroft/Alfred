from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension('alfred._orbit_funcs', ['alfred/_orbit_funcs.pyx'], 
                  include_dirs=[numpy.get_include()], 
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    )
)