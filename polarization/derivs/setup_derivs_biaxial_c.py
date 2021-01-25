from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('derivs_biaxial_c.pyx'))
