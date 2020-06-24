import sys
import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

# where to find opencv headers and libraries
cv_include_dir = os.path.join(sys.prefix, 'include')
cv_library_dir = os.path.join(sys.prefix, 'lib')

ext_modules = [
    Extension(
        "pnpransac",
        sources=["pnpransacpy.pyx"],
        language="c++",
        include_dirs=[cv_include_dir],
        library_dirs=[cv_library_dir],
        libraries=['opencv_core','opencv_calib3d'],
        extra_compile_args=['-fopenmp','-std=c++11'],
    )
]

setup(
    name='pnpransac',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    )