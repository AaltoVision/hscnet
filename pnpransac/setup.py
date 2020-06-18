import sys
import os
from glob import glob
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

# Find opencv libraries in lib_folder
lib_folder = os.path.join(sys.prefix, 'lib')
cvlibs = list()
for file_name in glob(os.path.join(lib_folder, 'libopencv_*')):
    cvlibs.append(file_name.split('.')[0])
cvlibs = list(set(cvlibs))
cvlibs = ['-L{}'.format(lib_folder)] \
            + ['opencv_{}'.format(cvlib.split(os.path.sep)[-1]\
                .split('libopencv_')[-1]) for cvlib in cvlibs]

ext_modules = [
    Extension(
        "pnpransac",
        sources=["pnpransacpy.pyx"],
        language="c++",
        include_dirs=[np.get_include(),
                        os.path.join(sys.prefix, 'include'),
                        ],
        library_dirs=[lib_folder],
        libraries=cvlibs,
        extra_compile_args=['-fopenmp','-std=c++11'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='pnpransac',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    )