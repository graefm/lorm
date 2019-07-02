import os
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

package_name = 'lorm'
nfft_name = 'nfft'
setup_dir = dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(setup_dir, package_name)
nfft_dir = os.path.join(setup_dir, nfft_name)
#nfft_libs = 'nfft3 fftw3'.split()
nfft_libs = 'nfft3_threads fftw3_threads'.split()

ext_modules = []
ext_modules.append(Extension(
            name=package_name+'.cmanif',
            sources=[os.path.join(package_dir, 'cmanif.pyx')],
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math'.split()
            ))
ext_modules.append(Extension(
            name=package_name+'.cfuncs',
            sources=[os.path.join(package_dir, 'cfuncs.pyx')],
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math'.split()
            ))
ext_modules.append(Extension(
            name=nfft_name+'.cnfft',
            sources=[os.path.join(nfft_dir, 'cnfft.pyx')],
            libraries=nfft_libs,
            library_dirs=[],
            include_dirs=[numpy.get_include()],
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math'.split()
            ))
ext_modules.append(Extension(
            name=nfft_name+'.cnfsft',
            sources=[os.path.join(nfft_dir, 'cnfsft.pyx')],
            libraries=nfft_libs,
            library_dirs=[],
            include_dirs=[numpy.get_include()],
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math'.split()
            ))
ext_modules.append(Extension(
            name=nfft_name+'.cnfsoft',
            sources=[os.path.join(nfft_dir, 'cnfsoft.pyx')],
            libraries=nfft_libs,
            library_dirs=[],
            include_dirs=[numpy.get_include()],
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math'.split()
            ))
ext_modules.append(Extension(
            name=nfft_name+'.cnfdsft',
            sources=[os.path.join(nfft_dir, 'cnfdsft.pyx')],
            libraries=nfft_libs,
            library_dirs=[],
            include_dirs=[numpy.get_include()],
            extra_compile_args='-O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math'.split()
            ))

setup(name="lorm",
      version="0.0.1",
      packages=["lorm"],
      ext_modules=cythonize(ext_modules)
      )
