import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

# os.environ["CC"] = "g++-5"
# os.environ["CXX"] = "g++-5"

setup(name='fatwalrus',
      version='1.0',
      description='Code for laying out on the beach all day with the homies.',
      author='Aaron Schein',
      packages=['fatwalrus'],
      install_requires=[],
      include_dirs=[np.get_include(),
                    os.path.expanduser('~/anaconda/include/')],
      ext_modules=cythonize(['fatwalrus/**/*.pyx']))
